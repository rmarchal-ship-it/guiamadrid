#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           📊 COMPARATIVA: 3 vs 5 POSICIONES SIMULTÁNEAS                       ║
║                                                                               ║
║           Capital: €10,000                                                    ║
║           Períodos: 6 y 12 meses                                              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code')
from momentum_breakout import (
    MomentumEngine, DynamicStopManager, calculate_atr,
    calculate_position_size, ASSETS, TICKERS
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Excluir crypto (no disponible en yfinance) y crear lista completa
FULL_TICKERS = [t for t in TICKERS if not ASSETS[t].get('is_crypto', False)]

# Subset líquido (igual que backtest_real_capital.py)
LIQUID_TICKERS = [
    # US Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # ETFs
    'QQQ', 'SPY', 'IWM',
    # Commodities
    'GLD', 'SLV', 'USO',
    # Fixed Income
    'TLT',
    # EU
    'SAP', 'ASML',
]

CONFIG = {
    'initial_capital': 10000,
    'currency': 'EUR',
    'longs_only': True,

    # Señal
    'ker_threshold': 0.40,
    'volume_threshold': 1.3,
    'rsi_threshold': 50,
    'breakout_period': 20,

    # Stops
    'initial_atr_mult': 2.0,
    'max_hold_bars': 30,

    # Risk
    'target_vol_annual': 0.40,
    'slippage_pct': 0.10,

    # Tickers - Cambiar según test
    'test_tickers': LIQUID_TICKERS,  # Usar FULL_TICKERS para universo completo
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    ticker: str
    direction: str
    entry_price: float
    entry_date: datetime
    entry_atr: float
    position_units: float
    position_notional: float

    R: float = field(init=False)
    initial_stop: float = field(init=False)
    target_2r: float = field(init=False)
    target_4r: float = field(init=False)

    current_stop: float = field(init=False)
    stop_phase: str = field(default='initial')
    highest_since: float = field(init=False)
    lowest_since: float = field(init=False)
    bars_held: int = field(default=0)

    remaining_units: float = field(init=False)
    partial_exits: List = field(default_factory=list)
    hit_2r: bool = field(default=False)
    hit_4r: bool = field(default=False)

    exit_price: Optional[float] = field(default=None)
    exit_date: Optional[datetime] = field(default=None)
    exit_reason: Optional[str] = field(default=None)
    pnl_euros: float = field(default=0.0)
    pnl_pct: float = field(default=0.0)
    r_multiple: float = field(default=0.0)

    def __post_init__(self):
        atr_mult = CONFIG['initial_atr_mult']
        self.R = self.entry_atr * atr_mult

        if self.direction == 'long':
            self.initial_stop = self.entry_price - self.R
            self.target_2r = self.entry_price + self.R * 2
            self.target_4r = self.entry_price + self.R * 4
        else:
            self.initial_stop = self.entry_price + self.R
            self.target_2r = self.entry_price - self.R * 2
            self.target_4r = self.entry_price - self.R * 4

        self.current_stop = self.initial_stop
        self.highest_since = self.entry_price
        self.lowest_since = self.entry_price
        self.remaining_units = self.position_units
        self.partial_exits = []

    def update_extremes(self, high: float, low: float):
        self.highest_since = max(self.highest_since, high)
        self.lowest_since = min(self.lowest_since, low)
        self.bars_held += 1

    def check_stop_hit(self, low: float, high: float) -> bool:
        if self.direction == 'long':
            return low <= self.current_stop
        return high >= self.current_stop

    def check_partial_2r(self, low: float, high: float) -> bool:
        if self.hit_2r:
            return False
        if self.direction == 'long':
            return high >= self.target_2r
        return low <= self.target_2r

    def check_partial_4r(self, low: float, high: float) -> bool:
        if self.hit_4r:
            return False
        if self.direction == 'long':
            return high >= self.target_4r
        return low <= self.target_4r

    def execute_partial_exit(self, exit_price: float, pct: float, reason: str, date) -> float:
        units_to_exit = self.position_units * pct
        if units_to_exit > self.remaining_units:
            units_to_exit = self.remaining_units

        if self.direction == 'long':
            partial_pnl = (exit_price - self.entry_price) * units_to_exit
        else:
            partial_pnl = (self.entry_price - exit_price) * units_to_exit

        self.remaining_units -= units_to_exit
        self.partial_exits.append({
            'date': date,
            'price': exit_price,
            'units': units_to_exit,
            'pnl_euros': partial_pnl,
            'reason': reason
        })
        return partial_pnl

    def calculate_final_pnl(self):
        total_pnl = sum(pe['pnl_euros'] for pe in self.partial_exits)
        self.pnl_euros = total_pnl
        if self.position_notional > 0:
            self.pnl_pct = (total_pnl / self.position_notional) * 100
        if self.R > 0 and self.position_units > 0:
            total_r = 0
            for pe in self.partial_exits:
                if self.direction == 'long':
                    r = (pe['price'] - self.entry_price) / self.R
                else:
                    r = (self.entry_price - pe['price']) / self.R
                total_r += r * pe['units']
            self.r_multiple = total_r / self.position_units


# ═══════════════════════════════════════════════════════════════════════════════
# EQUITY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class EquityTracker:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = [(datetime.now(), initial_capital)]
        self.max_equity = initial_capital
        self.open_positions = 0

    def update_equity(self, pnl_euros: float, timestamp):
        self.equity += pnl_euros
        self.equity_curve.append((timestamp, self.equity))
        self.max_equity = max(self.max_equity, self.equity)

    def get_max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0
        equity_values = [e[1] for e in self.equity_curve]
        equity = np.array(equity_values)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / np.where(running_max > 0, running_max, 1) * 100
        return drawdown.max()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_data(ticker: str, months: int) -> Optional[pd.DataFrame]:
    try:
        if months <= 2:
            period = f'{months}mo'
            interval = '4h'
        else:
            period = f'{months}mo'
            interval = '1h'

        df = yf.download(ticker, period=period, interval=interval, progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if interval == '1h' and len(df) > 0:
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        return df if len(df) >= 50 else None
    except:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(data: Dict[str, pd.DataFrame], max_positions: int, initial_capital: float = 10000) -> Dict:
    """Ejecuta backtest con número específico de posiciones máximas."""

    tracker = EquityTracker(initial_capital)
    all_trades = []
    active_trades: Dict[str, Trade] = {}

    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG['longs_only'],
    )
    stop_mgr = DynamicStopManager(initial_atr_mult=CONFIG['initial_atr_mult'])

    slippage = CONFIG['slippage_pct'] / 100
    max_hold = CONFIG['max_hold_bars']

    # Generar señales
    signals_data = {}
    for ticker, df in data.items():
        df = df.copy()
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
        signals = engine.generate_signals(df)
        signals_data[ticker] = {'df': df, 'signals': signals}

    # Timeline unificado
    all_dates = set()
    for ticker, sd in signals_data.items():
        all_dates.update(sd['df'].index.tolist())
    all_dates = sorted(all_dates)

    # Iterar
    for current_date in all_dates:
        trades_to_close = []

        # Gestionar trades activos
        for ticker, trade in active_trades.items():
            if ticker not in signals_data:
                continue

            df = signals_data[ticker]['df']
            if current_date not in df.index:
                continue

            idx = df.index.get_loc(current_date)
            bar = df.iloc[idx]
            current_atr = df['ATR'].iloc[idx]

            trade.update_extremes(bar['High'], bar['Low'])

            # Actualizar stop
            stop_info = stop_mgr.calculate_stop(
                position_type=trade.direction,
                entry_price=trade.entry_price,
                current_price=bar['Close'],
                current_atr=current_atr,
                highest_since_entry=trade.highest_since,
                lowest_since_entry=trade.lowest_since,
                entry_atr=trade.entry_atr
            )

            if trade.direction == 'long':
                if stop_info['stop_price'] > trade.current_stop:
                    trade.current_stop = stop_info['stop_price']
                    trade.stop_phase = stop_info['phase']
            else:
                if stop_info['stop_price'] < trade.current_stop:
                    trade.current_stop = stop_info['stop_price']
                    trade.stop_phase = stop_info['phase']

            # Partial @ 2R
            if trade.check_partial_2r(bar['Low'], bar['High']) and trade.remaining_units > 0:
                exit_price = trade.target_2r * (1 - slippage if trade.direction == 'long' else 1 + slippage)
                partial_pnl = trade.execute_partial_exit(exit_price, 0.33, 'partial_2R', current_date)
                tracker.update_equity(partial_pnl, current_date)
                trade.hit_2r = True

            # Partial @ 4R
            if trade.check_partial_4r(bar['Low'], bar['High']) and trade.remaining_units > 0:
                exit_price = trade.target_4r * (1 - slippage if trade.direction == 'long' else 1 + slippage)
                partial_pnl = trade.execute_partial_exit(exit_price, 0.33, 'partial_4R', current_date)
                tracker.update_equity(partial_pnl, current_date)
                trade.hit_4r = True

            # Stop hit
            if trade.check_stop_hit(bar['Low'], bar['High']) and trade.remaining_units > 0:
                exit_price = trade.current_stop * (1 - slippage if trade.direction == 'long' else 1 + slippage)
                remaining_pct = trade.remaining_units / trade.position_units
                partial_pnl = trade.execute_partial_exit(exit_price, remaining_pct, f'stop_{trade.stop_phase}', current_date)
                tracker.update_equity(partial_pnl, current_date)
                trade.exit_price = exit_price
                trade.exit_date = current_date
                trade.exit_reason = f'stop_{trade.stop_phase}'
                trade.calculate_final_pnl()
                trades_to_close.append(ticker)
                continue

            # Time exit
            if trade.bars_held >= max_hold and trade.remaining_units > 0:
                exit_price = bar['Close'] * (1 - slippage if trade.direction == 'long' else 1 + slippage)
                remaining_pct = trade.remaining_units / trade.position_units
                partial_pnl = trade.execute_partial_exit(exit_price, remaining_pct, 'time_exit', current_date)
                tracker.update_equity(partial_pnl, current_date)
                trade.exit_price = exit_price
                trade.exit_date = current_date
                trade.exit_reason = 'time_exit'
                trade.calculate_final_pnl()
                trades_to_close.append(ticker)

        # Cerrar trades
        for ticker in trades_to_close:
            trade = active_trades.pop(ticker)
            tracker.open_positions -= 1
            all_trades.append(trade)

        # Nuevas entradas
        for ticker, sd in signals_data.items():
            if ticker in active_trades:
                continue
            if tracker.open_positions >= max_positions:
                continue

            df = sd['df']
            signals = sd['signals']

            if current_date not in df.index:
                continue

            idx = df.index.get_loc(current_date)
            if idx < 1:
                continue

            prev_signal = signals.iloc[idx - 1]
            if prev_signal == 0:
                continue

            bar = df.iloc[idx]
            prev_atr = df['ATR'].iloc[idx - 1]

            if prev_atr <= 0 or np.isnan(prev_atr):
                continue

            direction = 'long' if prev_signal == 1 else 'short'
            entry_price = bar['Open'] * (1 + slippage if direction == 'long' else 1 - slippage)

            asset_info = ASSETS.get(ticker, {})
            is_crypto = asset_info.get('is_crypto', False)

            # Position sizing: usar equity completo con volatilidad objetivo
            # El sistema de volatilidad inversa ajusta automáticamente
            size_info = calculate_position_size(
                account_balance=tracker.equity,
                current_atr=prev_atr,
                price=entry_price,
                target_vol_annual=CONFIG['target_vol_annual'],
                is_crypto=is_crypto
            )

            trade = Trade(
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                entry_date=current_date,
                entry_atr=prev_atr,
                position_units=size_info['units'],
                position_notional=size_info['notional']
            )

            active_trades[ticker] = trade
            tracker.open_positions += 1

    # Cerrar trades abiertos
    for ticker, trade in active_trades.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            exit_price = df['Close'].iloc[-1]
            if trade.remaining_units > 0:
                remaining_pct = trade.remaining_units / trade.position_units
                partial_pnl = trade.execute_partial_exit(exit_price, remaining_pct, 'end_of_data', df.index[-1])
                tracker.update_equity(partial_pnl, df.index[-1])
            trade.exit_price = exit_price
            trade.exit_date = df.index[-1]
            trade.exit_reason = 'end_of_data'
            trade.calculate_final_pnl()
            all_trades.append(trade)

    # Métricas
    metrics = calculate_metrics(all_trades, tracker, initial_capital)

    return {
        'trades': all_trades,
        'equity_tracker': tracker,
        'metrics': metrics,
        'max_positions': max_positions
    }


def calculate_metrics(trades: List[Trade], tracker: EquityTracker, initial_capital: float) -> Dict:
    if not trades:
        return {'total_trades': 0, 'total_pnl_euros': 0, 'total_return_pct': 0}

    total_trades = len(trades)
    winners = [t for t in trades if t.pnl_euros > 0]
    losers = [t for t in trades if t.pnl_euros <= 0]

    total_pnl = sum(t.pnl_euros for t in trades)
    final_equity = tracker.equity

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    avg_trade = total_pnl / total_trades if total_trades > 0 else 0
    avg_r = np.mean([t.r_multiple for t in trades]) if trades else 0

    max_consec = 0
    current_streak = 0
    for t in trades:
        if t.pnl_euros <= 0:
            current_streak += 1
            max_consec = max(max_consec, current_streak)
        else:
            current_streak = 0

    trades_with_2r = sum(1 for t in trades if t.hit_2r)
    trades_with_4r = sum(1 for t in trades if t.hit_4r)

    return {
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'total_pnl_euros': total_pnl,
        'total_return_pct': (final_equity - initial_capital) / initial_capital * 100,
        'final_equity': final_equity,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade_euros': avg_trade,
        'avg_win_euros': np.mean([t.pnl_euros for t in winners]) if winners else 0,
        'avg_loss_euros': np.mean([t.pnl_euros for t in losers]) if losers else 0,
        'avg_r_multiple': avg_r,
        'max_drawdown_pct': tracker.get_max_drawdown(),
        'max_consec_losses': max_consec,
        'trades_hit_2r': trades_with_2r,
        'trades_hit_4r': trades_with_4r,
        'pct_hit_2r': trades_with_2r / total_trades * 100 if total_trades > 0 else 0,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║      📊 COMPARATIVA POSICIONES - 16 TICKERS LÍQUIDOS                         ║
║                                                                               ║
║           Capital Inicial: €10,000                                            ║
║           Períodos: 6 y 12 meses                                              ║
║           Estrategia: Momentum Breakout (LONGS ONLY)                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    results = []
    position_configs = [3, 5, 8, 10]

    for months in [6, 12]:
        print(f"\n{'='*70}")
        print(f"  📥 Descargando datos para {months} MESES...")
        print(f"{'='*70}")

        # Descargar datos
        data = {}
        for ticker in CONFIG['test_tickers']:
            print(f"\r  Descargando {ticker}...", end='', flush=True)
            df = download_data(ticker, months)
            if df is not None:
                data[ticker] = df

        print(f"\r  ✅ {len(data)} tickers descargados                    ")

        if not data:
            print("  ⚠️ No hay datos disponibles")
            continue

        # Test con cada configuración de posiciones
        period_results = {}
        for max_pos in position_configs:
            print(f"  🔄 Ejecutando backtest con {max_pos} POSICIONES...")
            result = run_backtest(data, max_positions=max_pos, initial_capital=CONFIG['initial_capital'])
            period_results[max_pos] = result['metrics']
            results.append({
                'months': months,
                'positions': max_pos,
                **result['metrics']
            })

        # Mostrar resultados en tabla compacta
        print(f"""
{'─'*90}
  📊 RESULTADOS {months} MESES
{'─'*90}

  {'Métrica':<20} {'3 Pos':>14} {'5 Pos':>14} {'8 Pos':>14} {'10 Pos':>14}
  {'─'*76}""")

        # Capital Final
        print(f"  {'Capital Final':<20}", end='')
        for pos in position_configs:
            print(f" {'€{:,.0f}'.format(period_results[pos]['final_equity']):>13}", end='')
        print()

        # Rentabilidad
        print(f"  {'Rentabilidad %':<20}", end='')
        for pos in position_configs:
            print(f" {'{:+.1f}%'.format(period_results[pos]['total_return_pct']):>13}", end='')
        print()

        print(f"  {'─'*76}")

        # Total Trades
        print(f"  {'Total Trades':<20}", end='')
        for pos in position_configs:
            print(f" {period_results[pos]['total_trades']:>13}", end='')
        print()

        # Win Rate
        print(f"  {'Win Rate':<20}", end='')
        for pos in position_configs:
            print(f" {'{:.1f}%'.format(period_results[pos]['win_rate']):>13}", end='')
        print()

        # Profit Factor
        print(f"  {'Profit Factor':<20}", end='')
        for pos in position_configs:
            print(f" {'{:.2f}'.format(period_results[pos]['profit_factor']):>13}", end='')
        print()

        # Avg R-Multiple
        print(f"  {'Avg R-Multiple':<20}", end='')
        for pos in position_configs:
            print(f" {'{:+.2f}R'.format(period_results[pos]['avg_r_multiple']):>13}", end='')
        print()

        print(f"  {'─'*76}")

        # Max Drawdown
        print(f"  {'Max Drawdown':<20}", end='')
        for pos in position_configs:
            print(f" {'{:.1f}%'.format(period_results[pos]['max_drawdown_pct']):>13}", end='')
        print()

        # Max Consec Losses
        print(f"  {'Max Consec Losses':<20}", end='')
        for pos in position_configs:
            print(f" {period_results[pos]['max_consec_losses']:>13}", end='')
        print()

        print(f"  {'─'*76}")

        # Trades Hit +2R
        print(f"  {'Trades Hit +2R':<20}", end='')
        for pos in position_configs:
            print(f" {'{:.1f}%'.format(period_results[pos]['pct_hit_2r']):>13}", end='')
        print()

        print()

    # Resumen final
    print(f"""
{'='*90}
  📊 RESUMEN FINAL - COMPARATIVA
{'='*90}

  {'Período':<8} {'Pos':<6} {'P&L €':<14} {'Return %':<10} {'MaxDD %':<10} {'PF':<8} {'Sharpe*':<10}
  {'─'*80}""")

    for r in results:
        # Calcular ratio return/drawdown como proxy de Sharpe
        risk_adj = r['total_return_pct'] / r['max_drawdown_pct'] if r['max_drawdown_pct'] > 0 else 0
        print(f"  {r['months']}m{'':<6} {r['positions']:<6} €{r['total_pnl_euros']:>+10,.0f}   {r['total_return_pct']:>+7.1f}%    {r['max_drawdown_pct']:>6.1f}%    {r['profit_factor']:>6.2f}  {risk_adj:>8.2f}")

    print(f"""
  *Sharpe proxy = Return% / MaxDD%

{'='*90}
  💡 CONCLUSIONES
{'='*90}
""")

    # Análisis por período
    for months in [6, 12]:
        period_results = [r for r in results if r['months'] == months]
        if not period_results:
            continue

        # Encontrar mejor en cada métrica
        best_return = max(period_results, key=lambda x: x['total_return_pct'])
        best_dd = min(period_results, key=lambda x: x['max_drawdown_pct'])
        best_pf = max(period_results, key=lambda x: x['profit_factor'])
        best_risk_adj = max(period_results, key=lambda x: x['total_return_pct'] / x['max_drawdown_pct'] if x['max_drawdown_pct'] > 0 else 0)

        print(f"""
  📅 {months} MESES:
     • Mayor rentabilidad:    {best_return['positions']} posiciones ({best_return['total_return_pct']:+.1f}%)
     • Menor drawdown:        {best_dd['positions']} posiciones ({best_dd['max_drawdown_pct']:.1f}%)
     • Mejor Profit Factor:   {best_pf['positions']} posiciones (PF {best_pf['profit_factor']:.2f})
     • Mejor risk-adjusted:   {best_risk_adj['positions']} posiciones (ratio {best_risk_adj['total_return_pct']/best_risk_adj['max_drawdown_pct']:.2f})
""")


if __name__ == "__main__":
    main()
