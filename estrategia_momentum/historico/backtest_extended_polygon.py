#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           📊 BACKTEST EXTENDIDO CON POLYGON.IO                                ║
║                                                                               ║
║           15 MESES de datos 4H (vs 6 meses de Yahoo)                          ║
║           Capital: €10,000 | 5 Posiciones                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code')
from momentum_breakout import (
    MomentumEngine, DynamicStopManager, calculate_atr,
    calculate_position_size, ASSETS
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

LIQUID_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM',
                  'V', 'MA', 'DIS', 'NFLX', 'AMD', 'CRM', 'PYPL', 'INTC']

CONFIG = {
    'initial_capital': 10000,
    'max_positions': 5,
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
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(data: Dict[str, pd.DataFrame], max_positions: int = 5,
                 initial_capital: float = 10000, months_filter: int = None) -> Dict:
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

    # Filtrar datos por meses si se especifica
    filtered_data = {}
    for ticker, df in data.items():
        if months_filter:
            cutoff = df.index.max() - timedelta(days=months_filter * 30)
            df_filtered = df[df.index >= cutoff].copy()
            if len(df_filtered) >= 50:
                filtered_data[ticker] = df_filtered
        else:
            filtered_data[ticker] = df.copy()

    if not filtered_data:
        return {'trades': [], 'equity_tracker': tracker, 'metrics': {}, 'max_positions': max_positions}

    # Generar señales
    signals_data = {}
    for ticker, df in filtered_data.items():
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

    # Holding period medio
    holding_bars = [t.bars_held for t in trades]
    avg_holding = np.mean(holding_bars) if holding_bars else 0

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
        'avg_holding_bars': avg_holding,
        'avg_holding_days': avg_holding * 4 / 24,  # 4H bars to days
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
║      📊 BACKTEST EXTENDIDO - POLYGON.IO DATA                                  ║
║                                                                               ║
║           ~15 MESES de datos 4H (vs 6 meses Yahoo)                            ║
║           Capital Inicial: €10,000                                            ║
║           5 Posiciones Simultáneas                                            ║
║           Estrategia: Momentum Breakout (LONGS ONLY)                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Cargar datos de Polygon
    cache_file = '/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code/polygon_data_cache.pkl'

    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✅ Datos cargados: {len(data)} tickers")
    except FileNotFoundError:
        print("  ❌ No se encontró polygon_data_cache.pkl")
        print("     Ejecuta primero la descarga de datos de Polygon")
        return

    # Mostrar rango de datos
    print("\n  📅 Rango de datos disponibles:")
    for ticker, df in list(data.items())[:3]:
        days = (df.index[-1] - df.index[0]).days
        print(f"     {ticker}: {df.index[0].date()} a {df.index[-1].date()} ({days} días, {len(df)} barras)")
    print(f"     ...")

    # Ejecutar backtests para diferentes periodos
    periods = [6, 9, 12, 15]  # Ahora podemos probar hasta 15 meses

    results = []

    for months in periods:
        print(f"\n{'='*70}")
        print(f"  🔄 Ejecutando backtest {months} MESES...")
        print(f"{'='*70}")

        result = run_backtest(
            data,
            max_positions=5,
            initial_capital=CONFIG['initial_capital'],
            months_filter=months
        )

        m = result['metrics']
        if m.get('total_trades', 0) > 0:
            results.append({
                'months': months,
                **m
            })

            print(f"""
  📊 RESULTADOS {months} MESES:
  {'─'*50}
  Capital Final:     €{m['final_equity']:,.0f}
  Rentabilidad:      {m['total_return_pct']:+.1f}%
  Total Trades:      {m['total_trades']}
  Win Rate:          {m['win_rate']:.1f}%
  Profit Factor:     {m['profit_factor']:.2f}
  Avg R-Multiple:    {m['avg_r_multiple']:+.2f}R
  Max Drawdown:      {m['max_drawdown_pct']:.1f}%
  Avg Holding:       {m['avg_holding_bars']:.1f} barras (~{m['avg_holding_days']:.1f} días)
  Trades >2R:        {m['pct_hit_2r']:.1f}%
""")
        else:
            print(f"  ⚠️ Sin datos suficientes para {months} meses")

    # Resumen comparativo
    if results:
        print(f"""
{'='*90}
  📊 RESUMEN COMPARATIVO - POLYGON DATA
{'='*90}

  {'Período':<10} {'Return %':<12} {'Trades':<10} {'Win %':<10} {'PF':<10} {'MaxDD %':<10} {'Risk-Adj':<10}
  {'─'*80}""")

        for r in results:
            risk_adj = r['total_return_pct'] / r['max_drawdown_pct'] if r['max_drawdown_pct'] > 0 else 0
            print(f"  {r['months']}m{'':<8} {r['total_return_pct']:>+8.1f}%    {r['total_trades']:>6}     {r['win_rate']:>6.1f}%   {r['profit_factor']:>6.2f}    {r['max_drawdown_pct']:>6.1f}%    {risk_adj:>8.2f}")

        print(f"""
{'='*90}
  💡 CONCLUSIONES
{'='*90}
""")
        # Encontrar mejor periodo
        best = max(results, key=lambda x: x['total_return_pct'] / x['max_drawdown_pct'] if x['max_drawdown_pct'] > 0 else 0)
        print(f"  ✅ Mejor periodo (risk-adjusted): {best['months']} meses")
        print(f"     Rentabilidad: {best['total_return_pct']:+.1f}%")
        print(f"     Max Drawdown: {best['max_drawdown_pct']:.1f}%")
        print(f"     Profit Factor: {best['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
