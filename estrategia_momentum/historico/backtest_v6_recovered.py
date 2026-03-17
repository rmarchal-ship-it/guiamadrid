#!/usr/bin/env python3
"""
BACKTEST v6 RECUPERADO — Logica original restaurada

Diferencias vs backtest_definitivo.py (que fue modificado a v7/v8):
  1. max_hold_bars: 12 (no 8)
  2. Time exit: CIERRE FORZADO si pierde/flat a 12 barras (no solo trailing)
  3. Universo: 111 tickers originales (no 225)

MomentumEngine: IDENTICO (rsi_threshold=50, rsi_max=75, ker=0.40, vol=1.3, bp=20)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from momentum_breakout import (
    MomentumEngine, calculate_atr, calculate_position_size, ASSETS, TICKERS
)


# =============================================================================
# CATEGORIAS ORIGINALES v6 (sin las categorias NUEVO de v8)
# =============================================================================

V6_CATEGORIES = {
    'US_TECH', 'US_FINANCE', 'US_HEALTH', 'US_CONSUMER',
    'EU_GERMANY', 'EU_FRANCE',
    'EU_NETHERLANDS', 'EU_SPAIN', 'EU_ITALY', 'EU_BELGIUM', 'EU_FINLAND', 'EU_IRELAND',
    'COMMODITY_PRECIOUS', 'COMMODITY_ENERGY', 'COMMODITY_INDUSTRIAL', 'COMMODITY_AGRICULTURE',
    'US_INDEX', 'US_INDEX_LEV',
}

V6_TICKERS = [t for t, v in ASSETS.items() if v.get('category', '') in V6_CATEGORIES]


# =============================================================================
# CONFIGURACION v6 ORIGINAL
# =============================================================================

CONFIG = {
    # Capital
    'initial_capital': 10000,

    # Position sizing
    'target_risk_per_trade_pct': 2.0,
    'max_positions': 7,

    # Senales (IDENTICAS a v8)
    'ker_threshold': 0.40,
    'volume_threshold': 1.3,
    'rsi_threshold': 50,
    'rsi_max': 75,
    'breakout_period': 20,
    'longs_only': True,

    # Stop management
    'emergency_stop_pct': 0.15,
    'trail_trigger_r': 2.0,
    'trail_atr_mult': 4.0,

    # Time management — v6 ORIGINAL: 12 barras + cierre forzado
    'max_hold_bars': 12,

    # Filtro macro
    'use_macro_filter': True,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,

    # Costes
    'slippage_pct': 0.10,

    # Tickers: SOLO universo original v6 (111 tickers)
    'test_tickers': V6_TICKERS,
}


# =============================================================================
# TRADE CLASS — con time exit FORZADO original v6
# =============================================================================

@dataclass
class Trade:
    """
    Trade v6 original.
    Salidas:
    1. Emergency stop -15%
    2. Trailing Chandelier 4xATR (activado a +2R)
    3. Time exit a 12 barras: CIERRE FORZADO si perdiendo/flat,
       activar trailing si ganando
    """

    ticker: str
    entry_price: float
    entry_date: datetime
    entry_atr: float
    position_euros: float
    position_units: float

    R: float = field(init=False)

    trailing_stop: Optional[float] = field(default=None)
    trailing_active: bool = field(default=False)
    highest_since: float = field(init=False)
    max_r_mult: float = field(default=0.0)
    bars_held: int = field(default=0)

    exit_price: Optional[float] = field(default=None)
    exit_date: Optional[datetime] = field(default=None)
    exit_reason: Optional[str] = field(default=None)
    pnl_euros: float = field(default=0.0)
    pnl_pct: float = field(default=0.0)

    def __post_init__(self):
        self.R = self.entry_atr * 2.0
        self.highest_since = self.entry_price

    def update(self, high: float, low: float, close: float, current_atr: float) -> Optional[dict]:
        self.bars_held += 1
        self.highest_since = max(self.highest_since, high)
        r_mult = (close - self.entry_price) / self.R if self.R > 0 else 0
        self.max_r_mult = max(self.max_r_mult, r_mult)

        # 0. EMERGENCY STOP (-15%)
        emergency_level = self.entry_price * (1 - CONFIG['emergency_stop_pct'])
        if low <= emergency_level:
            self._close(emergency_level * (1 - CONFIG['slippage_pct'] / 100), 'emergency_stop')
            return {'type': 'full_exit', 'reason': 'emergency_stop'}

        # 1. TRAILING STOP CHECK
        if self.trailing_active and self.trailing_stop is not None:
            if low <= self.trailing_stop:
                self._close(self.trailing_stop * (1 - CONFIG['slippage_pct'] / 100), 'trailing_stop')
                return {'type': 'full_exit', 'reason': 'trailing_stop'}

        # 2. ACTUALIZAR TRAILING (Chandelier 4xATR)
        if r_mult >= CONFIG['trail_trigger_r']:
            chandelier = self.highest_since - (current_atr * CONFIG['trail_atr_mult'])
            if not self.trailing_active:
                self.trailing_active = True
                self.trailing_stop = chandelier
            elif chandelier > self.trailing_stop:
                self.trailing_stop = chandelier

        # 3. TIME EXIT v6 ORIGINAL: 12 barras
        #    - Perdiendo/flat → CIERRE FORZADO
        #    - Ganando → activar trailing (si no activo ya)
        if self.bars_held >= CONFIG['max_hold_bars']:
            if close <= self.entry_price:
                # PERDIENDO/FLAT → CIERRE FORZADO (esto es lo que se elimino en v7)
                self._close(close * (1 - CONFIG['slippage_pct'] / 100), 'time_exit')
                return {'type': 'full_exit', 'reason': 'time_exit'}
            else:
                # GANANDO → activar trailing si no esta activo
                if not self.trailing_active:
                    chandelier = self.highest_since - (current_atr * CONFIG['trail_atr_mult'])
                    self.trailing_active = True
                    self.trailing_stop = chandelier

        return None

    def _close(self, exit_price: float, reason: str):
        self.pnl_euros = (exit_price - self.entry_price) * self.position_units
        self.pnl_pct = (self.pnl_euros / self.position_euros) * 100 if self.position_euros > 0 else 0
        self.exit_price = exit_price
        self.exit_reason = reason


# =============================================================================
# EQUITY TRACKER
# =============================================================================

class EquityTracker:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.max_equity = initial_capital
        self.open_positions = 0

    def get_position_size(self, current_atr: float, price: float) -> Dict:
        risk_pct = CONFIG['target_risk_per_trade_pct'] / 100
        R = current_atr * 2.0
        if R <= 0 or price <= 0:
            return {'units': 0, 'notional': 0}

        dollar_risk = self.equity * risk_pct
        units = dollar_risk / R
        notional = units * price

        max_notional = self.equity / CONFIG['max_positions'] * 2
        if notional > max_notional:
            notional = max_notional
            units = notional / price

        return {'units': units, 'notional': notional}

    def update_equity(self, pnl: float, date):
        self.equity += pnl
        self.equity_curve.append((date, self.equity))
        self.max_equity = max(self.max_equity, self.equity)

    def get_max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        equity_values = [self.initial_capital] + [e[1] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        for eq in equity_values:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_data(ticker: str, months: int) -> Optional[pd.DataFrame]:
    try:
        if months > 60:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30.44)
            df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'), interval='1d', progress=False)
        else:
            df = yf.download(ticker, period=f'{months}mo', interval='1d', progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df if len(df) >= 50 else None

    except Exception:
        return None


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(months: int = 36, verbose: bool = True) -> Dict:
    n_tickers = len(CONFIG['test_tickers'])
    print(f"\n{'='*70}")
    print(f"  BACKTEST v6 RECUPERADO -- {months} MESES -- {n_tickers} tickers")
    print(f"{'='*70}")
    print(f"  Capital: EUR {CONFIG['initial_capital']:,}")
    print(f"  Sizing: {CONFIG['target_risk_per_trade_pct']}% risk/trade (inverse vol)")
    print(f"  Posiciones: max {CONFIG['max_positions']}")
    print(f"  Stops: emergency -{CONFIG['emergency_stop_pct']*100:.0f}% | trailing {CONFIG['trail_atr_mult']:.0f}xATR a +2R")
    print(f"  Time exit: {CONFIG['max_hold_bars']}d CIERRE FORZADO si pierde (v6 original)")
    macro_str = f"SPY > SMA{CONFIG.get('macro_sma_period', 50)}" if CONFIG.get('use_macro_filter') else "OFF"
    print(f"  Filtro macro: {macro_str}")
    print(f"{'='*70}\n")

    # Descargar datos
    print("  Descargando datos...")
    all_data = {}
    failed = []
    for i, ticker in enumerate(CONFIG['test_tickers']):
        df = download_data(ticker, months)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            all_data[ticker] = df
        else:
            failed.append(ticker)
        if (i + 1) % 10 == 0 or i == n_tickers - 1:
            print(f"\r  Descargados: {len(all_data)}/{n_tickers} OK, {len(failed)} fallidos", end='')

    print(f"\n  Tickers con datos: {len(all_data)}")
    if failed and verbose:
        print(f"  Fallidos: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
    print()

    if not all_data:
        return {'error': 'No data'}

    # Inicializar
    tracker = EquityTracker(CONFIG['initial_capital'])
    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG['rsi_max'],
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG['longs_only']
    )

    # Generar senales y metadata
    signals_data = {}
    total_signals = 0
    for ticker, df in all_data.items():
        meta = engine.generate_signals_with_metadata(df)
        signals = meta['signal']
        n_long = (signals == 1).sum()
        total_signals += n_long
        signals_data[ticker] = {
            'df': df,
            'signals': signals,
            'ker': meta['ker'],
            'rsi': meta['rsi'],
            'vol_ratio': meta['vol_ratio'],
        }

    print(f"  Senales LONG totales: {total_signals}\n")

    # Filtro macro
    macro_bullish = {}
    if CONFIG.get('use_macro_filter', False):
        macro_ticker = CONFIG.get('macro_ticker', 'SPY')
        if macro_ticker in all_data:
            macro_df = all_data[macro_ticker]
            sma_period = CONFIG.get('macro_sma_period', 50)
            macro_sma = macro_df['Close'].rolling(window=sma_period).mean()
            for date in macro_df.index:
                sma_val = macro_sma.loc[date] if date in macro_sma.index else None
                close_val = macro_df['Close'].loc[date] if date in macro_df.index else None
                if sma_val is not None and close_val is not None and not pd.isna(sma_val):
                    macro_bullish[date] = close_val > sma_val
                else:
                    macro_bullish[date] = True
            n_bullish = sum(1 for v in macro_bullish.values() if v)
            n_bearish = sum(1 for v in macro_bullish.values() if not v)
            print(f"  Filtro macro ({macro_ticker} > SMA{sma_period}): "
                  f"{n_bullish} dias bull / {n_bearish} dias bear\n")

    # Timeline unificado
    all_dates = set()
    for sd in signals_data.values():
        all_dates.update(sd['df'].index.tolist())
    all_dates = sorted(all_dates)

    active_trades: Dict[str, Trade] = {}
    all_trades: List[Trade] = []

    for current_date in all_dates:
        # 1. GESTIONAR TRADES ACTIVOS
        trades_to_close = []

        for ticker, trade in active_trades.items():
            if ticker not in signals_data:
                continue
            df = signals_data[ticker]['df']
            if current_date not in df.index:
                continue

            idx = df.index.get_loc(current_date)
            bar = df.iloc[idx]
            current_atr = df['ATR'].iloc[idx]

            result = trade.update(
                high=bar['High'], low=bar['Low'],
                close=bar['Close'], current_atr=current_atr
            )

            if result and result['type'] == 'full_exit':
                trade.exit_date = current_date
                trades_to_close.append(ticker)
                tracker.update_equity(trade.pnl_euros, current_date)

        for ticker in trades_to_close:
            trade = active_trades.pop(ticker)
            tracker.open_positions -= 1
            all_trades.append(trade)

        # 2. BUSCAR NUEVAS SENALES
        if CONFIG.get('use_macro_filter', False) and macro_bullish:
            if current_date in macro_bullish:
                is_macro_ok = macro_bullish[current_date]
            else:
                prev_dates = [d for d in macro_bullish if d < current_date]
                is_macro_ok = macro_bullish[prev_dates[-1]] if prev_dates else False
        else:
            is_macro_ok = True

        if tracker.open_positions < CONFIG['max_positions'] and is_macro_ok:
            candidates = []

            for ticker, sd in signals_data.items():
                if ticker in active_trades:
                    continue
                df = sd['df']
                signals = sd['signals']
                if current_date not in df.index:
                    continue
                idx = df.index.get_loc(current_date)
                if idx < 1:
                    continue

                prev_signal = signals.iloc[idx - 1]
                if prev_signal != 1:
                    continue

                prev_atr = df['ATR'].iloc[idx - 1]
                if pd.isna(prev_atr) or prev_atr <= 0:
                    continue

                candidates.append((ticker, idx, prev_atr))

            # RANKING MULTI-FACTOR
            ranked = []
            for ticker, idx, prev_atr in candidates:
                sd = signals_data[ticker]
                df_t = sd['df']
                prev_idx = idx - 1

                ker_val = sd['ker'].iloc[prev_idx] if prev_idx >= 0 else 0
                rsi_val = sd['rsi'].iloc[prev_idx] if prev_idx >= 0 else 50
                rsi_score = max(0, min(1, (rsi_val - CONFIG['rsi_threshold']) / (CONFIG['rsi_max'] - CONFIG['rsi_threshold'])))

                vol_val = sd['vol_ratio'].iloc[prev_idx] if prev_idx >= 0 else 1.0
                vol_score = min(1, max(0, (vol_val - 1.0) / 2.0))

                if prev_idx >= 1:
                    close_prev = df_t['Close'].iloc[prev_idx]
                    rolling_high_prev = df_t['High'].iloc[max(0, prev_idx - CONFIG['breakout_period']):prev_idx].max()
                    breakout_pct = (close_prev - rolling_high_prev) / rolling_high_prev if rolling_high_prev > 0 else 0
                    breakout_score = min(1, max(0, breakout_pct / 0.05))
                else:
                    breakout_score = 0

                price_prev = df_t['Close'].iloc[prev_idx] if prev_idx >= 0 else 1
                atr_pct = prev_atr / price_prev if price_prev > 0 else 0
                atr_score = min(1, atr_pct / 0.04)

                composite = (
                    0.30 * ker_val +
                    0.20 * rsi_score +
                    0.20 * vol_score +
                    0.15 * breakout_score +
                    0.15 * atr_score
                )

                ranked.append((ticker, idx, prev_atr, composite))

            ranked.sort(key=lambda x: x[3], reverse=True)

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= CONFIG['max_positions']:
                    break

                df = signals_data[ticker]['df']
                bar = df.iloc[idx]

                size_info = tracker.get_position_size(
                    current_atr=prev_atr,
                    price=bar['Open']
                )

                entry_price = bar['Open'] * (1 + CONFIG['slippage_pct'] / 100)
                position_euros = size_info['notional']
                position_units = size_info['units']

                max_per_position = tracker.equity / CONFIG['max_positions']
                if position_euros > max_per_position:
                    position_euros = max_per_position
                    position_units = position_euros / entry_price

                if position_euros < 100:
                    continue

                trade = Trade(
                    ticker=ticker,
                    entry_price=entry_price,
                    entry_date=current_date,
                    entry_atr=prev_atr,
                    position_euros=position_euros,
                    position_units=position_units
                )

                active_trades[ticker] = trade
                tracker.open_positions += 1

    # Cerrar trades abiertos al final
    for ticker, trade in active_trades.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            exit_price = df['Close'].iloc[-1]
            trade._close(exit_price, 'end_of_data')
            trade.exit_date = df.index[-1]
            tracker.update_equity(trade.pnl_euros, df.index[-1])
            all_trades.append(trade)

    # =================================================================
    # METRICAS
    # =================================================================

    if not all_trades:
        return {'error': 'No trades'}

    total_trades = len(all_trades)
    winners = [t for t in all_trades if t.pnl_euros > 0]
    losers = [t for t in all_trades if t.pnl_euros <= 0]

    total_pnl = sum(t.pnl_euros for t in all_trades)
    win_rate = len(winners) / total_trades * 100

    gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
    profit_factor = gross_profit / gross_loss

    trades_gt_3r = sum(1 for t in all_trades if t.max_r_mult >= 3.0)
    max_dd = tracker.get_max_drawdown()

    total_return_pct = (tracker.equity / CONFIG['initial_capital'] - 1) * 100
    annualized = ((1 + total_return_pct / 100) ** (12 / months) - 1) * 100 if months > 0 else 0

    avg_win_euros = np.mean([t.pnl_euros for t in winners]) if winners else 0
    avg_win_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss_euros = np.mean([t.pnl_euros for t in losers]) if losers else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0

    avg_bars_winners = np.mean([t.bars_held for t in winners]) if winners else 0
    avg_bars_losers = np.mean([t.bars_held for t in losers]) if losers else 0

    best_trade = max(all_trades, key=lambda t: t.pnl_pct)
    worst_trade = min(all_trades, key=lambda t: t.pnl_pct)

    # Contar time_exits forzados
    time_exits = [t for t in all_trades if t.exit_reason == 'time_exit']

    print(f"""
{'='*70}
  RESULTADOS v6 RECUPERADO -- {months} MESES ({len(all_data)} tickers)
{'='*70}

  CAPITAL:
     Inicial:        EUR {CONFIG['initial_capital']:,.2f}
     Final:          EUR {tracker.equity:,.2f}
     P&L Total:      EUR {total_pnl:+,.2f} ({total_return_pct:+.1f}%)
     Annualizado:    {annualized:+.1f}%
     Max Drawdown:   -{max_dd:.1f}%

  TRADES:
     Total:          {total_trades}
     Ganadores:      {len(winners)} ({win_rate:.1f}%)
     Perdedores:     {len(losers)}
     Profit Factor:  {profit_factor:.2f}
     Time exits:     {len(time_exits)} (cierre forzado v6)

  FAT TAILS:
     Alcanzaron +3R: {trades_gt_3r} ({trades_gt_3r/total_trades*100:.1f}%)
     Best trade:     {best_trade.ticker} {best_trade.pnl_pct:+.1f}%
     Worst trade:    {worst_trade.ticker} {worst_trade.pnl_pct:+.1f}%

  PROMEDIOS:
     Avg Win:        EUR {avg_win_euros:+.2f} ({avg_win_pct:+.1f}%) | {avg_bars_winners:.0f} dias
     Avg Loss:       EUR {avg_loss_euros:.2f} ({avg_loss_pct:.1f}%) | {avg_bars_losers:.0f} dias
""")

    exit_reasons = {}
    for t in all_trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    print("  RAZONES DE SALIDA:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"     {reason:20} {count:3} ({count/total_trades*100:.1f}%)")

    return {
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'total_pnl_euros': total_pnl,
        'total_return_pct': total_return_pct,
        'annualized_return_pct': annualized,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'trades_gt_3r': trades_gt_3r,
        'time_exits': len(time_exits),
        'final_equity': tracker.equity,
        'total_signals': total_signals,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Backtest v6 Recuperado')
    parser.add_argument('--months', type=int, default=60, help='Meses maximo (default: 60)')

    args = parser.parse_args()

    print(f"""
======================================================================
  BACKTEST v6 RECUPERADO — MOMENTUM BREAKOUT (Fat Tails)
======================================================================
  - {len(CONFIG['test_tickers'])} tickers (universo original v6)
  - Time exit: {CONFIG['max_hold_bars']}d CIERRE FORZADO si pierde (v6 original)
  - Max {CONFIG['max_positions']} posiciones
  - Trailing {CONFIG['trail_atr_mult']:.0f}xATR a +{CONFIG['trail_trigger_r']:.0f}R
  - Filtro macro: SPY > SMA{CONFIG.get('macro_sma_period', 50)}
======================================================================
    """)

    all_results = []

    for period in [6, 12, 18, 24, 36, 48, 60]:
        if period <= args.months:
            result = run_backtest(months=period, verbose=False)
            if 'error' not in result:
                all_results.append({'period': period, **result})

    if len(all_results) > 1:
        print(f"""
{'='*70}
  RESUMEN COMPARATIVO v6 RECUPERADO
{'='*70}

  {'Periodo':<8} {'Trades':<8} {'Win%':<7} {'PnL EUR':<11} {'Return%':<9} {'Annual%':<9} {'MaxDD%':<7} {'PF':<6} {'TimeEx':<7}
  {'-'*76}""")

        for r in all_results:
            print(f"  {r['period']}m{' '*5} {r['total_trades']:<8} "
                  f"{r['win_rate']:<7.1f} "
                  f"EUR{r['total_pnl_euros']:>+8,.0f}  {r['total_return_pct']:>+7.1f}%  "
                  f"{r['annualized_return_pct']:>+7.1f}%  "
                  f"{r['max_drawdown']:>5.1f}%  {r['profit_factor']:.2f}  "
                  f"{r.get('time_exits', '?')}")

        print()


if __name__ == "__main__":
    main()
