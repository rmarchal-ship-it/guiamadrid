#!/usr/bin/env python3
"""
BACKTEST CRYPTO — Momentum Breakout v8 adaptado a BTC/ETH/SOL
Creado: 28 Feb 2026
Objetivo: Validar si la estrategia Momentum Breakout es aplicable a un
          universo cerrado de 3 criptomonedas (BTC, ETH, SOL).

Adaptaciones respecto a v8 original:
  - Universo: 3 tickers (BTC-USD, ETH-USD, SOL-USD)
  - Max posiciones: 3 (ajustado al universo)
  - Sin opciones (no aplica a cripto en este contexto)
  - Macro filter: BTC > SMA50 (BTC como referencia macro del ecosistema)
  - Slippage: 0.05% (cripto tiene buena liquidez en spot)
  - Cripto opera 24/7 pero yfinance da barras diarias

Uso:
  python3 backtest_crypto.py --months 6 --verbose
  python3 backtest_crypto.py --months 12
  python3 backtest_crypto.py --months 36
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
import warnings
import argparse
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_breakout import MomentumEngine, calculate_atr


# =============================================================================
# UNIVERSO CRYPTO
# =============================================================================

CRYPTO_TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD']

CRYPTO_NAMES = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'SOL-USD': 'Solana',
}

# =============================================================================
# CONFIG CRYPTO (basado en v8, ajustado para 3 tickers)
# =============================================================================

CONFIG = {
    'initial_capital': 10000,
    'target_risk_per_trade_pct': 2.0,
    'max_positions': 3,           # 3 tickers = 3 max

    # Señales (mismos parametros v8)
    'ker_threshold': 0.40,
    'volume_threshold': 1.3,
    'rsi_threshold': 50,
    'rsi_max': 75,
    'breakout_period': 20,
    'longs_only': True,

    # Stops y trailing (mismos v8)
    'emergency_stop_pct': 0.15,
    'trail_trigger_r': 2.0,
    'trail_atr_mult': 4.0,

    # Time exit (mismo v8)
    'max_hold_bars': 8,
    'time_exit_trail_atr_mult': 3.0,

    # Macro filter: BTC como referencia
    'use_macro_filter': True,
    'macro_ticker': 'BTC-USD',
    'macro_sma_period': 50,

    # Costes (cripto spot, menor que acciones)
    'slippage_pct': 0.05,
}


# =============================================================================
# TRADE CLASS (idéntica a v8)
# =============================================================================

@dataclass
class Trade:
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

    def update(self, high, low, close, current_atr):
        self.bars_held += 1
        self.highest_since = max(self.highest_since, high)
        r_mult = (close - self.entry_price) / self.R if self.R > 0 else 0
        self.max_r_mult = max(self.max_r_mult, r_mult)

        # Emergency stop -15%
        emergency_level = self.entry_price * (1 - CONFIG['emergency_stop_pct'])
        if low <= emergency_level:
            self._close(emergency_level * (1 - CONFIG['slippage_pct'] / 100), 'emergency_stop')
            return {'type': 'full_exit', 'reason': 'emergency_stop'}

        # Trailing stop check
        if self.trailing_active and self.trailing_stop is not None:
            if low <= self.trailing_stop:
                self._close(self.trailing_stop * (1 - CONFIG['slippage_pct'] / 100), 'trailing_stop')
                return {'type': 'full_exit', 'reason': 'trailing_stop'}

        # Trailing activation at +2R
        if r_mult >= CONFIG['trail_trigger_r']:
            chandelier = self.highest_since - (current_atr * CONFIG['trail_atr_mult'])
            if not self.trailing_active:
                self.trailing_active = True
                self.trailing_stop = chandelier
            elif chandelier > self.trailing_stop:
                self.trailing_stop = chandelier

        # Time exit: trailing a 8 bars (nunca forzar salida)
        if self.bars_held >= CONFIG['max_hold_bars']:
            if not self.trailing_active:
                trail_mult = CONFIG.get('time_exit_trail_atr_mult', 3.0)
                chandelier = self.highest_since - (current_atr * trail_mult)
                breakeven = self.entry_price * (1 + CONFIG['slippage_pct'] / 100)
                self.trailing_active = True
                if close <= self.entry_price:
                    self.trailing_stop = max(chandelier, self.entry_price * 0.95)
                else:
                    self.trailing_stop = max(chandelier, breakeven)

        return None

    def _close(self, exit_price, reason):
        self.pnl_euros = (exit_price - self.entry_price) * self.position_units
        self.pnl_pct = (self.pnl_euros / self.position_euros) * 100 if self.position_euros > 0 else 0
        self.exit_price = exit_price
        self.exit_reason = reason


# =============================================================================
# EQUITY TRACKER
# =============================================================================

class EquityTracker:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.max_equity = initial_capital
        self.open_positions = 0

    def get_position_size(self, ticker, current_atr, price):
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

    def update_equity(self, pnl, date):
        self.equity += pnl
        self.equity_curve.append((date, self.equity))
        self.max_equity = max(self.max_equity, self.equity)

    def get_max_drawdown(self):
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

def download_data(ticker, months):
    try:
        if months > 60:
            end = datetime.now()
            start = end - timedelta(days=months * 30)
            df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                             end=end.strftime('%Y-%m-%d'), interval='1d', progress=False)
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
# SIGNAL GENERATION + RANKING
# =============================================================================

def generate_all_signals(all_data, engine):
    signals_data = {}
    total_signals = 0
    for ticker, df in all_data.items():
        meta = engine.generate_signals_with_metadata(df)
        signals = meta['signal']
        n_long = (signals == 1).sum()
        total_signals += n_long
        signals_data[ticker] = {
            'df': df, 'signals': signals,
            'ker': meta['ker'], 'rsi': meta['rsi'], 'vol_ratio': meta['vol_ratio'],
        }
    return signals_data, total_signals


def build_macro_filter(all_data):
    macro_bullish = {}
    macro_ticker = CONFIG['macro_ticker']
    if macro_ticker in all_data:
        macro_df = all_data[macro_ticker]
        sma = macro_df['Close'].rolling(window=CONFIG['macro_sma_period']).mean()
        for date in macro_df.index:
            sma_val = sma.loc[date] if date in sma.index else None
            close_val = macro_df['Close'].loc[date] if date in macro_df.index else None
            if sma_val is not None and close_val is not None and not pd.isna(sma_val):
                macro_bullish[date] = close_val > sma_val
            else:
                macro_bullish[date] = True
    return macro_bullish


def rank_candidates(candidates, signals_data):
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

        composite = (0.30 * ker_val + 0.20 * rsi_score + 0.20 * vol_score +
                     0.15 * breakout_score + 0.15 * atr_score)
        ranked.append((ticker, idx, prev_atr, composite))

    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked


def find_candidates(signals_data, active_trades, current_date, is_macro_ok):
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
        if not is_macro_ok:
            continue
        prev_atr = df['ATR'].iloc[idx - 1]
        if pd.isna(prev_atr) or prev_atr <= 0:
            continue
        candidates.append((ticker, idx, prev_atr))
    return candidates


# =============================================================================
# CORE BACKTEST
# =============================================================================

def run_backtest(months, verbose=False):
    tickers = CRYPTO_TICKERS
    n_tickers = len(tickers)
    label = f"CRYPTO v8 (BTC/ETH/SOL)"

    print(f"\n{'='*70}")
    print(f"  {label} -- {months} MESES -- {n_tickers} tickers")
    print(f"  Macro filter: {CONFIG['macro_ticker']} > SMA{CONFIG['macro_sma_period']}")
    print(f"  Max posiciones: {CONFIG['max_positions']}")
    print(f"{'='*70}")

    # Descargar datos
    print("  Descargando datos...")
    all_data = {}
    failed = []
    for ticker in tickers:
        df = download_data(ticker, months)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            all_data[ticker] = df
            print(f"    {ticker}: {len(df)} barras ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
        else:
            failed.append(ticker)
    print(f"  Tickers con datos: {len(all_data)}/{n_tickers}")
    if failed:
        print(f"  Fallidos: {', '.join(failed)}")

    if not all_data:
        print("  ERROR: No hay datos. Abortando.")
        return None

    # Engine + señales
    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG['rsi_max'],
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG['longs_only']
    )
    signals_data, total_signals = generate_all_signals(all_data, engine)

    # Detalle señales por ticker
    print(f"\n  Señales LONG por ticker:")
    for ticker in tickers:
        if ticker in signals_data:
            n = (signals_data[ticker]['signals'] == 1).sum()
            print(f"    {ticker:10} ({CRYPTO_NAMES.get(ticker, '')}): {n} señales")
    print(f"  Total señales: {total_signals}")

    # Macro filter
    macro_bullish = build_macro_filter(all_data)
    if macro_bullish:
        bull_days = sum(1 for v in macro_bullish.values() if v)
        bear_days = sum(1 for v in macro_bullish.values() if not v)
        total_days = bull_days + bear_days
        print(f"\n  Macro filter ({CONFIG['macro_ticker']} > SMA{CONFIG['macro_sma_period']}):")
        print(f"    BULL: {bull_days} dias ({bull_days/total_days*100:.1f}%)")
        print(f"    BEAR: {bear_days} dias ({bear_days/total_days*100:.1f}%)")

    # Timeline
    all_dates = sorted(set(d for sd in signals_data.values() for d in sd['df'].index.tolist()))

    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}
    all_trades = []

    # =================================================================
    # LOOP PRINCIPAL
    # =================================================================
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
            result = trade.update(bar['High'], bar['Low'], bar['Close'], df['ATR'].iloc[idx])
            if result and result['type'] == 'full_exit':
                trade.exit_date = current_date
                trades_to_close.append(ticker)
                tracker.update_equity(trade.pnl_euros, current_date)

        for ticker in trades_to_close:
            trade = active_trades.pop(ticker)
            tracker.open_positions -= 1
            all_trades.append(trade)
            if verbose:
                pnl_pct = (trade.pnl_euros / trade.position_euros * 100) if trade.position_euros else 0
                pnl_sign = '+' if trade.pnl_euros >= 0 else ''
                print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE {ticker:10} | "
                      f"{trade.exit_reason:<15} | P&L EUR {pnl_sign}{trade.pnl_euros:.0f} ({pnl_sign}{pnl_pct:.1f}%) | "
                      f"Pos: {tracker.open_positions}/{CONFIG['max_positions']} | Equity: EUR {tracker.equity:,.0f}")

        # 2. BUSCAR NUEVAS SEÑALES
        if CONFIG['use_macro_filter']:
            prev_dates = [d for d in macro_bullish if d < current_date]
            if len(prev_dates) >= 2:
                is_macro_ok = macro_bullish[prev_dates[-2]]
            elif prev_dates:
                is_macro_ok = macro_bullish[prev_dates[-1]]
            else:
                is_macro_ok = True
        else:
            is_macro_ok = True

        if tracker.open_positions < CONFIG['max_positions'] and is_macro_ok:
            candidates = find_candidates(signals_data, active_trades, current_date, is_macro_ok)
            ranked = rank_candidates(candidates, signals_data)

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= CONFIG['max_positions']:
                    break

                df = signals_data[ticker]['df']
                bar = df.iloc[idx]

                size_info = tracker.get_position_size(ticker, prev_atr, bar['Open'])
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
                    position_units=position_units,
                )
                active_trades[ticker] = trade
                tracker.open_positions += 1

                if verbose:
                    print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN  {ticker:10} | "
                          f"EUR {position_euros:.0f} ({position_units:.6f}u) @ ${entry_price:,.2f} | "
                          f"Score: {composite_score:.2f} | Pos: {tracker.open_positions}/{CONFIG['max_positions']}")

    # Cerrar trades abiertos al final
    for ticker, trade in active_trades.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            trade._close(df['Close'].iloc[-1], 'end_of_data')
            trade.exit_date = df.index[-1]
            tracker.update_equity(trade.pnl_euros, df.index[-1])
            all_trades.append(trade)

    # =================================================================
    # MÉTRICAS
    # =================================================================
    if not all_trades:
        print("\n  ⚠ SIN TRADES — La estrategia no generó ninguna operación.")
        print("  Posibles causas: universo muy reducido, filtro macro bloqueando,")
        print("  o parámetros de señal no se cumplen en el periodo.")
        return None

    total_count = len(all_trades)
    winners = [t for t in all_trades if t.pnl_euros > 0]
    losers = [t for t in all_trades if t.pnl_euros <= 0]

    total_pnl = sum(t.pnl_euros for t in all_trades)
    win_rate = len(winners) / total_count * 100 if total_count > 0 else 0

    gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
    profit_factor = gross_profit / gross_loss

    max_dd = tracker.get_max_drawdown()
    total_return_pct = (tracker.equity / CONFIG['initial_capital'] - 1) * 100
    annualized = ((1 + total_return_pct / 100) ** (12 / months) - 1) * 100 if months > 0 else 0

    avg_win_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0

    stock_gt_3r = sum(1 for t in all_trades if t.max_r_mult >= 3.0)

    best_trade = max(all_trades, key=lambda t: t.pnl_pct)
    worst_trade = min(all_trades, key=lambda t: t.pnl_pct)

    # Avg hold days
    hold_days = []
    for t in all_trades:
        if t.entry_date and t.exit_date:
            hold_days.append((t.exit_date - t.entry_date).days)
    avg_hold = np.mean(hold_days) if hold_days else 0

    print(f"""
{'='*70}
  RESULTADOS {label} -- {months} MESES
{'='*70}

  CAPITAL:
     Inicial:        EUR {CONFIG['initial_capital']:,.2f}
     Final:          EUR {tracker.equity:,.2f}
     P&L Total:      EUR {total_pnl:+,.2f} ({total_return_pct:+.1f}%)
     Anualizado:     {annualized:+.1f}%
     Max Drawdown:   -{max_dd:.1f}%

  TRADES:
     Total:          {total_count}
     Ganadores:      {len(winners)} ({win_rate:.1f}%)
     Perdedores:     {len(losers)}
     Profit Factor:  {profit_factor:.2f}
     Avg Hold:       {avg_hold:.1f} dias

  FAT TAILS:
     Trades >= +3R:  {stock_gt_3r}
     Avg Win:        {avg_win_pct:+.1f}%
     Avg Loss:       {avg_loss_pct:.1f}%
     Best:           {best_trade.ticker} {best_trade.pnl_pct:+.1f}%
     Worst:          {worst_trade.ticker} {worst_trade.pnl_pct:+.1f}%
""")

    # Razones de salida
    exit_reasons = {}
    for t in all_trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    print("  RAZONES DE SALIDA:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        pnl_reason = sum(t.pnl_euros for t in all_trades if t.exit_reason == reason)
        print(f"     {reason:20} {count:3} ({count/total_count*100:.1f}%)  P&L: EUR {pnl_reason:+,.0f}")

    # Detalle por ticker
    print(f"\n  DETALLE POR TICKER:")
    print(f"  {'Ticker':<12} {'Trades':>6} {'Win%':>6} {'P&L EUR':>10} {'Avg Win%':>9} {'Avg Loss%':>10} {'Best%':>8}")
    print(f"  {'-'*65}")
    for ticker in CRYPTO_TICKERS:
        ticker_trades = [t for t in all_trades if t.ticker == ticker]
        if not ticker_trades:
            print(f"  {ticker:<12} {'0':>6} {'—':>6} {'—':>10} {'—':>9} {'—':>10} {'—':>8}")
            continue
        t_wins = [t for t in ticker_trades if t.pnl_euros > 0]
        t_losses = [t for t in ticker_trades if t.pnl_euros <= 0]
        t_pnl = sum(t.pnl_euros for t in ticker_trades)
        t_wr = len(t_wins) / len(ticker_trades) * 100
        t_avg_win = np.mean([t.pnl_pct for t in t_wins]) if t_wins else 0
        t_avg_loss = np.mean([t.pnl_pct for t in t_losses]) if t_losses else 0
        t_best = max(t.pnl_pct for t in ticker_trades)
        print(f"  {ticker:<12} {len(ticker_trades):>6} {t_wr:>5.1f}% {t_pnl:>+10,.0f} {t_avg_win:>+8.1f}% {t_avg_loss:>+9.1f}% {t_best:>+7.1f}%")

    # Lista completa de trades
    if verbose or total_count <= 30:
        print(f"\n  LISTA DE TRADES ({total_count}):")
        print(f"  {'#':>3} {'Entry':>12} {'Exit':>12} {'Ticker':<10} {'Entry$':>10} {'Exit$':>10} "
              f"{'P&L EUR':>9} {'P&L%':>7} {'Days':>5} {'MaxR':>5} {'Razon':<15}")
        print(f"  {'-'*110}")
        for i, t in enumerate(sorted(all_trades, key=lambda x: x.entry_date), 1):
            entry_str = t.entry_date.strftime('%Y-%m-%d') if t.entry_date else '?'
            exit_str = t.exit_date.strftime('%Y-%m-%d') if t.exit_date else '?'
            days = (t.exit_date - t.entry_date).days if t.entry_date and t.exit_date else 0
            print(f"  {i:>3} {entry_str:>12} {exit_str:>12} {t.ticker:<10} "
                  f"${t.entry_price:>9,.2f} ${t.exit_price:>9,.2f} "
                  f"{t.pnl_euros:>+9,.0f} {t.pnl_pct:>+6.1f}% {days:>5} {t.max_r_mult:>5.1f} {t.exit_reason:<15}")

    # Comparativa con v8 referencia (6m)
    if months == 6:
        print(f"""
{'='*70}
  COMPARATIVA: CRYPTO vs v8 ORIGINAL (6 meses)
{'='*70}

  {'Metrica':<25} {'v8 (225 tickers)':>18} {'CRYPTO (3 tickers)':>20}
  {'-'*65}
  {'Return':.<25} {'+27.6%':>18} {f'{total_return_pct:+.1f}%':>20}
  {'Anualizado':.<25} {'+62.9%':>18} {f'{annualized:+.1f}%':>20}
  {'Profit Factor':.<25} {'4.74':>18} {f'{profit_factor:.2f}':>20}
  {'Max Drawdown':.<25} {'-3.2%':>18} {f'-{max_dd:.1f}%':>20}
  {'Win Rate':.<25} {'55.9%':>18} {f'{win_rate:.1f}%':>20}
  {'Trades':.<25} {'34':>18} {f'{total_count}':>20}
""")

    return {
        'label': label,
        'total_trades': total_count,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'total_pnl_euros': total_pnl,
        'total_return_pct': total_return_pct,
        'annualized_return_pct': annualized,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'avg_hold_days': avg_hold,
        'stock_gt_3r': stock_gt_3r,
        'best_ticker': best_trade.ticker,
        'best_pnl_pct': best_trade.pnl_pct,
        'final_equity': tracker.equity,
        'all_trades': all_trades,
    }


# =============================================================================
# VARIANTE SIN MACRO FILTER (para comparar)
# =============================================================================

def run_no_macro(months, verbose=False):
    """Misma estrategia pero sin filtro macro (siempre opera)."""
    original = CONFIG['use_macro_filter']
    CONFIG['use_macro_filter'] = False
    print(f"\n  --- VARIANTE: SIN MACRO FILTER ---")
    result = run_backtest(months, verbose)
    CONFIG['use_macro_filter'] = original
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Backtest Crypto: Momentum Breakout v8 para BTC/ETH/SOL')
    parser.add_argument('--months', type=int, default=6, help='Meses de historico (default: 6)')
    parser.add_argument('--verbose', action='store_true', help='Detalle de trades')
    parser.add_argument('--no-macro', action='store_true', help='Tambien testear sin filtro macro')
    args = parser.parse_args()

    print(f"""
======================================================================
  BACKTEST CRYPTO — MOMENTUM BREAKOUT v8
  Universo: BTC-USD, ETH-USD, SOL-USD
  Macro filter: BTC > SMA50
  Config: mismos parametros v8 (KER 0.40, RSI 50-75, Vol 1.3x, BP 20)
  Max posiciones: 3
======================================================================
    """)

    r1 = run_backtest(args.months, args.verbose)

    if args.no_macro:
        r2 = run_no_macro(args.months, args.verbose)

        if r1 and r2:
            print(f"""
{'='*70}
  COMPARATIVA: CON vs SIN MACRO FILTER
{'='*70}

  {'Metrica':<25} {'Con Macro':>15} {'Sin Macro':>15}
  {'-'*55}
  {'Return':.<25} {f"{r1['total_return_pct']:+.1f}%":>15} {f"{r2['total_return_pct']:+.1f}%":>15}
  {'Anualizado':.<25} {f"{r1['annualized_return_pct']:+.1f}%":>15} {f"{r2['annualized_return_pct']:+.1f}%":>15}
  {'Profit Factor':.<25} {f"{r1['profit_factor']:.2f}":>15} {f"{r2['profit_factor']:.2f}":>15}
  {'Max Drawdown':.<25} {f"-{r1['max_drawdown']:.1f}%":>15} {f"-{r2['max_drawdown']:.1f}%":>15}
  {'Win Rate':.<25} {f"{r1['win_rate']:.1f}%":>15} {f"{r2['win_rate']:.1f}%":>15}
  {'Trades':.<25} {f"{r1['total_trades']}":>15} {f"{r2['total_trades']}":>15}
""")


if __name__ == '__main__':
    main()
