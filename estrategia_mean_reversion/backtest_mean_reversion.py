#!/usr/bin/env python3
"""
MEAN REVERSION BACKTEST — Nasdaq 100
=====================================

Concepto: Comprar el stock más sobrevendido (por RSI) del Nasdaq 100 cada día al cierre.
Al día siguiente: vender a +X% (target), -Y% (stop), o al cierre si ninguno se activa.

Origen: Tweet que afirma +86% en 2.5 meses con 7 stocks del SP500 (RSI10, ±3%).
Tesis: El resultado original es overfitting masivo (170K combinaciones, ventana de 2.5 meses).
Este backtest prueba el concepto de forma honesta con universo completo y periodo largo.

NOTA: Survivorship bias presente — Nasdaq 100 es la composición actual (marzo 2026).
Stocks que salieron del índice en los últimos años no están incluidos.

Creado: 14-mar-2026
"""

import argparse
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    'initial_capital': 10000,

    # Strategy params (defaults — overridden per variant)
    'rsi_period': 10,
    'rsi_threshold': 30,
    'target_pct': 3.0,
    'stop_pct': 3.0,
    'max_positions': 1,

    # Risk / costs
    'position_size_pct': 10.0,      # % de equity por posición
    'slippage_pct': 0.10,           # 0.10% por trade (entry + exit)
    'commission_per_trade': 1.0,    # EUR por ejecución

    # Macro filter
    'use_macro_filter': False,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,

    # Backtest
    'months': 60,

    # Ambigüedad intraday
    'intraday_bias': 'worst_case',  # 'worst_case', 'best_case', 'open_bias'
}


# ═══════════════════════════════════════════════════════════════════
# NASDAQ 100 TICKERS (marzo 2026 — verificar antes de ejecutar)
# ═══════════════════════════════════════════════════════════════════

NASDAQ_100 = [
    'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
    'AMZN', 'ANSS', 'APP', 'ARM', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR',
    'CCEP', 'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO',
    'CSGP', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG',
    'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN',
    'INTC', 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR',
    'MCHP', 'MDB', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU',
    'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD',
    'PEP', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SMCI', 'SNPS', 'TEAM',
    'TMUS', 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS',
]


# ═══════════════════════════════════════════════════════════════════
# TRADE DATACLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MeanReversionTrade:
    ticker: str
    entry_price: float
    entry_date: datetime
    entry_rsi: float
    position_euros: float
    position_units: float
    direction: int = 1  # +1 = long, -1 = short

    exit_price: Optional[float] = field(default=None)
    exit_date: Optional[datetime] = field(default=None)
    exit_reason: Optional[str] = field(default=None)
    pnl_euros: float = field(default=0.0)
    pnl_pct: float = field(default=0.0)
    bars_held: int = field(default=1)

    def close(self, exit_price, reason, exit_date, slippage_pct, commission):
        self.exit_price = exit_price
        self.exit_reason = reason
        self.exit_date = exit_date
        gross_pnl = (exit_price - self.entry_price) * self.position_units * self.direction
        slippage_cost = self.position_euros * slippage_pct / 100 * 2  # entry + exit
        commission_cost = commission * 2  # entry + exit
        self.pnl_euros = gross_pnl - slippage_cost - commission_cost
        self.pnl_pct = (self.pnl_euros / self.position_euros) * 100 if self.position_euros > 0 else 0


# ═══════════════════════════════════════════════════════════════════
# EQUITY TRACKER
# ═══════════════════════════════════════════════════════════════════

class EquityTracker:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.max_equity = initial_capital

    def get_position_size(self, n_positions):
        pct = CONFIG['position_size_pct'] / 100
        return self.equity * pct / max(n_positions, 1)

    def update_equity(self, pnl, date):
        self.equity += pnl
        self.equity_curve.append((date, self.equity))
        self.max_equity = max(self.max_equity, self.equity)

    def record_date(self, date):
        """Registra equity en días sin trades."""
        if not self.equity_curve or self.equity_curve[-1][0] != date:
            self.equity_curve.append((date, self.equity))

    def get_max_drawdown(self):
        if not self.equity_curve:
            return 0.0
        peak = self.initial_capital
        max_dd = 0.0
        for _, eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (eq - peak) / peak * 100 if peak > 0 else 0
            max_dd = min(max_dd, dd)
        return max_dd


# ═══════════════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def download_data(ticker, months):
    """Descarga datos OHLC de yfinance."""
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


def download_all_data(tickers, months):
    """Descarga todos los tickers con progreso."""
    all_data = {}
    failed = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        if i % 20 == 0 or i == total:
            print(f"  Descargando... {i}/{total}")
        df = download_data(ticker, months)
        if df is not None:
            all_data[ticker] = df
        else:
            failed.append(ticker)
    if failed:
        print(f"  ⚠ Sin datos: {', '.join(failed)}")
    print(f"  ✓ {len(all_data)} tickers descargados correctamente")
    return all_data


def calculate_rsi(close, period=14):
    """RSI (SMA-based)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_rsi_for_all(all_data, rsi_period):
    """Pre-calcula RSI para todos los tickers."""
    rsi_data = {}
    for ticker, df in all_data.items():
        rsi_data[ticker] = calculate_rsi(df['Close'], rsi_period)
    return rsi_data


# ═══════════════════════════════════════════════════════════════════
# STRATEGY LOGIC
# ═══════════════════════════════════════════════════════════════════

def resolve_intraday_exit(open_price, high, low, close, entry_price, target_pct, stop_pct, bias='worst_case', direction=1):
    """
    Determina precio y razón de salida a partir del OHLC del día siguiente.

    direction=+1 (LONG): target = sube, stop = baja
    direction=-1 (SHORT): target = baja, stop = sube

    Para caso ambiguo: bias determina resolución.
    """
    if direction == 1:  # LONG
        target_price = entry_price * (1 + target_pct / 100)
        stop_price = entry_price * (1 - stop_pct / 100)
        hit_target = high >= target_price
        hit_stop = low <= stop_price
    else:  # SHORT
        target_price = entry_price * (1 - target_pct / 100)  # precio baja = ganancia
        stop_price = entry_price * (1 + stop_pct / 100)      # precio sube = pérdida
        hit_target = low <= target_price
        hit_stop = high >= stop_price

    if hit_target and hit_stop:
        if bias == 'worst_case':
            return stop_price, 'stop_ambiguous'
        elif bias == 'best_case':
            return target_price, 'target_ambiguous'
        elif bias == 'open_bias':
            dist_to_target = abs(target_price - open_price)
            dist_to_stop = abs(open_price - stop_price)
            if dist_to_stop < dist_to_target:
                return stop_price, 'stop_ambiguous'
            else:
                return target_price, 'target_ambiguous'
        else:
            return stop_price, 'stop_ambiguous'
    elif hit_target:
        return target_price, 'target'
    elif hit_stop:
        return stop_price, 'stop'
    else:
        return close, 'close'


def get_candidates(all_data, rsi_data, date, rsi_threshold, active_tickers, mode='oversold'):
    """
    Encuentra stocks según RSI en la fecha dada.
    mode='oversold': RSI < umbral, ordenado ascendente (más sobrevendido primero)
    mode='overbought': RSI > umbral, ordenado descendente (más sobrecomprado primero)
    """
    candidates = []
    for ticker in rsi_data:
        if ticker in active_tickers:
            continue
        df = all_data[ticker]
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        rsi_val = rsi_data[ticker].iloc[idx]
        if np.isnan(rsi_val):
            continue
        if mode == 'oversold' and rsi_val >= rsi_threshold:
            continue
        if mode == 'overbought' and rsi_val <= rsi_threshold:
            continue
        close = df.iloc[idx]['Close']
        if close <= 0 or np.isnan(close):
            continue
        candidates.append((ticker, rsi_val))
    # Oversold: ascendente (RSI más bajo primero). Overbought: descendente (RSI más alto primero)
    candidates.sort(key=lambda x: x[1], reverse=(mode == 'overbought'))
    return candidates


def build_macro_filter(all_data, macro_ticker='SPY', sma_period=50):
    """Construye filtro macro: SPY > SMA(50)."""
    if macro_ticker not in all_data:
        return {}
    df = all_data[macro_ticker]
    sma = df['Close'].rolling(window=sma_period).mean()
    bullish = df['Close'] > sma
    macro = {}
    last_val = True
    for date in df.index:
        val = bullish.get(date)
        if pd.isna(val):
            macro[date] = last_val
        else:
            macro[date] = bool(val)
            last_val = bool(val)
    return macro


def build_benchmark_qqq(all_data, months):
    """QQQ buy-and-hold como benchmark."""
    if 'QQQ' not in all_data:
        df = download_data('QQQ', months)
        if df is None:
            return None, None, None
    else:
        df = all_data['QQQ']

    if df is None or len(df) < 50:
        return None, None, None

    initial = CONFIG['initial_capital']
    returns = df['Close'].pct_change().fillna(0)
    equity = initial * (1 + returns).cumprod()
    final_equity = equity.iloc[-1]

    total_return = (final_equity / initial - 1) * 100
    years = months / 12.0
    cagr = ((final_equity / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    dd = ((equity - peak) / peak * 100)
    max_dd = dd.min()

    # Sharpe
    daily_ret = returns
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (cagr / 100) / vol if vol > 0 else 0

    return cagr, max_dd, sharpe


# ═══════════════════════════════════════════════════════════════════
# CORE BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_single_backtest(all_data, rsi_data, variant_config, label, months):
    """Ejecuta un backtest completo para una variante."""
    target_pct = variant_config['target_pct']
    stop_pct = variant_config['stop_pct']
    rsi_threshold = variant_config['rsi_threshold']
    max_positions = variant_config['max_positions']
    max_hold_days = variant_config.get('max_hold_days', 1)
    bias = variant_config.get('intraday_bias', 'worst_case')
    use_macro = variant_config.get('use_macro_filter', False)
    direction = variant_config.get('direction', 1)  # +1 long, -1 short
    rsi_mode = variant_config.get('rsi_mode', 'oversold')

    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}  # ticker -> MeanReversionTrade
    closed_trades = []

    # Macro filter
    macro_filter = {}
    if use_macro:
        macro_filter = build_macro_filter(all_data, CONFIG['macro_ticker'], CONFIG['macro_sma_period'])

    # Timeline: unión de todas las fechas
    all_dates = sorted(set(d for df in all_data.values() for d in df.index))

    for i, date in enumerate(all_dates):
        # ── FASE 1: Gestionar posiciones activas ──
        tickers_to_close = list(active_trades.keys())
        for ticker in tickers_to_close:
            trade = active_trades[ticker]
            df = all_data[ticker]
            if date not in df.index:
                continue
            row = df.loc[date]
            trade.bars_held += 1

            # Comprobar target/stop intraday
            exit_price, exit_reason = resolve_intraday_exit(
                row['Open'], row['High'], row['Low'], row['Close'],
                trade.entry_price, target_pct, stop_pct, bias, direction
            )

            if exit_reason in ('target', 'stop', 'target_ambiguous', 'stop_ambiguous'):
                # Target o stop activado → cerrar
                trade.close(exit_price, exit_reason, date,
                            CONFIG['slippage_pct'], CONFIG['commission_per_trade'])
                tracker.update_equity(trade.pnl_euros, date)
                closed_trades.append(trade)
                del active_trades[ticker]
            elif trade.bars_held >= max_hold_days:
                # Tiempo máximo → cerrar al cierre
                trade.close(row['Close'], 'time_exit', date,
                            CONFIG['slippage_pct'], CONFIG['commission_per_trade'])
                tracker.update_equity(trade.pnl_euros, date)
                closed_trades.append(trade)
                del active_trades[ticker]
            # else: mantener posición abierta

        # ── FASE 2: Nuevas entradas al cierre de hoy ──
        if use_macro and not macro_filter.get(date, True):
            tracker.record_date(date)
            continue

        active_tickers = set(active_trades.keys())
        candidates = get_candidates(all_data, rsi_data, date, rsi_threshold, active_tickers, rsi_mode)

        n_to_buy = min(max_positions - len(active_trades), len(candidates))
        for ticker, rsi_val in candidates[:n_to_buy]:
            df = all_data[ticker]
            entry_price = df.loc[date, 'Close']
            notional = tracker.get_position_size(max_positions)
            if notional < 50:  # Mínimo EUR 50 por posición
                continue
            units = notional / entry_price
            trade = MeanReversionTrade(
                ticker=ticker,
                entry_price=entry_price,
                entry_date=date,
                entry_rsi=rsi_val,
                position_euros=notional,
                position_units=units,
                direction=direction,
            )
            active_trades[ticker] = trade

        tracker.record_date(date)

    # Cerrar trades huérfanos al final
    for ticker, trade in active_trades.items():
        df = all_data[ticker]
        if len(df) > 0:
            last_date = df.index[-1]
            last_close = df.iloc[-1]['Close']
            trade.close(last_close, 'end_of_data', last_date,
                        CONFIG['slippage_pct'], CONFIG['commission_per_trade'])
            tracker.update_equity(trade.pnl_euros, last_date)
            closed_trades.append(trade)

    return compute_metrics(tracker, closed_trades, months, label)


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(tracker, all_trades, months, label):
    """Calcula métricas de rendimiento."""
    total = len(all_trades)
    if total == 0:
        return {
            'label': label, 'total_trades': 0, 'win_rate': 0, 'total_pnl_euros': 0,
            'total_return_pct': 0, 'cagr': 0, 'profit_factor': 0, 'max_drawdown': 0,
            'sharpe': 0, 'avg_win_pct': 0, 'avg_loss_pct': 0, 'n_ambiguous': 0,
            'final_equity': tracker.equity, 'equity_curve': tracker.equity_curve,
            'all_trades': all_trades,
        }

    winners = [t for t in all_trades if t.pnl_euros > 0]
    losers = [t for t in all_trades if t.pnl_euros <= 0]

    total_pnl = sum(t.pnl_euros for t in all_trades)
    win_rate = len(winners) / total * 100

    gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
    profit_factor = gross_profit / gross_loss

    max_dd = tracker.get_max_drawdown()
    total_return_pct = (tracker.equity / CONFIG['initial_capital'] - 1) * 100
    years = months / 12.0
    cagr = ((tracker.equity / CONFIG['initial_capital']) ** (1 / years) - 1) * 100 if years > 0 and tracker.equity > 0 else 0

    # Sharpe desde equity curve
    if len(tracker.equity_curve) > 10:
        eq_series = pd.Series(
            [e[1] for e in tracker.equity_curve],
            index=[e[0] for e in tracker.equity_curve]
        )
        daily_ret = eq_series.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = (cagr / 100) / vol if vol > 0 else 0
    else:
        sharpe = 0

    n_ambiguous = sum(1 for t in all_trades if 'ambiguous' in (t.exit_reason or ''))

    # Exit reasons breakdown
    exit_reasons = {}
    for t in all_trades:
        r = t.exit_reason or 'unknown'
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        'label': label,
        'total_trades': total,
        'win_rate': win_rate,
        'total_pnl_euros': total_pnl,
        'total_return_pct': total_return_pct,
        'cagr': cagr,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'avg_win_pct': np.mean([t.pnl_pct for t in winners]) if winners else 0,
        'avg_loss_pct': np.mean([t.pnl_pct for t in losers]) if losers else 0,
        'avg_days': np.mean([t.bars_held for t in all_trades]) if all_trades else 0,
        'n_ambiguous': n_ambiguous,
        'exit_reasons': exit_reasons,
        'final_equity': tracker.equity,
        'equity_curve': tracker.equity_curve,
        'all_trades': all_trades,
    }


# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════

def print_comparison(results, months, bias, qqq_metrics, dir_label='LONG'):
    """Imprime tabla comparativa de todas las variantes."""
    n_tickers = len(NASDAQ_100)
    cap = CONFIG['initial_capital']

    print(f"\n{'='*100}")
    print(f"  MEAN REVERSION BACKTEST — NASDAQ 100 — {months} MESES — {dir_label}")
    print(f"{'='*100}")
    print(f"  Universo: {n_tickers} tickers | Capital: EUR {cap:,.0f} | Bias: {bias}")
    print(f"  Slippage: {CONFIG['slippage_pct']:.2f}% | Comision: EUR {CONFIG['commission_per_trade']:.2f}/trade")
    print(f"  Position size: {CONFIG['position_size_pct']:.0f}% del equity por posicion")
    print(f"{'='*100}\n")

    # Ordenar por CAGR descendente
    sorted_results = sorted(results, key=lambda r: r['cagr'], reverse=True)

    header = (f"  {'Variante':<35} {'Trades':>6} {'Win%':>6} {'PnL EUR':>10} "
              f"{'CAGR':>7} {'MaxDD':>7} {'PF':>6} {'Sharpe':>7} {'AvgD':>5}")
    print(header)
    print(f"  {'-'*99}")

    for r in sorted_results:
        avg_d = r.get('avg_days', 1)
        print(f"  {r['label']:<35} {r['total_trades']:>6} {r['win_rate']:>5.1f}% "
              f"EUR{r['total_pnl_euros']:>+8,.0f} {r['cagr']:>+6.1f}% {r['max_drawdown']:>6.1f}% "
              f"{r['profit_factor']:>5.2f} {r['sharpe']:>6.2f} {avg_d:>5.1f}")

    # Benchmark
    if qqq_metrics:
        qqq_cagr, qqq_dd, qqq_sharpe = qqq_metrics
        print(f"  {'-'*99}")
        print(f"  {'QQQ Buy&Hold':<35} {'—':>6} {'—':>6} {'—':>10} "
              f"{qqq_cagr:>+6.1f}% {qqq_dd:>6.1f}% {'—':>5} {qqq_sharpe:>6.2f} {'—':>5}")

    print(f"  {'-'*99}\n")

    # Detalle de la mejor variante
    if sorted_results and sorted_results[0]['total_trades'] > 0:
        best = sorted_results[0]
        print(f"  MEJOR VARIANTE: {best['label']}")
        print(f"  {'─'*60}")
        print(f"    Capital inicial:   EUR {CONFIG['initial_capital']:>10,.2f}")
        print(f"    Capital final:     EUR {best['final_equity']:>10,.2f}")
        print(f"    PnL total:         EUR {best['total_pnl_euros']:>+10,.2f}")
        print(f"    CAGR:              {best['cagr']:>+.2f}%")
        print(f"    Avg win:           {best['avg_win_pct']:>+.2f}%")
        print(f"    Avg loss:          {best['avg_loss_pct']:>+.2f}%")
        print()

        # Exit reasons
        print(f"  RAZONES DE SALIDA:")
        total_t = best['total_trades']
        for reason, count in sorted(best['exit_reasons'].items(), key=lambda x: -x[1]):
            pct = count / total_t * 100
            print(f"    {reason:<20} {count:>5} ({pct:>5.1f}%)")
        print()


def export_csv(trades, filename):
    """Exporta trades a CSV."""
    rows = []
    for t in trades:
        rows.append({
            'Entry_Date': t.entry_date.strftime('%Y-%m-%d') if hasattr(t.entry_date, 'strftime') else str(t.entry_date),
            'Exit_Date': t.exit_date.strftime('%Y-%m-%d') if t.exit_date and hasattr(t.exit_date, 'strftime') else str(t.exit_date),
            'Ticker': t.ticker,
            'Entry_RSI': round(t.entry_rsi, 1),
            'Entry_Price': round(t.entry_price, 2),
            'Exit_Price': round(t.exit_price, 2) if t.exit_price else None,
            'Position_EUR': round(t.position_euros, 2),
            'PnL_EUR': round(t.pnl_euros, 2),
            'PnL_Pct': round(t.pnl_pct, 2),
            'Exit_Reason': t.exit_reason,
        })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"  CSV exportado: {filename} ({len(rows)} trades)")


# ═══════════════════════════════════════════════════════════════════
# VARIANT GRID
# ═══════════════════════════════════════════════════════════════════

TARGET_STOP_PAIRS = [(3.0, 3.0), (2.0, 2.0), (3.0, 2.0)]
TARGET_STOP_PAIRS_MULTIDAY = [(3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (7.0, 5.0)]
RSI_PERIODS = [10, 14]
RSI_THRESHOLDS_OVERSOLD = [20, 30, 40]
RSI_THRESHOLDS_OVERBOUGHT = [80, 85, 90]
RSI_THRESHOLDS_WIDE = [60, 65, 70, 75, 80, 85, 90]
MAX_POSITIONS_LIST = [1, 3]
HOLD_DAYS_LIST = [1, 3, 5, 10]


def run_all_variants(all_data, months, bias, use_macro, direction=1, rsi_mode='oversold'):
    """Ejecuta grid de variantes."""
    results = []
    rsi_thresholds = RSI_THRESHOLDS_OVERBOUGHT if rsi_mode == 'overbought' else RSI_THRESHOLDS_OVERSOLD
    ts_pairs = TARGET_STOP_PAIRS_MULTIDAY if rsi_mode == 'overbought' else TARGET_STOP_PAIRS
    hold_days_list = HOLD_DAYS_LIST if rsi_mode == 'overbought' else [1]
    positions_list = [1] if rsi_mode == 'overbought' else MAX_POSITIONS_LIST

    total_variants = (len(RSI_PERIODS) * len(ts_pairs) * len(rsi_thresholds)
                      * len(positions_list) * len(hold_days_list))
    count = 0
    op = '>' if rsi_mode == 'overbought' else '<'

    for rsi_period in RSI_PERIODS:
        print(f"\n  Calculando RSI({rsi_period}) para todos los tickers...")
        rsi_data = compute_rsi_for_all(all_data, rsi_period)

        for target_pct, stop_pct in ts_pairs:
            for rsi_threshold in rsi_thresholds:
                for max_positions in positions_list:
                    for max_hold_days in hold_days_list:
                        count += 1
                        hold_str = f" D{max_hold_days}" if max_hold_days > 1 else ""
                        label = f"RSI{rsi_period} T{target_pct:.0f}/S{stop_pct:.0f} {op}{rsi_threshold}{hold_str} N={max_positions}"

                        variant_config = {
                            'rsi_period': rsi_period,
                            'target_pct': target_pct,
                            'stop_pct': stop_pct,
                            'rsi_threshold': rsi_threshold,
                            'max_positions': max_positions,
                            'max_hold_days': max_hold_days,
                            'intraday_bias': bias,
                            'use_macro_filter': use_macro,
                            'direction': direction,
                            'rsi_mode': rsi_mode,
                        }

                        result = run_single_backtest(all_data, rsi_data, variant_config, label, months)
                        results.append(result)

                        if count % 12 == 0 or count == total_variants:
                            print(f"  Variante {count}/{total_variants} completada...")

    return results


def run_long_overbought_grid(all_data, months, bias, use_macro):
    """Grid LONG con RSI14 fijo, umbrales RSI desde 60, hold D5/D10."""
    results = []
    rsi_period = 14
    ts_pairs = TARGET_STOP_PAIRS_MULTIDAY
    rsi_thresholds = RSI_THRESHOLDS_WIDE
    hold_days_list = [5, 10]  # D1 y D3 siempre pierden

    total = len(ts_pairs) * len(rsi_thresholds) * len(hold_days_list)
    count = 0

    print(f"  RSI period: 14 | Dirección: LONG | Umbrales: {rsi_thresholds}")
    print(f"  T/S pairs: {ts_pairs} | Hold days: {hold_days_list}")
    print(f"  Total variantes: {total}")
    print(f"\n  Calculando RSI(14) para todos los tickers...")
    rsi_data = compute_rsi_for_all(all_data, rsi_period)

    for target_pct, stop_pct in ts_pairs:
        for rsi_threshold in rsi_thresholds:
            for max_hold_days in hold_days_list:
                count += 1
                label = f"L T{target_pct:.0f}/S{stop_pct:.0f} >{rsi_threshold} D{max_hold_days}"

                variant_config = {
                    'rsi_period': rsi_period,
                    'target_pct': target_pct,
                    'stop_pct': stop_pct,
                    'rsi_threshold': rsi_threshold,
                    'max_positions': 1,
                    'max_hold_days': max_hold_days,
                    'intraday_bias': bias,
                    'use_macro_filter': use_macro,
                    'direction': 1,
                    'rsi_mode': 'overbought',
                }

                result = run_single_backtest(all_data, rsi_data, variant_config, label, months)
                results.append(result)

                if count % 14 == 0 or count == total:
                    print(f"  Variante {count}/{total} completada...")

    return results


def run_bidirectional_grid(all_data, months, bias, use_macro):
    """Grid LONG + SHORT con RSI14 fijo, múltiples targets, hold days y thresholds."""
    results = []
    rsi_period = 14
    directions = [(1, 'L'), (-1, 'S')]  # Long y Short
    ts_pairs = TARGET_STOP_PAIRS_MULTIDAY
    rsi_thresholds = RSI_THRESHOLDS_OVERBOUGHT
    hold_days_list = HOLD_DAYS_LIST

    total = len(directions) * len(ts_pairs) * len(rsi_thresholds) * len(hold_days_list)
    count = 0

    print(f"  RSI period: 14 | Direcciones: LONG + SHORT | Variantes: {total}")
    print(f"\n  Calculando RSI(14) para todos los tickers...")
    rsi_data = compute_rsi_for_all(all_data, rsi_period)

    for direction, dir_char in directions:
        for target_pct, stop_pct in ts_pairs:
            for rsi_threshold in rsi_thresholds:
                for max_hold_days in hold_days_list:
                    count += 1
                    hold_str = f" D{max_hold_days}" if max_hold_days > 1 else ""
                    label = f"{dir_char} T{target_pct:.0f}/S{stop_pct:.0f} >{rsi_threshold}{hold_str}"

                    variant_config = {
                        'rsi_period': rsi_period,
                        'target_pct': target_pct,
                        'stop_pct': stop_pct,
                        'rsi_threshold': rsi_threshold,
                        'max_positions': 1,
                        'max_hold_days': max_hold_days,
                        'intraday_bias': bias,
                        'use_macro_filter': use_macro,
                        'direction': direction,
                        'rsi_mode': 'overbought',
                    }

                    result = run_single_backtest(all_data, rsi_data, variant_config, label, months)
                    results.append(result)

                    if count % 12 == 0 or count == total:
                        print(f"  Variante {count}/{total} completada...")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Mean Reversion Backtest — Nasdaq 100')
    parser.add_argument('--months', type=int, default=60, help='Meses de historico (default: 60)')
    parser.add_argument('--bias', choices=['worst_case', 'best_case', 'open_bias'],
                        default='worst_case', help='Resolucion de ambiguedad intraday (default: worst_case)')
    parser.add_argument('--verbose', action='store_true', help='Detalle de cada trade')
    parser.add_argument('--export-csv', action='store_true', help='Exportar trades de mejor variante a CSV')
    parser.add_argument('--single', action='store_true',
                        help='Solo ejecutar CONFIG default, sin grid de variantes')
    parser.add_argument('--macro', action='store_true', help='Activar filtro macro SPY > SMA50')
    parser.add_argument('--short', action='store_true', help='Ponerse corto en oversold (en vez de comprar)')
    parser.add_argument('--overbought', action='store_true',
                        help='Short en sobrecompra (RSI>80/85/90) en vez de oversold')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Grid LONG+SHORT con RSI14 fijo, multiples targets y hold days')
    parser.add_argument('--long-overbought', action='store_true',
                        help='Grid LONG RSI14, umbrales RSI desde 60, hold D5/D10')
    args = parser.parse_args()

    months = args.months
    bias = args.bias

    if args.long_overbought:
        direction = 1
        rsi_mode = 'overbought'
        dir_label = "LONG OVERBOUGHT (RSI>=60)"
    elif args.bidirectional:
        direction = 1  # se ignora, el grid prueba ambas
        rsi_mode = 'overbought'
        dir_label = "BIDIRECTIONAL OVERBOUGHT"
    elif args.overbought:
        direction = -1
        rsi_mode = 'overbought'
        dir_label = "SHORT OVERBOUGHT"
    elif args.short:
        direction = -1
        rsi_mode = 'oversold'
        dir_label = "SHORT OVERSOLD"
    else:
        direction = 1
        rsi_mode = 'oversold'
        dir_label = "LONG OVERSOLD"

    print(f"\n{'='*100}")
    print(f"  MEAN REVERSION BACKTEST — NASDAQ 100 — {dir_label}")
    print(f"{'='*100}")
    print(f"  Descargando {len(NASDAQ_100)} tickers + SPY + QQQ ({months} meses)...\n")

    # Descargar datos (una sola vez)
    tickers_to_download = list(set(NASDAQ_100 + ['SPY', 'QQQ']))
    all_data = download_all_data(tickers_to_download, months)

    if len(all_data) < 20:
        print("\n  ERROR: Demasiados tickers sin datos. Abortando.")
        return

    # Benchmark QQQ
    qqq_metrics = build_benchmark_qqq(all_data, months)

    if args.single:
        # Solo CONFIG default
        print(f"\n  Ejecutando variante única (RSI{CONFIG['rsi_period']} T{CONFIG['target_pct']}/S{CONFIG['stop_pct']} "
              f"RSI<{CONFIG['rsi_threshold']} N={CONFIG['max_positions']})...")
        rsi_data = compute_rsi_for_all(all_data, CONFIG['rsi_period'])
        variant_config = {
            'rsi_period': CONFIG['rsi_period'],
            'target_pct': CONFIG['target_pct'],
            'stop_pct': CONFIG['stop_pct'],
            'rsi_threshold': CONFIG['rsi_threshold'],
            'max_positions': CONFIG['max_positions'],
            'intraday_bias': bias,
            'use_macro_filter': args.macro,
            'direction': direction,
            'rsi_mode': rsi_mode,
        }
        op = '>' if rsi_mode == 'overbought' else '<'
        label = f"RSI{CONFIG['rsi_period']} T{CONFIG['target_pct']:.0f}/S{CONFIG['stop_pct']:.0f} {op}{CONFIG['rsi_threshold']} N={CONFIG['max_positions']}"
        result = run_single_backtest(all_data, rsi_data, variant_config, label, months)
        results = [result]
    elif args.long_overbought:
        # Grid LONG overbought: RSI14, umbrales desde 60, hold D5/D10
        print(f"\n  Ejecutando grid LONG overbought (RSI14, umbrales 60-90)...")
        results = run_long_overbought_grid(all_data, months, bias, args.macro)
    elif args.bidirectional:
        # Grid bidireccional: LONG + SHORT con RSI14, multiples targets/hold
        print(f"\n  Ejecutando grid bidireccional (LONG + SHORT, RSI14 fijo)...")
        results = run_bidirectional_grid(all_data, months, bias, args.macro)
    else:
        # Grid completo
        rsi_thresholds = RSI_THRESHOLDS_OVERBOUGHT if rsi_mode == 'overbought' else RSI_THRESHOLDS_OVERSOLD
        ts_pairs = TARGET_STOP_PAIRS_MULTIDAY if rsi_mode == 'overbought' else TARGET_STOP_PAIRS
        hold_days = HOLD_DAYS_LIST if rsi_mode == 'overbought' else [1]
        positions = [1] if rsi_mode == 'overbought' else MAX_POSITIONS_LIST
        n_variants = len(RSI_PERIODS) * len(ts_pairs) * len(rsi_thresholds) * len(positions) * len(hold_days)
        print(f"\n  Ejecutando grid de {n_variants} combinaciones...")
        results = run_all_variants(all_data, months, bias, args.macro, direction, rsi_mode)

    # Output
    print_comparison(results, months, bias, qqq_metrics, dir_label)

    # Verbose: detalle de trades de la mejor variante
    best = max(results, key=lambda r: r['cagr'])
    if args.verbose and best['all_trades']:
        print(f"\n  DETALLE DE TRADES — {best['label']}")
        print(f"  {'─'*90}")
        print(f"  {'Fecha':<12} {'Ticker':<7} {'RSI':>5} {'Entry':>8} {'Exit':>8} {'PnL EUR':>9} {'PnL%':>7} {'Razon':<18}")
        print(f"  {'─'*90}")
        for t in best['all_trades'][:50]:  # Primeros 50
            entry_str = t.entry_date.strftime('%Y-%m-%d') if hasattr(t.entry_date, 'strftime') else str(t.entry_date)[:10]
            print(f"  {entry_str:<12} {t.ticker:<7} {t.entry_rsi:>5.1f} {t.entry_price:>8.2f} "
                  f"{t.exit_price:>8.2f} EUR{t.pnl_euros:>+7.2f} {t.pnl_pct:>+6.2f}% {t.exit_reason:<18}")
        if len(best['all_trades']) > 50:
            print(f"  ... ({len(best['all_trades']) - 50} trades más)")
        print()

    # Export CSV
    if args.export_csv and best['all_trades']:
        filename = f"mean_reversion_trades_{months}m.csv"
        export_csv(best['all_trades'], filename)

    print(f"  Completado.\n")


if __name__ == '__main__':
    main()
