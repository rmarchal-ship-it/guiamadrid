#!/usr/bin/env python3
"""
TEST DE VARIANTES DE TIME EXIT — Momentum Breakout v6+

Analiza el impacto de distintas configuraciones de time exit
sobre el rendimiento a 240 meses (20 anos).

Variantes a probar:
  V0 (actual):  max_hold_bars=12, si price<=entry -> forzar salida
  V1:           max_hold_bars=20 (dar mas tiempo)
  V2:           max_hold_bars=15 (intermedio)
  V3:           Eliminar time exit forzado, siempre activar trailing a breakeven
  V4:           max_hold_bars=12 pero usar trailing tighter (2xATR) en vez de salida forzada
  V5:           max_hold_bars=8, pero solo activar trailing, nunca forzar
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_breakout import MomentumEngine, calculate_atr, ASSETS

from backtest_experimental import (
    CONFIG, OPTIONS_ELIGIBLE, LEVERAGE_FACTORS,
    OptionTradeV2, EquityTracker,
    download_data, historical_volatility, iv_rank,
    generate_all_signals, build_macro_filter, rank_candidates,
    find_candidates, black_scholes_call, monthly_expiration_dte,
    BASE_TICKERS,
)


# =============================================================================
# TRADE CLASS MODIFICABLE (para probar variantes de time exit)
# =============================================================================

@dataclass
class TradeVariant:
    """Trade con time exit configurable."""
    ticker: str
    entry_price: float
    entry_date: datetime
    entry_atr: float
    position_euros: float
    position_units: float

    # Config de time exit (inyectable)
    time_exit_mode: str = 'original'  # 'original', 'trailing_only', 'tighter_trail', 'none'
    max_hold_bars: int = 12
    trail_atr_mult_time: float = 4.0  # ATR multiplier para trailing en time exit

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

        # Emergency stop (siempre activo)
        emergency_level = self.entry_price * (1 - CONFIG['emergency_stop_pct'])
        if low <= emergency_level:
            self._close(emergency_level * (1 - CONFIG['slippage_pct'] / 100), 'emergency_stop')
            return {'type': 'full_exit', 'reason': 'emergency_stop'}

        # Trailing stop normal (si ya activo)
        if self.trailing_active and self.trailing_stop is not None:
            if low <= self.trailing_stop:
                self._close(self.trailing_stop * (1 - CONFIG['slippage_pct'] / 100), 'trailing_stop')
                return {'type': 'full_exit', 'reason': 'trailing_stop'}

        # Activar trailing normal a +2R
        if r_mult >= CONFIG['trail_trigger_r']:
            chandelier = self.highest_since - (current_atr * CONFIG['trail_atr_mult'])
            if not self.trailing_active:
                self.trailing_active = True
                self.trailing_stop = chandelier
            elif chandelier > self.trailing_stop:
                self.trailing_stop = chandelier

        # ===== TIME EXIT LOGIC (variable segun modo) =====
        if self.bars_held >= self.max_hold_bars:
            if self.time_exit_mode == 'original':
                # V0: Modo actual — forzar salida si price <= entry
                if close <= self.entry_price:
                    self._close(close * (1 - CONFIG['slippage_pct'] / 100), 'time_exit')
                    return {'type': 'full_exit', 'reason': 'time_exit'}
                elif not self.trailing_active:
                    chandelier = self.highest_since - (current_atr * CONFIG['trail_atr_mult'])
                    breakeven = self.entry_price * (1 + CONFIG['slippage_pct'] / 100)
                    self.trailing_active = True
                    self.trailing_stop = max(chandelier, breakeven)

            elif self.time_exit_mode == 'trailing_only':
                # V3/V5: Nunca forzar salida, siempre activar trailing a breakeven
                if not self.trailing_active:
                    chandelier = self.highest_since - (current_atr * self.trail_atr_mult_time)
                    breakeven = self.entry_price * (1 + CONFIG['slippage_pct'] / 100)
                    self.trailing_active = True
                    # Si esta en perdidas, trailing a breakeven (esperamos recuperacion)
                    # Si esta en ganancias, trailing normal
                    if close <= self.entry_price:
                        # Perdiendo: ponemos trailing apretado (2xATR desde maximo o breakeven)
                        tight_chandelier = self.highest_since - (current_atr * self.trail_atr_mult_time)
                        self.trailing_stop = max(tight_chandelier, self.entry_price * 0.95)
                    else:
                        self.trailing_stop = max(chandelier, breakeven)

            elif self.time_exit_mode == 'tighter_trail':
                # V4: Activar trailing mas apretado en vez de forzar salida
                if not self.trailing_active:
                    chandelier = self.highest_since - (current_atr * self.trail_atr_mult_time)
                    self.trailing_active = True
                    # Si perdiendo: trailing apretado bajo el precio actual
                    if close <= self.entry_price:
                        self.trailing_stop = close * 0.97  # 3% stop desde precio actual
                    else:
                        breakeven = self.entry_price * (1 + CONFIG['slippage_pct'] / 100)
                        self.trailing_stop = max(chandelier, breakeven)

            elif self.time_exit_mode == 'none':
                # Sin time exit, solo emergency stop y trailing normal
                pass

        return None

    def _close(self, exit_price, reason):
        self.pnl_euros = (exit_price - self.entry_price) * self.position_units
        self.pnl_pct = (self.pnl_euros / self.position_euros) * 100 if self.position_euros > 0 else 0
        self.exit_price = exit_price
        self.exit_reason = reason


# =============================================================================
# BACKTEST ENGINE PARA VARIANTES
# =============================================================================

def run_variant_backtest(months, all_data, signals_data, macro_bullish, all_dates,
                         variant_name, time_exit_mode, max_hold_bars, trail_atr_mult_time=4.0,
                         use_options=True):
    """Ejecuta backtest con una variante especifica de time exit."""
    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}
    active_options = {}
    all_trades = []
    all_option_trades = []

    for current_date in all_dates:
        # 1. Gestionar trades activos
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

        # 2. Gestionar opciones (sin cambios)
        options_to_close = []
        for ticker, opt in active_options.items():
            if ticker not in signals_data:
                continue
            df = signals_data[ticker]['df']
            if current_date not in df.index:
                continue
            idx = df.index.get_loc(current_date)
            bar = df.iloc[idx]
            days_elapsed = (current_date - opt.entry_date).days
            iv_val = df['HVOL'].iloc[idx]
            if pd.isna(iv_val) or iv_val <= 0:
                iv_val = opt.entry_iv
            result = opt.update(bar['Close'], iv_val, days_elapsed)
            if result and result['type'] == 'full_exit':
                opt.exit_date = current_date
                options_to_close.append(ticker)
                tracker.update_equity(opt.pnl_euros, current_date)
        for ticker in options_to_close:
            opt = active_options.pop(ticker)
            tracker.open_positions -= 1
            tracker.open_options -= 1
            all_option_trades.append(opt)

        # 3. Buscar nuevas senales
        if CONFIG['use_macro_filter']:
            if current_date in macro_bullish:
                is_macro_ok = macro_bullish[current_date]
            else:
                prev_dates = [d for d in macro_bullish if d < current_date]
                is_macro_ok = macro_bullish[prev_dates[-1]] if prev_dates else False
        else:
            is_macro_ok = True

        if tracker.open_positions < CONFIG['max_positions'] and is_macro_ok:
            candidates = find_candidates(signals_data, {**active_trades, **active_options}, current_date, is_macro_ok)
            ranked = rank_candidates(candidates, signals_data)

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= CONFIG['max_positions']:
                    break
                df = signals_data[ticker]['df']
                bar = df.iloc[idx]

                open_as_option = False
                current_ivr = None
                if use_options and ticker in OPTIONS_ELIGIBLE and tracker.open_options < CONFIG['max_option_positions']:
                    hvol_series = df['HVOL']
                    current_ivr = iv_rank(hvol_series, idx, CONFIG.get('option_ivr_window', 252))
                    if current_ivr < CONFIG.get('option_max_ivr', 40):
                        open_as_option = True

                if open_as_option:
                    stock_price = bar['Open']
                    iv_val = df['HVOL'].iloc[idx]
                    if pd.isna(iv_val) or iv_val <= 0:
                        iv_val = 0.30
                    strike = stock_price * (1 - CONFIG['option_itm_pct'])
                    dte = monthly_expiration_dte(current_date, CONFIG['option_dte'])
                    T = dte / 365
                    bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], iv_val)
                    premium = bs['price']
                    premium *= (1 + CONFIG['option_spread_pct'] / 100 / 2)  # spread de entrada
                    if premium <= 0.01:
                        continue
                    size = tracker.get_option_size(premium)
                    if size['premium'] < 50:
                        continue
                    opt = OptionTradeV2(
                        ticker=ticker, entry_date=current_date,
                        entry_stock_price=stock_price, strike=round(strike, 2),
                        dte_at_entry=dte, entry_option_price=premium,
                        entry_iv=iv_val, num_contracts=size['contracts'],
                        position_euros=size['premium'],
                    )
                    active_options[ticker] = opt
                    tracker.open_positions += 1
                    tracker.open_options += 1
                else:
                    entry_price = bar['Open'] * (1 + CONFIG['slippage_pct'] / 100)
                    R = prev_atr * 2.0
                    risk_pct = CONFIG['target_risk_per_trade_pct'] / 100
                    dollar_risk = tracker.equity * risk_pct
                    position_units = dollar_risk / R
                    position_euros = position_units * entry_price
                    max_notional = tracker.equity / CONFIG['max_positions']
                    if position_euros > max_notional:
                        position_euros = max_notional
                        position_units = position_euros / entry_price

                    if position_euros < 100:
                        continue

                    trade = TradeVariant(
                        ticker=ticker, entry_price=entry_price,
                        entry_date=current_date, entry_atr=prev_atr,
                        position_euros=position_euros, position_units=position_units,
                        time_exit_mode=time_exit_mode,
                        max_hold_bars=max_hold_bars,
                        trail_atr_mult_time=trail_atr_mult_time,
                    )
                    active_trades[ticker] = trade
                    tracker.open_positions += 1

    # Cerrar posiciones abiertas
    for ticker, trade in active_trades.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            trade._close(df['Close'].iloc[-1], 'end_of_data')
            trade.exit_date = df.index[-1]
            tracker.update_equity(trade.pnl_euros, df.index[-1])
            all_trades.append(trade)
    for ticker, opt in active_options.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            stock_price = df['Close'].iloc[-1]
            intrinsic = max(stock_price - opt.strike, 0)
            opt._close(intrinsic, 'end_of_data')
            opt.exit_date = df.index[-1]
            tracker.update_equity(opt.pnl_euros, df.index[-1])
            all_option_trades.append(opt)

    # Metricas
    combined = all_trades + all_option_trades
    if not combined:
        return None

    total_count = len(combined)
    winners = [t for t in combined if t.pnl_euros > 0]
    losers = [t for t in combined if t.pnl_euros <= 0]
    total_pnl = sum(t.pnl_euros for t in combined)
    win_rate = len(winners) / total_count * 100
    gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
    profit_factor = gross_profit / gross_loss
    max_dd = tracker.get_max_drawdown()
    total_return_pct = (tracker.equity / CONFIG['initial_capital'] - 1) * 100
    annualized = ((1 + total_return_pct / 100) ** (12 / months) - 1) * 100 if months > 0 else 0

    # Desglose por exit_reason
    exit_reasons = {}
    for t in combined:
        reason = t.exit_reason or 'unknown'
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'pnl': 0, 'wins': 0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += t.pnl_euros
        if t.pnl_euros > 0:
            exit_reasons[reason]['wins'] += 1

    return {
        'variant': variant_name,
        'total_trades': total_count,
        'stock_trades': len(all_trades),
        'option_trades': len(all_option_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'annualized_pct': annualized,
        'max_drawdown': max_dd,
        'final_equity': tracker.equity,
        'exit_reasons': exit_reasons,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--months', type=int, default=240)
    parser.add_argument('--no-options', action='store_true', help='Solo acciones, sin opciones')
    args = parser.parse_args()
    months = args.months
    use_options = not args.no_options
    mode_label = "SOLO ACCIONES" if not use_options else "CON OPCIONES"

    print("=" * 90)
    print(f"  TEST DE VARIANTES DE TIME EXIT — {mode_label} — {months} MESES")
    print("=" * 90)

    # --- DESCARGAR DATOS (una sola vez) ---
    print("\n  Descargando datos...")
    all_data = {}
    for i, ticker in enumerate(BASE_TICKERS):
        df = download_data(ticker, months)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            df['HVOL'] = historical_volatility(df['Close'], CONFIG['hvol_window'])
            all_data[ticker] = df
        if (i + 1) % 20 == 0 or i == len(BASE_TICKERS) - 1:
            print(f"\r  Descargados: {len(all_data)}/{len(BASE_TICKERS)}", end='')
    print(f"\n  Tickers con datos: {len(all_data)}")

    # --- ENGINE + SENALES (una sola vez) ---
    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG['rsi_max'],
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG['longs_only']
    )
    signals_data, total_signals = generate_all_signals(all_data, engine)
    macro_bullish = build_macro_filter(all_data)
    all_dates = sorted(set(d for sd in signals_data.values() for d in sd['df'].index.tolist()))
    print(f"  Senales LONG totales: {total_signals}")
    print(f"  Dias de mercado: {len(all_dates)}")

    # --- DEFINIR VARIANTES ---
    variants = [
        {
            'name': 'V0: Original (12 bars, forzar)',
            'mode': 'original',
            'max_hold': 12,
            'trail_atr': 4.0,
        },
        {
            'name': 'V1: 20 bars, forzar',
            'mode': 'original',
            'max_hold': 20,
            'trail_atr': 4.0,
        },
        {
            'name': 'V2: 15 bars, forzar',
            'mode': 'original',
            'max_hold': 15,
            'trail_atr': 4.0,
        },
        {
            'name': 'V3: 12 bars, trailing only 4xATR',
            'mode': 'trailing_only',
            'max_hold': 12,
            'trail_atr': 4.0,
        },
        {
            'name': 'V4: 12 bars, tighter trail 3%',
            'mode': 'tighter_trail',
            'max_hold': 12,
            'trail_atr': 2.0,
        },
        {
            'name': 'V5: 8 bars, trailing only 3xATR',
            'mode': 'trailing_only',
            'max_hold': 8,
            'trail_atr': 3.0,
        },
        {
            'name': 'V6: Sin time exit (solo emergency)',
            'mode': 'none',
            'max_hold': 999,
            'trail_atr': 4.0,
        },
        {
            'name': 'V7: 12 bars, trailing only 2xATR',
            'mode': 'trailing_only',
            'max_hold': 12,
            'trail_atr': 2.0,
        },
        {
            'name': 'V8: 10 bars, trailing only 3xATR',
            'mode': 'trailing_only',
            'max_hold': 10,
            'trail_atr': 3.0,
        },
    ]

    # --- EJECUTAR TODAS LAS VARIANTES ---
    results = []
    for v in variants:
        print(f"\n  Ejecutando {v['name']}...")
        r = run_variant_backtest(
            months, all_data, signals_data, macro_bullish, all_dates,
            v['name'], v['mode'], v['max_hold'], v['trail_atr'],
            use_options=use_options
        )
        if r:
            results.append(r)
            print(f"    -> PF={r['profit_factor']:.2f}, Return={r['total_return_pct']:+.1f}%, "
                  f"MaxDD={r['max_drawdown']:.1f}%, Trades={r['total_trades']}")

    # --- TABLA COMPARATIVA ---
    print(f"\n\n{'='*110}")
    print(f"  TABLA COMPARATIVA DE VARIANTES DE TIME EXIT")
    print(f"{'='*110}")
    print(f"  {'Variante':<38} {'Trades':>7} {'Win%':>6} {'PF':>6} {'P&L EUR':>12} {'Return%':>9} {'Annual%':>8} {'MaxDD%':>7} {'Equity':>10}")
    print(f"  {'-'*38} {'-'*7} {'-'*6} {'-'*6} {'-'*12} {'-'*9} {'-'*8} {'-'*7} {'-'*10}")

    # Ordenar por PF descendente
    results_sorted = sorted(results, key=lambda x: x['profit_factor'], reverse=True)
    for r in results_sorted:
        marker = " ***" if r == results_sorted[0] else ""
        print(f"  {r['variant']:<38} {r['total_trades']:>7} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
              f"{r['total_pnl']:>+11,.0f} {r['total_return_pct']:>+8.1f}% {r['annualized_pct']:>+7.1f}% "
              f"{r['max_drawdown']:>6.1f}% {r['final_equity']:>9,.0f}{marker}")

    # --- DESGLOSE POR EXIT REASON ---
    print(f"\n\n{'='*110}")
    print(f"  DESGLOSE POR RAZON DE SALIDA (cada variante)")
    print(f"{'='*110}")

    for r in results_sorted[:5]:  # Top 5 variantes
        print(f"\n  {r['variant']}:")
        print(f"    {'Razon':<20} {'N':>5} {'Win%':>6} {'P&L EUR':>12}")
        print(f"    {'-'*20} {'-'*5} {'-'*6} {'-'*12}")
        for reason, data in sorted(r['exit_reasons'].items(), key=lambda x: -x[1]['count']):
            wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
            print(f"    {reason:<20} {data['count']:>5} {wr:>5.1f}% {data['pnl']:>+11,.0f}")

    # --- CONCLUSIONES ---
    best = results_sorted[0]
    original = next(r for r in results if 'V0' in r['variant'])

    print(f"\n\n{'='*110}")
    print(f"  CONCLUSIONES")
    print(f"{'='*110}")
    print(f"  Mejor variante:   {best['variant']}")
    print(f"  PF mejora:        {original['profit_factor']:.2f} -> {best['profit_factor']:.2f} ({(best['profit_factor']/original['profit_factor']-1)*100:+.1f}%)")
    print(f"  P&L mejora:       EUR {original['total_pnl']:+,.0f} -> EUR {best['total_pnl']:+,.0f}")
    print(f"  MaxDD cambio:     {original['max_drawdown']:.1f}% -> {best['max_drawdown']:.1f}%")
    print(f"  Return cambio:    {original['total_return_pct']:+.1f}% -> {best['total_return_pct']:+.1f}%")

    time_exit_original = original['exit_reasons'].get('time_exit', {})
    print(f"\n  Time exits originales: {time_exit_original.get('count', 0)} trades, "
          f"EUR {time_exit_original.get('pnl', 0):+,.0f}")

    time_exit_best = best['exit_reasons'].get('time_exit', {})
    print(f"  Time exits mejor:     {time_exit_best.get('count', 0)} trades, "
          f"EUR {time_exit_best.get('pnl', 0):+,.0f}")


if __name__ == "__main__":
    main()
