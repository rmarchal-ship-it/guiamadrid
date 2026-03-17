#!/usr/bin/env python3
"""
BACKTEST v13 — Rolling Thunder
=============================

Version: v13 (28 Feb 2026)
Base: v12 (v8 + EU options, slots separados)
Cambio: Rolling de opciones GANADORAS a 45 DTE en vez de cerrar

Reglas Rolling:
  - A 45 DTE, si la opcion tiene P&L > 0 Y la accion sigue con senal LONG:
    → Cerrar opcion actual (registrar P&L como 'dte_roll')
    → Abrir nueva opcion: MISMO STRIKE, ~120 DTE nuevo
    → Re-sizing basado en equity actual
    → Pagar nuevo spread
  - Si la opcion pierde o no hay senal: exit normal (dte_exit)

Racional:
  - Opciones ITM profundas tienen valor extrinseco minimo → theta decay no duele
  - Same strike = mantener exposicion nominal = confiar en el trend (momentum)
  - Rolling permite capturar movimientos prolongados sin el corte artificial a 75 dias

Test: 24 meses (--months 24)

Archivos relacionados:
  - backtest_v12_eu_options.py — v12 base
  - backtest_experimental.py — v8 base (imports)
  - momentum_breakout.py — motor de senales, ASSETS 225 tickers

Uso:
  python3 backtest_v13_rolling.py                    # 24 meses, v12 vs v13
  python3 backtest_v13_rolling.py --months 60        # 60 meses
  python3 backtest_v13_rolling.py --months 24 --verbose
"""

import sys, os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Importar todo del backtest base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_experimental import (
    CONFIG, BASE_TICKERS, LEVERAGE_FACTORS, OPTIONS_ELIGIBLE,
    MACRO_EXEMPT, MACRO_EXEMPT_NEG,
    black_scholes_call, historical_volatility, monthly_expiration_dte, iv_rank,
    download_data, generate_all_signals, build_macro_filter,
    rank_candidates, find_candidates,
    Trade, EquityTracker,
    calculate_atr,
)
from momentum_breakout import MomentumEngine, ASSETS

# Importar de v12: tickers EU, spreads, funciones comunes
from backtest_v12_eu_options import (
    OPTIONS_ELIGIBLE_EU, OPTIONS_ALL,
    US_SPREAD_PCT, EU_SPREAD_PCT,
    get_option_spread,
    run_backtest_eu,          # para REF (sin rolling)
    simulate_gold_overlay,
    print_comparison,
)


# =============================================================================
# OPTION TRADE V2 — ROLLING THUNDER (extiende v12 OptionTradeV2EU)
# =============================================================================

@dataclass
class OptionTradeV2Roll:
    """
    Trade de opcion con soporte para rolling.
    Cuando remaining_dte <= 45 y la opcion es profitable,
    devuelve roll_candidate en vez de cerrar.
    """
    ticker: str
    entry_date: datetime
    entry_stock_price: float
    strike: float
    dte_at_entry: int
    entry_option_price: float
    entry_iv: float
    num_contracts: float
    position_euros: float   # premium pagada = max loss
    spread_pct: float       # spread especifico del ticker

    # Tracking de rolling chain
    roll_number: int = field(default=0)       # 0=original, 1=1er roll, 2=2do roll...
    original_entry_date: Optional[datetime] = field(default=None)  # fecha del trade original

    bars_held: int = field(default=0)
    max_option_value: float = field(init=False)
    max_r_mult: float = field(default=0.0)

    exit_date: Optional[datetime] = field(default=None)
    exit_option_price: float = field(default=0.0)
    exit_reason: Optional[str] = field(default=None)
    pnl_euros: float = field(default=0.0)
    pnl_pct: float = field(default=0.0)

    def __post_init__(self):
        self.max_option_value = self.entry_option_price
        if self.original_entry_date is None:
            self.original_entry_date = self.entry_date

    def update(self, stock_price, current_iv, days_elapsed):
        self.bars_held += 1

        remaining_dte = max(self.dte_at_entry - days_elapsed, 0)
        T = remaining_dte / 365.0

        bs = black_scholes_call(
            S=stock_price, K=self.strike, T=T,
            r=CONFIG['risk_free_rate'], sigma=current_iv
        )
        current_option_price = bs['price']
        current_option_price *= (1 - self.spread_pct / 100 / 2)

        self.max_option_value = max(self.max_option_value, current_option_price)

        option_return = (current_option_price / self.entry_option_price) - 1 if self.entry_option_price > 0 else 0
        self.max_r_mult = max(self.max_r_mult, option_return)

        # EXPIRACION (safety)
        if remaining_dte <= 0:
            intrinsic = max(stock_price - self.strike, 0)
            intrinsic *= (1 - self.spread_pct / 100 / 2)
            self._close(intrinsic, 'expiration')
            return {'type': 'full_exit', 'reason': 'expiration'}

        # 45 DTE CHECK — ROLLING THUNDER
        if remaining_dte <= CONFIG.get('option_close_dte', 45):
            is_profitable = current_option_price > self.entry_option_price
            if is_profitable:
                # NO cerramos — devolvemos roll_candidate para que el main loop decida
                return {
                    'type': 'roll_candidate',
                    'current_option_price': current_option_price,
                    'stock_price': stock_price,
                    'current_iv': current_iv,
                }
            else:
                # Perdedora: exit normal
                self._close(current_option_price, 'dte_exit')
                return {'type': 'full_exit', 'reason': 'dte_exit'}

        return None

    def _close(self, exit_option_price, reason):
        self.exit_option_price = exit_option_price
        self.exit_reason = reason
        self.pnl_euros = (exit_option_price - self.entry_option_price) * self.num_contracts * 100
        self.pnl_pct = ((exit_option_price / self.entry_option_price) - 1) * 100 if self.entry_option_price > 0 else 0


# =============================================================================
# SIGNAL CHECK para rolling
# =============================================================================

def check_signal_for_roll(signals_data, ticker, current_date, is_macro_ok,
                          macro_exempt_set=None, opt=None, mode='ker_check',
                          ker_roll_threshold=0.30):
    """
    Verifica si un ticker cumple condiciones para rolling.

    Modos:
      'strict'       — senal LONG fresca (breakout). Muy conservador. (0/10 rolls en test)
      'trend_intact'  — stock price > entry price AND macro OK. Demasiado laxo. (9/10 rolls)
      'ker_check'    — KER > threshold AND price > entry AND macro OK. El sweet spot.
      'relaxed'      — stock price > entry price (ignora macro). Mas agresivo.

    KER (Kaufman Efficiency Ratio):
      - Mide calidad del trend (0=choppy, 1=trend perfecto)
      - Entrada normal usa KER > 0.40
      - Para rolling, usamos KER > 0.30 (el trend no necesita ser tan fuerte
        como para entrar, solo necesita seguir vivo)
    """
    if ticker not in signals_data:
        return False
    sd = signals_data[ticker]
    df = sd['df']
    if current_date not in df.index:
        return False
    idx = df.index.get_loc(current_date)
    if idx < 1:
        return False

    if mode == 'strict':
        # Senal LONG fresca — identica a find_candidates()
        signals = sd['signals']
        prev_signal = signals.iloc[idx - 1]
        if prev_signal != 1:
            return False
        if not is_macro_ok and not (macro_exempt_set and ticker in macro_exempt_set):
            return False
        return True

    elif mode == 'ker_check':
        # KER todavia alto + precio por encima del entry + macro OK
        current_price = df['Close'].iloc[idx]
        if opt is None:
            return False
        if current_price <= opt.entry_stock_price:
            return False
        # KER check: el trend sigue siendo eficiente
        ker_series = sd['ker']
        prev_ker = ker_series.iloc[idx - 1] if idx >= 1 else 0
        if pd.isna(prev_ker) or prev_ker < ker_roll_threshold:
            return False
        # Macro filter
        if not is_macro_ok and not (macro_exempt_set and ticker in macro_exempt_set):
            return False
        return True

    elif mode == 'trend_intact':
        # Trend intacto: stock sigue por encima del entry + macro OK
        current_price = df['Close'].iloc[idx]
        if opt is None:
            return False
        if current_price <= opt.entry_stock_price:
            return False
        if not is_macro_ok and not (macro_exempt_set and ticker in macro_exempt_set):
            return False
        return True

    elif mode == 'relaxed':
        # Solo check de precio, ignora macro
        current_price = df['Close'].iloc[idx]
        if opt is None:
            return False
        return current_price > opt.entry_stock_price

    return False


# =============================================================================
# RUN BACKTEST — ROLLING THUNDER
# =============================================================================

def run_backtest_rolling(months, tickers, label, use_leverage_scaling=False,
                         use_options=False, options_eligible_set=None,
                         max_us_options=2, max_eu_options=0,
                         macro_exempt_set=None, verbose=False):
    """
    Backtest v13: identico a v12 run_backtest_eu() pero con rolling de opciones
    ganadoras a 45 DTE cuando la accion sigue con senal LONG.
    """
    if options_eligible_set is None:
        options_eligible_set = set(OPTIONS_ELIGIBLE)
    else:
        options_eligible_set = set(options_eligible_set)

    n_tickers = len(tickers)
    print(f"\n{'='*70}")
    print(f"  {label} -- {months} MESES -- {n_tickers} tickers")
    if use_options:
        n_us = len([t for t in options_eligible_set if t in OPTIONS_ELIGIBLE])
        n_eu = len([t for t in options_eligible_set if t in OPTIONS_ELIGIBLE_EU])
        print(f"  Options eligible: {len(options_eligible_set)} ({n_us} US @ {US_SPREAD_PCT}% + {n_eu} EU @ {EU_SPREAD_PCT}%)")
        print(f"  Slots: US max {max_us_options} + EU max {max_eu_options} = {max_us_options + max_eu_options} total")
        print(f"  ROLLING THUNDER: opciones ganadoras rolan a mismo strike, 120 DTE")
    print(f"{'='*70}")

    # Descargar datos
    print("  Descargando datos...")
    all_data = {}
    failed = []
    for i, ticker in enumerate(tickers):
        df = download_data(ticker, months)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            df['HVOL'] = historical_volatility(df['Close'], CONFIG['hvol_window'])
            all_data[ticker] = df
        else:
            failed.append(ticker)
        if (i + 1) % 20 == 0 or i == n_tickers - 1:
            print(f"\r  Descargados: {len(all_data)}/{n_tickers} OK, {len(failed)} fallidos", end='')
    print(f"\n  Tickers con datos: {len(all_data)}")
    if failed:
        print(f"  Fallidos: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

    if not all_data:
        return {'error': 'No data'}

    # Engine + senales
    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG['rsi_max'],
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG['longs_only']
    )
    signals_data, total_signals = generate_all_signals(all_data, engine)
    print(f"  Senales LONG totales: {total_signals}")

    # Filtro macro
    macro_bullish = build_macro_filter(all_data)

    # Timeline
    all_dates = sorted(set(d for sd in signals_data.values() for d in sd['df'].index.tolist()))

    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}       # ticker -> Trade
    active_options = {}      # ticker -> OptionTradeV2Roll
    all_trades = []
    all_option_trades = []

    # Contadores EU vs US (slots separados)
    option_opens_us = 0
    option_opens_eu = 0
    open_options_us = 0
    open_options_eu = 0

    # ========== ROLLING STATS ==========
    total_rolls = 0
    roll_attempts = 0       # veces que la opcion era roll_candidate
    roll_no_signal = 0      # veces que no rolo por falta de senal
    max_chain_length = 0    # cadena mas larga de rolls
    roll_chain_pnls = []    # P&L totales de cadenas completas

    # =================================================================
    # LOOP PRINCIPAL
    # =================================================================
    for current_date in all_dates:

        # 1. GESTIONAR TRADES ACTIVOS (acciones/ETFs) — identico a v12
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
                print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE {ticker:8} | "
                      f"{trade.exit_reason:<15} | P&L EUR {pnl_sign}{trade.pnl_euros:.0f} ({pnl_sign}{pnl_pct:.1f}%) | "
                      f"Pos: {tracker.open_positions}/10 | Equity: EUR {tracker.equity:,.0f}")

        # 2. GESTIONAR OPCIONES ACTIVAS — *** ROLLING THUNDER ***
        options_to_close = []
        options_to_roll = []   # NEW: lista de tickers a rolar

        # Macro filter para este dia (necesario para signal check)
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

        for ticker, opt in active_options.items():
            if ticker not in signals_data:
                continue
            df = signals_data[ticker]['df']
            if current_date not in df.index:
                continue
            idx = df.index.get_loc(current_date)
            bar = df.iloc[idx]
            days_elapsed = (current_date - opt.entry_date).days
            iv = df['HVOL'].iloc[idx]
            if pd.isna(iv) or iv <= 0:
                iv = opt.entry_iv
            result = opt.update(bar['Close'], iv, days_elapsed)

            if result:
                if result['type'] == 'roll_candidate':
                    # *** ROLLING DECISION ***
                    roll_attempts += 1
                    # Obtener KER actual para logging
                    ker_val = signals_data[ticker]['ker'].iloc[idx - 1] if idx >= 1 else 0
                    if pd.isna(ker_val):
                        ker_val = 0

                    has_signal = check_signal_for_roll(
                        signals_data, ticker, current_date,
                        is_macro_ok, macro_exempt_set,
                        opt=opt, mode='ker_check', ker_roll_threshold=0.40
                    )
                    if has_signal:
                        # ROLL! Cerrar actual y preparar re-apertura
                        opt._close(result['current_option_price'], 'dte_roll')
                        opt.exit_date = current_date
                        options_to_roll.append((ticker, result))
                        tracker.update_equity(opt.pnl_euros, current_date)
                        total_rolls += 1
                        if verbose:
                            region = "EU" if ticker in OPTIONS_ELIGIBLE_EU else "US"
                            print(f"  {current_date.strftime('%Y-%m-%d')} | ROLL OPT  {ticker:8} [{region}] | "
                                  f"P&L EUR {opt.pnl_euros:+.0f} ({opt.pnl_pct:+.1f}%) | "
                                  f"Roll #{opt.roll_number + 1} | Same K=${opt.strike:.2f} | "
                                  f"KER={ker_val:.2f} | Equity: EUR {tracker.equity:,.0f}")
                    else:
                        # KER bajo o precio bajo entry → exit normal
                        roll_no_signal += 1
                        opt._close(result['current_option_price'], 'dte_exit')
                        opt.exit_date = current_date
                        options_to_close.append(ticker)
                        tracker.update_equity(opt.pnl_euros, current_date)
                        if verbose:
                            region = "EU" if ticker in OPTIONS_ELIGIBLE_EU else "US"
                            stock_px = result['stock_price']
                            print(f"  {current_date.strftime('%Y-%m-%d')} | NOROLL OPT {ticker:8} [{region}] | "
                                  f"dte_exit | P&L EUR {opt.pnl_euros:+.0f} ({opt.pnl_pct:+.1f}%) | "
                                  f"KER={ker_val:.2f} Stock=${stock_px:.2f} Entry=${opt.entry_stock_price:.2f}")

                elif result['type'] == 'full_exit':
                    opt.exit_date = current_date
                    options_to_close.append(ticker)
                    tracker.update_equity(opt.pnl_euros, current_date)

        # Procesar cierres normales
        for ticker in options_to_close:
            opt = active_options.pop(ticker)
            tracker.open_positions -= 1
            tracker.open_options -= 1
            if ticker in OPTIONS_ELIGIBLE_EU:
                open_options_eu -= 1
            else:
                open_options_us -= 1
            all_option_trades.append(opt)
            # Track chain length
            if opt.roll_number > 0:
                max_chain_length = max(max_chain_length, opt.roll_number)

        # Procesar rolls: cerrar vieja, abrir nueva
        for ticker, roll_result in options_to_roll:
            old_opt = active_options.pop(ticker)
            # NO decrementamos open_positions — el slot se reutiliza
            all_option_trades.append(old_opt)  # registrar la opcion cerrada

            # Abrir nueva opcion al MISMO STRIKE, 120 DTE
            stock_price = roll_result['stock_price']
            current_iv = roll_result['current_iv']

            actual_dte = monthly_expiration_dte(current_date, CONFIG['option_dte'])
            T = actual_dte / 365.0

            bs = black_scholes_call(stock_price, old_opt.strike, T, CONFIG['risk_free_rate'], current_iv)
            new_option_price = bs['price']

            ticker_spread = get_option_spread(ticker)
            new_option_price *= (1 + ticker_spread / 100 / 2)  # spread de entrada

            # Re-sizing basado en equity actual
            size = tracker.get_option_size(new_option_price)
            if size['premium'] < 50:
                # Premium muy baja — no rolar, cerrar slot
                tracker.open_positions -= 1
                tracker.open_options -= 1
                if ticker in OPTIONS_ELIGIBLE_EU:
                    open_options_eu -= 1
                else:
                    open_options_us -= 1
                if verbose:
                    print(f"  {current_date.strftime('%Y-%m-%d')} | SKIP ROLL {ticker:8} | "
                          f"Premium too low (${new_option_price:.2f})")
                continue

            new_opt = OptionTradeV2Roll(
                ticker=ticker,
                entry_date=current_date,
                entry_stock_price=stock_price,
                strike=old_opt.strike,       # *** MISMO STRIKE ***
                dte_at_entry=actual_dte,
                entry_option_price=new_option_price,
                entry_iv=current_iv,
                num_contracts=size['contracts'],
                position_euros=size['premium'],
                spread_pct=ticker_spread,
                roll_number=old_opt.roll_number + 1,
                original_entry_date=old_opt.original_entry_date,
            )
            active_options[ticker] = new_opt
            # Slot ya contado (no incrementar open_positions)

            if verbose:
                region = "EU" if ticker in OPTIONS_ELIGIBLE_EU else "US"
                is_eu_ticker = ticker in OPTIONS_ELIGIBLE_EU
                print(f"  {current_date.strftime('%Y-%m-%d')} | REOPEN OPT {ticker:8} [{region} {ticker_spread}%] | "
                      f"Same K=${old_opt.strike:.2f} IV={current_iv:.0%} "
                      f"{actual_dte}DTE Prem=${new_option_price:.2f} x{size['contracts']:.2f}c = EUR {size['premium']:.0f}")

        # 3. BUSCAR NUEVAS SENALES — identico a v12
        total_open = tracker.open_positions

        if total_open < CONFIG['max_positions'] and (is_macro_ok or macro_exempt_set):
            candidates = find_candidates(signals_data, {**active_trades, **active_options}, current_date, is_macro_ok, macro_exempt_set)
            ranked = rank_candidates(candidates, signals_data)

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= CONFIG['max_positions']:
                    break

                df = signals_data[ticker]['df']
                bar = df.iloc[idx]

                # Decidir si abrir opcion o accion
                open_as_option = False
                current_ivr = None
                is_eu_ticker = ticker in OPTIONS_ELIGIBLE_EU
                if use_options and ticker in options_eligible_set:
                    has_slot = False
                    if is_eu_ticker and open_options_eu < max_eu_options:
                        has_slot = True
                    elif not is_eu_ticker and open_options_us < max_us_options:
                        has_slot = True

                    if has_slot:
                        hvol_series = df['HVOL']
                        current_ivr = iv_rank(hvol_series, idx, CONFIG.get('option_ivr_window', 252))
                        max_ivr = CONFIG.get('option_max_ivr', 40)
                        if current_ivr < max_ivr:
                            open_as_option = True

                if open_as_option:
                    # --- OPCION CALL (con soporte rolling) ---
                    stock_price = bar['Open']
                    strike = stock_price * (1 - CONFIG['option_itm_pct'])  # 5% ITM

                    actual_dte = monthly_expiration_dte(current_date, CONFIG['option_dte'])
                    T = actual_dte / 365.0

                    iv = df['HVOL'].iloc[idx]
                    if pd.isna(iv) or iv <= 0:
                        iv = 0.30
                    bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], iv)
                    option_price = bs['price']

                    ticker_spread = get_option_spread(ticker)
                    option_price *= (1 + ticker_spread / 100 / 2)

                    size = tracker.get_option_size(option_price)
                    if size['premium'] < 50:
                        continue

                    opt = OptionTradeV2Roll(
                        ticker=ticker,
                        entry_date=current_date,
                        entry_stock_price=stock_price,
                        strike=strike,
                        dte_at_entry=actual_dte,
                        entry_option_price=option_price,
                        entry_iv=iv,
                        num_contracts=size['contracts'],
                        position_euros=size['premium'],
                        spread_pct=ticker_spread,
                        roll_number=0,                    # original (no rolada)
                        original_entry_date=current_date,
                    )
                    active_options[ticker] = opt
                    tracker.open_positions += 1
                    tracker.open_options += 1

                    if is_eu_ticker:
                        open_options_eu += 1
                        option_opens_eu += 1
                    else:
                        open_options_us += 1
                        option_opens_us += 1

                    if verbose:
                        region = "EU" if ticker in OPTIONS_ELIGIBLE_EU else "US"
                        print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN OPT {ticker:8} [{region} {ticker_spread}%] | "
                              f"K=${strike:.2f} IV={iv:.0%} IVR={current_ivr:.0f} "
                              f"{actual_dte}DTE Prem=${option_price:.2f} x{size['contracts']:.2f}c = EUR {size['premium']:.0f}")
                else:
                    # --- ACCION/ETF --- (identico a v12)
                    use_lev = use_leverage_scaling
                    size_info = tracker.get_position_size(ticker, prev_atr, bar['Open'], use_lev)
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
                        lev = LEVERAGE_FACTORS.get(ticker, 1.0)
                        lev_str = f" [{lev:.0f}x]" if lev > 1 else ""
                        print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN  {ticker:8}{lev_str} | "
                              f"EUR {position_euros:.0f} ({position_units:.2f}u) @ ${entry_price:.2f}")

    # Cerrar trades abiertos al final
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
            if opt.roll_number > 0:
                max_chain_length = max(max_chain_length, opt.roll_number)

    # =================================================================
    # METRICAS
    # =================================================================
    combined_trades = all_trades + all_option_trades
    if not combined_trades:
        return {'error': 'No trades'}

    total_count = len(combined_trades)
    winners = [t for t in combined_trades if (hasattr(t, 'pnl_euros') and t.pnl_euros > 0)]
    losers = [t for t in combined_trades if (hasattr(t, 'pnl_euros') and t.pnl_euros <= 0)]

    total_pnl = sum(t.pnl_euros for t in combined_trades)
    win_rate = len(winners) / total_count * 100 if total_count > 0 else 0

    gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
    profit_factor = gross_profit / gross_loss

    max_dd = tracker.get_max_drawdown()
    total_return_pct = (tracker.equity / CONFIG['initial_capital'] - 1) * 100
    annualized = ((1 + total_return_pct / 100) ** (12 / months) - 1) * 100 if months > 0 else 0

    avg_win_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0

    # Fat tails
    stock_gt_3r = sum(1 for t in all_trades if t.max_r_mult >= 3.0)
    opt_home_runs = sum(1 for t in all_option_trades if t.pnl_pct >= 100)

    best_trade = max(combined_trades, key=lambda t: t.pnl_pct)
    worst_trade = min(combined_trades, key=lambda t: t.pnl_pct)

    # Desglose opciones US vs EU
    opt_us = [t for t in all_option_trades if t.ticker not in OPTIONS_ELIGIBLE_EU]
    opt_eu = [t for t in all_option_trades if t.ticker in OPTIONS_ELIGIBLE_EU]
    pnl_opt_us = sum(t.pnl_euros for t in opt_us) if opt_us else 0
    pnl_opt_eu = sum(t.pnl_euros for t in opt_eu) if opt_eu else 0

    # Rolling stats
    rolled_trades = [t for t in all_option_trades if hasattr(t, 'roll_number') and t.roll_number > 0]
    non_rolled_trades = [t for t in all_option_trades if not hasattr(t, 'roll_number') or t.roll_number == 0]
    pnl_rolled = sum(t.pnl_euros for t in rolled_trades) if rolled_trades else 0
    pnl_non_rolled = sum(t.pnl_euros for t in non_rolled_trades) if non_rolled_trades else 0

    print(f"""
{'='*70}
  RESULTADOS {label} -- {months} MESES
{'='*70}

  CAPITAL:
     Inicial:        EUR {CONFIG['initial_capital']:,.2f}
     Final:          EUR {tracker.equity:,.2f}
     P&L Total:      EUR {total_pnl:+,.2f} ({total_return_pct:+.1f}%)
     Annualizado:    {annualized:+.1f}%
     Max Drawdown:   -{max_dd:.1f}%

  TRADES:
     Total:          {total_count} (stocks: {len(all_trades)}, opciones: {len(all_option_trades)})
     Ganadores:      {len(winners)} ({win_rate:.1f}%)
     Perdedores:     {len(losers)}
     Profit Factor:  {profit_factor:.2f}

  FAT TAILS:
     Stocks >= +3R:  {stock_gt_3r}
     Options >= +100%: {opt_home_runs}
     Avg Win:        {avg_win_pct:+.1f}%
     Avg Loss:       {avg_loss_pct:.1f}%
     Best:           {best_trade.ticker} {best_trade.pnl_pct:+.1f}%
     Worst:          {worst_trade.ticker} {worst_trade.pnl_pct:+.1f}%

  OPCIONES DESGLOSE:
     US trades:      {len(opt_us)} opens ({option_opens_us} total) | P&L EUR {pnl_opt_us:+,.0f}
     EU trades:      {len(opt_eu)} opens ({option_opens_eu} total) | P&L EUR {pnl_opt_eu:+,.0f}

  ROLLING THUNDER STATS:
     Roll attempts:  {roll_attempts} (opciones ganadoras a 45 DTE)
     Rolls ejecutados: {total_rolls} (con senal activa)
     No-roll (sin senal): {roll_no_signal}
     Roll rate:      {total_rolls/roll_attempts*100:.0f}% (de intentos) {'N/A' if roll_attempts == 0 else ''}
     Max chain:      {max_chain_length} rolls consecutivos
     Trades rolados: {len(rolled_trades)} (P&L EUR {pnl_rolled:+,.0f})
     Trades frescos: {len(non_rolled_trades)} (P&L EUR {pnl_non_rolled:+,.0f})
""")

    # Razones de salida
    exit_reasons = {}
    for t in combined_trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    print("  RAZONES DE SALIDA:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"     {reason:20} {count:3} ({count/total_count*100:.1f}%)")

    # Detalle opciones con rolling
    if verbose and all_option_trades:
        print(f"\n  DETALLE OPCIONES ({len(all_option_trades)} trades):")
        for i, opt in enumerate(sorted(all_option_trades, key=lambda x: x.entry_date), 1):
            entry_str = opt.entry_date.strftime('%Y-%m-%d') if opt.entry_date else '?'
            exit_str = opt.exit_date.strftime('%Y-%m-%d') if opt.exit_date else '?'
            marker = '+' if opt.pnl_euros > 0 else '-'
            region = "EU" if opt.ticker in OPTIONS_ELIGIBLE_EU else "US"
            roll_str = f"R{opt.roll_number}" if hasattr(opt, 'roll_number') and opt.roll_number > 0 else "  "
            print(f"     {i}. {entry_str} -> {exit_str} | {opt.ticker:10} [{region}] {roll_str} | "
                  f"K=${opt.strike:.2f} | Prem ${opt.entry_option_price:.2f} -> ${opt.exit_option_price:.2f} | "
                  f"P&L EUR {opt.pnl_euros:+.0f} ({opt.pnl_pct:+.1f}%) | {opt.bars_held}d | {opt.exit_reason} {marker}")

    return {
        'label': label,
        'total_trades': total_count,
        'stock_trades': len(all_trades),
        'option_trades': len(all_option_trades),
        'option_trades_us': len(opt_us),
        'option_trades_eu': len(opt_eu),
        'option_pnl_us': pnl_opt_us,
        'option_pnl_eu': pnl_opt_eu,
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
        'best_ticker': best_trade.ticker,
        'best_pnl_pct': best_trade.pnl_pct,
        'stock_gt_3r': stock_gt_3r,
        'opt_home_runs': opt_home_runs,
        'final_equity': tracker.equity,
        'equity_curve': tracker.equity_curve,
        'all_trades': all_trades,
        'all_option_trades': all_option_trades,
        'combined_trades': combined_trades,
        # Rolling-specific
        'total_rolls': total_rolls,
        'roll_attempts': roll_attempts,
        'roll_no_signal': roll_no_signal,
        'max_chain_length': max_chain_length,
        'rolled_trades': len(rolled_trades),
        'rolled_pnl': pnl_rolled,
    }


# =============================================================================
# TABLA COMPARATIVA v12 vs v13
# =============================================================================

def print_rolling_comparison(ref_result, rolling_result):
    """Comparativa detallada v12 vs v13."""
    r1 = ref_result
    r2 = rolling_result

    print(f"""
{'='*100}
  v12 vs v13 ROLLING THUNDER — COMPARATIVA
{'='*100}

  {'Metrica':<25} {'v12 (no roll)':<20} {'v13 (rolling)':<20} {'Delta':<15}
  {'-'*80}
  {'Equity Final':<25} EUR {r1['final_equity']:>12,.0f} EUR {r2['final_equity']:>12,.0f}
  {'CAGR':<25} {r1['annualized_return_pct']:>+8.1f}%{'':>11} {r2['annualized_return_pct']:>+8.1f}%{'':>11} {r2['annualized_return_pct']-r1['annualized_return_pct']:>+.1f}pp
  {'Max Drawdown':<25} {r1['max_drawdown']:>8.1f}%{'':>11} {r2['max_drawdown']:>8.1f}%{'':>11} {r2['max_drawdown']-r1['max_drawdown']:>+.1f}pp
  {'Profit Factor':<25} {r1['profit_factor']:>8.2f}{'':>11} {r2['profit_factor']:>8.2f}{'':>11} {r2['profit_factor']-r1['profit_factor']:>+.2f}
  {'Win Rate':<25} {r1['win_rate']:>8.1f}%{'':>11} {r2['win_rate']:>8.1f}%{'':>11} {r2['win_rate']-r1['win_rate']:>+.1f}pp
  {'-'*80}
  {'Total Trades':<25} {r1['total_trades']:>8}{'':>11} {r2['total_trades']:>8}{'':>11} {r2['total_trades']-r1['total_trades']:>+d}
  {'  Stock Trades':<25} {r1['stock_trades']:>8}{'':>11} {r2['stock_trades']:>8}
  {'  Option Trades':<25} {r1['option_trades']:>8}{'':>11} {r2['option_trades']:>8}{'':>11} {r2['option_trades']-r1['option_trades']:>+d}
  {'  Opt P&L US':<25} EUR {r1.get('option_pnl_us',0):>+9,.0f}{'':>6} EUR {r2.get('option_pnl_us',0):>+9,.0f}
  {'  Opt P&L EU':<25} EUR {r1.get('option_pnl_eu',0):>+9,.0f}{'':>6} EUR {r2.get('option_pnl_eu',0):>+9,.0f}
  {'-'*80}""")

    if 'total_rolls' in r2:
        print(f"""  ROLLING STATS (v13 only):
  {'  Roll attempts':<25} {r2['roll_attempts']:>8}       (opciones ganadoras a 45 DTE)
  {'  Rolls ejecutados':<25} {r2['total_rolls']:>8}       (con senal activa)
  {'  No-roll (sin senal)':<25} {r2['roll_no_signal']:>8}
  {'  Roll rate':<25} {r2['total_rolls']/max(r2['roll_attempts'],1)*100:>8.0f}%
  {'  Max chain':<25} {r2['max_chain_length']:>8}       rolls consecutivos
  {'  Trades rolados':<25} {r2['rolled_trades']:>8}       P&L EUR {r2['rolled_pnl']:>+,.0f}
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backtest v13 — Rolling Thunder')
    parser.add_argument('--months', type=int, default=24, help='Meses de historico (default: 24)')
    parser.add_argument('--verbose', action='store_true', help='Detalle de trades')
    args = parser.parse_args()

    months = args.months
    v = args.verbose

    print(f"""
{'='*70}
  BACKTEST v13 — ROLLING THUNDER
{'='*70}
  Base: v12 (US2 + EU2, slots separados)
  Cambio: opciones ganadoras rolan a mismo strike, 120 DTE
  Condicion: P&L > 0 AND senal LONG activa
  Periodo: {months} meses
{'='*70}
    """)

    tickers = list(ASSETS.keys())

    # =========================================================
    # 1. REF: v12 sin rolling (US2 + EU2)
    # =========================================================
    print(f"\n{'='*70}")
    print(f"  EJECUTANDO v12 REF (sin rolling)")
    print(f"{'='*70}")

    ref = run_backtest_eu(
        months=months,
        tickers=tickers,
        label=f"v12 REF: US2+EU2 (no roll)",
        use_options=True,
        options_eligible_set=OPTIONS_ALL,
        max_us_options=2,
        max_eu_options=2,
        macro_exempt_set=MACRO_EXEMPT,
        verbose=v,
    )

    # =========================================================
    # 2. v13: Rolling Thunder (US2 + EU2)
    # =========================================================
    print(f"\n{'='*70}")
    print(f"  EJECUTANDO v13 ROLLING THUNDER")
    print(f"{'='*70}")

    rolling = run_backtest_rolling(
        months=months,
        tickers=tickers,
        label=f"v13 Rolling: US2+EU2",
        use_options=True,
        options_eligible_set=OPTIONS_ALL,
        max_us_options=2,
        max_eu_options=2,
        macro_exempt_set=MACRO_EXEMPT,
        verbose=v,
    )

    # =========================================================
    # 3. COMPARATIVA
    # =========================================================
    if 'error' not in ref and 'error' not in rolling:
        print_rolling_comparison(ref, rolling)

    print(f"\n  {'='*70}")
    print(f"  CONCLUSION: v13 Rolling Thunder — {months} meses")
    print(f"  {'='*70}")

    if 'error' not in ref and 'error' not in rolling:
        delta_cagr = rolling['annualized_return_pct'] - ref['annualized_return_pct']
        delta_dd = rolling['max_drawdown'] - ref['max_drawdown']
        delta_pf = rolling['profit_factor'] - ref['profit_factor']

        if delta_cagr > 0 and delta_dd <= 5:
            verdict = "POSITIVO — Rolling mejora CAGR sin deteriorar drawdown"
        elif delta_cagr > 0 and delta_dd > 5:
            verdict = "MIXTO — Rolling mejora CAGR pero aumenta drawdown"
        elif delta_cagr <= 0:
            verdict = "NEGATIVO — Rolling no mejora, mantener v12"
        else:
            verdict = "NEUTRAL"

        print(f"     ΔCAGR:  {delta_cagr:+.1f}pp")
        print(f"     ΔMaxDD: {delta_dd:+.1f}pp")
        print(f"     ΔPF:    {delta_pf:+.2f}")
        print(f"     Rolls:  {rolling.get('total_rolls', 0)} ejecutados")
        print(f"     Veredicto: {verdict}")


if __name__ == '__main__':
    main()
