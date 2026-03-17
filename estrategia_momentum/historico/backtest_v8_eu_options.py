#!/usr/bin/env python3
"""
BACKTEST v8 EU OPTIONS — Test de opciones europeas con spread diferenciado

Creado: 2026-02-27
Objetivo: Evaluar si expandir OPTIONS_ELIGIBLE a tickers europeos mejora el CAGR,
          incluso con spreads mas amplios (10% EU vs 3% US).

Compara:
  REF:  v8 referencia (opciones solo US, 104 tickers, spread 3%)
  EU:   v8 + opciones EU (~30 tickers adicionales, spread 10%)

Cambios respecto a backtest_experimental.py:
  1. OPTIONS_ELIGIBLE_EU: ~30 tickers europeos con opciones disponibles
  2. OptionTradeV2EU: almacena spread_pct por trade (no global)
  3. get_option_spread(ticker): devuelve spread segun region
  4. run_backtest modificado: usa spread diferenciado

Notas:
  - Opciones europeas confirmadas disponibles en DEGIRO (ej: NESN.SW, AI.PA)
  - Spreads EU tipicamente 8-12%, usamos 10% conservador
  - El backtest usa contratos fraccionales (ficcion necesaria a EUR 10K)
  - En paper trading real, solo tickers con precio accesible (stock * 0.09 * 100 < budget)

Uso:
  python3 backtest_v8_eu_options.py --months 240
  python3 backtest_v8_eu_options.py --months 240 --verbose
  python3 backtest_v8_eu_options.py --months 36 --verbose
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


# =============================================================================
# OPCIONES EUROPEAS — TICKERS ELEGIBLES
# =============================================================================
# Criterios de inclusion:
#   1. Esta en ASSETS (universo 225 tickers)
#   2. Opciones disponibles en exchanges europeos (Eurex, Euronext, LSE, SIX)
#   3. Volumen de opciones suficiente para al menos 1 contrato
#   4. Confirmado por screenshots DEGIRO: NESN.SW, AI.PA

# Spread por region (half-turn, bid-ask)
US_SPREAD_PCT = 3.0      # Spread US: ~3% (muy liquido)
EU_SPREAD_PCT = 10.0     # Spread EU: ~10% (menos liquido, confirmado DEGIRO)

OPTIONS_ELIGIBLE_EU = [
    # Eurex — Alemania
    'SIE.DE',    # Siemens
    'ALV.DE',    # Allianz
    'DTE.DE',    # Deutsche Telekom
    'MUV2.DE',   # Munich Re
    'BAS.DE',    # BASF
    'BMW.DE',    # BMW
    'MBG.DE',    # Mercedes-Benz
    'ADS.DE',    # Adidas
    'IFX.DE',    # Infineon

    # Euronext — Francia
    'OR.PA',     # L'Oreal
    'MC.PA',     # LVMH
    'SAN.PA',    # Sanofi
    'AI.PA',     # Air Liquide (confirmado DEGIRO)
    'BNP.PA',    # BNP Paribas
    'SU.PA',     # Schneider Electric
    'AIR.PA',    # Airbus
    'CS.PA',     # AXA
    'DG.PA',     # Vinci
    'RI.PA',     # Pernod Ricard

    # Euronext — Holanda/Belgica
    'INGA.AS',   # ING Group
    'PHIA.AS',   # Philips
    'AD.AS',     # Ahold Delhaize
    'KBC.BR',    # KBC Group
    'ABI.BR',    # AB InBev

    # Borsa Italiana
    'ENEL.MI',   # Enel
    'ISP.MI',    # Intesa Sanpaolo
    'UCG.MI',    # UniCredit
    'ENI.MI',    # Eni

    # LSE — Reino Unido
    'ULVR.L',    # Unilever
    'LSEG.L',    # London Stock Exchange
    'BATS.L',    # BAT
    'DGE.L',     # Diageo

    # SIX — Suiza (Eurex)
    'NESN.SW',   # Nestle (confirmado DEGIRO)
    'ROG.SW',    # Roche
    'NOVN.SW',   # Novartis
    'UBSG.SW',   # UBS
    'ZURN.SW',   # Zurich Insurance
    'ABBN.SW',   # ABB

    # Nordicos (OMX)
    'ERIC-B.ST', # Ericsson
]

# Combinado: todos los tickers option-eligible
OPTIONS_ALL = OPTIONS_ELIGIBLE + OPTIONS_ELIGIBLE_EU


def get_option_spread(ticker):
    """Devuelve el spread (%) segun la region del ticker."""
    if ticker in OPTIONS_ELIGIBLE_EU:
        return EU_SPREAD_PCT
    return US_SPREAD_PCT


# =============================================================================
# OPTION TRADE V2 — MODIFICADO CON SPREAD POR TICKER
# =============================================================================
# Cambio clave: spread_pct se almacena en cada trade, no se lee de CONFIG.
# Esto permite usar 3% para US y 10% para EU en la misma ejecucion.

@dataclass
class OptionTradeV2EU:
    ticker: str
    entry_date: datetime
    entry_stock_price: float
    strike: float
    dte_at_entry: int
    entry_option_price: float
    entry_iv: float
    num_contracts: float
    position_euros: float   # premium pagada = max loss
    spread_pct: float       # NUEVO: spread especifico del ticker

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

    def update(self, stock_price, current_iv, days_elapsed):
        self.bars_held += 1

        remaining_dte = max(self.dte_at_entry - days_elapsed, 0)
        T = remaining_dte / 365.0

        bs = black_scholes_call(
            S=stock_price, K=self.strike, T=T,
            r=CONFIG['risk_free_rate'], sigma=current_iv
        )
        current_option_price = bs['price']
        # Usar spread del ticker, no de CONFIG
        current_option_price *= (1 - self.spread_pct / 100 / 2)

        self.max_option_value = max(self.max_option_value, current_option_price)

        option_return = (current_option_price / self.entry_option_price) - 1 if self.entry_option_price > 0 else 0
        self.max_r_mult = max(self.max_r_mult, option_return)

        # EXPIRACION
        if remaining_dte <= 0:
            intrinsic = max(stock_price - self.strike, 0)
            intrinsic *= (1 - self.spread_pct / 100 / 2)
            self._close(intrinsic, 'expiration')
            return {'type': 'full_exit', 'reason': 'expiration'}

        # CIERRE A 45 DTE RESTANTES
        if remaining_dte <= CONFIG.get('option_close_dte', 45):
            self._close(current_option_price, 'dte_exit')
            return {'type': 'full_exit', 'reason': 'dte_exit'}

        return None

    def _close(self, exit_option_price, reason):
        self.exit_option_price = exit_option_price
        self.exit_reason = reason
        self.pnl_euros = (exit_option_price - self.entry_option_price) * self.num_contracts * 100
        self.pnl_pct = ((exit_option_price / self.entry_option_price) - 1) * 100 if self.entry_option_price > 0 else 0


# =============================================================================
# RUN BACKTEST — MODIFICADO PARA SPREAD DIFERENCIADO
# =============================================================================

def run_backtest_eu(months, tickers, label, use_leverage_scaling=False,
                    use_options=False, options_eligible_set=None,
                    max_us_options=2, max_eu_options=0,
                    macro_exempt_set=None, verbose=False):
    """
    Backtest con soporte para spread diferenciado por ticker.
    Slots de opciones SEPARADOS para US y EU (no compiten entre si).

    Args:
        options_eligible_set: set/list de tickers elegibles para opciones.
        max_us_options: max slots para opciones US (default: 2)
        max_eu_options: max slots para opciones EU (default: 0 = sin EU)
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
    active_options = {}      # ticker -> OptionTradeV2EU
    all_trades = []
    all_option_trades = []

    # Contadores EU vs US (slots separados)
    option_opens_us = 0
    option_opens_eu = 0
    open_options_us = 0  # actualmente abiertas US
    open_options_eu = 0  # actualmente abiertas EU

    # =================================================================
    # LOOP PRINCIPAL
    # =================================================================
    for current_date in all_dates:

        # 1. GESTIONAR TRADES ACTIVOS (acciones/ETFs)
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

        # 2. GESTIONAR OPCIONES ACTIVAS
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
            iv = df['HVOL'].iloc[idx]
            if pd.isna(iv) or iv <= 0:
                iv = opt.entry_iv
            result = opt.update(bar['Close'], iv, days_elapsed)
            if result and result['type'] == 'full_exit':
                opt.exit_date = current_date
                options_to_close.append(ticker)
                tracker.update_equity(opt.pnl_euros, current_date)

        for ticker in options_to_close:
            opt = active_options.pop(ticker)
            tracker.open_positions -= 1
            tracker.open_options -= 1
            # Actualizar contadores separados
            if ticker in OPTIONS_ELIGIBLE_EU:
                open_options_eu -= 1
            else:
                open_options_us -= 1
            all_option_trades.append(opt)
            if verbose:
                pnl_pct = (opt.pnl_euros / opt.position_euros * 100) if opt.position_euros else 0
                pnl_sign = '+' if opt.pnl_euros >= 0 else ''
                region = "EU" if opt.ticker in OPTIONS_ELIGIBLE_EU else "US"
                print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE OPT {ticker:8} [{region}] | "
                      f"{opt.exit_reason:<15} | P&L EUR {pnl_sign}{opt.pnl_euros:.0f} ({pnl_sign}{pnl_pct:.1f}%) | "
                      f"Pos: {tracker.open_positions}/10 | Equity: EUR {tracker.equity:,.0f}")

        # 3. BUSCAR NUEVAS SENALES
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
                # SLOTS SEPARADOS: US options (max_us_options) y EU options (max_eu_options)
                open_as_option = False
                current_ivr = None
                is_eu_ticker = ticker in OPTIONS_ELIGIBLE_EU
                if use_options and ticker in options_eligible_set:
                    # Verificar slot disponible segun region
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
                    # --- OPCION CALL (spread diferenciado por ticker) ---
                    stock_price = bar['Open']
                    strike = stock_price * (1 - CONFIG['option_itm_pct'])  # 5% ITM

                    actual_dte = monthly_expiration_dte(current_date, CONFIG['option_dte'])
                    T = actual_dte / 365.0

                    iv = df['HVOL'].iloc[idx]
                    if pd.isna(iv) or iv <= 0:
                        iv = 0.30  # fallback
                    bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], iv)
                    option_price = bs['price']

                    # SPREAD DIFERENCIADO: US 3% vs EU 10%
                    ticker_spread = get_option_spread(ticker)
                    option_price *= (1 + ticker_spread / 100 / 2)  # spread de entrada

                    size = tracker.get_option_size(option_price)
                    if size['premium'] < 50:
                        continue

                    opt = OptionTradeV2EU(
                        ticker=ticker,
                        entry_date=current_date,
                        entry_stock_price=stock_price,
                        strike=strike,
                        dte_at_entry=actual_dte,
                        entry_option_price=option_price,
                        entry_iv=iv,
                        num_contracts=size['contracts'],
                        position_euros=size['premium'],
                        spread_pct=ticker_spread,  # NUEVO: spread por trade
                    )
                    active_options[ticker] = opt
                    tracker.open_positions += 1
                    tracker.open_options += 1

                    # Contador US/EU (slots separados)
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
                    # --- ACCION/ETF ---
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
""")

    # Razones de salida
    exit_reasons = {}
    for t in combined_trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    print("  RAZONES DE SALIDA:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"     {reason:20} {count:3} ({count/total_count*100:.1f}%)")

    # Detalle opciones EU si hay
    if opt_eu:
        print(f"\n  DETALLE OPCIONES EU ({len(opt_eu)} trades, spread {EU_SPREAD_PCT}%):")
        for i, opt in enumerate(sorted(opt_eu, key=lambda x: x.entry_date), 1):
            entry_str = opt.entry_date.strftime('%Y-%m-%d') if opt.entry_date else '?'
            exit_str = opt.exit_date.strftime('%Y-%m-%d') if opt.exit_date else '?'
            marker = '+' if opt.pnl_euros > 0 else '-'
            print(f"     {i}. {entry_str} -> {exit_str} | {opt.ticker:10} | "
                  f"K=${opt.strike:.2f} | Prem ${opt.entry_option_price:.2f} -> ${opt.exit_option_price:.2f} | "
                  f"P&L EUR {opt.pnl_euros:+.0f} ({opt.pnl_pct:+.1f}%) | {opt.bars_held}d | {opt.exit_reason} {marker}")

    # Detalle opciones US
    if opt_us and verbose:
        print(f"\n  DETALLE OPCIONES US ({len(opt_us)} trades, spread {US_SPREAD_PCT}%):")
        for i, opt in enumerate(sorted(opt_us, key=lambda x: x.entry_date), 1):
            entry_str = opt.entry_date.strftime('%Y-%m-%d') if opt.entry_date else '?'
            exit_str = opt.exit_date.strftime('%Y-%m-%d') if opt.exit_date else '?'
            marker = '+' if opt.pnl_euros > 0 else '-'
            print(f"     {i}. {entry_str} -> {exit_str} | {opt.ticker:10} | "
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
    }


# =============================================================================
# TABLA COMPARATIVA
# =============================================================================

def print_comparison(results):
    print(f"""
{'='*100}
  COMPARATIVA: OPCIONES US-ONLY vs US+EU
{'='*100}

  {'Variante':<30} {'Trades':<8} {'Win%':<7} {'PnL EUR':<12} {'Return%':<9} {'Annual%':<9} {'PF':<6} {'MaxDD%':<7}
  {'-'*95}""")

    for r in results:
        print(f"  {r['label']:<30} {r['total_trades']:<8} {r['win_rate']:<7.1f} "
              f"EUR{r['total_pnl_euros']:>+9,.0f}  {r['total_return_pct']:>+8.1f}%  "
              f"{r['annualized_return_pct']:>+7.1f}%  {r['profit_factor']:.2f}  {r['max_drawdown']:>5.1f}%")

    print(f"""
  {'Variante':<30} {'Stocks':<8} {'Opts':<6} {'OptUS':<6} {'OptEU':<6} {'>3R':<5} {'OptHR':<6} {'PnL OptUS':<11} {'PnL OptEU':<11}
  {'-'*95}""")

    for r in results:
        print(f"  {r['label']:<30} {r['stock_trades']:<8} {r['option_trades']:<6} "
              f"{r.get('option_trades_us', r['option_trades']):<6} {r.get('option_trades_eu', 0):<6} "
              f"{r['stock_gt_3r']:<5} {r['opt_home_runs']:<6} "
              f"EUR{r.get('option_pnl_us', 0):>+8,.0f}  EUR{r.get('option_pnl_eu', 0):>+8,.0f}")

    print()

    # Delta analysis
    if len(results) >= 2:
        ref = results[0]
        test = results[1]
        delta_return = test['total_return_pct'] - ref['total_return_pct']
        delta_annual = test['annualized_return_pct'] - ref['annualized_return_pct']
        delta_dd = test['max_drawdown'] - ref['max_drawdown']
        delta_pf = test['profit_factor'] - ref['profit_factor']
        print(f"  DELTA (EU - REF):")
        print(f"     Return:     {delta_return:+.1f}pp")
        print(f"     Annualized: {delta_annual:+.1f}pp")
        print(f"     MaxDD:      {delta_dd:+.1f}pp")
        print(f"     PF:         {delta_pf:+.2f}")
        print(f"     Extra EU option trades: {test.get('option_trades_eu', 0)}")
        print(f"     EU option P&L: EUR {test.get('option_pnl_eu', 0):+,.0f}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backtest v8 EU Options — Opciones europeas con spread diferenciado')
    parser.add_argument('--months', type=int, default=240, help='Meses de historico (default: 240)')
    parser.add_argument('--verbose', action='store_true', help='Detalle de trades')
    parser.add_argument('--eu-spread', type=float, default=10.0, help='Spread EU en %% (default: 10.0)')
    parser.add_argument('--test-spreads', action='store_true', help='Test varios spreads EU: 5%%, 10%%, 15%%')
    parser.add_argument('--multi-period', action='store_true',
                        help='Test US2+EU0 vs US2+EU2 a 36,60,120,180,240 meses')
    args = parser.parse_args()

    global EU_SPREAD_PCT
    EU_SPREAD_PCT = args.eu_spread

    months = args.months
    v = args.verbose

    print(f"""
======================================================================
  BACKTEST v8 EU OPTIONS — OPCIONES EUROPEAS (SLOTS SEPARADOS)
======================================================================
  Referencia: 2 slots US ({len(OPTIONS_ELIGIBLE)} tickers, spread {US_SPREAD_PCT}%)
  Test:       2 slots US + 1-2 slots EU ({len(OPTIONS_ALL)} tickers, EU {EU_SPREAD_PCT}%)
  Tickers EU: {len(OPTIONS_ELIGIBLE_EU)}
  Periodo:    {months} meses
  NOTA:       slots separados = EU no desplaza US
======================================================================
    """)

    results = []

    if args.multi_period:
        # ============================================================
        # MULTI-PERIOD: REF (US2+EU0) vs US2+EU2 a distintos períodos
        # ============================================================
        periods = [6, 12, 36, 60]
        summary_rows = []

        print(f"""
======================================================================
  MULTI-PERIOD TEST: REF (US2+EU0) vs US2+EU2
  Períodos: {periods}
  EU spread: {EU_SPREAD_PCT}%
======================================================================
        """)

        for m in periods:
            print(f"\n{'#'*70}")
            print(f"  PERÍODO: {m} MESES ({m/12:.0f} años)")
            print(f"{'#'*70}")

            # REF: US2+EU0
            r_ref = run_backtest_eu(m, BASE_TICKERS, f"REF US2+EU0 ({m}m)",
                                    use_options=True,
                                    options_eligible_set=OPTIONS_ELIGIBLE,
                                    max_us_options=2, max_eu_options=0,
                                    verbose=False)

            # TEST: US2+EU2
            r_eu2 = run_backtest_eu(m, BASE_TICKERS, f"US2+EU2 ({m}m)",
                                    use_options=True,
                                    options_eligible_set=OPTIONS_ALL,
                                    max_us_options=2, max_eu_options=2,
                                    verbose=False)

            if 'error' not in r_ref and 'error' not in r_eu2:
                delta_cagr = r_eu2['annualized_return_pct'] - r_ref['annualized_return_pct']
                delta_dd = r_eu2['max_drawdown'] - r_ref['max_drawdown']
                delta_pf = r_eu2['profit_factor'] - r_ref['profit_factor']
                summary_rows.append({
                    'months': m,
                    'years': m / 12,
                    'ref_equity': r_ref['final_equity'],
                    'ref_cagr': r_ref['annualized_return_pct'],
                    'ref_dd': r_ref['max_drawdown'],
                    'ref_pf': r_ref['profit_factor'],
                    'ref_opt_us': r_ref.get('option_trades_us', r_ref['option_trades']),
                    'eu2_equity': r_eu2['final_equity'],
                    'eu2_cagr': r_eu2['annualized_return_pct'],
                    'eu2_dd': r_eu2['max_drawdown'],
                    'eu2_pf': r_eu2['profit_factor'],
                    'eu2_opt_us': r_eu2.get('option_trades_us', r_eu2['option_trades']),
                    'eu2_opt_eu': r_eu2.get('option_trades_eu', 0),
                    'eu2_pnl_eu': r_eu2.get('option_pnl_eu', 0),
                    'delta_cagr': delta_cagr,
                    'delta_dd': delta_dd,
                    'delta_pf': delta_pf,
                })

        # TABLA RESUMEN MULTI-PERIOD
        if summary_rows:
            print(f"""
{'='*110}
  RESUMEN MULTI-PERIOD: REF (US2+EU0) vs US2+EU2 (spread EU {EU_SPREAD_PCT}%)
{'='*110}

  {'Período':<10} {'REF Equity':<13} {'REF CAGR':<10} {'REF DD':<8} {'REF PF':<8} | {'EU2 Equity':<13} {'EU2 CAGR':<10} {'EU2 DD':<8} {'EU2 PF':<8} | {'ΔCAGR':<8} {'ΔDD':<8} {'ΔPF':<8}
  {'-'*108}""")

            for row in summary_rows:
                print(f"  {row['months']:>3}m ({row['years']:.0f}y)  "
                      f"EUR {row['ref_equity']:>10,.0f}  {row['ref_cagr']:>+7.1f}%  "
                      f"{row['ref_dd']:>5.1f}%  {row['ref_pf']:>5.2f}   | "
                      f"EUR {row['eu2_equity']:>10,.0f}  {row['eu2_cagr']:>+7.1f}%  "
                      f"{row['eu2_dd']:>5.1f}%  {row['eu2_pf']:>5.2f}   | "
                      f"{row['delta_cagr']:>+6.1f}pp  {row['delta_dd']:>+5.1f}pp  {row['delta_pf']:>+5.2f}")

            print(f"""
  {'Período':<10} {'Opt US ref':<12} {'Opt US eu2':<12} {'Opt EU':<8} {'PnL EU':<12}
  {'-'*60}""")
            for row in summary_rows:
                print(f"  {row['months']:>3}m ({row['years']:.0f}y)  "
                      f"{row['ref_opt_us']:>8}     {row['eu2_opt_us']:>8}     "
                      f"{row['eu2_opt_eu']:>5}    EUR {row['eu2_pnl_eu']:>+8,.0f}")

            print()

            # Consistencia check
            all_positive = all(r['delta_cagr'] > 0 for r in summary_rows)
            print(f"  CONSISTENCIA: ΔCAGR positivo en {'TODOS' if all_positive else 'NO todos'} los períodos "
                  f"({sum(1 for r in summary_rows if r['delta_cagr'] > 0)}/{len(summary_rows)})")
            avg_delta = np.mean([r['delta_cagr'] for r in summary_rows])
            avg_dd_delta = np.mean([r['delta_dd'] for r in summary_rows])
            print(f"  MEDIA ΔCAGR: {avg_delta:+.1f}pp | MEDIA ΔDD: {avg_dd_delta:+.1f}pp")
            print()

        return  # No seguir con el flujo normal

    elif args.test_spreads:
        # Test con varios spreads EU
        # Primero la referencia US-only (2 slots US, 0 EU)
        r_ref = run_backtest_eu(months, BASE_TICKERS, "REF: US-only (3%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ELIGIBLE,
                                max_us_options=2, max_eu_options=0,
                                verbose=v)
        if 'error' not in r_ref:
            results.append(r_ref)

        for spread in [5.0, 10.0, 15.0]:
            EU_SPREAD_PCT = spread
            r = run_backtest_eu(months, BASE_TICKERS, f"US2+EU1 (EU {spread}%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ALL,
                                max_us_options=2, max_eu_options=1,
                                verbose=v)
            if 'error' not in r:
                results.append(r)
    else:
        # Test principal: slots SEPARADOS (US no compite con EU)
        # A) Referencia: 2 slots US, 0 EU
        r_ref = run_backtest_eu(months, BASE_TICKERS, "REF: US2 EU0 (3%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ELIGIBLE,
                                max_us_options=2, max_eu_options=0,
                                verbose=v)
        if 'error' not in r_ref:
            results.append(r_ref)

        # B) 2 slots US + 1 slot EU (no compiten)
        r_eu1 = run_backtest_eu(months, BASE_TICKERS, f"US2+EU1 ({EU_SPREAD_PCT}%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ALL,
                                max_us_options=2, max_eu_options=1,
                                verbose=v)
        if 'error' not in r_eu1:
            results.append(r_eu1)

        # C) 2 slots US + 2 slots EU
        r_eu2 = run_backtest_eu(months, BASE_TICKERS, f"US2+EU2 ({EU_SPREAD_PCT}%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ALL,
                                max_us_options=2, max_eu_options=2,
                                verbose=v)
        if 'error' not in r_eu2:
            results.append(r_eu2)

    if len(results) >= 2:
        print_comparison(results)


if __name__ == "__main__":
    main()
