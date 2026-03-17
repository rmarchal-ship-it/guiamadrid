#!/usr/bin/env python3
"""
BACKTEST v12 — Momentum Breakout v8 + Opciones EU (slots separados)

Version: v12 (semidefinitiva, 27 Feb 2026)
Mod 01 Mar 2026: +120m en multi-period, +Sharpe ratio, +equity curve en gold overlay
Base: v8 (backtest_experimental.py)
Cambio: Expansion opciones a Europa con spread diferenciado y slots independientes

Configuracion v12:
  - Motor de senales: identico a v8 (RSI 50, RSI max 75, KER 0.40, VOL 1.3, BP 20)
  - Opciones US: 104 tickers, spread 3%, max 2 slots (IBKR)
  - Opciones EU: 39 tickers, spread 10%, max 2 slots (DEGIRO)
  - Slots SEPARADOS: EU nunca desplaza US
  - Gold 30% overlay: NO incluido en backtest (capa operativa de portfolio allocation)

Resultados validados multi-period:
  | Periodo | REF CAGR | EU2 CAGR | ΔCAGR   |
  |---------|----------|----------|---------|
  |  6m     | +104.8%  | +135.6%  | +30.8pp |
  | 12m     |  +68.7%  |  +47.7%  | -21.0pp |  <-- unico periodo negativo
  | 36m     |  +81.8%  | +113.9%  | +32.1pp |
  | 60m     |  +43.6%  |  +84.7%  | +41.1pp |
  | 240m    |  +30.4%  |  +36.3%  |  +5.9pp |

Archivos relacionados:
  - backtest_experimental.py — v8 base (imports)
  - momentum_breakout.py — motor de senales, ASSETS 225 tickers
  - backtest_v8_eu_options.py — version de desarrollo (este archivo renombrado)

Uso:
  python3 backtest_v12_eu_options.py --months 240
  python3 backtest_v12_eu_options.py --multi-period
  python3 backtest_v12_eu_options.py --test-spreads
  python3 backtest_v12_eu_options.py --months 60 --verbose
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
    save_data_cache, load_data_cache, list_data_caches,
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
                    macro_exempt_set=None, verbose=False,
                    preloaded_data=None, download_fn=None):
    """
    Backtest con soporte para spread diferenciado por ticker.
    Slots de opciones SEPARADOS para US y EU (no compiten entre si).

    Args:
        options_eligible_set: set/list de tickers elegibles para opciones.
        max_us_options: max slots para opciones US (default: 2)
        max_eu_options: max slots para opciones EU (default: 0 = sin EU)
        preloaded_data: dict {ticker: DataFrame} precargado (cache). Si se pasa, no descarga.
        download_fn: función de descarga (default: download_data de yfinance).
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

    if preloaded_data is not None:
        # Usar datos precargados (cache)
        all_data = {}
        for ticker in tickers:
            if ticker in preloaded_data:
                df = preloaded_data[ticker].copy()
                if 'ATR' not in df.columns:
                    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
                if 'HVOL' not in df.columns:
                    df['HVOL'] = historical_volatility(df['Close'], CONFIG['hvol_window'])
                all_data[ticker] = df
        print(f"  Datos precargados: {len(all_data)}/{n_tickers} tickers")
    else:
        # Descargar datos frescos
        if download_fn is None:
            download_fn = download_data
        print("  Descargando datos...")
        all_data = {}
        failed = []
        for i, ticker in enumerate(tickers):
            df = download_fn(ticker, months)
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
# SHARPE RATIO
# =============================================================================

def compute_sharpe(equity_curve, initial_capital=None):
    """Calcula Sharpe ratio anualizado desde equity curve [(date, equity), ...].
    Convierte a serie diaria con ffill para comparabilidad."""
    if not equity_curve or len(equity_curve) < 10:
        return 0.0
    dates = [e[0] for e in equity_curve]
    values = [e[1] for e in equity_curve]
    ts = pd.Series(values, index=pd.DatetimeIndex(dates))
    ts = ts[~ts.index.duplicated(keep='last')]
    full_range = pd.bdate_range(ts.index.min(), ts.index.max())
    if initial_capital:
        first_day = full_range[0]
        if first_day not in ts.index:
            ts.loc[first_day] = initial_capital
            ts = ts.sort_index()
    ts = ts.reindex(full_range).ffill().bfill()
    daily_returns = ts.pct_change().dropna()
    if len(daily_returns) < 10 or daily_returns.std() == 0:
        return 0.0
    return float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))


# =============================================================================
# GOLD 30% OVERLAY (post-hoc, de test_v10_gold_hedge.py)
# =============================================================================

def simulate_gold_overlay(result, gld_data, gold_reserve_pct=0.30, initial_capital=None):
    """
    Overlay post-hoc: 30% equity siempre en GLD + cash idle en GLD.
    Momentum P&L escalado al 70%.
    """
    from collections import defaultdict

    if initial_capital is None:
        initial_capital = CONFIG['initial_capital']

    max_positions = CONFIG['max_positions']
    trading_pct = 1.0 - gold_reserve_pct

    all_trades = result.get('all_trades', []) + result.get('all_option_trades', [])
    if not all_trades:
        return None

    gld_close = gld_data['Close'].copy()
    if isinstance(gld_close, pd.DataFrame):
        gld_close = gld_close.squeeze()
    gld_returns = gld_close.pct_change().fillna(0)

    close_pnl_by_date = defaultdict(float)
    open_delta_by_date = defaultdict(int)

    for t in all_trades:
        if t.entry_date:
            open_delta_by_date[pd.Timestamp(t.entry_date)] += 1
        if t.exit_date:
            close_pnl_by_date[pd.Timestamp(t.exit_date)] += t.pnl_euros
            open_delta_by_date[pd.Timestamp(t.exit_date)] -= 1

    all_entry_dates = [pd.Timestamp(t.entry_date) for t in all_trades if t.entry_date]
    all_exit_dates = [pd.Timestamp(t.exit_date) for t in all_trades if t.exit_date]
    start_date = min(all_entry_dates)
    end_date = max(all_exit_dates + all_entry_dates)

    trading_days = pd.bdate_range(start_date, end_date)

    equity_gold = initial_capital
    max_eq_gold = initial_capital
    max_dd_gold = 0.0
    open_positions = 0
    gold_total_pnl = 0.0
    gold_equity_curve = []

    for day in trading_days:
        raw_pnl = close_pnl_by_date.get(day, 0.0)
        scaled_pnl = raw_pnl * trading_pct

        delta = open_delta_by_date.get(day, 0)
        open_positions += delta
        open_positions = max(0, open_positions)

        invested_pct = min(open_positions / max_positions, 1.0) * trading_pct
        gold_pct = max(gold_reserve_pct, 1.0 - invested_pct)

        gld_ret = 0.0
        if day in gld_returns.index:
            gld_ret = gld_returns.loc[day]
            if isinstance(gld_ret, pd.Series):
                gld_ret = gld_ret.iloc[0]

        gold_pnl = equity_gold * gold_pct * gld_ret
        equity_gold += scaled_pnl + gold_pnl
        gold_total_pnl += gold_pnl

        gold_equity_curve.append((day, equity_gold))

        max_eq_gold = max(max_eq_gold, equity_gold)
        dd_gold = (max_eq_gold - equity_gold) / max_eq_gold * 100
        max_dd_gold = max(max_dd_gold, dd_gold)

    days = (end_date - start_date).days
    years = days / 365.25
    ann_gold = ((equity_gold / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        'equity_gold': equity_gold,
        'ann_gold': ann_gold,
        'maxdd_gold': max_dd_gold,
        'gold_total_pnl': gold_total_pnl,
        'equity_curve': gold_equity_curve,
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
                        help='Test US2+EU0 vs US2+EU2 a distintos periodos')
    parser.add_argument('--gold', action='store_true',
                        help='Aplicar Gold 30%% overlay post-hoc')
    parser.add_argument('--gold-pct', type=float, default=0.30,
                        help='Fraccion en oro (default: 0.30)')
    parser.add_argument('--export-csv', action='store_true',
                        help='Exportar trades completos a CSV + PnL por año')
    parser.add_argument('--save-cache', action='store_true',
                        help='Guardar datos descargados en cache para reproducibilidad')
    parser.add_argument('--cache-date', type=str, default=None,
                        help='Cargar datos de cache existente (formato: YYYY-MM-DD)')
    parser.add_argument('--list-caches', action='store_true',
                        help='Listar caches disponibles y salir')
    parser.add_argument('--data-source', choices=['yahoo', 'eodhd'], default='yahoo',
                        help='Fuente de datos: yahoo (default) o eodhd')
    args = parser.parse_args()

    if args.list_caches:
        caches = list_data_caches()
        if caches:
            print("\n  Caches disponibles:")
            for name, meta in caches:
                print(f"    {name}")
                for line in meta.split('\n'):
                    print(f"      {line}")
        else:
            print("  No hay caches guardados.")
        sys.exit(0)

    global EU_SPREAD_PCT
    EU_SPREAD_PCT = args.eu_spread

    months = args.months
    v = args.verbose
    source = args.data_source

    # Seleccionar función de descarga según fuente
    if source == 'eodhd':
        from data_eodhd import download_data_eodhd
        download_fn = download_data_eodhd
    else:
        download_fn = download_data

    print(f"""
======================================================================
  BACKTEST v8 EU OPTIONS — OPCIONES EUROPEAS (SLOTS SEPARADOS)
======================================================================
  Referencia: 2 slots US ({len(OPTIONS_ELIGIBLE)} tickers, spread {US_SPREAD_PCT}%)
  Test:       2 slots US + 1-2 slots EU ({len(OPTIONS_ALL)} tickers, EU {EU_SPREAD_PCT}%)
  Tickers EU: {len(OPTIONS_ELIGIBLE_EU)}
  Periodo:    {months} meses
  Fuente:     {source.upper()}
  NOTA:       slots separados = EU no desplaza US
======================================================================
    """)

    results = []

    # --- Preload datos (cache o descarga fresca, una sola vez) ---
    preloaded = None
    if not args.multi_period:  # multi-period usa distintos months, no se puede precargar
        if args.cache_date:
            preloaded = load_data_cache(args.cache_date, months, source=source)
            if preloaded is None:
                print(f"  ERROR: Cache {args.cache_date} no encontrado para {months}m ({source})")
                return
        else:
            print(f"  Descargando datos de {source.upper()} (una sola vez para todas las variantes)...")
            preloaded = {}
            failed = []
            fallback_used = []
            for i, ticker in enumerate(BASE_TICKERS):
                df = download_fn(ticker, months)
                if df is None and source == 'eodhd':
                    # Fallback a Yahoo para tickers que EODHD no tiene
                    df = download_data(ticker, months)
                    if df is not None:
                        fallback_used.append(ticker)
                if df is not None:
                    preloaded[ticker] = df
                else:
                    failed.append(ticker)
                if (i + 1) % 20 == 0 or i == len(BASE_TICKERS) - 1:
                    print(f"\r  Descargados: {len(preloaded)}/{len(BASE_TICKERS)} OK, {len(failed)} fallidos", end='')
            print(f"\n  Tickers con datos: {len(preloaded)}")
            if fallback_used:
                print(f"  Fallback Yahoo: {', '.join(fallback_used[:15])}{'...' if len(fallback_used) > 15 else ''} ({len(fallback_used)} tickers)")
            if failed:
                print(f"  Fallidos: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
            if args.save_cache:
                save_data_cache(preloaded, months, source=source)

    if args.multi_period:
        # ============================================================
        # MULTI-PERIOD: REF (US2+EU0) vs US2+EU2, con/sin Gold overlay
        # ============================================================
        periods = [6, 12, 36, 60, 120]
        summary_rows = []
        use_gold = args.gold
        gold_pct = args.gold_pct

        gold_label = f" + Gold {gold_pct*100:.0f}%" if use_gold else ""

        print(f"""
======================================================================
  MULTI-PERIOD TEST: REF (US2+EU0) vs US2+EU2{gold_label}
  Períodos: {periods}
  EU spread: {EU_SPREAD_PCT}%
  Gold overlay: {'SI (' + str(gold_pct*100) + '%)' if use_gold else 'NO'}
======================================================================
        """)

        # Descargar GLD si gold overlay
        gld_data = None
        if use_gold:
            print("  Descargando GLD para overlay...")
            gld_data = download_fn('GLD', max(periods) + 6)
            if gld_data is None:
                print("  ERROR: No se pudo descargar GLD. Abortando.")
                return
            print(f"  GLD: {len(gld_data)} dias")

        for m in periods:
            print(f"\n{'#'*70}")
            print(f"  PERÍODO: {m} MESES ({m/12:.0f} años)")
            print(f"{'#'*70}")

            # REF: US2+EU0
            r_ref = run_backtest_eu(m, BASE_TICKERS, f"REF US2+EU0 ({m}m)",
                                    use_options=True,
                                    options_eligible_set=OPTIONS_ELIGIBLE,
                                    max_us_options=2, max_eu_options=0,
                                    verbose=False, download_fn=download_fn)

            # TEST: US2+EU2
            r_eu2 = run_backtest_eu(m, BASE_TICKERS, f"US2+EU2 ({m}m)",
                                    use_options=True,
                                    options_eligible_set=OPTIONS_ALL,
                                    max_us_options=2, max_eu_options=2,
                                    verbose=False, download_fn=download_fn)

            if 'error' not in r_ref and 'error' not in r_eu2:
                row = {
                    'months': m,
                    'years': m / 12,
                    'ref_equity': r_ref['final_equity'],
                    'ref_cagr': r_ref['annualized_return_pct'],
                    'ref_dd': r_ref['max_drawdown'],
                    'ref_pf': r_ref['profit_factor'],
                    'ref_sharpe': compute_sharpe(r_ref.get('equity_curve', []), CONFIG['initial_capital']),
                    'ref_opt_us': r_ref.get('option_trades_us', r_ref['option_trades']),
                    'eu2_equity': r_eu2['final_equity'],
                    'eu2_cagr': r_eu2['annualized_return_pct'],
                    'eu2_dd': r_eu2['max_drawdown'],
                    'eu2_pf': r_eu2['profit_factor'],
                    'eu2_sharpe': compute_sharpe(r_eu2.get('equity_curve', []), CONFIG['initial_capital']),
                    'eu2_opt_us': r_eu2.get('option_trades_us', r_eu2['option_trades']),
                    'eu2_opt_eu': r_eu2.get('option_trades_eu', 0),
                    'eu2_pnl_eu': r_eu2.get('option_pnl_eu', 0),
                }

                # Gold overlay si activado
                if use_gold and gld_data is not None:
                    g_ref = simulate_gold_overlay(r_ref, gld_data, gold_pct)
                    g_eu2 = simulate_gold_overlay(r_eu2, gld_data, gold_pct)
                    if g_ref and g_eu2:
                        row['ref_gold_equity'] = g_ref['equity_gold']
                        row['ref_gold_cagr'] = g_ref['ann_gold']
                        row['ref_gold_dd'] = g_ref['maxdd_gold']
                        row['ref_gold_pnl'] = g_ref['gold_total_pnl']
                        row['ref_gold_sharpe'] = compute_sharpe(g_ref.get('equity_curve', []), CONFIG['initial_capital'])
                        row['eu2_gold_equity'] = g_eu2['equity_gold']
                        row['eu2_gold_cagr'] = g_eu2['ann_gold']
                        row['eu2_gold_dd'] = g_eu2['maxdd_gold']
                        row['eu2_gold_pnl'] = g_eu2['gold_total_pnl']
                        row['eu2_gold_sharpe'] = compute_sharpe(g_eu2.get('equity_curve', []), CONFIG['initial_capital'])

                row['delta_cagr'] = r_eu2['annualized_return_pct'] - r_ref['annualized_return_pct']
                row['delta_dd'] = r_eu2['max_drawdown'] - r_ref['max_drawdown']
                row['delta_pf'] = r_eu2['profit_factor'] - r_ref['profit_factor']

                summary_rows.append(row)

        # TABLA RESUMEN MULTI-PERIOD
        if summary_rows:
            # --- Tabla sin gold ---
            print(f"""
{'='*110}
  RESUMEN MULTI-PERIOD: REF (US2+EU0) vs US2+EU2 (spread EU {EU_SPREAD_PCT}%)
{'='*110}

  {'Período':<10} {'REF Equity':<13} {'REF CAGR':<10} {'REF DD':<8} {'REF PF':<8} {'REF Sh':<7} | {'EU2 Equity':<13} {'EU2 CAGR':<10} {'EU2 DD':<8} {'EU2 PF':<8} {'EU2 Sh':<7} | {'ΔCAGR':<8} {'ΔDD':<8}
  {'-'*120}""")

            for row in summary_rows:
                print(f"  {row['months']:>3}m ({row['years']:.0f}y)  "
                      f"EUR {row['ref_equity']:>10,.0f}  {row['ref_cagr']:>+7.1f}%  "
                      f"{row['ref_dd']:>5.1f}%  {row['ref_pf']:>5.2f}  {row.get('ref_sharpe', 0):>5.2f}  | "
                      f"EUR {row['eu2_equity']:>10,.0f}  {row['eu2_cagr']:>+7.1f}%  "
                      f"{row['eu2_dd']:>5.1f}%  {row['eu2_pf']:>5.2f}  {row.get('eu2_sharpe', 0):>5.2f}  | "
                      f"{row['delta_cagr']:>+6.1f}pp  {row['delta_dd']:>+5.1f}pp")

            # --- Tabla con gold overlay ---
            if use_gold and 'ref_gold_cagr' in summary_rows[0]:
                print(f"""
{'='*110}
  CON GOLD {gold_pct*100:.0f}% OVERLAY
{'='*110}

  {'Período':<10} {'REF+G Eq':<14} {'REF+G CAGR':<11} {'REF+G DD':<9} {'REF+G Sh':<9} | {'EU2+G Eq':<14} {'EU2+G CAGR':<11} {'EU2+G DD':<9} {'EU2+G Sh':<9} | {'ΔCAGR':<8} {'ΔDD':<8}
  {'-'*120}""")

                for row in summary_rows:
                    delta_g_cagr = row['eu2_gold_cagr'] - row['ref_gold_cagr']
                    delta_g_dd = row['eu2_gold_dd'] - row['ref_gold_dd']
                    print(f"  {row['months']:>3}m ({row['years']:.0f}y)  "
                          f"EUR {row['ref_gold_equity']:>10,.0f}   {row['ref_gold_cagr']:>+7.1f}%   "
                          f"{row['ref_gold_dd']:>5.1f}%   {row.get('ref_gold_sharpe', 0):>5.2f}   | "
                          f"EUR {row['eu2_gold_equity']:>10,.0f}   {row['eu2_gold_cagr']:>+7.1f}%   "
                          f"{row['eu2_gold_dd']:>5.1f}%   {row.get('eu2_gold_sharpe', 0):>5.2f}   | "
                          f"{delta_g_cagr:>+6.1f}pp  {delta_g_dd:>+5.1f}pp")

                # Tabla eficiencia (CAGR/DD)
                print(f"""
  {'Período':<10} {'REF Eff':<10} {'REF+G Eff':<10} {'EU2 Eff':<10} {'EU2+G Eff':<10} | {'REF Sh':<8} {'REF+G Sh':<8} {'EU2 Sh':<8} {'EU2+G Sh':<8} | {'Gold PnL EU2':<14}
  {'-'*120}""")

                for row in summary_rows:
                    eff_ref = row['ref_cagr'] / row['ref_dd'] if row['ref_dd'] > 0 else 0
                    eff_ref_g = row['ref_gold_cagr'] / row['ref_gold_dd'] if row['ref_gold_dd'] > 0 else 0
                    eff_eu2 = row['eu2_cagr'] / row['eu2_dd'] if row['eu2_dd'] > 0 else 0
                    eff_eu2_g = row['eu2_gold_cagr'] / row['eu2_gold_dd'] if row['eu2_gold_dd'] > 0 else 0
                    print(f"  {row['months']:>3}m ({row['years']:.0f}y)  "
                          f"{eff_ref:>8.2f}    {eff_ref_g:>8.2f}    "
                          f"{eff_eu2:>8.2f}    {eff_eu2_g:>8.2f}    | "
                          f"{row.get('ref_sharpe', 0):>6.2f}  {row.get('ref_gold_sharpe', 0):>6.2f}  "
                          f"{row.get('eu2_sharpe', 0):>6.2f}  {row.get('eu2_gold_sharpe', 0):>6.2f}  | "
                          f"EUR {row['eu2_gold_pnl']:>+8,.0f}")

            # --- Opciones desglose ---
            print(f"""
  {'Período':<10} {'Opt US ref':<12} {'Opt US eu2':<12} {'Opt EU':<8} {'PnL EU':<12}
  {'-'*60}""")
            for row in summary_rows:
                print(f"  {row['months']:>3}m ({row['years']:.0f}y)  "
                      f"{row['ref_opt_us']:>8}     {row['eu2_opt_us']:>8}     "
                      f"{row['eu2_opt_eu']:>5}    EUR {row['eu2_pnl_eu']:>+8,.0f}")

            print()

            # Consistencia check
            if use_gold and 'eu2_gold_cagr' in summary_rows[0]:
                gold_deltas = [r['eu2_gold_cagr'] - r['ref_gold_cagr'] for r in summary_rows]
                all_pos_gold = all(d > 0 for d in gold_deltas)
                print(f"  CONSISTENCIA (con Gold): ΔCAGR positivo en {'TODOS' if all_pos_gold else 'NO todos'} "
                      f"({sum(1 for d in gold_deltas if d > 0)}/{len(gold_deltas)})")
                print(f"  MEDIA ΔCAGR (con Gold): {np.mean(gold_deltas):+.1f}pp")
                gold_dd_deltas = [r['eu2_gold_dd'] - r['ref_gold_dd'] for r in summary_rows]
                print(f"  MEDIA ΔDD (con Gold): {np.mean(gold_dd_deltas):+.1f}pp")
            else:
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
                                verbose=v, preloaded_data=preloaded,
                                download_fn=download_fn)
        if 'error' not in r_ref:
            results.append(r_ref)

        for spread in [5.0, 10.0, 15.0]:
            EU_SPREAD_PCT = spread
            r = run_backtest_eu(months, BASE_TICKERS, f"US2+EU1 (EU {spread}%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ALL,
                                max_us_options=2, max_eu_options=1,
                                verbose=v, preloaded_data=preloaded,
                                download_fn=download_fn)
            if 'error' not in r:
                results.append(r)
    else:
        # Test principal: slots SEPARADOS (US no compite con EU)
        # A) Referencia: 2 slots US, 0 EU
        r_ref = run_backtest_eu(months, BASE_TICKERS, "REF: US2 EU0 (3%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ELIGIBLE,
                                max_us_options=2, max_eu_options=0,
                                verbose=v, preloaded_data=preloaded,
                                download_fn=download_fn)
        if 'error' not in r_ref:
            results.append(r_ref)

        # B) 2 slots US + 1 slot EU (no compiten)
        r_eu1 = run_backtest_eu(months, BASE_TICKERS, f"US2+EU1 ({EU_SPREAD_PCT}%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ALL,
                                max_us_options=2, max_eu_options=1,
                                verbose=v, preloaded_data=preloaded,
                                download_fn=download_fn)
        if 'error' not in r_eu1:
            results.append(r_eu1)

        # C) 2 slots US + 2 slots EU
        r_eu2 = run_backtest_eu(months, BASE_TICKERS, f"US2+EU2 ({EU_SPREAD_PCT}%)",
                                use_options=True,
                                options_eligible_set=OPTIONS_ALL,
                                max_us_options=2, max_eu_options=2,
                                verbose=v, preloaded_data=preloaded,
                                download_fn=download_fn)
        if 'error' not in r_eu2:
            results.append(r_eu2)

    if len(results) >= 2:
        print_comparison(results)

    # Gold overlay en modo single-run
    if args.gold and not args.multi_period and not args.test_spreads and len(results) >= 2:
        gold_pct = args.gold_pct
        print(f"\n{'='*100}")
        print(f"  GOLD {gold_pct*100:.0f}% OVERLAY (single-run)")
        print(f"{'='*100}")
        gld_data = download_fn('GLD', months + 6)
        if gld_data is not None:
            print(f"  GLD: {len(gld_data)} dias\n")
            print(f"  {'Config':<30} {'Equity':<14} {'CAGR':<9} {'MaxDD':<8} {'Sharpe':<8} {'Eff':<8} {'Gold PnL':<12}")
            print(f"  {'-'*95}")
            for r in results:
                g = simulate_gold_overlay(r, gld_data, gold_pct)
                if g:
                    sh_raw = compute_sharpe(r.get('equity_curve', []), CONFIG['initial_capital'])
                    sh_gold = compute_sharpe(g.get('equity_curve', []), CONFIG['initial_capital'])
                    eff_raw = r['annualized_return_pct'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0
                    eff_gold = g['ann_gold'] / g['maxdd_gold'] if g['maxdd_gold'] > 0 else 0
                    print(f"  {r['label']:<30} EUR {r['final_equity']:>10,.0f}  {r['annualized_return_pct']:>+7.1f}%  "
                          f"{r['max_drawdown']:>5.1f}%  {sh_raw:>6.2f}  {eff_raw:>6.2f}")
                    print(f"  {r['label'] + ' +Gold':<30} EUR {g['equity_gold']:>10,.0f}  {g['ann_gold']:>+7.1f}%  "
                          f"{g['maxdd_gold']:>5.1f}%  {sh_gold:>6.2f}  {eff_gold:>6.2f}  EUR {g['gold_total_pnl']:>+8,.0f}")
                    print()

    # --- Export CSV completo (stocks + opciones) + PnL por año ---
    if args.export_csv and results:
        import csv
        from collections import defaultdict

        # Usar el último resultado (US2+EU2 si hay varios)
        r = results[-1]
        combined = r.get('combined_trades', [])
        if not combined:
            print("  No hay trades para exportar.")
        else:
            stock_trades = r.get('all_trades', [])
            opt_trades = r.get('all_option_trades', [])
            stock_set = set(id(t) for t in stock_trades)
            opt_eu_set = set(id(t) for t in opt_trades if t.ticker in OPTIONS_ELIGIBLE_EU)

            fname = f"historico_trades_v12_completo_{months}m.csv"
            with open(fname, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['entry_date', 'exit_date', 'ticker', 'trade_type', 'entry_price',
                            'exit_price', 'strike', 'position_eur', 'units', 'pnl_eur',
                            'pnl_pct', 'bars_held', 'max_r_mult', 'exit_reason'])
                for t in sorted(combined, key=lambda x: x.entry_date if x.entry_date else datetime.min):
                    tid = id(t)
                    if tid in stock_set:
                        ttype = 'stock'
                        strike_val = ''
                        entry_p = round(t.entry_price, 2)
                        exit_p = round(t.exit_price, 2) if t.exit_price else ''
                        units = round(t.position_units, 4)
                    elif tid in opt_eu_set:
                        ttype = 'option_eu'
                        strike_val = round(t.strike, 2)
                        entry_p = round(t.entry_option_price, 2)
                        exit_p = round(t.exit_option_price, 2)
                        units = round(t.num_contracts, 2)
                    else:
                        ttype = 'option_us'
                        strike_val = round(t.strike, 2)
                        entry_p = round(t.entry_option_price, 2)
                        exit_p = round(t.exit_option_price, 2)
                        units = round(t.num_contracts, 2)
                    w.writerow([
                        t.entry_date.strftime('%Y-%m-%d') if t.entry_date else '',
                        t.exit_date.strftime('%Y-%m-%d') if t.exit_date else '',
                        t.ticker, ttype, entry_p, exit_p, strike_val,
                        round(t.position_euros, 0), units,
                        round(t.pnl_euros, 2), round(t.pnl_pct, 2),
                        t.bars_held, round(t.max_r_mult, 2),
                        t.exit_reason or ''
                    ])
            print(f"\n  CSV exportado: {fname} ({len(combined)} trades)")

            # --- PnL por año (desglosado stock / opt US / opt EU) ---
            pnl_by_year = defaultdict(lambda: {'stock': 0, 'option_us': 0, 'option_eu': 0, 'n': 0})
            for t in combined:
                if not t.exit_date:
                    continue
                yr = t.exit_date.year
                tid = id(t)
                pnl_by_year[yr]['n'] += 1
                if tid in stock_set:
                    pnl_by_year[yr]['stock'] += t.pnl_euros
                elif tid in opt_eu_set:
                    pnl_by_year[yr]['option_eu'] += t.pnl_euros
                else:
                    pnl_by_year[yr]['option_us'] += t.pnl_euros

            print(f"\n{'='*90}")
            print(f"  PnL POR AÑO — {r['label']} ({months}m)")
            print(f"{'='*90}")
            print(f"  {'Año':<6} {'Trades':<8} {'Stock PnL':>12} {'Opt US PnL':>12} {'Opt EU PnL':>12} {'TOTAL':>12}")
            print(f"  {'-'*80}")

            cumulative = 0
            for yr in sorted(pnl_by_year.keys()):
                d = pnl_by_year[yr]
                total = d['stock'] + d['option_us'] + d['option_eu']
                cumulative += total
                print(f"  {yr:<6} {d['n']:<8} EUR {d['stock']:>+9,.0f} EUR {d['option_us']:>+9,.0f} "
                      f"EUR {d['option_eu']:>+9,.0f} EUR {total:>+9,.0f}")
            print(f"  {'-'*80}")
            print(f"  {'TOTAL':<6} {sum(d['n'] for d in pnl_by_year.values()):<8} "
                  f"EUR {sum(d['stock'] for d in pnl_by_year.values()):>+9,.0f} "
                  f"EUR {sum(d['option_us'] for d in pnl_by_year.values()):>+9,.0f} "
                  f"EUR {sum(d['option_eu'] for d in pnl_by_year.values()):>+9,.0f} "
                  f"EUR {cumulative:>+9,.0f}")

            # --- Equity + retorno por año desde equity curve ---
            eq_curve = r.get('equity_curve', [])
            if eq_curve:
                eq_by_yr = {}
                for day, eq in eq_curve:
                    eq_by_yr[day.year] = eq
                print(f"\n  {'Año':<6} {'Equity fin año':>14} {'Retorno año':>12}")
                print(f"  {'-'*40}")
                prev_eq = CONFIG['initial_capital']
                for yr in sorted(eq_by_yr.keys()):
                    eq_end = eq_by_yr[yr]
                    ret = (eq_end / prev_eq - 1) * 100
                    print(f"  {yr:<6} EUR {eq_end:>11,.0f} {ret:>+10.1f}%")
                    prev_eq = eq_end


if __name__ == "__main__":
    main()
