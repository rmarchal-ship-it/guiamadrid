#!/usr/bin/env python3
"""
PAPER TRADING v3.0 — Momentum Breakout v8 (Simulacion en tiempo real)

Ejecuta la estrategia Momentum Breakout v8 con dinero simulado,
registrando todas las operaciones (acciones + opciones CALL) con costes realistas.

USO:
  python paper_trading.py --scan                # Escaneo diario + ejecutar trades
  python paper_trading.py --status              # Estado actual de la cartera
  python paper_trading.py --history             # Historial de trades cerrados
  python paper_trading.py --reset               # Reset completo (nueva cartera)
  python paper_trading.py --reset --capital 25000  # Reset con capital personalizado

COSTES REALISTAS:
  - Comision: EUR 1 fijo por trade (entrada + salida)
  - Slippage: 0.05% del notional
  - Spread: 0.03% US, 0.08% EU, 0.05% Commodities/ETFs

REGLAS:
  - Senal del dia anterior → compra al Open del dia siguiente
  - Trailing Chandelier 4xATR activado a +2R
  - Emergency stop -15%
  - Time exit 8d: activa trailing 3xATR (nunca fuerza salida)
  - Filtro macro: SPY > SMA50
  - Max 10 posiciones simultaneas (optimizado v8: 225 tickers)

v8 OPCIONES:
  - CALL 5% ITM, ~120 DTE, cierre a 45 DTE restantes
  - Solo si IVR < 40 (opciones baratas) y ticker elegible
  - Max 2 opciones simultaneas, prioridad sobre stocks
  - Position size: 14% del equity por opcion
"""

import json
import os
import sys
import argparse
import csv
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np

# Path al directorio del proyecto (autodetectar via __file__)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from momentum_breakout import (
    MomentumEngine,
    calculate_atr,
    ASSETS,
    TICKERS,
)

# Reutilizar funciones de run_scanner
from run_scanner import (
    download_data,
    download_batch,
    check_macro_filter,
    calculate_composite_score,
    calculate_position_size_v5,
    scan_signals,
    CONFIG as SCANNER_CONFIG,
    # v8 opciones
    OPTIONS_ELIGIBLE,
    OPTIONS_CONFIG,
    historical_volatility,
    iv_rank,
)

# Black-Scholes para pricing de opciones en paper trading
try:
    from scipy.stats import norm as scipy_norm

    def black_scholes_call(S, K, T, r, sigma):
        """Precio y delta de una CALL europea via Black-Scholes."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {'price': max(S - K, 0), 'delta': 1.0 if S > K else 0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S * scipy_norm.cdf(d1) - K * np.exp(-r * T) * scipy_norm.cdf(d2)
        delta = scipy_norm.cdf(d1)
        return {'price': max(price, 0.01), 'delta': delta}

    def monthly_expiration_dte(entry_date, target_dte=120):
        """DTE real al vencimiento mensual (3er viernes) mas cercano a target_dte."""
        target_date = entry_date + timedelta(days=target_dte)
        year, month = target_date.year, target_date.month
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        candidates = []
        for delta_months in [-1, 0, 1]:
            m = month + delta_months
            y = year
            if m < 1:
                m = 12
                y -= 1
            elif m > 12:
                m = 1
                y += 1
            first_day_m = datetime(y, m, 1)
            first_friday_m = first_day_m + timedelta(days=(4 - first_day_m.weekday()) % 7)
            third_friday_m = first_friday_m + timedelta(weeks=2)
            if third_friday_m > entry_date:
                candidates.append(third_friday_m)
        if not candidates:
            return target_dte
        best = min(candidates, key=lambda d: abs((d - entry_date).days - target_dte))
        return (best - entry_date).days

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# CONFIGURACION v8 — Identica al backtest v8
# =============================================================================

CONFIG = {
    'initial_capital': 10000,
    'target_risk_per_trade_pct': 2.0,
    'max_positions': 10,  # Optimizado v8: 10 > 7 con 225 tickers (PF 2.89, +34.6% anual)
    'ker_threshold': 0.40,
    'volume_threshold': 1.3,
    'rsi_threshold': 50,
    'rsi_max': 75,
    'breakout_period': 20,
    'longs_only': True,
    'emergency_stop_pct': 0.15,
    'trail_trigger_r': 2.0,
    'trail_atr_mult': 4.0,
    'max_hold_bars': 8,
    'time_exit_trail_atr_mult': 3.0,  # ATR mult para trailing activado por time exit
    'use_macro_filter': True,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,
    'test_tickers': [t for t, v in ASSETS.items()
                     if not v.get('is_crypto', False)
                     and not t.endswith('USDT')],
    # v8 Opciones
    'option_dte': 120,
    'option_itm_pct': 0.05,
    'option_close_dte': 45,
    'option_max_ivr': 40,
    'option_ivr_window': 252,
    'option_position_pct': 0.14,
    'max_option_positions': 2,
    'option_spread_pct': 3.0,
    'risk_free_rate': 0.043,
}

PORTFOLIO_FILE = os.path.join(PROJECT_DIR, 'paper_portfolio.json')
TRADES_CSV = os.path.join(PROJECT_DIR, 'paper_trades_log.csv')


# =============================================================================
# MODELO DE COSTES REALISTAS
# =============================================================================

# Spread por categoria de activo
SPREAD_BY_CATEGORY = {
    'US_TECH': 0.03,
    'US_FINANCE': 0.03,
    'US_HEALTH': 0.03,
    'US_CONSUMER': 0.03,
    'EU_GERMANY': 0.08,
    'EU_FRANCE': 0.08,
    'EU_VARIOUS': 0.08,
    'COMMODITY': 0.05,
    'US_INDEX': 0.05,
    'FIXED_INCOME': 0.03,
}

COMMISSION_PER_TRADE = 5.00  # EUR fijo por trade (broker EU tipo DEGIRO/Renta4)
SLIPPAGE_PCT = 0.02          # 0.02% del notional (ordenes limitadas al Open)

# Horarios de mercado (hora local Madrid/CET)
# EU: 9:00 - 17:30 | US: 15:30 - 22:00
# Commodities/ETFs US: mismo horario US
MARKET_HOURS = {
    'EU': (9, 17),    # EU abierto entre 9:00 y 17:30
    'US': (15, 22),   # US abierto entre 15:30 y 22:00
}

def get_market_zone(ticker: str) -> str:
    """Retorna la zona de mercado del ticker (EU o US)."""
    asset = ASSETS.get(ticker, {})
    category = asset.get('category', '')
    if category.startswith('EU_'):
        return 'EU'
    return 'US'  # US stocks, commodities, indices, fixed income


def is_market_open(ticker: str) -> bool:
    """Comprueba si el mercado del ticker esta abierto ahora (hora Madrid)."""
    zone = get_market_zone(ticker)
    hour = datetime.now().hour
    open_h, close_h = MARKET_HOURS[zone]
    return open_h <= hour < close_h


def filter_tickers_by_market_hours(tickers: list) -> list:
    """Filtra tickers cuyo mercado esta abierto ahora."""
    return [t for t in tickers if is_market_open(t)]


def get_spread_pct(ticker: str) -> float:
    """Retorna el spread estimado (%) segun la categoria del activo."""
    asset = ASSETS.get(ticker, {})
    category = asset.get('category', 'UNKNOWN')
    return SPREAD_BY_CATEGORY.get(category, 0.05) / 100


def calculate_entry_costs(ticker: str, price: float, units: float) -> dict:
    """Calcula costes de entrada (comision + slippage + spread)."""
    notional = price * units
    spread_pct = get_spread_pct(ticker)
    return {
        'commission': COMMISSION_PER_TRADE,
        'slippage': notional * SLIPPAGE_PCT / 100,
        'spread': notional * spread_pct,
    }


def calculate_exit_costs(ticker: str, price: float, units: float) -> dict:
    """Calcula costes de salida (comision + slippage + spread)."""
    notional = price * units
    spread_pct = get_spread_pct(ticker)
    return {
        'commission': COMMISSION_PER_TRADE,
        'slippage': notional * SLIPPAGE_PCT / 100,
        'spread': notional * spread_pct,
    }


def effective_entry_price(ticker: str, open_price: float) -> float:
    """Precio efectivo de compra: Open + spread + slippage."""
    spread_pct = get_spread_pct(ticker)
    return open_price * (1 + spread_pct + SLIPPAGE_PCT / 100)


def effective_exit_price(ticker: str, raw_price: float) -> float:
    """Precio efectivo de venta: precio - spread - slippage."""
    spread_pct = get_spread_pct(ticker)
    return raw_price * (1 - spread_pct - SLIPPAGE_PCT / 100)


# =============================================================================
# PORTFOLIO — Estado persistente en JSON
# =============================================================================

def create_portfolio(capital: float) -> dict:
    """Crea un portfolio vacio."""
    return {
        'created': datetime.now().isoformat(),
        'initial_capital': capital,
        'cash': capital,
        'positions': [],
        'option_positions': [],  # v8: posiciones de opciones CALL
        'closed_trades': [],
        'closed_option_trades': [],  # v8: opciones cerradas
        'pending_signals': [],  # Senales bloqueadas por mercado cerrado, ejecutar en proximo scan
        'log': [],
        'last_scan': None,
    }


def load_portfolio() -> Optional[dict]:
    """Carga el portfolio desde JSON. Migra portfolios antiguos anadiendo campos de opciones."""
    if not os.path.exists(PORTFOLIO_FILE):
        return None
    with open(PORTFOLIO_FILE, 'r') as f:
        portfolio = json.load(f)
    # Migracion: anadir campos de opciones si no existen
    if 'option_positions' not in portfolio:
        portfolio['option_positions'] = []
    if 'closed_option_trades' not in portfolio:
        portfolio['closed_option_trades'] = []
    return portfolio


def save_portfolio(portfolio: dict):
    """Guarda el portfolio en JSON."""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


def add_log(portfolio: dict, action: str, ticker: str, price: float,
            units: float = 0, details: str = ''):
    """Anade entrada al log de operaciones."""
    portfolio['log'].append({
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'ticker': ticker,
        'price': round(price, 2),
        'units': round(units, 4),
        'details': details,
    })


def get_equity(portfolio: dict, current_prices: dict) -> float:
    """Calcula el equity total (cash + valor de posiciones abiertas + opciones)."""
    equity = portfolio['cash']
    for pos in portfolio['positions']:
        ticker = pos['ticker']
        if ticker in current_prices:
            current_price = current_prices[ticker]
            equity += pos['units'] * current_price
        else:
            equity += pos['position_eur']  # fallback: valor de entrada
    # v8: Anadir valor de opciones (premium pagada como proxy conservador)
    for opt in portfolio.get('option_positions', []):
        equity += opt.get('current_value', opt['premium_paid'])
    return equity


def export_trades_csv(portfolio: dict):
    """Exporta trades cerrados a CSV."""
    if not portfolio['closed_trades']:
        return

    fieldnames = [
        'entry_date', 'exit_date', 'ticker',
        'entry_price', 'exit_price',
        'position_eur', 'units',
        'pnl_eur', 'pnl_pct',
        'bars_held', 'max_r_mult',
        'exit_reason', 'total_costs',
        'commission_total', 'slippage_total', 'spread_total',
    ]

    with open(TRADES_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trade in portfolio['closed_trades']:
            costs = trade.get('costs', {})
            row = {
                'entry_date': trade['entry_date'],
                'exit_date': trade['exit_date'],
                'ticker': trade['ticker'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'position_eur': trade['position_eur'],
                'units': trade['units'],
                'pnl_eur': trade['pnl_eur'],
                'pnl_pct': trade['pnl_pct'],
                'bars_held': trade['bars_held'],
                'max_r_mult': trade['max_r_mult'],
                'exit_reason': trade['exit_reason'],
                'total_costs': trade.get('total_costs', 0),
                'commission_total': costs.get('commission_entry', 0) + costs.get('commission_exit', 0),
                'slippage_total': costs.get('slippage_entry', 0) + costs.get('slippage_exit', 0),
                'spread_total': costs.get('spread_entry', 0) + costs.get('spread_exit', 0),
            }
            writer.writerow(row)


# =============================================================================
# SCAN — Flujo diario completo
# =============================================================================

def run_scan(portfolio: dict):
    """
    Flujo diario:
    1. Gestionar posiciones existentes (stops, trailing, time exits)
    2. Comprobar filtro macro
    3. Generar senales y abrir nuevas posiciones
    """
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')

    hour = now.hour
    eu_open = MARKET_HOURS['EU'][0] <= hour < MARKET_HOURS['EU'][1]
    us_open = MARKET_HOURS['US'][0] <= hour < MARKET_HOURS['US'][1]
    markets_str = []
    if eu_open:
        markets_str.append("EU ABIERTO")
    if us_open:
        markets_str.append("US ABIERTO")
    if not markets_str:
        markets_str.append("MERCADOS CERRADOS (usando ultimo cierre)")
    markets_display = " | ".join(markets_str)

    print(f"\n{'='*70}")
    print(f"  PAPER TRADING v3.0 (v8) — Scan {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    n_opts = len(portfolio.get('option_positions', []))
    n_stocks = len(portfolio['positions'])
    print(f"  Capital inicial: EUR {portfolio['initial_capital']:,.0f}")
    print(f"  Cash disponible: EUR {portfolio['cash']:,.2f}")
    print(f"  Posiciones abiertas: {n_stocks + n_opts}/{CONFIG['max_positions']} (stocks: {n_stocks}, opciones: {n_opts})")
    print(f"  Mercados: {markets_display}")
    print(f"{'='*70}\n")

    # 1. DESCARGAR DATOS
    print("  Descargando datos para 112 tickers...")
    tickers_to_download = CONFIG['test_tickers']
    data = download_batch(tickers_to_download, period='14mo')

    if not data:
        print("  ERROR: No se pudieron descargar datos")
        return

    # Obtener precios actuales
    current_prices = {}
    for ticker, df in data.items():
        if len(df) > 0:
            current_prices[ticker] = float(df['Close'].iloc[-1])

    # 2. GESTIONAR POSICIONES EXISTENTES
    print(f"\n  --- GESTION DE POSICIONES EXISTENTES ---\n")

    positions_to_close = []
    for i, pos in enumerate(portfolio['positions']):
        ticker = pos['ticker']
        if ticker not in data:
            print(f"  WARN: {ticker} sin datos — manteniendo posicion")
            continue

        df = data[ticker]
        if len(df) < 2:
            continue

        # Usar la ultima barra disponible
        bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        current_atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else float(prev_bar.get('ATR', pos['entry_atr']))

        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        # Obtener la fecha de la ultima barra disponible
        last_bar_date = df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])[:10]

        # Incrementar dias SOLO si es una barra nueva (evitar contar doble con multiples scans/dia)
        last_update = pos.get('last_bar_date', '')
        if last_bar_date != last_update:
            pos['bars_held'] += 1
            pos['last_bar_date'] = last_bar_date

        pos['highest_since'] = max(pos.get('highest_since', pos['entry_price']), high)

        # Calcular R-multiple actual
        R = pos['R']
        r_mult = (close - pos['entry_price']) / R if R > 0 else 0
        pos['max_r_mult'] = max(pos['max_r_mult'], r_mult)

        exit_reason = None
        exit_price_raw = None

        # a. EMERGENCY STOP (-15%)
        emergency_level = pos['entry_price'] * (1 - CONFIG['emergency_stop_pct'])
        if low <= emergency_level:
            exit_reason = 'emergency_stop'
            exit_price_raw = emergency_level

        # b. TRAILING STOP CHECK
        elif pos['trailing_active'] and pos['trailing_stop'] is not None:
            if low <= pos['trailing_stop']:
                exit_reason = 'trailing_stop'
                exit_price_raw = pos['trailing_stop']

        # c. ACTUALIZAR TRAILING (Chandelier 4xATR)
        if exit_reason is None and r_mult >= CONFIG['trail_trigger_r']:
            chandelier = pos['highest_since'] - (current_atr * CONFIG['trail_atr_mult'])
            if not pos['trailing_active']:
                pos['trailing_active'] = True
                pos['trailing_stop'] = chandelier
                print(f"  {ticker:8} | Trailing ACTIVADO a ${chandelier:.2f} (r_mult: {r_mult:.1f}R)")
            elif chandelier > pos['trailing_stop']:
                pos['trailing_stop'] = chandelier

        # d. TIME EXIT: tras max_hold_bars, activar trailing (nunca forzar salida)
        # v8: 8 bars, trailing 3xATR. Elimina time_exit forzados.
        if exit_reason is None and pos['bars_held'] >= CONFIG['max_hold_bars']:
            if not pos['trailing_active']:
                trail_mult = CONFIG.get('time_exit_trail_atr_mult', 3.0)
                chandelier = pos['highest_since'] - (current_atr * trail_mult)
                breakeven = pos['entry_price'] * (1 + SLIPPAGE_PCT / 100 + get_spread_pct(ticker) * 100)
                pos['trailing_active'] = True
                if close <= pos['entry_price']:
                    # Perdiendo: trailing apretado (3xATR o 5% bajo maximo)
                    pos['trailing_stop'] = max(chandelier, pos['entry_price'] * 0.95)
                    print(f"  {ticker:8} | Trailing activado (time limit, perdedor) a ${pos['trailing_stop']:.2f}")
                else:
                    # Ganando: trailing a breakeven minimo
                    pos['trailing_stop'] = max(chandelier, breakeven)
                    print(f"  {ticker:8} | Trailing activado (time limit, ganador) a ${pos['trailing_stop']:.2f}")

        # CERRAR POSICION
        if exit_reason is not None:
            exit_price = effective_exit_price(ticker, exit_price_raw)
            pnl_eur = (exit_price - pos['entry_price']) * pos['units']
            pnl_pct = (pnl_eur / pos['position_eur']) * 100 if pos['position_eur'] > 0 else 0

            # Costes de salida
            exit_costs = calculate_exit_costs(ticker, exit_price_raw, pos['units'])
            entry_costs = pos.get('costs', {})
            total_costs = (
                entry_costs.get('commission', 0) + entry_costs.get('slippage', 0) + entry_costs.get('spread', 0) +
                exit_costs['commission'] + exit_costs['slippage'] + exit_costs['spread']
            )

            # PnL neto (descontando costes de salida)
            pnl_eur_net = pnl_eur - exit_costs['commission'] - exit_costs['slippage'] - exit_costs['spread']

            closed_trade = {
                'ticker': ticker,
                'entry_date': pos['entry_date'],
                'exit_date': today_str,
                'entry_price': pos['entry_price'],
                'exit_price': round(exit_price, 2),
                'units': pos['units'],
                'position_eur': pos['position_eur'],
                'pnl_eur': round(pnl_eur_net, 2),
                'pnl_pct': round((pnl_eur_net / pos['position_eur']) * 100, 2) if pos['position_eur'] > 0 else 0,
                'bars_held': pos['bars_held'],
                'max_r_mult': round(pos['max_r_mult'], 2),
                'exit_reason': exit_reason,
                'total_costs': round(total_costs, 2),
                'costs': {
                    'commission_entry': entry_costs.get('commission', 0),
                    'slippage_entry': entry_costs.get('slippage', 0),
                    'spread_entry': entry_costs.get('spread', 0),
                    'commission_exit': round(exit_costs['commission'], 2),
                    'slippage_exit': round(exit_costs['slippage'], 2),
                    'spread_exit': round(exit_costs['spread'], 2),
                },
            }

            portfolio['closed_trades'].append(closed_trade)
            portfolio['cash'] += pos['position_eur'] + pnl_eur_net
            positions_to_close.append(i)

            marker = '+' if pnl_eur_net > 0 else '-'
            print(f"  {ticker:8} | CERRADA | {exit_reason:15} | PnL: EUR {pnl_eur_net:+.2f} ({pnl_pct:+.1f}%) "
                  f"| {pos['bars_held']}d | Costes: EUR {total_costs:.2f}")

            add_log(portfolio, 'SELL', ticker, exit_price, pos['units'],
                    f"{exit_reason} | PnL EUR {pnl_eur_net:+.2f} | Costes EUR {total_costs:.2f}")

        else:
            # Posicion activa — mostrar estado
            unrealized_pnl = (close - pos['entry_price']) * pos['units']
            print(f"  {ticker:8} | ABIERTA | ${close:>9.2f} | PnL: EUR {unrealized_pnl:+.2f} ({r_mult:+.1f}R) "
                  f"| {pos['bars_held']}d | Trail: {'${:.2f}'.format(pos['trailing_stop']) if pos['trailing_active'] else '---'}")

    # Eliminar posiciones cerradas (en orden inverso para no romper indices)
    for i in sorted(positions_to_close, reverse=True):
        portfolio['positions'].pop(i)

    # 2b. GESTIONAR OPCIONES EXISTENTES (v8)
    if HAS_SCIPY and portfolio.get('option_positions'):
        print(f"\n  --- GESTION DE OPCIONES v8 ---\n")
        options_to_close = []
        for i, opt in enumerate(portfolio['option_positions']):
            ticker = opt['ticker']
            if ticker not in data:
                print(f"  {ticker:8} | OPT ABIERTA | Sin datos — manteniendo")
                continue

            df = data[ticker]
            if len(df) < 2:
                continue

            current_price = float(df['Close'].iloc[-1])
            current_iv = float(df['HVOL'].iloc[-1]) if 'HVOL' in df.columns and not pd.isna(df['HVOL'].iloc[-1]) else opt['entry_iv']

            # Calcular DTE restante
            entry_date = datetime.fromisoformat(opt['entry_date'])
            days_elapsed = (now - entry_date).days
            remaining_dte = max(opt['dte_at_entry'] - days_elapsed, 0)

            # Repriciar opcion con Black-Scholes
            T = remaining_dte / 365.0
            bs = black_scholes_call(current_price, opt['strike'], T, CONFIG['risk_free_rate'], current_iv)
            current_option_price = bs['price'] * (1 - CONFIG['option_spread_pct'] / 100 / 2)
            current_value = current_option_price * opt['num_contracts'] * 100
            opt['current_value'] = round(current_value, 2)
            opt['current_option_price'] = round(current_option_price, 2)

            option_pnl = current_value - opt['premium_paid']
            option_pnl_pct = (option_pnl / opt['premium_paid']) * 100 if opt['premium_paid'] > 0 else 0

            exit_reason = None

            # Cierre a 45 DTE restantes
            if remaining_dte <= CONFIG.get('option_close_dte', 45):
                exit_reason = 'dte_exit'

            # Expiracion (safety)
            if remaining_dte <= 0:
                current_option_price = max(current_price - opt['strike'], 0)
                current_value = current_option_price * opt['num_contracts'] * 100
                exit_reason = 'expiration'

            if exit_reason:
                exit_costs = calculate_exit_costs(ticker, current_option_price * opt['num_contracts'] * 100, 1)
                total_exit_cost = exit_costs['commission']  # Solo comision para opciones
                pnl_net = current_value - opt['premium_paid'] - total_exit_cost

                closed_opt = {
                    'ticker': ticker,
                    'type': 'CALL',
                    'entry_date': opt['entry_date'],
                    'exit_date': today_str,
                    'strike': opt['strike'],
                    'dte_at_entry': opt['dte_at_entry'],
                    'entry_option_price': opt['entry_option_price'],
                    'exit_option_price': round(current_option_price, 4),
                    'num_contracts': opt['num_contracts'],
                    'premium_paid': opt['premium_paid'],
                    'exit_value': round(current_value, 2),
                    'pnl_eur': round(pnl_net, 2),
                    'pnl_pct': round((pnl_net / opt['premium_paid']) * 100, 2) if opt['premium_paid'] > 0 else 0,
                    'days_held': days_elapsed,
                    'exit_reason': exit_reason,
                    'total_costs': round(total_exit_cost, 2),
                }

                if 'closed_option_trades' not in portfolio:
                    portfolio['closed_option_trades'] = []
                portfolio['closed_option_trades'].append(closed_opt)
                portfolio['cash'] += current_value - total_exit_cost
                options_to_close.append(i)

                print(f"  {ticker:8} | OPT CERRADA | {exit_reason:12} | PnL: EUR {pnl_net:+.2f} ({option_pnl_pct:+.1f}%) "
                      f"| {days_elapsed}d | K=${opt['strike']:.2f}")
                add_log(portfolio, 'SELL_OPT', ticker, current_option_price, opt['num_contracts'],
                        f"{exit_reason} | PnL EUR {pnl_net:+.2f}")
            else:
                print(f"  {ticker:8} | OPT ABIERTA | ${current_price:>9.2f} | PnL: EUR {option_pnl:+.2f} ({option_pnl_pct:+.1f}%) "
                      f"| {days_elapsed}d/{opt['dte_at_entry']}d | DTE rest: {remaining_dte}d | K=${opt['strike']:.2f}")

        for i in sorted(options_to_close, reverse=True):
            portfolio['option_positions'].pop(i)

    # 3. COMPROBAR MACRO FILTER
    macro_ok = check_macro_filter(data)
    macro_status = "BULL" if macro_ok else "BEAR"
    print(f"\n  Filtro macro: {macro_status} (SPY {'>' if macro_ok else '<'} SMA50)")

    # 4. GENERAR SENALES Y ABRIR NUEVAS POSICIONES
    total_positions = len(portfolio['positions']) + len(portfolio.get('option_positions', []))
    slots_available = CONFIG['max_positions'] - total_positions

    if slots_available <= 0:
        print(f"\n  Cartera llena ({CONFIG['max_positions']}/{CONFIG['max_positions']} posiciones) — no se buscan nuevas senales")
    elif not macro_ok:
        print(f"\n  Mercado BEAR — no se abren posiciones nuevas")
        # Pero mostramos las senales como informacion
        result = scan_signals(data, portfolio['cash'])
        if result['signals']:
            print(f"  (Hay {len(result['signals'])} senales disponibles pero bloqueadas por filtro macro)")
    else:
        print(f"\n  --- BUSCANDO NUEVAS SENALES ({slots_available} slots disponibles) ---\n")

        # Calcular equity actual para sizing
        equity = get_equity(portfolio, current_prices)

        # Generar senales con el scanner v8
        result = scan_signals(data, equity)
        signals = result['signals']

        # Filtrar tickers que ya tenemos en cartera (acciones + opciones)
        open_tickers = {pos['ticker'] for pos in portfolio['positions']}
        open_tickers |= {opt['ticker'] for opt in portfolio.get('option_positions', [])}
        signals = [s for s in signals if s['ticker'] not in open_tickers]

        # Recuperar senales pendientes de scans anteriores (mercado cerrado)
        pending = portfolio.get('pending_signals', [])
        pending_tickers = {p['ticker'] for p in pending}
        # Solo mantener pendientes cuyo mercado ahora este abierto
        pending_ready = [p for p in pending if is_market_open(p['ticker']) and p['ticker'] not in open_tickers]
        pending_still_waiting = [p for p in pending if not is_market_open(p['ticker'])]

        if pending_ready:
            print(f"  Ejecutando {len(pending_ready)} senales pendientes: {', '.join(p['ticker'] for p in pending_ready)}")
            # Anadir pendientes listos al inicio de signals (prioridad)
            for p in pending_ready:
                if not any(s['ticker'] == p['ticker'] for s in signals):
                    signals.insert(0, p)

        # Filtrar por horario de mercado: solo comprar si el mercado del ticker esta abierto
        signals_blocked = [s for s in signals if not is_market_open(s['ticker'])]
        signals = [s for s in signals if is_market_open(s['ticker'])]

        # Guardar senales bloqueadas como pendientes (solo las de hoy, no acumular)
        today_str_check = now.strftime('%Y-%m-%d')
        new_pending = pending_still_waiting.copy()
        for s in signals_blocked:
            if s['ticker'] not in {p['ticker'] for p in new_pending}:
                s['pending_since'] = today_str_check
                new_pending.append(s)
        # Limpiar pendientes de dias anteriores (max 1 dia)
        new_pending = [p for p in new_pending if p.get('pending_since', '') >= today_str_check]
        portfolio['pending_signals'] = new_pending

        if signals_blocked:
            blocked_tickers = ', '.join(s['ticker'] for s in signals_blocked)
            print(f"  Senales pendientes (mercado cerrado): {blocked_tickers}")
            print(f"  (Se compraran en el proximo scan cuando su mercado abra)\n")

        if not signals:
            print("  No hay senales nuevas hoy")
        else:
            n_to_open = min(len(signals), slots_available)
            print(f"  {len(signals)} senales encontradas — abriendo Top {n_to_open}\n")

            # Contar opciones abiertas para el limite v8
            n_open_options = len(portfolio.get('option_positions', []))

            for s in signals[:n_to_open]:
                ticker = s['ticker']
                df = data[ticker]
                open_price = float(df['Open'].iloc[-1])

                # v8: Decidir si abrir opcion o accion
                open_as_option = False
                if (HAS_SCIPY and s.get('option_recommended')
                        and n_open_options < CONFIG.get('max_option_positions', 2)):
                    open_as_option = True

                if open_as_option:
                    # --- ABRIR OPCION CALL v8 ---
                    strike = open_price * (1 - CONFIG['option_itm_pct'])
                    actual_dte = monthly_expiration_dte(now, CONFIG['option_dte'])
                    T = actual_dte / 365.0

                    iv = float(df['HVOL'].iloc[-1]) if 'HVOL' in df.columns and not pd.isna(df['HVOL'].iloc[-1]) else 0.30
                    bs = black_scholes_call(open_price, strike, T, CONFIG['risk_free_rate'], iv)
                    option_price = bs['price'] * (1 + CONFIG['option_spread_pct'] / 100 / 2)

                    # Position sizing: 14% del equity
                    max_premium = equity * CONFIG.get('option_position_pct', 0.14)
                    contracts = max_premium / (option_price * 100)
                    premium_paid = contracts * option_price * 100

                    if premium_paid < 50:
                        print(f"  {ticker:8} | SKIP OPT — premium demasiado baja (EUR {premium_paid:.0f})")
                        continue

                    # Comision de entrada
                    entry_cost = COMMISSION_PER_TRADE

                    opt_position = {
                        'ticker': ticker,
                        'type': 'CALL',
                        'entry_date': now.isoformat(),
                        'strike': round(strike, 2),
                        'dte_at_entry': actual_dte,
                        'entry_stock_price': round(open_price, 2),
                        'entry_option_price': round(option_price, 4),
                        'entry_iv': round(iv, 4),
                        'num_contracts': round(contracts, 4),
                        'premium_paid': round(premium_paid, 2),
                        'current_value': round(premium_paid, 2),
                        'current_option_price': round(option_price, 4),
                    }

                    if 'option_positions' not in portfolio:
                        portfolio['option_positions'] = []
                    portfolio['option_positions'].append(opt_position)
                    portfolio['cash'] -= (premium_paid + entry_cost)
                    n_open_options += 1

                    print(f"  {ticker:8} | CALL    | K=${strike:.2f} IV={iv:.0%} {actual_dte}DTE | "
                          f"Prem=${option_price:.2f} x{contracts:.2f}c = EUR {premium_paid:.0f} "
                          f"| Score: {s['score']:.0f}/100 | IVR={s.get('option_ivr', '?')}")

                    add_log(portfolio, 'BUY_OPT', ticker, option_price, contracts,
                            f"CALL K=${strike:.2f} {actual_dte}DTE IV={iv:.0%} IVR={s.get('option_ivr', '?')} | "
                            f"Premium EUR {premium_paid:.0f}")
                else:
                    # --- ABRIR ACCION/ETF (logica v8) ---
                    entry_price = effective_entry_price(ticker, open_price)

                    current_atr = s['atr']
                    size_info = calculate_position_size_v5(equity, current_atr, open_price)

                    units = size_info['units']
                    notional = size_info['notional']

                    max_per_position = equity / CONFIG['max_positions'] * 2
                    if notional > max_per_position:
                        notional = max_per_position
                        units = notional / entry_price

                    if notional < 100:
                        print(f"  {ticker:8} | SKIP — posicion demasiado pequena (EUR {notional:.0f})")
                        continue

                    entry_costs = calculate_entry_costs(ticker, open_price, units)
                    total_entry_cost = entry_costs['commission'] + entry_costs['slippage'] + entry_costs['spread']

                    R = current_atr * 2.0

                    position = {
                        'ticker': ticker,
                        'entry_date': today_str,
                        'entry_price': round(entry_price, 4),
                        'units': round(units, 4),
                        'position_eur': round(notional, 2),
                        'entry_atr': round(current_atr, 4),
                        'R': round(R, 4),
                        'trailing_stop': None,
                        'trailing_active': False,
                        'highest_since': round(float(df['High'].iloc[-1]), 4),
                        'max_r_mult': 0.0,
                        'bars_held': 0,
                        'costs': {
                            'commission': round(entry_costs['commission'], 2),
                            'slippage': round(entry_costs['slippage'], 2),
                            'spread': round(entry_costs['spread'], 2),
                        },
                    }

                    portfolio['positions'].append(position)
                    portfolio['cash'] -= (notional + total_entry_cost)

                    print(f"  {ticker:8} | COMPRA  | {units:.2f}u @ ${entry_price:.2f} = EUR {notional:.0f} "
                          f"| Score: {s['score']:.0f}/100 | Costes: EUR {total_entry_cost:.2f}")
                    print(f"  {'':>10} Niveles → Emergency: ${entry_price * (1 - CONFIG['emergency_stop_pct']):.2f} "
                          f"| Trail activa a: ${entry_price + R * CONFIG['trail_trigger_r']:.2f}")

                    add_log(portfolio, 'BUY', ticker, entry_price, units,
                            f"Score {s['score']:.0f}/100 | KER {s['ker']:.3f} | RSI {s['rsi']:.0f} | "
                            f"Vol {s['vol_ratio']:.1f}x | Costes EUR {total_entry_cost:.2f}")

        # Watchlist
        if result['watchlist']:
            print(f"\n  WATCHLIST — {len(result['watchlist'])} tickers cerca de breakout:")
            for w in result['watchlist'][:5]:
                print(f"    {w['ticker']:<10} ${w['price']:>9.2f} → ${w['breakout_level']:>9.2f} "
                      f"({w['distance_pct']:.2f}%) KER {w['ker']:.3f}")

    # 5. RESUMEN FINAL
    equity = get_equity(portfolio, current_prices)
    pnl_total = equity - portfolio['initial_capital']
    pnl_pct = (pnl_total / portfolio['initial_capital']) * 100

    print(f"\n{'='*70}")
    print(f"  RESUMEN — {today_str}")
    print(f"{'='*70}")
    n_stock_pos = len(portfolio['positions'])
    n_opt_pos = len(portfolio.get('option_positions', []))
    n_closed_opts = len(portfolio.get('closed_option_trades', []))
    print(f"  Equity:     EUR {equity:,.2f} ({pnl_pct:+.1f}%)")
    print(f"  Cash:       EUR {portfolio['cash']:,.2f}")
    print(f"  Posiciones: {n_stock_pos + n_opt_pos}/{CONFIG['max_positions']} (stocks: {n_stock_pos}, opciones: {n_opt_pos})")
    print(f"  Trades cerrados: {len(portfolio['closed_trades'])} stocks + {n_closed_opts} opciones")
    print(f"{'='*70}\n")

    # Guardar
    portfolio['last_scan'] = now.isoformat()
    save_portfolio(portfolio)
    export_trades_csv(portfolio)
    print(f"  Guardado en: {PORTFOLIO_FILE}")
    print(f"  CSV exportado: {TRADES_CSV}")


# =============================================================================
# STATUS — Estado actual de la cartera
# =============================================================================

def show_status(portfolio: dict, manual_prices: dict = None):
    """Muestra el estado actual de la cartera con precios en tiempo real.

    manual_prices: dict {ticker: precio_eur} — precios manuales de DEGIRO
                   para tickers problemáticos (ej. 6861.T, BHP.AX en Tradegate).
                   Estos SOBREESCRIBEN los precios de yfinance.
    """
    if manual_prices is None:
        manual_prices = {}
    now = datetime.now()

    print(f"\n{'='*70}")
    print(f"  PAPER TRADING v8 — Estado a {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    if manual_prices:
        print(f"  Precios manuales DEGIRO: {', '.join(f'{t}=€{p:.2f}' for t, p in manual_prices.items())}")

    # Descargar precios actuales para posiciones abiertas (acciones + opciones)
    # FIX 2-mar-2026: tickers .T (JPY) y .AX (AUD) necesitan conversion FX a EUR
    FX_CONVERSION = {}  # ticker -> fx_pair needed
    for pos in portfolio['positions']:
        ccy = pos.get('currency', 'USD')
        t = pos['ticker']
        # Skip FX conversion for tickers with manual prices
        if t in manual_prices:
            continue
        if ccy == 'EUR' and t.endswith('.T'):
            FX_CONVERSION[t] = 'EURJPY=X'
        elif ccy == 'EUR' and t.endswith('.AX'):
            FX_CONVERSION[t] = 'EURAUD=X'

    # Download FX rates if needed
    fx_rates = {}
    for fx_pair in set(FX_CONVERSION.values()):
        fx_df = download_data(fx_pair, period='3mo')
        if fx_df is not None and len(fx_df) > 0:
            fx_rates[fx_pair] = float(fx_df['Close'].iloc[-1])

    current_prices = {}
    # Inyectar precios manuales primero (tienen prioridad)
    current_prices.update(manual_prices)

    tickers_needed = [pos['ticker'] for pos in portfolio['positions']]
    tickers_needed += [opt['ticker'] for opt in portfolio.get('option_positions', [])]
    tickers_needed = list(set(tickers_needed))
    # No descargar tickers que ya tienen precio manual
    tickers_to_download = [t for t in tickers_needed if t not in manual_prices]
    if tickers_to_download:
        print(f"  Descargando precios para {len(tickers_to_download)} posiciones...")
        for ticker in tickers_to_download:
            df = download_data(ticker, period='3mo')
            if df is not None and len(df) > 0:
                price = float(df['Close'].iloc[-1])
                # Convert to EUR if needed (e.g. 6861.T returns JPY, BHP.AX returns AUD)
                if ticker in FX_CONVERSION:
                    fx_pair = FX_CONVERSION[ticker]
                    if fx_pair in fx_rates:
                        price = price / fx_rates[fx_pair]
                current_prices[ticker] = price

    equity = get_equity(portfolio, current_prices)
    pnl_total = equity - portfolio['initial_capital']
    pnl_pct = (pnl_total / portfolio['initial_capital']) * 100

    n_stock_pos = len(portfolio['positions'])
    n_opt_pos = len(portfolio.get('option_positions', []))
    print(f"\n  Capital inicial: EUR {portfolio['initial_capital']:,.0f} | Cash: EUR {portfolio['cash']:,.2f}")
    print(f"  Equity total:    EUR {equity:,.2f} ({pnl_pct:+.1f}%)")
    print(f"  Posiciones:      {n_stock_pos + n_opt_pos}/{CONFIG['max_positions']} (stocks: {n_stock_pos}, opciones: {n_opt_pos})")

    if portfolio.get('last_scan'):
        print(f"  Ultimo scan:     {portfolio['last_scan'][:19]}")

    # Posiciones de acciones abiertas
    if portfolio['positions']:
        print(f"\n  POSICIONES ACCIONES:")
        print(f"  {'#':<3} {'Ticker':<8} {'Entrada':<12} {'P.Entry':>9} {'P.Actual':>9} "
              f"{'PnL EUR':>9} {'PnL%':>7} {'Dias':>5} {'R-Mult':>7} {'Trail':>10}")
        print(f"  {'-'*85}")

        total_unrealized = 0
        total_costs_open = 0
        for i, pos in enumerate(portfolio['positions'], 1):
            ticker = pos['ticker']
            current_price = current_prices.get(ticker, pos['entry_price'])
            unrealized_pnl = (current_price - pos['entry_price']) * pos['units']
            unrealized_pct = (unrealized_pnl / pos['position_eur']) * 100 if pos['position_eur'] > 0 else 0
            r_mult = (current_price - pos['entry_price']) / pos['R'] if pos['R'] > 0 else 0
            trail_str = f"${pos['trailing_stop']:.2f}" if pos['trailing_active'] and pos['trailing_stop'] else '---'

            entry_costs = pos.get('costs', {})
            pos_costs = entry_costs.get('commission', 0) + entry_costs.get('slippage', 0) + entry_costs.get('spread', 0)
            total_costs_open += pos_costs

            print(f"  {i:<3} {ticker:<8} {pos['entry_date']:<12} ${pos['entry_price']:>8.2f} ${current_price:>8.2f} "
                  f"EUR{unrealized_pnl:>+7.0f} {unrealized_pct:>+6.1f}% {pos['bars_held']:>4}d {r_mult:>+6.1f}R {trail_str:>10}")
            total_unrealized += unrealized_pnl

        print(f"  {'-'*85}")
        print(f"  {'':>3} {'TOTAL':<8} {'':>12} {'':>9} {'':>9} EUR{total_unrealized:>+7.0f}")

    # v8: Posiciones de opciones abiertas
    if portfolio.get('option_positions'):
        print(f"\n  POSICIONES OPCIONES (v8):")
        print(f"  {'#':<3} {'Ticker':<8} {'Entrada':<12} {'Strike':>9} {'Premium':>9} "
              f"{'ValActual':>9} {'PnL EUR':>9} {'PnL%':>7} {'DTE':>5}")
        print(f"  {'-'*85}")

        total_opt_unrealized = 0
        for i, opt in enumerate(portfolio['option_positions'], 1):
            ticker = opt['ticker']
            current_val = opt.get('current_value', opt['premium_paid'])
            opt_pnl = current_val - opt['premium_paid']
            opt_pnl_pct = (opt_pnl / opt['premium_paid']) * 100 if opt['premium_paid'] > 0 else 0
            entry_date = datetime.fromisoformat(opt['entry_date'])
            days_elapsed = (now - entry_date).days
            remaining_dte = max(opt['dte_at_entry'] - days_elapsed, 0)
            entry_date_str = entry_date.strftime('%Y-%m-%d')

            print(f"  {i:<3} {ticker:<8} {entry_date_str:<12} ${opt['strike']:>8.2f} EUR{opt['premium_paid']:>7.0f} "
                  f"EUR{current_val:>7.0f} EUR{opt_pnl:>+7.0f} {opt_pnl_pct:>+6.1f}% {remaining_dte:>4}d")
            total_opt_unrealized += opt_pnl

        print(f"  {'-'*85}")
        print(f"  {'':>3} {'TOTAL':<8} {'':>12} {'':>9} {'':>9} {'':>9} EUR{total_opt_unrealized:>+7.0f}")

    if not portfolio['positions'] and not portfolio.get('option_positions'):
        print(f"\n  No hay posiciones abiertas.")

    # Resumen de costes
    total_costs_closed = sum(t.get('total_costs', 0) for t in portfolio['closed_trades'])
    total_costs_open = sum(
        c.get('commission', 0) + c.get('slippage', 0) + c.get('spread', 0)
        for pos in portfolio['positions']
        for c in [pos.get('costs', {})]
    )

    if total_costs_closed > 0 or total_costs_open > 0:
        print(f"\n  COSTES ACUMULADOS:")
        total_comm = sum(
            t.get('costs', {}).get('commission_entry', 0) + t.get('costs', {}).get('commission_exit', 0)
            for t in portfolio['closed_trades']
        ) + sum(pos.get('costs', {}).get('commission', 0) for pos in portfolio['positions'])
        total_slip = sum(
            t.get('costs', {}).get('slippage_entry', 0) + t.get('costs', {}).get('slippage_exit', 0)
            for t in portfolio['closed_trades']
        ) + sum(pos.get('costs', {}).get('slippage', 0) for pos in portfolio['positions'])
        total_spread = sum(
            t.get('costs', {}).get('spread_entry', 0) + t.get('costs', {}).get('spread_exit', 0)
            for t in portfolio['closed_trades']
        ) + sum(pos.get('costs', {}).get('spread', 0) for pos in portfolio['positions'])

        print(f"    Comisiones:  EUR {total_comm:.2f}")
        print(f"    Slippage:    EUR {total_slip:.2f}")
        print(f"    Spread:      EUR {total_spread:.2f}")
        print(f"    TOTAL:       EUR {total_comm + total_slip + total_spread:.2f}")

    # Rendimiento de trades cerrados
    if portfolio['closed_trades']:
        winners = [t for t in portfolio['closed_trades'] if t['pnl_eur'] > 0]
        losers = [t for t in portfolio['closed_trades'] if t['pnl_eur'] <= 0]
        total_realized = sum(t['pnl_eur'] for t in portfolio['closed_trades'])
        gross_profit = sum(t['pnl_eur'] for t in winners) if winners else 0
        gross_loss = abs(sum(t['pnl_eur'] for t in losers)) if losers else 0.01

        print(f"\n  RENDIMIENTO (trades cerrados):")
        print(f"    Trades: {len(portfolio['closed_trades'])} | Win: {len(winners)} ({len(winners)/len(portfolio['closed_trades'])*100:.0f}%)")
        print(f"    PnL realizado: EUR {total_realized:+,.2f}")
        print(f"    Profit Factor: {gross_profit/gross_loss:.2f}")

    print()


# =============================================================================
# HISTORY — Historial de trades cerrados
# =============================================================================

def show_history(portfolio: dict):
    """Muestra el historial completo de trades cerrados (acciones + opciones)."""
    trades = portfolio['closed_trades']
    opt_trades = portfolio.get('closed_option_trades', [])

    print(f"\n{'='*110}")
    print(f"  PAPER TRADING v8 — Historial de trades ({len(trades)} stocks + {len(opt_trades)} opciones)")
    print(f"{'='*110}")

    if not trades and not opt_trades:
        print("  No hay trades cerrados todavia.\n")
        return

    # Trades de acciones
    if trades:
        print(f"\n  ACCIONES ({len(trades)} trades):")
        print(f"  {'#':<4} {'Entrada':<12} {'Salida':<12} {'Ticker':<8} {'P.Entry':>9} {'P.Exit':>9} "
              f"{'PnL EUR':>9} {'PnL%':>7} {'Dias':>5} {'MaxR':>6} {'Razon':<16} {'Costes':>8}")
        print(f"  {'-'*108}")

        for i, t in enumerate(trades, 1):
            print(f"  {i:<4} {t['entry_date']:<12} {t['exit_date']:<12} {t['ticker']:<8} "
                  f"${t['entry_price']:>8.2f} ${t['exit_price']:>8.2f} "
                  f"EUR{t['pnl_eur']:>+7.0f} {t['pnl_pct']:>+6.1f}% {t['bars_held']:>4}d "
                  f"{t['max_r_mult']:>+5.1f}R {t['exit_reason']:<16} EUR{t.get('total_costs', 0):>6.2f}")

        print(f"  {'-'*108}")
        total_pnl = sum(t['pnl_eur'] for t in trades)
        total_costs = sum(t.get('total_costs', 0) for t in trades)
        winners = [t for t in trades if t['pnl_eur'] > 0]
        losers = [t for t in trades if t['pnl_eur'] <= 0]
        gross_profit = sum(t['pnl_eur'] for t in winners) if winners else 0
        gross_loss = abs(sum(t['pnl_eur'] for t in losers)) if losers else 0.01

        print(f"\n  TOTALES ACCIONES:")
        print(f"    Trades: {len(trades)} | Win: {len(winners)} ({len(winners)/len(trades)*100:.0f}%) | "
              f"PnL: EUR {total_pnl:+,.2f} | PF: {gross_profit/gross_loss:.2f} | Costes: EUR {total_costs:.2f}")

        if winners:
            avg_win = np.mean([t['pnl_pct'] for t in winners])
            avg_win_days = np.mean([t['bars_held'] for t in winners])
            print(f"    Avg Win:  {avg_win:+.1f}% | {avg_win_days:.0f} dias")
        if losers:
            avg_loss = np.mean([t['pnl_pct'] for t in losers])
            avg_loss_days = np.mean([t['bars_held'] for t in losers])
            print(f"    Avg Loss: {avg_loss:+.1f}% | {avg_loss_days:.0f} dias")

        fat_tails = sum(1 for t in trades if t['max_r_mult'] >= 3.0)
        print(f"    Fat tails (+3R): {fat_tails}/{len(trades)} ({fat_tails/len(trades)*100:.0f}%)")

    # v8: Trades de opciones
    if opt_trades:
        print(f"\n  OPCIONES v8 ({len(opt_trades)} trades):")
        print(f"  {'#':<4} {'Entrada':<12} {'Salida':<12} {'Ticker':<8} {'Strike':>9} "
              f"{'PremIn':>8} {'PremOut':>8} {'PnL EUR':>9} {'PnL%':>7} {'Dias':>5} {'Razon':<12}")
        print(f"  {'-'*108}")

        for i, t in enumerate(opt_trades, 1):
            entry_str = t['entry_date'][:10] if len(t['entry_date']) > 10 else t['entry_date']
            print(f"  {i:<4} {entry_str:<12} {t['exit_date']:<12} {t['ticker']:<8} "
                  f"${t['strike']:>8.2f} ${t['entry_option_price']:>7.2f} ${t['exit_option_price']:>7.2f} "
                  f"EUR{t['pnl_eur']:>+7.0f} {t['pnl_pct']:>+6.1f}% {t['days_held']:>4}d {t['exit_reason']:<12}")

        total_opt_pnl = sum(t['pnl_eur'] for t in opt_trades)
        opt_winners = [t for t in opt_trades if t['pnl_eur'] > 0]
        print(f"\n  TOTALES OPCIONES:")
        print(f"    Trades: {len(opt_trades)} | Win: {len(opt_winners)} | PnL: EUR {total_opt_pnl:+,.2f}")

    # Resumen combinado
    all_pnl = sum(t['pnl_eur'] for t in trades) + sum(t['pnl_eur'] for t in opt_trades)
    all_count = len(trades) + len(opt_trades)
    if all_count > 0:
        print(f"\n  TOTAL COMBINADO: {all_count} trades | PnL: EUR {all_pnl:+,.2f}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paper Trading v3.0 — Momentum Breakout v8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python paper_trading.py --reset                  # Crear cartera nueva (EUR 10,000)
  python paper_trading.py --reset --capital 25000  # Crear con capital personalizado
  python paper_trading.py --scan                   # Escaneo diario + ejecutar trades
  python paper_trading.py --status                 # Ver estado actual
  python paper_trading.py --status -p 6861.T=343.60 -p BHP.AX=34.88  # Con precios DEGIRO
  python paper_trading.py --history                # Ver historial de trades
        """
    )

    parser.add_argument('--scan', action='store_true', help='Escaneo diario + ejecutar trades')
    parser.add_argument('--status', action='store_true', help='Estado actual de la cartera')
    parser.add_argument('--history', action='store_true', help='Historial de trades cerrados')
    parser.add_argument('--reset', action='store_true', help='Reset completo (nueva cartera)')
    parser.add_argument('--capital', type=float, default=10000, help='Capital inicial en EUR (default: 10000)')
    parser.add_argument('-p', '--price', action='append', metavar='TICKER=PRECIO',
                        help='Precio manual DEGIRO (EUR). Ej: -p 6861.T=343.60 -p BHP.AX=34.88')

    args = parser.parse_args()

    # RESET
    if args.reset:
        portfolio = create_portfolio(args.capital)
        save_portfolio(portfolio)
        print(f"\n  Portfolio creado: EUR {args.capital:,.0f}")
        print(f"  Guardado en: {PORTFOLIO_FILE}\n")
        return

    # Cargar portfolio existente
    portfolio = load_portfolio()
    if portfolio is None:
        print(f"\n  No existe portfolio. Ejecuta primero:")
        print(f"  python paper_trading.py --reset --capital {args.capital:.0f}\n")
        return

    # STATUS
    if args.status:
        # Parsear precios manuales: -p TICKER=PRECIO -p TICKER=PRECIO
        manual_prices = {}
        if args.price:
            for item in args.price:
                if '=' in item:
                    tk, pr = item.split('=', 1)
                    try:
                        manual_prices[tk.strip()] = float(pr.strip())
                    except ValueError:
                        print(f"  WARN: precio invalido ignorado: {item}")
        show_status(portfolio, manual_prices=manual_prices)
        return

    # HISTORY
    if args.history:
        show_history(portfolio)
        return

    # SCAN (default)
    if args.scan or not (args.status or args.history or args.reset):
        run_scan(portfolio)
        return


if __name__ == "__main__":
    main()
