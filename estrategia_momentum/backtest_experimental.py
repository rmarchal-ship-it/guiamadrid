#!/usr/bin/env python3
"""
BACKTEST v8 — Momentum Breakout con Opciones CALL + Universo Expandido

Variante ganadora: v7 base + opciones CALL + universo 225 tickers + 10 posiciones.
Validado a 240 meses: +37,780% total, +34.6% anualizado, PF 2.89, MaxDD -42.6%.
Validado a 36 meses: +346.2% total, PF 3.34, MaxDD -21.4%.

Cambios v7+ → v8 (13 Feb 2026):
  - Universo: 112 → 225 tickers (sectores + geografias nuevas)
  - Max posiciones: 7 → 10 (grid test 7/8/10/12 confirmo 10 optimo)
  - Resultado: PF 1.77 → 2.89, MaxDD -59.9% → -42.6%

Tests disponibles (usar --test):
  Baseline: v8 base (solo acciones/ETFs, 10 posiciones)
  Test A: ETFs apalancados con risk ajustado (DESCARTADO)
  Test B: v8 Opciones CALL 120 DTE 5% ITM sin stop — GANADOR
  Test C: Combinado A+B (DESCARTADO)

Reglas opciones (Test B):
  - CALL 5% ITM, vencimiento mensual ~120 DTE
  - IVR < 40 (solo comprar opciones baratas)
  - Cierre a 45 DTE restantes (antes de theta acelerado)
  - Sin stop loss (riesgo = prima pagada)
  - Max 2 opciones simultaneas, prioridad sobre stocks
  - Position size: 14% del equity por opcion

Uso recomendado:
  python3 backtest_experimental.py --months 240 --test b --verbose
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_breakout import (
    MomentumEngine, calculate_atr, ASSETS
)


# =============================================================================
# CONSTANTES
# =============================================================================

# Factor de apalancamiento por ticker
LEVERAGE_FACTORS = {
    # 2x
    'UGL': 2.0,    # 2x Gold
    'AGQ': 2.0,    # 2x Silver
    'UCO': 2.0,    # 2x Oil
    # 3x (ya en ASSETS)
    'TQQQ': 3.0,
    'SPXL': 3.0,
    'TNA': 3.0,
    # 3x (nuevos)
    'NUGT': 3.0,   # 3x Gold Miners
    'BOIL': 3.0,   # 3x Natural Gas
    'ERX': 3.0,    # 3x Energy
    'SOXL': 3.0,   # 3x Semiconductors
    'UDOW': 3.0,   # 3x Dow Jones
    'TMF': 3.0,    # 3x Bonds 20+
}

# Tickers apalancados nuevos (no estan en ASSETS)
LEVERAGED_ADDITIONS = ['UGL', 'AGQ', 'UCO', 'NUGT', 'BOIL', 'ERX', 'SOXL', 'UDOW', 'TMF']

# Tickers elegibles para opciones (US stocks + ETFs liquidos)
OPTIONS_ELIGIBLE = [
    # US Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
    'ORCL', 'CRM', 'ADBE', 'AMD', 'INTC', 'CSCO', 'QCOM',
    # US Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'AXP',
    # US Health
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT',
    # US Consumer
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT',
    # US Industrial — NUEVO
    'CAT', 'HON', 'GE', 'UNP', 'DE', 'RTX', 'LMT', 'MMM', 'EMR', 'ETN',
    # US Energy — NUEVO
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'OXY', 'HAL',
    # US Utility — NUEVO
    'NEE', 'SO', 'DUK', 'D', 'AEP',
    # US Real Estate — NUEVO
    'AMT', 'PLD', 'EQIX', 'SPG', 'PSA', 'DLR', 'O',
    # US Telecom — NUEVO
    'T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'TMUS',
    # China ADRs — NUEVO
    'BABA', 'JD', 'PDD', 'BIDU',
    # ETFs
    'QQQ', 'SPY', 'IWM', 'DIA', 'GLD', 'SLV', 'XLE', 'TLT',
    'TQQQ', 'SPXL', 'TNA', 'BITO',
    'SMH', 'XBI', 'XLU', 'XLI', 'XLF', 'XLV',
    'EEM', 'EFA', 'HYG',
]

# v8.1: Tickers exentos del filtro macro (correlacion <= 0.15 con SPY, 2 años)
MACRO_EXEMPT = {
    'ULVR.L', 'ED', 'NESN.SW', 'TEF.MC', 'SHY', 'CSL.AX', 'RIO.AX', 'BATS.L',
    'IBE.MC', 'AD.AS', 'DUK', '9432.T', 'IEF', 'SAN.PA', 'NOVN.SW', 'EXC',
    'ABI.BR', 'SO', 'DTE.DE', 'AMT', 'ENEL.MI', 'BHP.AX', 'JNJ', 'T', '4063.T',
    'WBC.AX', 'WDS.AX', 'AEP', 'CORN', 'MUV2.DE', 'KO', 'WEAT', 'AI.PA',
    'DGE.L', 'VZ', 'ROG.SW', 'FMG.AX', 'UNG', '6501.T', 'LSEG.L', 'CBA.AX',
    'NAB.AX', '6861.T', 'WEC', 'RI.PA', '6758.T', 'TLT', 'ERIC-B.ST', 'PG',
    '8035.T', 'PEP', 'ENI.MI', 'ZURN.SW', 'CS.PA', '7267.T', 'CCI', '8306.T',
    '7203.T', '9984.T', 'GLD', 'PHIA.AS', 'XEL', 'NOVO-B.CO', 'OR.PA', 'ALV.DE',
    'AGG', 'UNH', 'LMT', '0700.HK', 'BMW.DE', 'DG.PA', 'BNP.PA', 'TIP', 'D',
    'NESTE.HE', 'BMY', 'MC.PA', 'GSK', 'ADS.DE', 'MBG.DE', 'O',
}  # 81 tickers, corr <= 0.15

# v8.1b: Solo tickers con correlacion NEGATIVA con SPY (20 tickers)
MACRO_EXEMPT_NEG = {
    'ULVR.L', 'ED', 'NESN.SW', 'TEF.MC', 'SHY', 'CSL.AX', 'RIO.AX', 'BATS.L',
    'IBE.MC', 'AD.AS', 'DUK', '9432.T', 'IEF', 'SAN.PA', 'NOVN.SW', 'EXC',
    'ABI.BR', 'SO', 'DTE.DE', 'AMT',
}  # 20 tickers, corr < 0

# Tickers base (todo el universo, crypto incluido via yfinance -USD)
BASE_TICKERS = [t for t, v in ASSETS.items()]

# Universo expandido (base + apalancados nuevos)
EXPANDED_TICKERS = BASE_TICKERS + LEVERAGED_ADDITIONS

# Config v8 (10 posiciones, 225 tickers, time exit trailing only 3xATR a 8 bars)
CONFIG = {
    'initial_capital': 10000,
    'target_risk_per_trade_pct': 2.0,
    'max_positions': 10,
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
    'slippage_pct': 0.10,
    # Opciones
    'option_dte': 120,
    'option_itm_pct': 0.05,        # 5% ITM (strike = spot * 0.95)
    'risk_free_rate': 0.043,
    'hvol_window': 30,
    'option_spread_pct': 3.0,
    'option_close_dte': 45,         # Cerrar opcion cuando quedan 45 DTE (antes de theta acelerado)
    'option_max_ivr': 40,           # Solo comprar opciones si IV Rank < 40% (opciones baratas)
    'option_ivr_window': 252,       # Ventana para calcular IV Rank (1 ano)
    'option_position_pct': 0.14,    # ~14% del equity por posicion de opciones (1/7)
    'max_option_positions': 2,
}


# =============================================================================
# BLACK-SCHOLES (copiado de backtest_opciones.py)
# =============================================================================

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'price': max(S - K, 0), 'delta': 1.0 if S > K else 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return {'price': max(price, 0.01), 'delta': delta}


def historical_volatility(close_prices, window=30):
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


def monthly_expiration_dte(entry_date, target_dte=120):
    """
    Calcula el DTE real al vencimiento mensual mas cercano a target_dte.
    Opciones mensuales vencen el 3er viernes de cada mes.
    Devuelve el DTE ajustado al vencimiento mensual mas cercano.
    """
    target_date = entry_date + timedelta(days=target_dte)

    # Encontrar 3er viernes del mes del target_date
    year, month = target_date.year, target_date.month
    # Dia 1 del mes
    first_day = datetime(year, month, 1)
    # Dia de la semana del 1 (0=lunes, 4=viernes)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)

    # Tambien calcular 3er viernes del mes anterior y siguiente
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
        if third_friday_m > entry_date:  # solo futuros
            candidates.append(third_friday_m)

    if not candidates:
        return target_dte  # fallback

    # Elegir el mas cercano a target_dte
    best = min(candidates, key=lambda d: abs((d - entry_date).days - target_dte))
    return (best - entry_date).days


def iv_rank(hvol_series, current_idx, window=252):
    """
    IV Rank: percentil de la IV actual respecto a los ultimos 'window' dias.
    IVR = (IV_actual - IV_min) / (IV_max - IV_min) * 100
    Rango: 0-100. Bajo = opciones baratas. Alto = opciones caras.
    """
    start = max(0, current_idx - window)
    hist = hvol_series.iloc[start:current_idx + 1].dropna()
    if len(hist) < 20:
        return 50.0  # sin datos suficientes, valor neutral
    iv_now = hist.iloc[-1]
    iv_min = hist.min()
    iv_max = hist.max()
    if iv_max == iv_min:
        return 50.0
    return (iv_now - iv_min) / (iv_max - iv_min) * 100


# =============================================================================
# TRADE CLASS v8 (10 posiciones, trailing 4xATR, time exit trailing only)
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

        emergency_level = self.entry_price * (1 - CONFIG['emergency_stop_pct'])
        if low <= emergency_level:
            self._close(emergency_level * (1 - CONFIG['slippage_pct'] / 100), 'emergency_stop')
            return {'type': 'full_exit', 'reason': 'emergency_stop'}

        if self.trailing_active and self.trailing_stop is not None:
            if low <= self.trailing_stop:
                self._close(self.trailing_stop * (1 - CONFIG['slippage_pct'] / 100), 'trailing_stop')
                return {'type': 'full_exit', 'reason': 'trailing_stop'}

        if r_mult >= CONFIG['trail_trigger_r']:
            chandelier = self.highest_since - (current_atr * CONFIG['trail_atr_mult'])
            if not self.trailing_active:
                self.trailing_active = True
                self.trailing_stop = chandelier
            elif chandelier > self.trailing_stop:
                self.trailing_stop = chandelier

        # TIME EXIT: tras max_hold_bars, activar trailing (nunca forzar salida)
        # v8: 8 bars, trailing 3xATR. Elimina time_exit forzados
        # que tenian 0% win rate y -248k EUR en 20 anos.
        if self.bars_held >= CONFIG['max_hold_bars']:
            if not self.trailing_active:
                trail_mult = CONFIG.get('time_exit_trail_atr_mult', 3.0)
                chandelier = self.highest_since - (current_atr * trail_mult)
                breakeven = self.entry_price * (1 + CONFIG['slippage_pct'] / 100)
                self.trailing_active = True
                if close <= self.entry_price:
                    # Perdiendo: trailing apretado (3xATR o 5% bajo maximo)
                    self.trailing_stop = max(chandelier, self.entry_price * 0.95)
                else:
                    # Ganando: trailing a breakeven minimo
                    self.trailing_stop = max(chandelier, breakeven)

        return None

    def _close(self, exit_price, reason):
        self.pnl_euros = (exit_price - self.entry_price) * self.position_units
        self.pnl_pct = (self.pnl_euros / self.position_euros) * 100 if self.position_euros > 0 else 0
        self.exit_price = exit_price
        self.exit_reason = reason


# =============================================================================
# OPTION TRADE V2 (sin stop, cierre a 45 DTE restantes)
# =============================================================================
# Logica: comprar CALL 5% ITM a 120 DTE, cerrar cuando quedan 45 DTE.
# Holding = 75 dias. Sin trailing, sin stop. Sales antes de que theta acelere.
# El riesgo maximo es la prima pagada.

@dataclass
class OptionTradeV2:
    ticker: str
    entry_date: datetime
    entry_stock_price: float
    strike: float
    dte_at_entry: int
    entry_option_price: float
    entry_iv: float
    num_contracts: float
    position_euros: float   # premium pagada = max loss

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
        current_option_price *= (1 - CONFIG['option_spread_pct'] / 100 / 2)

        self.max_option_value = max(self.max_option_value, current_option_price)

        option_return = (current_option_price / self.entry_option_price) - 1 if self.entry_option_price > 0 else 0
        self.max_r_mult = max(self.max_r_mult, option_return)

        # EXPIRACION (safety)
        if remaining_dte <= 0:
            intrinsic = max(stock_price - self.strike, 0)
            intrinsic *= (1 - CONFIG['option_spread_pct'] / 100 / 2)
            self._close(intrinsic, 'expiration')
            return {'type': 'full_exit', 'reason': 'expiration'}

        # CIERRE A 45 DTE RESTANTES (antes de que theta acelere)
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
# EQUITY TRACKER (extendido)
# =============================================================================

class EquityTracker:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.max_equity = initial_capital
        self.open_positions = 0
        self.open_options = 0

    def get_position_size(self, ticker, current_atr, price, use_leverage_scaling=False):
        base_risk_pct = CONFIG['target_risk_per_trade_pct'] / 100
        if use_leverage_scaling:
            leverage = LEVERAGE_FACTORS.get(ticker, 1.0)
            risk_pct = base_risk_pct * leverage
        else:
            risk_pct = base_risk_pct

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

    def get_option_size(self, option_price):
        """Posicion de opciones = 20% del equity."""
        position_pct = CONFIG.get('option_position_pct', 0.20)
        max_premium = self.equity * position_pct
        if option_price <= 0:
            return {'contracts': 0, 'premium': 0}
        contracts = max_premium / (option_price * 100)
        premium = contracts * option_price * 100
        return {'contracts': contracts, 'premium': premium}

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
# DATA DOWNLOAD + CACHE
# =============================================================================
# Cache permite reproducir backtests exactos guardando datos de yfinance.
# Uso: download_data(ticker, months) descarga fresco.
#      download_data(ticker, months, cache_date='2026-03-01') lee del cache.
#      save_data_cache(all_data, months) guarda snapshot completo.
#      load_data_cache(cache_date, months) carga snapshot completo.

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')


def save_data_cache(all_data, months, cache_date=None, source='yahoo'):
    """Guarda todos los DataFrames descargados como CSV + ticker mapping."""
    import json
    if cache_date is None:
        cache_date = datetime.now().strftime('%Y-%m-%d')
    cache_dir = os.path.join(DATA_CACHE_DIR, f"{cache_date}_{months}m_{source}")
    os.makedirs(cache_dir, exist_ok=True)
    ticker_map = {}
    for ticker, df in all_data.items():
        safe_name = ticker.replace('.', '_').replace('/', '_')
        fname = f"{safe_name}.csv.gz"
        df.to_csv(os.path.join(cache_dir, fname), compression='gzip')
        ticker_map[fname] = ticker
    with open(os.path.join(cache_dir, '_tickers.json'), 'w') as f:
        json.dump(ticker_map, f, indent=2)
    meta_path = os.path.join(cache_dir, '_metadata.txt')
    with open(meta_path, 'w') as f:
        f.write(f"date: {cache_date}\n")
        f.write(f"months: {months}\n")
        f.write(f"source: {source}\n")
        f.write(f"tickers: {len(all_data)}\n")
        f.write(f"saved: {datetime.now().isoformat()}\n")
    print(f"  Cache guardado: {cache_dir} ({len(all_data)} tickers)")
    return cache_date


def load_data_cache(cache_date, months, source='yahoo'):
    """Carga todos los DataFrames desde cache CSV usando ticker mapping."""
    import json
    # Intentar con source tag primero, fallback a formato antiguo (sin source)
    cache_dir = os.path.join(DATA_CACHE_DIR, f"{cache_date}_{months}m_{source}")
    if not os.path.isdir(cache_dir):
        cache_dir_old = os.path.join(DATA_CACHE_DIR, f"{cache_date}_{months}m")
        if os.path.isdir(cache_dir_old):
            cache_dir = cache_dir_old
            print(f"  (usando cache formato antiguo sin tag de fuente)")
        else:
            print(f"  ERROR: Cache no encontrado: {cache_dir}")
            return None
    mapping_path = os.path.join(cache_dir, '_tickers.json')
    ticker_map = {}
    if os.path.isfile(mapping_path):
        with open(mapping_path) as f:
            ticker_map = json.load(f)
    all_data = {}
    for fname in os.listdir(cache_dir):
        if not fname.endswith('.csv.gz'):
            continue
        if fname in ticker_map:
            ticker = ticker_map[fname]
        else:
            ticker = fname.replace('.csv.gz', '').replace('_', '.', 1)
        df = pd.read_csv(os.path.join(cache_dir, fname), compression='gzip',
                         index_col=0, parse_dates=True)
        all_data[ticker] = df
    print(f"  Cache cargado: {cache_dir} ({len(all_data)} tickers)")
    return all_data if all_data else None


def list_data_caches():
    """Lista todos los caches disponibles."""
    if not os.path.isdir(DATA_CACHE_DIR):
        return []
    caches = []
    for d in sorted(os.listdir(DATA_CACHE_DIR)):
        meta_path = os.path.join(DATA_CACHE_DIR, d, '_metadata.txt')
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                caches.append((d, f.read().strip()))
    return caches


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
# SIGNAL GENERATION + RANKING (compartido por todos los tests)
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


def find_candidates(signals_data, active_trades, current_date, is_macro_ok,
                     macro_exempt_set=None):
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
        # v8.1: tickers exentos ignoran el filtro macro
        if not is_macro_ok and not (macro_exempt_set and ticker in macro_exempt_set):
            continue
        prev_atr = df['ATR'].iloc[idx - 1]
        if pd.isna(prev_atr) or prev_atr <= 0:
            continue
        candidates.append((ticker, idx, prev_atr))
    return candidates


# =============================================================================
# CORE BACKTEST ENGINE (parametrizado)
# =============================================================================

def run_backtest(months, tickers, label, use_leverage_scaling=False,
                 use_options=False, macro_exempt_set=None, verbose=False):
    n_tickers = len(tickers)
    print(f"\n{'='*70}")
    print(f"  {label} -- {months} MESES -- {n_tickers} tickers")
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
    active_options = {}      # ticker -> OptionTradeV2
    all_trades = []
    all_option_trades = []

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
            all_option_trades.append(opt)
            if verbose:
                pnl_pct = (opt.pnl_euros / opt.position_euros * 100) if opt.position_euros else 0
                pnl_sign = '+' if opt.pnl_euros >= 0 else ''
                print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE OPT {ticker:8} | "
                      f"{opt.exit_reason:<15} | P&L EUR {pnl_sign}{opt.pnl_euros:.0f} ({pnl_sign}{pnl_pct:.1f}%) | "
                      f"Pos: {tracker.open_positions}/10 | Equity: EUR {tracker.equity:,.0f}")

        # 3. BUSCAR NUEVAS SENALES
        # Filtro macro usa Close[T-2] (dia del breakout, alineado con datos de la senal)
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
                # PRIORIDAD: opcion antes que accion para tickers elegibles
                open_as_option = False
                current_ivr = None
                if use_options and ticker in OPTIONS_ELIGIBLE and tracker.open_options < CONFIG['max_option_positions']:
                    # Filtro IVR: solo comprar opciones si IV Rank < umbral
                    hvol_series = df['HVOL']
                    current_ivr = iv_rank(hvol_series, idx, CONFIG.get('option_ivr_window', 252))
                    max_ivr = CONFIG.get('option_max_ivr', 40)
                    if current_ivr < max_ivr:
                        open_as_option = True

                if open_as_option:
                    # --- OPCION CALL (prioridad sobre accion) ---
                    stock_price = bar['Open']
                    strike = stock_price * (1 - CONFIG['option_itm_pct'])  # 5% ITM

                    # DTE: vencimiento mensual mas cercano a 120 dias
                    actual_dte = monthly_expiration_dte(current_date, CONFIG['option_dte'])
                    T = actual_dte / 365.0

                    iv = df['HVOL'].iloc[idx]
                    if pd.isna(iv) or iv <= 0:
                        iv = 0.30  # fallback
                    bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], iv)
                    option_price = bs['price']
                    option_price *= (1 + CONFIG['option_spread_pct'] / 100 / 2)  # spread de entrada

                    size = tracker.get_option_size(option_price)
                    if size['premium'] < 50:
                        continue

                    opt = OptionTradeV2(
                        ticker=ticker,
                        entry_date=current_date,
                        entry_stock_price=stock_price,
                        strike=strike,
                        dte_at_entry=actual_dte,
                        entry_option_price=option_price,
                        entry_iv=iv,
                        num_contracts=size['contracts'],
                        position_euros=size['premium'],
                    )
                    active_options[ticker] = opt
                    tracker.open_positions += 1
                    tracker.open_options += 1

                    if verbose:
                        print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN OPT {ticker:8} | "
                              f"K=${strike:.2f} IV={iv:.0%} IVR={current_ivr:.0f} "
                              f"{actual_dte}DTE Prem=${option_price:.2f} x{size['contracts']:.2f}c = EUR {size['premium']:.0f}")
                else:
                    # --- ACCION/ETF ---
                    size_info = tracker.get_position_size(ticker, prev_atr, bar['Open'], use_leverage_scaling)
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

    # Fat tails (stock trades)
    stock_gt_3r = sum(1 for t in all_trades if t.max_r_mult >= 3.0)
    # Option home runs (>100% return)
    opt_home_runs = sum(1 for t in all_option_trades if t.pnl_pct >= 100)

    best_trade = max(combined_trades, key=lambda t: t.pnl_pct)
    worst_trade = min(combined_trades, key=lambda t: t.pnl_pct)

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
""")

    # Razones de salida
    exit_reasons = {}
    for t in combined_trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    print("  RAZONES DE SALIDA:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"     {reason:20} {count:3} ({count/total_count*100:.1f}%)")

    # Detalle de opciones si hay
    if all_option_trades:
        print(f"\n  DETALLE OPCIONES ({len(all_option_trades)} trades):")
        for i, opt in enumerate(sorted(all_option_trades, key=lambda x: x.entry_date), 1):
            entry_str = opt.entry_date.strftime('%Y-%m-%d') if opt.entry_date else '?'
            exit_str = opt.exit_date.strftime('%Y-%m-%d') if opt.exit_date else '?'
            marker = '+' if opt.pnl_euros > 0 else '-'
            print(f"     {i}. {entry_str} → {exit_str} | {opt.ticker:8} | "
                  f"K=${opt.strike:.2f} | Prem ${opt.entry_option_price:.2f} → ${opt.exit_option_price:.2f} | "
                  f"P&L EUR {opt.pnl_euros:+.0f} ({opt.pnl_pct:+.1f}%) | {opt.bars_held}d | {opt.exit_reason} {marker}")

    # Detalle de trades apalancados si hay
    if use_leverage_scaling:
        lev_trades = [t for t in all_trades if LEVERAGE_FACTORS.get(t.ticker, 1.0) > 1]
        normal_trades = [t for t in all_trades if LEVERAGE_FACTORS.get(t.ticker, 1.0) == 1]
        if lev_trades:
            lev_pnl = sum(t.pnl_euros for t in lev_trades)
            norm_pnl = sum(t.pnl_euros for t in normal_trades)
            lev_wr = sum(1 for t in lev_trades if t.pnl_euros > 0) / len(lev_trades) * 100
            print(f"\n  DESGLOSE APALANCADOS vs NORMALES:")
            print(f"     Apalancados: {len(lev_trades)} trades, P&L EUR {lev_pnl:+,.0f}, Win% {lev_wr:.1f}%")
            print(f"     Normales:    {len(normal_trades)} trades, P&L EUR {norm_pnl:+,.0f}")
            print(f"     Tickers apal. usados: {', '.join(set(t.ticker for t in lev_trades))}")

    return {
        'label': label,
        'total_trades': total_count,
        'stock_trades': len(all_trades),
        'option_trades': len(all_option_trades),
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
# MAIN
# =============================================================================

def print_comparison(results):
    print(f"""
{'='*90}
  TABLA COMPARATIVA v7/v7+
{'='*90}

  {'Variante':<25} {'Trades':<8} {'Win%':<7} {'PnL EUR':<11} {'Return%':<9} {'Annual%':<9} {'PF':<6} {'MaxDD%':<7}
  {'-'*85}""")

    for r in results:
        print(f"  {r['label']:<25} {r['total_trades']:<8} {r['win_rate']:<7.1f} "
              f"EUR{r['total_pnl_euros']:>+8,.0f}  {r['total_return_pct']:>+7.1f}%  "
              f"{r['annualized_return_pct']:>+7.1f}%  {r['profit_factor']:.2f}  {r['max_drawdown']:>5.1f}%")

    print(f"""
  {'Variante':<25} {'Stocks':<8} {'Opts':<6} {'>3R':<5} {'OptHR':<6} {'AvgWin%':<9} {'AvgLoss%':<9} {'Best Trade':<20}
  {'-'*85}""")

    for r in results:
        best_str = f"{r['best_ticker']} {r['best_pnl_pct']:+.1f}%"
        print(f"  {r['label']:<25} {r['stock_trades']:<8} {r['option_trades']:<6} "
              f"{r['stock_gt_3r']:<5} {r['opt_home_runs']:<6} "
              f"{r['avg_win_pct']:>+7.1f}%  {r['avg_loss_pct']:>+7.1f}%  {best_str:<20}")

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backtest Experimental: Apalancados + Opciones')
    parser.add_argument('--months', type=int, default=18, help='Meses de historico')
    parser.add_argument('--verbose', action='store_true', help='Detalle de trades')
    parser.add_argument('--test', choices=['all', 'a', 'b', 'c', 'baseline', 'v81', 'v81b'],
                        default='all', help='Test a ejecutar')
    parser.add_argument('--export-csv', action='store_true',
                        help='Exportar trades a CSV (historico_trades_{months}m.csv)')
    args = parser.parse_args()

    months = args.months
    v = args.verbose

    print(f"""
======================================================================
  BACKTEST EXPERIMENTAL — MOMENTUM BREAKOUT
======================================================================
  Test A: ETFs apalancados (risk 2x->4%, 3x->6%)
  Test B: Opciones CALL 120 DTE 5% ITM sin stop (max 2) — GANADOR
  Test C: Combinado (A + B)
  Baseline: v7 base (solo acciones, 7 posiciones)
  v8.1: v8 + exencion macro corr<=0.15 (81 tickers)
  v8.1b: v8 + exencion macro corr<0 solo (20 tickers)
======================================================================
    """)

    results = []

    if args.test in ('all', 'baseline'):
        r = run_backtest(months, BASE_TICKERS, "BASELINE (v7)",
                         use_leverage_scaling=False, use_options=False, verbose=v)
        if 'error' not in r:
            results.append(r)

    if args.test in ('all', 'a'):
        r = run_backtest(months, EXPANDED_TICKERS, "TEST A: Apalancados",
                         use_leverage_scaling=True, use_options=False, verbose=v)
        if 'error' not in r:
            results.append(r)

    if args.test in ('all', 'b'):
        r = run_backtest(months, BASE_TICKERS, "v7+ (Opciones CALL)",
                         use_leverage_scaling=False, use_options=True, verbose=v)
        if 'error' not in r:
            results.append(r)

    if args.test in ('all', 'c'):
        r = run_backtest(months, EXPANDED_TICKERS, "TEST C: Combinado",
                         use_leverage_scaling=True, use_options=True, verbose=v)
        if 'error' not in r:
            results.append(r)

    if args.test == 'v81':
        # v8.1: v8 con opciones + exencion macro para tickers descorrelacionados
        r_v8 = run_backtest(months, BASE_TICKERS, "v8 (referencia)",
                            use_leverage_scaling=False, use_options=True, verbose=v)
        if 'error' not in r_v8:
            results.append(r_v8)
        r_v81 = run_backtest(months, BASE_TICKERS, "v8.1 (macro exempt corr<=0.15)",
                             use_leverage_scaling=False, use_options=True,
                             macro_exempt_set=MACRO_EXEMPT, verbose=v)
        if 'error' not in r_v81:
            results.append(r_v81)

    if args.test == 'v81b':
        # v8.1b: solo tickers con correlacion NEGATIVA (20 tickers)
        r_v8 = run_backtest(months, BASE_TICKERS, "v8 (referencia)",
                            use_leverage_scaling=False, use_options=True, verbose=v)
        if 'error' not in r_v8:
            results.append(r_v8)
        r_v81b = run_backtest(months, BASE_TICKERS, "v8.1b (macro exempt corr<0)",
                              use_leverage_scaling=False, use_options=True,
                              macro_exempt_set=MACRO_EXEMPT_NEG, verbose=v)
        if 'error' not in r_v81b:
            results.append(r_v81b)

    if len(results) >= 2:
        print_comparison(results)

    # --- Export CSV si se pide ---
    if args.export_csv and results:
        import csv
        for r in results:
            all_t = r.get('all_trades', [])
            if not all_t:
                continue
            fname = f"historico_trades_{months}m.csv"
            with open(fname, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['entry_date', 'exit_date', 'ticker', 'entry_price', 'exit_price',
                            'position_eur', 'units', 'pnl_eur', 'pnl_pct', 'bars_held',
                            'max_r_mult', 'exit_reason'])
                for t in all_t:
                    w.writerow([
                        t.entry_date.strftime('%Y-%m-%d') if t.entry_date else '',
                        t.exit_date.strftime('%Y-%m-%d') if t.exit_date else '',
                        t.ticker,
                        round(t.entry_price, 2),
                        round(t.exit_price, 2) if t.exit_price else '',
                        round(t.position_euros, 0),
                        round(t.position_units, 4),
                        round(t.pnl_euros, 2),
                        round(t.pnl_pct, 2),
                        t.bars_held,
                        round(t.max_r_mult, 2),
                        t.exit_reason or ''
                    ])
            print(f"\n  CSV exportado: {fname} ({len(all_t)} trades)")
            break  # solo exportar el primer resultado


if __name__ == "__main__":
    main()
