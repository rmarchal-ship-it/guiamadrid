#!/usr/bin/env python3
"""
BACKTEST OUTSIDERS — Momentum Breakout v8 para activos NO incluidos en v12
Creado: 28 Feb 2026
Modificado: 28 Feb 2026 — Añadidas opciones ITM 5% DTE120/exit DTE45

Universo: 20 Crypto + 10 Commodity Futures = 30 tickers
  - 20 criptomonedas (operan 24/7 incluyendo fines de semana)
  - 10 futuros de commodities (L-V, subyacentes directos, no ETFs)

Opciones: 10 futuros con ETF proxy (GLD, SLV, USO, UNG, CPER, PPLT, WEAT, CORN, SOYB, JO)
  - Compra Call ITM 5%, DTE ~120 (vencimiento mensual)
  - Salida automática a DTE 45 (antes de theta acelerado)
  - Max opciones = max_positions // 3 (nunca > 1/3 del total)
  - Filtro IVR < 40 (solo comprar opciones baratas)

Macro filter: SPY > SMA50 (mismo que v8 original)
Config señales: identica a v8 (KER 0.40, RSI 50-75, Vol 1.3x, BP 20)

Uso:
  python3 backtest_outsiders.py --months 36 --verbose
  python3 backtest_outsiders.py --months 36 --grid        # Grid test posiciones
  python3 backtest_outsiders.py --months 36 --pos 5       # Override max posiciones
  python3 backtest_outsiders.py --months 36 --no-options  # Sin opciones (baseline)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from scipy.stats import norm
import warnings
import argparse
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_breakout import MomentumEngine, calculate_atr


# =============================================================================
# UNIVERSO OUTSIDERS — 27 tickers NO incluidos en v8/v12
# =============================================================================

OUTSIDER_TICKERS = {
    # === CRIPTO (20) — operan 24/7 ===
    'BTC-USD':       {'name': 'Bitcoin',          'category': 'CRYPTO'},
    'ETH-USD':       {'name': 'Ethereum',         'category': 'CRYPTO'},
    'SOL-USD':       {'name': 'Solana',           'category': 'CRYPTO'},
    'XRP-USD':       {'name': 'XRP',              'category': 'CRYPTO'},
    'BNB-USD':       {'name': 'BNB',              'category': 'CRYPTO'},
    'DOGE-USD':      {'name': 'Dogecoin',         'category': 'CRYPTO'},
    'ADA-USD':       {'name': 'Cardano',          'category': 'CRYPTO'},
    'AVAX-USD':      {'name': 'Avalanche',        'category': 'CRYPTO'},
    'DOT-USD':       {'name': 'Polkadot',         'category': 'CRYPTO'},
    'LINK-USD':      {'name': 'Chainlink',        'category': 'CRYPTO'},
    'NEAR-USD':      {'name': 'NEAR Protocol',    'category': 'CRYPTO'},
    'LTC-USD':       {'name': 'Litecoin',         'category': 'CRYPTO'},
    'BCH-USD':       {'name': 'Bitcoin Cash',     'category': 'CRYPTO'},
    'AAVE-USD':      {'name': 'Aave',             'category': 'CRYPTO'},
    'HBAR-USD':      {'name': 'Hedera',           'category': 'CRYPTO'},
    'FIL-USD':       {'name': 'Filecoin',         'category': 'CRYPTO'},
    'ATOM-USD':      {'name': 'Cosmos',           'category': 'CRYPTO'},
    'ICP-USD':       {'name': 'Internet Computer','category': 'CRYPTO'},
    'TRX-USD':       {'name': 'TRON',             'category': 'CRYPTO'},
    'MKR-USD':       {'name': 'Maker',            'category': 'CRYPTO'},

    # === COMMODITIES FUTUROS (10) — L-V, NO son los ETFs de v8 ===
    'CL=F':          {'name': 'Crude Oil WTI',    'category': 'FUTURES'},
    'NG=F':          {'name': 'Natural Gas',      'category': 'FUTURES'},
    'GC=F':          {'name': 'Gold Futures',     'category': 'FUTURES'},
    'SI=F':          {'name': 'Silver Futures',   'category': 'FUTURES'},
    'HG=F':          {'name': 'Copper Futures',   'category': 'FUTURES'},
    'PL=F':          {'name': 'Platinum Futures', 'category': 'FUTURES'},
    'ZW=F':          {'name': 'Wheat Futures',    'category': 'FUTURES'},
    'ZC=F':          {'name': 'Corn Futures',     'category': 'FUTURES'},
    'ZS=F':          {'name': 'Soybean Futures',  'category': 'FUTURES'},
    'KC=F':          {'name': 'Coffee Futures',   'category': 'FUTURES'},
}

TICKER_LIST = list(OUTSIDER_TICKERS.keys())


# =============================================================================
# OPCIONES — 6 tickers con ETF proxy más líquidos
# =============================================================================

OPTIONS_MAP = {
    # ticker señal  →  proxy ETF donde se operan las opciones en real
    # 10 futuros con ETF proxy — diversificado para evitar look-ahead bias
    'GC=F':    {'proxy': 'GLD',  'name': 'SPDR Gold Shares'},
    'SI=F':    {'proxy': 'SLV',  'name': 'iShares Silver Trust'},
    'CL=F':    {'proxy': 'USO',  'name': 'United States Oil Fund'},
    'NG=F':    {'proxy': 'UNG',  'name': 'United States NatGas Fund'},
    'HG=F':    {'proxy': 'CPER', 'name': 'United States Copper Fund'},
    'PL=F':    {'proxy': 'PPLT', 'name': 'Aberdeen Platinum ETF'},
    'ZW=F':    {'proxy': 'WEAT', 'name': 'Teucrium Wheat Fund'},
    'ZC=F':    {'proxy': 'CORN', 'name': 'Teucrium Corn Fund'},
    'ZS=F':    {'proxy': 'SOYB', 'name': 'Teucrium Soybean Fund'},
    'KC=F':    {'proxy': 'JO',   'name': 'iPath Coffee ETN'},
}


# =============================================================================
# CONFIG (identica a v8, macro = SPY, + opciones)
# =============================================================================

CONFIG = {
    'initial_capital': 10000,
    'target_risk_per_trade_pct': 2.0,
    'max_positions': 5,           # Default, override con --pos o --grid

    # Señales (mismos parametros v8)
    'ker_threshold': 0.40,
    'volume_threshold': 1.3,
    'rsi_threshold': 50,
    'rsi_max': 75,
    'breakout_period': 20,
    'longs_only': True,

    # Stops y trailing (mismos v8)
    'emergency_stop_pct': 0.15,           # futures default
    'emergency_stop_pct_crypto': 0.30,    # 2026-02-28: crypto necesita SL más amplio (ATR% ~5.6%)
    'trail_trigger_r': 2.0,
    'trail_atr_mult': 4.0,

    # Time exit (mismo v8)
    'max_hold_bars': 8,
    'time_exit_trail_atr_mult': 3.0,

    # Macro filter: SPY (mismo que v8 original)
    'use_macro_filter': True,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,

    # Costes
    'slippage_pct': 0.10,

    # Opciones (ITM 5%, DTE 120, exit DTE 45)
    'use_options': True,
    'option_dte': 120,
    'option_itm_pct': 0.05,        # 5% ITM → strike = spot * 0.95
    'option_close_dte': 45,         # cerrar cuando quedan 45 DTE
    'risk_free_rate': 0.043,
    'hvol_window': 30,
    'option_spread_pct': 3.0,       # spread bid-ask simulado
    'option_max_ivr': 40,           # solo comprar si IVR < 40%
    'option_ivr_window': 252,
    'option_position_pct': 0.20,    # 20% del equity por opcion (max loss)
}


# =============================================================================
# BLACK-SCHOLES + VOLATILIDAD
# =============================================================================

def black_scholes_call(S, K, T, r, sigma):
    """Precio call europea + delta via Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'price': max(S - K, 0), 'delta': 1.0 if S > K else 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return {'price': max(price, 0.01), 'delta': delta}


def historical_volatility(close_prices, window=30):
    """HV anualizada desde log-returns."""
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


def monthly_expiration_dte(entry_date, target_dte=120):
    """DTE real al vencimiento mensual (3er viernes) mas cercano a target_dte."""
    target_date = entry_date + timedelta(days=target_dte)
    year, month = target_date.year, target_date.month

    # Encontrar 3er viernes del mes
    first_day = datetime(year, month, 1)
    dow = first_day.weekday()  # 0=lun, 4=vie
    first_friday = first_day + timedelta(days=(4 - dow) % 7)
    third_friday = first_friday + timedelta(weeks=2)

    # Tambien revisar mes anterior y siguiente
    candidates = [third_friday]
    for delta_months in [-1, 1]:
        m = month + delta_months
        y = year
        if m < 1:
            m = 12
            y -= 1
        elif m > 12:
            m = 1
            y += 1
        fd = datetime(y, m, 1)
        dow2 = fd.weekday()
        ff = fd + timedelta(days=(4 - dow2) % 7)
        candidates.append(ff + timedelta(weeks=2))

    # Elegir el mas cercano a target_dte
    best = min(candidates, key=lambda d: abs((d - entry_date).days - target_dte))
    actual_dte = (best - entry_date).days
    return max(actual_dte, 1)


def iv_rank(hvol_series, current_idx, window=252):
    """IV Rank: percentil de IV actual vs ultimos 'window' dias. 0-100."""
    start = max(0, current_idx - window)
    hist = hvol_series.iloc[start:current_idx + 1].dropna()
    if len(hist) < 20:
        return 50.0
    iv_now = hist.iloc[-1]
    iv_min = hist.min()
    iv_max = hist.max()
    if iv_max == iv_min:
        return 50.0
    return (iv_now - iv_min) / (iv_max - iv_min) * 100


# =============================================================================
# TRADE CLASS — Spot (identica a v8)
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

        # SL diferenciado: crypto 30%, futures 15%
        cat = OUTSIDER_TICKERS.get(self.ticker, {}).get('category', 'FUTURES')
        if cat == 'CRYPTO':
            sl_pct = CONFIG.get('emergency_stop_pct_crypto', CONFIG['emergency_stop_pct'])
        else:
            sl_pct = CONFIG['emergency_stop_pct']
        emergency_level = self.entry_price * (1 - sl_pct)
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
# OPTION TRADE CLASS — Call ITM 5%, DTE 120, exit DTE 45
# =============================================================================

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
    position_euros: float       # premium pagada = max loss
    proxy_etf: str = ''

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

        # CIERRE A 45 DTE (antes de theta acelerado)
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
# EQUITY TRACKER (con soporte opciones)
# =============================================================================

class EquityTracker:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.max_equity = initial_capital
        self.open_positions = 0
        self.open_options = 0

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

    def get_option_size(self, option_price):
        """Premium = equity / max_positions (misma tajada que spot). Max loss = premium."""
        max_premium = self.equity / CONFIG['max_positions']
        if option_price <= 0:
            return {'contracts': 0, 'premium': 0}
        contracts = max_premium / (option_price * 100)
        premium = contracts * option_price * 100
        return {'contracts': contracts, 'premium': premium}

    def get_max_option_positions(self):
        """Max 1 opción si max_positions >= 3, 0 si menos."""
        mp = CONFIG['max_positions']
        if mp < 3:
            return 0
        return 1

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
# DATA DOWNLOAD (con cache para grid test + HVOL)
# =============================================================================

_data_cache = {}

def download_data(ticker, months):
    cache_key = (ticker, months)
    if cache_key in _data_cache:
        return _data_cache[cache_key]
    try:
        if months > 60:
            end = datetime.now()
            start = end - timedelta(days=months * 30)
            df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                             end=end.strftime('%Y-%m-%d'), interval='1d', progress=False)
        else:
            df = yf.download(ticker, period=f'{months}mo', interval='1d', progress=False)
        if df.empty:
            _data_cache[cache_key] = None
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        result = df if len(df) >= 50 else None
        _data_cache[cache_key] = result
        return result
    except Exception:
        _data_cache[cache_key] = None
        return None


def download_all(tickers, months, verbose=True):
    """Descarga todos los tickers + SPY (macro). Calcula ATR + HVOL."""
    all_data = {}
    failed = []
    download_list = list(tickers)
    if CONFIG['macro_ticker'] not in download_list:
        download_list.append(CONFIG['macro_ticker'])

    for i, ticker in enumerate(download_list):
        df = download_data(ticker, months)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            df['HVOL'] = historical_volatility(df['Close'], CONFIG['hvol_window'])
            all_data[ticker] = df
        else:
            failed.append(ticker)
        if verbose and ((i + 1) % 10 == 0 or i == len(download_list) - 1):
            print(f"\r  Descargados: {len(all_data)}/{len(download_list)} OK, {len(failed)} fallidos", end='')

    if verbose:
        print()
    return all_data, failed


# =============================================================================
# SIGNAL GENERATION + RANKING + MACRO
# =============================================================================

def generate_all_signals(all_data, engine):
    signals_data = {}
    total_signals = 0
    for ticker, df in all_data.items():
        if ticker == CONFIG['macro_ticker'] and ticker not in OUTSIDER_TICKERS:
            continue
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


def find_candidates(signals_data, active_trades, active_options, current_date, is_macro_ok):
    """Busca candidatos excluyendo tickers con posicion abierta (spot u opcion)."""
    occupied = set(active_trades.keys()) | set(active_options.keys())
    candidates = []
    for ticker, sd in signals_data.items():
        if ticker == CONFIG['macro_ticker'] and ticker not in OUTSIDER_TICKERS:
            continue
        if ticker in occupied:
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
# CORE BACKTEST (con opciones)
# =============================================================================

def run_backtest(months, all_data, signals_data, macro_bullish, max_positions=None,
                 verbose=False, quiet=False, use_options=True):
    """
    Ejecuta el backtest con soporte de opciones.
    use_options: True para activar opciones (default), False para baseline spot-only.
    """
    if max_positions is not None:
        CONFIG['max_positions'] = max_positions
    mp = CONFIG['max_positions']
    use_opt = use_options and CONFIG.get('use_options', True)

    n_tickers = len([t for t in OUTSIDER_TICKERS if t in all_data])
    opt_label = f"+OPT({len(OPTIONS_MAP)})" if use_opt else "SPOT"
    label = f"OUTSIDERS ({n_tickers}tk, {mp}pos, {opt_label})"

    max_opt = 0
    if use_opt:
        max_opt = max(1, mp // 3) if mp >= 3 else 0

    if not quiet:
        print(f"\n{'='*70}")
        print(f"  {label} -- {months} MESES")
        print(f"  Macro filter: {CONFIG['macro_ticker']} > SMA{CONFIG['macro_sma_period']}")
        print(f"  Max posiciones: {mp} (max opciones: {max_opt})")
        if use_opt:
            print(f"  Opciones: ITM {CONFIG['option_itm_pct']*100:.0f}%, "
                  f"DTE {CONFIG['option_dte']}, exit DTE {CONFIG['option_close_dte']}, "
                  f"IVR<{CONFIG['option_max_ivr']}")
            eligible_str = ', '.join(t + '→' + v['proxy'] for t, v in OPTIONS_MAP.items())
            print(f"  Eligible: {eligible_str}")
        print(f"{'='*70}")

        # Señales por ticker
        print(f"\n  Señales LONG por ticker:")
        total_signals = 0
        for ticker in sorted(OUTSIDER_TICKERS.keys()):
            if ticker in signals_data:
                n = (signals_data[ticker]['signals'] == 1).sum()
                cat = OUTSIDER_TICKERS[ticker]['category']
                name = OUTSIDER_TICKERS[ticker]['name']
                opt_tag = f" [OPT→{OPTIONS_MAP[ticker]['proxy']}]" if ticker in OPTIONS_MAP and use_opt else ""
                total_signals += n
                if n > 0:
                    print(f"    {ticker:16} {name:<20} [{cat}]  {n} señales{opt_tag}")
        print(f"  Total señales: {total_signals}")

        if macro_bullish:
            bull_days = sum(1 for v in macro_bullish.values() if v)
            bear_days = sum(1 for v in macro_bullish.values() if not v)
            total_days = bull_days + bear_days
            print(f"\n  Macro ({CONFIG['macro_ticker']} > SMA{CONFIG['macro_sma_period']}): "
                  f"BULL {bull_days}d ({bull_days/total_days*100:.0f}%) / "
                  f"BEAR {bear_days}d ({bear_days/total_days*100:.0f}%)")

    # Timeline
    all_dates = sorted(set(
        d for t, sd in signals_data.items()
        for d in sd['df'].index.tolist()
    ))

    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}       # spot
    active_options = {}      # opciones
    all_trades = []          # spot cerrados
    all_option_trades = []   # opciones cerradas

    for current_date in all_dates:

        # 1. GESTIONAR OPCIONES ACTIVAS
        if use_opt:
            opts_to_close = []
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
                    opts_to_close.append(ticker)
                    tracker.update_equity(opt.pnl_euros, current_date)

            for ticker in opts_to_close:
                opt = active_options.pop(ticker)
                tracker.open_positions -= 1
                tracker.open_options -= 1
                all_option_trades.append(opt)
                if verbose and not quiet:
                    pnl_pct = (opt.pnl_euros / opt.position_euros * 100) if opt.position_euros else 0
                    pnl_sign = '+' if opt.pnl_euros >= 0 else ''
                    print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE OPT {ticker:12} [{opt.proxy_etf}] | "
                          f"{opt.exit_reason:<12} | P&L EUR {pnl_sign}{opt.pnl_euros:.0f} ({pnl_sign}{pnl_pct:.1f}%) | "
                          f"Pos: {tracker.open_positions}/{mp} (opt:{tracker.open_options}) | Eq: EUR {tracker.equity:,.0f}")

        # 2. GESTIONAR TRADES SPOT ACTIVOS
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
            if verbose and not quiet:
                pnl_sign = '+' if trade.pnl_euros >= 0 else ''
                pnl_pct = (trade.pnl_euros / trade.position_euros * 100) if trade.position_euros else 0
                print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE     {ticker:16} | "
                      f"{trade.exit_reason:<12} | P&L EUR {pnl_sign}{trade.pnl_euros:.0f} ({pnl_sign}{pnl_pct:.1f}%) | "
                      f"Pos: {tracker.open_positions}/{mp} | Eq: EUR {tracker.equity:,.0f}")

        # 3. BUSCAR NUEVAS SEÑALES
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

        if tracker.open_positions < mp and is_macro_ok:
            candidates = find_candidates(signals_data, active_trades, active_options,
                                         current_date, is_macro_ok)
            ranked = rank_candidates(candidates, signals_data)

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= mp:
                    break

                df = signals_data[ticker]['df']
                bar = df.iloc[idx]

                # --- Decidir: opcion o spot ---
                open_as_option = False
                current_ivr = None

                if (use_opt and ticker in OPTIONS_MAP
                        and tracker.open_options < max_opt):
                    hvol_series = df['HVOL']
                    current_ivr = iv_rank(hvol_series, idx, CONFIG.get('option_ivr_window', 252))
                    if current_ivr < CONFIG.get('option_max_ivr', 40):
                        open_as_option = True

                if open_as_option:
                    # === ABRIR OPCION CALL ===
                    stock_price = bar['Open']
                    strike = stock_price * (1 - CONFIG['option_itm_pct'])
                    actual_dte = monthly_expiration_dte(current_date, CONFIG['option_dte'])
                    T = actual_dte / 365.0

                    iv = df['HVOL'].iloc[idx]
                    if pd.isna(iv) or iv <= 0:
                        iv = 0.30
                    bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], iv)
                    option_price = bs['price']
                    option_price *= (1 + CONFIG['option_spread_pct'] / 100 / 2)  # spread entrada

                    size = tracker.get_option_size(option_price)
                    if size['premium'] < 50:
                        continue

                    proxy = OPTIONS_MAP[ticker]['proxy']
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
                        proxy_etf=proxy,
                    )
                    active_options[ticker] = opt
                    tracker.open_positions += 1
                    tracker.open_options += 1

                    if verbose and not quiet:
                        print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN  OPT {ticker:12} [{proxy}] | "
                              f"K=${strike:,.2f} IV={iv:.0%} IVR={current_ivr:.0f} "
                              f"{actual_dte}DTE Prem=${option_price:.2f} x{size['contracts']:.2f}c = EUR {size['premium']:.0f} | "
                              f"Score: {composite_score:.2f} | Pos: {tracker.open_positions}/{mp} (opt:{tracker.open_options})")
                else:
                    # === ABRIR SPOT ===
                    size_info = tracker.get_position_size(ticker, prev_atr, bar['Open'])
                    entry_price = bar['Open'] * (1 + CONFIG['slippage_pct'] / 100)
                    position_euros = size_info['notional']
                    position_units = size_info['units']

                    max_per_position = tracker.equity / mp
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

                    if verbose and not quiet:
                        name = OUTSIDER_TICKERS.get(ticker, {}).get('name', '')
                        print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN      {ticker:16} | "
                              f"EUR {position_euros:.0f} @ ${entry_price:,.2f} | "
                              f"Score: {composite_score:.2f} | Pos: {tracker.open_positions}/{mp}")

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
    # METRICAS (spot + opciones combinados)
    # =================================================================
    all_combined = all_trades + all_option_trades
    if not all_combined:
        if not quiet:
            print("\n  SIN TRADES.")
        return None

    total_count = len(all_combined)
    winners = [t for t in all_combined if (hasattr(t, 'pnl_euros') and t.pnl_euros > 0)]
    losers = [t for t in all_combined if (hasattr(t, 'pnl_euros') and t.pnl_euros <= 0)]

    total_pnl = sum(t.pnl_euros for t in all_combined)
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
    opt_gt_50 = sum(1 for t in all_option_trades if t.pnl_pct >= 50)

    if not quiet:
        # Mejor/peor combinando spot + opciones
        best_combined = max(all_combined, key=lambda t: t.pnl_pct)
        worst_combined = min(all_combined, key=lambda t: t.pnl_pct)

        hold_days = []
        for t in all_combined:
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

  TRADES (spot + opciones):
     Total:          {total_count} (spot: {len(all_trades)}, opciones: {len(all_option_trades)})
     Ganadores:      {len(winners)} ({win_rate:.1f}%)
     Perdedores:     {len(losers)}
     Profit Factor:  {profit_factor:.2f}
     Avg Hold:       {avg_hold:.1f} dias
     Spot >= +3R:    {stock_gt_3r}
     Opt >= +50%:    {opt_gt_50}

  FAT TAILS:
     Avg Win:        {avg_win_pct:+.1f}%
     Avg Loss:       {avg_loss_pct:.1f}%
     Best:           {best_combined.ticker} {best_combined.pnl_pct:+.1f}%
     Worst:          {worst_combined.ticker} {worst_combined.pnl_pct:+.1f}%
""")

        # Razones de salida
        exit_reasons = {}
        for t in all_combined:
            reason = t.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        print("  RAZONES DE SALIDA:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pnl_reason = sum(t.pnl_euros for t in all_combined if t.exit_reason == reason)
            print(f"     {reason:20} {count:3} ({count/total_count*100:.1f}%)  P&L: EUR {pnl_reason:+,.0f}")

        # Detalle opciones
        if all_option_trades:
            opt_pnl = sum(o.pnl_euros for o in all_option_trades)
            opt_wins = sum(1 for o in all_option_trades if o.pnl_euros > 0)
            opt_wr = opt_wins / len(all_option_trades) * 100 if all_option_trades else 0
            print(f"\n  DETALLE OPCIONES ({len(all_option_trades)} trades, P&L EUR {opt_pnl:+,.0f}, Win% {opt_wr:.1f}%):")
            for i, opt in enumerate(sorted(all_option_trades, key=lambda x: x.entry_date), 1):
                entry_str = opt.entry_date.strftime('%Y-%m-%d')
                exit_str = opt.exit_date.strftime('%Y-%m-%d') if opt.exit_date else '?'
                marker = '+' if opt.pnl_euros > 0 else '-'
                print(f"     {i}. {entry_str} -> {exit_str} | {opt.ticker:12} [{opt.proxy_etf}] | "
                      f"K=${opt.strike:,.2f} | Prem ${opt.entry_option_price:.2f} -> ${opt.exit_option_price:.2f} | "
                      f"P&L EUR {opt.pnl_euros:+,.0f} ({opt.pnl_pct:+.1f}%) | {opt.bars_held}d | {opt.exit_reason} {marker}")

        # Detalle por ticker
        print(f"\n  DETALLE POR TICKER:")
        print(f"  {'Ticker':<16} {'Cat':<8} {'Tipo':<6} {'Trades':>6} {'Win%':>6} {'P&L EUR':>10} {'Best%':>8}")
        print(f"  {'-'*70}")
        ticker_summary = []
        for ticker in sorted(OUTSIDER_TICKERS.keys()):
            # Spot trades
            spot_t = [t for t in all_trades if t.ticker == ticker]
            opt_t = [t for t in all_option_trades if t.ticker == ticker]
            combined_t = spot_t + opt_t
            if not combined_t:
                continue
            c_wins = sum(1 for t in combined_t if t.pnl_euros > 0)
            c_pnl = sum(t.pnl_euros for t in combined_t)
            c_wr = c_wins / len(combined_t) * 100
            c_best = max(t.pnl_pct for t in combined_t)
            cat = OUTSIDER_TICKERS[ticker]['category']
            tipo = 'OPT' if opt_t and not spot_t else 'SPOT' if spot_t and not opt_t else 'MIX'
            ticker_summary.append((ticker, cat, tipo, len(combined_t), c_wr, c_pnl, c_best))

        for t, cat, tipo, n, wr, pnl, best in sorted(ticker_summary, key=lambda x: -x[5]):
            print(f"  {t:<16} {cat:<8} {tipo:<6} {n:>6} {wr:>5.1f}% {pnl:>+10,.0f} {best:>+7.1f}%")

        # Desglose por categoria
        for cat in ['CRYPTO', 'FUTURES']:
            cat_combined = [t for t in all_combined
                           if OUTSIDER_TICKERS.get(t.ticker, {}).get('category') == cat]
            if cat_combined:
                cat_pnl = sum(t.pnl_euros for t in cat_combined)
                cat_wins = sum(1 for t in cat_combined if t.pnl_euros > 0)
                cat_wr = cat_wins / len(cat_combined) * 100
                cat_spot = sum(1 for t in cat_combined if isinstance(t, Trade))
                cat_opt = sum(1 for t in cat_combined if isinstance(t, OptionTradeV2))
                print(f"\n  {cat}: {len(cat_combined)} trades (spot:{cat_spot}, opt:{cat_opt}), "
                      f"P&L EUR {cat_pnl:+,.0f}, Win% {cat_wr:.1f}%")

        # Lista de trades completa
        if verbose or total_count <= 40:
            print(f"\n  LISTA DE TRADES ({total_count}):")
            print(f"  {'#':>3} {'Entry':>12} {'Exit':>12} {'Ticker':<16} {'Tipo':<5} "
                  f"{'P&L EUR':>9} {'P&L%':>7} {'Days':>5} {'Razon':<15}")
            print(f"  {'-'*100}")
            all_sorted = sorted(all_combined, key=lambda x: x.entry_date)
            for i, t in enumerate(all_sorted, 1):
                entry_str = t.entry_date.strftime('%Y-%m-%d')
                exit_str = t.exit_date.strftime('%Y-%m-%d') if t.exit_date else '?'
                days = (t.exit_date - t.entry_date).days if t.exit_date else 0
                tipo = 'OPT' if isinstance(t, OptionTradeV2) else 'SPOT'
                print(f"  {i:>3} {entry_str:>12} {exit_str:>12} {t.ticker:<16} {tipo:<5} "
                      f"{t.pnl_euros:>+9,.0f} {t.pnl_pct:>+6.1f}% {days:>5} {t.exit_reason:<15}")

    return {
        'label': label,
        'max_positions': mp,
        'total_trades': total_count,
        'spot_trades': len(all_trades),
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
        'stock_gt_3r': stock_gt_3r,
        'opt_gt_50': opt_gt_50,
        'final_equity': tracker.equity,
    }


# =============================================================================
# GRID TEST POSICIONES (con y sin opciones)
# =============================================================================

def run_grid(months, all_data, signals_data, macro_bullish, verbose=False, use_options=True):
    positions_to_test = [3, 5, 7, 10, 12, 15]
    results_opt = []
    results_spot = []

    opt_label = "CON OPCIONES" if use_options else "SOLO SPOT"
    print(f"\n{'='*100}")
    print(f"  GRID TEST MAX_POSITIONS -- {months} MESES -- {len(OUTSIDER_TICKERS)} tickers -- {opt_label}")
    print(f"{'='*100}")

    if use_options:
        print(f"\n  --- CON OPCIONES (max_opt = pos//3) ---")
    for pos in positions_to_test:
        max_opt = max(1, pos // 3) if pos >= 3 else 0
        r = run_backtest(months, all_data, signals_data, macro_bullish,
                         max_positions=pos, verbose=False, quiet=True,
                         use_options=use_options)
        if r:
            results_opt.append(r)
            print(f"  pos={pos:>2} (opt:{max_opt}): Return {r['total_return_pct']:>+8.1f}% | "
                  f"Annual {r['annualized_return_pct']:>+7.1f}% | "
                  f"PF {r['profit_factor']:>5.2f} | "
                  f"MaxDD -{r['max_drawdown']:>5.1f}% | "
                  f"Win% {r['win_rate']:>5.1f}% | "
                  f"Trades {r['total_trades']:>3} (opt:{r['option_trades']:>2}) | "
                  f">3R {r['stock_gt_3r']:>2}")

    if not use_options:
        # Tambien correr baseline sin opciones para comparar
        print(f"\n  --- BASELINE SIN OPCIONES ---")
        for pos in positions_to_test:
            r = run_backtest(months, all_data, signals_data, macro_bullish,
                             max_positions=pos, verbose=False, quiet=True,
                             use_options=False)
            if r:
                results_spot.append(r)
                print(f"  pos={pos:>2} (spot): Return {r['total_return_pct']:>+8.1f}% | "
                      f"Annual {r['annualized_return_pct']:>+7.1f}% | "
                      f"PF {r['profit_factor']:>5.2f} | "
                      f"MaxDD -{r['max_drawdown']:>5.1f}% | "
                      f"Win% {r['win_rate']:>5.1f}% | "
                      f"Trades {r['total_trades']:>3}")

    results = results_opt
    if results:
        print(f"\n{'='*100}")
        print(f"  TABLA COMPARATIVA GRID — {opt_label}")
        print(f"{'='*100}")
        print(f"\n  {'Pos':>4} {'MaxOpt':>7} {'Return%':>9} {'Annual%':>9} {'PF':>6} {'MaxDD%':>8} {'Win%':>7} "
              f"{'Total':>6} {'Spot':>5} {'Opt':>4} {'>3R':>4} {'Eficiencia':>11}")
        print(f"  {'-'*100}")

        for r in results:
            eff = r['annualized_return_pct'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0
            mo = max(1, r['max_positions'] // 3) if r['max_positions'] >= 3 else 0
            print(f"  {r['max_positions']:>4} {mo:>7} {r['total_return_pct']:>+8.1f}% "
                  f"{r['annualized_return_pct']:>+8.1f}% {r['profit_factor']:>6.2f} "
                  f"-{r['max_drawdown']:>6.1f}% {r['win_rate']:>6.1f}% "
                  f"{r['total_trades']:>6} {r['spot_trades']:>5} {r['option_trades']:>4} "
                  f"{r['stock_gt_3r']:>4} {eff:>10.2f}")

        best_pf = max(results, key=lambda x: x['profit_factor'])
        best_ret = max(results, key=lambda x: x['total_return_pct'])
        best_eff = max(results, key=lambda x: x['annualized_return_pct'] / x['max_drawdown'] if x['max_drawdown'] > 0 else 0)
        print(f"\n  Mejor PF:          {best_pf['max_positions']} posiciones (PF {best_pf['profit_factor']:.2f})")
        print(f"  Mejor Return:      {best_ret['max_positions']} posiciones ({best_ret['total_return_pct']:+.1f}%)")
        print(f"  Mejor Eficiencia:  {best_eff['max_positions']} posiciones")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Backtest Outsiders: Momentum Breakout v8 para Crypto + Futuros + Opciones')
    parser.add_argument('--months', type=int, default=36, help='Meses de historico (default: 36)')
    parser.add_argument('--verbose', action='store_true', help='Detalle de trades')
    parser.add_argument('--grid', action='store_true', help='Grid test de max_positions')
    parser.add_argument('--pos', type=int, default=None, help='Override max posiciones')
    parser.add_argument('--no-options', action='store_true', help='Desactivar opciones (baseline spot-only)')
    args = parser.parse_args()

    use_options = not args.no_options

    opt_str = "CON OPCIONES (6 ETF proxy)" if use_options else "SOLO SPOT"
    print(f"""
{'='*70}
  BACKTEST OUTSIDERS — MOMENTUM BREAKOUT v8
  Universo: 20 Crypto + 7 Commodity Futures = 27 tickers
  Activos NO incluidos en v8/v12 (225 tickers)
  Macro filter: SPY > SMA50
  Config señales: identica a v8
  Modo: {opt_str}
{'='*70}
    """)

    # Descargar datos (una sola vez, cacheados para grid)
    print("  Descargando datos...")
    all_data, failed = download_all(TICKER_LIST, args.months)
    n_ok = len([t for t in OUTSIDER_TICKERS if t in all_data])
    print(f"  Tickers con datos: {n_ok}/{len(OUTSIDER_TICKERS)}")
    if failed:
        print(f"  Fallidos: {', '.join(failed)}")

    if not all_data:
        print("  ERROR: No hay datos.")
        return

    # Engine + señales (una sola vez)
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

    if args.grid:
        run_grid(args.months, all_data, signals_data, macro_bullish,
                 args.verbose, use_options=use_options)
    else:
        pos = args.pos if args.pos else 5
        run_backtest(args.months, all_data, signals_data, macro_bullish,
                     max_positions=pos, verbose=args.verbose,
                     use_options=use_options)


if __name__ == '__main__':
    main()
