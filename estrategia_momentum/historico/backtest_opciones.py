#!/usr/bin/env python3
"""
BACKTEST OPCIONES - Momentum Breakout con CALL Options

Simula la compra de CALL options en vez de acciones para los tickers US
con opciones liquidas. Usa Black-Scholes para estimar precios de entrada
y salida de las opciones.

Logica:
  - Misma generacion de senales que backtest_definitivo.py v5
  - Misma gestion (trailing, time exit, filtro macro)
  - Pero en vez de comprar acciones, compramos CALL ligeramente OTM a ~120 DTE
  - 120 DTE para minimizar theta decay durante el holding (~25d para ganadores)
  - La perdida maxima es la prima pagada (no necesitamos emergency stop)
  - Los ganadores se amplifican exponencialmente (gamma positivo)

Tickers elegibles para opciones: US stocks + ETFs con alta liquidez
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code')
from momentum_breakout import (
    MomentumEngine, calculate_atr, ASSETS, TICKERS
)


# =============================================================================
# BLACK-SCHOLES
# =============================================================================

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> Dict:
    """
    Black-Scholes para CALL option.

    Args:
        S: Precio del subyacente
        K: Strike price
        T: Tiempo a vencimiento en anos (30 DTE = 30/365)
        r: Tasa libre de riesgo anual
        sigma: Volatilidad anualizada

    Returns:
        dict con price, delta, gamma, theta, vega
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'price': max(S - K, 0), 'delta': 1.0 if S > K else 0,
                'gamma': 0, 'theta': 0, 'vega': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        'price': max(price, 0.01),
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
    }


def historical_volatility(close_prices: pd.Series, window: int = 30) -> pd.Series:
    """Volatilidad historica anualizada rolling."""
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


# =============================================================================
# CONFIG
# =============================================================================

# Tickers US con opciones liquidas (spread bajo)
OPTIONS_ELIGIBLE = [
    # Tech mega-cap (spreads <$0.05)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
    'ORCL', 'CRM', 'ADBE', 'AMD', 'INTC', 'CSCO', 'QCOM',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'AXP',
    # Health
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT',
    # Consumer
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT',
    # ETFs (muy liquidos)
    'QQQ', 'SPY', 'IWM', 'DIA', 'GLD', 'SLV', 'XLE', 'TLT',
    # ETFs apalancados (spreads mas amplios pero disponibles)
    'TQQQ', 'SPXL', 'TNA', 'BITO',
]

CONFIG = {
    'initial_capital': 10000,
    'target_risk_per_trade_pct': 2.0,
    'max_positions': 5,

    # Senales
    'ker_threshold': 0.40,
    'volume_threshold': 1.3,
    'rsi_threshold': 50,
    'rsi_max': 75,
    'breakout_period': 20,
    'longs_only': True,

    # Opciones
    'option_dte': 120,              # 120 DTE — theta decay minimo durante holding
    'option_otm_pct': 0.02,         # 2% OTM (strike = precio * 1.02)
    'option_contracts_sizing': True, # Usar misma logica de riesgo
    'risk_free_rate': 0.043,        # ~4.3% (Treasury 10Y actual)
    'hvol_window': 30,              # Ventana para volatilidad historica

    # Stop management (adaptado a opciones — mas amplio que acciones)
    'trail_trigger_r': 2.0,         # Activar trailing a +2R del valor de la opcion
    'trail_pct_of_max': 0.30,       # Trailing: cerrar si opcion cae al 30% de su max
    'max_hold_bars': 20,            # 20 dias (mas que acciones: la opcion a 120 DTE lo permite)
    'time_exit_loss_threshold': -0.40,  # Solo time exit si opcion pierde >40%
    'emergency_stop_pct': 0.15,

    # Filtro macro
    'use_macro_filter': True,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,

    # Costes
    'slippage_pct': 0.10,
    'option_spread_pct': 3.0,       # ~3% de spread en opciones (conservador)

    # Tickers: solo los elegibles para opciones
    'test_tickers': OPTIONS_ELIGIBLE,
}


# =============================================================================
# OPTION TRADE CLASS
# =============================================================================

@dataclass
class OptionTrade:
    """
    Simula compra de CALL option usando Black-Scholes.

    La opcion se compra al entrar y se valora cada dia con BS.
    El trailing stop se aplica sobre el VALOR DE LA OPCION, no sobre el subyacente.
    La perdida maxima es la prima pagada.
    """
    ticker: str
    entry_date: datetime
    entry_stock_price: float
    entry_atr: float
    strike: float
    dte_at_entry: int
    entry_option_price: float
    entry_iv: float
    num_contracts: float          # fraccionario para simplificar
    position_euros: float         # = num_contracts * entry_option_price * 100

    # Tracking
    R: float = field(init=False)
    bars_held: int = field(default=0)
    max_option_value: float = field(init=False)
    trailing_stop: Optional[float] = field(default=None)
    trailing_active: bool = field(default=False)
    max_r_mult: float = field(default=0.0)

    # Exit
    exit_date: Optional[datetime] = field(default=None)
    exit_option_price: float = field(default=0.0)
    exit_stock_price: float = field(default=0.0)
    exit_reason: Optional[str] = field(default=None)
    pnl_euros: float = field(default=0.0)
    pnl_pct: float = field(default=0.0)

    def __post_init__(self):
        self.R = self.entry_option_price * 0.5  # R = 50% de la prima
        self.max_option_value = self.entry_option_price

    def update(self, stock_price: float, stock_high: float, current_atr: float,
               current_iv: float, days_elapsed: int) -> Optional[dict]:
        """Valorar la opcion y gestionar trailing."""
        self.bars_held += 1

        # Calcular precio actual de la opcion via BS
        remaining_dte = max(self.dte_at_entry - days_elapsed, 0)
        T = remaining_dte / 365.0

        bs = black_scholes_call(
            S=stock_price,
            K=self.strike,
            T=T,
            r=CONFIG['risk_free_rate'],
            sigma=current_iv
        )
        current_option_price = bs['price']

        # Ajustar por spread
        current_option_price *= (1 - CONFIG['option_spread_pct'] / 100 / 2)

        self.max_option_value = max(self.max_option_value, current_option_price)

        r_mult = (current_option_price - self.entry_option_price) / self.R if self.R > 0 else 0
        self.max_r_mult = max(self.max_r_mult, r_mult)

        # 0. OPCION EXPIRA SIN VALOR (DTE = 0)
        if remaining_dte <= 0:
            intrinsic = max(stock_price - self.strike, 0)
            self._close(intrinsic, stock_price, 'expiration')
            return {'type': 'full_exit', 'reason': 'expiration'}

        # 1. TRAILING STOP CHECK (sobre el valor de la opcion)
        if self.trailing_active and self.trailing_stop is not None:
            if current_option_price <= self.trailing_stop:
                self._close(current_option_price, stock_price, 'trailing_stop')
                return {'type': 'full_exit', 'reason': 'trailing_stop'}

        # 2. ACTUALIZAR TRAILING
        # Trailing = X% del max valor opcion (mas amplio que acciones)
        trail_pct = CONFIG.get('trail_pct_of_max', 0.30)
        if r_mult >= CONFIG['trail_trigger_r']:
            trail_level = self.max_option_value * trail_pct
            if not self.trailing_active:
                self.trailing_active = True
                self.trailing_stop = trail_level
            elif trail_level > self.trailing_stop:
                self.trailing_stop = trail_level

        # 3. TIME EXIT — solo si opcion pierde mas del umbral (ej. -40%)
        loss_threshold = CONFIG.get('time_exit_loss_threshold', -0.40)
        option_return = (current_option_price / self.entry_option_price) - 1
        if self.bars_held >= CONFIG['max_hold_bars']:
            if option_return <= loss_threshold:
                # Opcion perdiendo mucho: cortar antes de destruccion total
                self._close(current_option_price, stock_price, 'time_exit')
                return {'type': 'full_exit', 'reason': 'time_exit'}
            elif option_return > 0 and not self.trailing_active:
                # En positivo pero sin trailing: activar con suelo en breakeven
                self.trailing_active = True
                self.trailing_stop = max(
                    self.max_option_value * trail_pct,
                    self.entry_option_price * 1.10
                )

        return None

    def _close(self, exit_option_price: float, exit_stock_price: float, reason: str):
        self.exit_option_price = exit_option_price
        self.exit_stock_price = exit_stock_price
        self.exit_reason = reason
        self.pnl_euros = (exit_option_price - self.entry_option_price) * self.num_contracts * 100
        self.pnl_pct = ((exit_option_price / self.entry_option_price) - 1) * 100 if self.entry_option_price > 0 else 0


# =============================================================================
# EQUITY TRACKER (reutilizado)
# =============================================================================

class EquityTracker:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.max_equity = initial_capital
        self.open_positions = 0

    def get_option_size(self, option_price: float, stock_price: float) -> Dict:
        """
        Cuanto invertir en opciones.

        Con opciones, la prima ES la perdida maxima.
        Win rate ~28%, avg loss ~44% de la prima → perdida esperada ~0.72 * 0.44 = 31.7% de la prima
        Para que la perdida esperada sea similar a acciones (2% del equity * 3.2% = 0.064%),
        usamos 1% del equity como prima maxima.
        """
        risk_pct = CONFIG['target_risk_per_trade_pct'] / 100
        # Prima maxima = 1% del equity (la mitad que en acciones)
        # porque la perdida media en opciones es mucho mayor
        max_premium = self.equity * risk_pct * 0.5
        if option_price <= 0:
            return {'contracts': 0, 'premium': 0}
        contracts = max_premium / (option_price * 100)
        premium = contracts * option_price * 100

        # Cap: no mas del 3% del equity por posicion
        max_per_pos = self.equity * 0.03
        if premium > max_per_pos:
            premium = max_per_pos
            contracts = premium / (option_price * 100)

        return {'contracts': contracts, 'premium': premium}

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
        df = yf.download(ticker, period=f'{min(months, 24)}mo', interval='1d', progress=False)
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

def run_backtest(months: int = 18, verbose: bool = True) -> Dict:
    n_tickers = len(CONFIG['test_tickers'])
    print(f"\n{'='*70}")
    print(f"  BACKTEST OPCIONES -- {months} MESES -- {n_tickers} tickers US")
    print(f"{'='*70}")
    print(f"  Capital: EUR {CONFIG['initial_capital']:,}")
    print(f"  Sizing: {CONFIG['target_risk_per_trade_pct']}% risk/trade (prima = max perdida)")
    print(f"  Opciones: CALL {CONFIG['option_otm_pct']*100:.0f}% OTM, {CONFIG['option_dte']}d DTE")
    print(f"  Trailing: 50% del max valor opcion a +2R")
    print(f"  Time exit: {CONFIG['max_hold_bars']}d solo perdedores")
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
            df['HVol'] = historical_volatility(df['Close'], CONFIG['hvol_window'])
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

    # Generar senales
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
            n_bull = sum(1 for v in macro_bullish.values() if v)
            n_bear = sum(1 for v in macro_bullish.values() if not v)
            print(f"  Filtro macro ({macro_ticker} > SMA{sma_period}): "
                  f"{n_bull} dias bull / {n_bear} dias bear\n")

    # Timeline
    all_dates = set()
    for sd in signals_data.values():
        all_dates.update(sd['df'].index.tolist())
    all_dates = sorted(all_dates)

    active_trades: Dict[str, OptionTrade] = {}
    all_trades: List[OptionTrade] = []

    # LOOP PRINCIPAL
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
            current_iv = df['HVol'].iloc[idx] if not pd.isna(df['HVol'].iloc[idx]) else 0.30

            # Dias transcurridos
            days_elapsed = (current_date - trade.entry_date).days

            result = trade.update(
                stock_price=bar['Close'],
                stock_high=bar['High'],
                current_atr=current_atr,
                current_iv=current_iv,
                days_elapsed=days_elapsed
            )

            if result and result['type'] == 'full_exit':
                trade.exit_date = current_date
                trades_to_close.append(ticker)
                tracker.update_equity(trade.pnl_euros, current_date)

        for ticker in trades_to_close:
            trade = active_trades.pop(ticker)
            tracker.open_positions -= 1
            all_trades.append(trade)

            if verbose and len(all_trades) <= 40:
                stock_move = ((trade.exit_stock_price / trade.entry_stock_price) - 1) * 100
                print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE {ticker:8} | "
                      f"{trade.exit_reason:15} | Stock {stock_move:+.1f}% | "
                      f"Option P&L: EUR {trade.pnl_euros:+.0f} ({trade.pnl_pct:+.1f}%) "
                      f"| {trade.bars_held}d")

        # 2. NUEVAS SENALES
        if CONFIG.get('use_macro_filter', False) and macro_bullish:
            is_macro_ok = macro_bullish.get(current_date, True)
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

            # Ranking multi-factor
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
                composite = (0.30 * ker_val + 0.20 * rsi_score + 0.20 * vol_score
                             + 0.15 * breakout_score + 0.15 * atr_score)
                ranked.append((ticker, idx, prev_atr, composite))

            ranked.sort(key=lambda x: x[3], reverse=True)

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= CONFIG['max_positions']:
                    break

                df = signals_data[ticker]['df']
                bar = df.iloc[idx]
                stock_price = bar['Open'] * (1 + CONFIG['slippage_pct'] / 100)

                # Calcular IV del dia anterior
                hvol = df['HVol'].iloc[idx - 1] if idx > 0 and not pd.isna(df['HVol'].iloc[idx - 1]) else 0.30

                # Strike: ATM o ligeramente OTM
                strike = stock_price * (1 + CONFIG['option_otm_pct'])

                # Precio opcion via Black-Scholes
                T = CONFIG['option_dte'] / 365.0
                bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], hvol)
                option_price = bs['price']

                # Aplicar spread
                option_price *= (1 + CONFIG['option_spread_pct'] / 100 / 2)

                if option_price < 0.10:
                    continue

                # Sizing
                size_info = tracker.get_option_size(option_price, stock_price)
                if size_info['premium'] < 50:
                    continue

                trade = OptionTrade(
                    ticker=ticker,
                    entry_date=current_date,
                    entry_stock_price=stock_price,
                    entry_atr=prev_atr,
                    strike=strike,
                    dte_at_entry=CONFIG['option_dte'],
                    entry_option_price=option_price,
                    entry_iv=hvol,
                    num_contracts=size_info['contracts'],
                    position_euros=size_info['premium'],
                )

                active_trades[ticker] = trade
                tracker.open_positions += 1

                if verbose and len(all_trades) + len(active_trades) <= 45:
                    print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN  {ticker:8} | "
                          f"Stock ${stock_price:.2f} | Strike ${strike:.2f} | "
                          f"CALL ${option_price:.2f} | IV {hvol*100:.0f}% | "
                          f"EUR {size_info['premium']:.0f} ({size_info['contracts']:.2f}c)")

    # Cerrar trades abiertos
    for ticker, trade in active_trades.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            last_price = df['Close'].iloc[-1]
            last_iv = df['HVol'].iloc[-1] if not pd.isna(df['HVol'].iloc[-1]) else 0.30
            days_elapsed = (df.index[-1] - trade.entry_date).days
            remaining_dte = max(trade.dte_at_entry - days_elapsed, 0)
            T = remaining_dte / 365.0

            bs = black_scholes_call(last_price, trade.strike, T, CONFIG['risk_free_rate'], last_iv)
            exit_price = bs['price'] * (1 - CONFIG['option_spread_pct'] / 100 / 2)

            trade._close(exit_price, last_price, 'end_of_data')
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

    total_return_pct = (tracker.equity / CONFIG['initial_capital'] - 1) * 100
    annualized = ((1 + total_return_pct / 100) ** (12 / months) - 1) * 100 if months > 0 else 0

    avg_win_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0
    avg_win_euros = np.mean([t.pnl_euros for t in winners]) if winners else 0
    avg_loss_euros = np.mean([t.pnl_euros for t in losers]) if losers else 0

    max_dd = tracker.get_max_drawdown()

    best_trade = max(all_trades, key=lambda t: t.pnl_pct)
    worst_trade = min(all_trades, key=lambda t: t.pnl_pct)

    # Trades con >100% return en la opcion
    home_runs = [t for t in all_trades if t.pnl_pct >= 100]

    print(f"""
{'='*70}
  RESULTADOS OPCIONES -- {months} MESES ({len(all_data)} tickers)
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

  OPCIONES:
     Home runs >100%: {len(home_runs)} ({len(home_runs)/total_trades*100:.1f}%)
     Best option:    {best_trade.ticker} {best_trade.pnl_pct:+.1f}% (EUR {best_trade.pnl_euros:+.0f})
     Worst option:   {worst_trade.ticker} {worst_trade.pnl_pct:+.1f}% (EUR {worst_trade.pnl_euros:+.0f})

  PROMEDIOS:
     Avg Win:        EUR {avg_win_euros:+.0f} ({avg_win_pct:+.1f}%)
     Avg Loss:       EUR {avg_loss_euros:.0f} ({avg_loss_pct:.1f}%)
     Avg Trade:      EUR {total_pnl/total_trades:+.0f}
""")

    exit_reasons = {}
    for t in all_trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    print("  RAZONES DE SALIDA:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"     {reason:20} {count:3} ({count/total_trades*100:.1f}%)")

    # Top trades
    print(f"\n  TOP 5 TRADES (por EUR):")
    sorted_trades = sorted(all_trades, key=lambda t: t.pnl_euros, reverse=True)
    for t in sorted_trades[:5]:
        stock_move = ((t.exit_stock_price / t.entry_stock_price) - 1) * 100 if t.exit_stock_price else 0
        print(f"     {t.ticker:8} | Stock {stock_move:+.1f}% → Option {t.pnl_pct:+.1f}% | "
              f"EUR {t.pnl_euros:+.0f} | {t.bars_held}d | IV {t.entry_iv*100:.0f}%")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl_euros': total_pnl,
        'total_return_pct': total_return_pct,
        'annualized_return_pct': annualized,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'home_runs': len(home_runs),
        'final_equity': tracker.equity,
        'trades': all_trades,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"""
======================================================================
  BACKTEST OPCIONES -- MOMENTUM BREAKOUT (Fat Tails + CALL Options)
======================================================================
  - {len(CONFIG['test_tickers'])} tickers US con opciones liquidas
  - CALL {CONFIG['option_otm_pct']*100:.0f}% OTM, {CONFIG['option_dte']}d DTE
  - Precio estimado via Black-Scholes + volatilidad historica
  - Trailing: 50% del max valor opcion
  - Perdida maxima = prima pagada (no necesita emergency stop)
======================================================================
    """)

    all_results = []
    for period in [6, 12, 18]:
        result = run_backtest(months=period, verbose=(period == 18))
        if 'error' not in result:
            all_results.append({'period': period, **result})

    if len(all_results) > 1:
        print(f"""
{'='*70}
  RESUMEN COMPARATIVO — OPCIONES
{'='*70}

  {'Periodo':<8} {'Trades':<8} {'Win%':<7} {'PnL EUR':<11} {'Return%':<9} {'Annual%':<9} {'MaxDD%':<7} {'PF':<6} {'HRuns':<6}
  {'-'*76}""")
        for r in all_results:
            print(f"  {r['period']}m{' '*5} {r['total_trades']:<8} {r['win_rate']:<7.1f} "
                  f"EUR{r['total_pnl_euros']:>+8,.0f}  {r['total_return_pct']:>+7.1f}%  "
                  f"{r['annualized_return_pct']:>+7.1f}%  "
                  f"{r['max_drawdown']:>5.1f}%  {r['profit_factor']:.2f}  {r['home_runs']}")
        print()


if __name__ == "__main__":
    main()
