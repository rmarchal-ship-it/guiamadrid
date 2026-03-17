#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           📊 BACKTEST REALISTA — MOMENTUM BREAKOUT                            ║
║                                                                               ║
║           Capital: €10,000                                                    ║
║           Períodos: 6, 12, 24 meses                                           ║
║           Position sizing dinámico basado en equity                           ║
║           P&L en euros (no solo %)                                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CARACTERÍSTICAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Entrada al OPEN de la barra siguiente a la señal
• Position sizing basado en equity ACTUAL (no fijo)
• Stops evaluados con High/Low intrabarra
• Curva de equity y drawdown
• Métricas: Sharpe, Sortino, Profit Factor, etc.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Import momentum strategy
import sys
sys.path.insert(0, '/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code')
from momentum_breakout import (
    MomentumEngine, DynamicStopManager, calculate_atr,
    calculate_position_size, ASSETS, DEFAULT_CONFIG
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Capital
    'initial_capital': 10000,      # €10,000
    'currency': 'EUR',

    # ═══════════════════════════════════════════════════════════════════════════
    # LONGS ONLY MODE (basado en backtest: LONGS +130.8% vs SHORTS -281.1%)
    # ═══════════════════════════════════════════════════════════════════════════
    'longs_only': DEFAULT_CONFIG.get('longs_only', True),

    # Parámetros de señal (optimizados)
    'ker_threshold': DEFAULT_CONFIG['ker_threshold'],
    'volume_threshold': DEFAULT_CONFIG['volume_threshold'],
    'rsi_threshold': DEFAULT_CONFIG['rsi_threshold'],
    'rsi_max': DEFAULT_CONFIG.get('rsi_max', 75),  # NUEVO: Evitar sobrecompra
    'breakout_period': DEFAULT_CONFIG['breakout_period'],

    # Stop/Target (optimizados)
    'initial_atr_mult': DEFAULT_CONFIG['initial_atr_mult'],
    'target_r_mult': DEFAULT_CONFIG['target_r_mult'],

    # Time management
    # ACTUALIZADO a 30 barras (TEST B ganador: +72.4% en 6m, +61.7% en 12m)
    'max_hold_bars': 30,  # Era 18, ahora 30 (~5 días en 4H)

    # Costes
    'slippage_pct': 0.10,          # 0.1%
    'commission_pct': 0.00,        # Sin comisión

    # Risk
    'max_positions': 5,
    'target_vol_annual': 0.40,      # CAMBIO: 0.20 → 0.40 (2x más exposición)

    # Períodos a testear
    'periods_months': [6, 12, 24],

    # Tickers a testear (subset líquido)
    'test_tickers': [
        # US Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        # ETFs
        'QQQ', 'SPY', 'IWM',
        # Commodities
        'GLD', 'SLV', 'USO',
        # Fixed Income (descorrelacionado)
        'TLT',
        # EU
        'SAP', 'ASML',
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIONES DE TEST
# ═══════════════════════════════════════════════════════════════════════════════
#
# RESULTADO FINAL (Feb 2026):
# ══════════════════════════════════════════════════════════════════════════════
# TEST B (40% vol, 30 barras) fue el GANADOR:
#   - 6 meses:  +72.4% return, PF 1.55, Max DD -22%
#   - 12 meses: +61.7% return, PF 1.42, Max DD -25%
#
# NOTA IMPORTANTE:
# Los backtests con datos diarios (24-36 meses) mostraron resultados pobres
# porque la estrategia está diseñada para 4H y pierde información intradiaria.
# Para validación a largo plazo, se necesitarían datos 4H de un proveedor
# como TradingView, Polygon, o Tastytrade API.
# ══════════════════════════════════════════════════════════════════════════════

TEST_CONFIGS = {
    # CONFIGURACIÓN RECOMENDADA (ganadora)
    'test_b_winner': {
        'target_vol_annual': 0.40,
        'max_hold_bars': 30,  # 5 días en 4H
    },
    # Alternativas para referencia
    'v2.0_baseline': {
        'target_vol_annual': 0.40,
        'max_hold_bars': 18,
    },
    'conservative': {
        'target_vol_annual': 0.30,
        'max_hold_bars': 30,
    },
}


def get_adjusted_params(interval: str, adjust_for_timeframe: bool = True) -> Dict:
    """
    Retorna parámetros ajustados según el intervalo de datos.

    Args:
        interval: '4h' o '1d'
        adjust_for_timeframe: Si True, ajusta proporcionalmente.
                              Si False, mantiene números originales.

    Returns:
        Dict con parámetros ajustados
    """
    base_params = {
        'max_hold_bars': CONFIG['max_hold_bars'],
        'breakout_period': CONFIG['breakout_period'],
        'ker_period': 10,
        'rsi_period': 14,
        'atr_period': 14,
    }

    if interval == '1d' and adjust_for_timeframe:
        # 1 día ≈ 1.5 barras 4H (6.5h mercado / 4h)
        # Ajustamos dividiendo por 1.5 y redondeando
        return {
            'max_hold_bars': max(8, int(base_params['max_hold_bars'] / 1.5)),
            'breakout_period': max(8, int(base_params['breakout_period'] / 1.5)),
            'ker_period': max(5, int(base_params['ker_period'] / 1.5)),
            'rsi_period': max(7, int(base_params['rsi_period'] / 1.5)),
            'atr_period': max(7, int(base_params['atr_period'] / 1.5)),
        }
    return base_params


# ═══════════════════════════════════════════════════════════════════════════════
# EQUITY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class EquityTracker:
    """Rastrea capital, posiciones abiertas y P&L."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = [(datetime.now(), initial_capital)]
        self.max_equity = initial_capital
        self.open_positions = 0

    def update_equity(self, pnl_euros: float, timestamp):
        """Actualiza equity tras cerrar un trade."""
        self.equity += pnl_euros
        self.equity_curve.append((timestamp, self.equity))
        self.max_equity = max(self.max_equity, self.equity)

    def get_current_drawdown(self) -> float:
        """Drawdown actual en %."""
        if self.max_equity <= 0:
            return 0
        return (self.max_equity - self.equity) / self.max_equity * 100

    def get_max_drawdown(self) -> float:
        """Máximo drawdown histórico en %."""
        if len(self.equity_curve) < 2:
            return 0
        equity_values = [e[1] for e in self.equity_curve]
        equity = np.array(equity_values)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / np.where(running_max > 0, running_max, 1) * 100
        return drawdown.max()

    def get_equity_series(self) -> pd.Series:
        """Retorna serie de equity con timestamps."""
        dates = [e[0] for e in self.equity_curve]
        values = [e[1] for e in self.equity_curve]
        return pd.Series(values, index=dates)


# ═══════════════════════════════════════════════════════════════════════════════
# REALISTIC TRADE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RealisticTrade:
    """
    Trade con P&L en euros, tracking completo y PARTIAL PROFIT TAKING.

    Scale-out strategy (del plan original):
    - 33% @ 2R (lock in profit)
    - 33% @ 4R (significant move)
    - 34% trailing (let it run for fat tails)
    """

    ticker: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_date: datetime
    entry_atr: float
    position_units: float
    position_notional: float

    # Calculados al crear
    R: float = field(init=False)
    initial_stop: float = field(init=False)
    target_2r: float = field(init=False)  # 33% exit
    target_4r: float = field(init=False)  # 33% exit
    # El 34% restante usa trailing stop

    # Estado durante trade
    current_stop: float = field(init=False)
    stop_phase: str = field(default='initial')
    highest_since: float = field(init=False)
    lowest_since: float = field(init=False)
    bars_held: int = field(default=0)

    # Partial exits tracking
    remaining_units: float = field(init=False)  # Units still in trade
    partial_exits: List = field(default_factory=list)  # List of partial exit dicts
    hit_2r: bool = field(default=False)
    hit_4r: bool = field(default=False)

    # Resultado final
    exit_price: Optional[float] = field(default=None)
    exit_date: Optional[datetime] = field(default=None)
    exit_reason: Optional[str] = field(default=None)
    pnl_euros: float = field(default=0.0)
    pnl_pct: float = field(default=0.0)
    r_multiple: float = field(default=0.0)

    def __post_init__(self):
        atr_mult = CONFIG['initial_atr_mult']

        self.R = self.entry_atr * atr_mult
        if self.direction == 'long':
            self.initial_stop = self.entry_price - self.R
            self.target_2r = self.entry_price + self.R * 2  # +2R
            self.target_4r = self.entry_price + self.R * 4  # +4R
        else:
            self.initial_stop = self.entry_price + self.R
            self.target_2r = self.entry_price - self.R * 2
            self.target_4r = self.entry_price - self.R * 4

        self.current_stop = self.initial_stop
        self.highest_since = self.entry_price
        self.lowest_since = self.entry_price
        self.remaining_units = self.position_units
        self.partial_exits = []

    def update_extremes(self, high: float, low: float):
        """Actualiza máximos/mínimos desde entrada."""
        self.highest_since = max(self.highest_since, high)
        self.lowest_since = min(self.lowest_since, low)
        self.bars_held += 1

    def check_stop_hit(self, low: float, high: float) -> bool:
        """Verifica si el stop fue tocado."""
        if self.direction == 'long':
            return low <= self.current_stop
        else:
            return high >= self.current_stop

    def check_partial_2r(self, low: float, high: float) -> bool:
        """Verifica si alcanzó +2R para primer partial exit."""
        if self.hit_2r:
            return False
        if self.direction == 'long':
            return high >= self.target_2r
        else:
            return low <= self.target_2r

    def check_partial_4r(self, low: float, high: float) -> bool:
        """Verifica si alcanzó +4R para segundo partial exit."""
        if self.hit_4r:
            return False
        if self.direction == 'long':
            return high >= self.target_4r
        else:
            return low <= self.target_4r

    def execute_partial_exit(self, exit_price: float, pct: float, reason: str, date) -> float:
        """
        Ejecuta un partial exit y retorna el P&L en euros.

        Args:
            exit_price: Precio de salida
            pct: Porcentaje de la posición a cerrar (0.33 = 33%)
            reason: Razón del exit
            date: Fecha del exit

        Returns:
            P&L en euros de esta porción
        """
        units_to_exit = self.position_units * pct
        if units_to_exit > self.remaining_units:
            units_to_exit = self.remaining_units

        if self.direction == 'long':
            partial_pnl = (exit_price - self.entry_price) * units_to_exit
        else:
            partial_pnl = (self.entry_price - exit_price) * units_to_exit

        self.remaining_units -= units_to_exit
        self.partial_exits.append({
            'date': date,
            'price': exit_price,
            'units': units_to_exit,
            'pnl_euros': partial_pnl,
            'reason': reason
        })

        return partial_pnl

    def calculate_final_pnl(self):
        """Calcula P&L total incluyendo todos los partial exits."""
        total_pnl = sum(pe['pnl_euros'] for pe in self.partial_exits)
        self.pnl_euros = total_pnl

        # Calcular % sobre notional original
        if self.position_notional > 0:
            self.pnl_pct = (total_pnl / self.position_notional) * 100

        # Calcular R-multiple promedio ponderado
        if self.R > 0 and self.position_units > 0:
            total_r = 0
            for pe in self.partial_exits:
                if self.direction == 'long':
                    r = (pe['price'] - self.entry_price) / self.R
                else:
                    r = (self.entry_price - pe['price']) / self.R
                total_r += r * pe['units']
            self.r_multiple = total_r / self.position_units


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_data(ticker: str, months: int, force_daily: bool = False) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Descarga datos históricos.

    Args:
        ticker: Símbolo del activo
        months: Meses de histórico
        force_daily: Forzar uso de datos diarios

    Returns:
        (DataFrame, interval_used) - interval_used es '4h' o '1d'

    Lógica:
    - <= 2 meses: 4H directo
    - 3-24 meses: 1H resampleado a 4H (límite yfinance ~730 días)
    - > 24 meses O force_daily: Daily (para backtests largos)
    """
    try:
        if force_daily or months > 24:
            # Forzar datos diarios para períodos largos
            period = f'{months}mo' if months <= 60 else 'max'
            interval = '1d'
        elif months <= 2:
            period = f'{months}mo'
            interval = '4h'
        else:
            period = f'{months}mo'
            interval = '1h'

        df = yf.download(ticker, period=period, interval=interval, progress=False)

        if df.empty:
            return None, interval

        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Resamplear 1H a 4H
        original_interval = interval
        if interval == '1h' and len(df) > 0:
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            interval = '4h'  # Actualizar después de resampleo

        return (df, interval) if len(df) >= 50 else (None, interval)

    except Exception as e:
        return None, 'error'


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_realistic_backtest(
    data: Dict[str, pd.DataFrame],
    initial_capital: float = 10000,
    verbose: bool = False
) -> Dict:
    """
    Backtest con capital real y position sizing dinámico.

    Returns:
        {
            'trades': list[RealisticTrade],
            'equity_tracker': EquityTracker,
            'metrics': dict
        }
    """
    tracker = EquityTracker(initial_capital)
    all_trades = []
    active_trades: Dict[str, RealisticTrade] = {}

    # Setup strategy
    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG.get('rsi_max', 75),  # NUEVO: Evitar sobrecompra
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG.get('longs_only', True),  # ⚠️ LONGS ONLY por defecto
    )
    stop_mgr = DynamicStopManager(initial_atr_mult=CONFIG['initial_atr_mult'])

    slippage = CONFIG['slippage_pct'] / 100
    max_hold = CONFIG['max_hold_bars']

    # Generar señales para todos los tickers
    signals_data = {}
    for ticker, df in data.items():
        df = df.copy()
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
        signals = engine.generate_signals(df)
        signals_data[ticker] = {
            'df': df,
            'signals': signals
        }

    # Crear timeline unificado
    all_dates = set()
    for ticker, sd in signals_data.items():
        all_dates.update(sd['df'].index.tolist())
    all_dates = sorted(all_dates)

    # Iterar barra por barra
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
            current_atr = df['ATR'].iloc[idx]

            # Actualizar extremos
            trade.update_extremes(bar['High'], bar['Low'])

            # Actualizar stop dinámico
            stop_info = stop_mgr.calculate_stop(
                position_type=trade.direction,
                entry_price=trade.entry_price,
                current_price=bar['Close'],
                current_atr=current_atr,
                highest_since_entry=trade.highest_since,
                lowest_since_entry=trade.lowest_since,
                entry_atr=trade.entry_atr
            )

            # Solo mover stop si mejora
            if trade.direction == 'long':
                if stop_info['stop_price'] > trade.current_stop:
                    trade.current_stop = stop_info['stop_price']
                    trade.stop_phase = stop_info['phase']
            else:
                if stop_info['stop_price'] < trade.current_stop:
                    trade.current_stop = stop_info['stop_price']
                    trade.stop_phase = stop_info['phase']

            # ═══════════════════════════════════════════════════════════════
            # PARTIAL PROFIT TAKING (Scale-out strategy)
            # 33% @ 2R, 33% @ 4R, 34% trailing
            # ═══════════════════════════════════════════════════════════════

            # Verificar partial exit @ 2R
            if trade.check_partial_2r(bar['Low'], bar['High']) and trade.remaining_units > 0:
                exit_price = trade.target_2r * (1 - slippage if trade.direction == 'long' else 1 + slippage)
                partial_pnl = trade.execute_partial_exit(exit_price, 0.33, 'partial_2R', current_date)
                tracker.update_equity(partial_pnl, current_date)
                trade.hit_2r = True
                if verbose:
                    print(f"  {current_date.strftime('%Y-%m-%d')} | PARTIAL {ticker} @ 2R | "
                          f"33% exit | P&L: €{partial_pnl:+.2f}")

            # Verificar partial exit @ 4R
            if trade.check_partial_4r(bar['Low'], bar['High']) and trade.remaining_units > 0:
                exit_price = trade.target_4r * (1 - slippage if trade.direction == 'long' else 1 + slippage)
                partial_pnl = trade.execute_partial_exit(exit_price, 0.33, 'partial_4R', current_date)
                tracker.update_equity(partial_pnl, current_date)
                trade.hit_4r = True
                if verbose:
                    print(f"  {current_date.strftime('%Y-%m-%d')} | PARTIAL {ticker} @ 4R | "
                          f"33% exit | P&L: €{partial_pnl:+.2f}")

            # Verificar stop hit (para remaining units)
            if trade.check_stop_hit(bar['Low'], bar['High']) and trade.remaining_units > 0:
                exit_price = trade.current_stop * (1 - slippage if trade.direction == 'long' else 1 + slippage)
                # Cerrar remaining units
                remaining_pct = trade.remaining_units / trade.position_units
                partial_pnl = trade.execute_partial_exit(exit_price, remaining_pct, f'stop_{trade.stop_phase}', current_date)
                tracker.update_equity(partial_pnl, current_date)

                trade.exit_price = exit_price
                trade.exit_date = current_date
                trade.exit_reason = f'stop_{trade.stop_phase}'
                trade.calculate_final_pnl()
                trades_to_close.append(ticker)
                continue

            # Verificar time exit (para remaining units)
            if trade.bars_held >= max_hold and trade.remaining_units > 0:
                exit_price = bar['Close'] * (1 - slippage if trade.direction == 'long' else 1 + slippage)
                remaining_pct = trade.remaining_units / trade.position_units
                partial_pnl = trade.execute_partial_exit(exit_price, remaining_pct, 'time_exit', current_date)
                tracker.update_equity(partial_pnl, current_date)

                trade.exit_price = exit_price
                trade.exit_date = current_date
                trade.exit_reason = 'time_exit'
                trade.calculate_final_pnl()
                trades_to_close.append(ticker)

        # Cerrar trades completados y actualizar
        for ticker in trades_to_close:
            trade = active_trades.pop(ticker)
            tracker.open_positions -= 1
            all_trades.append(trade)

            if verbose:
                partials_str = f" ({len(trade.partial_exits)} partials)" if trade.partial_exits else ""
                print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE {ticker} | "
                      f"{trade.exit_reason}{partials_str} | P&L: €{trade.pnl_euros:+.2f} ({trade.pnl_pct:+.2f}%)")

        # 2. Buscar nuevas señales
        for ticker, sd in signals_data.items():
            if ticker in active_trades:
                continue  # Ya tenemos posición

            if tracker.open_positions >= CONFIG['max_positions']:
                continue  # Max posiciones alcanzado

            df = sd['df']
            signals = sd['signals']

            if current_date not in df.index:
                continue

            idx = df.index.get_loc(current_date)
            if idx < 1:
                continue

            # Señal de barra anterior, entrada en esta barra
            prev_signal = signals.iloc[idx - 1]
            if prev_signal == 0:
                continue

            bar = df.iloc[idx]
            prev_atr = df['ATR'].iloc[idx - 1]

            if prev_atr <= 0 or np.isnan(prev_atr):
                continue

            direction = 'long' if prev_signal == 1 else 'short'
            entry_price = bar['Open'] * (1 + slippage if direction == 'long' else 1 - slippage)

            # Position sizing basado en equity actual
            asset_info = ASSETS.get(ticker, {})
            is_crypto = asset_info.get('is_crypto', False)

            size_info = calculate_position_size(
                account_balance=tracker.equity,
                current_atr=prev_atr,
                price=entry_price,
                target_vol_annual=CONFIG['target_vol_annual'],
                is_crypto=is_crypto
            )

            # Crear trade
            trade = RealisticTrade(
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                entry_date=current_date,
                entry_atr=prev_atr,
                position_units=size_info['units'],
                position_notional=size_info['notional']
            )

            active_trades[ticker] = trade
            tracker.open_positions += 1

            if verbose:
                print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN {direction.upper()} {ticker} | "
                      f"Entry: {entry_price:.2f} | Units: {size_info['units']:.4f} | "
                      f"Stop: {trade.initial_stop:.2f}")

    # Cerrar trades abiertos al final
    for ticker, trade in active_trades.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            exit_price = df['Close'].iloc[-1]

            # Cerrar remaining units si quedan
            if trade.remaining_units > 0:
                remaining_pct = trade.remaining_units / trade.position_units
                partial_pnl = trade.execute_partial_exit(exit_price, remaining_pct, 'end_of_data', df.index[-1])
                tracker.update_equity(partial_pnl, df.index[-1])

            trade.exit_price = exit_price
            trade.exit_date = df.index[-1]
            trade.exit_reason = 'end_of_data'
            trade.calculate_final_pnl()
            all_trades.append(trade)

    # Calcular métricas
    metrics = calculate_metrics(all_trades, tracker, initial_capital)

    return {
        'trades': all_trades,
        'equity_tracker': tracker,
        'metrics': metrics
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(trades: List[RealisticTrade], tracker: EquityTracker,
                      initial_capital: float) -> Dict:
    """Calcula métricas de performance."""
    if not trades:
        return {
            'total_trades': 0,
            'total_pnl_euros': 0,
            'total_return_pct': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_trade_euros': 0,
            'avg_r_multiple': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_consec_losses': 0,
        }

    total_trades = len(trades)
    winners = [t for t in trades if t.pnl_euros > 0]
    losers = [t for t in trades if t.pnl_euros <= 0]

    total_pnl = sum(t.pnl_euros for t in trades)
    final_equity = tracker.equity

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    avg_trade = total_pnl / total_trades if total_trades > 0 else 0
    avg_r = np.mean([t.r_multiple for t in trades]) if trades else 0

    # Max consecutive losses
    max_consec = 0
    current_streak = 0
    for t in trades:
        if t.pnl_euros <= 0:
            current_streak += 1
            max_consec = max(max_consec, current_streak)
        else:
            current_streak = 0

    # Sharpe y Sortino
    returns = [t.pnl_pct for t in trades]
    if len(returns) > 1:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret * np.sqrt(252 / 6) if std_ret > 0 else 0  # 6 trades/day approx

        negative_returns = [r for r in returns if r < 0]
        std_neg = np.std(negative_returns) if len(negative_returns) > 1 else std_ret
        sortino = mean_ret / std_neg * np.sqrt(252 / 6) if std_neg > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # Exit reasons
    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Avg holding time
    avg_bars = np.mean([t.bars_held for t in trades]) if trades else 0

    # Partial exit stats
    trades_with_2r = sum(1 for t in trades if t.hit_2r)
    trades_with_4r = sum(1 for t in trades if t.hit_4r)
    total_partial_exits = sum(len(t.partial_exits) for t in trades)

    return {
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'total_pnl_euros': total_pnl,
        'total_return_pct': (final_equity - initial_capital) / initial_capital * 100,
        'final_equity': final_equity,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade_euros': avg_trade,
        'avg_win_euros': np.mean([t.pnl_euros for t in winners]) if winners else 0,
        'avg_loss_euros': np.mean([t.pnl_euros for t in losers]) if losers else 0,
        'avg_r_multiple': avg_r,
        'max_drawdown_pct': tracker.get_max_drawdown(),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_consec_losses': max_consec,
        'avg_bars_held': avg_bars,
        'exit_reasons': exit_reasons,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        # Partial exit metrics
        'trades_hit_2r': trades_with_2r,
        'trades_hit_4r': trades_with_4r,
        'pct_hit_2r': trades_with_2r / total_trades * 100 if total_trades > 0 else 0,
        'pct_hit_4r': trades_with_4r / total_trades * 100 if total_trades > 0 else 0,
        'total_partial_exits': total_partial_exits,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(metrics: Dict, period_name: str):
    """Imprime resultados de forma legible."""
    if metrics['total_trades'] == 0:
        print(f"\n  {period_name}: Sin trades")
        return

    print(f"""
{'─'*70}
  📊 RESULTADOS: {period_name}
{'─'*70}

  💰 P&L:
     Capital inicial:   €{CONFIG['initial_capital']:,.2f}
     Capital final:     €{metrics['final_equity']:,.2f}
     P&L Total:         €{metrics['total_pnl_euros']:+,.2f} ({metrics['total_return_pct']:+.1f}%)

  📈 MÉTRICAS:
     Total trades:      {metrics['total_trades']}
     Win Rate:          {metrics['win_rate']:.1f}%
     Profit Factor:     {metrics['profit_factor']:.2f}
     Avg Trade:         €{metrics['avg_trade_euros']:+.2f}
     Avg R-Multiple:    {metrics['avg_r_multiple']:+.2f}R

  💵 DETALLE:
     Avg Win:           €{metrics['avg_win_euros']:+.2f}
     Avg Loss:          €{metrics['avg_loss_euros']:.2f}
     Gross Profit:      €{metrics['gross_profit']:,.2f}
     Gross Loss:        €{metrics['gross_loss']:,.2f}

  🎯 PARTIAL EXITS (Scale-out):
     Trades hit +2R:    {metrics.get('trades_hit_2r', 0)} ({metrics.get('pct_hit_2r', 0):.1f}%)
     Trades hit +4R:    {metrics.get('trades_hit_4r', 0)} ({metrics.get('pct_hit_4r', 0):.1f}%)
     Total partials:    {metrics.get('total_partial_exits', 0)}

  📉 RIESGO:
     Max Drawdown:      -{metrics['max_drawdown_pct']:.1f}% (€{CONFIG['initial_capital'] * metrics['max_drawdown_pct'] / 100:,.0f})
     Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}
     Sortino Ratio:     {metrics['sortino_ratio']:.2f}
     Max Consec Loss:   {metrics['max_consec_losses']}

  ⏱️ TIMING:
     Avg Hold:          {metrics['avg_bars_held']:.1f} barras (~{metrics['avg_bars_held']*4:.0f}h)

  🚪 EXIT REASONS:""")

    for reason, count in metrics['exit_reasons'].items():
        pct = count / metrics['total_trades'] * 100
        print(f"     {reason:20} {count:3} ({pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def download_all_tickers(tickers: List[str], months: int, force_daily: bool = False) -> Tuple[Dict, str]:
    """
    Descarga datos para múltiples tickers.

    Returns:
        (data_dict, interval_used)
    """
    data = {}
    interval_used = None

    for ticker in tickers:
        df, interval = download_data(ticker, months, force_daily)
        if df is not None:
            data[ticker] = df
            interval_used = interval

    return data, interval_used or '4h'


def run_comparison_backtest(
    months: int,
    configs: Dict[str, Dict] = None,
    tickers: List[str] = None,
    force_daily: bool = False,
    adjust_for_timeframe: bool = True
) -> pd.DataFrame:
    """
    Ejecuta backtest con múltiples configuraciones y compara.

    Args:
        months: Meses de histórico
        configs: Dict de configuraciones a probar
        tickers: Lista de tickers (default: CONFIG['test_tickers'])
        force_daily: Forzar uso de datos diarios
        adjust_for_timeframe: Ajustar parámetros para datos diarios

    Returns:
        DataFrame con métricas por config
    """
    if configs is None:
        configs = TEST_CONFIGS
    if tickers is None:
        tickers = CONFIG['test_tickers']

    # Descargar datos una sola vez
    print(f"\n  📥 Descargando datos para {len(tickers)} tickers ({months} meses)...")
    data, interval = download_all_tickers(tickers, months, force_daily)
    print(f"     Intervalo: {interval} | Tickers con datos: {len(data)}")

    if not data:
        print("  ⚠️ No hay datos disponibles")
        return pd.DataFrame()

    # Obtener ajuste de parámetros si es necesario
    adjusted_params = get_adjusted_params(interval, adjust_for_timeframe)

    results = []
    for name, config in configs.items():
        # Aplicar config temporalmente
        original_config = {k: CONFIG[k] for k in config.keys() if k in CONFIG}
        CONFIG.update(config)

        # Ajustar parámetros si usamos datos diarios
        if interval == '1d' and adjust_for_timeframe:
            CONFIG['max_hold_bars'] = adjusted_params['max_hold_bars']
            CONFIG['breakout_period'] = adjusted_params['breakout_period']

        # Ejecutar backtest
        result = run_realistic_backtest(
            data=data,
            initial_capital=CONFIG['initial_capital'],
            verbose=False
        )

        results.append({
            'config': name,
            'return_pct': result['metrics']['total_return_pct'],
            'max_dd': result['metrics']['max_drawdown_pct'],
            'pf': result['metrics']['profit_factor'],
            'sharpe': result['metrics']['sharpe_ratio'],
            'trades': result['metrics']['total_trades'],
            'win_rate': result['metrics']['win_rate'],
            'avg_r': result['metrics']['avg_r_multiple'],
            'hit_2r_pct': result['metrics'].get('pct_hit_2r', 0),
            'hit_4r_pct': result['metrics'].get('pct_hit_4r', 0),
        })

        # Restaurar config original
        CONFIG.update(original_config)

    return pd.DataFrame(results)


def run_tlt_analysis(months_list: List[int] = None) -> pd.DataFrame:
    """
    Ejecuta análisis separado para TLT (renta fija).

    Returns:
        DataFrame con resultados por período
    """
    if months_list is None:
        months_list = [6, 12, 24, 36]

    results = []
    for months in months_list:
        force_daily = months > 24
        data, interval = download_all_tickers(['TLT'], months, force_daily)

        if not data:
            continue

        result = run_realistic_backtest(
            data=data,
            initial_capital=CONFIG['initial_capital'],
            verbose=False
        )

        results.append({
            'period': f"{months}m",
            'months': months,
            'interval': interval,
            'trades': result['metrics']['total_trades'],
            'return_pct': result['metrics']['total_return_pct'],
            'max_dd': result['metrics']['max_drawdown_pct'],
            'pf': result['metrics']['profit_factor'],
            'sharpe': result['metrics']['sharpe_ratio'],
        })

    return pd.DataFrame(results)


def run_timeframe_comparison(months: int) -> pd.DataFrame:
    """
    Compara resultados con y sin ajuste de parámetros para datos diarios.
    """
    configs = {
        'proporcional': {'target_vol_annual': 0.40, 'max_hold_bars': 18},
        'directo': {'target_vol_annual': 0.40, 'max_hold_bars': 18},
    }

    results = []

    # Test con ajuste proporcional
    print(f"\n  📊 Test con ajuste proporcional (max_hold ajustado)...")
    df_adj = run_comparison_backtest(
        months=months,
        configs={'proporcional': configs['proporcional']},
        force_daily=True,
        adjust_for_timeframe=True
    )
    if not df_adj.empty:
        results.append(df_adj.iloc[0].to_dict())

    # Test sin ajuste (directo)
    print(f"\n  📊 Test directo (max_hold sin ajustar)...")
    df_direct = run_comparison_backtest(
        months=months,
        configs={'directo': configs['directo']},
        force_daily=True,
        adjust_for_timeframe=False
    )
    if not df_direct.empty:
        results.append(df_direct.iloc[0].to_dict())

    return pd.DataFrame(results)


def print_comparison_table(df: pd.DataFrame, title: str):
    """Imprime tabla de comparación formateada."""
    if df.empty:
        print(f"\n  ⚠️ Sin datos para {title}")
        return

    print(f"""
{'='*80}
  📊 {title}
{'='*80}

  {'Config':<20} {'Return%':<10} {'MaxDD%':<10} {'PF':<8} {'Sharpe':<8} {'Trades':<8} {'WinRate':<8}
  {'─'*80}""")

    for _, r in df.iterrows():
        print(f"  {r['config']:<20} {r['return_pct']:>+8.1f}%  {r['max_dd']:>8.1f}%  "
              f"{r['pf']:>6.2f}  {r['sharpe']:>6.2f}  {r['trades']:>6}  {r.get('win_rate', 0):>6.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Ejecuta backtest con argumentos CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Momentum Breakout Backtest v3.0')
    parser.add_argument('--compare', choices=['configs', 'timeframe', 'all'],
                        help='Tipo de comparación a ejecutar')
    parser.add_argument('--months', type=int, nargs='+', default=[6, 12],
                        help='Meses a testear (default: 6 12)')
    parser.add_argument('--ticker', type=str,
                        help='Ticker específico para análisis aislado')
    parser.add_argument('--tlt', action='store_true',
                        help='Ejecutar análisis TLT separado')
    parser.add_argument('--daily', action='store_true',
                        help='Forzar uso de datos diarios')

    args = parser.parse_args()

    mode = "LONGS ONLY" if CONFIG.get('longs_only', True) else "LONGS + SHORTS"
    target_vol = CONFIG.get('target_vol_annual', 0.40) * 100

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           📊 BACKTEST REALISTA — MOMENTUM BREAKOUT v3.0                       ║
║                                                                               ║
║           Capital: €10,000 | Modo: {mode:<12}                             ║
║                                                                               ║
║           🚀 MEJORAS v3.0:                                                    ║
║           • Target Vol: {target_vol:.0f}% | Partial Exits: 33%@2R, 33%@4R, 34%trailing ║
║           • Soporte 24-36 meses con datos diarios                             ║
║           • Comparación de configs (30% vol, 30 bars, combo)                  ║
║           • Análisis TLT separado (renta fija)                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Análisis TLT separado
    if args.tlt:
        print("\n" + "="*80)
        print("  📊 ANÁLISIS TLT (RENTA FIJA)")
        print("="*80)
        df_tlt = run_tlt_analysis([6, 12, 24, 36])
        if not df_tlt.empty:
            print(f"""
  {'Período':<10} {'Interval':<10} {'Trades':<8} {'Return%':<10} {'MaxDD%':<10} {'PF':<8} {'Sharpe':<8}
  {'─'*80}""")
            for _, r in df_tlt.iterrows():
                print(f"  {r['period']:<10} {r['interval']:<10} {r['trades']:<8} "
                      f"{r['return_pct']:>+8.1f}%  {r['max_dd']:>8.1f}%  {r['pf']:>6.2f}  {r['sharpe']:>6.2f}")
        return

    # Ticker específico
    if args.ticker:
        print(f"\n  📊 Análisis para ticker: {args.ticker}")
        for months in args.months:
            data, interval = download_all_tickers([args.ticker], months, args.daily)
            if data:
                result = run_realistic_backtest(data, CONFIG['initial_capital'], verbose=False)
                print_results(result['metrics'], f"{args.ticker} - {months}m ({interval})")
        return

    # Comparación de configs
    if args.compare == 'configs' or args.compare == 'all':
        for months in args.months:
            df = run_comparison_backtest(months=months, force_daily=args.daily)
            print_comparison_table(df, f"COMPARATIVA CONFIGS - {months} MESES")

    # Comparación de timeframe (24-36 meses)
    if args.compare == 'timeframe' or args.compare == 'all':
        for months in [24, 36]:
            if months in args.months or args.compare == 'all':
                print(f"\n  📊 Comparación timeframe para {months} meses...")
                df = run_timeframe_comparison(months)
                print_comparison_table(df, f"AJUSTE TIMEFRAME - {months} MESES")

    # Si no hay argumentos específicos, ejecutar backtest básico
    if not args.compare and not args.ticker and not args.tlt:
        all_results = []
        for months in args.months:
            force_daily = months > 24 or args.daily
            data, interval = download_all_tickers(CONFIG['test_tickers'], months, force_daily)

            if not data:
                print(f"  ⚠️ No hay datos para {months} meses")
                continue

            print(f"\n  🔄 Período: {months} meses ({interval}) | {len(data)} tickers")
            result = run_realistic_backtest(data, CONFIG['initial_capital'], verbose=False)
            print_results(result['metrics'], f"{months} MESES ({interval})")
            all_results.append({'months': months, **result['metrics']})

        # Resumen
        if len(all_results) > 1:
            print(f"""

{'='*80}
  📊 RESUMEN COMPARATIVO
{'='*80}

  {'Período':<10} {'Trades':<8} {'Return%':<10} {'MaxDD%':<10} {'PF':<8} {'Sharpe':<8}
  {'─'*80}""")
            for r in all_results:
                print(f"  {r['months']}m{' '*7} {r['total_trades']:<8} {r['total_return_pct']:>+8.1f}%  "
                      f"{r['max_drawdown_pct']:>8.1f}%  {r['profit_factor']:>6.2f}  {r['sharpe_ratio']:>6.2f}")


if __name__ == "__main__":
    main()
