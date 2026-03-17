#!/usr/bin/env python3
"""
BACKTEST DEFINITIVO v6 - MOMENTUM BREAKOUT (Fat Tails)

Estrategia: Momentum Breakout con fat tails (solo acciones/ETFs)
  - Win rate bajo (~31%) pero ratio avg win/loss ~5:1
  - Los ganadores corren indefinidamente con trailing dinamico
  - Los perdedores activan trailing a 8 dias (3xATR, sin salida forzada)

Configuracion optimizada (Feb 2026):
  - Datos DIARIOS directos (112 tickers, sin crypto)
  - Ranking multi-factor: KER + RSI + Volume + Breakout + ATR%
  - Position sizing: 2% equity risk/trade (inverse volatility)
  - Max 7 posiciones simultaneas (optimizado: mejor return, PF y drawdown que 5)
  - Trailing Chandelier Exit 4xATR (activado a +2R)
  - Emergency stop -15% (solo catastrofe)
  - Time exit 12 dias SOLO para perdedores/flat
  - Filtro macro: SPY > SMA50 (no entrar en correcciones)
  - Sin partial exits: posicion completa corre con trailing

Cambio v5 → v6:
  - Max posiciones: 5 → 7 (testado 5/7/8/10/15, optimo en 7)
  - Validado a 36 meses: v6 +77.6% vs v5 +58.6% (+19pp)

Resultados backtest:
  - 18m: +67.2% total, +40.9% anualizado, PF 3.42, MaxDD -6.1%
  - 36m: +77.6% total, +21.1% anualizado, PF 2.26, MaxDD -11.2%
  - Trimestres negativos: 3 de 13 (perdidas contenidas)

Variante v6+ (backtest_experimental.py):
  - Anade opciones CALL 5% ITM, 120 DTE, cierre a 45 DTE, IVR<40
  - 36m: +427.7% total, +74.1% anualizado, PF 2.87, MaxDD -37.0%
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_breakout import (
    MomentumEngine, calculate_atr, calculate_position_size, ASSETS, TICKERS
)


# =============================================================================
# CONFIGURACION v6 — Optimizada Feb 2026
# =============================================================================

CONFIG = {
    # Capital
    'initial_capital': 10000,

    # Position sizing — volatility-based (directo para datos diarios)
    'target_risk_per_trade_pct': 2.0,  # 2% del equity en riesgo por trade (= 2x ATR)
    'max_positions': 7,               # Optimizado: 7 > 5 en return, PF y drawdown

    # Senales (mismos umbrales, el engine funciona igual con datos diarios)
    'ker_threshold': 0.40,
    'volume_threshold': 1.3,
    'rsi_threshold': 50,
    'rsi_max': 75,
    'breakout_period': 20,
    'longs_only': True,

    # Stop management — SOLO trailing dinamico
    'emergency_stop_pct': 0.15,     # -15% desde entrada (solo catastrofe)
    'trail_trigger_r': 2.0,         # Activar trailing a +2R
    'trail_atr_mult': 4.0,         # Chandelier 4xATR (amplio para fat tails)
    # SIN partial exits — posicion completa corre con trailing

    # Time management
    # Re-optimizado a 240m: 8 bars + trailing only 3xATR >> 12 bars + forzar salida
    # El time exit forzado tenia 0% win rate y -248k EUR en 20 anos
    # Ahora: a los 8 bars se activa trailing apretado (3xATR), nunca se fuerza salida
    'max_hold_bars': 8,
    'time_exit_trail_atr_mult': 3.0,  # ATR mult para trailing activado por time exit

    # Filtro macro — solo entrar cuando SPY > SMA50 (mercado alcista)
    'use_macro_filter': True,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,

    # Costes
    'slippage_pct': 0.10,

    # Tickers: todo el universo de ASSETS (crypto incluido via yfinance -USD)
    'test_tickers': [t for t, v in ASSETS.items()],
}


# =============================================================================
# TRADE CLASS
# =============================================================================

@dataclass
class Trade:
    """
    Trade con trailing dinamico puro. Sin partial exits.
    Posicion completa entra y sale entera.

    Salidas:
    1. Emergency stop -15% (catastrofe)
    2. Trailing Chandelier 4xATR (activado a +2R, o al llegar a time limit si en positivo)
    3. Time exit a 12d SOLO si trade en negativo/flat
    """

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

    def update(self, high: float, low: float, close: float, current_atr: float) -> Optional[dict]:
        self.bars_held += 1
        self.highest_since = max(self.highest_since, high)
        r_mult = (close - self.entry_price) / self.R if self.R > 0 else 0
        self.max_r_mult = max(self.max_r_mult, r_mult)

        # 0. EMERGENCY STOP (-15%)
        emergency_level = self.entry_price * (1 - CONFIG['emergency_stop_pct'])
        if low <= emergency_level:
            self._close(emergency_level * (1 - CONFIG['slippage_pct'] / 100), 'emergency_stop')
            return {'type': 'full_exit', 'reason': 'emergency_stop'}

        # 1. TRAILING STOP CHECK
        if self.trailing_active and self.trailing_stop is not None:
            if low <= self.trailing_stop:
                self._close(self.trailing_stop * (1 - CONFIG['slippage_pct'] / 100), 'trailing_stop')
                return {'type': 'full_exit', 'reason': 'trailing_stop'}

        # 2. ACTUALIZAR TRAILING (Chandelier 4xATR)
        if r_mult >= CONFIG['trail_trigger_r']:
            chandelier = self.highest_since - (current_atr * CONFIG['trail_atr_mult'])
            if not self.trailing_active:
                self.trailing_active = True
                self.trailing_stop = chandelier
            elif chandelier > self.trailing_stop:
                self.trailing_stop = chandelier

        # 3. TIME EXIT: tras max_hold_bars, activar trailing (nunca forzar salida)
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

    def _close(self, exit_price: float, reason: str):
        self.pnl_euros = (exit_price - self.entry_price) * self.position_units
        self.pnl_pct = (self.pnl_euros / self.position_euros) * 100 if self.position_euros > 0 else 0
        self.exit_price = exit_price
        self.exit_reason = reason


# =============================================================================
# EQUITY TRACKER
# =============================================================================

class EquityTracker:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.max_equity = initial_capital
        self.open_positions = 0

    def get_position_size(self, current_atr: float, price: float) -> Dict:
        """
        Position sizing basado en riesgo fijo por trade.

        Logica: arriesgar X% del equity por trade.
        R = 2 * ATR (riesgo por unidad)
        units = (equity * risk_pct) / R
        notional = units * price

        Esto hace que activos volatiles tengan posiciones mas pequenas
        y activos estables tengan posiciones mas grandes (volatility inverse).
        """
        risk_pct = CONFIG['target_risk_per_trade_pct'] / 100
        R = current_atr * 2.0  # R = 2x ATR diario
        if R <= 0 or price <= 0:
            return {'units': 0, 'notional': 0}

        dollar_risk = self.equity * risk_pct
        units = dollar_risk / R
        notional = units * price

        # Cap: no mas de equity / max_positions por posicion
        max_notional = self.equity / CONFIG['max_positions'] * 2  # hasta 40% por posicion
        if notional > max_notional:
            notional = max_notional
            units = notional / price

        return {'units': units, 'notional': notional}

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
# DATA DOWNLOAD — Datos diarios directos
# =============================================================================

def download_data(ticker: str, months: int) -> Optional[pd.DataFrame]:
    """Descarga datos diarios. Sin barras sinteticas."""
    try:
        df = yf.download(ticker, period=f'{months}mo', interval='1d', progress=False)

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
    """
    Backtest con datos diarios y universo completo.

    REGLAS v6:
    1. Position size = 2% equity risk/trade (inverse volatility)
    2. Maximo 7 posiciones simultaneas (ranking multi-factor)
    3. Emergency stop -15% | Trailing Chandelier 4xATR a +2R
    4. Sin partial exits — posicion completa corre con trailing
    5. Time exit: 12 dias SOLO para perdedores/flat
    6. Filtro macro: SPY > SMA50 (no entrar en correcciones)
    """

    n_tickers = len(CONFIG['test_tickers'])
    print(f"\n{'='*70}")
    print(f"  BACKTEST v6 -- {months} MESES -- {n_tickers} tickers")
    print(f"{'='*70}")
    print(f"  Capital: EUR {CONFIG['initial_capital']:,}")
    print(f"  Sizing: {CONFIG['target_risk_per_trade_pct']}% risk/trade (inverse vol)")
    print(f"  Posiciones: max {CONFIG['max_positions']} (ranking multi-factor)")
    print(f"  Stops: emergency -{CONFIG['emergency_stop_pct']*100:.0f}% | trailing {CONFIG['trail_atr_mult']:.0f}xATR a +2R")
    print(f"  Exits: trailing dinamico | time {CONFIG['max_hold_bars']}d activa trailing 3xATR")
    macro_str = f"SPY > SMA{CONFIG.get('macro_sma_period', 50)}" if CONFIG.get('use_macro_filter') else "OFF"
    print(f"  Filtro macro: {macro_str}")
    print(f"  Datos: DIARIOS (sin barras sinteticas)")
    print(f"{'='*70}\n")

    # Descargar datos
    print("  Descargando datos...")
    all_data = {}
    failed = []
    for i, ticker in enumerate(CONFIG['test_tickers']):
        df = download_data(ticker, months)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
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

    # Generar senales y metadata (KER, RSI, vol_ratio para ranking multi-factor)
    signals_data = {}
    total_signals = 0
    for ticker, df in all_data.items():
        meta = engine.generate_signals_with_metadata(df)
        signals = meta['signal']
        n_long = (signals == 1).sum()
        total_signals += n_long
        signals_data[ticker] = {
            'df': df,
            'signals': signals,
            'ker': meta['ker'],
            'rsi': meta['rsi'],
            'vol_ratio': meta['vol_ratio'],
        }

    print(f"  Senales LONG totales: {total_signals} "
          f"({total_signals/sum(len(sd['df']) for sd in signals_data.values())*100:.1f}% de barras)\n")

    # Filtro macro: SPY > SMA50 para decidir si abrimos nuevas posiciones
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
                    macro_bullish[date] = True  # Sin datos = permitir
            n_bullish = sum(1 for v in macro_bullish.values() if v)
            n_bearish = sum(1 for v in macro_bullish.values() if not v)
            print(f"  Filtro macro ({macro_ticker} > SMA{sma_period}): "
                  f"{n_bullish} dias bull / {n_bearish} dias bear\n")
        else:
            print(f"  WARN: {macro_ticker} no disponible para filtro macro\n")

    # Timeline unificado
    all_dates = set()
    for sd in signals_data.values():
        all_dates.update(sd['df'].index.tolist())
    all_dates = sorted(all_dates)

    active_trades: Dict[str, Trade] = {}
    all_trades: List[Trade] = []

    # =================================================================
    # LOOP PRINCIPAL
    # =================================================================

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

            result = trade.update(
                high=bar['High'], low=bar['Low'],
                close=bar['Close'], current_atr=current_atr
            )

            if result and result['type'] == 'full_exit':
                trade.exit_date = current_date
                trades_to_close.append(ticker)
                tracker.update_equity(trade.pnl_euros, current_date)

        for ticker in trades_to_close:
            trade = active_trades.pop(ticker)
            tracker.open_positions -= 1
            all_trades.append(trade)

            if verbose and len(all_trades) <= 30:
                print(f"  {current_date.strftime('%Y-%m-%d')} | CLOSE {ticker:8} | "
                      f"{trade.exit_reason:15} | P&L: EUR {trade.pnl_euros:+.2f} ({trade.pnl_pct:+.1f}%) "
                      f"| {trade.bars_held}d")

        # 2. BUSCAR NUEVAS SENALES
        # Filtro macro: si SPY < SMA50, no abrir nuevas posiciones
        if CONFIG.get('use_macro_filter', False) and macro_bullish:
            if current_date in macro_bullish:
                is_macro_ok = macro_bullish[current_date]
            else:
                # Festivos US: usar el ultimo valor conocido de SPY
                prev_dates = [d for d in macro_bullish if d < current_date]
                is_macro_ok = macro_bullish[prev_dates[-1]] if prev_dates else False
        else:
            is_macro_ok = True

        if tracker.open_positions < CONFIG['max_positions'] and is_macro_ok:
            # Recopilar todas las senales del dia y ordenar por fuerza
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

            # RANKING MULTI-FACTOR COMPUESTO
            # Combina KER, RSI, Volume y Breakout Strength
            # para seleccionar las senales con mayor probabilidad de fat tail
            ranked = []
            for ticker, idx, prev_atr in candidates:
                sd = signals_data[ticker]
                df_t = sd['df']
                prev_idx = idx - 1

                # 1. KER — fuerza de tendencia (0-1)
                ker_val = sd['ker'].iloc[prev_idx] if prev_idx >= 0 else 0

                # 2. RSI normalizado — momentum (0-1, donde 75→1.0, 50→0.0)
                rsi_val = sd['rsi'].iloc[prev_idx] if prev_idx >= 0 else 50
                rsi_score = max(0, min(1, (rsi_val - CONFIG['rsi_threshold']) / (CONFIG['rsi_max'] - CONFIG['rsi_threshold'])))

                # 3. Volume ratio — confirmacion (min-capped a 1.3, normalizado)
                vol_val = sd['vol_ratio'].iloc[prev_idx] if prev_idx >= 0 else 1.0
                vol_score = min(1, max(0, (vol_val - 1.0) / 2.0))  # 1.0→0, 3.0→1

                # 4. Breakout strength — distancia del close sobre el rolling high (%)
                if prev_idx >= 1:
                    close_prev = df_t['Close'].iloc[prev_idx]
                    rolling_high_prev = df_t['High'].iloc[max(0, prev_idx - CONFIG['breakout_period']):prev_idx].max()
                    breakout_pct = (close_prev - rolling_high_prev) / rolling_high_prev if rolling_high_prev > 0 else 0
                    breakout_score = min(1, max(0, breakout_pct / 0.05))  # 0%→0, 5%→1
                else:
                    breakout_score = 0

                # 5. ATR% — volatilidad relativa (preferir activos con mas recorrido potencial)
                price_prev = df_t['Close'].iloc[prev_idx] if prev_idx >= 0 else 1
                atr_pct = prev_atr / price_prev if price_prev > 0 else 0
                atr_score = min(1, atr_pct / 0.04)  # 0%→0, 4%→1

                # SCORE COMPUESTO (ponderado)
                composite = (
                    0.30 * ker_val +           # Tendencia limpia
                    0.20 * rsi_score +         # Momentum
                    0.20 * vol_score +         # Confirmacion volumen
                    0.15 * breakout_score +    # Fuerza del breakout
                    0.15 * atr_score           # Potencial de recorrido
                )

                ranked.append((ticker, idx, prev_atr, composite))

            ranked.sort(key=lambda x: x[3], reverse=True)  # Score mas alto primero

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= CONFIG['max_positions']:
                    break

                df = signals_data[ticker]['df']
                bar = df.iloc[idx]

                size_info = tracker.get_position_size(
                    current_atr=prev_atr,
                    price=bar['Open']
                )

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
                    position_units=position_units
                )

                active_trades[ticker] = trade
                tracker.open_positions += 1

                if verbose and len(all_trades) + len(active_trades) <= 35:
                    print(f"  {current_date.strftime('%Y-%m-%d')} | OPEN  {ticker:8} | "
                          f"EUR {position_euros:.0f} ({position_units:.2f}u) @ ${entry_price:.2f}")

    # Cerrar trades abiertos al final
    for ticker, trade in active_trades.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            exit_price = df['Close'].iloc[-1]
            trade._close(exit_price, 'end_of_data')
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

    trades_gt_3r = sum(1 for t in all_trades if t.max_r_mult >= 3.0)
    max_dd = tracker.get_max_drawdown()

    total_return_pct = (tracker.equity / CONFIG['initial_capital'] - 1) * 100
    annualized = ((1 + total_return_pct / 100) ** (12 / months) - 1) * 100 if months > 0 else 0

    avg_win_euros = np.mean([t.pnl_euros for t in winners]) if winners else 0
    avg_win_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss_euros = np.mean([t.pnl_euros for t in losers]) if losers else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0

    # Avg bars held
    avg_bars_winners = np.mean([t.bars_held for t in winners]) if winners else 0
    avg_bars_losers = np.mean([t.bars_held for t in losers]) if losers else 0

    # Biggest winner / loser
    best_trade = max(all_trades, key=lambda t: t.pnl_pct)
    worst_trade = min(all_trades, key=lambda t: t.pnl_pct)

    print(f"""
{'='*70}
  RESULTADOS -- {months} MESES ({len(all_data)} tickers)
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

  FAT TAILS:
     Alcanzaron +3R: {trades_gt_3r} ({trades_gt_3r/total_trades*100:.1f}%)
     Best trade:     {best_trade.ticker} {best_trade.pnl_pct:+.1f}% (EUR {best_trade.pnl_euros:+.2f})
     Worst trade:    {worst_trade.ticker} {worst_trade.pnl_pct:+.1f}% (EUR {worst_trade.pnl_euros:+.2f})

  PROMEDIOS:
     Avg Win:        EUR {avg_win_euros:+.2f} ({avg_win_pct:+.1f}%) | {avg_bars_winners:.0f} dias
     Avg Loss:       EUR {avg_loss_euros:.2f} ({avg_loss_pct:.1f}%) | {avg_bars_losers:.0f} dias
     Avg Trade:      EUR {total_pnl/total_trades:+.2f}
""")

    exit_reasons = {}
    for t in all_trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    print("  RAZONES DE SALIDA:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"     {reason:20} {count:3} ({count/total_trades*100:.1f}%)")

    # =========================================================================
    # DIAGNOSTICO TRIMESTRAL — ver evolucion temporal
    # =========================================================================
    if months >= 12 and all_trades:
        print(f"\n  DIAGNOSTICO POR TRIMESTRE:")
        print(f"  {'Trimestre':<12} {'Trades':<8} {'Win%':<7} {'AvgWin':<9} {'AvgLoss':<9} "
              f"{'PnL EUR':<11} {'FatTails':<10} {'AvgBars':<8}")
        print(f"  {'-'*80}")

        # Asignar trimestre a cada trade
        trade_quarters = {}
        for t in all_trades:
            if t.entry_date is not None:
                q = f"{t.entry_date.year}Q{(t.entry_date.month - 1)//3 + 1}"
                trade_quarters.setdefault(q, []).append(t)

        for q in sorted(trade_quarters.keys()):
            q_trades = trade_quarters[q]
            q_winners = [t for t in q_trades if t.pnl_euros > 0]
            q_losers = [t for t in q_trades if t.pnl_euros <= 0]
            q_total = len(q_trades)
            q_wr = len(q_winners) / q_total * 100 if q_total > 0 else 0
            q_pnl = sum(t.pnl_euros for t in q_trades)
            q_avg_win = np.mean([t.pnl_pct for t in q_winners]) if q_winners else 0
            q_avg_loss = np.mean([t.pnl_pct for t in q_losers]) if q_losers else 0
            q_fat = sum(1 for t in q_trades if t.max_r_mult >= 3.0)
            q_avg_bars = np.mean([t.bars_held for t in q_trades])
            print(f"  {q:<12} {q_total:<8} {q_wr:<7.1f} {q_avg_win:>+7.1f}%  {q_avg_loss:>+7.1f}%  "
                  f"EUR{q_pnl:>+8.0f}  {q_fat}/{q_total:<7} {q_avg_bars:>5.1f}d")

        # Detalle de fat tails por trimestre
        print(f"\n  MEJORES TRADES POR TRIMESTRE:")
        for q in sorted(trade_quarters.keys()):
            q_trades = trade_quarters[q]
            best = max(q_trades, key=lambda t: t.pnl_pct)
            worst = min(q_trades, key=lambda t: t.pnl_pct)
            print(f"  {q}: Best {best.ticker} {best.pnl_pct:+.1f}% ({best.bars_held}d) | "
                  f"Worst {worst.ticker} {worst.pnl_pct:+.1f}% ({worst.bars_held}d)")

    # =========================================================================
    # HISTORICO DETALLADO DE TRADES
    # =========================================================================
    if verbose and all_trades:
        print(f"\n{'='*110}")
        print(f"  HISTORICO COMPLETO DE POSICIONES ({len(all_trades)} trades)")
        print(f"{'='*110}")
        print(f"  {'#':<4} {'Entrada':<12} {'Salida':<12} {'Ticker':<10} {'P.Entrada':>10} {'P.Salida':>10} "
              f"{'PnL EUR':>10} {'PnL%':>7} {'Dias':>5} {'MaxR':>5} {'Razon':<16}")
        print(f"  {'-'*108}")

        equity_running = CONFIG['initial_capital']
        for i, t in enumerate(sorted(all_trades, key=lambda x: x.entry_date), 1):
            entry_str = t.entry_date.strftime('%Y-%m-%d') if t.entry_date else '?'
            exit_str = t.exit_date.strftime('%Y-%m-%d') if t.exit_date else '?'
            equity_running += t.pnl_euros
            marker = '✓' if t.pnl_euros > 0 else '✗'
            print(f"  {i:<4} {entry_str:<12} {exit_str:<12} {t.ticker:<10} "
                  f"${t.entry_price:>9.2f} ${t.exit_price:>9.2f} "
                  f"EUR{t.pnl_euros:>+8.0f} {t.pnl_pct:>+6.1f}% {t.bars_held:>4}d "
                  f"{t.max_r_mult:>+5.1f}R {t.exit_reason:<16} {marker}")

        print(f"  {'-'*108}")
        print(f"  {'':>4} {'':>12} {'':>12} {'TOTAL':<10} {'':>10} {'':>10} "
              f"EUR{total_pnl:>+8.0f} {total_return_pct:>+6.1f}%")
        print()

        # Exportar a CSV
        csv_data = []
        for t in sorted(all_trades, key=lambda x: x.entry_date):
            csv_data.append({
                'entry_date': t.entry_date.strftime('%Y-%m-%d') if t.entry_date else '',
                'exit_date': t.exit_date.strftime('%Y-%m-%d') if t.exit_date else '',
                'ticker': t.ticker,
                'entry_price': round(t.entry_price, 2),
                'exit_price': round(t.exit_price, 2) if t.exit_price else 0,
                'position_eur': round(t.position_euros, 0),
                'units': round(t.position_units, 4),
                'pnl_eur': round(t.pnl_euros, 2),
                'pnl_pct': round(t.pnl_pct, 2),
                'bars_held': t.bars_held,
                'max_r_mult': round(t.max_r_mult, 2),
                'exit_reason': t.exit_reason,
            })

        csv_df = pd.DataFrame(csv_data)
        csv_path = f'historico_trades_{months}m.csv'
        csv_df.to_csv(csv_path, index=False)
        print(f"  📁 Exportado a: {csv_path}")
        print()

    return {
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'total_pnl_euros': total_pnl,
        'total_return_pct': total_return_pct,
        'annualized_return_pct': annualized,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'trades_gt_3r': trades_gt_3r,
        'final_equity': tracker.equity,
        'trades': all_trades,
        'total_signals': total_signals,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Backtest Definitivo v6 Momentum Breakout')
    parser.add_argument('--months', type=int, default=36, help='Meses de historico (default: 36)')
    parser.add_argument('--quiet', action='store_true', help='Modo silencioso')

    args = parser.parse_args()

    macro_str = f"SPY > SMA{CONFIG.get('macro_sma_period', 50)}" if CONFIG.get('use_macro_filter') else "OFF"
    print(f"""
======================================================================
  BACKTEST DEFINITIVO v6 -- MOMENTUM BREAKOUT (Fat Tails)
======================================================================
  - Datos DIARIOS directos ({len(CONFIG['test_tickers'])} tickers)
  - Ranking: multi-factor (KER + RSI + Vol + Breakout + ATR%)
  - Sizing: {CONFIG['target_risk_per_trade_pct']}% risk/trade (inverse vol)
  - Max {CONFIG['max_positions']} posiciones
  - Emergency stop -{CONFIG['emergency_stop_pct']*100:.0f}% | Trailing {CONFIG['trail_atr_mult']:.0f}xATR a +2R
  - Sin partial exits: posicion completa con trailing dinamico
  - Time exit: {CONFIG['max_hold_bars']}d activa trailing 3xATR (sin salida forzada)
  - Filtro macro: {macro_str}
======================================================================
    """)

    all_results = []

    for period in [6, 12, 18, 24, 36, 48, 60]:
        if period <= args.months:
            result = run_backtest(months=period, verbose=not args.quiet)
            if 'error' not in result:
                all_results.append({'period': period, **result})

    if len(all_results) > 1:
        print(f"""
{'='*70}
  RESUMEN COMPARATIVO
{'='*70}

  {'Periodo':<8} {'Trades':<8} {'Signals':<8} {'Win%':<7} {'PnL EUR':<11} {'Return%':<9} {'Annual%':<9} {'MaxDD%':<7} {'PF':<6}
  {'-'*76}""")

        for r in all_results:
            print(f"  {r['period']}m{' '*5} {r['total_trades']:<8} {r.get('total_signals','?'):<8} "
                  f"{r['win_rate']:<7.1f} "
                  f"EUR{r['total_pnl_euros']:>+8,.0f}  {r['total_return_pct']:>+7.1f}%  "
                  f"{r['annualized_return_pct']:>+7.1f}%  "
                  f"{r['max_drawdown']:>5.1f}%  {r['profit_factor']:.2f}")

        print()

        returns = [r['annualized_return_pct'] for r in all_results]
        if len(returns) >= 2:
            is_consistent = all(r > 0 for r in returns)
            spread = max(returns) - min(returns)
            print(f"  CONSISTENCIA:")
            print(f"     Todos periodos positivos: {'SI' if is_consistent else 'NO'}")
            print(f"     Spread annualizado:       {spread:.1f}pp")
            if spread < 15:
                print(f"     Veredicto:                CONSISTENTE")
            elif spread < 30:
                print(f"     Veredicto:                ACEPTABLE")
            else:
                print(f"     Veredicto:                INCONSISTENTE - revisar")
            print()


if __name__ == "__main__":
    main()
