#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           📊 BACKTEST MOMENTUM BREAKOUT — PRECIOS REALES                      ║
║                                                                               ║
║           Timeframe: 4H                                                       ║
║           Períodos: 6, 12, 24 meses                                           ║
║           Entrada: Open siguiente barra (no Close)                            ║
║           Stops: Evaluados con High/Low intrabarra                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

METODOLOGÍA REALISTA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Señal generada al CIERRE de barra t
2. Entrada al OPEN de barra t+1 (+ slippage)
3. Stop/Target evaluado con HIGH/LOW de cada barra (no solo Close)
4. Partial exits en targets (33%/33%/34%)
5. Trailing stop solo se mueve en dirección favorable
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar módulos de momentum_breakout
import sys
sys.path.insert(0, '/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code')
from momentum_breakout import (
    MomentumEngine, DynamicStopManager, calculate_atr,
    calculate_scale_out_levels, ASSETS, TICKERS
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_CONFIG = {
    # Costes de transacción
    'slippage_pct': 0.10,       # 0.1% slippage por entrada/salida
    'commission_pct': 0.00,     # Comisión (0 para la mayoría de brokers)

    # Capital
    'initial_capital': 10000,
    'max_positions': 5,
    'risk_per_trade_pct': 2.0,  # % del capital por trade

    # Estrategia
    'partial_exit_1_pct': 33,   # % a cerrar en Target 1
    'partial_exit_2_pct': 33,   # % a cerrar en Target 2

    # Períodos a testear
    'periods_months': [6, 12, 24],

    # Tickers a testear (subset líquido)
    'test_tickers': [
        # US Tech (más líquidos)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        # ETFs índices
        'QQQ', 'SPY', 'IWM',
        # Commodities
        'GLD', 'SLV', 'USO',
        # EU (los más líquidos)
        'SAP', 'ASML',
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE DE TRADE INDIVIDUAL
# ═══════════════════════════════════════════════════════════════════════════════

class Trade:
    """Representa un trade individual con gestión de parciales."""

    def __init__(
        self,
        ticker: str,
        direction: str,  # 'long' or 'short'
        entry_price: float,
        entry_date: datetime,
        entry_atr: float,
        position_size: float,
        stop_price: float,
        target_1: float,
        target_2: float,
    ):
        self.ticker = ticker
        self.direction = direction
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.entry_atr = entry_atr
        self.initial_size = position_size
        self.current_size = position_size
        self.stop_price = stop_price
        self.target_1 = target_1
        self.target_2 = target_2

        self.highest_since_entry = entry_price
        self.lowest_since_entry = entry_price
        self.stop_phase = 'initial'

        self.partial_exits = []
        self.exit_price = None
        self.exit_date = None
        self.exit_reason = None

        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.r_multiple = 0.0
        self.bars_held = 0

    def update_extremes(self, high: float, low: float):
        """Actualiza máximos/mínimos desde entrada."""
        self.highest_since_entry = max(self.highest_since_entry, high)
        self.lowest_since_entry = min(self.lowest_since_entry, low)
        self.bars_held += 1

    def check_stop_hit(self, low: float, high: float) -> bool:
        """Verifica si el stop fue tocado durante la barra."""
        if self.direction == 'long':
            return low <= self.stop_price
        else:
            return high >= self.stop_price

    def check_target_1_hit(self, high: float, low: float) -> bool:
        """Verifica si Target 1 fue alcanzado."""
        if self.direction == 'long':
            return high >= self.target_1
        else:
            return low <= self.target_1

    def check_target_2_hit(self, high: float, low: float) -> bool:
        """Verifica si Target 2 fue alcanzado."""
        if self.direction == 'long':
            return high >= self.target_2
        else:
            return low <= self.target_2


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """Motor de backtest con precios realistas."""

    def __init__(
        self,
        initial_capital: float = 10000,
        slippage_pct: float = 0.10,
        commission_pct: float = 0.0,
        max_positions: int = 5,
    ):
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct / 100
        self.commission_pct = commission_pct / 100
        self.max_positions = max_positions

        # Instanciar estrategia
        self.engine = MomentumEngine(
            breakout_period=20,
            volume_threshold=1.3,
            ker_threshold=0.40,
            rsi_threshold=55,
        )
        self.stop_mgr = DynamicStopManager()

    def apply_slippage(self, price: float, direction: str, is_entry: bool) -> float:
        """Aplica slippage realista."""
        if is_entry:
            # Entry: slippage en contra
            if direction == 'long':
                return price * (1 + self.slippage_pct)
            else:
                return price * (1 - self.slippage_pct)
        else:
            # Exit: slippage en contra
            if direction == 'long':
                return price * (1 - self.slippage_pct)
            else:
                return price * (1 + self.slippage_pct)

    def run_backtest(
        self,
        ticker: str,
        df: pd.DataFrame,
        verbose: bool = False
    ) -> List[Trade]:
        """
        Ejecuta backtest para un ticker.

        IMPORTANTE:
        - Señal en barra t → Entrada en Open de barra t+1
        - Stops evaluados con High/Low, no solo Close
        """
        trades = []
        active_trade = None

        # Generar señales
        signals = self.engine.generate_signals(df)

        # Calcular ATR
        df = df.copy()
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], period=14)

        # Iterar barra por barra
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            signal = signals.iloc[i-1]  # Señal de barra anterior

            current_open = current_bar['Open']
            current_high = current_bar['High']
            current_low = current_bar['Low']
            current_close = current_bar['Close']
            current_atr = df['ATR'].iloc[i]
            current_date = df.index[i]

            # ═══════════════════════════════════════════════════════════════
            # GESTIONAR TRADE ACTIVO
            # ═══════════════════════════════════════════════════════════════

            if active_trade is not None:
                active_trade.update_extremes(current_high, current_low)

                # Actualizar stop dinámico
                stop_info = self.stop_mgr.calculate_stop(
                    position_type=active_trade.direction,
                    entry_price=active_trade.entry_price,
                    current_price=current_close,
                    current_atr=current_atr,
                    highest_since_entry=active_trade.highest_since_entry,
                    lowest_since_entry=active_trade.lowest_since_entry,
                    entry_atr=active_trade.entry_atr
                )

                # Solo mover stop si mejora (nunca empeora)
                if active_trade.direction == 'long':
                    if stop_info['stop_price'] > active_trade.stop_price:
                        active_trade.stop_price = stop_info['stop_price']
                        active_trade.stop_phase = stop_info['phase']
                else:
                    if stop_info['stop_price'] < active_trade.stop_price:
                        active_trade.stop_price = stop_info['stop_price']
                        active_trade.stop_phase = stop_info['phase']

                # ─────────────────────────────────────────────────────────
                # VERIFICAR STOP (con High/Low, no solo Close)
                # ─────────────────────────────────────────────────────────
                if active_trade.check_stop_hit(current_low, current_high):
                    exit_price = self.apply_slippage(
                        active_trade.stop_price,
                        active_trade.direction,
                        is_entry=False
                    )
                    active_trade.exit_price = exit_price
                    active_trade.exit_date = current_date
                    active_trade.exit_reason = f'stop_{active_trade.stop_phase}'

                    # Calcular P&L
                    if active_trade.direction == 'long':
                        active_trade.pnl = (exit_price - active_trade.entry_price) * active_trade.current_size
                        active_trade.pnl_pct = (exit_price / active_trade.entry_price - 1) * 100
                    else:
                        active_trade.pnl = (active_trade.entry_price - exit_price) * active_trade.current_size
                        active_trade.pnl_pct = (active_trade.entry_price / exit_price - 1) * 100

                    R = active_trade.entry_atr * 1.5
                    active_trade.r_multiple = active_trade.pnl / (R * active_trade.initial_size) if R > 0 else 0

                    trades.append(active_trade)
                    active_trade = None
                    continue

                # ─────────────────────────────────────────────────────────
                # VERIFICAR TARGET 1 (partial exit 33%)
                # ─────────────────────────────────────────────────────────
                if (active_trade.check_target_1_hit(current_high, current_low) and
                    len(active_trade.partial_exits) == 0):

                    exit_size = active_trade.initial_size * 0.33
                    exit_price = self.apply_slippage(
                        active_trade.target_1,
                        active_trade.direction,
                        is_entry=False
                    )

                    if active_trade.direction == 'long':
                        partial_pnl = (exit_price - active_trade.entry_price) * exit_size
                    else:
                        partial_pnl = (active_trade.entry_price - exit_price) * exit_size

                    active_trade.partial_exits.append({
                        'date': current_date,
                        'price': exit_price,
                        'size': exit_size,
                        'pnl': partial_pnl,
                        'reason': 'target_1'
                    })
                    active_trade.current_size -= exit_size
                    active_trade.pnl += partial_pnl

                # ─────────────────────────────────────────────────────────
                # VERIFICAR TARGET 2 (partial exit 33%)
                # ─────────────────────────────────────────────────────────
                if (active_trade.check_target_2_hit(current_high, current_low) and
                    len(active_trade.partial_exits) == 1):

                    exit_size = active_trade.initial_size * 0.33
                    exit_price = self.apply_slippage(
                        active_trade.target_2,
                        active_trade.direction,
                        is_entry=False
                    )

                    if active_trade.direction == 'long':
                        partial_pnl = (exit_price - active_trade.entry_price) * exit_size
                    else:
                        partial_pnl = (active_trade.entry_price - exit_price) * exit_size

                    active_trade.partial_exits.append({
                        'date': current_date,
                        'price': exit_price,
                        'size': exit_size,
                        'pnl': partial_pnl,
                        'reason': 'target_2'
                    })
                    active_trade.current_size -= exit_size
                    active_trade.pnl += partial_pnl

                # ─────────────────────────────────────────────────────────
                # TIME EXIT (máximo 30 barras = ~5 días en 4H)
                # ─────────────────────────────────────────────────────────
                if active_trade.bars_held >= 30:
                    exit_price = self.apply_slippage(
                        current_close,
                        active_trade.direction,
                        is_entry=False
                    )

                    if active_trade.direction == 'long':
                        final_pnl = (exit_price - active_trade.entry_price) * active_trade.current_size
                    else:
                        final_pnl = (active_trade.entry_price - exit_price) * active_trade.current_size

                    active_trade.pnl += final_pnl
                    active_trade.exit_price = exit_price
                    active_trade.exit_date = current_date
                    active_trade.exit_reason = 'time_exit'
                    active_trade.pnl_pct = active_trade.pnl / (active_trade.entry_price * active_trade.initial_size) * 100

                    R = active_trade.entry_atr * 1.5
                    active_trade.r_multiple = active_trade.pnl / (R * active_trade.initial_size) if R > 0 else 0

                    trades.append(active_trade)
                    active_trade = None
                    continue

            # ═══════════════════════════════════════════════════════════════
            # NUEVA ENTRADA (si no hay trade activo)
            # ═══════════════════════════════════════════════════════════════

            if active_trade is None and signal != 0:
                direction = 'long' if signal == 1 else 'short'

                # Entrada al Open de esta barra + slippage
                entry_price = self.apply_slippage(current_open, direction, is_entry=True)
                entry_atr = df['ATR'].iloc[i-1]  # ATR de barra anterior

                # Calcular niveles
                levels = calculate_scale_out_levels(entry_price, entry_atr, direction)

                # Position size (simplificado: 1 unidad)
                position_size = 1.0

                active_trade = Trade(
                    ticker=ticker,
                    direction=direction,
                    entry_price=entry_price,
                    entry_date=current_date,
                    entry_atr=entry_atr,
                    position_size=position_size,
                    stop_price=levels['initial_stop'],
                    target_1=levels['target_1']['price'],
                    target_2=levels['target_2']['price'],
                )

                if verbose:
                    print(f"  {current_date.strftime('%Y-%m-%d')} | {direction.upper():5} | "
                          f"Entry: ${entry_price:.2f} | Stop: ${levels['initial_stop']:.2f} | "
                          f"T1: ${levels['target_1']['price']:.2f}")

        # Cerrar trade abierto al final
        if active_trade is not None:
            exit_price = df['Close'].iloc[-1]
            if active_trade.direction == 'long':
                final_pnl = (exit_price - active_trade.entry_price) * active_trade.current_size
            else:
                final_pnl = (active_trade.entry_price - exit_price) * active_trade.current_size

            active_trade.pnl += final_pnl
            active_trade.exit_price = exit_price
            active_trade.exit_date = df.index[-1]
            active_trade.exit_reason = 'end_of_data'
            active_trade.pnl_pct = active_trade.pnl / (active_trade.entry_price * active_trade.initial_size) * 100
            trades.append(active_trade)

        return trades


# ═══════════════════════════════════════════════════════════════════════════════
# DESCARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def download_data(ticker: str, months: int) -> Optional[pd.DataFrame]:
    """Descarga datos 4H para el período especificado."""
    try:
        # yfinance limita 4H a ~60 días, usar 1H y resamplear
        if months <= 2:
            period = f'{months}mo'
            interval = '4h'
        else:
            # Para períodos más largos, usar 1H y resamplear
            period = f'{min(months, 24)}mo'  # yfinance limita a ~730 días para 1H
            interval = '1h'

        df = yf.download(ticker, period=period, interval=interval, progress=False)

        if df.empty:
            return None

        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Si descargamos 1H, resamplear a 4H
        if interval == '1h' and len(df) > 0:
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        return df if len(df) >= 50 else None

    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_results(trades: List[Trade], period_name: str) -> Dict:
    """Analiza resultados del backtest."""
    if not trades:
        return None

    # Métricas básicas
    total_trades = len(trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0

    # R-multiples
    r_multiples = [t.r_multiple for t in trades]
    avg_r = np.mean(r_multiples)

    # Profit factor
    gross_profit = sum([t.pnl for t in winners]) if winners else 0
    gross_loss = abs(sum([t.pnl for t in losers])) if losers else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Expectancy
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Max consecutive losses
    max_consec_losses = 0
    current_streak = 0
    for t in trades:
        if t.pnl <= 0:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0

    # Exit reasons
    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason or 'unknown'
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Holding time
    avg_bars_held = np.mean([t.bars_held for t in trades])

    return {
        'period': period_name,
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'avg_r_multiple': avg_r,
        'profit_factor': profit_factor,
        'expectancy_pct': expectancy,
        'max_consec_losses': max_consec_losses,
        'avg_bars_held': avg_bars_held,
        'exit_reasons': exit_reasons,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
    }


def print_results(results: Dict):
    """Imprime resultados de forma legible."""
    if not results:
        print("  No hay trades para analizar")
        return

    print(f"\n{'─'*60}")
    print(f"  📊 RESULTADOS: {results['period']}")
    print(f"{'─'*60}")

    print(f"\n  📈 PERFORMANCE:")
    print(f"     Total trades:      {results['total_trades']}")
    print(f"     Win Rate:          {results['win_rate']:.1f}%")
    print(f"     Avg Win:           +{results['avg_win_pct']:.2f}%")
    print(f"     Avg Loss:          {results['avg_loss_pct']:.2f}%")
    print(f"     Profit Factor:     {results['profit_factor']:.2f}")
    print(f"     Expectancy:        {results['expectancy_pct']:.2f}%/trade")

    print(f"\n  📐 R-MULTIPLES:")
    print(f"     Avg R:             {results['avg_r_multiple']:.2f}R")
    print(f"     Max Consec Loss:   {results['max_consec_losses']}")

    print(f"\n  ⏱️  TIMING:")
    print(f"     Avg Hold:          {results['avg_bars_held']:.1f} barras (~{results['avg_bars_held']*4:.0f}h)")

    print(f"\n  🚪 EXIT REASONS:")
    for reason, count in results['exit_reasons'].items():
        pct = count / results['total_trades'] * 100
        print(f"     {reason:20} {count:3} ({pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_backtest():
    """Ejecuta backtest completo en múltiples períodos."""

    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           📊 BACKTEST MOMENTUM BREAKOUT — PRECIOS REALES                      ║
║                                                                               ║
║           • Entrada: Open siguiente barra + slippage                          ║
║           • Stops: Evaluados con High/Low intrabarra                          ║
║           • Partial Exits: 33%/33%/34%                                        ║
║           • Trailing Stop: Chandelier Exit                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    engine = BacktestEngine(
        initial_capital=10000,
        slippage_pct=0.10,
    )

    all_results = []

    for months in BACKTEST_CONFIG['periods_months']:
        print(f"\n{'='*70}")
        print(f"  🔄 PERÍODO: {months} MESES")
        print(f"{'='*70}")

        period_trades = []

        for ticker in BACKTEST_CONFIG['test_tickers']:
            print(f"\r  Procesando {ticker}...", end='', flush=True)

            df = download_data(ticker, months)
            if df is None:
                continue

            trades = engine.run_backtest(ticker, df, verbose=False)
            period_trades.extend(trades)

        print(f"\r  Procesados {len(BACKTEST_CONFIG['test_tickers'])} tickers" + " "*20)

        # Analizar resultados del período
        results = analyze_results(period_trades, f"{months} meses")
        if results:
            all_results.append(results)
            print_results(results)

    # ═══════════════════════════════════════════════════════════════════════════
    # RESUMEN COMPARATIVO
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*70}")
    print(f"  📊 RESUMEN COMPARATIVO")
    print(f"{'='*70}")

    print(f"\n  {'Período':<12} {'Trades':<8} {'Win%':<8} {'AvgR':<8} {'PF':<8} {'Expect':<10}")
    print(f"  {'─'*54}")

    for r in all_results:
        print(f"  {r['period']:<12} {r['total_trades']:<8} {r['win_rate']:<8.1f} "
              f"{r['avg_r_multiple']:<8.2f} {r['profit_factor']:<8.2f} {r['expectancy_pct']:<10.2f}%")

    # Conclusiones
    print(f"\n\n{'='*70}")
    print(f"  💡 CONCLUSIONES")
    print(f"{'='*70}")

    if all_results:
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_pf = np.mean([r['profit_factor'] for r in all_results])
        avg_expect = np.mean([r['expectancy_pct'] for r in all_results])

        print(f"""
  📊 Promedios across períodos:
     • Win Rate promedio:     {avg_win_rate:.1f}%
     • Profit Factor prom:    {avg_pf:.2f}
     • Expectancy promedio:   {avg_expect:.2f}%/trade

  ⚠️  NOTAS IMPORTANTES:
     • Este backtest NO incluye correlación entre posiciones
     • Slippage real puede variar según liquidez
     • Resultados pasados no garantizan futuros
     • Revisar períodos de alta volatilidad vs baja vol
        """)

    return all_results


if __name__ == "__main__":
    results = run_full_backtest()
