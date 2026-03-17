#!/usr/bin/env python3
"""
TEST: Backtest v5 con ETFs apalancados vs normales.

Sustituye los ETFs normales por sus versiones apalancadas donde existan.
Mantiene TODA la logica identica (ranking, trailing, filtro macro, etc).
El position sizing por inverse volatility se adapta automaticamente:
  - ETFs apalancados tienen ATR mas alto → posiciones mas pequenas
  - Esto protege contra el riesgo adicional del apalancamiento
"""

import sys
sys.path.insert(0, '/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code')

# Mapeo: ETF normal → ETF apalancado
LEVERAGED_MAP = {
    # Precious metals
    'GLD':  'UGL',   # 2x Gold
    'SLV':  'AGQ',   # 2x Silver
    'GDX':  'NUGT',  # 2x Gold Miners
    'GDXJ': 'JNUG',  # 2x Junior Gold Miners

    # Energy
    'USO':  'UCO',   # 2x Crude Oil WTI
    'BNO':  'UCO',   # 2x Crude (no hay Brent apalancado)
    'UNG':  'BOIL',  # 2x Natural Gas
    'XLE':  'ERX',   # 2x Energy Sector
    'XOP':  'GUSH',  # 2x Oil & Gas E&P

    # Bonds
    'TLT':  'TMF',   # 3x 20+ Year Treasury

    # Indices (ya existian en el universo pero asegurar que se usan)
    'QQQ':  'TQQQ',  # 3x Nasdaq
    'SPY':  'SPXL',  # 3x S&P 500
    'IWM':  'TNA',   # 3x Russell 2000
    'DIA':  'UDOW',  # 3x Dow Jones
}

# Importar el backtest
import backtest_definitivo as bt
from momentum_breakout import ASSETS

def build_leveraged_tickers():
    """Construir lista de tickers sustituyendo normales por apalancados.
    IMPORTANTE: mantener SPY en la lista para que el filtro macro funcione."""
    original = [t for t, v in ASSETS.items()
                if not v.get('is_crypto', False)
                and not t.endswith('USDT')]

    leveraged = []
    replaced = []
    for ticker in original:
        if ticker in LEVERAGED_MAP:
            lev = LEVERAGED_MAP[ticker]
            if lev not in leveraged:  # Evitar duplicados (BNO→UCO cuando USO→UCO ya existe)
                leveraged.append(lev)
                replaced.append(f"{ticker} → {lev}")
            # Si el ticker original es SPY, TAMBIEN mantenerlo para filtro macro
            if ticker == 'SPY' and 'SPY' not in leveraged:
                leveraged.append('SPY')
        else:
            leveraged.append(ticker)

    return leveraged, replaced


def run_comparison():
    """Ejecutar ambos backtests y comparar."""

    leveraged_tickers, replaced = build_leveraged_tickers()

    print("="*70)
    print("  COMPARATIVA: ETFs NORMALES vs ETFs APALANCADOS")
    print("="*70)
    print(f"\n  Sustituciones ({len(replaced)}):")
    for r in replaced:
        print(f"    {r}")
    print(f"\n  Tickers totales: {len(leveraged_tickers)} (vs {len(bt.CONFIG['test_tickers'])} original)")
    print()

    # =============================================
    # TEST 1: ORIGINAL (v5 tal cual)
    # =============================================
    print("\n" + "="*70)
    print("  [A] BACKTEST ORIGINAL — ETFs NORMALES")
    print("="*70)

    # Guardar config original
    original_tickers = bt.CONFIG['test_tickers']

    results_normal = {}
    for period in [6, 12, 18]:
        result = bt.run_backtest(months=period, verbose=False)
        if 'error' not in result:
            results_normal[period] = result

    # =============================================
    # TEST 2: APALANCADOS
    # =============================================
    print("\n" + "="*70)
    print("  [B] BACKTEST — ETFs APALANCADOS")
    print("="*70)

    bt.CONFIG['test_tickers'] = leveraged_tickers

    results_leveraged = {}
    for period in [6, 12, 18]:
        result = bt.run_backtest(months=period, verbose=False)
        if 'error' not in result:
            results_leveraged[period] = result

    # Restaurar
    bt.CONFIG['test_tickers'] = original_tickers

    # =============================================
    # COMPARATIVA
    # =============================================
    print(f"\n{'='*70}")
    print(f"  COMPARATIVA FINAL")
    print(f"{'='*70}")

    print(f"\n  {'':12} {'--- NORMALES ---':^30}  {'--- APALANCADOS ---':^30}")
    print(f"  {'Periodo':<12} {'Trades':<7} {'Win%':<6} {'Return':<9} {'Annual':<9} {'PF':<6}  "
          f"{'Trades':<7} {'Win%':<6} {'Return':<9} {'Annual':<9} {'PF':<6}")
    print(f"  {'-'*80}")

    for period in [6, 12, 18]:
        rn = results_normal.get(period, {})
        rl = results_leveraged.get(period, {})

        if rn and rl:
            print(f"  {period}m{' '*9}"
                  f"{rn['total_trades']:<7} {rn['win_rate']:<6.1f} "
                  f"{rn['total_return_pct']:>+7.1f}%  {rn['annualized_return_pct']:>+7.1f}%  "
                  f"{rn['profit_factor']:<6.2f} "
                  f"{rl['total_trades']:<7} {rl['win_rate']:<6.1f} "
                  f"{rl['total_return_pct']:>+7.1f}%  {rl['annualized_return_pct']:>+7.1f}%  "
                  f"{rl['profit_factor']:<6.2f}")

    print()

    # Detalle de fat tails comparado
    print(f"  {'':12} {'--- NORMALES ---':^30}  {'--- APALANCADOS ---':^30}")
    print(f"  {'Periodo':<12} {'MaxDD':<7} {'AvgWin':<9} {'Best':<14}  "
          f"{'MaxDD':<7} {'AvgWin':<9} {'Best':<14}")
    print(f"  {'-'*80}")

    for period in [6, 12, 18]:
        rn = results_normal.get(period, {})
        rl = results_leveraged.get(period, {})

        if rn and rl:
            rn_trades = rn.get('trades', [])
            rl_trades = rl.get('trades', [])

            rn_best = max(rn_trades, key=lambda t: t.pnl_pct) if rn_trades else None
            rl_best = max(rl_trades, key=lambda t: t.pnl_pct) if rl_trades else None

            rn_avg_win = sum(t.pnl_pct for t in rn_trades if t.pnl_euros > 0) / max(1, sum(1 for t in rn_trades if t.pnl_euros > 0))
            rl_avg_win = sum(t.pnl_pct for t in rl_trades if t.pnl_euros > 0) / max(1, sum(1 for t in rl_trades if t.pnl_euros > 0))

            rn_best_str = f"{rn_best.ticker} {rn_best.pnl_pct:+.0f}%" if rn_best else "N/A"
            rl_best_str = f"{rl_best.ticker} {rl_best.pnl_pct:+.0f}%" if rl_best else "N/A"

            print(f"  {period}m{' '*9}"
                  f"{rn['max_drawdown']:<7.1f} {rn_avg_win:>+7.1f}%  {rn_best_str:<14} "
                  f"{rl['max_drawdown']:<7.1f} {rl_avg_win:>+7.1f}%  {rl_best_str:<14}")

    print()

    # Trades apalancados mas destacados
    if 18 in results_leveraged:
        trades_18 = results_leveraged[18].get('trades', [])
        if trades_18:
            sorted_trades = sorted(trades_18, key=lambda t: t.pnl_pct, reverse=True)
            print(f"  TOP 10 TRADES APALANCADOS (18m):")
            for t in sorted_trades[:10]:
                lev_mult = ""
                for normal, lev in LEVERAGED_MAP.items():
                    if t.ticker == lev:
                        lev_mult = f" (reemplaza {normal})"
                        break
                print(f"    {t.ticker:8} {t.pnl_pct:>+7.1f}% | EUR {t.pnl_euros:>+8.0f} | "
                      f"{t.bars_held:3}d | {t.exit_reason}{lev_mult}")
            print()


if __name__ == "__main__":
    run_comparison()
