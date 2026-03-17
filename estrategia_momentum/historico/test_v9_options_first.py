#!/usr/bin/env python3
"""
TEST v9 — Options-First Strategy
Fecha: 26 Feb 2026

Hipotesis: Si el edge viene de fat tails con low win rate, las opciones CALL
son el vehiculo perfecto (riesgo limitado a prima, upside ilimitado).

Cambio vs v8:
  - max_option_positions: 2 → 10 (sin limite, todas las slots pueden ser opciones)
  - Opciones: mismas condiciones v8 (DTE 120, ITM 5%, IVR<40, cierre 45 DTE)
  - Si ticker elegible + IVR<40 → CALL. Si no → accion.
  - option_position_pct: probamos 0.10 (10% equity) y 0.14 (14% equity)

Comparativa: stock-only vs v8 (2 opts) vs v9 (options-first)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_experimental import (
    CONFIG, BASE_TICKERS, run_backtest, print_comparison
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test v9: Options-First')
    parser.add_argument('--months', type=int, default=60, help='Meses de historico')
    parser.add_argument('--verbose', action='store_true', help='Detalle de trades')
    args = parser.parse_args()

    months = args.months
    v = args.verbose

    print(f"""
======================================================================
  TEST v9 — OPTIONS-FIRST STRATEGY — {months} MESES
======================================================================
  Hipotesis: opciones CALL siempre que el ticker sea elegible + IVR<40.
  Comparamos: stock-only vs v8 (max 2 opts) vs v9 (sin limite opts)
  Option sizing: {CONFIG['option_position_pct']*100:.0f}% equity por opcion
======================================================================
    """)

    results = []

    # 1. BASELINE: stock-only (sin opciones)
    print("\n>>> TEST 1/3: STOCK-ONLY (baseline)")
    CONFIG['max_option_positions'] = 0  # forzar 0 opciones
    r_stock = run_backtest(months, BASE_TICKERS, "STOCK-ONLY",
                           use_leverage_scaling=False, use_options=False, verbose=v)
    if 'error' not in r_stock:
        results.append(r_stock)

    # 2. v8 ACTUAL: max 2 opciones
    print("\n>>> TEST 2/3: v8 (max 2 opciones)")
    CONFIG['max_option_positions'] = 2
    r_v8 = run_backtest(months, BASE_TICKERS, "v8 (max 2 opts)",
                        use_leverage_scaling=False, use_options=True, verbose=v)
    if 'error' not in r_v8:
        results.append(r_v8)

    # 3. v9 OPTIONS-FIRST: sin limite de opciones (max = max_positions)
    print("\n>>> TEST 3/3: v9 OPTIONS-FIRST (sin limite)")
    CONFIG['max_option_positions'] = CONFIG['max_positions']  # 10
    r_v9 = run_backtest(months, BASE_TICKERS, "v9 OPTIONS-FIRST",
                        use_leverage_scaling=False, use_options=True, verbose=v)
    if 'error' not in r_v9:
        results.append(r_v9)

    # Comparativa
    if len(results) >= 2:
        print_comparison(results)

    # Resumen adicional: desglose stocks vs opciones
    print(f"""
{'='*70}
  DESGLOSE DETALLADO
{'='*70}""")
    for r in results:
        opt_trades = r.get('all_option_trades', [])
        stk_trades = r.get('all_trades', [])
        opt_pnl = sum(t.pnl_euros for t in opt_trades) if opt_trades else 0
        stk_pnl = sum(t.pnl_euros for t in stk_trades) if stk_trades else 0
        opt_winners = sum(1 for t in opt_trades if t.pnl_euros > 0) if opt_trades else 0
        stk_winners = sum(1 for t in stk_trades if t.pnl_euros > 0) if stk_trades else 0
        opt_wr = (opt_winners / len(opt_trades) * 100) if opt_trades else 0
        stk_wr = (stk_winners / len(stk_trades) * 100) if stk_trades else 0

        # Avg option premium paid
        avg_premium = 0
        if opt_trades:
            avg_premium = sum(t.position_euros for t in opt_trades) / len(opt_trades)

        print(f"\n  {r['label']}:")
        print(f"    Stocks:   {len(stk_trades)} trades, PnL EUR {stk_pnl:+,.0f}, Win% {stk_wr:.1f}%")
        print(f"    Opciones: {len(opt_trades)} trades, PnL EUR {opt_pnl:+,.0f}, Win% {opt_wr:.1f}%")
        if opt_trades:
            print(f"    Avg premium pagada: EUR {avg_premium:,.0f}")
            # Home runs
            hr_100 = sum(1 for t in opt_trades if t.pnl_pct >= 100)
            hr_200 = sum(1 for t in opt_trades if t.pnl_pct >= 200)
            hr_500 = sum(1 for t in opt_trades if t.pnl_pct >= 500)
            print(f"    Home runs: >=100%: {hr_100}, >=200%: {hr_200}, >=500%: {hr_500}")
            # Worst option loss
            if opt_trades:
                worst_opt = min(opt_trades, key=lambda t: t.pnl_pct)
                print(f"    Peor opcion: {worst_opt.ticker} {worst_opt.pnl_pct:+.1f}% (EUR {worst_opt.pnl_euros:+,.0f})")

    # Restaurar CONFIG
    CONFIG['max_option_positions'] = 2


if __name__ == "__main__":
    main()
