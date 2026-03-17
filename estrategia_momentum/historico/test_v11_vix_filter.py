#!/usr/bin/env python3
"""
TEST v11 — VIX Filter + Gold Hedge
Fecha: 27 Feb 2026
Mejora E del informe Cowork cap. 12

Hipotesis: No abrir posiciones nuevas cuando VIX > umbral.
Cuando VIX esta alto (>25-30), los breakouts tienden a fallar.
Anyadir este filtro sobre v10 (v8 + Gold 30%) deberia reducir MaxDD
sin sacrificar mucho CAGR.

Grid test: VIX threshold 20, 25, 30, 35 (+ sin filtro como baseline)
Base: v8 (2 opts, 14%) + Gold 30% = v10 = configuracion ganadora actual.

Approach:
  1. Monkey-patch build_macro_filter para anydir condicion VIX
  2. Ejecuta backtest con cada umbral VIX
  3. Aplica gold overlay post-hoc (30% GLD)
  4. Compara eficiencia (Annual / MaxDD)

NOTA: No se modifica backtest_experimental.py. El VIX filter se inyecta
via monkey-patch de build_macro_filter, que es llamado una vez por
run_backtest() en linea 621.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_experimental as be
from backtest_experimental import CONFIG, BASE_TICKERS, run_backtest, download_data
from test_v10_gold_hedge import simulate_gold_overlay
import pandas as pd
import numpy as np


# Save original function BEFORE any patching
_original_build_macro = be.build_macro_filter


def make_vix_macro_filter(vix_data, vix_threshold):
    """
    Creates a patched build_macro_filter that also checks VIX level.
    When VIX > threshold on a given date, macro_bullish = False (no new entries).

    El filtro VIX es ADITIVO al filtro macro (SPY>SMA50):
    - Si SPY < SMA50 → ya no opera (macro bear)
    - Si SPY > SMA50 pero VIX > threshold → tampoco opera (VIX bear)
    - Si SPY > SMA50 y VIX <= threshold → opera normalmente
    """
    def wrapped(all_data):
        # Get standard SPY>SMA50 macro filter
        macro = _original_build_macro(all_data)

        # Overlay VIX filter
        vix_close = vix_data['Close']
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.squeeze()

        blocked_count = 0
        total_bull = sum(1 for v in macro.values() if v)

        for date in list(macro.keys()):
            if date in vix_close.index:
                vix_val = vix_close.loc[date]
                if isinstance(vix_val, pd.Series):
                    vix_val = vix_val.iloc[0]
                if not pd.isna(vix_val) and vix_val > vix_threshold:
                    if macro[date]:  # Only count if it was True (bull) and we're blocking
                        blocked_count += 1
                    macro[date] = False

        print(f"    VIX filter (>{vix_threshold}): blocked {blocked_count}/{total_bull} bull days "
              f"({blocked_count/max(total_bull,1)*100:.1f}%)")
        return macro
    return wrapped


def restore_macro_filter():
    """Restore original build_macro_filter (no VIX)"""
    be.build_macro_filter = _original_build_macro


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test v11: VIX Filter + Gold Hedge')
    parser.add_argument('--months', type=int, default=60)
    parser.add_argument('--gold-pct', type=float, default=0.30)
    parser.add_argument('--vix-only', type=int, default=None,
                        help='Test single VIX threshold (skip grid)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    months = args.months
    gold_pct = args.gold_pct

    if args.vix_only:
        vix_thresholds = [None, args.vix_only]
    else:
        vix_thresholds = [None, 20, 25, 30, 35]  # None = v10 baseline

    print(f"""
======================================================================
  TEST v11 — VIX FILTER + GOLD HEDGE — {months}m
======================================================================
  Base: v8 (2 opts, 14%) + Gold {gold_pct*100:.0f}% = v10
  Grid: VIX thresholds = {[t for t in vix_thresholds if t is not None]}
  Baseline: sin filtro VIX (v10 actual)

  Mejora E del informe Cowork cap. 12

  Mecanismo VIX filter:
    - Descarga ^VIX historico
    - Cuando VIX > threshold → macro_bullish = False → no abre posiciones
    - ADITIVO al filtro SPY>SMA50 existente
    - Posiciones ya abiertas NO se cierran (solo bloquea nuevas entradas)
======================================================================
""")

    # Download VIX
    print("  Descargando ^VIX...")
    vix_data = download_data('^VIX', months + 6)
    if vix_data is None:
        print("  ERROR: No se pudo descargar ^VIX. Abortando.")
        return
    print(f"  VIX: {len(vix_data)} dias, "
          f"{vix_data.index[0].strftime('%Y-%m-%d')} a "
          f"{vix_data.index[-1].strftime('%Y-%m-%d')}")

    vix_close = vix_data['Close']
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.squeeze()
    print(f"  VIX actual: {vix_close.iloc[-1]:.1f} | "
          f"Media: {vix_close.mean():.1f} | Max: {vix_close.max():.1f}")

    # VIX distribution
    print(f"\n  Distribucion VIX ({months}m):")
    for lvl in [20, 25, 30, 35]:
        pct_above = (vix_close > lvl).mean() * 100
        days_above = (vix_close > lvl).sum()
        print(f"    VIX > {lvl}: {pct_above:5.1f}% ({days_above} dias)")

    # Download GLD
    print("\n  Descargando GLD...")
    gld_data = download_data('GLD', months + 6)
    if gld_data is None:
        print("  ERROR: No se pudo descargar GLD. Abortando.")
        return
    print(f"  GLD: {len(gld_data)} dias")

    # v8 config (explicit — never rely on defaults)
    CONFIG['max_option_positions'] = 2
    CONFIG['option_position_pct'] = 0.14

    results_table = []

    for vix_th in vix_thresholds:
        label = f"v10+VIX<{vix_th}" if vix_th else "v10 baseline"

        print(f"\n{'='*70}")
        print(f"  >> {label}")
        print(f"{'='*70}")

        # Patch or restore macro filter
        if vix_th is not None:
            be.build_macro_filter = make_vix_macro_filter(vix_data, vix_th)
        else:
            restore_macro_filter()

        r = run_backtest(months, BASE_TICKERS, label,
                        use_leverage_scaling=False, use_options=True,
                        verbose=args.verbose)

        if 'error' in r:
            print(f"  ERROR en {label}")
            continue

        # Apply gold overlay
        gold = simulate_gold_overlay(r, gld_data, gold_pct)
        if gold is None:
            print(f"  ERROR gold overlay en {label}")
            continue

        eff_raw = r['annualized_return_pct'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0
        eff_gold = gold['ann_gold'] / gold['maxdd_gold'] if gold['maxdd_gold'] > 0 else 0

        results_table.append({
            'vix_th': vix_th,
            'label': label,
            'trades': r['total_trades'],
            'stock_trades': r['stock_trades'],
            'option_trades': r['option_trades'],
            'win_rate': r['win_rate'],
            'pf': r['profit_factor'],
            'return_raw': r['total_return_pct'],
            'annual_raw': r['annualized_return_pct'],
            'maxdd_raw': r['max_drawdown'],
            'eff_raw': eff_raw,
            'return_gold': gold['return_gold'],
            'annual_gold': gold['ann_gold'],
            'maxdd_gold': gold['maxdd_gold'],
            'eff_gold': eff_gold,
            'gold_pnl': gold['gold_total_pnl'],
            'avg_gold_alloc': gold['avg_gold_alloc_pct'],
        })

    # Restore original always
    restore_macro_filter()

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print(f"""
{'='*130}
  GRID RESULTS — VIX FILTER + GOLD 30% — {months}m
{'='*130}

  {'Config':<20} {'Trades':>7} {'Stk':>5} {'Opt':>5} {'WR%':>6} {'PF':>6} | {'Raw Ann%':>9} {'Raw DD%':>8} {'Raw Ef':>7} | {'Gold Ann%':>10} {'Gold DD%':>9} {'Gold Ef':>8} {'AvgGold%':>9}
  {'-'*125}""")

    for r in results_table:
        print(f"  {r['label']:<20} {r['trades']:>7} {r['stock_trades']:>5} {r['option_trades']:>5} "
              f"{r['win_rate']:>5.1f}% {r['pf']:>5.2f} | "
              f"{r['annual_raw']:>+8.1f}% {r['maxdd_raw']:>7.1f}% {r['eff_raw']:>6.2f} | "
              f"{r['annual_gold']:>+9.1f}% {r['maxdd_gold']:>8.1f}% {r['eff_gold']:>7.2f} "
              f"{r['avg_gold_alloc']:>8.1f}%")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    if len(results_table) >= 2:
        baseline = results_table[0]  # v10 baseline (no VIX filter)
        best = max(results_table, key=lambda x: x['eff_gold'])

        print(f"""
{'='*80}
  ANALISIS
{'='*80}

  Baseline (v10 sin VIX filter):
    Annual (gold):  {baseline['annual_gold']:+.1f}%
    MaxDD (gold):   {baseline['maxdd_gold']:.1f}%
    Eficiencia:     {baseline['eff_gold']:.2f}
    Trades:         {baseline['trades']}

  Mejor configuracion: {best['label']}
    Annual (gold):  {best['annual_gold']:+.1f}%
    MaxDD (gold):   {best['maxdd_gold']:.1f}%
    Eficiencia:     {best['eff_gold']:.2f}
    Trades:         {best['trades']}

  Deltas vs baseline:
    Annual:    {best['annual_gold'] - baseline['annual_gold']:+.1f}pp
    MaxDD:     {best['maxdd_gold'] - baseline['maxdd_gold']:+.1f}pp
    Eficiencia: {best['eff_gold'] - baseline['eff_gold']:+.2f}
    Trades:    {best['trades'] - baseline['trades']:+d}""")

        # Trade-off analysis for each threshold
        print(f"""
  Trade-offs por umbral VIX:
  {'Umbral':<10} {'Delta Trades':>14} {'Delta Ann':>11} {'Delta DD':>10} {'Delta Efic':>12} {'Veredicto':>12}
  {'-'*70}""")

        for r in results_table[1:]:  # skip baseline
            d_trades = r['trades'] - baseline['trades']
            d_ann = r['annual_gold'] - baseline['annual_gold']
            d_dd = r['maxdd_gold'] - baseline['maxdd_gold']
            d_eff = r['eff_gold'] - baseline['eff_gold']

            # Simple verdict
            if d_eff > 0.05 and d_dd < -1.0:
                verdict = "MEJOR ✓"
            elif d_eff > 0 and d_dd < 0:
                verdict = "Marginal"
            elif abs(d_eff) < 0.03:
                verdict = "Neutral"
            else:
                verdict = "PEOR ✗"

            print(f"  VIX<{r['vix_th']:<6} {d_trades:>+14d} {d_ann:>+10.1f}pp {d_dd:>+9.1f}pp "
                  f"{d_eff:>+11.2f} {verdict:>12}")

        # Conclusion
        if best['vix_th'] is not None:
            print(f"""
  CONCLUSION:
    VIX filter con umbral {best['vix_th']} mejora la eficiencia de {baseline['eff_gold']:.2f} a {best['eff_gold']:.2f}
    ({best['eff_gold']-baseline['eff_gold']:+.2f}).
    Coste: {best['trades']-baseline['trades']:+d} trades, {best['annual_gold']-baseline['annual_gold']:+.1f}pp annual.
    Beneficio: {best['maxdd_gold']-baseline['maxdd_gold']:+.1f}pp MaxDD.

    v11 = v10 + VIX<{best['vix_th']} = v8 + Gold 30% + VIX<{best['vix_th']}
""")
        else:
            print(f"""
  CONCLUSION:
    Ningun umbral VIX mejora significativamente la eficiencia.
    El filtro SPY>SMA50 existente ya captura la mayor parte del beneficio.
    v10 (v8 + Gold 30%) sigue siendo la configuracion ganadora.
""")


if __name__ == "__main__":
    main()
