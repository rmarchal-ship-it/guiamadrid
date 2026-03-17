#!/usr/bin/env python3
"""
Long Swing — Monte Carlo & Walk-Forward Validation Suite.

5 tests independientes para validar que el edge es REAL:
  1. Trade Shuffle MC — ¿depende del orden de trades?
  2. Bootstrap mensual — distribución de outcomes a 5 años
  3. Permutation test — ¿PF es estadísticamente significativo? (p-value)
  4. Walk-Forward — IS (70%) vs OOS (30%), ¿se mantiene el PF?
  5. Robustness — ¿depende de pocos home-runs?

Uso:
  uv run python -m long_swing.validation              # todos
  uv run python -m long_swing.validation --test mc     # solo Monte Carlo (1+2+3)
  uv run python -m long_swing.validation --test wf     # solo Walk-Forward
  uv run python -m long_swing.validation --test rob    # solo Robustness
  uv run python -m long_swing.validation --sims 500    # rápido
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from .config import UNIVERSE, INITIAL, LOOKBACK, FIXED_YEARS, MACRO_FILTER, BENCHMARKS
from .strategy import (
    download_daily, get_macro_context, compute_all_trades,
    select_daily_signals, apply_macro_filter, simulate,
)


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES DE PRESENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def print_percentiles(label, data, unit=''):
    data = np.array(data)
    print(f"  {label}:")
    print(f"     P5  (peor caso):   {np.percentile(data, 5):>10.1f}{unit}")
    print(f"     P25 (pesimista):   {np.percentile(data, 25):>10.1f}{unit}")
    print(f"     P50 (mediana):     {np.percentile(data, 50):>10.1f}{unit}")
    print(f"     P75 (optimista):   {np.percentile(data, 75):>10.1f}{unit}")
    print(f"     P95 (mejor caso):  {np.percentile(data, 95):>10.1f}{unit}")
    print(f"     Media:             {np.mean(data):>10.1f}{unit}")
    print(f"     Std:               {np.std(data):>10.1f}{unit}")


def print_histogram(data, label, bins=20, width=50):
    data = np.array(data)
    counts, edges = np.histogram(data, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    print(f"\n  {label} — Distribucion ({len(data):,} simulaciones)")
    print(f"  {'─' * (width + 25)}")
    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width)
        lo, hi = edges[i], edges[i + 1]
        bar = '#' * bar_len
        print(f"  {lo:>8.1f} - {hi:>8.1f} | {bar} ({count:,})")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: TRADE SHUFFLE MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════

def monte_carlo_trade_shuffle(log, initial, n_sims=5000):
    """
    Reordena PnLs aleatoriamente y recalcula equity curve.
    Si mediana CAGR ≈ real, el edge no depende del orden (timing).
    """
    pnls = np.array([t["pnl"] for t in log])
    entry_dates = [t["entry_ts"] for t in log]
    exit_dates = [t["exit_ts"] for t in log]
    total_days = (max(exit_dates) - min(entry_dates)).days
    years = total_days / 365.25

    results = {'final_equity': [], 'cagr': [], 'max_dd': [], 'profit_factor': []}
    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        shuffled = rng.permutation(pnls)
        equity = initial
        peak = initial
        max_dd = 0.0
        gross_profit = gross_loss = 0.0

        for pnl in shuffled:
            equity += pnl
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        cagr = ((equity / initial) ** (1 / years) - 1) * 100 if years > 0 and equity > 0 else -100
        pf = gross_profit / gross_loss if gross_loss > 0 else 99

        results['final_equity'].append(equity)
        results['cagr'].append(cagr)
        results['max_dd'].append(max_dd)
        results['profit_factor'].append(pf)

    return results, years


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: BOOTSTRAP RETORNOS MENSUALES
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_monthly_returns(log, initial, n_sims=5000, years_to_simulate=5):
    """
    Calcula retornos % mensuales reales, samplea con reemplazo
    para generar distribución de outcomes a N años.
    """
    monthly_pnl = defaultdict(float)
    for t in log:
        if t.get("reason") != "OPEN":
            month_key = pd.Timestamp(t["exit_ts"]).strftime('%Y-%m')
            monthly_pnl[month_key] += t["pnl"]

    sorted_months = sorted(monthly_pnl.keys())
    equity = initial
    monthly_returns = []
    for month in sorted_months:
        pnl = monthly_pnl[month]
        ret_pct = pnl / equity * 100 if equity > 0 else 0
        monthly_returns.append(ret_pct)
        equity += pnl

    monthly_returns = np.array(monthly_returns)
    n_months = int(years_to_simulate * 12)
    rng = np.random.default_rng(123)

    results = {
        'final_equity': [], 'cagr': [], 'max_dd': [],
        'worst_month': [], 'best_month': [], 'pct_negative_months': [],
    }

    for _ in range(n_sims):
        sampled = rng.choice(monthly_returns, size=n_months, replace=True)
        eq = initial
        peak = initial
        max_dd = 0.0
        for ret in sampled:
            eq *= (1 + ret / 100)
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        cagr = ((eq / initial) ** (1 / years_to_simulate) - 1) * 100 if eq > 0 else -100
        results['final_equity'].append(eq)
        results['cagr'].append(cagr)
        results['max_dd'].append(max_dd)
        results['worst_month'].append(sampled.min())
        results['best_month'].append(sampled.max())
        results['pct_negative_months'].append(np.sum(sampled < 0) / len(sampled) * 100)

    return results, monthly_returns


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def permutation_test(log, n_perms=10000):
    """
    H0: las señales NO tienen poder predictivo.
    Shuffle PnL% entre trades. Si PF real está en top 5%, p < 0.05.
    """
    sizes = np.array([t["alloc"] + t["pyr_alloc"] for t in log])
    pnl_pcts = np.array([t["pnl"] / s * 100 if s > 0 else 0 for t, s in zip(log, sizes)])

    # Real metrics
    real_pnls = pnl_pcts * sizes / 100
    real_gp = np.sum(real_pnls[real_pnls > 0])
    real_gl = np.abs(np.sum(real_pnls[real_pnls <= 0]))
    real_pf = real_gp / real_gl if real_gl > 0 else 99
    real_total = np.sum(real_pnls)
    real_wr = np.sum(pnl_pcts > 0) / len(pnl_pcts) * 100

    rng = np.random.default_rng(456)
    perm_pfs, perm_totals, perm_wrs = [], [], []

    for _ in range(n_perms):
        shuffled = rng.permutation(pnl_pcts)
        pnls = shuffled * sizes / 100
        gp = np.sum(pnls[pnls > 0])
        gl = np.abs(np.sum(pnls[pnls <= 0]))
        pf = gp / gl if gl > 0 else 99
        perm_pfs.append(pf)
        perm_totals.append(np.sum(pnls))
        perm_wrs.append(np.sum(shuffled > 0) / len(shuffled) * 100)

    perm_pfs = np.array(perm_pfs)
    perm_totals = np.array(perm_totals)

    return {
        'real_pf': real_pf, 'real_total_pnl': real_total, 'real_win_rate': real_wr,
        'p_value_pf': np.mean(perm_pfs >= real_pf),
        'p_value_total_pnl': np.mean(perm_totals >= real_total),
        'p_value_win_rate': np.mean(np.array(perm_wrs) >= real_wr),
        'perm_pf_median': np.median(perm_pfs),
        'perm_pf_p95': np.percentile(perm_pfs, 95),
        'perm_total_median': np.median(perm_totals),
        'perm_total_p95': np.percentile(perm_totals, 95),
        'perm_pfs': perm_pfs,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: WALK-FORWARD (70/30)
# ═══════════════════════════════════════════════════════════════════════════

def test_walk_forward(filtered_signals, all_data):
    """
    Split señales al 70% temporal. Ejecuta simulate() en cada mitad.
    Compara PF IS vs OOS.
    """
    n = len(filtered_signals)
    split_idx = int(n * 0.7)
    is_signals = filtered_signals.iloc[:split_idx].reset_index(drop=True)
    oos_signals = filtered_signals.iloc[split_idx:].reset_index(drop=True)

    is_start = is_signals["entry_ts"].min()
    is_end = is_signals["exit_ts"].max()
    oos_start = oos_signals["entry_ts"].min()
    oos_end = oos_signals["exit_ts"].max()

    is_years = (is_end - is_start).days / 365.25
    oos_years = (oos_end - oos_start).days / 365.25

    _, _, m_is = simulate(is_signals, all_data, initial=INITIAL, fixed_years=is_years)
    _, _, m_oos = simulate(oos_signals, all_data, initial=INITIAL, fixed_years=oos_years)

    return {
        'is': m_is, 'oos': m_oos,
        'is_signals': len(is_signals), 'oos_signals': len(oos_signals),
        'is_period': f"{is_start.date()} → {is_end.date()} ({is_years:.1f}y)",
        'oos_period': f"{oos_start.date()} → {oos_end.date()} ({oos_years:.1f}y)",
        'pf_ratio': m_oos['pf'] / m_is['pf'] if m_is['pf'] > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: ROBUSTNESS (quitar top N trades)
# ═══════════════════════════════════════════════════════════════════════════

def test_robustness(log):
    """
    Ordena trades por PnL y recalcula PF quitando los top N.
    Detecta dependencia de fat-tail outliers.
    """
    closed = [t for t in log if t.get("reason") != "OPEN"]
    sorted_trades = sorted(closed, key=lambda t: t["pnl"], reverse=True)
    total_pnl = sum(t["pnl"] for t in closed)

    rows = []
    for n in [0, 3, 5, 10, 15, 20, 30]:
        remaining = sorted_trades[n:]
        if not remaining:
            break
        winners = [t for t in remaining if t["pnl"] > 0]
        losers = [t for t in remaining if t["pnl"] <= 0]
        gp = sum(t["pnl"] for t in winners) if winners else 0
        gl = abs(sum(t["pnl"] for t in losers)) if losers else 0.01
        pf = gp / gl
        wr = len(winners) / len(remaining) * 100
        pnl = sum(t["pnl"] for t in remaining)
        rows.append({
            'removed': n, 'trades': len(remaining), 'pnl': pnl,
            'wr': wr, 'pf': pf,
        })

    top10 = sorted_trades[:10]
    return rows, top10, total_pnl


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Run full signal pipeline once, return data for all tests."""
    print("  Descargando datos...")
    all_data = download_daily(UNIVERSE)
    print(f"\n  Tickers loaded: {len(all_data)} / {len(UNIVERSE)}")

    spy_ctx, vix_map = get_macro_context(all_data)
    trades_df = compute_all_trades(all_data, spy_ctx, vix_map)
    signals = select_daily_signals(trades_df)
    filtered = apply_macro_filter(signals, spy_ctx)

    print(f"  Signals: {len(signals)} total -> {len(filtered)} after macro filter")
    if filtered.empty:
        print("  No signals — aborting.")
        sys.exit(1)

    log, still_open, metrics = simulate(filtered, all_data)
    return all_data, spy_ctx, filtered, log, metrics


def main():
    parser = argparse.ArgumentParser(description='Long Swing — Validation Suite')
    parser.add_argument('--test', choices=['all', 'mc', 'wf', 'rob'],
                        default='all', help='Test(s) a ejecutar')
    parser.add_argument('--sims', type=int, default=5000,
                        help='Simulaciones Monte Carlo (default: 5000)')
    parser.add_argument('--bootstrap-years', type=int, default=5,
                        help='Anos a simular en bootstrap (default: 5)')
    args = parser.parse_args()

    n_sims = args.sims
    bs_years = args.bootstrap_years
    run_mc = args.test in ('all', 'mc')
    run_wf = args.test in ('all', 'wf')
    run_rob = args.test in ('all', 'rob')

    print(f"""
{'='*70}
  LONG SWING — VALIDATION SUITE
{'='*70}
  Lookback: {LOOKBACK}d (~{FIXED_YEARS:.1f}y)
  Capital: {INITIAL:,.0f}EUR
  Macro: {MACRO_FILTER}
  Sims: {n_sims:,}
{'='*70}
""")

    # ── Pipeline ──
    all_data, spy_ctx, filtered, log, m = run_pipeline()

    closed_log = [t for t in log if t.get("reason") != "OPEN"]

    print(f"""
  BASELINE: {m['trades']} trades, CAGR {m['cagr']:+.1f}%, MaxDD {m['dd']:.1f}%,
            PF {m['pf']:.2f}, WR {m['wr']:.1f}%, Pyramids {m['n_pyr']}
            Final: {m['final']:,.0f}EUR (from {INITIAL:,.0f}EUR)
""")

    # ==================================================================
    # TEST 1: TRADE SHUFFLE
    # ==================================================================
    if run_mc:
        print(f"{'='*70}")
        print(f"  TEST 1: TRADE SHUFFLE MONTE CARLO ({n_sims:,} sims)")
        print(f"{'='*70}")
        print(f"  Pregunta: El resultado depende del ORDEN de los trades?")
        print(f"  Si mediana ~= real, el edge es robusto al timing.\n")

        mc, mc_years = monte_carlo_trade_shuffle(closed_log, INITIAL, n_sims)

        print_percentiles("CAGR (%)", mc['cagr'], '%')
        print()
        print_percentiles("MaxDD (%)", mc['max_dd'], '%')
        print()
        print_percentiles("Profit Factor", mc['profit_factor'])

        real_cagr = m['cagr']
        median_cagr = np.median(mc['cagr'])
        print(f"""
  --- COMPARACION CON REAL ---
  CAGR real:     {real_cagr:+.1f}%
  CAGR mediana:  {median_cagr:+.1f}%
  Delta:         {real_cagr - median_cagr:+.1f}pp

  Prob CAGR > 0:    {np.mean(np.array(mc['cagr']) > 0) * 100:.1f}%
  Prob CAGR > 15%:  {np.mean(np.array(mc['cagr']) > 15) * 100:.1f}%
  Prob CAGR > 25%:  {np.mean(np.array(mc['cagr']) > 25) * 100:.1f}%
  Prob MaxDD > 30%: {np.mean(np.array(mc['max_dd']) > 30) * 100:.1f}%
  Prob MaxDD > 50%: {np.mean(np.array(mc['max_dd']) > 50) * 100:.1f}%
""")
        print_histogram(mc['cagr'], 'CAGR (%)')
        print_histogram(mc['max_dd'], 'MaxDD (%)')

        # ==================================================================
        # TEST 2: BOOTSTRAP
        # ==================================================================
        print(f"\n{'='*70}")
        print(f"  TEST 2: BOOTSTRAP RETORNOS MENSUALES ({n_sims:,} sims, {bs_years}y)")
        print(f"{'='*70}")
        print(f"  Pregunta: Distribucion de outcomes a {bs_years} anos")
        print(f"  si los meses futuros se parecen a los pasados.\n")

        bs, monthly_rets = bootstrap_monthly_returns(closed_log, INITIAL, n_sims, bs_years)

        print(f"  Retornos mensuales reales ({len(monthly_rets)} meses):")
        print(f"     Media:    {np.mean(monthly_rets):+.2f}%")
        print(f"     Mediana:  {np.median(monthly_rets):+.2f}%")
        print(f"     Std:      {np.std(monthly_rets):.2f}%")
        print(f"     Min:      {np.min(monthly_rets):+.2f}%")
        print(f"     Max:      {np.max(monthly_rets):+.2f}%")
        print(f"     % negativos: {np.sum(monthly_rets < 0) / len(monthly_rets) * 100:.1f}%")
        print()

        print_percentiles(f"CAGR a {bs_years}y (%)", bs['cagr'], '%')
        print()
        print_percentiles(f"MaxDD a {bs_years}y (%)", bs['max_dd'], '%')
        print()
        print_percentiles(f"Final Equity a {bs_years}y (EUR)", bs['final_equity'])

        prob_loss = np.mean(np.array(bs['final_equity']) < INITIAL) * 100
        prob_double = np.mean(np.array(bs['final_equity']) > INITIAL * 2) * 100
        prob_10x = np.mean(np.array(bs['final_equity']) > INITIAL * 10) * 100

        print(f"""
  --- PROBABILIDADES A {bs_years} ANOS ---
  Prob perder dinero:     {prob_loss:.1f}%
  Prob duplicar capital:  {prob_double:.1f}%
  Prob 10x capital:       {prob_10x:.1f}%
  Prob CAGR > 15%:        {np.mean(np.array(bs['cagr']) > 15) * 100:.1f}%
  Prob CAGR > 25%:        {np.mean(np.array(bs['cagr']) > 25) * 100:.1f}%
""")
        print_histogram(bs['cagr'], f'CAGR a {bs_years}y (%)')

        # ==================================================================
        # TEST 3: PERMUTATION TEST
        # ==================================================================
        print(f"\n{'='*70}")
        print(f"  TEST 3: PERMUTATION TEST ({n_sims:,} permutaciones)")
        print(f"{'='*70}")
        print(f"  H0: Las senales NO tienen poder predictivo (PF = azar)")
        print(f"  Si p-value < 0.05, rechazamos H0 -> edge REAL.\n")

        perm = permutation_test(closed_log, n_sims)

        sig_pf = "SIGNIFICATIVO" if perm['p_value_pf'] < 0.05 else "NO significativo"
        sig_pnl = "SIGNIFICATIVO" if perm['p_value_total_pnl'] < 0.05 else "NO significativo"
        sig_wr = "SIGNIFICATIVO" if perm['p_value_win_rate'] < 0.05 else "NO significativo"

        print(f"  Profit Factor real:    {perm['real_pf']:.2f}")
        print(f"  PF median shuffled:    {perm['perm_pf_median']:.2f}")
        print(f"  PF P95 shuffled:       {perm['perm_pf_p95']:.2f}")
        print(f"  p-value PF:            {perm['p_value_pf']:.4f}  {sig_pf}")
        print()
        print(f"  Total PnL real:        EUR {perm['real_total_pnl']:+,.0f}")
        print(f"  PnL median shuffled:   EUR {perm['perm_total_median']:+,.0f}")
        print(f"  p-value PnL:           {perm['p_value_total_pnl']:.4f}  {sig_pnl}")
        print()
        print(f"  Win rate real:         {perm['real_win_rate']:.1f}%")
        print(f"  p-value Win Rate:      {perm['p_value_win_rate']:.4f}  {sig_wr}")

        print_histogram(perm['perm_pfs'], 'PF bajo H0 (shuffled)')

    # ==================================================================
    # TEST 4: WALK-FORWARD
    # ==================================================================
    if run_wf:
        print(f"\n{'='*70}")
        print(f"  TEST 4: WALK-FORWARD (70% IS / 30% OOS)")
        print(f"{'='*70}")
        print(f"  Mismos parametros en ambos periodos (sin re-optimizacion).\n")

        wf = test_walk_forward(filtered, all_data)

        print(f"  IN-SAMPLE:       {wf['is_period']}  ({wf['is_signals']} signals)")
        print(f"  OUT-OF-SAMPLE:   {wf['oos_period']}  ({wf['oos_signals']} signals)")

        is_m, oos_m = wf['is'], wf['oos']
        print(f"""
  {'Metrica':<20s} {'IN-SAMPLE':>12s} {'OOS':>12s} {'Ratio OOS/IS':>14s}
  {'-'*60}
  {'Trades':<20s} {is_m['trades']:>12d} {oos_m['trades']:>12d}
  {'Win Rate':<20s} {is_m['wr']:>11.1f}% {oos_m['wr']:>11.1f}%
  {'Profit Factor':<20s} {is_m['pf']:>12.2f} {oos_m['pf']:>12.2f} {wf['pf_ratio']:>13.1%}
  {'CAGR':<20s} {is_m['cagr']:>+11.1f}% {oos_m['cagr']:>+11.1f}%
  {'MaxDD':<20s} {is_m['dd']:>11.1f}% {oos_m['dd']:>11.1f}%
  {'Final':<20s} {is_m['final']:>11,.0f} {oos_m['final']:>11,.0f}
  {'Pyramids':<20s} {is_m['n_pyr']:>12d} {oos_m['n_pyr']:>12d}
""")

        pf_ratio = wf['pf_ratio']
        if pf_ratio >= 0.8:
            verdict = "PF OOS >= 80% del IS -> Estrategia ROBUSTA"
        elif pf_ratio >= 0.5:
            verdict = "PF OOS 50-80% del IS -> Overfitting MODERADO"
        else:
            verdict = "PF OOS < 50% del IS -> Overfitting SEVERO"
        print(f"  VEREDICTO: {verdict}")

    # ==================================================================
    # TEST 5: ROBUSTNESS
    # ==================================================================
    if run_rob:
        print(f"\n{'='*70}")
        print(f"  TEST 5: ROBUSTNESS (quitar top N trades)")
        print(f"{'='*70}")
        print(f"  Depende de pocos home-runs?\n")

        rows, top10, total_pnl = test_robustness(log)

        print(f"  TOP 10 TRADES (mayores ganadores):")
        print(f"  {'#':<4s} {'Sym':>8s} {'PnL EUR':>10s} {'Ret%':>8s} {'Reason':>10s} {'Pyr?':>5s}")
        print(f"  {'-'*50}")
        top10_pnl = 0
        for i, t in enumerate(top10, 1):
            top10_pnl += t["pnl"]
            pyr = "Y" if t["pyramided"] else "-"
            print(f"  {i:<4d} {t['symbol']:>8s} {t['pnl']:>+10,.0f} "
                  f"{t['eff_ret']*100:>+7.1f}% {str(t['reason'])[:10]:>10s} {pyr:>5s}")
        print(f"\n  Top 10 = EUR {top10_pnl:+,.0f} ({top10_pnl/total_pnl*100:.1f}% del PnL total)")

        orig_pf = rows[0]['pf'] if rows else 0
        print(f"\n  {'Quitar top N':<15s} {'Trades':>8s} {'PnL EUR':>12s} {'WR':>7s} {'PF':>7s} {'PF %orig':>10s}")
        print(f"  {'-'*60}")
        for r in rows:
            label = "Original" if r['removed'] == 0 else f"Sin top {r['removed']}"
            pf_pct = r['pf'] / orig_pf * 100 if orig_pf > 0 else 0
            print(f"  {label:<15s} {r['trades']:>8d} {r['pnl']:>+12,.0f} "
                  f"{r['wr']:>6.1f}% {r['pf']:>7.2f} {pf_pct:>9.0f}%")

    # ==================================================================
    # RESUMEN EJECUTIVO
    # ==================================================================
    print(f"""

{'='*70}
  RESUMEN EJECUTIVO — LONG SWING VALIDATION
{'='*70}

  Baseline: {m['trades']} trades, CAGR {m['cagr']:+.1f}%, PF {m['pf']:.2f}, DD {m['dd']:.1f}%
""")

    if run_mc:
        median_cagr = np.median(mc['cagr'])
        print(f"  T1 Trade Shuffle:  CAGR mediana {median_cagr:+.1f}% (real {m['cagr']:+.1f}%)"
              f" -> {'ROBUSTO' if abs(m['cagr'] - median_cagr) < 15 else 'FRAGIL'}")
        print(f"  T2 Bootstrap {bs_years}y:   CAGR mediana {np.median(bs['cagr']):+.1f}%, "
              f"P5 {np.percentile(bs['cagr'], 5):+.1f}%, "
              f"Prob perder {prob_loss:.1f}%")
        print(f"  T3 Permutation:    p-value PF={perm['p_value_pf']:.4f}, "
              f"PnL={perm['p_value_total_pnl']:.4f}"
              f" -> {'EDGE REAL' if perm['p_value_pf'] < 0.05 else 'NO significativo'}")

    if run_wf:
        print(f"  T4 Walk-Forward:   PF IS={is_m['pf']:.2f} -> OOS={oos_m['pf']:.2f} "
              f"(ratio {wf['pf_ratio']:.1%})"
              f" -> {verdict}")

    if run_rob:
        if len(rows) > 2:
            pf_sin5 = rows[2]['pf']  # sin top 5
            print(f"  T5 Robustness:     PF sin top5={pf_sin5:.2f} "
                  f"({pf_sin5/orig_pf*100:.0f}% del original)"
                  f" -> {'ROBUSTO' if pf_sin5/orig_pf > 0.6 else 'DEPENDIENTE de outliers'}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
