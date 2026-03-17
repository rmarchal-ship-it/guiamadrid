#!/usr/bin/env python3
"""
MONTE CARLO VALIDATION — v12 (Momentum Breakout + EU Options)

Fecha: 27 Feb 2026
Objetivo: Validar que el edge de v12 es REAL y no artefacto de overfitting.

Tres tests independientes:
  1. TRADE SHUFFLE (Monte Carlo): Reordenar aleatoriamente los trades N veces.
     Si el CAGR/PF mediano se mantiene ~igual, el edge no depende del orden
     (no es solo un compounding lucky). Si cae, depende de timing.

  2. BOOTSTRAP de retornos mensuales: Sampling con reemplazo de los retornos
     mensuales reales para generar K curvas. Mide la DISTRIBUCION de outcomes
     posibles: mediana CAGR, peor caso (P5), mejor caso (P95), prob de perder.

  3. PERMUTATION TEST (null hypothesis): Shuffle aleatorio de la relacion
     senal→retorno (romper causalidad). Si el PF real esta en el top 5% de
     los PF shuffled, rechazamos H0 (el edge es significativo, no azar).
     p-value < 0.05 = edge real.

Uso:
  python3 backtest_v12_montecarlo.py --months 60
  python3 backtest_v12_montecarlo.py --months 60 --sims 10000
  python3 backtest_v12_montecarlo.py --months 36
"""

import sys, os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_experimental import (
    CONFIG, BASE_TICKERS, OPTIONS_ELIGIBLE,
    download_data,
)
from backtest_v12_eu_options import (
    OPTIONS_ELIGIBLE_EU, OPTIONS_ALL, EU_SPREAD_PCT, US_SPREAD_PCT,
    run_backtest_eu,
)


# =============================================================================
# 1. TRADE SHUFFLE MONTE CARLO
# =============================================================================

def monte_carlo_trade_shuffle(trades, initial_capital, n_sims=5000):
    """
    Reordena aleatoriamente la secuencia de trades y recalcula equity curve.

    Cada trade tiene un pnl_euros fijo. Lo que cambia es el ORDEN en que se
    aplican al capital. Si el resultado depende fuertemente del orden
    (e.g., un home run temprano que compounds), el edge es fragil.

    Returns dict con distribucion de CAGR, MaxDD, final equity.
    """
    pnls = np.array([t.pnl_euros for t in trades])
    n_trades = len(pnls)

    # Calcular duracion total en anos (para annualizar)
    entry_dates = [t.entry_date for t in trades if t.entry_date]
    exit_dates = [t.exit_date for t in trades if t.exit_date]
    total_days = (max(exit_dates) - min(entry_dates)).days
    years = total_days / 365.25

    results = {
        'final_equity': [],
        'cagr': [],
        'max_dd': [],
        'profit_factor': [],
    }

    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        # Shuffle PnLs
        shuffled = rng.permutation(pnls)

        # Simulate equity curve
        equity = initial_capital
        peak = initial_capital
        max_dd = 0.0

        gross_profit = 0.0
        gross_loss = 0.0

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

        cagr = ((equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and equity > 0 else -100
        pf = gross_profit / gross_loss if gross_loss > 0 else 99

        results['final_equity'].append(equity)
        results['cagr'].append(cagr)
        results['max_dd'].append(max_dd)
        results['profit_factor'].append(pf)

    return results


# =============================================================================
# 2. BOOTSTRAP DE RETORNOS MENSUALES
# =============================================================================

def bootstrap_monthly_returns(trades, initial_capital, n_sims=5000, years_to_simulate=5):
    """
    Calcula retornos mensuales reales, luego samplea con reemplazo para
    generar equity curves sinteticas de 'years_to_simulate' anos.

    Esto responde: "Si el mercado se comporta como los meses que hemos visto,
    cual es la distribucion de outcomes a X anos?"
    """
    # Agrupar trades por mes de cierre
    monthly_pnl = defaultdict(float)
    for t in trades:
        if t.exit_date:
            month_key = t.exit_date.strftime('%Y-%m')
            monthly_pnl[month_key] += t.pnl_euros

    # Ordenar por mes y calcular returns %
    sorted_months = sorted(monthly_pnl.keys())

    # Reconstruir equity mes a mes para calcular returns %
    equity = initial_capital
    monthly_returns = []
    monthly_equity = [equity]

    for month in sorted_months:
        pnl = monthly_pnl[month]
        ret_pct = pnl / equity * 100 if equity > 0 else 0
        monthly_returns.append(ret_pct)
        equity += pnl
        monthly_equity.append(equity)

    monthly_returns = np.array(monthly_returns)
    n_months_to_sim = int(years_to_simulate * 12)

    rng = np.random.default_rng(123)

    results = {
        'final_equity': [],
        'cagr': [],
        'max_dd': [],
        'worst_month': [],
        'best_month': [],
        'pct_negative_months': [],
    }

    for _ in range(n_sims):
        # Sample n_months_to_sim months with replacement
        sampled = rng.choice(monthly_returns, size=n_months_to_sim, replace=True)

        eq = initial_capital
        peak = initial_capital
        max_dd = 0.0

        for ret in sampled:
            eq *= (1 + ret / 100)
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        cagr = ((eq / initial_capital) ** (1 / years_to_simulate) - 1) * 100 if eq > 0 else -100

        results['final_equity'].append(eq)
        results['cagr'].append(cagr)
        results['max_dd'].append(max_dd)
        results['worst_month'].append(sampled.min())
        results['best_month'].append(sampled.max())
        results['pct_negative_months'].append(np.sum(sampled < 0) / len(sampled) * 100)

    return results, monthly_returns


# =============================================================================
# 3. PERMUTATION TEST (test de significancia)
# =============================================================================

def permutation_test(trades, initial_capital, n_perms=10000):
    """
    Test de hipotesis nula: "El PF observado se puede obtener por azar."

    Mecanismo: shuffle la relacion entre senales y retornos. En la practica,
    asignamos aleatoriamente los PnL% a trades de tamano fijo.
    Si el PF real esta en el top 5% de los PF shuffled, p-value < 0.05.

    Tambien testea: CAGR y win rate.
    """
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    sizes = np.array([t.position_euros for t in trades])

    # PF real
    real_pnls = pnl_pcts * sizes / 100
    real_gross_profit = np.sum(real_pnls[real_pnls > 0])
    real_gross_loss = np.abs(np.sum(real_pnls[real_pnls <= 0]))
    real_pf = real_gross_profit / real_gross_loss if real_gross_loss > 0 else 99
    real_total_pnl = np.sum(real_pnls)
    real_win_rate = np.sum(pnl_pcts > 0) / len(pnl_pcts) * 100

    rng = np.random.default_rng(456)

    perm_pfs = []
    perm_totals = []
    perm_wrs = []

    for _ in range(n_perms):
        # Shuffle PnL% across all trades (rompe senal→retorno)
        shuffled_pcts = rng.permutation(pnl_pcts)
        pnls = shuffled_pcts * sizes / 100

        gp = np.sum(pnls[pnls > 0])
        gl = np.abs(np.sum(pnls[pnls <= 0]))
        pf = gp / gl if gl > 0 else 99

        perm_pfs.append(pf)
        perm_totals.append(np.sum(pnls))
        perm_wrs.append(np.sum(shuffled_pcts > 0) / len(shuffled_pcts) * 100)

    perm_pfs = np.array(perm_pfs)
    perm_totals = np.array(perm_totals)
    perm_wrs = np.array(perm_wrs)

    # p-values: fraccion de permutaciones >= valor real
    p_pf = np.mean(perm_pfs >= real_pf)
    p_total = np.mean(perm_totals >= real_total_pnl)
    p_wr = np.mean(perm_wrs >= real_win_rate)

    return {
        'real_pf': real_pf,
        'real_total_pnl': real_total_pnl,
        'real_win_rate': real_win_rate,
        'p_value_pf': p_pf,
        'p_value_total_pnl': p_total,
        'p_value_win_rate': p_wr,
        'perm_pf_median': np.median(perm_pfs),
        'perm_pf_p95': np.percentile(perm_pfs, 95),
        'perm_total_median': np.median(perm_totals),
        'perm_total_p95': np.percentile(perm_totals, 95),
        'perm_pfs': perm_pfs,
    }


# =============================================================================
# UTILIDADES DE PRESENTACION
# =============================================================================

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
    """ASCII histogram."""
    data = np.array(data)
    counts, edges = np.histogram(data, bins=bins)
    max_count = max(counts)

    print(f"\n  {label} — Distribución ({len(data):,} simulaciones)")
    print(f"  {'─' * (width + 25)}")

    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        lo = edges[i]
        hi = edges[i + 1]
        bar = '█' * bar_len
        print(f"  {lo:>8.1f} - {hi:>8.1f} | {bar} ({count:,})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo Validation — v12')
    parser.add_argument('--months', type=int, default=60,
                        help='Meses de backtest para generar trades (default: 60)')
    parser.add_argument('--sims', type=int, default=5000,
                        help='Numero de simulaciones Monte Carlo (default: 5000)')
    parser.add_argument('--bootstrap-years', type=int, default=5,
                        help='Anos a simular en bootstrap (default: 5)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    months = args.months
    n_sims = args.sims
    bootstrap_years = args.bootstrap_years

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║   MONTE CARLO VALIDATION — v12 (Momentum Breakout + EU Options)    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Objetivo: Validar que el edge es REAL, no artefacto de overfitting ║
║                                                                      ║
║  Test 1: Trade Shuffle ({n_sims:,} sims)                                 ║
║  Test 2: Bootstrap retornos mensuales ({n_sims:,} sims, {bootstrap_years}y)             ║
║  Test 3: Permutation test ({n_sims:,} perms)                              ║
║                                                                      ║
║  Periodo base: {months} meses                                            ║
║  Config: US2+EU2, spread US 3% / EU 10%                             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # =====================================================================
    # PASO 0: Ejecutar backtest v12 para obtener trades
    # =====================================================================
    print("=" * 70)
    print("  PASO 0: Ejecutando backtest v12 para obtener trades...")
    print("=" * 70)

    result = run_backtest_eu(
        months, BASE_TICKERS, f"v12 US2+EU2 ({months}m)",
        use_options=True,
        options_eligible_set=OPTIONS_ALL,
        max_us_options=2, max_eu_options=2,
        verbose=False
    )

    if 'error' in result:
        print(f"  ERROR: {result['error']}")
        return

    all_trades = result['combined_trades']
    stock_trades = result['all_trades']
    option_trades = result['all_option_trades']
    initial_capital = CONFIG['initial_capital']

    print(f"""
  Trades obtenidos: {len(all_trades)} ({len(stock_trades)} stocks + {len(option_trades)} opciones)
  CAGR real:        {result['annualized_return_pct']:+.1f}%
  MaxDD real:       -{result['max_drawdown']:.1f}%
  PF real:          {result['profit_factor']:.2f}
  Final equity:     EUR {result['final_equity']:,.0f}
  Win rate:         {result['win_rate']:.1f}%
    """)

    # =====================================================================
    # TEST 1: TRADE SHUFFLE MONTE CARLO
    # =====================================================================
    print("=" * 70)
    print(f"  TEST 1: TRADE SHUFFLE MONTE CARLO ({n_sims:,} simulaciones)")
    print("=" * 70)
    print("  Pregunta: ¿El resultado depende del ORDEN de los trades?")
    print("  Si la mediana ~= real, el edge es robusto al timing.\n")

    mc = monte_carlo_trade_shuffle(all_trades, initial_capital, n_sims)

    print_percentiles("CAGR (%)", mc['cagr'], '%')
    print()
    print_percentiles("MaxDD (%)", mc['max_dd'], '%')
    print()
    print_percentiles("Final Equity (EUR)", mc['final_equity'])
    print()
    print_percentiles("Profit Factor", mc['profit_factor'])

    # Comparar con real
    real_cagr = result['annualized_return_pct']
    median_cagr = np.median(mc['cagr'])
    print(f"""
  ─── COMPARACION CON REAL ───
  CAGR real:     {real_cagr:+.1f}%
  CAGR mediana:  {median_cagr:+.1f}%
  Delta:         {real_cagr - median_cagr:+.1f}pp

  MaxDD real:    -{result['max_drawdown']:.1f}%
  MaxDD mediana: -{np.median(mc['max_dd']):.1f}%

  Prob CAGR > 0:    {np.mean(np.array(mc['cagr']) > 0) * 100:.1f}%
  Prob CAGR > 20%:  {np.mean(np.array(mc['cagr']) > 20) * 100:.1f}%
  Prob CAGR > 40%:  {np.mean(np.array(mc['cagr']) > 40) * 100:.1f}%
  Prob MaxDD > 50%: {np.mean(np.array(mc['max_dd']) > 50) * 100:.1f}%
  Prob MaxDD > 70%: {np.mean(np.array(mc['max_dd']) > 70) * 100:.1f}%
    """)

    print_histogram(mc['cagr'], 'CAGR (%)')
    print_histogram(mc['max_dd'], 'MaxDD (%)')

    # =====================================================================
    # TEST 2: BOOTSTRAP RETORNOS MENSUALES
    # =====================================================================
    print(f"\n{'=' * 70}")
    print(f"  TEST 2: BOOTSTRAP RETORNOS MENSUALES ({n_sims:,} sims, {bootstrap_years}y)")
    print("=" * 70)
    print("  Pregunta: ¿Cuál es la distribución de outcomes a 5 años")
    print("  si los meses futuros se parecen a los pasados?\n")

    bs, monthly_rets = bootstrap_monthly_returns(
        all_trades, initial_capital, n_sims, bootstrap_years
    )

    print(f"  Retornos mensuales reales ({len(monthly_rets)} meses):")
    print(f"     Media:    {np.mean(monthly_rets):+.2f}%")
    print(f"     Mediana:  {np.median(monthly_rets):+.2f}%")
    print(f"     Std:      {np.std(monthly_rets):.2f}%")
    print(f"     Min:      {np.min(monthly_rets):+.2f}%")
    print(f"     Max:      {np.max(monthly_rets):+.2f}%")
    print(f"     % meses negativos: {np.sum(monthly_rets < 0) / len(monthly_rets) * 100:.1f}%")
    print()

    print_percentiles(f"CAGR a {bootstrap_years}y (%)", bs['cagr'], '%')
    print()
    print_percentiles(f"MaxDD a {bootstrap_years}y (%)", bs['max_dd'], '%')
    print()
    print_percentiles(f"Final Equity a {bootstrap_years}y (EUR)", bs['final_equity'])

    prob_loss = np.mean(np.array(bs['final_equity']) < initial_capital) * 100
    prob_double = np.mean(np.array(bs['final_equity']) > initial_capital * 2) * 100
    prob_10x = np.mean(np.array(bs['final_equity']) > initial_capital * 10) * 100

    print(f"""
  ─── PROBABILIDADES A {bootstrap_years} AÑOS ───
  Prob perder dinero:     {prob_loss:.1f}%
  Prob duplicar capital:  {prob_double:.1f}%
  Prob 10x capital:       {prob_10x:.1f}%
  Prob CAGR > 20%:        {np.mean(np.array(bs['cagr']) > 20) * 100:.1f}%
  Prob CAGR > 40%:        {np.mean(np.array(bs['cagr']) > 40) * 100:.1f}%
  Prob CAGR > 60%:        {np.mean(np.array(bs['cagr']) > 60) * 100:.1f}%
  Prob MaxDD > 50%:       {np.mean(np.array(bs['max_dd']) > 50) * 100:.1f}%
  Prob MaxDD > 70%:       {np.mean(np.array(bs['max_dd']) > 70) * 100:.1f}%
    """)

    print_histogram(bs['cagr'], f'CAGR a {bootstrap_years}y (%)')
    print_histogram(bs['max_dd'], f'MaxDD a {bootstrap_years}y (%)')

    # =====================================================================
    # TEST 3: PERMUTATION TEST
    # =====================================================================
    print(f"\n{'=' * 70}")
    print(f"  TEST 3: PERMUTATION TEST ({n_sims:,} permutaciones)")
    print("=" * 70)
    print("  Pregunta: ¿El PF observado es estadísticamente significativo?")
    print("  H0: Las señales NO tienen poder predictivo (PF observado = azar)")
    print("  Si p-value < 0.05, rechazamos H0 → el edge es REAL.\n")

    perm = permutation_test(all_trades, initial_capital, n_sims)

    print(f"  ─── RESULTADOS ───")
    print(f"  Profit Factor real:    {perm['real_pf']:.2f}")
    print(f"  PF median shuffled:    {perm['perm_pf_median']:.2f}")
    print(f"  PF P95 shuffled:       {perm['perm_pf_p95']:.2f}")
    print(f"  p-value PF:            {perm['p_value_pf']:.4f}  {'✅ SIGNIFICATIVO' if perm['p_value_pf'] < 0.05 else '❌ NO significativo'}")
    print()
    print(f"  Total PnL real:        EUR {perm['real_total_pnl']:+,.0f}")
    print(f"  PnL median shuffled:   EUR {perm['perm_total_median']:+,.0f}")
    print(f"  PnL P95 shuffled:      EUR {perm['perm_total_p95']:+,.0f}")
    print(f"  p-value PnL:           {perm['p_value_total_pnl']:.4f}  {'✅ SIGNIFICATIVO' if perm['p_value_total_pnl'] < 0.05 else '❌ NO significativo'}")
    print()
    print(f"  Win rate real:         {perm['real_win_rate']:.1f}%")
    print(f"  p-value Win Rate:      {perm['p_value_win_rate']:.4f}  {'✅ SIGNIFICATIVO' if perm['p_value_win_rate'] < 0.05 else '❌ NO significativo'}")

    print_histogram(perm['perm_pfs'], 'PF bajo H0 (shuffled)')

    # =====================================================================
    # RESUMEN EJECUTIVO
    # =====================================================================
    print(f"""

{'╔' + '═'*68 + '╗'}
{'║'} {'RESUMEN EJECUTIVO — VALIDACION MONTE CARLO v12':^66} {'║'}
{'╠' + '═'*68 + '╣'}
║                                                                    ║
║  Backtest base: {months}m, {len(all_trades)} trades, CAGR {real_cagr:+.1f}%, PF {result['profit_factor']:.2f}            ║
║                                                                    ║
║  TEST 1 — Trade Shuffle:                                           ║
║    CAGR mediana: {median_cagr:+.1f}% (real {real_cagr:+.1f}%)                           ║
║    → {'ROBUSTO al orden' if abs(real_cagr - median_cagr) < 15 else 'DEPENDIENTE del orden (fragil)'}                                        ║
║                                                                    ║
║  TEST 2 — Bootstrap {bootstrap_years}y:                                            ║
║    CAGR mediana: {np.median(bs['cagr']):+.1f}%, P5: {np.percentile(bs['cagr'], 5):+.1f}%                         ║
║    Prob perder: {prob_loss:.1f}%                                            ║
║    → {'MUY BAJA prob de perdida' if prob_loss < 5 else 'RIESGO de perdida significativo (' + f'{prob_loss:.0f}%' + ')'}                                 ║
║                                                                    ║
║  TEST 3 — Permutation:                                             ║
║    p-value PF: {perm['p_value_pf']:.4f}                                           ║
║    p-value PnL: {perm['p_value_total_pnl']:.4f}                                          ║
║    → {'EDGE ESTADISTICAMENTE SIGNIFICATIVO' if perm['p_value_pf'] < 0.05 and perm['p_value_total_pnl'] < 0.05 else 'Edge NO significativo — posible overfitting'}                       ║
║                                                                    ║
{'╚' + '═'*68 + '╝'}
    """)


if __name__ == "__main__":
    main()
