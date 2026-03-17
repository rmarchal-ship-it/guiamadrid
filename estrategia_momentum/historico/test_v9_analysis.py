#!/usr/bin/env python3
"""
TEST v9 — Analisis detallado: distribucion de opciones abiertas + grid sizing
Fecha: 27 Feb 2026

Objetivo:
  1. Cuantas opciones hay abiertas simultaneamente en v9 (histograma)
  2. Grid de option_position_pct: 8%, 10%, 12%, 14%
  3. Grid de max_option_positions: 3, 5, 7, 10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_experimental import (
    CONFIG, BASE_TICKERS, OPTIONS_ELIGIBLE, run_backtest,
    generate_all_signals, build_macro_filter, download_data,
    calculate_atr, historical_volatility, MomentumEngine,
    EquityTracker, Trade, OptionTradeV2, find_candidates,
    rank_candidates, black_scholes_call, iv_rank, monthly_expiration_dte,
    LEVERAGE_FACTORS
)
import pandas as pd
import numpy as np
from collections import Counter


def run_backtest_with_tracking(months, tickers, label, max_opts, opt_pct, verbose=False):
    """Run backtest tracking concurrent open options over time."""

    # Temporarily override CONFIG
    orig_max_opts = CONFIG['max_option_positions']
    orig_opt_pct = CONFIG['option_position_pct']
    CONFIG['max_option_positions'] = max_opts
    CONFIG['option_position_pct'] = opt_pct

    result = run_backtest(months, tickers, label,
                          use_leverage_scaling=False, use_options=True, verbose=verbose)

    # Restore CONFIG
    CONFIG['max_option_positions'] = orig_max_opts
    CONFIG['option_position_pct'] = orig_opt_pct

    return result


def analyze_concurrent_options(result):
    """Analyze how many options were open simultaneously over time."""
    opt_trades = result.get('all_option_trades', [])
    if not opt_trades:
        return {}

    # Build timeline of option opens/closes
    events = []
    for t in opt_trades:
        if t.entry_date:
            events.append((t.entry_date, +1, t.ticker))
        if t.exit_date:
            events.append((t.exit_date, -1, t.ticker))

    events.sort(key=lambda x: x[0])

    concurrent = 0
    max_concurrent = 0
    daily_concurrent = []

    for date, delta, ticker in events:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
        daily_concurrent.append(concurrent)

    # Distribution
    counter = Counter(daily_concurrent)

    return {
        'max_concurrent': max_concurrent,
        'avg_concurrent': np.mean(daily_concurrent) if daily_concurrent else 0,
        'distribution': counter,
        'total_option_trades': len(opt_trades),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--months', type=int, default=60)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    months = args.months

    print(f"""
======================================================================
  v9 OPTIONS-FIRST — ANALISIS DETALLADO — {months}m
======================================================================
""")

    # =====================================================================
    # GRID TEST 1: max_option_positions con sizing fijo a 14%
    # =====================================================================
    print("=" * 70)
    print("  GRID 1: max_option_positions (sizing = 14%)")
    print("=" * 70)

    grid_results = []
    for max_opts in [2, 4, 6, 8, 10]:
        label = f"max_opts={max_opts} pct=14%"
        r = run_backtest_with_tracking(months, BASE_TICKERS, label,
                                        max_opts=max_opts, opt_pct=0.14)
        if 'error' not in r:
            opt_analysis = analyze_concurrent_options(r)
            grid_results.append({
                'max_opts': max_opts,
                'opt_pct': 14,
                'return': r['total_return_pct'],
                'annual': r['annualized_return_pct'],
                'pf': r['profit_factor'],
                'maxdd': r['max_drawdown'],
                'trades': r['total_trades'],
                'opt_trades': r['option_trades'],
                'stk_trades': r['stock_trades'],
                'win_rate': r['win_rate'],
                'max_concurrent': opt_analysis.get('max_concurrent', 0),
                'avg_concurrent': opt_analysis.get('avg_concurrent', 0),
                'opt_pnl': sum(t.pnl_euros for t in r.get('all_option_trades', [])),
                'stk_pnl': sum(t.pnl_euros for t in r.get('all_trades', [])),
            })

    # =====================================================================
    # GRID TEST 2: option_position_pct con max_opts=10
    # =====================================================================
    print("\n" + "=" * 70)
    print("  GRID 2: option_position_pct (max_opts = 10)")
    print("=" * 70)

    for opt_pct in [0.06, 0.08, 0.10, 0.12]:
        label = f"max_opts=10 pct={int(opt_pct*100)}%"
        r = run_backtest_with_tracking(months, BASE_TICKERS, label,
                                        max_opts=10, opt_pct=opt_pct)
        if 'error' not in r:
            opt_analysis = analyze_concurrent_options(r)
            grid_results.append({
                'max_opts': 10,
                'opt_pct': int(opt_pct * 100),
                'return': r['total_return_pct'],
                'annual': r['annualized_return_pct'],
                'pf': r['profit_factor'],
                'maxdd': r['max_drawdown'],
                'trades': r['total_trades'],
                'opt_trades': r['option_trades'],
                'stk_trades': r['stock_trades'],
                'win_rate': r['win_rate'],
                'max_concurrent': opt_analysis.get('max_concurrent', 0),
                'avg_concurrent': opt_analysis.get('avg_concurrent', 0),
                'opt_pnl': sum(t.pnl_euros for t in r.get('all_option_trades', [])),
                'stk_pnl': sum(t.pnl_euros for t in r.get('all_trades', [])),
            })

    # =====================================================================
    # TABLA RESUMEN
    # =====================================================================
    print(f"""
{'='*100}
  TABLA RESUMEN COMPLETA — {months}m
{'='*100}

  {'Config':<22} {'Return%':>8} {'Annual%':>8} {'PF':>6} {'MaxDD%':>7} {'Trades':>7} {'Opts':>5} {'Stks':>5} {'MaxConc':>8} {'OptPnL':>10} {'StkPnL':>10}
  {'-'*95}""")

    for r in sorted(grid_results, key=lambda x: (x['max_opts'], x['opt_pct'])):
        config = f"opts={r['max_opts']:2d} pct={r['opt_pct']:2d}%"
        print(f"  {config:<22} {r['return']:>+7.1f}% {r['annual']:>+7.1f}% {r['pf']:>5.2f} {r['maxdd']:>6.1f}% "
              f"{r['trades']:>7} {r['opt_trades']:>5} {r['stk_trades']:>5} {r['max_concurrent']:>8} "
              f"EUR{r['opt_pnl']:>+9,.0f} EUR{r['stk_pnl']:>+9,.0f}")

    print(f"\n  Referencia stock-only: +119.9%, +17.1% ann, PF 2.30, MaxDD -14.6%")

    # Mejor relacion return/MaxDD
    print(f"\n  --- EFICIENCIA (Return / MaxDD) ---")
    for r in sorted(grid_results, key=lambda x: x['annual'] / max(x['maxdd'], 1), reverse=True):
        config = f"opts={r['max_opts']:2d} pct={r['opt_pct']:2d}%"
        efficiency = r['annual'] / r['maxdd'] if r['maxdd'] > 0 else 0
        print(f"  {config:<22} Eficiencia: {efficiency:.2f} (Ann {r['annual']:+.1f}% / DD {r['maxdd']:.1f}%)")


if __name__ == "__main__":
    main()
