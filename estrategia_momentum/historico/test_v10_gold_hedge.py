#!/usr/bin/env python3
"""
TEST v10 — Gold Hedge Overlay
Fecha: 27 Feb 2026

Hipotesis: mantener 30% del equity en GLD en todo momento + cash no utilizado
(posiciones vacias / filtro bear) tambien en GLD deberia reducir MaxDD.

Approach: post-hoc overlay sobre resultados del backtest standard.
  1. Ejecuta backtest normal (v8 o v9)
  2. Escala P&L momentum al 70% (30% reservado para oro)
  3. Calcula gold allocation diaria: 30% + idle cash → GLD returns
  4. Equity curve combinada → MaxDD real

Variantes:
  - v8 (2 opts, 14%): baseline actual
  - v8 + Gold 30%: v8 con hedge de oro
  - v9 (10 opts, 14%): options-first agresivo
  - v9 + Gold 30%: v9 con hedge de oro
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_experimental import (
    CONFIG, BASE_TICKERS, run_backtest, download_data
)
import pandas as pd
import numpy as np
from collections import defaultdict


def simulate_gold_overlay(result, gld_data, gold_reserve_pct=0.30, initial_capital=None):
    """
    Simulacion post-hoc: overlay gold sobre resultado de backtest.

    Mecanismo:
      - 30% equity SIEMPRE en GLD (reserva permanente)
      - Cash no invertido (posiciones vacias) TAMBIEN en GLD
      - P&L momentum escalado al 70% (solo 70% del equity disponible para trading)
      - Gold compound diario

    Args:
        result: dict con all_trades, all_option_trades del backtest
        gld_data: DataFrame con columna Close de GLD
        gold_reserve_pct: fraccion minima en GLD (default 0.30)

    Returns:
        dict con metricas del portfolio combinado
    """
    if initial_capital is None:
        initial_capital = CONFIG['initial_capital']

    max_positions = CONFIG['max_positions']
    trading_pct = 1.0 - gold_reserve_pct  # 0.70

    all_trades = result.get('all_trades', []) + result.get('all_option_trades', [])
    if not all_trades:
        return None

    # GLD daily returns
    gld_close = gld_data['Close'].copy()
    if isinstance(gld_close, pd.DataFrame):
        gld_close = gld_close.squeeze()
    gld_returns = gld_close.pct_change().fillna(0)

    # Build trade events by date
    close_pnl_by_date = defaultdict(float)     # date -> sum of pnl_euros closing
    open_delta_by_date = defaultdict(int)       # date -> net open/close count

    for t in all_trades:
        if t.entry_date:
            open_delta_by_date[pd.Timestamp(t.entry_date)] += 1
        if t.exit_date:
            close_pnl_by_date[pd.Timestamp(t.exit_date)] += t.pnl_euros
            open_delta_by_date[pd.Timestamp(t.exit_date)] -= 1

    # Timeline: all business days in the backtest period
    all_entry_dates = [pd.Timestamp(t.entry_date) for t in all_trades if t.entry_date]
    all_exit_dates = [pd.Timestamp(t.exit_date) for t in all_trades if t.exit_date]
    start_date = min(all_entry_dates)
    end_date = max(all_exit_dates + all_entry_dates)

    trading_days = pd.bdate_range(start_date, end_date)

    # === SIMULACION SIN ORO (referencia escalada) ===
    equity_no_gold = initial_capital
    max_eq_no_gold = initial_capital
    max_dd_no_gold = 0.0

    # === SIMULACION CON ORO ===
    equity_gold = initial_capital
    max_eq_gold = initial_capital
    max_dd_gold = 0.0

    open_positions = 0
    gold_total_pnl = 0.0
    momentum_total_pnl = 0.0

    curve_no_gold = []
    curve_gold = []
    daily_gold_alloc_pcts = []

    for day in trading_days:
        # --- 1. Momentum P&L (from trade closings) ---
        raw_pnl = close_pnl_by_date.get(day, 0.0)
        scaled_pnl = raw_pnl * trading_pct  # solo 70% del equity para trading

        # Update positions count
        delta = open_delta_by_date.get(day, 0)
        open_positions += delta
        open_positions = max(0, open_positions)

        # Update no-gold equity (scaled momentum only)
        equity_no_gold += scaled_pnl
        momentum_total_pnl += scaled_pnl

        # --- 2. Gold allocation ---
        # Invested fraction of total equity (max_positions slots, each ~7% of equity)
        invested_pct = min(open_positions / max_positions, 1.0) * trading_pct
        gold_pct = max(gold_reserve_pct, 1.0 - invested_pct)
        daily_gold_alloc_pcts.append(gold_pct)

        # Gold return for this day
        gld_ret = 0.0
        if day in gld_returns.index:
            gld_ret = gld_returns.loc[day]
            if isinstance(gld_ret, pd.Series):
                gld_ret = gld_ret.iloc[0]

        gold_allocation_eur = equity_gold * gold_pct
        gold_pnl = gold_allocation_eur * gld_ret

        # --- 3. Update equity with gold ---
        equity_gold += scaled_pnl + gold_pnl
        gold_total_pnl += gold_pnl

        # --- 4. Track curves and DD ---
        curve_no_gold.append((day, equity_no_gold))
        curve_gold.append((day, equity_gold))

        max_eq_no_gold = max(max_eq_no_gold, equity_no_gold)
        dd_no_gold = (max_eq_no_gold - equity_no_gold) / max_eq_no_gold * 100
        max_dd_no_gold = max(max_dd_no_gold, dd_no_gold)

        max_eq_gold = max(max_eq_gold, equity_gold)
        dd_gold = (max_eq_gold - equity_gold) / max_eq_gold * 100
        max_dd_gold = max(max_dd_gold, dd_gold)

    # Calcular metricas
    days = (end_date - start_date).days
    years = days / 365.25

    ret_no_gold = (equity_no_gold / initial_capital - 1) * 100
    ret_gold = (equity_gold / initial_capital - 1) * 100

    ann_no_gold = ((equity_no_gold / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    ann_gold = ((equity_gold / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    avg_gold_pct = np.mean(daily_gold_alloc_pcts) * 100 if daily_gold_alloc_pcts else 0

    # GLD total return in the period
    gld_period = gld_close.loc[start_date:end_date]
    if len(gld_period) >= 2:
        gld_total_ret = (gld_period.iloc[-1] / gld_period.iloc[0] - 1) * 100
    else:
        gld_total_ret = 0

    return {
        'equity_no_gold': equity_no_gold,
        'equity_gold': equity_gold,
        'return_no_gold': ret_no_gold,
        'return_gold': ret_gold,
        'ann_no_gold': ann_no_gold,
        'ann_gold': ann_gold,
        'maxdd_no_gold': max_dd_no_gold,
        'maxdd_gold': max_dd_gold,
        'gold_total_pnl': gold_total_pnl,
        'momentum_total_pnl': momentum_total_pnl,
        'avg_gold_alloc_pct': avg_gold_pct,
        'gld_total_return': gld_total_ret,
        'curve_no_gold': curve_no_gold,
        'curve_gold': curve_gold,
        'eff_no_gold': ann_no_gold / max_dd_no_gold if max_dd_no_gold > 0 else 0,
        'eff_gold': ann_gold / max_dd_gold if max_dd_gold > 0 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test v10: Gold Hedge Overlay')
    parser.add_argument('--months', type=int, default=60, help='Meses de historico')
    parser.add_argument('--gold-pct', type=float, default=0.30, help='Reserva en oro (default 30%%)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    months = args.months
    gold_pct = args.gold_pct

    print(f"""
======================================================================
  TEST v10 — GOLD HEDGE OVERLAY — {months}m — Gold reserve: {gold_pct*100:.0f}%
======================================================================
  Mecanismo:
    - {gold_pct*100:.0f}% equity SIEMPRE en GLD (reserva permanente)
    - Cash no invertido (posiciones vacias) TAMBIEN en GLD
    - Momentum P&L escalado al {(1-gold_pct)*100:.0f}% (menos capital para trading)

  Comparativa:
    1. v8 (2 opts, 14%) — baseline
    2. v8 + Gold {gold_pct*100:.0f}%
    3. v9 (10 opts, 14%) — options-first
    4. v9 + Gold {gold_pct*100:.0f}%
======================================================================
""")

    # Descargar GLD
    print("  Descargando GLD...")
    gld_data = download_data('GLD', months + 6)  # extra margen
    if gld_data is None:
        print("  ERROR: No se pudo descargar GLD. Abortando.")
        return
    print(f"  GLD: {len(gld_data)} dias, {gld_data.index[0].strftime('%Y-%m-%d')} a {gld_data.index[-1].strftime('%Y-%m-%d')}")

    gld_ret = (gld_data['Close'].iloc[-1] / gld_data['Close'].iloc[0] - 1) * 100
    if isinstance(gld_ret, pd.Series):
        gld_ret = gld_ret.iloc[0]
    print(f"  GLD return periodo: {gld_ret:+.1f}%")

    results_table = []

    # =====================================================================
    # TEST 1: v8 (2 opts, 14%)
    # =====================================================================
    print("\n>>> TEST 1/2: v8 (max 2 opciones, 14% sizing)")
    CONFIG['max_option_positions'] = 2
    CONFIG['option_position_pct'] = 0.14

    r_v8 = run_backtest(months, BASE_TICKERS, "v8 (2 opts 14%)",
                        use_leverage_scaling=False, use_options=True,
                        verbose=args.verbose)

    if 'error' not in r_v8:
        gold_v8 = simulate_gold_overlay(r_v8, gld_data, gold_pct)
        if gold_v8:
            results_table.append({
                'label': 'v8 original',
                'return': r_v8['total_return_pct'],
                'annual': r_v8['annualized_return_pct'],
                'pf': r_v8['profit_factor'],
                'maxdd': r_v8['max_drawdown'],
                'trades': r_v8['total_trades'],
            })
            results_table.append({
                'label': f'v8 + Gold {gold_pct*100:.0f}%',
                'return': gold_v8['return_gold'],
                'annual': gold_v8['ann_gold'],
                'pf': r_v8['profit_factor'],  # PF no cambia (mismos trades)
                'maxdd': gold_v8['maxdd_gold'],
                'trades': r_v8['total_trades'],
                'gold_pnl': gold_v8['gold_total_pnl'],
                'mom_pnl': gold_v8['momentum_total_pnl'],
                'avg_gold': gold_v8['avg_gold_alloc_pct'],
                'eff': gold_v8['eff_gold'],
            })
            # Tambien guardar v8 70% sin gold (referencia)
            results_table.append({
                'label': 'v8 70% (sin gold)',
                'return': gold_v8['return_no_gold'],
                'annual': gold_v8['ann_no_gold'],
                'pf': r_v8['profit_factor'],
                'maxdd': gold_v8['maxdd_no_gold'],
                'trades': r_v8['total_trades'],
            })

    # =====================================================================
    # TEST 2: v9 (10 opts, 14%)
    # =====================================================================
    print("\n>>> TEST 2/2: v9 OPTIONS-FIRST (max 10 opciones, 14% sizing)")
    CONFIG['max_option_positions'] = 10
    CONFIG['option_position_pct'] = 0.14

    r_v9 = run_backtest(months, BASE_TICKERS, "v9 (10 opts 14%)",
                        use_leverage_scaling=False, use_options=True,
                        verbose=args.verbose)

    if 'error' not in r_v9:
        gold_v9 = simulate_gold_overlay(r_v9, gld_data, gold_pct)
        if gold_v9:
            results_table.append({
                'label': 'v9 original',
                'return': r_v9['total_return_pct'],
                'annual': r_v9['annualized_return_pct'],
                'pf': r_v9['profit_factor'],
                'maxdd': r_v9['max_drawdown'],
                'trades': r_v9['total_trades'],
            })
            results_table.append({
                'label': f'v9 + Gold {gold_pct*100:.0f}%',
                'return': gold_v9['return_gold'],
                'annual': gold_v9['ann_gold'],
                'pf': r_v9['profit_factor'],
                'maxdd': gold_v9['maxdd_gold'],
                'trades': r_v9['total_trades'],
                'gold_pnl': gold_v9['gold_total_pnl'],
                'mom_pnl': gold_v9['momentum_total_pnl'],
                'avg_gold': gold_v9['avg_gold_alloc_pct'],
                'eff': gold_v9['eff_gold'],
            })
            results_table.append({
                'label': 'v9 70% (sin gold)',
                'return': gold_v9['return_no_gold'],
                'annual': gold_v9['ann_no_gold'],
                'pf': r_v9['profit_factor'],
                'maxdd': gold_v9['maxdd_no_gold'],
                'trades': r_v9['total_trades'],
            })

    # =====================================================================
    # TABLA RESUMEN
    # =====================================================================
    print(f"""
{'='*100}
  TABLA RESUMEN — Gold Hedge Overlay — {months}m — Reserve: {gold_pct*100:.0f}%
{'='*100}

  {'Config':<25} {'Return%':>9} {'Annual%':>9} {'PF':>6} {'MaxDD%':>8} {'Trades':>7} {'GoldPnL':>11} {'AvgGold%':>9} {'Efic':>6}
  {'-'*95}""")

    for r in results_table:
        gold_pnl_s = f"EUR{r.get('gold_pnl', 0):>+8,.0f}" if r.get('gold_pnl') else "       -"
        avg_gold_s = f"{r.get('avg_gold', 0):>7.1f}%" if r.get('avg_gold') else "       -"
        eff_s = f"{r.get('eff', r.get('annual', 0) / max(r.get('maxdd', 1), 0.01)):>5.2f}" if r.get('maxdd', 0) > 0 else "    -"
        print(f"  {r['label']:<25} {r['return']:>+8.1f}% {r['annual']:>+8.1f}% {r['pf']:>5.2f} "
              f"{r['maxdd']:>7.1f}% {r['trades']:>7} {gold_pnl_s} {avg_gold_s} {eff_s}")

    # Detalle gold overlay
    print(f"""
{'='*100}
  ANALISIS DEL GOLD OVERLAY
{'='*100}""")

    if 'error' not in r_v8 and gold_v8:
        print(f"""
  v8 + Gold {gold_pct*100:.0f}%:
    Gold P&L total:     EUR {gold_v8['gold_total_pnl']:+,.0f}
    Momentum P&L (70%): EUR {gold_v8['momentum_total_pnl']:+,.0f}
    Avg gold allocation: {gold_v8['avg_gold_alloc_pct']:.1f}%
    GLD return periodo:  {gold_v8['gld_total_return']:+.1f}%
    MaxDD sin gold:      {gold_v8['maxdd_no_gold']:.1f}% (momentum 70% solo)
    MaxDD con gold:      {gold_v8['maxdd_gold']:.1f}% (momentum 70% + gold)
    DD reduction:        {gold_v8['maxdd_no_gold'] - gold_v8['maxdd_gold']:+.1f}pp
    Eficiencia:          {gold_v8['eff_gold']:.2f} (vs v8 original: {r_v8['annualized_return_pct']/r_v8['max_drawdown']:.2f})""")

    if 'error' not in r_v9 and gold_v9:
        print(f"""
  v9 + Gold {gold_pct*100:.0f}%:
    Gold P&L total:     EUR {gold_v9['gold_total_pnl']:+,.0f}
    Momentum P&L (70%): EUR {gold_v9['momentum_total_pnl']:+,.0f}
    Avg gold allocation: {gold_v9['avg_gold_alloc_pct']:.1f}%
    GLD return periodo:  {gold_v9['gld_total_return']:+.1f}%
    MaxDD sin gold:      {gold_v9['maxdd_no_gold']:.1f}% (momentum 70% solo)
    MaxDD con gold:      {gold_v9['maxdd_gold']:.1f}% (momentum 70% + gold)
    DD reduction:        {gold_v9['maxdd_no_gold'] - gold_v9['maxdd_gold']:+.1f}pp
    Eficiencia:          {gold_v9['eff_gold']:.2f} (vs v9 original: {r_v9['annualized_return_pct']/r_v9['max_drawdown']:.2f})""")

    print(f"""
  CONCLUSION:
    - GLD return en los ultimos {months}m: {gld_ret:+.1f}%
    - Gold hedge reduce MaxDD? Ver diferencia MaxDD original vs gold overlay
    - Coste: momentum P&L reducido al 70%
    - Beneficio: gold returns + amortiguacion en caidas
""")

    # Restaurar CONFIG
    CONFIG['max_option_positions'] = 2
    CONFIG['option_position_pct'] = 0.14


if __name__ == "__main__":
    main()
