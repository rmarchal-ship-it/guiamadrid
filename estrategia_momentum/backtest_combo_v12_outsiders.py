#!/usr/bin/env python3
"""
BACKTEST COMBO: 70% v12 + 30% Outsiders
Creado: 28 Feb 2026

Combina dos estrategias independientes con rebalanceo mensual:
  - 70% capital en v12 (225 tickers, opciones US+EU, gold overlay)
  - 30% capital en Outsiders v2 (64 tickers alternativos, SPOT sin gold)

El rebalanceo mensual ajusta los pesos de vuelta a 70/30.

Uso:
  python3 backtest_combo_v12_outsiders.py --months 120
  python3 backtest_combo_v12_outsiders.py --months 120 --no-rebal
  python3 backtest_combo_v12_outsiders.py --multi-period
  python3 backtest_combo_v12_outsiders.py --months 120 --weights 80 20
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import argparse
import sys
import os

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Imports v12
from backtest_v12_eu_options import (
    run_backtest_eu, simulate_gold_overlay as gold_overlay_v12,
    OPTIONS_ELIGIBLE, OPTIONS_ALL, OPTIONS_ELIGIBLE_EU,
)
from backtest_experimental import (
    CONFIG as CONFIG_V8, BASE_TICKERS, download_data,
)

# Imports outsiders
from backtest_outsiders_v2 import (
    CONFIG as CONFIG_OUT, OUTSIDER_TICKERS, TICKER_LIST, OPTIONS_MAP,
    download_all as download_all_out,
    generate_all_signals as gen_signals_out,
    build_macro_filter as build_macro_out,
    run_backtest as run_backtest_out,
    simulate_gold_overlay as gold_overlay_out,
)
from momentum_breakout import MomentumEngine


# =============================================================================
# EQUITY CURVE DESDE TRADES (para estrategias que no devuelven equity_curve)
# =============================================================================

def equity_curve_from_result(result, initial_capital):
    """
    Construye equity curve diaria [(date, equity), ...] a partir de trades.
    Asume que el resultado tiene 'equity_curve' o lo reconstruye de trades.
    """
    if 'equity_curve' in result and result['equity_curve']:
        return result['equity_curve']

    # Reconstruir desde trades
    all_t = result.get('all_trades', []) + result.get('all_option_trades', [])
    if not all_t:
        return []

    pnl_by_date = defaultdict(float)
    for t in all_t:
        if t.exit_date:
            pnl_by_date[pd.Timestamp(t.exit_date)] += t.pnl_euros

    curve = []
    equity = initial_capital
    for dt in sorted(pnl_by_date.keys()):
        equity += pnl_by_date[dt]
        curve.append((dt, equity))

    return curve


def daily_returns_from_curve(curve, initial_capital):
    """
    Convierte equity curve [(date, equity)] en Series de retornos diarios.
    Rellena dias sin actividad con 0% return.
    """
    if not curve:
        return pd.Series(dtype=float)

    # Crear series con index de fechas
    dates = [c[0] for c in curve]
    equities = [c[1] for c in curve]

    eq_series = pd.Series(equities, index=pd.DatetimeIndex(dates))

    # Si hay duplicados (multiples trades cerrados el mismo dia), quedarnos con el ultimo
    eq_series = eq_series.groupby(eq_series.index).last()

    # Insertar capital inicial al principio
    start = eq_series.index[0] - pd.Timedelta(days=1)
    eq_series = pd.concat([pd.Series([initial_capital], index=[start]), eq_series])

    # Rellenar a dias habiles y forward-fill
    full_idx = pd.bdate_range(eq_series.index[0], eq_series.index[-1])
    eq_series = eq_series.reindex(full_idx).ffill()

    # Retornos diarios
    returns = eq_series.pct_change().fillna(0)
    return returns


# =============================================================================
# COMBINAR DOS ESTRATEGIAS CON REBALANCEO
# =============================================================================

def combine_strategies(returns_a, returns_b, weight_a=0.70, weight_b=0.30,
                       rebalance='monthly', initial_capital=10000):
    """
    Combina dos series de retornos diarios con pesos y rebalanceo.

    Args:
        returns_a, returns_b: pd.Series de retornos diarios (index = dates)
        weight_a, weight_b: pesos objetivo (deben sumar 1.0)
        rebalance: 'monthly', 'annual', 'none'
        initial_capital: capital inicial

    Returns:
        dict con metricas del combo
    """
    # Alinear fechas (interseccion)
    common = returns_a.index.intersection(returns_b.index)
    if len(common) < 30:
        return {'error': f'Solo {len(common)} dias en comun'}

    ra = returns_a.reindex(common).fillna(0)
    rb = returns_b.reindex(common).fillna(0)

    # Simular con rebalanceo
    equity = initial_capital
    equity_a = equity * weight_a
    equity_b = equity * weight_b

    max_equity = equity
    max_dd = 0.0
    equity_curve = [(common[0], equity)]

    daily_returns = []
    last_rebal_month = None

    for i, date in enumerate(common):
        if i == 0:
            last_rebal_month = date.month
            continue

        # Aplicar retornos del dia
        equity_a *= (1 + ra.iloc[i])
        equity_b *= (1 + rb.iloc[i])
        equity = equity_a + equity_b

        daily_ret = equity / (equity_curve[-1][1]) - 1
        daily_returns.append(daily_ret)

        equity_curve.append((date, equity))

        # Drawdown
        max_equity = max(max_equity, equity)
        dd = (max_equity - equity) / max_equity * 100
        max_dd = max(max_dd, dd)

        # Rebalanceo
        if rebalance == 'monthly' and date.month != last_rebal_month:
            equity_a = equity * weight_a
            equity_b = equity * weight_b
            last_rebal_month = date.month
        elif rebalance == 'annual' and date.month == 1 and last_rebal_month != 1:
            equity_a = equity * weight_a
            equity_b = equity * weight_b
            last_rebal_month = date.month
        else:
            last_rebal_month = date.month

    # Metricas
    total_return = (equity / initial_capital - 1) * 100
    days = (common[-1] - common[0]).days
    years = days / 365.25
    cagr = ((equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    daily_ret_arr = np.array(daily_returns)
    vol_annual = np.std(daily_ret_arr) * np.sqrt(252) * 100
    sharpe = (cagr - 4.3) / vol_annual if vol_annual > 0 else 0  # rf = 4.3%

    # Eficiencia
    efficiency = cagr / max_dd if max_dd > 0 else 0

    return {
        'total_return': total_return,
        'cagr': cagr,
        'max_dd': max_dd,
        'vol_annual': vol_annual,
        'sharpe': sharpe,
        'efficiency': efficiency,
        'final_equity': equity,
        'years': years,
        'equity_curve': equity_curve,
        'n_days': len(common),
    }


# =============================================================================
# CORRER V12
# =============================================================================

def run_v12(months, use_gold=True, gold_pct=0.30, verbose=False):
    """Corre v12 (US2+EU2) y devuelve resultado + equity curve."""
    print(f"\n{'#'*70}")
    print(f"  COMPONENTE A: v12 (US2+EU2) — {months} meses")
    print(f"{'#'*70}")

    result = run_backtest_eu(
        months=months,
        tickers=BASE_TICKERS,
        label=f"v12 US2+EU2 ({months}m)",
        use_options=True,
        options_eligible_set=OPTIONS_ALL,
        max_us_options=2,
        max_eu_options=2,
        verbose=verbose,
    )

    if 'error' in result:
        return None, None

    info = {
        'cagr': result['annualized_return_pct'],
        'dd': result['max_drawdown'],
        'pf': result['profit_factor'],
        'equity': result['final_equity'],
        'trades': result['total_trades'],
    }

    # Gold overlay
    if use_gold:
        gld_data = download_data('GLD', months + 6)
        if gld_data is not None:
            g = gold_overlay_v12(result, gld_data, gold_pct)
            if g:
                info['cagr_gold'] = g['ann_gold']
                info['dd_gold'] = g['maxdd_gold']
                info['equity_gold'] = g['equity_gold']

    return result, info


# =============================================================================
# CORRER OUTSIDERS
# =============================================================================

def run_outsiders(months, max_positions=12, use_options=False, verbose=False):
    """Corre outsiders v2 (SPOT, sin gold) y devuelve resultado + equity curve."""
    print(f"\n{'#'*70}")
    print(f"  COMPONENTE B: Outsiders v2 ({'SPOT+OPT' if use_options else 'SPOT'}) — {months} meses")
    print(f"{'#'*70}")

    # Descargar datos
    all_tickers = TICKER_LIST + [CONFIG_OUT['macro_ticker']]
    all_data, failed = download_all_out(all_tickers, months)

    if not all_data:
        print("  ERROR: No hay datos outsiders.")
        return None, None

    # Engine + señales
    engine = MomentumEngine(
        ker_threshold=CONFIG_OUT['ker_threshold'],
        volume_threshold=CONFIG_OUT['volume_threshold'],
        rsi_threshold=CONFIG_OUT['rsi_threshold'],
        rsi_max=CONFIG_OUT['rsi_max'],
        breakout_period=CONFIG_OUT['breakout_period'],
        longs_only=CONFIG_OUT['longs_only']
    )
    signals_data, total_signals = gen_signals_out(all_data, engine)
    macro_bullish = build_macro_out(all_data)

    result = run_backtest_out(
        months=months,
        all_data=all_data,
        signals_data=signals_data,
        macro_bullish=macro_bullish,
        max_positions=max_positions,
        verbose=verbose,
        quiet=True,
        use_options=use_options,
    )

    if 'error' in result:
        return None, None

    info = {
        'cagr': result['annualized_return_pct'],
        'dd': result['max_drawdown'],
        'pf': result['profit_factor'],
        'equity': result['final_equity'],
        'trades': result['total_trades'],
    }

    return result, info


# =============================================================================
# MAIN: COMBO TEST
# =============================================================================

def run_combo(months, weight_v12=0.70, weight_out=0.30,
              rebalance='monthly', use_gold_v12=True, gold_pct=0.30,
              out_use_options=False, verbose=False):
    """
    Ejecuta v12 + outsiders y combina con pesos dados.
    """
    initial = CONFIG_V8['initial_capital']

    # Correr ambas estrategias
    result_v12, info_v12 = run_v12(months, use_gold=use_gold_v12,
                                    gold_pct=gold_pct, verbose=verbose)
    result_out, info_out = run_outsiders(months, use_options=out_use_options,
                                         verbose=verbose)

    if result_v12 is None or result_out is None:
        print("  ERROR: Una o ambas estrategias fallaron.")
        return None

    # Extraer equity curves
    curve_v12 = equity_curve_from_result(result_v12, initial)
    curve_out = equity_curve_from_result(result_out, initial)

    if not curve_v12 or not curve_out:
        print("  ERROR: Equity curves vacias.")
        return None

    # Convertir a retornos diarios
    ret_v12 = daily_returns_from_curve(curve_v12, initial)
    ret_out = daily_returns_from_curve(curve_out, initial)

    # Combinar
    combo = combine_strategies(ret_v12, ret_out, weight_v12, weight_out,
                                rebalance=rebalance, initial_capital=initial)

    if 'error' in combo:
        print(f"  ERROR combo: {combo['error']}")
        return None

    # También calcular v12 standalone y outsiders standalone sobre el mismo periodo
    v12_alone = combine_strategies(ret_v12, ret_out, 1.0, 0.0,
                                    rebalance='none', initial_capital=initial)
    out_alone = combine_strategies(ret_v12, ret_out, 0.0, 1.0,
                                    rebalance='none', initial_capital=initial)

    # Correlacion diaria
    common = ret_v12.index.intersection(ret_out.index)
    corr = ret_v12.reindex(common).corr(ret_out.reindex(common))

    # Imprimir resultados
    w_v12 = int(weight_v12 * 100)
    w_out = int(weight_out * 100)

    print(f"""
{'='*75}
  COMBO {w_v12}/{w_out} — v12 ({w_v12}%) + Outsiders ({w_out}%) — {months}m ({combo['years']:.1f}y)
  Rebalanceo: {rebalance}
  Correlacion diaria v12 vs Outsiders: {corr:.3f}
{'='*75}

  {'Estrategia':<30} {'CAGR':>8} {'MaxDD':>8} {'Vol':>8} {'Sharpe':>8} {'Effic':>8} {'Equity':>12}
  {'-'*88}
  {'v12 standalone':<30} {v12_alone.get('cagr',0):>7.1f}% {v12_alone.get('max_dd',0):>7.1f}% {v12_alone.get('vol_annual',0):>7.1f}% {v12_alone.get('sharpe',0):>8.2f} {v12_alone.get('efficiency',0):>8.2f} €{v12_alone.get('final_equity',0):>10,.0f}
  {'Outsiders standalone':<30} {out_alone.get('cagr',0):>7.1f}% {out_alone.get('max_dd',0):>7.1f}% {out_alone.get('vol_annual',0):>7.1f}% {out_alone.get('sharpe',0):>8.2f} {out_alone.get('efficiency',0):>8.2f} €{out_alone.get('final_equity',0):>10,.0f}
  {'COMBO ' + f'{w_v12}/{w_out}':<30} {combo['cagr']:>7.1f}% {combo['max_dd']:>7.1f}% {combo['vol_annual']:>7.1f}% {combo['sharpe']:>8.2f} {combo['efficiency']:>8.2f} €{combo['final_equity']:>10,.0f}
  {'-'*88}""")

    # Bonus: info componentes
    print(f"""
  Detalle componentes:
    v12:  CAGR {info_v12['cagr']:.1f}%, DD -{info_v12['dd']:.1f}%, PF {info_v12['pf']:.2f}, {info_v12['trades']} trades""", end='')
    if 'cagr_gold' in info_v12:
        print(f", Gold30%: CAGR {info_v12['cagr_gold']:.1f}%, DD -{info_v12['dd_gold']:.1f}%")
    else:
        print()
    print(f"    Outsiders: CAGR {info_out['cagr']:.1f}%, DD -{info_out['dd']:.1f}%, PF {info_out['pf']:.2f}, {info_out['trades']} trades")

    return {
        'combo': combo,
        'v12_alone': v12_alone,
        'out_alone': out_alone,
        'info_v12': info_v12,
        'info_out': info_out,
        'correlation': corr,
        'weight_v12': weight_v12,
        'weight_out': weight_out,
        'months': months,
    }


def run_multi_period(weight_v12=0.70, weight_out=0.30, rebalance='monthly'):
    """Test combo a multiples periodos."""
    periods = [36, 60, 120]
    results = []

    w_v12 = int(weight_v12 * 100)
    w_out = int(weight_out * 100)

    for m in periods:
        print(f"\n\n{'#'*75}")
        print(f"  PERIODO: {m} MESES ({m/12:.0f} años)")
        print(f"{'#'*75}")

        r = run_combo(m, weight_v12=weight_v12, weight_out=weight_out,
                      rebalance=rebalance, out_use_options=False)
        if r:
            results.append(r)

    if results:
        print(f"""

{'='*80}
  RESUMEN MULTI-PERIODO: COMBO {w_v12}/{w_out} (rebal. {rebalance})
{'='*80}

  {'Meses':>6} {'Años':>5} | {'v12 CAGR':>10} {'v12 DD':>8} | {'OUT CAGR':>10} {'OUT DD':>8} | {'COMBO CAGR':>11} {'COMBO DD':>9} {'Sharpe':>8} {'Corr':>6}
  {'-'*100}""")

        for r in results:
            m = r['months']
            v = r['v12_alone']
            o = r['out_alone']
            c = r['combo']
            print(f"  {m:>6} {m/12:>4.0f}y | {v['cagr']:>9.1f}% {v['max_dd']:>7.1f}% | {o['cagr']:>9.1f}% {o['max_dd']:>7.1f}% | {c['cagr']:>10.1f}% {c['max_dd']:>8.1f}% {c['sharpe']:>8.2f} {r['correlation']:>5.2f}")

        print(f"  {'-'*100}")


def run_weight_grid(months=120, rebalance='monthly'):
    """Grid de pesos: 90/10, 80/20, 70/30, 60/40, 50/50.
    OPTIMIZADO: descarga datos y corre backtests UNA sola vez, luego recombina.
    Si rebalance='all', muestra none + annual lado a lado."""
    weights = [(0.90, 0.10), (0.80, 0.20), (0.70, 0.30), (0.60, 0.40), (0.50, 0.50)]
    initial = CONFIG_V8['initial_capital']

    rebal_modes = ['none', 'annual'] if rebalance == 'all' else [rebalance]

    print(f"\n\n{'='*80}")
    print(f"  GRID DE PESOS — {months} meses — rebal. {', '.join(rebal_modes)}")
    print(f"  Ejecutando backtests UNA vez, luego recombinando...")
    print(f"{'='*80}")

    # 1) Correr cada backtest UNA sola vez
    result_v12, info_v12 = run_v12(months, use_gold=True, gold_pct=0.30)
    result_out, info_out = run_outsiders(months, use_options=False)

    if result_v12 is None or result_out is None:
        print("  ERROR: Una o ambas estrategias fallaron.")
        return

    # 2) Extraer equity curves y retornos UNA vez
    curve_v12 = equity_curve_from_result(result_v12, initial)
    curve_out = equity_curve_from_result(result_out, initial)
    ret_v12 = daily_returns_from_curve(curve_v12, initial)
    ret_out = daily_returns_from_curve(curve_out, initial)

    # Correlacion
    common = ret_v12.index.intersection(ret_out.index)
    corr = ret_v12.reindex(common).corr(ret_out.reindex(common))

    # v12 standalone (para referencia)
    v12_alone = combine_strategies(ret_v12, ret_out, 1.0, 0.0,
                                    rebalance='none', initial_capital=initial)
    out_alone = combine_strategies(ret_v12, ret_out, 0.0, 1.0,
                                    rebalance='none', initial_capital=initial)

    # 3) Recombinar con cada peso x cada modo de rebalanceo
    for rebal in rebal_modes:
        results = []
        for w_v12, w_out in weights:
            combo = combine_strategies(ret_v12, ret_out, w_v12, w_out,
                                        rebalance=rebal, initial_capital=initial)
            if 'error' not in combo:
                results.append({
                    'weight_v12': w_v12, 'weight_out': w_out,
                    'combo': combo,
                })

        if results:
            print(f"""

{'='*80}
  GRID — {months}m ({results[0]['combo']['years']:.1f}y) — rebal. {rebal.upper()}
  Correlacion v12 vs Outsiders: {corr:.3f}
{'='*80}

  {'Pesos':>10} | {'CAGR':>8} {'MaxDD':>8} {'Vol':>8} {'Sharpe':>8} {'Effic':>8} {'Equity':>12}
  {'-'*75}
  {'v12 solo':>10} | {v12_alone['cagr']:>7.1f}% {v12_alone['max_dd']:>7.1f}% {v12_alone['vol_annual']:>7.1f}% {v12_alone['sharpe']:>8.2f} {v12_alone['efficiency']:>8.2f} €{v12_alone['final_equity']:>10,.0f}
  {'-'*75}""")

            for r in results:
                w = f"{int(r['weight_v12']*100)}/{int(r['weight_out']*100)}"
                c = r['combo']
                print(f"  {w:>10} | {c['cagr']:>7.1f}% {c['max_dd']:>7.1f}% {c['vol_annual']:>7.1f}% {c['sharpe']:>8.2f} {c['efficiency']:>8.2f} €{c['final_equity']:>10,.0f}")

            print(f"  {'-'*75}")
            print(f"  {'OUT solo':>10} | {out_alone['cagr']:>7.1f}% {out_alone['max_dd']:>7.1f}% {out_alone['vol_annual']:>7.1f}% {out_alone['sharpe']:>8.2f} {out_alone['efficiency']:>8.2f} €{out_alone['final_equity']:>10,.0f}")
            print(f"  {'-'*75}")

    # Info componentes
    print(f"""
  Detalle componentes:
    v12:  CAGR {info_v12['cagr']:.1f}%, DD -{info_v12['dd']:.1f}%, PF {info_v12['pf']:.2f}, {info_v12['trades']} trades""", end='')
    if 'cagr_gold' in info_v12:
        print(f", Gold30%: CAGR {info_v12['cagr_gold']:.1f}%, DD -{info_v12['dd_gold']:.1f}%")
    else:
        print()
    print(f"    Outsiders: CAGR {info_out['cagr']:.1f}%, DD -{info_out['dd']:.1f}%, PF {info_out['pf']:.2f}, {info_out['trades']} trades")


def main():
    parser = argparse.ArgumentParser(
        description='Backtest Combo: 70%% v12 + 30%% Outsiders')
    parser.add_argument('--months', type=int, default=120,
                        help='Meses de historico (default: 120)')
    parser.add_argument('--weights', nargs=2, type=int, default=[70, 30],
                        help='Pesos v12/outsiders (default: 70 30)')
    parser.add_argument('--rebalance', choices=['monthly', 'annual', 'none', 'all'],
                        default='annual', help='Frecuencia rebalanceo (default: annual, all=none+annual)')
    parser.add_argument('--multi-period', action='store_true',
                        help='Test 36/60/120 meses')
    parser.add_argument('--grid', action='store_true',
                        help='Grid test de pesos: 90/10 a 60/40')
    parser.add_argument('--verbose', action='store_true',
                        help='Detalle de trades')
    parser.add_argument('--no-gold', action='store_true',
                        help='Sin gold overlay en v12')
    args = parser.parse_args()

    w_v12 = args.weights[0] / 100.0
    w_out = args.weights[1] / 100.0

    if abs(w_v12 + w_out - 1.0) > 0.01:
        print(f"  ERROR: Pesos deben sumar 100 (actual: {args.weights[0]}+{args.weights[1]})")
        return

    if args.multi_period:
        run_multi_period(w_v12, w_out, args.rebalance)
    elif args.grid:
        run_weight_grid(args.months, args.rebalance)
    else:
        run_combo(args.months, weight_v12=w_v12, weight_out=w_out,
                  rebalance=args.rebalance, use_gold_v12=not args.no_gold,
                  out_use_options=False, verbose=args.verbose)


if __name__ == '__main__':
    main()
