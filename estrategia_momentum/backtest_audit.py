#!/usr/bin/env python3
"""
AUDIT BACKTEST v8 — Walk-Forward + Universo Dinamico (sin survivorship bias)

Test 1: Walk-Forward
  - IN-SAMPLE: primeros 70% de los datos (optimizacion)
  - OUT-OF-SAMPLE: ultimo 30% (validacion)
  - Compara PF in-sample vs out-of-sample

Test 2: Universo Dinamico (sin survivorship bias)
  - Solo usa tickers que existian en cada fecha
  - Excluye tickers pre-IPO/pre-lanzamiento
  - Compara con el backtest original (universo fijo)

Test 3: Robustez (quitar top N trades)
  - Ejecuta backtest completo
  - Recalcula PF quitando los 5, 10, 20 mejores trades

Uso:
  python3 backtest_audit.py --months 240 --test all
  python3 backtest_audit.py --months 240 --test walkforward
  python3 backtest_audit.py --months 240 --test survivorship
  python3 backtest_audit.py --months 240 --test robustness
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_experimental import (
    run_backtest, CONFIG, BASE_TICKERS, ASSETS,
    download_data, calculate_atr, historical_volatility,
    generate_all_signals, build_macro_filter, rank_candidates,
    find_candidates, EquityTracker, Trade, OptionTradeV2,
    MomentumEngine, black_scholes_call, monthly_expiration_dte,
    iv_rank, OPTIONS_ELIGIBLE,
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse


# =============================================================================
# TICKERS POST-2006 (excluir del universo dinamico antes de su fecha)
# =============================================================================

# Ticker -> ano en que empezo a cotizar (para excluir antes de esa fecha)
TICKER_START_YEAR = {
    # ETFs post-2006
    'DBB': 2007,
    'DBA': 2007,
    'HYG': 2007,
    'EMB': 2008,    # Dec 2007, conservador -> 2008
    'SPXL': 2009,   # Nov 2008 -> dar margen
    'TNA': 2009,
    'GDXJ': 2010,
    'TQQQ': 2010,
    'BNO': 2010,
    'CORN': 2010,
    'CPER': 2011,
    'WEAT': 2011,
    'SOYB': 2011,
    'INDA': 2012,
    'PICK': 2012,
    'SMH': 2012,    # Relanzado VanEck Dec 2011
    'XBI': 2006,    # Jan 2006, marginal
    'BITO': 2021,

    # Stocks post-2006
    'AVGO': 2009,
    'TSLA': 2010,
    'META': 2012,
    'NOW': 2012,
    'TMUS': 2013,   # Como T-Mobile US (MetroPCS desde 2007)
    'JD': 2014,
    'BABA': 2014,
    'PDD': 2018,
}

# Tickers que SIEMPRE existieron (pre-2006) -> todos los demas


# =============================================================================
# TEST 1: WALK-FORWARD
# =============================================================================

def test_walk_forward(months, verbose=False):
    """
    Walk-forward: divide los datos en IN-SAMPLE (70%) y OUT-OF-SAMPLE (30%).
    Ejecuta el backtest con los MISMOS parametros en ambos periodos.
    Si PF OOS es similar al IS -> estrategia robusta.
    Si PF OOS << IS -> overfitting.
    """
    print(f"""
{'='*70}
  TEST WALK-FORWARD — {months} MESES
{'='*70}
  IN-SAMPLE:  primeros 70% ({int(months*0.7)} meses)
  OUT-OF-SAMPLE: ultimo 30% ({int(months*0.3)} meses)
  Parametros: IDENTICOS en ambos periodos (sin re-optimizacion)
{'='*70}
""")

    # Periodo IS
    is_months = int(months * 0.7)
    oos_months = months - is_months

    print(f"  === IN-SAMPLE: {is_months} meses ===")
    r_is = run_backtest(is_months, BASE_TICKERS, f"IN-SAMPLE ({is_months}m)",
                        use_options=True, verbose=verbose)

    print(f"\n  === OUT-OF-SAMPLE: {oos_months} meses (los mas recientes) ===")
    r_oos = run_backtest(oos_months, BASE_TICKERS, f"OUT-OF-SAMPLE ({oos_months}m)",
                         use_options=True, verbose=verbose)

    if 'error' in r_is or 'error' in r_oos:
        print("  ERROR: No se pudieron ejecutar ambos periodos")
        return

    # Comparar
    print(f"""
{'='*70}
  RESULTADO WALK-FORWARD
{'='*70}

  {'Metrica':<25} {'IN-SAMPLE':>12} {'OUT-OF-SAMPLE':>14} {'Ratio OOS/IS':>14}
  {'-'*65}
  {'Trades':<25} {r_is['total_trades']:>12} {r_oos['total_trades']:>14}
  {'Win Rate':<25} {r_is['win_rate']:>11.1f}% {r_oos['win_rate']:>13.1f}%
  {'Profit Factor':<25} {r_is['profit_factor']:>12.2f} {r_oos['profit_factor']:>14.2f} {r_oos['profit_factor']/r_is['profit_factor']:>13.1%}
  {'Return Total':<25} {r_is['total_return_pct']:>+11.1f}% {r_oos['total_return_pct']:>+13.1f}%
  {'Anualizado':<25} {r_is['annualized_return_pct']:>+11.1f}% {r_oos['annualized_return_pct']:>+13.1f}%
  {'Max Drawdown':<25} {r_is['max_drawdown']:>11.1f}% {r_oos['max_drawdown']:>13.1f}%
""")

    pf_ratio = r_oos['profit_factor'] / r_is['profit_factor'] if r_is['profit_factor'] > 0 else 0
    if pf_ratio >= 0.8:
        print("  VEREDICTO: ✅ PF OOS >= 80% del IS → Estrategia ROBUSTA")
    elif pf_ratio >= 0.5:
        print("  VEREDICTO: ⚠️ PF OOS 50-80% del IS → Overfitting MODERADO")
    else:
        print("  VEREDICTO: ❌ PF OOS < 50% del IS → Overfitting SEVERO")

    return {'is': r_is, 'oos': r_oos, 'pf_ratio': pf_ratio}


# =============================================================================
# TEST 2: UNIVERSO DINAMICO / PROGRESIVO (sin survivorship bias)
# =============================================================================

def get_dynamic_universe(year):
    """Devuelve solo los tickers que existian en ese ano."""
    valid = []
    for ticker in BASE_TICKERS:
        start_year = TICKER_START_YEAR.get(ticker, 2000)  # default: existia antes de 2006
        if year >= start_year:
            valid.append(ticker)
    return valid


def run_backtest_progressive(months, tickers, label, ticker_start_years,
                              use_options=True, verbose=False):
    """
    Backtest con universo PROGRESIVO: descarga todos los tickers pero solo
    permite entradas en tickers que ya existian en la fecha actual del loop.
    Usa monkey-patching de find_candidates para filtrar por ano de IPO.
    """
    import backtest_experimental as be

    # Guardar original
    original_find_candidates = be.find_candidates

    # Variable para rastrear la fecha actual del loop
    _state = {'current_year': 2000}

    def find_candidates_progressive(signals_data, active_trades, current_date,
                                     is_macro_ok, macro_exempt_set=None):
        """Wrapper que filtra candidatos cuyo ticker no existia aun."""
        candidates = original_find_candidates(
            signals_data, active_trades, current_date,
            is_macro_ok, macro_exempt_set
        )
        current_year = current_date.year
        filtered = []
        for c in candidates:
            ticker = c[0]
            start_year = ticker_start_years.get(ticker, 2000)
            if current_year >= start_year:
                filtered.append(c)
        return filtered

    # Monkey-patch
    be.find_candidates = find_candidates_progressive

    try:
        result = run_backtest(months, tickers, label,
                              use_options=use_options, verbose=verbose)
    finally:
        # Restaurar siempre
        be.find_candidates = original_find_candidates

    return result


def test_survivorship(months, verbose=False):
    """
    Compara:
    A) Backtest original: 225 tickers fijos (survivorship bias)
    B) Backtest PROGRESIVO: tickers se incorporan al universo segun su IPO
       (descarga todos, pero solo genera entradas en tickers que ya existian)
    """
    print(f"""
{'='*70}
  TEST SURVIVORSHIP BIAS — {months} MESES
{'='*70}
  A) Universo FIJO: {len(BASE_TICKERS)} tickers (original, con bias)
  B) Universo PROGRESIVO: tickers se incorporan segun su IPO
     (descarga todos, filtra entradas por fecha de existencia)
{'='*70}
""")

    # Contar tickers excluidos por periodo
    excluded = {t: y for t, y in TICKER_START_YEAR.items() if t in BASE_TICKERS}
    print(f"  Tickers con survivorship bias (post-2006): {len(excluded)}")
    for t, y in sorted(excluded.items(), key=lambda x: x[1]):
        name = ASSETS.get(t, {}).get('name', t)
        print(f"    {t:12} ({name}) — disponible desde {y}")

    # Mostrar evolucion del universo
    print(f"\n  Evolucion del universo:")
    for year in range(2006, datetime.now().year + 1, 2):
        n = len(get_dynamic_universe(year))
        bar = '#' * (n // 5)
        print(f"    {year}: {n:3} tickers {bar}")

    # A) Original
    print(f"\n  === A) UNIVERSO FIJO (original, {len(BASE_TICKERS)} tickers) ===")
    r_fixed = run_backtest(months, BASE_TICKERS, "FIJO (225 tickers)",
                           use_options=True, verbose=verbose)

    # B) Progresivo: descarga todos, filtra por IPO
    print(f"\n  === B) UNIVERSO PROGRESIVO (225 descargados, filtrado por IPO) ===")
    r_progressive = run_backtest_progressive(
        months, BASE_TICKERS,
        f"PROGRESIVO (filtrado por IPO)",
        TICKER_START_YEAR,
        use_options=True, verbose=verbose
    )

    if 'error' in r_fixed or 'error' in r_progressive:
        print("  ERROR: No se pudieron ejecutar ambos tests")
        return

    # Comparar
    bias_pf = r_fixed['profit_factor'] - r_progressive['profit_factor']
    bias_ret = r_fixed['annualized_return_pct'] - r_progressive['annualized_return_pct']

    print(f"""
{'='*70}
  RESULTADO SURVIVORSHIP BIAS (PROGRESIVO)
{'='*70}

  {'Metrica':<25} {'FIJO (225)':>12} {'PROGRESIVO':>14} {'Diferencia':>14}
  {'-'*65}
  {'Trades':<25} {r_fixed['total_trades']:>12} {r_progressive['total_trades']:>14} {r_fixed['total_trades']-r_progressive['total_trades']:>+14}
  {'Win Rate':<25} {r_fixed['win_rate']:>11.1f}% {r_progressive['win_rate']:>13.1f}% {r_fixed['win_rate']-r_progressive['win_rate']:>+13.1f}%
  {'Profit Factor':<25} {r_fixed['profit_factor']:>12.2f} {r_progressive['profit_factor']:>14.2f} {bias_pf:>+14.2f}
  {'Anualizado':<25} {r_fixed['annualized_return_pct']:>+11.1f}% {r_progressive['annualized_return_pct']:>+13.1f}% {bias_ret:>+13.1f}%
  {'Max Drawdown':<25} {r_fixed['max_drawdown']:>11.1f}% {r_progressive['max_drawdown']:>13.1f}%
""")

    print(f"  SURVIVORSHIP BIAS en PF: {bias_pf:+.2f}")
    print(f"  SURVIVORSHIP BIAS en retorno anual: {bias_ret:+.1f}%")

    if abs(bias_pf) < 0.3:
        print("  VEREDICTO: ✅ Bias < 0.3 PF → Impacto BAJO")
    elif abs(bias_pf) < 0.7:
        print("  VEREDICTO: ⚠️ Bias 0.3-0.7 PF → Impacto MODERADO")
    else:
        print("  VEREDICTO: ❌ Bias > 0.7 PF → Impacto ALTO, resultados NO fiables")

    return {'fixed': r_fixed, 'progressive': r_progressive, 'bias_pf': bias_pf}


# =============================================================================
# TEST 3: ROBUSTEZ (quitar top N trades)
# =============================================================================

def test_robustness(months, verbose=False):
    """
    Ejecuta backtest completo y recalcula metricas quitando los mejores trades.
    Si el PF se desploma al quitar 10 trades de 1400+ -> dependencia de outliers.
    """
    print(f"""
{'='*70}
  TEST ROBUSTEZ — {months} MESES
{'='*70}
  Ejecuta backtest completo y recalcula PF quitando los N mejores trades.
  Mide dependencia de fat tails / home runs.
{'='*70}
""")

    r = run_backtest(months, BASE_TICKERS, "BASELINE para robustez",
                     use_options=True, verbose=False)

    if 'error' in r:
        print("  ERROR: No se pudo ejecutar")
        return

    all_trades = r.get('combined_trades', [])
    if not all_trades:
        print("  ERROR: No hay trades para analizar")
        return

    # Ordenar por PnL descendente
    sorted_trades = sorted(all_trades, key=lambda t: t.pnl_euros, reverse=True)

    total_pnl = sum(t.pnl_euros for t in all_trades)
    total_count = len(all_trades)

    print(f"\n  Total trades: {total_count}")
    print(f"  PnL total: EUR {total_pnl:+,.0f}")
    print(f"  PF original: {r['profit_factor']:.2f}")

    # Top 10 trades
    print(f"\n  TOP 10 TRADES (mayores ganadores):")
    print(f"  {'#':<4} {'Ticker':<10} {'PnL EUR':>10} {'PnL %':>8} {'Exit Reason':<15}")
    print(f"  {'-'*50}")
    top10_pnl = 0
    for i, t in enumerate(sorted_trades[:10], 1):
        top10_pnl += t.pnl_euros
        print(f"  {i:<4} {t.ticker:<10} {t.pnl_euros:>+10,.0f} {t.pnl_pct:>+7.1f}% {t.exit_reason or 'n/a':<15}")

    print(f"\n  Top 10 contribuyen: EUR {top10_pnl:+,.0f} ({top10_pnl/total_pnl*100:.1f}% del PnL total)")

    # Recalcular PF quitando top N
    print(f"\n  {'Quitar top N':<15} {'Trades':>8} {'PnL EUR':>12} {'Win%':>7} {'PF':>7} {'PF vs orig':>12}")
    print(f"  {'-'*65}")

    for n in [0, 5, 10, 20, 30, 50]:
        remaining = sorted_trades[n:]
        if not remaining:
            break
        winners = [t for t in remaining if t.pnl_euros > 0]
        losers = [t for t in remaining if t.pnl_euros <= 0]
        gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
        pf = gross_profit / gross_loss
        wr = len(winners) / len(remaining) * 100
        pnl = sum(t.pnl_euros for t in remaining)
        pf_pct = pf / r['profit_factor'] * 100 if r['profit_factor'] > 0 else 0

        label = "Original" if n == 0 else f"Sin top {n}"
        print(f"  {label:<15} {len(remaining):>8} {pnl:>+12,.0f} {wr:>6.1f}% {pf:>7.2f} {pf_pct:>11.0f}%")

    # Bottom 10 (peores trades)
    print(f"\n  BOTTOM 10 TRADES (mayores perdedores):")
    print(f"  {'#':<4} {'Ticker':<10} {'PnL EUR':>10} {'PnL %':>8} {'Exit Reason':<15}")
    print(f"  {'-'*50}")
    for i, t in enumerate(sorted_trades[-10:], 1):
        print(f"  {i:<4} {t.ticker:<10} {t.pnl_euros:>+10,.0f} {t.pnl_pct:>+7.1f}% {t.exit_reason or 'n/a':<15}")

    return r


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Audit Backtest v8')
    parser.add_argument('--months', type=int, default=240, help='Meses de historico')
    parser.add_argument('--test', choices=['all', 'walkforward', 'survivorship', 'robustness'],
                        default='all', help='Test a ejecutar')
    parser.add_argument('--verbose', action='store_true', help='Detalle de trades')
    args = parser.parse_args()

    print(f"""
======================================================================
  AUDIT BACKTEST v8 — MOMENTUM BREAKOUT
======================================================================
  Test 1: Walk-Forward (IS vs OOS)
  Test 2: Survivorship Bias (universo dinamico)
  Test 3: Robustez (quitar top N trades)
======================================================================
""")

    results = {}

    if args.test in ('all', 'walkforward'):
        results['walkforward'] = test_walk_forward(args.months, args.verbose)

    if args.test in ('all', 'survivorship'):
        results['survivorship'] = test_survivorship(args.months, args.verbose)

    if args.test in ('all', 'robustness'):
        results['robustness'] = test_robustness(args.months, args.verbose)

    # Resumen final
    if len(results) > 1:
        print(f"""
{'='*70}
  RESUMEN FINAL DE AUDITORIA
{'='*70}
""")
        if 'walkforward' in results and results['walkforward']:
            wf = results['walkforward']
            print(f"  Walk-Forward: PF IS={wf['is']['profit_factor']:.2f} → "
                  f"OOS={wf['oos']['profit_factor']:.2f} "
                  f"(ratio {wf['pf_ratio']:.1%})")

        if 'survivorship' in results and results['survivorship']:
            sv = results['survivorship']
            print(f"  Survivorship: PF fijo={sv['fixed']['profit_factor']:.2f} → "
                  f"progresivo={sv['progressive']['profit_factor']:.2f} "
                  f"(bias {sv['bias_pf']:+.2f})")

        if 'robustness' in results and results['robustness']:
            print(f"  Robustez: Ver tabla de sensibilidad arriba")


if __name__ == "__main__":
    main()
