#!/usr/bin/env python3
"""
BACKTEST v12 — TEST DEFINITIVO 40 AÑOS (480 meses, ~1986-2026)

Fecha: 27 Feb 2026
Objetivo: Validar v12 a 40 años de datos historicos para confirmar robustez.

Cambios respecto a v12 estandar:
  - macro_ticker: ^GSPC (S&P 500 Index, datos desde 1927) en vez de SPY (desde 1993)
  - Gold overlay: GC=F (oro futuros, desde 2000) en vez de GLD (desde 2004)
  - Antes de 2000: gold overlay asume 0% retorno (conservador, como cash)

Uso:
  python3 backtest_v12_40y.py
  python3 backtest_v12_40y.py --verbose
  python3 backtest_v12_40y.py --no-gold
"""

import sys, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Importar del v12 y base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_experimental import (
    CONFIG, BASE_TICKERS, OPTIONS_ELIGIBLE,
    download_data, calculate_atr, historical_volatility,
)
from backtest_v12_eu_options import (
    run_backtest_eu, simulate_gold_overlay, print_comparison,
    OPTIONS_ELIGIBLE_EU, OPTIONS_ALL,
    US_SPREAD_PCT, EU_SPREAD_PCT,
)

MONTHS = 480  # 40 años
GOLD_TICKER = 'GC=F'       # Oro futuros (desde 2000)
MACRO_TICKER = '^GSPC'      # S&P 500 Index (desde 1927)
GOLD_PCT = 0.30             # 30% gold overlay


def download_long_data(ticker, months):
    """Download con start/end para periodos muy largos."""
    import yfinance as yf
    try:
        end = datetime.now()
        start = end - timedelta(days=months * 31)  # Margen extra
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                         end=end.strftime('%Y-%m-%d'), interval='1d', progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if len(df) >= 50 else None
    except Exception as e:
        print(f"  Error descargando {ticker}: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backtest v12 — Test 40 años')
    parser.add_argument('--verbose', action='store_true', help='Detalle de trades')
    parser.add_argument('--no-gold', action='store_true', help='Sin gold overlay')
    parser.add_argument('--months', type=int, default=MONTHS, help=f'Meses (default: {MONTHS})')
    args = parser.parse_args()

    months = args.months
    use_gold = not args.no_gold

    # =========================================================================
    # 1. CONFIGURAR MACRO TICKER
    # =========================================================================
    original_macro = CONFIG['macro_ticker']
    CONFIG['macro_ticker'] = MACRO_TICKER
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  BACKTEST v12 — TEST DEFINITIVO {months//12} AÑOS ({months} meses)           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Macro filter: {MACRO_TICKER} (S&P 500 Index, datos desde 1927)          ║
║  Gold overlay: {'GC=F 30% (oro futuros, datos desde 2000)' if use_gold else 'DESACTIVADO':42s}  ║
║  Tickers:      {len(BASE_TICKERS)} activos                                       ║
║  Opciones:     US ({len(OPTIONS_ELIGIBLE)}) + EU ({len(OPTIONS_ELIGIBLE_EU)}) — slots 2+2 separados          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # =========================================================================
    # 2. DESCARGAR ^GSPC (necesario para macro filter, no está en BASE_TICKERS)
    # =========================================================================
    print(f"  Descargando {MACRO_TICKER} para macro filter...")
    gspc_data = download_long_data(MACRO_TICKER, months + 6)
    if gspc_data is None:
        print(f"  ERROR: No se pudo descargar {MACRO_TICKER}. Abortando.")
        sys.exit(1)
    print(f"  {MACRO_TICKER}: {len(gspc_data)} días, "
          f"desde {gspc_data.index[0].strftime('%Y-%m-%d')} "
          f"hasta {gspc_data.index[-1].strftime('%Y-%m-%d')}")

    # =========================================================================
    # 3. DESCARGAR ORO (si gold overlay activo)
    # =========================================================================
    gold_data = None
    if use_gold:
        print(f"\n  Descargando {GOLD_TICKER} para gold overlay...")
        gold_data = download_long_data(GOLD_TICKER, months + 6)
        if gold_data is None:
            print(f"  WARNING: No se pudo descargar {GOLD_TICKER}. Gold overlay desactivado.")
            use_gold = False
        else:
            print(f"  {GOLD_TICKER}: {len(gold_data)} días, "
                  f"desde {gold_data.index[0].strftime('%Y-%m-%d')} "
                  f"hasta {gold_data.index[-1].strftime('%Y-%m-%d')}")
            # Pre-2000: sin datos de oro → retorno 0% (como cash)
            first_gold_year = gold_data.index[0].year
            if first_gold_year > 1990:
                print(f"  NOTA: Antes de {first_gold_year}, gold overlay = 0% return (cash equivalente)")

    # =========================================================================
    # 4. MONKEY-PATCH: inyectar ^GSPC en download_data
    # =========================================================================
    # Necesitamos parchear AMBOS módulos porque `from X import download_data`
    # crea una referencia local independiente en cada módulo
    import backtest_experimental as be
    import backtest_v12_eu_options as v12mod

    _original_download_be = be.download_data
    _original_download_v12 = v12mod.download_data if hasattr(v12mod, 'download_data') else _original_download_be

    def patched_download(ticker, m):
        if ticker == MACRO_TICKER:
            return gspc_data.copy()
        return _original_download_be(ticker, m)

    # Parchear en AMBOS módulos
    be.download_data = patched_download
    v12mod.download_data = patched_download

    # ^GSPC debe estar en la lista de tickers para que build_macro_filter lo encuentre
    tickers_extended = list(BASE_TICKERS)
    if MACRO_TICKER not in tickers_extended:
        tickers_extended.append(MACRO_TICKER)

    # =========================================================================
    # 5. EJECUTAR BACKTESTS
    # =========================================================================

    results = []

    # A) Referencia: v8 (2 slots US, 0 EU)
    print("\n" + "="*70)
    print("  EJECUTANDO REF: US2 + EU0 (referencia v8)")
    print("="*70)
    r_ref = run_backtest_eu(months, tickers_extended, "REF: US2 EU0",
                            use_options=True,
                            options_eligible_set=OPTIONS_ELIGIBLE,
                            max_us_options=2, max_eu_options=0,
                            verbose=args.verbose)
    if 'error' not in r_ref:
        results.append(r_ref)

    # B) v12: 2 slots US + 2 slots EU
    print("\n" + "="*70)
    print(f"  EJECUTANDO v12: US2 + EU2 (EU spread {EU_SPREAD_PCT}%)")
    print("="*70)
    r_eu2 = run_backtest_eu(months, tickers_extended, f"v12: US2+EU2 ({EU_SPREAD_PCT}%)",
                            use_options=True,
                            options_eligible_set=OPTIONS_ALL,
                            max_us_options=2, max_eu_options=2,
                            verbose=args.verbose)
    if 'error' not in r_eu2:
        results.append(r_eu2)

    # =========================================================================
    # 6. RESULTADOS BASE (sin gold)
    # =========================================================================
    if len(results) >= 2:
        print_comparison(results)

    # =========================================================================
    # 7. GOLD OVERLAY
    # =========================================================================
    if use_gold and gold_data is not None and len(results) >= 2:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  GOLD {GOLD_PCT*100:.0f}% OVERLAY ({GOLD_TICKER})                                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Periodo oro: {gold_data.index[0].strftime('%Y-%m-%d')} → {gold_data.index[-1].strftime('%Y-%m-%d')}                       ║
║  Antes de {gold_data.index[0].year}: 0% retorno oro (conservador)                  ║
╚══════════════════════════════════════════════════════════════════════╝
        """)

        g_ref = simulate_gold_overlay(r_ref, gold_data, GOLD_PCT)
        g_eu2 = simulate_gold_overlay(r_eu2, gold_data, GOLD_PCT)

        if g_ref and g_eu2:
            print(f"  {'Config':<28} {'Equity':>14}   {'CAGR':>8}   {'MaxDD':>7}   {'Eficiencia':>10}   {'Gold PnL':>12}")
            print(f"  {'-'*95}")

            # Sin gold
            years = months / 12
            for r in results:
                ann = r['annualized_return_pct']
                dd = r['max_drawdown']
                eff = ann / dd if dd > 0 else 0
                print(f"  {r['label']:<28} EUR {r['final_equity']:>14,.0f}   {ann:>+7.1f}%   "
                      f"{dd:>6.1f}%   {eff:>9.2f}    {'—':>12}")

            print(f"  {'-'*95}")

            # Con gold
            for label, g in [("REF + Gold 30%", g_ref), (f"v12 + Gold 30%", g_eu2)]:
                eff_g = g['ann_gold'] / g['maxdd_gold'] if g['maxdd_gold'] > 0 else 0
                print(f"  {label:<28} EUR {g['equity_gold']:>14,.0f}   {g['ann_gold']:>+7.1f}%   "
                      f"{g['maxdd_gold']:>6.1f}%   {eff_g:>9.2f}    EUR {g['gold_total_pnl']:>+12,.0f}")

            # Delta
            delta_cagr = g_eu2['ann_gold'] - g_ref['ann_gold']
            delta_dd = g_eu2['maxdd_gold'] - g_ref['maxdd_gold']
            print(f"\n  DELTA v12 vs REF (con Gold): CAGR {delta_cagr:+.1f}pp | MaxDD {delta_dd:+.1f}pp")

            # Comparativa 4 estrategias
            print(f"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  RESUMEN FINAL — {months//12} AÑOS ({months} meses)                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════╣""")

            strategies = []
            for r in results:
                strategies.append((r['label'], r['final_equity'], r['annualized_return_pct'], r['max_drawdown']))

            strategies.append(("REF + Gold 30%", g_ref['equity_gold'], g_ref['ann_gold'], g_ref['maxdd_gold']))
            strategies.append(("v12 + Gold 30%", g_eu2['equity_gold'], g_eu2['ann_gold'], g_eu2['maxdd_gold']))

            for name, eq, ann, dd in strategies:
                eff = ann / dd if dd > 0 else 0
                mult = eq / CONFIG['initial_capital']
                print(f"║  {name:<28} EUR {eq:>14,.0f}  CAGR {ann:>+7.1f}%  DD {dd:>6.1f}%  Eff {eff:.2f}  x{mult:>12,.0f} ║")

            print(f"╚══════════════════════════════════════════════════════════════════════════════════════╝")

    # Restaurar
    CONFIG['macro_ticker'] = original_macro
    be.download_data = _original_download_be
    v12mod.download_data = _original_download_v12


if __name__ == "__main__":
    main()
