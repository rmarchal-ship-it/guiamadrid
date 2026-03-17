#!/usr/bin/env python3
"""
Comparativa: 60/40 TQQQ/Gold vs Momentum Breakout v8
=====================================================

Compara side-by-side:
  - 60/40 TQQQ/Gold (pasiva, rebalanceo anual)
  - Momentum Breakout v8 (activa, 225 tickers, 10 posiciones)

Focus: % de años negativos, perfil de riesgo/retorno.

Primera ejecución: ejecuta ambos backtests y guarda CSV.
Ejecuciones posteriores: lee CSV directamente (--force para re-ejecutar).
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOMENTUM_DIR = os.path.join(BASE_DIR, "estrategia_momentum")
CSV_FILE = os.path.join(BASE_DIR, "estrategia_60_40", "comparativa_year_by_year.csv")

MONTHS = 240

# =============================================================================
# 1. 60/40 TQQQ/Gold
# =============================================================================
def run_60_40(months=240):
    """Ejecuta 60/40 TQQQ/Gold y retorna serie de valores diarios."""
    import yfinance as yf
    import urllib.request
    import io
    from scipy.optimize import minimize_scalar
    from dateutil.relativedelta import relativedelta

    print("  [60/40] Cargando datos...")
    ndx = yf.download("^NDX", start="1985-01-01", end="2026-12-31", auto_adjust=True, progress=False)
    qqq = yf.download("QQQ", start="1999-03-01", end="2026-12-31", auto_adjust=True, progress=False)
    gld = yf.download("GLD", start="2004-11-01", end="2026-12-31", auto_adjust=True, progress=False)
    tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2026-12-31", auto_adjust=True, progress=False)

    for df in [ndx, qqq, gld, tqqq_real]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Gold composite
    url = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    gold_monthly = pd.read_csv(io.StringIO(resp.read().decode('utf-8')), parse_dates=['Date'])
    gold_monthly = gold_monthly.set_index('Date').sort_index()

    gold_daily_interp = gold_monthly['Price'].resample('B').interpolate(method='linear')
    gld_start = gld.index[0]
    gc_scale = gld["Close"].iloc[0] / gold_daily_interp.loc[:gld_start].iloc[-1]
    gold_pre = gold_daily_interp.loc[:gld_start] * gc_scale
    gold_composite = pd.concat([gold_pre.iloc[:-1], gld["Close"].loc[gld_start:]])
    gold_composite = gold_composite[~gold_composite.index.duplicated(keep='last')].sort_index()

    # Calibrar TQQQ
    qqq_ret = qqq["Close"].pct_change()
    ov = tqqq_real.index.intersection(qqq_ret.index)
    real_total = tqqq_real["Close"].loc[ov].iloc[-1] / tqqq_real["Close"].loc[ov].iloc[0]

    def sim_err(dc):
        sr = qqq_ret.loc[ov] * 3 - dc
        st = (1 + sr).cumprod()
        return (np.log(st.iloc[-1] / st.iloc[0]) - np.log(real_total)) ** 2

    daily_cost = minimize_scalar(sim_err, bounds=(0, 0.001), method='bounded').x

    ndx_ret = ndx["Close"].pct_change()
    qqq_start = qqq.index[0]
    nasdaq_ret = pd.concat([ndx_ret.loc[:qqq_start].iloc[:-1], qqq_ret.loc[qqq_start:]])
    nasdaq_ret = nasdaq_ret[~nasdaq_ret.index.duplicated(keep='last')].sort_index()

    tqqq_sim_ret = nasdaq_ret * 3 - daily_cost
    tqqq_sim = (1 + tqqq_sim_ret).cumprod()
    tqqq_sim.iloc[0] = 1.0

    common = tqqq_sim.index.intersection(gold_composite.index).sort_values()
    tqqq_p = tqqq_sim.loc[common] / tqqq_sim.loc[common].iloc[0]
    gold_p = gold_composite.loc[common] / gold_composite.loc[common].iloc[0]

    tqqq_d = tqqq_p.pct_change().fillna(0)
    gold_d = gold_p.pct_change().fillna(0)

    # Cortar a periodo
    end_date = common[-1]
    start_date = end_date - relativedelta(months=months)
    mask = (common >= start_date) & (common <= end_date)
    dates = common[mask]
    tqqq_ret_p = tqqq_d.loc[dates]
    gold_ret_p = gold_d.loc[dates]

    # Simular 60/40 con rebalanceo anual
    initial = 10000
    w = 0.60
    cur_t = initial * w
    cur_g = initial * (1 - w)
    values = [initial]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret_p.iloc[i])
        cur_g *= (1 + gold_ret_p.iloc[i])
        total = cur_t + cur_g
        if dates[i].year != dates[i-1].year:
            cur_t = total * w
            cur_g = total * (1 - w)
        values.append(total)

    equity_60_40 = pd.Series(values, index=dates)
    print(f"  [60/40] Periodo: {dates[0].strftime('%Y-%m-%d')} a {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  [60/40] Final: ${equity_60_40.iloc[-1]:,.0f} desde $10,000")

    return equity_60_40


# =============================================================================
# 2. Momentum Breakout v8
# =============================================================================
def run_v8(months=240):
    """Ejecuta backtest v8 y retorna equity curve como Series."""
    sys.path.insert(0, MOMENTUM_DIR)
    from backtest_experimental import run_backtest, BASE_TICKERS, CONFIG

    print(f"  [v8] Ejecutando backtest {months} meses (puede tardar varios minutos)...")
    result = run_backtest(months, BASE_TICKERS, "v8 (Opciones CALL)",
                          use_leverage_scaling=False, use_options=True, verbose=False)

    if 'error' in result:
        print(f"  [v8] ERROR: {result['error']}")
        return None

    # Construir equity curve desde tracker.equity_curve
    ec = result.get('equity_curve', [])
    if ec:
        dates = [e[0] for e in ec]
        values = [e[1] for e in ec]
        # Añadir punto inicial
        equity_v8 = pd.Series([CONFIG['initial_capital']] + values,
                               index=[dates[0] - pd.Timedelta(days=1)] + dates)
        # Eliminar duplicados
        equity_v8 = equity_v8[~equity_v8.index.duplicated(keep='last')]
    else:
        # Fallback: reconstruir desde trades
        print("  [v8] Reconstruyendo equity curve desde trades...")
        trades = sorted(result['combined_trades'], key=lambda t: t.exit_date)
        equity = CONFIG['initial_capital']
        points = [(trades[0].entry_date, equity)]
        for t in trades:
            equity += t.pnl_euros
            points.append((t.exit_date, equity))
        dates = [p[0] for p in points]
        values = [p[1] for p in points]
        equity_v8 = pd.Series(values, index=dates)

    print(f"  [v8] Final: EUR {equity_v8.iloc[-1]:,.0f} desde EUR 10,000")
    print(f"  [v8] Trades: {result['total_trades']}, PF: {result['profit_factor']:.2f}, MaxDD: -{result['max_drawdown']:.1f}%")

    return equity_v8, result


# =============================================================================
# 3. Year-by-year
# =============================================================================
def compute_year_by_year(equity_series, label):
    """Calcula retorno por año desde una equity curve."""
    rows = []
    all_years = sorted(set(d.year for d in equity_series.index))

    for year in all_years:
        yd = equity_series[equity_series.index.map(lambda d: d.year == year)]
        if len(yd) < 2:
            continue
        ret = yd.iloc[-1] / yd.iloc[0] - 1
        # Max drawdown del año
        rm = yd.cummax()
        dd = (yd - rm) / rm
        max_dd = dd.min()
        rows.append({
            'year': year,
            f'{label}_return': ret,
            f'{label}_maxdd': max_dd,
            f'{label}_start': yd.iloc[0],
            f'{label}_end': yd.iloc[-1],
        })

    return pd.DataFrame(rows)


# =============================================================================
# 4. Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Forzar re-ejecucion (ignorar CSV)')
    args = parser.parse_args()

    # Si existe CSV y no --force, leer directamente
    if os.path.exists(CSV_FILE) and not args.force:
        print(f"Leyendo datos de {CSV_FILE} (usar --force para re-ejecutar)")
        df = pd.read_csv(CSV_FILE)
        show_results(df)
        return

    print("=" * 80)
    print("COMPARATIVA: 60/40 TQQQ/Gold vs Momentum Breakout v8")
    print(f"Periodo: {MONTHS} meses")
    print("=" * 80)

    # Ejecutar 60/40
    print("\n--- 60/40 TQQQ/Gold ---")
    equity_60_40 = run_60_40(MONTHS)

    # Ejecutar v8
    print("\n--- Momentum Breakout v8 ---")
    v8_result = run_v8(MONTHS)
    if v8_result is None:
        print("Error ejecutando v8. Abortando.")
        return
    equity_v8, result_v8 = v8_result

    # Year-by-year
    print("\n--- Calculando year-by-year ---")
    df_60 = compute_year_by_year(equity_60_40, '60_40')
    df_v8 = compute_year_by_year(equity_v8, 'v8')

    # Merge
    df = pd.merge(df_60, df_v8, on='year', how='outer').sort_values('year')

    # Guardar CSV
    df.to_csv(CSV_FILE, index=False)
    print(f"\nGuardado: {CSV_FILE}")

    show_results(df)


def show_results(df):
    """Muestra la comparativa formateada."""

    print(f"\n{'=' * 90}")
    print("COMPARATIVA AÑO POR AÑO: 60/40 TQQQ/Gold vs Momentum Breakout v8")
    print(f"{'=' * 90}")

    # Calcular acumulados
    df['60_40_acum'] = (1 + df['60_40_return']).cumprod() - 1
    df['v8_acum'] = (1 + df['v8_return']).cumprod() - 1

    print(f"\n  {'Año':<7} {'60/40':>10} {'v8':>10} {'Ganador':>10} {'60/40 DD':>10} {'v8 DD':>10} {'60/40 acum':>12} {'v8 acum':>12}")
    print(f"  {'-' * 85}")

    for _, row in df.iterrows():
        year = int(row['year'])
        r60 = row.get('60_40_return', np.nan)
        rv8 = row.get('v8_return', np.nan)
        dd60 = row.get('60_40_maxdd', np.nan)
        ddv8 = row.get('v8_maxdd', np.nan)
        a60 = row.get('60_40_acum', np.nan)
        av8 = row.get('v8_acum', np.nan)

        r60_s = f"{r60:>9.2%}" if not np.isnan(r60) else f"{'N/A':>9}"
        rv8_s = f"{rv8:>9.2%}" if not np.isnan(rv8) else f"{'N/A':>9}"
        dd60_s = f"{dd60:>9.2%}" if not np.isnan(dd60) else f"{'N/A':>9}"
        ddv8_s = f"{ddv8:>9.2%}" if not np.isnan(ddv8) else f"{'N/A':>9}"
        a60_s = f"{a60:>11.1%}" if not np.isnan(a60) else f"{'N/A':>11}"
        av8_s = f"{av8:>11.1%}" if not np.isnan(av8) else f"{'N/A':>11}"

        if not np.isnan(r60) and not np.isnan(rv8):
            ganador = "60/40" if r60 > rv8 else "v8"
        elif not np.isnan(r60):
            ganador = "60/40"
        elif not np.isnan(rv8):
            ganador = "v8"
        else:
            ganador = "N/A"

        print(f"  {year:<7} {r60_s} {rv8_s} {ganador:>10} {dd60_s} {ddv8_s} {a60_s} {av8_s}")

    # Estadísticas
    print(f"\n{'=' * 90}")
    print("ESTADÍSTICAS")
    print(f"{'=' * 90}")

    valid_60 = df['60_40_return'].dropna()
    valid_v8 = df['v8_return'].dropna()

    neg_60 = (valid_60 < 0).sum()
    neg_v8 = (valid_v8 < 0).sum()
    total_60 = len(valid_60)
    total_v8 = len(valid_v8)

    gt50_60 = (valid_60 > 0.50).sum()
    gt50_v8 = (valid_v8 > 0.50).sum()

    print(f"\n  {'Métrica':<35} {'60/40':>12} {'v8':>12}")
    print(f"  {'-' * 62}")
    print(f"  {'Años totales':<35} {total_60:>12} {total_v8:>12}")
    print(f"  {'Años negativos':<35} {neg_60:>12} {neg_v8:>12}")
    print(f"  {'% años negativos':<35} {neg_60/total_60*100:>11.1f}% {neg_v8/total_v8*100:>11.1f}%")
    print(f"  {'Años > +50%':<35} {gt50_60:>12} {gt50_v8:>12}")
    print(f"  {'% años > +50%':<35} {gt50_60/total_60*100:>11.1f}% {gt50_v8/total_v8*100:>11.1f}%")
    print(f"  {'Media retorno anual':<35} {valid_60.mean():>11.2%} {valid_v8.mean():>11.2%}")
    print(f"  {'Mediana retorno anual':<35} {valid_60.median():>11.2%} {valid_v8.median():>11.2%}")
    print(f"  {'Mejor año':<35} {valid_60.max():>11.2%} {valid_v8.max():>11.2%}")
    print(f"  {'Peor año':<35} {valid_60.min():>11.2%} {valid_v8.min():>11.2%}")
    print(f"  {'Desv. estándar anual':<35} {valid_60.std():>11.2%} {valid_v8.std():>11.2%}")

    # CAGR
    if total_60 > 0:
        final_60 = (1 + valid_60).prod()
        cagr_60 = final_60 ** (1 / total_60) - 1
    else:
        cagr_60 = 0

    if total_v8 > 0:
        final_v8 = (1 + valid_v8).prod()
        cagr_v8 = final_v8 ** (1 / total_v8) - 1
    else:
        cagr_v8 = 0

    print(f"  {'CAGR (desde retornos anuales)':<35} {cagr_60:>11.2%} {cagr_v8:>11.2%}")

    # Peor drawdown anual
    dd_60 = df['60_40_maxdd'].dropna()
    dd_v8 = df['v8_maxdd'].dropna()
    print(f"  {'Peor DD intra-anual':<35} {dd_60.min():>11.2%} {dd_v8.min():>11.2%}")

    # Correlación
    common_years = df.dropna(subset=['60_40_return', 'v8_return'])
    if len(common_years) >= 3:
        corr = common_years['60_40_return'].corr(common_years['v8_return'])
        print(f"  {'Correlación retornos anuales':<35} {corr:>11.2f}")

    # Años donde gana cada una
    wins_60 = (common_years['60_40_return'] > common_years['v8_return']).sum()
    wins_v8 = (common_years['v8_return'] > common_years['60_40_return']).sum()
    print(f"  {'Años donde gana 60/40':<35} {wins_60:>12}")
    print(f"  {'Años donde gana v8':<35} {wins_v8:>12}")

    # Resumen final
    print(f"""

{'=' * 90}
RESUMEN
{'=' * 90}

  60/40 TQQQ/Gold (pasiva):
    - Estrategia mas simple posible: 60% TQQQ + 40% Gold, rebalanceo anual
    - CAGR: {cagr_60:.2%}
    - Años negativos: {neg_60}/{total_60} ({neg_60/total_60*100:.0f}%)
    - Peor año: {valid_60.min():.2%}

  Momentum Breakout v8 (activa):
    - 225 tickers, 10 posiciones, opciones CALL, filtro macro
    - CAGR: {cagr_v8:.2%}
    - Años negativos: {neg_v8}/{total_v8} ({neg_v8/total_v8*100:.0f}%)
    - Peor año: {valid_v8.min():.2%}
""")


if __name__ == "__main__":
    main()
