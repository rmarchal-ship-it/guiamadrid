#!/usr/bin/env python3
"""
Comparativa: 3 Estrategias de Inversion
========================================

Compara side-by-side (20 anos, periodo comun):
  1. Momentum Breakout v8 (activa, 225 tickers, 10 posiciones)
  2. 60/40 TQQQ/Gold (pasiva, rebalanceo anual)
  3. Multi-Asset optimizada: 45% Gold / 35% TLT / 20% TQQQ (rebalanceo anual)

Primera ejecucion: ejecuta los 3 backtests y guarda CSV.
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
CSV_FILE = os.path.join(BASE_DIR, "comparativa_3_estrategias.csv")

MONTHS = 240  # 20 anos


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

    equity = pd.Series(values, index=dates)
    print(f"  [60/40] Periodo: {dates[0].strftime('%Y-%m-%d')} a {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  [60/40] Final: ${equity.iloc[-1]:,.0f}")
    return equity


# =============================================================================
# 2. Multi-Asset: 45% Gold / 35% TLT / 20% TQQQ (optimizada)
# =============================================================================
def run_multi_asset(months=240):
    """Ejecuta la cartera multi-asset optimizada y retorna equity curve."""
    import yfinance as yf
    from dateutil.relativedelta import relativedelta

    print("  [Multi-Asset] Cargando datos...")

    # --- Datos ---
    GOLD_MONTHLY = {
        '1985-01': 303, '1985-02': 299, '1985-03': 304, '1985-04': 325,
        '1985-05': 317, '1985-06': 317, '1985-07': 317, '1985-08': 330,
        '1985-09': 323, '1985-10': 325, '1985-11': 325, '1985-12': 326,
        '1986-01': 345, '1986-02': 339, '1986-03': 346, '1986-04': 340,
        '1986-05': 342, '1986-06': 343, '1986-07': 349, '1986-08': 383,
        '1986-09': 418, '1986-10': 431, '1986-11': 399, '1986-12': 391,
        '1987-01': 408, '1987-02': 401, '1987-03': 408, '1987-04': 438,
        '1987-05': 461, '1987-06': 449, '1987-07': 451, '1987-08': 461,
        '1987-09': 461, '1987-10': 466, '1987-11': 470, '1987-12': 484,
        '1988-01': 477, '1988-02': 443, '1988-03': 444, '1988-04': 452,
        '1988-05': 451, '1988-06': 452, '1988-07': 437, '1988-08': 432,
        '1988-09': 413, '1988-10': 406, '1988-11': 420, '1988-12': 418,
        '1989-01': 404, '1989-02': 388, '1989-03': 390, '1989-04': 384,
        '1989-05': 371, '1989-06': 368, '1989-07': 375, '1989-08': 365,
        '1989-09': 362, '1989-10': 367, '1989-11': 394, '1989-12': 401,
        '1990-01': 410, '1990-02': 416, '1990-03': 393, '1990-04': 374,
        '1990-05': 367, '1990-06': 352, '1990-07': 362, '1990-08': 395,
        '1990-09': 389, '1990-10': 381, '1990-11': 381, '1990-12': 378,
        '1991-01': 383, '1991-02': 363, '1991-03': 363, '1991-04': 358,
        '1991-05': 357, '1991-06': 366, '1991-07': 368, '1991-08': 356,
        '1991-09': 349, '1991-10': 359, '1991-11': 360, '1991-12': 361,
        '1992-01': 354, '1992-02': 354, '1992-03': 345, '1992-04': 338,
        '1992-05': 337, '1992-06': 340, '1992-07': 353, '1992-08': 340,
        '1992-09': 345, '1992-10': 344, '1992-11': 335, '1992-12': 334,
        '1993-01': 329, '1993-02': 329, '1993-03': 330, '1993-04': 342,
        '1993-05': 368, '1993-06': 373, '1993-07': 392, '1993-08': 380,
        '1993-09': 355, '1993-10': 363, '1993-11': 374, '1993-12': 383,
        '1994-01': 387, '1994-02': 382, '1994-03': 384, '1994-04': 378,
        '1994-05': 381, '1994-06': 386, '1994-07': 386, '1994-08': 384,
        '1994-09': 394, '1994-10': 388, '1994-11': 384, '1994-12': 379,
        '1995-01': 379, '1995-02': 376, '1995-03': 382, '1995-04': 391,
        '1995-05': 386, '1995-06': 387, '1995-07': 386, '1995-08': 383,
        '1995-09': 383, '1995-10': 383, '1995-11': 386, '1995-12': 387,
        '1996-01': 399, '1996-02': 404, '1996-03': 397, '1996-04': 392,
        '1996-05': 392, '1996-06': 385, '1996-07': 384, '1996-08': 388,
        '1996-09': 383, '1996-10': 381, '1996-11': 378, '1996-12': 369,
        '1997-01': 354, '1997-02': 346, '1997-03': 352, '1997-04': 344,
        '1997-05': 344, '1997-06': 341, '1997-07': 325, '1997-08': 325,
        '1997-09': 323, '1997-10': 325, '1997-11': 306, '1997-12': 290,
        '1998-01': 289, '1998-02': 298, '1998-03': 296, '1998-04': 310,
        '1998-05': 300, '1998-06': 292, '1998-07': 293, '1998-08': 284,
        '1998-09': 291, '1998-10': 296, '1998-11': 294, '1998-12': 291,
        '1999-01': 287, '1999-02': 287, '1999-03': 286, '1999-04': 282,
        '1999-05': 277, '1999-06': 262, '1999-07': 256, '1999-08': 256,
        '1999-09': 265, '1999-10': 311, '1999-11': 293, '1999-12': 290,
    }

    ER_TQQQ = 0.0088
    ER_TLT = 0.0015
    ER_GOLD = 0.0040
    DURATION = 17
    NASDAQ_BETA = 1.25

    BORROWING_COST_BY_YEAR = {
        range(1985, 1990): 0.07, range(1990, 1995): 0.04,
        range(1995, 2000): 0.055, range(2000, 2005): 0.03,
        range(2005, 2008): 0.045, range(2008, 2016): 0.005,
        range(2016, 2020): 0.015, range(2020, 2022): 0.001,
        range(2022, 2027): 0.05,
    }

    def get_borrow(year):
        for r, c in BORROWING_COST_BY_YEAR.items():
            if year in r:
                return c
        return 0.03

    # Descargar datos
    sp500 = yf.download("^GSPC", start="1985-01-01", end="2026-12-31", progress=False)
    qqq_yf = yf.download("QQQ", start="1999-03-01", end="2010-02-28", progress=False)
    tqqq_yf = yf.download("TQQQ", start="2010-02-01", end="2026-12-31", progress=False)
    tlt_yf = yf.download("TLT", start="2002-07-01", end="2026-12-31", progress=False)
    tnx = yf.download("^TNX", start="1985-01-01", end="2002-08-01", progress=False)
    gold_yf = yf.download("GC=F", start="2000-01-01", end="2026-12-31", progress=False)

    for df in [sp500, qqq_yf, tqqq_yf, tlt_yf, tnx, gold_yf]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    sp500_close = sp500['Close'].dropna()

    # --- Oro ---
    gold_early = pd.Series(
        {pd.Timestamp(k + '-01'): float(v) for k, v in GOLD_MONTHLY.items()}
    ).sort_index()
    gold_early = gold_early.resample('B').interpolate(method='linear')
    gold_late = gold_yf['Close'].dropna()
    gold_close = pd.concat([gold_early.loc[:'1999-12-31'], gold_late.loc['2000-01-01':]])
    gold_close = gold_close[~gold_close.index.duplicated(keep='last')].sort_index()

    # --- TLT sintetico pre-2002 ---
    tlt_real = tlt_yf['Close'].dropna()
    tnx_close = tnx['Close'].dropna()
    yield_series = tnx_close / 100
    yield_change = yield_series.diff()
    bond_daily_ret = yield_series.shift(1) / 252 - DURATION * yield_change
    bond_daily_ret = bond_daily_ret.dropna().clip(-0.10, 0.10)
    tlt_synth_price = 100 * (1 + bond_daily_ret).cumprod()
    scale = tlt_real.iloc[0] / tlt_synth_price.iloc[-1]
    tlt_synth_price = tlt_synth_price * scale
    tlt_close = pd.concat([tlt_synth_price.loc[:'2002-07-29'], tlt_real.loc['2002-07-30':]])
    tlt_close = tlt_close[~tlt_close.index.duplicated(keep='last')].sort_index()

    # --- TQQQ sintetico pre-2010 ---
    tqqq_real = tqqq_yf['Close'].dropna()
    qqq_close = qqq_yf['Close'].dropna()
    sp500_pre99 = sp500_close.loc[:'1999-03-09']
    sp500_pre99_ret = sp500_pre99.pct_change().dropna()
    nasdaq_proxy_ret = sp500_pre99_ret * NASDAQ_BETA
    qqq_ret = qqq_close.pct_change().dropna()
    underlying_ret = pd.concat([nasdaq_proxy_ret, qqq_ret])
    underlying_ret = underlying_ret[~underlying_ret.index.duplicated(keep='last')].sort_index()

    tqqq_synth_ret = underlying_ret.copy()
    for idx in tqqq_synth_ret.index:
        yr = idx.year
        daily_borrow = get_borrow(yr) / 252
        tqqq_synth_ret.loc[idx] = 3.0 * underlying_ret.loc[idx] - ER_TQQQ / 252 - daily_borrow * 2

    tqqq_synth_price = 10 * (1 + tqqq_synth_ret).cumprod()
    scale_tqqq = tqqq_real.iloc[0] / tqqq_synth_price.iloc[-1]
    tqqq_synth_price = tqqq_synth_price * scale_tqqq
    tqqq_close = pd.concat([tqqq_synth_price.loc[:'2010-02-10'], tqqq_real.loc['2010-02-11':]])
    tqqq_close = tqqq_close[~tqqq_close.index.duplicated(keep='last')].sort_index()

    # --- Alinear series ---
    start = max(gold_close.index[0], sp500_close.index[0],
                tlt_close.index[0], tqqq_close.index[0])
    end = min(gold_close.index[-1], sp500_close.index[-1],
              tlt_close.index[-1], tqqq_close.index[-1])

    all_dates = sp500_close.loc[start:end].index
    data = pd.DataFrame(index=all_dates)
    data['Oro'] = gold_close.reindex(all_dates).ffill()
    data['TLT'] = tlt_close.reindex(all_dates).ffill()
    data['TQQQ'] = tqqq_close.reindex(all_dates).ffill()
    data = data.dropna()

    ret = data.pct_change().dropna()
    ret['Oro'] = ret['Oro'] - ER_GOLD / 252
    ret['TLT'] = ret['TLT'] - ER_TLT / 252

    # Cortar a periodo
    end_date = ret.index[-1]
    start_date = end_date - relativedelta(months=months)
    ret = ret.loc[start_date:]
    dates = ret.index

    # --- Simular 45% Gold / 35% TLT / 20% TQQQ ---
    W_GOLD, W_TLT, W_TQQQ = 0.45, 0.35, 0.20
    initial = 10000
    cur_g = initial * W_GOLD
    cur_t = initial * W_TLT
    cur_q = initial * W_TQQQ
    values = [initial]

    for i in range(len(dates)):
        cur_g *= (1 + ret['Oro'].iloc[i])
        cur_t *= (1 + ret['TLT'].iloc[i])
        cur_q *= (1 + ret['TQQQ'].iloc[i])
        total = cur_g + cur_t + cur_q

        if i > 0 and dates[i].year != dates[i-1].year:
            cur_g = total * W_GOLD
            cur_t = total * W_TLT
            cur_q = total * W_TQQQ

        values.append(total)

    equity = pd.Series(values[1:], index=dates)
    print(f"  [Multi-Asset] Periodo: {dates[0].strftime('%Y-%m-%d')} a {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  [Multi-Asset] Final: ${equity.iloc[-1]:,.0f}")
    return equity


# =============================================================================
# 3. Momentum Breakout v8
# =============================================================================
def run_v8(months=240):
    """Ejecuta backtest v8 y retorna equity curve como Series."""
    sys.path.insert(0, MOMENTUM_DIR)
    from backtest_experimental import run_backtest, BASE_TICKERS, CONFIG

    print(f"  [v8] Ejecutando backtest {months} meses...")
    result = run_backtest(months, BASE_TICKERS, "v8 (Opciones CALL)",
                          use_leverage_scaling=False, use_options=True, verbose=False)

    if 'error' in result:
        print(f"  [v8] ERROR: {result['error']}")
        return None

    ec = result.get('equity_curve', [])
    if ec:
        dates = [e[0] for e in ec]
        values = [e[1] for e in ec]
        equity_v8 = pd.Series([CONFIG['initial_capital']] + values,
                               index=[dates[0] - pd.Timedelta(days=1)] + dates)
        equity_v8 = equity_v8[~equity_v8.index.duplicated(keep='last')]
    else:
        trades = sorted(result['combined_trades'], key=lambda t: t.exit_date)
        equity = CONFIG['initial_capital']
        points = [(trades[0].entry_date, equity)]
        for t in trades:
            equity += t.pnl_euros
            points.append((t.exit_date, equity))
        equity_v8 = pd.Series([p[1] for p in points], index=[p[0] for p in points])

    print(f"  [v8] Final: EUR {equity_v8.iloc[-1]:,.0f}")
    print(f"  [v8] Trades: {result['total_trades']}, PF: {result['profit_factor']:.2f}")

    return equity_v8, result


# =============================================================================
# 4. Utilidades
# =============================================================================
def compute_year_by_year(equity_series, label):
    """Calcula retorno por ano desde una equity curve."""
    rows = []
    all_years = sorted(set(d.year for d in equity_series.index))
    for year in all_years:
        yd = equity_series[equity_series.index.map(lambda d: d.year == year)]
        if len(yd) < 2:
            continue
        ret = yd.iloc[-1] / yd.iloc[0] - 1
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


def compute_metrics(series, years):
    """Metricas de una equity curve."""
    total = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
    daily = series.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    rm = series.cummax()
    dd = (series - rm) / rm
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        'final': series.iloc[-1],
        'total_ret': total,
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
    }


# =============================================================================
# 5. Mostrar resultados
# =============================================================================
def show_results(df):
    """Muestra la comparativa formateada de las 3 estrategias."""

    print(f"\n{'=' * 115}")
    print("COMPARATIVA ANO POR ANO: Momentum v8 vs 60/40 TQQQ/Gold vs Multi-Asset (45/35/20)")
    print(f"{'=' * 115}")

    labels = ['v8', '60_40', 'multi']
    names = ['Momentum v8', '60/40 TQQQ/Gold', 'Multi-Asset']

    # Calcular acumulados
    for label in labels:
        col = f'{label}_return'
        if col in df.columns:
            df[f'{label}_acum'] = (1 + df[col].fillna(0)).cumprod() - 1

    print(f"\n  {'Ano':<7}", end="")
    for n in names:
        print(f" {n:>16}", end="")
    print(f" {'Ganador':>16}")
    print(f"  {'-' * 75}")

    for _, row in df.iterrows():
        year = int(row['year'])
        rets = {}
        print(f"  {year:<7}", end="")
        for label, name in zip(labels, names):
            r = row.get(f'{label}_return', np.nan)
            rets[name] = r
            r_s = f"{r:>15.2%}" if not np.isnan(r) else f"{'N/A':>15}"
            print(r_s, end=" ")

        # Ganador
        valid = {k: v for k, v in rets.items() if not np.isnan(v)}
        if valid:
            winner = max(valid, key=valid.get)
            # Abreviar nombre
            short = {'Momentum v8': 'v8', '60/40 TQQQ/Gold': '60/40', 'Multi-Asset': 'Multi'}
            print(f"{short.get(winner, winner):>16}")
        else:
            print(f"{'N/A':>16}")

    # === Estadisticas globales ===
    print(f"\n{'=' * 115}")
    print("ESTADISTICAS GLOBALES")
    print(f"{'=' * 115}")

    print(f"\n  {'Metrica':<35}", end="")
    for n in names:
        print(f" {n:>16}", end="")
    print()
    print(f"  {'-' * 85}")

    stats = {}
    for label, name in zip(labels, names):
        col = f'{label}_return'
        valid = df[col].dropna()
        neg = (valid < 0).sum()
        total = len(valid)
        gt50 = (valid > 0.50).sum()
        final = (1 + valid).prod()
        cagr = final ** (1 / total) - 1 if total > 0 else 0

        stats[name] = {
            'total': total,
            'neg': neg,
            'pct_neg': neg/total*100 if total > 0 else 0,
            'gt50': gt50,
            'media': valid.mean(),
            'mediana': valid.median(),
            'mejor': valid.max(),
            'peor': valid.min(),
            'std': valid.std(),
            'cagr': cagr,
        }

    metric_rows = [
        ('Anos totales', 'total', 'd'),
        ('Anos negativos', 'neg', 'd'),
        ('% anos negativos', 'pct_neg', '.1f', '%'),
        ('Anos > +50%', 'gt50', 'd'),
        ('Media retorno anual', 'media', '.2%'),
        ('Mediana retorno anual', 'mediana', '.2%'),
        ('Mejor ano', 'mejor', '.2%'),
        ('Peor ano', 'peor', '.2%'),
        ('Desv. estandar anual', 'std', '.2%'),
        ('CAGR', 'cagr', '.2%'),
    ]

    for row_def in metric_rows:
        label_name = row_def[0]
        key = row_def[1]
        fmt = row_def[2]
        suffix = row_def[3] if len(row_def) > 3 else ''
        print(f"  {label_name:<35}", end="")
        for name in names:
            val = stats[name].get(key, 0)
            if fmt == 'd':
                print(f" {int(val):>16}", end="")
            elif fmt == '.1f':
                print(f" {val:>15.1f}{suffix}", end="")
            else:
                print(f" {val:>16{fmt}}", end="")
        print()

    # Drawdowns intra-anuales
    print(f"\n  {'Peor DD intra-anual':<35}", end="")
    for label in labels:
        col = f'{label}_maxdd'
        if col in df.columns:
            dd = df[col].dropna()
            if len(dd) > 0:
                print(f" {dd.min():>16.2%}", end="")
            else:
                print(f" {'N/A':>16}", end="")
        else:
            print(f" {'N/A':>16}", end="")
    print()

    # Correlaciones entre estrategias
    print(f"\n{'=' * 115}")
    print("CORRELACIONES ENTRE ESTRATEGIAS (retornos anuales)")
    print(f"{'=' * 115}\n")

    corr_data = {}
    for label, name in zip(labels, names):
        col = f'{label}_return'
        if col in df.columns:
            corr_data[name] = df.set_index('year')[col]

    if len(corr_data) >= 2:
        corr_df = pd.DataFrame(corr_data).dropna()
        if len(corr_df) >= 3:
            print(corr_df.corr().round(2).to_string())

    # Resumen final
    print(f"\n{'=' * 115}")
    print("RESUMEN")
    print(f"{'=' * 115}")

    for name in names:
        s = stats[name]
        print(f"\n  {name}:")
        print(f"    CAGR: {s['cagr']:.2%} | Anos negativos: {s['neg']}/{s['total']} ({s['pct_neg']:.0f}%) | Peor ano: {s['peor']:.2%}")


# =============================================================================
# 6. Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Forzar re-ejecucion')
    args = parser.parse_args()

    if os.path.exists(CSV_FILE) and not args.force:
        print(f"Leyendo datos de {CSV_FILE} (usar --force para re-ejecutar)")
        df = pd.read_csv(CSV_FILE)
        show_results(df)
        return

    print("=" * 115)
    print("COMPARATIVA 3 ESTRATEGIAS")
    print(f"Periodo: {MONTHS} meses ({MONTHS//12} anos)")
    print("=" * 115)

    # Ejecutar las 3
    print("\n--- 60/40 TQQQ/Gold ---")
    equity_60_40 = run_60_40(MONTHS)

    print("\n--- Multi-Asset (45% Gold / 35% TLT / 20% TQQQ) ---")
    equity_multi = run_multi_asset(MONTHS)

    print("\n--- Momentum Breakout v8 ---")
    v8_result = run_v8(MONTHS)
    if v8_result is None:
        print("Error ejecutando v8. Continuando sin ella.")
        equity_v8 = None
    else:
        equity_v8, _ = v8_result

    # Year-by-year
    print("\n--- Calculando year-by-year ---")
    df_60 = compute_year_by_year(equity_60_40, '60_40')
    df_multi = compute_year_by_year(equity_multi, 'multi')

    if equity_v8 is not None:
        df_v8 = compute_year_by_year(equity_v8, 'v8')
        df = df_60.merge(df_multi, on='year', how='outer').merge(df_v8, on='year', how='outer')
    else:
        df = df_60.merge(df_multi, on='year', how='outer')

    df = df.sort_values('year')
    df.to_csv(CSV_FILE, index=False)
    print(f"\nGuardado: {CSV_FILE}")

    show_results(df)


if __name__ == "__main__":
    main()
