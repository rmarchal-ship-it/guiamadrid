#!/usr/bin/env python3
"""
Sweep de asignación TQQQ/Gold: desde 80/20 hasta 20/80.
Se prueba a 36, 60 y 240 meses, con peso fijo y con Vol Target 20%.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
import io
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Load data
# =============================================================================
print("Cargando datos...")

ndx = yf.download("^NDX", start="1985-01-01", end="2026-12-31", auto_adjust=True, progress=False)
qqq = yf.download("QQQ", start="1999-03-01", end="2026-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2026-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2026-12-31", auto_adjust=True, progress=False)

for df in [ndx, qqq, gld, tqqq_real]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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

print(f"Datos disponibles: {common[0].strftime('%Y-%m-%d')} a {common[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 2. Funciones
# =============================================================================
def metrics(series):
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    total = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    daily = series.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    rm = series.cummax()
    dd = (series - rm) / rm
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {"final": series.iloc[-1], "total_ret": total, "cagr": cagr,
            "vol": vol, "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar}


def strat_baseline(dates, tqqq_ret, gold_ret, w_tqqq=0.60, initial=10000):
    cur_t = initial * w_tqqq
    cur_g = initial * (1 - w_tqqq)
    values = [initial]
    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g
        if dates[i].year != dates[i-1].year:
            cur_t = total * w_tqqq
            cur_g = total * (1 - w_tqqq)
        values.append(total)
    return pd.Series(values, index=dates)


def strat_vol_target(dates, tqqq_ret, gold_ret, w_tqqq=0.60,
                     target_vol=0.20, lookback=63, max_w=0.80, min_w=0.10,
                     initial=10000):
    W = w_tqqq
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    weights = [W]
    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g
        d, pd_ = dates[i], dates[i-1]
        if d.month != pd_.month and i >= lookback:
            recent_vol = tqqq_ret.iloc[i-lookback:i].std() * np.sqrt(252)
            if recent_vol > 0:
                ideal_w = target_vol / recent_vol
                W = np.clip(ideal_w, min_w, max_w)
            cur_t = total * W
            cur_g = total * (1 - W)
        elif d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)
        values.append(total)
        weights.append(W)
    return pd.Series(values, index=dates), pd.Series(weights, index=dates)


# =============================================================================
# 3. Periodos
# =============================================================================
allocations = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]

from dateutil.relativedelta import relativedelta

end_date = common[-1]
periods = {
    '36m': 36,
    '60m': 60,
    '240m': 240,
}

for period_label, months in periods.items():
    start_date = end_date - relativedelta(months=months)
    mask = (common >= start_date) & (common <= end_date)
    dates_p = common[mask]

    if len(dates_p) < 100:
        print(f"\n*** {period_label}: datos insuficientes, saltando ***")
        continue

    yrs_p = (dates_p[-1] - dates_p[0]).days / 365.25

    tqqq_ret_p = tqqq_d.loc[dates_p]
    gold_ret_p = gold_d.loc[dates_p]

    # ==== PESO FIJO ====
    print("\n" + "=" * 95)
    print(f"  {period_label.upper()} — {dates_p[0].strftime('%Y-%m-%d')} a {dates_p[-1].strftime('%Y-%m-%d')} ({yrs_p:.1f} años)")
    print("=" * 95)

    print(f"\n  PESO FIJO + rebalanceo anual:")
    print(f"  {'TQQQ/Gold':<12} {'Final ($10k)':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>9} {'Calmar':>8}")
    print(f"  {'-'*70}")

    base_r = {}
    for w in allocations:
        label = f"{int(w*100)}/{int((1-w)*100)}"
        s = strat_baseline(dates_p, tqqq_ret_p, gold_ret_p, w_tqqq=w)
        m = metrics(s)
        base_r[label] = m
        print(f"  {label:<12} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

    # ==== VOL TARGET 20% ====
    print(f"\n  VOL TARGET 20%:")
    print(f"  {'TQQQ/Gold':<12} {'Final ($10k)':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>9} {'Calmar':>8} {'AvgW':>7}")
    print(f"  {'-'*78}")

    vt_r = {}
    for w in allocations:
        label = f"{int(w*100)}/{int((1-w)*100)}"
        s, ws = strat_vol_target(dates_p, tqqq_ret_p, gold_ret_p, w_tqqq=w, target_vol=0.20)
        m = metrics(s)
        vt_r[label] = m
        avg_w = ws.mean()
        print(f"  {label:<12} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f} {avg_w:>6.1%}")

    # ==== COMPARATIVA ====
    print(f"\n  COMPARATIVA peso fijo vs VT20%:")
    print(f"  {'TQQQ/Gold':<12} {'--PESO FIJO--':^26}  {'--VOL TARGET 20%--':^26}")
    print(f"  {'':12} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8}  {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8}")
    print(f"  {'-'*78}")

    for w in allocations:
        label = f"{int(w*100)}/{int((1-w)*100)}"
        b = base_r[label]
        v = vt_r[label]
        print(f"  {label:<12} {b['cagr']:>7.2%} {b['max_dd']:>7.2%} {b['sharpe']:>7.2f}  {v['cagr']:>7.2%} {v['max_dd']:>7.2%} {v['sharpe']:>7.2f}")

    # ==== MEJORES ====
    best_sharpe_b = max(base_r, key=lambda k: base_r[k]['sharpe'])
    best_cagr_b = max(base_r, key=lambda k: base_r[k]['cagr'])
    best_sharpe_v = max(vt_r, key=lambda k: vt_r[k]['sharpe'])

    print(f"\n  GANADORES {period_label}:")
    b_bs = base_r[best_sharpe_b]
    b_bc = base_r[best_cagr_b]
    v_bs = vt_r[best_sharpe_v]
    print(f"    Peso fijo — Mejor CAGR:   {best_cagr_b} ({b_bc['cagr']:.2%}, DD {b_bc['max_dd']:.1%})")
    print(f"    Peso fijo — Mejor Sharpe: {best_sharpe_b} (Sharpe {b_bs['sharpe']:.2f}, CAGR {b_bs['cagr']:.2%})")
    print(f"    VT20%     — Mejor Sharpe: {best_sharpe_v} (Sharpe {v_bs['sharpe']:.2f}, CAGR {v_bs['cagr']:.2%})")

# =============================================================================
# 4. Resumen cross-period
# =============================================================================
print("\n" + "=" * 95)
print("  RESUMEN CROSS-PERIOD: 60/40 vs 50/50 vs 40/60")
print("=" * 95)

focus = [0.60, 0.50, 0.40]

print(f"\n  {'':16} {'--- 36 MESES ---':^24} {'--- 60 MESES ---':^24} {'--- 240 MESES ---':^24}")
print(f"  {'Estrategia':<16} {'CAGR':>8} {'MaxDD':>8} {'Shrp':>6} {'CAGR':>8} {'MaxDD':>8} {'Shrp':>6} {'CAGR':>8} {'MaxDD':>8} {'Shrp':>6}")
print(f"  {'-'*88}")

for w in focus:
    label = f"{int(w*100)}/{int((1-w)*100)}"

    row_base = f"  {label + ' fijo':<16}"
    row_vt = f"  {label + ' +VT20':<16}"

    for months in [36, 60, 240]:
        start_date = end_date - relativedelta(months=months)
        mask = (common >= start_date) & (common <= end_date)
        dates_p = common[mask]
        tqqq_ret_p = tqqq_d.loc[dates_p]
        gold_ret_p = gold_d.loc[dates_p]

        sb = strat_baseline(dates_p, tqqq_ret_p, gold_ret_p, w_tqqq=w)
        mb = metrics(sb)
        row_base += f" {mb['cagr']:>7.1%} {mb['max_dd']:>7.1%} {mb['sharpe']:>5.2f}"

        sv, _ = strat_vol_target(dates_p, tqqq_ret_p, gold_ret_p, w_tqqq=w, target_vol=0.20)
        mv = metrics(sv)
        row_vt += f" {mv['cagr']:>7.1%} {mv['max_dd']:>7.1%} {mv['sharpe']:>5.2f}"

    print(row_base)
    print(row_vt)
    print()
