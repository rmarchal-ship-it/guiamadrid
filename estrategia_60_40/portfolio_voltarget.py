#!/usr/bin/env python3
"""
Vol Target sweep: compare different target volatilities
for the 60% TQQQ / 40% Gold portfolio.

Focus on Vol Target 25% vs 20% vs baseline.
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
print("Loading data...")

ndx = yf.download("^NDX", start="1985-01-01", end="2025-12-31", auto_adjust=True, progress=False)
qqq = yf.download("QQQ", start="1999-03-01", end="2025-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2025-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2025-12-31", auto_adjust=True, progress=False)

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

common = tqqq_sim.index.intersection(gold_composite.index)
common = common.sort_values()

tqqq_p = tqqq_sim.loc[common] / tqqq_sim.loc[common].iloc[0]
gold_p = gold_composite.loc[common] / gold_composite.loc[common].iloc[0]

tqqq_d = tqqq_p.pct_change().fillna(0)
gold_d = gold_p.pct_change().fillna(0)

years = (common[-1] - common[0]).days / 365.25
print(f"Period: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")

# =============================================================================
# 2. Metrics
# =============================================================================
def metrics(series, yrs):
    total = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/yrs) - 1
    daily = series.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    rm = series.cummax()
    dd = (series - rm) / rm
    max_dd = dd.min()
    max_dd_date = dd.idxmin()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        "final": series.iloc[-1], "total_ret": total, "cagr": cagr,
        "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "max_dd_date": max_dd_date, "calmar": calmar,
    }

# =============================================================================
# 3. Strategies
# =============================================================================
def strat_baseline(dates, tqqq_ret, gold_ret, initial=10000):
    W = 0.60
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g
        if dates[i].year != dates[i-1].year:
            cur_t = total * W
            cur_g = total * (1 - W)
        values.append(total)
    return pd.Series(values, index=dates)


def strat_vol_target(dates, tqqq_ret, gold_ret, initial=10000,
                      target_vol=0.25, lookback=63, max_w=0.80, min_w=0.10):
    W = 0.60
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    weights = [W]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        check = d.month != pd_.month

        if check and i >= lookback:
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
# 4. Run sweep
# =============================================================================
print("\n" + "=" * 70)
print("VOL TARGET SWEEP")
print("=" * 70)

results = {}
weight_series = {}

results['Baseline 60/40'] = strat_baseline(common, tqqq_d, gold_d)

targets = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
for tv in targets:
    r, w = strat_vol_target(common, tqqq_d, gold_d, target_vol=tv)
    label = f'VolTgt {int(tv*100)}%'
    results[label] = r
    weight_series[label] = w

print(f"\n{'Strategy':<20} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8} {'Avg W':>7}")
print("-" * 95)

m_base = metrics(results['Baseline 60/40'], years)
print(f"{'Baseline 60/40':<20} ${m_base['final']:>12,.0f} {m_base['cagr']:>7.2%} {m_base['vol']:>7.2%} {m_base['sharpe']:>7.2f} {m_base['max_dd']:>8.2%} {m_base['calmar']:>7.2f} {'60.0%':>7}")

all_metrics = {'Baseline 60/40': m_base}
for tv in targets:
    label = f'VolTgt {int(tv*100)}%'
    m = metrics(results[label], years)
    all_metrics[label] = m
    avg_w = weight_series[label].mean()
    print(f"{label:<20} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f} {avg_w:>6.1%}")

# =============================================================================
# 5. Detail on 20% vs 25%
# =============================================================================
print("\n" + "=" * 70)
print("DETAIL: Vol Target 20% vs 25% vs Baseline")
print("=" * 70)

m20 = all_metrics['VolTgt 20%']
m25 = all_metrics['VolTgt 25%']

print(f"""
                    {'Baseline':>14} {'VolTgt 20%':>14} {'VolTgt 25%':>14}
  CAGR              {m_base['cagr']:>13.2%} {m20['cagr']:>13.2%} {m25['cagr']:>13.2%}
  Max DD            {m_base['max_dd']:>13.2%} {m20['max_dd']:>13.2%} {m25['max_dd']:>13.2%}
  Sharpe            {m_base['sharpe']:>13.2f} {m20['sharpe']:>13.2f} {m25['sharpe']:>13.2f}
  Calmar            {m_base['calmar']:>13.2f} {m20['calmar']:>13.2f} {m25['calmar']:>13.2f}
  Vol               {m_base['vol']:>13.2%} {m20['vol']:>13.2%} {m25['vol']:>13.2%}
  Final ($10k)      ${m_base['final']:>12,.0f} ${m20['final']:>12,.0f} ${m25['final']:>12,.0f}
  Avg TQQQ weight   {'60.0%':>13} {weight_series['VolTgt 20%'].mean():>12.1%} {weight_series['VolTgt 25%'].mean():>12.1%}
""")

# =============================================================================
# 6. Weight distribution for 20% and 25%
# =============================================================================
print("=" * 70)
print("TQQQ WEIGHT DISTRIBUTION")
print("=" * 70)

for label in ['VolTgt 20%', 'VolTgt 25%']:
    w = weight_series[label]
    print(f"\n  {label}:")
    print(f"    Min:    {w.min():.1%}")
    print(f"    p10:    {w.quantile(0.10):.1%}")
    print(f"    p25:    {w.quantile(0.25):.1%}")
    print(f"    Median: {w.quantile(0.50):.1%}")
    print(f"    p75:    {w.quantile(0.75):.1%}")
    print(f"    p90:    {w.quantile(0.90):.1%}")
    print(f"    Max:    {w.max():.1%}")
    print(f"    Mean:   {w.mean():.1%}")

    # Time in ranges
    print(f"    Time at min (10%):  {(w <= 0.11).sum()/len(w)*100:.1f}%")
    print(f"    Time at max (80%):  {(w >= 0.79).sum()/len(w)*100:.1f}%")
    print(f"    Time 10-30%:        {((w > 0.10) & (w <= 0.30)).sum()/len(w)*100:.1f}%")
    print(f"    Time 30-50%:        {((w > 0.30) & (w <= 0.50)).sum()/len(w)*100:.1f}%")
    print(f"    Time 50-80%:        {((w > 0.50) & (w <= 0.80)).sum()/len(w)*100:.1f}%")

# =============================================================================
# 7. Crisis performance
# =============================================================================
print("\n" + "=" * 70)
print("CRISIS PERFORMANCE")
print("=" * 70)

crisis_periods = [
    ("Black Monday 1987", "1987-09-01", "1988-06-30"),
    ("Dot-com crash", "2000-01-01", "2003-12-31"),
    ("Financial crisis 2008", "2007-10-01", "2009-12-31"),
    ("COVID crash 2020", "2020-01-01", "2020-12-31"),
    ("2022 bear market", "2021-11-01", "2023-06-30"),
]

compare = ['Baseline 60/40', 'VolTgt 20%', 'VolTgt 25%', 'VolTgt 30%']

for crisis_name, start, end in crisis_periods:
    print(f"\n--- {crisis_name} ---")
    crisis_dates = [d for d in common if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(crisis_dates) < 2:
        continue

    f, l = crisis_dates[0], crisis_dates[-1]

    # Show weight at crisis start for vol targets
    for label in ['VolTgt 20%', 'VolTgt 25%']:
        w_at_start = weight_series[label].loc[f]
        print(f"  {label} TQQQ weight at start: {w_at_start:.1%}")

    print(f"  {'Strategy':<20} {'Return':>10} {'Max DD':>10}")
    print(f"  {'-'*44}")
    for name in compare:
        if name in results:
            series = results[name]
            ret = series.loc[l] / series.loc[f] - 1
            crisis_s = series.loc[crisis_dates]
            rm = crisis_s.cummax()
            dd = (crisis_s - rm) / rm
            mdd = dd.min()
            print(f"  {name:<20} {ret:>9.2%} {mdd:>9.2%}")

# =============================================================================
# 8. Year-by-year
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR")
print("=" * 70)

yby = ['Baseline 60/40', 'VolTgt 20%', 'VolTgt 25%']
yby_labels = ['Baseline', 'VT20%', 'VT25%']

print(f"\n{'Year':<7}", end="")
for l in yby_labels:
    print(f" {l:>10}", end="")
print(f" {'W_20%':>7} {'W_25%':>7} {'Winner':>10}")
print("-" * 70)

all_years = sorted(set(d.year for d in common))
for year in all_years:
    yd = [d for d in common if d.year == year]
    if len(yd) < 2:
        continue
    f, l_d = yd[0], yd[-1]

    w20 = weight_series['VolTgt 20%'].loc[f]
    w25 = weight_series['VolTgt 25%'].loc[f]

    rets = {}
    print(f"{year:<7}", end="")
    for sname, label in zip(yby, yby_labels):
        r = results[sname].loc[l_d] / results[sname].loc[f] - 1
        rets[label] = r
        print(f" {r:>9.2%}", end="")

    print(f" {w20:>6.1%} {w25:>6.1%}", end="")
    best = max(rets, key=rets.get)
    print(f" {best:>10}")

# =============================================================================
# 9. Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Subir de Vol Target 20% a 25% significa:

  CAGR:       {m20['cagr']:.2%}  ->  {m25['cagr']:.2%}  (+{m25['cagr']-m20['cagr']:.2%})
  Max DD:     {m20['max_dd']:.2%}  ->  {m25['max_dd']:.2%}  ({m25['max_dd']-m20['max_dd']:+.2%})
  Sharpe:     {m20['sharpe']:.2f}  ->  {m25['sharpe']:.2f}
  Calmar:     {m20['calmar']:.2f}  ->  {m25['calmar']:.2f}
  Final:      ${m20['final']:>12,.0f}  ->  ${m25['final']:>12,.0f}
  Avg weight: {weight_series['VolTgt 20%'].mean():.1%}  ->  {weight_series['VolTgt 25%'].mean():.1%}

  El peso medio de TQQQ sube de {weight_series['VolTgt 20%'].mean():.1%} a {weight_series['VolTgt 25%'].mean():.1%}.
  Ganas +{m25['cagr']-m20['cagr']:.2%} de CAGR a cambio de {m25['max_dd']-m20['max_dd']:+.2%} más de drawdown.
""")
