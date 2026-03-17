#!/usr/bin/env python3
"""
ROC Deceleration Strategy for 60% TQQQ / 40% Gold

Signal: NDX ROC acceleration (2nd derivative of price)
  - 6-month ROC minus 6-month ROC from 6 months ago
  - When negative = market decelerating (still rising but slowing)
  - This was the ONLY signal with clean separation between all 3 pre-crash
    periods (2000, 2007, 2021) and all bull markets

Strategy:
  - ROC accel >= 0: hold 60% TQQQ / 40% Gold (normal)
  - ROC accel < 0:  hold 40% TQQQ / 60% Gold (defensive)
  - Check monthly, annual rebalancing within current allocation

Also test variants:
  - Different defensive allocations (20%, 30%, 40% TQQQ)
  - Different ROC lookback windows (3m, 6m, 9m, 12m)
  - ROC < threshold instead of < 0 (small buffer)
  - With/without hysteresis
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
# 1. Load all data
# =============================================================================
print("Loading data...")

ndx = yf.download("^NDX", start="1985-01-01", end="2025-12-31", auto_adjust=True, progress=False)
qqq = yf.download("QQQ", start="1999-03-01", end="2025-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2025-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2025-12-31", auto_adjust=True, progress=False)

for df in [ndx, qqq, gld, tqqq_real]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

# Gold
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

# Calibrate TQQQ
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

# Nasdaq price
ndx_price = ndx["Close"].copy()
qqq_price = qqq["Close"].copy()
scale_factor = qqq_price.iloc[0] / ndx_price.loc[:qqq_start].iloc[-1]
ndx_pre = ndx_price.loc[:qqq_start] * scale_factor
nasdaq_price = pd.concat([ndx_pre.iloc[:-1], qqq_price.loc[qqq_start:]])
nasdaq_price = nasdaq_price[~nasdaq_price.index.duplicated(keep='last')].sort_index()

# Align
common = tqqq_sim.index.intersection(gold_composite.index).intersection(nasdaq_price.index)
common = common.sort_values()

tqqq_p = tqqq_sim.loc[common] / tqqq_sim.loc[common].iloc[0]
gold_p = gold_composite.loc[common] / gold_composite.loc[common].iloc[0]
nasdaq_px = nasdaq_price.loc[common]

tqqq_d = tqqq_p.pct_change().fillna(0)
gold_d = gold_p.pct_change().fillna(0)

years = (common[-1] - common[0]).days / 365.25
print(f"Period: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")

# =============================================================================
# 2. Compute ROC acceleration variants
# =============================================================================
print("\n" + "=" * 70)
print("COMPUTING ROC SIGNALS...")
print("=" * 70)

# ROC acceleration = ROC(now) - ROC(N months ago)
# ROC = price / price_N_months_ago - 1

def compute_roc_accel(prices, half_period_days):
    """
    Acceleration of momentum:
    roc_accel = ROC(t, half_period) - ROC(t - half_period, half_period)
    where ROC(t, n) = price(t) / price(t-n) - 1
    """
    roc = prices / prices.shift(half_period_days) - 1
    roc_accel = roc - roc.shift(half_period_days)
    return roc_accel

# Standard: 6-month half-period (126 trading days)
roc_accel_6m = compute_roc_accel(nasdaq_px, 126)

# Also try other half-periods
roc_accel_3m = compute_roc_accel(nasdaq_px, 63)
roc_accel_9m = compute_roc_accel(nasdaq_px, 189)
roc_accel_12m = compute_roc_accel(nasdaq_px, 252)

# Show values at key dates
print(f"\nROC Acceleration at key moments:")
print(f"{'Date':<22} {'3m':>8} {'6m':>8} {'9m':>8} {'12m':>8}")
print("-" * 55)

key_dates = [
    ("Pre-Dotcom 1999-06", "1999-06-01"),
    ("Dotcom peak 2000-03", "2000-03-01"),
    ("GFC 2007-06", "2007-06-01"),
    ("GFC peak 2007-10", "2007-10-01"),
    ("Pre-2022 2021-06", "2021-06-01"),
    ("2021 peak 2021-11", "2021-11-01"),
    ("Bull 1995-06", "1995-06-01"),
    ("Bull 2013-06", "2013-06-01"),
    ("Bull 2017-06", "2017-06-01"),
    ("Bull 2019-06", "2019-06-01"),
    ("Bull 2024-06", "2024-06-01"),
    ("Current", common[-1].strftime('%Y-%m-%d')),
]

for name, date_str in key_dates:
    d = pd.Timestamp(date_str)
    idx = common.searchsorted(d)
    if idx < len(common):
        dd = common[idx]
        v3 = roc_accel_3m.loc[dd] if dd in roc_accel_3m.index and not np.isnan(roc_accel_3m.loc[dd]) else np.nan
        v6 = roc_accel_6m.loc[dd] if dd in roc_accel_6m.index and not np.isnan(roc_accel_6m.loc[dd]) else np.nan
        v9 = roc_accel_9m.loc[dd] if dd in roc_accel_9m.index and not np.isnan(roc_accel_9m.loc[dd]) else np.nan
        v12 = roc_accel_12m.loc[dd] if dd in roc_accel_12m.index and not np.isnan(roc_accel_12m.loc[dd]) else np.nan
        print(f"  {name:<22} {v3:>7.2%} {v6:>7.2%} {v9:>7.2%} {v12:>7.2%}")

# =============================================================================
# 3. Time spent in negative ROC accel
# =============================================================================
print("\n" + "=" * 70)
print("TIME IN NEGATIVE ROC ACCELERATION")
print("=" * 70)

for label, roc in [("3m", roc_accel_3m), ("6m", roc_accel_6m), ("9m", roc_accel_9m), ("12m", roc_accel_12m)]:
    valid = roc.loc[common].dropna()
    neg_pct = (valid < 0).sum() / len(valid) * 100
    print(f"  ROC accel {label}: negative {neg_pct:.1f}% of time ({(valid < 0).sum()}/{len(valid)} days)")

# =============================================================================
# 4. Metrics function
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
# 5. Strategy implementations
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


def strat_roc(dates, tqqq_ret, gold_ret, roc_accel, initial=10000,
              w_normal=0.60, w_defensive=0.40, threshold=0.0,
              check_freq='monthly'):
    """
    Simple ROC acceleration strategy:
    - ROC accel >= threshold: normal allocation
    - ROC accel < threshold: defensive allocation
    - Checked monthly (or weekly)
    """
    W = w_normal
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    signals = []
    state = 'normal'

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]

        if check_freq == 'monthly':
            check = d.month != pd_.month
        elif check_freq == 'weekly':
            check = d.isocalendar()[1] != pd_.isocalendar()[1]
        else:
            check = d.month != pd_.month

        if check and not np.isnan(roc_accel.iloc[i]):
            roc_val = roc_accel.iloc[i]
            old_state = state

            if roc_val < threshold:
                state = 'defensive'
                W = w_defensive
            else:
                state = 'normal'
                W = w_normal

            if state != old_state:
                signals.append((d, f"ROC={roc_val:.2%}, State={state}, W={W:.0%}"))

            cur_t = total * W
            cur_g = total * (1 - W)

        # Annual rebalance within current allocation
        elif d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)

        values.append(total)
    return pd.Series(values, index=dates), signals


def strat_vol_target(dates, tqqq_ret, gold_ret, initial=10000,
                      target_vol=0.20, lookback=63, max_w=0.80, min_w=0.10):
    W = 0.60
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
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
    return pd.Series(values, index=dates)


# =============================================================================
# 6. Run all variants
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING STRATEGIES...")
print("=" * 70)

results = {}

# Baseline
results['Baseline 60/40'] = strat_baseline(common, tqqq_d, gold_d)

# --- Main test: ROC 6m, threshold=0, defensive=40/60 ---
r_main, sig_main = strat_roc(common, tqqq_d, gold_d, roc_accel_6m,
                               w_normal=0.60, w_defensive=0.40, threshold=0.0)
results['ROC6m 60->40'] = r_main

# --- Defensive allocation sweep ---
for w_def in [0.50, 0.40, 0.30, 0.20, 0.10, 0.0]:
    r, _ = strat_roc(common, tqqq_d, gold_d, roc_accel_6m,
                      w_normal=0.60, w_defensive=w_def, threshold=0.0)
    results[f'ROC6m 60->{int(w_def*100)}'] = r

# --- Lookback sweep ---
for label, roc in [("3m", roc_accel_3m), ("6m", roc_accel_6m), ("9m", roc_accel_9m), ("12m", roc_accel_12m)]:
    r, _ = strat_roc(common, tqqq_d, gold_d, roc,
                      w_normal=0.60, w_defensive=0.40, threshold=0.0)
    results[f'ROC{label} 60->40'] = r

# --- Threshold sweep (small negative buffer to reduce whipsaw) ---
for thresh in [-0.02, -0.05, -0.08, -0.10]:
    r, _ = strat_roc(common, tqqq_d, gold_d, roc_accel_6m,
                      w_normal=0.60, w_defensive=0.40, threshold=thresh)
    results[f'ROC6m thr{int(thresh*100)}% 60->40'] = r

# --- Weekly check instead of monthly ---
r_weekly, _ = strat_roc(common, tqqq_d, gold_d, roc_accel_6m,
                          w_normal=0.60, w_defensive=0.40, threshold=0.0,
                          check_freq='weekly')
results['ROC6m weekly 60->40'] = r_weekly

# Vol Target for comparison
results['Vol Target 20%'] = strat_vol_target(common, tqqq_d, gold_d)

# =============================================================================
# 7. Results
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: DEFENSIVE ALLOCATION SWEEP (ROC 6m, threshold=0)")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

# Show allocation sweep first
alloc_names = ['Baseline 60/40'] + [f'ROC6m 60->{w}' for w in [50, 40, 30, 20, 10, 0]]
for name in alloc_names:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

print(f"\n" + "=" * 70)
print("RESULTS: LOOKBACK PERIOD SWEEP (defensive=40%, threshold=0)")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

lb_names = [f'ROC{lb} 60->40' for lb in ['3m', '6m', '9m', '12m']]
for name in lb_names:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

print(f"\n" + "=" * 70)
print("RESULTS: THRESHOLD SWEEP (ROC 6m, defensive=40%)")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

thresh_names = ['ROC6m 60->40'] + [f'ROC6m thr{t}% 60->40' for t in [-2, -5, -8, -10]]
for name in thresh_names:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

print(f"\n" + "=" * 70)
print("RESULTS: ALL STRATEGIES RANKED BY SHARPE")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

all_metrics = {}
for name, series in results.items():
    all_metrics[name] = metrics(series, years)

ranked = sorted(all_metrics.items(), key=lambda x: x[1]['sharpe'], reverse=True)
for name, m in ranked:
    marker = " ***" if name in ['Baseline 60/40', 'ROC6m 60->40', 'Vol Target 20%'] else ""
    print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}{marker}")

# =============================================================================
# 8. Signal log for main strategy
# =============================================================================
print("\n" + "=" * 70)
print("ROC 6m SIGNAL LOG (60->40 strategy)")
print("=" * 70)

for date, sig in sig_main:
    print(f"  {date.strftime('%Y-%m-%d')}: {sig}")
print(f"\nTotal state changes: {len(sig_main)}")

# Count time in each state
roc_valid = roc_accel_6m.loc[common].dropna()
normal_time = (roc_valid >= 0).sum() / len(roc_valid) * 100
defensive_time = (roc_valid < 0).sum() / len(roc_valid) * 100
print(f"\nTime in Normal (60%):    {normal_time:.1f}%")
print(f"Time in Defensive (40%): {defensive_time:.1f}%")

# =============================================================================
# 9. Crisis performance
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

compare_names = ['Baseline 60/40', 'ROC6m 60->40', 'ROC6m 60->30', 'ROC6m 60->20', 'Vol Target 20%']

for crisis_name, start, end in crisis_periods:
    print(f"\n--- {crisis_name} ---")
    crisis_dates = [d for d in common if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(crisis_dates) < 2:
        continue

    # Show ROC at crisis start
    f_d = crisis_dates[0]
    roc_val = roc_accel_6m.loc[f_d] if f_d in roc_accel_6m.index and not np.isnan(roc_accel_6m.loc[f_d]) else np.nan
    print(f"  ROC accel at start: {roc_val:.2%}" if not np.isnan(roc_val) else "  ROC accel at start: N/A")

    f, l = crisis_dates[0], crisis_dates[-1]
    print(f"  {'Strategy':<28} {'Return':>10} {'Max DD':>10}")
    print(f"  {'-'*52}")
    for name in compare_names:
        if name in results:
            series = results[name]
            ret = series.loc[l] / series.loc[f] - 1
            crisis_s = series.loc[crisis_dates]
            rm = crisis_s.cummax()
            dd = (crisis_s - rm) / rm
            mdd = dd.min()
            print(f"  {name:<28} {ret:>9.2%} {mdd:>9.2%}")

# =============================================================================
# 10. Year-by-year
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR COMPARISON")
print("=" * 70)

yby_names = ['Baseline 60/40', 'ROC6m 60->40', 'Vol Target 20%']
yby_labels = ['Baseline', 'ROC6m40', 'VolTgt']

print(f"\n{'Year':<7}", end="")
for l in yby_labels:
    print(f" {l:>10}", end="")
print(f" {'ROC6m':>8} {'State':>10} {'Winner':>10}")
print("-" * 70)

all_years = sorted(set(d.year for d in common))
for year in all_years:
    yd = [d for d in common if d.year == year]
    if len(yd) < 2:
        continue
    f, l = yd[0], yd[-1]

    roc_val = roc_accel_6m.loc[f] if f in roc_accel_6m.index and not np.isnan(roc_accel_6m.loc[f]) else np.nan
    state = "DEF(40%)" if (not np.isnan(roc_val) and roc_val < 0) else "NRM(60%)"

    rets = {}
    print(f"{year:<7}", end="")
    for sname, label in zip(yby_names, yby_labels):
        r = results[sname].loc[l] / results[sname].loc[f] - 1
        rets[label] = r
        print(f" {r:>9.2%}", end="")

    if not np.isnan(roc_val):
        print(f" {roc_val:>7.2%}", end="")
    else:
        print(f" {'N/A':>7}", end="")

    print(f" {state:>10}", end="")
    best = max(rets, key=rets.get)
    print(f" {best:>10}")

# =============================================================================
# 11. Final analysis
# =============================================================================
print("\n" + "=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)

m_base = all_metrics['Baseline 60/40']
m_roc = all_metrics['ROC6m 60->40']
m_vol = all_metrics['Vol Target 20%']

# Calculate CAGR difference
cagr_cost = m_base['cagr'] - m_roc['cagr']
dd_improvement = m_roc['max_dd'] - m_base['max_dd']

print(f"""
=== THE ROC DECELERATION STRATEGY ===

Rule: Check monthly.
  - If 6-month NDX momentum is ACCELERATING (ROC accel >= 0): hold 60% TQQQ / 40% Gold
  - If 6-month NDX momentum is DECELERATING (ROC accel < 0):  hold 40% TQQQ / 60% Gold
  - Annual rebalancing within current allocation

BASELINE 60/40:         CAGR {m_base['cagr']:.2%}  MaxDD {m_base['max_dd']:.2%}  Sharpe {m_base['sharpe']:.2f}
ROC6m 60->40:           CAGR {m_roc['cagr']:.2%}  MaxDD {m_roc['max_dd']:.2%}  Sharpe {m_roc['sharpe']:.2f}
Vol Target 20%:         CAGR {m_vol['cagr']:.2%}  MaxDD {m_vol['max_dd']:.2%}  Sharpe {m_vol['sharpe']:.2f}

COST:    CAGR drops by {cagr_cost:.2%} ({m_base['cagr']:.2%} -> {m_roc['cagr']:.2%})
BENEFIT: Max DD improves by {dd_improvement:.2%} ({m_base['max_dd']:.2%} -> {m_roc['max_dd']:.2%})

TIME IN EACH STATE:
  Normal (60%):    {normal_time:.1f}% of days
  Defensive (40%): {defensive_time:.1f}% of days

CURRENT SIGNAL: ROC accel 6m = {roc_accel_6m.iloc[-1]:.2%} -> {'DEFENSIVE' if roc_accel_6m.iloc[-1] < 0 else 'NORMAL'}

$10,000 after 40 years:
  Baseline:   ${m_base['final']:>14,.0f}
  ROC 60->40: ${m_roc['final']:>14,.0f}
  Vol Target: ${m_vol['final']:>14,.0f}
""")
