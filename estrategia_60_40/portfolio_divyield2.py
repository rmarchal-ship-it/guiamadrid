#!/usr/bin/env python3
"""
Dividend Yield Signal v2: Rolling Window Percentiles

Fix: instead of comparing DY vs all history since 1871 (structurally broken
because DY regime-shifted permanently in the 1990s due to buybacks),
compute percentiles over a ROLLING WINDOW (e.g., 20 years).

This way DY=1.17% today is compared to the 1.2%-2.1% range of the last
20 years, not the 4%+ of 1900.

Strategy:
  - DY percentile < 5 (within rolling window): EXPENSIVE -> reduce TQQQ
  - Stay reduced until DY percentile > 50 (within rolling window): CORRECTED
  - Asymmetric hysteresis preserved
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
# 1. Load market data
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

nasdaq_price = pd.concat([
    ndx["Close"].loc[:qqq_start].iloc[:-1] * (qqq["Close"].iloc[0] / ndx["Close"].loc[:qqq_start].iloc[-1]),
    qqq["Close"].loc[qqq_start:]
])
nasdaq_price = nasdaq_price[~nasdaq_price.index.duplicated(keep='last')].sort_index()

common = tqqq_sim.index.intersection(gold_composite.index).intersection(nasdaq_price.index)
common = common.sort_values()

tqqq_p = tqqq_sim.loc[common] / tqqq_sim.loc[common].iloc[0]
gold_p = gold_composite.loc[common] / gold_composite.loc[common].iloc[0]

tqqq_d = tqqq_p.pct_change().fillna(0)
gold_d = gold_p.pct_change().fillna(0)

years = (common[-1] - common[0]).days / 365.25
print(f"Period: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")

# =============================================================================
# 2. Load S&P 500 Dividend Yield
# =============================================================================
print("\nLoading Dividend Yield data...")

dy_tables = pd.read_html('https://www.multpl.com/s-p-500-dividend-yield/table/by-month')
df_dy_raw = dy_tables[0]
df_dy_raw.columns = ['Date', 'Value']
df_dy_raw['Date'] = pd.to_datetime(
    df_dy_raw['Date'].astype(str).str.encode('ascii', 'ignore').str.decode('ascii').str.strip(),
    format='mixed', errors='coerce'
)
df_dy_raw['DY'] = pd.to_numeric(
    df_dy_raw['Value'].astype(str).str.replace('%', '').str.replace('[^0-9.]', '', regex=True),
    errors='coerce'
)
df_dy = df_dy_raw[['Date', 'DY']].dropna().sort_values('Date').set_index('Date')

dy_full = df_dy['DY'].copy()
dy_full.index = pd.to_datetime(dy_full.index)

dy_daily = dy_full.resample('B').ffill()
dy_aligned = dy_daily.reindex(common).ffill()

print(f"DY data: {df_dy.index[0].strftime('%Y-%m')} to {df_dy.index[-1].strftime('%Y-%m')} ({len(df_dy)} months)")

# =============================================================================
# 3. Compute rolling-window percentiles
# =============================================================================
print("\n" + "=" * 70)
print("COMPUTING ROLLING-WINDOW DY PERCENTILES")
print("=" * 70)

# For each month, compute DY percentile within a rolling window of N years
# This adapts to the structural regime shift in dividends

def compute_rolling_dy_pct(dy_monthly, window_years):
    """Compute percentile of current DY within rolling window of monthly data."""
    window_months = window_years * 12
    dy_pct = pd.Series(index=dy_monthly.index, dtype=float)
    vals = dy_monthly.values
    for i in range(len(vals)):
        start_idx = max(0, i - window_months)
        window = vals[start_idx:i+1]
        window = window[~np.isnan(window)]
        if len(window) >= 24:  # need at least 2 years
            dy_pct.iloc[i] = (window < vals[i]).sum() / len(window) * 100
    return dy_pct

# Compute for different windows
dy_monthly_aligned = dy_full.copy()
windows = [10, 15, 20, 25, 30]

rolling_pcts = {}
for w in windows:
    print(f"  Computing {w}-year rolling percentile...")
    rolling_pcts[w] = compute_rolling_dy_pct(dy_monthly_aligned, w)

# Show values at key dates
print(f"\nDY rolling percentile at key moments:")
print(f"{'Date':<25} {'DY%':>6}", end="")
for w in windows:
    print(f" {'p'+str(w)+'yr':>8}", end="")
print()
print("-" * (32 + 9 * len(windows)))

key_dates = [
    ("Black Monday 1987-09", "1987-09-01"),
    ("Pre-Dotcom 1998-01", "1998-01-01"),
    ("Dotcom peak 2000-03", "2000-03-01"),
    ("Dotcom trough 2002-10", "2002-10-01"),
    ("Pre-GFC 2007-06", "2007-06-01"),
    ("GFC trough 2009-03", "2009-03-01"),
    ("Bull 2013-06", "2013-06-01"),
    ("Bull 2017-06", "2017-06-01"),
    ("Pre-COVID 2020-01", "2020-01-01"),
    ("Pre-2022 2021-11", "2021-11-01"),
    ("2022 trough 2022-10", "2022-10-01"),
    ("Current 2025-12", "2025-12-01"),
]

for name, date_str in key_dates:
    d = pd.Timestamp(date_str)
    # Find nearest monthly date
    idx = dy_monthly_aligned.index.searchsorted(d)
    if idx < len(dy_monthly_aligned):
        nearest = dy_monthly_aligned.index[idx]
        dy_val = dy_monthly_aligned.iloc[idx]
        print(f"  {name:<23} {dy_val:>5.2f}", end="")
        for w in windows:
            pct = rolling_pcts[w].iloc[idx]
            if not np.isnan(pct):
                marker = " *" if pct < 5 else ""
                print(f" {pct:>6.1f}{marker}", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()

# =============================================================================
# 4. Metrics
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
# 5. Strategies
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


def strat_dy_rolling(dates, tqqq_ret, gold_ret, dy_series, dy_monthly_full,
                      initial=10000, window_years=20,
                      w_normal=0.60, w_defensive=0.40,
                      trigger_pct=5, recover_pct=50):
    """
    DY strategy with rolling-window percentiles.
    - trigger when DY percentile (within rolling window) < trigger_pct
    - recover when DY percentile > recover_pct
    """
    window_months = window_years * 12
    W = w_normal
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    signals = []
    state = 'normal'

    # Pre-compute monthly DY array for fast percentile lookups
    dy_m_vals = dy_monthly_full.values
    dy_m_idx = dy_monthly_full.index

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        check = d.month != pd_.month

        if check and not np.isnan(dy_series.iloc[i]):
            current_dy = dy_series.iloc[i]

            # Find position in monthly data
            m_idx = dy_m_idx.searchsorted(d)
            if m_idx > window_months:
                window = dy_m_vals[m_idx - window_months:m_idx + 1]
                window = window[~np.isnan(window)]
                if len(window) >= 24:
                    dy_pct = (window < current_dy).sum() / len(window) * 100

                    old_state = state
                    if state == 'normal' and dy_pct < trigger_pct:
                        state = 'defensive'
                        W = w_defensive
                    elif state == 'defensive' and dy_pct > recover_pct:
                        state = 'normal'
                        W = w_normal

                    if state != old_state:
                        signals.append((d, f"DY={current_dy:.2f}%, pct({window_years}yr)={dy_pct:.1f}%, "
                                          f"-> {state} (W={W:.0%})"))

            cur_t = total * W
            cur_g = total * (1 - W)
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
results['Baseline 60/40'] = strat_baseline(common, tqqq_d, gold_d)
results['Vol Target 20%'] = strat_vol_target(common, tqqq_d, gold_d)

# --- Window sweep (trigger p5, recover p50, defensive 40%) ---
for w_yr in [10, 15, 20, 25, 30]:
    r, sigs = strat_dy_rolling(common, tqqq_d, gold_d, dy_aligned, dy_full,
                                 window_years=w_yr, w_defensive=0.40,
                                 trigger_pct=5, recover_pct=50)
    results[f'DY {w_yr}yr p5/p50 ->40'] = r
    if w_yr == 20:
        sig_20yr = sigs

# --- Defensive weight sweep (20yr window, trigger p5, recover p50) ---
for w_def in [0.50, 0.40, 0.30, 0.20, 0.10, 0.0]:
    r, _ = strat_dy_rolling(common, tqqq_d, gold_d, dy_aligned, dy_full,
                              window_years=20, w_defensive=w_def,
                              trigger_pct=5, recover_pct=50)
    results[f'DY 20yr p5/p50 ->{int(w_def*100)}'] = r

# --- Trigger percentile sweep (20yr, defensive 40%, recover p50) ---
for trig in [3, 5, 8, 10, 15]:
    r, _ = strat_dy_rolling(common, tqqq_d, gold_d, dy_aligned, dy_full,
                              window_years=20, w_defensive=0.40,
                              trigger_pct=trig, recover_pct=50)
    results[f'DY 20yr p{trig}/p50 ->40'] = r

# --- Recovery percentile sweep (20yr, trigger p5, defensive 40%) ---
for rec in [30, 40, 50, 60, 70]:
    r, _ = strat_dy_rolling(common, tqqq_d, gold_d, dy_aligned, dy_full,
                              window_years=20, w_defensive=0.40,
                              trigger_pct=5, recover_pct=rec)
    results[f'DY 20yr p5/p{rec} ->40'] = r

# =============================================================================
# 7. Results
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: ROLLING WINDOW SWEEP (trigger p5, recover p50, def 40%)")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

for name in ['Baseline 60/40'] + [f'DY {w}yr p5/p50 ->40' for w in [10, 15, 20, 25, 30]] + ['Vol Target 20%']:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

print(f"\n" + "=" * 70)
print("RESULTS: DEFENSIVE WEIGHT SWEEP (20yr window, trigger p5, recover p50)")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

for name in ['Baseline 60/40'] + [f'DY 20yr p5/p50 ->{w}' for w in [50, 40, 30, 20, 10, 0]] + ['Vol Target 20%']:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

print(f"\n" + "=" * 70)
print("RESULTS: TRIGGER PERCENTILE SWEEP (20yr, def 40%, recover p50)")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

for name in ['Baseline 60/40'] + [f'DY 20yr p{t}/p50 ->40' for t in [3, 5, 8, 10, 15]] + ['Vol Target 20%']:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

print(f"\n" + "=" * 70)
print("RESULTS: RECOVERY PERCENTILE SWEEP (20yr, trigger p5, def 40%)")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

for name in ['Baseline 60/40'] + [f'DY 20yr p5/p{r} ->40' for r in [30, 40, 50, 60, 70]] + ['Vol Target 20%']:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# 8. Signal log (20yr window)
# =============================================================================
print("\n" + "=" * 70)
print("SIGNAL LOG: DY 20yr p5/p50 ->40%")
print("=" * 70)

if sig_20yr:
    for date, sig in sig_20yr:
        print(f"  {date.strftime('%Y-%m-%d')}: {sig}")
    print(f"\nTotal state changes: {len(sig_20yr)}")
else:
    print("  No signals triggered")

# Compute time in states for 20yr version
state_20yr = pd.Series('normal', index=common)
current_state = 'normal'
window_months = 20 * 12
dy_m_vals = dy_full.values
dy_m_idx = dy_full.index

for i in range(len(common)):
    d = common[i]
    dy_val = dy_aligned.iloc[i]
    if not np.isnan(dy_val):
        m_idx = dy_m_idx.searchsorted(d)
        if m_idx > window_months:
            window = dy_m_vals[m_idx - window_months:m_idx + 1]
            window = window[~np.isnan(window)]
            if len(window) >= 24:
                dy_pct = (window < dy_val).sum() / len(window) * 100
                if current_state == 'normal' and dy_pct < 5:
                    current_state = 'defensive'
                elif current_state == 'defensive' and dy_pct > 50:
                    current_state = 'normal'
    state_20yr.iloc[i] = current_state

normal_pct = (state_20yr == 'normal').sum() / len(state_20yr) * 100
defensive_pct = (state_20yr == 'defensive').sum() / len(state_20yr) * 100
print(f"\nTime in Normal (60%):    {normal_pct:.1f}%")
print(f"Time in Defensive (40%): {defensive_pct:.1f}%")

# Current state
current_dy = dy_aligned.iloc[-1]
m_idx = dy_m_idx.searchsorted(common[-1])
if m_idx > window_months:
    window = dy_m_vals[m_idx - window_months:m_idx + 1]
    window = window[~np.isnan(window)]
    current_pct_20yr = (window < current_dy).sum() / len(window) * 100
else:
    current_pct_20yr = np.nan

print(f"\nCurrent: DY = {current_dy:.2f}%, 20yr percentile = {current_pct_20yr:.1f}%")
print(f"  -> State: {state_20yr.iloc[-1].upper()}")

# =============================================================================
# 9. Defensive periods timeline (20yr)
# =============================================================================
print("\n" + "=" * 70)
print("DEFENSIVE PERIODS TIMELINE (20yr window)")
print("=" * 70)

in_def = False
def_start = None
periods = []
for i, d in enumerate(common):
    if state_20yr.iloc[i] == 'defensive' and not in_def:
        in_def = True
        def_start = d
    elif state_20yr.iloc[i] == 'normal' and in_def:
        in_def = False
        dur = (d - def_start).days
        periods.append((def_start, d, dur))
        print(f"  {def_start.strftime('%Y-%m-%d')} to {d.strftime('%Y-%m-%d')} "
              f"({dur:>5} days, {dur/365:.1f} yrs)")

if in_def:
    dur = (common[-1] - def_start).days
    periods.append((def_start, common[-1], dur))
    print(f"  {def_start.strftime('%Y-%m-%d')} to ONGOING          "
          f"({dur:>5} days, {dur/365:.1f} yrs)")

# =============================================================================
# 10. Crisis performance
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

compare = ['Baseline 60/40', 'DY 20yr p5/p50 ->40', 'DY 20yr p5/p50 ->20',
           'DY 20yr p5/p50 ->0', 'Vol Target 20%']

for crisis_name, start, end in crisis_periods:
    print(f"\n--- {crisis_name} ---")
    crisis_dates = [d for d in common if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(crisis_dates) < 2:
        continue

    f_d = crisis_dates[0]
    st = state_20yr.loc[f_d]
    dy_val = dy_aligned.loc[f_d]
    print(f"  State at start: {st.upper()}, DY={dy_val:.2f}%")

    f, l = crisis_dates[0], crisis_dates[-1]
    print(f"  {'Strategy':<28} {'Return':>10} {'Max DD':>10}")
    print(f"  {'-'*52}")
    for name in compare:
        if name in results:
            series = results[name]
            ret = series.loc[l] / series.loc[f] - 1
            crisis_s = series.loc[crisis_dates]
            rm = crisis_s.cummax()
            dd = (crisis_s - rm) / rm
            mdd = dd.min()
            print(f"  {name:<28} {ret:>9.2%} {mdd:>9.2%}")

# =============================================================================
# 11. Year-by-year
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR COMPARISON")
print("=" * 70)

yby = ['Baseline 60/40', 'DY 20yr p5/p50 ->40', 'Vol Target 20%']
yby_labels = ['Baseline', 'DY20yr', 'VolTgt']

print(f"\n{'Year':<7}", end="")
for l in yby_labels:
    print(f" {l:>10}", end="")
print(f" {'DY%':>6} {'p20yr':>6} {'State':>10} {'Winner':>10}")
print("-" * 78)

all_years = sorted(set(d.year for d in common))
for year in all_years:
    yd = [d for d in common if d.year == year]
    if len(yd) < 2:
        continue
    f, l_d = yd[0], yd[-1]

    dy_val = dy_aligned.loc[f] if not np.isnan(dy_aligned.loc[f]) else np.nan
    st = state_20yr.loc[f]

    # Get rolling pct for display
    m_idx = dy_m_idx.searchsorted(f)
    if m_idx > window_months and not np.isnan(dy_val):
        window = dy_m_vals[m_idx - window_months:m_idx + 1]
        window = window[~np.isnan(window)]
        dy_pct_20 = (window < dy_val).sum() / len(window) * 100
    else:
        dy_pct_20 = np.nan

    rets = {}
    print(f"{year:<7}", end="")
    for sname, label in zip(yby, yby_labels):
        r = results[sname].loc[l_d] / results[sname].loc[f] - 1
        rets[label] = r
        print(f" {r:>9.2%}", end="")

    if not np.isnan(dy_val):
        print(f" {dy_val:>5.2f}", end="")
    else:
        print(f" {'N/A':>5}", end="")

    if not np.isnan(dy_pct_20):
        print(f" {dy_pct_20:>5.1f}", end="")
    else:
        print(f" {'N/A':>5}", end="")

    print(f" {st:>10}", end="")
    best = max(rets, key=rets.get)
    print(f" {best:>10}")

# =============================================================================
# 12. Grand comparison of ALL best strategies across ALL backtests
# =============================================================================
print("\n" + "=" * 70)
print("GRAND COMPARISON: ALL BEST STRATEGIES")
print("=" * 70)
print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

grand = ['Baseline 60/40', 'DY 20yr p5/p50 ->40', 'DY 20yr p5/p50 ->20',
         'DY 20yr p5/p50 ->0', 'DY 20yr p5/p30 ->40', 'DY 20yr p10/p50 ->40',
         'Vol Target 20%']
for name in grand:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# 13. Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)

m_base = metrics(results['Baseline 60/40'], years)
m_dy = metrics(results['DY 20yr p5/p50 ->40'], years)
m_vol = metrics(results['Vol Target 20%'], years)

# Find best DY variant by Sharpe
dy_variants = {k: metrics(v, years) for k, v in results.items() if 'DY' in k}
best_dy = max(dy_variants.items(), key=lambda x: x[1]['sharpe'])

print(f"""
=== DIVIDEND YIELD v2: ROLLING WINDOW PERCENTILES ===

Fix applied: percentiles computed over a rolling 20-year window,
not vs all history since 1871. This adapts to the structural shift
in dividend policy (from high dividends to buybacks).

RULE:
  1. Monthly: compute S&P 500 DY percentile within last 20 years
  2. If DY < 5th percentile of last 20 years -> DEFENSIVE (reduce TQQQ)
  3. Stay DEFENSIVE until DY > 50th percentile of last 20 years -> NORMAL
  4. Annual rebalancing within current allocation

RESULTS (40 years):

  Baseline 60/40:         CAGR {m_base['cagr']:.2%}  MaxDD {m_base['max_dd']:.2%}  Sharpe {m_base['sharpe']:.2f}  ${m_base['final']:>12,.0f}
  DY 20yr p5/p50 ->40%:  CAGR {m_dy['cagr']:.2%}  MaxDD {m_dy['max_dd']:.2%}  Sharpe {m_dy['sharpe']:.2f}  ${m_dy['final']:>12,.0f}
  Vol Target 20%:         CAGR {m_vol['cagr']:.2%}  MaxDD {m_vol['max_dd']:.2%}  Sharpe {m_vol['sharpe']:.2f}  ${m_vol['final']:>12,.0f}

  Best DY variant: {best_dy[0]}
    CAGR {best_dy[1]['cagr']:.2%}  MaxDD {best_dy[1]['max_dd']:.2%}  Sharpe {best_dy[1]['sharpe']:.2f}

TIME IN STATES (20yr window):
  Normal (60%):    {normal_pct:.1f}%
  Defensive (40%): {defensive_pct:.1f}%

CURRENT: DY = {current_dy:.2f}%, 20yr rolling percentile = {current_pct_20yr:.1f}%
  -> {state_20yr.iloc[-1].upper()}
""")
