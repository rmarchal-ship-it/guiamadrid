#!/usr/bin/env python3
"""
Dividend Yield Signal Strategy for 60% TQQQ / 40% Gold

Signal: S&P 500 Dividend Yield
  - When DY is very LOW (percentil 95+) = market is expensive -> reduce TQQQ
  - Stay reduced until DY drops below percentil 50 = market has corrected
  - This creates ASYMMETRIC hysteresis: hard to trigger, easy to recover
  - The signal is sticky: once in defensive, stays until real correction

Data source: multpl.com (S&P 500 dividend yield, monthly from 1871)
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

# Align
common = tqqq_sim.index.intersection(gold_composite.index).intersection(nasdaq_price.index)
common = common.sort_values()

tqqq_p = tqqq_sim.loc[common] / tqqq_sim.loc[common].iloc[0]
gold_p = gold_composite.loc[common] / gold_composite.loc[common].iloc[0]

tqqq_d = tqqq_p.pct_change().fillna(0)
gold_d = gold_p.pct_change().fillna(0)

years = (common[-1] - common[0]).days / 365.25
print(f"Period: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")

# =============================================================================
# 2. Load S&P 500 Dividend Yield from multpl.com
# =============================================================================
print("\n" + "=" * 70)
print("LOADING S&P 500 DIVIDEND YIELD DATA...")
print("=" * 70)

dy_tables = pd.read_html('https://www.multpl.com/s-p-500-dividend-yield/table/by-month')
df_dy_raw = dy_tables[0]
df_dy_raw.columns = ['Date', 'Value']
df_dy_raw['Date'] = pd.to_datetime(
    df_dy_raw['Date'].astype(str).str.encode('ascii', 'ignore').str.decode('ascii').str.strip(),
    format='mixed', errors='coerce'
)
# Parse percentage values like "1.23%"
df_dy_raw['DY'] = pd.to_numeric(
    df_dy_raw['Value'].astype(str).str.replace('%', '').str.replace('[^0-9.]', '', regex=True),
    errors='coerce'
)
df_dy = df_dy_raw[['Date', 'DY']].dropna().sort_values('Date').set_index('Date')

print(f"Dividend Yield data: {df_dy.index[0].strftime('%Y-%m')} to {df_dy.index[-1].strftime('%Y-%m')} ({len(df_dy)} months)")

# Full monthly history
dy_full = df_dy['DY'].copy()
dy_full.index = pd.to_datetime(dy_full.index)

# Interpolate to business days (forward fill — DY doesn't change intra-month)
dy_daily = dy_full.resample('B').ffill()
dy_aligned = dy_daily.reindex(common).ffill()

print(f"DY aligned to portfolio: {dy_aligned.dropna().index[0].strftime('%Y-%m-%d')} to {dy_aligned.dropna().index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 3. DY analysis - percentiles and key dates
# =============================================================================
print("\n" + "=" * 70)
print("DIVIDEND YIELD ANALYSIS")
print("=" * 70)

# NOTE: Low DY = expensive market. So percentile 95 of "expensiveness"
# means DY is in the BOTTOM 5% (very low yield = very high prices).
# We need to invert: percentile 95 of "low DY" = DY < 5th percentile of DY values.
# OR equivalently: rank DY ascending, bottom 5% = market expensive.

# Let's compute it clearly:
# "DY percentile" = what % of historical DY values are LOWER than current
# If DY_pct > 95 -> DY is very HIGH (cheap market, good)
# If DY_pct < 5 -> DY is very LOW (expensive market, bubble)

# So the trigger is: DY_pct < 5 (= DY in bottom 5% = percentil 95 of expensiveness)
# Recovery is: DY_pct > 50 (= DY above median = market has corrected)

print(f"\nKey DY percentiles (full history 1871-present, {len(dy_full)} months):")
for p in [5, 10, 15, 20, 25, 50, 75, 90, 95]:
    val = np.percentile(dy_full.dropna(), p)
    print(f"  {p:>3}th percentile: DY = {val:.2f}%")

print(f"\nDY at key moments:")
key_dates = [
    ("Black Monday 1987", "1987-10-01"),
    ("Pre-Dotcom 1998", "1998-01-01"),
    ("Dotcom peak 2000-03", "2000-03-01"),
    ("Dotcom trough 2002-10", "2002-10-01"),
    ("Pre-GFC 2007-10", "2007-10-01"),
    ("GFC trough 2009-03", "2009-03-01"),
    ("Bull 2013", "2013-06-01"),
    ("Bull 2017", "2017-06-01"),
    ("Pre-COVID 2020", "2020-01-01"),
    ("Pre-2022 2021-11", "2021-11-01"),
    ("2022 trough", "2022-10-01"),
    ("Current", common[-1].strftime('%Y-%m-%d')),
]

for name, date_str in key_dates:
    d = pd.Timestamp(date_str)
    nearest = dy_aligned.index[dy_aligned.index.searchsorted(d)]
    if nearest in dy_aligned.index:
        dy_val = dy_aligned.loc[nearest]
        # Expanding percentile: what % of ALL history up to this date is LOWER
        hist = dy_full.loc[:d].dropna()
        if len(hist) > 0 and not np.isnan(dy_val):
            pct = (hist < dy_val).sum() / len(hist) * 100
            print(f"  {name:<25} DY={dy_val:>5.2f}%  pct={pct:>5.1f}%  {'<-- EXPENSIVE' if pct < 5 else '<-- CHEAP' if pct > 50 else ''}")

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


def strat_divyield(dates, tqqq_ret, gold_ret, dy_series, dy_history,
                    initial=10000,
                    w_normal=0.60, w_defensive=0.40,
                    trigger_pct=5, recover_pct=50):
    """
    Dividend Yield strategy with asymmetric hysteresis:
    - ENTER defensive when DY percentile < trigger_pct (DY very low = expensive)
    - EXIT defensive when DY percentile > recover_pct (DY above median = corrected)
    - Checked monthly

    trigger_pct: DY must be in bottom X% to trigger (e.g., 5 = bottom 5%)
    recover_pct: DY must rise above this percentile to recover (e.g., 50 = median)
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
        check = d.month != pd_.month

        if check and not np.isnan(dy_series.iloc[i]):
            current_dy = dy_series.iloc[i]
            hist = dy_history.loc[:d].dropna()
            if len(hist) > 24:
                dy_pct = (hist < current_dy).sum() / len(hist) * 100

                old_state = state
                if state == 'normal' and dy_pct < trigger_pct:
                    state = 'defensive'
                    W = w_defensive
                elif state == 'defensive' and dy_pct > recover_pct:
                    state = 'normal'
                    W = w_normal

                if state != old_state:
                    signals.append((d, f"DY={current_dy:.2f}%, pct={dy_pct:.1f}%, -> {state} (W={W:.0%})"))

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
# 6. Run strategies
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING STRATEGIES...")
print("=" * 70)

results = {}

# Baseline
results['Baseline 60/40'] = strat_baseline(common, tqqq_d, gold_d)

# --- Main: DY p5 trigger, p50 recovery, 40% defensive ---
r_main, sig_main = strat_divyield(common, tqqq_d, gold_d, dy_aligned, dy_full,
                                    w_defensive=0.40, trigger_pct=5, recover_pct=50)
results['DY p5/p50 60->40'] = r_main

# --- Defensive weight sweep ---
for w_def in [0.50, 0.40, 0.30, 0.20, 0.10, 0.0]:
    r, _ = strat_divyield(common, tqqq_d, gold_d, dy_aligned, dy_full,
                           w_defensive=w_def, trigger_pct=5, recover_pct=50)
    results[f'DY p5/p50 60->{int(w_def*100)}'] = r

# --- Trigger percentile sweep ---
for trig in [3, 5, 8, 10, 15]:
    r, _ = strat_divyield(common, tqqq_d, gold_d, dy_aligned, dy_full,
                           w_defensive=0.40, trigger_pct=trig, recover_pct=50)
    results[f'DY p{trig}/p50 60->40'] = r

# --- Recovery percentile sweep ---
for rec in [30, 40, 50, 60, 70]:
    r, _ = strat_divyield(common, tqqq_d, gold_d, dy_aligned, dy_full,
                           w_defensive=0.40, trigger_pct=5, recover_pct=rec)
    results[f'DY p5/p{rec} 60->40'] = r

# Vol Target for comparison
results['Vol Target 20%'] = strat_vol_target(common, tqqq_d, gold_d)

# =============================================================================
# 7. Results
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: DEFENSIVE WEIGHT SWEEP (trigger p5, recovery p50)")
print("=" * 70)
print(f"\n{'Strategy':<26} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

alloc_names = ['Baseline 60/40'] + [f'DY p5/p50 60->{w}' for w in [50, 40, 30, 20, 10, 0]] + ['Vol Target 20%']
for name in alloc_names:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<26} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

print(f"\n" + "=" * 70)
print("RESULTS: TRIGGER PERCENTILE SWEEP (defensive=40%, recovery p50)")
print("=" * 70)
print(f"\n{'Strategy':<26} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

for name in ['Baseline 60/40'] + [f'DY p{t}/p50 60->40' for t in [3, 5, 8, 10, 15]] + ['Vol Target 20%']:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<26} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

print(f"\n" + "=" * 70)
print("RESULTS: RECOVERY PERCENTILE SWEEP (trigger p5, defensive=40%)")
print("=" * 70)
print(f"\n{'Strategy':<26} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

for name in ['Baseline 60/40'] + [f'DY p5/p{r} 60->40' for r in [30, 40, 50, 60, 70]] + ['Vol Target 20%']:
    if name in results:
        m = metrics(results[name], years)
        print(f"{name:<26} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# 8. Signal log
# =============================================================================
print("\n" + "=" * 70)
print("DY p5/p50 SIGNAL LOG (60->40)")
print("=" * 70)

for date, sig in sig_main:
    print(f"  {date.strftime('%Y-%m-%d')}: {sig}")
print(f"\nTotal state changes: {len(sig_main)}")

# Time in each state
state_track = pd.Series('normal', index=common)
current_state = 'normal'
for i in range(len(common)):
    d = common[i]
    dy_val = dy_aligned.iloc[i]
    if not np.isnan(dy_val):
        hist = dy_full.loc[:d].dropna()
        if len(hist) > 24:
            dy_pct = (hist < dy_val).sum() / len(hist) * 100
            if current_state == 'normal' and dy_pct < 5:
                current_state = 'defensive'
            elif current_state == 'defensive' and dy_pct > 50:
                current_state = 'normal'
    state_track.iloc[i] = current_state

normal_pct = (state_track == 'normal').sum() / len(state_track) * 100
defensive_pct = (state_track == 'defensive').sum() / len(state_track) * 100
print(f"\nTime in Normal (60%):    {normal_pct:.1f}%")
print(f"Time in Defensive (40%): {defensive_pct:.1f}%")

# Current state
current_dy = dy_aligned.iloc[-1]
hist_now = dy_full.loc[:common[-1]].dropna()
current_pct = (hist_now < current_dy).sum() / len(hist_now) * 100
print(f"\nCurrent: DY = {current_dy:.2f}%, percentile = {current_pct:.1f}%, state = {state_track.iloc[-1]}")

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

compare = ['Baseline 60/40', 'DY p5/p50 60->40', 'DY p5/p50 60->20', 'DY p5/p50 60->0', 'Vol Target 20%']

for crisis_name, start, end in crisis_periods:
    print(f"\n--- {crisis_name} ---")
    crisis_dates = [d for d in common if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(crisis_dates) < 2:
        continue

    # DY and state at crisis start
    f_d = crisis_dates[0]
    dy_val = dy_aligned.loc[f_d] if f_d in dy_aligned.index else np.nan
    st = state_track.loc[f_d]
    if not np.isnan(dy_val):
        h = dy_full.loc[:f_d].dropna()
        p = (h < dy_val).sum() / len(h) * 100
        print(f"  DY at start: {dy_val:.2f}% (pct={p:.1f}%), state={st}")

    f, l = crisis_dates[0], crisis_dates[-1]
    print(f"  {'Strategy':<26} {'Return':>10} {'Max DD':>10}")
    print(f"  {'-'*50}")
    for name in compare:
        if name in results:
            series = results[name]
            ret = series.loc[l] / series.loc[f] - 1
            crisis_s = series.loc[crisis_dates]
            rm = crisis_s.cummax()
            dd = (crisis_s - rm) / rm
            mdd = dd.min()
            print(f"  {name:<26} {ret:>9.2%} {mdd:>9.2%}")

# =============================================================================
# 10. Year-by-year
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR COMPARISON")
print("=" * 70)

yby = ['Baseline 60/40', 'DY p5/p50 60->40', 'Vol Target 20%']
yby_labels = ['Baseline', 'DY60->40', 'VolTgt']

print(f"\n{'Year':<7}", end="")
for l in yby_labels:
    print(f" {l:>10}", end="")
print(f" {'DY%':>7} {'DYpct':>7} {'State':>10} {'Winner':>10}")
print("-" * 80)

all_years = sorted(set(d.year for d in common))
for year in all_years:
    yd = [d for d in common if d.year == year]
    if len(yd) < 2:
        continue
    f, l_d = yd[0], yd[-1]

    dy_val = dy_aligned.loc[f] if f in dy_aligned.index and not np.isnan(dy_aligned.loc[f]) else np.nan
    st = state_track.loc[f]
    if not np.isnan(dy_val):
        h = dy_full.loc[:f].dropna()
        pct = (h < dy_val).sum() / len(h) * 100
    else:
        pct = np.nan

    rets = {}
    print(f"{year:<7}", end="")
    for sname, label in zip(yby, yby_labels):
        r = results[sname].loc[l_d] / results[sname].loc[f] - 1
        rets[label] = r
        print(f" {r:>9.2%}", end="")

    if not np.isnan(dy_val):
        print(f" {dy_val:>6.2f}", end="")
    else:
        print(f" {'N/A':>6}", end="")

    if not np.isnan(pct):
        print(f" {pct:>6.1f}", end="")
    else:
        print(f" {'N/A':>6}", end="")

    print(f" {st:>10}", end="")
    best = max(rets, key=rets.get)
    print(f" {best:>10}")

# =============================================================================
# 11. Detailed state timeline
# =============================================================================
print("\n" + "=" * 70)
print("DEFENSIVE PERIODS TIMELINE")
print("=" * 70)

in_def = False
def_start = None
for i, d in enumerate(common):
    if state_track.iloc[i] == 'defensive' and not in_def:
        in_def = True
        def_start = d
    elif state_track.iloc[i] == 'normal' and in_def:
        in_def = False
        dur = (d - def_start).days
        dy_at_start = dy_aligned.loc[def_start]
        dy_at_end = dy_aligned.loc[d]
        print(f"  {def_start.strftime('%Y-%m-%d')} to {d.strftime('%Y-%m-%d')} ({dur:>5} days, {dur/365:.1f} yrs)  "
              f"DY: {dy_at_start:.2f}% -> {dy_at_end:.2f}%")

if in_def:
    dur = (common[-1] - def_start).days
    dy_at_start = dy_aligned.loc[def_start]
    print(f"  {def_start.strftime('%Y-%m-%d')} to ONGOING         ({dur:>5} days, {dur/365:.1f} yrs)  "
          f"DY at start: {dy_at_start:.2f}%")

# =============================================================================
# 12. Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)

m_base = metrics(results['Baseline 60/40'], years)
m_dy = metrics(results['DY p5/p50 60->40'], years)
m_vol = metrics(results['Vol Target 20%'], years)

print(f"""
=== DIVIDEND YIELD STRATEGY ===

Rule:
  1. Check monthly: compute S&P 500 Dividend Yield percentile vs all history
  2. If DY falls below 5th percentile (very low yield = expensive market):
     -> Switch to 40% TQQQ / 60% Gold
  3. Stay defensive UNTIL DY rises above 50th percentile (median):
     -> Switch back to 60% TQQQ / 40% Gold
  4. Annual rebalancing within current allocation

This creates ASYMMETRIC hysteresis:
  - Hard to trigger (needs extreme cheapness, DY in bottom 5%)
  - Sticky once triggered (stays defensive until real correction)
  - Easy to recover (DY just needs to reach median)

RESULTS (40 years):

  {'Strategy':<26} {'CAGR':>8} {'Max DD':>9} {'Sharpe':>8} {'Final ($10k)':>14}
  {'-'*70}
  {'Baseline 60/40':<26} {m_base['cagr']:>7.2%} {m_base['max_dd']:>8.2%} {m_base['sharpe']:>7.2f} ${m_base['final']:>12,.0f}
  {'DY p5/p50 60->40':<26} {m_dy['cagr']:>7.2%} {m_dy['max_dd']:>8.2%} {m_dy['sharpe']:>7.2f} ${m_dy['final']:>12,.0f}
  {'Vol Target 20%':<26} {m_vol['cagr']:>7.2%} {m_vol['max_dd']:>8.2%} {m_vol['sharpe']:>7.2f} ${m_vol['final']:>12,.0f}

TIME IN STATES:
  Normal (60%):    {normal_pct:.1f}%
  Defensive (40%): {defensive_pct:.1f}%

CURRENT STATE: DY = {current_dy:.2f}%, percentile = {current_pct:.1f}%
  -> {'DEFENSIVE (40% TQQQ)' if state_track.iloc[-1] == 'defensive' else 'NORMAL (60% TQQQ)'}
""")
