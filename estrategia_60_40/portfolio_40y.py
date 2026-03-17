#!/usr/bin/env python3
"""
40-year backtest: 60% TQQQ (simulated) / 40% GLD (gold spot)
- TQQQ simulated from ^NDX (Nasdaq-100 index) daily returns x3
- Gold: monthly spot prices interpolated to daily (no leverage = no issue)
- GLD used where available (Nov 2004+) for better accuracy
- Period: Oct 1985 - Dec 2025 (~40 years)
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
# 1. Download data
# =============================================================================
print("Downloading data...")

ndx = yf.download("^NDX", start="1985-01-01", end="2025-12-31", auto_adjust=True, progress=False)
qqq = yf.download("QQQ", start="1999-03-01", end="2025-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2025-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2025-12-31", auto_adjust=True, progress=False)
spy = yf.download("SPY", start="1993-01-01", end="2025-12-31", auto_adjust=True, progress=False)

for df in [ndx, qqq, gld, tqqq_real, spy]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

# Gold monthly from GitHub
url = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=30)
gold_monthly = pd.read_csv(io.StringIO(resp.read().decode('utf-8')), parse_dates=['Date'])
gold_monthly = gold_monthly.set_index('Date').sort_index()

print(f"^NDX:       {ndx.index[0].strftime('%Y-%m-%d')} to {ndx.index[-1].strftime('%Y-%m-%d')} ({len(ndx)} days)")
print(f"QQQ:        {qqq.index[0].strftime('%Y-%m-%d')} to {qqq.index[-1].strftime('%Y-%m-%d')} ({len(qqq)} days)")
print(f"GLD:        {gld.index[0].strftime('%Y-%m-%d')} to {gld.index[-1].strftime('%Y-%m-%d')} ({len(gld)} days)")
print(f"TQQQ real:  {tqqq_real.index[0].strftime('%Y-%m-%d')} to {tqqq_real.index[-1].strftime('%Y-%m-%d')} ({len(tqqq_real)} days)")
print(f"Gold monthly: {gold_monthly.index[0].strftime('%Y-%m-%d')} to {gold_monthly.index[-1].strftime('%Y-%m-%d')} ({len(gold_monthly)} months)")

# =============================================================================
# 2. Build composite gold daily series
# =============================================================================
print("\n" + "=" * 70)
print("BUILDING GOLD DAILY SERIES")
print("=" * 70)

# Strategy:
# - Before GLD (pre Nov 2004): monthly gold spot -> interpolate to business days
# - From GLD onwards: use GLD directly (better daily accuracy)

# Interpolate monthly gold to business days
# For each business day, use linear interpolation between monthly points
gold_m = gold_monthly['Price'].copy()
gold_m.index = pd.to_datetime(gold_m.index)

# Resample to business days with linear interpolation (not just ffill)
gold_daily_interp = gold_m.resample('B').interpolate(method='linear')

# Now build composite: interpolated before GLD, GLD after
gld_start = gld.index[0]

# Scale interpolated gold to match GLD at junction
# Find the monthly gold value closest to GLD start
gold_at_gld_start = gold_daily_interp.loc[:gld_start].iloc[-1]
gld_at_start = gld["Close"].iloc[0]
scale = gld_at_start / gold_at_gld_start

gold_pre = gold_daily_interp.loc[:gld_start] * scale
gold_post = gld["Close"].loc[gld_start:]

# Combine
gold_composite = pd.concat([gold_pre.iloc[:-1], gold_post])
gold_composite = gold_composite[~gold_composite.index.duplicated(keep='last')]
gold_composite = gold_composite.sort_index()

print(f"Gold composite: {gold_composite.index[0].strftime('%Y-%m-%d')} to {gold_composite.index[-1].strftime('%Y-%m-%d')}")
print(f"  Pre-GLD (interpolated monthly): {gold_composite.loc[:gld_start].index[0].strftime('%Y-%m-%d')} to {gld_start.strftime('%Y-%m-%d')}")
print(f"  Post-GLD (actual daily):        {gld_start.strftime('%Y-%m-%d')} to {gold_composite.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 3. Build NDX-based series and calibrate TQQQ
# =============================================================================
print("\n" + "=" * 70)
print("CALIBRATING TQQQ SIMULATION")
print("=" * 70)

# NDX daily returns -> simulate TQQQ as 3x daily
# But first, calibrate the daily cost using real TQQQ (which tracks QQQ, not NDX directly)
# Since NDX and QQQ are ~0.98 correlated, we calibrate on QQQ period then apply to NDX

qqq_ret = qqq["Close"].pct_change()
overlap_tqqq_dates = tqqq_real.index.intersection(qqq_ret.index)
qqq_ret_ov = qqq_ret.loc[overlap_tqqq_dates]
tqqq_real_ov = tqqq_real["Close"].loc[overlap_tqqq_dates]
real_total = tqqq_real_ov.iloc[-1] / tqqq_real_ov.iloc[0]

def sim_error(dc):
    sr = qqq_ret_ov * 3 - dc
    st = (1 + sr).cumprod()
    return (np.log(st.iloc[-1] / st.iloc[0]) - np.log(real_total)) ** 2

result = minimize_scalar(sim_error, bounds=(0, 0.001), method='bounded')
daily_cost = result.x
annual_cost = daily_cost * 252
print(f"Calibrated annual friction: {annual_cost:.2%}")

# Now build TQQQ from NDX for full period, and from QQQ where available
# Use NDX pre-QQQ, QQQ post-QQQ for maximum accuracy
ndx_ret = ndx["Close"].pct_change()
qqq_start = qqq.index[0]

# Combined Nasdaq returns: NDX before QQQ, QQQ after
nasdaq_ret = pd.concat([
    ndx_ret.loc[:qqq_start].iloc[:-1],  # NDX up to QQQ start (exclusive)
    qqq_ret.loc[qqq_start:]              # QQQ from its start
])
nasdaq_ret = nasdaq_ret[~nasdaq_ret.index.duplicated(keep='last')]
nasdaq_ret = nasdaq_ret.sort_index()

# Build TQQQ simulated
tqqq_sim_ret = nasdaq_ret * 3 - daily_cost
tqqq_sim = (1 + tqqq_sim_ret).cumprod()
tqqq_sim.iloc[0] = 1.0

print(f"\nTQQQ simulated: {tqqq_sim.index[0].strftime('%Y-%m-%d')} to {tqqq_sim.index[-1].strftime('%Y-%m-%d')} ({len(tqqq_sim)} days)")

# Validate against real TQQQ
sim_ov = tqqq_sim.loc[overlap_tqqq_dates]
sim_total = sim_ov.iloc[-1] / sim_ov.iloc[0] - 1
real_t = real_total - 1
print(f"\nValidation (2010-2025): TQQQ real = {real_t:.2%}, simulated = {sim_total:.2%}")

# =============================================================================
# 4. Align series
# =============================================================================
common = tqqq_sim.index.intersection(gold_composite.index)
common = common.sort_values()

tqqq_p = tqqq_sim.loc[common]
gold_p = gold_composite.loc[common]

# Normalize
tqqq_p = tqqq_p / tqqq_p.iloc[0]
gold_n = gold_p / gold_p.iloc[0]

tqqq_d = tqqq_p.pct_change().fillna(0)
gold_d = gold_n.pct_change().fillna(0)

# Also get NDX (as proxy for QQQ buy&hold) and SPY
ndx_aligned = ndx["Close"].reindex(common).ffill().bfill()
ndx_n = ndx_aligned / ndx_aligned.iloc[0]

spy_aligned = spy["Close"].reindex(common).ffill()
# SPY starts Jan 1993, so before that we'll have NaN
spy_start_idx = spy_aligned.first_valid_index()

years = (common[-1] - common[0]).days / 365.25

print(f"\n" + "=" * 70)
print(f"BACKTEST PERIOD: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")
print(f"Trading days: {len(common)}")
print("=" * 70)

# =============================================================================
# 5. Portfolio simulation
# =============================================================================
def simulate(w_tqqq, freq, dates, tqqq_ret, gold_ret, initial=10000):
    w_gold = 1 - w_tqqq
    cur_t = initial * w_tqqq
    cur_g = initial * w_gold
    values = [initial]
    t_alloc = [cur_t]
    g_alloc = [cur_g]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        rebal = False
        if freq == 'annual':
            rebal = d.year != pd_.year
        elif freq == 'quarterly':
            rebal = (d.month != pd_.month) and d.month in [1,4,7,10]
        elif freq == 'monthly':
            rebal = d.month != pd_.month

        if rebal:
            cur_t = total * w_tqqq
            cur_g = total * w_gold

        values.append(total)
        t_alloc.append(cur_t)
        g_alloc.append(cur_g)

    return pd.Series(values, index=dates), pd.Series(t_alloc, index=dates), pd.Series(g_alloc, index=dates)


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
    peak_date = series.loc[:max_dd_date].idxmax()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        "final": series.iloc[-1], "total_ret": total, "cagr": cagr,
        "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "max_dd_date": max_dd_date, "peak_date": peak_date, "calmar": calmar
    }

# =============================================================================
# 6. Main results: 60/40 annual
# =============================================================================
port_60, t_alloc, g_alloc = simulate(0.60, 'annual', common, tqqq_d, gold_d)

# Benchmarks
tqqq_bh = pd.Series((tqqq_p / tqqq_p.iloc[0] * 10000).values, index=common)
gold_bh = pd.Series((gold_n / gold_n.iloc[0] * 10000).values, index=common)
ndx_bh = pd.Series((ndx_n / ndx_n.iloc[0] * 10000).values, index=common)

print(f"\n{'Strategy':<42} {'Final Value':>14} {'Total Ret':>12} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9}")
print("-" * 105)

for name, s in [
    ("60% TQQQ / 40% Gold (anual)", port_60),
    ("100% TQQQ sim (buy & hold)", tqqq_bh),
    ("100% Gold (buy & hold)", gold_bh),
    ("100% Nasdaq-100 (buy & hold)", ndx_bh),
]:
    m = metrics(s, years)
    print(f"{name:<42} ${m['final']:>12,.0f} {m['total_ret']:>11.2%} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%}")

# SPY (shorter period)
if spy_start_idx is not None:
    spy_dates = common[common >= spy_start_idx]
    spy_s = spy_aligned.loc[spy_dates]
    spy_s_n = spy_s / spy_s.iloc[0] * 10000
    spy_yrs = (spy_dates[-1] - spy_dates[0]).days / 365.25
    m = metrics(spy_s_n, spy_yrs)
    print(f"{'100% SPY (from 1993)':<42} ${m['final']:>12,.0f} {m['total_ret']:>11.2%} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%}")

# =============================================================================
# 7. Weight sweep (annual rebalancing)
# =============================================================================
print("\n" + "=" * 70)
print("WEIGHT SWEEP (annual rebalancing)")
print("=" * 70)

print(f"\n{'TQQQ %':>8} {'Gold %':>8} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 80)

sweep_results = []
for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    s, _, _ = simulate(w, 'annual', common, tqqq_d, gold_d)
    m = metrics(s, years)
    m['weight'] = w
    sweep_results.append(m)
    print(f"{w:>7.0%} {1-w:>7.0%} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

df_sweep = pd.DataFrame(sweep_results)
best_cagr = df_sweep.loc[df_sweep['cagr'].idxmax()]
best_sharpe = df_sweep.loc[df_sweep['sharpe'].idxmax()]
best_calmar = df_sweep.loc[df_sweep['calmar'].idxmax()]

print(f"\n  Best CAGR:   {best_cagr['weight']:.0%} TQQQ -> CAGR {best_cagr['cagr']:.2%}, Sharpe {best_cagr['sharpe']:.2f}, MaxDD {best_cagr['max_dd']:.2%}")
print(f"  Best Sharpe: {best_sharpe['weight']:.0%} TQQQ -> CAGR {best_sharpe['cagr']:.2%}, Sharpe {best_sharpe['sharpe']:.2f}, MaxDD {best_sharpe['max_dd']:.2%}")
print(f"  Best Calmar: {best_calmar['weight']:.0%} TQQQ -> CAGR {best_calmar['cagr']:.2%}, Sharpe {best_calmar['sharpe']:.2f}, MaxDD {best_calmar['max_dd']:.2%}")

# =============================================================================
# 8. Year-by-year
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR PERFORMANCE (60/40, annual rebalancing)")
print("=" * 70)

print(f"\n{'Year':<8} {'Portfolio':>12} {'TQQQ Sim':>12} {'Gold':>12} {'NDX':>12} {'Port Value':>14} {'TQQQ%':>8} {'Gold%':>8}")
print("-" * 92)

all_years = sorted(set(d.year for d in common))
for year in all_years:
    yd = [d for d in common if d.year == year]
    if len(yd) < 2:
        continue
    f, l = yd[0], yd[-1]

    pr = port_60.loc[l] / port_60.loc[f] - 1
    tr = tqqq_bh.loc[l] / tqqq_bh.loc[f] - 1
    gr = gold_bh.loc[l] / gold_bh.loc[f] - 1
    nr = ndx_bh.loc[l] / ndx_bh.loc[f] - 1

    pv = port_60.loc[l]
    tp = t_alloc.loc[l] / port_60.loc[l] * 100
    gp = g_alloc.loc[l] / port_60.loc[l] * 100

    print(f"{year:<8} {pr:>11.2%} {tr:>11.2%} {gr:>11.2%} {nr:>11.2%} ${pv:>12,.0f} {tp:>7.1f}% {gp:>7.1f}%")

# =============================================================================
# 9. Major drawdowns
# =============================================================================
print("\n" + "=" * 70)
print("MAJOR DRAWDOWNS (> -25%)")
print("=" * 70)

rolling_max = port_60.cummax()
drawdown = (port_60 - rolling_max) / rolling_max

in_dd = False
dd_start = None
dd_events = []

for i, date in enumerate(common):
    if drawdown.iloc[i] < -0.05 and not in_dd:
        in_dd = True
        dd_start = port_60.loc[:date].idxmax()
    elif drawdown.iloc[i] >= -0.01 and in_dd:
        in_dd = False
        dd_end = date
        trough = drawdown.loc[dd_start:dd_end].idxmin()
        depth = drawdown.loc[trough]
        if depth < -0.25:
            days_down = (trough - dd_start).days
            days_rec = (dd_end - trough).days
            dd_events.append({
                "peak": dd_start, "trough": trough, "recovery": dd_end,
                "depth": depth, "days_down": days_down, "days_rec": days_rec,
                "peak_val": port_60.loc[dd_start], "trough_val": port_60.loc[trough],
            })

if in_dd and dd_start is not None:
    trough = drawdown.loc[dd_start:].idxmin()
    depth = drawdown.loc[trough]
    if depth < -0.25:
        dd_events.append({
            "peak": dd_start, "trough": trough, "recovery": None,
            "depth": depth, "days_down": (trough - dd_start).days, "days_rec": None,
            "peak_val": port_60.loc[dd_start], "trough_val": port_60.loc[trough],
        })

print(f"\n{'#':<4} {'Peak':<14} {'Trough':<14} {'Recovery':<14} {'DD':>9} {'Days Down':>10} {'Days Rec':>10} {'Peak$':>12} {'Trough$':>12}")
print("-" * 110)
for i, dd in enumerate(dd_events, 1):
    rec = dd["recovery"].strftime('%Y-%m-%d') if dd["recovery"] else "Ongoing"
    dr = str(dd["days_rec"]) if dd["days_rec"] else "N/A"
    print(f"{i:<4} {dd['peak'].strftime('%Y-%m-%d'):<14} {dd['trough'].strftime('%Y-%m-%d'):<14} {rec:<14} {dd['depth']:>8.2%} {dd['days_down']:>10} {dr:>10} ${dd['peak_val']:>10,.0f} ${dd['trough_val']:>10,.0f}")

# =============================================================================
# 10. Decade analysis
# =============================================================================
print("\n" + "=" * 70)
print("PERFORMANCE BY DECADE")
print("=" * 70)

decades = [(1986, 1995), (1996, 2005), (2006, 2015), (2016, 2025)]
print(f"\n{'Decade':<14} {'Portfolio':>12} {'TQQQ BH':>12} {'Gold BH':>12} {'NDX BH':>12} {'Port CAGR':>10} {'Port MaxDD':>11}")
print("-" * 85)

for start_y, end_y in decades:
    dec_dates = [d for d in common if start_y <= d.year <= end_y]
    if len(dec_dates) < 2:
        continue
    f, l = dec_dates[0], dec_dates[-1]
    dec_yrs = (l - f).days / 365.25

    pr = port_60.loc[l] / port_60.loc[f] - 1
    tr = tqqq_bh.loc[l] / tqqq_bh.loc[f] - 1
    gr = gold_bh.loc[l] / gold_bh.loc[f] - 1
    nr = ndx_bh.loc[l] / ndx_bh.loc[f] - 1

    p_cagr = (port_60.loc[l] / port_60.loc[f]) ** (1/dec_yrs) - 1

    dec_port = port_60.loc[dec_dates]
    rm = dec_port.cummax()
    dd = (dec_port - rm) / rm
    p_mdd = dd.min()

    print(f"{start_y}-{end_y}     {pr:>11.2%} {tr:>11.2%} {gr:>11.2%} {nr:>11.2%} {p_cagr:>9.2%} {p_mdd:>10.2%}")

# =============================================================================
# 11. Comparison across all backtest periods
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: 60/40 ANNUAL ACROSS DIFFERENT BACKTEST PERIODS")
print("=" * 70)

# 40-year
m40 = metrics(port_60, years)

# 25-year (from 2000)
dates_25 = [d for d in common if d.year >= 2000]
if len(dates_25) > 1:
    port_25, _, _ = simulate(0.60, 'annual', pd.DatetimeIndex(dates_25),
                              tqqq_d.loc[dates_25], gold_d.loc[dates_25])
    yrs_25 = (dates_25[-1] - dates_25[0]).days / 365.25
    m25 = metrics(port_25, yrs_25)

# 20-year (from 2005)
dates_20 = [d for d in common if d.year >= 2005]
if len(dates_20) > 1:
    port_20, _, _ = simulate(0.60, 'annual', pd.DatetimeIndex(dates_20),
                              tqqq_d.loc[dates_20], gold_d.loc[dates_20])
    yrs_20 = (dates_20[-1] - dates_20[0]).days / 365.25
    m20 = metrics(port_20, yrs_20)

print(f"\n{'Period':<25} {'Years':>6} {'Final ($10k)':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9}")
print("-" * 85)
print(f"{'~40yr (1985-2025)':<25} {years:>5.1f} ${m40['final']:>12,.0f} {m40['cagr']:>7.2%} {m40['vol']:>7.2%} {m40['sharpe']:>7.2f} {m40['max_dd']:>8.2%}")
if 'm25' in dir():
    print(f"{'~25yr (2000-2025)':<25} {yrs_25:>5.1f} ${m25['final']:>12,.0f} {m25['cagr']:>7.2%} {m25['vol']:>7.2%} {m25['sharpe']:>7.2f} {m25['max_dd']:>8.2%}")
if 'm20' in dir():
    print(f"{'~20yr (2005-2025)':<25} {yrs_20:>5.1f} ${m20['final']:>12,.0f} {m20['cagr']:>7.2%} {m20['vol']:>7.2%} {m20['sharpe']:>7.2f} {m20['max_dd']:>8.2%}")

# =============================================================================
# 12. Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
40-YEAR BACKTEST: Oct 1985 - Dec 2025

  Data sources:
  - Nasdaq-100 (^NDX) daily from Oct 1985 -> TQQQ simulated as 3x daily - {annual_cost:.2%} friction
  - QQQ daily used instead of ^NDX where available (from Mar 1999)
  - Gold spot monthly (1833-2025) interpolated to daily (pre-Nov 2004)
  - GLD actual daily prices (Nov 2004 onwards)
  - Monthly interpolation for gold is fine because gold has NO leverage/decay

  60/40 TQQQ/Gold Portfolio (annual rebalancing):
    $10,000 -> ${m40['final']:>,.0f}
    CAGR: {m40['cagr']:.2%}
    Max Drawdown: {m40['max_dd']:.2%}
    Sharpe: {m40['sharpe']:.2f}

  Optimal weights (40-year perspective):
    Best CAGR:   {best_cagr['weight']:.0%} TQQQ ({best_cagr['cagr']:.2%} CAGR, {best_cagr['max_dd']:.2%} MaxDD)
    Best Sharpe: {best_sharpe['weight']:.0%} TQQQ ({best_sharpe['cagr']:.2%} CAGR, {best_sharpe['max_dd']:.2%} MaxDD)
    Best Calmar: {best_calmar['weight']:.0%} TQQQ ({best_calmar['cagr']:.2%} CAGR, {best_calmar['max_dd']:.2%} MaxDD)

  Key insight: Over 40 years (vs 20), the strategy looks very different
  because it includes the dot-com crash AND the full 1990s tech bubble.
  The optimal TQQQ weight shifts lower with more history.
""")
