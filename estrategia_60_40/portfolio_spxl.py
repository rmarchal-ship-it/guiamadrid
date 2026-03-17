#!/usr/bin/env python3
"""
Backtest: 20% SPXL (3x SPY) / 40% SPY / 40% Gold
vs 20% TQQQ (3x QQQ) / 40% QQQ / 40% Gold
Using ^GSPC (S&P 500 index) for maximum history back to 1950.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request, io
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Load all data
# =============================================================================
print("Loading data...")

gspc = yf.download("^GSPC", start="1950-01-01", end="2025-12-31", auto_adjust=True, progress=False)
spy = yf.download("SPY", start="1993-01-01", end="2025-12-31", auto_adjust=True, progress=False)
spxl_real = yf.download("SPXL", start="2008-11-01", end="2025-12-31", auto_adjust=True, progress=False)
ndx = yf.download("^NDX", start="1985-01-01", end="2025-12-31", auto_adjust=True, progress=False)
qqq = yf.download("QQQ", start="1999-03-01", end="2025-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2025-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2025-12-31", auto_adjust=True, progress=False)

for df in [gspc, spy, spxl_real, ndx, qqq, tqqq_real, gld]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

# Gold monthly
url = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=30)
gold_monthly = pd.read_csv(io.StringIO(resp.read().decode('utf-8')), parse_dates=['Date'])
gold_monthly = gold_monthly.set_index('Date').sort_index()

# Build composite gold (interpolated monthly -> GLD where available)
gold_interp = gold_monthly['Price'].resample('B').interpolate(method='linear')
gld_start = gld.index[0]
gc_scale = gld["Close"].iloc[0] / gold_interp.loc[:gld_start].iloc[-1]
gold_pre = gold_interp.loc[:gld_start] * gc_scale
gold_composite = pd.concat([gold_pre.iloc[:-1], gld["Close"].loc[gld_start:]])
gold_composite = gold_composite[~gold_composite.index.duplicated(keep='last')].sort_index()

print(f"^GSPC:      {gspc.index[0].strftime('%Y-%m-%d')} to {gspc.index[-1].strftime('%Y-%m-%d')} ({len(gspc)} days)")
print(f"SPY:        {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")
print(f"SPXL real:  {spxl_real.index[0].strftime('%Y-%m-%d')} to {spxl_real.index[-1].strftime('%Y-%m-%d')}")
print(f"^NDX:       {ndx.index[0].strftime('%Y-%m-%d')} to {ndx.index[-1].strftime('%Y-%m-%d')}")
print(f"Gold:       {gold_composite.index[0].strftime('%Y-%m-%d')} to {gold_composite.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 2. Calibrate both 3x leveraged products
# =============================================================================
print("\n" + "=" * 80)
print("CALIBRATING 3x SIMULATIONS")
print("=" * 80)

# --- Calibrate SPXL (3x SPY) ---
spy_ret = spy["Close"].pct_change()
ov_spxl = spxl_real.index.intersection(spy_ret.index)
spxl_real_total = spxl_real["Close"].loc[ov_spxl].iloc[-1] / spxl_real["Close"].loc[ov_spxl].iloc[0]

def spxl_err(dc):
    sr = spy_ret.loc[ov_spxl] * 3 - dc
    st = (1 + sr).cumprod()
    return (np.log(st.iloc[-1] / st.iloc[0]) - np.log(spxl_real_total)) ** 2

spxl_daily_cost = minimize_scalar(spxl_err, bounds=(0, 0.001), method='bounded').x
spxl_annual_cost = spxl_daily_cost * 252

# Verify
sr_check = spy_ret.loc[ov_spxl] * 3 - spxl_daily_cost
sim_total_spxl = (1 + sr_check).cumprod().iloc[-1] / (1 + sr_check).cumprod().iloc[0]
print(f"\nSPXL (3x SPY):")
print(f"  Calibration period: {ov_spxl[0].strftime('%Y-%m-%d')} to {ov_spxl[-1].strftime('%Y-%m-%d')}")
print(f"  Real SPXL return: {(spxl_real_total-1):.2%}")
print(f"  Simulated return: {(sim_total_spxl-1):.2%}")
print(f"  Annual friction:  {spxl_annual_cost:.2%}")

# --- Calibrate TQQQ (3x QQQ) ---
qqq_ret = qqq["Close"].pct_change()
ov_tqqq = tqqq_real.index.intersection(qqq_ret.index)
tqqq_real_total = tqqq_real["Close"].loc[ov_tqqq].iloc[-1] / tqqq_real["Close"].loc[ov_tqqq].iloc[0]

def tqqq_err(dc):
    sr = qqq_ret.loc[ov_tqqq] * 3 - dc
    st = (1 + sr).cumprod()
    return (np.log(st.iloc[-1] / st.iloc[0]) - np.log(tqqq_real_total)) ** 2

tqqq_daily_cost = minimize_scalar(tqqq_err, bounds=(0, 0.001), method='bounded').x
tqqq_annual_cost = tqqq_daily_cost * 252

print(f"\nTQQQ (3x QQQ):")
print(f"  Annual friction:  {tqqq_annual_cost:.2%}")
print(f"\n  -> SPXL is {'cheaper' if spxl_annual_cost < tqqq_annual_cost else 'more expensive'} than TQQQ ({spxl_annual_cost:.2%} vs {tqqq_annual_cost:.2%})")

# =============================================================================
# 3. Build all price series
# =============================================================================

# --- S&P 500 based series ---
# Use ^GSPC for pre-SPY, SPY where available
gspc_ret = gspc["Close"].pct_change()
spy_start = spy.index[0]
sp500_ret = pd.concat([gspc_ret.loc[:spy_start].iloc[:-1], spy_ret.loc[spy_start:]])
sp500_ret = sp500_ret[~sp500_ret.index.duplicated(keep='last')].sort_index()

# S&P 500 price (for 1x component)
gspc_price = gspc["Close"].copy()
spy_price = spy["Close"].copy()
scale_spy = spy_price.iloc[0] / gspc_price.loc[:spy_start].iloc[-1]
sp500_price = pd.concat([gspc_price.loc[:spy_start] * scale_spy, spy_price.loc[spy_start:].iloc[1:]])
sp500_price = sp500_price[~sp500_price.index.duplicated(keep='last')].sort_index()

# SPXL simulated (3x S&P 500)
spxl_sim_ret = sp500_ret * 3 - spxl_daily_cost
spxl_sim = (1 + spxl_sim_ret).cumprod()
spxl_sim.iloc[0] = 1.0

# --- Nasdaq based series ---
ndx_ret = ndx["Close"].pct_change()
qqq_start = qqq.index[0]
nasdaq_ret = pd.concat([ndx_ret.loc[:qqq_start].iloc[:-1], qqq_ret.loc[qqq_start:]])
nasdaq_ret = nasdaq_ret[~nasdaq_ret.index.duplicated(keep='last')].sort_index()

ndx_price = ndx["Close"].copy()
qqq_price = qqq["Close"].copy()
scale_qqq = qqq_price.iloc[0] / ndx_price.loc[:qqq_start].iloc[-1]
nasdaq_price = pd.concat([ndx_price.loc[:qqq_start] * scale_qqq, qqq_price.loc[qqq_start:].iloc[1:]])
nasdaq_price = nasdaq_price[~nasdaq_price.index.duplicated(keep='last')].sort_index()

tqqq_sim_ret = nasdaq_ret * 3 - tqqq_daily_cost
tqqq_sim = (1 + tqqq_sim_ret).cumprod()
tqqq_sim.iloc[0] = 1.0

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

    post_trough = series.loc[max_dd_date:]
    peak_date = series.loc[:max_dd_date].idxmax()
    rec = post_trough[post_trough >= series.loc[peak_date]]
    recovery_days = (rec.index[0] - max_dd_date).days if len(rec) > 0 else None
    total_dd_days = ((rec.index[0] if len(rec) > 0 else series.index[-1]) - peak_date).days

    return {
        "final": series.iloc[-1], "total_ret": total, "cagr": cagr,
        "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "max_dd_date": max_dd_date, "calmar": calmar,
        "recovery_days": recovery_days, "total_dd_days": total_dd_days,
    }

# =============================================================================
# 5. Simulation function
# =============================================================================
def simulate_3(w_3x, w_1x, w_gold, dates, ret_3x, ret_1x, ret_gold, initial=10000):
    cur_3x = initial * w_3x
    cur_1x = initial * w_1x
    cur_g = initial * w_gold
    values = [initial]
    for i in range(1, len(dates)):
        cur_3x *= (1 + ret_3x.iloc[i])
        cur_1x *= (1 + ret_1x.iloc[i])
        cur_g *= (1 + ret_gold.iloc[i])
        total = cur_3x + cur_1x + cur_g
        if dates[i].year != dates[i-1].year:  # annual rebal
            cur_3x = total * w_3x
            cur_1x = total * w_1x
            cur_g = total * w_gold
        values.append(total)
    return pd.Series(values, index=dates)

def simulate_2(w_a, w_b, dates, ret_a, ret_b, initial=10000):
    cur_a = initial * w_a
    cur_b = initial * w_b
    values = [initial]
    for i in range(1, len(dates)):
        cur_a *= (1 + ret_a.iloc[i])
        cur_b *= (1 + ret_b.iloc[i])
        total = cur_a + cur_b
        if dates[i].year != dates[i-1].year:
            cur_a = total * w_a
            cur_b = total * w_b
        values.append(total)
    return pd.Series(values, index=dates)

# =============================================================================
# PART A: COMPARISON OVER COMMON PERIOD (1985-2025, ~40 years)
# =============================================================================
print("\n" + "=" * 80)
print("PART A: SPY vs QQQ COMPARISON (1985-2025, ~40 years)")
print("=" * 80)

# Common dates for all Nasdaq-based series (from NDX start)
common_ndx = tqqq_sim.index.intersection(nasdaq_price.index).intersection(gold_composite.index).intersection(spxl_sim.index).intersection(sp500_price.index)
common_ndx = common_ndx.sort_values()

# Normalize
def norm(s, dates):
    a = s.loc[dates]
    return (a / a.iloc[0]), a.pct_change().fillna(0)

tqqq_n, tqqq_r = norm(tqqq_sim, common_ndx)
qqq_n, qqq_r = norm(nasdaq_price, common_ndx)
spxl_n, spxl_r = norm(spxl_sim, common_ndx)
spy_n, spy_r = norm(sp500_price, common_ndx)
gold_n, gold_r = norm(gold_composite, common_ndx)

years_40 = (common_ndx[-1] - common_ndx[0]).days / 365.25
print(f"Period: {common_ndx[0].strftime('%Y-%m-%d')} to {common_ndx[-1].strftime('%Y-%m-%d')} ({years_40:.1f} years)")

# Run strategies
strats_40 = {}

# SPY-based
strats_40["20% SPXL/40% SPY/40% Gold"] = simulate_3(0.20, 0.40, 0.40, common_ndx, spxl_r, spy_r, gold_r)
strats_40["60% SPXL/40% Gold"] = simulate_2(0.60, 0.40, common_ndx, spxl_r, gold_r)

# QQQ-based
strats_40["20% TQQQ/40% QQQ/40% Gold"] = simulate_3(0.20, 0.40, 0.40, common_ndx, tqqq_r, qqq_r, gold_r)
strats_40["60% TQQQ/40% Gold"] = simulate_2(0.60, 0.40, common_ndx, tqqq_r, gold_r)

# Benchmarks
strats_40["100% SPY B&H"] = pd.Series((spy_n * 10000).values, index=common_ndx)
strats_40["100% QQQ B&H"] = pd.Series((qqq_n * 10000).values, index=common_ndx)
strats_40["100% Gold B&H"] = pd.Series((gold_n * 10000).values, index=common_ndx)

# Additional SPY-based weights
for t, s, g in [(0.10, 0.50, 0.40), (0.15, 0.45, 0.40), (0.25, 0.35, 0.40),
                 (0.30, 0.30, 0.40), (0.20, 0.50, 0.30), (0.20, 0.30, 0.50)]:
    strats_40[f"{int(t*100)}SPXL/{int(s*100)}SPY/{int(g*100)}G"] = simulate_3(t, s, g, common_ndx, spxl_r, spy_r, gold_r)

print(f"\n{'Strategy':<32} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8} {'RecDays':>8}")
print("-" * 110)

main_order = [
    "20% SPXL/40% SPY/40% Gold",
    "20% TQQQ/40% QQQ/40% Gold",
    "60% SPXL/40% Gold",
    "60% TQQQ/40% Gold",
    "100% SPY B&H",
    "100% QQQ B&H",
    "100% Gold B&H",
]

for name in main_order:
    m = metrics(strats_40[name], years_40)
    rd = str(m['recovery_days']) if m['recovery_days'] else "N/A"
    print(f"{name:<32} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f} {rd:>8}")

# Weight sweep for SPY-based
print(f"\n--- SPXL-based weight sweep (annual rebalancing) ---")
print(f"{'SPXL':>6} {'SPY':>6} {'Gold':>6} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 82)

spxl_sweep = []
for name in sorted(strats_40.keys()):
    if 'SPXL/' in name and 'SPY/' in name and name[0].isdigit():
        m = metrics(strats_40[name], years_40)
        clean = name.replace('SPXL/', ' ').replace('SPY/', ' ').replace('G', '').replace('%', '')
        parts = clean.split()
        t, s, g = int(parts[0])/100, int(parts[1])/100, int(parts[2])/100
        m['w_3x'], m['w_1x'], m['w_g'] = t, s, g
        spxl_sweep.append(m)

spxl_sweep.sort(key=lambda x: x['w_3x'])
for m in spxl_sweep:
    print(f"{m['w_3x']:>5.0%} {m['w_1x']:>5.0%} {m['w_g']:>5.0%} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# PART B: EXTENDED BACKTEST (1950-2025, ~75 years) - SPY only
# =============================================================================
print("\n" + "=" * 80)
print("PART B: EXTENDED BACKTEST S&P 500 (1950-2025, ~75 years)")
print("=" * 80)

# Common dates using ^GSPC + gold
common_ext = spxl_sim.index.intersection(sp500_price.index).intersection(gold_composite.index)
# Gold monthly starts in 1833, but let's use from 1950 where S&P 500 data starts
common_ext = common_ext[common_ext >= '1950-01-01']
common_ext = common_ext.sort_values()

spxl_ne, spxl_re = norm(spxl_sim, common_ext)
spy_ne, spy_re = norm(sp500_price, common_ext)
gold_ne, gold_re = norm(gold_composite, common_ext)

years_ext = (common_ext[-1] - common_ext[0]).days / 365.25
print(f"Period: {common_ext[0].strftime('%Y-%m-%d')} to {common_ext[-1].strftime('%Y-%m-%d')} ({years_ext:.1f} years)")

strats_ext = {}
strats_ext["20% SPXL/40% SPY/40% Gold"] = simulate_3(0.20, 0.40, 0.40, common_ext, spxl_re, spy_re, gold_re)
strats_ext["60% SPXL/40% Gold"] = simulate_2(0.60, 0.40, common_ext, spxl_re, gold_re)
strats_ext["100% SPY B&H"] = pd.Series((spy_ne * 10000).values, index=common_ext)
strats_ext["100% Gold B&H"] = pd.Series((gold_ne * 10000).values, index=common_ext)

# Additional weights
for t, s, g in [(0.10, 0.50, 0.40), (0.15, 0.45, 0.40), (0.20, 0.40, 0.40),
                 (0.25, 0.35, 0.40), (0.30, 0.30, 0.40), (0.35, 0.25, 0.40),
                 (0.40, 0.20, 0.40), (0.20, 0.50, 0.30), (0.20, 0.30, 0.50)]:
    label = f"{int(t*100)}SPXL/{int(s*100)}SPY/{int(g*100)}G"
    if label not in strats_ext:
        strats_ext[label] = simulate_3(t, s, g, common_ext, spxl_re, spy_re, gold_re)

print(f"\n{'Strategy':<32} {'Final Value':>16} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 100)

for name in ["20% SPXL/40% SPY/40% Gold", "60% SPXL/40% Gold", "100% SPY B&H", "100% Gold B&H"]:
    m = metrics(strats_ext[name], years_ext)
    print(f"{name:<32} ${m['final']:>14,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# Extended weight sweep
print(f"\n--- Weight sweep (75 years, annual rebalancing) ---")
print(f"{'SPXL':>6} {'SPY':>6} {'Gold':>6} {'Final Value':>16} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 90)

ext_sweep = []
for name in sorted(strats_ext.keys()):
    if 'SPXL/' in name and 'SPY/' in name and name[0].isdigit():
        m = metrics(strats_ext[name], years_ext)
        clean = name.replace('SPXL/', ' ').replace('SPY/', ' ').replace('G', '').replace('%', '')
        parts = clean.split()
        t, s, g = int(parts[0])/100, int(parts[1])/100, int(parts[2])/100
        m['w_3x'], m['w_1x'], m['w_g'] = t, s, g
        ext_sweep.append(m)

ext_sweep.sort(key=lambda x: x['w_3x'])
for m in ext_sweep:
    print(f"{m['w_3x']:>5.0%} {m['w_1x']:>5.0%} {m['w_g']:>5.0%} ${m['final']:>14,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# PART C: CRISIS COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("PART C: CRISIS COMPARISON (SPY vs QQQ based)")
print("=" * 80)

crises = [
    ("Black Monday 1987", "1987-09-01", "1988-12-31"),
    ("Dot-com crash", "2000-01-01", "2003-12-31"),
    ("Financial crisis 2008", "2007-10-01", "2009-12-31"),
    ("COVID crash 2020", "2020-01-01", "2020-12-31"),
    ("2022 bear market", "2021-11-01", "2023-06-30"),
]

compare = {
    "20%SPXL/40%SPY/40%G": strats_40["20% SPXL/40% SPY/40% Gold"],
    "20%TQQQ/40%QQQ/40%G": strats_40["20% TQQQ/40% QQQ/40% Gold"],
    "60%SPXL/40%Gold": strats_40["60% SPXL/40% Gold"],
    "60%TQQQ/40%Gold": strats_40["60% TQQQ/40% Gold"],
}

for crisis_name, start, end in crises:
    print(f"\n--- {crisis_name} ---")
    dates = [d for d in common_ndx if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates) < 2:
        continue
    f, l = dates[0], dates[-1]
    print(f"{'Strategy':<28} {'Return':>10} {'Max DD':>10}")
    print("-" * 52)
    for name, s in compare.items():
        ret = s.loc[l] / s.loc[f] - 1
        crisis_s = s.loc[dates]
        rm = crisis_s.cummax()
        dd = (crisis_s - rm) / rm
        print(f"{name:<28} {ret:>9.2%} {dd.min():>9.2%}")

# =============================================================================
# PART D: YEAR-BY-YEAR
# =============================================================================
print("\n" + "=" * 80)
print("PART D: YEAR-BY-YEAR (SPY-based vs QQQ-based, 20/40/40)")
print("=" * 80)

port_spy = strats_40["20% SPXL/40% SPY/40% Gold"]
port_qqq = strats_40["20% TQQQ/40% QQQ/40% Gold"]

print(f"\n{'Year':<7} {'SPY-based':>11} {'QQQ-based':>11} {'Diff':>9} {'SPY':>9} {'QQQ':>9} {'Gold':>9}")
print("-" * 72)

spy_wins = 0
qqq_wins = 0
all_years = sorted(set(d.year for d in common_ndx))
for year in all_years:
    yd = [d for d in common_ndx if d.year == year]
    if len(yd) < 2:
        continue
    f, l = yd[0], yd[-1]
    r_spy = port_spy.loc[l] / port_spy.loc[f] - 1
    r_qqq = port_qqq.loc[l] / port_qqq.loc[f] - 1
    r_s = strats_40["100% SPY B&H"].loc[l] / strats_40["100% SPY B&H"].loc[f] - 1
    r_q = strats_40["100% QQQ B&H"].loc[l] / strats_40["100% QQQ B&H"].loc[f] - 1
    r_g = strats_40["100% Gold B&H"].loc[l] / strats_40["100% Gold B&H"].loc[f] - 1
    diff = r_spy - r_qqq
    if r_spy > r_qqq:
        spy_wins += 1
    else:
        qqq_wins += 1
    print(f"{year:<7} {r_spy:>10.2%} {r_qqq:>10.2%} {diff:>8.2%} {r_s:>8.2%} {r_q:>8.2%} {r_g:>8.2%}")

print(f"\nSPY-based wins: {spy_wins} years, QQQ-based wins: {qqq_wins} years")

# =============================================================================
# PART E: DECADE ANALYSIS (extended period)
# =============================================================================
print("\n" + "=" * 80)
print("PART E: DECADE ANALYSIS (20% SPXL/40% SPY/40% Gold, from 1950)")
print("=" * 80)

port_ext = strats_ext["20% SPXL/40% SPY/40% Gold"]
spy_ext = strats_ext["100% SPY B&H"]
gold_ext = strats_ext["100% Gold B&H"]

decades = [(1950,1959),(1960,1969),(1970,1979),(1980,1989),(1990,1999),(2000,2009),(2010,2019),(2020,2025)]

print(f"\n{'Decade':<14} {'Portfolio':>12} {'S&P 500':>12} {'Gold':>12} {'Port CAGR':>10} {'Port MaxDD':>11}")
print("-" * 75)

for sy, ey in decades:
    dec = [d for d in common_ext if sy <= d.year <= ey]
    if len(dec) < 2:
        continue
    f, l = dec[0], dec[-1]
    yrs = (l - f).days / 365.25
    if yrs < 0.5:
        continue

    pr = port_ext.loc[l] / port_ext.loc[f] - 1
    sr = spy_ext.loc[l] / spy_ext.loc[f] - 1
    gr = gold_ext.loc[l] / gold_ext.loc[f] - 1
    pc = (port_ext.loc[l] / port_ext.loc[f]) ** (1/yrs) - 1

    dec_p = port_ext.loc[dec]
    rm = dec_p.cummax()
    dd = (dec_p - rm) / rm
    mdd = dd.min()

    print(f"{sy}-{ey}     {pr:>11.2%} {sr:>11.2%} {gr:>11.2%} {pc:>9.2%} {mdd:>10.2%}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

m_spy34 = metrics(strats_40["20% SPXL/40% SPY/40% Gold"], years_40)
m_qqq34 = metrics(strats_40["20% TQQQ/40% QQQ/40% Gold"], years_40)
m_spy_ext = metrics(strats_ext["20% SPXL/40% SPY/40% Gold"], years_ext)

print(f"""
┌───────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Metric                │ SPY-based (40yr)     │ QQQ-based (40yr)     │ SPY-based (75yr)     │
│                       │ 20%SPXL/40%SPY/40%G  │ 20%TQQQ/40%QQQ/40%G │ 20%SPXL/40%SPY/40%G  │
├───────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Final value ($10k)    │ ${m_spy34['final']:>18,.0f} │ ${m_qqq34['final']:>18,.0f} │ ${m_spy_ext['final']:>18,.0f} │
│ CAGR                  │ {m_spy34['cagr']:>19.2%}  │ {m_qqq34['cagr']:>19.2%}  │ {m_spy_ext['cagr']:>19.2%}  │
│ Volatility            │ {m_spy34['vol']:>19.2%}  │ {m_qqq34['vol']:>19.2%}  │ {m_spy_ext['vol']:>19.2%}  │
│ Sharpe                │ {m_spy34['sharpe']:>19.2f}  │ {m_qqq34['sharpe']:>19.2f}  │ {m_spy_ext['sharpe']:>19.2f}  │
│ Max Drawdown          │ {m_spy34['max_dd']:>19.2%}  │ {m_qqq34['max_dd']:>19.2%}  │ {m_spy_ext['max_dd']:>19.2%}  │
│ Calmar                │ {m_spy34['calmar']:>19.2f}  │ {m_qqq34['calmar']:>19.2f}  │ {m_spy_ext['calmar']:>19.2f}  │
│ Eff. Nasdaq/SP exp.   │ {'100% S&P 500':>19}  │ {'100% Nasdaq':>19}  │ {'100% S&P 500':>19}  │
└───────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘

Key differences:
  - S&P 500 based: lower volatility, smaller drawdowns, more diversified
  - Nasdaq based: higher returns, but much more concentrated in tech
  - The 75-year backtest includes recessions, oil crises, stagflation of the 70s,
    Black Monday, etc. — a much more complete stress test.

Effective exposure comparison:
  20% SPXL + 40% SPY = 20%×3 + 40%×1 = 100% S&P 500 equivalent + 40% Gold
  20% TQQQ + 40% QQQ = 20%×3 + 40%×1 = 100% Nasdaq equivalent + 40% Gold
""")
