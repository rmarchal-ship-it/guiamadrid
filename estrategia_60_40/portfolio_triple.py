#!/usr/bin/env python3
"""
Portfolio comparison: 3-asset strategy (40% Gold / 40% QQQ / 20% TQQQ)
vs previous strategies, 40-year backtest.
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
spy = yf.download("SPY", start="1993-01-01", end="2025-12-31", auto_adjust=True, progress=False)

for df in [ndx, qqq, gld, tqqq_real, spy]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

# Gold monthly
url = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=30)
gold_monthly = pd.read_csv(io.StringIO(resp.read().decode('utf-8')), parse_dates=['Date'])
gold_monthly = gold_monthly.set_index('Date').sort_index()

# Build composite gold
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
annual_cost = daily_cost * 252

# Build TQQQ and QQQ/NDX series
ndx_ret = ndx["Close"].pct_change()
qqq_start = qqq.index[0]
nasdaq_ret = pd.concat([ndx_ret.loc[:qqq_start].iloc[:-1], qqq_ret.loc[qqq_start:]])
nasdaq_ret = nasdaq_ret[~nasdaq_ret.index.duplicated(keep='last')].sort_index()

tqqq_sim_ret = nasdaq_ret * 3 - daily_cost
tqqq_sim = (1 + tqqq_sim_ret).cumprod()
tqqq_sim.iloc[0] = 1.0

# QQQ/NDX price (for the 1x Nasdaq component)
ndx_price = ndx["Close"].copy()
qqq_price = qqq["Close"].copy()
scale_factor = qqq_price.iloc[0] / ndx_price.loc[:qqq_start].iloc[-1]
ndx_pre = ndx_price.loc[:qqq_start] * scale_factor
nasdaq_price = pd.concat([ndx_pre.iloc[:-1], qqq_price.loc[qqq_start:]])
nasdaq_price = nasdaq_price[~nasdaq_price.index.duplicated(keep='last')].sort_index()

# Align all series
common = tqqq_sim.index.intersection(gold_composite.index).intersection(nasdaq_price.index)
common = common.sort_values()

tqqq_p = tqqq_sim.loc[common] / tqqq_sim.loc[common].iloc[0]
gold_p = gold_composite.loc[common] / gold_composite.loc[common].iloc[0]
qqq_p = nasdaq_price.loc[common] / nasdaq_price.loc[common].iloc[0]

tqqq_d = tqqq_p.pct_change().fillna(0)
gold_d = gold_p.pct_change().fillna(0)
qqq_d = qqq_p.pct_change().fillna(0)

# SPY
spy_aligned = spy["Close"].reindex(common).ffill()
spy_start = spy_aligned.first_valid_index()

years = (common[-1] - common[0]).days / 365.25
print(f"Period: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")

# =============================================================================
# 2. Portfolio simulation (3 assets)
# =============================================================================
def simulate_3(w_tqqq, w_qqq, w_gold, freq, dates, tqqq_ret, qqq_ret, gold_ret, initial=10000):
    cur_t = initial * w_tqqq
    cur_q = initial * w_qqq
    cur_g = initial * w_gold
    values = [initial]
    alloc_t = [cur_t]
    alloc_q = [cur_q]
    alloc_g = [cur_g]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_q *= (1 + qqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_q + cur_g

        d, pd_ = dates[i], dates[i-1]
        rebal = False
        if freq == 'annual':
            rebal = d.year != pd_.year
        elif freq == 'semi':
            rebal = (d.month != pd_.month) and d.month in [1, 7]
        elif freq == 'quarterly':
            rebal = (d.month != pd_.month) and d.month in [1, 4, 7, 10]
        elif freq == 'monthly':
            rebal = d.month != pd_.month

        if rebal:
            cur_t = total * w_tqqq
            cur_q = total * w_qqq
            cur_g = total * w_gold

        values.append(total)
        alloc_t.append(cur_t)
        alloc_q.append(cur_q)
        alloc_g.append(cur_g)

    return (pd.Series(values, index=dates),
            pd.Series(alloc_t, index=dates),
            pd.Series(alloc_q, index=dates),
            pd.Series(alloc_g, index=dates))


def simulate_2(w_a, w_b, freq, dates, ret_a, ret_b, initial=10000):
    cur_a = initial * w_a
    cur_b = initial * w_b
    values = [initial]
    for i in range(1, len(dates)):
        cur_a *= (1 + ret_a.iloc[i])
        cur_b *= (1 + ret_b.iloc[i])
        total = cur_a + cur_b
        d, pd_ = dates[i], dates[i-1]
        rebal = False
        if freq == 'annual':
            rebal = d.year != pd_.year
        elif freq == 'quarterly':
            rebal = (d.month != pd_.month) and d.month in [1,4,7,10]
        elif freq == 'monthly':
            rebal = d.month != pd_.month
        if rebal:
            cur_a = total * w_a
            cur_b = total * w_b
        values.append(total)
    return pd.Series(values, index=dates)


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

    # Time to recover from max DD
    post_trough = series.loc[max_dd_date:]
    recovery_candidates = post_trough[post_trough >= series.loc[peak_date]]
    if len(recovery_candidates) > 0:
        recovery_date = recovery_candidates.index[0]
        recovery_days = (recovery_date - max_dd_date).days
    else:
        recovery_date = None
        recovery_days = None

    return {
        "final": series.iloc[-1], "total_ret": total, "cagr": cagr,
        "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "max_dd_date": max_dd_date, "peak_date": peak_date,
        "calmar": calmar, "recovery_days": recovery_days,
    }

# =============================================================================
# 3. Run strategies
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING STRATEGIES...")
print("=" * 80)

strategies = {}

# YOUR PROPOSAL: 40% Gold / 40% QQQ / 20% TQQQ
for freq in ['annual', 'semi', 'quarterly', 'monthly']:
    s, at, aq, ag = simulate_3(0.20, 0.40, 0.40, freq, common, tqqq_d, qqq_d, gold_d)
    label = f"20T/40Q/40G ({freq})"
    strategies[label] = s

# Previous baseline: 60% TQQQ / 40% Gold
strategies["60T/40G (annual)"] = simulate_2(0.60, 0.40, 'annual', common, tqqq_d, gold_d)

# Vol Target 20% (previous winner)
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
                W = np.clip(target_vol / recent_vol, min_w, max_w)
            cur_t = total * W
            cur_g = total * (1 - W)
        elif d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)
        values.append(total)
    return pd.Series(values, index=dates)

strategies["Vol Target 20%"] = strat_vol_target(common, tqqq_d, gold_d)

# Benchmarks
strategies["100% QQQ (B&H)"] = pd.Series((qqq_p * 10000).values, index=common)
strategies["100% Gold (B&H)"] = pd.Series((gold_p * 10000).values, index=common)

# Also test variations of the 3-asset approach
for t, q, g in [(0.10, 0.50, 0.40), (0.15, 0.45, 0.40),
                 (0.20, 0.40, 0.40), (0.25, 0.35, 0.40),
                 (0.30, 0.30, 0.40), (0.20, 0.50, 0.30),
                 (0.20, 0.30, 0.50), (0.30, 0.40, 0.30),
                 (0.10, 0.40, 0.50)]:
    s, _, _, _ = simulate_3(t, q, g, 'annual', common, tqqq_d, qqq_d, gold_d)
    strategies[f"{int(t*100)}T/{int(q*100)}Q/{int(g*100)}G"] = s

# =============================================================================
# 4. Main comparison table
# =============================================================================
print("\n" + "=" * 80)
print("MAIN COMPARISON: YOUR PROPOSAL vs PREVIOUS STRATEGIES")
print("=" * 80)

main_strats = [
    "20T/40Q/40G (annual)",
    "60T/40G (annual)",
    "Vol Target 20%",
    "100% QQQ (B&H)",
    "100% Gold (B&H)",
]

print(f"\n{'Strategy':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8} {'Rec Days':>9}")
print("-" * 105)

for name in main_strats:
    m = metrics(strategies[name], years)
    rd = f"{m['recovery_days']}" if m['recovery_days'] else "N/A"
    marker = " ***" if "20T/40Q/40G" in name else ""
    print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f} {rd:>9}{marker}")

# =============================================================================
# 5. Rebalancing frequency for 20/40/40
# =============================================================================
print("\n" + "=" * 80)
print("YOUR PROPOSAL: REBALANCING FREQUENCY COMPARISON")
print("=" * 80)

print(f"\n{'Frequency':<28} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 90)
for freq in ['annual', 'semi', 'quarterly', 'monthly']:
    name = f"20T/40Q/40G ({freq})"
    m = metrics(strategies[name], years)
    print(f"{name:<28} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# 6. 3-asset weight sweep (with 40% gold fixed)
# =============================================================================
print("\n" + "=" * 80)
print("3-ASSET WEIGHT SWEEP (annual rebalancing)")
print("=" * 80)

print(f"\n{'TQQQ':>6} {'QQQ':>6} {'Gold':>6} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 82)

sweep_3 = []
for name in sorted(strategies.keys()):
    if 'T/' in name and 'Q/' in name and 'G' in name and '(' not in name:
        parts = name.replace('T/', ' ').replace('Q/', ' ').replace('G', '').split()
        t, q, g = int(parts[0])/100, int(parts[1])/100, int(parts[2])/100
        m = metrics(strategies[name], years)
        m['w_t'] = t
        m['w_q'] = q
        m['w_g'] = g
        sweep_3.append(m)
        print(f"{t:>5.0%} {q:>5.0%} {g:>5.0%} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# 7. Effective leverage analysis
# =============================================================================
print("\n" + "=" * 80)
print("EFFECTIVE LEVERAGE ANALYSIS")
print("=" * 80)

print(f"""
Your 20/40/40 portfolio has an "effective Nasdaq exposure" of:
  20% × 3x + 40% × 1x = 100% Nasdaq equivalent
  Plus 40% gold as hedge

Compare to the original 60/40:
  60% × 3x = 180% Nasdaq equivalent
  Plus 40% gold as hedge

So your proposal cuts Nasdaq exposure from 180% to 100% — almost halved.
This means in a -50% Nasdaq drawdown:
  60/40 TQQQ/Gold: Nasdaq part loses ~97% (3x leverage decay), portfolio ~-58%
  20/40/40:        TQQQ part loses ~97% but QQQ part loses ~50%,
                   portfolio ~ 20%×(-97%) + 40%×(-50%) + 40%×(gold) ≈ -39% + gold hedge
""")

# =============================================================================
# 8. Year-by-year comparison
# =============================================================================
print("=" * 80)
print("YEAR-BY-YEAR: YOUR PROPOSAL vs BASELINE vs VOL TARGET")
print("=" * 80)

port_20_40_40 = strategies["20T/40Q/40G (annual)"]
port_60_40 = strategies["60T/40G (annual)"]
port_vol = strategies["Vol Target 20%"]

print(f"\n{'Year':<7} {'20/40/40':>10} {'60T/40G':>10} {'VolTgt20':>10} {'QQQ':>10} {'Gold':>10} {'20/40/40 Val':>14}")
print("-" * 80)

all_years = sorted(set(d.year for d in common))
for year in all_years:
    yd = [d for d in common if d.year == year]
    if len(yd) < 2:
        continue
    f, l = yd[0], yd[-1]

    r1 = port_20_40_40.loc[l] / port_20_40_40.loc[f] - 1
    r2 = port_60_40.loc[l] / port_60_40.loc[f] - 1
    r3 = port_vol.loc[l] / port_vol.loc[f] - 1
    rq = strategies["100% QQQ (B&H)"].loc[l] / strategies["100% QQQ (B&H)"].loc[f] - 1
    rg = strategies["100% Gold (B&H)"].loc[l] / strategies["100% Gold (B&H)"].loc[f] - 1

    pv = port_20_40_40.loc[l]
    print(f"{year:<7} {r1:>9.2%} {r2:>9.2%} {r3:>9.2%} {rq:>9.2%} {rg:>9.2%} ${pv:>12,.0f}")

# =============================================================================
# 9. Drawdown analysis
# =============================================================================
print("\n" + "=" * 80)
print("DRAWDOWN COMPARISON IN KEY CRISES")
print("=" * 80)

crisis_periods = [
    ("Black Monday 1987", "1987-09-01", "1988-12-31"),
    ("Dot-com crash", "2000-01-01", "2003-12-31"),
    ("Financial crisis", "2007-10-01", "2009-12-31"),
    ("COVID crash", "2020-01-01", "2020-12-31"),
    ("2022 bear market", "2021-11-01", "2023-06-30"),
]

compare_strats = {
    "20/40/40": port_20_40_40,
    "60T/40G": port_60_40,
    "VolTgt20": port_vol,
}

for crisis_name, start, end in crisis_periods:
    print(f"\n--- {crisis_name} ---")
    dates = [d for d in common if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates) < 2:
        continue

    f, l = dates[0], dates[-1]
    print(f"{'Strategy':<15} {'Return':>10} {'Max DD':>10}")
    print("-" * 40)

    for name, s in compare_strats.items():
        ret = s.loc[l] / s.loc[f] - 1
        crisis_s = s.loc[dates]
        rm = crisis_s.cummax()
        dd = (crisis_s - rm) / rm
        mdd = dd.min()
        print(f"{name:<15} {ret:>9.2%} {mdd:>9.2%}")

# =============================================================================
# 10. Major drawdowns for 20/40/40
# =============================================================================
print("\n" + "=" * 80)
print("ALL MAJOR DRAWDOWNS > -20% (20/40/40 annual)")
print("=" * 80)

rolling_max = port_20_40_40.cummax()
drawdown = (port_20_40_40 - rolling_max) / rolling_max

in_dd = False
dd_start = None
dd_events = []

for i, date in enumerate(common):
    if drawdown.iloc[i] < -0.05 and not in_dd:
        in_dd = True
        dd_start = port_20_40_40.loc[:date].idxmax()
    elif drawdown.iloc[i] >= -0.01 and in_dd:
        in_dd = False
        dd_end = date
        trough = drawdown.loc[dd_start:dd_end].idxmin()
        depth = drawdown.loc[trough]
        if depth < -0.20:
            dd_events.append({
                "peak": dd_start, "trough": trough, "recovery": dd_end,
                "depth": depth,
                "days_down": (trough - dd_start).days,
                "days_rec": (dd_end - trough).days,
            })

if in_dd and dd_start is not None:
    trough = drawdown.loc[dd_start:].idxmin()
    depth = drawdown.loc[trough]
    if depth < -0.20:
        dd_events.append({
            "peak": dd_start, "trough": trough, "recovery": None,
            "depth": depth, "days_down": (trough - dd_start).days, "days_rec": None,
        })

print(f"\n{'#':<4} {'Peak':<14} {'Trough':<14} {'Recovery':<14} {'DD':>9} {'Days ↓':>8} {'Days ↑':>8}")
print("-" * 80)
for i, dd in enumerate(dd_events, 1):
    rec = dd["recovery"].strftime('%Y-%m-%d') if dd["recovery"] else "Ongoing"
    dr = str(dd["days_rec"]) if dd["days_rec"] else "N/A"
    print(f"{i:<4} {dd['peak'].strftime('%Y-%m-%d'):<14} {dd['trough'].strftime('%Y-%m-%d'):<14} {rec:<14} {dd['depth']:>8.2%} {dd['days_down']:>8} {dr:>8}")


# =============================================================================
# 11. Final summary
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

m1 = metrics(port_20_40_40, years)
m2 = metrics(port_60_40, years)
m3 = metrics(port_vol, years)

print(f"""
40-YEAR BACKTEST (Oct 1985 - Dec 2025):

  ┌─────────────────────────┬────────────────┬────────────┬────────────┐
  │ Metric                  │ 20/40/40 (tuya)│ 60T/40G    │ VolTgt 20% │
  ├─────────────────────────┼────────────────┼────────────┼────────────┤
  │ Final value ($10k)      │ ${m1['final']:>12,.0f}│ ${m2['final']:>8,.0f}│ ${m3['final']:>8,.0f}│
  │ CAGR                    │ {m1['cagr']:>13.2%} │ {m2['cagr']:>9.2%} │ {m3['cagr']:>9.2%} │
  │ Volatility              │ {m1['vol']:>13.2%} │ {m2['vol']:>9.2%} │ {m3['vol']:>9.2%} │
  │ Sharpe                  │ {m1['sharpe']:>13.2f} │ {m2['sharpe']:>9.2f} │ {m3['sharpe']:>9.2f} │
  │ Max Drawdown            │ {m1['max_dd']:>13.2%} │ {m2['max_dd']:>9.2%} │ {m3['max_dd']:>9.2%} │
  │ Calmar                  │ {m1['calmar']:>13.2f} │ {m2['calmar']:>9.2f} │ {m3['calmar']:>9.2f} │
  │ Recovery (max DD, days) │ {str(m1['recovery_days']):>13} │ {str(m2['recovery_days']):>9} │ {str(m3['recovery_days']):>9} │
  └─────────────────────────┴────────────────┴────────────┴────────────┘

  Effective Nasdaq exposure:
    20/40/40: 20%×3 + 40%×1 = 100% (+ 40% gold)
    60T/40G:  60%×3          = 180% (+ 40% gold)
    VolTgt:   variable       = ~30-120% (+ rest in gold)
""")
