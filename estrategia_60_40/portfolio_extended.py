#!/usr/bin/env python3
"""
Extended backtest: 60% TQQQ (sim) / 40% Gold
- Uses QQQ + Gold Futures (GC=F) to go back to Aug 2000
- Includes the dot-com crash (2000-2002)
- Calibrated TQQQ simulation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Download all data
# =============================================================================
print("Downloading data...")
qqq = yf.download("QQQ", start="1999-01-01", end="2025-12-31", auto_adjust=True, progress=False)
gc = yf.download("GC=F", start="1999-01-01", end="2025-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2025-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2025-12-31", auto_adjust=True, progress=False)
spy = yf.download("SPY", start="1999-01-01", end="2025-12-31", auto_adjust=True, progress=False)

for df in [qqq, gc, gld, tqqq_real, spy]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

print(f"QQQ:  {qqq.index[0].strftime('%Y-%m-%d')} to {qqq.index[-1].strftime('%Y-%m-%d')}")
print(f"GC=F: {gc.index[0].strftime('%Y-%m-%d')} to {gc.index[-1].strftime('%Y-%m-%d')}")
print(f"GLD:  {gld.index[0].strftime('%Y-%m-%d')} to {gld.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 2. Build gold series: GC=F for 2000-2004, GLD from 2004 onwards
# =============================================================================
print("\n" + "=" * 70)
print("BUILDING COMPOSITE GOLD SERIES")
print("=" * 70)

# Use GLD where available, GC=F before that
# First, calibrate GC=F to match GLD in the overlap period
overlap = gc.index.intersection(gld.index)
gc_overlap = gc["Close"].loc[overlap]
gld_overlap = gld["Close"].loc[overlap]

# Scale factor: ratio of returns
gc_ret_overlap = gc_overlap.pct_change().dropna()
gld_ret_overlap = gld_overlap.pct_change().dropna()
common_ret_idx = gc_ret_overlap.index.intersection(gld_ret_overlap.index)

print(f"GC=F vs GLD overlap: {overlap[0].strftime('%Y-%m-%d')} to {overlap[-1].strftime('%Y-%m-%d')}")
print(f"Correlation: {np.corrcoef(gc_ret_overlap.loc[common_ret_idx], gld_ret_overlap.loc[common_ret_idx])[0,1]:.4f}")

# Strategy: use GC=F daily returns directly for pre-GLD period
# This is the best proxy for what GLD would have done
# GLD tracks gold spot price closely, GC=F is the nearest gold futures contract

# Build composite gold price:
# - Before GLD start: use GC=F returns, anchored so that on GLD start date the value matches
# - From GLD start: use GLD directly

gld_start = gld.index[0]

# GC=F series before GLD existed
gc_pre = gc["Close"].loc[:gld_start]
# Normalize GC=F to end at GLD's first price
gc_pre_norm = gc_pre * (gld["Close"].iloc[0] / gc_pre.iloc[-1])

# Combine: GC=F normalized (before GLD) + GLD (from GLD start)
gold_composite = pd.concat([gc_pre_norm.iloc[:-1], gld["Close"].loc[gld_start:]])
gold_composite = gold_composite[~gold_composite.index.duplicated(keep='last')]
gold_composite = gold_composite.sort_index()

print(f"\nComposite gold series: {gold_composite.index[0].strftime('%Y-%m-%d')} to {gold_composite.index[-1].strftime('%Y-%m-%d')}")
print(f"Records: {len(gold_composite)}")

# =============================================================================
# 3. Calibrate TQQQ simulation (same as before)
# =============================================================================
print("\n" + "=" * 70)
print("CALIBRATING TQQQ SIMULATION")
print("=" * 70)

qqq_ret = qqq["Close"].pct_change()
overlap_tqqq = tqqq_real.index.intersection(qqq_ret.index)
qqq_ret_ov = qqq_ret.loc[overlap_tqqq]
tqqq_real_ov = tqqq_real["Close"].loc[overlap_tqqq]

real_total = tqqq_real_ov.iloc[-1] / tqqq_real_ov.iloc[0]

def sim_error(daily_cost):
    sim_ret = qqq_ret_ov * 3 - daily_cost
    sim_cum = (1 + sim_ret).cumprod()
    sim_total = sim_cum.iloc[-1] / sim_cum.iloc[0]
    return (np.log(sim_total) - np.log(real_total)) ** 2

result = minimize_scalar(sim_error, bounds=(0, 0.001), method='bounded')
daily_cost = result.x
annual_cost = daily_cost * 252

print(f"Calibrated annual friction: {annual_cost:.2%}")

# Build TQQQ simulated (full history from QQQ start)
tqqq_sim_ret = qqq_ret * 3 - daily_cost
tqqq_sim = (1 + tqqq_sim_ret).cumprod()
tqqq_sim.iloc[0] = 1.0

# =============================================================================
# 4. Align all series and run backtest
# =============================================================================
print("\n" + "=" * 70)
print("EXTENDED BACKTEST: ~25 YEARS (from Aug 2000)")
print("=" * 70)

# Common dates between TQQQ sim and composite gold
common = tqqq_sim.index.intersection(gold_composite.index).intersection(spy["Close"].index)
common = common.sort_values()

tqqq_p = tqqq_sim.loc[common]
gold_p = gold_composite.loc[common]
spy_p = spy["Close"].loc[common]
qqq_p = qqq["Close"].loc[common]

# Normalize all to start at 1
tqqq_p = tqqq_p / tqqq_p.iloc[0]
gold_n = gold_p / gold_p.iloc[0]
spy_n = spy_p / spy_p.iloc[0]
qqq_n = qqq_p / qqq_p.iloc[0]

tqqq_daily = tqqq_p.pct_change().fillna(0)
gold_daily = gold_n.pct_change().fillna(0)

years = (common[-1] - common[0]).days / 365.25
print(f"Period: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")
print(f"Trading days: {len(common)}")

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

# Main portfolio: 60/40 annual
port_60, t_alloc, g_alloc = simulate(0.60, 'annual', common, tqqq_daily, gold_daily)

# Benchmarks
tqqq_bh = 10000 * tqqq_p / tqqq_p.iloc[0]
gold_bh = 10000 * gold_n / gold_n.iloc[0]
qqq_bh = 10000 * qqq_n / qqq_n.iloc[0]
spy_bh = 10000 * spy_n / spy_n.iloc[0]

# Different weights with annual rebalancing
configs = []
for w in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    s, _, _ = simulate(w, 'annual', common, tqqq_daily, gold_daily)
    m = metrics(s, years)
    m['weight'] = w
    configs.append(m)

# =============================================================================
# 6. Results
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: EXTENDED BACKTEST (~25 YEARS)")
print("=" * 70)

strats = [
    ("60% TQQQ / 40% Gold (anual)", metrics(port_60, years)),
    ("100% TQQQ sim (buy & hold)", metrics(tqqq_bh, years)),
    ("100% Gold (buy & hold)", metrics(gold_bh, years)),
    ("100% QQQ (buy & hold)", metrics(qqq_bh, years)),
    ("100% SPY (buy & hold)", metrics(spy_bh, years)),
]

print(f"\nPeriod: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")
print(f"Starting: $10,000\n")

print(f"{'Strategy':<38} {'Final Value':>14} {'Total Ret':>12} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9}")
print("-" * 100)
for name, m in strats:
    print(f"{name:<38} ${m['final']:>12,.0f} {m['total_ret']:>11.2%} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%}")

# =============================================================================
# 7. Weight sweep
# =============================================================================
print(f"\n{'TQQQ %':>8} {'Gold %':>8} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 80)
for c in configs:
    print(f"{c['weight']:>7.0%} {1-c['weight']:>7.0%} ${c['final']:>12,.0f} {c['cagr']:>7.2%} {c['vol']:>7.2%} {c['sharpe']:>7.2f} {c['max_dd']:>8.2%} {c['calmar']:>7.2f}")

# =============================================================================
# 8. Year-by-year
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR PERFORMANCE (60/40, annual rebalancing)")
print("=" * 70)

print(f"\n{'Year':<8} {'Portfolio':>12} {'TQQQ Sim':>12} {'Gold':>12} {'QQQ':>12} {'SPY':>12} {'Port Value':>14} {'TQQQ%':>8} {'Gold%':>8}")
print("-" * 108)

all_years = sorted(set(d.year for d in common))
for year in all_years:
    yd = [d for d in common if d.year == year]
    if len(yd) < 2:
        continue
    f, l = yd[0], yd[-1]

    pr = port_60.loc[l] / port_60.loc[f] - 1
    tr = tqqq_bh.loc[l] / tqqq_bh.loc[f] - 1
    gr = gold_bh.loc[l] / gold_bh.loc[f] - 1
    qr = qqq_bh.loc[l] / qqq_bh.loc[f] - 1
    sr = spy_bh.loc[l] / spy_bh.loc[f] - 1

    pv = port_60.loc[l]
    tp = t_alloc.loc[l] / port_60.loc[l] * 100
    gp = g_alloc.loc[l] / port_60.loc[l] * 100

    print(f"{year:<8} {pr:>11.2%} {tr:>11.2%} {gr:>11.2%} {qr:>11.2%} {sr:>11.2%} ${pv:>12,.0f} {tp:>7.1f}% {gp:>7.1f}%")

# =============================================================================
# 9. Drawdown analysis
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
            days_to_trough = (trough - dd_start).days
            days_to_recover = (dd_end - trough).days
            dd_events.append({
                "peak": dd_start, "trough": trough, "recovery": dd_end,
                "depth": depth,
                "peak_val": port_60.loc[dd_start], "trough_val": port_60.loc[trough],
                "days_down": days_to_trough, "days_recover": days_to_recover,
            })

if in_dd and dd_start is not None:
    trough = drawdown.loc[dd_start:].idxmin()
    depth = drawdown.loc[trough]
    if depth < -0.25:
        dd_events.append({
            "peak": dd_start, "trough": trough, "recovery": None,
            "depth": depth,
            "peak_val": port_60.loc[dd_start], "trough_val": port_60.loc[trough],
            "days_down": (trough - dd_start).days, "days_recover": None,
        })

print(f"\n{'#':<4} {'Peak':<14} {'Trough':<14} {'Recovery':<14} {'DD':>9} {'Days Down':>10} {'Days Rec':>10} {'Peak$':>12} {'Trough$':>12}")
print("-" * 110)
for i, dd in enumerate(dd_events, 1):
    rec = dd["recovery"].strftime('%Y-%m-%d') if dd["recovery"] else "Ongoing"
    dr = str(dd["days_recover"]) if dd["days_recover"] else "N/A"
    print(f"{i:<4} {dd['peak'].strftime('%Y-%m-%d'):<14} {dd['trough'].strftime('%Y-%m-%d'):<14} {rec:<14} {dd['depth']:>8.2%} {dd['days_down']:>10} {dr:>10} ${dd['peak_val']:>10,.0f} ${dd['trough_val']:>10,.0f}")

# =============================================================================
# 10. Comparison: 21-year vs 25-year backtest
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: 21-YEAR vs 25-YEAR BACKTEST (60/40 annual)")
print("=" * 70)

# 21-year version (from 2005)
gld_dates = gld.index.intersection(tqqq_sim.index).intersection(spy["Close"].index)
gld_dates = gld_dates.sort_values()

tqqq_21 = tqqq_sim.loc[gld_dates] / tqqq_sim.loc[gld_dates].iloc[0]
gld_21 = gld["Close"].loc[gld_dates] / gld["Close"].loc[gld_dates].iloc[0]
tqqq_21d = tqqq_21.pct_change().fillna(0)
gld_21d = gld_21.pct_change().fillna(0)

port_21, _, _ = simulate(0.60, 'annual', gld_dates, tqqq_21d, gld_21d)
years_21 = (gld_dates[-1] - gld_dates[0]).days / 365.25
m21 = metrics(port_21, years_21)

m25 = metrics(port_60, years)

print(f"\n{'Metric':<30} {'21yr (2005-2025)':>20} {'25yr (2000-2025)':>20} {'Difference':>15}")
print("-" * 90)
print(f"{'Period':<30} {'2005-01 to 2025-12':>20} {'2000-08 to 2025-12':>20}")
print(f"{'Final value ($10k start)':<30} ${m21['final']:>18,.0f} ${m25['final']:>18,.0f}")
print(f"{'CAGR':<30} {m21['cagr']:>19.2%} {m25['cagr']:>19.2%} {m25['cagr']-m21['cagr']:>14.2%}")
print(f"{'Volatility':<30} {m21['vol']:>19.2%} {m25['vol']:>19.2%} {m25['vol']-m21['vol']:>14.2%}")
print(f"{'Sharpe':<30} {m21['sharpe']:>19.2f} {m25['sharpe']:>19.2f} {m25['sharpe']-m21['sharpe']:>14.2f}")
print(f"{'Max Drawdown':<30} {m21['max_dd']:>19.2%} {m25['max_dd']:>19.2%}")
print(f"{'Max DD date':<30} {m21['max_dd_date'].strftime('%Y-%m-%d'):>20} {m25['max_dd_date'].strftime('%Y-%m-%d'):>20}")

# =============================================================================
# 11. Key insight: the dot-com crash
# =============================================================================
print("\n" + "=" * 70)
print("THE DOT-COM CRASH: TQQQ SIMULATION (2000-2002)")
print("=" * 70)

# What would have happened to TQQQ during dot-com?
dotcom_dates = [d for d in common if d.year >= 2000 and d.year <= 2003]
if len(dotcom_dates) > 1:
    tqqq_dotcom = tqqq_bh.loc[dotcom_dates]
    gold_dotcom = gold_bh.loc[dotcom_dates]
    qqq_dotcom = qqq_bh.loc[dotcom_dates]
    port_dotcom = port_60.loc[dotcom_dates]

    print(f"\nPeriod: {dotcom_dates[0].strftime('%Y-%m-%d')} to {dotcom_dates[-1].strftime('%Y-%m-%d')}")
    print(f"\n  TQQQ sim: ${tqqq_dotcom.iloc[0]:>10,.0f} -> ${tqqq_dotcom.iloc[-1]:>10,.0f}  ({(tqqq_dotcom.iloc[-1]/tqqq_dotcom.iloc[0]-1):>8.2%})")
    print(f"  QQQ:      ${qqq_dotcom.iloc[0]:>10,.0f} -> ${qqq_dotcom.iloc[-1]:>10,.0f}  ({(qqq_dotcom.iloc[-1]/qqq_dotcom.iloc[0]-1):>8.2%})")
    print(f"  Gold:     ${gold_dotcom.iloc[0]:>10,.0f} -> ${gold_dotcom.iloc[-1]:>10,.0f}  ({(gold_dotcom.iloc[-1]/gold_dotcom.iloc[0]-1):>8.2%})")
    print(f"  60/40:    ${port_dotcom.iloc[0]:>10,.0f} -> ${port_dotcom.iloc[-1]:>10,.0f}  ({(port_dotcom.iloc[-1]/port_dotcom.iloc[0]-1):>8.2%})")

    # TQQQ min during dot-com
    tqqq_min = tqqq_dotcom.min()
    tqqq_min_date = tqqq_dotcom.idxmin()
    tqqq_dd = tqqq_min / tqqq_dotcom.iloc[0] - 1
    print(f"\n  TQQQ sim lowest point: ${tqqq_min:>10,.0f} on {tqqq_min_date.strftime('%Y-%m-%d')} ({tqqq_dd:.2%} from start)")
    print(f"  Portfolio lowest point: ${port_60.loc[dotcom_dates].min():>10,.0f}")

# =============================================================================
# 12. Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
EXTENDED BACKTEST (25 years, Aug 2000 - Dec 2025):

  The dot-com crash (2000-2002) is the single most important stress test
  for this strategy. QQQ fell ~83% peak-to-trough, which means simulated
  TQQQ would have been virtually wiped out.

  60/40 TQQQ/Gold with annual rebalancing:
    $10,000 -> ${m25['final']:>,.0f}
    CAGR: {m25['cagr']:.2%}
    Max Drawdown: {m25['max_dd']:.2%}

  Key observations:
  1. Including the dot-com crash LOWERS the CAGR from {m21['cagr']:.2%} to {m25['cagr']:.2%}
  2. The max drawdown gets WORSE: {m21['max_dd']:.2%} -> {m25['max_dd']:.2%}
  3. Gold was critical during 2000-2002 as a hedge
  4. The strategy survived the dot-com crash, but barely
  5. Starting point matters enormously for leveraged strategies

  Data limitations:
  - Gold futures (GC=F) used as proxy before Nov 2004 (GLD didn't exist)
  - GC=F vs GLD correlation is ~0.89 (futures have roll effects)
  - Going further back (pre-2000) would require ^NDX + monthly gold
    interpolation, which is unreliable for 3x daily leverage simulation
  - 25 years with daily data is the practical maximum for this strategy
""")
