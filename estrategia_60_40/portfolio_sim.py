#!/usr/bin/env python3
"""
Portfolio simulation: 60% TQQQ (simulated) / 40% GLD
- TQQQ simulated as 3x daily return of QQQ
- Annual rebalancing on first trading day of each year
- 20-year backtest
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =============================================================================
# 1. Download historical data
# =============================================================================
print("=" * 70)
print("DOWNLOADING HISTORICAL DATA...")
print("=" * 70)

# GLD launched Nov 2004, QQQ has data from 1999
# We'll use ~20 years of data
start_date = "2005-01-01"
end_date = "2025-12-31"

qqq = yf.download("QQQ", start=start_date, end=end_date, auto_adjust=True, progress=False)
gld = yf.download("GLD", start=start_date, end=end_date, auto_adjust=True, progress=False)

# Also download actual TQQQ for comparison (launched Feb 2010)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end=end_date, auto_adjust=True, progress=False)

# Handle multi-level columns if present
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
if isinstance(gld.columns, pd.MultiIndex):
    gld.columns = gld.columns.get_level_values(0)
if isinstance(tqqq_real.columns, pd.MultiIndex):
    tqqq_real.columns = tqqq_real.columns.get_level_values(0)

print(f"QQQ: {qqq.index[0].strftime('%Y-%m-%d')} to {qqq.index[-1].strftime('%Y-%m-%d')} ({len(qqq)} days)")
print(f"GLD: {gld.index[0].strftime('%Y-%m-%d')} to {gld.index[-1].strftime('%Y-%m-%d')} ({len(gld)} days)")
print(f"TQQQ real: {tqqq_real.index[0].strftime('%Y-%m-%d')} to {tqqq_real.index[-1].strftime('%Y-%m-%d')} ({len(tqqq_real)} days)")

# =============================================================================
# 2. Simulate TQQQ from QQQ
# =============================================================================
print("\n" + "=" * 70)
print("SIMULATING TQQQ (3x daily leverage on QQQ)...")
print("=" * 70)

qqq_returns = qqq["Close"].pct_change()
tqqq_sim_returns = qqq_returns * 3  # 3x daily leverage (simplified, no expense ratio)

# Build TQQQ simulated price series starting at $1
tqqq_sim_price = (1 + tqqq_sim_returns).cumprod()
tqqq_sim_price.iloc[0] = 1.0  # first day

# Also build a version with a rough expense ratio drag (~0.95% annual -> ~0.0038% daily)
daily_expense = 0.0095 / 252
tqqq_sim_returns_exp = qqq_returns * 3 - daily_expense
tqqq_sim_price_exp = (1 + tqqq_sim_returns_exp).cumprod()
tqqq_sim_price_exp.iloc[0] = 1.0

# Validate against real TQQQ
overlap_start = tqqq_real.index[0]
overlap_end = min(tqqq_real.index[-1], tqqq_sim_price.index[-1])
real_period = tqqq_real.loc[overlap_start:overlap_end, "Close"]
sim_period = tqqq_sim_price.loc[overlap_start:overlap_end]

real_total_ret = real_period.iloc[-1] / real_period.iloc[0] - 1
sim_total_ret = sim_period.iloc[-1] / sim_period.iloc[0] - 1
sim_exp_period = tqqq_sim_price_exp.loc[overlap_start:overlap_end]
sim_exp_total_ret = sim_exp_period.iloc[-1] / sim_exp_period.iloc[0] - 1

print(f"\nValidation period: {overlap_start.strftime('%Y-%m-%d')} to {overlap_end.strftime('%Y-%m-%d')}")
print(f"  TQQQ real total return:              {real_total_ret:>10.2%}")
print(f"  TQQQ simulated (no costs):           {sim_total_ret:>10.2%}")
print(f"  TQQQ simulated (with expense ratio): {sim_exp_total_ret:>10.2%}")

# =============================================================================
# 3. Portfolio simulation with annual rebalancing
# =============================================================================
print("\n" + "=" * 70)
print("PORTFOLIO SIMULATION: 60% TQQQ (sim) / 40% GLD")
print("Annual rebalancing on first trading day of each year")
print("=" * 70)

# Use the expense-ratio version for more realism
gld_price = gld["Close"]

# Align dates
common_dates = tqqq_sim_price_exp.index.intersection(gld_price.index)
tqqq_prices = tqqq_sim_price_exp.loc[common_dates]
gld_prices = gld_price.loc[common_dates]

# Normalize GLD to start at 1
gld_prices_norm = gld_prices / gld_prices.iloc[0]

print(f"\nSimulation period: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
print(f"Trading days: {len(common_dates)}")

# Initial portfolio value
initial_value = 10000.0
W_TQQQ = 0.60
W_GLD = 0.40

# Compute daily returns for both assets
tqqq_daily_ret = tqqq_prices.pct_change().fillna(0)
gld_daily_ret = gld_prices_norm.pct_change().fillna(0)

# Simulation
portfolio_value = [initial_value]
tqqq_allocation = [initial_value * W_TQQQ]
gld_allocation = [initial_value * W_GLD]
rebalance_dates = []

current_tqqq = initial_value * W_TQQQ
current_gld = initial_value * W_GLD

for i in range(1, len(common_dates)):
    date = common_dates[i]
    prev_date = common_dates[i - 1]

    # Apply daily returns
    current_tqqq *= (1 + tqqq_daily_ret.iloc[i])
    current_gld *= (1 + gld_daily_ret.iloc[i])

    total = current_tqqq + current_gld

    # Check if we need to rebalance (first trading day of a new year)
    if date.year != prev_date.year:
        current_tqqq = total * W_TQQQ
        current_gld = total * W_GLD
        rebalance_dates.append(date)

    portfolio_value.append(total)
    tqqq_allocation.append(current_tqqq)
    gld_allocation.append(current_gld)

portfolio_series = pd.Series(portfolio_value, index=common_dates)
tqqq_alloc_series = pd.Series(tqqq_allocation, index=common_dates)
gld_alloc_series = pd.Series(gld_allocation, index=common_dates)

# =============================================================================
# 4. Also simulate individual assets and benchmarks
# =============================================================================

# TQQQ only (buy and hold)
tqqq_only = initial_value * tqqq_prices / tqqq_prices.iloc[0]

# GLD only (buy and hold)
gld_only = initial_value * gld_prices_norm / gld_prices_norm.iloc[0]

# QQQ (buy and hold)
qqq_price_aligned = qqq["Close"].loc[common_dates]
qqq_only = initial_value * qqq_price_aligned / qqq_price_aligned.iloc[0]

# SPY for reference
spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_aligned = spy["Close"].reindex(common_dates).ffill()
spy_only = initial_value * spy_aligned / spy_aligned.iloc[0]

# =============================================================================
# 5. Results
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

years = (common_dates[-1] - common_dates[0]).days / 365.25

def calc_stats(series, name):
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
    daily_ret = series.pct_change().dropna()
    volatility = daily_ret.std() * np.sqrt(252)
    sharpe = cagr / volatility if volatility > 0 else 0

    # Max drawdown
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    # Find drawdown start (peak before max drawdown)
    peak_date = series.loc[:max_dd_date].idxmax()

    return {
        "name": name,
        "start_val": series.iloc[0],
        "end_val": series.iloc[-1],
        "total_ret": total_ret,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "max_dd_date": max_dd_date,
        "peak_date": peak_date,
    }

strategies = [
    calc_stats(portfolio_series, "60% TQQQ / 40% GLD (rebal. anual)"),
    calc_stats(tqqq_only, "100% TQQQ (buy & hold)"),
    calc_stats(gld_only, "100% GLD (buy & hold)"),
    calc_stats(qqq_only, "100% QQQ (buy & hold)"),
    calc_stats(spy_only, "100% SPY (buy & hold)"),
]

print(f"\nPeriod: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')} ({years:.1f} years)")
print(f"Initial investment: ${initial_value:,.0f}")
print(f"Rebalancing dates: {len(rebalance_dates)}")
print()

# Print table header
print(f"{'Strategy':<42} {'Final Value':>14} {'Total Return':>14} {'CAGR':>8} {'Volatility':>12} {'Sharpe':>8} {'Max DD':>9}")
print("-" * 110)

for s in strategies:
    print(f"{s['name']:<42} ${s['end_val']:>12,.0f} {s['total_ret']:>13.2%} {s['cagr']:>7.2%} {s['volatility']:>11.2%} {s['sharpe']:>7.2f} {s['max_dd']:>8.2%}")

# =============================================================================
# 6. Year-by-year breakdown
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR PERFORMANCE")
print("=" * 70)

print(f"\n{'Year':<8} {'Portfolio':>12} {'TQQQ Sim':>12} {'GLD':>12} {'QQQ':>12} {'SPY':>12} {'Port. Value':>14} {'TQQQ %':>8} {'GLD %':>8}")
print("-" * 105)

all_years = sorted(set(d.year for d in common_dates))

for year in all_years:
    year_dates = [d for d in common_dates if d.year == year]
    if len(year_dates) < 2:
        continue

    first_day = year_dates[0]
    last_day = year_dates[-1]

    port_ret = portfolio_series.loc[last_day] / portfolio_series.loc[first_day] - 1
    tqqq_ret = tqqq_only.loc[last_day] / tqqq_only.loc[first_day] - 1
    gld_ret = gld_only.loc[last_day] / gld_only.loc[first_day] - 1
    qqq_ret = qqq_only.loc[last_day] / qqq_only.loc[first_day] - 1
    spy_ret = spy_only.loc[last_day] / spy_only.loc[first_day] - 1

    port_val = portfolio_series.loc[last_day]
    tqqq_pct = tqqq_alloc_series.loc[last_day] / portfolio_series.loc[last_day] * 100
    gld_pct = gld_alloc_series.loc[last_day] / portfolio_series.loc[last_day] * 100

    print(f"{year:<8} {port_ret:>11.2%} {tqqq_ret:>11.2%} {gld_ret:>11.2%} {qqq_ret:>11.2%} {spy_ret:>11.2%} ${port_val:>12,.0f} {tqqq_pct:>7.1f}% {gld_pct:>7.1f}%")

# =============================================================================
# 7. Drawdown analysis
# =============================================================================
print("\n" + "=" * 70)
print("MAJOR DRAWDOWNS (Portfolio > -20%)")
print("=" * 70)

rolling_max = portfolio_series.cummax()
drawdown = (portfolio_series - rolling_max) / rolling_max

# Find drawdown periods
in_drawdown = False
dd_start = None
dd_events = []

for i, date in enumerate(common_dates):
    if drawdown.iloc[i] < -0.05 and not in_drawdown:
        in_drawdown = True
        dd_start = portfolio_series.loc[:date].idxmax()
    elif drawdown.iloc[i] >= -0.01 and in_drawdown:
        in_drawdown = False
        dd_end = date
        dd_trough_idx = drawdown.loc[dd_start:dd_end].idxmin()
        dd_depth = drawdown.loc[dd_trough_idx]
        if dd_depth < -0.20:
            dd_events.append({
                "peak": dd_start,
                "trough": dd_trough_idx,
                "recovery": dd_end,
                "depth": dd_depth,
                "peak_val": portfolio_series.loc[dd_start],
                "trough_val": portfolio_series.loc[dd_trough_idx],
            })

if in_drawdown and dd_start is not None:
    dd_trough_idx = drawdown.loc[dd_start:].idxmin()
    dd_depth = drawdown.loc[dd_trough_idx]
    if dd_depth < -0.20:
        dd_events.append({
            "peak": dd_start,
            "trough": dd_trough_idx,
            "recovery": None,
            "depth": dd_depth,
            "peak_val": portfolio_series.loc[dd_start],
            "trough_val": portfolio_series.loc[dd_trough_idx],
        })

print(f"\n{'#':<4} {'Peak Date':<14} {'Trough Date':<14} {'Recovery':<14} {'Drawdown':>10} {'Peak Val':>12} {'Trough Val':>12}")
print("-" * 85)

for i, dd in enumerate(dd_events, 1):
    rec = dd["recovery"].strftime('%Y-%m-%d') if dd["recovery"] else "Not recovered"
    print(f"{i:<4} {dd['peak'].strftime('%Y-%m-%d'):<14} {dd['trough'].strftime('%Y-%m-%d'):<14} {rec:<14} {dd['depth']:>9.2%} ${dd['peak_val']:>10,.0f} ${dd['trough_val']:>10,.0f}")

# =============================================================================
# 8. Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

port = strategies[0]
qqq_s = strategies[3]
spy_s = strategies[4]

print(f"""
Starting with ${initial_value:,.0f} on {common_dates[0].strftime('%Y-%m-%d')}:

  60/40 TQQQ/GLD Portfolio:
    Final value:     ${port['end_val']:>14,.0f}
    Total return:    {port['total_ret']:>14.2%}
    CAGR:            {port['cagr']:>14.2%}
    Volatility:      {port['volatility']:>14.2%}
    Max Drawdown:    {port['max_dd']:>14.2%} (peak: {port['peak_date'].strftime('%Y-%m-%d')}, trough: {port['max_dd_date'].strftime('%Y-%m-%d')})

  vs. QQQ Buy & Hold:
    Final value:     ${qqq_s['end_val']:>14,.0f}
    CAGR:            {qqq_s['cagr']:>14.2%}

  vs. SPY Buy & Hold:
    Final value:     ${spy_s['end_val']:>14,.0f}
    CAGR:            {spy_s['cagr']:>14.2%}

Notes:
  - TQQQ is simulated as 3x daily QQQ return minus ~0.95% annual expense ratio
  - This simulation does NOT account for: borrowing costs, tracking error,
    bid-ask spreads, or potential ETF termination risk
  - Real-world TQQQ returns may differ from simulation
  - The 3x daily reset causes "volatility decay" which can significantly
    erode returns in sideways/volatile markets
""")
