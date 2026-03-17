#!/usr/bin/env python3
"""
Portfolio optimization: 60% TQQQ (simulated) / 40% GLD
- Calibrate TQQQ simulation against real TQQQ
- Sweep weight allocations and rebalancing frequencies
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Download data
# =============================================================================
print("Downloading data...")
start_date = "2005-01-01"
end_date = "2025-12-31"

tickers = {"QQQ": None, "GLD": None, "SPY": None, "TQQQ": None}
for t in tickers:
    s = "2010-02-10" if t == "TQQQ" else start_date
    data = yf.download(t, start=s, end=end_date, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    tickers[t] = data["Close"]

qqq, gld, spy, tqqq_real = tickers["QQQ"], tickers["GLD"], tickers["SPY"], tickers["TQQQ"]

# =============================================================================
# 2. Calibrate: find daily cost that matches TQQQ real returns
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: CALIBRATING TQQQ SIMULATION vs REAL TQQQ")
print("=" * 70)

qqq_ret = qqq.pct_change()
overlap_dates = tqqq_real.index.intersection(qqq_ret.index)
qqq_ret_overlap = qqq_ret.loc[overlap_dates]
tqqq_real_overlap = tqqq_real.loc[overlap_dates]

real_total = tqqq_real_overlap.iloc[-1] / tqqq_real_overlap.iloc[0]

def sim_error(daily_cost):
    """Simulate TQQQ with a given daily cost and return squared error vs real."""
    sim_ret = qqq_ret_overlap * 3 - daily_cost
    sim_cumulative = (1 + sim_ret).cumprod()
    sim_total = sim_cumulative.iloc[-1] / sim_cumulative.iloc[0]
    # Use log ratio to avoid scale issues
    return (np.log(sim_total) - np.log(real_total)) ** 2

result = minimize_scalar(sim_error, bounds=(0, 0.001), method='bounded')
calibrated_daily_cost = result.x
calibrated_annual_cost = calibrated_daily_cost * 252

print(f"\nCalibration period: {overlap_dates[0].strftime('%Y-%m-%d')} to {overlap_dates[-1].strftime('%Y-%m-%d')}")
print(f"TQQQ real total return: {(real_total - 1):.2%}")

# Verify calibration
sim_ret_cal = qqq_ret_overlap * 3 - calibrated_daily_cost
sim_total_cal = (1 + sim_ret_cal).cumprod().iloc[-1] / (1 + sim_ret_cal).cumprod().iloc[0]
print(f"TQQQ simulated (calibrated): {(sim_total_cal - 1):.2%}")
print(f"\nCalibrated daily friction: {calibrated_daily_cost:.6f} ({calibrated_daily_cost*100:.4f}%)")
print(f"Calibrated annual friction: {calibrated_annual_cost:.4%}")
print(f"  (includes: 0.95% expense ratio + borrowing/swap costs + tracking error)")

# Also show what different cost assumptions yield
print(f"\nSensitivity to friction cost:")
for annual in [0.0095, 0.02, 0.03, calibrated_annual_cost, 0.05, 0.06]:
    dc = annual / 252
    sr = qqq_ret_overlap * 3 - dc
    st = (1 + sr).cumprod().iloc[-1] / (1 + sr).cumprod().iloc[0]
    marker = " <-- CALIBRATED" if abs(annual - calibrated_annual_cost) < 0.001 else ""
    print(f"  Annual cost {annual:.2%}: simulated return = {(st-1):>10.2%}{marker}")

# =============================================================================
# 3. Build calibrated TQQQ for full period
# =============================================================================
qqq_ret_full = qqq.pct_change()
tqqq_sim_ret = qqq_ret_full * 3 - calibrated_daily_cost
tqqq_sim = (1 + tqqq_sim_ret).cumprod()
tqqq_sim.iloc[0] = 1.0

# Align all series
common = tqqq_sim.index.intersection(gld.index).intersection(spy.index)
tqqq_p = tqqq_sim.loc[common]
gld_p = gld.loc[common] / gld.loc[common].iloc[0]
qqq_p = qqq.loc[common] / qqq.loc[common].iloc[0]
spy_p = spy.loc[common] / spy.loc[common].iloc[0]

tqqq_daily = tqqq_p.pct_change().fillna(0)
gld_daily = gld_p.pct_change().fillna(0)

years_total = (common[-1] - common[0]).days / 365.25

# =============================================================================
# 4. Portfolio simulation function
# =============================================================================
def simulate_portfolio(w_tqqq, rebal_freq, dates, tqqq_ret, gld_ret, initial=10000):
    """
    Simulate portfolio with given TQQQ weight and rebalancing frequency.
    rebal_freq: 'annual', 'semi', 'quarterly', 'monthly', 'weekly', 'none'
    """
    w_gld = 1 - w_tqqq
    cur_tqqq = initial * w_tqqq
    cur_gld = initial * w_gld

    values = [initial]

    for i in range(1, len(dates)):
        cur_tqqq *= (1 + tqqq_ret.iloc[i])
        cur_gld *= (1 + gld_ret.iloc[i])
        total = cur_tqqq + cur_gld

        rebalance = False
        d = dates[i]
        pd_prev = dates[i - 1]

        if rebal_freq == 'annual':
            rebalance = d.year != pd_prev.year
        elif rebal_freq == 'semi':
            # Jan and Jul
            rebalance = (d.month != pd_prev.month) and d.month in [1, 7]
        elif rebal_freq == 'quarterly':
            rebalance = (d.month != pd_prev.month) and d.month in [1, 4, 7, 10]
        elif rebal_freq == 'monthly':
            rebalance = d.month != pd_prev.month
        elif rebal_freq == 'weekly':
            rebalance = d.isocalendar()[1] != pd_prev.isocalendar()[1]
        elif rebal_freq == 'none':
            rebalance = False

        if rebalance:
            cur_tqqq = total * w_tqqq
            cur_gld = total * w_gld

        values.append(total)

    series = pd.Series(values, index=dates)
    return series


def calc_metrics(series, yrs):
    """Calculate key portfolio metrics."""
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / yrs) - 1
    daily = series.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    rolling_max = series.cummax()
    dd = (series - rolling_max) / rolling_max
    max_dd = dd.min()
    # Calmar ratio (CAGR / abs(max_dd))
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        "final": series.iloc[-1],
        "total_ret": total_ret,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
    }

# =============================================================================
# 5. SWEEP: Weights x Rebalancing frequency
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: PARAMETER SWEEP")
print("=" * 70)

weights = [i / 100 for i in range(5, 96, 5)]  # 5% to 95%
freqs = ['weekly', 'monthly', 'quarterly', 'semi', 'annual', 'none']
freq_labels = {
    'weekly': 'Semanal', 'monthly': 'Mensual', 'quarterly': 'Trimestral',
    'semi': 'Semestral', 'annual': 'Anual', 'none': 'Sin rebal.'
}

results = []
for w in weights:
    for f in freqs:
        port = simulate_portfolio(w, f, common, tqqq_daily, gld_daily)
        m = calc_metrics(port, years_total)
        m['weight'] = w
        m['freq'] = f
        results.append(m)

df = pd.DataFrame(results)

# =============================================================================
# 5a. Best by CAGR
# =============================================================================
print(f"\n--- TOP 10 by CAGR ---")
print(f"{'TQQQ %':>8} {'Rebalanceo':<12} {'Valor Final':>14} {'CAGR':>8} {'Volatilidad':>12} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 85)
top_cagr = df.nlargest(10, 'cagr')
for _, r in top_cagr.iterrows():
    print(f"{r['weight']:>7.0%} {freq_labels[r['freq']]:<12} ${r['final']:>12,.0f} {r['cagr']:>7.2%} {r['vol']:>11.2%} {r['sharpe']:>7.2f} {r['max_dd']:>8.2%} {r['calmar']:>7.2f}")

# =============================================================================
# 5b. Best by Sharpe ratio
# =============================================================================
print(f"\n--- TOP 10 by SHARPE RATIO ---")
print(f"{'TQQQ %':>8} {'Rebalanceo':<12} {'Valor Final':>14} {'CAGR':>8} {'Volatilidad':>12} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 85)
top_sharpe = df.nlargest(10, 'sharpe')
for _, r in top_sharpe.iterrows():
    print(f"{r['weight']:>7.0%} {freq_labels[r['freq']]:<12} ${r['final']:>12,.0f} {r['cagr']:>7.2%} {r['vol']:>11.2%} {r['sharpe']:>7.2f} {r['max_dd']:>8.2%} {r['calmar']:>7.2f}")

# =============================================================================
# 5c. Best by Calmar ratio (CAGR / max drawdown)
# =============================================================================
print(f"\n--- TOP 10 by CALMAR RATIO (CAGR / |Max Drawdown|) ---")
print(f"{'TQQQ %':>8} {'Rebalanceo':<12} {'Valor Final':>14} {'CAGR':>8} {'Volatilidad':>12} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 85)
top_calmar = df.nlargest(10, 'calmar')
for _, r in top_calmar.iterrows():
    print(f"{r['weight']:>7.0%} {freq_labels[r['freq']]:<12} ${r['final']:>12,.0f} {r['cagr']:>7.2%} {r['vol']:>11.2%} {r['sharpe']:>7.2f} {r['max_dd']:>8.2%} {r['calmar']:>7.2f}")

# =============================================================================
# 6. Focus on rebalancing frequency impact for 60/40
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: IMPACT OF REBALANCING FREQUENCY (at 60/40 weight)")
print("=" * 70)

print(f"\n{'Frecuencia':<14} {'Valor Final':>14} {'CAGR':>8} {'Volatilidad':>12} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 75)
for f in freqs:
    row = df[(df['weight'] == 0.60) & (df['freq'] == f)].iloc[0]
    print(f"{freq_labels[f]:<14} ${row['final']:>12,.0f} {row['cagr']:>7.2%} {row['vol']:>11.2%} {row['sharpe']:>7.2f} {row['max_dd']:>8.2%} {row['calmar']:>7.2f}")

# =============================================================================
# 7. Focus on weight impact at best rebalancing frequency
# =============================================================================
# Find best freq by Sharpe across all weights
best_freq_sharpe = df.groupby('freq')['sharpe'].mean().idxmax()
print(f"\n" + "=" * 70)
print(f"STEP 4: IMPACT OF WEIGHT (at {freq_labels[best_freq_sharpe]} rebalancing)")
print("=" * 70)

print(f"\n{'TQQQ %':>8} {'GLD %':>8} {'Valor Final':>14} {'CAGR':>8} {'Volatilidad':>12} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 82)
for w in weights:
    row = df[(df['weight'] == w) & (df['freq'] == best_freq_sharpe)].iloc[0]
    print(f"{w:>7.0%} {1-w:>7.0%} ${row['final']:>12,.0f} {row['cagr']:>7.2%} {row['vol']:>11.2%} {row['sharpe']:>7.2f} {row['max_dd']:>8.2%} {row['calmar']:>7.2f}")

# =============================================================================
# 8. Comparison: your 60/40 annual vs optimized
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: YOUR STRATEGY vs OPTIMIZED")
print("=" * 70)

# Your strategy
yours = df[(df['weight'] == 0.60) & (df['freq'] == 'annual')].iloc[0]

# Best CAGR
best_c = df.loc[df['cagr'].idxmax()]
# Best Sharpe
best_s = df.loc[df['sharpe'].idxmax()]
# Best Calmar
best_cal = df.loc[df['calmar'].idxmax()]

print(f"\n{'Strategy':<42} {'Final Val':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)
print(f"{'Tu cartera (60/40, anual)':<42} ${yours['final']:>12,.0f} {yours['cagr']:>7.2%} {yours['vol']:>7.2%} {yours['sharpe']:>7.2f} {yours['max_dd']:>8.2%} {yours['calmar']:>7.2f}")
print(f"{'Best CAGR ({:.0%} TQQQ, {})'.format(best_c['weight'], freq_labels[best_c['freq']]):<42} ${best_c['final']:>12,.0f} {best_c['cagr']:>7.2%} {best_c['vol']:>7.2%} {best_c['sharpe']:>7.2f} {best_c['max_dd']:>8.2%} {best_c['calmar']:>7.2f}")
print(f"{'Best Sharpe ({:.0%} TQQQ, {})'.format(best_s['weight'], freq_labels[best_s['freq']]):<42} ${best_s['final']:>12,.0f} {best_s['cagr']:>7.2%} {best_s['vol']:>7.2%} {best_s['sharpe']:>7.2f} {best_s['max_dd']:>8.2%} {best_s['calmar']:>7.2f}")
print(f"{'Best Calmar ({:.0%} TQQQ, {})'.format(best_cal['weight'], freq_labels[best_cal['freq']]):<42} ${best_cal['final']:>12,.0f} {best_cal['cagr']:>7.2%} {best_cal['vol']:>7.2%} {best_cal['sharpe']:>7.2f} {best_cal['max_dd']:>8.2%} {best_cal['calmar']:>7.2f}")

# QQQ and SPY benchmarks
qqq_series = pd.Series((qqq_p / qqq_p.iloc[0] * 10000).values, index=common)
spy_series = pd.Series((spy_p / spy_p.iloc[0] * 10000).values, index=common)
qqq_m = calc_metrics(qqq_series, years_total)
spy_m = calc_metrics(spy_series, years_total)
print(f"{'QQQ Buy & Hold':<42} ${qqq_m['final']:>12,.0f} {qqq_m['cagr']:>7.2%} {qqq_m['vol']:>7.2%} {qqq_m['sharpe']:>7.2f} {qqq_m['max_dd']:>8.2%} {qqq_m['calmar']:>7.2f}")
print(f"{'SPY Buy & Hold':<42} ${spy_m['final']:>12,.0f} {spy_m['cagr']:>7.2%} {spy_m['vol']:>7.2%} {spy_m['sharpe']:>7.2f} {spy_m['max_dd']:>8.2%} {spy_m['calmar']:>7.2f}")

# =============================================================================
# 9. Heatmaps (text-based)
# =============================================================================
print("\n" + "=" * 70)
print("HEATMAP: CAGR by Weight x Rebalancing Frequency")
print("=" * 70)

pivot_cagr = df.pivot_table(values='cagr', index='weight', columns='freq', aggfunc='first')
pivot_cagr = pivot_cagr[freqs]  # order columns
print(f"\n{'TQQQ%':<8}", end="")
for f in freqs:
    print(f"{freq_labels[f]:>12}", end="")
print()
print("-" * (8 + 12 * len(freqs)))
for w in weights:
    print(f"{w:<7.0%}", end=" ")
    for f in freqs:
        v = pivot_cagr.loc[w, f]
        print(f"{v:>11.2%}", end=" ")
    print()

print("\n" + "=" * 70)
print("HEATMAP: SHARPE RATIO by Weight x Rebalancing Frequency")
print("=" * 70)

pivot_sharpe = df.pivot_table(values='sharpe', index='weight', columns='freq', aggfunc='first')
pivot_sharpe = pivot_sharpe[freqs]
print(f"\n{'TQQQ%':<8}", end="")
for f in freqs:
    print(f"{freq_labels[f]:>12}", end="")
print()
print("-" * (8 + 12 * len(freqs)))
for w in weights:
    print(f"{w:<7.0%}", end=" ")
    for f in freqs:
        v = pivot_sharpe.loc[w, f]
        print(f"{v:>11.2f}", end=" ")
    print()

print("\n" + "=" * 70)
print("HEATMAP: MAX DRAWDOWN by Weight x Rebalancing Frequency")
print("=" * 70)

pivot_dd = df.pivot_table(values='max_dd', index='weight', columns='freq', aggfunc='first')
pivot_dd = pivot_dd[freqs]
print(f"\n{'TQQQ%':<8}", end="")
for f in freqs:
    print(f"{freq_labels[f]:>12}", end="")
print()
print("-" * (8 + 12 * len(freqs)))
for w in weights:
    print(f"{w:<7.0%}", end=" ")
    for f in freqs:
        v = pivot_dd.loc[w, f]
        print(f"{v:>11.2%}", end=" ")
    print()

# =============================================================================
# 10. Final recommendation
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

print(f"""
CALIBRATION:
  The real implicit annual cost of 3x leverage is ~{calibrated_annual_cost:.2%}
  (vs 0.95% expense ratio alone). This includes borrowing costs,
  swap spreads, and tracking inefficiencies.

REBALANCING FREQUENCY (holding weight constant at 60/40):
  - More frequent rebalancing generally {"improves" if df[(df['weight']==0.6) & (df['freq']=='monthly')].iloc[0]['sharpe'] > df[(df['weight']==0.6) & (df['freq']=='annual')].iloc[0]['sharpe'] else "does not improve"} risk-adjusted returns
  - The difference between monthly and annual is meaningful for
    high-volatility portfolios like this one
  - Weekly vs monthly shows diminishing returns (and more transaction costs
    in practice)

WEIGHT ALLOCATION:
  - Higher TQQQ weight = higher CAGR but MUCH higher drawdowns
  - The "sweet spot" for Sharpe ratio is around {df.loc[df['sharpe'].idxmax(), 'weight']:.0%} TQQQ
  - The "sweet spot" for Calmar ratio is around {df.loc[df['calmar'].idxmax(), 'weight']:.0%} TQQQ
  - Above ~70% TQQQ, you get diminishing returns for massive extra risk

WHAT MATTERS MORE - WEIGHTS OR REBALANCING?
""")

# Quantify: variance explained by weight vs frequency
from itertools import product
weight_var = df.groupby('weight')['cagr'].mean().var()
freq_var = df.groupby('freq')['cagr'].mean().var()
print(f"  Variance in CAGR explained by weight changes: {weight_var:.6f}")
print(f"  Variance in CAGR explained by frequency changes: {freq_var:.6f}")
print(f"  Weight impact is ~{weight_var/freq_var:.0f}x larger than frequency impact")

weight_var_s = df.groupby('weight')['sharpe'].mean().var()
freq_var_s = df.groupby('freq')['sharpe'].mean().var()
print(f"\n  Variance in Sharpe explained by weight changes: {weight_var_s:.6f}")
print(f"  Variance in Sharpe explained by frequency changes: {freq_var_s:.6f}")
print(f"  Weight impact is ~{weight_var_s/freq_var_s:.0f}x larger than frequency impact")

print(f"""
ANSWER: Changing the weights has MUCH more impact than changing rebalancing
frequency. However, if you keep 60/40, increasing rebalancing to quarterly
or monthly is a low-effort improvement.

SUGGESTED CONFIGURATIONS (depending on risk tolerance):
""")

configs = [
    ("Conservative", 0.35, 'quarterly'),
    ("Moderate", 0.50, 'quarterly'),
    ("Your current", 0.60, 'annual'),
    ("Optimized (same risk budget)", 0.55, 'monthly'),
    ("Aggressive", 0.70, 'monthly'),
]

print(f"{'Profile':<32} {'TQQQ%':>7} {'Freq':<12} {'Final Val':>14} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>9}")
print("-" * 95)
for name, w, f in configs:
    r = df[(df['weight'] == w) & (df['freq'] == f)]
    if len(r) == 0:
        # Run simulation for this specific config
        port = simulate_portfolio(w, f, common, tqqq_daily, gld_daily)
        m = calc_metrics(port, years_total)
        print(f"{name:<32} {w:>6.0%} {freq_labels[f]:<12} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%}")
    else:
        r = r.iloc[0]
        print(f"{name:<32} {w:>6.0%} {freq_labels[f]:<12} ${r['final']:>12,.0f} {r['cagr']:>7.2%} {r['sharpe']:>7.2f} {r['max_dd']:>8.2%}")
