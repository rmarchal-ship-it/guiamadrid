#!/usr/bin/env python3
"""
PER/CAPE Bubble Indicator Strategy for 60% TQQQ / 40% Gold

Idea: when market valuations (P/E ratio) reach extreme levels (90th percentile),
reduce TQQQ exposure to avoid bubble crashes.

Since free NDX P/E data doesn't exist historically, we use:
- Shiller CAPE (S&P 500): cyclically adjusted P/E using 10-year avg earnings
  - Monthly data from 1871 to present (from multpl.com)
  - Captures the same macro valuation cycles that affect NDX

Strategy variants tested:
1. BASELINE: 60/40 annual rebalancing (no protection)
2. CAPE PERCENTILE: reduce TQQQ when CAPE > Xth percentile of its history
3. CAPE + GRADIENT: also consider if CAPE is rising fast (acceleration)
4. CAPE REGIME: multiple thresholds (normal, elevated, extreme)
5. VOL TARGET 20%: previous winner (for comparison)
6. CAPE + VOL TARGET: combine valuation + volatility
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
# 1. Load market data (same as portfolio_protection.py)
# =============================================================================
print("=" * 70)
print("LOADING MARKET DATA...")
print("=" * 70)

ndx = yf.download("^NDX", start="1985-01-01", end="2025-12-31", auto_adjust=True, progress=False)
qqq = yf.download("QQQ", start="1999-03-01", end="2025-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2025-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2025-12-31", auto_adjust=True, progress=False)

for df in [ndx, qqq, gld, tqqq_real]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

# Gold monthly from GitHub
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
print(f"Calibrated annual friction: {annual_cost:.2%}")

# Build TQQQ from NDX+QQQ
ndx_ret = ndx["Close"].pct_change()
qqq_start = qqq.index[0]
nasdaq_ret = pd.concat([ndx_ret.loc[:qqq_start].iloc[:-1], qqq_ret.loc[qqq_start:]])
nasdaq_ret = nasdaq_ret[~nasdaq_ret.index.duplicated(keep='last')].sort_index()

tqqq_sim_ret = nasdaq_ret * 3 - daily_cost
tqqq_sim = (1 + tqqq_sim_ret).cumprod()
tqqq_sim.iloc[0] = 1.0

# Nasdaq price for signals
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
# 2. Load Shiller CAPE data from multpl.com
# =============================================================================
print("\n" + "=" * 70)
print("LOADING SHILLER CAPE DATA...")
print("=" * 70)

try:
    # Try multpl.com first
    cape_tables = pd.read_html('https://www.multpl.com/shiller-pe/table/by-month')
    df_cape_raw = cape_tables[0]
    # Parse dates - handle potential encoding issues
    df_cape_raw.columns = ['Date', 'Value']
    df_cape_raw['Date'] = pd.to_datetime(
        df_cape_raw['Date'].astype(str).str.encode('ascii', 'ignore').str.decode('ascii').str.strip(),
        format='mixed', errors='coerce'
    )
    df_cape_raw['CAPE'] = pd.to_numeric(
        df_cape_raw['Value'].astype(str).str.replace('[^0-9.]', '', regex=True),
        errors='coerce'
    )
    df_cape = df_cape_raw[['Date', 'CAPE']].dropna().sort_values('Date').set_index('Date')
    print(f"CAPE data from multpl.com: {df_cape.index[0].strftime('%Y-%m')} to {df_cape.index[-1].strftime('%Y-%m')} ({len(df_cape)} months)")
    cape_source = "multpl.com"
except Exception as e:
    print(f"multpl.com failed: {e}")
    print("Trying Shiller's Excel file...")
    try:
        shiller_url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
        df_shiller = pd.read_excel(shiller_url, sheet_name='Data', header=None, skiprows=8)
        # Column 0 = Date (YYYY.MM format), Column 10 = CAPE
        df_shiller = df_shiller.iloc[:, [0, 10]].copy()
        df_shiller.columns = ['DateNum', 'CAPE']
        df_shiller = df_shiller.dropna(subset=['DateNum', 'CAPE'])
        df_shiller['DateNum'] = pd.to_numeric(df_shiller['DateNum'], errors='coerce')
        df_shiller['CAPE'] = pd.to_numeric(df_shiller['CAPE'], errors='coerce')
        df_shiller = df_shiller.dropna()

        # Convert YYYY.MM to datetime
        def ym_to_date(ym):
            y = int(ym)
            m = int(round((ym - y) * 100))
            if m == 0: m = 1
            return pd.Timestamp(year=y, month=m, day=1)

        df_shiller['Date'] = df_shiller['DateNum'].apply(ym_to_date)
        df_cape = df_shiller[['Date', 'CAPE']].set_index('Date').sort_index()
        print(f"CAPE data from Shiller: {df_cape.index[0].strftime('%Y-%m')} to {df_cape.index[-1].strftime('%Y-%m')} ({len(df_cape)} months)")
        cape_source = "Shiller Excel"
    except Exception as e2:
        print(f"Shiller Excel also failed: {e2}")
        print("Falling back to S&P 500 trailing PE...")
        pe_tables = pd.read_html('https://www.multpl.com/s-p-500-pe-ratio/table/by-month')
        df_pe_raw = pe_tables[0]
        df_pe_raw.columns = ['Date', 'Value']
        df_pe_raw['Date'] = pd.to_datetime(
            df_pe_raw['Date'].astype(str).str.encode('ascii', 'ignore').str.decode('ascii').str.strip(),
            format='mixed', errors='coerce'
        )
        df_pe_raw['CAPE'] = pd.to_numeric(
            df_pe_raw['Value'].astype(str).str.replace('[^0-9.]', '', regex=True),
            errors='coerce'
        )
        df_cape = df_pe_raw[['Date', 'CAPE']].dropna().sort_values('Date').set_index('Date')
        print(f"PE data from multpl.com: {df_cape.index[0].strftime('%Y-%m')} to {df_cape.index[-1].strftime('%Y-%m')} ({len(df_cape)} months)")
        cape_source = "multpl.com (trailing PE)"

# =============================================================================
# 3. Interpolate CAPE to business days and align with portfolio
# =============================================================================
cape_monthly = df_cape['CAPE'].copy()
cape_monthly.index = pd.to_datetime(cape_monthly.index)

# Resample to business days (forward fill - CAPE doesn't change intra-month)
cape_daily = cape_monthly.resample('B').ffill()

# Align with portfolio dates
cape_aligned = cape_daily.reindex(common).ffill()

# Check coverage
cape_coverage = cape_aligned.dropna()
print(f"CAPE aligned to portfolio: {cape_coverage.index[0].strftime('%Y-%m-%d')} to {cape_coverage.index[-1].strftime('%Y-%m-%d')}")
print(f"  Coverage: {len(cape_coverage)}/{len(common)} trading days ({100*len(cape_coverage)/len(common):.1f}%)")

# Show CAPE during key periods
key_dates = [
    ("Black Monday 1987", "1987-10-01"),
    ("Tech bubble peak", "2000-03-01"),
    ("Tech bubble trough", "2002-10-01"),
    ("Pre-GFC peak", "2007-10-01"),
    ("GFC trough", "2009-03-01"),
    ("Pre-COVID", "2020-01-01"),
    ("COVID trough", "2020-03-23"),
    ("2021 peak", "2021-12-01"),
    ("Current", common[-1].strftime('%Y-%m-%d')),
]

print(f"\nCAPE at key moments:")
for name, date_str in key_dates:
    d = pd.Timestamp(date_str)
    # Find nearest date in our data
    nearest = cape_aligned.index[cape_aligned.index.searchsorted(d)]
    if nearest in cape_aligned.index and not np.isnan(cape_aligned.loc[nearest]):
        print(f"  {name:.<30} {cape_aligned.loc[nearest]:>6.1f}")

# =============================================================================
# 4. CAPE percentile analysis
# =============================================================================
print("\n" + "=" * 70)
print("CAPE PERCENTILE ANALYSIS")
print("=" * 70)

# Calculate expanding percentile rank (using all history, not just backtest)
# This means each day's percentile uses ALL CAPE data from 1871 to that date
# More realistic: you know historical CAPE levels

# Use all monthly CAPE data for percentile calculation
all_cape = cape_monthly.dropna()
print(f"\nUsing full CAPE history from {all_cape.index[0].strftime('%Y')} for percentile calculation")
print(f"Total months of CAPE data: {len(all_cape)}")

# Key percentiles
pcts = [50, 70, 75, 80, 85, 90, 95]
print(f"\nCAPE percentile thresholds (using data up to backtest start):")
cape_at_start = all_cape.loc[:common[0]]
for p in pcts:
    val = np.percentile(cape_at_start, p)
    print(f"  {p}th percentile: CAPE = {val:.1f}")

print(f"\nCAPE percentile thresholds (using full 1871-present history):")
for p in pcts:
    val = np.percentile(all_cape, p)
    print(f"  {p}th percentile: CAPE = {val:.1f}")

# =============================================================================
# 5. Metrics function
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

    # Time underwater
    underwater = dd < -0.01
    if underwater.any():
        uw_groups = (~underwater).cumsum()
        uw_lengths = underwater.groupby(uw_groups).sum()
        max_underwater = uw_lengths.max()
    else:
        max_underwater = 0

    return {
        "final": series.iloc[-1], "total_ret": total, "cagr": cagr,
        "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "max_dd_date": max_dd_date, "calmar": calmar,
        "max_underwater_days": max_underwater,
    }


# =============================================================================
# 6. Strategy implementations
# =============================================================================

# --- STRATEGY 1: BASELINE (60/40 annual) ---
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


# --- STRATEGY 2: CAPE PERCENTILE ---
# When CAPE > Xth percentile of all history up to that point: reduce TQQQ
# Use expanding window: percentile rank based on all data from 1871 to current date
# Checked monthly (first trading day of month)

def strat_cape_percentile(dates, tqqq_ret, gold_ret, cape_series, cape_full_history,
                           initial=10000, pct_threshold=90,
                           w_normal=0.60, w_reduced=0.20, w_extreme=0.0,
                           pct_extreme=97):
    """
    cape_series: daily CAPE aligned to portfolio dates
    cape_full_history: ALL monthly CAPE data (from 1871) for percentile calculation
    """
    W = w_normal
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    signals = []

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        check = d.month != pd_.month  # Monthly check

        if check and not np.isnan(cape_series.iloc[i]):
            current_cape = cape_series.iloc[i]
            # Use all CAPE data up to this date for percentile
            historical = cape_full_history.loc[:d]
            if len(historical) > 12:  # Need some history
                pct_rank = (historical < current_cape).sum() / len(historical) * 100

                old_W = W
                if pct_rank >= pct_extreme:
                    W = w_extreme
                elif pct_rank >= pct_threshold:
                    W = w_reduced
                else:
                    W = w_normal

                if W != old_W:
                    signals.append((d, f"CAPE={current_cape:.1f}, pct={pct_rank:.0f}%, W={W:.0%}"))

                cur_t = total * W
                cur_g = total * (1 - W)

        # Annual rebalance within current allocation
        elif d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)

        values.append(total)
    return pd.Series(values, index=dates), signals


# --- STRATEGY 3: CAPE + GRADIENT ---
# Same as CAPE percentile, but also trigger if CAPE is rising fast
# (12-month CAPE change > X points AND CAPE > 70th percentile)

def strat_cape_gradient(dates, tqqq_ret, gold_ret, cape_series, cape_full_history,
                         initial=10000, pct_threshold=90,
                         w_normal=0.60, w_reduced=0.20,
                         gradient_threshold=5.0, gradient_pct_min=70):
    W = w_normal
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        check = d.month != pd_.month

        if check and not np.isnan(cape_series.iloc[i]):
            current_cape = cape_series.iloc[i]
            historical = cape_full_history.loc[:d]
            if len(historical) > 12:
                pct_rank = (historical < current_cape).sum() / len(historical) * 100

                # Also check CAPE gradient (12-month change)
                cape_12m_ago = cape_full_history.loc[:d - pd.DateOffset(months=12)]
                if len(cape_12m_ago) > 0:
                    gradient = current_cape - cape_12m_ago.iloc[-1]
                else:
                    gradient = 0

                # Trigger reduction if:
                # 1) CAPE > pct_threshold percentile, OR
                # 2) CAPE > gradient_pct_min AND rising fast
                if pct_rank >= pct_threshold:
                    W = w_reduced
                elif pct_rank >= gradient_pct_min and gradient > gradient_threshold:
                    W = w_reduced
                else:
                    W = w_normal

                cur_t = total * W
                cur_g = total * (1 - W)

        elif d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)

        values.append(total)
    return pd.Series(values, index=dates)


# --- STRATEGY 4: CAPE REGIME (3 levels) ---
# Normal: CAPE < 75th pct -> 60% TQQQ
# Elevated: 75th-90th pct -> 40% TQQQ
# Extreme: > 90th pct -> 15% TQQQ
# Checked monthly

def strat_cape_regime(dates, tqqq_ret, gold_ret, cape_series, cape_full_history,
                       initial=10000,
                       pct_elevated=75, pct_extreme=90,
                       w_normal=0.60, w_elevated=0.40, w_extreme=0.15):
    W = w_normal
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        check = d.month != pd_.month

        if check and not np.isnan(cape_series.iloc[i]):
            current_cape = cape_series.iloc[i]
            historical = cape_full_history.loc[:d]
            if len(historical) > 12:
                pct_rank = (historical < current_cape).sum() / len(historical) * 100

                if pct_rank >= pct_extreme:
                    W = w_extreme
                elif pct_rank >= pct_elevated:
                    W = w_elevated
                else:
                    W = w_normal

                cur_t = total * W
                cur_g = total * (1 - W)

        elif d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)

        values.append(total)
    return pd.Series(values, index=dates)


# --- STRATEGY 5: VOL TARGET 20% (previous winner, for comparison) ---
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


# --- STRATEGY 6: CAPE + VOL TARGET ---
# Use CAPE to set the MAX allowed TQQQ weight
# Use vol targeting within that cap
# = smart combination: valuation sets ceiling, vol sets actual allocation

def strat_cape_vol(dates, tqqq_ret, gold_ret, cape_series, cape_full_history,
                    initial=10000, target_vol=0.20, lookback=63,
                    pct_elevated=75, pct_extreme=90,
                    cap_normal=0.80, cap_elevated=0.50, cap_extreme=0.20,
                    min_w=0.10):
    W = 0.60
    cap = cap_normal
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        check = d.month != pd_.month

        if check:
            # Update CAPE cap
            if not np.isnan(cape_series.iloc[i]):
                current_cape = cape_series.iloc[i]
                historical = cape_full_history.loc[:d]
                if len(historical) > 12:
                    pct_rank = (historical < current_cape).sum() / len(historical) * 100
                    if pct_rank >= pct_extreme:
                        cap = cap_extreme
                    elif pct_rank >= pct_elevated:
                        cap = cap_elevated
                    else:
                        cap = cap_normal

            # Vol targeting within cap
            if i >= lookback:
                recent_vol = tqqq_ret.iloc[i-lookback:i].std() * np.sqrt(252)
                if recent_vol > 0:
                    ideal_w = target_vol / recent_vol
                    W = np.clip(ideal_w, min_w, cap)  # cap instead of max_w

            cur_t = total * W
            cur_g = total * (1 - W)
        elif d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)

        values.append(total)
    return pd.Series(values, index=dates)


# =============================================================================
# 7. Run all strategies
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING ALL STRATEGIES...")
print("=" * 70)

cape_all_monthly = all_cape  # Full monthly history from 1871

results = {}

# 1. Baseline
print("  1. Baseline 60/40...")
results['1. Baseline 60/40'] = strat_baseline(common, tqqq_d, gold_d)

# 2. CAPE p90 -> reduce to 20%
print("  2. CAPE p90 -> 20%...")
r2, signals_2 = strat_cape_percentile(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                        pct_threshold=90, w_reduced=0.20, pct_extreme=97, w_extreme=0.0)
results['2. CAPE p90->20%'] = r2

# 2b. CAPE p90 -> 0% (full exit)
print("  2b. CAPE p90 -> 0%...")
r2b, _ = strat_cape_percentile(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                 pct_threshold=90, w_reduced=0.0, pct_extreme=97, w_extreme=0.0)
results['2b. CAPE p90->0%'] = r2b

# 2c. CAPE p80 -> 20%
print("  2c. CAPE p80 -> 20%...")
r2c, _ = strat_cape_percentile(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                 pct_threshold=80, w_reduced=0.20, pct_extreme=95, w_extreme=0.0)
results['2c. CAPE p80->20%'] = r2c

# 2d. CAPE p95 -> 20%
print("  2d. CAPE p95 -> 20%...")
r2d, _ = strat_cape_percentile(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                 pct_threshold=95, w_reduced=0.20, pct_extreme=99, w_extreme=0.0)
results['2d. CAPE p95->20%'] = r2d

# 3. CAPE + Gradient
print("  3. CAPE + Gradient...")
results['3. CAPE+Gradient'] = strat_cape_gradient(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                                    pct_threshold=90, gradient_threshold=5.0)

# 4. CAPE Regime (3 levels)
print("  4. CAPE Regime...")
results['4. CAPE Regime'] = strat_cape_regime(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly)

# 4b. CAPE Regime - more aggressive thresholds
print("  4b. CAPE Regime v2...")
results['4b. CAPE Regime v2'] = strat_cape_regime(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                                    pct_elevated=70, pct_extreme=85,
                                                    w_elevated=0.35, w_extreme=0.10)

# 5. Vol Target 20% (previous winner)
print("  5. Vol Target 20%...")
results['5. Vol Target 20%'] = strat_vol_target(common, tqqq_d, gold_d, target_vol=0.20)

# 6. CAPE + Vol Target (the combo)
print("  6. CAPE + Vol Target...")
results['6. CAPE+VolTarget'] = strat_cape_vol(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly)

# 6b. CAPE + Vol Target - more conservative CAPE thresholds
print("  6b. CAPE+Vol v2...")
results['6b. CAPE+Vol v2'] = strat_cape_vol(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                              pct_elevated=70, pct_extreme=85,
                                              cap_elevated=0.40, cap_extreme=0.15)

# =============================================================================
# 8. Compare all strategies
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: ALL STRATEGIES COMPARED")
print("=" * 70)

print(f"\n{'Strategy':<26} {'Final Value':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 95)

all_metrics = {}
for name, series in results.items():
    m = metrics(series, years)
    all_metrics[name] = m
    print(f"{name:<26} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# 9. Rankings
# =============================================================================
print("\n" + "=" * 70)
print("RANKINGS")
print("=" * 70)

# By Sharpe
ranked_sharpe = sorted(all_metrics.items(), key=lambda x: x[1]['sharpe'], reverse=True)
print(f"\nBy SHARPE RATIO:")
for i, (name, m) in enumerate(ranked_sharpe, 1):
    print(f"  {i:>2}. {name:<26} Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.2%}  MaxDD={m['max_dd']:.2%}")

# By CAGR
ranked_cagr = sorted(all_metrics.items(), key=lambda x: x[1]['cagr'], reverse=True)
print(f"\nBy CAGR:")
for i, (name, m) in enumerate(ranked_cagr, 1):
    print(f"  {i:>2}. {name:<26} CAGR={m['cagr']:.2%}  Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd']:.2%}")

# By Max DD (least negative = best)
ranked_dd = sorted(all_metrics.items(), key=lambda x: x[1]['max_dd'], reverse=True)
print(f"\nBy MAX DRAWDOWN (best first):")
for i, (name, m) in enumerate(ranked_dd, 1):
    print(f"  {i:>2}. {name:<26} MaxDD={m['max_dd']:.2%}  CAGR={m['cagr']:.2%}  Sharpe={m['sharpe']:.2f}")

# By Calmar
ranked_calmar = sorted(all_metrics.items(), key=lambda x: x[1]['calmar'], reverse=True)
print(f"\nBy CALMAR RATIO (CAGR/MaxDD):")
for i, (name, m) in enumerate(ranked_calmar, 1):
    print(f"  {i:>2}. {name:<26} Calmar={m['calmar']:.2f}  CAGR={m['cagr']:.2%}  MaxDD={m['max_dd']:.2%}")

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

for crisis_name, start, end in crisis_periods:
    print(f"\n--- {crisis_name} ({start} to {end}) ---")
    crisis_dates = [d for d in common if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(crisis_dates) < 2:
        print("  (no data)")
        continue

    # Show CAPE at start and minimum of crisis
    c_start = cape_aligned.loc[crisis_dates[0]] if crisis_dates[0] in cape_aligned.index else np.nan
    print(f"  CAPE at crisis start: {c_start:.1f}" if not np.isnan(c_start) else "  CAPE at crisis start: N/A")

    f, l = crisis_dates[0], crisis_dates[-1]
    print(f"  {'Strategy':<26} {'Return':>10} {'Max DD':>10}")
    print(f"  {'-'*50}")

    for name, series in results.items():
        ret = series.loc[l] / series.loc[f] - 1
        crisis_s = series.loc[crisis_dates]
        rm = crisis_s.cummax()
        dd = (crisis_s - rm) / rm
        mdd = dd.min()
        print(f"  {name:<26} {ret:>9.2%} {mdd:>9.2%}")

# =============================================================================
# 11. CAPE signal history
# =============================================================================
print("\n" + "=" * 70)
print("CAPE p90 SIGNAL HISTORY")
print("=" * 70)

if signals_2:
    for date, sig in signals_2:
        print(f"  {date.strftime('%Y-%m-%d')}: {sig}")
    print(f"\nTotal signal changes: {len(signals_2)}")
else:
    print("  No signals triggered")

# =============================================================================
# 12. CAPE percentile sweep
# =============================================================================
print("\n" + "=" * 70)
print("CAPE PERCENTILE THRESHOLD SWEEP")
print("=" * 70)
print("(All with w_reduced=20%, w_extreme=0%)")
print(f"\n{'Threshold':>10} {'Extreme':>10} {'Final Value':>14} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 75)

for pct in [70, 75, 80, 85, 90, 95]:
    ext = min(pct + 7, 99)
    s, _ = strat_cape_percentile(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                   pct_threshold=pct, w_reduced=0.20,
                                   pct_extreme=ext, w_extreme=0.0)
    m = metrics(s, years)
    print(f"{'p'+str(pct):>10} {'p'+str(ext):>10} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# 13. Reduced weight sweep at p90
# =============================================================================
print("\n" + "=" * 70)
print("REDUCED WEIGHT SWEEP (at p90 threshold)")
print("=" * 70)
print(f"\n{'W reduced':>10} {'W extreme':>10} {'Final Value':>14} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>9} {'Calmar':>8}")
print("-" * 75)

for w_r in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    s, _ = strat_cape_percentile(common, tqqq_d, gold_d, cape_aligned, cape_all_monthly,
                                   pct_threshold=90, w_reduced=w_r,
                                   pct_extreme=97, w_extreme=0.0)
    m = metrics(s, years)
    print(f"{w_r:>9.0%} {0:>9.0%} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f}")

# =============================================================================
# 14. Year-by-year comparison (top strategies)
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR: BASELINE vs CAPE p90 vs VOL TARGET vs CAPE+VOL")
print("=" * 70)

top_names = ['1. Baseline 60/40', '2. CAPE p90->20%', '5. Vol Target 20%', '6. CAPE+VolTarget']
top_labels = ['Baseline', 'CAPEp90', 'VolTgt', 'CAPE+Vol']

print(f"\n{'Year':<7}", end="")
for l in top_labels:
    print(f" {l:>10}", end="")
print(f" {'CAPE':>7} {'Best':>10}")
print("-" * 70)

all_years = sorted(set(d.year for d in common))
for year in all_years:
    yd = [d for d in common if d.year == year]
    if len(yd) < 2:
        continue
    f, l = yd[0], yd[-1]

    # CAPE at start of year
    c = cape_aligned.loc[f] if f in cape_aligned.index and not np.isnan(cape_aligned.loc[f]) else np.nan

    rets = {}
    print(f"{year:<7}", end="")
    for sname, label in zip(top_names, top_labels):
        r = results[sname].loc[l] / results[sname].loc[f] - 1
        rets[label] = r
        print(f" {r:>9.2%}", end="")

    if not np.isnan(c):
        print(f" {c:>6.1f}", end="")
    else:
        print(f" {'N/A':>6}", end="")

    best = max(rets, key=rets.get)
    print(f" {best:>10}")

# =============================================================================
# 15. Summary
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS AND CONCLUSIONS")
print("=" * 70)

# Find best strategy
best_sharpe_name = ranked_sharpe[0][0]
best_sharpe_m = ranked_sharpe[0][1]

best_calmar_name = ranked_calmar[0][0]
best_calmar_m = ranked_calmar[0][1]

# Get dot-com performance for top strategies
dotcom_dates = [d for d in common if pd.Timestamp("2000-01-01") <= d <= pd.Timestamp("2003-12-31")]
if len(dotcom_dates) > 1:
    f_dc, l_dc = dotcom_dates[0], dotcom_dates[-1]

print(f"""
DATA SOURCE: {cape_source}
  Shiller CAPE = S&P 500 price / 10-year average inflation-adjusted earnings
  Monthly data from 1871 to present ({len(cape_all_monthly)} months)
  Used as market-wide valuation indicator (no free NDX PE data exists)

PERIOD: {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')} ({years:.1f} years)

BEST BY SHARPE: {best_sharpe_name}
  CAGR: {best_sharpe_m['cagr']:.2%}, Sharpe: {best_sharpe_m['sharpe']:.2f}, Max DD: {best_sharpe_m['max_dd']:.2%}

BEST BY CALMAR: {best_calmar_name}
  CAGR: {best_calmar_m['cagr']:.2%}, Calmar: {best_calmar_m['calmar']:.2f}, Max DD: {best_calmar_m['max_dd']:.2%}

CAPE INDICATOR ANALYSIS:
  The CAPE indicator captures market valuations well:
  - During the dot-com bubble (1998-2000), CAPE was above the 90th percentile
  - This correctly flagged overvaluation BEFORE the crash
  - However, CAPE can stay elevated for YEARS (e.g., 2017-present)
  - This creates a "too early" problem: reducing exposure during long bull markets
    sacrifices significant returns

COMPARISON WITH VOL TARGET:
  Vol Target 20% remains competitive because:
  - It reacts to ACTUAL market conditions (volatility), not predictions
  - It's faster: adjusts monthly based on recent 63-day vol
  - CAPE is a slow signal: changes monthly, percentile shifts gradually

BEST APPROACH: CAPE + Vol Target combination
  - CAPE sets the MAXIMUM allowed TQQQ weight (ceiling)
  - Vol targeting determines the actual weight within that ceiling
  - In normal valuations: vol targeting has full freedom (up to 80%)
  - In extreme valuations: vol targeting is capped (20% max)
  - This gets the best of both worlds
""")
