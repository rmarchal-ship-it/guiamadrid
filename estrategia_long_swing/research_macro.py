"""
Long Swing — Auto-research: macro filters + position sizing.

Downloads data ONCE (120m), computes trades ONCE, then iterates simulate()
with different configs. Fast iteration: ~2s per variant after initial download.

Usage: uv run python -m long_swing.research_macro
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta

from long_swing import strategy as strat
from long_swing.config import UNIVERSE, INITIAL, TUNED_RSI, BEST_EMA

# ── Research grid ────────────────────────────────────────────────────────
LOOKBACK_RESEARCH = 3652  # 120 months (~10 years)

MACRO_FILTERS = {
    "NO_FILTER":    lambda sig, spy_ctx: True,
    "SPY>SMA50":    lambda sig, spy_ctx: spy_ctx.get(sig["entry_date"], {}).get("spy_above_sma50", 1) == 1,
    "SPY>SMA200":   lambda sig, spy_ctx: spy_ctx.get(sig["entry_date"], {}).get("spy_above_sma200", 1) == 1,
    "VIX<30":       lambda sig, spy_ctx: sig.get("vix", 0) < 30,
    "VIX<25":       lambda sig, spy_ctx: sig.get("vix", 0) < 25,
    "SMA50+VIX30":  lambda sig, spy_ctx: (spy_ctx.get(sig["entry_date"], {}).get("spy_above_sma50", 1) == 1) and sig.get("vix", 0) < 30,
    "SMA200+VIX30": lambda sig, spy_ctx: (spy_ctx.get(sig["entry_date"], {}).get("spy_above_sma200", 1) == 1) and sig.get("vix", 0) < 30,
}

MAX_POS_GRID = [3, 4, 5, 6]
RESERVE_GRID = [0.15, 0.20]


def get_macro_context_extended(all_data, lookback):
    """Like get_macro_context but adds SMA200 and uses correct lookback for VIX."""
    from trading.data import DataRequest, download_ohlcv

    spy_df = all_data.get("SPY")
    spy_context = {}
    if spy_df is not None:
        close = spy_df["Close"].astype(float)
        high_52w = close.rolling(252, min_periods=50).max()
        pct_from_high = ((close - high_52w) / high_52w * 100)
        sma50 = ta.sma(close, length=50)
        sma200 = ta.sma(close, length=200)
        for dt in spy_df.index:
            spy_context[dt.date()] = {
                "spy_pct_from_high": pct_from_high.get(dt, 0),
                "spy_above_sma50": 1 if (sma50 is not None and dt in sma50.index and close[dt] > sma50[dt]) else 0,
                "spy_above_sma200": 1 if (sma200 is not None and dt in sma200.index and close[dt] > sma200[dt]) else 0,
            }

    vix_df = download_ohlcv(DataRequest("^VIX", interval="1d", lookback_days=lookback + 60))
    vix_map = {}
    if vix_df is not None and not vix_df.empty:
        for dt in vix_df.index:
            vix_map[dt.date()] = float(vix_df.loc[dt, "Close"])

    return spy_context, vix_map


def simulate_variant(signals_df, all_data, spy_ctx, macro_filter_fn,
                     max_pos, reserve_pct, pyr_threshold=0.15, pyr_size=1.00,
                     initial=INITIAL, cost_bps=4, fixed_years=None):
    """Run simulation with specific params. Filters signals by macro condition first."""
    if fixed_years is None:
        fixed_years = LOOKBACK_RESEARCH / 365.25

    # Apply macro filter to signals
    mask = signals_df.apply(lambda row: macro_filter_fn(row, spy_ctx), axis=1)
    filtered = signals_df[mask].reset_index(drop=True)

    if filtered.empty:
        return {"final": initial, "cagr": 0, "dd": 0, "wr": 0, "trades": 0,
                "pf": 0, "skipped": 0, "n_pyr": 0, "avg_hold_d": 0, "signals_used": 0}

    # Monkey-patch strategy module for simulate()
    orig = {
        "MAX_POS": strat.MAX_POS, "RESERVE_PCT": strat.RESERVE_PCT,
        "PYR_THRESHOLD": strat.PYR_THRESHOLD, "PYR_SIZE": strat.PYR_SIZE,
        "INITIAL": strat.INITIAL, "COST_BPS": strat.COST_BPS,
        "FIXED_YEARS": strat.FIXED_YEARS, "LOOKBACK": strat.LOOKBACK,
    }
    try:
        strat.MAX_POS = max_pos
        strat.RESERVE_PCT = reserve_pct
        strat.PYR_THRESHOLD = pyr_threshold
        strat.PYR_SIZE = pyr_size
        strat.INITIAL = initial
        strat.COST_BPS = cost_bps
        strat.FIXED_YEARS = fixed_years
        strat.LOOKBACK = LOOKBACK_RESEARCH

        log, still_open, metrics = strat.simulate(filtered, all_data)
        metrics["signals_used"] = len(filtered)
        return metrics
    finally:
        # Restore originals
        for k, v in orig.items():
            setattr(strat, k, v)


def main():
    print("=" * 130)
    print("  LONG SWING — Auto-Research: Macro Filters + Position Sizing")
    print(f"  Lookback: {LOOKBACK_RESEARCH}d (~{LOOKBACK_RESEARCH/365.25:.0f}y) | Initial: {INITIAL:,.0f}€")
    print("=" * 130)

    # ── Step 1: Download once ──
    strat.LOOKBACK = LOOKBACK_RESEARCH
    all_data = strat.download_daily(UNIVERSE, lookback_days=LOOKBACK_RESEARCH + 60)
    print(f"\n  Tickers loaded: {len(all_data)} / {len(UNIVERSE)}")

    # ── Step 2: Macro context (extended with SMA200) ──
    spy_ctx, vix_map = get_macro_context_extended(all_data, LOOKBACK_RESEARCH)

    # ── Step 3: Compute all trades once ──
    # Need to patch LOOKBACK for VIX download inside compute_all_trades
    orig_lookback = strat.LOOKBACK
    strat.LOOKBACK = LOOKBACK_RESEARCH
    trades_df = strat.compute_all_trades(all_data, spy_ctx, vix_map)
    strat.LOOKBACK = orig_lookback

    signals = strat.select_daily_signals(trades_df)
    print(f"  Total signals: {len(signals)}")
    if signals.empty:
        print("  No signals — aborting.")
        return
    print(f"  Period: {signals['entry_ts'].min().date()} → {signals['exit_ts'].max().date()}")

    # ── Step 4: Grid search ──
    results = []
    total = len(MACRO_FILTERS) * len(MAX_POS_GRID) * len(RESERVE_GRID)
    print(f"\n  Running {total} variants...\n")

    i = 0
    for macro_name, macro_fn in MACRO_FILTERS.items():
        for max_pos in MAX_POS_GRID:
            for reserve in RESERVE_GRID:
                i += 1
                m = simulate_variant(
                    signals, all_data, spy_ctx, macro_fn,
                    max_pos=max_pos, reserve_pct=reserve,
                )
                results.append({
                    "macro": macro_name,
                    "max_pos": max_pos,
                    "reserve": reserve,
                    **m,
                })
                tag = f"[{i:3d}/{total}]"
                print(f"  {tag} {macro_name:<16s} pos={max_pos} rsv={reserve:.0%}  →  "
                      f"{m['final']:>9,.0f}€  CAGR {m['cagr']:>+6.1f}%  DD {m['dd']:>6.1f}%  "
                      f"WR {m['wr']:>5.1f}%  PF {m['pf']:>5.2f}  "
                      f"Tr {m['trades']:>3d}  Pyr {m['n_pyr']:>2d}  Sig {m['signals_used']:>3d}")

    # ── Step 5: Summary — top 20 by CAGR ──
    df = pd.DataFrame(results)
    df["efficiency"] = df["cagr"] / df["dd"].abs().clip(lower=0.1)

    print(f"\n{'='*130}")
    print("  TOP 20 by CAGR")
    print(f"{'='*130}")
    print(f"  {'Macro':<16s} {'Pos':>3s} {'Rsv':>4s} {'Final':>10s} {'CAGR':>7s} {'DD':>7s} "
          f"{'WR':>6s} {'PF':>6s} {'Tr':>4s} {'Pyr':>3s} {'Eff':>6s}")
    print("  " + "-" * 110)

    top = df.nlargest(20, "cagr")
    for _, r in top.iterrows():
        print(f"  {r['macro']:<16s} {r['max_pos']:>3.0f} {r['reserve']:>4.0%} "
              f"{r['final']:>10,.0f}€ {r['cagr']:>+6.1f}% {r['dd']:>6.1f}% "
              f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['trades']:>4.0f} {r['n_pyr']:>3.0f} "
              f"{r['efficiency']:>5.2f}")

    # ── Top 10 by efficiency (CAGR/DD) ──
    print(f"\n{'='*130}")
    print("  TOP 10 by EFFICIENCY (CAGR / |DD|)")
    print(f"{'='*130}")
    print(f"  {'Macro':<16s} {'Pos':>3s} {'Rsv':>4s} {'Final':>10s} {'CAGR':>7s} {'DD':>7s} "
          f"{'WR':>6s} {'PF':>6s} {'Tr':>4s} {'Pyr':>3s} {'Eff':>6s}")
    print("  " + "-" * 110)

    top_eff = df.nlargest(10, "efficiency")
    for _, r in top_eff.iterrows():
        print(f"  {r['macro']:<16s} {r['max_pos']:>3.0f} {r['reserve']:>4.0%} "
              f"{r['final']:>10,.0f}€ {r['cagr']:>+6.1f}% {r['dd']:>6.1f}% "
              f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['trades']:>4.0f} {r['n_pyr']:>3.0f} "
              f"{r['efficiency']:>5.2f}")

    # ── Baseline comparison ──
    baseline = df[(df["macro"] == "NO_FILTER") & (df["max_pos"] == 3) & (df["reserve"] == 0.20)]
    if not baseline.empty:
        b = baseline.iloc[0]
        print(f"\n  BASELINE (NO_FILTER, 3pos, 20%rsv): {b['final']:,.0f}€ | "
              f"CAGR {b['cagr']:+.1f}% | DD {b['dd']:.1f}% | Eff {b['efficiency']:.2f}")

    best = df.loc[df["efficiency"].idxmax()]
    print(f"  BEST EFF: {best['macro']}, {best['max_pos']:.0f}pos, {best['reserve']:.0%}rsv: "
          f"{best['final']:,.0f}€ | CAGR {best['cagr']:+.1f}% | DD {best['dd']:.1f}% | Eff {best['efficiency']:.2f}")

    best_cagr = df.loc[df["cagr"].idxmax()]
    print(f"  BEST CAGR: {best_cagr['macro']}, {best_cagr['max_pos']:.0f}pos, {best_cagr['reserve']:.0%}rsv: "
          f"{best_cagr['final']:,.0f}€ | CAGR {best_cagr['cagr']:+.1f}% | DD {best_cagr['dd']:.1f}%")

    print(f"\n{'='*130}")


if __name__ == "__main__":
    main()
