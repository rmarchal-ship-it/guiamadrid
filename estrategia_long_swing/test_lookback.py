"""
Long Swing — Extended lookback test (120m / 240m).

Runs the full pipeline for LOOKBACK=3652 and LOOKBACK=7305 without modifying config.py.
Monkey-patches strategy module globals that depend on LOOKBACK / FIXED_YEARS.

Usage: uv run python -m long_swing.test_lookback
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from . import strategy as strat
from .config import (
    UNIVERSE, INITIAL, MAX_POS, RESERVE_PCT, PYR_THRESHOLD, PYR_SIZE, COST_BPS,
)


def run_one(lookback_days: int):
    fixed_years = lookback_days / 365.25

    # ── Monkey-patch strategy module globals ──
    strat.LOOKBACK = lookback_days
    strat.FIXED_YEARS = fixed_years

    label = f"~{lookback_days // 365}y ({lookback_days}d)"
    print("=" * 130)
    print(f"  LONG SWING — LOOKBACK {label}")
    print(f"  max{MAX_POS}, Rsv {RESERVE_PCT*100:.0f}%, "
          f"+{PYR_THRESHOLD*100:.0f}% trigger, {PYR_SIZE*100:.0f}% pyramid")
    print(f"  Capital: {INITIAL:,.0f}€ | FIXED_YEARS: {fixed_years:.2f} | Cost: {COST_BPS}bps")
    print("=" * 130)

    # Download with extra margin for indicator warmup
    all_data = strat.download_daily(UNIVERSE, lookback_days=lookback_days + 60)
    print(f"\n  Tickers cargados: {len(all_data)} / {len(UNIVERSE)}")
    missing = [s for s in UNIVERSE if s not in all_data]
    if missing:
        print(f"  Missing: {', '.join(missing)}")

    spy_ctx, vix_map = strat.get_macro_context(all_data)
    trades_df = strat.compute_all_trades(all_data, spy_ctx, vix_map)
    signals = strat.select_daily_signals(trades_df)

    if signals.empty:
        print("  No signals — skipping.\n")
        return

    print(f"  Senales: {len(signals)}")
    print(f"  Periodo: {signals['entry_ts'].min().date()} -> {signals['exit_ts'].max().date()}")

    log, still_open, m = strat.simulate(signals, all_data)

    # ── Summary ──
    print(f"\n{'='*130}")
    print(f"  RESULTADO ({label}): {INITIAL:,.0f}€ -> {m['final']:,.0f}€ | "
          f"CAGR {m['cagr']:+.1f}% | DD {m['dd']:.1f}% | WR {m['wr']:.1f}% | "
          f"PF {m['pf']:.2f} | Trades {m['trades']} | Pyramids {m['n_pyr']} | "
          f"Skip {m['skipped']} | AvgHold {m['avg_hold_d']:.0f}d")
    print(f"{'='*130}")

    # ── Pyramid stats ──
    pyr_trades = [t for t in log if t["pyramided"]]
    nopyr_trades = [t for t in log if not t["pyramided"] and t["reason"] != "OPEN"]
    if pyr_trades:
        pw = sum(1 for t in pyr_trades if t["pnl"] > 0)
        print(f"  Pyramided:  {len(pyr_trades)} trades, WR {pw/len(pyr_trades)*100:.0f}%, "
              f"avg PnL {np.mean([t['pnl'] for t in pyr_trades]):+,.0f}€, "
              f"total PnL {sum(t['pnl'] for t in pyr_trades):+,.0f}€")
    if nopyr_trades:
        nw = sum(1 for t in nopyr_trades if t["pnl"] > 0)
        print(f"  No pyramid: {len(nopyr_trades)} trades, WR {nw/len(nopyr_trades)*100:.0f}%, "
              f"avg PnL {np.mean([t['pnl'] for t in nopyr_trades]):+,.0f}€, "
              f"total PnL {sum(t['pnl'] for t in nopyr_trades):+,.0f}€")

    print()
    return m


def main():
    results = {}
    for lb in [3652, 7305]:
        m = run_one(lb)
        if m:
            results[lb] = m

    # ── Comparison table ──
    if len(results) >= 2:
        print("\n" + "=" * 90)
        print("  COMPARISON TABLE")
        print("=" * 90)
        print(f"  {'Lookback':>12s} {'Final€':>10s} {'CAGR%':>8s} {'DD%':>8s} "
              f"{'WR%':>6s} {'PF':>6s} {'Trades':>7s} {'Pyr':>5s} {'Skip':>6s} {'AvgH':>6s}")
        print("  " + "-" * 85)
        for lb, m in sorted(results.items()):
            yrs = lb / 365.25
            print(f"  {f'~{lb//365}y ({lb}d)':>12s} {m['final']:>10,.0f} {m['cagr']:>+7.1f}% "
                  f"{m['dd']:>7.1f}% {m['wr']:>5.1f}% {m['pf']:>6.2f} "
                  f"{m['trades']:>7d} {m['n_pyr']:>5d} {m['skipped']:>6d} {m['avg_hold_d']:>5.0f}d")
        print("=" * 90)


if __name__ == "__main__":
    main()
