"""
Long Swing — Backtest runner.

Usage: uv run python -m long_swing
       uv run python -m long_swing.backtest
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from .config import (
    UNIVERSE, INITIAL, LOOKBACK, FIXED_YEARS, MAX_POS, RESERVE_PCT,
    PYR_THRESHOLD, PYR_SIZE, COST_BPS, MACRO_FILTER, VIX_MAX, BENCHMARKS,
)
from .strategy import (
    download_daily, get_macro_context, compute_all_trades,
    select_daily_signals, apply_macro_filter, simulate,
)


def compute_benchmark(all_data, sym, initial, fixed_years):
    """Compute buy-and-hold CAGR and MaxDD for a benchmark."""
    df = all_data.get(sym)
    if df is None or len(df) < 100:
        return None
    close = df["Close"].astype(float)
    total_ret = close.iloc[-1] / close.iloc[0]
    years = (close.index[-1] - close.index[0]).days / 365.25
    cagr = (total_ret ** (1 / years) - 1) * 100
    final = initial * total_ret
    peak = close.cummax()
    dd = ((close - peak) / peak).min() * 100
    return {"final": round(final, 0), "cagr": round(cagr, 1), "dd": round(dd, 1),
            "years": round(years, 1), "start": close.index[0].date(), "end": close.index[-1].date()}


def main():
    print("=" * 130)
    print(f"  LONG SWING — max{MAX_POS}, Rsv {RESERVE_PCT*100:.0f}%, "
          f"+{PYR_THRESHOLD*100:.0f}% trigger, {PYR_SIZE*100:.0f}% pyramid | "
          f"Macro: {MACRO_FILTER} (VIX<{VIX_MAX})")
    print(f"  Capital: {INITIAL:,.0f}€ | Lookback: {LOOKBACK}d (~{FIXED_YEARS:.1f}y) | Cost: {COST_BPS}bps")
    print("=" * 130)

    all_data = download_daily(UNIVERSE)
    print(f"\n  Tickers loaded: {len(all_data)} / {len(UNIVERSE)}")

    spy_ctx, vix_map = get_macro_context(all_data)
    trades_df = compute_all_trades(all_data, spy_ctx, vix_map)
    signals = select_daily_signals(trades_df)
    total_signals = len(signals)

    # Apply macro filter
    filtered = apply_macro_filter(signals, spy_ctx)
    filtered_out = total_signals - len(filtered)

    print(f"  Signals: {total_signals} total → {len(filtered)} after macro filter ({filtered_out} filtered)")
    if filtered.empty:
        print("  No signals — aborting.")
        return
    print(f"  Period: {filtered['entry_ts'].min().date()} → {filtered['exit_ts'].max().date()}")

    log, still_open, m = simulate(filtered, all_data)

    # ── Benchmarks ──
    print(f"\n{'='*130}")
    print(f"  {'Strategy':<16s} {'Final':>10s} {'CAGR':>7s} {'MaxDD':>7s} {'WR':>6s} {'PF':>6s} {'Trades':>6s} {'Pyr':>4s}")
    print("  " + "-" * 70)
    print(f"  {'LONG SWING':<16s} {m['final']:>10,.0f}€ {m['cagr']:>+6.1f}% {m['dd']:>6.1f}% "
          f"{m['wr']:>5.1f}% {m['pf']:>5.2f} {m['trades']:>6d} {m['n_pyr']:>4d}")

    for bm_sym in BENCHMARKS:
        bm = compute_benchmark(all_data, bm_sym, INITIAL, FIXED_YEARS)
        if bm:
            print(f"  {bm_sym + ' (B&H)':<16s} {bm['final']:>10,.0f}€ {bm['cagr']:>+6.1f}% {bm['dd']:>6.1f}%"
                  f"     —     —      —    —")

    print(f"{'='*130}")
    print(f"  Skip {m['skipped']} | AvgHold {m['avg_hold_d']:.0f}d")

    # ── Trade detail ──
    print(f"\n  {'#':>3s} {'Sym':>8s} {'Signal':>8s} {'Entry':>12s} {'Exit':>12s} "
          f"{'Reason':>8s} {'Alloc€':>8s} {'Pyr€':>7s} {'Ret%':>7s} "
          f"{'PnL€':>8s} {'PnlPyr€':>7s} {'Hold':>5s} {'':>3s}")
    print("  " + "-" * 120)

    for i, t in enumerate(log, 1):
        wl = "W" if t["pnl"] > 0 else ("L" if t["pnl"] < 0 else "-")
        if t["reason"] == "OPEN":
            wl = "*"
        ed = str(t["entry_ts"])[:10]
        xd = str(t["exit_ts"])[:10] if t["reason"] != "OPEN" else "    OPEN   "
        pyr_s = f"{t['pyr_alloc']:>7,.0f}" if t["pyramided"] else "      -"
        pnlp_s = f"{t['pnl_pyr']:>+7,.0f}" if t["pyramided"] else "      -"
        print(f"  {i:>3d} {t['symbol']:>8s} {t['signal']:>8s} {ed:>12s} {xd:>12s} "
              f"{str(t['reason'])[:8]:>8s} {t['alloc']:>8,.0f} {pyr_s} "
              f"{t['eff_ret']*100:>+6.1f}% {t['pnl']:>+8,.0f} {pnlp_s} "
              f"{t['hold_days']:>4.0f}d {wl:>3s}")

    # ── Pyramid stats ──
    pyr_trades = [t for t in log if t["pyramided"]]
    nopyr_trades = [t for t in log if not t["pyramided"] and t["reason"] != "OPEN"]
    if pyr_trades:
        pw = sum(1 for t in pyr_trades if t["pnl"] > 0)
        print(f"\n  Pyramided:     {len(pyr_trades)} trades, WR {pw/len(pyr_trades)*100:.0f}%, "
              f"avg PnL {np.mean([t['pnl'] for t in pyr_trades]):+,.0f}€, "
              f"total PnL {sum(t['pnl'] for t in pyr_trades):+,.0f}€")
    if nopyr_trades:
        nw = sum(1 for t in nopyr_trades if t["pnl"] > 0)
        print(f"  No pyramid:    {len(nopyr_trades)} trades, WR {nw/len(nopyr_trades)*100:.0f}%, "
              f"avg PnL {np.mean([t['pnl'] for t in nopyr_trades]):+,.0f}€, "
              f"total PnL {sum(t['pnl'] for t in nopyr_trades):+,.0f}€")

    # ── Open positions ──
    last_date = filtered["exit_ts"].max().date()
    open_trades = [t for t in log if str(t["exit_ts"])[:10] == str(last_date)]
    if open_trades:
        print(f"\n{'='*130}")
        print(f"  POSICIONES ABIERTAS ({len(open_trades)}) — exit_ts = {last_date} (fin de datos)")
        print(f"{'='*130}")
        for t in open_trades:
            sym = t["symbol"]
            df = all_data.get(sym)
            if df is not None and len(df) > 50:
                ema21 = float(df["Close"].ewm(span=21).mean().iloc[-1])
                ema50 = float(df["Close"].ewm(span=50).mean().iloc[-1])
                last_close = float(df["Close"].iloc[-1])
                ema_status = "EMA21>50" if ema21 > ema50 else "EMA21<50 !!EXIT!!"
                pyr_info = f" | Pyr {t['pyr_alloc']:,.0f}€" if t["pyramided"] else ""
                print(f"  {sym:>8s} | Entry {str(t['entry_ts'])[:10]} | Alloc {t['alloc']:,.0f}€{pyr_info}"
                      f" | Close {last_close:.2f} | EMA21 {ema21:.2f} vs EMA50 {ema50:.2f} | {ema_status}"
                      f" | PnL {t['pnl']:+,.0f}€ ({t['eff_ret']*100:+.1f}%)")

    print(f"\n{'='*130}")
    print("  FIN")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
