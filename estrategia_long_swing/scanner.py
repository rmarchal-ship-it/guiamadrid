"""
Long Swing — Daily scanner.

Shows: today's signals (macro-filtered), open position status (EMA check), pyramid triggers.
Usage: uv run python -m long_swing.scanner
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pandas_ta as ta
from .config import (
    UNIVERSE, LOOKBACK, EMA_EXIT_FAST, EMA_EXIT_SLOW, PYR_THRESHOLD,
    TUNED_RSI, BEST_EMA, MACRO_FILTER, VIX_MAX, MAX_POS,
)
from .strategy import (
    download_daily, get_macro_context, compute_all_trades,
    select_daily_signals, apply_macro_filter,
)


def main():
    print("=" * 100)
    print(f"  LONG SWING — Scanner diario | Macro: {MACRO_FILTER} | Max {MAX_POS} pos")
    print("=" * 100)

    all_data = download_daily(UNIVERSE, lookback_days=300)
    print(f"\n  Simbolos: {len(all_data)}")

    spy_ctx, vix_map = get_macro_context(all_data)

    # ── Today's signals ──
    trades_df = compute_all_trades(all_data, spy_ctx, vix_map)
    if trades_df.empty:
        print("  No hay trades generados.")
        return

    signals = select_daily_signals(trades_df)
    if signals.empty:
        print("  No hay senales.")
        return

    # Apply macro filter
    filtered = apply_macro_filter(signals, spy_ctx)

    # Signals from last 7 days
    today = pd.Timestamp.now().normalize()
    last_7d_all = signals[signals["entry_ts"] >= today - pd.Timedelta(days=7)]

    if not last_7d_all.empty:
        print(f"\n  SENALES RECIENTES (ultimos 7 dias)")
        print(f"  {'Sym':>8s} {'Signal':>8s} {'Entry':>12s} {'ADX':>5s} {'VIX':>5s} {'SPY%':>5s} {'Macro':>8s}")
        print("  " + "-" * 60)
        for _, s in last_7d_all.iterrows():
            in_filtered = any(
                (filtered["entry_ts"] == s["entry_ts"]) & (filtered["symbol"] == s["symbol"])
            ) if not filtered.empty else False
            macro_tag = "  OK" if in_filtered else " BLOCK"
            print(f"  {s['symbol']:>8s} {s['signal']:>8s} {str(s['entry_ts'])[:10]:>12s} "
                  f"{s['adx']:>5.1f} {s['vix']:>5.1f} {s['spy_pct_from_high']:>4.1f}%{macro_tag:>8s}")
    else:
        print("\n  No hay senales en los ultimos 7 dias.")

    # ── EMA status for open positions ──
    print(f"\n  EMA STATUS — {EMA_EXIT_FAST}/{EMA_EXIT_SLOW}")
    print(f"  {'Sym':>8s} {'Close':>8s} {'EMA21':>8s} {'EMA50':>8s} {'Status':>12s} {'%fromEntry':>10s}")
    print("  " + "-" * 60)

    last_7d_filtered = filtered[filtered["entry_ts"] >= today - pd.Timedelta(days=7)]
    active_syms = set(last_7d_filtered["symbol"].tolist()) if not last_7d_filtered.empty else set()
    recent_entries = filtered[filtered["entry_ts"] >= today - pd.Timedelta(days=90)]
    open_syms = set()
    for _, s in recent_entries.iterrows():
        if str(s["exit_ts"])[:10] >= str(today.date()):
            open_syms.add(s["symbol"])

    for sym in sorted(open_syms | active_syms):
        df = all_data.get(sym)
        if df is None or len(df) < 60:
            continue
        close = df["Close"].astype(float)
        ema21 = float(ta.ema(close, length=EMA_EXIT_FAST).iloc[-1])
        ema50 = float(ta.ema(close, length=EMA_EXIT_SLOW).iloc[-1])
        last_close = float(close.iloc[-1])
        status = "EMA21>50" if ema21 > ema50 else "!!EXIT!!"

        sym_signals = filtered[filtered["symbol"] == sym]
        if not sym_signals.empty:
            entry_px = sym_signals.iloc[-1]["entry_px"]
            pct = (last_close / entry_px - 1.0) * 100
            pyr_status = f" [PYR +{PYR_THRESHOLD*100:.0f}%]" if pct >= PYR_THRESHOLD * 100 else ""
            print(f"  {sym:>8s} {last_close:>8.2f} {ema21:>8.2f} {ema50:>8.2f} {status:>12s} "
                  f"{pct:>+9.1f}%{pyr_status}")

    print(f"\n{'='*100}")
    print("  FIN")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
