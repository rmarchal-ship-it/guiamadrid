"""
Long Swing — Core strategy engine.

Provides:
  download_daily()       — fetch OHLCV for universe
  get_macro_context()    — SPY SMA50/SMA200, VIX levels per day
  compute_all_trades()   — generate entry/exit signals with EMA cross exit
  select_daily_signals() — pick 1 signal/day (RSI priority, ADX ranked)
  apply_macro_filter()   — filter signals by macro regime
  compute_pyramid_data() — detect +15% pyramid triggers
  simulate()             — event-based portfolio sim with pyramiding
"""
import hashlib
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

from trading.data import DataRequest, download_ohlcv
from trading.signals import rsi_reversal_macd_entry, ema_cross_entry

from .config import (
    TUNED_RSI, BEST_EMA, LOOKBACK, INITIAL, COST_BPS, FIXED_YEARS,
    MAX_POS, RESERVE_PCT, PYR_THRESHOLD, PYR_SIZE,
    EMA_EXIT_FAST, EMA_EXIT_SLOW, GRACE_BARS,
    MACRO_FILTER, VIX_MAX,
)

# ── Data cache ────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"


def _cache_key(sym, lookback_days):
    """Deterministic cache file path for a symbol + lookback."""
    return CACHE_DIR / f"{sym.replace('.', '_')}_{lookback_days}d.parquet"


def download_daily(symbols, lookback_days=None, use_cache=True, refresh=False):
    """
    Download daily OHLCV for all symbols.

    Cache behaviour:
      use_cache=True, refresh=False  → load from cache if exists, else download & save
      use_cache=True, refresh=True   → always download & overwrite cache
      use_cache=False                → download only, no disk I/O
    """
    if lookback_days is None:
        lookback_days = LOOKBACK + 60

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)

    all_data = {}
    cached = downloaded = skipped = 0

    for i, sym in enumerate(symbols, 1):
        cache_path = _cache_key(sym, lookback_days)
        print(f"  [{i:2d}/{len(symbols)}] {sym:<8s} ... ", end="", flush=True)

        # Try cache first
        if use_cache and not refresh and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if len(df) >= 100:
                    all_data[sym] = df
                    cached += 1
                    print(f"{len(df)} bars [cache]")
                    continue
            except Exception:
                pass  # corrupted cache, re-download

        # Download with retries
        df = None
        for attempt in range(3):
            df = download_ohlcv(DataRequest(sym, interval="1d", lookback_days=lookback_days))
            if df is not None and not df.empty and len(df) >= 100:
                break
            if attempt < 2:
                import time; time.sleep(1)
        if df is None or df.empty or len(df) < 100:
            print("skip (3 retries failed)")
            skipped += 1
            continue

        all_data[sym] = df
        downloaded += 1
        print(f"{len(df)} bars")

        # Save to cache
        if use_cache:
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                print(f"    [cache write failed: {e}]")

    print(f"  Data: {cached} cached + {downloaded} downloaded + {skipped} skipped = {len(all_data)} tickers")
    return all_data


def get_macro_context(all_data):
    """Compute SPY SMA50/SMA200 regime and VIX level for each trading day."""
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

    vix_df = download_ohlcv(DataRequest("^VIX", interval="1d", lookback_days=LOOKBACK + 60))
    vix_map = {}
    if vix_df is not None and not vix_df.empty:
        for dt in vix_df.index:
            vix_map[dt.date()] = float(vix_df.loc[dt, "Close"])

    return spy_context, vix_map


# ── Signals & Trades ──────────────────────────────────────────────────────

def compute_all_trades(all_data, spy_context, vix_map):
    """
    For each symbol, compute entry signals on daily bars.
    Entry = next day's OPEN. Exit = EMA21 < EMA50 close (with grace period).
    """
    cost_mult = 1.0 - (COST_BPS / 10_000)
    all_trades = []

    for sym, df in all_data.items():
        if sym in ("^VIX",):
            continue

        close = df["Close"].astype(float)
        ema21 = ta.ema(close, length=EMA_EXIT_FAST)
        ema50 = ta.ema(close, length=EMA_EXIT_SLOW)

        adx_df = df.ta.adx(length=14)
        adx_s = adx_df["ADX_14"].astype(float) if adx_df is not None and "ADX_14" in adx_df.columns else pd.Series(np.nan, index=df.index)

        for sig_name, sig_fn in [
            ("rsi_macd", rsi_reversal_macd_entry(TUNED_RSI)),
            ("ema_cross", ema_cross_entry(BEST_EMA)),
        ]:
            signal = sig_fn(df)
            entry_mask = signal == 1

            for loc_idx in np.where(entry_mask.values)[0]:
                entry_day_idx = loc_idx + 1
                if entry_day_idx >= len(df):
                    continue

                signal_ts = df.index[loc_idx]
                entry_ts = df.index[entry_day_idx]
                entry_px = float(df.iloc[entry_day_idx]["Open"])

                adx_val = 0.0
                if signal_ts in adx_s.index:
                    v = adx_s.loc[signal_ts]
                    adx_val = float(v) if np.isfinite(v) else 0.0

                ema21_at_entry = ema21.iloc[loc_idx] if ema21 is not None and loc_idx < len(ema21) else np.nan
                ema50_at_entry = ema50.iloc[loc_idx] if ema50 is not None and loc_idx < len(ema50) else np.nan
                ema21_above_50 = 1 if (np.isfinite(ema21_at_entry) and np.isfinite(ema50_at_entry) and ema21_at_entry >= ema50_at_entry) else 0

                # Find exit: walk forward from entry day
                exit_idx = None
                exit_reason = "end_of_data"
                ema_was_above = False

                for j in range(entry_day_idx, len(df)):
                    bars_since_entry = j - entry_day_idx
                    if ema21 is not None and ema50 is not None:
                        e21 = ema21.iloc[j] if j < len(ema21) else np.nan
                        e50 = ema50.iloc[j] if j < len(ema50) else np.nan
                        if np.isfinite(e21) and np.isfinite(e50):
                            if e21 >= e50:
                                ema_was_above = True
                            elif ema_was_above and e21 < e50:
                                exit_idx = j
                                exit_reason = "ema_cross"
                                break
                            elif not ema_was_above and bars_since_entry >= GRACE_BARS:
                                exit_idx = j
                                exit_reason = "grace_exit"
                                break

                if exit_idx is None:
                    exit_idx = len(df) - 1

                exit_ts = df.index[exit_idx]
                exit_px = float(df.iloc[exit_idx]["Close"])
                ret = (exit_px / entry_px - 1.0) * cost_mult
                hold_days = (exit_ts - entry_ts).days

                sig_date = signal_ts.date() if hasattr(signal_ts, 'date') else signal_ts
                spy_ctx = spy_context.get(sig_date, {})
                vix_val = vix_map.get(sig_date, 0)

                min_low = entry_px
                if entry_day_idx < len(df):
                    lows = df["Low"].iloc[entry_day_idx:exit_idx+1].astype(float)
                    if len(lows) > 0:
                        min_low = lows.min()
                max_dd_from_entry = (min_low / entry_px - 1.0) if entry_px > 0 else 0

                all_trades.append({
                    "entry_ts": entry_ts, "exit_ts": exit_ts,
                    "signal_ts": signal_ts,
                    "entry_px": entry_px, "exit_px": exit_px,
                    "ret": ret, "hold_days": hold_days,
                    "reason": exit_reason,
                    "symbol": sym, "signal": sig_name,
                    "adx": adx_val,
                    "ema21_above_50": ema21_above_50,
                    "spy_pct_from_high": spy_ctx.get("spy_pct_from_high", 0),
                    "spy_above_sma200": spy_ctx.get("spy_above_sma200", 1),
                    "vix": vix_val,
                    "max_dd_from_entry": max_dd_from_entry,
                    "entry_date": entry_ts.date() if hasattr(entry_ts, 'date') else entry_ts,
                })

    return pd.DataFrame(all_trades)


def select_daily_signals(trades_df):
    """1 trade/day max: rsi_macd priority, then best ADX."""
    selected = []
    for date, day in trades_df.groupby("entry_date"):
        rsi = day[day["signal"] == "rsi_macd"]
        pool = rsi if not rsi.empty else day[day["signal"] == "ema_cross"]
        if pool.empty:
            continue
        selected.append(pool.sort_values("adx", ascending=False).head(1))
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected).sort_values("entry_ts").reset_index(drop=True)


def apply_macro_filter(signals_df, spy_context, macro_filter=None):
    """Filter signals by macro regime. Returns filtered DataFrame."""
    if macro_filter is None:
        macro_filter = MACRO_FILTER

    if macro_filter == "NO_FILTER":
        return signals_df

    def passes(row):
        ctx = spy_context.get(row["entry_date"], {})
        if "SMA200" in macro_filter:
            if ctx.get("spy_above_sma200", 1) != 1:
                return False
        elif "SMA50" in macro_filter:
            if ctx.get("spy_above_sma50", 1) != 1:
                return False
        if "VIX" in macro_filter:
            if row.get("vix", 0) >= VIX_MAX:
                return False
        return True

    mask = signals_df.apply(passes, axis=1)
    return signals_df[mask].reset_index(drop=True)


# ── Pyramiding ────────────────────────────────────────────────────────────

def compute_pyramid_data(signals_df, all_data, threshold=PYR_THRESHOLD):
    """
    For each trade, find when close first reaches entry_px * (1+threshold).
    Pyramid entry = next day's OPEN after trigger.
    """
    cost_mult = 1.0 - COST_BPS / 10_000
    records = []

    for i, row in signals_df.iterrows():
        sym = row["symbol"]
        df = all_data.get(sym)
        if df is None:
            records.append({"pyr_ts": pd.NaT, "pyr_px": 0, "pyr_ret": 0, "pyr_hold": 0})
            continue

        entry_px = row["entry_px"]
        target_px = entry_px * (1.0 + threshold)
        exit_px = row["exit_px"]
        exit_ts = row["exit_ts"]

        try:
            entry_loc = df.index.get_loc(row["entry_ts"])
        except KeyError:
            entry_loc = min(df.index.searchsorted(row["entry_ts"]), len(df) - 1)
        try:
            exit_loc = df.index.get_loc(exit_ts)
        except KeyError:
            exit_loc = min(df.index.searchsorted(exit_ts), len(df) - 1)

        pyr_ts = pd.NaT
        pyr_px = 0.0
        for j in range(entry_loc, exit_loc + 1):
            if float(df.iloc[j]["Close"]) >= target_px:
                if j + 1 < len(df) and df.index[j + 1] <= exit_ts:
                    pyr_ts = df.index[j + 1]
                    pyr_px = float(df.iloc[j + 1]["Open"])
                break

        if pd.notna(pyr_ts) and pyr_px > 0:
            pyr_ret = (exit_px / pyr_px - 1.0) * cost_mult
            pyr_hold = (exit_ts - pyr_ts).days
        else:
            pyr_ts = pd.NaT
            pyr_ret = 0.0
            pyr_hold = 0

        records.append({
            "pyr_ts": pyr_ts, "pyr_px": pyr_px,
            "pyr_ret": pyr_ret, "pyr_hold": pyr_hold,
        })

    return pd.DataFrame(records, index=signals_df.index)


# ── Portfolio Simulation ──────────────────────────────────────────────────

def simulate(signals_df, all_data, initial=None, fixed_years=None):
    """
    Event-based portfolio sim: no equity cap, 20% reserve, pyramiding +15%/100%.
    Returns (log, still_open_list, metrics_dict).
    Optional overrides for initial capital and fixed_years (for walk-forward).
    """
    if initial is None:
        initial = INITIAL
    if fixed_years is None:
        fixed_years = FIXED_YEARS
    pyr_df = compute_pyramid_data(signals_df, all_data)
    enriched = pd.concat([signals_df, pyr_df], axis=1)

    cost_mult = 1.0 - COST_BPS / 10_000

    events = []
    for idx, row in enriched.iterrows():
        events.append(("entry", row["entry_ts"], idx))
        events.append(("exit", row["exit_ts"], idx))
        if pd.notna(row["pyr_ts"]):
            events.append(("pyramid", row["pyr_ts"], idx))

    events.sort(key=lambda e: (e[1], {"exit": 0, "pyramid": 1, "entry": 2}[e[0]]))

    cash = initial
    open_pos = {}
    log = []
    skipped = 0
    gp = gl = 0.0
    wins = 0
    eqs = []

    for etype, ts, idx in events:
        row = enriched.loc[idx]

        if etype == "exit":
            if idx not in open_pos:
                continue
            pos = open_pos.pop(idx)
            pnl_orig = pos["alloc"] * row["ret"]
            pnl_pyr = pos["pyr_alloc"] * row["pyr_ret"] if pos["pyr_alloc"] > 0 else 0
            pnl = pnl_orig + pnl_pyr
            total_invested = pos["alloc"] + pos["pyr_alloc"]
            cash += total_invested + pnl

            if pnl > 0:
                gp += pnl; wins += 1
            elif pnl < 0:
                gl += abs(pnl)

            ov = sum(p["alloc"] + p["pyr_alloc"] for p in open_pos.values())
            eqs.append(cash + ov)
            eff_ret = pnl / total_invested if total_invested > 0 else 0

            log.append({
                "entry_ts": row["entry_ts"], "exit_ts": row["exit_ts"],
                "symbol": row["symbol"], "signal": row["signal"],
                "alloc": pos["alloc"], "pyr_alloc": pos["pyr_alloc"],
                "ret": row["ret"], "pyr_ret": row["pyr_ret"],
                "pnl_orig": pnl_orig, "pnl_pyr": pnl_pyr, "pnl": pnl,
                "eff_ret": eff_ret,
                "hold_days": row["hold_days"], "reason": row["reason"],
                "cash_after": cash,
                "pyramided": pos["pyr_alloc"] > 0,
                "vix": row["vix"], "spy_pct_from_high": row["spy_pct_from_high"],
            })

        elif etype == "pyramid":
            if idx not in open_pos:
                continue
            pos = open_pos[idx]
            if pos["pyr_alloc"] > 0:
                continue
            pyr_alloc = min(pos["alloc"] * PYR_SIZE, cash * 0.95)
            if pyr_alloc <= 100:
                continue
            cash -= pyr_alloc
            pos["pyr_alloc"] = pyr_alloc

        elif etype == "entry":
            if len(open_pos) >= MAX_POS:
                skipped += 1
                continue
            free = MAX_POS - len(open_pos)
            deployable = cash * (1.0 - RESERVE_PCT)
            alloc = deployable / free if free > 0 else 0
            if alloc <= 0:
                skipped += 1
                continue
            cash -= alloc
            open_pos[idx] = {"alloc": alloc, "pyr_alloc": 0.0}

    # Force-close remaining
    still_open = list(open_pos.items())
    for idx, pos in still_open:
        row = enriched.loc[idx]
        pnl_orig = pos["alloc"] * row["ret"]
        pnl_pyr = pos["pyr_alloc"] * row["pyr_ret"] if pos["pyr_alloc"] > 0 else 0
        pnl = pnl_orig + pnl_pyr
        total_invested = pos["alloc"] + pos["pyr_alloc"]
        cash += total_invested + pnl
        if pnl > 0:
            gp += pnl; wins += 1
        elif pnl < 0:
            gl += abs(pnl)
        eff_ret = pnl / total_invested if total_invested > 0 else 0
        log.append({
            "entry_ts": row["entry_ts"], "exit_ts": row["exit_ts"],
            "symbol": row["symbol"], "signal": row["signal"],
            "alloc": pos["alloc"], "pyr_alloc": pos["pyr_alloc"],
            "ret": row["ret"], "pyr_ret": row["pyr_ret"],
            "pnl_orig": pnl_orig, "pnl_pyr": pnl_pyr, "pnl": pnl,
            "eff_ret": eff_ret,
            "hold_days": row["hold_days"], "reason": "OPEN",
            "cash_after": cash,
            "pyramided": pos["pyr_alloc"] > 0,
            "vix": row.get("vix", 0), "spy_pct_from_high": row.get("spy_pct_from_high", 0),
        })
        eqs.append(cash)
    open_pos.clear()

    total = len(log); final = cash
    log.sort(key=lambda t: t["exit_ts"])
    if eqs:
        ea = np.array(eqs); pk = np.maximum.accumulate(ea)
        dd = ((ea - pk) / pk).min() * 100
    else:
        dd = 0
    cagr = ((final / initial) ** (1 / fixed_years) - 1) * 100 if final > 0 and fixed_years > 0 else -100
    pf = gp / gl if gl > 0 else (99 if gp > 0 else 0)
    avg_hold = np.mean([t["hold_days"] for t in log]) if log else 0

    return log, still_open, {
        "final": round(final, 0), "cagr": round(cagr, 1), "dd": round(dd, 1),
        "wr": round(wins / total * 100, 1) if total else 0,
        "trades": total, "pf": round(min(pf, 99), 2),
        "skipped": skipped, "avg_hold_d": round(avg_hold, 1),
        "n_pyr": sum(1 for t in log if t["pyramided"]),
    }
