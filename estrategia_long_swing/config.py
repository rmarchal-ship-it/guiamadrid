"""
Long Swing — Strategy configuration.

All tunable parameters in one place. Import CONFIG dict for backtest/scanner.

Best config found via research_macro.py grid search (15 Mar 2026):
  SMA200+VIX30 macro filter, 4 positions, 20% reserve, +15% pyramid.
  240m: 1,003,889€ / CAGR +24.8% / DD -17.6% / PF 4.38 / 200 trades (cached 15-mar-2026)
  120m:        TBD (pending rerun with cache)
   60m:        TBD (pending rerun with cache)
"""
from trading.signals import RsiMacdConfig, EmaCrossConfig

# ── Universe ──────────────────────────────────────────────────────────────
# 76 tickers: US (ETFs + mega-caps) + EU large caps + BTC-USD for signals.
# BTC-USD for backtest signals; IBIT for live paper trading.
UNIVERSE = [
    # -- US Index ETFs --
    "SPY", "QQQ", "IWM", "DIA",
    # -- Bond / Credit ETFs --
    "TLT", "IEF", "HYG", "LQD",
    # -- Commodity ETFs --
    "GLD", "SLV", "USO",
    # -- BTC spot (signal source; trade via IBIT in paper trading) --
    "BTC-USD",
    # -- US Sector ETFs --
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    "XBI", "SMH", "SOXX", "ARKK",
    # -- US Mega-cap equities --
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "BRK-B", "JPM", "V", "MA", "AVGO", "LLY", "UNH", "XOM",
    "COST", "WMT", "PG", "HD", "KO", "PEP", "ORCL", "NFLX",
    "AMD", "INTC", "QCOM", "ADBE", "CRM", "CSCO",
    "BAC", "GS", "MS", "WFC",
    "PFE", "ABBV", "JNJ", "MRK",
    "NKE", "DIS", "ABNB", "UBER", "SHOP", "PLTR",
    # -- Europe large caps --
    "ASML", "SAP", "MC.PA", "AIR.PA", "OR.PA", "SIE.DE", "DTE.DE", "NESN.SW",
]

# ── Signal configs ────────────────────────────────────────────────────────
TUNED_RSI = RsiMacdConfig(
    rsi_reversal_min=46, adx_min=18, bb_bw_min=0.012, vol_z_min=0.25,
)
BEST_EMA = EmaCrossConfig(
    fast=9, slow=32, rsi_min=49.0, adx_min=17.0, vol_z_min=0.25, bb_bw_min=0.008,
)

# ── Macro filter ─────────────────────────────────────────────────────────
MACRO_FILTER = "SMA200+VIX30"  # skip entries when SPY < SMA200 or VIX >= 30
VIX_MAX = 30                    # max VIX level for new entries

# ── Portfolio ─────────────────────────────────────────────────────────────
MAX_POS = 4                     # max simultaneous positions (was 3)
RESERVE_PCT = 0.20              # 20% cash reserved for pyramiding
COST_BPS = 4                    # round-trip cost in basis points

# ── Pyramiding ────────────────────────────────────────────────────────────
PYR_THRESHOLD = 0.15            # +15% from entry triggers pyramid
PYR_SIZE = 1.00                 # add 100% of original alloc

# ── Exit ──────────────────────────────────────────────────────────────────
EMA_EXIT_FAST = 21
EMA_EXIT_SLOW = 50
GRACE_BARS = 60                 # force exit if EMA21 never goes above EMA50

# ── Backtest ──────────────────────────────────────────────────────────────
LOOKBACK = 7305                 # ~20 years of daily bars (240 months)
INITIAL = 12_000.0              # starting capital in EUR
FIXED_YEARS = LOOKBACK / 365.25

# ── Benchmarks ────────────────────────────────────────────────────────────
BENCHMARKS = ["SPY", "QQQ"]

# ── Convenience dict ──────────────────────────────────────────────────────
CONFIG = {
    "tuned_rsi": TUNED_RSI,
    "best_ema": BEST_EMA,
    "macro_filter": MACRO_FILTER,
    "vix_max": VIX_MAX,
    "max_pos": MAX_POS,
    "reserve_pct": RESERVE_PCT,
    "cost_bps": COST_BPS,
    "pyr_threshold": PYR_THRESHOLD,
    "pyr_size": PYR_SIZE,
    "ema_exit_fast": EMA_EXIT_FAST,
    "ema_exit_slow": EMA_EXIT_SLOW,
    "grace_bars": GRACE_BARS,
    "lookback": LOOKBACK,
    "initial": INITIAL,
    "fixed_years": FIXED_YEARS,
}
