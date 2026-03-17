"""
Test: expanded universe (~150 tickers) — same criteria US + EU + ADRs.
One-off test, not part of the strategy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from .config import INITIAL, LOOKBACK, FIXED_YEARS, MAX_POS, RESERVE_PCT, PYR_THRESHOLD, PYR_SIZE, COST_BPS
from .strategy import download_daily, get_macro_context, compute_all_trades, select_daily_signals, simulate

# ── Expanded universe: ~150 tickers ──────────────────────────────────────
EXPANDED = [
    # -- US Index ETFs --
    "SPY", "QQQ", "IWM", "DIA", "MDY", "RSP",
    # -- Bond / Credit ETFs --
    "TLT", "IEF", "HYG", "LQD", "BND", "TIP",
    # -- Commodity ETFs --
    "GLD", "SLV", "USO", "DBA", "PDBC",
    # -- BTC --
    "BTC-USD",
    # -- US Sector ETFs --
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    "XBI", "SMH", "SOXX", "ARKK", "XHB", "XRT", "KRE", "ITB",
    # -- US Mega-cap equities (original) --
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "BRK-B", "JPM", "V", "MA", "AVGO", "LLY", "UNH", "XOM",
    "COST", "WMT", "PG", "HD", "KO", "PEP", "ORCL", "NFLX",
    "AMD", "INTC", "QCOM", "ADBE", "CRM", "CSCO",
    "BAC", "GS", "MS", "WFC",
    "PFE", "ABBV", "JNJ", "MRK",
    "NKE", "DIS", "ABNB", "UBER", "SHOP", "PLTR",
    # -- US additional large-cap --
    "CVX", "LIN", "TMO", "ABT", "DHR", "ISRG", "NOW", "PANW",
    "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS",
    "AXP", "SCHW", "BLK", "CME",
    "DE", "CAT", "HON", "RTX", "GE", "BA",
    "LOW", "TJX", "MCD", "SBUX", "YUM",
    "T", "VZ", "TMUS",
    "COP", "EOG", "SLB", "PSX",
    # -- ADRs (large international companies trading on US exchanges) --
    "TSM",      # TSMC (Taiwan)
    "BABA",     # Alibaba (China)
    "NVO",      # Novo Nordisk (Denmark)
    "TM",       # Toyota (Japan)
    "SONY",     # Sony (Japan)
    "UL",       # Unilever (UK)
    "AZN",      # AstraZeneca (UK)
    "HSBC",     # HSBC (UK)
    "SHEL",     # Shell (UK/NL)
    "BP",       # BP (UK)
    "RIO",      # Rio Tinto (UK/AU)
    "BHP",      # BHP (AU)
    "DEO",      # Diageo (UK)
    "SNY",      # Sanofi (France)
    "SAP",      # SAP (Germany) — already dual-listed
    "ASML",     # ASML (Netherlands) — already dual-listed
    "INFY",     # Infosys (India)
    "WIT",      # Wipro (India)
    "MELI",     # MercadoLibre (LatAm)
    "NU",       # Nubank (Brazil)
    "SE",       # Sea Limited (Singapore)
    "GRAB",     # Grab (Singapore)
    # -- Europe large caps (local exchanges) --
    "MC.PA", "AIR.PA", "OR.PA",     # France: LVMH, Airbus, L'Oréal
    "SIE.DE", "DTE.DE",             # Germany: Siemens, Dt Telekom
    "NESN.SW",                       # Switzerland: Nestlé
    "ALV.DE", "BMW.DE", "MBG.DE",   # Germany: Allianz, BMW, Mercedes
    "SAN.PA", "BNP.PA", "TTE.PA",   # France: Sanofi, BNP, TotalEnergies
    "ABI.BR",                        # Belgium: AB InBev
    "NOVO-B.CO",                     # Denmark: Novo Nordisk (local)
    "AZN.L", "SHEL.L", "ULVR.L", "LSEG.L", "RIO.L",  # UK
]

# Deduplicate
EXPANDED = list(dict.fromkeys(EXPANDED))


def main():
    print("=" * 130)
    print(f"  LONG SWING TEST — Expanded Universe ({len(EXPANDED)} tickers)")
    print(f"  max{MAX_POS}, Rsv {RESERVE_PCT*100:.0f}%, +{PYR_THRESHOLD*100:.0f}% trigger, {PYR_SIZE*100:.0f}% pyramid")
    print(f"  Capital: {INITIAL:,.0f}€ | Lookback: {LOOKBACK}d (~{FIXED_YEARS:.1f}y) | Cost: {COST_BPS}bps")
    print("=" * 130)

    all_data = download_daily(EXPANDED)
    print(f"\n  Simbolos cargados: {len(all_data)} / {len(EXPANDED)}")

    spy_ctx, vix_map = get_macro_context(all_data)
    trades_df = compute_all_trades(all_data, spy_ctx, vix_map)
    signals = select_daily_signals(trades_df)
    print(f"  Senales: {len(signals)}")
    print(f"  Periodo: {signals['entry_ts'].min().date()} -> {signals['exit_ts'].max().date()}")

    log, still_open, m = simulate(signals, all_data)

    print(f"\n{'='*130}")
    print(f"  RESULTADO: {INITIAL:,.0f}€ -> {m['final']:,.0f}€ | CAGR {m['cagr']:+.1f}% | "
          f"DD {m['dd']:.1f}% | WR {m['wr']:.1f}% | PF {m['pf']:.2f} | "
          f"Trades {m['trades']} | Pyramids {m['n_pyr']} | Skip {m['skipped']}")
    print(f"{'='*130}")

    # Trade detail
    for i, t in enumerate(log, 1):
        wl = "W" if t["pnl"] > 0 else ("L" if t["pnl"] < 0 else "-")
        if t["reason"] == "OPEN":
            wl = "*"
        ed = str(t["entry_ts"])[:10]
        xd = str(t["exit_ts"])[:10] if t["reason"] != "OPEN" else "    OPEN   "
        pyr_s = f"{t['pyr_alloc']:>7,.0f}" if t["pyramided"] else "      -"
        print(f"  {i:>3d} {t['symbol']:>8s} {t['signal']:>8s} {ed:>12s} {xd:>12s} "
              f"{str(t['reason'])[:8]:>8s} {t['alloc']:>8,.0f} {pyr_s} "
              f"{t['eff_ret']*100:>+6.1f}% {t['pnl']:>+8,.0f} "
              f"{t['hold_days']:>4.0f}d {wl:>3s}")

    # Pyramid stats
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

    # Count by region
    us_trades = [t for t in log if not any(t["symbol"].endswith(s) for s in [".PA", ".DE", ".SW", ".L", ".BR", ".CO"])]
    eu_trades = [t for t in log if any(t["symbol"].endswith(s) for s in [".PA", ".DE", ".SW", ".L", ".BR", ".CO"])]
    adr_syms = {"TSM", "BABA", "NVO", "TM", "SONY", "UL", "AZN", "HSBC", "SHEL", "BP",
                "RIO", "BHP", "DEO", "SNY", "INFY", "WIT", "MELI", "NU", "SE", "GRAB"}
    adr_trades = [t for t in us_trades if t["symbol"] in adr_syms]
    print(f"\n  Por region: US {len(us_trades)-len(adr_trades)} | ADR {len(adr_trades)} | EU local {len(eu_trades)}")

    print(f"\n{'='*130}")


if __name__ == "__main__":
    main()
