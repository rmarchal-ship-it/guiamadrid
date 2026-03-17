#!/usr/bin/env python3
"""
Grid test: vary max_hold_bars in backtest_definitivo and compare results.
Tests hold periods of 8, 10, 12, 15, 20 days over 18-month backtest.
"""

import sys
import importlib

# Add the Code directory so we can import backtest_definitivo
CODE_DIR = "/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code"
sys.path.insert(0, CODE_DIR)

import backtest_definitivo as bt

HOLD_VALUES = [8, 10, 12, 15, 20]
MONTHS = 18

results = []

for hold in HOLD_VALUES:
    print(f"\n{'#'*70}")
    print(f"### RUNNING: max_hold_bars = {hold}")
    print(f"{'#'*70}")

    # Modify CONFIG in-place
    bt.CONFIG['max_hold_bars'] = hold

    res = bt.run_backtest(months=MONTHS, verbose=False)

    if 'error' in res:
        print(f"  ERROR for hold={hold}: {res['error']}")
        continue

    # Count exit reasons from trade objects
    trades_list = res.get('trades', [])
    time_exit_count = sum(1 for t in trades_list if getattr(t, 'exit_reason', '') == 'time_exit')
    trailing_stop_count = sum(1 for t in trades_list if getattr(t, 'exit_reason', '') == 'trailing_stop')

    results.append({
        'hold_days': hold,
        'total_trades': res['total_trades'],
        'win_rate': res['win_rate'],
        'total_return_pct': res['total_return_pct'],
        'annualized': res['annualized_return_pct'],
        'profit_factor': res['profit_factor'],
        'max_drawdown': res['max_drawdown'],
        'time_exit_count': time_exit_count,
        'trailing_stop_count': trailing_stop_count,
    })

# ---- Print comparison table ----
print(f"\n\n{'='*110}")
print(f"  GRID TEST RESULTS: max_hold_bars  |  {MONTHS}-month backtest")
print(f"{'='*110}")

header = (
    f"{'Hold':>5}  "
    f"{'Trades':>7}  "
    f"{'WinRate%':>8}  "
    f"{'Return%':>8}  "
    f"{'Annual%':>8}  "
    f"{'PF':>6}  "
    f"{'MaxDD%':>7}  "
    f"{'TimeEx':>7}  "
    f"{'TrailSt':>7}"
)
print(header)
print("-" * 110)

for r in results:
    row = (
        f"{r['hold_days']:>5}  "
        f"{r['total_trades']:>7}  "
        f"{r['win_rate']:>8.1f}  "
        f"{r['total_return_pct']:>+8.1f}  "
        f"{r['annualized']:>+8.1f}  "
        f"{r['profit_factor']:>6.2f}  "
        f"{r['max_drawdown']:>7.1f}  "
        f"{r['time_exit_count']:>7}  "
        f"{r['trailing_stop_count']:>7}"
    )
    print(row)

print("-" * 110)

# Highlight best
if results:
    best_return = max(results, key=lambda x: x['total_return_pct'])
    best_pf = max(results, key=lambda x: x['profit_factor'])
    best_dd = min(results, key=lambda x: x['max_drawdown'])
    print(f"\n  Best Total Return:   hold={best_return['hold_days']}d  ({best_return['total_return_pct']:+.1f}%)")
    print(f"  Best Profit Factor:  hold={best_pf['hold_days']}d  (PF={best_pf['profit_factor']:.2f})")
    print(f"  Lowest Max Drawdown: hold={best_dd['hold_days']}d  (DD={best_dd['max_drawdown']:.1f}%)")

print(f"\n{'='*110}\n")
