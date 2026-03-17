#!/usr/bin/env python3
"""
Monthly Equity Report — Mark-to-Market
Corre el backtest v8 a 36 meses y genera un informe mensual con:
- Equity realizada (cash)
- Valor mark-to-market de posiciones abiertas
- Equity total (cash + MTM)
- Posiciones abiertas con detalle
- Return mensual y acumulado
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from momentum_breakout import MomentumEngine, calculate_atr, ASSETS
from backtest_experimental import (
    CONFIG, BASE_TICKERS, OPTIONS_ELIGIBLE, LEVERAGE_FACTORS,
    download_data, historical_volatility, find_candidates, rank_candidates,
    Trade, OptionTradeV2, EquityTracker, black_scholes_call,
    generate_all_signals, iv_rank,
)


def run_monthly_report(months=36):
    tickers = BASE_TICKERS
    n_tickers = len(tickers)

    print(f"\n{'='*90}")
    print(f"  MONTHLY EQUITY REPORT — v8 (Opciones CALL) — {months} MESES")
    print(f"{'='*90}")

    # Descargar datos
    print("  Descargando datos...")
    all_data = {}
    for i, ticker in enumerate(tickers):
        df = download_data(ticker, months)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            df['HVOL'] = historical_volatility(df['Close'], 30)
            all_data[ticker] = df
        if (i + 1) % 50 == 0:
            print(f"\r  Descargados: {len(all_data)}/{n_tickers}", end='')
    print(f"\n  Tickers con datos: {len(all_data)}")

    if not all_data:
        return

    # Generar señales
    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG['rsi_max'],
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG['longs_only']
    )
    signals_data, total_signals = generate_all_signals(all_data, engine)

    # Macro filter
    spy_data = all_data.get('SPY')
    if spy_data is None:
        print("ERROR: No SPY data")
        return
    sma = spy_data['Close'].rolling(CONFIG['macro_sma_period']).mean()
    macro_bullish = (spy_data['Close'] > sma).to_dict()

    # Fechas de trading
    all_dates = sorted(set().union(*[df.index for df in all_data.values()]))
    all_dates = [d for d in all_dates if d >= spy_data.index[CONFIG['macro_sma_period'] + 10]]

    # Tracker
    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}
    active_options = {}
    all_trades = []
    all_option_trades = []

    # Monthly snapshots
    monthly_data = []
    last_month = None

    use_options = True
    use_leverage_scaling = False

    for current_date in all_dates:

        # 1. GESTIONAR TRADES ACTIVOS
        trades_to_close = []
        for ticker, trade in active_trades.items():
            if ticker not in signals_data:
                continue
            df = signals_data[ticker]['df']
            if current_date not in df.index:
                continue
            idx = df.index.get_loc(current_date)
            bar = df.iloc[idx]
            result = trade.update(bar['High'], bar['Low'], bar['Close'], df['ATR'].iloc[idx])
            if result and result['type'] == 'full_exit':
                trade.exit_date = current_date
                trades_to_close.append(ticker)
                tracker.update_equity(trade.pnl_euros, current_date)

        for ticker in trades_to_close:
            trade = active_trades.pop(ticker)
            tracker.open_positions -= 1
            all_trades.append(trade)

        # 2. GESTIONAR OPCIONES
        options_to_close = []
        for ticker, opt in active_options.items():
            if ticker not in signals_data:
                continue
            df = signals_data[ticker]['df']
            if current_date not in df.index:
                continue
            idx = df.index.get_loc(current_date)
            bar = df.iloc[idx]
            days_elapsed = (current_date - opt.entry_date).days
            iv = df['HVOL'].iloc[idx]
            if pd.isna(iv) or iv <= 0:
                iv = opt.entry_iv
            result = opt.update(bar['Close'], iv, days_elapsed)
            if result and result['type'] == 'full_exit':
                opt.exit_date = current_date
                options_to_close.append(ticker)
                tracker.update_equity(opt.pnl_euros, current_date)

        for ticker in options_to_close:
            opt = active_options.pop(ticker)
            tracker.open_positions -= 1
            tracker.open_options -= 1
            all_option_trades.append(opt)

        # 3. BUSCAR NUEVAS SEÑALES
        if CONFIG['use_macro_filter']:
            if current_date in macro_bullish:
                is_macro_ok = macro_bullish[current_date]
            else:
                prev_dates = [d for d in macro_bullish if d < current_date]
                is_macro_ok = macro_bullish[prev_dates[-1]] if prev_dates else False
        else:
            is_macro_ok = True

        if tracker.open_positions < CONFIG['max_positions'] and is_macro_ok:
            candidates = find_candidates(signals_data, {**active_trades, **active_options}, current_date, is_macro_ok)
            ranked = rank_candidates(candidates, signals_data)

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= CONFIG['max_positions']:
                    break

                df = signals_data[ticker]['df']
                bar = df.iloc[idx]

                open_as_option = False
                current_ivr = None
                if use_options and ticker in OPTIONS_ELIGIBLE and tracker.open_options < CONFIG['max_option_positions']:
                    hvol_series = df['HVOL']
                    current_ivr = iv_rank(hvol_series, idx, CONFIG.get('option_ivr_window', 252))
                    max_ivr = CONFIG.get('option_max_ivr', 40)
                    if current_ivr < max_ivr:
                        open_as_option = True

                if open_as_option:
                    stock_price = bar['Open']
                    strike = stock_price * (1 - CONFIG['option_itm_pct'])
                    dte = CONFIG['option_dte']
                    T = dte / 365.0
                    iv_val = df['HVOL'].iloc[idx]
                    if pd.isna(iv_val) or iv_val <= 0:
                        iv_val = 0.30
                    bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], iv_val)
                    option_price = bs['price']
                    option_price *= (1 + CONFIG['option_spread_pct'] / 100 / 2)
                    actual_dte = dte
                    size = tracker.get_option_size(option_price)
                    if size['premium'] < 50:
                        continue

                    opt = OptionTradeV2(
                        ticker=ticker, entry_date=current_date,
                        entry_stock_price=stock_price, strike=strike,
                        dte_at_entry=actual_dte, entry_option_price=option_price,
                        entry_iv=iv_val, num_contracts=size['contracts'],
                        position_euros=size['premium'],
                    )
                    active_options[ticker] = opt
                    tracker.open_positions += 1
                    tracker.open_options += 1
                else:
                    size_info = tracker.get_position_size(ticker, prev_atr, bar['Open'], use_leverage_scaling)
                    entry_price = bar['Open'] * (1 + CONFIG['slippage_pct'] / 100)
                    position_euros = size_info['notional']
                    position_units = size_info['units']

                    max_per_position = tracker.equity / CONFIG['max_positions']
                    if position_euros > max_per_position:
                        position_euros = max_per_position
                        position_units = position_euros / entry_price

                    if position_euros < 100:
                        continue

                    trade = Trade(
                        ticker=ticker, entry_price=entry_price,
                        entry_date=current_date, entry_atr=prev_atr,
                        position_euros=position_euros, position_units=position_units,
                    )
                    active_trades[ticker] = trade
                    tracker.open_positions += 1

        # MONTHLY SNAPSHOT (último día de trading del mes)
        current_month = (current_date.year, current_date.month)
        next_idx = all_dates.index(current_date) + 1
        if next_idx < len(all_dates):
            next_month = (all_dates[next_idx].year, all_dates[next_idx].month)
        else:
            next_month = None  # último día del backtest

        is_month_end = (next_month != current_month) or (next_month is None)

        if is_month_end:
            # Calcular MTM de posiciones abiertas
            mtm_stocks = 0
            stock_positions = []
            for ticker, trade in active_trades.items():
                if ticker in signals_data:
                    df = signals_data[ticker]['df']
                    if current_date in df.index:
                        price = df['Close'].loc[current_date]
                        unrealized = (price - trade.entry_price) * trade.position_units
                        mtm_stocks += unrealized
                        pnl_pct = unrealized / trade.position_euros * 100 if trade.position_euros > 0 else 0
                        stock_positions.append({
                            'ticker': ticker, 'entry': trade.entry_price,
                            'current': price, 'units': trade.position_units,
                            'notional': trade.position_euros,
                            'unrealized': unrealized, 'pnl_pct': pnl_pct,
                        })

            mtm_options = 0
            opt_positions = []
            for ticker, opt in active_options.items():
                if ticker in signals_data:
                    df = signals_data[ticker]['df']
                    if current_date in df.index:
                        stock_price = df['Close'].loc[current_date]
                        days_elapsed = (current_date - opt.entry_date).days
                        remaining_dte = opt.dte_at_entry - days_elapsed
                        T = max(remaining_dte, 1) / 365.0
                        iv_val = df['HVOL'].loc[current_date] if not pd.isna(df['HVOL'].loc[current_date]) else opt.entry_iv
                        bs = black_scholes_call(stock_price, opt.strike, T, CONFIG['risk_free_rate'], iv_val)
                        current_opt_price = bs['price']
                        unrealized = (current_opt_price - opt.entry_option_price) * opt.num_contracts * 100
                        mtm_options += unrealized
                        pnl_pct = unrealized / opt.position_euros * 100 if opt.position_euros > 0 else 0
                        opt_positions.append({
                            'ticker': f"OPT {ticker}", 'strike': opt.strike,
                            'entry_prem': opt.entry_option_price,
                            'current_prem': current_opt_price,
                            'contracts': opt.num_contracts,
                            'notional': opt.position_euros,
                            'unrealized': unrealized, 'pnl_pct': pnl_pct,
                        })

            # Macro status
            if current_date in macro_bullish:
                macro_now = macro_bullish[current_date]
            else:
                prev = [d for d in macro_bullish if d < current_date]
                macro_now = macro_bullish[prev[-1]] if prev else False

            total_mtm = mtm_stocks + mtm_options
            equity_realized = tracker.equity
            equity_total = equity_realized + total_mtm

            monthly_data.append({
                'date': current_date,
                'equity_realized': equity_realized,
                'mtm_stocks': mtm_stocks,
                'mtm_options': mtm_options,
                'equity_total': equity_total,
                'n_stocks': len(active_trades),
                'n_options': len(active_options),
                'n_total': len(active_trades) + len(active_options),
                'macro': 'BULL' if macro_now else 'BEAR',
                'stock_positions': stock_positions,
                'opt_positions': opt_positions,
            })

    # =====================================================================
    # IMPRIMIR INFORME
    # =====================================================================
    print(f"\n{'='*90}")
    print(f"  INFORME MENSUAL MARK-TO-MARKET")
    print(f"{'='*90}")
    print()
    print(f"  {'Mes':<10} {'Macro':<5} {'Pos':<5} {'Equity Real':>12} {'MTM Stocks':>12} {'MTM Opts':>10} "
          f"{'TOTAL':>12} {'Ret Mes':>8} {'Ret Acum':>9}")
    print(f"  {'-'*10} {'-'*5} {'-'*5} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*8} {'-'*9}")

    prev_total = CONFIG['initial_capital']
    for m in monthly_data:
        ret_month = (m['equity_total'] / prev_total - 1) * 100
        ret_acum = (m['equity_total'] / CONFIG['initial_capital'] - 1) * 100
        print(f"  {m['date'].strftime('%Y-%m'):<10} {m['macro']:<5} {m['n_total']:>3}/10 "
              f"EUR {m['equity_realized']:>10,.0f} EUR {m['mtm_stocks']:>+10,.0f} "
              f"EUR {m['mtm_options']:>+8,.0f} "
              f"EUR {m['equity_total']:>10,.0f} {ret_month:>+7.1f}% {ret_acum:>+8.1f}%")
        prev_total = m['equity_total']

    # Detalle de posiciones en meses clave
    print(f"\n{'='*90}")
    print(f"  DETALLE DE POSICIONES ABIERTAS (meses seleccionados)")
    print(f"{'='*90}")

    for m in monthly_data:
        month_str = m['date'].strftime('%Y-%m')
        if m['stock_positions'] or m['opt_positions']:
            print(f"\n  --- {month_str} | {m['macro']} | {m['n_total']}/10 pos | "
                  f"Equity total: EUR {m['equity_total']:,.0f} ---")

            for p in sorted(m['stock_positions'], key=lambda x: x['unrealized'], reverse=True):
                sign = '+' if p['unrealized'] >= 0 else ''
                print(f"    {p['ticker']:<10} Entry ${p['entry']:>9,.2f} → ${p['current']:>9,.2f} | "
                      f"{p['units']:.2f}u x EUR {p['notional']:>7,.0f} | "
                      f"P&L EUR {sign}{p['unrealized']:>7,.0f} ({sign}{p['pnl_pct']:.1f}%)")

            for p in m['opt_positions']:
                sign = '+' if p['unrealized'] >= 0 else ''
                print(f"    {p['ticker']:<10} K=${p['strike']:>8,.2f} | Prem ${p['entry_prem']:.2f} → ${p['current_prem']:.2f} | "
                      f"{p['contracts']:.2f}c x EUR {p['notional']:>7,.0f} | "
                      f"P&L EUR {sign}{p['unrealized']:>7,.0f} ({sign}{p['pnl_pct']:.1f}%)")

    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--months', type=int, default=36)
    args = parser.parse_args()
    run_monthly_report(args.months)
