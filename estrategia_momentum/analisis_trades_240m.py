#!/usr/bin/env python3
"""
ANALISIS PROFUNDO — Backtest v6+ 240 meses
Destripa trades por epoca, categoria y patrones para diagnosticar
por que el PF decae en periodos largos.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_breakout import MomentumEngine, calculate_atr, ASSETS

# Importar todo del backtest experimental
from backtest_experimental import (
    CONFIG, OPTIONS_ELIGIBLE, LEVERAGE_FACTORS,
    Trade, OptionTradeV2, EquityTracker,
    download_data, historical_volatility, iv_rank,
    generate_all_signals, build_macro_filter, rank_candidates,
    find_candidates, black_scholes_call, monthly_expiration_dte,
    BASE_TICKERS,
)

def run_analysis():
    months = 240
    tickers = BASE_TICKERS

    print("=" * 80)
    print("  ANALISIS PROFUNDO v6+ — 240 MESES")
    print("=" * 80)

    # --- DESCARGAR DATOS ---
    print("\n  Descargando datos...")
    all_data = {}
    for i, ticker in enumerate(tickers):
        df = download_data(ticker, months)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            df['HVOL'] = historical_volatility(df['Close'], CONFIG['hvol_window'])
            all_data[ticker] = df
        if (i + 1) % 20 == 0 or i == len(tickers) - 1:
            print(f"\r  Descargados: {len(all_data)}/{len(tickers)}", end='')
    print(f"\n  Tickers con datos: {len(all_data)}")

    # --- ENGINE + SENALES ---
    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG['rsi_max'],
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG['longs_only']
    )
    signals_data, total_signals = generate_all_signals(all_data, engine)
    macro_bullish = build_macro_filter(all_data)
    all_dates = sorted(set(d for sd in signals_data.values() for d in sd['df'].index.tolist()))

    # --- EJECUTAR BACKTEST (recolectar todos los trades) ---
    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}
    active_options = {}
    all_trades = []
    all_option_trades = []

    for current_date in all_dates:
        # 1. Gestionar trades activos
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

        # 2. Gestionar opciones
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
            iv_val = df['HVOL'].iloc[idx]
            if pd.isna(iv_val) or iv_val <= 0:
                iv_val = opt.entry_iv
            result = opt.update(bar['Close'], iv_val, days_elapsed)
            if result and result['type'] == 'full_exit':
                opt.exit_date = current_date
                options_to_close.append(ticker)
                tracker.update_equity(opt.pnl_euros, current_date)
        for ticker in options_to_close:
            opt = active_options.pop(ticker)
            tracker.open_positions -= 1
            tracker.open_options -= 1
            all_option_trades.append(opt)

        # 3. Buscar nuevas senales
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
                if ticker in OPTIONS_ELIGIBLE and tracker.open_options < CONFIG['max_option_positions']:
                    hvol_series = df['HVOL']
                    current_ivr = iv_rank(hvol_series, idx, CONFIG.get('option_ivr_window', 252))
                    if current_ivr < CONFIG.get('option_max_ivr', 40):
                        open_as_option = True

                if open_as_option:
                    stock_price = bar['Close']
                    iv_val = df['HVOL'].iloc[idx]
                    if pd.isna(iv_val) or iv_val <= 0:
                        iv_val = 0.25
                    strike = stock_price * (1 - CONFIG['option_itm_pct'])
                    dte = monthly_expiration_dte(current_date, CONFIG['option_dte'])
                    T = dte / 365
                    bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], iv_val)
                    premium = bs['price']
                    if premium <= 0.01:
                        continue
                    position_euros = tracker.equity * CONFIG['option_position_pct']
                    n_contracts = position_euros / (premium * 100)
                    if n_contracts < 0.01:
                        continue
                    opt = OptionTradeV2(
                        ticker=ticker, entry_date=current_date,
                        entry_stock_price=stock_price, strike=round(strike, 2),
                        dte_at_entry=dte, entry_option_price=premium,
                        entry_iv=iv_val, num_contracts=n_contracts,
                        position_euros=position_euros,
                    )
                    active_options[ticker] = opt
                    tracker.open_positions += 1
                    tracker.open_options += 1
                else:
                    entry_price = bar['Close']
                    R = prev_atr * 2.0
                    risk_pct = CONFIG['target_risk_per_trade_pct'] / 100
                    lev = LEVERAGE_FACTORS.get(ticker, 1.0)
                    dollar_risk = tracker.equity * risk_pct
                    position_units = dollar_risk / R
                    position_euros = position_units * entry_price
                    max_notional = tracker.equity / CONFIG['max_positions'] * 2
                    if position_euros > max_notional:
                        position_euros = max_notional
                        position_units = position_euros / entry_price
                    trade = Trade(
                        ticker=ticker, entry_price=entry_price,
                        entry_date=current_date, entry_atr=prev_atr,
                        position_euros=position_euros, position_units=position_units,
                    )
                    active_trades[ticker] = trade
                    tracker.open_positions += 1

    # Cerrar posiciones abiertas
    for ticker, trade in active_trades.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            trade._close(df['Close'].iloc[-1], 'end_of_data')
            trade.exit_date = df.index[-1]
            tracker.update_equity(trade.pnl_euros, df.index[-1])
            all_trades.append(trade)
    for ticker, opt in active_options.items():
        if ticker in signals_data:
            df = signals_data[ticker]['df']
            stock_price = df['Close'].iloc[-1]
            intrinsic = max(stock_price - opt.strike, 0)
            opt._close(intrinsic, 'end_of_data')
            opt.exit_date = df.index[-1]
            tracker.update_equity(opt.pnl_euros, df.index[-1])
            all_option_trades.append(opt)

    # =================================================================
    # ANALISIS PROFUNDO
    # =================================================================
    combined = []
    for t in all_trades:
        combined.append({
            'ticker': t.ticker,
            'type': 'stock',
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl_eur': t.pnl_euros,
            'pnl_pct': t.pnl_pct,
            'bars_held': t.bars_held,
            'exit_reason': t.exit_reason,
            'max_r_mult': t.max_r_mult,
            'position_eur': t.position_euros,
            'category': ASSETS.get(t.ticker, {}).get('category', 'UNKNOWN'),
        })
    for t in all_option_trades:
        combined.append({
            'ticker': t.ticker,
            'type': 'option',
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_stock_price,
            'exit_price': t.exit_option_price,
            'pnl_eur': t.pnl_euros,
            'pnl_pct': t.pnl_pct,
            'bars_held': t.bars_held,
            'exit_reason': t.exit_reason,
            'max_r_mult': 0,
            'position_eur': t.position_euros,
            'category': ASSETS.get(t.ticker, {}).get('category', 'UNKNOWN'),
        })

    df_trades = pd.DataFrame(combined)
    df_trades['year'] = df_trades['entry_date'].apply(lambda x: x.year)
    df_trades['win'] = df_trades['pnl_eur'] > 0

    # -----------------------------------------------------------------
    # 1. ANALISIS POR ANO
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  1. RENDIMIENTO POR ANO")
    print(f"{'='*90}")
    print(f"  {'Ano':<6} {'Trades':>7} {'Win%':>6} {'PF':>7} {'P&L EUR':>12} {'AvgWin%':>8} {'AvgLoss%':>9} {'Stocks':>7} {'Opts':>5} {'EmStop':>7}")
    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*8} {'-'*9} {'-'*7} {'-'*5} {'-'*7}")

    for year in sorted(df_trades['year'].unique()):
        yr = df_trades[df_trades['year'] == year]
        n = len(yr)
        wins = yr[yr['win']]
        losses = yr[~yr['win']]
        wr = len(wins) / n * 100 if n > 0 else 0
        gp = wins['pnl_eur'].sum() if len(wins) > 0 else 0
        gl = abs(losses['pnl_eur'].sum()) if len(losses) > 0 else 0.01
        pf = gp / gl if gl > 0 else 999
        pnl = yr['pnl_eur'].sum()
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        n_stocks = len(yr[yr['type'] == 'stock'])
        n_opts = len(yr[yr['type'] == 'option'])
        n_emstop = len(yr[yr['exit_reason'] == 'emergency_stop'])
        marker = " <<<" if pf < 1.0 else ""
        print(f"  {year:<6} {n:>7} {wr:>5.1f}% {pf:>6.2f} {pnl:>+11,.0f} {avg_win:>+7.1f}% {avg_loss:>+8.1f}% {n_stocks:>7} {n_opts:>5} {n_emstop:>7}{marker}")

    # -----------------------------------------------------------------
    # 2. ANALISIS POR REGIMEN DE MERCADO
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  2. RENDIMIENTO POR REGIMEN DE MERCADO")
    print(f"{'='*90}")

    # Clasificar regimenes usando SPY
    if 'SPY' in all_data:
        spy_df = all_data['SPY']
        spy_sma50 = spy_df['Close'].rolling(50).mean()
        spy_sma200 = spy_df['Close'].rolling(200).mean()

        def get_regime(date):
            if date not in spy_df.index:
                # Buscar dia anterior
                prev = spy_df.index[spy_df.index < date]
                if len(prev) == 0:
                    return 'unknown'
                date = prev[-1]
            idx = spy_df.index.get_loc(date)
            close = spy_df['Close'].iloc[idx]
            sma50 = spy_sma50.iloc[idx]
            sma200 = spy_sma200.iloc[idx]
            if pd.isna(sma200):
                return 'unknown'
            if close > sma50 and close > sma200:
                return 'BULL_FUERTE'
            elif close > sma200 and close <= sma50:
                return 'CORRECCION'
            elif close <= sma200 and close > sma50:
                return 'RECUPERACION'
            else:
                return 'BEAR'

        df_trades['regime'] = df_trades['entry_date'].apply(get_regime)

        print(f"  {'Regimen':<16} {'Trades':>7} {'Win%':>6} {'PF':>7} {'P&L EUR':>12} {'AvgWin%':>8} {'AvgLoss%':>9} {'EmStop':>7}")
        print(f"  {'-'*16} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*8} {'-'*9} {'-'*7}")
        for regime in ['BULL_FUERTE', 'CORRECCION', 'RECUPERACION', 'BEAR', 'unknown']:
            rg = df_trades[df_trades['regime'] == regime]
            if len(rg) == 0:
                continue
            n = len(rg)
            wins = rg[rg['win']]
            losses = rg[~rg['win']]
            wr = len(wins) / n * 100
            gp = wins['pnl_eur'].sum() if len(wins) > 0 else 0
            gl = abs(losses['pnl_eur'].sum()) if len(losses) > 0 else 0.01
            pf = gp / gl
            pnl = rg['pnl_eur'].sum()
            avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
            n_emstop = len(rg[rg['exit_reason'] == 'emergency_stop'])
            marker = " <<<" if pf < 1.0 else ""
            print(f"  {regime:<16} {n:>7} {wr:>5.1f}% {pf:>6.2f} {pnl:>+11,.0f} {avg_win:>+7.1f}% {avg_loss:>+8.1f}% {n_emstop:>7}{marker}")

    # -----------------------------------------------------------------
    # 3. ANALISIS POR CATEGORIA
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  3. RENDIMIENTO POR CATEGORIA DE ACTIVO")
    print(f"{'='*90}")
    print(f"  {'Categoria':<20} {'Trades':>7} {'Win%':>6} {'PF':>7} {'P&L EUR':>12} {'AvgWin%':>8} {'AvgLoss%':>9}")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*8} {'-'*9}")
    for cat in sorted(df_trades['category'].unique()):
        cg = df_trades[df_trades['category'] == cat]
        n = len(cg)
        wins = cg[cg['win']]
        losses = cg[~cg['win']]
        wr = len(wins) / n * 100
        gp = wins['pnl_eur'].sum() if len(wins) > 0 else 0
        gl = abs(losses['pnl_eur'].sum()) if len(losses) > 0 else 0.01
        pf = gp / gl
        pnl = cg['pnl_eur'].sum()
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        marker = " <<<" if pf < 1.0 else ""
        print(f"  {cat:<20} {n:>7} {wr:>5.1f}% {pf:>6.2f} {pnl:>+11,.0f} {avg_win:>+7.1f}% {avg_loss:>+8.1f}%{marker}")

    # -----------------------------------------------------------------
    # 4. ANALISIS POR RAZON DE SALIDA
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  4. RENDIMIENTO POR RAZON DE SALIDA")
    print(f"{'='*90}")
    print(f"  {'Razon':<20} {'Trades':>7} {'Win%':>6} {'PF':>7} {'P&L EUR':>12} {'AvgPnl%':>8} {'AvgBars':>8}")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*8} {'-'*8}")
    for reason in sorted(df_trades['exit_reason'].unique()):
        rg = df_trades[df_trades['exit_reason'] == reason]
        n = len(rg)
        wins = rg[rg['win']]
        losses = rg[~rg['win']]
        wr = len(wins) / n * 100
        gp = wins['pnl_eur'].sum() if len(wins) > 0 else 0
        gl = abs(losses['pnl_eur'].sum()) if len(losses) > 0 else 0.01
        pf = gp / gl
        pnl = rg['pnl_eur'].sum()
        avg_pnl = rg['pnl_pct'].mean()
        avg_bars = rg['bars_held'].mean()
        print(f"  {reason:<20} {n:>7} {wr:>5.1f}% {pf:>6.2f} {pnl:>+11,.0f} {avg_pnl:>+7.1f}% {avg_bars:>7.1f}")

    # -----------------------------------------------------------------
    # 5. STOCKS vs OPCIONES POR PERIODO
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  5. STOCKS vs OPCIONES — POR PERIODO")
    print(f"{'='*90}")

    periods = [
        ('2006-2010', 2006, 2010),
        ('2011-2015', 2011, 2015),
        ('2016-2020', 2016, 2020),
        ('2021-2026', 2021, 2026),
    ]

    print(f"  {'Periodo':<12} {'Tipo':<8} {'Trades':>7} {'Win%':>6} {'PF':>7} {'P&L EUR':>12} {'AvgWin%':>8} {'AvgLoss%':>9}")
    print(f"  {'-'*12} {'-'*8} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*8} {'-'*9}")
    for label, y_start, y_end in periods:
        for ttype in ['stock', 'option']:
            subset = df_trades[(df_trades['year'] >= y_start) & (df_trades['year'] <= y_end) & (df_trades['type'] == ttype)]
            if len(subset) == 0:
                continue
            n = len(subset)
            wins = subset[subset['win']]
            losses = subset[~subset['win']]
            wr = len(wins) / n * 100
            gp = wins['pnl_eur'].sum() if len(wins) > 0 else 0
            gl = abs(losses['pnl_eur'].sum()) if len(losses) > 0 else 0.01
            pf = gp / gl
            pnl = subset['pnl_eur'].sum()
            avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
            marker = " <<<" if pf < 1.0 else ""
            print(f"  {label:<12} {ttype:<8} {n:>7} {wr:>5.1f}% {pf:>6.2f} {pnl:>+11,.0f} {avg_win:>+7.1f}% {avg_loss:>+8.1f}%{marker}")

    # -----------------------------------------------------------------
    # 6. TOP PERDEDORES Y GANADORES
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  6. TOP 15 PEORES TRADES (por P&L EUR)")
    print(f"{'='*90}")
    worst = df_trades.nsmallest(15, 'pnl_eur')
    print(f"  {'Fecha':<12} {'Ticker':<10} {'Tipo':<7} {'P&L EUR':>10} {'P&L%':>8} {'Bars':>5} {'Salida':<16} {'Cat':<15}")
    print(f"  {'-'*12} {'-'*10} {'-'*7} {'-'*10} {'-'*8} {'-'*5} {'-'*16} {'-'*15}")
    for _, r in worst.iterrows():
        d = r['entry_date'].strftime('%Y-%m-%d')
        print(f"  {d:<12} {r['ticker']:<10} {r['type']:<7} {r['pnl_eur']:>+9,.0f} {r['pnl_pct']:>+7.1f}% {r['bars_held']:>5} {r['exit_reason']:<16} {r['category']:<15}")

    print(f"\n{'='*90}")
    print(f"  7. TOP 15 MEJORES TRADES (por P&L EUR)")
    print(f"{'='*90}")
    best = df_trades.nlargest(15, 'pnl_eur')
    print(f"  {'Fecha':<12} {'Ticker':<10} {'Tipo':<7} {'P&L EUR':>10} {'P&L%':>8} {'Bars':>5} {'Salida':<16} {'Cat':<15}")
    print(f"  {'-'*12} {'-'*10} {'-'*7} {'-'*10} {'-'*8} {'-'*5} {'-'*16} {'-'*15}")
    for _, r in best.iterrows():
        d = r['entry_date'].strftime('%Y-%m-%d')
        print(f"  {d:<12} {r['ticker']:<10} {r['type']:<7} {r['pnl_eur']:>+9,.0f} {r['pnl_pct']:>+7.1f}% {r['bars_held']:>5} {r['exit_reason']:<16} {r['category']:<15}")

    # -----------------------------------------------------------------
    # 8. ANALISIS DE PERDEDORES POR TIME EXIT
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  8. TIME EXITS — El mayor lastre")
    print(f"{'='*90}")
    time_exits = df_trades[df_trades['exit_reason'] == 'time_exit']
    te_by_year = time_exits.groupby('year').agg(
        n=('pnl_eur', 'count'),
        pnl_total=('pnl_eur', 'sum'),
        avg_pnl_pct=('pnl_pct', 'mean'),
    )
    print(f"  {'Ano':<6} {'N':>5} {'P&L EUR':>12} {'AvgPnl%':>8}")
    print(f"  {'-'*6} {'-'*5} {'-'*12} {'-'*8}")
    for year, row in te_by_year.iterrows():
        print(f"  {year:<6} {row['n']:>5} {row['pnl_total']:>+11,.0f} {row['avg_pnl_pct']:>+7.1f}%")
    print(f"\n  TOTAL time_exit: {len(time_exits)} trades, P&L EUR {time_exits['pnl_eur'].sum():+,.0f}")
    print(f"  % del portfolio: son el {len(time_exits)/len(df_trades)*100:.1f}% de los trades")
    print(f"  Dano total: EUR {time_exits['pnl_eur'].sum():+,.0f} de EUR {df_trades['pnl_eur'].sum():+,.0f} netos")

    # -----------------------------------------------------------------
    # 9. TICKERS PROBLEMATICOS
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  9. TICKERS MAS PROBLEMATICOS (>5 trades, PF < 1.0)")
    print(f"{'='*90}")
    ticker_stats = df_trades.groupby('ticker').agg(
        n=('pnl_eur', 'count'),
        wins=('win', 'sum'),
        pnl_total=('pnl_eur', 'sum'),
        avg_pnl_pct=('pnl_pct', 'mean'),
    )
    ticker_stats['wr'] = ticker_stats['wins'] / ticker_stats['n'] * 100
    # Calcular PF por ticker
    for ticker in ticker_stats.index:
        tg = df_trades[df_trades['ticker'] == ticker]
        gp = tg[tg['win']]['pnl_eur'].sum()
        gl = abs(tg[~tg['win']]['pnl_eur'].sum())
        ticker_stats.loc[ticker, 'pf'] = gp / gl if gl > 0 else 999

    bad_tickers = ticker_stats[(ticker_stats['n'] >= 5) & (ticker_stats['pf'] < 1.0)].sort_values('pnl_total')
    print(f"  {'Ticker':<10} {'Trades':>7} {'Win%':>6} {'PF':>7} {'P&L EUR':>12} {'AvgPnl%':>8} {'Cat':<15}")
    print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*8} {'-'*15}")
    for ticker, row in bad_tickers.iterrows():
        cat = ASSETS.get(ticker, {}).get('category', 'UNKNOWN')
        print(f"  {ticker:<10} {row['n']:>7} {row['wr']:>5.1f}% {row['pf']:>6.2f} {row['pnl_total']:>+11,.0f} {row['avg_pnl_pct']:>+7.1f}% {cat:<15}")

    good_tickers = ticker_stats[(ticker_stats['n'] >= 5) & (ticker_stats['pf'] >= 1.5)].sort_values('pnl_total', ascending=False)
    print(f"\n  TICKERS RENTABLES (>5 trades, PF >= 1.5):")
    print(f"  {'Ticker':<10} {'Trades':>7} {'Win%':>6} {'PF':>7} {'P&L EUR':>12} {'AvgPnl%':>8} {'Cat':<15}")
    print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*8} {'-'*15}")
    for ticker, row in good_tickers.iterrows():
        cat = ASSETS.get(ticker, {}).get('category', 'UNKNOWN')
        print(f"  {ticker:<10} {row['n']:>7} {row['wr']:>5.1f}% {row['pf']:>6.2f} {row['pnl_total']:>+11,.0f} {row['avg_pnl_pct']:>+7.1f}% {cat:<15}")

    # -----------------------------------------------------------------
    # 10. PATRON CLAVE: Tamano de posicion vs resultado
    # -----------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  10. EVOLUCION DEL EQUITY Y TAMANO DE POSICION")
    print(f"{'='*90}")
    df_trades_sorted = df_trades.sort_values('entry_date')
    # Analizar en ventanas de 100 trades
    window = 100
    print(f"  {'Ventana':<15} {'Trades':>7} {'Win%':>6} {'PF':>7} {'P&L EUR':>12} {'AvgPos EUR':>12} {'AvgPnl%':>8}")
    print(f"  {'-'*15} {'-'*7} {'-'*6} {'-'*7} {'-'*12} {'-'*12} {'-'*8}")
    for i in range(0, len(df_trades_sorted), window):
        chunk = df_trades_sorted.iloc[i:i+window]
        if len(chunk) < 10:
            continue
        n = len(chunk)
        start_yr = chunk.iloc[0]['entry_date'].year
        end_yr = chunk.iloc[-1]['entry_date'].year
        wins = chunk[chunk['win']]
        losses = chunk[~chunk['win']]
        wr = len(wins) / n * 100
        gp = wins['pnl_eur'].sum() if len(wins) > 0 else 0
        gl = abs(losses['pnl_eur'].sum()) if len(losses) > 0 else 0.01
        pf = gp / gl
        pnl = chunk['pnl_eur'].sum()
        avg_pos = chunk['position_eur'].mean()
        avg_pnl = chunk['pnl_pct'].mean()
        label = f"{start_yr}-{end_yr}"
        marker = " <<<" if pf < 1.0 else ""
        print(f"  {label:<15} {n:>7} {wr:>5.1f}% {pf:>6.2f} {pnl:>+11,.0f} {avg_pos:>11,.0f} {avg_pnl:>+7.1f}%{marker}")

    print(f"\n{'='*90}")
    print(f"  FIN DEL ANALISIS")
    print(f"{'='*90}")


if __name__ == "__main__":
    run_analysis()
