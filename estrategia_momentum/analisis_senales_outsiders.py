#!/usr/bin/env python3
"""
Análisis de componentes de señal: ¿por qué fallan las señales crypto?
Compara KER, RSI, Volume, Breakout, ATR% entre:
  - Crypto ganadores vs perdedores
  - Futures ganadores vs perdedores

Fecha: 2026-02-28
"""

import sys
import os
import numpy as np
import pandas as pd

# Importar desde backtest_outsiders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_outsiders import (
    OUTSIDER_TICKERS, OPTIONS_MAP, CONFIG, TICKER_LIST,
    download_all, generate_all_signals, build_macro_filter,
    rank_candidates, find_candidates, EquityTracker, Trade,
)
from momentum_breakout import MomentumEngine, calculate_atr


def analyze_signals(months=60, max_positions=10):
    """
    Ejecuta backtest spot-only capturando componentes de señal en cada entrada.
    Usa pos=10 para maximizar el número de trades y tener mejor muestra estadística.
    """
    CONFIG['max_positions'] = max_positions
    mp = CONFIG['max_positions']

    print(f"Descargando datos ({months} meses)...")
    all_data, failed = download_all(TICKER_LIST, months)

    print("Generando señales...")
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

    # Timeline
    all_dates = sorted(set(
        d for t, sd in signals_data.items()
        for d in sd['df'].index.tolist()
    ))

    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}
    active_options = {}  # vacío (spot-only)

    # Almacenar componentes de señal por trade
    trade_diagnostics = []

    for current_date in all_dates:
        # Gestionar trades activos
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

            # Buscar diagnóstico guardado para este trade
            for diag in trade_diagnostics:
                if (diag['ticker'] == ticker and
                    diag['entry_date'] == trade.entry_date):
                    diag['pnl_euros'] = trade.pnl_euros
                    diag['pnl_pct'] = (trade.pnl_euros / trade.position_euros * 100) if trade.position_euros else 0
                    diag['exit_reason'] = trade.exit_reason
                    diag['hold_days'] = (trade.exit_date - trade.entry_date).days
                    diag['winner'] = trade.pnl_euros > 0
                    break

        # Buscar nuevas señales
        if CONFIG['use_macro_filter']:
            prev_dates = [d for d in macro_bullish if d < current_date]
            if len(prev_dates) >= 2:
                is_macro_ok = macro_bullish[prev_dates[-2]]
            elif prev_dates:
                is_macro_ok = macro_bullish[prev_dates[-1]]
            else:
                is_macro_ok = True
        else:
            is_macro_ok = True

        if tracker.open_positions < mp and is_macro_ok:
            candidates = find_candidates(signals_data, active_trades, active_options,
                                         current_date, is_macro_ok)
            ranked = rank_candidates(candidates, signals_data)

            for ticker, idx, prev_atr, composite_score in ranked:
                if tracker.open_positions >= mp:
                    break

                df = signals_data[ticker]['df']
                bar = df.iloc[idx]
                prev_idx = idx - 1

                # === CAPTURAR COMPONENTES DE SEÑAL ===
                sd = signals_data[ticker]
                ker_val = sd['ker'].iloc[prev_idx] if prev_idx >= 0 else 0
                rsi_val = sd['rsi'].iloc[prev_idx] if prev_idx >= 0 else 50
                rsi_score = max(0, min(1, (rsi_val - CONFIG['rsi_threshold']) / (CONFIG['rsi_max'] - CONFIG['rsi_threshold'])))
                vol_val = sd['vol_ratio'].iloc[prev_idx] if prev_idx >= 0 else 1.0
                vol_score = min(1, max(0, (vol_val - 1.0) / 2.0))

                if prev_idx >= 1:
                    close_prev = df['Close'].iloc[prev_idx]
                    rolling_high_prev = df['High'].iloc[max(0, prev_idx - CONFIG['breakout_period']):prev_idx].max()
                    breakout_pct = (close_prev - rolling_high_prev) / rolling_high_prev if rolling_high_prev > 0 else 0
                    breakout_score = min(1, max(0, breakout_pct / 0.05))
                else:
                    breakout_pct = 0
                    breakout_score = 0

                price_prev = df['Close'].iloc[prev_idx] if prev_idx >= 0 else 1
                atr_pct = prev_atr / price_prev if price_prev > 0 else 0
                atr_score = min(1, atr_pct / 0.04)

                # Forward return (20 bars después de entrada) para ver "calidad" de la señal
                fwd_idx = min(idx + 20, len(df) - 1)
                fwd_return = (df['Close'].iloc[fwd_idx] / bar['Open'] - 1) * 100 if bar['Open'] > 0 else 0

                # Max adverse excursion (MAE) - máxima caída desde entrada en 20 bars
                if idx + 1 < len(df):
                    future_lows = df['Low'].iloc[idx:min(idx+20, len(df))]
                    mae = (future_lows.min() / bar['Open'] - 1) * 100 if bar['Open'] > 0 else 0
                else:
                    mae = 0

                # Max favorable excursion (MFE)
                if idx + 1 < len(df):
                    future_highs = df['High'].iloc[idx:min(idx+20, len(df))]
                    mfe = (future_highs.max() / bar['Open'] - 1) * 100 if bar['Open'] > 0 else 0
                else:
                    mfe = 0

                cat = OUTSIDER_TICKERS.get(ticker, {}).get('category', 'UNKNOWN')

                trade_diagnostics.append({
                    'ticker': ticker,
                    'category': cat,
                    'entry_date': current_date,
                    'composite_score': composite_score,
                    'ker': ker_val,
                    'rsi': rsi_val,
                    'rsi_score': rsi_score,
                    'vol_ratio': vol_val,
                    'vol_score': vol_score,
                    'breakout_pct': breakout_pct * 100,
                    'breakout_score': breakout_score,
                    'atr_pct': atr_pct * 100,
                    'atr_score': atr_score,
                    'fwd_return_20b': fwd_return,
                    'mae_20b': mae,
                    'mfe_20b': mfe,
                    # Estos se rellenan al cerrar:
                    'pnl_euros': None,
                    'pnl_pct': None,
                    'exit_reason': None,
                    'hold_days': None,
                    'winner': None,
                })

                # Abrir el trade
                position_size = tracker.get_position_size(ticker, prev_atr, bar['Open'])
                notional = position_size['notional']
                units = position_size['units']
                if notional < 50:
                    trade_diagnostics.pop()
                    continue
                trade = Trade(
                    ticker=ticker, entry_date=current_date,
                    entry_price=bar['Open'], entry_atr=prev_atr,
                    position_euros=notional,
                    position_units=units,
                )
                active_trades[ticker] = trade
                tracker.open_positions += 1

    # Cerrar trades abiertos al final
    for ticker, trade in active_trades.items():
        df = signals_data[ticker]['df']
        if len(df) > 0:
            last_bar = df.iloc[-1]
            trade.exit_date = df.index[-1]
            trade.exit_reason = 'end_of_data'
            pnl = (last_bar['Close'] - trade.entry_price) * trade.position_units
            trade.pnl_euros = round(pnl)
            tracker.update_equity(trade.pnl_euros, trade.exit_date)
            for diag in trade_diagnostics:
                if diag['ticker'] == ticker and diag['entry_date'] == trade.entry_date:
                    diag['pnl_euros'] = trade.pnl_euros
                    diag['pnl_pct'] = (trade.pnl_euros / trade.position_euros * 100) if trade.position_euros else 0
                    diag['exit_reason'] = trade.exit_reason
                    diag['hold_days'] = (trade.exit_date - trade.entry_date).days
                    diag['winner'] = trade.pnl_euros > 0

    return trade_diagnostics


def print_analysis(diagnostics):
    df = pd.DataFrame(diagnostics)
    df = df.dropna(subset=['pnl_euros'])

    print(f"\n{'='*80}")
    print(f"  ANÁLISIS DE COMPONENTES DE SEÑAL — OUTSIDERS 60m")
    print(f"  Total trades analizados: {len(df)}")
    print(f"{'='*80}")

    # === 1. MEDIAS POR CATEGORÍA Y RESULTADO ===
    print(f"\n{'─'*80}")
    print(f"  1. COMPONENTES DE SEÑAL: MEDIAS POR CATEGORÍA Y RESULTADO")
    print(f"{'─'*80}")

    for cat in ['CRYPTO', 'FUTURES']:
        cat_df = df[df['category'] == cat]
        wins = cat_df[cat_df['winner'] == True]
        losses = cat_df[cat_df['winner'] == False]

        print(f"\n  {'═'*35} {cat} {'═'*35}")
        print(f"  {'Componente':<18} {'WINS':>8} {'LOSSES':>8} {'TODOS':>8} │ {'Delta W-L':>10}")
        print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} │ {'─'*10}")

        for col, fmt, label in [
            ('ker', '.3f', 'KER'),
            ('rsi', '.1f', 'RSI'),
            ('vol_ratio', '.2f', 'Vol Ratio'),
            ('breakout_pct', '.2f', 'Breakout %'),
            ('atr_pct', '.2f', 'ATR %'),
            ('composite_score', '.3f', 'Score Total'),
        ]:
            w_mean = wins[col].mean() if len(wins) > 0 else 0
            l_mean = losses[col].mean() if len(losses) > 0 else 0
            all_mean = cat_df[col].mean()
            delta = w_mean - l_mean
            print(f"  {label:<18} {w_mean:>8{fmt}} {l_mean:>8{fmt}} {all_mean:>8{fmt}} │ {delta:>+10{fmt}}")

        # Forward returns
        print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} │ {'─'*10}")
        for col, fmt, label in [
            ('fwd_return_20b', '.1f', 'Fwd Ret 20b %'),
            ('mae_20b', '.1f', 'MAE 20b %'),
            ('mfe_20b', '.1f', 'MFE 20b %'),
        ]:
            w_mean = wins[col].mean() if len(wins) > 0 else 0
            l_mean = losses[col].mean() if len(losses) > 0 else 0
            all_mean = cat_df[col].mean()
            delta = w_mean - l_mean
            print(f"  {label:<18} {w_mean:>8{fmt}} {l_mean:>8{fmt}} {all_mean:>8{fmt}} │ {delta:>+10{fmt}}")

        print(f"\n  Trades: {len(cat_df)} total, {len(wins)} wins ({len(wins)/len(cat_df)*100:.1f}%), "
              f"{len(losses)} losses ({len(losses)/len(cat_df)*100:.1f}%)")

    # === 2. COMPARACIÓN DIRECTA CRYPTO vs FUTURES ===
    print(f"\n{'─'*80}")
    print(f"  2. COMPARACIÓN DIRECTA: CRYPTO vs FUTURES (todos los trades)")
    print(f"{'─'*80}")

    crypto = df[df['category'] == 'CRYPTO']
    futures = df[df['category'] == 'FUTURES']

    print(f"\n  {'Componente':<18} {'CRYPTO':>10} {'FUTURES':>10} │ {'Diferencia':>10}")
    print(f"  {'─'*18} {'─'*10} {'─'*10} │ {'─'*10}")

    for col, fmt, label in [
        ('ker', '.3f', 'KER'),
        ('rsi', '.1f', 'RSI'),
        ('vol_ratio', '.2f', 'Vol Ratio'),
        ('breakout_pct', '.2f', 'Breakout %'),
        ('atr_pct', '.2f', 'ATR %'),
        ('composite_score', '.3f', 'Score Total'),
        ('fwd_return_20b', '.1f', 'Fwd Ret 20b %'),
        ('mae_20b', '.1f', 'MAE 20b %'),
        ('mfe_20b', '.1f', 'MFE 20b %'),
    ]:
        c_mean = crypto[col].mean()
        f_mean = futures[col].mean()
        delta = c_mean - f_mean
        print(f"  {label:<18} {c_mean:>10{fmt}} {f_mean:>10{fmt}} │ {delta:>+10{fmt}}")

    # === 3. DISTRIBUCIÓN KER: ¿false trends en crypto? ===
    print(f"\n{'─'*80}")
    print(f"  3. DISTRIBUCIÓN KER POR CATEGORÍA Y RESULTADO")
    print(f"{'─'*80}")

    for cat in ['CRYPTO', 'FUTURES']:
        cat_df = df[df['category'] == cat]
        wins = cat_df[cat_df['winner'] == True]
        losses = cat_df[cat_df['winner'] == False]

        print(f"\n  {cat}:")
        bins = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0)]
        print(f"  {'KER Range':<14} {'Total':>6} {'Wins':>6} {'Win%':>6} {'Avg P&L%':>10} {'Avg Fwd20b':>10}")
        print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*6} {'─'*10} {'─'*10}")

        for low, high in bins:
            in_bin = cat_df[(cat_df['ker'] >= low) & (cat_df['ker'] < high)]
            if len(in_bin) == 0:
                continue
            n_wins = (in_bin['winner'] == True).sum()
            win_pct = n_wins / len(in_bin) * 100
            avg_pnl = in_bin['pnl_pct'].mean()
            avg_fwd = in_bin['fwd_return_20b'].mean()
            print(f"  {low:.2f} - {high:.2f}   {len(in_bin):>6} {n_wins:>6} {win_pct:>5.1f}% {avg_pnl:>+9.1f}% {avg_fwd:>+9.1f}%")

    # === 4. EXIT REASONS: ¿dónde mueren los trades crypto? ===
    print(f"\n{'─'*80}")
    print(f"  4. RAZONES DE SALIDA POR CATEGORÍA")
    print(f"{'─'*80}")

    for cat in ['CRYPTO', 'FUTURES']:
        cat_df = df[df['category'] == cat]
        print(f"\n  {cat}:")
        print(f"  {'Exit Reason':<20} {'Count':>6} {'Win%':>6} {'Avg P&L%':>10} {'Avg Hold':>10} {'Avg KER':>8}")
        print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*8}")

        for reason in ['trailing_stop', 'emergency_stop', 'end_of_data']:
            r_df = cat_df[cat_df['exit_reason'] == reason]
            if len(r_df) == 0:
                continue
            n_wins = (r_df['winner'] == True).sum()
            win_pct = n_wins / len(r_df) * 100
            avg_pnl = r_df['pnl_pct'].mean()
            avg_hold = r_df['hold_days'].mean()
            avg_ker = r_df['ker'].mean()
            print(f"  {reason:<20} {len(r_df):>6} {win_pct:>5.1f}% {avg_pnl:>+9.1f}% {avg_hold:>9.1f}d {avg_ker:>8.3f}")

    # === 5. VOLUMEN: ¿el volumen es fiable en crypto? ===
    print(f"\n{'─'*80}")
    print(f"  5. DISTRIBUCIÓN VOLUME RATIO POR CATEGORÍA Y RESULTADO")
    print(f"{'─'*80}")

    for cat in ['CRYPTO', 'FUTURES']:
        cat_df = df[df['category'] == cat]
        print(f"\n  {cat}:")
        bins = [(1.3, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 100.0)]
        print(f"  {'Vol Range':<14} {'Total':>6} {'Wins':>6} {'Win%':>6} {'Avg P&L%':>10} {'Avg KER':>8}")
        print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*6} {'─'*10} {'─'*8}")

        for low, high in bins:
            in_bin = cat_df[(cat_df['vol_ratio'] >= low) & (cat_df['vol_ratio'] < high)]
            if len(in_bin) == 0:
                continue
            n_wins = (in_bin['winner'] == True).sum()
            win_pct = n_wins / len(in_bin) * 100
            avg_pnl = in_bin['pnl_pct'].mean()
            avg_ker = in_bin['ker'].mean()
            print(f"  {low:.1f} - {high:.1f}    {len(in_bin):>6} {n_wins:>6} {win_pct:>5.1f}% {avg_pnl:>+9.1f}% {avg_ker:>8.3f}")

    # === 6. ATR%: ¿crypto demasiado volátil? ===
    print(f"\n{'─'*80}")
    print(f"  6. DISTRIBUCIÓN ATR% POR CATEGORÍA")
    print(f"{'─'*80}")

    for cat in ['CRYPTO', 'FUTURES']:
        cat_df = df[df['category'] == cat]
        print(f"\n  {cat}:")

        pcts = [0, 25, 50, 75, 90, 100]
        atr_pcts = np.percentile(cat_df['atr_pct'].dropna(), pcts)
        print(f"  Percentiles ATR%: ", end="")
        for p, v in zip(pcts, atr_pcts):
            print(f"P{p}={v:.1f}%  ", end="")
        print()

        bins = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 50)]
        print(f"  {'ATR% Range':<14} {'Total':>6} {'Wins':>6} {'Win%':>6} {'Avg P&L%':>10}")
        print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*6} {'─'*10}")

        for low, high in bins:
            in_bin = cat_df[(cat_df['atr_pct'] >= low) & (cat_df['atr_pct'] < high)]
            if len(in_bin) == 0:
                continue
            n_wins = (in_bin['winner'] == True).sum()
            win_pct = n_wins / len(in_bin) * 100
            avg_pnl = in_bin['pnl_pct'].mean()
            print(f"  {low:.0f}% - {high:.0f}%     {len(in_bin):>6} {n_wins:>6} {win_pct:>5.1f}% {avg_pnl:>+9.1f}%")

    # === 7. TOP/BOTTOM TICKERS ===
    print(f"\n{'─'*80}")
    print(f"  7. RESUMEN POR TICKER (ordenado por Win%)")
    print(f"{'─'*80}")

    ticker_stats = df.groupby(['ticker', 'category']).agg(
        trades=('pnl_euros', 'count'),
        wins=('winner', 'sum'),
        avg_ker=('ker', 'mean'),
        avg_rsi=('rsi', 'mean'),
        avg_vol=('vol_ratio', 'mean'),
        avg_atr_pct=('atr_pct', 'mean'),
        avg_fwd20=('fwd_return_20b', 'mean'),
        avg_pnl_pct=('pnl_pct', 'mean'),
        total_pnl=('pnl_euros', 'sum'),
    ).reset_index()
    ticker_stats['win_pct'] = ticker_stats['wins'] / ticker_stats['trades'] * 100
    ticker_stats = ticker_stats.sort_values('win_pct', ascending=False)

    print(f"\n  {'Ticker':<14} {'Cat':<8} {'Trades':>6} {'Win%':>6} {'KER':>6} {'RSI':>6} {'Vol':>6} "
          f"{'ATR%':>6} {'Fwd20b':>8} {'Avg P&L':>8} {'Tot P&L':>8}")
    print(f"  {'─'*14} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

    for _, row in ticker_stats.iterrows():
        print(f"  {row['ticker']:<14} {row['category']:<8} {row['trades']:>6.0f} {row['win_pct']:>5.1f}% "
              f"{row['avg_ker']:>6.3f} {row['avg_rsi']:>6.1f} {row['avg_vol']:>6.2f} "
              f"{row['avg_atr_pct']:>5.1f}% {row['avg_fwd20']:>+7.1f}% {row['avg_pnl_pct']:>+7.1f}% "
              f"€{row['total_pnl']:>+7.0f}")


if __name__ == '__main__':
    print("Análisis de señales Outsiders — 60 meses, 10 posiciones (spot-only)")
    print("Objetivo: entender por qué crypto falla vs futures\n")

    diagnostics = analyze_signals(months=60, max_positions=10)
    print_analysis(diagnostics)
