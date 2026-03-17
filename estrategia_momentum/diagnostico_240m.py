#!/usr/bin/env python3
"""
DIAGNÓSTICO: ¿Por qué la rentabilidad se degrada de 48-60m a 240m?

Análisis forense por ventanas temporales de 5 años.
Ejecuta el MISMO backtest v7+ pero registra cada trade con timestamp
para luego desglosar métricas por período.

Además detecta problemas de calidad de datos yfinance:
  - Tickers con datos parciales (no cubren 20 años)
  - Saltos de volumen absurdos (>10x de un día para otro)
  - Gaps de precio inexplicables
  - Splits posiblemente mal ajustados

Uso:
  python3 diagnostico_240m.py
  python3 diagnostico_240m.py --months 240
  python3 diagnostico_240m.py --months 240 --data-quality   # solo calidad de datos
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_breakout import MomentumEngine, calculate_atr, ASSETS

# Reutilizar config y clases del backtest principal
from backtest_experimental import (
    CONFIG, OPTIONS_ELIGIBLE, BASE_TICKERS,
    black_scholes_call, historical_volatility, monthly_expiration_dte,
    iv_rank, Trade, OptionTradeV2, EquityTracker,
    generate_all_signals, build_macro_filter, rank_candidates, find_candidates,
)


# =============================================================================
# PARTE 1: ANÁLISIS DE CALIDAD DE DATOS yfinance
# =============================================================================

def analyze_data_quality(all_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Detecta problemas en los datos de yfinance para cada ticker.

    Checks:
    1. Cobertura temporal (¿desde cuándo hay datos?)
    2. Gaps de precio (>20% de un día a otro sin split)
    3. Saltos de volumen (>10x o caída a 0)
    4. Días sin datos (huecos en el calendario)
    5. Precios negativos o cero
    6. ATR absurdo (>50% del precio)
    """
    quality_report = {}

    for ticker, df in all_data.items():
        issues = []

        # 1. Cobertura temporal
        first_date = df.index[0]
        last_date = df.index[-1]
        total_days = (last_date - first_date).days
        trading_days = len(df)
        expected_trading_days = total_days * 252 / 365
        coverage_ratio = trading_days / expected_trading_days if expected_trading_days > 0 else 0

        # 2. Gaps de precio diarios
        daily_returns = df['Close'].pct_change().dropna()
        extreme_gaps = daily_returns[daily_returns.abs() > 0.20]

        # 3. Saltos de volumen
        vol_ratio = df['Volume'].pct_change().dropna()
        vol_spikes = vol_ratio[vol_ratio.abs() > 9.0]  # >10x cambio
        zero_vol_days = (df['Volume'] == 0).sum()

        # 4. Precios sospechosos
        zero_prices = (df['Close'] <= 0).sum()

        # 5. ATR absurdo
        if 'ATR' in df.columns:
            atr_pct = (df['ATR'] / df['Close']).dropna()
            extreme_atr = (atr_pct > 0.50).sum()  # ATR > 50% del precio
        else:
            extreme_atr = 0

        # 6. Volumen medio por período (detectar cambios estructurales)
        vol_by_year = {}
        for year in range(first_date.year, last_date.year + 1):
            year_data = df[df.index.year == year]
            if len(year_data) > 20:
                vol_by_year[year] = year_data['Volume'].mean()

        # Detectar si el volumen histórico es <<< volumen reciente
        if vol_by_year:
            years_sorted = sorted(vol_by_year.keys())
            if len(years_sorted) >= 4:
                early_vol = np.mean([vol_by_year[y] for y in years_sorted[:2]])
                recent_vol = np.mean([vol_by_year[y] for y in years_sorted[-2:]])
                if early_vol > 0 and recent_vol / early_vol > 20:
                    issues.append(f"VOL_RATIO_HISTORICO: vol reciente {recent_vol/early_vol:.0f}x vs primeros años")
                if early_vol > 0 and recent_vol / early_vol < 0.05:
                    issues.append(f"VOL_COLAPSO: vol reciente solo {recent_vol/early_vol:.1%} vs primeros años")

        if len(extreme_gaps) > 0:
            worst_gap = extreme_gaps.abs().max()
            worst_gap_date = extreme_gaps.abs().idxmax()
            issues.append(f"PRICE_GAP: {len(extreme_gaps)} gaps >20%, peor {worst_gap:.1%} en {worst_gap_date.strftime('%Y-%m-%d')}")

        if zero_vol_days > trading_days * 0.05:
            issues.append(f"ZERO_VOL: {zero_vol_days} días con volumen=0 ({zero_vol_days/trading_days*100:.1f}%)")

        if len(vol_spikes) > 10:
            issues.append(f"VOL_SPIKES: {len(vol_spikes)} saltos de volumen >10x")

        if zero_prices > 0:
            issues.append(f"ZERO_PRICE: {zero_prices} días con precio <= 0")

        if extreme_atr > 0:
            issues.append(f"EXTREME_ATR: {extreme_atr} días con ATR > 50% del precio")

        if coverage_ratio < 0.80:
            issues.append(f"LOW_COVERAGE: solo {coverage_ratio:.1%} de días esperados")

        quality_report[ticker] = {
            'first_date': first_date,
            'last_date': last_date,
            'trading_days': trading_days,
            'total_calendar_days': total_days,
            'issues': issues,
            'n_issues': len(issues),
            'n_extreme_gaps': len(extreme_gaps),
            'zero_vol_days': zero_vol_days,
            'vol_by_year': vol_by_year,
        }

    return quality_report


def print_data_quality_report(quality_report: Dict):
    """Imprime reporte de calidad de datos."""
    print(f"""
{'='*90}
  ANÁLISIS DE CALIDAD DE DATOS yfinance
{'='*90}
""")

    # Clasificar tickers por cobertura
    full_coverage = []    # >= 18 años
    partial = []          # 5-18 años
    short_only = []       # < 5 años

    for ticker, info in quality_report.items():
        years = info['total_calendar_days'] / 365.25
        if years >= 18:
            full_coverage.append((ticker, years, info))
        elif years >= 5:
            partial.append((ticker, years, info))
        else:
            short_only.append((ticker, years, info))

    print(f"  COBERTURA TEMPORAL:")
    print(f"    Cobertura completa (≥18 años): {len(full_coverage)} tickers")
    print(f"    Parcial (5-18 años):           {len(partial)} tickers")
    print(f"    Solo reciente (<5 años):       {len(short_only)} tickers")

    if partial:
        print(f"\n  TICKERS PARCIALES (contribuyen SOLO en períodos recientes):")
        for ticker, years, info in sorted(partial, key=lambda x: x[1]):
            print(f"    {ticker:10} desde {info['first_date'].strftime('%Y-%m-%d')} ({years:.1f} años)")

    if short_only:
        print(f"\n  TICKERS MUY CORTOS (<5 años, NO contribuyen en ventanas antiguas):")
        for ticker, years, info in sorted(short_only, key=lambda x: x[1]):
            print(f"    {ticker:10} desde {info['first_date'].strftime('%Y-%m-%d')} ({years:.1f} años)")

    # Tickers con problemas de datos
    problematic = [(t, info) for t, info in quality_report.items() if info['n_issues'] > 0]
    problematic.sort(key=lambda x: x[1]['n_issues'], reverse=True)

    if problematic:
        print(f"\n  TICKERS CON PROBLEMAS DE DATOS ({len(problematic)}):")
        print(f"  {'Ticker':<12} {'Issues':<6} {'Detalles'}")
        print(f"  {'-'*80}")
        for ticker, info in problematic[:30]:
            for i, issue in enumerate(info['issues']):
                prefix = f"  {ticker:<12} {info['n_issues']:<6}" if i == 0 else f"  {'':<12} {'':<6}"
                print(f"{prefix} {issue}")

    # Resumen de volumen por período para top tickers
    print(f"\n  EVOLUCIÓN DE VOLUMEN MEDIO (millones) — TOP 20 tickers US:")
    us_tickers = [t for t in quality_report if not any(x in t for x in ['.DE', '.PA', '.MC', '.MI', '.AS', '.BR'])]
    sample_tickers = us_tickers[:20]

    # Encontrar rango de años común
    all_years = set()
    for t in sample_tickers:
        all_years.update(quality_report[t]['vol_by_year'].keys())
    if all_years:
        year_range = sorted(all_years)
        # Agrupar en ventanas de 5 años
        windows = []
        for start_y in range(min(year_range), max(year_range), 5):
            end_y = min(start_y + 4, max(year_range))
            windows.append((start_y, end_y))

        header = f"  {'Ticker':<10}"
        for s, e in windows:
            header += f" {s}-{e}  "
        print(header)
        print(f"  {'-'*len(header)}")

        for ticker in sample_tickers:
            vol_data = quality_report[ticker]['vol_by_year']
            row = f"  {ticker:<10}"
            for s, e in windows:
                vals = [vol_data.get(y, 0) for y in range(s, e + 1) if y in vol_data]
                if vals:
                    avg = np.mean(vals) / 1e6
                    row += f" {avg:>7.1f}M "
                else:
                    row += f"     N/A  "
            print(row)


# =============================================================================
# PARTE 2: BACKTEST CON TRACKING POR VENTANA TEMPORAL
# =============================================================================

def run_diagnostic_backtest(months, verbose=False):
    """
    Ejecuta el backtest v7+ completo pero guarda cada trade con su
    fecha de entrada/salida para análisis por ventana temporal.
    """
    tickers = BASE_TICKERS
    use_options = True

    print(f"\n{'='*90}")
    print(f"  DIAGNÓSTICO: BACKTEST v7+ — {months} MESES — {len(tickers)} tickers")
    print(f"{'='*90}")

    # Descargar datos
    print("  Descargando datos...")
    all_data = {}
    failed = []
    for i, ticker in enumerate(tickers):
        try:
            if months > 60:
                end = datetime.now()
                start = end - timedelta(days=months * 30)
                df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                                 end=end.strftime('%Y-%m-%d'), interval='1d', progress=False)
            else:
                df = yf.download(ticker, period=f'{months}mo', interval='1d', progress=False)
            if df.empty:
                failed.append(ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= 50:
                df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
                df['HVOL'] = historical_volatility(df['Close'], CONFIG['hvol_window'])
                all_data[ticker] = df
            else:
                failed.append(ticker)
        except Exception:
            failed.append(ticker)
        if (i + 1) % 20 == 0 or i == len(tickers) - 1:
            print(f"\r  Descargados: {len(all_data)}/{len(tickers)} OK, {len(failed)} fallidos", end='')

    print(f"\n  Tickers con datos: {len(all_data)}")
    if failed:
        print(f"  Fallidos: {', '.join(failed[:15])}{'...' if len(failed) > 15 else ''}")

    if not all_data:
        return None, None

    # =========================================================================
    # ANÁLISIS DE CALIDAD DE DATOS
    # =========================================================================
    quality_report = analyze_data_quality(all_data)
    print_data_quality_report(quality_report)

    # =========================================================================
    # BACKTEST v7+ (copia exacta del backtest_experimental.py)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  EJECUTANDO BACKTEST v7+...")
    print(f"{'='*90}")

    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG['rsi_max'],
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG['longs_only']
    )
    signals_data, total_signals = generate_all_signals(all_data, engine)
    print(f"  Señales LONG totales: {total_signals}")

    macro_bullish = build_macro_filter(all_data)
    all_dates = sorted(set(d for sd in signals_data.values() for d in sd['df'].index.tolist()))

    tracker = EquityTracker(CONFIG['initial_capital'])
    active_trades = {}
    active_options = {}
    all_trades = []
    all_option_trades = []

    # Equity curve diaria para análisis por ventana
    daily_equity = []

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

        # 2. Gestionar opciones activas
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

        # 3. Buscar nuevas señales
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
                    actual_dte = monthly_expiration_dte(current_date, CONFIG['option_dte'])
                    T = actual_dte / 365.0
                    iv = df['HVOL'].iloc[idx]
                    if pd.isna(iv) or iv <= 0:
                        iv = 0.30
                    bs = black_scholes_call(stock_price, strike, T, CONFIG['risk_free_rate'], iv)
                    option_price = bs['price']
                    option_price *= (1 + CONFIG['option_spread_pct'] / 100 / 2)
                    size = tracker.get_option_size(option_price)
                    if size['premium'] < 50:
                        continue

                    opt = OptionTradeV2(
                        ticker=ticker, entry_date=current_date,
                        entry_stock_price=stock_price, strike=strike,
                        dte_at_entry=actual_dte, entry_option_price=option_price,
                        entry_iv=iv, num_contracts=size['contracts'],
                        position_euros=size['premium'],
                    )
                    active_options[ticker] = opt
                    tracker.open_positions += 1
                    tracker.open_options += 1
                else:
                    size_info = tracker.get_position_size(ticker, prev_atr, bar['Open'], False)
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

        # Registrar equity diario
        daily_equity.append((current_date, tracker.equity, len(active_trades) + len(active_options)))

    # Cerrar trades abiertos al final
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

    return {
        'all_trades': all_trades,
        'all_option_trades': all_option_trades,
        'daily_equity': daily_equity,
        'tracker': tracker,
        'quality_report': quality_report,
        'all_data': all_data,
        'signals_data': signals_data,
    }, quality_report


# =============================================================================
# PARTE 3: ANÁLISIS POR VENTANA TEMPORAL
# =============================================================================

def analyze_by_window(results: Dict, window_years: int = 5):
    """
    Desglosa métricas por ventanas temporales.
    """
    all_trades = results['all_trades']
    all_option_trades = results['all_option_trades']
    daily_equity = results['daily_equity']
    combined = all_trades + all_option_trades

    if not combined:
        print("  Sin trades para analizar.")
        return

    # Determinar rango temporal
    min_date = min(t.entry_date for t in combined)
    max_date = max(t.exit_date for t in combined if t.exit_date)

    # Crear ventanas
    windows = []
    start_year = min_date.year
    end_year = max_date.year

    for y in range(start_year, end_year + 1, window_years):
        w_start = datetime(y, 1, 1)
        w_end = datetime(min(y + window_years - 1, end_year), 12, 31)
        windows.append((w_start, w_end, f"{y}-{min(y + window_years - 1, end_year)}"))

    # También análisis por año individual
    years = list(range(start_year, end_year + 1))

    # =========================================================================
    # MÉTRICAS POR VENTANA DE N AÑOS
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  MÉTRICAS POR VENTANA DE {window_years} AÑOS")
    print(f"{'='*90}")

    print(f"\n  {'Ventana':<12} {'Trades':<8} {'Win%':<7} {'PF':<7} {'P&L EUR':<12} "
          f"{'AvgWin%':<9} {'AvgLoss%':<9} {'Stocks':<8} {'Opts':<6} {'MaxR':<6}")
    print(f"  {'-'*95}")

    for w_start, w_end, label in windows:
        # Trades cuya ENTRADA cae en esta ventana
        w_trades = [t for t in combined
                    if t.entry_date >= w_start and t.entry_date <= w_end]

        if not w_trades:
            print(f"  {label:<12} {'N/A':>8}")
            continue

        w_stocks = [t for t in w_trades if isinstance(t, Trade)]
        w_opts = [t for t in w_trades if isinstance(t, OptionTradeV2)]
        winners = [t for t in w_trades if t.pnl_euros > 0]
        losers = [t for t in w_trades if t.pnl_euros <= 0]

        total_pnl = sum(t.pnl_euros for t in w_trades)
        win_rate = len(winners) / len(w_trades) * 100

        gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
        pf = gross_profit / gross_loss

        avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0

        max_r = max(t.max_r_mult for t in w_trades)

        print(f"  {label:<12} {len(w_trades):<8} {win_rate:<7.1f} {pf:<7.2f} "
              f"EUR{total_pnl:>+9,.0f}  {avg_win:>+7.1f}%  {avg_loss:>+7.1f}%  "
              f"{len(w_stocks):<8} {len(w_opts):<6} {max_r:<6.1f}")

    # =========================================================================
    # MÉTRICAS POR AÑO INDIVIDUAL
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  MÉTRICAS POR AÑO INDIVIDUAL")
    print(f"{'='*90}")

    print(f"\n  {'Año':<6} {'Trades':<8} {'Win%':<7} {'PF':<7} {'P&L EUR':<12} "
          f"{'AvgWin%':<9} {'AvgLoss%':<9} {'Peor Trade':<20} {'Mejor Trade':<20}")
    print(f"  {'-'*110}")

    yearly_data = []
    for year in years:
        y_trades = [t for t in combined
                    if t.entry_date.year == year]

        if not y_trades:
            continue

        winners = [t for t in y_trades if t.pnl_euros > 0]
        losers = [t for t in y_trades if t.pnl_euros <= 0]

        total_pnl = sum(t.pnl_euros for t in y_trades)
        win_rate = len(winners) / len(y_trades) * 100

        gross_profit = sum(t.pnl_euros for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_euros for t in losers)) if losers else 0.01
        pf = gross_profit / gross_loss

        avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0

        worst = min(y_trades, key=lambda t: t.pnl_euros)
        best = max(y_trades, key=lambda t: t.pnl_euros)
        worst_str = f"{worst.ticker} EUR{worst.pnl_euros:+,.0f}"
        best_str = f"{best.ticker} EUR{best.pnl_euros:+,.0f}"

        yearly_data.append({
            'year': year, 'trades': len(y_trades), 'win_rate': win_rate,
            'pf': pf, 'pnl': total_pnl
        })

        print(f"  {year:<6} {len(y_trades):<8} {win_rate:<7.1f} {pf:<7.2f} "
              f"EUR{total_pnl:>+9,.0f}  {avg_win:>+7.1f}%  {avg_loss:>+7.1f}%  "
              f"{worst_str:<20} {best_str:<20}")

    # =========================================================================
    # TOP TICKERS DESTRUCTORES vs CONSTRUCTORES (acumulado 240m)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  TOP 15 TICKERS DESTRUCTORES DE VALOR (P&L acumulado)")
    print(f"{'='*90}")

    ticker_pnl = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0, 'first_year': 9999, 'last_year': 0})
    for t in combined:
        tk = t.ticker
        ticker_pnl[tk]['pnl'] += t.pnl_euros
        ticker_pnl[tk]['trades'] += 1
        if t.pnl_euros > 0:
            ticker_pnl[tk]['wins'] += 1
        ticker_pnl[tk]['first_year'] = min(ticker_pnl[tk]['first_year'], t.entry_date.year)
        ticker_pnl[tk]['last_year'] = max(ticker_pnl[tk]['last_year'], t.entry_date.year)

    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1]['pnl'])

    print(f"\n  {'Ticker':<10} {'P&L EUR':<12} {'Trades':<8} {'Win%':<7} {'Años activo':<15} {'Categoría'}")
    print(f"  {'-'*80}")

    for ticker, info in sorted_tickers[:15]:
        wr = info['wins'] / info['trades'] * 100 if info['trades'] > 0 else 0
        cat = ASSETS.get(ticker, {}).get('category', '?')
        years_str = f"{info['first_year']}-{info['last_year']}"
        print(f"  {ticker:<10} EUR{info['pnl']:>+9,.0f} {info['trades']:<8} {wr:<7.1f} {years_str:<15} {cat}")

    print(f"\n{'='*90}")
    print(f"  TOP 15 TICKERS CONSTRUCTORES DE VALOR")
    print(f"{'='*90}")

    print(f"\n  {'Ticker':<10} {'P&L EUR':<12} {'Trades':<8} {'Win%':<7} {'Años activo':<15} {'Categoría'}")
    print(f"  {'-'*80}")

    for ticker, info in reversed(sorted_tickers[-15:]):
        wr = info['wins'] / info['trades'] * 100 if info['trades'] > 0 else 0
        cat = ASSETS.get(ticker, {}).get('category', '?')
        years_str = f"{info['first_year']}-{info['last_year']}"
        print(f"  {ticker:<10} EUR{info['pnl']:>+9,.0f} {info['trades']:<8} {wr:<7.1f} {years_str:<15} {cat}")

    # =========================================================================
    # ANÁLISIS DE SEÑALES POR PERÍODO
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  NÚMERO DE SEÑALES GENERADAS POR AÑO Y CATEGORÍA")
    print(f"{'='*90}")

    signals_by_year_cat = defaultdict(lambda: defaultdict(int))
    for t in combined:
        year = t.entry_date.year
        cat = ASSETS.get(t.ticker, {}).get('category', '?')
        signals_by_year_cat[year][cat] += 1

    all_cats = sorted(set(cat for yd in signals_by_year_cat.values() for cat in yd))

    # Agrupar categorías para legibilidad
    cat_groups = {
        'US_TECH': 'US_TECH', 'US_FINANCE': 'US_FIN', 'US_HEALTH': 'US_HLTH',
        'US_CONSUMER': 'US_CONS', 'US_INDEX': 'US_IDX', 'US_INDEX_LEV': 'US_LEV',
    }

    print(f"\n  {'Año':<6}", end='')
    for cat in all_cats:
        short = cat_groups.get(cat, cat[:8])
        print(f" {short:>8}", end='')
    print(f" {'TOTAL':>8}")
    print(f"  {'-'*(6 + 9*len(all_cats) + 9)}")

    for year in sorted(signals_by_year_cat.keys()):
        print(f"  {year:<6}", end='')
        total = 0
        for cat in all_cats:
            n = signals_by_year_cat[year].get(cat, 0)
            total += n
            print(f" {n:>8}", end='')
        print(f" {total:>8}")

    # =========================================================================
    # EQUITY CURVE POR VENTANA
    # =========================================================================
    if daily_equity:
        print(f"\n{'='*90}")
        print(f"  EQUITY CURVE — SNAPSHOTS")
        print(f"{'='*90}\n")

        # Mostrar equity al inicio de cada año
        eq_df = pd.DataFrame(daily_equity, columns=['date', 'equity', 'positions'])
        eq_df['year'] = eq_df['date'].apply(lambda d: d.year)

        for year in sorted(eq_df['year'].unique()):
            year_data = eq_df[eq_df['year'] == year]
            if len(year_data) > 0:
                start_eq = year_data['equity'].iloc[0]
                end_eq = year_data['equity'].iloc[-1]
                peak = year_data['equity'].max()
                trough = year_data['equity'].min()
                dd = (peak - trough) / peak * 100 if peak > 0 else 0
                yoy_ret = (end_eq / start_eq - 1) * 100 if start_eq > 0 else 0
                avg_pos = year_data['positions'].mean()

                bar_len = max(0, int(yoy_ret / 5))  # escala: 5% = 1 char
                bar_neg = max(0, int(-yoy_ret / 5))
                bar = '█' * bar_len if yoy_ret >= 0 else '░' * bar_neg

                print(f"  {year} | EUR {start_eq:>10,.0f} → {end_eq:>10,.0f} | "
                      f"YoY {yoy_ret:>+7.1f}% | DD -{dd:>5.1f}% | "
                      f"Pos {avg_pos:.1f} | {bar}")

    # =========================================================================
    # DIAGNÓSTICO CRUZADO: calidad datos vs performance
    # =========================================================================
    quality_report = results['quality_report']

    print(f"\n{'='*90}")
    print(f"  CRUCE: CALIDAD DE DATOS vs PERFORMANCE POR TICKER")
    print(f"{'='*90}")
    print(f"\n  ¿Los tickers con peor calidad de datos son los destructores?")
    print(f"\n  {'Ticker':<10} {'P&L EUR':<12} {'Issues':<8} {'Datos desde':<14} {'Problemas principales'}")
    print(f"  {'-'*90}")

    # Mostrar los 15 peores tickers con su calidad de datos
    for ticker, info in sorted_tickers[:15]:
        q = quality_report.get(ticker, {})
        n_issues = q.get('n_issues', 0)
        first = q.get('first_date', None)
        first_str = first.strftime('%Y-%m-%d') if first else 'N/A'
        issues_str = '; '.join(q.get('issues', [])[:2]) if q.get('issues') else 'OK'
        if len(issues_str) > 50:
            issues_str = issues_str[:50] + '...'

        print(f"  {ticker:<10} EUR{info['pnl']:>+9,.0f} {n_issues:<8} {first_str:<14} {issues_str}")

    # =========================================================================
    # CONCLUSIONES AUTOMÁTICAS
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"  CONCLUSIONES PRELIMINARES")
    print(f"{'='*90}\n")

    # 1. ¿Cuántos tickers tienen cobertura completa?
    full_tickers = [t for t, q in quality_report.items()
                    if q['total_calendar_days'] / 365.25 >= 18]
    partial_tickers = [t for t, q in quality_report.items()
                       if 5 <= q['total_calendar_days'] / 365.25 < 18]

    print(f"  1. COBERTURA: {len(full_tickers)} tickers con ≥18 años, "
          f"{len(partial_tickers)} parciales, "
          f"{len(quality_report) - len(full_tickers) - len(partial_tickers)} cortos")

    # 2. ¿Los años malos coinciden con problemas de datos?
    bad_years = [y for y in yearly_data if y['pf'] < 1.0]
    good_years = [y for y in yearly_data if y['pf'] >= 1.5]
    print(f"  2. AÑOS MALOS (PF<1): {', '.join(str(y['year']) for y in bad_years) if bad_years else 'Ninguno'}")
    print(f"     AÑOS BUENOS (PF≥1.5): {', '.join(str(y['year']) for y in good_years) if good_years else 'Ninguno'}")

    # 3. Contribución de tickers con datos problemáticos
    problematic_tickers = [t for t, q in quality_report.items() if q['n_issues'] >= 2]
    prob_pnl = sum(ticker_pnl[t]['pnl'] for t in problematic_tickers if t in ticker_pnl)
    clean_pnl = sum(info['pnl'] for t, info in ticker_pnl.items() if t not in problematic_tickers)
    print(f"  3. TICKERS PROBLEMÁTICOS ({len(problematic_tickers)}): P&L acumulado EUR {prob_pnl:+,.0f}")
    print(f"     TICKERS LIMPIOS ({len(quality_report) - len(problematic_tickers)}): P&L acumulado EUR {clean_pnl:+,.0f}")

    # 4. ¿El universo activo es muy diferente entre ventanas?
    first_window_tickers = set(t.ticker for t in combined if t.entry_date.year <= start_year + 4)
    last_window_tickers = set(t.ticker for t in combined if t.entry_date.year >= end_year - 4)
    overlap = first_window_tickers & last_window_tickers
    only_early = first_window_tickers - last_window_tickers
    only_late = last_window_tickers - first_window_tickers

    print(f"  4. UNIVERSO ACTIVO:")
    print(f"     Primera ventana: {len(first_window_tickers)} tickers")
    print(f"     Última ventana:  {len(last_window_tickers)} tickers")
    print(f"     En ambas:        {len(overlap)} tickers")
    print(f"     Solo al inicio:  {len(only_early)} — {', '.join(sorted(only_early)[:10])}")
    print(f"     Solo al final:   {len(only_late)} — {', '.join(sorted(only_late)[:10])}")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnóstico: degradación 48-60m vs 240m')
    parser.add_argument('--months', type=int, default=240, help='Meses de histórico (default 240)')
    parser.add_argument('--window', type=int, default=5, help='Tamaño ventana en años (default 5)')
    parser.add_argument('--data-quality', action='store_true', help='Solo análisis de calidad de datos')
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DIAGNÓSTICO: ¿Por qué la rentabilidad se degrada a 240 meses?            ║
║                                                                              ║
║  Hipótesis bajo test:                                                        ║
║    H1: Calidad de datos yfinance (splits, volumen, gaps)                     ║
║    H2: Régimen de mercado 2006-2012 (crisis, whipsaws)                       ║
║    H3: Universo de tickers diferente (survivorship bias)                     ║
║    H4: Filtro macro SPY>SMA50 menos eficaz en bears largos                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    if args.data_quality:
        # Solo calidad de datos (rápido, sin backtest)
        print("  Modo: Solo análisis de calidad de datos (sin backtest)")
        print("  Descargando datos...")
        all_data = {}
        for i, ticker in enumerate(BASE_TICKERS):
            try:
                end = datetime.now()
                start = end - timedelta(days=args.months * 30)
                df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                                 end=end.strftime('%Y-%m-%d'), interval='1d', progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if len(df) >= 50:
                        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
                        all_data[ticker] = df
            except Exception:
                pass
            if (i + 1) % 20 == 0 or i == len(BASE_TICKERS) - 1:
                print(f"\r  Descargados: {len(all_data)}/{len(BASE_TICKERS)}", end='')
        print()

        quality_report = analyze_data_quality(all_data)
        print_data_quality_report(quality_report)
    else:
        # Backtest completo + diagnóstico
        results, quality_report = run_diagnostic_backtest(args.months, verbose=False)
        if results:
            analyze_by_window(results, window_years=args.window)


if __name__ == "__main__":
    main()
