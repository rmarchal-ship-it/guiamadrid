#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║              BACKTEST RADAR V7.1 — INTRADÍA                       ║
║              MOMENTUM + DIP BUY                                   ║
╚═══════════════════════════════════════════════════════════════════╝

Backtests:
- 3 meses (datos 1H)
- 6 meses (datos 1H)
- 12 meses (datos 1H, límite yfinance ~730 días)
- 24 meses (datos diarios como proxy)

Estrategias:
1. MOMENTUM: Comprar cuando sube >2% en 4 horas
2. DIP BUY: Comprar cuando cae >3% en 4 horas (mean reversion)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    'momentum_threshold': 2.0,   # % para señal momentum
    'dip_threshold': -3.0,       # % para señal dip buy
    'stop_loss': -2.0,           # % stop loss
    'take_profit': 3.0,          # % take profit
    'max_hold_hours': 8,         # Máximo horas en trade
}

ASSETS = {
    'QQQ': {'name': 'Nasdaq-100', 'future': '/MNQ'},
    'TQQQ': {'name': 'Nasdaq 3x', 'future': '/MNQ'},
    'SPY': {'name': 'S&P 500', 'future': '/MES'},
    'SPXL': {'name': 'S&P 3x', 'future': '/MES'},
    'TNA': {'name': 'Russell 3x', 'future': '/M2K'},
    'GLD': {'name': 'Gold', 'future': '/MGC'},
    'SLV': {'name': 'Silver', 'future': '/SIL'},
    'UNG': {'name': 'Natural Gas', 'future': '/NG'},
    'TSLA': {'name': 'Tesla', 'future': None},
    'SMCI': {'name': 'Super Micro', 'future': None},
    'UPST': {'name': 'Upstart', 'future': None},
    'GME': {'name': 'GameStop', 'future': None},
}

TICKERS = list(ASSETS.keys())

# ═══════════════════════════════════════════════════════════════════
# FUNCIONES BASE
# ═══════════════════════════════════════════════════════════════════

def download_data(ticker, period='6mo', interval='1h'):
    """Descarga datos con fix para MultiIndex"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

# ═══════════════════════════════════════════════════════════════════
# BACKTEST INTRADÍA
# ═══════════════════════════════════════════════════════════════════

def backtest_intraday(months=3, verbose=True):
    """
    Backtest de estrategia intradía
    
    MOMENTUM: Comprar cuando sube >2% en 4 horas
    DIP BUY: Comprar cuando cae >3% en 4 horas
    Exit: Stop -2%, Target +3%, o máximo 8 horas
    """
    
    # Determinar periodo e intervalo
    if months <= 6:
        period = f'{months}mo'
        interval = '1h'
        bars_4h = 4  # 4 barras de 1h = 4 horas
    elif months <= 24:
        period = f'{months}mo'
        interval = '1h'
        bars_4h = 4
    else:
        period = f'{months}mo'
        interval = '1d'
        bars_4h = 1  # 1 día como proxy
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📈 BACKTEST RADAR V7.1 — {months} MESES")
        print(f"{'='*70}")
        print(f"Datos: {interval} | Lookback: {bars_4h} barras")
        print(f"Momentum: >{CONFIG['momentum_threshold']}% | Dip: <{CONFIG['dip_threshold']}%")
        print(f"Stop: {CONFIG['stop_loss']}% | Target: {CONFIG['take_profit']}%\n")
    
    all_trades = []
    
    for ticker in TICKERS:
        data = download_data(ticker, period=period, interval=interval)
        
        if data is None or len(data) < 50:
            continue
        
        asset = ASSETS[ticker]
        
        # Calcular cambio en últimas 4 barras
        data['Change_4h'] = data['Close'].pct_change(periods=bars_4h) * 100
        
        if verbose:
            signals_up = len(data[data['Change_4h'] > CONFIG['momentum_threshold']])
            signals_down = len(data[data['Change_4h'] < CONFIG['dip_threshold']])
            print(f"✅ {ticker:5}: {len(data)} barras | Mom: {signals_up}, Dip: {signals_down}")
        
        # Generar trades
        i = bars_4h + 1
        while i < len(data) - CONFIG['max_hold_hours']:
            change = data['Change_4h'].iloc[i]
            
            if pd.isna(change):
                i += 1
                continue
            
            entry_price = data['Close'].iloc[i]
            entry_time = data.index[i]
            strategy = None
            
            # MOMENTUM: Subió >2%
            if change > CONFIG['momentum_threshold']:
                strategy = 'MOMENTUM'
            # DIP BUY: Cayó >3%
            elif change < CONFIG['dip_threshold']:
                strategy = 'DIP_BUY'
            
            if strategy:
                # Simular trade
                stop = entry_price * (1 + CONFIG['stop_loss']/100)
                target = entry_price * (1 + CONFIG['take_profit']/100)
                
                exit_price = None
                exit_reason = None
                exit_idx = i
                
                for j in range(i + 1, min(i + CONFIG['max_hold_hours'] + 1, len(data))):
                    bar = data.iloc[j]
                    
                    # Check stop
                    if bar['Low'] <= stop:
                        exit_price = stop
                        exit_reason = 'STOP'
                        exit_idx = j
                        break
                    # Check target
                    elif bar['High'] >= target:
                        exit_price = target
                        exit_reason = 'TARGET'
                        exit_idx = j
                        break
                
                # Time exit
                if exit_price is None:
                    exit_idx = min(i + CONFIG['max_hold_hours'], len(data) - 1)
                    exit_price = data['Close'].iloc[exit_idx]
                    exit_reason = 'TIME'
                
                pnl = (exit_price - entry_price) / entry_price * 100
                
                all_trades.append({
                    'Ticker': ticker,
                    'Strategy': strategy,
                    'Date': entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(entry_time, 'strftime') else str(entry_time)[:16],
                    'Entry': round(entry_price, 2),
                    'Exit': round(exit_price, 2),
                    'P&L%': round(pnl, 2),
                    'Result': exit_reason,
                    'Bars': exit_idx - i,
                })
                
                i = exit_idx + 1  # Skip barras del trade
            else:
                i += 1
    
    # ══════════════════════════════════════════════════════════════
    # RESULTADOS
    # ══════════════════════════════════════════════════════════════
    
    if all_trades:
        df = pd.DataFrame(all_trades)
        
        total = len(df)
        wins = len(df[df['P&L%'] > 0])
        losses = total - wins
        win_rate = wins / total * 100 if total > 0 else 0
        
        avg_win = df[df['P&L%'] > 0]['P&L%'].mean() if wins > 0 else 0
        avg_loss = df[df['P&L%'] <= 0]['P&L%'].mean() if losses > 0 else 0
        
        total_return = df['P&L%'].sum()
        avg_return = df['P&L%'].mean()
        
        gross_profit = df[df['P&L%'] > 0]['P&L%'].sum() if wins > 0 else 0
        gross_loss = abs(df[df['P&L%'] <= 0]['P&L%'].sum()) if losses > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
        
        # Por estrategia
        momentum = df[df['Strategy'] == 'MOMENTUM']
        dip_buy = df[df['Strategy'] == 'DIP_BUY']
        
        # Annual return
        annual_return = (total_return / months) * 12
        
        results = {
            'months': months,
            'trades': total,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annual_return': annual_return,
            'avg_trade': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'momentum_trades': len(momentum),
            'momentum_return': momentum['P&L%'].sum() if len(momentum) > 0 else 0,
            'dip_trades': len(dip_buy),
            'dip_return': dip_buy['P&L%'].sum() if len(dip_buy) > 0 else 0,
        }
        
        if verbose:
            print(f"""
{'='*70}
📊 RESULTADOS — {months} MESES
{'='*70}

╔══════════════════════════════════════════════════════════════════╗
║  TRADES: {total:<55}║
║  WIN RATE: {win_rate:.1f}% ({wins}W / {losses}L)                            ║
║  PROFIT FACTOR: {profit_factor:.2f}                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  RETURN TOTAL: {total_return:.1f}%                                        ║
║  RETURN ANUALIZADO: {annual_return:.1f}%                                   ║
║  AVG POR TRADE: {avg_return:.2f}%                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  MOMENTUM: {len(momentum)} trades → {momentum['P&L%'].sum():.1f}%                           ║
║  DIP BUY: {len(dip_buy)} trades → {dip_buy['P&L%'].sum():.1f}%                            ║
╚══════════════════════════════════════════════════════════════════╝
""")
            
            # Por resultado
            stops = len(df[df['Result'] == 'STOP'])
            targets = len(df[df['Result'] == 'TARGET'])
            time_exits = len(df[df['Result'] == 'TIME'])
            
            print(f"Salidas: STOP {stops} ({stops/total*100:.0f}%) | TARGET {targets} ({targets/total*100:.0f}%) | TIME {time_exits} ({time_exits/total*100:.0f}%)")
            
            # Por ticker top 5
            print(f"\n📊 TOP 5 TICKERS:")
            by_ticker = df.groupby('Ticker')['P&L%'].agg(['sum', 'count']).round(2)
            by_ticker.columns = ['Total%', 'Trades']
            print(by_ticker.sort_values('Total%', ascending=False).head(5).to_string())
        
        return results, df
    else:
        if verbose:
            print("❌ No se generaron trades")
        return None, None

# ═══════════════════════════════════════════════════════════════════
# COMPARACIÓN MULTI-PERIODO
# ═══════════════════════════════════════════════════════════════════

def compare_periods():
    """Compara backtest a diferentes periodos"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           📊 BACKTEST MULTI-PERIODO RADAR V7.1 📊                 ║
║           MOMENTUM + DIP BUY                                      ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    periods = [3, 6]  # yfinance limita datos 1H a ~730 días
    
    # Nota: Para 12 y 24 meses usamos datos diarios
    results_all = []
    
    # Backtests con datos 1H (hasta 6 meses con buena calidad)
    for months in periods:
        print(f"\n{'#'*70}")
        print(f"# PERIODO: {months} MESES")
        print(f"{'#'*70}")
        
        results, df = backtest_intraday(months=months, verbose=True)
        if results:
            results_all.append(results)
    
    # Backtest con datos diarios (12 y 24 meses)
    print(f"\n{'#'*70}")
    print(f"# PERIODO: 12 MESES (datos diarios)")
    print(f"{'#'*70}")
    results_12, df_12 = backtest_daily(months=12, verbose=True)
    if results_12:
        results_all.append(results_12)
    
    print(f"\n{'#'*70}")
    print(f"# PERIODO: 24 MESES (datos diarios)")
    print(f"{'#'*70}")
    results_24, df_24 = backtest_daily(months=24, verbose=True)
    if results_24:
        results_all.append(results_24)
    
    # ══════════════════════════════════════════════════════════════
    # RESUMEN COMPARATIVO
    # ══════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("📊 COMPARACIÓN FINAL — TODOS LOS PERIODOS")
    print("="*70 + "\n")
    
    if results_all:
        summary = pd.DataFrame(results_all)
        summary = summary[['months', 'trades', 'win_rate', 'profit_factor', 'total_return', 'annual_return', 'momentum_return', 'dip_return']]
        summary.columns = ['Meses', 'Trades', 'WinRate%', 'PF', 'Return%', 'Annual%', 'Momentum%', 'DipBuy%']
        summary = summary.round(2)
        
        print(summary.to_string(index=False))
        
        print(f"""

╔══════════════════════════════════════════════════════════════════╗
║  📋 CONCLUSIONES                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Win Rate promedio: {summary['WinRate%'].mean():.1f}%                                 ║
║  Profit Factor promedio: {summary['PF'].mean():.2f}                               ║
║  Return anualizado promedio: {summary['Annual%'].mean():.1f}%                        ║
║                                                                  ║
║  MOMENTUM vs DIP BUY:                                            ║
║    Momentum total: {summary['Momentum%'].sum():.1f}%                                 ║
║    Dip Buy total: {summary['DipBuy%'].sum():.1f}%                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    return results_all

# ═══════════════════════════════════════════════════════════════════
# BACKTEST CON DATOS DIARIOS (para periodos largos)
# ═══════════════════════════════════════════════════════════════════

def backtest_daily(months=12, verbose=True):
    """
    Backtest con datos diarios para periodos largos
    
    MOMENTUM: Comprar cuando sube >2% en 1 día
    DIP BUY: Comprar cuando cae >3% en 1 día
    Exit: Al cierre del día siguiente
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📈 BACKTEST DIARIO — {months} MESES")
        print(f"{'='*70}")
        print(f"Datos: Diarios | Lookback: 1 día")
        print(f"Momentum: >{CONFIG['momentum_threshold']}% | Dip: <{CONFIG['dip_threshold']}%\n")
    
    all_trades = []
    
    for ticker in TICKERS:
        data = download_data(ticker, period=f'{months}mo', interval='1d')
        
        if data is None or len(data) < 30:
            continue
        
        asset = ASSETS[ticker]
        
        # Calcular cambio diario
        data['Change'] = data['Close'].pct_change() * 100
        
        if verbose:
            signals_up = len(data[data['Change'] > CONFIG['momentum_threshold']])
            signals_down = len(data[data['Change'] < CONFIG['dip_threshold']])
            print(f"✅ {ticker:5}: {len(data)} días | Mom: {signals_up}, Dip: {signals_down}")
        
        # Generar trades
        for i in range(21, len(data) - 1):
            change = data['Change'].iloc[i]
            
            if pd.isna(change):
                continue
            
            entry_price = data['Close'].iloc[i]
            exit_price = data['Close'].iloc[i + 1]
            entry_date = data.index[i]
            strategy = None
            
            # MOMENTUM: Subió >2%
            if change > CONFIG['momentum_threshold']:
                strategy = 'MOMENTUM'
            # DIP BUY: Cayó >3%
            elif change < CONFIG['dip_threshold']:
                strategy = 'DIP_BUY'
            
            if strategy:
                pnl = (exit_price - entry_price) / entry_price * 100
                
                all_trades.append({
                    'Ticker': ticker,
                    'Strategy': strategy,
                    'Date': entry_date.strftime('%Y-%m-%d'),
                    'Entry': round(entry_price, 2),
                    'Exit': round(exit_price, 2),
                    'P&L%': round(pnl, 2),
                    'Result': 'NEXT_DAY',
                    'Bars': 1,
                })
    
    if all_trades:
        df = pd.DataFrame(all_trades)
        
        total = len(df)
        wins = len(df[df['P&L%'] > 0])
        losses = total - wins
        win_rate = wins / total * 100 if total > 0 else 0
        
        avg_win = df[df['P&L%'] > 0]['P&L%'].mean() if wins > 0 else 0
        avg_loss = df[df['P&L%'] <= 0]['P&L%'].mean() if losses > 0 else 0
        
        total_return = df['P&L%'].sum()
        avg_return = df['P&L%'].mean()
        
        gross_profit = df[df['P&L%'] > 0]['P&L%'].sum() if wins > 0 else 0
        gross_loss = abs(df[df['P&L%'] <= 0]['P&L%'].sum()) if losses > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
        
        momentum = df[df['Strategy'] == 'MOMENTUM']
        dip_buy = df[df['Strategy'] == 'DIP_BUY']
        
        annual_return = (total_return / months) * 12
        
        results = {
            'months': months,
            'trades': total,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annual_return': annual_return,
            'avg_trade': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'momentum_trades': len(momentum),
            'momentum_return': momentum['P&L%'].sum() if len(momentum) > 0 else 0,
            'dip_trades': len(dip_buy),
            'dip_return': dip_buy['P&L%'].sum() if len(dip_buy) > 0 else 0,
        }
        
        if verbose:
            print(f"""
{'='*70}
📊 RESULTADOS — {months} MESES (DIARIO)
{'='*70}

╔══════════════════════════════════════════════════════════════════╗
║  TRADES: {total:<55}║
║  WIN RATE: {win_rate:.1f}% ({wins}W / {losses}L)                            ║
║  PROFIT FACTOR: {profit_factor:.2f}                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  RETURN TOTAL: {total_return:.1f}%                                        ║
║  RETURN ANUALIZADO: {annual_return:.1f}%                                   ║
║  AVG POR TRADE: {avg_return:.2f}%                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  MOMENTUM: {len(momentum)} trades → {momentum['P&L%'].sum():.1f}%                           ║
║  DIP BUY: {len(dip_buy)} trades → {dip_buy['P&L%'].sum():.1f}%                            ║
╚══════════════════════════════════════════════════════════════════╝
""")
            
            # Top 5 tickers
            print(f"📊 TOP 5 TICKERS:")
            by_ticker = df.groupby('Ticker')['P&L%'].agg(['sum', 'count']).round(2)
            by_ticker.columns = ['Total%', 'Trades']
            print(by_ticker.sort_values('Total%', ascending=False).head(5).to_string())
        
        return results, df
    else:
        if verbose:
            print("❌ No se generaron trades")
        return None, None

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    compare_periods()
