#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           📊 BACKTEST V7.3 — MULTI-PERIODO                        ║
║                                                                   ║
║           Estrategia: MOMENTUM 1H + DIP BUY 1H                    ║
║           Periodos: 3, 6, 12, 24, 36 meses                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

ESTRATEGIA TESTEADA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOMENTUM: Comprar cuando sube >1% en 1 hora (o >2% en 4 horas)
DIP BUY: Comprar cuando cae >1.5% en 1 hora (o >3% en 4 horas)
EXIT: Stop -1.5%, Target +2.5%, o máximo 8 barras

DATOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 3-6 meses: Datos 1H (alta precisión)
• 12-36 meses: Datos diarios (proxy, yfinance limita 1H a ~730 días)
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
    # Thresholds intradía
    'momentum_1h': 1.0,
    'momentum_4h': 2.0,
    'dip_1h': -1.5,
    'dip_4h': -3.0,
    
    # Thresholds diarios (para periodos largos)
    'momentum_daily': 2.0,
    'dip_daily': -3.0,
    
    # Risk management
    'stop_pct': 1.5,
    'target_pct': 2.5,
    'max_hold_bars': 8,
}

# ═══════════════════════════════════════════════════════════════════
# ACTIVOS (igual que V7.3)
# ═══════════════════════════════════════════════════════════════════

FUTURES_ETFS = ['QQQ', 'TQQQ', 'SPY', 'SPXL', 'IWM', 'TNA', 'GLD', 'SLV', 'UNG']
STOCKS = ['TSLA', 'NVDA', 'SMCI', 'GME', 'UPST', 'COIN', 'AMD']
ALL_TICKERS = FUTURES_ETFS + STOCKS

# ═══════════════════════════════════════════════════════════════════
# FUNCIONES BASE
# ═══════════════════════════════════════════════════════════════════

def download_data(ticker, period, interval):
    """Descarga datos con manejo de errores"""
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
# BACKTEST INTRADÍA (1H) — Para 3-6 meses
# ═══════════════════════════════════════════════════════════════════

def backtest_intraday(months, verbose=True):
    """
    Backtest con datos de 1 hora
    
    Estrategia:
    - MOMENTUM: >1% en 1H o >2% en 4H
    - DIP BUY: <-1.5% en 1H o <-3% en 4H
    - Exit: Stop -1.5%, Target +2.5%, o 8 horas max
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📈 BACKTEST INTRADÍA V7.3 — {months} MESES")
        print(f"{'='*70}")
        print(f"Datos: 1H | MOM >+{CONFIG['momentum_1h']}% (1H) | DIP <{CONFIG['dip_1h']}% (1H)")
        print(f"Exit: Stop -{CONFIG['stop_pct']}% | Target +{CONFIG['target_pct']}% | Max {CONFIG['max_hold_bars']}H\n")
    
    all_trades = []
    
    for ticker in ALL_TICKERS:
        data = download_data(ticker, period=f'{months}mo', interval='1h')
        
        if data is None or len(data) < 50:
            continue
        
        # Calcular cambios
        data['chg_1h'] = data['Close'].pct_change() * 100
        data['chg_4h'] = data['Close'].pct_change(periods=4) * 100
        
        if verbose:
            mom_1h = len(data[data['chg_1h'] > CONFIG['momentum_1h']])
            mom_4h = len(data[data['chg_4h'] > CONFIG['momentum_4h']])
            dip_1h = len(data[data['chg_1h'] < CONFIG['dip_1h']])
            print(f"✅ {ticker:5}: {len(data):4} barras | MOM: {mom_1h+mom_4h:3} | DIP: {dip_1h:3}")
        
        # Generar trades
        i = 5
        while i < len(data) - CONFIG['max_hold_bars']:
            chg_1h = data['chg_1h'].iloc[i]
            chg_4h = data['chg_4h'].iloc[i]
            
            if pd.isna(chg_1h) or pd.isna(chg_4h):
                i += 1
                continue
            
            signal = None
            
            # MOMENTUM
            if chg_1h > CONFIG['momentum_1h']:
                signal = 'MOM_1H'
            elif chg_4h > CONFIG['momentum_4h'] and chg_1h > 0:
                signal = 'MOM_4H'
            # DIP BUY
            elif chg_1h < CONFIG['dip_1h']:
                signal = 'DIP_1H'
            elif chg_4h < CONFIG['dip_4h'] and chg_1h < 0:
                signal = 'DIP_4H'
            
            if signal:
                entry_price = data['Close'].iloc[i]
                entry_time = data.index[i]
                
                stop = entry_price * (1 - CONFIG['stop_pct']/100)
                target = entry_price * (1 + CONFIG['target_pct']/100)
                
                exit_price = None
                exit_reason = None
                
                for j in range(i + 1, min(i + CONFIG['max_hold_bars'] + 1, len(data))):
                    bar = data.iloc[j]
                    
                    if bar['Low'] <= stop:
                        exit_price = stop
                        exit_reason = 'STOP'
                        break
                    elif bar['High'] >= target:
                        exit_price = target
                        exit_reason = 'TARGET'
                        break
                
                if exit_price is None:
                    exit_idx = min(i + CONFIG['max_hold_bars'], len(data) - 1)
                    exit_price = data['Close'].iloc[exit_idx]
                    exit_reason = 'TIME'
                
                pnl = (exit_price - entry_price) / entry_price * 100
                
                strategy = 'MOMENTUM' if 'MOM' in signal else 'DIP_BUY'
                
                all_trades.append({
                    'Ticker': ticker,
                    'Strategy': strategy,
                    'Signal': signal,
                    'Date': str(entry_time)[:16],
                    'Entry': round(entry_price, 2),
                    'Exit': round(exit_price, 2),
                    'P&L%': round(pnl, 2),
                    'Result': exit_reason,
                })
                
                i += CONFIG['max_hold_bars']
            else:
                i += 1
    
    return process_results(all_trades, months, 'INTRADÍA', verbose)

# ═══════════════════════════════════════════════════════════════════
# BACKTEST DIARIO — Para 12-36 meses
# ═══════════════════════════════════════════════════════════════════

def backtest_daily(months, verbose=True):
    """
    Backtest con datos diarios (para periodos largos)
    
    Estrategia:
    - MOMENTUM: >2% en 1 día
    - DIP BUY: <-3% en 1 día
    - Exit: Al cierre del día siguiente
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📈 BACKTEST DIARIO V7.3 — {months} MESES")
        print(f"{'='*70}")
        print(f"Datos: Diarios | MOM >+{CONFIG['momentum_daily']}% | DIP <{CONFIG['dip_daily']}%")
        print(f"Exit: Cierre día siguiente\n")
    
    all_trades = []
    
    for ticker in ALL_TICKERS:
        # Para >24 meses, usar max period
        if months > 24:
            period = 'max'
        else:
            period = f'{months}mo'
        
        data = download_data(ticker, period=period, interval='1d')
        
        if data is None or len(data) < 30:
            continue
        
        # Limitar a los meses solicitados
        if months <= 36 and len(data) > months * 21:
            data = data.tail(months * 21)
        
        # Calcular cambio diario
        data['chg'] = data['Close'].pct_change() * 100
        
        if verbose:
            mom = len(data[data['chg'] > CONFIG['momentum_daily']])
            dip = len(data[data['chg'] < CONFIG['dip_daily']])
            print(f"✅ {ticker:5}: {len(data):4} días | MOM: {mom:3} | DIP: {dip:3}")
        
        # Generar trades
        for i in range(21, len(data) - 1):
            chg = data['chg'].iloc[i]
            
            if pd.isna(chg):
                continue
            
            signal = None
            strategy = None
            
            if chg > CONFIG['momentum_daily']:
                signal = 'MOM_DAY'
                strategy = 'MOMENTUM'
            elif chg < CONFIG['dip_daily']:
                signal = 'DIP_DAY'
                strategy = 'DIP_BUY'
            
            if signal:
                entry_price = data['Close'].iloc[i]
                exit_price = data['Close'].iloc[i + 1]
                entry_date = data.index[i]
                
                pnl = (exit_price - entry_price) / entry_price * 100
                
                all_trades.append({
                    'Ticker': ticker,
                    'Strategy': strategy,
                    'Signal': signal,
                    'Date': entry_date.strftime('%Y-%m-%d'),
                    'Entry': round(entry_price, 2),
                    'Exit': round(exit_price, 2),
                    'P&L%': round(pnl, 2),
                    'Result': 'NEXT_DAY',
                })
    
    return process_results(all_trades, months, 'DIARIO', verbose)

# ═══════════════════════════════════════════════════════════════════
# PROCESAR RESULTADOS
# ═══════════════════════════════════════════════════════════════════

def process_results(all_trades, months, data_type, verbose=True):
    """Procesa y muestra resultados del backtest"""
    
    if not all_trades:
        if verbose:
            print("❌ No se generaron trades")
        return None
    
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
        'data_type': data_type,
        'trades': total,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'total_return': round(total_return, 1),
        'annual_return': round(annual_return, 1),
        'avg_trade': round(avg_return, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'momentum_trades': len(momentum),
        'momentum_return': round(momentum['P&L%'].sum(), 1) if len(momentum) > 0 else 0,
        'dip_trades': len(dip_buy),
        'dip_return': round(dip_buy['P&L%'].sum(), 1) if len(dip_buy) > 0 else 0,
    }
    
    if verbose:
        print(f"""
{'='*70}
📊 RESULTADOS — {months} MESES ({data_type})
{'='*70}

╔══════════════════════════════════════════════════════════════════╗
║  TRADES: {total:<55}║
║  WIN RATE: {win_rate:.1f}% ({wins}W / {losses}L)                            ║
║  AVG WIN: +{avg_win:.2f}% | AVG LOSS: {avg_loss:.2f}%                      ║
║  PROFIT FACTOR: {profit_factor:.2f}                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  RETURN TOTAL: {total_return:.1f}%                                        ║
║  RETURN ANUALIZADO: {annual_return:.1f}%                                   ║
║  RETURN POR TRADE: {avg_return:.2f}%                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  MOMENTUM: {len(momentum):4} trades → {momentum['P&L%'].sum():.1f}%                          ║
║  DIP BUY:  {len(dip_buy):4} trades → {dip_buy['P&L%'].sum() if len(dip_buy) > 0 else 0:.1f}%                          ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        # Top tickers
        print("📊 TOP 5 TICKERS:")
        by_ticker = df.groupby('Ticker')['P&L%'].agg(['sum', 'count']).round(2)
        by_ticker.columns = ['Total%', 'Trades']
        by_ticker['Avg%'] = (by_ticker['Total%'] / by_ticker['Trades']).round(2)
        print(by_ticker.sort_values('Total%', ascending=False).head(5).to_string())
        
        # Por resultado (si intradía)
        if 'Result' in df.columns and data_type == 'INTRADÍA':
            print(f"\n📊 SALIDAS:")
            for result in ['STOP', 'TARGET', 'TIME']:
                subset = df[df['Result'] == result]
                if len(subset) > 0:
                    print(f"  {result:6}: {len(subset):4} ({len(subset)/total*100:4.1f}%) → {subset['P&L%'].sum():+.1f}%")
    
    return results

# ═══════════════════════════════════════════════════════════════════
# BACKTEST MULTI-PERIODO
# ═══════════════════════════════════════════════════════════════════

def run_all_backtests():
    """Ejecuta backtests para todos los periodos"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           📊 BACKTEST MULTI-PERIODO V7.3 📊                       ║
║                                                                   ║
║           Estrategia: MOMENTUM + DIP BUY (tiempo real)            ║
║           Periodos: 3, 6, 12, 24, 36 meses                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    results_all = []
    
    # 3 meses - Intradía
    print("\n" + "#"*70)
    print("# PERIODO: 3 MESES (datos 1H)")
    print("#"*70)
    r = backtest_intraday(3)
    if r:
        results_all.append(r)
    
    # 6 meses - Intradía
    print("\n" + "#"*70)
    print("# PERIODO: 6 MESES (datos 1H)")
    print("#"*70)
    r = backtest_intraday(6)
    if r:
        results_all.append(r)
    
    # 12 meses - Diario
    print("\n" + "#"*70)
    print("# PERIODO: 12 MESES (datos diarios)")
    print("#"*70)
    r = backtest_daily(12)
    if r:
        results_all.append(r)
    
    # 24 meses - Diario
    print("\n" + "#"*70)
    print("# PERIODO: 24 MESES (datos diarios)")
    print("#"*70)
    r = backtest_daily(24)
    if r:
        results_all.append(r)
    
    # 36 meses - Diario
    print("\n" + "#"*70)
    print("# PERIODO: 36 MESES (datos diarios)")
    print("#"*70)
    r = backtest_daily(36)
    if r:
        results_all.append(r)
    
    # ═══════════════════════════════════════════════════════════════
    # RESUMEN COMPARATIVO
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("📊 COMPARACIÓN FINAL — TODOS LOS PERIODOS")
    print("="*70 + "\n")
    
    if results_all:
        summary = pd.DataFrame(results_all)
        display_cols = ['months', 'data_type', 'trades', 'win_rate', 'profit_factor', 
                       'total_return', 'annual_return', 'momentum_return', 'dip_return']
        summary_display = summary[display_cols].copy()
        summary_display.columns = ['Meses', 'Datos', 'Trades', 'WR%', 'PF', 
                                  'Return%', 'Annual%', 'MOM%', 'DIP%']
        
        print(summary_display.to_string(index=False))
        
        # Promedios
        avg_wr = summary['win_rate'].mean()
        avg_pf = summary['profit_factor'].mean()
        avg_annual = summary['annual_return'].mean()
        total_mom = summary['momentum_return'].sum()
        total_dip = summary['dip_return'].sum()
        
        print(f"""

╔══════════════════════════════════════════════════════════════════╗
║  📋 CONCLUSIONES V7.3 TIEMPO REAL                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WIN RATE PROMEDIO: {avg_wr:.1f}%                                       ║
║  PROFIT FACTOR PROMEDIO: {avg_pf:.2f}                                 ║
║  RETURN ANUALIZADO PROMEDIO: {avg_annual:.1f}%                          ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  MOMENTUM vs DIP BUY:                                            ║
║    Momentum total: {total_mom:+.1f}%                                     ║
║    Dip Buy total:  {total_dip:+.1f}%                                     ║
║                                                                  ║
║  RECOMENDACIÓN:                                                  ║""")
        
        if total_mom > total_dip * 1.5:
            print("║    → MOMENTUM es significativamente mejor                        ║")
        elif total_dip > total_mom * 1.5:
            print("║    → DIP BUY es significativamente mejor                         ║")
        else:
            print("║    → Ambas estrategias funcionan, usar combinadas                ║")
        
        if avg_pf > 1.3:
            print("║    → Profit Factor >1.3 = Estrategia SÓLIDA ✅                   ║")
        elif avg_pf > 1.0:
            print("║    → Profit Factor >1.0 = Estrategia RENTABLE ⚠️                 ║")
        else:
            print("║    → Profit Factor <1.0 = Estrategia PERDEDORA ❌                ║")
        
        print("║                                                                  ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
    
    return results_all

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest V7.3')
    parser.add_argument('--all', action='store_true', help='Todos los periodos')
    parser.add_argument('--months', type=int, help='Periodo específico')
    parser.add_argument('--intraday', action='store_true', help='Forzar datos intradía')
    args = parser.parse_args()
    
    if args.months:
        if args.intraday or args.months <= 6:
            backtest_intraday(args.months)
        else:
            backtest_daily(args.months)
    else:
        run_all_backtests()
