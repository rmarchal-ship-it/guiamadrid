#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🏆 HYPER-CONVEX V7.3 FINAL 🏆                           ║
║                                                                   ║
║           TIEMPO REAL + BACKTEST MULTI-PERIODO                    ║
║           35+ ACTIVOS DIVERSIFICADOS                              ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

ACTIVOS (35+):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÍNDICES US:     QQQ, TQQQ, SPY, SPXL, IWM, TNA
ÍNDICES EU:     EWG (DAX), EWP (IBEX), FEZ (EuroStoxx), EWU (UK)
ÍNDICES ASIA:   EWJ (Japón), FXI (China), EWT (Taiwan), EWY (Korea)
EMERGENTES:     EEM, EWZ (Brasil), EWW (México), TUR (Turquía)
BONOS:          TLT (20Y), IEF (7-10Y), HYG (High Yield), LQD (Corp)
COMMODITIES:    GLD, SLV, UNG, USO (Oil), BNO (Brent), CPER (Copper)
AGRICULTURA:    DBA, WEAT, CORN, SOYB
SECTORES:       XLE (Energy), XLF (Finance), XLV (Health), GDX (Miners)
STOCKS:         TSLA, NVDA, SMCI, GME, UPST, COIN, AMD, MSTR

USO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python v73_final.py --scan              # Escaneo tiempo real
  python v73_final.py --watch             # Monitoreo continuo
  python v73_final.py --backtest          # Backtest todos periodos
  python v73_final.py --backtest --months 12  # Periodo específico
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # Thresholds tiempo real (intradía)
    'momentum_1h': 1.0,
    'momentum_4h': 2.0,
    'dip_1h': -1.5,
    'dip_4h': -3.0,
    
    # Thresholds diarios
    'momentum_daily': 2.0,
    'dip_daily': -3.0,
    
    # Risk management
    'stop_pct': 1.5,
    'target_pct': 2.5,
    'max_hold_bars': 8,
    
    # Refresh
    'scan_interval': 60,
}

# ═══════════════════════════════════════════════════════════════════
# ACTIVOS DIVERSIFICADOS (35+)
# ═══════════════════════════════════════════════════════════════════

ASSETS = {
    # ═══════════════════════════════════════════════════════════════
    # ÍNDICES US + LEVERAGED
    # ═══════════════════════════════════════════════════════════════
    'QQQ': {'name': 'Nasdaq-100', 'future': '/MNQ', 'category': 'US_INDEX'},
    'TQQQ': {'name': 'Nasdaq 3x', 'future': '/MNQ', 'category': 'US_LEVERAGED'},
    'SPY': {'name': 'S&P 500', 'future': '/MES', 'category': 'US_INDEX'},
    'SPXL': {'name': 'S&P 3x', 'future': '/MES', 'category': 'US_LEVERAGED'},
    'IWM': {'name': 'Russell 2000', 'future': '/M2K', 'category': 'US_INDEX'},
    'TNA': {'name': 'Russell 3x', 'future': '/M2K', 'category': 'US_LEVERAGED'},
    'DIA': {'name': 'Dow Jones', 'future': '/MYM', 'category': 'US_INDEX'},
    
    # ═══════════════════════════════════════════════════════════════
    # ÍNDICES EUROPEOS
    # ═══════════════════════════════════════════════════════════════
    'EWG': {'name': 'Germany (DAX)', 'future': '/FDAX', 'category': 'EU_INDEX'},
    'EWP': {'name': 'Spain (IBEX)', 'future': None, 'category': 'EU_INDEX'},
    'FEZ': {'name': 'EuroStoxx 50', 'future': '/FESX', 'category': 'EU_INDEX'},
    'EWU': {'name': 'UK (FTSE)', 'future': '/Z', 'category': 'EU_INDEX'},
    'EWQ': {'name': 'France (CAC)', 'future': '/FCE', 'category': 'EU_INDEX'},
    'EWI': {'name': 'Italy (FTSE MIB)', 'future': None, 'category': 'EU_INDEX'},
    'EWL': {'name': 'Switzerland (SMI)', 'future': None, 'category': 'EU_INDEX'},
    
    # ═══════════════════════════════════════════════════════════════
    # ÍNDICES ASIA/PACÍFICO
    # ═══════════════════════════════════════════════════════════════
    'EWJ': {'name': 'Japan (Nikkei)', 'future': '/NKD', 'category': 'ASIA_INDEX'},
    'FXI': {'name': 'China Large Cap', 'future': None, 'category': 'ASIA_INDEX'},
    'KWEB': {'name': 'China Internet', 'future': None, 'category': 'ASIA_INDEX'},
    'EWT': {'name': 'Taiwan', 'future': None, 'category': 'ASIA_INDEX'},
    'EWY': {'name': 'South Korea', 'future': None, 'category': 'ASIA_INDEX'},
    'EWA': {'name': 'Australia', 'future': None, 'category': 'ASIA_INDEX'},
    
    # ═══════════════════════════════════════════════════════════════
    # MERCADOS EMERGENTES
    # ═══════════════════════════════════════════════════════════════
    'EEM': {'name': 'Emerging Markets', 'future': None, 'category': 'EMERGING'},
    'EWZ': {'name': 'Brazil', 'future': None, 'category': 'EMERGING'},
    'EWW': {'name': 'Mexico', 'future': None, 'category': 'EMERGING'},
    'TUR': {'name': 'Turkey', 'future': None, 'category': 'EMERGING'},
    'INDA': {'name': 'India', 'future': None, 'category': 'EMERGING'},
    
    # ═══════════════════════════════════════════════════════════════
    # BONOS
    # ═══════════════════════════════════════════════════════════════
    'TLT': {'name': 'Treasury 20Y+', 'future': '/ZB', 'category': 'BONDS'},
    'IEF': {'name': 'Treasury 7-10Y', 'future': '/ZN', 'category': 'BONDS'},
    'HYG': {'name': 'High Yield Corp', 'future': None, 'category': 'BONDS'},
    'LQD': {'name': 'Invest Grade Corp', 'future': None, 'category': 'BONDS'},
    'TMF': {'name': 'Treasury 20Y 3x', 'future': '/ZB', 'category': 'BONDS_LEV'},
    
    # ═══════════════════════════════════════════════════════════════
    # COMMODITIES - METALES
    # ═══════════════════════════════════════════════════════════════
    'GLD': {'name': 'Gold', 'future': '/MGC', 'category': 'COMMODITY'},
    'SLV': {'name': 'Silver', 'future': '/SIL', 'category': 'COMMODITY'},
    'CPER': {'name': 'Copper', 'future': '/HG', 'category': 'COMMODITY'},
    'PPLT': {'name': 'Platinum', 'future': '/PL', 'category': 'COMMODITY'},
    'GDX': {'name': 'Gold Miners', 'future': None, 'category': 'COMMODITY'},
    
    # ═══════════════════════════════════════════════════════════════
    # COMMODITIES - ENERGÍA
    # ═══════════════════════════════════════════════════════════════
    'USO': {'name': 'WTI Oil', 'future': '/CL', 'category': 'ENERGY'},
    'BNO': {'name': 'Brent Oil', 'future': '/BZ', 'category': 'ENERGY'},
    'UNG': {'name': 'Natural Gas', 'future': '/NG', 'category': 'ENERGY'},
    'XLE': {'name': 'Energy Sector', 'future': None, 'category': 'ENERGY'},
    'XOP': {'name': 'Oil & Gas Explor', 'future': None, 'category': 'ENERGY'},
    
    # ═══════════════════════════════════════════════════════════════
    # COMMODITIES - AGRICULTURA
    # ═══════════════════════════════════════════════════════════════
    'DBA': {'name': 'Agriculture', 'future': None, 'category': 'AGRICULTURE'},
    'WEAT': {'name': 'Wheat', 'future': '/ZW', 'category': 'AGRICULTURE'},
    'CORN': {'name': 'Corn', 'future': '/ZC', 'category': 'AGRICULTURE'},
    'SOYB': {'name': 'Soybeans', 'future': '/ZS', 'category': 'AGRICULTURE'},
    
    # ═══════════════════════════════════════════════════════════════
    # SECTORES US
    # ═══════════════════════════════════════════════════════════════
    'XLF': {'name': 'Financials', 'future': None, 'category': 'SECTOR'},
    'XLV': {'name': 'Healthcare', 'future': None, 'category': 'SECTOR'},
    'XLK': {'name': 'Technology', 'future': None, 'category': 'SECTOR'},
    'XLI': {'name': 'Industrials', 'future': None, 'category': 'SECTOR'},
    'XLP': {'name': 'Consumer Staples', 'future': None, 'category': 'SECTOR'},
    'XLY': {'name': 'Consumer Discret', 'future': None, 'category': 'SECTOR'},
    'XLU': {'name': 'Utilities', 'future': None, 'category': 'SECTOR'},
    'XLRE': {'name': 'Real Estate', 'future': None, 'category': 'SECTOR'},
    
    # ═══════════════════════════════════════════════════════════════
    # STOCKS INDIVIDUALES
    # ═══════════════════════════════════════════════════════════════
    'TSLA': {'name': 'Tesla', 'future': None, 'category': 'STOCK'},
    'NVDA': {'name': 'NVIDIA', 'future': None, 'category': 'STOCK'},
    'SMCI': {'name': 'Super Micro', 'future': None, 'category': 'STOCK'},
    'GME': {'name': 'GameStop', 'future': None, 'category': 'STOCK'},
    'UPST': {'name': 'Upstart', 'future': None, 'category': 'STOCK'},
    'COIN': {'name': 'Coinbase', 'future': None, 'category': 'STOCK'},
    'AMD': {'name': 'AMD', 'future': None, 'category': 'STOCK'},
    'MSTR': {'name': 'MicroStrategy', 'future': '/MBT', 'category': 'STOCK'},
    'HIMS': {'name': 'Hims & Hers', 'future': None, 'category': 'STOCK'},
}

TICKERS = list(ASSETS.keys())
CATEGORIES = list(set(a['category'] for a in ASSETS.values()))

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

def calculate_rsi(prices, period=7):
    """RSI rápido"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def play_alert():
    os.system('afplay /System/Library/Sounds/Ping.aiff 2>/dev/null || echo -e "\a"')

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

# ═══════════════════════════════════════════════════════════════════
# SCANNER TIEMPO REAL
# ═══════════════════════════════════════════════════════════════════

def scan_realtime(category_filter=None):
    """Scanner en tiempo real con datos de 5 minutos"""
    results = []
    
    tickers = TICKERS
    if category_filter:
        tickers = [t for t in TICKERS if ASSETS[t]['category'] == category_filter]
    
    for ticker in tickers:
        data = download_data(ticker, period='5d', interval='5m')
        
        if data is None or len(data) < 50:
            continue
        
        asset = ASSETS[ticker]
        current = data['Close'].iloc[-1]
        
        # Cambios
        chg_15m = (current / data['Close'].iloc[-4] - 1) * 100 if len(data) >= 4 else 0
        chg_1h = (current / data['Close'].iloc[-13] - 1) * 100 if len(data) >= 13 else 0
        chg_4h = (current / data['Close'].iloc[-49] - 1) * 100 if len(data) >= 49 else 0
        
        today = data[data.index.date == data.index[-1].date()]
        chg_day = (current / today['Open'].iloc[0] - 1) * 100 if len(today) > 0 else 0
        
        rsi = calculate_rsi(data['Close']).iloc[-1]
        
        # Señales
        signal = None
        if chg_1h > CONFIG['momentum_1h']:
            signal = '🟢 MOM 1H'
        elif chg_4h > CONFIG['momentum_4h'] and chg_1h > 0:
            signal = '🟢 MOM 4H'
        elif chg_1h < CONFIG['dip_1h']:
            signal = '🔵 DIP 1H'
        elif chg_4h < CONFIG['dip_4h'] and chg_1h < 0:
            signal = '🔵 DIP 4H'
        
        trend = '📈' if chg_15m > 0.15 else '📉' if chg_15m < -0.15 else '➡️'
        
        results.append({
            'Ticker': ticker,
            'Name': asset['name'][:15],
            'Cat': asset['category'][:8],
            'Price': current,
            '15m': chg_15m,
            '1H': chg_1h,
            '4H': chg_4h,
            'Day': chg_day,
            'RSI': rsi,
            'Trend': trend,
            'Signal': signal,
            'Future': asset.get('future'),
        })
    
    return results

def display_realtime(results, sound=False):
    """Muestra resultados tiempo real"""
    now = datetime.now()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║  🔴 RADAR V7.3 FINAL — {now.strftime('%H:%M:%S')}                            ║
║  {len(TICKERS)} Activos | MOM >+{CONFIG['momentum_1h']}% (1H) | DIP <{CONFIG['dip_1h']}% (1H)            ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    # Separar por señal
    momentum = [r for r in results if r['Signal'] and 'MOM' in r['Signal']]
    dips = [r for r in results if r['Signal'] and 'DIP' in r['Signal']]
    
    if momentum or dips:
        print("═"*70)
        print(f"  🎯 {len(momentum) + len(dips)} SEÑALES ACTIVAS")
        print("═"*70 + "\n")
        
        if momentum:
            print("🟢 MOMENTUM:")
            for r in sorted(momentum, key=lambda x: x['1H'], reverse=True)[:10]:
                future_str = f" → {r['Future']}" if r['Future'] else ""
                print(f"  {r['Ticker']:6} {r['Name']:15} {r['Signal']:10} 1H:{r['1H']:+.1f}% Day:{r['Day']:+.1f}%{future_str}")
        
        if dips:
            print("\n🔵 DIP BUY:")
            for r in sorted(dips, key=lambda x: x['1H'])[:10]:
                future_str = f" → {r['Future']}" if r['Future'] else ""
                print(f"  {r['Ticker']:6} {r['Name']:15} {r['Signal']:10} 1H:{r['1H']:+.1f}% Day:{r['Day']:+.1f}%{future_str}")
        
        if sound:
            play_alert()
    else:
        print("❌ Sin señales activas\n")
    
    # Tabla completa por categoría
    print("\n" + "═"*70)
    print("  📊 ESTADO POR CATEGORÍA")
    print("═"*70 + "\n")
    
    for cat in sorted(CATEGORIES):
        cat_results = [r for r in results if ASSETS[r['Ticker']]['category'] == cat]
        if not cat_results:
            continue
        
        # Resumen de categoría
        avg_day = np.mean([r['Day'] for r in cat_results])
        up = len([r for r in cat_results if r['1H'] > 0.3])
        down = len([r for r in cat_results if r['1H'] < -0.3])
        signals = len([r for r in cat_results if r['Signal']])
        
        status = '🟢' if avg_day > 0.5 else '🔴' if avg_day < -0.5 else '➡️'
        
        print(f"{status} {cat:12} │ Day:{avg_day:+.1f}% │ ↑{up} ↓{down} │ Signals:{signals}")
    
    return results

def watch_mode(interval=60, sound=False):
    """Monitoreo continuo"""
    try:
        while True:
            clear_screen()
            results = scan_realtime()
            display_realtime(results, sound=sound)
            print(f"\n⏰ Próxima actualización en {interval}s... (Ctrl+C para salir)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Radar detenido.")

# ═══════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════

def backtest_intraday(months, verbose=True):
    """Backtest con datos 1H"""
    if verbose:
        print(f"\n{'='*70}")
        print(f"📈 BACKTEST INTRADÍA — {months} MESES ({len(TICKERS)} activos)")
        print(f"{'='*70}")
        print(f"MOM >+{CONFIG['momentum_1h']}% (1H) | DIP <{CONFIG['dip_1h']}% (1H)")
        print(f"Stop -{CONFIG['stop_pct']}% | Target +{CONFIG['target_pct']}% | Max {CONFIG['max_hold_bars']}H\n")
    
    all_trades = []
    stats_by_cat = {}
    
    for ticker in TICKERS:
        data = download_data(ticker, period=f'{months}mo', interval='1h')
        
        if data is None or len(data) < 50:
            continue
        
        asset = ASSETS[ticker]
        cat = asset['category']
        
        data['chg_1h'] = data['Close'].pct_change() * 100
        data['chg_4h'] = data['Close'].pct_change(periods=4) * 100
        
        if verbose:
            mom = len(data[data['chg_1h'] > CONFIG['momentum_1h']])
            dip = len(data[data['chg_1h'] < CONFIG['dip_1h']])
            print(f"✅ {ticker:6} ({cat[:6]:6}): {len(data):4} bars | MOM:{mom:3} DIP:{dip:3}")
        
        i = 5
        while i < len(data) - CONFIG['max_hold_bars']:
            chg_1h = data['chg_1h'].iloc[i]
            chg_4h = data['chg_4h'].iloc[i]
            
            if pd.isna(chg_1h) or pd.isna(chg_4h):
                i += 1
                continue
            
            signal = None
            if chg_1h > CONFIG['momentum_1h']:
                signal = 'MOM_1H'
            elif chg_4h > CONFIG['momentum_4h'] and chg_1h > 0:
                signal = 'MOM_4H'
            elif chg_1h < CONFIG['dip_1h']:
                signal = 'DIP_1H'
            elif chg_4h < CONFIG['dip_4h'] and chg_1h < 0:
                signal = 'DIP_4H'
            
            if signal:
                entry_price = data['Close'].iloc[i]
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
                    'Category': cat,
                    'Strategy': strategy,
                    'P&L%': round(pnl, 2),
                    'Result': exit_reason,
                })
                
                i += CONFIG['max_hold_bars']
            else:
                i += 1
    
    return process_results(all_trades, months, 'INTRADÍA', verbose)

def backtest_daily(months, verbose=True):
    """Backtest con datos diarios"""
    if verbose:
        print(f"\n{'='*70}")
        print(f"📈 BACKTEST DIARIO — {months} MESES ({len(TICKERS)} activos)")
        print(f"{'='*70}")
        print(f"MOM >+{CONFIG['momentum_daily']}% | DIP <{CONFIG['dip_daily']}%")
        print(f"Exit: Cierre día siguiente\n")
    
    all_trades = []
    
    for ticker in TICKERS:
        period = 'max' if months > 24 else f'{months}mo'
        data = download_data(ticker, period=period, interval='1d')
        
        if data is None or len(data) < 30:
            continue
        
        if months <= 36 and len(data) > months * 21:
            data = data.tail(months * 21)
        
        asset = ASSETS[ticker]
        cat = asset['category']
        
        data['chg'] = data['Close'].pct_change() * 100
        
        if verbose:
            mom = len(data[data['chg'] > CONFIG['momentum_daily']])
            dip = len(data[data['chg'] < CONFIG['dip_daily']])
            print(f"✅ {ticker:6} ({cat[:6]:6}): {len(data):4} days | MOM:{mom:3} DIP:{dip:3}")
        
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
                pnl = (exit_price - entry_price) / entry_price * 100
                
                all_trades.append({
                    'Ticker': ticker,
                    'Category': cat,
                    'Strategy': strategy,
                    'P&L%': round(pnl, 2),
                    'Result': 'NEXT_DAY',
                })
    
    return process_results(all_trades, months, 'DIARIO', verbose)

def process_results(all_trades, months, data_type, verbose=True):
    """Procesa resultados"""
    if not all_trades:
        if verbose:
            print("❌ No se generaron trades")
        return None
    
    df = pd.DataFrame(all_trades)
    
    total = len(df)
    wins = len(df[df['P&L%'] > 0])
    losses = total - wins
    win_rate = wins / total * 100
    
    avg_win = df[df['P&L%'] > 0]['P&L%'].mean() if wins > 0 else 0
    avg_loss = df[df['P&L%'] <= 0]['P&L%'].mean() if losses > 0 else 0
    
    total_return = df['P&L%'].sum()
    
    gross_profit = df[df['P&L%'] > 0]['P&L%'].sum() if wins > 0 else 0
    gross_loss = abs(df[df['P&L%'] <= 0]['P&L%'].sum()) if losses > 0 else 1
    profit_factor = gross_profit / gross_loss
    
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
        'momentum_return': round(momentum['P&L%'].sum(), 1) if len(momentum) > 0 else 0,
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
║  PROFIT FACTOR: {profit_factor:.2f}                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  RETURN TOTAL: {total_return:.1f}%                                        ║
║  RETURN ANUALIZADO: {annual_return:.1f}%                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  MOMENTUM: {len(momentum):4} trades → {momentum['P&L%'].sum():.1f}%                          ║
║  DIP BUY:  {len(dip_buy):4} trades → {dip_buy['P&L%'].sum() if len(dip_buy) > 0 else 0:.1f}%                          ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        # Por categoría
        print("📊 POR CATEGORÍA:")
        by_cat = df.groupby('Category')['P&L%'].agg(['sum', 'count']).round(1)
        by_cat.columns = ['Return%', 'Trades']
        by_cat['Avg%'] = (by_cat['Return%'] / by_cat['Trades']).round(2)
        by_cat = by_cat.sort_values('Return%', ascending=False)
        print(by_cat.to_string())
        
        # Top tickers
        print("\n📊 TOP 10 TICKERS:")
        by_ticker = df.groupby('Ticker')['P&L%'].agg(['sum', 'count']).round(1)
        by_ticker.columns = ['Return%', 'Trades']
        print(by_ticker.sort_values('Return%', ascending=False).head(10).to_string())
        
        # Bottom tickers
        print("\n📊 BOTTOM 5 TICKERS:")
        print(by_ticker.sort_values('Return%', ascending=True).head(5).to_string())
    
    return results

def run_all_backtests():
    """Ejecuta todos los backtests"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           📊 BACKTEST MULTI-PERIODO V7.3 FINAL 📊                 ║
║                                                                   ║
║           {0} Activos Diversificados                              ║
║           Periodos: 3, 6, 12, 24, 36 meses                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """.format(len(TICKERS)))
    
    results_all = []
    
    # 3 meses
    print("\n" + "#"*70)
    print("# PERIODO: 3 MESES (datos 1H)")
    print("#"*70)
    r = backtest_intraday(3)
    if r: results_all.append(r)
    
    # 6 meses
    print("\n" + "#"*70)
    print("# PERIODO: 6 MESES (datos 1H)")
    print("#"*70)
    r = backtest_intraday(6)
    if r: results_all.append(r)
    
    # 12 meses
    print("\n" + "#"*70)
    print("# PERIODO: 12 MESES (datos diarios)")
    print("#"*70)
    r = backtest_daily(12)
    if r: results_all.append(r)
    
    # 24 meses
    print("\n" + "#"*70)
    print("# PERIODO: 24 MESES (datos diarios)")
    print("#"*70)
    r = backtest_daily(24)
    if r: results_all.append(r)
    
    # 36 meses
    print("\n" + "#"*70)
    print("# PERIODO: 36 MESES (datos diarios)")
    print("#"*70)
    r = backtest_daily(36)
    if r: results_all.append(r)
    
    # Resumen
    print("\n" + "="*70)
    print("📊 COMPARACIÓN FINAL — TODOS LOS PERIODOS")
    print("="*70 + "\n")
    
    if results_all:
        summary = pd.DataFrame(results_all)
        display_cols = ['months', 'data_type', 'trades', 'win_rate', 'profit_factor', 
                       'total_return', 'annual_return', 'momentum_return', 'dip_return']
        summary_display = summary[display_cols]
        summary_display.columns = ['Meses', 'Datos', 'Trades', 'WR%', 'PF', 
                                  'Return%', 'Annual%', 'MOM%', 'DIP%']
        print(summary_display.to_string(index=False))
        
        avg_wr = summary['win_rate'].mean()
        avg_pf = summary['profit_factor'].mean()
        avg_annual = summary['annual_return'].mean()
        
        print(f"""

╔══════════════════════════════════════════════════════════════════╗
║  📋 CONCLUSIONES V7.3 FINAL                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  WIN RATE PROMEDIO: {avg_wr:.1f}%                                       ║
║  PROFIT FACTOR PROMEDIO: {avg_pf:.2f}                                 ║
║  RETURN ANUALIZADO PROMEDIO: {avg_annual:.1f}%                          ║
╠══════════════════════════════════════════════════════════════════╣
║  ACTIVOS TESTEADOS: {len(TICKERS)}                                         ║
║  CATEGORÍAS: {len(CATEGORIES)}                                              ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    return results_all

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Hyper-Convex V7.3 FINAL')
    parser.add_argument('--scan', action='store_true', help='Escaneo tiempo real')
    parser.add_argument('--watch', action='store_true', help='Monitoreo continuo')
    parser.add_argument('--backtest', action='store_true', help='Backtest completo')
    parser.add_argument('--months', type=int, help='Backtest periodo específico')
    parser.add_argument('--interval', type=int, default=60, help='Segundos entre scans')
    parser.add_argument('--sound', action='store_true', help='Alertas sonoras')
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🏆 HYPER-CONVEX V7.3 FINAL 🏆                           ║
║                                                                   ║
║           {len(TICKERS)} Activos Diversificados                            ║
║           {len(CATEGORIES)} Categorías (US, EU, Asia, EM, Bonds, Commodities)    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    if args.backtest:
        if args.months:
            if args.months <= 6:
                backtest_intraday(args.months)
            else:
                backtest_daily(args.months)
        else:
            run_all_backtests()
    elif args.watch:
        watch_mode(interval=args.interval, sound=args.sound)
    elif args.scan:
        results = scan_realtime()
        display_realtime(results, sound=args.sound)
    else:
        print("""
USO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIEMPO REAL:
  python v73_final.py --scan              # Escaneo único
  python v73_final.py --watch             # Monitoreo continuo
  python v73_final.py --watch --sound     # Con alertas

BACKTEST:
  python v73_final.py --backtest          # Todos los periodos
  python v73_final.py --backtest --months 12  # Periodo específico
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        print("Ejecutando escaneo rápido...\n")
        results = scan_realtime()
        display_realtime(results)

if __name__ == "__main__":
    main()
