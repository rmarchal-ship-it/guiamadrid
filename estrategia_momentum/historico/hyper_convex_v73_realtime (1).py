#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🔴 HYPER-CONVEX V7.3 — TIEMPO REAL 🔴                   ║
║                                                                   ║
║           DATOS EN VIVO (no cierre de ayer)                       ║
║           Cambios: 15min, 1H, 4H, desde apertura                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

CAMBIOS VS V7.2:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Datos de 5 minutos (no diarios)
✅ Cambio ACTUAL vs hace 15min, 1H, 4H, apertura
✅ Señales basadas en movimiento AHORA
✅ Refresh automático configurable
✅ Enfocado en FUTUROS que cotizan ahora

FUTUROS MAPEADOS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  /MNQ → QQQ, TQQQ  │  /MES → SPY, SPXL  │  /M2K → IWM, TNA
  /MGC → GLD        │  /SIL → SLV        │  /NG  → UNG

USO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python v73.py --scan             # Escaneo único AHORA
  python v73.py --watch            # Monitoreo continuo
  python v73.py --watch --interval 30  # Cada 30 segundos
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
    # Thresholds para señales REAL-TIME
    'momentum_1h': 1.0,         # >1% en 1H = momentum
    'momentum_4h': 2.0,         # >2% en 4H = momentum fuerte
    'dip_1h': -1.5,             # <-1.5% en 1H = dip
    'dip_4h': -3.0,             # <-3% en 4H = dip fuerte
    
    # Risk
    'stop_pct': 1.5,
    'target_pct': 2.5,
    
    # Refresh
    'default_interval': 60,     # Segundos
}

# ═══════════════════════════════════════════════════════════════════
# MAPEO FUTUROS ↔ ETFs
# ═══════════════════════════════════════════════════════════════════

FUTURES = {
    '/MNQ': {'etf': 'QQQ', 'lev': 'TQQQ', 'name': 'Micro Nasdaq'},
    '/MES': {'etf': 'SPY', 'lev': 'SPXL', 'name': 'Micro S&P'},
    '/M2K': {'etf': 'IWM', 'lev': 'TNA', 'name': 'Micro Russell'},
    '/MGC': {'etf': 'GLD', 'lev': None, 'name': 'Micro Gold'},
    '/SIL': {'etf': 'SLV', 'lev': None, 'name': 'Micro Silver'},
    '/NG': {'etf': 'UNG', 'lev': None, 'name': 'Natural Gas'},
}

STOCKS = {
    'TSLA': 'Tesla',
    'NVDA': 'NVIDIA',
    'SMCI': 'Super Micro',
    'GME': 'GameStop',
    'UPST': 'Upstart',
    'COIN': 'Coinbase',
    'AMD': 'AMD',
}

# ═══════════════════════════════════════════════════════════════════
# FUNCIONES BASE
# ═══════════════════════════════════════════════════════════════════

def download_5min(ticker, days=5):
    """Descarga datos de 5 minutos"""
    try:
        data = yf.download(ticker, period=f'{days}d', interval='5m', progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def calculate_rsi(prices, period=7):
    """RSI rápido para intradía"""
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

def scan_realtime():
    """
    Escanea FUTUROS y STOCKS en tiempo real
    
    Calcula cambios desde:
    - Hace 15 minutos (3 velas)
    - Hace 1 hora (12 velas)
    - Hace 4 horas (48 velas)
    - Apertura del día
    """
    now = datetime.now()
    futures_results = []
    stocks_results = []
    
    # ═══════════════════════════════════════════════════════════════
    # ESCANEAR FUTUROS (vía ETF proxy)
    # ═══════════════════════════════════════════════════════════════
    
    for future, info in FUTURES.items():
        etf = info['etf']
        data = download_5min(etf)
        
        if data is None or len(data) < 50:
            continue
        
        current = data['Close'].iloc[-1]
        
        # Cambios
        chg_15m = (current / data['Close'].iloc[-4] - 1) * 100 if len(data) >= 4 else 0
        chg_1h = (current / data['Close'].iloc[-13] - 1) * 100 if len(data) >= 13 else 0
        chg_4h = (current / data['Close'].iloc[-49] - 1) * 100 if len(data) >= 49 else 0
        
        # Cambio desde apertura HOY
        today = data[data.index.date == data.index[-1].date()]
        chg_day = (current / today['Open'].iloc[0] - 1) * 100 if len(today) > 0 else 0
        
        # RSI
        rsi = calculate_rsi(data['Close']).iloc[-1]
        
        # High/Low de hoy
        if len(today) > 0:
            high = today['High'].max()
            low = today['Low'].min()
            pos = (current - low) / (high - low) * 100 if high != low else 50
        else:
            high = low = current
            pos = 50
        
        # Señal
        signal = None
        if chg_1h > CONFIG['momentum_1h']:
            signal = '🟢 MOM 1H'
        elif chg_4h > CONFIG['momentum_4h'] and chg_1h > 0:
            signal = '🟢 MOM 4H'
        elif chg_1h < CONFIG['dip_1h']:
            signal = '🔵 DIP 1H'
        elif chg_4h < CONFIG['dip_4h'] and chg_1h < 0:
            signal = '🔵 DIP 4H'
        
        # Trend icon
        trend = '📈' if chg_15m > 0.15 else '📉' if chg_15m < -0.15 else '➡️'
        
        futures_results.append({
            'Future': future,
            'ETF': etf,
            'Name': info['name'],
            'Price': current,
            '15m': chg_15m,
            '1H': chg_1h,
            '4H': chg_4h,
            'Day': chg_day,
            'RSI': rsi,
            'Pos': pos,
            'Trend': trend,
            'Signal': signal,
        })
    
    # ═══════════════════════════════════════════════════════════════
    # ESCANEAR STOCKS
    # ═══════════════════════════════════════════════════════════════
    
    for ticker, name in STOCKS.items():
        data = download_5min(ticker)
        
        if data is None or len(data) < 50:
            continue
        
        current = data['Close'].iloc[-1]
        
        chg_15m = (current / data['Close'].iloc[-4] - 1) * 100 if len(data) >= 4 else 0
        chg_1h = (current / data['Close'].iloc[-13] - 1) * 100 if len(data) >= 13 else 0
        chg_4h = (current / data['Close'].iloc[-49] - 1) * 100 if len(data) >= 49 else 0
        
        today = data[data.index.date == data.index[-1].date()]
        chg_day = (current / today['Open'].iloc[0] - 1) * 100 if len(today) > 0 else 0
        
        rsi = calculate_rsi(data['Close']).iloc[-1]
        
        signal = None
        if chg_1h > CONFIG['momentum_1h']:
            signal = '🟢 MOM'
        elif chg_1h < CONFIG['dip_1h']:
            signal = '🔵 DIP'
        
        trend = '📈' if chg_15m > 0.15 else '📉' if chg_15m < -0.15 else '➡️'
        
        stocks_results.append({
            'Ticker': ticker,
            'Name': name,
            'Price': current,
            '15m': chg_15m,
            '1H': chg_1h,
            '4H': chg_4h,
            'Day': chg_day,
            'RSI': rsi,
            'Trend': trend,
            'Signal': signal,
        })
    
    return futures_results, stocks_results

# ═══════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════

def display_results(futures, stocks, sound=False):
    """Muestra resultados en tiempo real"""
    now = datetime.now()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║  🔴 RADAR TIEMPO REAL V7.3 — {now.strftime('%H:%M:%S')}                       ║
║  Datos: 5min | MOM >+{CONFIG['momentum_1h']}% (1H) | DIP <{CONFIG['dip_1h']}% (1H)             ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    # ═══════════════════════════════════════════════════════════════
    # FUTUROS
    # ═══════════════════════════════════════════════════════════════
    
    print("══════════════════════════════════════════════════════════════════")
    print("                        📊 FUTUROS")
    print("══════════════════════════════════════════════════════════════════\n")
    
    rows = []
    for f in futures:
        rows.append({
            'Future': f['Future'],
            'ETF': f['ETF'],
            'Price': f"${f['Price']:.2f}",
            '15m': f"{f['15m']:+.2f}%",
            '1H': f"{f['1H']:+.2f}%",
            '4H': f"{f['4H']:+.2f}%",
            'Day': f"{f['Day']:+.2f}%",
            'RSI': f"{f['RSI']:.0f}" if not pd.isna(f['RSI']) else '-',
            '': f['Trend'],
            'Signal': f['Signal'] or '—',
        })
    
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    
    # Señales activas futuros
    active_futures = [f for f in futures if f['Signal']]
    
    # ═══════════════════════════════════════════════════════════════
    # STOCKS
    # ═══════════════════════════════════════════════════════════════
    
    print("\n══════════════════════════════════════════════════════════════════")
    print("                        📈 STOCKS")
    print("══════════════════════════════════════════════════════════════════\n")
    
    rows = []
    for s in stocks:
        rows.append({
            'Ticker': s['Ticker'],
            'Price': f"${s['Price']:.2f}",
            '15m': f"{s['15m']:+.2f}%",
            '1H': f"{s['1H']:+.2f}%",
            '4H': f"{s['4H']:+.2f}%",
            'Day': f"{s['Day']:+.2f}%",
            'RSI': f"{s['RSI']:.0f}" if not pd.isna(s['RSI']) else '-',
            '': s['Trend'],
            'Signal': s['Signal'] or '—',
        })
    
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    
    active_stocks = [s for s in stocks if s['Signal']]
    
    # ═══════════════════════════════════════════════════════════════
    # SEÑALES ACTIVAS
    # ═══════════════════════════════════════════════════════════════
    
    if active_futures or active_stocks:
        print("\n══════════════════════════════════════════════════════════════════")
        print(f"  🎯 {len(active_futures) + len(active_stocks)} SEÑALES ACTIVAS")
        print("══════════════════════════════════════════════════════════════════\n")
        
        for f in active_futures:
            stop = f['Price'] * (1 - CONFIG['stop_pct']/100)
            target = f['Price'] * (1 + CONFIG['target_pct']/100)
            print(f"""  {f['Signal']} {f['Future']} ({f['ETF']})
     Entry: ${f['Price']:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}
     1H: {f['1H']:+.2f}% | 4H: {f['4H']:+.2f}% | Day: {f['Day']:+.2f}%
""")
        
        for s in active_stocks:
            stop = s['Price'] * (1 - CONFIG['stop_pct']/100)
            target = s['Price'] * (1 + CONFIG['target_pct']/100)
            print(f"""  {s['Signal']} {s['Ticker']}
     Entry: ${s['Price']:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}
     1H: {s['1H']:+.2f}% | 4H: {s['4H']:+.2f}% | Day: {s['Day']:+.2f}%
""")
        
        if sound:
            play_alert()
    else:
        print("\n❌ Sin señales activas ahora mismo")
    
    print(f"""
══════════════════════════════════════════════════════════════════
💡 LEYENDA:
   🟢 MOM = Momentum (subiendo) | 🔵 DIP = Dip Buy (cayendo)
   📈 Subiendo 15min | 📉 Bajando 15min | ➡️ Lateral
   
⚠️  yfinance tiene ~15min delay. Para real-time usar Polygon.io
""")

# ═══════════════════════════════════════════════════════════════════
# MODO VIGILANCIA
# ═══════════════════════════════════════════════════════════════════

def watch_mode(interval=60, sound=False):
    """Monitoreo continuo"""
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║           🔴 RADAR V7.3 — TIEMPO REAL                             ║
║           Actualizando cada {interval} segundos                          ║
║           Ctrl+C para salir                                       ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        while True:
            clear_screen()
            futures, stocks = scan_realtime()
            display_results(futures, stocks, sound=sound)
            print(f"⏰ Próxima actualización en {interval} segundos...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Radar detenido.")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Radar Tiempo Real V7.3')
    parser.add_argument('--scan', action='store_true', help='Escaneo único')
    parser.add_argument('--watch', action='store_true', help='Monitoreo continuo')
    parser.add_argument('--interval', type=int, default=60, help='Segundos entre updates')
    parser.add_argument('--sound', action='store_true', help='Alertas sonoras')
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🔴 HYPER-CONVEX V7.3 — TIEMPO REAL 🔴                   ║
║                                                                   ║
║           ✓ Datos de 5 minutos (NO cierre de ayer)                ║
║           ✓ Cambios: 15min, 1H, 4H, desde apertura                ║
║           ✓ Señales basadas en movimiento ACTUAL                  ║
║           ✓ Futuros mapeados a ETFs                               ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    if args.watch:
        watch_mode(interval=args.interval, sound=args.sound)
    elif args.scan:
        futures, stocks = scan_realtime()
        display_results(futures, stocks, sound=args.sound)
    else:
        print("""
USO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python v73.py --scan             # Escaneo único AHORA
  python v73.py --watch            # Monitoreo continuo (60s)
  python v73.py --watch --interval 30  # Cada 30 segundos
  python v73.py --watch --sound    # Con alertas sonoras
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        print("Ejecutando escaneo único...\n")
        futures, stocks = scan_realtime()
        display_results(futures, stocks)

if __name__ == "__main__":
    main()
