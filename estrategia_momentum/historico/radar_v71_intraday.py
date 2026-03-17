#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║              HYPER-CONVEX RADAR V7.1 — INTRADÍA                   ║
║              MONITOREO EN TIEMPO REAL                             ║
╚═══════════════════════════════════════════════════════════════════╝

CARACTERÍSTICAS:
• Monitorea cada 5 minutos (configurable)
• Detecta movimientos >1% en última hora
• Detecta movimientos >2% en últimas 4 horas
• Señales MOMENTUM (sube fuerte) y DIP BUY (cae fuerte)
• Alerta sonora cuando hay señal (macOS)

USO:
  python radar.py --watch          # Monitoreo continuo
  python radar.py --scan           # Escaneo único
  python radar.py --watch --sound  # Con alertas sonoras

FUTUROS MONITOREADOS (vía ETFs proxy):
  /MNQ → QQQ, TQQQ
  /MES → SPY, SPXL  
  /MGC → GLD
  /SIL → SLV
  /NG  → UNG
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
    'scan_interval': 300,          # Segundos entre escaneos (5 min)
    'momentum_threshold_1h': 1.0,  # % para señal en 1 hora
    'momentum_threshold_4h': 2.0,  # % para señal en 4 horas
    'dip_threshold_1h': -1.5,      # % para dip buy en 1 hora
    'dip_threshold_4h': -3.0,      # % para dip buy en 4 horas
    'volume_threshold': 1.0,       # Ratio mínimo de volumen
}

# ═══════════════════════════════════════════════════════════════════
# ACTIVOS MONITOREADOS
# ═══════════════════════════════════════════════════════════════════

ASSETS = {
    # ETFs que sirven como proxy para futuros
    'QQQ': {'name': 'Nasdaq-100', 'future': '/MNQ', 'type': 'index'},
    'TQQQ': {'name': 'Nasdaq 3x', 'future': '/MNQ', 'type': 'leveraged'},
    'SPY': {'name': 'S&P 500', 'future': '/MES', 'type': 'index'},
    'SPXL': {'name': 'S&P 3x', 'future': '/MES', 'type': 'leveraged'},
    'IWM': {'name': 'Russell 2000', 'future': '/M2K', 'type': 'index'},
    'TNA': {'name': 'Russell 3x', 'future': '/M2K', 'type': 'leveraged'},
    'GLD': {'name': 'Gold', 'future': '/MGC', 'type': 'commodity'},
    'SLV': {'name': 'Silver', 'future': '/SIL', 'type': 'commodity'},
    'UNG': {'name': 'Natural Gas', 'future': '/NG', 'type': 'commodity'},
    # Stocks volátiles
    'TSLA': {'name': 'Tesla', 'future': None, 'type': 'stock'},
    'NVDA': {'name': 'NVIDIA', 'future': None, 'type': 'stock'},
    'SMCI': {'name': 'Super Micro', 'future': None, 'type': 'stock'},
    'COIN': {'name': 'Coinbase', 'future': None, 'type': 'crypto'},
}

TICKERS = list(ASSETS.keys())

# ═══════════════════════════════════════════════════════════════════
# FUNCIONES BASE
# ═══════════════════════════════════════════════════════════════════

def download_intraday(ticker, period='5d', interval='15m'):
    """Descarga datos intradía con fix para MultiIndex"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def calculate_rsi(prices, period=14):
    """RSI rápido"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def play_alert():
    """Alerta sonora (macOS)"""
    os.system('afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || echo -e "\a"')

# ═══════════════════════════════════════════════════════════════════
# SCANNER INTRADÍA
# ═══════════════════════════════════════════════════════════════════

def scan_intraday(sound=False):
    """
    Escanea movimientos intradía
    
    Detecta:
    - MOMENTUM: Subida >1% en última hora o >2% en 4 horas
    - DIP BUY: Caída >1.5% en última hora o >3% en 4 horas
    """
    now = datetime.now()
    
    signals = []
    
    for ticker in TICKERS:
        data = download_intraday(ticker, period='5d', interval='15m')
        
        if data is None or len(data) < 20:
            continue
        
        asset = ASSETS[ticker]
        
        # Precio actual
        current = data['Close'].iloc[-1]
        
        # Cambio en última hora (4 velas de 15min)
        if len(data) >= 4:
            price_1h_ago = data['Close'].iloc[-5]
            change_1h = (current - price_1h_ago) / price_1h_ago * 100
        else:
            change_1h = 0
        
        # Cambio en últimas 4 horas (16 velas de 15min)
        if len(data) >= 16:
            price_4h_ago = data['Close'].iloc[-17]
            change_4h = (current - price_4h_ago) / price_4h_ago * 100
        else:
            change_4h = 0
        
        # Cambio desde apertura del día
        today = data[data.index.date == data.index[-1].date()]
        if len(today) > 0:
            open_price = today['Open'].iloc[0]
            change_day = (current - open_price) / open_price * 100
        else:
            change_day = 0
        
        # Volumen relativo
        vol_avg = data['Volume'].tail(20).mean()
        vol_current = data['Volume'].iloc[-1]
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1
        
        # RSI
        rsi = calculate_rsi(data['Close']).iloc[-1]
        
        # Detectar señales
        signal = None
        signal_type = None
        timeframe = None
        
        # MOMENTUM 1H
        if change_1h > CONFIG['momentum_threshold_1h']:
            signal = '🟢 MOMENTUM'
            signal_type = 'LONG'
            timeframe = '1H'
        # MOMENTUM 4H
        elif change_4h > CONFIG['momentum_threshold_4h'] and change_1h > 0:
            signal = '🟢 MOMENTUM'
            signal_type = 'LONG'
            timeframe = '4H'
        # DIP BUY 1H
        elif change_1h < CONFIG['dip_threshold_1h']:
            signal = '🔵 DIP BUY'
            signal_type = 'LONG'
            timeframe = '1H'
        # DIP BUY 4H
        elif change_4h < CONFIG['dip_threshold_4h'] and change_1h < 0:
            signal = '🔵 DIP BUY'
            signal_type = 'LONG'
            timeframe = '4H'
        
        if signal:
            signals.append({
                'Time': now.strftime('%H:%M'),
                'Ticker': ticker,
                'Signal': signal,
                'TF': timeframe,
                'Price': f"${current:.2f}",
                '1H%': f"{change_1h:+.2f}%",
                '4H%': f"{change_4h:+.2f}%",
                'Day%': f"{change_day:+.2f}%",
                'RSI': round(rsi, 0) if not pd.isna(rsi) else '-',
                'Vol': f"{vol_ratio:.1f}x",
                'Future': asset.get('future', '-'),
            })
    
    return signals

def display_signals(signals, sound=False):
    """Muestra señales en consola"""
    now = datetime.now()
    
    print("\n" + "="*80)
    print(f"🔍 RADAR V7.1 — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    if signals:
        # Separar por tipo
        momentum = [s for s in signals if 'MOMENTUM' in s['Signal']]
        dips = [s for s in signals if 'DIP' in s['Signal']]
        
        if momentum:
            print(f"\n🟢 MOMENTUM ({len(momentum)} señales):")
            print("-"*80)
            df = pd.DataFrame(momentum)
            print(df.to_string(index=False))
        
        if dips:
            print(f"\n🔵 DIP BUY ({len(dips)} señales):")
            print("-"*80)
            df = pd.DataFrame(dips)
            print(df.to_string(index=False))
        
        # Alerta sonora
        if sound:
            play_alert()
            
        # Instrucciones
        print("\n" + "="*80)
        print("💡 ACCIÓN:")
        print("="*80)
        print("""
MOMENTUM: El activo está subiendo con fuerza
  → Entry: AHORA (o pullback menor)
  → Stop: -1.5% desde entrada
  → Target: +2-3% o resistencia próxima

DIP BUY: El activo cayó fuerte, posible rebote
  → Entry: Confirmar que deja de caer (vela verde)
  → Stop: Debajo del mínimo del día
  → Target: Recuperar 50% de la caída
""")
    else:
        print("\n❌ Sin señales en este momento")
        print("   Monitoreando movimientos >1% (1H) o >2% (4H)")
    
    print(f"\n⏰ Próximo escaneo en {CONFIG['scan_interval']//60} minutos...")

# ═══════════════════════════════════════════════════════════════════
# MONITOREO CONTINUO
# ═══════════════════════════════════════════════════════════════════

def watch_mode(sound=False):
    """Monitoreo continuo"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🔍 RADAR V7.1 — MODO VIGILANCIA 🔍                      ║
║                                                                   ║
║           Escaneando cada 5 minutos                               ║
║           Activos: """ + str(len(TICKERS)) + """ ETFs/Stocks                               ║
║           Ctrl+C para salir                                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        while True:
            signals = scan_intraday(sound=sound)
            display_signals(signals, sound=sound)
            time.sleep(CONFIG['scan_interval'])
    except KeyboardInterrupt:
        print("\n\n👋 Radar detenido.")

# ═══════════════════════════════════════════════════════════════════
# ESCANEO RÁPIDO DE FUTUROS
# ═══════════════════════════════════════════════════════════════════

def quick_scan():
    """Escaneo rápido de situación actual"""
    print("\n" + "="*80)
    print("📊 SITUACIÓN ACTUAL DEL MERCADO")
    print("="*80 + "\n")
    
    # Descargar datos de los principales
    main_tickers = ['QQQ', 'SPY', 'IWM', 'GLD', 'SLV', 'UNG', 'TQQQ', 'TNA']
    
    results = []
    
    for ticker in main_tickers:
        data = download_intraday(ticker, period='5d', interval='15m')
        
        if data is None or len(data) < 20:
            continue
        
        asset = ASSETS.get(ticker, {})
        current = data['Close'].iloc[-1]
        
        # Cambios
        change_1h = 0
        change_4h = 0
        change_day = 0
        
        if len(data) >= 5:
            change_1h = (current - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
        if len(data) >= 17:
            change_4h = (current - data['Close'].iloc[-17]) / data['Close'].iloc[-17] * 100
        
        today = data[data.index.date == data.index[-1].date()]
        if len(today) > 0:
            change_day = (current - today['Open'].iloc[0]) / today['Open'].iloc[0] * 100
        
        # High/Low del día
        if len(today) > 0:
            high = today['High'].max()
            low = today['Low'].min()
            range_pct = (high - low) / low * 100
            position = (current - low) / (high - low) * 100 if high != low else 50
        else:
            high = low = current
            range_pct = 0
            position = 50
        
        # RSI
        rsi = calculate_rsi(data['Close']).iloc[-1]
        
        # Determinar estado
        if change_1h > 1:
            status = '🟢 SUBIENDO'
        elif change_1h < -1:
            status = '🔴 CAYENDO'
        elif change_1h > 0.3:
            status = '📈 Alcista'
        elif change_1h < -0.3:
            status = '📉 Bajista'
        else:
            status = '➡️ Lateral'
        
        results.append({
            'Ticker': ticker,
            'Future': asset.get('future', '-'),
            'Price': f"${current:.2f}",
            '1H': f"{change_1h:+.2f}%",
            '4H': f"{change_4h:+.2f}%",
            'Day': f"{change_day:+.2f}%",
            'RSI': round(rsi, 0) if not pd.isna(rsi) else '-',
            'Pos%': f"{position:.0f}%",
            'Status': status,
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Resumen
    print("\n" + "="*80)
    print("📋 RESUMEN")
    print("="*80)
    
    subiendo = len([r for r in results if 'SUBIENDO' in r['Status']])
    cayendo = len([r for r in results if 'CAYENDO' in r['Status']])
    
    if subiendo > cayendo + 2:
        print("🟢 MERCADO ALCISTA — Buscar LONGS en momentum")
    elif cayendo > subiendo + 2:
        print("🔴 MERCADO BAJISTA — Buscar DIP BUYS o esperar")
    else:
        print("➡️ MERCADO MIXTO — Ser selectivo con las entradas")
    
    print(f"\n   Activos subiendo: {subiendo}")
    print(f"   Activos cayendo: {cayendo}")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Radar Intradía V7.1')
    parser.add_argument('--watch', action='store_true', help='Modo vigilancia continua')
    parser.add_argument('--scan', action='store_true', help='Escaneo único')
    parser.add_argument('--quick', action='store_true', help='Vista rápida del mercado')
    parser.add_argument('--sound', action='store_true', help='Alertas sonoras')
    parser.add_argument('--interval', type=int, default=300, help='Intervalo en segundos')
    args = parser.parse_args()
    
    if args.interval:
        CONFIG['scan_interval'] = args.interval
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🔍 HYPER-CONVEX RADAR V7.1 🔍                           ║
║           MONITOREO INTRADÍA EN TIEMPO REAL                       ║
║                                                                   ║
║           ✓ Datos cada 15 minutos                                 ║
║           ✓ Detecta movimientos >1% (1H) y >2% (4H)               ║
║           ✓ Señales MOMENTUM y DIP BUY                            ║
║           ✓ Monitorea {len(TICKERS)} activos                                ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.watch:
        watch_mode(sound=args.sound)
    elif args.scan:
        signals = scan_intraday(sound=args.sound)
        display_signals(signals, sound=args.sound)
    elif args.quick:
        quick_scan()
    else:
        print("\nUso:")
        print("  python radar.py --quick        → Vista rápida del mercado")
        print("  python radar.py --scan         → Escaneo único de señales")
        print("  python radar.py --watch        → Monitoreo continuo")
        print("  python radar.py --watch --sound → Con alertas sonoras")
        print("\nEjecutando vista rápida...\n")
        quick_scan()
        print("\n")
        signals = scan_intraday()
        display_signals(signals)

if __name__ == "__main__":
    main()
