#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🏆 HYPER-CONVEX V7.2 FINAL 🏆                           ║
║                                                                   ║
║           SCANNER DIARIO + RADAR INTRADÍA + BACKTEST              ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

BASADO EN BACKTEST DE 24 MESES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 1,912 trades analizados
• Win Rate: 52.2%
• Profit Factor: 1.24
• Return: +689% (24mo) = 344% anualizado

ESTRATEGIAS VALIDADAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIARIO:
  ✅ MOMENTUM: Comprar cuando sube >2% → +368%
  ✅ DIP BUY: Comprar cuando cae >3% → +320%

INTRADÍA:
  ✅ MOMENTUM: Comprar cuando sube >2% (4H) → Funciona
  ❌ DIP BUY: NO USAR intradía → Pierde dinero (-28%)

TOP TICKERS (probados 24 meses):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TSLA  +129%  │  TQQQ  +118%  │  TNA   +105%
  GME   +104%  │  SMCI  +74%   │  SPXL  +70%
  SLV   +24%   │  UNG   +14%   │  UPST  +52%

USO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python v72.py --daily           # Scanner diario (EOD)
  python v72.py --radar           # Radar intradía (tiempo real)
  python v72.py --watch           # Monitoreo continuo
  python v72.py --backtest        # Backtest completo
  python v72.py --backtest --months 12
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
# CONFIGURACIÓN OPTIMIZADA (basada en backtest 24 meses)
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # General
    'capital': 10000,
    'max_risk_per_trade': 0.02,
    'max_positions': 3,
    
    # Señales diarias
    'daily_momentum_threshold': 2.0,    # % para momentum
    'daily_dip_threshold': -3.0,        # % para dip buy
    
    # Señales intradía (solo momentum)
    'intraday_momentum_1h': 1.0,        # % en 1 hora
    'intraday_momentum_4h': 2.0,        # % en 4 horas
    
    # Risk management
    'stop_loss_pct': 2.0,               # % stop loss
    'take_profit_pct': 4.0,             # % take profit
    
    # Radar
    'scan_interval': 300,               # Segundos entre escaneos (5 min)
}

# ═══════════════════════════════════════════════════════════════════
# ACTIVOS OPTIMIZADOS (Top performers de backtest 24 meses)
# ═══════════════════════════════════════════════════════════════════

ASSETS = {
    # TIER 1: TOP PERFORMERS (+100% en 24mo)
    'TSLA': {'name': 'Tesla', 'tier': 1, 'return_24mo': 129.43, 'future': None},
    'TQQQ': {'name': 'Nasdaq 3x', 'tier': 1, 'return_24mo': 118.02, 'future': '/MNQ'},
    'TNA': {'name': 'Russell 3x', 'tier': 1, 'return_24mo': 104.77, 'future': '/M2K'},
    'GME': {'name': 'GameStop', 'tier': 1, 'return_24mo': 103.72, 'future': None},
    
    # TIER 2: BUENOS PERFORMERS (+50% en 24mo)
    'SMCI': {'name': 'Super Micro', 'tier': 2, 'return_24mo': 73.96, 'future': None},
    'SPXL': {'name': 'S&P 3x', 'tier': 2, 'return_24mo': 69.51, 'future': '/MES'},
    'UPST': {'name': 'Upstart', 'tier': 2, 'return_24mo': 52.00, 'future': None},
    
    # TIER 3: COMMODITIES (Buenos intradía)
    'SLV': {'name': 'Silver', 'tier': 3, 'return_24mo': 24.00, 'future': '/SIL'},
    'UNG': {'name': 'Natural Gas', 'tier': 3, 'return_24mo': 14.00, 'future': '/NG'},
    'GLD': {'name': 'Gold', 'tier': 3, 'return_24mo': 8.00, 'future': '/MGC'},
    
    # TIER 4: ÍNDICES (Para referencia)
    'QQQ': {'name': 'Nasdaq-100', 'tier': 4, 'return_24mo': 5.00, 'future': '/MNQ'},
    'SPY': {'name': 'S&P 500', 'tier': 4, 'return_24mo': 3.00, 'future': '/MES'},
}

TICKERS = list(ASSETS.keys())

# ═══════════════════════════════════════════════════════════════════
# FUNCIONES BASE
# ═══════════════════════════════════════════════════════════════════

def download_data(ticker, period='6mo', interval='1d'):
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

def calculate_rsi(prices, period=14):
    """RSI estándar"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def play_alert():
    """Alerta sonora (macOS/Linux)"""
    os.system('afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || echo -e "\a"')

# ═══════════════════════════════════════════════════════════════════
# 1. SCANNER DIARIO (MOMENTUM + DIP BUY)
# ═══════════════════════════════════════════════════════════════════

def scan_daily():
    """
    Scanner diario — Para ejecutar al cierre del mercado (EOD)
    
    MOMENTUM: Comprar si hoy subió >2%
    DIP BUY: Comprar si hoy cayó >3%
    """
    print("\n" + "="*70)
    print("📊 SCANNER DIARIO V7.2 — MOMENTUM + DIP BUY")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Señales: Momentum >+{CONFIG['daily_momentum_threshold']}% | Dip <{CONFIG['daily_dip_threshold']}%\n")
    
    signals = []
    
    for ticker in TICKERS:
        data = download_data(ticker, period='1mo', interval='1d')
        
        if data is None or len(data) < 5:
            continue
        
        asset = ASSETS[ticker]
        
        # Precio actual y cambio de hoy
        current = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change_today = (current - prev_close) / prev_close * 100
        
        # ATR para stops
        atr = calculate_atr(data['High'], data['Low'], data['Close']).iloc[-1]
        atr = atr if atr > 0 else current * 0.02
        
        # RSI
        rsi = calculate_rsi(data['Close']).iloc[-1]
        
        # Volumen
        vol_avg = data['Volume'].tail(20).mean()
        vol_ratio = data['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1
        
        signal = None
        strategy = None
        
        # MOMENTUM: Subió >2%
        if change_today > CONFIG['daily_momentum_threshold']:
            signal = '🟢 MOMENTUM'
            strategy = 'MOMENTUM'
            stop = current - atr * 1.5
            target = current + atr * 3
        
        # DIP BUY: Cayó >3%
        elif change_today < CONFIG['daily_dip_threshold']:
            signal = '🔵 DIP BUY'
            strategy = 'DIP_BUY'
            stop = current - atr * 2
            target = current + atr * 4
        
        if signal:
            signals.append({
                'Ticker': ticker,
                'Tier': f"T{asset['tier']}",
                'Signal': signal,
                'Price': f"${current:.2f}",
                'Change': f"{change_today:+.1f}%",
                'RSI': round(rsi, 0) if not pd.isna(rsi) else '-',
                'Vol': f"{vol_ratio:.1f}x",
                'Stop': f"${stop:.2f}",
                'Target': f"${target:.2f}",
                'Future': asset.get('future') or '-',
            })
    
    # Mostrar resultados
    if signals:
        # Separar por tipo
        momentum = [s for s in signals if 'MOMENTUM' in s['Signal']]
        dips = [s for s in signals if 'DIP' in s['Signal']]
        
        if momentum:
            print(f"🟢 MOMENTUM — {len(momentum)} señales:")
            print("-"*70)
            df = pd.DataFrame(momentum)
            df = df.sort_values('Tier')
            print(df.to_string(index=False))
        
        if dips:
            print(f"\n🔵 DIP BUY — {len(dips)} señales:")
            print("-"*70)
            df = pd.DataFrame(dips)
            df = df.sort_values('Tier')
            print(df.to_string(index=False))
        
        print("\n" + "="*70)
        print("💡 EJECUCIÓN MAÑANA:")
        print("="*70)
        print("""
• Entrada: En apertura del mercado (9:30 ET / 15:30 Madrid)
• Stop: Precio indicado (bracket order)
• Target: Precio indicado
• Salida alternativa: Cierre del día si no toca stop/target
""")
    else:
        print("❌ Sin señales diarias hoy")
        print(f"   Ningún activo movió +{CONFIG['daily_momentum_threshold']}% o {CONFIG['daily_dip_threshold']}%")
    
    return signals

# ═══════════════════════════════════════════════════════════════════
# 2. RADAR INTRADÍA (SOLO MOMENTUM)
# ═══════════════════════════════════════════════════════════════════

def scan_intraday():
    """
    Radar intradía — Para monitoreo en tiempo real
    
    SOLO MOMENTUM: Comprar cuando sube >1% (1H) o >2% (4H)
    NO DIP BUY: Probado que pierde dinero intradía
    """
    now = datetime.now()
    signals = []
    
    for ticker in TICKERS:
        data = download_data(ticker, period='5d', interval='15m')
        
        if data is None or len(data) < 20:
            continue
        
        asset = ASSETS[ticker]
        
        current = data['Close'].iloc[-1]
        
        # Cambio en 1 hora (4 velas de 15min)
        if len(data) >= 5:
            price_1h = data['Close'].iloc[-5]
            change_1h = (current - price_1h) / price_1h * 100
        else:
            change_1h = 0
        
        # Cambio en 4 horas (16 velas de 15min)
        if len(data) >= 17:
            price_4h = data['Close'].iloc[-17]
            change_4h = (current - price_4h) / price_4h * 100
        else:
            change_4h = 0
        
        # Cambio desde apertura
        today = data[data.index.date == data.index[-1].date()]
        if len(today) > 0:
            change_day = (current - today['Open'].iloc[0]) / today['Open'].iloc[0] * 100
        else:
            change_day = 0
        
        # RSI rápido
        rsi = calculate_rsi(data['Close'], period=7).iloc[-1]
        
        # Detectar SOLO MOMENTUM (no dip buy intradía)
        signal = None
        timeframe = None
        
        if change_1h > CONFIG['intraday_momentum_1h']:
            signal = '🟢 MOMENTUM'
            timeframe = '1H'
        elif change_4h > CONFIG['intraday_momentum_4h'] and change_1h > 0:
            signal = '🟢 MOMENTUM'
            timeframe = '4H'
        
        if signal:
            # ATR para stops
            atr = (data['High'] - data['Low']).tail(20).mean()
            stop = current - atr * 2
            target = current + atr * 3
            
            signals.append({
                'Time': now.strftime('%H:%M'),
                'Ticker': ticker,
                'Tier': f"T{asset['tier']}",
                'Signal': signal,
                'TF': timeframe,
                'Price': f"${current:.2f}",
                '1H': f"{change_1h:+.1f}%",
                '4H': f"{change_4h:+.1f}%",
                'Day': f"{change_day:+.1f}%",
                'RSI': round(rsi, 0) if not pd.isna(rsi) else '-',
                'Stop': f"${stop:.2f}",
                'Target': f"${target:.2f}",
                'Future': asset.get('future') or '-',
            })
    
    return signals

def display_radar(signals, sound=False):
    """Muestra señales del radar"""
    now = datetime.now()
    
    print("\n" + "="*70)
    print(f"🔍 RADAR INTRADÍA V7.2 — {now.strftime('%H:%M:%S')}")
    print("="*70)
    print("⚠️  SOLO MOMENTUM (Dip Buy deshabilitado intradía)\n")
    
    if signals:
        df = pd.DataFrame(signals)
        df = df.sort_values(['Tier', 'TF'])
        print(df.to_string(index=False))
        
        if sound:
            play_alert()
        
        print("\n" + "-"*70)
        print("💡 ACCIÓN INMEDIATA:")
        print("-"*70)
        print("""
• T1 señales: Prioridad máxima (TSLA, TQQQ, TNA, GME)
• Entry: AHORA o en siguiente pullback menor
• Stop: Precio indicado
• Target: Precio indicado o cierre de sesión
""")
    else:
        print("❌ Sin señales de momentum intradía")
        print(f"   Buscando: >+{CONFIG['intraday_momentum_1h']}% (1H) o >+{CONFIG['intraday_momentum_4h']}% (4H)")
    
    return signals

def watch_mode(sound=False):
    """Modo vigilancia continua"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║           🔍 RADAR V7.2 — MODO VIGILANCIA                         ║
║           Escaneando cada 5 minutos                               ║
║           SOLO MOMENTUM (Dip Buy deshabilitado)                   ║
║           Ctrl+C para salir                                       ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        while True:
            signals = scan_intraday()
            display_radar(signals, sound=sound)
            print(f"\n⏰ Próximo escaneo en {CONFIG['scan_interval']//60} minutos...")
            time.sleep(CONFIG['scan_interval'])
    except KeyboardInterrupt:
        print("\n\n👋 Radar detenido.")

# ═══════════════════════════════════════════════════════════════════
# 3. BACKTEST COMPLETO
# ═══════════════════════════════════════════════════════════════════

def backtest_full(months=6):
    """
    Backtest completo de ambas estrategias
    
    DIARIO: MOMENTUM + DIP BUY
    """
    print("\n" + "="*70)
    print(f"📈 BACKTEST V7.2 — {months} MESES")
    print("="*70)
    print(f"Activos: {len(TICKERS)}")
    print(f"Estrategias: MOMENTUM (>+{CONFIG['daily_momentum_threshold']}%) + DIP BUY (<{CONFIG['daily_dip_threshold']}%)\n")
    
    all_trades = []
    
    for ticker in TICKERS:
        data = download_data(ticker, period=f'{months}mo', interval='1d')
        
        if data is None or len(data) < 30:
            continue
        
        asset = ASSETS[ticker]
        
        # Calcular cambio diario
        data['Change'] = data['Close'].pct_change() * 100
        
        signals_mom = len(data[data['Change'] > CONFIG['daily_momentum_threshold']])
        signals_dip = len(data[data['Change'] < CONFIG['daily_dip_threshold']])
        
        print(f"✅ {ticker:5} (T{asset['tier']}): {len(data)} días | Mom: {signals_mom}, Dip: {signals_dip}")
        
        # Generar trades
        for i in range(21, len(data) - 1):
            change = data['Change'].iloc[i]
            
            if pd.isna(change):
                continue
            
            entry_price = data['Close'].iloc[i]
            exit_price = data['Close'].iloc[i + 1]
            entry_date = data.index[i]
            strategy = None
            
            # MOMENTUM
            if change > CONFIG['daily_momentum_threshold']:
                strategy = 'MOMENTUM'
            # DIP BUY
            elif change < CONFIG['daily_dip_threshold']:
                strategy = 'DIP_BUY'
            
            if strategy:
                pnl = (exit_price - entry_price) / entry_price * 100
                
                all_trades.append({
                    'Ticker': ticker,
                    'Tier': asset['tier'],
                    'Strategy': strategy,
                    'Date': entry_date.strftime('%Y-%m-%d'),
                    'Entry': round(entry_price, 2),
                    'Exit': round(exit_price, 2),
                    'P&L%': round(pnl, 2),
                })
    
    # Resultados
    print("\n" + "="*70)
    print("📊 RESULTADOS")
    print("="*70)
    
    if all_trades:
        df = pd.DataFrame(all_trades)
        
        total = len(df)
        wins = len(df[df['P&L%'] > 0])
        losses = total - wins
        win_rate = wins / total * 100
        
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
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  🏆 RESULTADOS BACKTEST V7.2                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Periodo: {months} meses | Activos: {len(TICKERS)}                            ║
║  TRADES: {total:<55}║
╠══════════════════════════════════════════════════════════════════╣
║  WIN RATE: {win_rate:.1f}% ({wins}W / {losses}L)                            ║
║  AVG WIN: +{avg_win:.2f}% | AVG LOSS: {avg_loss:.2f}%                       ║
║  PROFIT FACTOR: {profit_factor:.2f}                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  RETURN TOTAL: {total_return:.1f}%                                        ║
║  RETURN ANUALIZADO: {annual_return:.1f}%                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  MOMENTUM: {len(momentum)} trades → {momentum['P&L%'].sum():.1f}%                          ║
║  DIP BUY: {len(dip_buy)} trades → {dip_buy['P&L%'].sum():.1f}%                           ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        # Por Tier
        print("📊 POR TIER:")
        by_tier = df.groupby('Tier').agg({
            'P&L%': ['sum', 'count', 'mean']
        }).round(2)
        by_tier.columns = ['Total%', 'Trades', 'Avg%']
        by_tier['WinRate'] = df.groupby('Tier').apply(
            lambda x: len(x[x['P&L%'] > 0]) / len(x) * 100
        ).round(1)
        print(by_tier.to_string())
        
        # Por Ticker
        print("\n📊 POR TICKER:")
        by_ticker = df.groupby('Ticker').agg({
            'P&L%': ['sum', 'count', 'mean']
        }).round(2)
        by_ticker.columns = ['Total%', 'Trades', 'Avg%']
        by_ticker = by_ticker.sort_values('Total%', ascending=False)
        print(by_ticker.to_string())
        
        # Por estrategia
        print("\n📊 MOMENTUM vs DIP BUY:")
        for strat in ['MOMENTUM', 'DIP_BUY']:
            subset = df[df['Strategy'] == strat]
            if len(subset) > 0:
                wr = len(subset[subset['P&L%'] > 0]) / len(subset) * 100
                print(f"  {strat:10}: {len(subset):3} trades | WR: {wr:.1f}% | Return: {subset['P&L%'].sum():.1f}%")
        
        return df
    else:
        print("❌ No se generaron trades")
        return None

# ═══════════════════════════════════════════════════════════════════
# 4. VISTA RÁPIDA DEL MERCADO
# ═══════════════════════════════════════════════════════════════════

def quick_view():
    """Vista rápida del estado actual del mercado"""
    print("\n" + "="*70)
    print("📊 ESTADO ACTUAL DEL MERCADO")
    print("="*70 + "\n")
    
    results = []
    
    for ticker in TICKERS:
        data = download_data(ticker, period='5d', interval='1d')
        
        if data is None or len(data) < 3:
            continue
        
        asset = ASSETS[ticker]
        current = data['Close'].iloc[-1]
        
        # Cambios
        change_1d = (current - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
        change_5d = (current - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
        
        # RSI
        rsi = calculate_rsi(data['Close']).iloc[-1]
        
        # Status
        if change_1d > 2:
            status = '🟢 +2%'
        elif change_1d < -3:
            status = '🔵 DIP'
        elif change_1d > 0:
            status = '📈 Up'
        elif change_1d < 0:
            status = '📉 Down'
        else:
            status = '➡️ Flat'
        
        results.append({
            'Ticker': ticker,
            'Tier': f"T{asset['tier']}",
            'Price': f"${current:.2f}",
            '1D': f"{change_1d:+.1f}%",
            '5D': f"{change_5d:+.1f}%",
            'RSI': round(rsi, 0) if not pd.isna(rsi) else '-',
            'Status': status,
            'Future': asset.get('future') or '-',
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Tier')
    print(df.to_string(index=False))
    
    # Resumen
    signals_mom = len([r for r in results if '+2%' in r['Status']])
    signals_dip = len([r for r in results if 'DIP' in r['Status']])
    
    print(f"\n📋 Señales MOMENTUM (+2%): {signals_mom}")
    print(f"📋 Señales DIP BUY (-3%): {signals_dip}")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Hyper-Convex V7.2 FINAL')
    parser.add_argument('--daily', action='store_true', help='Scanner diario (EOD)')
    parser.add_argument('--radar', action='store_true', help='Radar intradía')
    parser.add_argument('--watch', action='store_true', help='Monitoreo continuo')
    parser.add_argument('--backtest', action='store_true', help='Backtest completo')
    parser.add_argument('--quick', action='store_true', help='Vista rápida')
    parser.add_argument('--months', type=int, default=6, help='Meses para backtest')
    parser.add_argument('--sound', action='store_true', help='Alertas sonoras')
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🏆 HYPER-CONVEX V7.2 FINAL 🏆                           ║
║                                                                   ║
║           SCANNER DIARIO + RADAR INTRADÍA                         ║
║           Optimizado con 24 meses de backtest                     ║
║                                                                   ║
║           📊 24mo: +689% return, 52% WR, 1.24 PF                  ║
║                                                                   ║
║           DIARIO: MOMENTUM + DIP BUY (ambos funcionan)            ║
║           INTRADÍA: SOLO MOMENTUM (dip buy pierde)                ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Activos: {len(TICKERS)} | Capital: ${CONFIG['capital']:,}")
    
    if args.daily:
        scan_daily()
    elif args.radar:
        signals = scan_intraday()
        display_radar(signals, sound=args.sound)
    elif args.watch:
        watch_mode(sound=args.sound)
    elif args.backtest:
        backtest_full(args.months)
    elif args.quick:
        quick_view()
    else:
        print("\n" + "="*70)
        print("📖 USO")
        print("="*70)
        print("""
  python v72.py --quick              # Vista rápida del mercado
  python v72.py --daily              # Scanner diario (para EOD)
  python v72.py --radar              # Radar intradía (señales ahora)
  python v72.py --watch              # Monitoreo continuo
  python v72.py --watch --sound      # Con alertas sonoras
  python v72.py --backtest           # Backtest 6 meses
  python v72.py --backtest --months 12  # Backtest 12 meses
""")
        print("Mostrando vista rápida del mercado...\n")
        quick_view()
        print("\n")
        scan_daily()

if __name__ == "__main__":
    main()
