#!/usr/bin/env python3
"""
MOMENTUM BREAKOUT SCANNER v4.0 — Alineado con v12 (EU options + Gold overlay)

Radar en tiempo real para la estrategia Momentum Breakout v12 (Fat Tails).
Escanea 225 tickers (acciones + ETFs + commodities + renta fija) con datos DIARIOS.

USO:
  python run_scanner.py --scan              # Escaneo unico
  python run_scanner.py --watch             # Monitoreo continuo (cada 5 min)
  python run_scanner.py --scan --category US_TECH   # Solo una categoria
  python run_scanner.py --scan --ticker NVDA,TSLA   # Tickers especificos
  python run_scanner.py --capital 25000     # Capital personalizado

ESTRATEGIA v12:
  - Solo LONGS (shorts destruyen el portfolio)
  - Filtro macro: SPY > SMA50 (no entrar en correcciones)
  - Ranking multi-factor: KER + RSI + Volume + Breakout + ATR%
  - Position sizing: 2% equity risk/trade (inverse volatility)
  - Trailing Chandelier 4xATR activado a +2R
  - Time exit 8d trailing only (nunca forzado)
  - Max 10 posiciones simultaneas (225 tickers)
  - Opciones CALL US: max 2 slots, spread 3% (via IBKR)
  - Opciones CALL EU: max 2 slots SEPARADOS, spread 10% (via DEGIRO)
  - Gold overlay: 30% equity permanente en GLD + cash idle en GLD
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

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_breakout import (
    MomentumEngine,
    calculate_atr,
    calculate_position_size,
    ASSETS,
    TICKERS,
)


# =============================================================================
# v12 — OPCIONES CALL US + EU (slots separados)
# =============================================================================

# Tickers elegibles para opciones US (via IBKR)
OPTIONS_ELIGIBLE_US = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
    'ORCL', 'CRM', 'ADBE', 'AMD', 'INTC', 'CSCO', 'QCOM',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'AXP',
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT',
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT',
    'QQQ', 'SPY', 'IWM', 'DIA', 'GLD', 'SLV', 'XLE', 'TLT',
    'TQQQ', 'SPXL', 'TNA', 'BITO',
]

# Tickers elegibles para opciones EU (via DEGIRO — Eurex, Euronext, LSE, SIX, OMX)
OPTIONS_ELIGIBLE_EU = [
    # Eurex — Alemania
    'SIE.DE', 'ALV.DE', 'DTE.DE', 'MUV2.DE', 'BAS.DE',
    'BMW.DE', 'MBG.DE', 'ADS.DE', 'IFX.DE',
    # Euronext — Francia
    'OR.PA', 'MC.PA', 'SAN.PA', 'AI.PA', 'BNP.PA',
    'SU.PA', 'AIR.PA', 'CS.PA', 'DG.PA', 'RI.PA',
    # Euronext — Holanda/Belgica
    'INGA.AS', 'PHIA.AS', 'AD.AS', 'KBC.BR', 'ABI.BR',
    # Borsa Italiana
    'ENEL.MI', 'ISP.MI', 'UCG.MI', 'ENI.MI',
    # LSE — Reino Unido
    'ULVR.L', 'LSEG.L', 'BATS.L', 'DGE.L',
    # SIX — Suiza (Eurex)
    'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'ZURN.SW', 'ABBN.SW',
    # Nordicos (OMX)
    'ERIC-B.ST',
]

# Compatibilidad con codigo que usa OPTIONS_ELIGIBLE
OPTIONS_ELIGIBLE = OPTIONS_ELIGIBLE_US

# Spreads por region
US_SPREAD_PCT = 3.0      # Spread US: ~3% (muy liquido)
EU_SPREAD_PCT = 10.0     # Spread EU: ~10% (menos liquido, confirmado DEGIRO)

OPTIONS_CONFIG = {
    'option_dte': 120,
    'option_itm_pct': 0.05,          # 5% ITM (strike = spot * 0.95)
    'option_close_dte': 45,          # Cerrar a 45 DTE restantes
    'option_max_ivr': 40,            # Solo comprar si IVR < 40
    'option_ivr_window': 252,        # Ventana IV Rank: 1 ano
    'option_position_pct': 0.14,     # 14% del equity por opcion
    'max_us_option_positions': 2,    # Max 2 opciones US simultaneas (IBKR)
    'max_eu_option_positions': 2,    # Max 2 opciones EU simultaneas (DEGIRO) — SEPARADOS
}

# Gold overlay v10: 30% equity permanente en GLD + cash idle en GLD
GOLD_OVERLAY_PCT = 0.30


def historical_volatility(close_prices, window=30):
    """Calcula volatilidad historica anualizada."""
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


def iv_rank(hvol_series, window=252):
    """
    IV Rank: percentil de la IV actual respecto a los ultimos 'window' dias.
    IVR = (IV_actual - IV_min) / (IV_max - IV_min) * 100
    Bajo = opciones baratas. Alto = opciones caras.
    """
    hist = hvol_series.dropna()
    if len(hist) < 20:
        return None  # sin datos suficientes
    recent = hist.iloc[-window:] if len(hist) >= window else hist
    iv_now = recent.iloc[-1]
    iv_min = recent.min()
    iv_max = recent.max()
    if iv_max == iv_min:
        return 50.0
    return (iv_now - iv_min) / (iv_max - iv_min) * 100


# =============================================================================
# CONFIGURACION v3.0 — Alineada con Backtest v6 (Feb 2026)
# =============================================================================

CONFIG = {
    # Capital
    'capital': 10000,

    # Position sizing — volatility-based
    'target_risk_per_trade_pct': 2.0,  # 2% del equity en riesgo por trade
    'max_positions': 10,  # Optimizado v8: 10 > 7 con 225 tickers (PF 2.89, +34.6% anual)

    # Senales (identicas al backtest v6)
    'ker_threshold': 0.40,
    'volume_threshold': 1.3,
    'rsi_threshold': 50,
    'rsi_max': 75,
    'breakout_period': 20,
    'longs_only': True,

    # Stop management — Chandelier trailing
    'emergency_stop_pct': 0.15,     # -15% desde entrada
    'trail_trigger_r': 2.0,         # Activar trailing a +2R
    'trail_atr_mult': 4.0,         # Chandelier 4xATR

    # Time management
    'max_hold_bars': 12,            # 12 dias (solo perdedores salen por tiempo)

    # Filtro macro
    'use_macro_filter': True,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,

    # Costes
    'slippage_pct': 0.10,

    # Scanner
    'scan_interval': 300,  # 5 minutos entre escaneos en watch mode

    # Tickers: todo el universo (excluir crypto)
    'test_tickers': [t for t, v in ASSETS.items()
                     if not v.get('is_crypto', False)
                     and not t.endswith('USDT')],
}


# =============================================================================
# DATA DOWNLOAD — Datos DIARIOS (alineado con backtest v6)
# =============================================================================

def download_data(ticker: str, period: str = '14mo') -> pd.DataFrame:
    """Descarga datos DIARIOS de yfinance."""
    try:
        yf_ticker = ticker
        if ticker.endswith('USDT'):
            yf_ticker = ticker.replace('USDT', '-USD')

        df = yf.download(yf_ticker, period=period, interval='1d', progress=False)

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df if len(df) >= 50 else None

    except Exception:
        return None


def download_batch(tickers: list, period: str = '14mo') -> dict:
    """Descarga datos para multiples tickers. Periodo 14mo para IVR (252 dias)."""
    data = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        print(f"\r  Descargando {i+1}/{total}: {ticker:12}", end='', flush=True)
        df = download_data(ticker, period)
        if df is not None:
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            df['HVOL'] = historical_volatility(df['Close'], 30)
            data[ticker] = df

    print(f"\r  Descargados {len(data)}/{total} tickers OK" + " " * 30)
    return data


# =============================================================================
# FILTRO MACRO — SPY > SMA50
# =============================================================================

def check_macro_filter(data: dict) -> bool:
    """
    Verifica si SPY esta por encima de su SMA50.
    True = mercado alcista, se puede operar.
    False = mercado en correccion, no abrir posiciones.
    """
    if not CONFIG.get('use_macro_filter', False):
        return True

    macro_ticker = CONFIG.get('macro_ticker', 'SPY')
    if macro_ticker not in data:
        print(f"  WARN: {macro_ticker} no disponible para filtro macro — permitiendo operaciones")
        return True

    df = data[macro_ticker]
    sma_period = CONFIG.get('macro_sma_period', 50)
    sma = df['Close'].rolling(window=sma_period).mean()

    if len(sma) < sma_period:
        return True

    # Filtro macro: Close[-1] vs SMA[-1] (ultimo dato disponible)
    # Equivale a backtest prev_dates[-2] porque:
    #   - Backtest: current_date=entry_day, prev_dates[-2]=dia del breakout (2 dias antes)
    #   - Scanner: ultimo dato=dia del breakout (mercado ya cerro), [-1]=ese mismo dia
    # Ambos leen el cierre del dia del breakout.
    if len(df) < 1 or len(sma) < 1:
        return True

    macro_close = df['Close'].iloc[-1]
    macro_sma = sma.iloc[-1]

    if pd.isna(macro_sma):
        return True

    return macro_close > macro_sma


# =============================================================================
# RANKING MULTI-FACTOR v6
# =============================================================================

def calculate_composite_score(ticker: str, df: pd.DataFrame, meta: dict, idx: int) -> float:
    """
    Score compuesto multi-factor (identico al backtest v6).

    Score = 0.30 x KER           (tendencia limpia)
          + 0.20 x RSI_norm      (momentum)
          + 0.20 x Vol_norm      (confirmacion volumen)
          + 0.15 x Breakout_str  (fuerza del breakout)
          + 0.15 x ATR%          (potencial de recorrido)
    """
    # 1. KER (0-1)
    ker_val = meta['ker'].iloc[idx] if idx >= 0 else 0

    # 2. RSI normalizado (0-1, donde 50→0, 75→1)
    rsi_val = meta['rsi'].iloc[idx] if idx >= 0 else 50
    rsi_score = max(0, min(1, (rsi_val - CONFIG['rsi_threshold']) / (CONFIG['rsi_max'] - CONFIG['rsi_threshold'])))

    # 3. Volume ratio normalizado (1.0→0, 3.0→1)
    vol_val = meta['vol_ratio'].iloc[idx] if idx >= 0 else 1.0
    vol_score = min(1, max(0, (vol_val - 1.0) / 2.0))

    # 4. Breakout strength (distancia del close sobre rolling high)
    if idx >= 1:
        close_val = df['Close'].iloc[idx]
        bp = CONFIG['breakout_period']
        rolling_high = df['High'].iloc[max(0, idx - bp):idx].max()
        breakout_pct = (close_val - rolling_high) / rolling_high if rolling_high > 0 else 0
        breakout_score = min(1, max(0, breakout_pct / 0.05))
    else:
        breakout_score = 0

    # 5. ATR% (volatilidad relativa → potencial de recorrido)
    price = df['Close'].iloc[idx] if idx >= 0 else 1
    atr_val = df['ATR'].iloc[idx] if 'ATR' in df.columns and idx >= 0 else 0
    atr_pct = atr_val / price if price > 0 else 0
    atr_score = min(1, atr_pct / 0.04)

    # SCORE COMPUESTO
    composite = (
        0.30 * ker_val +
        0.20 * rsi_score +
        0.20 * vol_score +
        0.15 * breakout_score +
        0.15 * atr_score
    )

    return composite


# =============================================================================
# POSITION SIZING — Inverse Volatility (v6)
# =============================================================================

def calculate_position_size_v5(equity: float, current_atr: float, price: float) -> dict:
    """
    Position sizing basado en riesgo fijo por trade (v6).

    R = 2 x ATR (riesgo por unidad)
    units = (equity * 2%) / R
    notional = units * price
    Tope: equity/7 por posicion (v6)

    Activos volatiles → posiciones pequenas
    Activos estables → posiciones grandes
    """
    risk_pct = CONFIG['target_risk_per_trade_pct'] / 100
    R = current_atr * 2.0
    if R <= 0 or price <= 0:
        return {'units': 0, 'notional': 0, 'R': 0, 'risk_eur': 0}

    dollar_risk = equity * risk_pct
    units = dollar_risk / R
    notional = units * price

    # Cap: no mas de equity / max_positions * 2
    max_notional = equity / CONFIG['max_positions'] * 2
    if notional > max_notional:
        notional = max_notional
        units = notional / price

    return {
        'units': units,
        'notional': notional,
        'R': R,
        'risk_eur': dollar_risk,
    }


# =============================================================================
# SCANNER — Escaneo de senales activas
# =============================================================================

def _is_asia_market(ticker: str) -> bool:
    """
    Determina si un ticker opera en un mercado asiatico que ya ha cerrado
    cuando el scanner se ejecuta (~8:00 CET).

    Mercados afectados (cierre antes de 8 CET):
      - Japon (.T) — cierra 8:00 CET / 15:00 JST
      - Australia (.AX) — cierra 7:00 CET / 16:00 AEDT
      - Hong Kong (.HK) — cierra 8:00 CET / 16:00 HKT

    Estos mercados ya tienen el dato de HOY completo, mientras US/EU
    solo tienen hasta ayer. Esto afecta el offset de senal necesario.

    Nota: ADRs chinos (BABA, JD, PDD, BIDU) cotizan en NYSE → NO son asia.
    """
    return ticker.endswith('.T') or ticker.endswith('.AX') or ticker.endswith('.HK')


def scan_signals(data: dict, capital: float = 10000) -> dict:
    """
    Escanea todos los tickers buscando senales activas.

    Retorna diccionario con:
    - signals: lista de senales activas (rankeadas por score multi-factor)
    - watchlist: tickers cerca de breakout
    - macro_ok: estado del filtro macro
    """
    engine = MomentumEngine(
        ker_threshold=CONFIG['ker_threshold'],
        volume_threshold=CONFIG['volume_threshold'],
        rsi_threshold=CONFIG['rsi_threshold'],
        rsi_max=CONFIG.get('rsi_max', 75),
        breakout_period=CONFIG['breakout_period'],
        longs_only=CONFIG.get('longs_only', True)
    )

    # Filtro macro
    macro_ok = check_macro_filter(data)

    signals = []
    watchlist = []

    for ticker, df in data.items():
        try:
            asset = ASSETS.get(ticker, {})
            category = asset.get('category', 'UNKNOWN')

            meta = engine.generate_signals_with_metadata(df)

            # --- Offset de senal por zona horaria ---
            # A las ~8:00 CET (hora tipica del scanner):
            #   US/EU: ultimo dato = ayer → signals[-1] = senal de ayer = T+2 ✓
            #   Asia:  ultimo dato = hoy  → signals[-1] = senal de hoy = T+3 ✗
            #          → necesitan signals[-2] = senal de ayer = T+2 ✓
            asia = _is_asia_market(ticker)
            sig_offset = -2 if asia else -1

            if len(meta['signal']) < abs(sig_offset):
                continue
            last_signal = meta['signal'].iloc[sig_offset]
            # Indice para score/metadata: alineado con la senal leida
            last_idx = len(df) + sig_offset

            if last_signal == 1:  # Solo LONG
                # Precio/ATR: ultimo dato disponible (para sizing y niveles)
                current_price = df['Close'].iloc[-1]
                current_atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else meta['atr'].iloc[-1]

                # Score multi-factor v6 (usa last_idx = dia de la senal)
                composite_score = calculate_composite_score(ticker, df, meta, last_idx)

                # Position sizing v6
                size_info = calculate_position_size_v5(capital, current_atr, current_price)

                # Calcular niveles
                R = current_atr * 2.0
                emergency_stop = current_price * (1 - CONFIG['emergency_stop_pct'])
                trail_activation = current_price + (R * CONFIG['trail_trigger_r'])
                chandelier_initial = current_price - (current_atr * CONFIG['trail_atr_mult'])

                # v12: Evaluar elegibilidad para opciones CALL (US y EU separados)
                option_us = ticker in OPTIONS_ELIGIBLE_US
                option_eu = ticker in OPTIONS_ELIGIBLE_EU
                option_eligible = option_us or option_eu
                option_region = 'US' if option_us else ('EU' if option_eu else None)
                option_spread = US_SPREAD_PCT if option_us else (EU_SPREAD_PCT if option_eu else 0)
                option_ivr = None
                option_recommended = False
                if option_eligible and 'HVOL' in df.columns:
                    hvol = df['HVOL']
                    option_ivr_val = iv_rank(hvol, OPTIONS_CONFIG['option_ivr_window'])
                    if option_ivr_val is not None:
                        option_ivr = round(option_ivr_val, 1)
                        option_recommended = option_ivr < OPTIONS_CONFIG['option_max_ivr']

                # Fecha de la senal (para referencia)
                signal_date = str(df.index[last_idx].date()) if last_idx >= 0 else 'N/A'

                signals.append({
                    'ticker': ticker,
                    'name': asset.get('name', ticker),
                    'category': category,
                    'signal': 'LONG',
                    'score': round(composite_score * 100, 1),  # 0-100
                    'price': round(current_price, 2),
                    'atr': round(current_atr, 2),
                    'atr_pct': round(current_atr / current_price * 100, 2),
                    'ker': round(meta['ker'].iloc[last_idx], 3),
                    'rsi': round(meta['rsi'].iloc[last_idx], 1),
                    'vol_ratio': round(meta['vol_ratio'].iloc[last_idx], 2),
                    'units': round(size_info['units'], 4),
                    'notional': round(size_info['notional'], 0),
                    'risk_eur': round(size_info['risk_eur'], 0),
                    'R': round(R, 2),
                    'emergency_stop': round(emergency_stop, 2),
                    'trail_activation': round(trail_activation, 2),
                    'chandelier': round(chandelier_initial, 2),
                    'signal_date': signal_date,
                    'asia_offset': asia,
                    # v12 opciones (US + EU separados)
                    'option_eligible': option_eligible,
                    'option_region': option_region,
                    'option_spread': option_spread,
                    'option_ivr': option_ivr,
                    'option_recommended': option_recommended,
                })

            # Watchlist: tickers trending cerca de breakout
            elif meta.get('is_trending') is not None and last_idx >= 0:
                is_trending = meta['is_trending'].iloc[last_idx] if 'is_trending' in meta else False
                if is_trending:
                    current_price = df['Close'].iloc[last_idx]
                    bp = CONFIG['breakout_period']
                    if last_idx >= bp:
                        rolling_high = df['High'].iloc[max(0, last_idx - bp):last_idx].max()
                        dist_to_breakout = (rolling_high - current_price) / current_price * 100

                        if 0 < dist_to_breakout < 2.0:  # Dentro del 2% de breakout
                            watchlist.append({
                                'ticker': ticker,
                                'name': asset.get('name', ticker),
                                'category': category,
                                'price': round(current_price, 2),
                                'breakout_level': round(rolling_high, 2),
                                'distance_pct': round(dist_to_breakout, 2),
                                'ker': round(meta['ker'].iloc[last_idx], 3),
                                'rsi': round(meta['rsi'].iloc[last_idx], 1),
                            })

        except Exception:
            continue

    # Ordenar por score (mayor primero)
    signals.sort(key=lambda x: x['score'], reverse=True)
    watchlist.sort(key=lambda x: x['distance_pct'])

    return {
        'signals': signals,
        'watchlist': watchlist,
        'macro_ok': macro_ok,
    }


# =============================================================================
# DISPLAY — Mostrar resultados del escaneo
# =============================================================================

def display_signals(result: dict, capital: float = 10000):
    """Muestra los resultados del escaneo en formato tabla."""
    now = datetime.now()
    signals = result['signals']
    watchlist = result['watchlist']
    macro_ok = result['macro_ok']

    macro_ticker = CONFIG.get('macro_ticker', 'SPY')
    macro_sma = CONFIG.get('macro_sma_period', 50)
    macro_status = f"BULL ({macro_ticker} > SMA{macro_sma})" if macro_ok else f"BEAR ({macro_ticker} < SMA{macro_sma})"

    print(f"""
{'='*80}
  MOMENTUM BREAKOUT SCANNER v4.0 (v12) — {now.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
  Capital: EUR {capital:,.0f} | Max {CONFIG['max_positions']} posiciones | Risk: {CONFIG['target_risk_per_trade_pct']}%/trade
  Filtro macro: {macro_status}
  Opciones: US {len(OPTIONS_ELIGIBLE_US)} tickers (max 2 slots, spread 3%) | EU {len(OPTIONS_ELIGIBLE_EU)} tickers (max 2 slots, spread 10%)
  Gold overlay: {GOLD_OVERLAY_PCT*100:.0f}% equity permanente en GLD + cash idle en GLD
  Datos: DIARIOS | Solo LONGS | Trailing {CONFIG['trail_atr_mult']:.0f}xATR a +2R
{'='*80}
""")

    # Filtro macro
    if not macro_ok:
        print("  ⚠️  MERCADO BEAR — No se abren posiciones nuevas")
        print(f"  ({macro_ticker} esta por debajo de su SMA{macro_sma})")
        print()

    # Senales activas
    if signals:
        n_show = min(len(signals), CONFIG['max_positions'])
        print(f"  🎯 SENALES ACTIVAS: {len(signals)} encontradas (Top {n_show} por ranking multi-factor)")
        print(f"  {'-'*76}")
        print()

        for rank, s in enumerate(signals[:n_show], 1):
            if macro_ok:
                action_str = "✅ EJECUTAR"
            else:
                action_str = "⚠️ NO EJECUTAR (bear)"
            tz_tag = " [ASIA→T-2]" if s.get('asia_offset') else ""
            sig_date = s.get('signal_date', '')
            print(f"  #{rank} LONG {s['ticker']:10} | Score: {s['score']:.0f}/100 | {action_str}{tz_tag}")
            print(f"  {'':>5} Senal: {sig_date} | Precio: ${s['price']:>10,.2f} | ATR: ${s['atr']:.2f} ({s['atr_pct']:.1f}%)")
            print(f"  {'':>5} KER: {s['ker']:.3f} | RSI: {s['rsi']:.0f} | Vol: {s['vol_ratio']:.1f}x")
            print(f"  {'':>5} Posicion: {s['units']:.4f} unid = EUR {s['notional']:,.0f} | Riesgo: EUR {s['risk_eur']:,.0f}")
            print(f"  {'':>5} Niveles → Entry: ${s['price']:.2f} | Emergency: ${s['emergency_stop']:.2f} | "
                  f"Trail activa a: ${s['trail_activation']:.2f}")
            # v12: Indicacion de opciones CALL (US / EU)
            if s.get('option_eligible'):
                region = s.get('option_region', '??')
                spread = s.get('option_spread', 0)
                broker = 'IBKR' if region == 'US' else 'DEGIRO'
                ivr_str = f"IVR: {s['option_ivr']:.0f}" if s['option_ivr'] is not None else "IVR: N/A"
                if s.get('option_recommended'):
                    print(f"  {'':>5} 📋 CALL {region} 5% ITM 120DTE | {ivr_str} | spread {spread:.0f}% | {broker} | RECOMENDADA")
                else:
                    print(f"  {'':>5} 📋 CALL {region} elegible | {ivr_str} | spread {spread:.0f}% | {broker} | Caras (IVR≥40) → accion")
            print()

        # Senales que no entran en el top N
        if len(signals) > n_show:
            print(f"  --- {len(signals) - n_show} senales adicionales fuera del ranking ---")
            for s in signals[n_show:]:
                opt_tag = " [CALL]" if s.get('option_recommended') else ""
                print(f"      {s['ticker']:10} Score: {s['score']:.0f} | ${s['price']:>9,.2f} | "
                      f"KER {s['ker']:.3f} RSI {s['rsi']:.0f}{opt_tag}")
            print()

    else:
        print("  ❌ No hay senales activas hoy")
        print()

    # Watchlist
    if watchlist:
        print(f"  👀 WATCHLIST — Cerca de breakout ({len(watchlist)} tickers)")
        print(f"  {'-'*76}")
        print(f"  {'Ticker':<12} {'Precio':>10} {'Breakout':>10} {'Dist':>6} {'KER':>6} {'RSI':>5}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*5}")

        for w in watchlist[:10]:
            print(f"  {w['ticker']:<12} ${w['price']:>9,.2f} ${w['breakout_level']:>9,.2f} "
                  f"{w['distance_pct']:>5.2f}% {w['ker']:>5.3f} {w['rsi']:>4.0f}")
        print()

    # v12: Resumen de opciones (US + EU separados)
    opt_recommended = [s for s in signals if s.get('option_recommended')]
    opt_us = [s for s in opt_recommended if s.get('option_region') == 'US']
    opt_eu = [s for s in opt_recommended if s.get('option_region') == 'EU']
    opt_eligible = [s for s in signals if s.get('option_eligible')]
    if opt_eligible:
        print(f"  📈 OPCIONES v12: {len(opt_recommended)} recomendadas (IVR<40) de {len(opt_eligible)} elegibles")
        if opt_us:
            max_us = OPTIONS_CONFIG['max_us_option_positions']
            print(f"      US (max {max_us} slots, IBKR):")
            for s in opt_us[:max_us]:
                strike = s['price'] * (1 - OPTIONS_CONFIG['option_itm_pct'])
                print(f"        {s['ticker']:8} CALL K=${strike:.2f} (5% ITM) ~120DTE | IVR={s['option_ivr']:.0f} | spread 3%")
        if opt_eu:
            max_eu = OPTIONS_CONFIG['max_eu_option_positions']
            print(f"      EU (max {max_eu} slots, DEGIRO):")
            for s in opt_eu[:max_eu]:
                strike = s['price'] * (1 - OPTIONS_CONFIG['option_itm_pct'])
                print(f"        {s['ticker']:8} CALL K={strike:.2f} (5% ITM) ~120DTE | IVR={s['option_ivr']:.0f} | spread 10%")
        print()

    # Resumen
    print(f"  📊 RESUMEN: {len(signals)} senales | {len(watchlist)} en watchlist | "
          f"Macro: {'BULL ✅' if macro_ok else 'BEAR ⚠️'}")
    print()


# =============================================================================
# WATCH MODE — Monitoreo continuo
# =============================================================================

def watch_mode(tickers: list, capital: float, interval: int = 300):
    """Modo monitoreo continuo. Escanea cada N segundos."""
    print(f"\n  🔴 MODO WATCH ACTIVO — Escaneando cada {interval}s")
    print("     Ctrl+C para detener\n")

    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')

            print(f"  Descargando datos para {len(tickers)} tickers...")
            data = download_batch(tickers, period='14mo')

            result = scan_signals(data, capital)
            display_signals(result, capital)

            # Alerta sonora si hay senal activa en mercado bull
            if result['signals'] and result['macro_ok']:
                os.system('afplay /System/Library/Sounds/Ping.aiff 2>/dev/null || echo -e "\\a"')

            print(f"  Proximo escaneo en {interval}s... (Ctrl+C para detener)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n  👋 Scanner detenido.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Momentum Breakout Scanner v4.0 (alineado con v12: EU options + Gold)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_scanner.py --scan                       # Escaneo unico
  python run_scanner.py --scan --category US_TECH    # Solo tech USA
  python run_scanner.py --scan --ticker NVDA,TSLA    # Tickers especificos
  python run_scanner.py --watch                      # Monitoreo continuo
  python run_scanner.py --watch --interval 60        # Cada 60 segundos
  python run_scanner.py --capital 25000              # Capital personalizado
        """
    )

    # Modo
    parser.add_argument('--scan', action='store_true', help='Escaneo unico')
    parser.add_argument('--watch', action='store_true', help='Monitoreo continuo')

    # Filtros
    parser.add_argument('--category', type=str,
                        help='Filtrar por categoria (US_TECH, US_FINANCE, EU_GERMANY, COMMODITY, etc.)')
    parser.add_argument('--ticker', type=str,
                        help='Tickers especificos (separados por coma)')

    # Parametros
    parser.add_argument('--capital', type=float, default=10000,
                        help='Capital en EUR (default: 10000)')
    parser.add_argument('--interval', type=int, default=300,
                        help='Intervalo en watch mode en segundos (default: 300)')

    args = parser.parse_args()

    # Determinar tickers
    if args.ticker:
        tickers = [t.strip().upper() for t in args.ticker.split(',')]
    elif args.category:
        tickers = [t for t, v in ASSETS.items()
                   if v.get('category', '').upper().startswith(args.category.upper())
                   and not v.get('is_crypto', False)
                   and not t.endswith('USDT')]
        if not tickers:
            categories = sorted(set(v['category'] for v in ASSETS.values()))
            print(f"  ❌ No hay tickers para categoria: {args.category}")
            print(f"  Disponibles: {', '.join(categories)}")
            return
    else:
        tickers = CONFIG['test_tickers']

    CONFIG['capital'] = args.capital

    macro_str = f"SPY > SMA{CONFIG.get('macro_sma_period', 50)}" if CONFIG.get('use_macro_filter') else "OFF"
    n_opt_total = len(OPTIONS_ELIGIBLE_US) + len(OPTIONS_ELIGIBLE_EU)
    print(f"""
{'='*70}
  MOMENTUM BREAKOUT SCANNER v4.0 (v12)
{'='*70}
  Tickers: {len(tickers)} | Capital: EUR {args.capital:,.0f}
  Estrategia: v12 (v8 + opciones EU + Gold overlay 30%)
  Filtro macro: {macro_str}
  Datos: DIARIOS | Solo LONGS | Max {CONFIG['max_positions']} posiciones
  Opciones US: {len(OPTIONS_ELIGIBLE_US)} tickers | max 2 slots | spread 3% (IBKR)
  Opciones EU: {len(OPTIONS_ELIGIBLE_EU)} tickers | max 2 slots | spread 10% (DEGIRO)
  Total option-eligible: {n_opt_total} | IVR < {OPTIONS_CONFIG['option_max_ivr']}
  Gold overlay: {GOLD_OVERLAY_PCT*100:.0f}% equity en GLD permanente
{'='*70}
""")

    if args.watch:
        watch_mode(tickers, args.capital, args.interval)

    elif args.scan or True:  # Default = scan
        print(f"  Descargando datos para {len(tickers)} tickers...")
        data = download_batch(tickers, period='14mo')

        if data:
            result = scan_signals(data, args.capital)
            display_signals(result, args.capital)
        else:
            print("  ❌ No se pudieron descargar datos")


if __name__ == "__main__":
    main()
