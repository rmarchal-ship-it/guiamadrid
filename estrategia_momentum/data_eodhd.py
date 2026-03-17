#!/usr/bin/env python3
"""
EODHD Data Source — Drop-in replacement para download_data() de yfinance.

Uso:
    from data_eodhd import download_data_eodhd
    df = download_data_eodhd('AAPL', 120)      # equivale a download_data('AAPL', 120)
    df = download_data_eodhd('SIE.DE', 240)     # convierte a SIE.XETRA internamente

Requiere:
    export EODHD_API_KEY=tu_api_key
    (o archivo .env en el directorio del proyecto)

Diferencias con yfinance:
    - EODHD devuelve OHLC raw (sin ajustar) + adjusted_close
    - Esta función ajusta O/H/L usando ratio = adjusted_close / close
    - El resultado es idéntico en formato al de yfinance: DatetimeIndex + Open/High/Low/Close/Volume
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# .env support (opcional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EODHD_API_KEY = os.environ.get('EODHD_API_KEY')
EODHD_BASE_URL = 'https://eodhd.com/api/eod'

# Mapping de sufijos Yahoo → códigos de exchange EODHD
# Solo los que cambian. Los que no están aquí se usan tal cual.
EXCHANGE_MAP = {
    '': 'US',          # AAPL → AAPL.US
    '.DE': 'XETRA',    # SIE.DE → SIE.XETRA
    '.L': 'LSE',        # ULVR.L → ULVR.LSE
    '.T': 'TSE',        # 6861.T → 6861.TSE
    '.AX': 'AU',        # BHP.AX → BHP.AU
}

# Sufijos que se mantienen igual en EODHD
SAME_SUFFIX = {'.PA', '.SW', '.ST', '.MI', '.AS', '.BR', '.MC', '.HE', '.CO', '.HK'}


def yahoo_to_eodhd(ticker):
    """Convierte ticker Yahoo Finance a formato EODHD (SYMBOL.EXCHANGE)."""
    # Buscar sufijo más largo primero (e.g., .AX antes de buscar vacío)
    for suffix in sorted(EXCHANGE_MAP.keys(), key=len, reverse=True):
        if suffix and ticker.endswith(suffix):
            symbol = ticker[:-len(suffix)]
            exchange = EXCHANGE_MAP[suffix]
            return f"{symbol}.{exchange}"

    # Sufijos que no cambian
    for suffix in SAME_SUFFIX:
        if ticker.endswith(suffix):
            symbol = ticker[:-len(suffix)]
            exchange = suffix[1:]  # quitar el punto
            return f"{symbol}.{exchange}"

    # Sin sufijo → US
    return f"{ticker}.{EXCHANGE_MAP['']}"


def _adjust_ohlc(df):
    """Ajusta Open/High/Low usando ratio adjusted_close/close.

    EODHD devuelve OHLC raw y solo adjusted_close está ajustado a splits+dividendos.
    El ratio adjusted_close/close captura el factor de ajuste acumulado.
    """
    if 'adjusted_close' not in df.columns or 'close' not in df.columns:
        return df

    # Evitar división por cero
    ratio = df['adjusted_close'] / df['close'].replace(0, float('nan'))
    ratio = ratio.fillna(1.0)

    df['open'] = df['open'] * ratio
    df['high'] = df['high'] * ratio
    df['low'] = df['low'] * ratio
    df['close'] = df['adjusted_close']  # Close = adjusted

    return df


def download_data_eodhd(ticker, months):
    """Descarga datos históricos diarios de EODHD.

    Drop-in replacement de download_data() de backtest_experimental.py.
    Devuelve DataFrame con DatetimeIndex y columnas Open/High/Low/Close/Volume,
    o None si falla.
    """
    if not EODHD_API_KEY:
        raise ValueError(
            "EODHD_API_KEY no configurada. "
            "Usa: export EODHD_API_KEY=tu_key (o crea .env)"
        )

    eodhd_ticker = yahoo_to_eodhd(ticker)

    end = datetime.now()
    start = end - timedelta(days=months * 30)

    url = f"{EODHD_BASE_URL}/{eodhd_ticker}"
    params = {
        'from': start.strftime('%Y-%m-%d'),
        'to': end.strftime('%Y-%m-%d'),
        'period': 'd',
        'api_token': EODHD_API_KEY,
        'fmt': 'json',
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, list):
            return None

        df = pd.DataFrame(data)

        if df.empty or len(df) < 50:
            return None

        # Index = DatetimeIndex
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()

        # Ajustar OHLC con ratio adjusted_close/close
        df = _adjust_ohlc(df)

        # Renombrar a formato yfinance (capital first letter)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        })

        # Mantener solo las columnas estándar
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[c for c in cols if c in df.columns]]

        # Rate limiting: 1000 req/min → 60ms entre calls
        time.sleep(0.06)

        return df if len(df) >= 50 else None

    except Exception as e:
        # Silencioso, igual que download_data() de yfinance
        return None
