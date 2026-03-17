"""
Polygon.io Data Provider for Extended Historical Data
Supports 4H bars with up to 2 years of history (free plan)
"""

from polygon import RESTClient
from datetime import datetime, timedelta
import pandas as pd
import time

POLYGON_API_KEY = 'zktUOU6om3a4_ApgWOd0QW9naA7y_zpj'

def get_polygon_4h_data(ticker: str, days_back: int = 730, verbose: bool = True) -> pd.DataFrame:
    """
    Download 4H OHLCV data from Polygon.io

    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        days_back: Number of days to look back (max ~730 for free plan)
        verbose: Print progress

    Returns:
        DataFrame with OHLCV data indexed by datetime
    """
    client = RESTClient(POLYGON_API_KEY)

    end = datetime.now()
    all_bars = []
    chunk_days = 60  # Smaller chunks to avoid rate limits

    current_end = end
    total_requests = 0
    target_start = end - timedelta(days=days_back)

    if verbose:
        print(f'📊 Descargando {ticker} 4H ({days_back} días)...')

    while current_end > target_start:
        current_start = max(current_end - timedelta(days=chunk_days), target_start)

        try:
            aggs = list(client.get_aggs(
                ticker=ticker,
                multiplier=4,
                timespan='hour',
                from_=current_start.strftime('%Y-%m-%d'),
                to=current_end.strftime('%Y-%m-%d'),
                limit=50000
            ))
            all_bars.extend(aggs)
            total_requests += 1

            if verbose:
                print(f'   {current_start.date()} - {current_end.date()}: {len(aggs)} barras')

            current_end = current_start - timedelta(days=1)

            # Rate limit: 5 req/min en plan gratuito
            if total_requests % 4 == 0:
                if verbose:
                    print('   ⏳ Rate limit pause...')
                time.sleep(15)
            else:
                time.sleep(2)

        except Exception as e:
            if '429' in str(e):
                if verbose:
                    print(f'   ⚠️ Rate limit alcanzado, esperando 60s...')
                time.sleep(60)
                continue
            else:
                if verbose:
                    print(f'   ❌ Error: {e}')
                break

    if not all_bars:
        return pd.DataFrame()

    # Convertir a DataFrame
    data = []
    seen = set()
    for bar in all_bars:
        if bar.timestamp not in seen:
            seen.add(bar.timestamp)
            data.append({
                'datetime': datetime.fromtimestamp(bar.timestamp / 1000),
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume
            })

    df = pd.DataFrame(data)
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.set_index('datetime')

    if verbose:
        print(f'✅ {ticker}: {len(df)} barras 4H ({df.index[0].date()} - {df.index[-1].date()})')

    return df


def get_multiple_tickers_4h(tickers: list, days_back: int = 450) -> dict:
    """
    Download 4H data for multiple tickers

    Args:
        tickers: List of stock symbols
        days_back: Number of days to look back

    Returns:
        Dictionary of ticker -> DataFrame
    """
    data = {}

    for i, ticker in enumerate(tickers):
        print(f'\n[{i+1}/{len(tickers)}] {ticker}')
        try:
            df = get_polygon_4h_data(ticker, days_back=days_back, verbose=True)
            if len(df) > 100:  # Mínimo 100 barras
                data[ticker] = df
            else:
                print(f'   ⚠️ Insuficientes datos ({len(df)} barras)')
        except Exception as e:
            print(f'   ❌ Error: {e}')

        # Pausa entre tickers
        if (i + 1) % 3 == 0:
            print('   ⏳ Pausa entre tickers (30s)...')
            time.sleep(30)

    return data


if __name__ == '__main__':
    # Test con AAPL
    df = get_polygon_4h_data('AAPL', days_back=450)
    print(f'\nResumen:')
    print(df.head())
    print(f'\n...\n')
    print(df.tail())
