#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           TASTYTRADE DATA PROVIDER - Datos Históricos 4H                      ║
║                                                                               ║
║           Permite obtener datos históricos de candles con timeframes          ║
║           específicos (1h, 4h, daily) sin las limitaciones de Yahoo           ║
║                                                                               ║
║           REQUISITOS:                                                         ║
║           pip install tastytrade                                              ║
║           Cuenta en Tastytrade (puede ser paper trading)                      ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

DOCUMENTACIÓN:
- SDK: https://github.com/tastyware/tastytrade
- API: https://developer.tastytrade.com/

TIMEFRAMES SOPORTADOS:
- '15s', '1m', '5m', '15m', '30m'  (segundos/minutos)
- '1h', '2h', '4h'                  (horas)
- '1d', '3d', '1w', '1mo'           (días/semanas/meses)

USO:
    from tastytrade_data import TastytradeDataProvider

    provider = TastytradeDataProvider('usuario', 'password')
    df = await provider.get_candles('AAPL', '4h', days_back=365)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd

# Verificar si tastytrade está instalado
try:
    from tastytrade import Session, DXLinkStreamer
    from tastytrade.dxfeed import Candle
    TASTYTRADE_AVAILABLE = True
except ImportError:
    TASTYTRADE_AVAILABLE = False
    print("⚠️  tastytrade no instalado. Ejecuta: pip install tastytrade")


class TastytradeDataProvider:
    """
    Proveedor de datos históricos usando la API de Tastytrade/dxFeed.

    Ventajas sobre Yahoo Finance:
    - Datos 4H históricos sin límite de 730 días
    - Datos más precisos y consistentes
    - Acceso a datos extended hours
    """

    def __init__(self, username: str, password: str, is_paper: bool = True):
        """
        Inicializa el proveedor de datos.

        Args:
            username: Usuario de Tastytrade
            password: Contraseña de Tastytrade
            is_paper: True para cuenta paper trading, False para cuenta real
        """
        if not TASTYTRADE_AVAILABLE:
            raise ImportError("Instala tastytrade: pip install tastytrade")

        self.username = username
        self.password = password
        self.is_paper = is_paper
        self.session = None

    async def connect(self) -> bool:
        """Establece conexión con Tastytrade."""
        try:
            self.session = Session(self.username, self.password)
            print(f"✅ Conectado a Tastytrade {'(Paper)' if self.is_paper else '(Real)'}")
            return True
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False

    async def get_candles(
        self,
        symbol: str,
        interval: str = '4h',
        days_back: int = 365,
        extended_hours: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de candles.

        Args:
            symbol: Símbolo del activo (ej: 'AAPL', 'SPY')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            days_back: Días de histórico a obtener
            extended_hours: Incluir horas extendidas

        Returns:
            DataFrame con OHLCV o None si hay error
        """
        if not self.session:
            await self.connect()

        start_time = datetime.now() - timedelta(days=days_back)
        candles = []

        try:
            async with DXLinkStreamer(self.session) as streamer:
                # Suscribirse a candles históricos
                await streamer.subscribe_candle(
                    symbols=[symbol],
                    interval=interval,
                    start_time=start_time,
                    extended_trading_hours=extended_hours
                )

                # Recoger datos (esperar hasta que lleguen todos)
                timeout = 30  # segundos
                start = datetime.now()

                async for candle in streamer.listen(Candle):
                    candles.append({
                        'timestamp': candle.time,
                        'open': candle.open,
                        'high': candle.high,
                        'low': candle.low,
                        'close': candle.close,
                        'volume': candle.volume
                    })

                    # Timeout para evitar espera infinita
                    if (datetime.now() - start).seconds > timeout:
                        break

                # Desuscribirse
                await streamer.unsubscribe_candle([symbol])

        except Exception as e:
            print(f"❌ Error obteniendo candles para {symbol}: {e}")
            return None

        if not candles:
            print(f"⚠️  No se obtuvieron candles para {symbol}")
            return None

        # Convertir a DataFrame
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df = df.sort_index()

        print(f"✅ {symbol}: {len(df)} candles ({interval}) desde {df.index[0]} hasta {df.index[-1]}")
        return df

    async def get_multiple_candles(
        self,
        symbols: List[str],
        interval: str = '4h',
        days_back: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos históricos para múltiples símbolos.

        Args:
            symbols: Lista de símbolos
            interval: Timeframe
            days_back: Días de histórico

        Returns:
            Diccionario {symbol: DataFrame}
        """
        data = {}
        for symbol in symbols:
            print(f"  Descargando {symbol}...", end='\r')
            df = await self.get_candles(symbol, interval, days_back)
            if df is not None:
                data[symbol] = df
        return data


async def test_connection():
    """Test de conexión y descarga de datos."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  TASTYTRADE DATA PROVIDER - Test de Conexión                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Pedir credenciales
    print("Ingresa tus credenciales de Tastytrade:")
    username = input("  Usuario: ")
    password = input("  Password: ")

    if not username or not password:
        print("❌ Credenciales requeridas")
        return

    provider = TastytradeDataProvider(username, password)

    if await provider.connect():
        # Test con AAPL
        print("\n📊 Descargando datos de prueba (AAPL, 4H, 30 días)...")
        df = await provider.get_candles('AAPL', '4h', days_back=30)

        if df is not None:
            print(f"\n✅ Test exitoso!")
            print(f"   Rango: {df.index[0]} a {df.index[-1]}")
            print(f"   Barras: {len(df)}")
            print(f"\n   Últimas 5 barras:")
            print(df.tail())
        else:
            print("❌ No se pudieron obtener datos")


# ═══════════════════════════════════════════════════════════════════════════════
# ALTERNATIVA: POLYGON.IO (API de pago con datos históricos extensos)
# ═══════════════════════════════════════════════════════════════════════════════

class PolygonDataProvider:
    """
    Proveedor alternativo usando Polygon.io.

    Requiere suscripción ($29/mes básico, $199/mes completo)
    Ofrece datos históricos completos sin límites.

    pip install polygon-api-client
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    # TODO: Implementar si se decide usar Polygon


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not TASTYTRADE_AVAILABLE:
        print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  INSTALACIÓN REQUERIDA                                                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Ejecuta: pip install tastytrade                                              ║
║                                                                               ║
║  Luego vuelve a ejecutar este script.                                         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """)
    else:
        asyncio.run(test_connection())
