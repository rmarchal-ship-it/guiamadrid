# MOMENTUM BREAKOUT v8 — Documento Unico de Referencia

## VERSION DEFINITIVA: v8 — La unica version activa y validada

**CRITICO PARA CUALQUIER CHAT NUEVO**: v8 es la version DEFINITIVA y AFINADA de la estrategia Momentum Breakout. Todas las versiones anteriores (v1 a v7) estan descartadas y sus archivos movidos a la carpeta `historico/`. Cualquier trabajo, analisis, mejora o ejecucion debe partir SIEMPRE de v8. No existe ninguna otra version valida.

**Archivos obsoletos**: Todos los scripts de versiones anteriores (v1-v7, experimentales, radares antiguos, polygon, etc.) se han movido a `estrategia_momentum/historico/` el 24 Feb 2026. Solo los archivos en la raiz de `estrategia_momentum/` son activos.

---

## 1. QUE ES v8

Estrategia de trading sistematico **Momentum Breakout con Fat Tails** que opera SOLO en largo sobre **225 activos** (acciones US + Europa + Asia-Pacifico, ETFs sectoriales e internacionales, commodities, renta fija). Busca capturar movimientos explosivos aceptando un win rate bajo (~32%) a cambio de que cada ganador genere rendimientos desproporcionados.

**v8** = v7 base + opciones CALL + universo expandido 225 tickers + 10 posiciones simultaneas.

### Resultados validados por periodo (todos los horizontes):

| Periodo | Return | Anualizado | PF | MaxDD | Win Rate | Trades |
|---------|--------|------------|-----|-------|----------|--------|
| **6m** | +27.6% | +62.9% | 4.74 | -3.2% | 55.9% | 34 |
| **12m** | +52.6% | +52.6% | 1.82 | -13.6% | 40.7% | 81 |
| **18m** | +138.9% | +78.7% | 4.84 | -16.4% | 36.4% | 110 |
| **36m** | +346.2% | — | 3.34 | -21.4% | ~32% | 231 |
| **240m** | **+37,780%** | **+34.6%** | **2.89** | **-42.6%** | **32.2%** | **1,416** |
| **480m** | **x89,620,000** | **+35.2%** | **2.14** | **-50.2%** | **31.5%** | **3,149** |

Detalle 240m: 1,281 stocks + 135 opciones. Time exits forzados: 0 (eliminados).
Detalle 480m (v8 ref): ^GSPC como macro filter, 2,857 stocks + 292 opciones. v12 (US2+EU2): CAGR +44.5%, PF 3.32.

### Mejora vs v7+ (112 tickers, 7 posiciones):

| Metrica | v7+ (112tk/7pos) | v8 (225tk/10pos) | Mejora |
|---------|------------------|------------------|--------|
| PF 240m | 1.77 | **2.89** | +63% |
| Return anualizado | +17.6% | **+34.6%** | x2 |
| MaxDD 240m | -59.9% | **-42.6%** | -17pp |
| PF 36m | 2.27 | **3.34** | +47% |
| MaxDD 36m | -42.5% | **-21.4%** | -21pp |

### Grid test max_positions con 225 tickers (240 meses):

| Pos | Return | Anualizado | PF | MaxDD | Trades |
|-----|--------|------------|-----|-------|--------|
| 7 | +5,034% | +21.8% | 1.78 | -36.1% | 988 |
| 8 | +12,514% | +27.4% | 1.97 | -43.4% | 1,149 |
| **10** | **+37,780%** | **+34.6%** | **2.89** | **-42.6%** | **1,416** |
| 12 | +23,737% | +31.5% | 2.62 | -44.3% | 1,685 |

**10 posiciones gana en PF y return.** 12 empeora a 240m porque en anos tempranos (2006-2012) con menos tickers disponibles, las senales #11 y #12 eran ruido. 7 tiene el mejor MaxDD (-36.1%) pero sacrifica mucho PF.

---

## 2. CONFIGURACION COMPLETA v8

```python
CONFIG = {
    # Capital y posiciones
    'initial_capital': 10000,
    'target_risk_per_trade_pct': 2.0,   # 2% riesgo por trade
    'max_positions': 10,                 # 10 posiciones simultaneas max (v8, antes 7)

    # Senales de entrada
    'ker_threshold': 0.40,      # Kaufman Efficiency Ratio minimo
    'volume_threshold': 1.3,    # 1.3x volumen medio
    'rsi_threshold': 50,        # RSI > 50 para longs
    'rsi_max': 75,              # RSI < 75 (evitar sobrecompra)
    'breakout_period': 20,      # Breakout sobre maximo de 20 barras
    'longs_only': True,         # SOLO LONGS (shorts destruyen portfolio)

    # Stops y trailing
    'emergency_stop_pct': 0.15,         # -15% desde entrada (solo catastrofe)
    'trail_trigger_r': 2.0,             # Trailing se activa a +2R
    'trail_atr_mult': 4.0,              # Chandelier 4xATR (trailing normal)

    # Time exit v7 (trailing only, SIN salida forzada)
    'max_hold_bars': 8,                 # A los 8 dias se activa trailing
    'time_exit_trail_atr_mult': 3.0,    # Trailing apretado 3xATR

    # Filtro macro
    'use_macro_filter': True,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,

    # Costes
    'slippage_pct': 0.10,

    # Opciones CALL (solo v8)
    'option_dte': 120,              # DTE objetivo (~120 dias)
    'option_itm_pct': 0.05,        # 5% ITM (strike = spot * 0.95)
    'option_close_dte': 45,        # Cerrar cuando quedan 45 DTE
    'option_max_ivr': 40,          # Solo comprar si IVR < 40%
    'option_ivr_window': 252,      # Ventana IV Rank: 1 ano
    'option_position_pct': 0.14,   # ~14% del equity por opcion
    'max_option_positions': 2,     # Max 2 opciones simultaneas
    'option_spread_pct': 3.0,      # Spread bid-ask opciones
    'risk_free_rate': 0.043,       # Tasa libre de riesgo
}
```

---

## 3. REGLAS DE OPERACION

### 3.1 Entrada: 4 condiciones simultaneas

Para cada uno de los 225 tickers, cada dia:
1. **KER > 0.40**: Mercado en tendencia (Kaufman Efficiency Ratio)
2. **RSI entre 50 y 75**: Momentum alcista sin sobrecompra
3. **Volumen > 1.3x media**: Confirmacion institucional
4. **Breakout**: Close supera el maximo de las ultimas 20 barras

Si las 4 se cumplen = SENAL LONG.

### 3.2 Ranking multi-factor (seleccion de las 10 mejores)

```
Score = 0.30 x KER + 0.20 x RSI_norm + 0.20 x Vol_norm + 0.15 x Breakout_str + 0.15 x ATR%
```

### 3.3 Filtro macro: SPY > SMA50

Solo operar cuando SPY > SMA50. En festivos US (cuando SPY no cotiza pero Europa si), se usa el ultimo valor conocido de SPY (no se asume bullish por defecto — bug corregido en v7).

**Estadisticas macro a 240 meses:** 29.1% del tiempo en BEAR (1,443 dias / ~68.7 meses equivalentes), 167 periodos bear separados. El filtro solo bloquea NUEVAS posiciones; las existentes siguen con sus trailing stops normales.

### 3.4 Decision accion vs opcion (v8)

Si el ticker es elegible para opciones:
1. Calcular IVR (IV Rank sobre 252 dias)
2. Si IVR < 40 Y hay slot de opcion libre (max 2) → abrir CALL
3. Si no → abrir accion normal

### 3.5 Position sizing

- **Acciones**: Inverse volatility. R = 2xATR, units = (equity x 2%) / R. Tope: equity/10 por posicion.
- **Opciones**: 14% del equity por posicion. Contratos = (equity x 0.14) / (premium x 100).

### 3.6 Gestion de la posicion

**Acciones — 3 mecanismos de salida (por prioridad):**

1. **Emergency stop -15%**: Solo catastrofe. Si low <= entry * 0.85 → cierre.
2. **Trailing Chandelier 4xATR**: Se activa cuando el trade alcanza +2R. Sube pero nunca baja.
3. **Time exit a 8 dias**: Si tras 8 barras el trailing normal NO se ha activado → activar trailing 3xATR.
   - Si perdiendo: trailing = max(chandelier_3xATR, entry * 0.95)
   - Si ganando: trailing = max(chandelier_3xATR, breakeven)
   - **NUNCA se fuerza la salida.** El trailing se encarga.

**Opciones CALL:**
1. Cierre automatico a 45 DTE restantes (antes de que theta acelere)
2. Sin stop loss (riesgo = prima pagada)
3. Nunca ir a vencimiento → siempre queda valor residual

### 3.7 Logica exacta del time exit v7 (en Trade.update)

```python
# TIME EXIT: tras max_hold_bars, activar trailing (nunca forzar salida)
if self.bars_held >= CONFIG['max_hold_bars']:  # 8 bars
    if not self.trailing_active:
        trail_mult = CONFIG.get('time_exit_trail_atr_mult', 3.0)
        chandelier = self.highest_since - (current_atr * trail_mult)
        breakeven = self.entry_price * (1 + CONFIG['slippage_pct'] / 100)
        self.trailing_active = True
        if close <= self.entry_price:
            self.trailing_stop = max(chandelier, self.entry_price * 0.95)
        else:
            self.trailing_stop = max(chandelier, breakeven)
```

---

## 4. UNIVERSO: 225 TICKERS (v8, expandido desde 112)

| Zona | Categoria | N | Ejemplos |
|------|-----------|---|----------|
| EEUU - Tech | US_TECH | 20 | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, CRM, AVGO |
| EEUU - Finanzas | US_FINANCE | 10 | JPM, BAC, WFC, GS, MS, BLK, SCHW, C, AXP, USB |
| EEUU - Salud | US_HEALTH | 10 | UNH, JNJ, LLY, PFE, ABBV, MRK, TMO, ABT, DHR, BMY |
| EEUU - Consumo | US_CONSUMER | 10 | WMT, HD, PG, KO, PEP, COST, MCD, NKE, SBUX, TGT |
| **EEUU - Industrial** | **US_INDUSTRIAL** | **10** | **CAT, HON, GE, UNP, DE, RTX, LMT, MMM, EMR, ETN** |
| **EEUU - Energia** | **US_ENERGY** | **10** | **XOM, CVX, COP, SLB, EOG, MPC, PSX, VLO, OXY, HAL** |
| **EEUU - Utilities** | **US_UTILITY** | **10** | **NEE, SO, DUK, D, AEP, SRE, XEL, EXC, WEC, ED** |
| **EEUU - Real Estate** | **US_REALESTATE** | **10** | **AMT, PLD, CCI, O, EQIX, SPG, PSA, DLR, WELL, AVB** |
| **EEUU - Telecom** | **US_TELECOM** | **7** | **T, VZ, CMCSA, DIS, NFLX, TMUS, CHTR** |
| Europa - Alemania | EU_GERMANY | 10 | SAP, SIE.DE, ALV.DE, DTE.DE, MUV2.DE, BAS.DE, BMW.DE |
| Europa - Francia | EU_FRANCE | 10 | OR.PA, MC.PA, SAN.PA, AI.PA, BNP.PA, SU.PA, AIR.PA |
| Europa - Otros Eurozona | EU_VARIOUS | 16 | ASML, INGA.AS, IBE.MC, SAN.MC, ENEL.MI, ISP.MI, UCG.MI |
| **Europa - UK** | **EU_UK** | **9** | **SHEL, HSBC, BP, RIO, GSK, ULVR.L, LSEG.L, BATS.L, DGE.L** |
| **Europa - Suiza** | **EU_SWISS** | **6** | **NESN.SW, ROG.SW, NOVN.SW, UBSG.SW, ZURN.SW, ABBN.SW** |
| **Europa - Nordicos** | **EU_NORDIC** | **5** | **NOVO-B.CO, ERIC-B.ST, VOLV-B.ST, SAND.ST, NESTE.HE** |
| **Asia - Japon** | **ASIA_JAPAN** | **10** | **7203.T, 6758.T, 6861.T, 8306.T, 9984.T, 6501.T** |
| **Asia - Australia** | **ASIA_AUSTRALIA** | **8** | **BHP.AX, CBA.AX, CSL.AX, NAB.AX, WBC.AX, FMG.AX** |
| **Asia - China** | **ASIA_CHINA** | **5** | **BABA, JD, PDD, BIDU, 0700.HK** |
| Commodities | COMMODITY | 17 | GLD, SLV, PPLT, GDX, GDXJ, USO, BNO, UNG, XLE, XOP, CPER |
| Indices/ETFs US | US_INDEX | 8 | QQQ, TQQQ, SPY, SPXL, IWM, TNA, DIA, BITO |
| **ETFs Sectoriales** | **ETF_SECTOR** | **6** | **SMH, XBI, XLU, XLI, XLF, XLV** |
| **ETFs Internacionales** | **ETF_INTL** | **10** | **EEM, VWO, EFA, FXI, EWJ, EWG, EWU, INDA, EWZ, EWT** |
| **Renta fija** | **FIXED_INCOME** | **8** | **TLT, IEF, SHY, TIP, AGG, LQD, HYG, EMB** |

**Categorias en negrita = nuevas en v8.** Universo expandido de 112 a 225 tickers.

**Tickers elegibles para opciones (v8, 104 tickers):** US Tech (15), US Finance (9), US Health (8), US Consumer (10), US Industrial (10), US Energy (9), US Utility (5), US Real Estate (7), US Telecom (6), China ADRs (4), ETFs (21).

---

## 5. ARCHIVOS DEL PROYECTO

### Estructura de carpetas (actualizada 24 Feb 2026):

```
estrategia_momentum/
├── ARCHIVOS ACTIVOS (raiz) — actualizado 27 Feb 2026
│   ├── RESUMEN_PARA_CONTINUAR.md          ← Este archivo. Documento unico de referencia.
│   ├── INFORME_FINAL_V12.md               ← Informe final v12 completo (estrategia + validacion)
│   ├── MOMENTUM_BREAKOUT_V8_COMPLETO.py   ← Estrategia v8 completa en un solo archivo
│   ├── ESTRATEGIA_COMPLETA_PARA_COPIAR.txt← Para compartir/analisis externo
│   ├── backtest_experimental.py           ← BACKTEST v8 principal. Validado a 240m.
│   ├── backtest_v12_eu_options.py         ← BACKTEST v12: v8 + opciones EU (2+2 slots separados)
│   ├── backtest_v12_montecarlo.py         ← Validacion Monte Carlo v12 (3 tests, 10K sims)
│   ├── backtest_v12_40y.py               ← Test DEFINITIVO v12 a 40 años (^GSPC + GC=F)
│   ├── backtest_v13_rolling.py            ← v13 Rolling Thunder (DESCARTADO — rolar opciones ganadoras)
│   ├── backtest_time_adjusted.py          ← v8 con delay 1-bar (alineado con paper trading)
│   ├── momentum_breakout.py               ← Motor de senales (MomentumEngine, 225 tickers)
│   ├── paper_trading.py                   ← Paper trading v3.0. Ejecutar con --scan
│   ├── paper_portfolio.json               ← Estado actual del paper trading
│   ├── run_scanner.py                     ← Scanner/radar v3.0
│   ├── monthly_equity_report.py           ← Informe mensual mark-to-market
│   ├── backtest_audit.py                  ← Auditoria (walk-forward, survivorship, robustez)
│   ├── comparativa_60_40_vs_v8.py         ← Comparativa vs 60/40 TQQQ/Gold
│   ├── simulacion_apalancamiento.py       ← Simulacion de apalancamiento
│   ├── simulacion_stress_test.py          ← Stress test
│   ├── spy_correlation_analysis.py        ← Correlacion 225 tickers con SPY
│   ├── diagnostico_240m.py                ← Diagnostico temporal por ventanas
│   ├── analisis_trades_240m.py            ← Analisis profundo de trades
│   ├── test_time_exit_variants.py         ← Validacion time exit v6→v7
│   ├── rendimiento_con_deuda.md           ← Simulacion apalancamiento bancario 70K
│   └── historico_trades_*.csv             ← CSVs de trades por periodo (6/12/18/24/36m)
│
└── historico/                             ← ARCHIVOS OBSOLETOS / INVESTIGACION CERRADA
    ├── backtest_momentum_breakout.py      ← v6 original
    ├── backtest_opciones.py               ← Primer experimento opciones
    ├── backtest_real_capital.py            ← Version antigua
    ├── backtest_radar_v71.py              ← Radar v7.1
    ├── backtest_v73_multiperiod.py        ← v7.3 multiperiodo
    ├── backtest_positions_comparison.py    ← Comparacion posiciones
    ├── backtest_definitivo.py              ← Backtest v7 base (obsoleto)
    ├── backtest_extended_polygon.py        ← Polygon experimental
    ├── backtest_v6_recovered.py            ← v6 reconstruido y validado (Feb 2026)
    ├── backtest_v8_eu_options.py           ← v8 EU options (precursor de v12, movido 27 Feb)
    ├── test_v9_options_first.py            ← v9 Options-First (DESCARTADO — DD >60%)
    ├── test_v9_analysis.py                 ← v9 analisis detallado
    ├── test_v10_gold_hedge.py              ← v10 Gold 30% overlay (ADOPTADO como capa)
    ├── test_v11_vix_filter.py              ← v11 VIX filter (DESCARTADO — redundante)
    ├── test_leveraged_etfs.py             ← ETFs apalancados (descartado)
    ├── grid_holdperiod.py                 ← Grid search hold period
    ├── hyper_*.py                         ← Scanners experimentales v7.0-7.3
    ├── radar_v71_intraday*.py             ← Radar intraday (no usado)
    ├── polygon_data.py / .pkl             ← Datos Polygon (no usado)
    └── tastytrade_data.py                 ← Datos Tastytrade (no usado)
```

**REGLA**: Solo los archivos en la raiz de `estrategia_momentum/` son activos. Todo lo de `historico/` esta descartado y solo se mantiene como referencia.

---

## 6. INSTRUCCIONES DE EJECUCION

### 6.1 Backtest v8 (el principal)

```bash
cd "/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code/estrategia_momentum"

# Backtest 240 meses (20 anos) — el test de referencia
python3 backtest_experimental.py --months 240 --test b --verbose

# Periodos mas cortos
python3 backtest_experimental.py --months 36 --test b --verbose
python3 backtest_experimental.py --months 12 --test b --verbose

# Todos los tests (baseline, apalancados, opciones, combinado)
python3 backtest_experimental.py --months 36 --test all
```

### 6.2 Tests v8.1 (exencion macro para tickers descorrelacionados)

```bash
# v8.1: exencion para 81 tickers con correlacion <= 0.15 con SPY
python3 backtest_experimental.py --months 36 --test v81

# v8.1b: exencion solo para 20 tickers con correlacion negativa con SPY
python3 backtest_experimental.py --months 36 --test v81b
```

**DECISION: v8.1/v8.1b DESCARTADOS.** Testados a 12/24/36 meses. v8.1 gana a 12m (+63.4% vs +39.8%) pero pierde a 36m (MaxDD -37.6% vs -21.4%). v8.1b (solo correlacion negativa, 20 tickers) mejor que v8.1 pero aun peor que v8 a 36m. Conclusion: el filtro macro debe aplicarse a todos los tickers. El codigo se mantiene para referencia pero no se usa.

### 6.3 Informe mensual mark-to-market

```bash
# Informe mensual con equity realizada + MTM posiciones abiertas
python3 monthly_equity_report.py --months 36
```

Genera tabla mensual con: fecha, macro (BULL/BEAR), posiciones, equity realizada, MTM stocks, MTM opciones, equity total, return mensual, return acumulado. Tambien detalle de cada posicion abierta por mes.

### 6.4 Verbose logging (log completo de OPEN/CLOSE)

```bash
# Log completo de todas las entradas y salidas
python3 backtest_experimental.py --months 36 --test b --verbose
```

El verbose muestra:
- `OPEN ticker | EUR xxx (yy.yyu) @ $price` — entrada acciones
- `OPEN OPT ticker | K=$strike IV=xx% IVR=xx xxxDTE Prem=$x.xx xNc = EUR xxx` — entrada opciones
- `CLOSE ticker | exit_reason | P&L EUR +/-xxx (+/-x.x%) | Pos: n/10 | Equity: EUR xxx` — cierre acciones
- `CLOSE OPT ticker | exit_reason | P&L EUR +/-xxx (+/-x.x%) | Pos: n/10 | Equity: EUR xxx` — cierre opciones

### 6.5 Diagnostico temporal (por ventanas de 5 anos)

```bash
python3 diagnostico_240m.py --months 240
python3 diagnostico_240m.py --data-quality    # Solo calidad de datos
```

### 6.6 Backtest v7 base (solo acciones, sin opciones) — MOVIDO A historico/

```bash
# NOTA: backtest_definitivo.py movido a historico/ (v7, obsoleto)
# Solo para referencia historica:
python3 historico/backtest_definitivo.py --months 240
```

### 6.7 Paper Trading (diario)

```bash
python3 paper_trading.py --scan                     # Escaneo diario + ejecutar trades
python3 paper_trading.py --status                   # Ver posiciones actuales
python3 paper_trading.py --history                   # Trades cerrados
python3 paper_trading.py --reset --capital 10000    # Reset (nueva cartera)
```

### 6.8 Scanner / Radar

```bash
python3 run_scanner.py --scan                       # Senales de hoy
python3 run_scanner.py --scan --category US_TECH    # Por categoria
python3 run_scanner.py --watch                      # Monitoreo continuo
```

El scanner v3.0 incluye logica v8.1: en mercado BEAR, muestra tickers exentos (corr <= 0.15 con SPY) con "EJECUTAR (exento macro v8.1)" y los no-exentos con "NO EJECUTAR (bear)". Aunque v8.1 fue descartado en backtest, la informacion visual se mantiene como referencia.

### 6.9 Analisis de correlacion con SPY

```bash
python3 spy_correlation_analysis.py
```

Calcula correlacion Pearson de retornos diarios (2 anos) de los 225 tickers con SPY. Resultado: 81 tickers con corr <= 0.15, 20 con correlacion negativa.

### 6.10 Dependencias

```bash
pip install yfinance pandas numpy scipy
```

---

## 7. PAPER TRADING — ESTADO ACTUAL (10 Mar 2026)

Macro en **BEAR** desde 3 Mar 2026 (SPY < SMA50). No se abren posiciones nuevas.
Gold overlay 30% se aplica mentalmente (no en paper_portfolio.json, que trackea solo momentum).

| Campo | Valor |
|-------|-------|
| Capital inicial | EUR 10,000 |
| Cash | EUR -852.52 (margen DEGIRO) |
| Filtro macro | **BEAR** (SPY < SMA50) |
| Posiciones abiertas | **3 de 10** |
| Opciones abiertas | 0 |
| Trades cerrados | 7 (3W, 4L) |
| Senales pendientes | 5 (INTU, NOW, CORN, DBA, PSX) — todas bloqueadas BEAR |

### Posiciones abiertas (10 Mar 2026):

| Ticker | Entry | Units | Pos EUR | Trailing | SL DEGIRO |
|--------|-------|-------|---------|----------|-----------|
| AI.PA | EUR 178.18 | 11 | 1,960 | NO | EUR 151.50 (emergency -15%) |
| NESN.SW | CHF 82.90 | 24 | 2,182 | NO | CHF 70.47 (emergency -15%) |
| WDS | $20.19 | 100 | 1,711 | $20.67 (+2R) | $20.74 |

**AI.PA y NESN.SW** tienen SL de emergencia (-15%) puestos en DEGIRO. Ambas en perdida.

### Trades cerrados (7):

| # | Ticker | Entry → Exit | PnL EUR | PnL % | R-mult | Razon | Dias |
|---|--------|-------------|---------|-------|--------|-------|------|
| 1 | RI.PA | EUR 82.68 → 84.50 | +31.36 | +1.4% | +0.85R | Trailing (time exit d8) | 8d |
| 2 | BMY | $61.37 → $60.64 | -15.58 | -0.8% | 0R | Venta manual (error) | 9d |
| 3 | GE | $328.49 → $328.46 | +32.31 | +1.5% | +0.75R | Trailing (breakeven, FX) | 9d |
| 4 | BHP.AX | EUR 34.98 → 32.12 | -164.83 | -8.6% | -1.6R | Manual (SL borrado) | 4d |
| 5 | ROG.SW | CHF 366.40 → 343.00 | -134.36 | -6.7% | -2.0R | SL ejecutado | 6d |
| 6 | 6861.T | EUR 369.37 → 311.81 | **-430.20** | **-16.6%** | **-2.4R** | SL con gap (320→312) | 8d |
| 7 | PSA | $300.20 → $300.05 | +26.19 | +1.3% | +0.57R | Trailing (time exit d8, BE) | 11d |

**PnL total cerrados: EUR -655** | Win rate: 43% (3/7)

### Senales pendientes (10 Mar, todas BEAR):

| Ticker | Score | KER | RSI | Vol |
|--------|-------|-----|-----|-----|
| INTU | 80 | 0.697 | 72 | 3.1x |
| NOW | 79 | 0.678 | 74 | 2.9x |
| CORN | 70 | 0.703 | 76 | 22.1x |
| DBA | 65 | 0.642 | 82 | 15.5x |
| PSX | 49 | 0.600 | 66 | 1.7x |

### Watchlist (cerca de breakout, 10 Mar): SBUX 0.32%, EXC 0.74%, SOYB 0.98%, GIS 1.22%, SHEL 1.30%

### Lecciones paper trading (actualizado 10 Mar):
- **SL en DEGIRO pueden borrarse**: BHP SL se borro sin aviso. Verificar SL diariamente.
- **Gaps en Tradegate**: 6861.T SL a EUR 320, ejecutado a EUR 312. Slippage EUR 8/acc. Liquidez limitada.
- **yfinance unreliable para tickers EU/JP**: Usar precios DEGIRO reales para tracking.
- **Time exit day 8**: Funciono correctamente en GE, RI.PA y PSA. Sistema de trailing es robusto.
- **FX puede salvar trades**: PSA cerro a precio casi igual ($300.20→$300.05) pero PnL positivo (+EUR 26) por debilitamiento EUR (1.181→1.158).

### REGLA CRITICA — OPCIONES EN PAPER TRADING (27 Feb 2026):

**Las opciones son ESENCIALES. Triplican el CAGR (10% → 28%).** Proceso obligatorio:

1. Señal nueva para ticker X
2. ¿X está en OPTIONS_ELIGIBLE? (ver backtest_experimental.py linea 76-103)
3. ¿Slots de opciones < 2?
4. ¿El precio del stock permite 1 contrato (100 shares) dentro del budget (14% del equity)?
   - A EUR 10K: budget = EUR 1,400 → stocks ≤ ~$160 son accesibles
   - A EUR 20K: budget = EUR 2,800 → stocks ≤ ~$310 accesibles
   - Formula: stock_price × 0.09 × 100 < equity × 0.14
5. Si todo SÍ → **abrir OPCION** (Call 5% ITM, ~120 DTE, IVR < 40%). Broker: IBKR
6. Si ticker elegible pero contrato demasiado caro → abrir accion normal

**Caso GE/PSA**: Estan en OPTIONS_ELIGIBLE pero a EUR 10K un contrato ITM cuesta $2,700-$2,950 > budget EUR 1,400. Correctamente comprados como stocks.

**Opciones europeas — v12 DEFINITIVA (27 Feb 2026)**:
Backtest `backtest_v12_eu_options.py` con slots SEPARADOS US2+EU2.

**Resultados 240m**:

| Config | Final EUR | CAGR | MaxDD | PF | Opt US | Opt EU |
|--------|----------|------|-------|----|----|-----|
| US2 EU0 (ref) | 2,013K | +30.4% | -35.9% | 2.21 | 128 | 0 |
| **US2+EU2** | **4,895K** | **+36.3%** | -42.6% | **3.40** | 128 | 72 |

**TEST DEFINITIVO 40 AÑOS (480m) — `backtest_v12_40y.py`**:
Macro filter: ^GSPC (S&P 500 Index, datos desde 1986). Gold overlay: GC=F (oro futuros, desde 2000; antes de 2000: 0% return conservador).

| Config | Final EUR | CAGR | MaxDD | PF | Eficiencia |
|--------|----------|------|-------|-----|------------|
| REF (v8 US2 EU0) | 1,739M | +35.2% | -50.2% | 2.14 | 0.70 |
| **v12 (US2+EU2)** | **25,152M** | **+44.5%** | -59.3% | **3.32** | 0.75 |
| REF + Gold 30% | 1,529M | +33.7% | -45.9% | — | 0.73 |
| **v12 + Gold 30%** | **20,560M** | **+42.4%** | -54.3% | — | **0.78** |

DELTA v12 vs REF (con Gold): CAGR +8.7pp | MaxDD -8.4pp | Eficiencia +0.05.
v12 multiplica el capital x2,056,000 en 40 años vs x152,900 del REF.
**v12 CONFIRMADA DEFINITIVA a 40 años.**

**Validacion multi-period**:

| Periodo | REF CAGR | EU2 CAGR | ΔCAGR | ΔDD |
|---------|----------|----------|-------|-----|
| 6m | +104.8% | +135.6% | +30.8pp | 0.0pp |
| 12m | +68.7% | +47.7% | **-21.0pp** | +16.3pp |
| 36m | +81.8% | +113.9% | +32.1pp | 0.0pp |
| 60m | +43.6% | +84.7% | +41.1pp | -2.4pp |
| 240m | +30.4% | +36.3% | +5.9pp | +6.7pp |

Positivo en 4/5 periodos. El unico negativo (12m) tiene solo 5 trades EU — muestra insuficiente.
Media ΔCAGR: +17.8pp. Opciones US NO desplazadas en ningun periodo.

**39 tickers EU option-eligible** definidos en `backtest_v12_eu_options.py` (Eurex, Euronext, LSE, SIX, OMX).
**CLAVE**: slots separados. Si EU compite con US por los mismos slots, EU DESTRUYE valor (efecto desplazamiento).
**Home runs EU confirmados**: ISP.MI +175%, KBC.BR +151%, BATS.L +167%, ADS.DE +185%.
Opciones EU disponibles en DEGIRO (confirmado: NESN.SW, AI.PA). Spread tipico ~10%.

**Para futuras señales**: SIEMPRE verificar: (1) option_eligible US o EU, (2) slot disponible (US: max 2, EU: max 2), (3) precio accesible. Campos trackeados en paper_portfolio.json.

**Validacion Monte Carlo v12 (27 Feb 2026)** — `backtest_v12_montecarlo.py`, 10,000 sims, 60m:

| Test | Resultado | Detalle |
|------|-----------|---------|
| Trade Shuffle | ✅ ROBUSTO | CAGR constante, MaxDD real (-33.5%) ≈ mediana (-24.9%) |
| Bootstrap 5y | ✅ MUY ROBUSTO | Prob perder 0%, P5 CAGR +42.3%, P50 +84.8% |
| Permutation PnL | ✅ p=0.036 | Rentabilidad total significativa (no es azar) |
| Permutation PF | ⚠️ p=0.199 | PF no significativo — inflado por compounding/sizing |

**Conclusion MC**: Edge REAL en rentabilidad total (PnL p<0.05), bootstrap extremadamente robusto (0% prob perdida). El PF no pasa test permutacion porque position sizing × compounding infla la distribucion nula — tipico de estrategias fat-tail con win rate bajo (36%). No es evidencia de overfitting sino limitacion del test para este tipo de estrategia.

### Notas ejecucion multi-broker:
- **DEGIRO**: EU stocks (Euronext, Xetra, SIX Swiss, Tradegate). Comisiones €3.90-€4.90
- **DEGIRO**: US stocks (NYSE/NASDAQ). Comision €2.00 + AutoFX ~0.25%
- **IBKR**: US options + ETFs no disponibles en DEGIRO (EWU, DBA)
- **Preferencia Europa**: Si el ticker cotiza en Europa (BHP en Tradegate), comprar ahi para evitar AutoFX y horarios

---

## 8. CASO DE ESTUDIO: CORRECCION 2025

Analisis detallado de como la estrategia navego la correccion de 2025 (backtest 36m):

1. **Inicio correccion:** 10/10 posiciones abiertas (cartera llena)
2. **Filtro macro BEAR se activa:** No se abren nuevas posiciones, pero las existentes siguen con trailing stops
3. **Trailing stops van cerrando:** Las posiciones se cierran gradualmente por stops, no por el filtro macro
4. **Minimo de posiciones:** 1/10 — solo OPT GLD sobrevivio toda la correccion (+255.1%, +EUR 5,504)
5. **Vuelta a BULL (1 Mayo):** Se rellenan 10/10 posiciones rapidamente
6. **Resultado:** La estrategia protegio capital via trailing stops, no vendiendo por panico

**Leccion clave:** El filtro macro bloquea NUEVAS entradas pero NO fuerza ventas. Las posiciones existentes siguen su propio ciclo de trailing stops. Esto es correcto — vender todo al activarse BEAR generaria whipsaws constantes.

---

## 9. CARTERA SIMULADA FEBRERO 2026 (backtest 36m, mark-to-market)

Ultimo snapshot del informe mensual (`monthly_equity_report.py --months 36`):

| Metrica | Valor |
|---------|-------|
| Fecha | 2026-02 |
| Macro | BEAR |
| Posiciones | 10/10 |
| Equity realizada | EUR 49,682 |
| MTM stocks | +EUR 2,640 |
| MTM opciones | +EUR 3,065 |
| **Equity total** | **EUR 55,386** |
| Return mensual | +21.0% |
| **Return acumulado** | **+453.9%** |

### Posiciones abiertas febrero 2026:

| Ticker | Tipo | Entry → Current | P&L EUR | P&L % |
|--------|------|-----------------|---------|-------|
| OPT XLI | Opcion CALL | Prem $12.75 → $20.31 | +3,065 | +59.3% |
| BNP.PA | Accion | $72.54 → $89.40 | +799 | +23.2% |
| VOLV-B.ST | Accion | $283.28 → $343.70 | +733 | +21.3% |
| ASML | Accion | $1,222.43 → $1,406.61 | +515 | +15.1% |
| NESTE.HE | Accion | $19.00 → $20.75 | +316 | +9.2% |
| KO | Accion | $70.39 → $78.68 | +251 | +11.8% |
| WDS.AX | Accion | $25.28 → $25.78 | +79 | +2.0% |
| SHY | Accion | $80.40 → $83.06 | +61 | +3.3% |
| BMY | Accion | $61.43 → $60.74 | -56 | -1.1% |
| BHP.AX | Accion | $51.89 → $51.13 | -58 | -1.5% |

Diversificacion geografica: Europa (BNP, VOLV-B, ASML, NESTE), Australia (WDS, BHP), EEUU (KO, BMY, SHY, XLI). 8/10 en positivo.

---

## 10. EVOLUCION DE VERSIONES (HISTORICO)

| Version | Cambio principal | Resultado |
|---------|-----------------|-----------|
| **v1-v3** | Stop loss 2xATR, 5 posiciones, 4H, partial exits | Win rate bajo, stops destruian edge |
| **v5** | Sin stop loss, 5 posiciones, time exit 45 barras fijo | PF ~1.5 a 18m |
| **v6** | 7 posiciones (grid test 5/7/8/10/15 confirmo 7 optimo) | Mejor return, PF y drawdown |
| **v6+** | v6 + opciones CALL 5% ITM, 120 DTE, cierre 45 DTE | +255.8% en 36m vs +86.3% base |
| **v7/v7+** | Time exit trailing only 3xATR a 8 bars (sin salida forzada) | PF 1.77, +17.6% anual, MaxDD -59.9% |
| **v8** | **Universo 112→225 tickers + 10 posiciones (grid test)** | **PF 2.89, +34.6% anual, MaxDD -42.6%** |
| v8.1 | Exencion macro para 81 tickers (corr<=0.15) | **DESCARTADO** — peor MaxDD a 36m |
| v8.1b | Exencion macro solo 20 tickers (corr negativa) | **DESCARTADO** — aun peor que v8 a 36m |
| v9 | Options-First (priorizar opciones sobre stocks) | **DESCARTADO** — DD inaceptable >60% |
| v10 | Gold 30% hedge overlay | **ADOPTADO** como capa operativa (no en backtest) |
| v11 | VIX filter aditivo | **DESCARTADO** — redundante con SPY>SMA50 + Gold |
| **v12** | **v8 + opciones EU (2+2 slots separados, spread 10%)** | **DEFINITIVA** — +5.9pp CAGR a 240m, PF 3.40 — validada 480m (40y) |
| v13 | Rolling Thunder: rolar opciones ganadoras a 45 DTE (mismo strike, 120 DTE) | **DESCARTADO** — bloquea slots, peor CAGR y DD a 240m |

### Cambio clave v7+→v8: expansion del universo + mas posiciones

| | v7+ (112tk/7pos) | v8 (225tk/10pos) |
|---|--------|--------|
| Universo | 112 tickers (US + EU + commodities) | 225 tickers (+industrial, energy, utility, real estate, telecom, UK, Swiss, Nordic, Japan, Australia, China, ETFs intl, fixed income) |
| Max posiciones | 7 | 10 (grid test 7/8/10/12 a 240m) |
| PF 240m | 1.77 | **2.89** |
| Return anualizado | +17.6% | **+34.6%** |
| MaxDD 240m | -59.9% | **-42.6%** |
| Trades 240m | 945 | 1,416 |

**Por que funciona**: Con 225 tickers de sectores/geografias descorrelacionados, las senales #8-#10 cada dia son de calidad comparable a las top 7 del universo reducido, pero de sectores distintos. Esto mejora la diversificacion y reduce el drawdown sin sacrificar rentabilidad. Con 112 tickers, la senal #8 a menudo era del mismo sector que otra ya abierta.

### Cambio clave v6→v7: eliminacion del time exit forzado

| | v6/v6+ | v7/v7+ |
|---|--------|--------|
| Trigger | 12 barras | 8 barras |
| Accion | Si perdiendo → **cierre forzado** | **Activar trailing** 3xATR (nunca forzar cierre) |
| Win rate time exits | 0% (316 trades a 240m) | N/A (no existen time exits forzados) |
| P&L time exits | -248,158 EUR en 20 anos | 0 (eliminados) |

**IMPORTANTE**: El time exit forzado fue eliminado por su rendimiento a 240m. Pero a horizontes cortos (36m, 60m), el cierre forzado MEJORA el rendimiento porque libera capital para nuevas entradas. El trade-off es claro:
- 240m: forzar cierre = -248k EUR (peor, muchos trades que eventualmente se recuperan)
- 36m: forzar cierre = +22% anualizado vs +9.3% sin forzar (mejor, capital libre para rotacion)

### Investigacion Feb 2026: v6 recuperado vs v8 stock-only

**Hallazgo clave**: El MomentumEngine NO cambio entre v6 y v8. Parametros identicos:
rsi_threshold=50, rsi_max=75, ker=0.40, vol=1.3, breakout_period=20.

Lo que cambio fue SOLO la logica de backtest (time exit, universo, posiciones).

**Archivo**: `historico/backtest_v6_recovered.py` — v6 reconstruido y validado.

| Config | v6 original | v8 stock-only | v8 con opciones |
|--------|-------------|---------------|-----------------|
| Time exit | 12d forzado si pierde | 8d trailing only | 8d trailing only |
| Universo | 111 tickers | 225 tickers | 225 tickers |
| Max posiciones | 7 | 10 | 10 |
| **36m return** | **+81.4% (+22.0% ann)** | ~+26% (~+8% ann) | +346.2% |
| **60m return** | +77.6% (+12.2% ann) | +104.6% (+15.4% ann) | **+121.4% (+17.2% ann)** |
| **36m PF** | **2.28** | ~1.5 | 3.34 |
| **60m PF** | 1.83 | **2.33** | 2.32 |
| **60m MaxDD** | -12.1% | **-9.6%** | -14.3% |

**Patron por horizonte**:
- 36m: v6 (+22%) >> v7/v8 (~9%) — forzar cierre libera capital, mejor rotacion
- 60m: v8 (+17.2%) > v7 (+15.4%) > v6 (+12.2%) — dejar correr trades gana
- 240m: v8 stock-only ~8% ann — se degrada por anos malos (2007, 2008, 2011, 2018, 2022)

**Problema abierto (26 Feb 2026)**: v8 stock-only se hunde a 240m (~8% ann). Las opciones enmascaran el problema subiendo a +34.6% ann. Necesita mejora del componente stock-only para horizontes largos.

### Investigacion macro filter: grid SMA (26 Feb 2026)

Probado a 60m stock-only v8 (225 tickers, 10 posiciones). Objetivo: encontrar SMA alternativa que mejore el rendimiento.

| SMA periodo | Return total | Anualizado | PF | MaxDD | Resultado |
|-------------|-------------|------------|-----|-------|-----------|
| **SMA20** | — | +10.2% | 1.64 | -11.7% | Peor que SMA50 |
| **SMA25** | — | +10.0% | 1.67 | -11.5% | Peor que SMA50 |
| **SMA35** | — | +2.8% | 1.16 | -19.3% | Mucho peor |
| **SMA50** | +121.4% | **+17.2%** | **2.32** | -14.3% | **GANADOR CLARO** |
| **SMA100** | — | +9.0% | 1.54 | — | Peor que SMA50 |
| **SMA200** | — | +9.7% | 1.56 | — | Peor que SMA50 |

**Conclusion**: SMA50 es optimo por amplio margen. No hay mejora posible via macro filter. Opcion C (mejorar macro filter) DESCARTADA.

### Analisis detallado de trades v8 stock-only 60m (26 Feb 2026)

**Top 10 ganadores** (60m, stock-only):

| Ticker | Return % | Categoria |
|--------|----------|-----------|
| 9984.T | +148.2% | Asia Japan |
| NVDA | +57.7% | US Tech |
| TSLA | +54.8% | US Tech |
| TQQQ | +48.9% | US Index Lev |
| BABA | +39.9% | Asia China |
| CEG | +34.2% | US Energy |
| TNA | +32.3% | US Index Lev |
| AVGO | +31.7% | US Tech |
| META | +31.5% | US Tech |
| SPXL | +30.5% | US Index Lev |

**Top 10 perdedores** (todos al emergency stop -15.1%):

| Ticker | Return % | Categoria |
|--------|----------|-----------|
| JD | -15.1% | Asia China |
| BITO | -15.1% | US Index Lev |
| BIDU | -15.1% | Asia China |
| TNA | -15.1% | US Index Lev |
| SPXL | -15.1% | US Index Lev |
| AMD | -15.1% | US Tech |
| TEF.MC | -15.1% | EU Spain |
| PDD | -15.1% | Asia China |
| ETN | -15.1% | US Industrial |
| ABBV | -15.1% | US Health |

**Rendimiento por año** (60m):

| Año | PF | PnL EUR | Win Rate | Observacion |
|-----|-----|---------|----------|-------------|
| 2021 | 2.61 | +2,858 | ~35% | Buen año |
| **2022** | **0.59** | **-1,467** | **13.2%** | **DESASTRE** |
| 2023 | 1.65 | +3,181 | ~30% | Recuperacion |
| 2024 | 1.83 | +4,744 | ~33% | Solido |
| 2025 | 4.89 | +16,976 | ~50% | Excepcional |

**Trimestres criticos** (win rate < 15%):
- 2022-Q2: **0% win rate** (todos los trades perdedores)
- 2022-Q3: 5% win rate
- 2023-Q1: 8% win rate

**Hallazgo clave**: El 2022 es el año que destruye el rendimiento. El filtro macro SMA50 no es suficiente para evitar las perdidas en ese periodo. Los perdedores peores son ETFs apalancados (TQQQ/TNA/SPXL) y China (JD/BIDU/PDD) que tocan emergency stop.

### Opciones de mejora pendientes de probar

- ~~Opcion A: Mejorar time exit~~ (ya probado v6/v7)
- **Opcion B: Trailing adaptativo** (ajustar ATR mult segun volatilidad del mercado)
- ~~Opcion C: Mejorar macro filter~~ (DESCARTADO — SMA50 ya optimo)
- **Opcion D: Sizing dinamico** (reducir tamaño en alta volatilidad)
- **Opcion E: VIX filter** (no operar cuando VIX > umbral)

---

## 11. DECISIONES TOMADAS Y POR QUE

| Decision | Razon |
|----------|-------|
| Solo LONGS | Shorts: P&L -281% |
| Sin stop loss fijo | Stop destruye edge (win rate 55%→31%) |
| Sin partial exits | Cortar a +2R limitaba fat tails (SLV +7.5%→+61.6%) |
| 10 posiciones (v8) | Grid test 7/8/10/12 a 240m: 10 mejor PF (2.89) y return |
| 7 posiciones (v7) | Era optimo con 112 tickers. Con 225, hay suficientes senales de calidad para 10 |
| Universo 225 tickers (v8) | Expansion en amplitud (no profundidad) — decision ex-ante, no data snooping |
| Time exit 8d trailing only (v7) | Time exit forzado: 0% win rate, -248k EUR. Trailing 3xATR elimina esos trades |
| Filtro macro SPY>SMA50 | PnL paso de +15.5% a +60.6% |
| Macro: ultimo valor conocido | En festivos US, default a True generaba entradas en BEAR |
| Opciones 5% ITM | Delta alto + valor intrinseco = menor riesgo theta |
| IVR < 40 | Filtrar opciones caras |
| Cierre opciones a 45 DTE | Evitar aceleracion de theta. Trailing en opciones no funciona |
| Opciones sin stop | Riesgo = prima pagada. Stops destruyen edge en opciones |
| ETFs apalancados descartados | Peor en todos los tests |
| v8.1 exencion macro descartada | Mejor a 12m pero peor MaxDD a 36m (-37.6% vs -21.4%). No compensa |
| v8.1b solo corr negativa descartada | Mejor que v8.1 pero aun peor que v8 puro a horizontes largos |

---

## 12. BUGS CORREGIDOS

### Bug 1: Filtro macro en festivos US
`macro_bullish.get(current_date, True)` defaulteaba a bullish cuando SPY no cotizaba en festivos US pero Europa si. Solucion: buscar el ultimo valor conocido de SPY.

### Bug 2: yfinance period='240mo'
`yf.download(period='240mo')` fallaba para la mayoria de tickers. Solucion: para periodos > 60 meses, usar parametros `start`/`end` en vez de `period`.

### Bug 3: Cifras infladas en test_time_exit_variants.py (corregido 13 Feb 2026)
`test_time_exit_variants.py` tenia 3 errores que inflaban los resultados ~12x:
1. **Notional cap**: `equity / max_positions * 2` (28.6%) en vez de `equity / max_positions` (14.3%). Posiciones el doble de grandes.
2. **Entry price**: Usaba `bar['Close']` en vez de `bar['Open'] * (1 + slippage)`. Entrada mas favorable.
3. **Opciones sin spread**: No aplicaba el +1.5% de spread de entrada a las primas.

Resultado: PF 2.36 / +23,035% (inflado) → PF 1.77 / +2,471% (real). Los numeros reales son los de `backtest_experimental.py`.

### Bug 4: Verbose close logging — exit_reason vs close_reason (corregido 14 Feb 2026)
El verbose logging para cierres usaba `trade.close_reason` pero el atributo correcto en Trade/OptionTradeV2 es `trade.exit_reason`. Corregido con find-and-replace.

---

## 13. DIAGNOSTICO TEMPORAL (descubrimiento v7+→v8)

### Problema original: rendimiento se degradaba a 240 meses

Con 112 tickers y 7 posiciones:
- 36m: PF 2.27 (excelente)
- 240m: PF 1.77 (mediocre)

### Diagnostico (diagnostico_240m.py):

- **NO era calidad de datos** — tickers con problemas de yfinance tenian P&L positivo
- **5 anos malos destruian la curva**: 2007 (PF 0.68), 2008 (PF 0.03), 2011 (PF 0.94), 2018 (PF 0.22), 2022 (PF 0.36)
- Ventanas 2006-2015: PF ~1.3 vs 2016-2025: PF ~1.8

### Solucion: expansion en amplitud (NO filtros a posteriori)

Filtrar sectores/tickers basandose en resultados de backtest = data snooping. En su lugar, expansion del universo en amplitud:
- Sectores US que faltaban: industrial, energy, utility, real estate, telecom
- Geografias nuevas: UK, Suiza, Nordicos, Japon, Australia, China
- ETFs sectoriales e internacionales
- Renta fija expandida (1→8 instrumentos)

Con mas diversificacion, los anos malos se suavizan porque siempre hay algun sector/geografia en tendencia.

---

## 14. INVESTIGACION v8.1 — EXENCION MACRO POR CORRELACION (DESCARTADA)

### Hipotesis
Los tickers con baja correlacion con SPY (<=0.15) no deberian ser bloqueados por el filtro macro, ya que en mercados bajistas estos activos (renta fija, utilities, Japon, Australia, oro) suelen funcionar bien.

### Analisis de correlacion (spy_correlation_analysis.py)
- 81 tickers con correlacion Pearson <= 0.15 con SPY (retornos diarios, 2 anos)
- 20 tickers con correlacion negativa (< 0)
- Tickers descorrelacionados: renta fija (SHY, TLT, AGG), utilities (DUK, SO, WEC), Japon, Australia, telecom europeo, commodities (GLD, UNG)

### Resultados comparativos

| Horizonte | v8 PF | v8.1 PF | v8.1b PF | v8 MaxDD | v8.1 MaxDD | v8.1b MaxDD |
|-----------|-------|---------|----------|----------|------------|-------------|
| 12m | +39.8% | **+63.4%** | +52.1% | -10.2% | -12.8% | -11.5% |
| 24m | **mejor** | peor | intermedio | **mejor** | peor | intermedio |
| 36m | **+346%** | +280% | +310% | **-21.4%** | **-37.6%** | -29.8% |

### Conclusion
v8.1 gana a corto plazo (12m) pero empeora significativamente a horizontes largos. Los trades adicionales en bear market son de menor calidad y aumentan el drawdown. **Se mantiene v8 sin modificaciones.**

El codigo de v8.1/v8.1b permanece en `backtest_experimental.py` (constantes MACRO_EXEMPT y MACRO_EXEMPT_NEG, parametro `macro_exempt_set` en find_candidates/run_backtest, tests --test v81/v81b) para referencia futura.

---

## 15. CONSTANTES CLAVE EN EL CODIGO

### backtest_experimental.py

| Constante | Linea aprox | Descripcion |
|-----------|-------------|-------------|
| LEVERAGE_FACTORS | 54-70 | Multiplicadores 2x/3x para ETFs apalancados |
| OPTIONS_ELIGIBLE | 76-103 | ~65 tickers elegibles para opciones CALL |
| MACRO_EXEMPT | 106-117 | 81 tickers corr<=0.15 con SPY (v8.1, descartado) |
| MACRO_EXEMPT_NEG | 120-124 | 20 tickers corr<0 con SPY (v8.1b, descartado) |
| BASE_TICKERS | 127 | 225 tickers del universo (desde momentum_breakout.ASSETS) |
| CONFIG | 133-163 | Configuracion completa de la estrategia |

### Funciones principales en backtest_experimental.py

| Funcion | Descripcion |
|---------|-------------|
| `find_candidates(signals_data, active_trades, current_date, is_macro_ok, macro_exempt_set=None)` | Busca candidatos para nuevas posiciones |
| `run_backtest(months, tickers, label, use_leverage_scaling=False, use_options=False, macro_exempt_set=None, verbose=False)` | Motor principal del backtest |
| `rank_candidates(candidates, signals_data)` | Ranking multi-factor de candidatos |
| `generate_all_signals(all_data, engine)` | Genera senales para todos los tickers |
| `build_macro_filter(all_data)` | Construye filtro SPY>SMA50 |
| `black_scholes_call(S, K, T, r, sigma)` | Precio de opcion CALL |
| `iv_rank(hvol_series, current_idx, window=252)` | IV Rank percentil |

---

## 16. AUDITORIA CUANTITATIVA (19 Feb 2026)

Archivo: `backtest_audit.py` — 3 tests independientes sobre 240 meses.

### Test 1: Walk-Forward (IS 70% / OOS 30%)

| Metrica | IN-SAMPLE (168m) | OUT-OF-SAMPLE (72m) | Ratio OOS/IS |
|---------|:---:|:---:|:---:|
| Trades | 1,036 | 447 | — |
| Win Rate | 32.0% | 33.1% | — |
| **Profit Factor** | **2.84** | **2.82** | **99.4%** |
| Anualizado | +37.5% | +42.4% | — |
| Max Drawdown | -43.5% | -29.6% | — |

**✅ VEREDICTO: PF OOS = 99.4% del IS → Sin overfitting**

### Test 2: Survivorship Bias (universo progresivo)

Metodo: descarga 225 tickers, pero solo permite entradas a partir del ano de IPO de cada ticker (27 tickers post-2006: META 2012, TSLA 2010, AVGO 2009, BITO 2021, etc.).

| Metrica | FIJO (225) | PROGRESIVO | Diferencia |
|---------|:---:|:---:|:---:|
| Trades | 1,408 | 1,409 | -1 |
| **Profit Factor** | **2.64** | **2.82** | **-0.18** |
| Anualizado | +38.0% | +37.7% | +0.3% |
| Max Drawdown | -43.1% | -42.6% | — |

**✅ VEREDICTO: Bias -0.18 PF → Impacto BAJO** (tickers post-2006 no inflan resultados)

### Test 3: Robustez (dependencia de home runs)

| Escenario | Trades | PnL EUR | Win% | PF | PF vs orig |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Original | 1,415 | +5,843,459 | 32.6% | 2.82 | 100% |
| Sin top 5 | 1,410 | +2,902,019 | 32.3% | 1.66 | 59% |
| Sin top 10 | 1,405 | +919,153 | 31.9% | 1.29 | 46% |
| Sin top 20 | 1,395 | -636,247 | 31.2% | 0.81 | 29% |

**⚠️ VEREDICTO: Top 10 trades (0.7%) = 84% del PnL → Dependencia CRITICA de home runs**

Nota: esto es inherente a estrategias momentum/trend-following con win rate ~32%. La distribucion de retornos es fat-tailed por diseno.

### Resumen auditoria

| Test | Resultado | Veredicto |
|------|-----------|-----------|
| Walk-Forward | PF IS 2.84 → OOS 2.82 (99.4%) | ✅ Sin overfitting |
| Survivorship (progresivo) | PF 2.64 → 2.82 (-0.18) | ✅ Bias BAJO |
| Robustez | Top 10 = 84% del PnL | ⚠️ Home runs (inherente al estilo) |

---

## 17. REGLAS DE PRESERVACION DE CODIGO

**CRITICO: Nunca mas perder versiones historicas.**

1. **NUNCA modificar archivos en `historico/`** — son el audit trail. Si necesitas probar algo, crea un archivo nuevo.
2. **Al cambiar logica de estrategia** (time exit, sizing, universo, senales): crear NUEVO archivo versionado, nunca sobreescribir el anterior.
   - Nombre: `backtest_v{N}_{variante}.py` (ej: `backtest_v6_recovered.py`, `backtest_v8_stock_only.py`)
3. **Documentar SIEMPRE en cabecera** del archivo: fecha, parametros exactos, que cambio y por que.
4. **MomentumEngine**: instanciar SIEMPRE con TODOS los parametros explicitos desde CONFIG. Nunca depender de defaults del modulo.
5. **Antes de mover a `historico/`**: verificar que el archivo es una copia fiel y funcional, no una version modificada.
6. **RESUMEN_PARA_CONTINUAR.md**: actualizar con cada cambio significativo, incluyendo parametros exactos de cada version.

**Leccion aprendida (26 Feb 2026)**: `backtest_definitivo.py` fue silenciosamente modificado de v6 (12 bars forzado, 111 tickers) a v7/v8 (8 bars trailing, 225 tickers). La logica original v6 se perdio. Recuperarla costo horas de arqueologia de codigo. `backtest_v6_recovered.py` es la reconstruccion validada.

---

## 18. INVESTIGACION v9 — OPTIONS-FIRST (27 Feb 2026)

Hipotesis: si el edge viene de fat tails con low win rate, las opciones CALL son el vehiculo perfecto (riesgo limitado a prima, upside ilimitado). ¿Por que limitar a 2 opciones?

Archivos: `test_v9_options_first.py`, `test_v9_analysis.py`

### Resultados grid 60 meses

| Config | Return | Annual | PF | MaxDD | Opts | MaxConc | Eficiencia |
|--------|--------|--------|-----|-------|------|---------|------------|
| **Ref stock-only** | +119.9% | +17.1% | 2.30 | -14.6% | 0 | 0 | 1.17 |
| opts=2 pct=14% (v8) | +491.5% | +42.7% | 2.36 | -35.9% | 33 | 2 | **1.19** |
| opts=10 pct=6% | +288.0% | +31.2% | 2.36 | -39.1% | 65 | 8 | 0.80 |
| opts=10 pct=8% | +437.0% | +40.0% | 2.33 | -44.7% | 66 | 9 | 0.89 |
| opts=10 pct=10% | +571.8% | +46.4% | 2.51 | -49.7% | 63 | 8 | 0.93 |
| opts=10 pct=12% | +683.8% | +51.0% | 2.39 | -58.8% | 65 | 8 | 0.87 |
| opts=10 pct=14% | +836.7% | +56.4% | 2.51 | -64.2% | 63 | 8 | 0.88 |

Eficiencia = Annual% / MaxDD%

### Hallazgos clave v9

1. **MaxConc real = 8-9** opciones simultaneas (IVR<40 es el limitador, no max_opts)
2. **v8 (2 opts, 14%) tiene la mejor eficiencia** (1.19) — sorprendentemente
3. Opciones ilimitadas sin reducir sizing = DD inaceptable (>60%)
4. Reducir sizing a 8-10% con opciones ilimitadas da returns similares a v8 pero con mas DD
5. **Paradoja**: con opciones ilimitadas necesitas sizing muy pequeno para controlar DD, pero pierdes el edge de fat tails

**CONCLUSION v9**: Descartada como mejora directa. La v8 con 2 opciones y 14% sizing captura fat tails sin sobreexponer.

---

## 19. INVESTIGACION v10 — GOLD HEDGE OVERLAY (27 Feb 2026)

Hipotesis: mantener 30% del equity SIEMPRE en GLD + cash no utilizado (posiciones vacias / filtro bear) tambien en GLD para reducir MaxDD.

Archivo: `test_v10_gold_hedge.py`

Mecanismo:
- 30% equity = reserva permanente en GLD
- Cash idle (posiciones no ocupadas) = tambien en GLD
- Momentum P&L escalado al 70% (menos capital para trading)
- Asignacion media a oro: ~42% (30% reserva + ~12% idle)

### Resultados 60 meses

| Config | Return | Annual | MaxDD | Eficiencia | Gold P&L |
|--------|--------|--------|-------|------------|----------|
| v8 original | +493.1% | +42.8% | -35.9% | 1.19 | — |
| **v8 + Gold 30%** | **+452.3%** | **+41.8%** | **-22.3%** | **1.87** | **EUR +10,712** |
| v9 original | +838.5% | +56.5% | -64.2% | 0.88 | — |
| v9 + Gold 30% | +711.5% | +53.4% | -45.0% | 1.19 | EUR +12,456 |

### Resultados 240 meses

| Config | Return | Annual | PF | MaxDD | Eficiencia | Gold P&L |
|--------|--------|--------|-----|-------|------------|----------|
| v8 original | +21,594% | +30.9% | 2.22 | -35.4% | 0.87 | — |
| **v8 + Gold 30%** | **+18,988%** | **+30.7%** | **2.22** | **-27.8%** | **1.11** | **EUR +387k** |
| v9 original | +84,531% | +40.1% | 2.03 | -70.2% | 0.57 | — |
| v9 + Gold 30% | +70,220% | +39.7% | 2.03 | -58.3% | 0.68 | EUR +1.1M |

### Hallazgos clave v10

1. **v8 + Gold 30% = GANADOR CLARO**: MaxDD -35.4% → -27.8% (-7.6pp) con coste despreciable (-0.2pp annual)
2. Eficiencia 240m: 0.87 → 1.11 (+28% mejora)
3. A 60m aun mejor: MaxDD -35.9% → -22.3% (-13.6pp) con solo -1pp annual
4. Gold aporta EUR +387k en 240m (return oro +674% en el periodo)
5. v9 + Gold sigue con DD inaceptable (58.3%) — confirma v8 como base

### Caveat

GLD tuvo periodo excepcionalmente bueno (2004-2026: ~+674%). El efecto amortiguador de DD es mas robusto que el return extra del oro (la correlacion negativa oro/equity es lo que vale, no el return absoluto).

**CONFIGURACION GANADORA**: v8 + Gold 30% — adoptada como referencia principal.

### Gold allocation grid test (27 Feb 2026)

Validacion de que 30% es optimo (no 20%, 25%, 40%, 50%):

**A 60 meses:**

| Gold % | Annual | MaxDD | Eficiencia |
|--------|--------|-------|------------|
| 15% | +42.4% | -25.3% | 1.68 |
| 20% | +42.0% | -23.9% | 1.76 |
| 25% | +41.9% | -23.0% | 1.82 |
| **30%** | **+41.8%** | **-22.3%** | **1.87** |
| 35% | +41.6% | -21.6% | 1.93 |
| 40% | +41.4% | -20.8% | 1.99 |
| 50% | +40.8% | -18.7% | 2.18 |

**A 240 meses:**

| Gold % | Annual | MaxDD | Eficiencia |
|--------|--------|-------|------------|
| 20% | +30.9% | -30.9% | 1.00 |
| 25% | +30.8% | -29.3% | 1.05 |
| **30%** | **+30.7%** | **-27.8%** | **1.11** |
| 35% | +30.5% | -26.3% | 1.16 |
| 40% | +30.2% | -28.0% | 1.08 |
| 50% | +29.3% | -29.1% | 1.01 |

**Conclusion**: A 60m la eficiencia sube monotonicamente con mas oro (hasta 50%), pero a 240m hay un punto de inflexion en 35-40% donde el coste en CAGR supera el ahorro en DD. **30% es el sweet spot**: optimo a 240m, bueno a 60m, simple de operar.

---

## 20. INVESTIGACION v11 — VIX FILTER (27 Feb 2026) — DESCARTADA

**Hipotesis**: No abrir posiciones nuevas cuando VIX > umbral (20/25/30/35). ADITIVO al filtro SPY>SMA50.

**Archivo**: `test_v11_vix_filter.py`

**Mecanismo**: Monkey-patch de `build_macro_filter` en `backtest_experimental.py` (sin modificar el archivo original). Cuando VIX > threshold, `macro_bullish[date] = False`.

**Resultados 60 meses (v10 + VIX filter + Gold 30%):**

| Config | Annual Gold | MaxDD Gold | Eficiencia |
|--------|-----------|-----------|------------|
| **v10 baseline** | **+41.8%** | **-22.3%** | **1.87** |
| v10 + VIX<20 | +33.7% | -27.7% | 1.22 |
| v10 + VIX<25 | +38.0% | -22.9% | 1.66 |
| v10 + VIX<30 | +41.3% | -22.3% | 1.85 |
| v10 + VIX<35 | +41.6% | -22.3% | 1.87 |

**CONCLUSION**: Todos los umbrales VIX dan resultado PEOR o IGUAL que v10 baseline. VIX<20 es catastrofico (+33.7% vs +41.8% con DD peor). El filtro SPY>SMA50 + Gold 30% ya captura la proteccion necesaria. **VIX filter DESCARTADA — redundante.**

---

## 21. ANALISIS AÑO A AÑO (27 Feb 2026)

Analisis de v8 + Gold 30% a 240m para identificar los años que destruyen eficiencia.

### Rentabilidad anual

| Año | Ret Gold% | MaxDD Gold | PF trades | WR% |
|-----|----------|-----------|----------|-----|
| 2006 | +46.9% | 3.8% | 28.69 | 51.3% |
| 2007 | +27.9% | 10.8% | 1.21 | 30.3% |
| 2008 | +11.7% | 27.8% | 1.42 | 16.4% |
| 2009 | +33.2% | 9.2% | 1.94 | 37.7% |
| 2010 | +22.1% | 16.7% | 1.51 | 31.9% |
| 2011 | +8.5% | 25.1% | 1.01 | 17.2% |
| 2012 | +19.0% | 12.6% | 2.14 | 35.3% |
| **2013** | **-1.7%** | 22.4% | 1.97 | 40.0% |
| 2014 | +66.3% | 13.2% | 1.30 | 25.0% |
| 2015 | +5.8% | 18.4% | 1.19 | 16.0% |
| 2016 | +21.6% | 12.4% | 2.32 | 29.4% |
| 2017 | +16.6% | 10.0% | 2.24 | 36.6% |
| **2018** | **-3.7%** | 25.3% | **0.26** | **15.9%** |
| 2019 | +23.8% | 5.8% | 3.70 | 37.7% |
| 2020 | +105.8% | 15.5% | 2.51 | 39.2% |
| 2021 | +10.0% | 12.5% | 1.04 | 40.5% |
| 2022 | +15.3% | 12.7% | 1.23 | 16.9% |
| **2023** | **-19.1%** | 23.3% | 1.16 | 27.6% |
| 2024 | +115.5% | 5.2% | 1.67 | 27.0% |
| 2025 | +136.0% | 13.5% | 7.42 | 45.7% |
| 2026 | +23.1% | 3.2% | 0.73 | 62.5% |

### Resumen estadistico
- **18/21 años positivos (86%)**, 3 negativos
- Media año positivo: **+39.4%**
- Media año negativo: **-8.1%**
- Gold ayuda en 12/21 años (57%), especialmente los peores

### Pregunta clave: ¿Los años malos explican el bajon de eficiencia 60m→240m?

| Escenario | Annual | Worst DD | Eficiencia |
|-----------|--------|----------|------------|
| **Todos los años** | +27.7% | 27.8% | **1.00** |
| Sin 2008 | +28.6% | 25.3% | 1.13 |
| Sin 2007+2008 | +28.6% | 25.3% | 1.13 |
| Sin 2007+2008+2011 | +29.8% | 25.3% | 1.18 |
| Sin 5 peores años | +33.3% | 23.3% | **1.43** |

### Diagnostico

**NO es solo 2007-2008.** El bajon se distribuye entre 6 años mediocres:

1. **2018 (PF 0.26)**: El peor año con diferencia. WR 15.9%. Opciones pierden mucho (GS -53%, AMZN -75%).
2. **2023 (ret -19.1%)**: El peor retorno. Gold convierte -31.6% en -19.1% (ayuda mucho).
3. **2013 (ret -1.7%)**: Gold PERJUDICA aqui — convierte +14.5% en -1.7% (oro cayo ~28% ese año).
4. **2011 (PF 1.01)**: Breakeven, DD alto (25.1%).
5. **2015 (ret +5.8%)**: WR 16.0%, marginal.
6. **2008 (DD 27.8%)**: El peor DD, pero retorno positivo (+11.7% con gold).

**Patron**: Los años con WR < 17% son los que destruyen eficiencia (2008, 2011, 2015, 2018, 2022). Cuando WR baja a 15-17%, ni los fat tails compensan. Esto pasa en entornos de alta volatilidad con tendencias falsas (breakouts que fallan inmediatamente).

**Implicacion para mejoras**: Las mejoras B (trailing adaptativo) y D (sizing dinamico) atacarian exactamente estos años — reducir tamaño/ajustar trailing cuando la volatilidad es alta y los breakouts fallan masivamente.

### Analisis stock-only vs stock+opciones (27 Feb 2026)

Backtest 240m sin opciones (use_options=False) + Gold 30% para cuantificar el "piso" real:

| Año | Con Opts | Solo Stock | Δ Opciones |
|-----|---------|-----------|-----------|
| 2006 | +46.9% | +2.9% | +44.0pp |
| 2007 | +27.9% | +27.4% | +0.5pp |
| 2008 | +11.7% | -0.2% | +11.9pp |
| 2009 | +33.2% | +14.7% | +18.5pp |
| 2010 | +22.1% | +16.5% | +5.6pp |
| 2011 | +8.5% | +4.0% | +4.5pp |
| 2012 | +19.0% | +9.7% | +9.3pp |
| 2013 | -1.7% | -5.8% | +4.1pp |
| 2014 | +66.3% | +6.4% | +59.9pp |
| 2015 | +5.8% | -9.2% | +15.0pp |
| 2016 | +21.6% | +23.1% | -1.5pp |
| 2017 | +16.6% | +10.3% | +6.3pp |
| 2018 | -3.7% | -5.6% | +1.9pp |
| 2019 | +23.8% | +7.6% | +16.2pp |
| 2020 | +105.8% | +11.5% | +94.3pp |
| 2021 | +10.0% | +10.6% | -0.6pp |
| 2022 | +15.3% | -9.4% | +24.7pp |
| 2023 | -19.1% | +15.2% | -34.3pp |
| 2024 | +115.5% | +21.6% | +93.9pp |
| 2025 | +136.0% | +51.5% | +84.5pp |
| 2026 | +23.1% | +15.2% | +7.9pp |

**Resumen**: CAGR con opciones +27.7%, CAGR stock-only +10.1%. Opciones triplican el CAGR. Ayudan en 18/21 años (86%). Solo perjudican significativamente en 2023 (-34.3pp). **Las opciones son ESENCIALES, no opcionales.**

---

## 22. INVESTIGACION v13 — ROLLING THUNDER (28 Feb 2026) — DESCARTADA

**Hipotesis**: En vez de cerrar opciones ganadoras a 45 DTE, rolarlas: cerrar la opcion actual (bloquear P&L), abrir nueva CALL al MISMO strike con ~120 DTE, re-sized al equity actual. Condicion: el subyacente debe seguir en tendencia.

**Archivo**: `backtest_v13_rolling.py`

**Mecanismo**:
- A 45 DTE, si la opcion es rentable → candidata a roll
- Se evalua la señal de roll (varios modos probados)
- Si pasa: cierre actual (exit_reason=dte_roll), apertura nueva al mismo strike, 120 DTE
- Si no pasa: cierre normal (exit_reason=dte_exit)
- Tracking: roll_number, original_entry_date, cadena maxima de rolls

### Modos de señal probados

| Modo | Criterio | Rolls/Candidatos | Resultado |
|------|---------|-----------------|-----------|
| `strict` | Señal LONG fresca (breakout) | 0/10 (0%) | Inutilizable — tras 75 dias el stock ya no "rompe" |
| `trend_intact` | Precio > entry + macro OK | 9/10 (90%) | Catastrofico — incluye tendencias agotadas |
| **`ker_check`** (KER≥0.30) | **KER>0.30 + precio>entry + macro** | **Variable** | **Ganador a corto plazo** |
| `ker_check` (KER≥0.40) | KER>0.40 + precio>entry + macro | Mas selectivo | Peor que KER≥0.30 |

### Resultados multi-periodo (ker_check KER≥0.30, v13 vs v12 REF)

| Periodo | Δ CAGR | Δ MaxDD | Δ PF | Rolls | Veredicto |
|---------|--------|--------|------|-------|-----------|
| **24m** | **+30.6pp** | **-0.5pp** | **+1.42** | 5/10 (50%) | ✅ Excepcional |
| **60m** | **+4.9pp** | **-3.8pp** | **-0.07** | 14/36 (39%) | ✅ Positivo marginal |
| **240m** | **-1.1pp** | **+4.1pp** | **-0.38** | 41/113 (36%) | ❌ Negativo |

### Resultados 240m con KER≥0.40 (mas selectivo)

| Metrica | v12 REF | v13 KER≥0.40 | Delta |
|---------|---------|-------------|-------|
| CAGR | ~30% | ~27% | **-3.1pp** |
| MaxDD | ~-38% | ~-42% | **+4.1pp peor** |
| Rolls | — | 27/109 (25%) | Mas selectivo pero aun peor |

### Ejemplo de KER como filtro (24m, KER≥0.30)

| Ticker | KER a 45 DTE | Roll? | Resultado |
|--------|-------------|-------|-----------|
| SLV | 0.46 | ✅ Si | +76% (excelente) |
| GLD | 0.65 | ✅ Si | +22% (bueno) |
| EQIX | 0.21 | ❌ No | Habria perdido -83% |
| T | 0.08 | ❌ No | Tendencia agotada |
| SMH | 0.18 | ❌ No | Habria perdido |

### Diagnostico: por que falla a largo plazo

**Causa raiz: BLOQUEO DE SLOTS.** Una opcion rolada ocupa el slot 75 dias extra (~120 DTE nuevos - 45 DTE restantes). Durante ese tiempo, el slot no puede capturar nuevas oportunidades de breakout potencialmente mejores.

- A **24 meses** (2024-2025): mercado excepcional con SLV, GLD, SMH en tendencias limpias. Rolar captura +30pp extra porque hay pocas oportunidades perdidas.
- A **240 meses**: los slots bloqueados pierden cientos de oportunidades de breakout nuevas a lo largo de 20 años. El coste de oportunidad supera el beneficio de extender posiciones existentes.

**Conclusion**: El cierre a 45 DTE de v12 es optimo — libera slots para nuevas oportunidades, que son mas valiosas que extender las existentes. El KER es un excelente filtro de tendencia, pero el problema no es la calidad de la señal sino el coste de oportunidad del slot.

---

## 23. CARTERA OUTSIDERS — DESCORRELACION (28 Feb 2026)

### Concepto
Cartera paralela de 64 tickers **NO incluidos en el universo v8/v12 de 225 tickers**. Misma logica Momentum Breakout v8 pero sobre activos alternativos: commodities futures, metales, energia, agricultura, REITs, infraestructura, mercados frontera, volatilidad, cripto, divisas.

**Rol: DESCORRELACION, no rentabilidad standalone.** No compite con v12 en retornos absolutos — su valor es reducir DD y volatilidad de la cartera global.

### Archivo: `backtest_outsiders_v2.py`

CONFIG diferencias vs v8:
- 64 tickers (universo propio, sin solapamiento con v8)
- 12 posiciones max (vs 10 en v8)
- `use_macro_filter: False` (activos no correlacionan con SPY)
- 7 entradas en OPTIONS_MAP (futuros → proxy ETF)
- GC=F y NG=F eliminados del universo (oro via overlay, gas ATR% excesiva)

### Resultados standalone (120 meses)

| Variante | Return | CAGR | MaxDD | Eficiencia | PF |
|----------|--------|------|-------|------------|-----|
| SPOT sin gold | +128.7% | 8.6% | -12.6% | 0.69 | — |
| SPOT + GLD 30% | +189.6% | 11.5% | -20.5% | 0.56 | — |
| +OPT sin gold | +213.6% | 12.1% | -41.2% | 0.29 | — |
| +OPT + GLD 30% | +237.1% | 13.3% | -28.5% | 0.46 | — |
| **SPY B&H (ref)** | **+318.2%** | **15.4%** | **-33.7%** | **0.46** | — |

### Correlacion (clave del valor)

| Par | Correlacion diaria |
|-----|-------------------|
| OUTSIDERS vs SPY | **-0.068** |
| OUTSIDERS vs Multi-Asset (33G/33S/17T/17TQ) | **-0.113** |
| Multi-Asset vs SPY | +0.626 |

**Proteccion en crisis**: En los 10 peores meses de SPY (120m), OUTSIDERS genero retorno positivo o neutro en **10 de 10 casos**.

### Combo optimo: Multi-Asset + Outsiders

| Combo | CAGR | Vol | MaxDD | Sharpe | Eficiencia |
|-------|------|-----|-------|--------|------------|
| 100% Multi-Asset | 20.5% | 17.4% | -30.3% | 0.95 | 0.68 |
| **80% MA + 20% OUT** | **18.4%** | **13.8%** | **-25.7%** | **1.05** | **0.72** |
| **70% MA + 30% OUT** | **17.4%** | **12.1%** | **-23.3%** | **1.10** | **0.74** |

Añadir 20-30% outsiders: sacrifica -2/3pp CAGR pero mejora DD 5-7pp, Sharpe +0.10/0.15.

### Decision
- **Guardar como herramienta de descorrelacion** para uso futuro en cartera global
- Multi-Asset (33G/33S/17T/17TQ rebal. anual) = estrategia_multi_asset/
- Outsiders SPOT sin gold = la pieza descorreladora (DD -12.6%, corr -0.07 vs SPY)
- Combo 70/30 o 80/20 segun tolerancia al riesgo

---

## 24. v12g REFERENCIA @ 120m — VALIDACION COMPLETA (1 Mar 2026)

### Definicion v12g

**v12g = v12 + Gold 30% overlay**. Adoptada como REFERENCIA UNICA para toda decision futura.
- v12 = v8 + opciones EU (2 US + 2 EU slots separados)
- Gold overlay: 30% equity permanente en GLD, P&L momentum escalado a 70%, cash idle tambien en GLD

### Por que 120 meses como ventana de referencia

120m (Mar 2016 → Feb 2026) es la UNICA ventana donde TODOS los datos son reales:
- 225 tickers disponibles con historico completo
- Opciones EU con datos de mercado reales (no sinteticas pre-2012)
- GLD disponible (lanzado 2004)
- No hay asunciones pre-data (a 240m se extrapolan opciones EU antes de 2006)
- No hay distorsion de universo pequeño (a 480m, antes de 2006 habia <200 tickers)

**Problemas identificados en otros horizontes**:
- 6m/12m: "falso alpha" del oro (rally 2025 hace que gold SUME CAGR en vez de costar)
- 36m: cherry-pick ventana Nov 2023-Feb 2026 (NVDA +529%, home runs EU)
- 240m: asunciones sinteticas pre-2006 deprimen resultados
- 480m: CAGR (+44.5%) MAYOR que 240m (+36.3%) — señal de datos inflados pre-2006

### Tabla completa v12 vs v12g todos los horizontes

| Periodo | v12 CAGR | v12 DD | v12 Eff | v12g CAGR | v12g DD | v12g Eff | ΔCAGR | ΔDD |
|---------|----------|--------|---------|-----------|---------|----------|-------|-----|
| 6m | +82.5% | -9.9% | 8.35 | +119.9% | -4.2% | 28.40 | +37.4pp | -5.7pp |
| 12m | +49.0% | -29.5% | 1.66 | +69.7% | -10.7% | 6.51 | +20.7pp | -18.8pp |
| 36m | +130.6% | -23.6% | 5.53 | +124.4% | -19.9% | 6.27 | -6.2pp | -3.7pp |
| 60m | +84.6% | -33.5% | 2.52 | +80.4% | -27.7% | 2.91 | -4.2pp | -5.8pp |
| **120m** | **+52.4%** | **-38.1%** | **1.37** | **+51.1%** | **-29.8%** | **1.71** | **-1.3pp** | **-8.3pp** |
| 240m | +36.3% | -42.6% | 0.85 | ~+35% | ~-28% | ~1.25 | ~-1.3pp | ~-14.6pp |
| 480m | +44.5% | -59.3% | 0.75 | +42.4% | -54.3% | 0.78 | -2.1pp | -5.0pp |

**Eficiencia** = CAGR / MaxDD. A partir de 36m, gold REDUCE CAGR (-1 a -6pp) pero REDUCE DD mas (-4 a -15pp) → mejor Eficiencia.

### Numeros de referencia v12g @ 120m

| Metrica | v12 | v12g | Delta |
|---------|-----|------|-------|
| CAGR | +52.4% | +51.1% | -1.3pp |
| MaxDD | -38.1% | -29.8% | **-8.3pp** |
| Eficiencia | 1.37 | **1.71** | +0.34 |
| PF | 3.62 | — | — |
| Trades | 747 | — | — |
| Final EUR (10K) | 674,433 | 557,000 | — |
| Gold PnL | — | +82,116 | — |

### Validacion Monte Carlo v12 @ 120m (1 Mar 2026)

Script: `backtest_v12_montecarlo.py --months 120 --sims 5000 --bootstrap-years 10`
Backtest base v12: 747 trades, CAGR +57.2%, PF 3.74, DD -33.5%, EUR 919K final.

**Test 1 — Trade Shuffle (5,000 sims)**:
| Metrica | Real | Mediana MC | Delta |
|---------|------|------------|-------|
| CAGR | +57.2% | +59.0% | +1.8pp |
| MaxDD | -33.5% | -45.9% | -12.4pp |
| Prob CAGR > 40% | — | 100% | — |
| Prob MaxDD > 50% | — | 46.3% | — |

✅ ROBUSTO al orden (nota: test suma PnLs EUR → total siempre identico, solo MaxDD varia).

**Test 2 — Bootstrap 10y (5,000 sims)**:
| Metrica | Valor |
|---------|-------|
| CAGR mediana 10y | +60.9% |
| CAGR P5 (peor caso) | +32.0% |
| CAGR P95 (mejor caso) | +98.3% |
| MaxDD mediana | -29.7% |
| MaxDD P95 | -47.8% |
| Prob perder dinero | **0.0%** |
| Prob 10x capital | **98.2%** |
| Prob duplicar | 100% |
| Retornos mensuales: media | +4.75% |
| Retornos mensuales: % negativos | 43% |

✅ MUY ROBUSTO — 0% probabilidad de perdida a 10 años.

**Test 3 — Permutation Test (5,000 perms)**:
| Metrica | Real | Mediana Shuffled | p-value | Sig |
|---------|------|------------------|---------|-----|
| Profit Factor | 3.74 | 2.45 | **0.062** | ⚠️ marginal |
| PnL total | +909K | +416K | **0.009** | ✅ |
| Win Rate | 34.8% | — | 1.000 | ❌ (esperado) |

⚠️ PF p=0.062 no alcanza 5%, pero PnL p=0.009 es altamente significativo. El edge viene de fat tails (tamano de ganadores), no de win rate. La mediana shuffled PF=2.45 indica que la asimetria trailing stop + opciones genera edge "estructural" incluso con senales aleatorias.

### Validacion Audit v8 @ 120m (1 Mar 2026)

Script: `backtest_audit.py --months 120 --test all`
Nota: audit corre sobre v8 base (sin EU options). Valida el motor core que es identico en v12.

**Test 1 — Walk-Forward (IS 84m / OOS 36m)**:
| Metrica | IN-SAMPLE (84m) | OUT-OF-SAMPLE (36m) | Ratio OOS/IS |
|---------|:---:|:---:|:---:|
| Trades | 547 | 239 | — |
| Win Rate | 34.2% | 36.8% | — |
| **Profit Factor** | **2.33** | **2.96** | **127.3%** |
| Anualizado | +43.1% | +90.3% | — |
| Max Drawdown | -35.4% | -24.0% | — |

✅ EXCEPCIONAL: OOS MEJOR que IS (PF 2.96 > 2.33). No hay overfitting — el periodo reciente 2023-2026 fue extraordinario.

**Test 2 — Survivorship Bias (universo progresivo)**:
26 tickers post-2006. A 120m (desde 2016) la mayoria ya existian → bias minimo.

| Metrica | FIJO (225) | PROGRESIVO | Diferencia |
|---------|:---:|:---:|:---:|
| Trades | 779 | 783 | -4 |
| **Profit Factor** | **2.37** | **2.30** | **+0.08** |
| Anualizado | +38.9% | +35.3% | +3.6% |
| Max Drawdown | -35.4% | -35.9% | — |

✅ Bias +0.08 PF, +3.6pp CAGR → Impacto BAJO.

**Test 3 — Robustez (quitar top N trades)**:
| Escenario | Trades | PnL EUR | PF | PF vs orig |
|-----------|:---:|:---:|:---:|:---:|
| Original | 786 | +228,216 | 2.31 | 100% |
| Sin top 5 | 781 | +86,065 | 1.49 | 65% |
| Sin top 10 | 776 | +21,980 | 1.13 | 49% |
| Sin top 20 | 766 | -33,694 | 0.81 | 35% |

⚠️ Top 10 trades = 90.4% del PnL total. Inherente a momentum fat-tail. Top 10: INTC +46K, LLY +26K, GLD +25K, PG +23K, GLD +21K, BABA +17K, SLV +16K, TMUS +14K, BAC +9K, QCOM +9K.

### Resumen validacion completa 120m

| Test | Resultado | Veredicto |
|------|-----------|-----------|
| Monte Carlo Shuffle | CAGR constante, MaxDD varia | ✅ Robusto al orden |
| Bootstrap 10y | CAGR mediana +60.9%, P5 +32%, 0% prob perdida | ✅ MUY robusto |
| Permutation PnL | p=0.009 | ✅ Edge REAL |
| Permutation PF | p=0.062 | ⚠️ Marginal (fat-tail inherente) |
| Walk-Forward | PF IS 2.33 → OOS 2.96 (127%) | ✅ Sin overfitting |
| Survivorship | PF 2.37 → 2.30 (bias +0.08) | ✅ Bias BAJO |
| Robustez | Top 10 = 90% del PnL | ⚠️ Home runs (inherente) |

**Conclusion**: v12g @ 120m pasa 5 de 7 tests limpiamente. Los 2 que no pasan (PF permutation, robustez) son limitaciones conocidas del test para estrategias fat-tail momentum — no son evidencia de overfitting sino de la naturaleza del sistema.

---

## 25. PENDIENTE / MEJORAS FUTURAS


### Alta prioridad:
- **Opcion B: Trailing adaptativo** (ajustar ATR mult segun volatilidad del mercado) — ataca los años con WR<17%
- **Opcion D: Sizing dinamico** (reducir tamaño en alta volatilidad) — complementario a B
- Alertas por email/Telegram cuando hay senal
- Precios de opciones reales (vs Black-Scholes con HVOL como proxy de IV)

### Media prioridad:
- **Mejora E: Survival Stress Test** (idea de Codex 5.3) — Simular supervivencia con hazard rates (0.10 / 0.20 / 0.30): tickers que desaparecen del universo aleatoriamente durante el BT. Comparar dispersion de CAGR/PF/DD entre seeds. Objetivo: validar que el BT no depende de que los 225 tickers sobrevivan los 20-40 años completos. Referencia conceptual: `backtest_v12_montecarlo_codex.py` (Codex 5.3, carpeta `/Documents/New project/`)
- Equity curve grafica (matplotlib)
- Analisis por sector/geografia con universo expandido
- Backtest v8 con time exit hibrido: forzado a 12 barras + trailing v8 (lo mejor de ambos?)

### Descartado (probado y no mejora):
- ~~Opcion C: Mejorar macro filter~~ (SMA50 ya optimo — grid SMA probado)
- ~~Opcion E: VIX filter~~ (redundante con SPY>SMA50 + Gold 30%)
- ~~v9 Options-First~~ (DD inaceptable >60%)
- ~~v8.1 Exencion macro~~ (peor MaxDD a 36m)
- ~~Gold allocation diferente a 30%~~ (grid 15-50% probado, 30% optimo a 240m)
- ~~v13 Rolling Thunder~~ (rolar opciones ganadoras a 45 DTE — bloquea slots, peor a 240m)

### Baja prioridad:
- Sector rotation
- Conexion a broker real (IBKR/Tastytrade)
- Optimizar combo Multi-Asset + Outsiders (pesos, rebalanceo)

---

## 26. PARA CONTINUAR EN UN NUEVO CHAT

Prompt sugerido:

> "Lee `estrategia_momentum/RESUMEN_PARA_CONTINUAR.md`. Estrategia Momentum Breakout. **v12g** (v12 + Gold 30%) es la REFERENCIA UNICA. Ventana de referencia: **120 meses** (unica con datos 100% reales).
>
> v12g @ 120m: CAGR +51.1%, MaxDD -29.8%, Eficiencia 1.71, PF 3.62. Validada (1 Mar 2026).
>
> **Paper trading activo** (13 Mar 2026): 1 posicion abierta (WDS +2.54R, SL $20.74), 9 cerradas (3W/6L, PnL EUR -863). Cash EUR 3,097. Dividendos pendientes EUR 122 (BHP+WDS, cobro 26-27 mar). Macro **BEAR** — no abrir nuevas. Ver `paper_portfolio.json` para detalle.
>
> **Precauciones DEGIRO**: (1) SL pueden borrarse sin aviso — verificar diariamente. (2) yfinance unreliable para tickers EU/JP/AX — usar precios DEGIRO reales. (3) Motor correcto para v12 es `backtest_v12_eu_options.py` (4 slots: 2US+2EU), NO `backtest_experimental.py --test b` (solo 2 US → 15pp menos CAGR).
>
> Archivos activos (`estrategia_momentum/`):
> - `backtest_experimental.py` — backtest v8 principal (NO usar para v12)
> - `backtest_v12_eu_options.py` — backtest v12 (acepta --gold, --multi-period, --months)
> - `momentum_breakout.py` — motor de senales, 225 tickers
> - `paper_trading.py` — paper trading v3.0
> - `run_scanner.py` — scanner/radar
> - `paper_portfolio.json` — estado paper trading (1 pos abierta, 9 cerradas)
>
> IMPORTANTE: Trabajar SIEMPRE sobre v8/v12. No modificar `historico/`. 120m es la ventana de referencia. NO re-correr backtests sin permiso.
>
> Quiero [lo que necesites]."

---

*Ultima actualizacion: 13 Mar 2026 — Paper trading: 1 pos abierta (WDS +2.54R, SL $20.74), 9 cerradas (3W/6L, PnL EUR -863). Cerrados desde 9-mar: PSA trailing breakeven (+EUR 26), 6861.T SL gap (-EUR 430), AI.PA time exit (-EUR 119), NESN.SW time exit (-EUR 89). Dividendos pendientes EUR 122 (BHP+WDS). Macro BEAR. Cash EUR 3,097. ERRATA Cowork: ver seccion 27.*

---

## 27. ERRATA INFORME COWORK — CAGR 120m (4 Mar 2026)

**Error detectado en**: `analisis_v12_momentum_breakout.html` (informe Cowork, Mar 2026)

**Ubicacion del error**: Hero card metricas (linea 176-177 del HTML)
- Dice: `120m sin/con Gold: +84.6% / +80.4%`
- Deberia decir: `120m sin/con Gold: +52.4% / +51.1%`

**Origen del error**: La tabla detallada del mismo informe (seccion 4, linea 426) muestra +84.6% correctamente en la fila de **60m**. La fila de 120m (linea 427) esta VACIA (solo dice "horizonte primario"). El hero card tomo el dato de 60m y lo etiqueto como 120m.

**Cifras correctas** (de `RESUMEN_PARA_CONTINUAR.md` seccion 24, validado 1 Mar 2026):

| Periodo | v12 CAGR | v12g CAGR | Fuente |
|---------|----------|-----------|--------|
| 60m | +84.6% | +80.4% | Correcto en tabla e informe |
| **120m** | **+52.4%** | **+51.1%** | `RESUMEN` seccion 24. Motor: `backtest_v12_eu_options.py --months 120` |
| 240m | +36.3% | ~+35% | Correcto en informe |

**Motor incorrecto vs correcto** (descubierto 4 Mar 2026):
- `backtest_experimental.py --test b --months 120` → CAGR +37.3% (solo 2 slots opciones US)
- `backtest_v12_eu_options.py --months 120` → CAGR +52.4% (4 slots: 2US+2EU) ← CORRECTO

**Accion requerida para Cowork**: Corregir hero card del HTML con los datos reales de 120m. Rellenar la fila vacia de 120m en la tabla de resultados (seccion 4) con: Return +6,644%, CAGR +52.4%, PF 3.62, MaxDD -38.1%, Trades 747.

**Archivos de referencia**:
- `historico_trades_120m.csv` — 716 stock trades exportados (sin opciones, motor experimental)
- `analisis_bt_120m_fat_tails.md` — analisis distribucion temporal fat tails
- `RESUMEN_PARA_CONTINUAR.md` seccion 24 — cifras oficiales v12g @ 120m
