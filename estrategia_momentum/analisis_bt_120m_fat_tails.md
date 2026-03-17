# Backtest 120 Meses — Analisis Fat Tails y Distribucion Temporal
**Fecha**: 4 marzo 2026
**Periodo**: Mar 2016 — Mar 2026 (120 meses)
**Estrategia**: Momentum Breakout v8 + Opciones CALL US (solo 2 slots US)
**Motor**: backtest_experimental.py --months 120 --test b --export-csv

**NOTA**: Este analisis usa el motor `backtest_experimental.py` (v8, solo opciones US, 2 slots).
Los numeros de REFERENCIA de v12g usan `backtest_v12_eu_options.py` (4 slots: 2US+2EU) → CAGR +52.4% sin Gold, +51.1% con Gold.
La diferencia de ~15pp viene de las opciones EU adicionales. El analisis de fat tails (distribucion temporal de R-multiplos) es valido para ambos motores ya que solo usa stock trades.

---

## Resumen del Backtest (motor experimental, NO referencia v12)

| Metrica | Valor (experimental) | Valor v12 (referencia) |
|---------|---------------------|----------------------|
| Capital Inicial | EUR 10,000 | EUR 10,000 |
| Capital Final | EUR 237,999 | EUR 674,433 |
| Return Total | +2,280% | +6,644% |
| CAGR | +37.3% | **+52.4%** |
| Max Drawdown | -35.9% | -38.1% |
| Total Trades | 782 (716 stocks + 66 opts US) | 747 |
| Win Rate | 32.5% | — |
| Profit Factor | 2.25 | 3.62 |
| Avg Win | +23.3% |
| Avg Loss | -5.3% |
| Stocks >= 3R | 127 |
| Options >= 100% | 14 |

---

## Concentracion de Ganancias (Fat Tails)

| Grupo | PnL EUR | % del Total |
|-------|---------|-------------|
| Top 10 trades (por %) | +31,018 | 45% |
| Top 20 trades (por %) | +50,634 | 74% |
| Restantes 696 trades | +18,085 | 26% |

**Conclusion**: El 2.8% de los trades (top 20 de 716) genera el 74% de la rentabilidad total.
Esto es CONSISTENTE con los resultados de 36 meses (76% top 20).

---

## Pregunta Clave: ¿Estan los ganadores distribuidos en el tiempo?

### Test 1: Top 20 por PnL% — Distribucion por Año

| Año | Trades en Top 20 |
|-----|------------------|
| 2017 | 1 |
| 2019 | 2 |
| 2020 | 3 |
| 2021 | 1 |
| 2022 | 1 |
| 2023 | 2 |
| 2024 | 3 |
| 2025 | 7 |

**Primera mitad (2016-2020)**: 6 trades en top 20
**Segunda mitad (2021-2026)**: 14 trades en top 20

Hay sesgo hacia la segunda mitad, pero NO es por compounding. Es por:
1. **ETFs apalancados (TQQQ, SPXL)**: aparecen 3x en top 20, y sus movimientos % son mayores
2. **Universo**: mas tickers disponibles en el periodo reciente
3. **2025 fue excepcionalmente bueno**: win rate 43.2% vs media de ~30%

### Test 2: Top 20 por R-MULTIPLO (medida pura, independiente del tamaño)

El R-multiplo = ganancia / riesgo asumido. Es la medida MAS LIMPIA de calidad del trade.

| Año | Trades en Top 20 por R |
|-----|------------------------|
| 2017 | 5 |
| 2019 | 1 |
| 2020 | 3 |
| 2021 | 2 |
| 2023 | 2 |
| 2024 | 2 |
| 2025 | 5 |

**Primera mitad**: 9 trades | **Segunda mitad**: 11 trades

**Top 50 por R**: 21 vs 29 (42% / 58%)

**CONCLUSION CRITICA**: Por R-multiplo, la distribucion es CASI equilibrada.
Los fat tails son ESTRUCTURALES — no son producto del compounding.

### Test 3: Tamaño de Posicion por Año (efecto compounding)

| Año | Pos Media (EUR) | vs Inicial |
|-----|-----------------|------------|
| 2016 | 1,066 | 1.0x |
| 2017 | 1,448 | 1.4x |
| 2018 | 1,790 | 1.7x |
| 2019 | 1,608 | 1.5x |
| 2020 | 3,063 | 2.9x |
| 2021 | 4,072 | 3.8x |
| 2022 | 5,546 | 5.2x |
| 2023 | 4,713 | 4.4x |
| 2024 | 6,799 | 6.4x |
| 2025 | 13,138 | 12.3x |
| 2026 | 24,256 | 22.7x |

Esto confirma que el PnL ABSOLUTO si escala con compounding, pero el PnL% y el R-multiplo NO.

---

## Top 20 Trades por R-Multiplo (lista completa)

| # | Entry | Ticker | R mult | PnL% | PnL EUR |
|---|-------|--------|--------|------|---------|
| 1 | 2020-11-09 | AMAT | 16.9R | +78.7% | +3,029 |
| 2 | 2024-01-10 | NVDA | 15.1R | +57.7% | +2,290 |
| 3 | 2019-08-02 | AAPL | 14.9R | +48.4% | +827 |
| 4 | 2025-10-10 | NOK | 11.6R | +30.8% | +4,211 |
| 5 | 2023-11-07 | ISP.MI | 11.6R | +33.9% | +1,319 |
| 6 | 2025-12-23 | SLV | 10.9R | +34.4% | +7,187 |
| 7 | 2020-10-13 | SOYB | 10.7R | +22.3% | +925 |
| 8 | 2017-09-08 | HD | 10.7R | +23.7% | +366 |
| 9 | 2025-01-29 | BABA | 10.4R | +29.2% | +2,610 |
| 10 | 2017-03-03 | SAP | 10.0R | +12.7% | +155 |
| 11 | 2025-08-11 | GDX | 9.9R | +31.5% | +3,976 |
| 12 | 2025-05-22 | PPLT | 9.9R | +23.1% | +2,482 |
| 13 | 2024-10-29 | TSLA | 9.5R | +54.8% | +4,446 |
| 14 | 2023-12-12 | CRH | 9.5R | +28.6% | +1,123 |
| 15 | 2020-10-23 | CORN | 9.4R | +14.6% | +597 |
| 16 | 2017-07-18 | ASML | 9.1R | +24.4% | +386 |
| 17 | 2021-12-02 | BHP.AX | 8.4R | +23.2% | +943 |
| 18 | 2017-03-21 | 0700.HK | 8.2R | +18.2% | +225 |
| 19 | 2017-11-30 | SPY | 8.0R | +5.8% | +101 |
| 20 | 2021-03-29 | COST | 7.8R | +27.9% | +1,127 |

---

## Win Rate y PnL por Año

| Año | Trades | Win% | PnL EUR | Mejor% |
|-----|--------|------|---------|--------|
| 2016 | 57 | 24.6% | -200 | +11.8% |
| 2017 | 83 | 32.5% | +1,422 | +24.4% |
| 2018 | 78 | 15.4% | -3,336 | +12.2% |
| 2019 | 70 | 34.3% | +1,048 | +48.4% |
| 2020 | 72 | 38.9% | +4,040 | +78.7% |
| 2021 | 67 | 44.8% | +5,626 | +27.9% |
| 2022 | 58 | 12.1% | -3,646 | +39.9% |
| 2023 | 68 | 26.5% | +2,259 | +33.9% |
| 2024 | 83 | 25.3% | +4,735 | +57.7% |
| 2025 | 74 | 43.2% | +47,307 | +48.9% |
| 2026 | 6 | 66.7% | +9,463 | +20.6% |

**Patron**: Se pierde dinero en BEAR markets (2018, 2022) pero se recupera rapido en los rebotes.
Los mejores % aparecen en TODOS los años excepto 2016 (primer año, sin compounding ni opciones disponibles).

---

## Conclusiones Finales

1. **Fat tails CONFIRMADOS como estructurales**: Los top 20 trades por R-multiplo estan distribuidos uniformemente (9 primera mitad, 11 segunda). No son producto del compounding.

2. **El compounding afecta solo al PnL absoluto**: La posicion media pasa de EUR 1,066 a EUR 24,256 (x22.7), lo que amplifica las ganancias absolutas pero NO el porcentaje o R-multiplo.

3. **Concentracion extrema es NORMAL**: Top 20 de 716 trades = 74% del profit. Esto es la firma de un sistema trend-following de fat tails. Se repite en 36m (76%) y 120m (74%).

4. **Cada periodo tiene su "lottery ticket"**: AMAT en 2020 (16.9R), NVDA en 2024 (15.1R), AAPL en 2019 (14.9R), HD en 2017 (10.7R). No puedes predecir CUAL sera, pero el sistema los captura consistentemente.

5. **La paciencia es OBLIGATORIA**: Win rate medio de 30-33%. Pierdes 2 de cada 3, pero los ganadores pagan 5-17x el riesgo asumido. Salirse antes de un fat tail (por impaciencia o miedo) destruye la rentabilidad del sistema.

---

## Archivos Generados

- `historico_trades_120m.csv` — 716 stock trades (entry, exit, ticker, prices, pnl, R-mult, exit_reason)
- `analisis_bt_120m_fat_tails.md` — este archivo
- Motor: `backtest_experimental.py --months 120 --test b --export-csv`
