# Rendimiento Momentum v8 con Apalancamiento Bancario

> Actualizado 2026-02-25 con retornos corregidos del backtest (t+2, macro T-2).
> CAGR v8 corregido: 27.5% (21 anos, 2006-2026).

---

## 1. Estructura base del prestamo

| Concepto | Importe |
|----------|---------|
| Equity propio | EUR 10,000 |
| Prestamo bancario | EUR 70,000 (7 anos, sin garantia real) |
| Total invertido | EUR 80,000 |
| Coste deuda | Euribor 12M + 3% |
| Cuota anual (~tipo 5.5%) | ~EUR 12,318 |
| Colchon (2 anos cuotas) | ~EUR 24,635 en cash |
| Capital invertido real | ~EUR 55,365 |

## 2. Tabla de amortizacion (Euribor 2.5% + 3%)

| Ano | Cuotas | Intereses | Amort. capital | Deuda pendiente |
|-----|--------|-----------|----------------|-----------------|
| 1 | 12,318 | 3,850 | 8,468 | 61,532 |
| 2 | 12,318 | 3,384 | 8,934 | 52,598 |
| 3 | 12,318 | 2,893 | 9,425 | 43,173 |
| 4 | 12,318 | 2,375 | 9,943 | 33,230 |
| 5 | 12,318 | 1,828 | 10,490 | 22,740 |
| 6 | 12,318 | 1,251 | 11,067 | 11,672 |
| 7 | 12,318 | 642 | 11,672 | 0 |

Rentabilidad minima necesaria: ~15.4% anual sobre EUR 80,000 para cubrir cuotas.

## 3. Opciones simuladas — prestamo unico (15 ventanas de 7 anos)

### Equity neto final tras 7 anos

| Opcion | Media | Mediana | Mejor | Peor | Impagos |
|--------|-------|---------|-------|------|---------|
| A) 100% invertido | EUR 222,227 | EUR 140,186 | EUR 700,188 | EUR 64,654 | 0/15 |
| B) Colchon 2 anos cuotas | EUR 121,923 | EUR 75,888 | EUR 421,992 | EUR 29,096 | 0/15 |
| C) Amort. acelerada | EUR 128,197 | EUR 59,828 | EUR 566,350 | EUR 33,644 | 0/15 |
| D) Solo equity 10K | EUR 50,095 | EUR 37,975 | EUR 133,473 | EUR 21,808 | N/A |

### Todas las ventanas (opcion B colchon, Euribor +3%)

| Inicio | Fin | D) Solo 10K | B) Colchon +3% | Mult |
|--------|-----|-------------|----------------|------|
| 2006 | 2012 | 52,275 | 151,961 | 2.9x |
| 2007 | 2013 | 35,632 | 67,279 | 1.9x |
| 2008 | 2014 | 50,157 | 78,813 | 1.6x |
| 2009 | 2015 | 49,745 | 118,248 | 2.4x |
| 2010 | 2016 | 37,395 | 63,250 | 1.7x |
| 2011 | 2017 | 37,975 | 75,888 | 2.0x |
| 2012 | 2018 | 27,357 | 79,487 | 2.9x |
| 2013 | 2019 | 26,367 | 74,008 | 2.8x |
| 2014 | 2020 | 49,919 | 136,454 | 2.7x |
| 2015 | 2021 | 31,087 | 29,096 | 0.9x |
| 2016 | 2022 | 35,002 | 38,273 | 1.1x |
| 2017 | 2023 | 21,808 | 33,872 | 1.6x |
| 2018 | 2024 | 44,013 | 51,581 | 1.2x |
| 2019 | 2025 | 133,473 | 408,641 | 3.1x |
| 2020 | 2026 | 119,223 | 421,992 | 3.5x |

En 14 de 15 ventanas la deuda supera al solo equity. Mediana: 2.0x.

---

## 4. Estrategia rolling: renovacion del prestamo al vencimiento

### Concepto

Al liquidar cada prestamo (cada 7 anos), se renueva con un nuevo prestamo proporcional al equity acumulado. Ratio recomendado: **0.7x** (prestamo = 70% del equity neto).

Esto mantiene una estructura de deuda sana similar al margen de un broker, pero con prestamos bancarios a tipo fijo y sin riesgo de margin call.

### Calibracion del ratio: supervivencia a DD -50%

| Ratio | Tras DD -50%: equity/deuda | Status |
|-------|---------------------------|--------|
| 0.5x | 68% | OK - muy conservador |
| **0.7x** | **39%** | **OK - recomendado** |
| 1.0x | 18% | Justo |
| 1.5x | 1% | Peligro |

Con ratio 0.7x, tras un drawdown del -50% (peor que cualquier DD historico del v8), el equity neto sigue siendo el 39% de la deuda. Margen amplio, sin riesgo de impago.

### Prestamos en la simulacion (ratio 0.7x, Euribor +3%)

| # | Ano | Equity en ese momento | Prestamo | Ratio real |
|---|-----|----------------------|----------|------------|
| 1 | 2006 | EUR 10,000 | EUR 70,000 | 7.0x (inicial) |
| 2 | 2013 | EUR 151,961 | EUR 106,373 | 0.7x |
| 3 | 2020 | EUR 473,075 | EUR 331,153 | 0.7x |

### Evolucion ano a ano (backtest completo 21 anos)

| Ano | Ret v8 | Solo 10K | Rolling 0.7x | Mult |
|-----|--------|----------|--------------|------|
| 2006 | +70.3% | 17,026 | 43,803 | 2.6x |
| 2007 | +30.2% | 22,175 | 63,427 | 2.9x |
| 2008 | +11.1% | 24,629 | 69,870 | 2.8x |
| 2009 | +45.7% | 35,884 | 108,375 | 3.0x |
| 2010 | +11.6% | 40,045 | 120,426 | 3.0x |
| 2011 | +5.7% | 42,332 | 126,008 | 3.0x |
| 2012 | +23.5% | 52,275 | **151,961** | 2.9x |
| | | | *→ Renovacion #2: prestamo EUR 106K* | |
| 2013 | +16.1% | 60,667 | 184,075 | 3.0x |
| 2014 | +83.3% | 111,224 | 382,615 | 3.4x |
| 2015 | +10.2% | 122,518 | 423,495 | 3.5x |
| 2016 | +9.5% | 134,187 | 464,672 | 3.5x |
| 2017 | +13.3% | 152,070 | 526,961 | 3.5x |
| 2018 | -23.8% | 115,808 | 400,804 | 3.5x |
| 2019 | +19.0% | 137,834 | **473,075** | 3.4x |
| | | | *→ Renovacion #3: prestamo EUR 331K* | |
| 2020 | +119.7% | 302,844 | 1,298,670 | 4.3x |
| 2021 | +14.2% | 345,765 | 1,499,833 | 4.3x |
| 2022 | +24.0% | 428,831 | 1,878,436 | 4.4x |
| 2023 | -31.8% | 292,633 | 1,238,264 | 4.2x |
| 2024 | +128.7% | 669,303 | 2,885,640 | 4.3x |
| 2025 | +130.9% | 1,545,714 | 6,658,396 | 4.3x |
| 2026 | +6.3% | **1,643,292** | **7,072,460** | **4.3x** |

### Comparativa final (21 anos, 2006-2026)

| Estrategia | Capital final | CAGR | vs Solo 10K | Intereses totales |
|---|---|---|---|---|
| Solo equity 10K | EUR 1,643,292 | 27.5% | 1.0x | 0 |
| Deuda unica 70K | EUR 4,777,009 | 34.1% | 2.9x | EUR 17,651 |
| **Rolling 0.7x** | **EUR 7,072,460** | **36.7%** | **4.3x** | EUR 96,814 |
| Rolling 1.0x | EUR 8,190,425 | 37.6% | 5.0x | EUR 136,836 |

### Valor anadido de cada opcion rolling vs deuda unica

| Estrategia | Capital final | vs Deuda unica |
|---|---|---|
| Rolling 0.3x | EUR 5,707,094 | +19.5% |
| Rolling 0.5x | EUR 6,371,884 | +33.4% |
| **Rolling 0.7x** | **EUR 7,072,460** | **+48.1%** |
| Rolling 1.0x | EUR 8,190,425 | +71.5% |

### Stress test: DD -50% justo despues de renovar

| Prestamo | Equity pre | Deuda | Equity tras DD-50% | Status |
|----------|-----------|-------|-------------------|--------|
| #1 (2006) | EUR 10,000 | EUR 70,000 | EUR -17,682 | Riesgo aceptado (7x inicial) |
| #2 (2013) | EUR 151,961 | EUR 106,373 | EUR 41,512 | OK |
| #3 (2020) | EUR 473,075 | EUR 331,153 | EUR 129,232 | OK |

El prestamo inicial (7x) es el unico que no sobreviviria un DD -50%, pero ese riesgo se asume con solo EUR 10K en juego. Las renovaciones al 0.7x sobreviven con holgura.

---

## 5. Stress Test del prestamo inicial

### Primer ano malo (simulacion mensual, despues 35%/a)

| Retorno 1er ano | Equity 12m | Equity 7 anos | Impago? |
|-----------------|------------|---------------|---------|
| -10% | -427 | 174,303 | No |
| -20% | -7,859 | 129,314 | No |
| -30% | -15,263 | 84,490 | No |
| -40% | -22,635 | 39,867 | No |
| -43.5% | -25,205 | 24,306 | No |

Equity negativo NO es impago. Impago = no poder pagar cuota mensual.

### Drawdowns concentrados (caida rapida + rebote + normal)

| Escenario | Equity min | Equity final | Impago? |
|-----------|-----------|--------------|---------|
| DD -30% en 4m, rebote +20%, luego 35%/a | -14,564 | 175,565 | No |
| DD -40% en 5m, rebote +30%, luego 35%/a | -22,428 | 123,378 | No |
| DD -43.5% en 6m, rebote +35%, luego 35%/a | -25,160 | 96,914 | No |
| DD -30%, SIN rebote, luego 20%/a | -14,719 | 26,016 | No |
| **DD -40%, luego SOLO 15%/a** | -22,388 | -10,411 | **Si (mes 74)** |
| **2 anos malos: -25% + -15%** | -20,770 | -2,677 | **Si (mes 82)** |

### Colchon necesario (escenario DD -40%, rebote, 35%/a)

| Colchon | Cash inicial | Invertido | Equity min | Equity final | Impago? |
|---------|-------------|-----------|-----------|--------------|---------|
| 0 meses | 0 | 80,000 | -22,428 | 123,378 | No |
| 12 meses | 11,676 | 68,324 | -17,758 | 77,096 | No |
| 24 meses | 23,352 | 56,648 | -13,087 | 30,813 | No |

---

## 6. Anos en perdidas (v8 corregido, 21 anos)

| Ano | Retorno v8 | Negativo? |
|-----|-----------|-----------|
| 2006 | +70.26% | |
| 2007 | +30.24% | |
| 2008 | +11.06% | |
| 2009 | +45.70% | |
| 2010 | +11.60% | |
| 2011 | +5.71% | |
| 2012 | +23.49% | |
| 2013 | +16.05% | |
| 2014 | +83.34% | |
| 2015 | +10.15% | |
| 2016 | +9.53% | |
| 2017 | +13.33% | |
| **2018** | **-23.85%** | **Si** |
| 2019 | +19.02% | |
| 2020 | +119.72% | |
| 2021 | +14.17% | |
| 2022 | +24.02% | |
| **2023** | **-31.76%** | **Si** |
| 2024 | +128.72% | |
| 2025 | +130.94% | |
| 2026 | +6.31% | |

- **Anos negativos**: solo 2 de 21 (9.5%)
- **Racha maxima de perdidas**: 1 ano (nunca consecutivos)
- El peor ano (-31.76% en 2023) fue seguido por +128.72% en 2024

---

## 7. Conclusion y recomendacion

### Estrategia recomendada: Rolling 0.7x con colchon

1. **Prestamo inicial**: EUR 70,000 sobre EUR 10,000 de equity (7 anos, Euribor +3%)
2. **Colchon**: 2 anos de cuotas en cash (~EUR 25K)
3. **Al vencimiento (ano 7)**: renovar con prestamo = 70% del equity acumulado
4. **Repetir** cada 7 anos mientras la estrategia sea rentable

### Por que funciona

- El prestamo inicial (7x) maximiza el impacto cuando tienes menos que perder (EUR 10K)
- Las renovaciones al 0.7x mantienen apalancamiento sano que sobrevive DD -50%
- El colchon de 2 anos cubre las cuotas en periodos de drawdown
- Cada renovacion amplifica el efecto del interes compuesto sin riesgo excesivo

### Cifras clave

- **Capital final 21 anos**: EUR 7,072,460 (rolling 0.7x) vs EUR 1,643,292 (solo equity)
- **Multiplicador**: 4.3x sobre invertir solo los EUR 10,000 propios
- **CAGR**: 36.7% (vs 27.5% sin deuda)
- **Intereses totales pagados**: EUR 96,814 (1.4% del capital final)
- **Cero impagos** en todas las ventanas historicas del prestamo inicial
- **Renovaciones sobreviven DD -50%** con equity/deuda del 39%

### Riesgos asumidos

- El prestamo inicial (7x) no sobreviviria un DD -50% en terminos de equity neto, pero se asume porque solo hay EUR 10K en riesgo personal
- Impago solo posible con retornos permanentemente < 15%/a o 2+ anos negativos consecutivos (nunca ocurrido en 21 anos de backtest)

## Archivos relacionados

- `simulacion_apalancamiento.py` — simulacion completa con retornos reales v8
- `simulacion_stress_test.py` — stress test mensual con escenarios sinteticos
- `comparativa_3_estrategias.csv` — fuente de retornos anuales v8 corregidos
