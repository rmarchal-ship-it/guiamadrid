# Momentum Breakout — Evolucion de Versiones y Razones de Cada Cambio

Documento generado 5 Mar 2026. Fuente: RESUMEN_PARA_CONTINUAR.md, diagnostico_240m.py, momentum_breakout.py, spy_correlation_analysis.py.

---

## Resumen ejecutivo

| Version | CAGR 240m | PF | MaxDD | Cambio clave |
|---------|-----------|-----|-------|-------------|
| v5 | ~8% | ~1.5 | — | Sin SL, time exit 45d fijo, 5 pos |
| v6 | ~12%* | 1.83* | -12.1%* | 7 posiciones (grid test) |
| v7/v7+ | +17.6% | 1.77 | -59.9% | Time exit 8d trailing (sin forzar) |
| v8 stock-only | ~8% | ~1.5 | — | Universo 225tk + 10 pos (sin opciones) |
| v8 con opciones | +34.6% | 2.89 | -42.6% | + CALL US (2 slots) |
| v12 (US2+EU2) | +36.3% | 3.40 | -42.6% | + opciones EU (slots separados) |
| v12 (480m) | +44.5% | 3.32 | -59.3% | Test 40 anos |
| **v12g (120m)** | **+51.1%** | **3.62** | **-29.8%** | **v12 + Gold 30% overlay. REFERENCIA** |
| v12g (480m) | +42.4% | — | -54.3% | v12 + Gold 30% (40 anos) |

*v6 solo testeado a 60m (+12.2% ann). No hay dato fiable a 240m.

**v12g a 120 meses es la REFERENCIA UNICA** (datos 100% reales, sin extrapolacion).

---

## v1-v3: Los origenes (descartadas)

**Configuracion**: Stop loss 2xATR, 5 posiciones, timeframe 4H, partial exits.

**Problema**: Win rate bajisimo. Los stops a 2xATR cortaban trades que luego se recuperaban. Las salidas parciales limitaban los fat tails que son la esencia de la estrategia.

**Leccion**: En una estrategia de momentum con win rate ~32%, el stop loss tradicional destruye el edge. Cada trade cortado prematuramente es un potencial home run perdido.

---

## v5: Sin stop loss (primera version viable)

**Cambio**: Eliminar el stop loss fijo. Time exit forzado a 45 barras. 5 posiciones max.

**Resultado**: PF ~1.5 a 18m. Primera version rentable.

**Leccion**: Dejar correr los trades es fundamental. El riesgo se gestiona por diversificacion (multiples posiciones) y time exits, no por stops.

---

## v6: Optimizacion de posiciones (7 pos)

**Cambio**: Grid test de 5/7/8/10/15 posiciones con 111 tickers. Time exit forzado a 12 barras.

**Resultado a 36m**: +81.4% (+22.0% ann), PF 2.28.
**Resultado a 60m**: +77.6% (+12.2% ann), PF 1.83, MaxDD -12.1%.

**Por que 7**: Con 111 tickers, la senal #8 era frecuentemente del mismo sector que otra abierta. 7 era el punto optimo de diversificacion/concentracion.

**Leccion**: El numero de posiciones depende del tamano del universo. Mas tickers = mas posiciones optimas.

---

## v6+: Opciones CALL (primer experimento)

**Cambio**: Anadir opciones CALL 5% ITM, 120 DTE, cierre a 45 DTE sobre tickers elegibles.

**Resultado a 36m**: +255.8% vs +86.3% base (v6 sin opciones).

**Por que funciona**: Las opciones amplifican los home runs (apalancamiento implicito ~3-5x) mientras el riesgo esta limitado a la prima pagada. En una estrategia fat-tail con win rate 32%, esta asimetria es ideal.

**Parametros opciones** (mantenidos hasta v12):
- Strike: 5% ITM (delta alto + valor intrinseco)
- DTE objetivo: ~120 dias
- Cierre: a 45 DTE restantes (antes de aceleracion theta)
- IVR < 40% (solo comprar volatilidad barata)
- Sin stop loss (riesgo = prima)

---

## v7/v7+: Eliminacion del time exit forzado

**Cambio critico**: A los 8 dias, en vez de cerrar la posicion (v6), se activa trailing 3xATR.

### Antes (v6) vs Despues (v7)

| | v6 | v7 |
|---|---|---|
| Trigger | 12 barras | 8 barras |
| Accion | Si perdiendo: **cierre forzado** | **Activar trailing** 3xATR |
| Win rate time exits | 0% (316 trades a 240m) | N/A (eliminados) |
| PnL time exits | -248,158 EUR en 20 anos | 0 EUR |

### Por que se cambio

A 240 meses, los cierres forzados tenian **0% win rate** y destruian -248K EUR. Muchos trades que se cerraban en perdida a dia 12 se habrian recuperado si se les hubiera dejado correr con trailing.

### Trade-off importante (documentado)

- **240m**: eliminar forzado = mejor (+17.6% vs ~12% ann)
- **36m**: forzar cierre = mejor (+22% vs ~9% ann)

A horizontes cortos, forzar la salida libera capital para nuevas entradas (rotacion). A horizontes largos, dejar correr gana porque muchos trades se recuperan.

**Decision**: Priorizar el horizonte largo (240m) ya que es mas representativo estadisticamente.

### Logica exacta del time exit v7

```python
if self.bars_held >= 8:  # CONFIG['max_hold_bars']
    if not self.trailing_active:
        chandelier = self.highest_since - (current_atr * 3.0)
        breakeven = self.entry_price * (1 + slippage/100)
        self.trailing_active = True
        if close <= self.entry_price:
            self.trailing_stop = max(chandelier, entry * 0.95)  # perdiendo: 5% bajo entry
        else:
            self.trailing_stop = max(chandelier, breakeven)      # ganando: breakeven
```

---

## v7+ -> v8: La gran expansion (112 -> 225 tickers + 10 posiciones)

### El problema diagnosticado

Con `diagnostico_240m.py` se descubrio que 5 anos destruian la curva:

| Ano | PF | Observacion |
|-----|-----|-------------|
| 2007 | 0.68 | Crisis subprime, todos los sectores US caen juntos |
| 2008 | 0.03 | Catastrofe. DD -27.8% |
| 2011 | 0.94 | Crisis deuda europea |
| 2018 | 0.22 | Peor ano. Fed sube tipos |
| 2022 | 0.36 | Tech crash, 0% win rate en Q2 |

**Causa raiz**: 112 tickers concentrados en US Tech/Finance/Health/Consumer. Las 7 posiciones venian del mismo sector/geografia, compounding el drawdown.

### La solucion: expansion en amplitud (NO data snooping)

**Principio**: "Filtrar sectores/tickers basandose en resultados de backtest = data snooping. En su lugar, expansion del universo en amplitud."

La lista de 225 tickers se definio ANTES de correr el backtest, basandose en:
1. Cobertura sistematica de todos los sectores GICS
2. Diversificacion geografica (US + Europa + Asia-Pacifico)
3. Clases de activos multiples (equity, bonds, commodities, ETFs)

### Categorias anadidas en v8

| Categoria nueva | N | Logica economica |
|----------------|---|-----------------|
| US Industrial | 10 | Ciclo economico diferente a tech (CAT, GE, LMT...) |
| US Energy | 10 | Contraciclico — petroleo sube cuando tech cae (XOM, CVX...) |
| US Utilities | 10 | Defensivo, baja correlacion con SPY (NEE, SO, DUK...) |
| US Real Estate | 10 | Ciclo inmobiliario propio (AMT, PLD, PSA...) |
| US Telecom | 7 | Dividendos estables (T, VZ, NFLX, DIS...) |
| UK | 9 | Commodities + finanzas, ciclo distinto a EU (SHEL, HSBC, RIO...) |
| Suiza | 6 | Defensivo: pharma, food, seguros (NESN.SW, ROG.SW...) |
| Nordicos | 5 | Telco, industrial, energia verde (NOVO-B.CO, VOLV-B.ST...) |
| Japon | 10 | Descorrelacionado con S&P, yen carry trade (7203.T, 6861.T...) |
| Australia | 8 | Commodities + banca local (BHP.AX, CBA.AX...) |
| China | 5 | Solo ADRs liquidos, ciclo propio (BABA, JD, PDD...) |
| ETFs sectoriales | 6 | Capturar tendencias sectoriales broad (SMH, XBI, XLU...) |
| ETFs internacionales | 10 | EM, India, Brasil, Taiwan (EEM, INDA, EWZ...) |
| Renta fija | 1->8 | Proteccion en crisis (TLT, AGG, HYG, EMB...) |

### Criterios de seleccion por ticker

1. **Liquidez**: solo large-caps y ETFs con volumen alto diario
2. **Datos**: disponibilidad en yfinance >= 18 anos (para backtest 240m)
3. **Cobertura sectorial**: cubrir TODOS los sectores GICS principales
4. **Diversificacion geografica**: US + Europa + Asia-Pacifico
5. **Sin survivorship bias**: tickers se anaden al universo desde su IPO date
6. **No optimizacion**: la lista se definio ANTES del backtest

### Grid test de posiciones con 225 tickers (240m)

| Pos | CAGR | PF | MaxDD |
|-----|------|-----|-------|
| 7 | +21.8% | 1.78 | -36.1% |
| 8 | +27.4% | 1.97 | -43.4% |
| **10** | **+34.6%** | **2.89** | **-42.6%** |
| 12 | +31.5% | 2.62 | -44.3% |

**10 posiciones gana en PF y return.** Con 225 tickers, las senales #8-#10 son de calidad comparable a las top 7 pero de sectores distintos. Con 112 tickers, la senal #8 era del mismo sector.

### Resultado

| Metrica | v7+ (112tk/7pos) | v8 (225tk/10pos) | Mejora |
|---------|-----------------|------------------|--------|
| CAGR | +17.6% | **+34.6%** | +17pp |
| PF | 1.77 | **2.89** | +63% |
| MaxDD | -59.9% | **-42.6%** | +17pp |

### Validacion de correlacion (spy_correlation_analysis.py)

81 de los 225 tickers tienen correlacion Pearson <= 0.15 con SPY:
- Renta fija (SHY, TLT, AGG)
- Utilities (DUK, SO, WEC)
- Japon, Australia
- Commodities (GLD, UNG)

Estos son los que protegen en bear markets.

---

## v8.1 / v8.1b: Exencion macro por correlacion (DESCARTADAS)

### Hipotesis

Tickers con baja correlacion con SPY (<=0.15) no deberian ser bloqueados por el macro filter, ya que funcionan bien en mercados bajistas.

### Resultados

| Horizonte | v8 PF | v8.1 MaxDD | v8 MaxDD |
|-----------|-------|------------|----------|
| 12m | +39.8% | -12.8% | -10.2% |
| 36m | **+346%** | -37.6% | **-21.4%** |

v8.1 gana a 12m (+63.4% vs +39.8%) pero **MaxDD a 36m inaceptable** (-37.6% vs -21.4%).

**Decision**: DESCARTADA. El macro filter debe aplicarse a todos los tickers.

---

## v9: Options-First (DESCARTADA)

**Concepto**: Priorizar opciones sobre stocks en todas las senales.

**Resultado**: DD inaceptable > 60%. Las opciones amplifican tanto las ganancias como las perdidas. Cuando todo son opciones, un mes malo destruye la cartera.

**Leccion**: Las opciones deben ser un COMPLEMENTO (2 de 10 posiciones), no el core.

---

## v10: Gold 30% overlay (ADOPTADA como capa)

**Concepto**: Mantener permanentemente 30% de la equity en oro fisico (GLD / Invesco Physical Gold).

**Resultado**: Reduccion dramatica de drawdown con impacto minimo en CAGR.

**Por que funciona**: Oro esta descorrelacionado con momentum equity y tiende a subir en crisis (flight to safety). El 30% actua como amortiguador.

**Implementacion**: No es parte del motor de senales. Es una capa pasiva sobre la cartera de momentum.

---

## v11: VIX filter (DESCARTADA)

**Concepto**: Anadir filtro VIX > umbral como condicion adicional para no operar.

**Resultado**: Redundante con SPY > SMA50. Cuando VIX sube, SPY ya esta bajo SMA50. El filtro adicional no anade valor y aumenta complejidad.

---

## v12: Opciones EU con slots separados (DEFINITIVA)

### Cambio

Anadir 2 slots de opciones europeas (Eurex, Euronext, LSE, SIX) SEPARADOS de los 2 slots US.

**Motor**: `backtest_v12_eu_options.py` (4 slots: 2US + 2EU).

### Por que slots separados es critico

Si EU compite con US por los mismos 2 slots, EU DESTRUYE valor (efecto desplazamiento). Las opciones EU tienen spread ~10% (vs ~3% US) y menor liquidez. Cuando desplazan una opcion US, el resultado neto es peor.

Con slots separados, las opciones EU capturan home runs ADICIONALES sin canibalizar US.

### Resultados

| Config | CAGR 240m | PF | MaxDD |
|--------|-----------|-----|-------|
| v8 (US2 only) | +30.4% | 2.21 | -35.9% |
| **v12 (US2+EU2)** | **+36.3%** | **3.40** | -42.6% |

**Home runs EU confirmados**: ISP.MI +175%, KBC.BR +151%, BATS.L +167%, ADS.DE +185%.

### Test 40 anos (480m)

| Config | CAGR | MaxDD | PF |
|--------|------|-------|-----|
| v8 REF | +35.2% | -50.2% | 2.14 |
| **v12** | **+44.5%** | -59.3% | **3.32** |
| v8 + Gold 30% | +33.7% | -45.9% | — |
| **v12 + Gold 30%** | **+42.4%** | -54.3% | — |

### Validacion Monte Carlo (10,000 sims, 60m)

| Test | Resultado |
|------|-----------|
| Trade Shuffle | ROBUSTO — CAGR constante |
| Bootstrap 5y | MUY ROBUSTO — 0% probabilidad de perdida |
| Permutation PnL | p=0.036 — edge REAL |
| Permutation PF | p=0.199 — no significativo (limitacion del test para fat-tail) |

### 39 tickers EU option-eligible

Definidos en `backtest_v12_eu_options.py`. Mercados: Eurex, Euronext, LSE, SIX, OMX.
Confirmados disponibles en DEGIRO: NESN.SW, AI.PA.

---

## v13: Rolling Thunder (DESCARTADA)

**Concepto**: Rolar opciones ganadoras a 45 DTE — en vez de cerrar, abrir nueva opcion mismo strike, 120 DTE.

**Resultado**: Bloquea slots (una opcion rolada ocupa el slot indefinidamente), peor CAGR y DD a 240m.

**Leccion**: Es mejor cerrar la opcion a 45 DTE, liberar el slot, y dejar que el motor asigne la siguiente mejor senal.

---

## v12g: v12 + Gold 30% (REFERENCIA FINAL)

**v12g a 120 meses** es la version de referencia unica:

| Metrica | Valor |
|---------|-------|
| CAGR | **+51.1%** |
| MaxDD | **-29.8%** |
| Eficiencia (CAGR/MaxDD) | **1.71** |
| PF | **3.62** |

**Por que 120m es la referencia**: Es la ventana con datos 100% reales para Gold (GC=F desde ~2000). A 480m, Gold pre-2000 se extrapola con 0% return (conservador), lo que subestima el beneficio.

---

## Decisiones clave y sus razones

| Decision | Razon | Evidencia |
|----------|-------|-----------|
| Solo LONGS | Shorts: P&L -281% | Backtest exhaustivo |
| Sin stop loss fijo | Stop destruye edge (win rate 55%->31%) | Grid test SL vs no SL |
| Sin partial exits | Cortaba fat tails (SLV +7.5% -> +61.6%) | Analisis trade-by-trade |
| 10 posiciones | Grid test 7/8/10/12 a 240m: 10 mejor PF | Grid test con 225 tickers |
| 225 tickers | Diversificacion en amplitud, no data snooping | Diagnostico temporal |
| Time exit 8d trailing | Forzar cierre: 0% win rate, -248K EUR | Comparacion v6 vs v7 |
| Macro SPY > SMA50 | PnL +15.5% -> +60.6% con filtro | Grid SMA20/25/35/50/100/200 |
| SMA50 (no otra) | Ganador claro vs SMA20/25/35/100/200 | Grid test a 60m |
| Opciones 5% ITM | Delta alto + valor intrinseco = menor theta | Comparativa ITM vs OTM |
| IVR < 40 | Filtrar opciones caras | Backtest con/sin filtro |
| Cierre opciones 45 DTE | Evitar aceleracion theta | Curva theta teorica |
| Gold 30% overlay | Descorrelacion + proteccion crisis | Backtest 120m/480m |
| Slots EU separados | EU desplaza US si compiten | Test combinado vs separado |

---

## Diagrama conceptual de CAGR

```
v5    ████████ ~8%
v6    ████████████ ~12%
v7    █████████████████ +17.6%
v8    ██████████████████████████████████ +34.6%
v12   ████████████████████████████████████ +36.3%
v12g  █████████████████████████████████████████████████ +51.1% (120m, REFERENCIA)
```

---

## Archivos de referencia

| Archivo | Contenido |
|---------|-----------|
| `RESUMEN_PARA_CONTINUAR.md` | Documento unico de referencia (todo el proyecto) |
| `backtest_experimental.py` | Backtest v8 principal |
| `backtest_v12_eu_options.py` | Backtest v12 (motor correcto, 4 slots) |
| `momentum_breakout.py` | Motor de senales + ASSETS (225 tickers) |
| `diagnostico_240m.py` | Diagnostico temporal que revelo el problema |
| `spy_correlation_analysis.py` | Correlacion 225 tickers con SPY |
| `backtest_v12_montecarlo.py` | Validacion Monte Carlo (10K sims) |
| `backtest_v12_40y.py` | Test definitivo 40 anos |
| `historico/backtest_v6_recovered.py` | v6 reconstruido y validado |
| `historico/test_v9_options_first.py` | v9 Options-First (descartado) |
| `historico/test_v10_gold_hedge.py` | v10 Gold overlay (adoptado) |
| `historico/test_v11_vix_filter.py` | v11 VIX filter (descartado) |
| `backtest_v13_rolling.py` | v13 Rolling Thunder (descartado) |
