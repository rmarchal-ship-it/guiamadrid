# Estrategia Multi-Asset: Gold / SP500 / TLT / TQQQ

## Concepto
Cartera pasiva de 4 activos descorrelados con rebalanceo anual.
El apalancamiento (TQQQ 3x) se controla limitando su peso y rebalanceando.

---

## Backtest Original — 41 anos (1985-2025)

### Pesos iniciales: 1/3 Gold + 1/3 SP500 + 1/6 TLT + 1/6 TQQQ

| Cartera | Valor Final | CAGR | Volatilidad | Max DD | Sharpe |
|---------|------------|------|-------------|--------|--------|
| **CARTERA (rebal. anual)** | **$2,121,091** | **13.96%** | **17.5%** | **-50.2%** | **0.57** |
| CARTERA (buy & hold) | $342,957 | 9.01% | 30.6% | -93.8% | 0.16 |
| Oro solo | $122,567 | 6.31% | 13.7% | -50.0% | 0.17 |
| S&P 500 solo | $403,877 | 9.44% | 18.2% | -56.8% | 0.30 |
| TLT solo | $194,041 | 7.50% | 15.8% | -48.6% | 0.22 |
| TQQQ solo | $810,816 | 11.32% | 74.1% | -100.0% | 0.10 |

### Matriz de correlaciones

|      | Oro   | SP500 | TLT   | TQQQ  |
|------|-------|-------|-------|-------|
| Oro  | 1.00  | -0.01 | 0.12  | -0.03 |
| SP500| -0.01 | 1.00  | 0.05  | 0.88  |
| TLT  | 0.12  | 0.05  | 1.00  | 0.02  |
| TQQQ | -0.03 | 0.88  | 0.02  | 1.00  |

---

## Anos con perdidas: 10 de 40 (25%)

| Ano  | Cartera | Oro    | SP500  | TLT    | TQQQ   | Contexto |
|------|---------|--------|--------|--------|--------|----------|
| 1987 | -8.0%   | +16.7% | +1.9%  | -21.9% | -63.1% | Black Monday |
| 1990 | -10.1%  | -6.9%  | -6.6%  | +4.5%  | -37.9% | Recesion + Guerra Golfo |
| 1994 | -8.4%   | -1.7%  | -1.6%  | -25.8% | -17.9% | Subida tipos Fed |
| 2000 | -15.5%  | -6.6%  | -10.2% | +31.5% | -91.1% | Burbuja .com |
| 2001 | -17.7%  | +2.1%  | -13.1% | +4.4%  | -88.6% | 9/11 |
| 2002 | -10.0%  | +24.2% | -23.4% | +24.8% | -86.3% | Bear market tech |
| 2008 | -20.1%  | +5.4%  | -38.5% | +33.8% | -88.1% | Crisis financiera |
| 2015 | -1.3%   | -10.8% | -0.8%  | -1.9%  | +17.2% | Desaceleracion China |
| 2018 | -6.5%   | -2.5%  | -6.3%  | -1.8%  | -19.8% | Subida tipos + trade war |
| 2022 | -25.2%  | -0.8%  | -19.5% | -31.3% | -79.1% | Inflacion + tipos agresivos |

- **Peor racha**: 3 anos consecutivos (2000-2002), cartera -37% acumulado
- **Mejor racha**: 6 anos consecutivos ganando (2009-2014)
- **75% de los anos gana dinero**, retorno medio +15.2%
- **TQQQ es culpable en 8 de 10 anos negativos**

---

## 10 Mejores vs 10 Peores Anos

### 10 PEORES
| Ano  | Cartera | Culpable | Contribucion |
|------|---------|----------|-------------|
| 2022 | -25.2%  | TQQQ -79% | -13.2% |
| 2008 | -20.1%  | TQQQ -88% | -14.7% |
| 2001 | -17.7%  | TQQQ -89% | -14.8% |
| 2000 | -15.5%  | TQQQ -91% | -15.2% |
| 1990 | -10.1%  | TQQQ -38% | -6.3% |
| 2002 | -10.0%  | TQQQ -86% | -14.4% |
| 1994 | -8.4%   | TLT -26%  | -4.3% |
| 1987 | -8.0%   | TQQQ -63% | -10.5% |
| 2018 | -6.5%   | TQQQ -20% | -3.3% |
| 2015 | -1.3%   | Oro -11%  | -3.6% |

**Media peores: -12.3%**

### 10 MEJORES
| Ano  | Cartera | Motor | Contribucion |
|------|---------|-------|-------------|
| 1999 | +60.8%  | TQQQ +348% | +58.0% |
| 1995 | +48.5%  | TQQQ +158% | +26.4% |
| 2023 | +45.8%  | TQQQ +198% | +33.0% |
| 2009 | +45.1%  | TQQQ +199% | +33.1% |
| 2003 | +42.2%  | TQQQ +161% | +26.8% |
| 2019 | +40.3%  | TQQQ +134% | +22.3% |
| 2020 | +34.8%  | TQQQ +110% | +18.3% |
| 2025 | +34.7%  | Oro +66%   | +21.9% |
| 2017 | +32.0%  | TQQQ +118% | +19.7% |
| 1989 | +30.4%  | TQQQ +94%  | +15.7% |

**Media mejores: +41.5%. Ratio asimetrico: mejores ganan 3.4x mas que lo que pierden los peores.**

---

## Optimizacion de Pesos (Grid Search ~1,200 combinaciones)

| Cartera | Oro | SP500 | TLT | TQQQ | CAGR | Vol | Sharpe | Max DD | Valor Final |
|---------|-----|-------|-----|------|------|-----|--------|--------|-------------|
| ACTUAL | 33% | 33% | 17% | 17% | 13.96% | 17.5% | 0.57 | -50.2% | $2.12M |
| **MAX SHARPE** | **45%** | **0%** | **35%** | **20%** | **14.66%** | **16.5%** | **0.65** | **-41.6%** | **$2.72M** |
| EQUILIBRADA | 30% | 20% | 35% | 15% | 13.58% | 15.3% | 0.63 | -39.6% | $1.85M |
| MAX CAGR | 25% | 5% | 35% | 35% | 18.40% | 25.2% | 0.57 | -67.7% | $10.2M |

### Por que eliminar SP500 mejora la cartera (MAX SHARPE)

1. SP500 y TQQQ correlacion 0.88 — son casi el mismo activo. Diversificacion redundante.
2. TQQQ ya captura bolsa americana (con esteroides). SP500 al lado solo diluye sin reducir riesgo.
3. Subir TLT al 35% amortigua las caidas de TQQQ (correlacion ~0) y genera "rebalancing alpha" comprando TQQQ barato en crisis.
4. Subir Oro al 45% aporta el ancla descorrelada que estabiliza todo.

### Cartera optima recomendada: 45% Gold / 35% TLT / 20% TQQQ

- Elimina SP500 (redundante con TQQQ)
- CAGR +14.66%, Sharpe 0.65, Max DD -41.6%
- Solo 3 activos, rebalanceo anual
- Exposicion efectiva Nasdaq: 20% x 3 = 60% (moderada)

---

## Evolucian por quinquenios ($10K iniciales, cartera original 33/33/17/17)

| Ano  | Cartera | Oro    | SP500  | TLT    | TQQQ   |
|------|---------|--------|--------|--------|--------|
| 1990 | $19K    | $12K   | $20K   | $26K   | $14K   |
| 1995 | $38K    | $13K   | $37K   | $53K   | $66K   |
| 2000 | $91K    | $8K    | $79K   | $71K   | $156K  |
| 2005 | $111K   | $16K   | $74K   | $111K  | $7K    |
| 2010 | $218K   | $42K   | $75K   | $140K  | $6K    |
| 2015 | $380K   | $31K   | $121K  | $206K  | $34K   |
| 2020 | $957K   | $54K   | $221K  | $300K  | $326K  |
| 2025 | $2.12M  | $123K  | $404K  | $194K  | $811K  |

---

## Research de soporte

- `gold_50y_simulation.py` — Oro 1x/2x/3x a 50 anos. Apalancamiento destruye valor en oro.
- `gold_vs_inflation.py` — Oro vs CPI-U. Protege a largo plazo (+2.76% real/ano) pero no fiable a corto.

---

## Archivos

| Archivo | Descripcion |
|---------|-------------|
| `portfolio_40y_simulation.py` | Backtest principal: 4 activos, 41 anos, rebalanceo anual |
| `gold_50y_simulation.py` | Simulacion oro 1x/2x/3x, 50 anos |
| `gold_vs_inflation.py` | Oro vs inflacion EEUU, 50 anos |
| `portfolio_40y_simulation.png` | Grafico equity curve cartera |
| `portfolio_40y_composition.png` | Grafico composicion por activo |
| `portfolio_optimization.png` | Heatmap optimizacion de pesos |
| `portfolio_sensitivity.png` | Analisis de sensibilidad |
| `gold_50y_simulation.png` | Grafico oro 1x/2x/3x |
| `gold_vs_inflation.png` | Grafico oro vs CPI |
| `RESULTADOS_BACKTEST.md` | Este documento — resultados completos |
