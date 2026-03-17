"""
Simulación Cartera 40 Años (1985-2025)
======================================
Composición:
  - 1/3 Oro (GLD equivalente)
  - 1/3 S&P 500 (SPY equivalente)
  - 1/6 Bonos largo plazo (TLT equivalente)
  - 1/6 TQQQ (Nasdaq-100 3x apalancado diario)

Datos sintéticos pre-ETF:
  - Oro: precios mensuales interpolados 1985-1999 + GC=F 2000+
  - S&P 500: ^GSPC desde 1985
  - TLT: reconstruido con rendimiento bonos 20y (^TLT no existía hasta 2002)
  - TQQQ: simulado como 3x diario del Nasdaq-100 (^NDX/QQQ)

Rebalanceo anual a pesos objetivo.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code"

# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 1: DATOS HISTÓRICOS
# ═══════════════════════════════════════════════════════════════════

# ─── Oro mensual 1985-1999 ───
GOLD_MONTHLY = {
    '1985-01': 303, '1985-02': 299, '1985-03': 304, '1985-04': 325,
    '1985-05': 317, '1985-06': 317, '1985-07': 317, '1985-08': 330,
    '1985-09': 323, '1985-10': 325, '1985-11': 325, '1985-12': 326,
    '1986-01': 345, '1986-02': 339, '1986-03': 346, '1986-04': 340,
    '1986-05': 342, '1986-06': 343, '1986-07': 349, '1986-08': 383,
    '1986-09': 418, '1986-10': 431, '1986-11': 399, '1986-12': 391,
    '1987-01': 408, '1987-02': 401, '1987-03': 408, '1987-04': 438,
    '1987-05': 461, '1987-06': 449, '1987-07': 451, '1987-08': 461,
    '1987-09': 461, '1987-10': 466, '1987-11': 470, '1987-12': 484,
    '1988-01': 477, '1988-02': 443, '1988-03': 444, '1988-04': 452,
    '1988-05': 451, '1988-06': 452, '1988-07': 437, '1988-08': 432,
    '1988-09': 413, '1988-10': 406, '1988-11': 420, '1988-12': 418,
    '1989-01': 404, '1989-02': 388, '1989-03': 390, '1989-04': 384,
    '1989-05': 371, '1989-06': 368, '1989-07': 375, '1989-08': 365,
    '1989-09': 362, '1989-10': 367, '1989-11': 394, '1989-12': 401,
    '1990-01': 410, '1990-02': 416, '1990-03': 393, '1990-04': 374,
    '1990-05': 367, '1990-06': 352, '1990-07': 362, '1990-08': 395,
    '1990-09': 389, '1990-10': 381, '1990-11': 381, '1990-12': 378,
    '1991-01': 383, '1991-02': 363, '1991-03': 363, '1991-04': 358,
    '1991-05': 357, '1991-06': 366, '1991-07': 368, '1991-08': 356,
    '1991-09': 349, '1991-10': 359, '1991-11': 360, '1991-12': 361,
    '1992-01': 354, '1992-02': 354, '1992-03': 345, '1992-04': 338,
    '1992-05': 337, '1992-06': 340, '1992-07': 353, '1992-08': 340,
    '1992-09': 345, '1992-10': 344, '1992-11': 335, '1992-12': 334,
    '1993-01': 329, '1993-02': 329, '1993-03': 330, '1993-04': 342,
    '1993-05': 368, '1993-06': 373, '1993-07': 392, '1993-08': 380,
    '1993-09': 355, '1993-10': 363, '1993-11': 374, '1993-12': 383,
    '1994-01': 387, '1994-02': 382, '1994-03': 384, '1994-04': 378,
    '1994-05': 381, '1994-06': 386, '1994-07': 386, '1994-08': 384,
    '1994-09': 394, '1994-10': 388, '1994-11': 384, '1994-12': 379,
    '1995-01': 379, '1995-02': 376, '1995-03': 382, '1995-04': 391,
    '1995-05': 386, '1995-06': 387, '1995-07': 386, '1995-08': 383,
    '1995-09': 383, '1995-10': 383, '1995-11': 386, '1995-12': 387,
    '1996-01': 399, '1996-02': 404, '1996-03': 397, '1996-04': 392,
    '1996-05': 392, '1996-06': 385, '1996-07': 384, '1996-08': 388,
    '1996-09': 383, '1996-10': 381, '1996-11': 378, '1996-12': 369,
    '1997-01': 354, '1997-02': 346, '1997-03': 352, '1997-04': 344,
    '1997-05': 344, '1997-06': 341, '1997-07': 325, '1997-08': 325,
    '1997-09': 323, '1997-10': 325, '1997-11': 306, '1997-12': 290,
    '1998-01': 289, '1998-02': 298, '1998-03': 296, '1998-04': 310,
    '1998-05': 300, '1998-06': 292, '1998-07': 293, '1998-08': 284,
    '1998-09': 291, '1998-10': 296, '1998-11': 294, '1998-12': 291,
    '1999-01': 287, '1999-02': 287, '1999-03': 286, '1999-04': 282,
    '1999-05': 277, '1999-06': 262, '1999-07': 256, '1999-08': 256,
    '1999-09': 265, '1999-10': 311, '1999-11': 293, '1999-12': 290,
}

# Costes por producto
ER_GOLD = 0.0040    # GLD expense ratio
ER_SPY  = 0.0009    # SPY expense ratio
ER_TLT  = 0.0015    # TLT expense ratio
ER_TQQQ = 0.0088    # TQQQ expense ratio

# Borrowing cost para TQQQ (3x leverage → financia 2x)
BORROWING_COST_BY_YEAR = {
    range(1985, 1990): 0.07,
    range(1990, 1995): 0.04,
    range(1995, 2000): 0.055,
    range(2000, 2005): 0.03,
    range(2005, 2008): 0.045,
    range(2008, 2016): 0.005,
    range(2016, 2020): 0.015,
    range(2020, 2022): 0.001,
    range(2022, 2026): 0.05,
}

def get_borrow(year):
    for r, c in BORROWING_COST_BY_YEAR.items():
        if year in r:
            return c
    return 0.03

# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 2: DESCARGAR Y CONSTRUIR SERIES
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("DESCARGANDO DATOS...")
print("=" * 80)

# ─── S&P 500 ───
print("  S&P 500 (^GSPC)...")
sp500 = yf.download("^GSPC", start="1985-01-01", end="2025-12-31", progress=False)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
sp500_close = sp500['Close'].dropna()
print(f"    {sp500_close.index[0].strftime('%Y-%m-%d')} → {sp500_close.index[-1].strftime('%Y-%m-%d')}")

# ─── Oro: mensual interpolado + GC=F ───
print("  Oro (mensual 1985-99 + GC=F 2000+)...")
gold_early = pd.Series(
    {pd.Timestamp(k + '-01'): float(v) for k, v in GOLD_MONTHLY.items()}
).sort_index()
gold_early = gold_early.resample('B').interpolate(method='linear')

gold_yf = yf.download("GC=F", start="2000-01-01", end="2025-12-31", progress=False)
if isinstance(gold_yf.columns, pd.MultiIndex):
    gold_yf.columns = gold_yf.columns.get_level_values(0)
gold_late = gold_yf['Close'].dropna()

gold_close = pd.concat([
    gold_early.loc[:'1999-12-31'],
    gold_late.loc['2000-01-01':]
])
gold_close = gold_close[~gold_close.index.duplicated(keep='last')].sort_index()
print(f"    {gold_close.index[0].strftime('%Y-%m-%d')} → {gold_close.index[-1].strftime('%Y-%m-%d')}")

# ─── TLT: real desde 2002 + sintético antes ───
# Antes de TLT, simulamos con la duración de bonos 20y y el yield de ^TNX
print("  TLT (real 2002+ / sintético 1985-2002 vía ^TNX)...")
tlt_yf = yf.download("TLT", start="2002-07-01", end="2025-12-31", progress=False)
if isinstance(tlt_yf.columns, pd.MultiIndex):
    tlt_yf.columns = tlt_yf.columns.get_level_values(0)
tlt_real = tlt_yf['Close'].dropna()

# Sintético pre-2002: usar ^TNX (10y yield) como proxy
# Retorno diario de un bono largo ≈ yield/252 - duration × Δyield
# Duración efectiva TLT ≈ 17 años
tnx = yf.download("^TNX", start="1985-01-01", end="2002-08-01", progress=False)
if isinstance(tnx.columns, pd.MultiIndex):
    tnx.columns = tnx.columns.get_level_values(0)
tnx_close = tnx['Close'].dropna()

# Yield en decimal
yield_series = tnx_close / 100
yield_change = yield_series.diff()

# Retorno diario del bono: cupón diario + efecto precio (duration)
DURATION = 17  # duración efectiva TLT
bond_daily_ret = yield_series.shift(1) / 252 - DURATION * yield_change
bond_daily_ret = bond_daily_ret.dropna().clip(-0.10, 0.10)

# Construir precio sintético de TLT
tlt_synth_price = 100 * (1 + bond_daily_ret).cumprod()
# Reescalar para que empate con TLT real en julio 2002
tlt_real_start = tlt_real.iloc[0]
tlt_synth_end = tlt_synth_price.iloc[-1]
scale = tlt_real_start / tlt_synth_end
tlt_synth_price = tlt_synth_price * scale

tlt_close = pd.concat([
    tlt_synth_price.loc[:'2002-07-29'],
    tlt_real.loc['2002-07-30':]
])
tlt_close = tlt_close[~tlt_close.index.duplicated(keep='last')].sort_index()
print(f"    {tlt_close.index[0].strftime('%Y-%m-%d')} → {tlt_close.index[-1].strftime('%Y-%m-%d')}")

# ─── TQQQ: real desde 2010 + sintético 3x diario QQQ/NDX antes ───
print("  TQQQ (real 2010+ / sintético 3x diario 1985-2010)...")
tqqq_yf = yf.download("TQQQ", start="2010-02-01", end="2025-12-31", progress=False)
if isinstance(tqqq_yf.columns, pd.MultiIndex):
    tqqq_yf.columns = tqqq_yf.columns.get_level_values(0)
tqqq_real = tqqq_yf['Close'].dropna()

# Pre-2010: usar QQQ (desde 1999) + ^GSPC como proxy del growth antes de 1999
qqq_yf = yf.download("QQQ", start="1999-03-01", end="2010-02-28", progress=False)
if isinstance(qqq_yf.columns, pd.MultiIndex):
    qqq_yf.columns = qqq_yf.columns.get_level_values(0)
qqq_close = qqq_yf['Close'].dropna()

# Pre-1999: usar ^GSPC como proxy (no ideal pero es lo mejor que hay)
# El Nasdaq-100 se creó en 1985 pero no tenemos datos diarios fiables
# Usamos S&P 500 con un ajuste de beta (Nasdaq históricamente ~1.2x beta vs S&P)
NASDAQ_BETA = 1.25  # beta aproximado del Nasdaq vs S&P pre-1999
sp500_pre99 = sp500_close.loc[:'1999-03-09']
sp500_pre99_ret = sp500_pre99.pct_change().dropna()
nasdaq_proxy_ret = sp500_pre99_ret * NASDAQ_BETA

# QQQ retornos 1999-2010
qqq_ret = qqq_close.pct_change().dropna()

# Combinar retornos del subyacente
underlying_ret = pd.concat([nasdaq_proxy_ret, qqq_ret])
underlying_ret = underlying_ret[~underlying_ret.index.duplicated(keep='last')].sort_index()

# Simular TQQQ = 3x diario - expense - borrowing cost
trading_days = 252
tqqq_synth_ret = underlying_ret.copy()
for idx in tqqq_synth_ret.index:
    yr = idx.year
    daily_borrow = get_borrow(yr) / trading_days
    tqqq_synth_ret.loc[idx] = 3.0 * underlying_ret.loc[idx] - ER_TQQQ / trading_days - daily_borrow * 2

tqqq_synth_price = 10 * (1 + tqqq_synth_ret).cumprod()
# Reescalar para empatar con TQQQ real
tqqq_real_start = tqqq_real.iloc[0]
tqqq_synth_end = tqqq_synth_price.iloc[-1]
scale_tqqq = tqqq_real_start / tqqq_synth_end
tqqq_synth_price = tqqq_synth_price * scale_tqqq

tqqq_close = pd.concat([
    tqqq_synth_price.loc[:'2010-02-10'],
    tqqq_real.loc['2010-02-11':]
])
tqqq_close = tqqq_close[~tqqq_close.index.duplicated(keep='last')].sort_index()
print(f"    {tqqq_close.index[0].strftime('%Y-%m-%d')} → {tqqq_close.index[-1].strftime('%Y-%m-%d')}")

# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 3: ALINEAR TODAS LAS SERIES AL MISMO RANGO
# ═══════════════════════════════════════════════════════════════════

start_date = max(gold_close.index[0], sp500_close.index[0],
                 tlt_close.index[0], tqqq_close.index[0])
end_date = min(gold_close.index[-1], sp500_close.index[-1],
               tlt_close.index[-1], tqqq_close.index[-1])

print(f"\nRango común: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")

# Construir DataFrame común
all_dates = sp500_close.loc[start_date:end_date].index
df = pd.DataFrame(index=all_dates)
df['Oro'] = gold_close.reindex(all_dates).ffill()
df['SP500'] = sp500_close.reindex(all_dates).ffill()
df['TLT'] = tlt_close.reindex(all_dates).ffill()
df['TQQQ'] = tqqq_close.reindex(all_dates).ffill()
df = df.dropna()

print(f"Datos alineados: {len(df)} días de trading")
print(f"  Oro:   ${df['Oro'].iloc[0]:,.0f} → ${df['Oro'].iloc[-1]:,.0f}")
print(f"  SP500: ${df['SP500'].iloc[0]:,.0f} → ${df['SP500'].iloc[-1]:,.0f}")
print(f"  TLT:   ${df['TLT'].iloc[0]:,.2f} → ${df['TLT'].iloc[-1]:,.2f}")
print(f"  TQQQ:  ${df['TQQQ'].iloc[0]:,.4f} → ${df['TQQQ'].iloc[-1]:,.2f}")

# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 4: RETORNOS DIARIOS (con expense ratios)
# ═══════════════════════════════════════════════════════════════════

ret = df.pct_change().dropna()

# Aplicar expense ratios (ya están incluidos en TQQQ y TLT reales post-lanzamiento,
# pero los aplicamos uniformemente para consistencia con la parte sintética)
ret_net = ret.copy()
ret_net['Oro'] = ret['Oro'] - ER_GOLD / trading_days
ret_net['SP500'] = ret['SP500'] - ER_SPY / trading_days
ret_net['TLT'] = ret['TLT'] - ER_TLT / trading_days
# TQQQ ya tiene costes incluidos en la simulación sintética; el real ya los tiene
ret_net['TQQQ'] = ret['TQQQ']

# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 5: SIMULACIÓN DE CARTERAS
# ═══════════════════════════════════════════════════════════════════

INITIAL = 10000

# Pesos objetivo
W_GOLD = 1/3
W_SP500 = 1/3
W_TLT = 1/6
W_TQQQ = 1/6

weights = {'Oro': W_GOLD, 'SP500': W_SP500, 'TLT': W_TLT, 'TQQQ': W_TQQQ}

# ─── A) Cartera con rebalanceo anual ───
print("\nSimulando cartera con rebalanceo anual...")

# Inicializar posiciones
portfolio_value = [INITIAL]
alloc = {k: INITIAL * w for k, w in weights.items()}
last_rebal_year = ret_net.index[0].year

for i, date in enumerate(ret_net.index):
    # Rebalancear si cambió el año
    if date.year != last_rebal_year:
        total = sum(alloc.values())
        alloc = {k: total * w for k, w in weights.items()}
        last_rebal_year = date.year

    # Aplicar retornos del día
    for k in alloc:
        alloc[k] *= (1 + ret_net[k].iloc[i])

    portfolio_value.append(sum(alloc.values()))

portfolio_eq = pd.Series(portfolio_value[1:], index=ret_net.index)

# ─── B) Cartera sin rebalanceo (buy & hold) ───
print("Simulando cartera buy & hold (sin rebalanceo)...")

alloc_bh = {k: INITIAL * w for k, w in weights.items()}
portfolio_bh_values = [INITIAL]

for i, date in enumerate(ret_net.index):
    for k in alloc_bh:
        alloc_bh[k] *= (1 + ret_net[k].iloc[i])
    portfolio_bh_values.append(sum(alloc_bh.values()))

portfolio_bh = pd.Series(portfolio_bh_values[1:], index=ret_net.index)

# ─── C) Equity curves individuales ───
eq_individual = {}
for k in ['Oro', 'SP500', 'TLT', 'TQQQ']:
    eq_individual[k] = INITIAL * (1 + ret_net[k]).cumprod()

# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 6: MÉTRICAS
# ═══════════════════════════════════════════════════════════════════

def calc_metrics(equity, name):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_ret = equity.iloc[-1] / INITIAL - 1
    cagr = (equity.iloc[-1] / INITIAL) ** (1 / years) - 1
    daily_ret = equity.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = dd.min()
    sharpe = (cagr - 0.04) / vol if vol > 0 else 0
    yearly = equity.resample('YE').last().pct_change().dropna()
    return {
        'Nombre': name,
        'Valor Final': f"${equity.iloc[-1]:,.0f}",
        'CAGR': f"{cagr:.2%}",
        'Volatilidad': f"{vol:.1%}",
        'Max DD': f"{max_dd:.1%}",
        'Sharpe': f"{sharpe:.2f}",
        'Mejor Año': f"{yearly.max():.1%}",
        'Peor Año': f"{yearly.min():.1%}",
    }

all_metrics = []
all_metrics.append(calc_metrics(portfolio_eq, "📊 CARTERA (rebal.)"))
all_metrics.append(calc_metrics(portfolio_bh, "📊 CARTERA (B&H)"))
all_metrics.append(calc_metrics(eq_individual['Oro'], "Oro (1/3)"))
all_metrics.append(calc_metrics(eq_individual['SP500'], "S&P 500 (1/3)"))
all_metrics.append(calc_metrics(eq_individual['TLT'], "TLT (1/6)"))
all_metrics.append(calc_metrics(eq_individual['TQQQ'], "TQQQ (1/6)"))

df_metrics = pd.DataFrame(all_metrics).set_index('Nombre')

years_total = (portfolio_eq.index[-1] - portfolio_eq.index[0]).days / 365.25
print(f"\n{'='*90}")
print(f"SIMULACIÓN CARTERA — {years_total:.0f} AÑOS ({portfolio_eq.index[0].strftime('%Y')}–{portfolio_eq.index[-1].strftime('%Y')})")
print(f"Pesos: Oro 33% | S&P 500 33% | TLT 17% | TQQQ 17%")
print(f"{'='*90}")
print(df_metrics.to_string())

# ─── Tabla por quinquenios ───
print(f"\n{'='*90}")
print("VALOR DE $10,000 POR QUINQUENIOS")
print(f"{'='*90}")

years_check = list(range(1985, 2030, 5))
quinq_data = []
for eq, name in [(portfolio_eq, "CARTERA (rebal.)"), (eq_individual['Oro'], "Oro"),
                  (eq_individual['SP500'], "S&P 500"),
                  (eq_individual['TLT'], "TLT"),
                  (eq_individual['TQQQ'], "TQQQ")]:
    row = {'': name}
    for yr in years_check:
        mask = eq.index.year == yr
        if mask.any():
            row[str(yr)] = f"${eq[mask].iloc[-1]:,.0f}"
        else:
            row[str(yr)] = "—"
    quinq_data.append(row)

df_quinq = pd.DataFrame(quinq_data).set_index('')
print(df_quinq.to_string())

# ─── Composición de la cartera en el tiempo ───
print(f"\n{'='*90}")
print("COMPOSICIÓN REAL DE LA CARTERA (buy & hold, sin rebalanceo)")
print(f"{'='*90}")

# Recalcular pesos reales en el tiempo
alloc_track = {k: INITIAL * w for k, w in weights.items()}
weight_history = []

for i, date in enumerate(ret_net.index):
    for k in alloc_track:
        alloc_track[k] *= (1 + ret_net[k].iloc[i])
    total = sum(alloc_track.values())
    if date.month == 12 and date.day >= 28:  # fin de año
        wts = {f"w_{k}": alloc_track[k]/total for k in alloc_track}
        wts['Fecha'] = date.strftime('%Y')
        weight_history.append(wts)

if weight_history:
    df_wts = pd.DataFrame(weight_history).set_index('Fecha')
    for col in df_wts.columns:
        df_wts[col] = df_wts[col].apply(lambda x: f"{x:.1%}")
    print(df_wts.to_string())

# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 7: GRÁFICOS
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Cartera Oro/SP500/TLT/TQQQ — {years_total:.0f} Años ({portfolio_eq.index[0].year}–{portfolio_eq.index[-1].year})\n'
             f'Pesos: Oro 33% | S&P 500 33% | TLT 17% | TQQQ 17% | Rebalanceo anual',
             fontsize=13, fontweight='bold', y=0.99)

colors = {'Cartera': '#1a1a2e', 'Oro': '#FFD700', 'SP500': '#2196F3',
          'TLT': '#4CAF50', 'TQQQ': '#FF5722'}

# Panel 1: Equity curves (log)
ax1 = axes[0, 0]
ax1.semilogy(portfolio_eq.index, portfolio_eq, color=colors['Cartera'],
             linewidth=3, label='CARTERA (rebal.)', zorder=5)
ax1.semilogy(portfolio_bh.index, portfolio_bh, color=colors['Cartera'],
             linewidth=1.5, linestyle='--', alpha=0.5, label='CARTERA (B&H)')
for k, c in [('Oro', colors['Oro']), ('SP500', colors['SP500']),
              ('TLT', colors['TLT']), ('TQQQ', colors['TQQQ'])]:
    ax1.semilogy(eq_individual[k].index, eq_individual[k], color=c,
                 linewidth=1.5, alpha=0.7, label=k)
ax1.set_title('Equity Curves — $10,000 (Log)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Valor ($)')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

# Anotar valores finales
for eq, name, c in [(portfolio_eq, 'Cartera', colors['Cartera']),
                     (eq_individual['SP500'], 'SP500', colors['SP500'])]:
    ax1.annotate(f"${eq.iloc[-1]:,.0f}", xy=(eq.index[-1], eq.iloc[-1]),
                fontsize=8, color=c, fontweight='bold',
                xytext=(5, 0), textcoords='offset points')

# Panel 2: Equity curves (linear)
ax2 = axes[0, 1]
ax2.plot(portfolio_eq.index, portfolio_eq/1000, color=colors['Cartera'],
         linewidth=3, label='CARTERA (rebal.)', zorder=5)
for k, c in [('Oro', colors['Oro']), ('SP500', colors['SP500']),
              ('TLT', colors['TLT']), ('TQQQ', colors['TQQQ'])]:
    ax2.plot(eq_individual[k].index, eq_individual[k]/1000, color=c,
             linewidth=1.5, alpha=0.7, label=k)
ax2.set_title('Equity Curves — $10,000 (Lineal)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Valor ($K)')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)

# Panel 3: Drawdowns
ax3 = axes[1, 0]
for eq, name, c in [(portfolio_eq, 'CARTERA', colors['Cartera']),
                      (eq_individual['SP500'], 'SP500', colors['SP500']),
                      (eq_individual['TQQQ'], 'TQQQ', colors['TQQQ'])]:
    rm = eq.cummax()
    dd = (eq - rm) / rm * 100
    ax3.plot(dd.index, dd, color=c, linewidth=1.5 if name != 'CARTERA' else 2.5,
             alpha=0.7, label=name)
ax3.fill_between(portfolio_eq.index,
                 (portfolio_eq - portfolio_eq.cummax()) / portfolio_eq.cummax() * 100,
                 0, color=colors['Cartera'], alpha=0.15)
ax3.set_title('Drawdowns', fontsize=11, fontweight='bold')
ax3.set_ylabel('Drawdown (%)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Retornos anuales de la cartera
ax4 = axes[1, 1]
yearly_ret = portfolio_eq.resample('YE').last().pct_change().dropna() * 100
bar_colors = ['#2196F3' if r >= 0 else '#F44336' for r in yearly_ret.values]
ax4.bar(yearly_ret.index.year, yearly_ret.values, color=bar_colors, alpha=0.8)
ax4.axhline(y=0, color='black', linewidth=0.5)
avg_ret = yearly_ret.mean()
ax4.axhline(y=avg_ret, color='#FF9800', linewidth=2, linestyle='--',
            label=f'Media: {avg_ret:.1f}%')
ax4.set_title('Retornos Anuales de la Cartera', fontsize=11, fontweight='bold')
ax4.set_ylabel('Retorno (%)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = f"{OUTPUT_DIR}/portfolio_40y_simulation.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nGráfico guardado: {output_path}")

# ─── Gráfico adicional: composición de la cartera ───
fig2, ax5 = plt.subplots(figsize=(16, 5))

# Recalcular composición diaria (rebalanceo anual)
alloc_daily = {k: INITIAL * w for k, w in weights.items()}
last_rb_yr = ret_net.index[0].year
w_history = {'Oro': [], 'SP500': [], 'TLT': [], 'TQQQ': []}
dates_h = []

for i, date in enumerate(ret_net.index):
    if date.year != last_rb_yr:
        total = sum(alloc_daily.values())
        alloc_daily = {k: total * w for k, w in weights.items()}
        last_rb_yr = date.year

    for k in alloc_daily:
        alloc_daily[k] *= (1 + ret_net[k].iloc[i])

    total = sum(alloc_daily.values())
    for k in alloc_daily:
        w_history[k].append(alloc_daily[k] / total)
    dates_h.append(date)

df_w = pd.DataFrame(w_history, index=dates_h)
ax5.stackplot(df_w.index,
              df_w['Oro'], df_w['SP500'], df_w['TLT'], df_w['TQQQ'],
              labels=['Oro', 'S&P 500', 'TLT', 'TQQQ'],
              colors=[colors['Oro'], colors['SP500'], colors['TLT'], colors['TQQQ']],
              alpha=0.8)
ax5.set_title('Composición de la Cartera en el Tiempo (con rebalanceo anual)',
              fontsize=12, fontweight='bold')
ax5.set_ylabel('Peso (%)')
ax5.set_ylim(0, 1)
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

output_path2 = f"{OUTPUT_DIR}/portfolio_40y_composition.png"
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Gráfico composición: {output_path2}")

# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 8: ANÁLISIS DE CORRELACIONES
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*90}")
print("MATRIZ DE CORRELACIONES (retornos mensuales)")
print(f"{'='*90}")

monthly_ret = df.resample('ME').last().pct_change().dropna()
corr_matrix = monthly_ret.corr()
print(corr_matrix.round(2).to_string())

print(f"\n{'='*90}")
print("BENEFICIO DE DIVERSIFICACIÓN")
print(f"{'='*90}")

port_cagr = (portfolio_eq.iloc[-1] / INITIAL) ** (1 / years_total) - 1
port_vol = portfolio_eq.pct_change().dropna().std() * np.sqrt(252)
avg_component_vol = np.mean([eq_individual[k].pct_change().dropna().std() * np.sqrt(252)
                             for k in eq_individual])
vol_reduction = 1 - port_vol / avg_component_vol

print(f"  Volatilidad media componentes: {avg_component_vol:.1%}")
print(f"  Volatilidad cartera:           {port_vol:.1%}")
print(f"  Reducción de volatilidad:      {vol_reduction:.0%}")

print("\n✓ Simulación completada")
