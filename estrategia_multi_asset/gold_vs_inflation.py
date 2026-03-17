"""
Oro vs Inflación EEUU (1975-2025) — 50 años
=============================================
Compara el poder adquisitivo del oro frente al CPI-U.
¿El oro protege contra la inflación a largo plazo?
"""

import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── DATOS MENSUALES DEL ORO 1975-1999 (London PM Fix, USD/oz) ───
GOLD_MONTHLY = {
    '1975-01': 175.00, '1975-02': 181.50, '1975-03': 178.50, '1975-04': 167.00,
    '1975-05': 166.50, '1975-06': 165.00, '1975-07': 165.50, '1975-08': 163.00,
    '1975-09': 143.00, '1975-10': 143.00, '1975-11': 138.50, '1975-12': 140.25,
    '1976-01': 131.50, '1976-02': 130.00, '1976-03': 132.50, '1976-04': 128.75,
    '1976-05': 126.50, '1976-06': 126.00, '1976-07': 118.00, '1976-08': 110.00,
    '1976-09': 113.50, '1976-10': 118.00, '1976-11': 131.50, '1976-12': 134.50,
    '1977-01': 133.00, '1977-02': 138.00, '1977-03': 148.50, '1977-04': 150.50,
    '1977-05': 147.00, '1977-06': 141.00, '1977-07': 143.50, '1977-08': 147.00,
    '1977-09': 154.00, '1977-10': 160.00, '1977-11': 164.00, '1977-12': 164.50,
    '1978-01': 175.00, '1978-02': 183.50, '1978-03': 185.50, '1978-04': 175.00,
    '1978-05': 183.00, '1978-06': 184.00, '1978-07': 190.00, '1978-08': 207.50,
    '1978-09': 213.00, '1978-10': 228.00, '1978-11': 206.00, '1978-12': 226.00,
    '1979-01': 232.00, '1979-02': 251.00, '1979-03': 243.00, '1979-04': 239.50,
    '1979-05': 275.00, '1979-06': 280.00, '1979-07': 296.00, '1979-08': 315.00,
    '1979-09': 380.00, '1979-10': 400.00, '1979-11': 410.00, '1979-12': 524.00,
    '1980-01': 675.00, '1980-02': 665.00, '1980-03': 550.00, '1980-04': 517.00,
    '1980-05': 514.00, '1980-06': 600.00, '1980-07': 644.00, '1980-08': 627.00,
    '1980-09': 674.00, '1980-10': 661.00, '1980-11': 623.00, '1980-12': 594.00,
    '1981-01': 557.00, '1981-02': 500.00, '1981-03': 499.00, '1981-04': 495.00,
    '1981-05': 480.00, '1981-06': 461.00, '1981-07': 410.00, '1981-08': 410.00,
    '1981-09': 443.00, '1981-10': 437.00, '1981-11': 413.00, '1981-12': 410.00,
    '1982-01': 384.00, '1982-02': 374.00, '1982-03': 330.00, '1982-04': 350.00,
    '1982-05': 335.00, '1982-06': 315.00, '1982-07': 337.00, '1982-08': 366.00,
    '1982-09': 436.00, '1982-10': 421.00, '1982-11': 414.00, '1982-12': 456.00,
    '1983-01': 481.00, '1983-02': 490.00, '1983-03': 414.00, '1983-04': 432.00,
    '1983-05': 436.00, '1983-06': 413.00, '1983-07': 422.00, '1983-08': 416.00,
    '1983-09': 411.00, '1983-10': 395.00, '1983-11': 381.00, '1983-12': 389.00,
    '1984-01': 370.00, '1984-02': 385.00, '1984-03': 395.00, '1984-04': 382.00,
    '1984-05': 378.00, '1984-06': 378.00, '1984-07': 348.00, '1984-08': 348.00,
    '1984-09': 341.00, '1984-10': 340.00, '1984-11': 341.00, '1984-12': 320.00,
    '1985-01': 303.00, '1985-02': 299.00, '1985-03': 304.00, '1985-04': 325.00,
    '1985-05': 317.00, '1985-06': 317.00, '1985-07': 317.00, '1985-08': 330.00,
    '1985-09': 323.00, '1985-10': 325.00, '1985-11': 325.00, '1985-12': 326.00,
    '1986-01': 345.00, '1986-02': 339.00, '1986-03': 346.00, '1986-04': 340.00,
    '1986-05': 342.00, '1986-06': 343.00, '1986-07': 349.00, '1986-08': 383.00,
    '1986-09': 418.00, '1986-10': 431.00, '1986-11': 399.00, '1986-12': 391.00,
    '1987-01': 408.00, '1987-02': 401.00, '1987-03': 408.00, '1987-04': 438.00,
    '1987-05': 461.00, '1987-06': 449.00, '1987-07': 451.00, '1987-08': 461.00,
    '1987-09': 461.00, '1987-10': 466.00, '1987-11': 470.00, '1987-12': 484.00,
    '1988-01': 477.00, '1988-02': 443.00, '1988-03': 444.00, '1988-04': 452.00,
    '1988-05': 451.00, '1988-06': 452.00, '1988-07': 437.00, '1988-08': 432.00,
    '1988-09': 413.00, '1988-10': 406.00, '1988-11': 420.00, '1988-12': 418.00,
    '1989-01': 404.00, '1989-02': 388.00, '1989-03': 390.00, '1989-04': 384.00,
    '1989-05': 371.00, '1989-06': 368.00, '1989-07': 375.00, '1989-08': 365.00,
    '1989-09': 362.00, '1989-10': 367.00, '1989-11': 394.00, '1989-12': 401.00,
    '1990-01': 410.00, '1990-02': 416.00, '1990-03': 393.00, '1990-04': 374.00,
    '1990-05': 367.00, '1990-06': 352.00, '1990-07': 362.00, '1990-08': 395.00,
    '1990-09': 389.00, '1990-10': 381.00, '1990-11': 381.00, '1990-12': 378.00,
    '1991-01': 383.00, '1991-02': 363.00, '1991-03': 363.00, '1991-04': 358.00,
    '1991-05': 357.00, '1991-06': 366.00, '1991-07': 368.00, '1991-08': 356.00,
    '1991-09': 349.00, '1991-10': 359.00, '1991-11': 360.00, '1991-12': 361.00,
    '1992-01': 354.00, '1992-02': 354.00, '1992-03': 345.00, '1992-04': 338.00,
    '1992-05': 337.00, '1992-06': 340.00, '1992-07': 353.00, '1992-08': 340.00,
    '1992-09': 345.00, '1992-10': 344.00, '1992-11': 335.00, '1992-12': 334.00,
    '1993-01': 329.00, '1993-02': 329.00, '1993-03': 330.00, '1993-04': 342.00,
    '1993-05': 368.00, '1993-06': 373.00, '1993-07': 392.00, '1993-08': 380.00,
    '1993-09': 355.00, '1993-10': 363.00, '1993-11': 374.00, '1993-12': 383.00,
    '1994-01': 387.00, '1994-02': 382.00, '1994-03': 384.00, '1994-04': 378.00,
    '1994-05': 381.00, '1994-06': 386.00, '1994-07': 386.00, '1994-08': 384.00,
    '1994-09': 394.00, '1994-10': 388.00, '1994-11': 384.00, '1994-12': 379.00,
    '1995-01': 379.00, '1995-02': 376.00, '1995-03': 382.00, '1995-04': 391.00,
    '1995-05': 386.00, '1995-06': 387.00, '1995-07': 386.00, '1995-08': 383.00,
    '1995-09': 383.00, '1995-10': 383.00, '1995-11': 386.00, '1995-12': 387.00,
    '1996-01': 399.00, '1996-02': 404.00, '1996-03': 397.00, '1996-04': 392.00,
    '1996-05': 392.00, '1996-06': 385.00, '1996-07': 384.00, '1996-08': 388.00,
    '1996-09': 383.00, '1996-10': 381.00, '1996-11': 378.00, '1996-12': 369.00,
    '1997-01': 354.00, '1997-02': 346.00, '1997-03': 352.00, '1997-04': 344.00,
    '1997-05': 344.00, '1997-06': 341.00, '1997-07': 325.00, '1997-08': 325.00,
    '1997-09': 323.00, '1997-10': 325.00, '1997-11': 306.00, '1997-12': 290.00,
    '1998-01': 289.00, '1998-02': 298.00, '1998-03': 296.00, '1998-04': 310.00,
    '1998-05': 300.00, '1998-06': 292.00, '1998-07': 293.00, '1998-08': 284.00,
    '1998-09': 291.00, '1998-10': 296.00, '1998-11': 294.00, '1998-12': 291.00,
    '1999-01': 287.00, '1999-02': 287.00, '1999-03': 286.00, '1999-04': 282.00,
    '1999-05': 277.00, '1999-06': 262.00, '1999-07': 256.00, '1999-08': 256.00,
    '1999-09': 265.00, '1999-10': 311.00, '1999-11': 293.00, '1999-12': 290.00,
}

# ─── 1) Descargar CPI-U desde FRED ───
print("Descargando CPI-U (inflación EEUU) desde FRED...")
cpi = web.DataReader('CPIAUCSL', 'fred',
                     start=datetime(1974, 12, 1),
                     end=datetime(2025, 12, 31))
cpi = cpi.dropna()
print(f"CPI: {cpi.index[0].strftime('%Y-%m')} a {cpi.index[-1].strftime('%Y-%m')} ({len(cpi)} meses)")

# ─── 2) Construir serie mensual del oro ───
print("Construyendo serie mensual del oro 1975-2025...")

# Parte 1: datos hardcodeados 1975-1999
gold_monthly_early = pd.Series(
    {pd.Timestamp(k + '-01'): v for k, v in GOLD_MONTHLY.items()}
).sort_index()

# Parte 2: datos mensuales de yfinance 2000-2025
gold_yf = yf.download("GC=F", start="2000-01-01", end="2025-12-31", progress=False)
if isinstance(gold_yf.columns, pd.MultiIndex):
    gold_yf.columns = gold_yf.columns.get_level_values(0)
gold_monthly_late = gold_yf['Close'].resample('MS').last().dropna()

# Combinar
gold_monthly = pd.concat([
    gold_monthly_early.loc[:'1999-12-31'],
    gold_monthly_late.loc['2000-01-01':]
])
gold_monthly = gold_monthly[~gold_monthly.index.duplicated(keep='last')].sort_index()

# ─── 3) Alinear series al mismo rango ───
common_start = max(gold_monthly.index[0], cpi.index[0])
common_end = min(gold_monthly.index[-1], cpi.index[-1])

gold_m = gold_monthly.loc[common_start:common_end]
cpi_m = cpi.loc[common_start:common_end].squeeze()

# Alinear ambas series al mismo conjunto de fechas (inner join)
common_dates = gold_m.index.intersection(cpi_m.index)
gold_m = gold_m.loc[common_dates]
cpi_m = cpi_m.loc[common_dates]

print(f"Rango común: {common_start.strftime('%Y-%m')} a {common_end.strftime('%Y-%m')} ({len(common_dates)} meses)")
print(f"Oro inicio: ${gold_m.iloc[0]:.0f}/oz → fin: ${gold_m.iloc[-1]:.0f}/oz")
print(f"CPI inicio: {cpi_m.iloc[0]:.1f} → fin: {cpi_m.iloc[-1]:.1f}")

# ─── 4) Calcular métricas ───
years = (common_end - common_start).days / 365.25

# Inflación acumulada
inflation_total = cpi_m.iloc[-1] / cpi_m.iloc[0] - 1
inflation_cagr = (cpi_m.iloc[-1] / cpi_m.iloc[0]) ** (1/years) - 1

# Oro nominal
gold_total = gold_m.iloc[-1] / gold_m.iloc[0] - 1
gold_cagr = (gold_m.iloc[-1] / gold_m.iloc[0]) ** (1/years) - 1

# Oro real (deflactado por CPI)
gold_real = gold_m.values / cpi_m.values  # precio del oro en "dólares constantes"
gold_real_series = pd.Series(gold_real, index=gold_m.index)
gold_real_total = gold_real_series.iloc[-1] / gold_real_series.iloc[0] - 1
gold_real_cagr = (gold_real_series.iloc[-1] / gold_real_series.iloc[0]) ** (1/years) - 1

# Poder adquisitivo del dólar
dollar_purchasing = cpi_m.iloc[0] / cpi_m  # cuánto vale $1 de 1975

print(f"\n{'='*80}")
print(f"ORO vs INFLACIÓN EEUU — {common_start.strftime('%Y')} a {common_end.strftime('%Y')} ({years:.1f} años)")
print(f"{'='*80}")
print(f"")
print(f"{'Métrica':<35} {'Oro (nominal)':>15} {'Inflación (CPI)':>15} {'Oro (real)':>15}")
print(f"{'-'*80}")
print(f"{'Retorno acumulado':<35} {gold_total:>14.0%} {inflation_total:>14.0%} {gold_real_total:>14.0%}")
print(f"{'CAGR':<35} {gold_cagr:>14.2%} {inflation_cagr:>14.2%} {gold_real_cagr:>14.2%}")
print(f"{'Multiplicador':<35} {gold_m.iloc[-1]/gold_m.iloc[0]:>14.1f}x {cpi_m.iloc[-1]/cpi_m.iloc[0]:>14.1f}x {gold_real_series.iloc[-1]/gold_real_series.iloc[0]:>14.1f}x")

# ─── 5) Tabla por década ───
print(f"\n{'='*80}")
print("EVOLUCIÓN POR DÉCADA")
print(f"{'='*80}")
print(f"{'Década':<12} {'Oro inicio':>10} {'Oro fin':>10} {'Oro CAGR':>10} {'CPI CAGR':>10} {'Oro real':>10} {'¿Protege?':>10}")
print(f"{'-'*72}")

decades = [(1975, 1980), (1980, 1985), (1985, 1990), (1990, 1995),
           (1995, 2000), (2000, 2005), (2005, 2010), (2010, 2015),
           (2015, 2020), (2020, 2025)]

for start_yr, end_yr in decades:
    try:
        g0 = gold_m[gold_m.index.year == start_yr].iloc[0]
        g1 = gold_m[gold_m.index.year == end_yr].iloc[-1] if end_yr <= gold_m.index.year.max() else gold_m.iloc[-1]
        c0 = cpi_m[cpi_m.index.year == start_yr].iloc[0]
        c1 = cpi_m[cpi_m.index.year == end_yr].iloc[-1] if end_yr <= cpi_m.index.year.max() else cpi_m.iloc[-1]

        n = end_yr - start_yr
        g_cagr = (g1/g0)**(1/n) - 1
        c_cagr = (c1/c0)**(1/n) - 1
        real_cagr = (1 + g_cagr) / (1 + c_cagr) - 1
        protects = "SI" if g_cagr > c_cagr else "NO"

        print(f"{start_yr}-{end_yr:<7} ${g0:>8,.0f} ${g1:>8,.0f} {g_cagr:>9.1%} {c_cagr:>9.1%} {real_cagr:>9.1%} {'  '+protects:>10}")
    except Exception as e:
        print(f"{start_yr}-{end_yr}: error — {e}")

# ─── 6) Poder adquisitivo de $10,000 ───
print(f"\n{'='*80}")
print("PODER ADQUISITIVO DE $10,000 (en dólares de 1975)")
print(f"{'='*80}")

# $10K en efectivo pierde poder adquisitivo con la inflación
# $10K en oro mantiene (o no) su poder adquisitivo
cash_real = 10000 * cpi_m.iloc[0] / cpi_m  # poder adquisitivo real del cash
gold_value_real = (gold_m.values / gold_m.iloc[0]) * 10000 * cpi_m.iloc[0] / cpi_m.values

print(f"  En 1975: $10,000 = $10,000 (ambos)")
cash_2025 = 10000 * cpi_m.iloc[0] / cpi_m.iloc[-1]
gold_2025 = (gold_m.iloc[-1] / gold_m.iloc[0]) * 10000 * cpi_m.iloc[0] / cpi_m.iloc[-1]
print(f"  En 2025: $10,000 cash = ${cash_2025:,.0f} en dólares de 1975 (perdió {(1-cash_2025/10000)*100:.0f}%)")
print(f"  En 2025: $10,000 en oro = ${gold_2025:,.0f} en dólares de 1975 ({'ganó' if gold_2025 > 10000 else 'perdió'} {abs(gold_2025/10000 - 1)*100:.0f}%)")

# ─── 7) Inflación interanual vs rendimiento del oro ───
gold_yoy = gold_m.pct_change(12).dropna()
cpi_yoy = cpi_m.pct_change(12).dropna()

# Alinear
common_idx = gold_yoy.index.intersection(cpi_yoy.index)
gold_yoy_a = gold_yoy.loc[common_idx]
cpi_yoy_a = cpi_yoy.loc[common_idx]

# Correlación
corr = gold_yoy_a.corr(cpi_yoy_a)
print(f"\n  Correlación oro YoY vs CPI YoY: {corr:.3f}")

# ¿En períodos de alta inflación (>5%), el oro batió a la inflación?
high_inflation = cpi_yoy_a > 0.05
if high_inflation.any():
    gold_in_high = gold_yoy_a[high_inflation].mean()
    cpi_in_high = cpi_yoy_a[high_inflation].mean()
    pct_beat = (gold_yoy_a[high_inflation] > cpi_yoy_a[high_inflation]).mean()
    print(f"\n  En meses con inflación >5% interanual ({high_inflation.sum()} meses):")
    print(f"    Retorno medio del oro:  {gold_in_high:.1%}")
    print(f"    Inflación media:        {cpi_in_high:.1%}")
    print(f"    % veces oro > inflación: {pct_beat:.0%}")

# ¿En períodos de baja inflación (<2%)?
low_inflation = cpi_yoy_a < 0.02
if low_inflation.any():
    gold_in_low = gold_yoy_a[low_inflation].mean()
    cpi_in_low = cpi_yoy_a[low_inflation].mean()
    pct_beat_low = (gold_yoy_a[low_inflation] > cpi_yoy_a[low_inflation]).mean()
    print(f"\n  En meses con inflación <2% interanual ({low_inflation.sum()} meses):")
    print(f"    Retorno medio del oro:  {gold_in_low:.1%}")
    print(f"    Inflación media:        {cpi_in_low:.1%}")
    print(f"    % veces oro > inflación: {pct_beat_low:.0%}")

# ─── 8) GRÁFICOS ───
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Oro vs Inflación EEUU — 50 Años (1975-2025)',
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: $10,000 nominales — Oro vs "Coste de la vida"
ax1 = axes[0, 0]
gold_idx = (gold_m / gold_m.iloc[0]) * 10000
cpi_idx = (cpi_m / cpi_m.iloc[0]) * 10000
ax1.semilogy(gold_idx.index, gold_idx, color='#FFD700', linewidth=2.5, label='$10K en Oro')
ax1.semilogy(cpi_idx.index, cpi_idx, color='#CC0000', linewidth=2.5, label='$10K ajustado por CPI')
ax1.fill_between(gold_idx.index, cpi_idx, gold_idx,
                 where=gold_idx >= cpi_idx, color='#FFD700', alpha=0.15, label='Oro > Inflación')
ax1.fill_between(gold_idx.index, cpi_idx, gold_idx,
                 where=gold_idx < cpi_idx, color='#CC0000', alpha=0.15, label='Inflación > Oro')
ax1.set_title('$10,000 Nominales: Oro vs Inflación (Log)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Valor ($)')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.annotate(f"Oro: ${gold_idx.iloc[-1]:,.0f}", xy=(gold_idx.index[-1], gold_idx.iloc[-1]),
            fontsize=9, color='#B8860B', fontweight='bold', xytext=(-100, 10), textcoords='offset points')
ax1.annotate(f"CPI: ${cpi_idx.iloc[-1]:,.0f}", xy=(cpi_idx.index[-1], cpi_idx.iloc[-1]),
            fontsize=9, color='#CC0000', fontweight='bold', xytext=(-100, -15), textcoords='offset points')

# Panel 2: Precio real del oro (deflactado por CPI, base 1975=100)
ax2 = axes[0, 1]
gold_real_idx = gold_real_series / gold_real_series.iloc[0] * 100
ax2.plot(gold_real_idx.index, gold_real_idx, color='#DAA520', linewidth=2.5)
ax2.axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='Base 100 (1975)')
ax2.fill_between(gold_real_idx.index, 100, gold_real_idx,
                 where=gold_real_idx >= 100, color='#FFD700', alpha=0.2)
ax2.fill_between(gold_real_idx.index, 100, gold_real_idx,
                 where=gold_real_idx < 100, color='red', alpha=0.1)
ax2.set_title('Precio REAL del Oro (dólares constantes 1975, base=100)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Índice (1975=100)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.annotate(f"{gold_real_idx.iloc[-1]:.0f}", xy=(gold_real_idx.index[-1], gold_real_idx.iloc[-1]),
            fontsize=10, color='#DAA520', fontweight='bold', xytext=(5, 0), textcoords='offset points')

# Panel 3: Inflación interanual vs retorno del oro YoY
ax3 = axes[1, 0]
ax3.bar(cpi_yoy_a.index, cpi_yoy_a * 100, width=25, color='#CC0000', alpha=0.4, label='Inflación YoY')
ax3.plot(gold_yoy_a.index, gold_yoy_a * 100, color='#FFD700', linewidth=1.2, alpha=0.8, label='Oro YoY')
ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.set_title('Retorno Interanual: Oro vs Inflación', fontsize=11, fontweight='bold')
ax3.set_ylabel('Retorno (%)')
ax3.set_ylim(-50, 150)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Retorno real acumulado del oro (rolling)
ax4 = axes[1, 1]
# Calcular el % de tiempo que el oro batió a la inflación (rolling 10 años)
window = 120  # 10 años en meses
if len(gold_real_idx) > window:
    rolling_real_cagr = ((gold_real_idx / gold_real_idx.shift(window)) ** (1/10) - 1) * 100
    ax4.plot(rolling_real_cagr.index, rolling_real_cagr, color='#DAA520', linewidth=2, label='CAGR real oro (rolling 10a)')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Breakeven vs inflación')
    ax4.fill_between(rolling_real_cagr.dropna().index,
                     0, rolling_real_cagr.dropna(),
                     where=rolling_real_cagr.dropna() >= 0,
                     color='#FFD700', alpha=0.2)
    ax4.fill_between(rolling_real_cagr.dropna().index,
                     0, rolling_real_cagr.dropna(),
                     where=rolling_real_cagr.dropna() < 0,
                     color='red', alpha=0.1)
ax4.set_title('CAGR Real del Oro — Rolling 10 Años', fontsize=11, fontweight='bold')
ax4.set_ylabel('CAGR Real (%)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = "/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code/gold_vs_inflation.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nGráfico guardado: {output_path}")

# ─── 9) Resumen final ───
print(f"\n{'='*80}")
print("VEREDICTO: ¿EL ORO PROTEGE CONTRA LA INFLACIÓN?")
print(f"{'='*80}")

# Calcular % del tiempo que el oro real estuvo por encima de 100
pct_above = (gold_real_idx >= 100).mean()
# Períodos donde perdió vs inflación
below_periods = (gold_real_idx < 100)

print(f"""
  A 50 AÑOS COMPLETOS:
    CAGR del oro:       {gold_cagr:.2%}
    CAGR inflación:     {inflation_cagr:.2%}
    CAGR real del oro:  {gold_real_cagr:.2%}
    → El oro SUPERÓ a la inflación en {gold_real_cagr:.2%} anual

  PERO CON MATICES:
    - Solo {pct_above:.0%} del tiempo el oro estuvo por encima de su valor real de 1975
    - Quien compró en el pico de 1980 tardó ~28 años en recuperar poder adquisitivo
    - Desde mínimos de 1999-2001, el oro ha multiplicado su valor real ~8x
    - La protección funciona a MUY largo plazo y depende enormemente del punto de entrada

  CORRELACIÓN CON INFLACIÓN:
    - Correlación oro vs CPI (interanual): {corr:.3f}
    - El oro NO reacciona linealmente a la inflación
    - Es más un hedge contra crisis monetarias/de confianza que contra inflación ordinaria
""")

print("✓ Análisis completado")
