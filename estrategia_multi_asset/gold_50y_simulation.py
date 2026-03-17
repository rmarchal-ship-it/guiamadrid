"""
Simulación a 50 años: GLD (oro) vs Oro Apalancado 2x y 3x
==========================================================
Datos:
  - 1975-1999: Precios mensuales del oro (London PM Fix) → interpolados a diario
  - 2000-2025: Datos diarios reales de futuros del oro (GC=F via yfinance)

Simula:
  - GLD (1x): precio del oro - expense ratio 0.40%
  - Oro 2x:   apalancamiento diario 2x - expense ratio 0.95% - borrowing cost
  - Oro 3x:   apalancamiento diario 3x - expense ratio 1.35% - borrowing cost
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

# ─── DATOS MENSUALES DEL ORO 1975-1999 (London PM Fix, USD/oz) ───
# Fuente: World Gold Council / Kitco / FRED GOLDAMGBD228NLBM (promedios mensuales)
GOLD_MONTHLY = {
    # 1975
    '1975-01': 175.00, '1975-02': 181.50, '1975-03': 178.50, '1975-04': 167.00,
    '1975-05': 166.50, '1975-06': 165.00, '1975-07': 165.50, '1975-08': 163.00,
    '1975-09': 143.00, '1975-10': 143.00, '1975-11': 138.50, '1975-12': 140.25,
    # 1976
    '1976-01': 131.50, '1976-02': 130.00, '1976-03': 132.50, '1976-04': 128.75,
    '1976-05': 126.50, '1976-06': 126.00, '1976-07': 118.00, '1976-08': 110.00,
    '1976-09': 113.50, '1976-10': 118.00, '1976-11': 131.50, '1976-12': 134.50,
    # 1977
    '1977-01': 133.00, '1977-02': 138.00, '1977-03': 148.50, '1977-04': 150.50,
    '1977-05': 147.00, '1977-06': 141.00, '1977-07': 143.50, '1977-08': 147.00,
    '1977-09': 154.00, '1977-10': 160.00, '1977-11': 164.00, '1977-12': 164.50,
    # 1978
    '1978-01': 175.00, '1978-02': 183.50, '1978-03': 185.50, '1978-04': 175.00,
    '1978-05': 183.00, '1978-06': 184.00, '1978-07': 190.00, '1978-08': 207.50,
    '1978-09': 213.00, '1978-10': 228.00, '1978-11': 206.00, '1978-12': 226.00,
    # 1979
    '1979-01': 232.00, '1979-02': 251.00, '1979-03': 243.00, '1979-04': 239.50,
    '1979-05': 275.00, '1979-06': 280.00, '1979-07': 296.00, '1979-08': 315.00,
    '1979-09': 380.00, '1979-10': 400.00, '1979-11': 410.00, '1979-12': 524.00,
    # 1980
    '1980-01': 675.00, '1980-02': 665.00, '1980-03': 550.00, '1980-04': 517.00,
    '1980-05': 514.00, '1980-06': 600.00, '1980-07': 644.00, '1980-08': 627.00,
    '1980-09': 674.00, '1980-10': 661.00, '1980-11': 623.00, '1980-12': 594.00,
    # 1981
    '1981-01': 557.00, '1981-02': 500.00, '1981-03': 499.00, '1981-04': 495.00,
    '1981-05': 480.00, '1981-06': 461.00, '1981-07': 410.00, '1981-08': 410.00,
    '1981-09': 443.00, '1981-10': 437.00, '1981-11': 413.00, '1981-12': 410.00,
    # 1982
    '1982-01': 384.00, '1982-02': 374.00, '1982-03': 330.00, '1982-04': 350.00,
    '1982-05': 335.00, '1982-06': 315.00, '1982-07': 337.00, '1982-08': 366.00,
    '1982-09': 436.00, '1982-10': 421.00, '1982-11': 414.00, '1982-12': 456.00,
    # 1983
    '1983-01': 481.00, '1983-02': 490.00, '1983-03': 414.00, '1983-04': 432.00,
    '1983-05': 436.00, '1983-06': 413.00, '1983-07': 422.00, '1983-08': 416.00,
    '1983-09': 411.00, '1983-10': 395.00, '1983-11': 381.00, '1983-12': 389.00,
    # 1984
    '1984-01': 370.00, '1984-02': 385.00, '1984-03': 395.00, '1984-04': 382.00,
    '1984-05': 378.00, '1984-06': 378.00, '1984-07': 348.00, '1984-08': 348.00,
    '1984-09': 341.00, '1984-10': 340.00, '1984-11': 341.00, '1984-12': 320.00,
    # 1985
    '1985-01': 303.00, '1985-02': 299.00, '1985-03': 304.00, '1985-04': 325.00,
    '1985-05': 317.00, '1985-06': 317.00, '1985-07': 317.00, '1985-08': 330.00,
    '1985-09': 323.00, '1985-10': 325.00, '1985-11': 325.00, '1985-12': 326.00,
    # 1986
    '1986-01': 345.00, '1986-02': 339.00, '1986-03': 346.00, '1986-04': 340.00,
    '1986-05': 342.00, '1986-06': 343.00, '1986-07': 349.00, '1986-08': 383.00,
    '1986-09': 418.00, '1986-10': 431.00, '1986-11': 399.00, '1986-12': 391.00,
    # 1987
    '1987-01': 408.00, '1987-02': 401.00, '1987-03': 408.00, '1987-04': 438.00,
    '1987-05': 461.00, '1987-06': 449.00, '1987-07': 451.00, '1987-08': 461.00,
    '1987-09': 461.00, '1987-10': 466.00, '1987-11': 470.00, '1987-12': 484.00,
    # 1988
    '1988-01': 477.00, '1988-02': 443.00, '1988-03': 444.00, '1988-04': 452.00,
    '1988-05': 451.00, '1988-06': 452.00, '1988-07': 437.00, '1988-08': 432.00,
    '1988-09': 413.00, '1988-10': 406.00, '1988-11': 420.00, '1988-12': 418.00,
    # 1989
    '1989-01': 404.00, '1989-02': 388.00, '1989-03': 390.00, '1989-04': 384.00,
    '1989-05': 371.00, '1989-06': 368.00, '1989-07': 375.00, '1989-08': 365.00,
    '1989-09': 362.00, '1989-10': 367.00, '1989-11': 394.00, '1989-12': 401.00,
    # 1990
    '1990-01': 410.00, '1990-02': 416.00, '1990-03': 393.00, '1990-04': 374.00,
    '1990-05': 367.00, '1990-06': 352.00, '1990-07': 362.00, '1990-08': 395.00,
    '1990-09': 389.00, '1990-10': 381.00, '1990-11': 381.00, '1990-12': 378.00,
    # 1991
    '1991-01': 383.00, '1991-02': 363.00, '1991-03': 363.00, '1991-04': 358.00,
    '1991-05': 357.00, '1991-06': 366.00, '1991-07': 368.00, '1991-08': 356.00,
    '1991-09': 349.00, '1991-10': 359.00, '1991-11': 360.00, '1991-12': 361.00,
    # 1992
    '1992-01': 354.00, '1992-02': 354.00, '1992-03': 345.00, '1992-04': 338.00,
    '1992-05': 337.00, '1992-06': 340.00, '1992-07': 353.00, '1992-08': 340.00,
    '1992-09': 345.00, '1992-10': 344.00, '1992-11': 335.00, '1992-12': 334.00,
    # 1993
    '1993-01': 329.00, '1993-02': 329.00, '1993-03': 330.00, '1993-04': 342.00,
    '1993-05': 368.00, '1993-06': 373.00, '1993-07': 392.00, '1993-08': 380.00,
    '1993-09': 355.00, '1993-10': 363.00, '1993-11': 374.00, '1993-12': 383.00,
    # 1994
    '1994-01': 387.00, '1994-02': 382.00, '1994-03': 384.00, '1994-04': 378.00,
    '1994-05': 381.00, '1994-06': 386.00, '1994-07': 386.00, '1994-08': 384.00,
    '1994-09': 394.00, '1994-10': 388.00, '1994-11': 384.00, '1994-12': 379.00,
    # 1995
    '1995-01': 379.00, '1995-02': 376.00, '1995-03': 382.00, '1995-04': 391.00,
    '1995-05': 386.00, '1995-06': 387.00, '1995-07': 386.00, '1995-08': 383.00,
    '1995-09': 383.00, '1995-10': 383.00, '1995-11': 386.00, '1995-12': 387.00,
    # 1996
    '1996-01': 399.00, '1996-02': 404.00, '1996-03': 397.00, '1996-04': 392.00,
    '1996-05': 392.00, '1996-06': 385.00, '1996-07': 384.00, '1996-08': 388.00,
    '1996-09': 383.00, '1996-10': 381.00, '1996-11': 378.00, '1996-12': 369.00,
    # 1997
    '1997-01': 354.00, '1997-02': 346.00, '1997-03': 352.00, '1997-04': 344.00,
    '1997-05': 344.00, '1997-06': 341.00, '1997-07': 325.00, '1997-08': 325.00,
    '1997-09': 323.00, '1997-10': 325.00, '1997-11': 306.00, '1997-12': 290.00,
    # 1998
    '1998-01': 289.00, '1998-02': 298.00, '1998-03': 296.00, '1998-04': 310.00,
    '1998-05': 300.00, '1998-06': 292.00, '1998-07': 293.00, '1998-08': 284.00,
    '1998-09': 291.00, '1998-10': 296.00, '1998-11': 294.00, '1998-12': 291.00,
    # 1999
    '1999-01': 287.00, '1999-02': 287.00, '1999-03': 286.00, '1999-04': 282.00,
    '1999-05': 277.00, '1999-06': 262.00, '1999-07': 256.00, '1999-08': 256.00,
    '1999-09': 265.00, '1999-10': 311.00, '1999-11': 293.00, '1999-12': 290.00,
}

# ─── Parámetros ───
ER_1X = 0.0040   # GLD: 0.40%
ER_2X = 0.0095   # UGL: 0.95%
ER_3X = 0.0135   # Estimado para 3x

# Coste de financiación variable por década (aprox. Fed Funds Rate promedio)
BORROWING_COST_BY_YEAR = {
    range(1975, 1980): 0.07,   # Alta inflación, tipos altos
    range(1980, 1985): 0.12,   # Volcker, tipos muy altos
    range(1985, 1990): 0.07,   # Bajada gradual
    range(1990, 1995): 0.04,   # Tipos moderados
    range(1995, 2000): 0.055,  # Expansión
    range(2000, 2005): 0.03,   # Post dot-com, tipos bajos
    range(2005, 2008): 0.045,  # Pre-crisis
    range(2008, 2016): 0.005,  # ZIRP
    range(2016, 2020): 0.015,  # Normalización lenta
    range(2020, 2022): 0.001,  # COVID ZIRP
    range(2022, 2026): 0.05,   # Subida agresiva
}

def get_borrowing_cost(year):
    for yr_range, cost in BORROWING_COST_BY_YEAR.items():
        if year in yr_range:
            return cost
    return 0.03  # default

# ─── Construir serie histórica combinada ───
print("Construyendo serie histórica del oro 1975-2025...")

# 1) Datos mensuales 1975-1999 → interpolados a diario
monthly_dates = []
monthly_prices = []
for date_str, price in sorted(GOLD_MONTHLY.items()):
    monthly_dates.append(pd.Timestamp(date_str + '-01'))
    monthly_prices.append(price)

monthly_series = pd.Series(monthly_prices, index=monthly_dates)
# Resample a diario (días laborables) e interpolar
daily_early = monthly_series.resample('B').interpolate(method='linear')
daily_early = daily_early.loc[:'1999-12-31']

# 2) Datos diarios 2000-2025 de yfinance
print("Descargando datos diarios GC=F (2000-2025)...")
gold_yf = yf.download("GC=F", start="2000-01-01", end="2025-12-31", progress=False)
if isinstance(gold_yf.columns, pd.MultiIndex):
    gold_yf.columns = gold_yf.columns.get_level_values(0)
daily_late = gold_yf['Close'].dropna()

# 3) Combinar: ajustar nivel para que empate en la transición
# Usar el ratio entre el último precio interpolado y el primer precio real
transition_date = daily_late.index[0]
last_early = daily_early.iloc[-1]
first_late = daily_late.iloc[0]
# No ajustamos — los precios mensuales ya representan el precio spot real
# Solo concatenamos
combined = pd.concat([daily_early.loc[:transition_date - pd.Timedelta(days=1)], daily_late])
combined = combined[~combined.index.duplicated(keep='last')]
combined = combined.sort_index()
combined.name = 'Gold'

print(f"Serie combinada: {combined.index[0].strftime('%Y-%m-%d')} a {combined.index[-1].strftime('%Y-%m-%d')}")
print(f"Total puntos de datos: {len(combined)}")

# ─── Calcular retornos diarios ───
daily_ret = combined.pct_change().dropna()
daily_ret = daily_ret.clip(-0.15, 0.15)  # Eliminar outliers de datos

print(f"\nEstadísticas del oro (diario → anualizado):")
print(f"  Retorno medio: {daily_ret.mean()*252:.2%}")
print(f"  Volatilidad:   {daily_ret.std()*np.sqrt(252):.2%}")

# ─── Simular carteras con borrowing cost variable ───
INITIAL = 10000
trading_days = 252

# Crear arrays de costes diarios que varían por año
daily_er_1x = ER_1X / trading_days
daily_er_2x = ER_2X / trading_days
daily_er_3x = ER_3X / trading_days

# Borrowing cost diario variable por año
daily_borrow = pd.Series(
    [get_borrowing_cost(d.year) / trading_days for d in daily_ret.index],
    index=daily_ret.index
)

# Retornos diarios de cada estrategia
ret_gold = daily_ret.copy()  # Oro puro
ret_1x = daily_ret - daily_er_1x
ret_2x = 2 * daily_ret - daily_er_2x - daily_borrow * 1
ret_3x = 3 * daily_ret - daily_er_3x - daily_borrow * 2

# Equity curves
equity_gold = INITIAL * (1 + ret_gold).cumprod()
equity_1x = INITIAL * (1 + ret_1x).cumprod()
equity_2x = INITIAL * (1 + ret_2x).cumprod()
equity_3x = INITIAL * (1 + ret_3x).cumprod()

# ─── Métricas ───
def calc_metrics(equity, name, daily_returns):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_ret = equity.iloc[-1] / INITIAL - 1
    cagr = (equity.iloc[-1] / INITIAL) ** (1 / years) - 1
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol if vol > 0 else 0
    yearly = equity.resample('YE').last().pct_change().dropna()
    worst_year = yearly.min()
    best_year = yearly.max()
    return {
        'Nombre': name,
        'Valor Final': f"${equity.iloc[-1]:,.0f}",
        'Retorno Total': f"{total_ret:,.0%}",
        'CAGR': f"{cagr:.2%}",
        'Volatilidad': f"{vol:.1%}",
        'Max Drawdown': f"{max_dd:.1%}",
        'Sharpe': f"{sharpe:.2f}",
        'Mejor Año': f"{best_year:.1%}",
        'Peor Año': f"{worst_year:.1%}",
    }

metrics = [
    calc_metrics(equity_gold, "Oro Puro (spot)", ret_gold),
    calc_metrics(equity_1x, "GLD (1x)", ret_1x),
    calc_metrics(equity_2x, "Oro 2x apalancado", ret_2x),
    calc_metrics(equity_3x, "Oro 3x apalancado", ret_3x),
]

df_metrics = pd.DataFrame(metrics).set_index('Nombre')
print("\n" + "="*90)
print("SIMULACIÓN ORO 50 AÑOS (1975-2025) — $10,000 INICIALES")
print("="*90)
print(df_metrics.to_string())

# ─── Tabla por quinquenios ───
print("\n" + "="*90)
print("VALOR DE LA CARTERA POR QUINQUENIOS ($10,000 iniciales)")
print("="*90)

decades_data = []
years_check = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
for eq, name in [(equity_gold, "Oro Puro"), (equity_1x, "GLD 1x"),
                  (equity_2x, "Oro 2x"), (equity_3x, "Oro 3x")]:
    row = {'Cartera': name}
    for year in years_check:
        mask = eq.index.year == year
        if mask.any():
            val = eq[mask].iloc[-1]
            row[str(year)] = f"${val:,.0f}"
        else:
            row[str(year)] = "—"
    decades_data.append(row)

df_decades = pd.DataFrame(decades_data).set_index('Cartera')
print(df_decades.to_string())

# ─── Retornos por década ───
print("\n" + "="*90)
print("CAGR POR DÉCADA")
print("="*90)

decade_cagr = []
for eq, name in [(equity_gold, "Oro Puro"), (equity_1x, "GLD 1x"),
                  (equity_2x, "Oro 2x"), (equity_3x, "Oro 3x")]:
    row = {'Cartera': name}
    for start_yr, end_yr in [(1975,1985), (1985,1995), (1995,2005), (2005,2015), (2015,2025)]:
        try:
            start_mask = eq.index.year == start_yr
            end_mask = eq.index.year == end_yr
            if start_mask.any() and end_mask.any():
                v0 = eq[start_mask].iloc[0]
                v1 = eq[end_mask].iloc[-1]
                cagr = (v1/v0)**(1/10) - 1
                row[f"{start_yr}-{end_yr}"] = f"{cagr:.1%}"
            else:
                row[f"{start_yr}-{end_yr}"] = "—"
        except:
            row[f"{start_yr}-{end_yr}"] = "—"
    decade_cagr.append(row)

df_cagr = pd.DataFrame(decade_cagr).set_index('Cartera')
print(df_cagr.to_string())

# ─── GRÁFICOS ───
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Simulación Oro 50 Años (1975-2025) — $10,000 Iniciales',
             fontsize=14, fontweight='bold', y=0.98)

colors = {'gold_pure': '#888888', '1x': '#FFD700', '2x': '#FF8C00', '3x': '#DC143C'}

# 1. Equity curves (log)
ax1 = axes[0, 0]
ax1.semilogy(equity_gold.index, equity_gold, color=colors['gold_pure'], alpha=0.5,
             label='Oro Puro', linewidth=1)
ax1.semilogy(equity_1x.index, equity_1x, color=colors['1x'], label='GLD (1x)', linewidth=2)
ax1.semilogy(equity_2x.index, equity_2x, color=colors['2x'], label='Oro 2x', linewidth=2)
ax1.semilogy(equity_3x.index, equity_3x, color=colors['3x'], label='Oro 3x', linewidth=2)
ax1.set_title('Equity Curves (Escala Logarítmica)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Valor ($)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
# Annotate final values
for eq, name, color in [(equity_1x, 'GLD', colors['1x']),
                         (equity_2x, '2x', colors['2x']),
                         (equity_3x, '3x', colors['3x'])]:
    ax1.annotate(f"${eq.iloc[-1]:,.0f}", xy=(eq.index[-1], eq.iloc[-1]),
                fontsize=8, color=color, fontweight='bold',
                xytext=(5, 0), textcoords='offset points')

# 2. Equity curves (linear)
ax2 = axes[0, 1]
ax2.plot(equity_gold.index, equity_gold/1000, color=colors['gold_pure'], alpha=0.5,
         label='Oro Puro', linewidth=1)
ax2.plot(equity_1x.index, equity_1x/1000, color=colors['1x'], label='GLD (1x)', linewidth=2)
ax2.plot(equity_2x.index, equity_2x/1000, color=colors['2x'], label='Oro 2x', linewidth=2)
ax2.plot(equity_3x.index, equity_3x/1000, color=colors['3x'], label='Oro 3x', linewidth=2)
ax2.set_title('Equity Curves (Escala Lineal)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Valor ($K)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Drawdowns
ax3 = axes[1, 0]
for eq, name, color in [(equity_1x, 'GLD 1x', colors['1x']),
                          (equity_2x, 'Oro 2x', colors['2x']),
                          (equity_3x, 'Oro 3x', colors['3x'])]:
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max * 100
    ax3.fill_between(dd.index, dd, 0, alpha=0.3, color=color, label=name)
    ax3.plot(dd.index, dd, color=color, alpha=0.5, linewidth=0.5)
ax3.set_title('Drawdowns', fontsize=11, fontweight='bold')
ax3.set_ylabel('Drawdown (%)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Rolling 10-year CAGR
ax4 = axes[1, 1]
window = 252 * 10  # 10 años
for eq, name, color in [(equity_1x, 'GLD 1x', colors['1x']),
                          (equity_2x, 'Oro 2x', colors['2x']),
                          (equity_3x, 'Oro 3x', colors['3x'])]:
    if len(eq) > window:
        rolling_cagr = ((eq / eq.shift(window)) ** (1/10) - 1) * 100
        ax4.plot(rolling_cagr.index, rolling_cagr, color=color, label=name, linewidth=1.5)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.set_title('CAGR Rolling 10 Años (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('CAGR (%)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = "/Users/rodrigomarchalpuchol/Library/CloudStorage/GoogleDrive-rmarchal75@gmail.com/Mi unidad/Claude/Code/gold_50y_simulation.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nGráfico guardado: {output_path}")

# ─── Análisis del Volatility Drag ───
print("\n" + "="*90)
print("ANÁLISIS DEL VOLATILITY DRAG Y COSTES")
print("="*90)

gold_vol = daily_ret.std() * np.sqrt(252)
print(f"Volatilidad anualizada del oro (50 años): {gold_vol:.1%}")
print(f"\nVolatility drag (teórico = 0.5 × σ² × leverage²):")
print(f"  1x: {0.5 * gold_vol**2 * 100:.2f}% anual")
print(f"  2x: {0.5 * (2*gold_vol)**2 * 100:.2f}% anual  (4× el de 1x)")
print(f"  3x: {0.5 * (3*gold_vol)**2 * 100:.2f}% anual  (9× el de 1x)")

avg_borrow = np.mean([get_borrowing_cost(y) for y in range(1975, 2026)])
print(f"\nCoste medio de financiación (Fed Funds avg): {avg_borrow:.2%}")
print(f"\nCoste total anual estimado (expense + borrowing + vol drag):")
print(f"  GLD 1x: {ER_1X + 0.5*gold_vol**2:.2%}")
print(f"  Oro 2x:  {ER_2X + avg_borrow*1 + 0.5*(2*gold_vol)**2:.2%}")
print(f"  Oro 3x:  {ER_3X + avg_borrow*2 + 0.5*(3*gold_vol)**2:.2%}")

# ─── Comparativa con inflación ───
print("\n" + "="*90)
print("RETORNO REAL (AJUSTADO POR INFLACIÓN ~3.5% anual promedio 1975-2025)")
print("="*90)
avg_inflation = 0.035
for eq, name in [(equity_gold, "Oro Puro"), (equity_1x, "GLD 1x"),
                  (equity_2x, "Oro 2x"), (equity_3x, "Oro 3x")]:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    nominal_cagr = (eq.iloc[-1] / INITIAL) ** (1 / years) - 1
    real_cagr = (1 + nominal_cagr) / (1 + avg_inflation) - 1
    real_final = INITIAL * (1 + real_cagr) ** years
    print(f"  {name:20s}: CAGR nominal {nominal_cagr:.2%} → real {real_cagr:.2%} → valor real ${real_final:,.0f}")

print("\n✓ Simulación completada")
