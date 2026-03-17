#!/usr/bin/env python3
"""
Stress Test: Apalancamiento Bancario + Momentum v8
===================================================

La simulacion anterior usa retornos anuales, que suavizan el riesgo.
En realidad:
  - Los pagos del prestamo son MENSUALES
  - Puedes empezar justo antes de un drawdown del -43%
  - Dos anos malos seguidos son posibles

Este script simula:
  1. Escenarios sinteticos: que pasa si el primer ano es -20%, -30%, -40%
  2. Cuotas mensuales vs anuales
  3. Combinacion de primer ano malo + retornos reales despues
  4. El peor escenario: empezar en el peor momento posible
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

EQUITY = 10_000
PRESTAMO = 70_000
TOTAL = EQUITY + PRESTAMO
PLAZO = 7  # anos
SPREAD = 2.0  # sobre Euribor

# Euribor actual/reciente para escenarios forward-looking
EURIBOR_ACTUAL = 2.50  # %
TIPO_ANUAL = (EURIBOR_ACTUAL + SPREAD) / 100  # 4.5%
TIPO_MENSUAL = TIPO_ANUAL / 12

# Cuota mensual francesa
n_meses = PLAZO * 12  # 84 meses
CUOTA_MENSUAL = PRESTAMO * TIPO_MENSUAL / (1 - (1 + TIPO_MENSUAL) ** (-n_meses))
CUOTA_ANUAL_TOTAL = CUOTA_MENSUAL * 12

# Retornos v8 historicos
V8_RETURNS = {
    2006: 0.5499, 2007: 0.6645, 2008: -0.2377, 2009: 0.5538, 2010: 1.2449,
    2011: 0.2969, 2012: -0.0094, 2013: 0.4185, 2014: 0.7278, 2015: 0.1881,
    2016: 0.0788, 2017: 0.2628, 2018: -0.2922, 2019: 0.0622, 2020: 0.9133,
    2021: 0.4671, 2022: 0.0275, 2023: 0.1741, 2024: 1.1643, 2025: 0.7681,
    2026: 0.6486,
}

# Max drawdown historico del v8: -43.5%
MAX_DD = -0.435

print("=" * 100)
print("STRESS TEST: APALANCAMIENTO BANCARIO + MOMENTUM v8")
print("=" * 100)
print(f"""
  Equity: EUR {EQUITY:,} | Prestamo: EUR {PRESTAMO:,} | Total: EUR {TOTAL:,}
  Plazo: {PLAZO} anos ({n_meses} meses) | Tipo: {TIPO_ANUAL:.1%} (Euribor {EURIBOR_ACTUAL}% + {SPREAD}%)
  Cuota mensual: EUR {CUOTA_MENSUAL:,.0f} | Cuota anual total: EUR {CUOTA_ANUAL_TOTAL:,.0f}
  Max drawdown historico v8: {MAX_DD:.1%}
""")


# =============================================================================
# 1. REALIDAD BASICA: cuanto cuesta el prestamo y cuanto necesitas ganar
# =============================================================================
print("=" * 100)
print("1. REALIDAD BASICA: COSTE DEL PRESTAMO")
print("=" * 100)

# Tabla de amortizacion completa
deuda = PRESTAMO
total_intereses = 0
total_pagado = 0

print(f"\n  {'Ano':<6} {'Cuotas pag.':>12} {'Intereses':>12} {'Amort. cap.':>12} {'Deuda pend.':>12}")
print(f"  {'-' * 58}")

for year in range(1, PLAZO + 1):
    intereses_ano = 0
    amort_ano = 0
    for month in range(12):
        interes_mes = deuda * TIPO_MENSUAL
        amort_mes = CUOTA_MENSUAL - interes_mes
        deuda -= amort_mes
        intereses_ano += interes_mes
        amort_ano += amort_mes

    total_intereses += intereses_ano
    total_pagado += CUOTA_MENSUAL * 12
    print(f"  {year:<6} EUR {CUOTA_MENSUAL*12:>8,.0f} EUR {intereses_ano:>8,.0f} EUR {amort_ano:>8,.0f} EUR {max(deuda,0):>8,.0f}")

print(f"\n  Total pagado:    EUR {total_pagado:,.0f}")
print(f"  Total intereses: EUR {total_intereses:,.0f}")
print(f"  Coste real:      {total_intereses/PRESTAMO*100:.1f}% sobre el prestamo")
print(f"\n  Rentabilidad MINIMA necesaria para no perder dinero:")
print(f"    Solo cubrir intereses: {total_intereses / TOTAL / PLAZO * 100:.1f}% anual sobre EUR {TOTAL:,}")
print(f"    Cubrir cuotas completas: {CUOTA_ANUAL_TOTAL / TOTAL * 100:.1f}% anual sobre EUR {TOTAL:,}")


# =============================================================================
# 2. QUE PASA SI EL PRIMER ANO ES MALO (simulacion mensual)
# =============================================================================
print(f"\n{'=' * 100}")
print("2. PRIMER ANO MALO: SIMULACION MENSUAL")
print("=" * 100)

print(f"\n  Escenario: el primer ano el v8 tiene un retorno negativo.")
print(f"  Despues, retornos medios historicos (~35% anual).")
print(f"  Pagos: cuota mensual de EUR {CUOTA_MENSUAL:,.0f}")

ret_post_crisis = 0.35  # CAGR medio v8 para anos posteriores
ret_mensual_normal = (1 + ret_post_crisis) ** (1/12) - 1

first_year_scenarios = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.435]

print(f"\n  {'1er ano':>10} {'Port. 12m':>12} {'Deuda 12m':>12} {'Equity 12m':>12} "
      f"{'Port. 84m':>12} {'Deuda 84m':>10} {'Equity 84m':>12} {'Impago?':>10}")
print(f"  {'-' * 98}")

for first_ret in first_year_scenarios:
    # Mes a mes, primer ano: distribuir la caida uniformemente
    ret_mensual_crisis = (1 + first_ret) ** (1/12) - 1

    portfolio = float(TOTAL)
    deuda = float(PRESTAMO)
    impago = False
    impago_mes = None

    for mes in range(1, n_meses + 1):
        # Retorno mensual
        if mes <= 12:
            ret = ret_mensual_crisis
        else:
            ret = ret_mensual_normal

        portfolio *= (1 + ret)

        # Pagar cuota mensual
        if portfolio >= CUOTA_MENSUAL:
            interes = deuda * TIPO_MENSUAL
            amort = CUOTA_MENSUAL - interes
            portfolio -= CUOTA_MENSUAL
            deuda -= amort
            deuda = max(deuda, 0)
        else:
            impago = True
            impago_mes = mes
            break

        # Guardar valores a 12 meses
        if mes == 12:
            port_12 = portfolio
            deuda_12 = deuda
            eq_12 = portfolio - deuda

    if not impago:
        port_final = portfolio
        deuda_final = deuda
        eq_final = portfolio - deuda
    else:
        port_final = portfolio
        deuda_final = deuda
        eq_final = portfolio - deuda

    imp_str = f"Mes {impago_mes}" if impago else "No"
    if impago:
        print(f"  {first_ret:>9.0%} EUR {port_12 if not impago or impago_mes > 12 else 0:>8,.0f} "
              f"EUR {deuda_12 if not impago or impago_mes > 12 else deuda:>8,.0f} "
              f"EUR {eq_12 if not impago or impago_mes > 12 else -deuda:>8,.0f} "
              f"{'---':>12} {'---':>10} {'---':>12} {imp_str:>10}")
    else:
        print(f"  {first_ret:>9.0%} EUR {port_12:>8,.0f} EUR {deuda_12:>8,.0f} EUR {eq_12:>8,.0f} "
              f"EUR {port_final:>8,.0f} EUR {deuda_final:>6,.0f} EUR {eq_final:>8,.0f} {imp_str:>10}")


# =============================================================================
# 3. PEOR ESCENARIO REALISTA: drawdown concentrado + recuperacion
# =============================================================================
print(f"\n{'=' * 100}")
print("3. ESCENARIO REALISTA: DRAWDOWN CONCENTRADO EN 3-6 MESES")
print("=" * 100)

print(f"\n  El v8 no cae uniformemente. El max DD de -43.5% puede ocurrir en 3-6 meses.")
print(f"  Despues hay rebote. Simulamos: caida rapida + recuperacion + retorno normal.\n")

def sim_mensual(monthly_returns, prestamo=PRESTAMO, total_inv=TOTAL):
    """Simula mes a mes con cuota francesa mensual."""
    portfolio = float(total_inv)
    deuda = float(prestamo)
    tipo_m = TIPO_MENSUAL
    cuota_m = prestamo * tipo_m / (1 - (1 + tipo_m) ** (-len(monthly_returns)))

    history = []
    impago = False
    impago_mes = None

    for mes, ret in enumerate(monthly_returns, 1):
        portfolio *= (1 + ret)

        if deuda > 0:
            if portfolio >= cuota_m:
                interes = deuda * tipo_m
                amort = cuota_m - interes
                portfolio -= cuota_m
                deuda -= amort
                deuda = max(deuda, 0)
            else:
                impago = True
                impago_mes = mes

        history.append({
            'mes': mes, 'ret': ret, 'portfolio': portfolio,
            'deuda': deuda, 'equity': portfolio - deuda,
            'impago': impago,
        })

        if impago:
            break

    return history


# Escenario A: Caida del -35% en 4 meses, rebote 20% en 4 meses, despues normal
def gen_scenario(dd_pct, dd_months, bounce_pct, bounce_months, normal_annual, total_months=84):
    """Genera retornos mensuales para un escenario."""
    rets = []
    # Caida
    monthly_dd = (1 + dd_pct) ** (1/dd_months) - 1
    rets.extend([monthly_dd] * dd_months)
    # Rebote
    monthly_bounce = (1 + bounce_pct) ** (1/bounce_months) - 1
    rets.extend([monthly_bounce] * bounce_months)
    # Normal
    monthly_normal = (1 + normal_annual) ** (1/12) - 1
    remaining = total_months - dd_months - bounce_months
    rets.extend([monthly_normal] * remaining)
    return rets[:total_months]


scenarios = [
    ("DD -20% en 3m, rebote +15% en 3m, luego 35%/a",
     gen_scenario(-0.20, 3, 0.15, 3, 0.35)),
    ("DD -30% en 4m, rebote +20% en 4m, luego 35%/a",
     gen_scenario(-0.30, 4, 0.20, 4, 0.35)),
    ("DD -35% en 4m, rebote +25% en 4m, luego 35%/a",
     gen_scenario(-0.35, 4, 0.25, 4, 0.35)),
    ("DD -40% en 5m, rebote +30% en 5m, luego 35%/a",
     gen_scenario(-0.40, 5, 0.30, 5, 0.35)),
    ("DD -43.5% en 6m, rebote +35% en 6m, luego 35%/a",
     gen_scenario(-0.435, 6, 0.35, 6, 0.35)),
    ("DD -30% en 3m, SIN rebote, luego 20%/a (pesimista)",
     gen_scenario(-0.30, 3, 0.0, 1, 0.20)),
    ("DD -40% en 4m, luego SOLO 15%/a (muy pesimista)",
     gen_scenario(-0.40, 4, 0.05, 2, 0.15)),
    ("Dos anos malos: -25% a1, -15% a2, luego 30%/a",
     gen_scenario(-0.25, 12, -0.15, 12, 0.30)),
]

print(f"  {'Escenario':<55} {'Eq. 12m':>10} {'Eq. min':>10} {'Eq. final':>12} {'Impago':>10}")
print(f"  {'-' * 100}")

for name, rets in scenarios:
    hist = sim_mensual(rets)
    eq_12 = hist[11]['equity'] if len(hist) > 11 else hist[-1]['equity']
    eq_min = min(h['equity'] for h in hist)
    eq_final = hist[-1]['equity']
    impago = hist[-1]['impago']
    imp_str = f"Mes {hist[-1]['mes']}" if impago else "No"

    print(f"  {name:<55} EUR {eq_12:>7,.0f} EUR {eq_min:>7,.0f} EUR {eq_final:>8,.0f} {imp_str:>10}")


# =============================================================================
# 4. COLCHON NECESARIO PARA SOBREVIVIR
# =============================================================================
print(f"\n{'=' * 100}")
print("4. COLCHON NECESARIO: cuanto cash reservar para sobrevivir al peor caso")
print("=" * 100)

print(f"\n  Si reservamos X meses de cuotas en cash (no invertido), el portfolio")
print(f"  tiene menos capital pero hay un colchon para pagar cuotas en meses malos.\n")

# Escenario: DD -40% en 5 meses, rebote +30% en 5 meses, luego 35%/a
test_rets = gen_scenario(-0.40, 5, 0.30, 5, 0.35)

print(f"  Escenario test: DD -40% en 5m, rebote +30% en 5m, luego 35%/a")
print(f"\n  {'Colchon':>10} {'Cash inic.':>12} {'Invertido':>12} {'Eq. min':>10} {'Eq. final':>12} {'Impago':>8}")
print(f"  {'-' * 70}")

for meses_colchon in [0, 6, 12, 18, 24, 30, 36]:
    cash_reserva = CUOTA_MENSUAL * meses_colchon
    invertido = TOTAL - cash_reserva
    if invertido <= 0:
        continue

    portfolio = float(invertido)
    deuda = float(PRESTAMO)
    cash = float(cash_reserva)
    impago = False
    eq_min = invertido + cash - PRESTAMO
    history = []

    cuota_m = PRESTAMO * TIPO_MENSUAL / (1 - (1 + TIPO_MENSUAL) ** (-n_meses))

    for mes in range(n_meses):
        ret = test_rets[mes] if mes < len(test_rets) else ((1.35)**(1/12)-1)
        portfolio *= (1 + ret)

        if deuda > 0:
            interes = deuda * TIPO_MENSUAL
            amort = cuota_m - interes

            if portfolio >= cuota_m:
                portfolio -= cuota_m
                deuda -= amort
            elif portfolio + cash >= cuota_m:
                falta = cuota_m - portfolio
                portfolio = 0
                cash -= falta
                deuda -= amort
            else:
                impago = True
                break

            deuda = max(deuda, 0)

        eq = portfolio + cash - deuda
        eq_min = min(eq_min, eq)

    eq_final = portfolio + cash - deuda
    imp_str = "SI" if impago else "No"

    print(f"  {meses_colchon:>7} m  EUR {cash_reserva:>8,.0f} EUR {invertido:>8,.0f} "
          f"EUR {eq_min:>7,.0f} EUR {eq_final:>8,.0f} {imp_str:>8}")


# =============================================================================
# 5. ESCENARIO MAS PELIGROSO: 2018 real mes a mes (aproximado)
# =============================================================================
print(f"\n{'=' * 100}")
print("5. RECONSTRUCCION: ANO 2018 DEL V8 (peor ano historico, -29.2%)")
print("=" * 100)

# Aproximacion mensual del v8 en 2018 (basado en comportamiento tipico del mercado)
# El v8 cayo ~29% ese ano. El mercado tuvo:
# - Ene-Sep: plano/ligeramente positivo con pico en septiembre
# - Oct-Dic: caida fuerte (-20% del mercado, amplificado en momentum)
# Aproximacion del v8 mensual 2018:
v8_2018_monthly = [0.05, -0.08, -0.03, 0.04, 0.03, 0.01, 0.02, 0.03, 0.02, -0.15, -0.12, -0.08]
# Verificar que da ~-29%
total_2018 = 1.0
for r in v8_2018_monthly:
    total_2018 *= (1 + r)
print(f"\n  Retorno 2018 reconstruido: {total_2018-1:.1%} (real: -29.2%)")

# Seguido de 2019 (+6.2%) y 2020 (+91.3%)
v8_2019_monthly = [(1.0622)**(1/12)-1] * 12  # distribuido uniformemente
v8_2020_monthly = [(1.9133)**(1/12)-1] * 12

# Simulamos 3 anos (36 meses)
rets_2018_2020 = v8_2018_monthly + v8_2019_monthly + v8_2020_monthly
# Extender a 7 anos con 35% anual
normal_monthly = (1.35)**(1/12) - 1
rets_2018_full = rets_2018_2020 + [normal_monthly] * (84 - 36)

# Sin colchon
hist_no_colchon = sim_mensual(rets_2018_full)

# Con 12 meses de colchon
cash_12 = CUOTA_MENSUAL * 12
inv_12 = TOTAL - cash_12

portfolio_c = float(inv_12)
deuda_c = float(PRESTAMO)
cash_c = float(cash_12)
cuota_m = PRESTAMO * TIPO_MENSUAL / (1 - (1 + TIPO_MENSUAL) ** (-84))

print(f"\n  Sin colchon (EUR {TOTAL:,} invertidos):")
print(f"  {'Mes':<6} {'Portfolio':>12} {'Deuda':>12} {'Equity':>12}")
print(f"  {'-' * 44}")
for h in hist_no_colchon:
    if h['mes'] in [1, 3, 6, 9, 12, 18, 24, 36, 48, 60, 72, 84] or h['impago']:
        print(f"  {h['mes']:<6} EUR {h['portfolio']:>8,.0f} EUR {h['deuda']:>8,.0f} EUR {h['equity']:>8,.0f}"
              f"{'  *** IMPAGO' if h['impago'] else ''}")

# Con colchon
print(f"\n  Con colchon 12 meses (EUR {cash_12:,.0f} cash + EUR {inv_12:,.0f} invertidos):")
print(f"  {'Mes':<6} {'Portfolio':>12} {'Cash':>10} {'Deuda':>12} {'Equity':>12}")
print(f"  {'-' * 54}")

history_c = []
impago_c = False
for mes in range(84):
    ret = rets_2018_full[mes] if mes < len(rets_2018_full) else normal_monthly
    portfolio_c *= (1 + ret)

    if deuda_c > 0:
        interes = deuda_c * TIPO_MENSUAL
        amort = cuota_m - interes

        if portfolio_c >= cuota_m:
            portfolio_c -= cuota_m
            deuda_c -= amort
        elif portfolio_c + cash_c >= cuota_m:
            falta = cuota_m - portfolio_c
            portfolio_c = 0
            cash_c -= falta
            deuda_c -= amort
        else:
            impago_c = True

        deuda_c = max(deuda_c, 0)

    eq = portfolio_c + cash_c - deuda_c
    m = mes + 1
    history_c.append({'mes': m, 'portfolio': portfolio_c, 'cash': cash_c,
                       'deuda': deuda_c, 'equity': eq, 'impago': impago_c})

    if m in [1, 3, 6, 9, 12, 18, 24, 36, 48, 60, 72, 84] or impago_c:
        print(f"  {m:<6} EUR {portfolio_c:>8,.0f} EUR {cash_c:>6,.0f} EUR {deuda_c:>8,.0f} EUR {eq:>8,.0f}"
              f"{'  *** IMPAGO' if impago_c else ''}")

    if impago_c:
        break


# =============================================================================
# 6. RESUMEN FINAL
# =============================================================================
print(f"\n{'=' * 100}")
print("RESUMEN Y RECOMENDACION")
print(f"{'=' * 100}")

print(f"""
  RIESGOS REALES:
  ===============
  1. Cuota mensual: EUR {CUOTA_MENSUAL:,.0f}/mes = EUR {CUOTA_ANUAL_TOTAL:,.0f}/ano
     → Necesitas que el portfolio genere al menos eso para sobrevivir
     → Sobre EUR {TOTAL:,} invertidos = {CUOTA_ANUAL_TOTAL/TOTAL*100:.1f}% anual minimo

  2. Si el v8 cae -40% en los primeros 5 meses:
     → Portfolio baja de EUR {TOTAL:,} a ~EUR {TOTAL*0.60:,.0f}
     → Sigues debiendo EUR {PRESTAMO:,} + intereses
     → Equity negativo: EUR {TOTAL*0.60 - PRESTAMO:,.0f}
     → PERO puedes seguir pagando cuotas (EUR {CUOTA_MENSUAL:,.0f}/mes)
       desde los EUR {TOTAL*0.60:,.0f} que quedan

  3. El peligro real es: caida fuerte + SIN recuperacion rapida
     → Dos anos consecutivos malos (-25% + -15%) = stress severo
     → Pero en 20 anos de v8, NUNCA ha habido 2 anos negativos seguidos

  COLCHON RECOMENDADO:
  ====================
  - 12 meses de cuotas = EUR {CUOTA_MENSUAL*12:,.0f} en cash
  - Inviertes EUR {TOTAL - CUOTA_MENSUAL*12:,.0f} en v8
  - Sobrevives incluso al peor escenario historico (-43.5% DD)
  - Coste de oportunidad: reduces retorno ~15% vs invertir todo

  ESTRUCTURA OPTIMA:
  ==================
  - EUR {EQUITY:,} equity propio
  - EUR {PRESTAMO:,} prestamo a 7 anos (Euribor + 2%)
  - EUR {CUOTA_MENSUAL*12:,.0f} en deposito/cuenta (colchon 12 meses)
  - EUR {TOTAL - CUOTA_MENSUAL*12:,.0f} en Momentum v8
  - Cuota mensual: EUR {CUOTA_MENSUAL:,.0f}
  - Coste total intereses: EUR {total_intereses:,.0f} (en 7 anos)
""")
