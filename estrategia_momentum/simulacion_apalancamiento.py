#!/usr/bin/env python3
"""
Simulacion Apalancamiento Bancario + Momentum Breakout v8
==========================================================

Escenario:
  - Equity propio: EUR 10,000
  - Prestamo bancario: EUR 70,000 (7 anos, sin garantia real)
  - Total invertido: EUR 80,000
  - Coste: Euribor 12M + 2% (spread por no tener garantia)
  - Rentabilidad: retornos reales ano a ano del backtest v8

Opciones simuladas:
  A) 100% invertido, amortizacion francesa (cuota constante anual)
  B) Colchon de 2 anos de cuotas en cash, resto invertido
  C) Amortizacion acelerada: pagar deuda lo antes posible con beneficios
  D) Solo equity (sin deuda) - benchmark

Usa retornos reales del v8 para todas las ventanas de 7 anos posibles.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(os.path.dirname(BASE_DIR), "comparativa_3_estrategias.csv")

# =============================================================================
# 1. Datos
# =============================================================================

# Euribor 12M historico (media anual)
EURIBOR = {
    2006: 3.44, 2007: 4.45, 2008: 3.85, 2009: 1.25, 2010: 1.35,
    2011: 2.01, 2012: 0.55, 2013: 0.54, 2014: 0.33, 2015: 0.06,
    2016: -0.03, 2017: -0.19, 2018: -0.13, 2019: -0.26, 2020: -0.30,
    2021: -0.50, 2022: 3.02, 2023: 3.86, 2024: 2.96, 2025: 2.50,
    2026: 2.50,
}
SPREAD = 2.0  # puntos sobre Euribor

EQUITY_INICIAL = 10_000
PRESTAMO = 70_000
TOTAL_INVERTIDO = EQUITY_INICIAL + PRESTAMO
PLAZO_ANOS = 7


def loan_rate(year):
    """Tipo de interes del prestamo = Euribor 12M + spread."""
    euribor = EURIBOR.get(year, 2.50)
    return max(euribor + SPREAD, SPREAD) / 100  # floor en el spread (si Euribor negativo)


def cuota_francesa(principal, rate, anos):
    """Cuota anual constante (amortizacion francesa)."""
    if rate == 0:
        return principal / anos
    return principal * rate / (1 - (1 + rate) ** (-anos))


# =============================================================================
# 2. Cargar retornos v8
# =============================================================================
df = pd.read_csv(CSV_FILE)
v8_returns = df[['year', 'v8_return']].dropna().set_index('year')['v8_return'].to_dict()
all_years = sorted(v8_returns.keys())

print("=" * 100)
print("SIMULACION: APALANCAMIENTO BANCARIO + MOMENTUM BREAKOUT v8")
print("=" * 100)
print(f"\n  Equity propio:    EUR {EQUITY_INICIAL:>10,}")
print(f"  Prestamo:         EUR {PRESTAMO:>10,}  (plazo {PLAZO_ANOS} anos)")
print(f"  Total invertido:  EUR {TOTAL_INVERTIDO:>10,}")
print(f"  Coste deuda:      Euribor 12M + {SPREAD:.0f}%")
print(f"  Retornos v8:      {len(all_years)} anos ({all_years[0]}-{all_years[-1]})")


# =============================================================================
# 3. Simulacion de cada opcion para una ventana de 7 anos
# =============================================================================
def simular_opcion_A(start_year, returns, plazo=7):
    """
    Opcion A: 100% invertido, amortizacion francesa.
    Cada ano: portfolio crece por retorno v8, se paga cuota del prestamo.
    Si portfolio < cuota → IMPAGO.
    """
    portfolio = float(TOTAL_INVERTIDO)
    deuda = float(PRESTAMO)
    cash = 0.0
    years_data = []
    impago = False

    for i in range(plazo):
        year = start_year + i
        ret = returns.get(year, 0.0)
        rate = loan_rate(year)

        # Calcular cuota francesa sobre deuda restante y anos restantes
        anos_rest = plazo - i
        if deuda > 0 and anos_rest > 0:
            cuota = cuota_francesa(deuda, rate, anos_rest)
            intereses = deuda * rate
            amort_capital = cuota - intereses
        else:
            cuota = intereses = amort_capital = 0

        # Portfolio crece
        portfolio_pre = portfolio
        portfolio *= (1 + ret)

        # Pagar cuota
        if deuda > 0:
            if portfolio >= cuota:
                portfolio -= cuota
                deuda -= amort_capital
                deuda = max(deuda, 0)
            else:
                impago = True
                # Paga lo que puede
                pagado = portfolio
                portfolio = 0
                deuda -= max(pagado - intereses, 0)

        equity_neto = portfolio - deuda

        years_data.append({
            'year': year, 'return': ret, 'rate': rate,
            'portfolio': portfolio, 'deuda': deuda,
            'cuota': cuota, 'intereses': intereses,
            'equity_neto': equity_neto, 'impago': impago,
        })

        if impago:
            break

    return years_data


def simular_opcion_B(start_year, returns, plazo=7):
    """
    Opcion B: Colchon de 2 anos de cuotas en cash.
    Calcula cuota inicial, reserva 2x en cash, invierte el resto.
    Si portfolio baja, usa colchon para pagar cuotas.
    """
    rate_inicial = loan_rate(start_year)
    cuota_est = cuota_francesa(PRESTAMO, rate_inicial, plazo)
    colchon_inicial = cuota_est * 2  # 2 anos de cuotas
    invertido = TOTAL_INVERTIDO - colchon_inicial
    cash = colchon_inicial

    portfolio = float(invertido)
    deuda = float(PRESTAMO)
    years_data = []
    impago = False

    for i in range(plazo):
        year = start_year + i
        ret = returns.get(year, 0.0)
        rate = loan_rate(year)

        anos_rest = plazo - i
        if deuda > 0 and anos_rest > 0:
            cuota = cuota_francesa(deuda, rate, anos_rest)
            intereses = deuda * rate
            amort_capital = cuota - intereses
        else:
            cuota = intereses = amort_capital = 0

        # Portfolio crece
        portfolio *= (1 + ret)
        # Cash no renta (cuenta corriente)

        # Pagar cuota: primero del portfolio, si no alcanza del colchon
        if deuda > 0:
            if portfolio >= cuota:
                portfolio -= cuota
                deuda -= amort_capital
                deuda = max(deuda, 0)
            elif portfolio + cash >= cuota:
                falta = cuota - portfolio
                portfolio = 0
                cash -= falta
                deuda -= amort_capital
                deuda = max(deuda, 0)
            else:
                impago = True
                pagado = portfolio + cash
                portfolio = 0
                cash = 0
                deuda -= max(pagado - intereses, 0)

        equity_neto = portfolio + cash - deuda

        years_data.append({
            'year': year, 'return': ret, 'rate': rate,
            'portfolio': portfolio, 'deuda': deuda,
            'cuota': cuota, 'intereses': intereses,
            'cash': cash, 'equity_neto': equity_neto,
            'impago': impago,
        })

        if impago:
            break

    return years_data


def simular_opcion_C(start_year, returns, plazo=7):
    """
    Opcion C: Amortizacion acelerada.
    Paga cuota francesa normal + todo el excedente por encima de 20,000 EUR
    va a amortizar deuda anticipadamente. Objetivo: liquidar deuda ASAP.
    """
    MINIMO_PORTFOLIO = 20_000  # mantener minimo en cartera

    portfolio = float(TOTAL_INVERTIDO)
    deuda = float(PRESTAMO)
    years_data = []
    impago = False

    for i in range(plazo):
        year = start_year + i
        ret = returns.get(year, 0.0)
        rate = loan_rate(year)

        anos_rest = plazo - i
        if deuda > 0 and anos_rest > 0:
            cuota = cuota_francesa(deuda, rate, anos_rest)
            intereses = deuda * rate
            amort_capital = cuota - intereses
        else:
            cuota = intereses = amort_capital = 0

        # Portfolio crece
        portfolio *= (1 + ret)

        # Pagar cuota normal
        pago_total = 0
        amort_extra = 0
        if deuda > 0:
            if portfolio >= cuota:
                portfolio -= cuota
                deuda -= amort_capital
                deuda = max(deuda, 0)
                pago_total = cuota

                # Amortizacion anticipada: todo por encima del minimo
                if portfolio > MINIMO_PORTFOLIO and deuda > 0:
                    excedente = portfolio - MINIMO_PORTFOLIO
                    amort_extra = min(excedente, deuda)
                    portfolio -= amort_extra
                    deuda -= amort_extra
                    pago_total += amort_extra
            else:
                impago = True
                pagado = portfolio
                portfolio = 0
                deuda -= max(pagado - intereses, 0)

        equity_neto = portfolio - deuda

        years_data.append({
            'year': year, 'return': ret, 'rate': rate,
            'portfolio': portfolio, 'deuda': deuda,
            'cuota': cuota, 'intereses': intereses,
            'amort_extra': amort_extra, 'pago_total': pago_total,
            'equity_neto': equity_neto, 'impago': impago,
        })

        if impago:
            break

    return years_data


def simular_opcion_D(start_year, returns, plazo=7):
    """
    Opcion D: Solo equity propio, sin deuda (benchmark).
    """
    portfolio = float(EQUITY_INICIAL)
    years_data = []

    for i in range(plazo):
        year = start_year + i
        ret = returns.get(year, 0.0)
        portfolio *= (1 + ret)
        years_data.append({
            'year': year, 'return': ret,
            'portfolio': portfolio, 'equity_neto': portfolio,
        })

    return years_data


# =============================================================================
# 4. Ejecutar todas las ventanas de 7 anos
# =============================================================================
print(f"\n{'=' * 100}")
print("TODAS LAS VENTANAS DE 7 ANOS POSIBLES")
print(f"{'=' * 100}")

# Ventanas posibles
windows = []
for start in all_years:
    end = start + PLAZO_ANOS - 1
    if end <= all_years[-1]:
        windows.append(start)

print(f"\n  Ventanas posibles: {len(windows)} (de {windows[0]} a {windows[-1]})")

# Tabla resumen de todas las ventanas
header = (f"  {'Inicio':<8} {'Fin':<6} "
          f"{'A) 100% inv':>14} {'B) Colchon':>14} {'C) Acelerada':>14} {'D) Solo 10K':>14} "
          f"{'A imp':>6} {'B imp':>6} {'C imp':>6}")
print(f"\n{header}")
print(f"  {'-' * 106}")

summary = {'A': [], 'B': [], 'C': [], 'D': []}
impagos = {'A': 0, 'B': 0, 'C': 0}

for start in windows:
    end = start + PLAZO_ANOS - 1
    rA = simular_opcion_A(start, v8_returns)
    rB = simular_opcion_B(start, v8_returns)
    rC = simular_opcion_C(start, v8_returns)
    rD = simular_opcion_D(start, v8_returns)

    eA = rA[-1]['equity_neto']
    eB = rB[-1]['equity_neto']
    eC = rC[-1]['equity_neto']
    eD = rD[-1]['equity_neto']

    impA = any(r['impago'] for r in rA)
    impB = any(r['impago'] for r in rB)
    impC = any(r['impago'] for r in rC)

    if impA: impagos['A'] += 1
    if impB: impagos['B'] += 1
    if impC: impagos['C'] += 1

    summary['A'].append(eA)
    summary['B'].append(eB)
    summary['C'].append(eC)
    summary['D'].append(eD)

    impA_s = "IMPAGO" if impA else ""
    impB_s = "IMPAGO" if impB else ""
    impC_s = "IMPAGO" if impC else ""

    print(f"  {start:<8} {end:<6} "
          f"EUR {eA:>10,.0f} EUR {eB:>10,.0f} EUR {eC:>10,.0f} EUR {eD:>10,.0f} "
          f"{impA_s:>6} {impB_s:>6} {impC_s:>6}")

# Estadisticas
print(f"\n{'=' * 100}")
print("ESTADISTICAS RESUMEN (equity neto final tras 7 anos)")
print(f"{'=' * 100}")

print(f"\n  {'Metrica':<30} {'A) 100% inv':>14} {'B) Colchon':>14} {'C) Acelerada':>14} {'D) Solo 10K':>14}")
print(f"  {'-' * 90}")

for label, key in [('Media', 'mean'), ('Mediana', 'median'),
                    ('Mejor caso', 'max'), ('Peor caso', 'min'),
                    ('Desv. estandar', 'std')]:
    vals = []
    for opt in ['A', 'B', 'C', 'D']:
        arr = np.array(summary[opt])
        if key == 'mean': v = arr.mean()
        elif key == 'median': v = np.median(arr)
        elif key == 'max': v = arr.max()
        elif key == 'min': v = arr.min()
        elif key == 'std': v = arr.std()
        vals.append(v)
    print(f"  {label:<30} EUR {vals[0]:>10,.0f} EUR {vals[1]:>10,.0f} EUR {vals[2]:>10,.0f} EUR {vals[3]:>10,.0f}")

print(f"\n  {'Casos de impago':<30} {impagos['A']:>14} {impagos['B']:>14} {impagos['C']:>14} {'N/A':>14}")
print(f"  {'% impago':<30} {impagos['A']/len(windows)*100:>13.0f}% {impagos['B']/len(windows)*100:>13.0f}% {impagos['C']/len(windows)*100:>13.0f}% {'N/A':>14}")

# Multiplicador vs solo equity
print(f"\n  {'Multiplicador medio vs D':<30}", end="")
for opt in ['A', 'B', 'C']:
    mult = np.mean(summary[opt]) / np.mean(summary['D'])
    print(f" {mult:>14.1f}x", end="")
print(f" {'1.0x':>14}")

# CAGR medio
print(f"\n  {'CAGR medio 7 anos':<30}", end="")
for opt in ['A', 'B', 'C', 'D']:
    arr = np.array(summary[opt])
    # CAGR desde 10,000 de equity propio
    cagrs = [(max(v, 0) / EQUITY_INICIAL) ** (1/PLAZO_ANOS) - 1 if v > 0 else -1.0 for v in arr]
    mean_cagr = np.mean([c for c in cagrs if c > -1])
    print(f" {mean_cagr:>13.1%}", end="")
print()

# ROE (return on equity)
print(f"  {'ROE medio (sobre 10K)':<30}", end="")
for opt in ['A', 'B', 'C', 'D']:
    arr = np.array(summary[opt])
    roes = [(v - EQUITY_INICIAL) / EQUITY_INICIAL for v in arr]
    print(f" {np.mean(roes):>13.0%}", end="")
print()


# =============================================================================
# 5. Detalle de escenarios clave
# =============================================================================

def print_detalle(data, titulo, opcion_label):
    """Imprime detalle ano a ano de una simulacion."""
    print(f"\n  --- {titulo} ---")
    if opcion_label == 'C':
        print(f"  {'Ano':<6} {'Ret v8':>8} {'Tipo':>6} {'Portfolio':>12} {'Deuda':>12} "
              f"{'Cuota':>10} {'Amort+':>10} {'Equity':>12}")
        print(f"  {'-' * 82}")
        for r in data:
            ae = r.get('amort_extra', 0)
            print(f"  {r['year']:<6} {r['return']:>7.1%} {r['rate']:>5.1%} "
                  f"EUR {r['portfolio']:>9,.0f} EUR {r['deuda']:>9,.0f} "
                  f"EUR {r['cuota']:>7,.0f} EUR {ae:>7,.0f} EUR {r['equity_neto']:>9,.0f}"
                  f"{'  *** IMPAGO' if r['impago'] else ''}")
    elif opcion_label == 'B':
        print(f"  {'Ano':<6} {'Ret v8':>8} {'Tipo':>6} {'Portfolio':>12} {'Cash':>10} {'Deuda':>12} "
              f"{'Cuota':>10} {'Equity':>12}")
        print(f"  {'-' * 82}")
        for r in data:
            print(f"  {r['year']:<6} {r['return']:>7.1%} {r['rate']:>5.1%} "
                  f"EUR {r['portfolio']:>9,.0f} EUR {r.get('cash',0):>7,.0f} EUR {r['deuda']:>9,.0f} "
                  f"EUR {r['cuota']:>7,.0f} EUR {r['equity_neto']:>9,.0f}"
                  f"{'  *** IMPAGO' if r['impago'] else ''}")
    else:
        print(f"  {'Ano':<6} {'Ret v8':>8} {'Tipo':>6} {'Portfolio':>12} {'Deuda':>12} "
              f"{'Cuota':>10} {'Intereses':>10} {'Equity':>12}")
        print(f"  {'-' * 82}")
        for r in data:
            print(f"  {r['year']:<6} {r['return']:>7.1%} {r.get('rate',0):>5.1%} "
                  f"EUR {r['portfolio']:>9,.0f} EUR {r.get('deuda',0):>9,.0f} "
                  f"EUR {r.get('cuota',0):>7,.0f} EUR {r.get('intereses',0):>7,.0f} EUR {r['equity_neto']:>9,.0f}"
                  f"{'  *** IMPAGO' if r.get('impago') else ''}")


# Encontrar mejor, peor, y caso "tipico" (mediana)
median_idx = np.argsort(summary['A'])[len(summary['A'])//2]
best_idx = np.argmax(summary['A'])
worst_idx = np.argmin(summary['A'])

scenarios = [
    (windows[best_idx], "MEJOR CASO"),
    (windows[median_idx], "CASO MEDIANO"),
    (windows[worst_idx], "PEOR CASO"),
]

for start, label in scenarios:
    end = start + PLAZO_ANOS - 1
    print(f"\n{'=' * 100}")
    print(f"DETALLE {label}: {start}-{end}")
    print(f"{'=' * 100}")

    rA = simular_opcion_A(start, v8_returns)
    rB = simular_opcion_B(start, v8_returns)
    rC = simular_opcion_C(start, v8_returns)
    rD = simular_opcion_D(start, v8_returns)

    print_detalle(rA, "A) 100% invertido, amortizacion francesa", 'A')
    print_detalle(rB, "B) Colchon 2 anos de cuotas", 'B')
    print_detalle(rC, "C) Amortizacion acelerada (minimo 20K en cartera)", 'C')

    print(f"\n  --- D) Solo equity 10K (sin deuda) ---")
    print(f"  Equity final: EUR {rD[-1]['equity_neto']:,.0f}")

    print(f"\n  COMPARATIVA FINAL {label} ({start}-{end}):")
    print(f"  {'Opcion':<35} {'Equity final':>14} {'vs Solo equity':>16}")
    print(f"  {'-' * 68}")
    for name, data, key in [
        ("A) 100% inv, amort. francesa", rA, 'A'),
        ("B) Colchon 2 anos", rB, 'B'),
        ("C) Amort. acelerada", rC, 'C'),
        ("D) Solo 10K equity", rD, 'D'),
    ]:
        eq = data[-1]['equity_neto']
        mult = eq / rD[-1]['equity_neto']
        print(f"  {name:<35} EUR {eq:>10,.0f} {mult:>15.1f}x")


# =============================================================================
# 6. Analisis de riesgo
# =============================================================================
print(f"\n{'=' * 100}")
print("ANALISIS DE RIESGO")
print(f"{'=' * 100}")

# Para cada opcion, contar cuantas veces equity neto < 0 en algun momento
def check_underwater(sim_func, returns):
    """Cuenta ventanas donde equity neto fue negativo en algun momento."""
    underwater = 0
    for start in windows:
        data = sim_func(start, returns)
        if any(r['equity_neto'] < 0 for r in data):
            underwater += 1
    return underwater

uw_A = check_underwater(simular_opcion_A, v8_returns)
uw_B = check_underwater(simular_opcion_B, v8_returns)
uw_C = check_underwater(simular_opcion_C, v8_returns)

print(f"\n  {'Metrica':<45} {'A) 100%':>10} {'B) Colch':>10} {'C) Acel':>10}")
print(f"  {'-' * 78}")
print(f"  {'Ventanas con impago':.<45} {impagos['A']:>10} {impagos['B']:>10} {impagos['C']:>10}")
print(f"  {'Ventanas con equity neto < 0 (algun ano)':.<45} {uw_A:>10} {uw_B:>10} {uw_C:>10}")
print(f"  {'% equity negativo':.<45} {uw_A/len(windows)*100:>9.0f}% {uw_B/len(windows)*100:>9.0f}% {uw_C/len(windows)*100:>9.0f}%")

# Intereses totales pagados
print(f"\n  Intereses totales pagados (media todas las ventanas):")
for opt_name, sim_func in [("A) 100% invertido", simular_opcion_A),
                             ("B) Colchon", simular_opcion_B),
                             ("C) Acelerada", simular_opcion_C)]:
    total_int = []
    for start in windows:
        data = sim_func(start, v8_returns)
        ti = sum(r.get('intereses', 0) for r in data)
        total_int.append(ti)
    print(f"    {opt_name:<30} EUR {np.mean(total_int):>8,.0f} (media) | EUR {min(total_int):>8,.0f} - {max(total_int):>8,.0f}")

# Deuda pendiente tras 7 anos (si no ha pagado todo)
print(f"\n  Deuda pendiente al final (deberia ser 0 si no hay impago):")
for opt_name, sim_func in [("A) 100% invertido", simular_opcion_A),
                             ("B) Colchon", simular_opcion_B),
                             ("C) Acelerada", simular_opcion_C)]:
    deudas = []
    for start in windows:
        data = sim_func(start, v8_returns)
        deudas.append(data[-1]['deuda'])
    no_zero = [d for d in deudas if d > 0]
    if no_zero:
        print(f"    {opt_name:<30} {len(no_zero)} ventanas con deuda pendiente (max EUR {max(no_zero):,.0f})")
    else:
        print(f"    {opt_name:<30} Todas las ventanas: deuda liquidada")

# Ano en que se liquida la deuda (opcion C)
print(f"\n  Opcion C - Ano en que se liquida la deuda:")
for start in windows:
    data = simular_opcion_C(start, v8_returns)
    for r in data:
        if r['deuda'] <= 0:
            anos_para_pagar = r['year'] - start + 1
            print(f"    Inicio {start}: deuda liquidada en {anos_para_pagar} anos (ano {r['year']})")
            break
    else:
        if data[-1]['impago']:
            print(f"    Inicio {start}: IMPAGO")
        else:
            print(f"    Inicio {start}: deuda pendiente EUR {data[-1]['deuda']:,.0f}")


# =============================================================================
# 7. Conclusiones
# =============================================================================
print(f"\n{'=' * 100}")
print("CONCLUSIONES")
print(f"{'=' * 100}")

mean_A = np.mean(summary['A'])
mean_B = np.mean(summary['B'])
mean_C = np.mean(summary['C'])
mean_D = np.mean(summary['D'])

print(f"""
  CAPITAL PROPIO: EUR 10,000 | PRESTAMO: EUR 70,000 | PLAZO: 7 anos

  Equity neto medio tras 7 anos:
    A) 100% invertido:     EUR {mean_A:>10,.0f}  ({mean_A/EQUITY_INICIAL:.0f}x capital propio)
    B) Colchon 2 anos:     EUR {mean_B:>10,.0f}  ({mean_B/EQUITY_INICIAL:.0f}x capital propio)
    C) Amort. acelerada:   EUR {mean_C:>10,.0f}  ({mean_C/EQUITY_INICIAL:.0f}x capital propio)
    D) Solo equity:        EUR {mean_D:>10,.0f}  ({mean_D/EQUITY_INICIAL:.0f}x capital propio)

  Multiplicador medio del apalancamiento:
    A vs D: {mean_A/mean_D:.1f}x  |  B vs D: {mean_B/mean_D:.1f}x  |  C vs D: {mean_C/mean_D:.1f}x

  Riesgo:
    Impagos: A={impagos['A']}/{len(windows)}, B={impagos['B']}/{len(windows)}, C={impagos['C']}/{len(windows)}
    Equity negativo (algun ano): A={uw_A}/{len(windows)}, B={uw_B}/{len(windows)}, C={uw_C}/{len(windows)}
""")
