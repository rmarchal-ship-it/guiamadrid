#!/usr/bin/env python3
"""
Comparativa: 60/40 TQQQ/Gold (Baseline) vs Momentum Breakout v7/v7+

Compara las dos estrategias en el máximo periodo comparable.
Datos de v7+ obtenidos ejecutando backtest_experimental.py directamente.

Periodos de comparación:
- 36 meses (mar 2023 - feb 2026): Baseline vs v7+
- 240 meses (20 años): Baseline vs v7+ (métricas reales del backtest)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Cargar datos para Baseline 60/40 TQQQ/Gold
# =============================================================================
print("=" * 80)
print("COMPARATIVA: Baseline 60/40 TQQQ/Gold  vs  Momentum Breakout v7/v7+")
print("=" * 80)

print("\nCargando datos...")

qqq = yf.download("QQQ", start="1999-03-01", end="2026-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2026-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2026-12-31", auto_adjust=True, progress=False)

for df in [qqq, gld, tqqq_real]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

# Calibrar coste diario TQQQ
qqq_ret = qqq["Close"].pct_change()
ov = tqqq_real.index.intersection(qqq_ret.index)
real_total = tqqq_real["Close"].loc[ov].iloc[-1] / tqqq_real["Close"].loc[ov].iloc[0]

def sim_err(dc):
    sr = qqq_ret.loc[ov] * 3 - dc
    st = (1 + sr).cumprod()
    return (np.log(st.iloc[-1] / st.iloc[0]) - np.log(real_total)) ** 2

daily_cost = minimize_scalar(sim_err, bounds=(0, 0.001), method='bounded').x

# TQQQ simulado desde QQQ
tqqq_sim_ret = qqq_ret * 3 - daily_cost
tqqq_sim = (1 + tqqq_sim_ret).cumprod()
tqqq_sim.iloc[0] = 1.0

# Periodo común TQQQ sim + GLD
common_all = tqqq_sim.index.intersection(gld.index).sort_values()
tqqq_p = tqqq_sim.loc[common_all] / tqqq_sim.loc[common_all].iloc[0]
gold_p = gld["Close"].loc[common_all] / gld["Close"].loc[common_all].iloc[0]
tqqq_d = tqqq_p.pct_change().fillna(0)
gold_d = gold_p.pct_change().fillna(0)

# =============================================================================
# 2. Funciones
# =============================================================================
def metrics(series, label=""):
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    total = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    daily = series.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    rm = series.cummax()
    dd = (series - rm) / rm
    max_dd = dd.min()
    max_dd_date = dd.idxmin()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        "label": label, "final": series.iloc[-1], "total_ret": total,
        "cagr": cagr, "vol": vol, "sharpe": sharpe,
        "max_dd": max_dd, "max_dd_date": max_dd_date, "calmar": calmar,
        "years": yrs,
    }


def run_baseline(dates, tqqq_ret, gold_ret, initial=10000):
    W = 0.60
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g
        if dates[i].year != dates[i-1].year:
            cur_t = total * W
            cur_g = total * (1 - W)
        values.append(total)
    return pd.Series(values, index=dates)


def run_voltarget(dates, tqqq_ret, gold_ret, initial=10000,
                  target_vol=0.20, lookback=63, max_w=0.80, min_w=0.10):
    W = 0.60
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g
        d, pd_ = dates[i], dates[i-1]
        check = d.month != pd_.month
        if check and i >= lookback:
            recent_vol = tqqq_ret.iloc[i-lookback:i].std() * np.sqrt(252)
            if recent_vol > 0:
                ideal_w = target_vol / recent_vol
                W = np.clip(ideal_w, min_w, max_w)
            cur_t = total * W
            cur_g = total * (1 - W)
        elif d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)
        values.append(total)
    return pd.Series(values, index=dates)


# =============================================================================
# 3. Reconstruir equity curve v7 base desde CSV de trades (36m)
# =============================================================================
print("Reconstruyendo equity curve v7 base desde historico_trades_36m.csv...")

import os
base_path = os.path.dirname(os.path.abspath(__file__))
trades_path = os.path.join(base_path, "historico_trades_36m.csv")
trades_df = pd.read_csv(trades_path, parse_dates=['entry_date', 'exit_date'])

trade_events = [(t['exit_date'], t['pnl_eur']) for _, t in trades_df.iterrows()]
trade_events.sort(key=lambda x: x[0])

running_equity = 10000
equity_points = [(trades_df['entry_date'].min(), 10000)]
for date, pnl in trade_events:
    running_equity += pnl
    equity_points.append((date, running_equity))

v7_equity = pd.Series([p[1] for p in equity_points],
                       index=pd.DatetimeIndex([p[0] for p in equity_points]))
v7_equity = v7_equity.groupby(v7_equity.index).last()
v7_daily = v7_equity.resample('B').ffill()

# =============================================================================
# 4. COMPARACIÓN A 36 MESES
# =============================================================================
print("\n" + "=" * 80)
print("COMPARACIÓN A 36 MESES  (marzo 2023 — febrero 2026)")
print("=" * 80)

v7_start = v7_daily.index[0]
v7_end = v7_daily.index[-1]
mask_36m = (common_all >= v7_start) & (common_all <= v7_end)
dates_36m = common_all[mask_36m]

baseline_36m = run_baseline(dates_36m, tqqq_d.loc[dates_36m], gold_d.loc[dates_36m])
vt20_36m = run_voltarget(dates_36m, tqqq_d.loc[dates_36m], gold_d.loc[dates_36m], target_vol=0.20)
vt25_36m = run_voltarget(dates_36m, tqqq_d.loc[dates_36m], gold_d.loc[dates_36m], target_vol=0.25)

m_base_36 = metrics(baseline_36m)
m_vt20_36 = metrics(vt20_36m)
m_vt25_36 = metrics(vt25_36m)
m_v7_36 = metrics(v7_daily)

# Datos REALES de v7+ a 36m (de backtest_experimental.py --months 36 --test b)
v7p_36_cagr = 0.583
v7p_36_total = 2.966       # +296.6%
v7p_36_maxdd = -0.425
v7p_36_pf = 2.27
v7p_36_final = 39656.81
v7p_36_winrate = 0.373
v7p_36_trades = 161

print(f"\nPeriodo: {dates_36m[0].strftime('%Y-%m-%d')} a {dates_36m[-1].strftime('%Y-%m-%d')}")
print(f"\n{'Métrica':<18} {'Baseline 60/40':>15} {'VolTgt 20%':>15} {'VolTgt 25%':>15} {'v7 base':>15} {'v7+ real':>15}")
print("-" * 95)
print(f"{'CAGR':<18} {m_base_36['cagr']:>14.1%} {m_vt20_36['cagr']:>14.1%} {m_vt25_36['cagr']:>14.1%} {m_v7_36['cagr']:>14.1%} {v7p_36_cagr:>14.1%}")
print(f"{'Total Return':<18} {m_base_36['total_ret']:>14.1%} {m_vt20_36['total_ret']:>14.1%} {m_vt25_36['total_ret']:>14.1%} {m_v7_36['total_ret']:>14.1%} {v7p_36_total:>14.1%}")
print(f"{'Volatility':<18} {m_base_36['vol']:>14.1%} {m_vt20_36['vol']:>14.1%} {m_vt25_36['vol']:>14.1%} {m_v7_36['vol']:>14.1%} {'—':>15}")
print(f"{'Sharpe':<18} {m_base_36['sharpe']:>14.2f} {m_vt20_36['sharpe']:>14.2f} {m_vt25_36['sharpe']:>14.2f} {m_v7_36['sharpe']:>14.2f} {'—':>15}")
print(f"{'Max Drawdown':<18} {m_base_36['max_dd']:>14.1%} {m_vt20_36['max_dd']:>14.1%} {m_vt25_36['max_dd']:>14.1%} {m_v7_36['max_dd']:>14.1%} {v7p_36_maxdd:>14.1%}")
print(f"{'Calmar':<18} {m_base_36['calmar']:>14.2f} {m_vt20_36['calmar']:>14.2f} {m_vt25_36['calmar']:>14.2f} {m_v7_36['calmar']:>14.2f} {v7p_36_cagr/abs(v7p_36_maxdd):>14.2f}")
print(f"{'Final (€10k)':<18} {'$':>1}{m_base_36['final']:>13,.0f} {'$':>1}{m_vt20_36['final']:>13,.0f} {'$':>1}{m_vt25_36['final']:>13,.0f} {'€':>1}{m_v7_36['final']:>13,.0f} {'€':>1}{v7p_36_final:>13,.0f}")
print(f"{'Profit Factor':<18} {'—':>15} {'—':>15} {'—':>15} {'—':>15} {v7p_36_pf:>14.2f}")
print(f"{'Win Rate':<18} {'—':>15} {'—':>15} {'—':>15} {'—':>15} {v7p_36_winrate:>14.1%}")
print(f"{'Trades':<18} {'—':>15} {'—':>15} {'—':>15} {len(trades_df):>15} {v7p_36_trades:>15}")

# =============================================================================
# 5. AÑO A AÑO (36 meses)
# =============================================================================
print("\n" + "=" * 80)
print("AÑO A AÑO (periodo 36 meses)")
print("=" * 80)

strategies_36m = {
    'Baseline': baseline_36m,
    'VT20%': vt20_36m,
    'VT25%': vt25_36m,
    'v7 base': v7_daily,
}

print(f"\n{'Año':<7}", end="")
for s in strategies_36m:
    print(f" {s:>12}", end="")
print(f" {'Ganador':>12}")
print("-" * 65)

for year in sorted(set(d.year for d in dates_36m)):
    yd_base = [d for d in dates_36m if d.year == year]
    yd_v7 = [d for d in v7_daily.index if d.year == year]
    if len(yd_base) < 2:
        continue
    print(f"{year:<7}", end="")
    rets = {}
    for name, series in strategies_36m.items():
        yd = yd_v7 if name == 'v7 base' else yd_base
        if len(yd) >= 2:
            f, l = yd[0], yd[-1]
            if f in series.index and l in series.index:
                r = series.loc[l] / series.loc[f] - 1
                rets[name] = r
                print(f" {r:>11.1%}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        else:
            print(f" {'N/A':>12}", end="")
    if rets:
        print(f" {max(rets, key=rets.get):>12}")
    else:
        print()

# =============================================================================
# 6. COMPARACIÓN A 240 MESES (~20 AÑOS)
# =============================================================================
print("\n" + "=" * 80)
print("COMPARACIÓN A 240 MESES (~20 AÑOS)")
print("Datos REALES de backtest_experimental.py --months 240 --test b")
print("=" * 80)

# Baseline 20 años
start_20y = pd.Timestamp("2006-02-11")
end_20y = pd.Timestamp("2026-02-11")
mask_20y = (common_all >= start_20y) & (common_all <= end_20y)
dates_20y = common_all[mask_20y]

baseline_20y = run_baseline(dates_20y, tqqq_d.loc[dates_20y], gold_d.loc[dates_20y])
vt20_20y = run_voltarget(dates_20y, tqqq_d.loc[dates_20y], gold_d.loc[dates_20y], target_vol=0.20)
vt25_20y = run_voltarget(dates_20y, tqqq_d.loc[dates_20y], gold_d.loc[dates_20y], target_vol=0.25)

m_base_20y = metrics(baseline_20y)
m_vt20_20y = metrics(vt20_20y)
m_vt25_20y = metrics(vt25_20y)

yrs_20y = (dates_20y[-1] - dates_20y[0]).days / 365.25

# Datos REALES de v7+ a 240m (de backtest_experimental.py --months 240 --test b)
v7p_240_cagr = 0.172
v7p_240_total = 23.109      # +2310.9%
v7p_240_maxdd = -0.599
v7p_240_pf = 1.76
v7p_240_final = 241091.38
v7p_240_winrate = 0.316
v7p_240_trades = 942

print(f"\nPeriodo: {dates_20y[0].strftime('%Y-%m-%d')} a {dates_20y[-1].strftime('%Y-%m-%d')} ({yrs_20y:.1f} años)")
print(f"\n{'Métrica':<18} {'Baseline 60/40':>15} {'VolTgt 20%':>15} {'VolTgt 25%':>15} {'v7+ real':>15}")
print("-" * 80)
print(f"{'CAGR':<18} {m_base_20y['cagr']:>14.1%} {m_vt20_20y['cagr']:>14.1%} {m_vt25_20y['cagr']:>14.1%} {v7p_240_cagr:>14.1%}")
print(f"{'Total Return':<18} {m_base_20y['total_ret']:>13.0%}% {m_vt20_20y['total_ret']:>13.0%}% {m_vt25_20y['total_ret']:>13.0%}% {v7p_240_total:>13.0%}%")
print(f"{'Volatility':<18} {m_base_20y['vol']:>14.1%} {m_vt20_20y['vol']:>14.1%} {m_vt25_20y['vol']:>14.1%} {'—':>15}")
print(f"{'Sharpe':<18} {m_base_20y['sharpe']:>14.2f} {m_vt20_20y['sharpe']:>14.2f} {m_vt25_20y['sharpe']:>14.2f} {'—':>15}")
print(f"{'Max Drawdown':<18} {m_base_20y['max_dd']:>14.1%} {m_vt20_20y['max_dd']:>14.1%} {m_vt25_20y['max_dd']:>14.1%} {v7p_240_maxdd:>14.1%}")
print(f"{'Calmar':<18} {m_base_20y['calmar']:>14.2f} {m_vt20_20y['calmar']:>14.2f} {m_vt25_20y['calmar']:>14.2f} {v7p_240_cagr/abs(v7p_240_maxdd):>14.2f}")
print(f"{'Final (€10k)':<18} {'$':>1}{m_base_20y['final']:>13,.0f} {'$':>1}{m_vt20_20y['final']:>13,.0f} {'$':>1}{m_vt25_20y['final']:>13,.0f} {'€':>1}{v7p_240_final:>13,.0f}")
print(f"{'Profit Factor':<18} {'—':>15} {'—':>15} {'—':>15} {v7p_240_pf:>14.2f}")
print(f"{'Trades':<18} {'—':>15} {'—':>15} {'—':>15} {v7p_240_trades:>15}")

# =============================================================================
# 7. TABLA RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("TABLA RESUMEN FINAL")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  36 MESES (marzo 2023 — febrero 2026)                                 ║
╠══════════════════╦═════════╦═════════╦═════════╦══════════╦════════════╣
║ Estrategia       ║  CAGR   ║ Max DD  ║ Sharpe  ║ Final€10k║ Complejidad║
╠══════════════════╬═════════╬═════════╬═════════╬══════════╬════════════╣
║ Baseline 60/40   ║ {m_base_36['cagr']:>6.1%} ║{m_base_36['max_dd']:>7.1%}  ║  {m_base_36['sharpe']:>5.2f}  ║ ${m_base_36['final']:>7,.0f}║  Ninguna   ║
║ VolTarget 20%    ║ {m_vt20_36['cagr']:>6.1%} ║{m_vt20_36['max_dd']:>7.1%}  ║  {m_vt20_36['sharpe']:>5.2f}  ║ ${m_vt20_36['final']:>7,.0f}║  Baja      ║
║ VolTarget 25%    ║ {m_vt25_36['cagr']:>6.1%} ║{m_vt25_36['max_dd']:>7.1%}  ║  {m_vt25_36['sharpe']:>5.2f}  ║ ${m_vt25_36['final']:>7,.0f}║  Baja      ║
║ v7 base          ║ {m_v7_36['cagr']:>6.1%} ║{m_v7_36['max_dd']:>7.1%}  ║  {m_v7_36['sharpe']:>5.2f}  ║ €{m_v7_36['final']:>7,.0f}║  Alta      ║
║ v7+ (opciones)   ║ {v7p_36_cagr:>6.1%} ║{v7p_36_maxdd:>7.1%}  ║    —    ║ €{v7p_36_final:>7,.0f}║  Muy alta  ║
╚══════════════════╩═════════╩═════════╩═════════╩══════════╩════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║  20 AÑOS (~240 meses)                                                  ║
╠══════════════════╦═════════╦═════════╦═════════╦══════════════╦════════╣
║ Estrategia       ║  CAGR   ║ Max DD  ║ Sharpe  ║  Final (€10k)║Calmar ║
╠══════════════════╬═════════╬═════════╬═════════╬══════════════╬════════╣
║ Baseline 60/40   ║ {m_base_20y['cagr']:>6.1%} ║{m_base_20y['max_dd']:>7.1%}  ║  {m_base_20y['sharpe']:>5.2f}  ║ ${m_base_20y['final']:>11,.0f}║ {m_base_20y['calmar']:>5.2f}║
║ VolTarget 20%    ║ {m_vt20_20y['cagr']:>6.1%} ║{m_vt20_20y['max_dd']:>7.1%}  ║  {m_vt20_20y['sharpe']:>5.2f}  ║ ${m_vt20_20y['final']:>11,.0f}║ {m_vt20_20y['calmar']:>5.2f}║
║ VolTarget 25%    ║ {m_vt25_20y['cagr']:>6.1%} ║{m_vt25_20y['max_dd']:>7.1%}  ║  {m_vt25_20y['sharpe']:>5.2f}  ║ ${m_vt25_20y['final']:>11,.0f}║ {m_vt25_20y['calmar']:>5.2f}║
║ v7+ (opciones)   ║ {v7p_240_cagr:>6.1%} ║{v7p_240_maxdd:>7.1%}  ║    —    ║ €{v7p_240_final:>11,.0f}║ {v7p_240_cagr/abs(v7p_240_maxdd):>5.2f}║
╚══════════════════╩═════════╩═════════╩═════════╩══════════════╩════════╝
""")

# =============================================================================
# 8. CONCLUSIONES
# =============================================================================
print("=" * 80)
print("CONCLUSIONES")
print("=" * 80)

print(f"""
1. A 36 MESES (bull market reciente):
   - Baseline 60/40 aplasta a todo: {m_base_36['cagr']:.1%} CAGR, ${m_base_36['final']:,.0f}
   - v7+ es comparable en retorno ({v7p_36_cagr:.1%}) pero con drawdown peor (-42.5% vs -34.0%)
   - VolTarget 20% mejor Sharpe ({m_vt20_36['sharpe']:.2f}) y mejor drawdown ({m_vt20_36['max_dd']:.1%})
   - v7 base (solo acciones, sin opciones) rinde la mitad ({m_v7_36['cagr']:.1%})

2. A 20 AÑOS (incluye crisis 2008, COVID, bear 2022):
   - Baseline 60/40: {m_base_20y['cagr']:.1%} CAGR pero drawdown brutal ({m_base_20y['max_dd']:.1%})
   - VolTarget 20%: {m_vt20_20y['cagr']:.1%} CAGR, drawdown controlado ({m_vt20_20y['max_dd']:.1%}), mejor Sharpe ({m_vt20_20y['sharpe']:.2f})
   - v7+: {v7p_240_cagr:.1%} CAGR con drawdown {v7p_240_maxdd:.1%} — peor que VolTarget en ambas métricas
   - v7+ tiene MENOS retorno y MAS drawdown que VolTarget 20%

3. NOTA SOBRE v7+ A 240 MESES:
   - Los datos del RESUMEN decían: +31.3% anual, -40.0% DD, PF 2.36
   - Ejecutando HOY: +17.2% anual, -59.9% DD, PF 1.76
   - Posible causa: datos de yfinance actualizados, splits, ajustes retroactivos
   - Los resultados del RESUMEN pueden estar desactualizados

4. GANADOR POR CATEGORÍA:
   - Retorno absoluto 20 años: Baseline 60/40 ({m_base_20y['cagr']:.1%})
   - Retorno ajustado riesgo: VolTarget 20% (Sharpe {m_vt20_20y['sharpe']:.2f})
   - Menor drawdown: VolTarget 20% ({m_vt20_20y['max_dd']:.1%})
   - Complejidad mínima: Baseline 60/40

5. VEREDICTO:
   La familia TQQQ/Gold (especialmente con VolTarget 20%) domina a v7+
   tanto en retorno como en riesgo a 20 años. v7+ aporta diversificación
   pero no mejora las métricas.

   RANKING:
   1. Baseline 60/40:  {m_base_20y['cagr']:.1%} CAGR, {m_base_20y['max_dd']:.1%} DD → máximo retorno, sin esfuerzo
   2. VolTarget 20%:   {m_vt20_20y['cagr']:.1%} CAGR, {m_vt20_20y['max_dd']:.1%} DD → mejor ajustado al riesgo
   3. VolTarget 25%:   {m_vt25_20y['cagr']:.1%} CAGR, {m_vt25_20y['max_dd']:.1%} DD → equilibrio
   4. v7+ Momentum:    {v7p_240_cagr:.1%} CAGR, {v7p_240_maxdd:.1%} DD → peor y más compleja
""")
