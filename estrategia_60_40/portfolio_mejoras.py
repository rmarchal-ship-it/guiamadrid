#!/usr/bin/env python3
"""
Portfolio Mejoras: 4 nuevas estrategias para mejorar 60% TQQQ / 40% Gold
=========================================================================

Objetivo: reducir MaxDD del -63% a niveles tolerables (-35% a -45%)
          SIN destruir el CAGR del ~28% a 240 meses.

Estrategias:
  1. SMA200 + Vol Target Asimétrico ("Trend-Gated Protection")
  2. Rebalanceo por Bandas Asimétricas + Momentum
  3. Risk Parity Dinámico 3 Activos (TQQQ/QQQ/Gold)
  4. Regime Score Continuo ("Composite Dimmer")

Referencia: portfolio_protection.py, portfolio_voltarget.py, portfolio_triple.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
import io
from scipy.optimize import minimize_scalar
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
print("=" * 80)
print("CARGANDO DATOS...")
print("=" * 80)

ndx = yf.download("^NDX", start="1985-01-01", end="2026-12-31", auto_adjust=True, progress=False)
qqq = yf.download("QQQ", start="1999-03-01", end="2026-12-31", auto_adjust=True, progress=False)
gld = yf.download("GLD", start="2004-11-01", end="2026-12-31", auto_adjust=True, progress=False)
tqqq_real = yf.download("TQQQ", start="2010-02-10", end="2026-12-31", auto_adjust=True, progress=False)

for df in [ndx, qqq, gld, tqqq_real]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

# Gold composite (monthly spot pre-2004, GLD post-2004)
url = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req, timeout=30)
gold_monthly = pd.read_csv(io.StringIO(resp.read().decode('utf-8')), parse_dates=['Date'])
gold_monthly = gold_monthly.set_index('Date').sort_index()

gold_daily_interp = gold_monthly['Price'].resample('B').interpolate(method='linear')
gld_start = gld.index[0]
gc_scale = gld["Close"].iloc[0] / gold_daily_interp.loc[:gld_start].iloc[-1]
gold_pre = gold_daily_interp.loc[:gld_start] * gc_scale
gold_composite = pd.concat([gold_pre.iloc[:-1], gld["Close"].loc[gld_start:]])
gold_composite = gold_composite[~gold_composite.index.duplicated(keep='last')].sort_index()

# Calibrar TQQQ simulado vs real
qqq_ret = qqq["Close"].pct_change()
ov = tqqq_real.index.intersection(qqq_ret.index)
real_total = tqqq_real["Close"].loc[ov].iloc[-1] / tqqq_real["Close"].loc[ov].iloc[0]

def sim_err(dc):
    sr = qqq_ret.loc[ov] * 3 - dc
    st = (1 + sr).cumprod()
    return (np.log(st.iloc[-1] / st.iloc[0]) - np.log(real_total)) ** 2

daily_cost = minimize_scalar(sim_err, bounds=(0, 0.001), method='bounded').x
print(f"  Fricción diaria calibrada: {daily_cost:.6f} ({daily_cost*252:.2%} anual)")

# Build TQQQ simulado desde NDX+QQQ
ndx_ret = ndx["Close"].pct_change()
qqq_start = qqq.index[0]
nasdaq_ret = pd.concat([ndx_ret.loc[:qqq_start].iloc[:-1], qqq_ret.loc[qqq_start:]])
nasdaq_ret = nasdaq_ret[~nasdaq_ret.index.duplicated(keep='last')].sort_index()

tqqq_sim_ret = nasdaq_ret * 3 - daily_cost
tqqq_sim = (1 + tqqq_sim_ret).cumprod()
tqqq_sim.iloc[0] = 1.0

# Nasdaq price (para señales SMA, momentum)
ndx_price = ndx["Close"].copy()
qqq_price = qqq["Close"].copy()
scale_factor = qqq_price.iloc[0] / ndx_price.loc[:qqq_start].iloc[-1]
ndx_pre = ndx_price.loc[:qqq_start] * scale_factor
nasdaq_price = pd.concat([ndx_pre.iloc[:-1], qqq_price.loc[qqq_start:]])
nasdaq_price = nasdaq_price[~nasdaq_price.index.duplicated(keep='last')].sort_index()

# Alinear todas las series
common = tqqq_sim.index.intersection(gold_composite.index).intersection(nasdaq_price.index)
common = common.sort_values()

tqqq_p = tqqq_sim.loc[common] / tqqq_sim.loc[common].iloc[0]
gold_p = gold_composite.loc[common] / gold_composite.loc[common].iloc[0]
nasdaq_px = nasdaq_price.loc[common]

# QQQ normalizado (para Risk Parity)
qqq_p = qqq_price.loc[qqq_price.index.intersection(common)]
qqq_p = qqq_p / qqq_p.iloc[0]

tqqq_d = tqqq_p.pct_change().fillna(0)
gold_d = gold_p.pct_change().fillna(0)
qqq_d_ret = qqq_p.pct_change().fillna(0)

print(f"  Datos: {common[0].strftime('%Y-%m-%d')} a {common[-1].strftime('%Y-%m-%d')}")
print(f"  Días: {len(common)}, Años: {(common[-1] - common[0]).days / 365.25:.1f}")
print(f"  QQQ disponible desde: {qqq_p.index[0].strftime('%Y-%m-%d')} ({len(qqq_p)} días)")

# =============================================================================
# 2. MÉTRICAS
# =============================================================================
def metrics(series):
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    if yrs <= 0:
        return None
    total = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/yrs) - 1
    daily = series.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    rm = series.cummax()
    dd = (series - rm) / rm
    max_dd = dd.min()
    max_dd_date = dd.idxmin()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Max underwater days
    underwater = dd < -0.01
    if underwater.any():
        uw_groups = (~underwater).cumsum()
        uw_lengths = underwater.groupby(uw_groups).sum()
        max_underwater = uw_lengths.max()
    else:
        max_underwater = 0

    return {
        "final": series.iloc[-1], "total_ret": total, "cagr": cagr,
        "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "max_dd_date": max_dd_date, "calmar": calmar,
        "max_uw_days": max_underwater, "years": yrs,
    }

# =============================================================================
# 3. ESTRATEGIAS
# =============================================================================

# --- BASELINE 60/40 ---
def strat_baseline(dates, tqqq_ret, gold_ret, w_tqqq=0.60, initial=10000):
    cur_t = initial * w_tqqq
    cur_g = initial * (1 - w_tqqq)
    values = [initial]
    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g
        if dates[i].year != dates[i-1].year:
            cur_t = total * w_tqqq
            cur_g = total * (1 - w_tqqq)
        values.append(total)
    return pd.Series(values, index=dates)


# --- ESTRATEGIA 1: SMA200 + Vol Target Asimétrico ---
# Risk-on (NDX > SMA200): mantener 60/40 FIJO (captura CAGR completo)
# Risk-off (NDX < SMA200): Vol Target 20%, cap 25% TQQQ
def strat_sma_voltarget(dates, tqqq_ret, gold_ret, nasdaq_prices, initial=10000,
                         sma_days=200, w_risk_on=0.60,
                         vt_target=0.20, vt_lookback=63, vt_max=0.25, vt_min=0.05):
    sma = nasdaq_prices.rolling(sma_days).mean()

    W = w_risk_on
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]
    risk_on = True

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        check_monthly = d.month != pd_.month

        if check_monthly and not np.isnan(sma.loc[dates[i]]):
            price = nasdaq_prices.loc[dates[i]]
            sma_val = sma.loc[dates[i]]
            new_risk_on = price > sma_val

            if new_risk_on:
                # RISK ON: 60/40 fijo, sin vol target
                W = w_risk_on
            else:
                # RISK OFF: vol target con cap bajo
                if i >= vt_lookback:
                    recent_vol = tqqq_ret.iloc[i-vt_lookback:i].std() * np.sqrt(252)
                    if recent_vol > 0:
                        W = np.clip(vt_target / recent_vol, vt_min, vt_max)
                    else:
                        W = vt_min
                else:
                    W = vt_min

            if new_risk_on != risk_on:
                # Cambio de régimen: rebalancear
                cur_t = total * W
                cur_g = total * (1 - W)
                risk_on = new_risk_on
            elif not risk_on:
                # En risk-off: rebalancear mensualmente al peso VT
                cur_t = total * W
                cur_g = total * (1 - W)

        # Rebalanceo anual solo en risk-on
        if risk_on and d.year != pd_.year:
            cur_t = total * W
            cur_g = total * (1 - W)

        values.append(total)
    return pd.Series(values, index=dates)


# --- ESTRATEGIA 2: Bandas Asimétricas + Momentum ---
# En vez de rebalanceo anual fijo:
#   TQQQ > 72%: rebalancear a 60% (tomar ganancias)
#   TQQQ < 45%: rebalancear a 60% SOLO si NDX 3m return > 0
#   Dentro de bandas: dejar correr
def strat_bandas_momentum(dates, tqqq_ret, gold_ret, nasdaq_prices, initial=10000,
                           w_target=0.60, upper_band=0.72, lower_band=0.45,
                           mom_lookback=63):
    cur_t = initial * w_target
    cur_g = initial * (1 - w_target)
    values = [initial]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        if total > 0:
            w_actual = cur_t / total
        else:
            w_actual = 0.5

        rebalance = False

        # Banda superior: tomar ganancias
        if w_actual > upper_band:
            rebalance = True

        # Banda inferior: solo si momentum positivo
        elif w_actual < lower_band and i >= mom_lookback:
            ndx_now = nasdaq_prices.iloc[i]
            ndx_prev = nasdaq_prices.iloc[i - mom_lookback]
            if ndx_prev > 0 and (ndx_now / ndx_prev - 1) > 0:
                rebalance = True
            # Si momentum negativo: NO rebalancear, dejar Gold dominar

        if rebalance:
            cur_t = total * w_target
            cur_g = total * (1 - w_target)

        values.append(total)
    return pd.Series(values, index=dates)


# --- ESTRATEGIA 3: Risk Parity Dinámico 3 Activos ---
# Peso = (1/vol_i) / sum(1/vol_j) con caps
# Solo aplica desde que QQQ está disponible
def strat_risk_parity_3(dates, tqqq_ret, gold_ret, qqq_ret, initial=10000,
                         lookback=63,
                         cap_tqqq=0.45, cap_qqq=0.50, min_gold=0.15, max_gold=0.60):
    # Pesos iniciales por defecto
    w_t, w_q, w_g = 0.30, 0.30, 0.40
    cur_t = initial * w_t
    cur_q = initial * w_q
    cur_g = initial * w_g
    values = [initial]

    # Alinear QQQ con las fechas
    qqq_available = qqq_ret.index

    for i in range(1, len(dates)):
        d = dates[i]
        pd_ = dates[i-1]

        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])

        # QQQ: solo si está disponible en esta fecha
        if d in qqq_available:
            cur_q *= (1 + qqq_ret.loc[d])
        # Si no hay QQQ, cur_q no cambia (como cash)

        total = cur_t + cur_q + cur_g

        check_monthly = d.month != pd_.month

        if check_monthly and i >= lookback:
            # Calcular vol de cada activo
            vol_t = tqqq_ret.iloc[i-lookback:i].std() * np.sqrt(252)

            # Vol QQQ
            qqq_slice = qqq_ret.reindex(dates[max(0,i-lookback):i]).dropna()
            vol_q = qqq_slice.std() * np.sqrt(252) if len(qqq_slice) > 20 else vol_t / 3

            vol_g = gold_ret.iloc[i-lookback:i].std() * np.sqrt(252)

            # Risk parity: peso inversamente proporcional a vol
            if vol_t > 0 and vol_q > 0 and vol_g > 0:
                inv_t = 1.0 / vol_t
                inv_q = 1.0 / vol_q
                inv_g = 1.0 / vol_g
                total_inv = inv_t + inv_q + inv_g

                w_t = inv_t / total_inv
                w_q = inv_q / total_inv
                w_g = inv_g / total_inv

                # Aplicar caps
                w_t = min(w_t, cap_tqqq)
                w_q = min(w_q, cap_qqq)
                w_g = max(min(w_g, max_gold), min_gold)

                # Renormalizar
                total_w = w_t + w_q + w_g
                w_t /= total_w
                w_q /= total_w
                w_g /= total_w

            cur_t = total * w_t
            cur_q = total * w_q
            cur_g = total * w_g

        values.append(total)
    return pd.Series(values, index=dates)


# --- ESTRATEGIA 4: Regime Score Continuo ---
# Score 0-1 combinando trend, vol, momentum
# Peso TQQQ = 20% + score * 50% (rango 20%-70%)
def strat_regime_score(dates, tqqq_ret, gold_ret, nasdaq_prices, initial=10000,
                        sma_days=200, vol_lookback=63, mom_lookback=126,
                        w_trend=0.40, w_vol=0.35, w_mom=0.25,
                        w_min=0.20, w_range=0.50,
                        vol_target=0.25):
    sma = nasdaq_prices.rolling(sma_days).mean()

    W = 0.60  # inicial
    cur_t = initial * W
    cur_g = initial * (1 - W)
    values = [initial]

    for i in range(1, len(dates)):
        cur_t *= (1 + tqqq_ret.iloc[i])
        cur_g *= (1 + gold_ret.iloc[i])
        total = cur_t + cur_g

        d, pd_ = dates[i], dates[i-1]
        check_monthly = d.month != pd_.month

        if check_monthly and i >= max(sma_days, vol_lookback, mom_lookback):
            # Componente 1: TREND (NDX vs SMA200)
            price = nasdaq_prices.iloc[i]
            sma_val = sma.iloc[i]
            if not np.isnan(sma_val) and sma_val > 0:
                trend_score = np.clip((price / sma_val - 1) / 0.20, 0, 1)
            else:
                trend_score = 0.5

            # Componente 2: VOLATILITY (inversa de vol reciente)
            recent_vol = tqqq_ret.iloc[i-vol_lookback:i].std() * np.sqrt(252)
            if recent_vol > 0:
                vol_score = np.clip(vol_target / recent_vol, 0, 1)
            else:
                vol_score = 1.0

            # Componente 3: MOMENTUM (NDX 6-month return)
            if i >= mom_lookback:
                ndx_now = nasdaq_prices.iloc[i]
                ndx_prev = nasdaq_prices.iloc[i - mom_lookback]
                if ndx_prev > 0:
                    mom_ret = ndx_now / ndx_prev - 1
                    mom_score = np.clip(mom_ret / 0.30, 0, 1)
                else:
                    mom_score = 0.5
            else:
                mom_score = 0.5

            # Score compuesto
            score = w_trend * trend_score + w_vol * vol_score + w_mom * mom_score

            # Peso TQQQ: rango [w_min, w_min + w_range]
            W = w_min + score * w_range

            cur_t = total * W
            cur_g = total * (1 - W)

        values.append(total)
    return pd.Series(values, index=dates)


# =============================================================================
# 4. EJECUTAR BACKTESTS
# =============================================================================
print("\n" + "=" * 80)
print("EJECUTANDO BACKTESTS...")
print("=" * 80)

# Periodos de análisis
end_date = common[-1]
periods = {
    '36m': 36,
    '60m': 60,
    '240m': 240,
}

nasdaq_aligned = nasdaq_px.loc[common]

# QQQ retornos alineados con common (solo donde QQQ existe)
qqq_common = qqq_p.index.intersection(common)
qqq_d_aligned = qqq_d_ret.reindex(common).fillna(0)

for period_label, months in periods.items():
    start_date = end_date - relativedelta(months=months)
    mask = (common >= start_date) & (common <= end_date)
    dates_p = common[mask]

    if len(dates_p) < 100:
        print(f"\n*** {period_label}: datos insuficientes ***")
        continue

    yrs_p = (dates_p[-1] - dates_p[0]).days / 365.25
    tqqq_ret_p = tqqq_d.loc[dates_p]
    gold_ret_p = gold_d.loc[dates_p]
    nasdaq_p = nasdaq_aligned.loc[dates_p]
    qqq_ret_p = qqq_d_aligned.loc[dates_p]

    print(f"\n{'=' * 80}")
    print(f"  PERÍODO: {period_label.upper()} — {dates_p[0].strftime('%Y-%m-%d')} a {dates_p[-1].strftime('%Y-%m-%d')} ({yrs_p:.1f} años)")
    print(f"{'=' * 80}")

    results = {}

    # Baseline
    results['Baseline 60/40'] = strat_baseline(dates_p, tqqq_ret_p, gold_ret_p)

    # Estrategia 1: SMA200 + Vol Target Asimétrico
    results['1.SMA+VT asim'] = strat_sma_voltarget(
        dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p)

    # Variante: SMA200 + VT con cap más alto (35%)
    results['1b.SMA+VT cap35'] = strat_sma_voltarget(
        dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p, vt_max=0.35)

    # Estrategia 2: Bandas Asimétricas + Momentum
    results['2.Bandas+Mom'] = strat_bandas_momentum(
        dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p)

    # Variante: bandas más estrechas
    results['2b.Bandas estrech'] = strat_bandas_momentum(
        dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p,
        upper_band=0.68, lower_band=0.50)

    # Estrategia 3: Risk Parity 3 Activos
    results['3.RiskParity 3'] = strat_risk_parity_3(
        dates_p, tqqq_ret_p, gold_ret_p, qqq_ret_p)

    # Variante: Risk Parity con cap TQQQ más alto
    results['3b.RP3 capTQQQ55'] = strat_risk_parity_3(
        dates_p, tqqq_ret_p, gold_ret_p, qqq_ret_p, cap_tqqq=0.55)

    # Estrategia 4: Regime Score Continuo
    results['4.RegimeScore'] = strat_regime_score(
        dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p)

    # Variante: Score con rango más amplio (10%-75%)
    results['4b.Score 10-75'] = strat_regime_score(
        dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p,
        w_min=0.10, w_range=0.65)

    # ===== TABLA PRINCIPAL =====
    print(f"\n  {'Estrategia':<22} {'Final ($10k)':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>9} {'Calmar':>8} {'UW días':>8}")
    print(f"  {'-' * 100}")

    all_m = {}
    for name, series in results.items():
        m = metrics(series)
        if m:
            all_m[name] = m
            print(f"  {name:<22} ${m['final']:>12,.0f} {m['cagr']:>7.2%} {m['vol']:>7.2%} {m['sharpe']:>7.2f} {m['max_dd']:>8.2%} {m['calmar']:>7.2f} {m['max_uw_days']:>7.0f}")

    # ===== RANKINGS =====
    print(f"\n  --- Rankings {period_label} ---")

    ranked_sharpe = sorted(all_m.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    print(f"\n  Por SHARPE:")
    for i, (name, m) in enumerate(ranked_sharpe, 1):
        marker = " <-- BASELINE" if name == 'Baseline 60/40' else ""
        print(f"    {i}. {name:<22} Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.2%}  MaxDD={m['max_dd']:.2%}{marker}")

    ranked_cagr = sorted(all_m.items(), key=lambda x: x[1]['cagr'], reverse=True)
    print(f"\n  Por CAGR:")
    for i, (name, m) in enumerate(ranked_cagr, 1):
        marker = " <-- BASELINE" if name == 'Baseline 60/40' else ""
        print(f"    {i}. {name:<22} CAGR={m['cagr']:.2%}  Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd']:.2%}{marker}")

    ranked_dd = sorted(all_m.items(), key=lambda x: x[1]['max_dd'], reverse=True)
    print(f"\n  Por MENOR DRAWDOWN:")
    for i, (name, m) in enumerate(ranked_dd, 1):
        marker = " <-- BASELINE" if name == 'Baseline 60/40' else ""
        print(f"    {i}. {name:<22} MaxDD={m['max_dd']:.2%}  CAGR={m['cagr']:.2%}  Sharpe={m['sharpe']:.2f}{marker}")

    # ===== CRISIS PERFORMANCE =====
    if months >= 240:
        print(f"\n  {'=' * 80}")
        print(f"  RENDIMIENTO EN CRISIS ({period_label})")
        print(f"  {'=' * 80}")

        crisis_periods = [
            ("Dot-com 2000-2003", "2000-01-01", "2003-12-31"),
            ("GFC 2007-2009", "2007-10-01", "2009-12-31"),
            ("COVID 2020", "2020-01-01", "2020-12-31"),
            ("Bear 2022", "2021-11-01", "2023-06-30"),
        ]

        for crisis_name, c_start, c_end in crisis_periods:
            crisis_dates = [d for d in dates_p if pd.Timestamp(c_start) <= d <= pd.Timestamp(c_end)]
            if len(crisis_dates) < 10:
                continue

            f, l = crisis_dates[0], crisis_dates[-1]
            print(f"\n  --- {crisis_name} ---")
            print(f"  {'Estrategia':<22} {'Return':>10} {'Max DD':>10}")
            print(f"  {'-' * 46}")

            for name, series in results.items():
                if f in series.index and l in series.index:
                    ret = series.loc[l] / series.loc[f] - 1
                    crisis_s = series.loc[crisis_dates]
                    rm = crisis_s.cummax()
                    dd = (crisis_s - rm) / rm
                    mdd = dd.min()
                    print(f"  {name:<22} {ret:>9.2%} {mdd:>9.2%}")

    # ===== YEAR-BY-YEAR =====
    print(f"\n  {'=' * 80}")
    print(f"  AÑO POR AÑO ({period_label})")
    print(f"  {'=' * 80}")

    top_names = ['Baseline 60/40', '1.SMA+VT asim', '2.Bandas+Mom', '4.RegimeScore']
    top_labels = ['Base', 'SMA+VT', 'Bandas', 'Score']

    print(f"\n  {'Año':<7}", end="")
    for lb in top_labels:
        print(f" {lb:>10}", end="")
    print(f" {'Mejor':>10}")
    print(f"  {'-' * 55}")

    all_years = sorted(set(d.year for d in dates_p))
    for year in all_years:
        yd = [d for d in dates_p if d.year == year]
        if len(yd) < 2:
            continue
        f, l = yd[0], yd[-1]

        rets = {}
        print(f"  {year:<7}", end="")
        for sname, label in zip(top_names, top_labels):
            if sname in results:
                r = results[sname].loc[l] / results[sname].loc[f] - 1
                rets[label] = r
                print(f" {r:>9.2%}", end="")
            else:
                print(f" {'N/A':>9}", end="")

        if rets:
            best = max(rets, key=rets.get)
            print(f" {best:>10}")
        else:
            print()

# =============================================================================
# 5. COMPARATIVA FINAL CROSS-PERIOD
# =============================================================================
print(f"\n{'=' * 80}")
print("COMPARATIVA FINAL: MEJOR ESTRATEGIA POR PERÍODO")
print(f"{'=' * 80}")

strategy_names = [
    'Baseline 60/40', '1.SMA+VT asim', '1b.SMA+VT cap35',
    '2.Bandas+Mom', '2b.Bandas estrech',
    '3.RiskParity 3', '3b.RP3 capTQQQ55',
    '4.RegimeScore', '4b.Score 10-75'
]

print(f"\n{'Estrategia':<22}", end="")
for pl in periods:
    print(f" {'---' + pl.upper() + '---':^24}", end="")
print()

print(f"{'':22}", end="")
for _ in periods:
    print(f" {'CAGR':>8} {'MaxDD':>8} {'Shrp':>6}", end="")
print()
print("-" * (22 + 24 * len(periods)))

for sname in strategy_names:
    print(f"{sname:<22}", end="")
    for period_label, months in periods.items():
        start_date = end_date - relativedelta(months=months)
        mask = (common >= start_date) & (common <= end_date)
        dates_p = common[mask]
        if len(dates_p) < 100:
            print(f" {'N/A':>8} {'N/A':>8} {'N/A':>6}", end="")
            continue

        tqqq_ret_p = tqqq_d.loc[dates_p]
        gold_ret_p = gold_d.loc[dates_p]
        nasdaq_p = nasdaq_aligned.loc[dates_p]
        qqq_ret_p = qqq_d_aligned.loc[dates_p]

        if sname == 'Baseline 60/40':
            s = strat_baseline(dates_p, tqqq_ret_p, gold_ret_p)
        elif sname == '1.SMA+VT asim':
            s = strat_sma_voltarget(dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p)
        elif sname == '1b.SMA+VT cap35':
            s = strat_sma_voltarget(dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p, vt_max=0.35)
        elif sname == '2.Bandas+Mom':
            s = strat_bandas_momentum(dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p)
        elif sname == '2b.Bandas estrech':
            s = strat_bandas_momentum(dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p, upper_band=0.68, lower_band=0.50)
        elif sname == '3.RiskParity 3':
            s = strat_risk_parity_3(dates_p, tqqq_ret_p, gold_ret_p, qqq_ret_p)
        elif sname == '3b.RP3 capTQQQ55':
            s = strat_risk_parity_3(dates_p, tqqq_ret_p, gold_ret_p, qqq_ret_p, cap_tqqq=0.55)
        elif sname == '4.RegimeScore':
            s = strat_regime_score(dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p)
        elif sname == '4b.Score 10-75':
            s = strat_regime_score(dates_p, tqqq_ret_p, gold_ret_p, nasdaq_p, w_min=0.10, w_range=0.65)
        else:
            print(f" {'N/A':>8} {'N/A':>8} {'N/A':>6}", end="")
            continue

        m = metrics(s)
        if m:
            print(f" {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>5.2f}", end="")
        else:
            print(f" {'ERR':>8} {'ERR':>8} {'ERR':>6}", end="")
    print()

print(f"""

RESUMEN
=======

Baseline 60/40 con rebalanceo anual:
  - CAGR extraordinario (~28% a 240m) pero MaxDD brutal (~-63%)

Las 4 estrategias buscan diferentes trade-offs:

1. SMA200 + Vol Target Asimétrico:
   → Protección SOLO en risk-off. Mantiene 60/40 puro en bull markets.
   → Menor drawdown con coste moderado de CAGR.

2. Bandas + Momentum:
   → Mejor timing de rebalanceo. Deja correr ganadores, no compra cuchillos.
   → Potencialmente puede SUPERAR el baseline en CAGR.

3. Risk Parity 3 Activos:
   → QQQ como amortiguador. Pesos dinámicos por inversa de volatilidad.
   → Mejor Sharpe pero menor CAGR (el diversificador tiene coste).

4. Regime Score Continuo:
   → Allocation continua sin whipsaws. Dimmer en vez de interruptor.
   → Buen balance entre CAGR y protección.
""")
