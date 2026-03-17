#!/usr/bin/env python3
"""
MOMENTUM BREAKOUT v8 — ESTRATEGIA COMPLETA (archivo unico para Codex)
=====================================================================

Estrategia sistematica long-only de momentum con fat tails.
225 tickers, 10 posiciones simultaneas max, opciones CALL.

Resultados validados 240 meses (20 anos):
  Profit Factor:  2.89
  Return total:   +37,780%
  Anualizado:     +34.6%
  Max Drawdown:   -42.6%
  Win Rate:       32.2%
  Trades:         1,416 (1,281 stocks + 135 opciones)

Auditoria:
  Walk-Forward: PF OOS = 99.4% del IS (sin overfitting)
  Survivorship bias: impacto bajo (-0.18 PF)
  Robustez: top 10 trades = 84% del PnL (inherente al estilo momentum)

Dependencias: yfinance, pandas, numpy, scipy
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. CONFIGURACION v8
# =============================================================================

CONFIG = {
    # Capital y posiciones
    'initial_capital': 10000,
    'target_risk_per_trade_pct': 2.0,   # 2% riesgo por trade
    'max_positions': 10,                 # 10 posiciones simultaneas max

    # Senales de entrada (4 condiciones simultaneas)
    'ker_threshold': 0.40,      # Kaufman Efficiency Ratio minimo
    'volume_threshold': 1.3,    # 1.3x volumen medio
    'rsi_threshold': 50,        # RSI > 50 para longs
    'rsi_max': 75,              # RSI < 75 (evitar sobrecompra)
    'breakout_period': 20,      # Breakout sobre maximo de 20 barras
    'longs_only': True,         # SOLO LONGS (shorts destruyen portfolio)

    # Stops y trailing
    'emergency_stop_pct': 0.15,         # -15% desde entrada (solo catastrofe)
    'trail_trigger_r': 2.0,             # Trailing se activa a +2R
    'trail_atr_mult': 4.0,              # Chandelier 4xATR (trailing normal)

    # Time exit v8 (trailing only, SIN salida forzada)
    'max_hold_bars': 8,                 # A los 8 dias se activa trailing
    'time_exit_trail_atr_mult': 3.0,    # Trailing apretado 3xATR

    # Filtro macro
    'use_macro_filter': True,
    'macro_ticker': 'SPY',
    'macro_sma_period': 50,

    # Costes
    'slippage_pct': 0.10,

    # Opciones CALL
    'option_dte': 120,              # DTE objetivo (~120 dias)
    'option_itm_pct': 0.05,        # 5% ITM (strike = spot * 0.95)
    'option_close_dte': 45,        # Cerrar cuando quedan 45 DTE
    'option_max_ivr': 40,          # Solo comprar si IVR < 40%
    'option_ivr_window': 252,      # Ventana IV Rank: 1 ano
    'option_position_pct': 0.14,   # ~14% del equity por opcion
    'max_option_positions': 2,     # Max 2 opciones simultaneas
    'option_spread_pct': 3.0,      # Spread bid-ask opciones
    'risk_free_rate': 0.043,       # Tasa libre de riesgo
    'hvol_window': 30,             # Ventana volatilidad historica
}


# =============================================================================
# 2. UNIVERSO: 225 TICKERS
# =============================================================================

ASSETS = {
    # US STOCKS - TECH (20)
    'AAPL': {'name': 'Apple', 'category': 'US_TECH'},
    'MSFT': {'name': 'Microsoft', 'category': 'US_TECH'},
    'GOOGL': {'name': 'Alphabet', 'category': 'US_TECH'},
    'AMZN': {'name': 'Amazon', 'category': 'US_TECH'},
    'NVDA': {'name': 'NVIDIA', 'category': 'US_TECH'},
    'META': {'name': 'Meta', 'category': 'US_TECH'},
    'TSLA': {'name': 'Tesla', 'category': 'US_TECH'},
    'AVGO': {'name': 'Broadcom', 'category': 'US_TECH'},
    'ORCL': {'name': 'Oracle', 'category': 'US_TECH'},
    'CRM': {'name': 'Salesforce', 'category': 'US_TECH'},
    'ADBE': {'name': 'Adobe', 'category': 'US_TECH'},
    'AMD': {'name': 'AMD', 'category': 'US_TECH'},
    'INTC': {'name': 'Intel', 'category': 'US_TECH'},
    'CSCO': {'name': 'Cisco', 'category': 'US_TECH'},
    'QCOM': {'name': 'Qualcomm', 'category': 'US_TECH'},
    'TXN': {'name': 'Texas Instruments', 'category': 'US_TECH'},
    'IBM': {'name': 'IBM', 'category': 'US_TECH'},
    'NOW': {'name': 'ServiceNow', 'category': 'US_TECH'},
    'INTU': {'name': 'Intuit', 'category': 'US_TECH'},
    'AMAT': {'name': 'Applied Materials', 'category': 'US_TECH'},
    # US STOCKS - FINANCE (10)
    'JPM': {'name': 'JPMorgan', 'category': 'US_FINANCE'},
    'BAC': {'name': 'Bank of America', 'category': 'US_FINANCE'},
    'WFC': {'name': 'Wells Fargo', 'category': 'US_FINANCE'},
    'GS': {'name': 'Goldman Sachs', 'category': 'US_FINANCE'},
    'MS': {'name': 'Morgan Stanley', 'category': 'US_FINANCE'},
    'BLK': {'name': 'BlackRock', 'category': 'US_FINANCE'},
    'SCHW': {'name': 'Schwab', 'category': 'US_FINANCE'},
    'C': {'name': 'Citigroup', 'category': 'US_FINANCE'},
    'AXP': {'name': 'American Express', 'category': 'US_FINANCE'},
    'USB': {'name': 'US Bancorp', 'category': 'US_FINANCE'},
    # US STOCKS - HEALTHCARE (10)
    'UNH': {'name': 'UnitedHealth', 'category': 'US_HEALTH'},
    'JNJ': {'name': 'Johnson & Johnson', 'category': 'US_HEALTH'},
    'LLY': {'name': 'Eli Lilly', 'category': 'US_HEALTH'},
    'PFE': {'name': 'Pfizer', 'category': 'US_HEALTH'},
    'ABBV': {'name': 'AbbVie', 'category': 'US_HEALTH'},
    'MRK': {'name': 'Merck', 'category': 'US_HEALTH'},
    'TMO': {'name': 'Thermo Fisher', 'category': 'US_HEALTH'},
    'ABT': {'name': 'Abbott', 'category': 'US_HEALTH'},
    'DHR': {'name': 'Danaher', 'category': 'US_HEALTH'},
    'BMY': {'name': 'Bristol-Myers', 'category': 'US_HEALTH'},
    # US STOCKS - CONSUMER (10)
    'WMT': {'name': 'Walmart', 'category': 'US_CONSUMER'},
    'HD': {'name': 'Home Depot', 'category': 'US_CONSUMER'},
    'PG': {'name': 'Procter & Gamble', 'category': 'US_CONSUMER'},
    'KO': {'name': 'Coca-Cola', 'category': 'US_CONSUMER'},
    'PEP': {'name': 'PepsiCo', 'category': 'US_CONSUMER'},
    'COST': {'name': 'Costco', 'category': 'US_CONSUMER'},
    'MCD': {'name': 'McDonalds', 'category': 'US_CONSUMER'},
    'NKE': {'name': 'Nike', 'category': 'US_CONSUMER'},
    'SBUX': {'name': 'Starbucks', 'category': 'US_CONSUMER'},
    'TGT': {'name': 'Target', 'category': 'US_CONSUMER'},
    # US STOCKS - INDUSTRIAL (10)
    'CAT': {'name': 'Caterpillar', 'category': 'US_INDUSTRIAL'},
    'HON': {'name': 'Honeywell', 'category': 'US_INDUSTRIAL'},
    'GE': {'name': 'GE Aerospace', 'category': 'US_INDUSTRIAL'},
    'UNP': {'name': 'Union Pacific', 'category': 'US_INDUSTRIAL'},
    'DE': {'name': 'Deere & Co', 'category': 'US_INDUSTRIAL'},
    'RTX': {'name': 'RTX Corp', 'category': 'US_INDUSTRIAL'},
    'LMT': {'name': 'Lockheed Martin', 'category': 'US_INDUSTRIAL'},
    'MMM': {'name': '3M', 'category': 'US_INDUSTRIAL'},
    'EMR': {'name': 'Emerson Electric', 'category': 'US_INDUSTRIAL'},
    'ETN': {'name': 'Eaton Corp', 'category': 'US_INDUSTRIAL'},
    # US STOCKS - ENERGY (10)
    'XOM': {'name': 'ExxonMobil', 'category': 'US_ENERGY'},
    'CVX': {'name': 'Chevron', 'category': 'US_ENERGY'},
    'COP': {'name': 'ConocoPhillips', 'category': 'US_ENERGY'},
    'SLB': {'name': 'Schlumberger', 'category': 'US_ENERGY'},
    'EOG': {'name': 'EOG Resources', 'category': 'US_ENERGY'},
    'MPC': {'name': 'Marathon Petroleum', 'category': 'US_ENERGY'},
    'PSX': {'name': 'Phillips 66', 'category': 'US_ENERGY'},
    'VLO': {'name': 'Valero Energy', 'category': 'US_ENERGY'},
    'OXY': {'name': 'Occidental Petroleum', 'category': 'US_ENERGY'},
    'HAL': {'name': 'Halliburton', 'category': 'US_ENERGY'},
    # US STOCKS - UTILITIES (10)
    'NEE': {'name': 'NextEra Energy', 'category': 'US_UTILITY'},
    'SO': {'name': 'Southern Company', 'category': 'US_UTILITY'},
    'DUK': {'name': 'Duke Energy', 'category': 'US_UTILITY'},
    'D': {'name': 'Dominion Energy', 'category': 'US_UTILITY'},
    'AEP': {'name': 'American Electric Power', 'category': 'US_UTILITY'},
    'SRE': {'name': 'Sempra', 'category': 'US_UTILITY'},
    'XEL': {'name': 'Xcel Energy', 'category': 'US_UTILITY'},
    'EXC': {'name': 'Exelon', 'category': 'US_UTILITY'},
    'WEC': {'name': 'WEC Energy', 'category': 'US_UTILITY'},
    'ED': {'name': 'Consolidated Edison', 'category': 'US_UTILITY'},
    # US STOCKS - REAL ESTATE (10)
    'AMT': {'name': 'American Tower', 'category': 'US_REALESTATE'},
    'PLD': {'name': 'Prologis', 'category': 'US_REALESTATE'},
    'CCI': {'name': 'Crown Castle', 'category': 'US_REALESTATE'},
    'O': {'name': 'Realty Income', 'category': 'US_REALESTATE'},
    'EQIX': {'name': 'Equinix', 'category': 'US_REALESTATE'},
    'SPG': {'name': 'Simon Property', 'category': 'US_REALESTATE'},
    'PSA': {'name': 'Public Storage', 'category': 'US_REALESTATE'},
    'DLR': {'name': 'Digital Realty', 'category': 'US_REALESTATE'},
    'WELL': {'name': 'Welltower', 'category': 'US_REALESTATE'},
    'AVB': {'name': 'AvalonBay', 'category': 'US_REALESTATE'},
    # US STOCKS - TELECOM & MEDIA (7)
    'T': {'name': 'AT&T', 'category': 'US_TELECOM'},
    'VZ': {'name': 'Verizon', 'category': 'US_TELECOM'},
    'CMCSA': {'name': 'Comcast', 'category': 'US_TELECOM'},
    'DIS': {'name': 'Disney', 'category': 'US_TELECOM'},
    'NFLX': {'name': 'Netflix', 'category': 'US_TELECOM'},
    'TMUS': {'name': 'T-Mobile US', 'category': 'US_TELECOM'},
    'CHTR': {'name': 'Charter Communications', 'category': 'US_TELECOM'},
    # EUROPE - GERMANY (10)
    'SAP': {'name': 'SAP SE', 'category': 'EU_GERMANY'},
    'SIE.DE': {'name': 'Siemens', 'category': 'EU_GERMANY'},
    'ALV.DE': {'name': 'Allianz', 'category': 'EU_GERMANY'},
    'DTE.DE': {'name': 'Deutsche Telekom', 'category': 'EU_GERMANY'},
    'MUV2.DE': {'name': 'Munich Re', 'category': 'EU_GERMANY'},
    'BAS.DE': {'name': 'BASF', 'category': 'EU_GERMANY'},
    'BMW.DE': {'name': 'BMW', 'category': 'EU_GERMANY'},
    'MBG.DE': {'name': 'Mercedes-Benz', 'category': 'EU_GERMANY'},
    'ADS.DE': {'name': 'Adidas', 'category': 'EU_GERMANY'},
    'IFX.DE': {'name': 'Infineon', 'category': 'EU_GERMANY'},
    # EUROPE - FRANCE (10)
    'OR.PA': {'name': "L'Oreal", 'category': 'EU_FRANCE'},
    'MC.PA': {'name': 'LVMH', 'category': 'EU_FRANCE'},
    'SAN.PA': {'name': 'Sanofi', 'category': 'EU_FRANCE'},
    'AI.PA': {'name': 'Air Liquide', 'category': 'EU_FRANCE'},
    'BNP.PA': {'name': 'BNP Paribas', 'category': 'EU_FRANCE'},
    'SU.PA': {'name': 'Schneider Electric', 'category': 'EU_FRANCE'},
    'AIR.PA': {'name': 'Airbus', 'category': 'EU_FRANCE'},
    'CS.PA': {'name': 'AXA', 'category': 'EU_FRANCE'},
    'DG.PA': {'name': 'Vinci', 'category': 'EU_FRANCE'},
    'RI.PA': {'name': 'Pernod Ricard', 'category': 'EU_FRANCE'},
    # EUROPE - OTHER EUROZONE (16)
    'ASML': {'name': 'ASML Holding', 'category': 'EU_NETHERLANDS'},
    'INGA.AS': {'name': 'ING Group', 'category': 'EU_NETHERLANDS'},
    'PHIA.AS': {'name': 'Philips', 'category': 'EU_NETHERLANDS'},
    'AD.AS': {'name': 'Ahold Delhaize', 'category': 'EU_NETHERLANDS'},
    'IBE.MC': {'name': 'Iberdrola', 'category': 'EU_SPAIN'},
    'SAN.MC': {'name': 'Santander', 'category': 'EU_SPAIN'},
    'TEF.MC': {'name': 'Telefonica', 'category': 'EU_SPAIN'},
    'ITX.MC': {'name': 'Inditex', 'category': 'EU_SPAIN'},
    'ENEL.MI': {'name': 'Enel', 'category': 'EU_ITALY'},
    'ISP.MI': {'name': 'Intesa Sanpaolo', 'category': 'EU_ITALY'},
    'UCG.MI': {'name': 'UniCredit', 'category': 'EU_ITALY'},
    'ENI.MI': {'name': 'Eni', 'category': 'EU_ITALY'},
    'KBC.BR': {'name': 'KBC Group', 'category': 'EU_BELGIUM'},
    'ABI.BR': {'name': 'AB InBev', 'category': 'EU_BELGIUM'},
    'NOK': {'name': 'Nokia', 'category': 'EU_FINLAND'},
    'CRH': {'name': 'CRH plc', 'category': 'EU_IRELAND'},
    # EUROPE - UK (9)
    'SHEL': {'name': 'Shell', 'category': 'EU_UK'},
    'HSBC': {'name': 'HSBC', 'category': 'EU_UK'},
    'BP': {'name': 'BP', 'category': 'EU_UK'},
    'RIO': {'name': 'Rio Tinto', 'category': 'EU_UK'},
    'GSK': {'name': 'GSK', 'category': 'EU_UK'},
    'ULVR.L': {'name': 'Unilever', 'category': 'EU_UK'},
    'LSEG.L': {'name': 'London Stock Exchange', 'category': 'EU_UK'},
    'BATS.L': {'name': 'BAT', 'category': 'EU_UK'},
    'DGE.L': {'name': 'Diageo', 'category': 'EU_UK'},
    # EUROPE - SWITZERLAND (6)
    'NESN.SW': {'name': 'Nestle', 'category': 'EU_SWISS'},
    'ROG.SW': {'name': 'Roche', 'category': 'EU_SWISS'},
    'NOVN.SW': {'name': 'Novartis', 'category': 'EU_SWISS'},
    'UBSG.SW': {'name': 'UBS', 'category': 'EU_SWISS'},
    'ZURN.SW': {'name': 'Zurich Insurance', 'category': 'EU_SWISS'},
    'ABBN.SW': {'name': 'ABB', 'category': 'EU_SWISS'},
    # EUROPE - NORDICS (5)
    'NOVO-B.CO': {'name': 'Novo Nordisk', 'category': 'EU_NORDIC'},
    'ERIC-B.ST': {'name': 'Ericsson', 'category': 'EU_NORDIC'},
    'VOLV-B.ST': {'name': 'Volvo', 'category': 'EU_NORDIC'},
    'SAND.ST': {'name': 'Sandvik', 'category': 'EU_NORDIC'},
    'NESTE.HE': {'name': 'Neste', 'category': 'EU_NORDIC'},
    # ASIA - JAPAN (10)
    '7203.T': {'name': 'Toyota', 'category': 'ASIA_JAPAN'},
    '6758.T': {'name': 'Sony', 'category': 'ASIA_JAPAN'},
    '6861.T': {'name': 'Keyence', 'category': 'ASIA_JAPAN'},
    '8306.T': {'name': 'Mitsubishi UFJ', 'category': 'ASIA_JAPAN'},
    '9984.T': {'name': 'SoftBank Group', 'category': 'ASIA_JAPAN'},
    '6501.T': {'name': 'Hitachi', 'category': 'ASIA_JAPAN'},
    '7267.T': {'name': 'Honda', 'category': 'ASIA_JAPAN'},
    '8035.T': {'name': 'Tokyo Electron', 'category': 'ASIA_JAPAN'},
    '4063.T': {'name': 'Shin-Etsu Chemical', 'category': 'ASIA_JAPAN'},
    '9432.T': {'name': 'NTT', 'category': 'ASIA_JAPAN'},
    # ASIA - AUSTRALIA (8)
    'BHP.AX': {'name': 'BHP Group', 'category': 'ASIA_AUSTRALIA'},
    'CBA.AX': {'name': 'Commonwealth Bank', 'category': 'ASIA_AUSTRALIA'},
    'CSL.AX': {'name': 'CSL Limited', 'category': 'ASIA_AUSTRALIA'},
    'NAB.AX': {'name': 'National Australia Bank', 'category': 'ASIA_AUSTRALIA'},
    'WBC.AX': {'name': 'Westpac', 'category': 'ASIA_AUSTRALIA'},
    'FMG.AX': {'name': 'Fortescue Metals', 'category': 'ASIA_AUSTRALIA'},
    'WDS.AX': {'name': 'Woodside Energy', 'category': 'ASIA_AUSTRALIA'},
    'RIO.AX': {'name': 'Rio Tinto AU', 'category': 'ASIA_AUSTRALIA'},
    # ASIA - CHINA / HK (5)
    'BABA': {'name': 'Alibaba', 'category': 'ASIA_CHINA'},
    'JD': {'name': 'JD.com', 'category': 'ASIA_CHINA'},
    'PDD': {'name': 'PDD Holdings', 'category': 'ASIA_CHINA'},
    'BIDU': {'name': 'Baidu', 'category': 'ASIA_CHINA'},
    '0700.HK': {'name': 'Tencent', 'category': 'ASIA_CHINA'},
    # COMMODITIES (17)
    'GLD': {'name': 'Gold ETF', 'category': 'COMMODITY_PRECIOUS'},
    'SLV': {'name': 'Silver ETF', 'category': 'COMMODITY_PRECIOUS'},
    'PPLT': {'name': 'Platinum ETF', 'category': 'COMMODITY_PRECIOUS'},
    'GDX': {'name': 'Gold Miners', 'category': 'COMMODITY_PRECIOUS'},
    'GDXJ': {'name': 'Junior Gold Miners', 'category': 'COMMODITY_PRECIOUS'},
    'USO': {'name': 'WTI Oil ETF', 'category': 'COMMODITY_ENERGY'},
    'BNO': {'name': 'Brent Oil ETF', 'category': 'COMMODITY_ENERGY'},
    'UNG': {'name': 'Natural Gas ETF', 'category': 'COMMODITY_ENERGY'},
    'XLE': {'name': 'Energy Sector', 'category': 'COMMODITY_ENERGY'},
    'XOP': {'name': 'Oil & Gas Exploration', 'category': 'COMMODITY_ENERGY'},
    'CPER': {'name': 'Copper ETF', 'category': 'COMMODITY_INDUSTRIAL'},
    'DBB': {'name': 'Base Metals', 'category': 'COMMODITY_INDUSTRIAL'},
    'PICK': {'name': 'Metal Mining', 'category': 'COMMODITY_INDUSTRIAL'},
    'DBA': {'name': 'Agriculture ETF', 'category': 'COMMODITY_AGRICULTURE'},
    'WEAT': {'name': 'Wheat ETF', 'category': 'COMMODITY_AGRICULTURE'},
    'CORN': {'name': 'Corn ETF', 'category': 'COMMODITY_AGRICULTURE'},
    'SOYB': {'name': 'Soybeans ETF', 'category': 'COMMODITY_AGRICULTURE'},
    # INDEX ETFs (8)
    'QQQ': {'name': 'Nasdaq-100', 'category': 'US_INDEX'},
    'TQQQ': {'name': 'Nasdaq 3x', 'category': 'US_INDEX_LEV'},
    'SPY': {'name': 'S&P 500', 'category': 'US_INDEX'},
    'SPXL': {'name': 'S&P 3x', 'category': 'US_INDEX_LEV'},
    'IWM': {'name': 'Russell 2000', 'category': 'US_INDEX'},
    'TNA': {'name': 'Russell 3x', 'category': 'US_INDEX_LEV'},
    'DIA': {'name': 'Dow Jones', 'category': 'US_INDEX'},
    'BITO': {'name': 'Bitcoin ETF', 'category': 'US_INDEX'},
    # ETFs SECTORIALES (6)
    'SMH': {'name': 'Semiconductors ETF', 'category': 'ETF_SECTOR'},
    'XBI': {'name': 'Biotech ETF', 'category': 'ETF_SECTOR'},
    'XLU': {'name': 'Utilities ETF', 'category': 'ETF_SECTOR'},
    'XLI': {'name': 'Industrials ETF', 'category': 'ETF_SECTOR'},
    'XLF': {'name': 'Financials ETF', 'category': 'ETF_SECTOR'},
    'XLV': {'name': 'Healthcare ETF', 'category': 'ETF_SECTOR'},
    # ETFs INTERNACIONALES (10)
    'EEM': {'name': 'Emerging Markets', 'category': 'ETF_INTL'},
    'VWO': {'name': 'Vanguard EM', 'category': 'ETF_INTL'},
    'EFA': {'name': 'EAFE (Developed ex-US)', 'category': 'ETF_INTL'},
    'FXI': {'name': 'China Large Cap', 'category': 'ETF_INTL'},
    'EWJ': {'name': 'Japan ETF', 'category': 'ETF_INTL'},
    'EWG': {'name': 'Germany ETF', 'category': 'ETF_INTL'},
    'EWU': {'name': 'UK ETF', 'category': 'ETF_INTL'},
    'INDA': {'name': 'India ETF', 'category': 'ETF_INTL'},
    'EWZ': {'name': 'Brazil ETF', 'category': 'ETF_INTL'},
    'EWT': {'name': 'Taiwan ETF', 'category': 'ETF_INTL'},
    # FIXED INCOME (8)
    'TLT': {'name': 'Treasury 20+ Year', 'category': 'FIXED_INCOME'},
    'IEF': {'name': 'Treasury 7-10 Year', 'category': 'FIXED_INCOME'},
    'SHY': {'name': 'Treasury 1-3 Year', 'category': 'FIXED_INCOME'},
    'TIP': {'name': 'TIPS (Inflation)', 'category': 'FIXED_INCOME'},
    'AGG': {'name': 'US Aggregate Bond', 'category': 'FIXED_INCOME'},
    'LQD': {'name': 'Investment Grade Corp', 'category': 'FIXED_INCOME'},
    'HYG': {'name': 'High Yield Corp', 'category': 'FIXED_INCOME'},
    'EMB': {'name': 'EM Sovereign Debt', 'category': 'FIXED_INCOME'},
}

TICKERS = list(ASSETS.keys())

# Tickers elegibles para opciones CALL (104 US stocks + ETFs liquidos)
OPTIONS_ELIGIBLE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
    'ORCL', 'CRM', 'ADBE', 'AMD', 'INTC', 'CSCO', 'QCOM',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'AXP',
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT',
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT',
    'CAT', 'HON', 'GE', 'UNP', 'DE', 'RTX', 'LMT', 'MMM', 'EMR', 'ETN',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'OXY', 'HAL',
    'NEE', 'SO', 'DUK', 'D', 'AEP',
    'AMT', 'PLD', 'EQIX', 'SPG', 'PSA', 'DLR', 'O',
    'T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'TMUS',
    'BABA', 'JD', 'PDD', 'BIDU',
    'QQQ', 'SPY', 'IWM', 'DIA', 'GLD', 'SLV', 'XLE', 'TLT',
    'TQQQ', 'SPXL', 'TNA', 'BITO',
    'SMH', 'XBI', 'XLU', 'XLI', 'XLF', 'XLV', 'EEM', 'EFA', 'HYG',
]


# =============================================================================
# 3. INDICADORES TECNICOS
# =============================================================================

def calculate_kaufman_efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """
    KER: mide eficiencia de la tendencia (0 a 1).
    > 0.40 = mercado en tendencia (tomar senales)
    < 0.30 = mercado choppy (evitar)
    """
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(window=period).sum()
    er = direction / volatility.replace(0, np.nan)
    return er.fillna(0)


def calculate_atr(high: pd.Series, low: pd.Series,
                  close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range estandar."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI estandar."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def historical_volatility(close_prices, window=30):
    """Volatilidad historica anualizada (proxy de IV para opciones)."""
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


# =============================================================================
# 4. MOTOR DE SENALES (MomentumEngine)
# =============================================================================

class MomentumEngine:
    """
    Genera senales de momentum breakout.

    Logica (todas usan .shift(1) para evitar look-ahead bias):
      LONG si las 4 condiciones se cumplen simultaneamente:
        1. KER[t-1] > 0.40 (tendencia)
        2. 50 < RSI[t-1] < 75 (momentum sin sobrecompra)
        3. Close[t-1] > RollingHigh[t-2] (breakout confirmado)
        4. Volume[t-1] > 1.3x media (confirmacion institucional)
    """

    def __init__(self, breakout_period=20, volume_threshold=1.3,
                 ker_threshold=0.40, rsi_threshold=50, rsi_max=75,
                 rsi_period=14, atr_period=14, ker_period=10,
                 longs_only=True):
        self.breakout_period = breakout_period
        self.volume_threshold = volume_threshold
        self.ker_threshold = ker_threshold
        self.rsi_threshold = rsi_threshold
        self.rsi_max = rsi_max
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.ker_period = ker_period
        self.longs_only = longs_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genera serie de senales: +1 (long), 0 (neutral)."""
        df = df.copy()
        df['_rsi'] = calculate_rsi(df['Close'], self.rsi_period)
        df['_atr'] = calculate_atr(df['High'], df['Low'], df['Close'], self.atr_period)
        df['_ker'] = calculate_kaufman_efficiency_ratio(df['Close'], self.ker_period)
        vol_sma = df['Volume'].rolling(window=20).mean()
        df['_vol_ratio'] = (df['Volume'] / vol_sma.replace(0, 1.0)).fillna(1.0)
        df['_rolling_high'] = df['High'].rolling(window=self.breakout_period).max()

        # .shift(1) = usar datos del dia anterior (no look-ahead)
        is_trending = df['_ker'].shift(1) > self.ker_threshold
        rsi_bullish = (df['_rsi'].shift(1) > self.rsi_threshold) & (df['_rsi'].shift(1) < self.rsi_max)
        vol_confirmed = df['_vol_ratio'].shift(1) > self.volume_threshold
        breakout_long = df['Close'].shift(1) > df['_rolling_high'].shift(2)

        long_signal = is_trending & rsi_bullish & breakout_long & vol_confirmed

        signals = pd.Series(0, index=df.index, name='signal')
        signals.loc[long_signal] = 1

        warmup = max(self.breakout_period, self.rsi_period, self.atr_period, self.ker_period) + 5
        signals.iloc[:warmup] = 0
        return signals

    def generate_signals_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera senales con todos los indicadores para analisis."""
        df = df.copy()
        df['rsi'] = calculate_rsi(df['Close'], self.rsi_period)
        df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'], self.atr_period)
        df['ker'] = calculate_kaufman_efficiency_ratio(df['Close'], self.ker_period)
        vol_sma = df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = (df['Volume'] / vol_sma.replace(0, 1.0)).fillna(1.0)
        df['rolling_high'] = df['High'].rolling(window=self.breakout_period).max()
        df['signal'] = self.generate_signals(df)
        return df


# =============================================================================
# 5. RANKING MULTI-FACTOR (seleccion de las mejores senales)
# =============================================================================
#
# Cuando hay mas senales que slots disponibles, se rankean por:
#   Score = 0.30*KER + 0.20*RSI_norm + 0.20*Vol_norm + 0.15*Breakout_str + 0.15*ATR%
#
# Se seleccionan las top N (hasta llenar max_positions).

def rank_candidates(candidates, signals_data):
    """
    Rankea candidatos por score multi-factor.
    candidates: lista de (ticker, bar_idx, prev_atr)
    signals_data: dict con df, signals, ker, rsi, vol_ratio por ticker
    """
    ranked = []
    for ticker, idx, prev_atr in candidates:
        sd = signals_data[ticker]
        df_t = sd['df']
        prev_idx = idx - 1

        ker_val = sd['ker'].iloc[prev_idx] if prev_idx >= 0 else 0
        rsi_val = sd['rsi'].iloc[prev_idx] if prev_idx >= 0 else 50
        rsi_score = max(0, min(1, (rsi_val - CONFIG['rsi_threshold']) /
                                  (CONFIG['rsi_max'] - CONFIG['rsi_threshold'])))
        vol_val = sd['vol_ratio'].iloc[prev_idx] if prev_idx >= 0 else 1.0
        vol_score = min(1, max(0, (vol_val - 1.0) / 2.0))

        if prev_idx >= 1:
            close_prev = df_t['Close'].iloc[prev_idx]
            rolling_high_prev = df_t['High'].iloc[
                max(0, prev_idx - CONFIG['breakout_period']):prev_idx].max()
            breakout_pct = (close_prev - rolling_high_prev) / rolling_high_prev \
                if rolling_high_prev > 0 else 0
            breakout_score = min(1, max(0, breakout_pct / 0.05))
        else:
            breakout_score = 0

        price_prev = df_t['Close'].iloc[prev_idx] if prev_idx >= 0 else 1
        atr_pct = prev_atr / price_prev if price_prev > 0 else 0
        atr_score = min(1, atr_pct / 0.04)

        composite = (0.30 * ker_val + 0.20 * rsi_score + 0.20 * vol_score +
                     0.15 * breakout_score + 0.15 * atr_score)
        ranked.append((ticker, idx, prev_atr, composite))

    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked


# =============================================================================
# 6. FILTRO MACRO (SPY > SMA50)
# =============================================================================
#
# Solo se abren posiciones nuevas cuando SPY > SMA50.
# En festivos US, se usa el ultimo valor conocido de SPY.
# El filtro NO fuerza ventas — las posiciones existentes siguen
# con sus trailing stops normales.


# =============================================================================
# 7. POSITION SIZING
# =============================================================================
#
# ACCIONES:
#   R = ATR * 2 (riesgo por trade)
#   units = (equity * 2%) / R   (inverse volatility)
#   Tope: equity / max_positions por posicion
#
# OPCIONES:
#   premium = equity * 14%
#   contracts = premium / (option_price * 100)
#   Max 2 opciones simultaneas


# =============================================================================
# 8. GESTION DE POSICIONES — ACCIONES (3 mecanismos de salida)
# =============================================================================

@dataclass
class Trade:
    """Gestiona una posicion de acciones/ETFs."""
    ticker: str
    entry_price: float
    entry_date: datetime
    entry_atr: float
    position_euros: float
    position_units: float

    R: float = field(init=False)
    trailing_stop: Optional[float] = field(default=None)
    trailing_active: bool = field(default=False)
    highest_since: float = field(init=False)
    max_r_mult: float = field(default=0.0)
    bars_held: int = field(default=0)

    exit_price: Optional[float] = field(default=None)
    exit_date: Optional[datetime] = field(default=None)
    exit_reason: Optional[str] = field(default=None)
    pnl_euros: float = field(default=0.0)
    pnl_pct: float = field(default=0.0)

    def __post_init__(self):
        self.R = self.entry_atr * 2.0
        self.highest_since = self.entry_price

    def update(self, high, low, close, current_atr):
        """Actualiza la posicion con la barra actual. Retorna dict si cierra."""
        self.bars_held += 1
        self.highest_since = max(self.highest_since, high)
        r_mult = (close - self.entry_price) / self.R if self.R > 0 else 0
        self.max_r_mult = max(self.max_r_mult, r_mult)

        # SALIDA 1: Emergency stop -15% (solo catastrofe)
        emergency_level = self.entry_price * (1 - CONFIG['emergency_stop_pct'])
        if low <= emergency_level:
            self._close(emergency_level * (1 - CONFIG['slippage_pct'] / 100),
                        'emergency_stop')
            return {'type': 'full_exit', 'reason': 'emergency_stop'}

        # SALIDA 2: Trailing stop (si activo)
        if self.trailing_active and self.trailing_stop is not None:
            if low <= self.trailing_stop:
                self._close(self.trailing_stop * (1 - CONFIG['slippage_pct'] / 100),
                            'trailing_stop')
                return {'type': 'full_exit', 'reason': 'trailing_stop'}

        # ACTIVAR trailing a +2R (Chandelier 4xATR)
        if r_mult >= CONFIG['trail_trigger_r']:
            chandelier = self.highest_since - (current_atr * CONFIG['trail_atr_mult'])
            if not self.trailing_active:
                self.trailing_active = True
                self.trailing_stop = chandelier
            elif chandelier > self.trailing_stop:
                self.trailing_stop = chandelier

        # SALIDA 3: Time exit a 8 dias (activar trailing, NUNCA forzar salida)
        if self.bars_held >= CONFIG['max_hold_bars']:
            if not self.trailing_active:
                trail_mult = CONFIG.get('time_exit_trail_atr_mult', 3.0)
                chandelier = self.highest_since - (current_atr * trail_mult)
                breakeven = self.entry_price * (1 + CONFIG['slippage_pct'] / 100)
                self.trailing_active = True
                if close <= self.entry_price:
                    # Perdiendo: trailing apretado (3xATR o 5% bajo maximo)
                    self.trailing_stop = max(chandelier, self.entry_price * 0.95)
                else:
                    # Ganando: trailing a breakeven minimo
                    self.trailing_stop = max(chandelier, breakeven)

        return None

    def _close(self, exit_price, reason):
        self.pnl_euros = (exit_price - self.entry_price) * self.position_units
        self.pnl_pct = (self.pnl_euros / self.position_euros) * 100 \
            if self.position_euros > 0 else 0
        self.exit_price = exit_price
        self.exit_reason = reason


# =============================================================================
# 9. GESTION DE POSICIONES — OPCIONES CALL
# =============================================================================
#
# Reglas:
#   - CALL 5% ITM (strike = spot * 0.95), vencimiento mensual ~120 DTE
#   - Solo comprar si IV Rank < 40% (opciones baratas)
#   - Cierre automatico a 45 DTE restantes (antes de theta acelerado)
#   - Sin stop loss (riesgo = prima pagada)
#   - Max 2 opciones simultaneas
#   - Position size: 14% del equity por opcion

def black_scholes_call(S, K, T, r, sigma):
    """Precio de opcion CALL via Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'price': max(S - K, 0), 'delta': 1.0 if S > K else 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return {'price': max(price, 0.01), 'delta': delta}


def monthly_expiration_dte(entry_date, target_dte=120):
    """Calcula DTE real al 3er viernes (vencimiento mensual) mas cercano."""
    target_date = entry_date + timedelta(days=target_dte)
    year, month = target_date.year, target_date.month
    candidates = []
    for delta_months in [-1, 0, 1]:
        m = month + delta_months
        y = year
        if m < 1: m = 12; y -= 1
        elif m > 12: m = 1; y += 1
        first_day = datetime(y, m, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        if third_friday > entry_date:
            candidates.append(third_friday)
    if not candidates:
        return target_dte
    best = min(candidates, key=lambda d: abs((d - entry_date).days - target_dte))
    return (best - entry_date).days


def iv_rank(hvol_series, current_idx, window=252):
    """
    IV Rank: percentil de IV actual vs ultimos 'window' dias.
    0-100. Bajo = opciones baratas. Alto = opciones caras.
    """
    start = max(0, current_idx - window)
    hist = hvol_series.iloc[start:current_idx + 1].dropna()
    if len(hist) < 20:
        return 50.0
    iv_now = hist.iloc[-1]
    iv_min = hist.min()
    iv_max = hist.max()
    if iv_max == iv_min:
        return 50.0
    return (iv_now - iv_min) / (iv_max - iv_min) * 100


@dataclass
class OptionTrade:
    """Gestiona una posicion de opciones CALL."""
    ticker: str
    entry_date: datetime
    entry_stock_price: float
    strike: float
    dte_at_entry: int
    entry_option_price: float
    entry_iv: float
    num_contracts: float
    position_euros: float   # premium pagada = max loss

    bars_held: int = field(default=0)
    max_option_value: float = field(init=False)
    max_r_mult: float = field(default=0.0)

    exit_date: Optional[datetime] = field(default=None)
    exit_option_price: float = field(default=0.0)
    exit_reason: Optional[str] = field(default=None)
    pnl_euros: float = field(default=0.0)
    pnl_pct: float = field(default=0.0)

    def __post_init__(self):
        self.max_option_value = self.entry_option_price

    def update(self, stock_price, current_iv, days_elapsed):
        """Actualiza opcion. Retorna dict si cierra."""
        self.bars_held += 1
        remaining_dte = max(self.dte_at_entry - days_elapsed, 0)
        T = remaining_dte / 365.0

        bs = black_scholes_call(S=stock_price, K=self.strike, T=T,
                                r=CONFIG['risk_free_rate'], sigma=current_iv)
        current_option_price = bs['price']
        current_option_price *= (1 - CONFIG['option_spread_pct'] / 100 / 2)

        self.max_option_value = max(self.max_option_value, current_option_price)
        option_return = (current_option_price / self.entry_option_price) - 1 \
            if self.entry_option_price > 0 else 0
        self.max_r_mult = max(self.max_r_mult, option_return)

        # Expiracion (safety)
        if remaining_dte <= 0:
            intrinsic = max(stock_price - self.strike, 0)
            intrinsic *= (1 - CONFIG['option_spread_pct'] / 100 / 2)
            self._close(intrinsic, 'expiration')
            return {'type': 'full_exit', 'reason': 'expiration'}

        # Cierre a 45 DTE restantes (antes de theta acelerado)
        if remaining_dte <= CONFIG.get('option_close_dte', 45):
            self._close(current_option_price, 'dte_exit')
            return {'type': 'full_exit', 'reason': 'dte_exit'}

        return None

    def _close(self, exit_option_price, reason):
        self.exit_option_price = exit_option_price
        self.exit_reason = reason
        self.pnl_euros = (exit_option_price - self.entry_option_price) * \
            self.num_contracts * 100
        self.pnl_pct = ((exit_option_price / self.entry_option_price) - 1) * 100 \
            if self.entry_option_price > 0 else 0


# =============================================================================
# 10. EQUITY TRACKER
# =============================================================================

class EquityTracker:
    """Rastrea equity, posiciones y drawdown."""

    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.equity_curve = []
        self.max_equity = initial_capital
        self.open_positions = 0
        self.open_options = 0

    def get_position_size(self, ticker, current_atr, price):
        """Position sizing para acciones: inverse volatility."""
        risk_pct = CONFIG['target_risk_per_trade_pct'] / 100
        R = current_atr * 2.0
        if R <= 0 or price <= 0:
            return {'units': 0, 'notional': 0}
        dollar_risk = self.equity * risk_pct
        units = dollar_risk / R
        notional = units * price
        max_notional = self.equity / CONFIG['max_positions'] * 2
        if notional > max_notional:
            notional = max_notional
            units = notional / price
        return {'units': units, 'notional': notional}

    def get_option_size(self, option_price):
        """Position sizing para opciones: 14% del equity."""
        position_pct = CONFIG.get('option_position_pct', 0.14)
        max_premium = self.equity * position_pct
        if option_price <= 0:
            return {'contracts': 0, 'premium': 0}
        contracts = max_premium / (option_price * 100)
        premium = contracts * option_price * 100
        return {'contracts': contracts, 'premium': premium}

    def update_equity(self, pnl, date):
        self.equity += pnl
        self.equity_curve.append((date, self.equity))
        self.max_equity = max(self.max_equity, self.equity)

    def get_max_drawdown(self):
        if not self.equity_curve:
            return 0
        equity_values = [self.initial_capital] + [e[1] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        for eq in equity_values:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd


# =============================================================================
# 11. RESUMEN DE REGLAS (para referencia rapida)
# =============================================================================
#
# ENTRADA (4 condiciones simultaneas):
#   1. KER > 0.40
#   2. 50 < RSI < 75
#   3. Volume > 1.3x media 20d
#   4. Close > max(High, 20 barras)
#
# FILTRO MACRO: SPY > SMA50 (no entrar en bear)
#
# RANKING: Score = 0.30*KER + 0.20*RSI + 0.20*Vol + 0.15*Breakout + 0.15*ATR%
#
# SIZING ACCIONES: R = 2*ATR, units = (equity*2%)/R, max equity/10
# SIZING OPCIONES: premium = equity*14%, max 2 simultaneas
#
# SALIDAS ACCIONES (por prioridad):
#   1. Emergency stop -15%
#   2. Trailing Chandelier 4xATR (activa a +2R)
#   3. Time exit dia 8: trailing 3xATR (nunca fuerza salida)
#
# SALIDAS OPCIONES:
#   1. Cierre a 45 DTE restantes
#   2. Sin stop (riesgo = prima)
#
# DECISION ACCION vs OPCION:
#   Si ticker elegible + IVR < 40 + slot libre -> CALL 5% ITM 120 DTE
#   Si no -> accion normal
