#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║           MOMENTUM BREAKOUT STRATEGY MODULE v1.0                              ║
║                                                                               ║
║           Multi-Asset Trading Engine for 4H Timeframe                         ║
║           Target: High Skew (Low Win Rate, Fat Tails)                         ║
║           Capital: $10k | Volatility Target: 20% Annual                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

MODULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module 1: Whipsaw Filter (Regime Detection)
    - Kaufman Efficiency Ratio (KER)
    - Choppiness Index (CI)
    - Boolean mask: is_trending

Module 2: Dynamic Sizing (Inverse Volatility)
    - calculate_position_size()
    - Target 20% annual portfolio volatility
    - Adapts to varying daily volatility

Module 3: Signal Vectorization (MomentumEngine)
    - Fully vectorized pandas/numpy
    - Look-ahead bias prevention
    - Time-normalized volume for 24/7 markets

Module 4: Entry/Exit & Dynamic Stop Loss
    - Entry levels (market/limit)
    - 3-tier exit system (breakeven, partial, trailing)
    - DynamicStopManager with Chandelier Exit

USAGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from momentum_breakout import MomentumEngine, DynamicStopManager, calculate_position_size

    # Generate signals
    engine = MomentumEngine(ker_threshold=0.40, volume_threshold=1.3)
    signals = engine.generate_signals(df)

    # Calculate position size
    size = calculate_position_size(
        account_balance=10000,
        current_atr=df['ATR'].iloc[-1],
        price=df['Close'].iloc[-1],
        is_crypto=True
    )

    # Manage stops
    stop_mgr = DynamicStopManager()
    stop_info = stop_mgr.calculate_stop(
        position_type='long',
        entry_price=100,
        current_price=106,
        current_atr=2,
        highest_since_entry=108,
        lowest_since_entry=99
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET UNIVERSE (120+ instruments)
# ═══════════════════════════════════════════════════════════════════════════════

ASSETS = {
    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - TECH (20)
    # ═══════════════════════════════════════════════════════════════════════════
    'AAPL': {'name': 'Apple', 'category': 'US_TECH', 'is_crypto': False},
    'MSFT': {'name': 'Microsoft', 'category': 'US_TECH', 'is_crypto': False},
    'GOOGL': {'name': 'Alphabet', 'category': 'US_TECH', 'is_crypto': False},
    'AMZN': {'name': 'Amazon', 'category': 'US_TECH', 'is_crypto': False},
    'NVDA': {'name': 'NVIDIA', 'category': 'US_TECH', 'is_crypto': False},
    'META': {'name': 'Meta', 'category': 'US_TECH', 'is_crypto': False},
    'TSLA': {'name': 'Tesla', 'category': 'US_TECH', 'is_crypto': False},
    'AVGO': {'name': 'Broadcom', 'category': 'US_TECH', 'is_crypto': False},
    'ORCL': {'name': 'Oracle', 'category': 'US_TECH', 'is_crypto': False},
    'CRM': {'name': 'Salesforce', 'category': 'US_TECH', 'is_crypto': False},
    'ADBE': {'name': 'Adobe', 'category': 'US_TECH', 'is_crypto': False},
    'AMD': {'name': 'AMD', 'category': 'US_TECH', 'is_crypto': False},
    'INTC': {'name': 'Intel', 'category': 'US_TECH', 'is_crypto': False},
    'CSCO': {'name': 'Cisco', 'category': 'US_TECH', 'is_crypto': False},
    'QCOM': {'name': 'Qualcomm', 'category': 'US_TECH', 'is_crypto': False},
    'TXN': {'name': 'Texas Instruments', 'category': 'US_TECH', 'is_crypto': False},
    'IBM': {'name': 'IBM', 'category': 'US_TECH', 'is_crypto': False},
    'NOW': {'name': 'ServiceNow', 'category': 'US_TECH', 'is_crypto': False},
    'INTU': {'name': 'Intuit', 'category': 'US_TECH', 'is_crypto': False},
    'AMAT': {'name': 'Applied Materials', 'category': 'US_TECH', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - FINANCE (10)
    # ═══════════════════════════════════════════════════════════════════════════
    'JPM': {'name': 'JPMorgan', 'category': 'US_FINANCE', 'is_crypto': False},
    'BAC': {'name': 'Bank of America', 'category': 'US_FINANCE', 'is_crypto': False},
    'WFC': {'name': 'Wells Fargo', 'category': 'US_FINANCE', 'is_crypto': False},
    'GS': {'name': 'Goldman Sachs', 'category': 'US_FINANCE', 'is_crypto': False},
    'MS': {'name': 'Morgan Stanley', 'category': 'US_FINANCE', 'is_crypto': False},
    'BLK': {'name': 'BlackRock', 'category': 'US_FINANCE', 'is_crypto': False},
    'SCHW': {'name': 'Schwab', 'category': 'US_FINANCE', 'is_crypto': False},
    'C': {'name': 'Citigroup', 'category': 'US_FINANCE', 'is_crypto': False},
    'AXP': {'name': 'American Express', 'category': 'US_FINANCE', 'is_crypto': False},
    'USB': {'name': 'US Bancorp', 'category': 'US_FINANCE', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - HEALTHCARE (10)
    # ═══════════════════════════════════════════════════════════════════════════
    'UNH': {'name': 'UnitedHealth', 'category': 'US_HEALTH', 'is_crypto': False},
    'JNJ': {'name': 'Johnson & Johnson', 'category': 'US_HEALTH', 'is_crypto': False},
    'LLY': {'name': 'Eli Lilly', 'category': 'US_HEALTH', 'is_crypto': False},
    'PFE': {'name': 'Pfizer', 'category': 'US_HEALTH', 'is_crypto': False},
    'ABBV': {'name': 'AbbVie', 'category': 'US_HEALTH', 'is_crypto': False},
    'MRK': {'name': 'Merck', 'category': 'US_HEALTH', 'is_crypto': False},
    'TMO': {'name': 'Thermo Fisher', 'category': 'US_HEALTH', 'is_crypto': False},
    'ABT': {'name': 'Abbott', 'category': 'US_HEALTH', 'is_crypto': False},
    'DHR': {'name': 'Danaher', 'category': 'US_HEALTH', 'is_crypto': False},
    'BMY': {'name': 'Bristol-Myers', 'category': 'US_HEALTH', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - CONSUMER (10)
    # ═══════════════════════════════════════════════════════════════════════════
    'WMT': {'name': 'Walmart', 'category': 'US_CONSUMER', 'is_crypto': False},
    'HD': {'name': 'Home Depot', 'category': 'US_CONSUMER', 'is_crypto': False},
    'PG': {'name': 'Procter & Gamble', 'category': 'US_CONSUMER', 'is_crypto': False},
    'KO': {'name': 'Coca-Cola', 'category': 'US_CONSUMER', 'is_crypto': False},
    'PEP': {'name': 'PepsiCo', 'category': 'US_CONSUMER', 'is_crypto': False},
    'COST': {'name': 'Costco', 'category': 'US_CONSUMER', 'is_crypto': False},
    'MCD': {'name': 'McDonalds', 'category': 'US_CONSUMER', 'is_crypto': False},
    'NKE': {'name': 'Nike', 'category': 'US_CONSUMER', 'is_crypto': False},
    'SBUX': {'name': 'Starbucks', 'category': 'US_CONSUMER', 'is_crypto': False},
    'TGT': {'name': 'Target', 'category': 'US_CONSUMER', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - INDUSTRIAL (10) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'CAT': {'name': 'Caterpillar', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'HON': {'name': 'Honeywell', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'GE': {'name': 'GE Aerospace', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'UNP': {'name': 'Union Pacific', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'DE': {'name': 'Deere & Co', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'RTX': {'name': 'RTX Corp', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'LMT': {'name': 'Lockheed Martin', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'MMM': {'name': '3M', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'EMR': {'name': 'Emerson Electric', 'category': 'US_INDUSTRIAL', 'is_crypto': False},
    'ETN': {'name': 'Eaton Corp', 'category': 'US_INDUSTRIAL', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - ENERGY (10) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'XOM': {'name': 'ExxonMobil', 'category': 'US_ENERGY', 'is_crypto': False},
    'CVX': {'name': 'Chevron', 'category': 'US_ENERGY', 'is_crypto': False},
    'COP': {'name': 'ConocoPhillips', 'category': 'US_ENERGY', 'is_crypto': False},
    'SLB': {'name': 'Schlumberger', 'category': 'US_ENERGY', 'is_crypto': False},
    'EOG': {'name': 'EOG Resources', 'category': 'US_ENERGY', 'is_crypto': False},
    'MPC': {'name': 'Marathon Petroleum', 'category': 'US_ENERGY', 'is_crypto': False},
    'PSX': {'name': 'Phillips 66', 'category': 'US_ENERGY', 'is_crypto': False},
    'VLO': {'name': 'Valero Energy', 'category': 'US_ENERGY', 'is_crypto': False},
    'OXY': {'name': 'Occidental Petroleum', 'category': 'US_ENERGY', 'is_crypto': False},
    'HAL': {'name': 'Halliburton', 'category': 'US_ENERGY', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - UTILITIES (10) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'NEE': {'name': 'NextEra Energy', 'category': 'US_UTILITY', 'is_crypto': False},
    'SO': {'name': 'Southern Company', 'category': 'US_UTILITY', 'is_crypto': False},
    'DUK': {'name': 'Duke Energy', 'category': 'US_UTILITY', 'is_crypto': False},
    'D': {'name': 'Dominion Energy', 'category': 'US_UTILITY', 'is_crypto': False},
    'AEP': {'name': 'American Electric Power', 'category': 'US_UTILITY', 'is_crypto': False},
    'SRE': {'name': 'Sempra', 'category': 'US_UTILITY', 'is_crypto': False},
    'XEL': {'name': 'Xcel Energy', 'category': 'US_UTILITY', 'is_crypto': False},
    'EXC': {'name': 'Exelon', 'category': 'US_UTILITY', 'is_crypto': False},
    'WEC': {'name': 'WEC Energy', 'category': 'US_UTILITY', 'is_crypto': False},
    'ED': {'name': 'Consolidated Edison', 'category': 'US_UTILITY', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - REAL ESTATE (10) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'AMT': {'name': 'American Tower', 'category': 'US_REALESTATE', 'is_crypto': False},
    'PLD': {'name': 'Prologis', 'category': 'US_REALESTATE', 'is_crypto': False},
    'CCI': {'name': 'Crown Castle', 'category': 'US_REALESTATE', 'is_crypto': False},
    'O': {'name': 'Realty Income', 'category': 'US_REALESTATE', 'is_crypto': False},
    'EQIX': {'name': 'Equinix', 'category': 'US_REALESTATE', 'is_crypto': False},
    'SPG': {'name': 'Simon Property', 'category': 'US_REALESTATE', 'is_crypto': False},
    'PSA': {'name': 'Public Storage', 'category': 'US_REALESTATE', 'is_crypto': False},
    'DLR': {'name': 'Digital Realty', 'category': 'US_REALESTATE', 'is_crypto': False},
    'WELL': {'name': 'Welltower', 'category': 'US_REALESTATE', 'is_crypto': False},
    'AVB': {'name': 'AvalonBay', 'category': 'US_REALESTATE', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # US STOCKS - TELECOM & MEDIA (7) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'T': {'name': 'AT&T', 'category': 'US_TELECOM', 'is_crypto': False},
    'VZ': {'name': 'Verizon', 'category': 'US_TELECOM', 'is_crypto': False},
    'CMCSA': {'name': 'Comcast', 'category': 'US_TELECOM', 'is_crypto': False},
    'DIS': {'name': 'Disney', 'category': 'US_TELECOM', 'is_crypto': False},
    'NFLX': {'name': 'Netflix', 'category': 'US_TELECOM', 'is_crypto': False},
    'TMUS': {'name': 'T-Mobile US', 'category': 'US_TELECOM', 'is_crypto': False},
    'CHTR': {'name': 'Charter Communications', 'category': 'US_TELECOM', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # EUROPE - GERMANY (10)
    # ═══════════════════════════════════════════════════════════════════════════
    'SAP': {'name': 'SAP SE', 'category': 'EU_GERMANY', 'is_crypto': False},
    'SIE.DE': {'name': 'Siemens', 'category': 'EU_GERMANY', 'is_crypto': False},
    'ALV.DE': {'name': 'Allianz', 'category': 'EU_GERMANY', 'is_crypto': False},
    'DTE.DE': {'name': 'Deutsche Telekom', 'category': 'EU_GERMANY', 'is_crypto': False},
    'MUV2.DE': {'name': 'Munich Re', 'category': 'EU_GERMANY', 'is_crypto': False},
    'BAS.DE': {'name': 'BASF', 'category': 'EU_GERMANY', 'is_crypto': False},
    'BMW.DE': {'name': 'BMW', 'category': 'EU_GERMANY', 'is_crypto': False},
    'MBG.DE': {'name': 'Mercedes-Benz', 'category': 'EU_GERMANY', 'is_crypto': False},
    'ADS.DE': {'name': 'Adidas', 'category': 'EU_GERMANY', 'is_crypto': False},
    'IFX.DE': {'name': 'Infineon', 'category': 'EU_GERMANY', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # EUROPE - FRANCE (10)
    # ═══════════════════════════════════════════════════════════════════════════
    'OR.PA': {'name': "L'Oreal", 'category': 'EU_FRANCE', 'is_crypto': False},
    'MC.PA': {'name': 'LVMH', 'category': 'EU_FRANCE', 'is_crypto': False},
    'SAN.PA': {'name': 'Sanofi', 'category': 'EU_FRANCE', 'is_crypto': False},
    'AI.PA': {'name': 'Air Liquide', 'category': 'EU_FRANCE', 'is_crypto': False},
    'BNP.PA': {'name': 'BNP Paribas', 'category': 'EU_FRANCE', 'is_crypto': False},
    'SU.PA': {'name': 'Schneider Electric', 'category': 'EU_FRANCE', 'is_crypto': False},
    'AIR.PA': {'name': 'Airbus', 'category': 'EU_FRANCE', 'is_crypto': False},
    'CS.PA': {'name': 'AXA', 'category': 'EU_FRANCE', 'is_crypto': False},
    'DG.PA': {'name': 'Vinci', 'category': 'EU_FRANCE', 'is_crypto': False},
    'RI.PA': {'name': 'Pernod Ricard', 'category': 'EU_FRANCE', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # EUROPE - OTHER EUROZONE (16)
    # ═══════════════════════════════════════════════════════════════════════════
    'ASML': {'name': 'ASML Holding', 'category': 'EU_NETHERLANDS', 'is_crypto': False},
    'INGA.AS': {'name': 'ING Group', 'category': 'EU_NETHERLANDS', 'is_crypto': False},
    'PHIA.AS': {'name': 'Philips', 'category': 'EU_NETHERLANDS', 'is_crypto': False},
    'AD.AS': {'name': 'Ahold Delhaize', 'category': 'EU_NETHERLANDS', 'is_crypto': False},
    'IBE.MC': {'name': 'Iberdrola', 'category': 'EU_SPAIN', 'is_crypto': False},
    'SAN.MC': {'name': 'Santander', 'category': 'EU_SPAIN', 'is_crypto': False},
    'TEF.MC': {'name': 'Telefonica', 'category': 'EU_SPAIN', 'is_crypto': False},
    'ITX.MC': {'name': 'Inditex', 'category': 'EU_SPAIN', 'is_crypto': False},
    'ENEL.MI': {'name': 'Enel', 'category': 'EU_ITALY', 'is_crypto': False},
    'ISP.MI': {'name': 'Intesa Sanpaolo', 'category': 'EU_ITALY', 'is_crypto': False},
    'UCG.MI': {'name': 'UniCredit', 'category': 'EU_ITALY', 'is_crypto': False},
    'ENI.MI': {'name': 'Eni', 'category': 'EU_ITALY', 'is_crypto': False},
    'KBC.BR': {'name': 'KBC Group', 'category': 'EU_BELGIUM', 'is_crypto': False},
    'ABI.BR': {'name': 'AB InBev', 'category': 'EU_BELGIUM', 'is_crypto': False},
    'NOK': {'name': 'Nokia', 'category': 'EU_FINLAND', 'is_crypto': False},
    'CRH': {'name': 'CRH plc', 'category': 'EU_IRELAND', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # EUROPE - UK (10) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'SHEL': {'name': 'Shell', 'category': 'EU_UK', 'is_crypto': False},
    'HSBC': {'name': 'HSBC', 'category': 'EU_UK', 'is_crypto': False},
    'BP': {'name': 'BP', 'category': 'EU_UK', 'is_crypto': False},
    'RIO': {'name': 'Rio Tinto', 'category': 'EU_UK', 'is_crypto': False},
    'GSK': {'name': 'GSK', 'category': 'EU_UK', 'is_crypto': False},
    'ULVR.L': {'name': 'Unilever', 'category': 'EU_UK', 'is_crypto': False},
    'LSEG.L': {'name': 'London Stock Exchange', 'category': 'EU_UK', 'is_crypto': False},
    'BATS.L': {'name': 'BAT', 'category': 'EU_UK', 'is_crypto': False},
    'DGE.L': {'name': 'Diageo', 'category': 'EU_UK', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # EUROPE - SWITZERLAND (6) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'NESN.SW': {'name': 'Nestle', 'category': 'EU_SWISS', 'is_crypto': False},
    'ROG.SW': {'name': 'Roche', 'category': 'EU_SWISS', 'is_crypto': False},
    'NOVN.SW': {'name': 'Novartis', 'category': 'EU_SWISS', 'is_crypto': False},
    'UBSG.SW': {'name': 'UBS', 'category': 'EU_SWISS', 'is_crypto': False},
    'ZURN.SW': {'name': 'Zurich Insurance', 'category': 'EU_SWISS', 'is_crypto': False},
    'ABBN.SW': {'name': 'ABB', 'category': 'EU_SWISS', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # EUROPE - NORDICS (5) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'NOVO-B.CO': {'name': 'Novo Nordisk', 'category': 'EU_NORDIC', 'is_crypto': False},
    'ERIC-B.ST': {'name': 'Ericsson', 'category': 'EU_NORDIC', 'is_crypto': False},
    'VOLV-B.ST': {'name': 'Volvo', 'category': 'EU_NORDIC', 'is_crypto': False},
    'SAND.ST': {'name': 'Sandvik', 'category': 'EU_NORDIC', 'is_crypto': False},
    'NESTE.HE': {'name': 'Neste', 'category': 'EU_NORDIC', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # ASIA - JAPAN (10) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    '7203.T': {'name': 'Toyota', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '6758.T': {'name': 'Sony', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '6861.T': {'name': 'Keyence', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '8306.T': {'name': 'Mitsubishi UFJ', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '9984.T': {'name': 'SoftBank Group', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '6501.T': {'name': 'Hitachi', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '7267.T': {'name': 'Honda', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '8035.T': {'name': 'Tokyo Electron', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '4063.T': {'name': 'Shin-Etsu Chemical', 'category': 'ASIA_JAPAN', 'is_crypto': False},
    '9432.T': {'name': 'NTT', 'category': 'ASIA_JAPAN', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # ASIA - AUSTRALIA (8) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'BHP.AX': {'name': 'BHP Group', 'category': 'ASIA_AUSTRALIA', 'is_crypto': False},
    'CBA.AX': {'name': 'Commonwealth Bank', 'category': 'ASIA_AUSTRALIA', 'is_crypto': False},
    'CSL.AX': {'name': 'CSL Limited', 'category': 'ASIA_AUSTRALIA', 'is_crypto': False},
    'NAB.AX': {'name': 'National Australia Bank', 'category': 'ASIA_AUSTRALIA', 'is_crypto': False},
    'WBC.AX': {'name': 'Westpac', 'category': 'ASIA_AUSTRALIA', 'is_crypto': False},
    'FMG.AX': {'name': 'Fortescue Metals', 'category': 'ASIA_AUSTRALIA', 'is_crypto': False},
    'WDS.AX': {'name': 'Woodside Energy', 'category': 'ASIA_AUSTRALIA', 'is_crypto': False},
    'RIO.AX': {'name': 'Rio Tinto AU', 'category': 'ASIA_AUSTRALIA', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # ASIA - CHINA / HK (5) — NUEVO (solo ADRs US + HK líquidos)
    # ═══════════════════════════════════════════════════════════════════════════
    'BABA': {'name': 'Alibaba', 'category': 'ASIA_CHINA', 'is_crypto': False},
    'JD': {'name': 'JD.com', 'category': 'ASIA_CHINA', 'is_crypto': False},
    'PDD': {'name': 'PDD Holdings', 'category': 'ASIA_CHINA', 'is_crypto': False},
    'BIDU': {'name': 'Baidu', 'category': 'ASIA_CHINA', 'is_crypto': False},
    '0700.HK': {'name': 'Tencent', 'category': 'ASIA_CHINA', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # COMMODITIES - ETFs (17)
    # ═══════════════════════════════════════════════════════════════════════════
    'GLD': {'name': 'Gold ETF', 'category': 'COMMODITY_PRECIOUS', 'is_crypto': False, 'future': '/MGC'},
    'SLV': {'name': 'Silver ETF', 'category': 'COMMODITY_PRECIOUS', 'is_crypto': False, 'future': '/SIL'},
    'PPLT': {'name': 'Platinum ETF', 'category': 'COMMODITY_PRECIOUS', 'is_crypto': False},
    'GDX': {'name': 'Gold Miners', 'category': 'COMMODITY_PRECIOUS', 'is_crypto': False},
    'GDXJ': {'name': 'Junior Gold Miners', 'category': 'COMMODITY_PRECIOUS', 'is_crypto': False},
    'USO': {'name': 'WTI Oil ETF', 'category': 'COMMODITY_ENERGY', 'is_crypto': False, 'future': '/CL'},
    'BNO': {'name': 'Brent Oil ETF', 'category': 'COMMODITY_ENERGY', 'is_crypto': False},
    'UNG': {'name': 'Natural Gas ETF', 'category': 'COMMODITY_ENERGY', 'is_crypto': False, 'future': '/NG'},
    'XLE': {'name': 'Energy Sector', 'category': 'COMMODITY_ENERGY', 'is_crypto': False},
    'XOP': {'name': 'Oil & Gas Exploration', 'category': 'COMMODITY_ENERGY', 'is_crypto': False},
    'CPER': {'name': 'Copper ETF', 'category': 'COMMODITY_INDUSTRIAL', 'is_crypto': False},
    'DBB': {'name': 'Base Metals', 'category': 'COMMODITY_INDUSTRIAL', 'is_crypto': False},
    'PICK': {'name': 'Metal Mining', 'category': 'COMMODITY_INDUSTRIAL', 'is_crypto': False},
    'DBA': {'name': 'Agriculture ETF', 'category': 'COMMODITY_AGRICULTURE', 'is_crypto': False},
    'WEAT': {'name': 'Wheat ETF', 'category': 'COMMODITY_AGRICULTURE', 'is_crypto': False},
    'CORN': {'name': 'Corn ETF', 'category': 'COMMODITY_AGRICULTURE', 'is_crypto': False},
    'SOYB': {'name': 'Soybeans ETF', 'category': 'COMMODITY_AGRICULTURE', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # INDEX ETFs - US (5) & LEVERAGED (3)
    # ═══════════════════════════════════════════════════════════════════════════
    'QQQ': {'name': 'Nasdaq-100', 'category': 'US_INDEX', 'is_crypto': False, 'future': '/MNQ'},
    'TQQQ': {'name': 'Nasdaq 3x', 'category': 'US_INDEX_LEV', 'is_crypto': False},
    'SPY': {'name': 'S&P 500', 'category': 'US_INDEX', 'is_crypto': False, 'future': '/MES'},
    'SPXL': {'name': 'S&P 3x', 'category': 'US_INDEX_LEV', 'is_crypto': False},
    'IWM': {'name': 'Russell 2000', 'category': 'US_INDEX', 'is_crypto': False, 'future': '/M2K'},
    'TNA': {'name': 'Russell 3x', 'category': 'US_INDEX_LEV', 'is_crypto': False},
    'DIA': {'name': 'Dow Jones', 'category': 'US_INDEX', 'is_crypto': False, 'future': '/MYM'},
    'BITO': {'name': 'Bitcoin ETF', 'category': 'US_INDEX', 'is_crypto': False, 'future': '/MBT'},

    # ═══════════════════════════════════════════════════════════════════════════
    # ETFs SECTORIALES US (6) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'SMH': {'name': 'Semiconductors ETF', 'category': 'ETF_SECTOR', 'is_crypto': False},
    'XBI': {'name': 'Biotech ETF', 'category': 'ETF_SECTOR', 'is_crypto': False},
    'XLU': {'name': 'Utilities ETF', 'category': 'ETF_SECTOR', 'is_crypto': False},
    'XLI': {'name': 'Industrials ETF', 'category': 'ETF_SECTOR', 'is_crypto': False},
    'XLF': {'name': 'Financials ETF', 'category': 'ETF_SECTOR', 'is_crypto': False},
    'XLV': {'name': 'Healthcare ETF', 'category': 'ETF_SECTOR', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # ETFs INTERNACIONALES (10) — NUEVO
    # ═══════════════════════════════════════════════════════════════════════════
    'EEM': {'name': 'Emerging Markets', 'category': 'ETF_INTL', 'is_crypto': False},
    'VWO': {'name': 'Vanguard EM', 'category': 'ETF_INTL', 'is_crypto': False},
    'EFA': {'name': 'EAFE (Developed ex-US)', 'category': 'ETF_INTL', 'is_crypto': False},
    'FXI': {'name': 'China Large Cap', 'category': 'ETF_INTL', 'is_crypto': False},
    'EWJ': {'name': 'Japan ETF', 'category': 'ETF_INTL', 'is_crypto': False},
    'EWG': {'name': 'Germany ETF', 'category': 'ETF_INTL', 'is_crypto': False},
    'EWU': {'name': 'UK ETF', 'category': 'ETF_INTL', 'is_crypto': False},
    'INDA': {'name': 'India ETF', 'category': 'ETF_INTL', 'is_crypto': False},
    'EWZ': {'name': 'Brazil ETF', 'category': 'ETF_INTL', 'is_crypto': False},
    'EWT': {'name': 'Taiwan ETF', 'category': 'ETF_INTL', 'is_crypto': False},

    # ═══════════════════════════════════════════════════════════════════════════
    # FIXED INCOME & DIVERSIFICADORES (8) — AMPLIADO
    # ═══════════════════════════════════════════════════════════════════════════
    'TLT': {'name': 'Treasury 20+ Year', 'category': 'FIXED_INCOME', 'is_crypto': False},
    'IEF': {'name': 'Treasury 7-10 Year', 'category': 'FIXED_INCOME', 'is_crypto': False},
    'SHY': {'name': 'Treasury 1-3 Year', 'category': 'FIXED_INCOME', 'is_crypto': False},
    'TIP': {'name': 'TIPS (Inflation)', 'category': 'FIXED_INCOME', 'is_crypto': False},
    'AGG': {'name': 'US Aggregate Bond', 'category': 'FIXED_INCOME', 'is_crypto': False},
    'LQD': {'name': 'Investment Grade Corp', 'category': 'FIXED_INCOME', 'is_crypto': False},
    'HYG': {'name': 'High Yield Corp', 'category': 'FIXED_INCOME', 'is_crypto': False},
    'EMB': {'name': 'EM Sovereign Debt', 'category': 'FIXED_INCOME', 'is_crypto': False},
}

# Helper lists
TICKERS = list(ASSETS.keys())
CRYPTO_TICKERS = [t for t, v in ASSETS.items() if v.get('is_crypto', False)]
US_STOCK_TICKERS = [t for t, v in ASSETS.items() if v.get('category', '').startswith('US_') and not v.get('is_crypto')]
EU_TICKERS = [t for t, v in ASSETS.items() if v.get('category', '').startswith('EU_')]
ASIA_TICKERS = [t for t, v in ASSETS.items() if v.get('category', '').startswith('ASIA_')]
COMMODITY_TICKERS = [t for t, v in ASSETS.items() if v.get('category', '').startswith('COMMODITY_')]
ETF_TICKERS = [t for t, v in ASSETS.items() if v.get('category', '').startswith('ETF_')]
FIXED_INCOME_TICKERS = [t for t, v in ASSETS.items() if v.get('category', '') == 'FIXED_INCOME']
CATEGORIES = list(set(v['category'] for v in ASSETS.values()))


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: WHIPSAW FILTER (REGIME DETECTION)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_kaufman_efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Kaufman Efficiency Ratio (KER) - Measures trend efficiency.

    Formula:
        Direction = |Close[t] - Close[t-N]|
        Volatility = Sum(|Close[i] - Close[i-1]|) for i in (t-N+1 to t)
        ER = Direction / Volatility

    Args:
        close: Series of closing prices
        period: Lookback period (default 10 for 4H timeframe)

    Returns:
        Series of efficiency ratio values (0 to 1)
        - Values > 0.4: Trending market (take breakout signals)
        - Values < 0.3: Choppy market (avoid signals)

    Why KER > EMA crossover for Crypto/Tech:
        - EMA crossovers lag badly in 24/7 markets (no overnight gap to reset)
        - KER measures price efficiency instantly
        - Detects when momentum is "clean" vs "noisy"
    """
    # Direction: absolute price change over period
    direction = (close - close.shift(period)).abs()

    # Volatility: sum of absolute bar-to-bar changes
    volatility = close.diff().abs().rolling(window=period).sum()

    # Efficiency Ratio (avoid division by zero)
    er = direction / volatility.replace(0, np.nan)

    return er.fillna(0)


def calculate_choppiness_index(high: pd.Series, low: pd.Series,
                                close: pd.Series, period: int = 14) -> pd.Series:
    """
    Choppiness Index - Measures market consolidation vs trending.

    Formula:
        ATR_Sum = Sum(True Range) over N periods
        Range_N = Highest_High - Lowest_Low over N periods
        CI = 100 * LOG10(ATR_Sum / Range_N) / LOG10(N)

    Args:
        high, low, close: OHLC price series
        period: Lookback period (default 14)

    Returns:
        Series of CI values (0 to 100)
        - Values < 45: Trending market (take signals)
        - Values > 55: Choppy market (avoid signals)
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Sum of TR over period
    atr_sum = true_range.rolling(window=period).sum()

    # Highest high and lowest low over period
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    range_n = highest - lowest

    # Choppiness Index (avoid log of zero/negative)
    range_n = range_n.replace(0, np.nan)
    ci = 100 * np.log10(atr_sum / range_n) / np.log10(period)

    return ci.fillna(50)  # Neutral value for NaN


def get_regime_filter(df: pd.DataFrame,
                      method: str = 'ker',
                      ker_threshold: float = 0.40,
                      ci_threshold: float = 45.0,
                      ker_period: int = 10,
                      ci_period: int = 14) -> pd.Series:
    """
    Creates boolean mask for trending regime.

    Args:
        df: DataFrame with OHLC data
        method: 'ker' (recommended) or 'choppiness'
        ker_threshold: Minimum KER for trending (default 0.40)
        ci_threshold: Maximum CI for trending (default 45)
        ker_period: KER lookback period
        ci_period: CI lookback period

    Returns:
        Boolean Series - True when market is trending (safe to take signals)
    """
    if method == 'ker':
        er = calculate_kaufman_efficiency_ratio(df['Close'], period=ker_period)
        is_trending = er > ker_threshold
    elif method == 'choppiness':
        ci = calculate_choppiness_index(
            df['High'], df['Low'], df['Close'], period=ci_period
        )
        is_trending = ci < ci_threshold
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ker' or 'choppiness'")

    return is_trending


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: DYNAMIC SIZING (INVERSE VOLATILITY)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_atr(high: pd.Series, low: pd.Series,
                  close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range - Standard implementation.

    Args:
        high, low, close: OHLC price series
        period: ATR period (default 14)

    Returns:
        Series of ATR values in price terms
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(window=period).mean()


def calculate_position_size(
    account_balance: float,
    current_atr: float,
    price: float,
    target_vol_annual: float = 0.20,
    timeframe_hours: int = 4,
    is_crypto: bool = False,
    contract_multiplier: float = 1.0,
    min_size: float = 0.001,
    max_leverage: float = 1.0
) -> Dict:
    """
    Calculate position size using inverse volatility scaling.

    Goal: Keep portfolio heat constant regardless of whether BTC moves 2% or 10% daily.

    Formula:
        target_daily_vol = target_vol_annual / sqrt(trading_days)
        atr_pct = current_atr / price
        daily_vol_estimate = atr_pct * sqrt(bars_per_day)
        dollar_risk = account_balance * target_daily_vol
        position_size = (dollar_risk / daily_vol_estimate) / price

    Args:
        account_balance: Total account equity (e.g., 10000)
        current_atr: Current ATR value in price terms
        price: Current asset price
        target_vol_annual: Target annual portfolio volatility (default 20%)
        timeframe_hours: Bar timeframe in hours (default 4)
        is_crypto: If True, use 365 days; if False, use 252 days
        contract_multiplier: Futures multiplier (e.g., 2 for MNQ $2/point)
        min_size: Minimum position size
        max_leverage: Maximum allowed leverage (1.0 = no leverage)

    Returns:
        dict with:
            - 'units': Position size in units/contracts
            - 'notional': Dollar value of position
            - 'risk_pct': Actual portfolio risk percentage
            - 'atr_pct': ATR as percentage of price
            - 'daily_vol_est': Estimated daily volatility
            - 'leverage': Implied leverage

    Example for MNQ:
        size = calculate_position_size(
            account_balance=10000,
            current_atr=50,  # 50 points ATR
            price=18000,     # NQ at 18000
            is_crypto=False,
            contract_multiplier=2  # MNQ is $2/point
        )

    Example for BTCUSDT:
        size = calculate_position_size(
            account_balance=10000,
            current_atr=1500,  # $1500 ATR
            price=45000,       # BTC at 45000
            is_crypto=True,
            contract_multiplier=1
        )
    """
    # Validate inputs
    if account_balance <= 0 or price <= 0:
        return {
            'units': 0, 'notional': 0, 'risk_pct': 0,
            'atr_pct': 0, 'daily_vol_est': 0, 'leverage': 0
        }

    # Trading days per year
    trading_days = 365 if is_crypto else 252

    # Trading hours per day
    trading_hours_per_day = 24 if is_crypto else 6.5  # US market hours
    bars_per_day = trading_hours_per_day / timeframe_hours

    # Target daily volatility
    target_daily_vol = target_vol_annual / np.sqrt(trading_days)

    # ATR as percentage of price
    atr_pct = current_atr / price if price > 0 else 0.02

    # Estimate daily volatility from ATR
    # 4H ATR approximates daily range / sqrt(bars_per_day)
    daily_vol_estimate = atr_pct * np.sqrt(bars_per_day)

    # Floor for safety
    if daily_vol_estimate < 0.005:
        daily_vol_estimate = 0.02  # Default 2% daily vol

    # Dollar amount to risk for target volatility
    dollar_risk = account_balance * target_daily_vol

    # Position value to achieve target vol
    position_value = dollar_risk / daily_vol_estimate

    # Apply leverage cap
    max_position_value = account_balance * max_leverage
    position_value = min(position_value, max_position_value)

    # Convert to units
    units = position_value / (price * contract_multiplier)

    # Apply minimum size
    units = max(units, min_size)

    # Recalculate actual values
    actual_notional = units * price * contract_multiplier
    actual_leverage = actual_notional / account_balance
    actual_risk_pct = (actual_notional * daily_vol_estimate) / account_balance

    return {
        'units': round(units, 6),
        'notional': round(actual_notional, 2),
        'risk_pct': round(actual_risk_pct * 100, 2),
        'atr_pct': round(atr_pct * 100, 4),
        'daily_vol_est': round(daily_vol_estimate * 100, 2),
        'leverage': round(actual_leverage, 2)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: SIGNAL VECTORIZATION (MomentumEngine)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_time_normalized_volume(df: pd.DataFrame,
                                      lookback_weeks: int = 4) -> pd.Series:
    """
    Time-normalized volume that accounts for intraday and weekly patterns.

    FIXES the "Volume > SMA(Volume, 20) * 1.5" flaw for 24/7 crypto markets
    where weekend volume naturally dips 30-50%.

    Method:
        1. Group historical volume by (hour, day_of_week)
        2. Calculate rolling median for each time slot
        3. Express current volume as ratio to expected volume for that slot

    Args:
        df: DataFrame with 'Volume' column and DatetimeIndex
        lookback_weeks: Weeks of history to use (default 4)

    Returns:
        Series of normalized volume ratios
        - 1.0 = normal volume for this time slot
        - 2.0 = 2x normal volume (strong confirmation)
        - 0.5 = 50% of normal (weak, ignore signal)
    """
    df = df.copy()

    # Check for datetime index
    if not hasattr(df.index, 'hour'):
        # Fall back to simple normalization
        vol_sma = df['Volume'].rolling(window=20).mean()
        return (df['Volume'] / vol_sma.replace(0, 1.0)).fillna(1.0)

    # Extract time components
    df['_hour'] = df.index.hour
    df['_dow'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['_time_slot'] = df['_hour'].astype(str) + '_' + df['_dow'].astype(str)

    # Calculate normalized volume
    lookback_bars = lookback_weeks * 7 * 6  # ~6 bars per day on 4H
    normalized_vol = pd.Series(1.0, index=df.index)

    for idx in range(20, len(df)):
        current_slot = df['_time_slot'].iloc[idx]
        current_vol = df['Volume'].iloc[idx]

        # Get historical data for same time slot
        start_idx = max(0, idx - lookback_bars)
        historical = df.iloc[start_idx:idx]
        same_slot_vol = historical[historical['_time_slot'] == current_slot]['Volume']

        if len(same_slot_vol) >= 2:
            expected_vol = same_slot_vol.median()
            if expected_vol > 0:
                normalized_vol.iloc[idx] = current_vol / expected_vol
        else:
            # Fallback: use simple 20-bar SMA
            expected_vol = df['Volume'].iloc[max(0, idx-20):idx].mean()
            if expected_vol > 0:
                normalized_vol.iloc[idx] = current_vol / expected_vol

    return normalized_vol


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


class MomentumEngine:
    """
    Vectorized momentum breakout signal generator.

    Strategy (High Skew / Fat Tails):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. REGIME: KER > threshold (trending, not choppy)
    2. SETUP: RSI > threshold (momentum confirmation)
    3. TRIGGER: Close > RollingHigh (breakout)
    4. FILTER: Normalized Volume > threshold

    Output: Signal Series (-1 short, 0 neutral, +1 long)

    Features:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Fully vectorized (fast backtesting)
    - Look-ahead bias prevention via .shift(1)
    - Time-normalized volume (fixes 24/7 market issues)
    - Proper NaN handling for warmup period

    Usage:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        engine = MomentumEngine(
            breakout_period=20,
            volume_threshold=1.3,
            ker_threshold=0.40,
            rsi_threshold=55
        )
        signals = engine.generate_signals(df)

        # With full metadata for analysis
        result_df = engine.generate_signals_with_metadata(df)
    """

    def __init__(
        self,
        breakout_period: int = 20,
        volume_threshold: float = 1.3,
        ker_threshold: float = 0.40,
        rsi_threshold: float = 55,
        rsi_max: float = 75,
        rsi_period: int = 14,
        atr_period: int = 14,
        ker_period: int = 10,
        use_time_normalized_volume: bool = True,
        longs_only: bool = True
    ):
        """
        Initialize MomentumEngine.

        Args:
            breakout_period: Lookback for rolling high/low (default 20)
            volume_threshold: Min normalized volume ratio (default 1.3)
            ker_threshold: Min KER for trending regime (default 0.40)
            rsi_threshold: Min RSI for long setup (default 55)
            rsi_max: Max RSI for long setup to avoid overbought (default 75)
            rsi_period: RSI calculation period (default 14)
            atr_period: ATR calculation period (default 14)
            ker_period: KER calculation period (default 10)
            use_time_normalized_volume: Use time-aware volume (default True)
            longs_only: If True, only generate LONG signals (default True)
                       Backtest showed shorts destroy portfolio in bull markets:
                       LONGS +130.8% P&L vs SHORTS -281.1% P&L
        """
        self.breakout_period = breakout_period
        self.volume_threshold = volume_threshold
        self.ker_threshold = ker_threshold
        self.rsi_threshold = rsi_threshold
        self.rsi_max = rsi_max
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.ker_period = ker_period
        self.use_time_normalized_volume = use_time_normalized_volume
        self.longs_only = longs_only

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        return calculate_rsi(close, self.rsi_period)

    def _calculate_atr(self, high: pd.Series, low: pd.Series,
                       close: pd.Series) -> pd.Series:
        """Calculate ATR indicator."""
        return calculate_atr(high, low, close, self.atr_period)

    def _calculate_ker(self, close: pd.Series) -> pd.Series:
        """Calculate Kaufman Efficiency Ratio."""
        return calculate_kaufman_efficiency_ratio(close, self.ker_period)

    def _calculate_normalized_volume(self, df: pd.DataFrame) -> pd.Series:
        """Calculate time-normalized or simple volume ratio."""
        if self.use_time_normalized_volume:
            return calculate_time_normalized_volume(df)
        else:
            vol_sma = df['Volume'].rolling(window=20).mean()
            return (df['Volume'] / vol_sma.replace(0, 1.0)).fillna(1.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate momentum breakout signals.

        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                Must have DatetimeIndex for time-normalized volume.

        Returns:
            Series of signals: +1 (long), 0 (neutral), -1 (short)

        Signal Logic (all use .shift(1) to prevent look-ahead bias):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        LONG (+1):
            - KER[t-1] > ker_threshold (trending regime)
            - RSI[t-1] > rsi_threshold (bullish momentum)
            - Close[t-1] > RollingHigh[t-2] (breakout confirmed)
            - NormalizedVolume[t-1] > volume_threshold

        SHORT (-1):
            - KER[t-1] > ker_threshold (trending regime)
            - RSI[t-1] < (100 - rsi_threshold) (bearish momentum)
            - Close[t-1] < RollingLow[t-2] (breakdown confirmed)
            - NormalizedVolume[t-1] > volume_threshold
        """
        # Validate input
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Create working copy
        df = df.copy()

        # Calculate indicators
        df['_rsi'] = self._calculate_rsi(df['Close'])
        df['_atr'] = self._calculate_atr(df['High'], df['Low'], df['Close'])
        df['_ker'] = self._calculate_ker(df['Close'])
        df['_vol_ratio'] = self._calculate_normalized_volume(df)

        # Rolling high/low for breakout detection
        df['_rolling_high'] = df['High'].rolling(window=self.breakout_period).max()
        df['_rolling_low'] = df['Low'].rolling(window=self.breakout_period).min()

        # ════════════════════════════════════════════════════════════
        # CRITICAL: .shift(1) prevents look-ahead bias
        # Signal at bar t uses data from bar t-1 and earlier
        # ════════════════════════════════════════════════════════════

        # Regime filter (previous bar)
        is_trending = df['_ker'].shift(1) > self.ker_threshold

        # RSI conditions (previous bar)
        # For longs: RSI > threshold AND RSI < rsi_max (avoid overbought)
        rsi_bullish = (df['_rsi'].shift(1) > self.rsi_threshold) & (df['_rsi'].shift(1) < self.rsi_max)
        rsi_bearish = df['_rsi'].shift(1) < (100 - self.rsi_threshold)

        # Volume confirmation (previous bar)
        vol_confirmed = df['_vol_ratio'].shift(1) > self.volume_threshold

        # Breakout conditions
        # Close at t-1 compared to rolling level at t-2
        breakout_long = df['Close'].shift(1) > df['_rolling_high'].shift(2)
        breakout_short = df['Close'].shift(1) < df['_rolling_low'].shift(2)

        # Combine conditions
        long_signal = is_trending & rsi_bullish & breakout_long & vol_confirmed
        short_signal = is_trending & rsi_bearish & breakout_short & vol_confirmed

        # Create signal series
        signals = pd.Series(0, index=df.index, name='signal')
        signals.loc[long_signal] = 1

        # Only add shorts if longs_only is False
        # IMPORTANT: Backtest showed SHORTS -281.1% vs LONGS +130.8%
        if not self.longs_only:
            signals.loc[short_signal] = -1

        # Zero out warmup period (insufficient data for indicators)
        warmup = max(self.breakout_period, self.rsi_period,
                     self.atr_period, self.ker_period) + 5
        signals.iloc[:warmup] = 0

        return signals

    def generate_signals_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals with full indicator metadata for analysis/debugging.

        Returns DataFrame with additional columns:
            - signal: Trading signal (-1, 0, +1)
            - rsi, ker, atr, vol_ratio: Indicator values
            - is_trending: Boolean regime filter
            - rolling_high, rolling_low: Breakout levels
        """
        df = df.copy()

        # Calculate all indicators
        df['rsi'] = self._calculate_rsi(df['Close'])
        df['atr'] = self._calculate_atr(df['High'], df['Low'], df['Close'])
        df['ker'] = self._calculate_ker(df['Close'])
        df['vol_ratio'] = self._calculate_normalized_volume(df)
        df['rolling_high'] = df['High'].rolling(window=self.breakout_period).max()
        df['rolling_low'] = df['Low'].rolling(window=self.breakout_period).min()

        # Generate signals (uses internal copy)
        df['signal'] = self.generate_signals(df)

        # Add diagnostic flags
        df['is_trending'] = df['ker'].shift(1) > self.ker_threshold
        df['is_vol_confirmed'] = df['vol_ratio'].shift(1) > self.volume_threshold

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: ENTRY/EXIT & DYNAMIC STOP LOSS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeState:
    """Represents the current state of a trade."""
    ticker: str
    position_type: str  # 'long' or 'short'
    entry_price: float
    entry_bar: int
    entry_atr: float
    current_stop: float
    stop_phase: str  # 'initial', 'breakeven', 'trailing'
    highest_since_entry: float
    lowest_since_entry: float
    partial_exits: List[Dict]
    r_multiple: float = 0.0


class DynamicStopManager:
    """
    Manages multi-phase stop loss for letting winners run.

    Philosophy: "Cut losers fast, let winners run"

    Phases:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. INITIAL: Hard stop at Entry - (ATR * 1.5)
    2. BREAKEVEN: At +1R profit, move stop to entry + small buffer
    3. TRAILING: At +2R profit, use Chandelier Exit (only moves in profit direction)

    Usage:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        stop_mgr = DynamicStopManager()
        stop_info = stop_mgr.calculate_stop(
            position_type='long',
            entry_price=100,
            current_price=106,
            current_atr=2,
            highest_since_entry=108,
            lowest_since_entry=99
        )
    """

    def __init__(
        self,
        initial_atr_mult: float = 2.0,      # OPTIMIZADO: era 1.5
        breakeven_trigger_r: float = 1.0,
        trail_trigger_r: float = 2.0,
        trail_atr_mult: float = 2.5,
        chandelier_period: int = 22,
        breakeven_buffer_atr: float = 0.1
    ):
        """
        Initialize DynamicStopManager.

        Args:
            initial_atr_mult: ATR multiplier for initial stop (default 2.0, optimizado)
            breakeven_trigger_r: R-multiple to trigger breakeven (default 1.0)
            trail_trigger_r: R-multiple to start trailing (default 2.0)
            trail_atr_mult: ATR multiplier for trailing stop (default 2.5)
            chandelier_period: Lookback for highest high/lowest low (default 22)
            breakeven_buffer_atr: Buffer above/below entry for BE stop (default 0.1)
        """
        self.initial_atr_mult = initial_atr_mult
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trail_trigger_r = trail_trigger_r
        self.trail_atr_mult = trail_atr_mult
        self.chandelier_period = chandelier_period
        self.breakeven_buffer_atr = breakeven_buffer_atr

    def calculate_initial_stop(self, entry_price: float, entry_atr: float,
                                position_type: str) -> float:
        """Calculate initial hard stop."""
        risk = entry_atr * self.initial_atr_mult
        if position_type == 'long':
            return entry_price - risk
        else:
            return entry_price + risk

    def calculate_r_multiple(self, entry_price: float, current_price: float,
                              entry_atr: float, position_type: str) -> float:
        """Calculate current R-multiple (profit in terms of initial risk)."""
        initial_risk = entry_atr * self.initial_atr_mult
        if initial_risk <= 0:
            return 0.0

        if position_type == 'long':
            profit = current_price - entry_price
        else:
            profit = entry_price - current_price

        return profit / initial_risk

    def calculate_stop(
        self,
        position_type: str,
        entry_price: float,
        current_price: float,
        current_atr: float,
        highest_since_entry: float,
        lowest_since_entry: float,
        entry_atr: Optional[float] = None
    ) -> Dict:
        """
        Calculate current stop level based on trade progress.

        Args:
            position_type: 'long' or 'short'
            entry_price: Original entry price
            current_price: Current market price
            current_atr: Current ATR value
            highest_since_entry: Highest price since entry
            lowest_since_entry: Lowest price since entry
            entry_atr: ATR at entry (defaults to current_atr if not provided)

        Returns:
            dict with:
                - stop_price: Current stop level
                - phase: 'initial', 'breakeven', or 'trailing'
                - r_multiple: Current profit in R terms
                - risk_pct: Distance to stop as % of current price
        """
        if entry_atr is None:
            entry_atr = current_atr

        # Calculate R-multiple
        initial_risk = entry_atr * self.initial_atr_mult
        r_multiple = self.calculate_r_multiple(
            entry_price, current_price, entry_atr, position_type
        )

        if position_type == 'long':
            # Phase 1: Initial stop
            initial_stop = entry_price - initial_risk
            stop_price = initial_stop
            phase = 'initial'

            # Phase 2: Breakeven
            if r_multiple >= self.breakeven_trigger_r:
                stop_price = entry_price + (current_atr * self.breakeven_buffer_atr)
                phase = 'breakeven'

            # Phase 3: Trailing (Chandelier)
            if r_multiple >= self.trail_trigger_r:
                chandelier = highest_since_entry - (current_atr * self.trail_atr_mult)
                stop_price = max(stop_price, chandelier)  # Only moves up
                phase = 'trailing'

        else:  # short
            initial_stop = entry_price + initial_risk
            stop_price = initial_stop
            phase = 'initial'

            if r_multiple >= self.breakeven_trigger_r:
                stop_price = entry_price - (current_atr * self.breakeven_buffer_atr)
                phase = 'breakeven'

            if r_multiple >= self.trail_trigger_r:
                chandelier = lowest_since_entry + (current_atr * self.trail_atr_mult)
                stop_price = min(stop_price, chandelier)  # Only moves down
                phase = 'trailing'

        # Calculate risk percentage
        risk_pct = abs(current_price - stop_price) / current_price * 100 if current_price > 0 else 0

        return {
            'stop_price': round(stop_price, 2),
            'phase': phase,
            'r_multiple': round(r_multiple, 2),
            'risk_pct': round(risk_pct, 2),
            'initial_risk': round(initial_risk, 2)
        }

    def check_stop_hit(self, stop_price: float, bar_low: float, bar_high: float,
                        position_type: str) -> bool:
        """Check if stop was hit during a bar."""
        if position_type == 'long':
            return bar_low <= stop_price
        else:
            return bar_high >= stop_price


def calculate_chandelier_exit(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    period: int = 22,
    multiplier: float = 2.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Chandelier Exit - trailing stop from highest high / lowest low.

    Args:
        high, low, close: OHLC data
        atr: ATR series
        period: Lookback for extremes
        multiplier: ATR multiplier

    Returns:
        Tuple of (long_exit, short_exit) Series
    """
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()

    long_exit = highest - (atr * multiplier)
    short_exit = lowest + (atr * multiplier)

    return long_exit, short_exit


def calculate_entry_levels(df: pd.DataFrame, signal: pd.Series,
                           atr: pd.Series) -> pd.DataFrame:
    """
    Calculate entry prices for each signal.

    Args:
        df: OHLCV DataFrame
        signal: Signal series from MomentumEngine
        atr: ATR series

    Returns:
        DataFrame with entry information
    """
    entries = pd.DataFrame(index=df.index)

    # Entry on next bar's open (no look-ahead)
    entries['entry_price'] = df['Open'].shift(-1)

    # Alternative: Limit at breakout level + buffer
    buffer = atr * 0.1  # 10% of ATR buffer

    rolling_high = df['High'].rolling(window=20).max()
    rolling_low = df['Low'].rolling(window=20).min()

    entries['limit_long'] = rolling_high + buffer
    entries['limit_short'] = rolling_low - buffer

    # Copy signal
    entries['signal'] = signal

    return entries


def calculate_scale_out_levels(
    entry_price: float,
    atr: float,
    position_type: str = 'long',
    atr_multiplier: float = 2.0,    # OPTIMIZADO: era 1.5
    target_r_mult: float = 3.0      # OPTIMIZADO: target a 3R
) -> Dict:
    """
    Calculate levels for partial profit taking.

    Scale-out strategy (OPTIMIZADO):
    - Target principal a 3R (PF=1.46)
    - 50% at 3R (lock in profit)
    - 50% trailing (let it run for fat tails)

    Args:
        entry_price: Entry price
        atr: ATR at entry
        position_type: 'long' or 'short'
        atr_multiplier: Multiplier for initial risk (R) - default 2.0
        target_r_mult: Target in R multiples - default 3.0

    Returns:
        dict with target levels and percentages
    """
    R = atr * atr_multiplier  # Initial risk

    if position_type == 'long':
        return {
            'target_1': {'price': round(entry_price + (R * target_r_mult), 2), 'pct': 50, 'r': target_r_mult},
            'target_2': {'price': round(entry_price + (R * target_r_mult * 1.5), 2), 'pct': 25, 'r': target_r_mult * 1.5},
            'target_3': {'price': None, 'pct': 25, 'r': 'trailing'},  # Runner
            'initial_stop': round(entry_price - R, 2),
            'breakeven_level': round(entry_price + R, 2),
        }
    else:
        return {
            'target_1': {'price': round(entry_price - (R * target_r_mult), 2), 'pct': 50, 'r': target_r_mult},
            'target_2': {'price': round(entry_price - (R * target_r_mult * 1.5), 2), 'pct': 25, 'r': target_r_mult * 1.5},
            'target_3': {'price': None, 'pct': 25, 'r': 'trailing'},
            'initial_stop': round(entry_price + R, 2),
            'breakeven_level': round(entry_price - R, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ═══════════════════════════════════════════════════════════════════════════
    # PARÁMETROS OPTIMIZADOS v3.0 - TEST B WINNER (Feb 2026)
    # ═══════════════════════════════════════════════════════════════════════════
    # Resultados TEST B (40% vol, 30 barras):
    #   - 6 meses:  +72.4% return, PF 1.55, Max DD -22%
    #   - 12 meses: +61.7% return, PF 1.42, Max DD -25%
    # ═══════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # LONGS ONLY MODE (CRITICAL - based on backtest analysis)
    # ═══════════════════════════════════════════════════════════════════════════
    # Backtest showed: LONGS +130.8% P&L vs SHORTS -281.1% P&L
    # In bull markets, shorts destroy portfolio. Default to LONGS ONLY.
    'longs_only': True,         # ⚠️ IMPORTANTE: Solo operar LONGS

    # Regime filter
    'ker_threshold': 0.40,      # Optimizado: 0.40 (trending filter)
    'ker_period': 10,

    # Signal generation
    'breakout_period': 20,      # Optimizado: 20 barras
    'volume_threshold': 1.3,    # Optimizado: 1.3x normalized volume
    'rsi_threshold': 50,        # Optimizado: 50 (más sensible que 55)
    'rsi_max': 75,              # NUEVO: Evitar entradas en sobrecompra (RSI > 75)
    'rsi_period': 14,

    # Position sizing - AUMENTADO para más exposición
    'target_vol_annual': 0.40,      # CAMBIO: 0.20 → 0.40 (2x más exposición)
    'max_leverage': 1.0,

    # Stop management - V3 FINAL (Feb 2026)
    # ⚠️ SIN STOP LOSS - El backtest demostró que el stop destruye el edge
    'use_stop': False,          # CRÍTICO: No usar stop loss
    'initial_atr_mult': 2.0,    # Solo referencia (no se usa)
    'breakeven_trigger_r': 1.0, # Solo referencia (no se usa)
    'trail_trigger_r': 2.0,     # Solo referencia (no se usa)
    'trail_atr_mult': 2.5,      # Solo referencia (no se usa)
    'chandelier_period': 22,    # Solo referencia (no se usa)

    # Target - Solo referencia (no se usa con time exit)
    'target_r_mult': 3.0,

    # Time management - V3 FINAL
    'max_hold_bars': 45,        # 45 barras (~7.5 días) - V3 winner

    # Risk
    'max_positions': 5,
    'max_correlation': 0.7,
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE / DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Run demonstration with synthetic data."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║           MOMENTUM BREAKOUT MODULE - DEMO                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='4h')

    price = 100
    prices = [price]
    for _ in range(499):
        price *= (1 + np.random.randn() * 0.015)
        prices.append(price)

    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.randn() * 0.008)) for p in prices],
        'Low': [p * (1 - abs(np.random.randn() * 0.008)) for p in prices],
        'Close': prices,
        'Volume': [1000000 * (1 + np.random.randn() * 0.3) for _ in prices]
    }, index=dates)

    # Ensure High >= Open/Close and Low <= Open/Close
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    print(f"Asset Universe: {len(ASSETS)} instruments")
    print(f"  - Crypto: {len(CRYPTO_TICKERS)}")
    print(f"  - US Stocks: {len(US_STOCK_TICKERS)}")
    print(f"  - EU Stocks: {len(EU_TICKERS)}")
    print(f"  - Commodities: {len(COMMODITY_TICKERS)}")

    # Run engine
    print("\n" + "="*70)
    print("SIGNAL GENERATION TEST")
    print("="*70)

    engine = MomentumEngine(ker_threshold=0.35, volume_threshold=1.2)
    signals = engine.generate_signals(df)

    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()

    print(f"Total bars: {len(df)}")
    print(f"Long signals: {long_signals}")
    print(f"Short signals: {short_signals}")
    print(f"Signal rate: {(long_signals + short_signals) / len(df) * 100:.1f}%")

    # Position sizing example
    print("\n" + "="*70)
    print("POSITION SIZING TEST")
    print("="*70)

    # BTC example
    print("\nBTC at $45,000, ATR $1,500 (high vol):")
    size = calculate_position_size(
        account_balance=10000,
        current_atr=1500,
        price=45000,
        is_crypto=True,
        target_vol_annual=0.20
    )
    print(f"  Units: {size['units']:.6f} BTC")
    print(f"  Notional: ${size['notional']:,.2f}")
    print(f"  Portfolio risk: {size['risk_pct']:.2f}%")
    print(f"  Leverage: {size['leverage']:.2f}x")

    # BTC low vol
    print("\nBTC at $45,000, ATR $800 (low vol):")
    size = calculate_position_size(
        account_balance=10000,
        current_atr=800,
        price=45000,
        is_crypto=True,
        target_vol_annual=0.20
    )
    print(f"  Units: {size['units']:.6f} BTC")
    print(f"  Notional: ${size['notional']:,.2f}")
    print(f"  Portfolio risk: {size['risk_pct']:.2f}%")

    # MNQ example
    print("\nMNQ at 18,000, ATR 50 pts:")
    size = calculate_position_size(
        account_balance=10000,
        current_atr=50,
        price=18000,
        is_crypto=False,
        contract_multiplier=2  # $2/point
    )
    print(f"  Contracts: {size['units']:.2f}")
    print(f"  Notional: ${size['notional']:,.2f}")
    print(f"  Portfolio risk: {size['risk_pct']:.2f}%")

    # Stop management example
    print("\n" + "="*70)
    print("DYNAMIC STOP MANAGEMENT TEST")
    print("="*70)

    stop_mgr = DynamicStopManager()

    # Simulate trade progression
    scenarios = [
        {'price': 100, 'high': 100, 'low': 98, 'desc': 'Entry'},
        {'price': 102, 'high': 102, 'low': 100, 'desc': '+0.7R'},
        {'price': 103, 'high': 103, 'low': 101, 'desc': '+1R (BE trigger)'},
        {'price': 106, 'high': 106, 'low': 104, 'desc': '+2R (Trail start)'},
        {'price': 110, 'high': 112, 'low': 108, 'desc': '+3.3R'},
        {'price': 108, 'high': 110, 'low': 107, 'desc': 'Pullback'},
    ]

    entry_price = 100
    entry_atr = 3
    print(f"\nLong entry at ${entry_price}, ATR=${entry_atr}")
    print(f"Initial Risk (R) = ${entry_atr * 1.5:.2f}")
    print("-" * 60)

    highest = entry_price
    for s in scenarios:
        highest = max(highest, s['high'])
        stop_info = stop_mgr.calculate_stop(
            position_type='long',
            entry_price=entry_price,
            current_price=s['price'],
            current_atr=entry_atr,
            highest_since_entry=highest,
            lowest_since_entry=98,
            entry_atr=entry_atr
        )
        print(f"{s['desc']:20} | Price: ${s['price']:6.2f} | "
              f"Stop: ${stop_info['stop_price']:6.2f} | "
              f"Phase: {stop_info['phase']:10} | "
              f"R: {stop_info['r_multiple']:+.2f}")

    # Scale-out levels
    print("\n" + "="*70)
    print("SCALE-OUT LEVELS")
    print("="*70)

    levels = calculate_scale_out_levels(entry_price=100, atr=3, position_type='long')
    print(f"\nEntry: $100, ATR: $3, Initial Stop: ${levels['initial_stop']}")
    print(f"Breakeven at: ${levels['breakeven_level']} (+1R)")
    print(f"Target 1 (33%): ${levels['target_1']['price']} (+{levels['target_1']['r']}R)")
    print(f"Target 2 (33%): ${levels['target_2']['price']} (+{levels['target_2']['r']}R)")
    print(f"Target 3 (34%): Trailing stop")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    demo()
