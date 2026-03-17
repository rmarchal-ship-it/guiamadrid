#!/usr/bin/env python3
"""
SPY Correlation Analysis for Momentum Breakout Universe
========================================================
Calculates Pearson correlation of daily returns with SPY over the last 2 years.
Goal: Identify tickers that could bypass the SPY > SMA50 macro filter.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# ASSET UNIVERSE (from momentum_breakout.py)
# ═══════════════════════════════════════════════════════════════════════════════

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
    # COMMODITIES - ETFs (17)
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
    # INDEX ETFs - US (5) & LEVERAGED (3)
    'QQQ': {'name': 'Nasdaq-100', 'category': 'US_INDEX'},
    'TQQQ': {'name': 'Nasdaq 3x', 'category': 'US_INDEX_LEV'},
    'SPY': {'name': 'S&P 500', 'category': 'US_INDEX'},
    'SPXL': {'name': 'S&P 3x', 'category': 'US_INDEX_LEV'},
    'IWM': {'name': 'Russell 2000', 'category': 'US_INDEX'},
    'TNA': {'name': 'Russell 3x', 'category': 'US_INDEX_LEV'},
    'DIA': {'name': 'Dow Jones', 'category': 'US_INDEX'},
    'BITO': {'name': 'Bitcoin ETF', 'category': 'US_INDEX'},
    # ETFs SECTORIALES US (6)
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
    # FIXED INCOME & DIVERSIFIERS (8)
    'TLT': {'name': 'Treasury 20+ Year', 'category': 'FIXED_INCOME'},
    'IEF': {'name': 'Treasury 7-10 Year', 'category': 'FIXED_INCOME'},
    'SHY': {'name': 'Treasury 1-3 Year', 'category': 'FIXED_INCOME'},
    'TIP': {'name': 'TIPS (Inflation)', 'category': 'FIXED_INCOME'},
    'AGG': {'name': 'US Aggregate Bond', 'category': 'FIXED_INCOME'},
    'LQD': {'name': 'Investment Grade Corp', 'category': 'FIXED_INCOME'},
    'HYG': {'name': 'High Yield Corp', 'category': 'FIXED_INCOME'},
    'EMB': {'name': 'EM Sovereign Debt', 'category': 'FIXED_INCOME'},
}


def main():
    print("=" * 100)
    print("  SPY CORRELATION ANALYSIS - MOMENTUM BREAKOUT UNIVERSE")
    print("  Pearson Correlation of Daily Returns with SPY (Last 2 Years)")
    print("=" * 100)
    print()

    # Remove SPY from analysis list (we correlate against it)
    tickers_to_analyze = [t for t in ASSETS.keys() if t != 'SPY']
    all_tickers = tickers_to_analyze + ['SPY']

    print(f"Universe: {len(ASSETS)} total tickers ({len(tickers_to_analyze)} to analyze vs SPY)")
    print(f"Period: Last 2 years of daily data")
    print()

    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    print("Downloading data from Yahoo Finance...")
    print("(This may take a minute for 225+ tickers)")
    print()

    # Download in batches to avoid timeout issues
    batch_size = 50
    all_data = {}
    failed_tickers = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        batch_str = ' '.join(batch)
        print(f"  Downloading batch {i//batch_size + 1}/{(len(all_tickers)-1)//batch_size + 1}: {len(batch)} tickers...")

        try:
            data = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if len(batch) == 1:
                # Single ticker returns different format
                ticker = batch[0]
                if not data.empty:
                    all_data[ticker] = data['Close']
                else:
                    failed_tickers.append(ticker)
            else:
                close_data = data['Close']
                for ticker in batch:
                    if ticker in close_data.columns and close_data[ticker].notna().sum() > 50:
                        all_data[ticker] = close_data[ticker]
                    else:
                        failed_tickers.append(ticker)
        except Exception as e:
            print(f"    Error downloading batch: {e}")
            failed_tickers.extend(batch)

    print()

    if 'SPY' not in all_data:
        print("CRITICAL ERROR: Could not download SPY data!")
        return

    # Build returns DataFrame
    close_df = pd.DataFrame(all_data)
    returns_df = close_df.pct_change().dropna(how='all')

    # Get SPY returns
    spy_returns = returns_df['SPY'].dropna()

    print(f"Data range: {close_df.index[0].strftime('%Y-%m-%d')} to {close_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading days: {len(spy_returns)}")
    print(f"Successfully downloaded: {len(all_data) - 1} tickers (excl. SPY)")
    if failed_tickers:
        print(f"Failed to download ({len(failed_tickers)}): {', '.join(failed_tickers)}")
    print()

    # Calculate correlations
    results = []
    for ticker in tickers_to_analyze:
        if ticker not in returns_df.columns:
            continue

        ticker_returns = returns_df[ticker].dropna()

        # Align with SPY
        common_idx = spy_returns.index.intersection(ticker_returns.index)
        if len(common_idx) < 50:
            continue

        corr = spy_returns.loc[common_idx].corr(ticker_returns.loc[common_idx])
        results.append({
            'ticker': ticker,
            'name': ASSETS[ticker]['name'],
            'category': ASSETS[ticker]['category'],
            'correlation': corr,
            'data_points': len(common_idx),
        })

    # Sort by correlation (lowest first)
    results_df = pd.DataFrame(results).sort_values('correlation')

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: FULL SORTED LIST
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  FULL CORRELATION TABLE (sorted by correlation, lowest first)")
    print("=" * 100)
    print()
    print(f"{'#':>4}  {'Ticker':<12} {'Name':<28} {'Category':<22} {'Corr':>8}  {'Days':>5}  {'Flag':<12}")
    print("-" * 100)

    for i, row in enumerate(results_df.itertuples(), 1):
        flag = ""
        if row.correlation < 0:
            flag = "*** NEGATIVE"
        elif row.correlation < 0.15:
            flag = "** V.LOW"
        elif row.correlation < 0.30:
            flag = "* LOW"

        print(f"{i:>4}  {row.ticker:<12} {row.name:<28} {row.category:<22} {row.correlation:>8.4f}  {row.data_points:>5}  {flag:<12}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: MACRO FILTER BYPASS CANDIDATES
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("  MACRO FILTER BYPASS CANDIDATES")
    print("  These tickers DON'T correlate with SPY => SPY > SMA50 filter is irrelevant for them")
    print("=" * 100)

    # Negative correlation
    negative = results_df[results_df['correlation'] < 0]
    print()
    print(f"  NEGATIVE CORRELATION (corr < 0): {len(negative)} tickers")
    print("  " + "-" * 90)
    if len(negative) > 0:
        for row in negative.itertuples():
            print(f"    {row.ticker:<12} {row.name:<28} {row.category:<22} corr = {row.correlation:>8.4f}")
    else:
        print("    (none)")

    # Very low correlation
    very_low = results_df[(results_df['correlation'] >= 0) & (results_df['correlation'] < 0.15)]
    print()
    print(f"  VERY LOW CORRELATION (0 <= corr < 0.15): {len(very_low)} tickers")
    print("  " + "-" * 90)
    if len(very_low) > 0:
        for row in very_low.itertuples():
            print(f"    {row.ticker:<12} {row.name:<28} {row.category:<22} corr = {row.correlation:>8.4f}")
    else:
        print("    (none)")

    # Low correlation
    low = results_df[(results_df['correlation'] >= 0.15) & (results_df['correlation'] < 0.30)]
    print()
    print(f"  LOW CORRELATION (0.15 <= corr < 0.30): {len(low)} tickers")
    print("  " + "-" * 90)
    if len(low) > 0:
        for row in low.itertuples():
            print(f"    {row.ticker:<12} {row.name:<28} {row.category:<22} corr = {row.correlation:>8.4f}")
    else:
        print("    (none)")

    # Moderate-low
    mod_low = results_df[(results_df['correlation'] >= 0.30) & (results_df['correlation'] < 0.50)]
    print()
    print(f"  MODERATE-LOW CORRELATION (0.30 <= corr < 0.50): {len(mod_low)} tickers")
    print("  " + "-" * 90)
    if len(mod_low) > 0:
        for row in mod_low.itertuples():
            print(f"    {row.ticker:<12} {row.name:<28} {row.category:<22} corr = {row.correlation:>8.4f}")
    else:
        print("    (none)")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: GROUPED BY CATEGORY
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("  AVERAGE CORRELATION BY CATEGORY")
    print("=" * 100)
    print()

    cat_stats = results_df.groupby('category').agg(
        avg_corr=('correlation', 'mean'),
        min_corr=('correlation', 'min'),
        max_corr=('correlation', 'max'),
        count=('correlation', 'count')
    ).sort_values('avg_corr')

    print(f"{'Category':<22} {'Avg Corr':>10} {'Min':>10} {'Max':>10} {'Count':>6}  {'Assessment':<20}")
    print("-" * 85)

    for cat, row in cat_stats.iterrows():
        assessment = ""
        if row['avg_corr'] < 0:
            assessment = "INVERSE to SPY"
        elif row['avg_corr'] < 0.15:
            assessment = "INDEPENDENT of SPY"
        elif row['avg_corr'] < 0.30:
            assessment = "WEAKLY linked"
        elif row['avg_corr'] < 0.50:
            assessment = "MODERATELY linked"
        elif row['avg_corr'] < 0.70:
            assessment = "CORRELATED"
        else:
            assessment = "HIGHLY CORRELATED"

        print(f"{cat:<22} {row['avg_corr']:>10.4f} {row['min_corr']:>10.4f} {row['max_corr']:>10.4f} {int(row['count']):>6}  {assessment:<20}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: SUMMARY STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("  SUMMARY STATISTICS")
    print("=" * 100)
    print()

    total = len(results_df)
    n_negative = len(results_df[results_df['correlation'] < 0])
    n_very_low = len(results_df[(results_df['correlation'] >= 0) & (results_df['correlation'] < 0.15)])
    n_low = len(results_df[(results_df['correlation'] >= 0.15) & (results_df['correlation'] < 0.30)])
    n_mod_low = len(results_df[(results_df['correlation'] >= 0.30) & (results_df['correlation'] < 0.50)])
    n_corr = len(results_df[(results_df['correlation'] >= 0.50) & (results_df['correlation'] < 0.70)])
    n_high = len(results_df[results_df['correlation'] >= 0.70])

    print(f"  Total tickers analyzed:     {total}")
    print(f"  Mean correlation:           {results_df['correlation'].mean():.4f}")
    print(f"  Median correlation:         {results_df['correlation'].median():.4f}")
    print(f"  Std dev:                    {results_df['correlation'].std():.4f}")
    print()
    print(f"  Distribution:")
    print(f"    Negative  (< 0.00):       {n_negative:>4}  ({n_negative/total*100:>5.1f}%)  => BYPASS macro filter")
    print(f"    Very Low  (0.00 - 0.15):  {n_very_low:>4}  ({n_very_low/total*100:>5.1f}%)  => BYPASS macro filter")
    print(f"    Low       (0.15 - 0.30):  {n_low:>4}  ({n_low/total*100:>5.1f}%)  => Consider bypassing")
    print(f"    Mod-Low   (0.30 - 0.50):  {n_mod_low:>4}  ({n_mod_low/total*100:>5.1f}%)  => Partial benefit from filter")
    print(f"    Correlated(0.50 - 0.70):  {n_corr:>4}  ({n_corr/total*100:>5.1f}%)  => Keep macro filter")
    print(f"    High      (>= 0.70):      {n_high:>4}  ({n_high/total*100:>5.1f}%)  => Macro filter essential")
    print()

    bypass_candidates = n_negative + n_very_low + n_low
    print(f"  TOTAL BYPASS CANDIDATES (corr < 0.30): {bypass_candidates} tickers ({bypass_candidates/total*100:.1f}%)")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  RECOMMENDATION FOR MACRO FILTER")
    print("=" * 100)
    print()
    print("  Tickers with correlation < 0.30 to SPY should be EXEMPT from the")
    print("  'SPY > SMA50' macro filter, as their price action is largely")
    print("  independent of the S&P 500 regime.")
    print()
    print("  Exempt tickers list (copy-paste for code):")
    bypass_list = results_df[results_df['correlation'] < 0.30]['ticker'].tolist()
    print(f"  MACRO_FILTER_EXEMPT = {bypass_list}")
    print()

    # Also show the corr < 0.50 list for a more aggressive bypass
    print("  Extended exempt list (corr < 0.50, more aggressive):")
    bypass_ext = results_df[results_df['correlation'] < 0.50]['ticker'].tolist()
    print(f"  MACRO_FILTER_EXEMPT_EXTENDED = {bypass_ext}")
    print()
    print("=" * 100)


if __name__ == '__main__':
    main()
