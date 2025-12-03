"""
Universe Screening Module (STEP 1 of Strategy)

Identifies dividend-paying US stocks meeting the strategy criteria:
- Market cap > $5B
- Minimum 5 consecutive years of dividend payments
- Average daily volume > $10M
- Excludes: REITs, MLPs, special dividends only
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Set
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Known REITs and MLPs to exclude (common examples - can be extended)
KNOWN_REITS = {
    'O', 'VICI', 'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'WELL', 'DLR',
    'AVB', 'EQR', 'VTR', 'ARE', 'BXP', 'UDR', 'HST', 'SUI', 'PEAK', 'MAA',
    'EXR', 'CUBE', 'REG', 'ESS', 'IRM', 'WPC', 'NNN', 'STOR', 'KIM', 'FRT',
    'CPT', 'AIV', 'INVH', 'ACC', 'SLG', 'DEI', 'HIW', 'JBGS', 'CUZ', 'OFC'
}

KNOWN_MLPS = {
    'EPD', 'ET', 'MPLX', 'WES', 'PAA', 'PAGP', 'USAC', 'NS', 'HESM', 'DCP',
    'AM', 'CEQP', 'ENLC', 'SMLP', 'GEL', 'HEP', 'BSM', 'KNOP', 'NGL', 'USDP'
}


class UniverseScreener:
    """Screens for dividend-paying stocks meeting strategy criteria."""

    def __init__(
        self,
        market_cap_min: float = 5_000_000_000,  # $5B
        min_dividend_years: int = 5,
        avg_daily_volume_min: float = 10_000_000,  # $10M
        data_dir: str = "data/universe"
    ):
        """
        Initialize universe screener.

        Args:
            market_cap_min: Minimum market cap in dollars
            min_dividend_years: Minimum years of consecutive dividend payments
            avg_daily_volume_min: Minimum average daily dollar volume
            data_dir: Directory to store universe data
        """
        self.market_cap_min = market_cap_min
        self.min_dividend_years = min_dividend_years
        self.avg_daily_volume_min = avg_daily_volume_min
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_sp500_constituents(self) -> List[str]:
        """
        Get S&P 500 constituents as a starting point.

        Returns:
            List of ticker symbols
        """
        try:
            # Use Wikipedia table for S&P 500 constituents
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            # Clean up symbols (remove dots for BRK.B -> BRK-B format for yfinance)
            symbols = [s.replace('.', '-') for s in symbols]
            return symbols
        except Exception as e:
            logger.error(f"Error fetching S&P 500 list: {e}")
            return []

    def get_sp400_midcap_constituents(self) -> List[str]:
        """
        Get S&P 400 Mid-Cap constituents for expanded universe.

        Returns:
            List of ticker symbols
        """
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
            tables = pd.read_html(url)
            sp400_table = tables[0]
            # The column might be 'Symbol' or 'Ticker Symbol'
            symbol_col = 'Symbol' if 'Symbol' in sp400_table.columns else 'Ticker symbol'
            symbols = sp400_table[symbol_col].tolist()
            symbols = [s.replace('.', '-') for s in symbols]
            return symbols
        except Exception as e:
            logger.error(f"Error fetching S&P 400 list: {e}")
            return []

    def get_sp600_smallcap_constituents(self) -> List[str]:
        """
        Get S&P 600 Small-Cap constituents for expanded universe.

        Returns:
            List of ticker symbols
        """
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
            tables = pd.read_html(url)
            sp600_table = tables[0]
            symbol_col = 'Symbol' if 'Symbol' in sp600_table.columns else 'Ticker symbol'
            symbols = sp600_table[symbol_col].tolist()
            symbols = [s.replace('.', '-') for s in symbols]
            return symbols
        except Exception as e:
            logger.error(f"Error fetching S&P 600 list: {e}")
            return []

    def get_dividend_aristocrats(self) -> List[str]:
        """
        Get Dividend Aristocrats (25+ years of dividend increases).
        These are guaranteed to pass the 5-year dividend requirement.

        Returns:
            List of ticker symbols
        """
        # Manually curated list of dividend aristocrats (as of 2024)
        aristocrats = [
            'MMM', 'ABT', 'ABBV', 'AFL', 'APD', 'ALB', 'AMCR', 'AOS', 'ATO',
            'ADP', 'BDX', 'BRO', 'BF-B', 'CAH', 'CAT', 'CVX', 'CB', 'CINF',
            'CTAS', 'CLX', 'KO', 'CL', 'ED', 'DOV', 'ECL', 'EMR', 'ESS',
            'EXPD', 'XOM', 'FAST', 'FRT', 'BEN', 'GD', 'GPC', 'HRL', 'ITW',
            'JNJ', 'KMB', 'KVUE', 'LEG', 'LIN', 'LOW', 'MKC', 'MCD', 'MDT',
            'NEE', 'NKE', 'NUE', 'PNR', 'PBCT', 'PEP', 'PPG', 'PG', 'O',
            'ROP', 'SPGI', 'SHW', 'SJM', 'SWK', 'SYY', 'TGT', 'T', 'TROW',
            'WBA', 'WMT', 'WST', 'WEC'
        ]
        return aristocrats

    def get_hardcoded_sp500(self) -> List[str]:
        """
        Hardcoded S&P 500 constituents (as fallback when Wikipedia is blocked).
        """
        return [
            'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE',
            'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL',
            'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET',
            'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI',
            'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBWI',
            'BBY', 'BDX', 'BEN', 'BF-B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK',
            'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH',
            'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE',
            'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX',
            'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO',
            'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX',
            'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL',
            'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR',
            'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM',
            'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH',
            'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY',
            'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FBHS',
            'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT', 'FMC',
            'FOX', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS',
            'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS',
            'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT',
            'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM',
            'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU',
            'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
            'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY',
            'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE',
            'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC',
            'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA',
            'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET',
            'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO',
            'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT',
            'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM',
            'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE',
            'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE',
            'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PARA', 'PAYC', 'PAYX', 'PCAR',
            'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM',
            'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL',
            'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL',
            'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP',
            'ROST', 'RSG', 'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM',
            'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX',
            'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY',
            'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR',
            'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT',
            'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
            'V', 'VFC', 'VICI', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR',
            'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC',
            'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WTW', 'WY', 'WYNN',
            'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS'
        ]

    def get_hardcoded_midcaps(self) -> List[str]:
        """
        Hardcoded list of mid-cap dividend stocks for expanded universe.
        Includes S&P 400 Mid-Cap stocks and other quality mid-cap dividend payers.
        """
        return [
            # S&P 400 Mid-Cap dividend payers
            'ACHC', 'ACIW', 'ACM', 'AEL', 'AFG', 'AGO', 'AIT', 'ALE', 'ALLY', 'AM',
            'AMG', 'AMED', 'AROC', 'ASB', 'ATGE', 'ATI', 'AVA', 'AXTA', 'AYI', 'BANF',
            'BC', 'BCO', 'BDC', 'BERY', 'BHE', 'BJ', 'BPOP', 'BRKR', 'BXC', 'CALM',
            'CASY', 'CBT', 'CBSH', 'CC', 'CHDN', 'CHE', 'CHH', 'CIT', 'CIVI', 'CKH',
            'CLH', 'CMD', 'CMP', 'CNM', 'CNO', 'CNX', 'CRI', 'CRS', 'CW', 'CWT',
            'CXW', 'DAR', 'DCI', 'DECK', 'DKS', 'DLX', 'DNB', 'DORM', 'EBC', 'EEFT',
            'EGP', 'EME', 'ENR', 'ENS', 'ENSG', 'EPC', 'EPRT', 'EVR', 'EXP', 'FAF',
            'FBP', 'FCFS', 'FHN', 'FL', 'FLO', 'FN', 'FNB', 'FRME', 'G', 'GATX',
            'GBCI', 'GEO', 'GFF', 'GGG', 'GHC', 'GMS', 'GNW', 'GNTX', 'GPK', 'GVA',
            'HBI', 'HE', 'HEI', 'HGV', 'HLI', 'HNI', 'HP', 'HR', 'HUN', 'HWC',
            'IBOC', 'ICUI', 'IDA', 'IESC', 'INGR', 'IPGP', 'ITCI', 'JBL', 'JBT', 'JEF',
            'JHG', 'JLL', 'JNPR', 'KAR', 'KBH', 'KEX', 'KGS', 'KMT', 'KNX', 'KRC',
            'KRG', 'KWR', 'LAD', 'LANC', 'LB', 'LFUS', 'LGF-A', 'LGIH', 'LHCG', 'LKFN',
            'LNN', 'LPG', 'LSTR', 'LXP', 'MANH', 'MAN', 'MAT', 'MASI', 'MATX', 'MHO',
            'MIDD', 'MKSI', 'MLKN', 'MLR', 'MMSI', 'MMS', 'MOD', 'MORN', 'MPW', 'MSA',
            'MSM', 'MTH', 'MTN', 'MTZ', 'MUR', 'MUSA', 'NCI', 'NEU', 'NFG', 'NJR',
            'NMIH', 'NOV', 'NR', 'NVT', 'NWE', 'NYCB', 'NYT', 'OC', 'OFG', 'OGE',
            'OHI', 'OI', 'OII', 'OLN', 'OLP', 'ORA', 'ORIT', 'OSK', 'OTTR', 'OUT',
            'OZK', 'PACW', 'PATK', 'PBF', 'PB', 'PEGA', 'PINC', 'PII', 'PIPR', 'PK',
            'PLNT', 'PNM', 'PNFP', 'POST', 'POR', 'POWL', 'PPC', 'PRGO', 'PRGS', 'PRI',
            'PRIM', 'PRK', 'PRSP', 'PSB', 'PSN', 'PTC', 'PVH', 'QGEN', 'R', 'RAMP',
            'RAVN', 'RC', 'RBC', 'RDN', 'REYN', 'RGA', 'RGS', 'RHP', 'RIG', 'RLI',
            'RNR', 'RPM', 'RRC', 'RYN', 'SAIC', 'SAM', 'SANM', 'SBCF', 'SCI', 'SEIC',
            'SF', 'SFBS', 'SFNC', 'SFM', 'SGRY', 'SHC', 'SI', 'SIGI', 'SITE', 'SKX',
            'SLG', 'SLM', 'SM', 'SMAR', 'SMG', 'SNV', 'SONO', 'SPNT', 'SR', 'SSD',
            'SSB', 'SSRM', 'STC', 'STNG', 'STOR', 'STRA', 'SUM', 'SXI', 'TBBK', 'TDC',
            'TCBI', 'TDW', 'TEN', 'TEX', 'TFSL', 'THC', 'THO', 'TILE', 'TNET', 'TNL',
            'TOL', 'TPH', 'TPX', 'TR', 'TRNO', 'TRN', 'TTC', 'TTEC', 'TTI', 'TUP',
            'TWST', 'TY', 'UBSI', 'UCBI', 'UE', 'UFPI', 'UGI', 'UMBF', 'UNFI', 'UNIT',
            'UNVR', 'UPBD', 'URBN', 'USM', 'UTHR', 'VALE', 'VAR', 'VCEL', 'VET',
            'VG', 'VIR', 'VIRT', 'VITR', 'VLY', 'VMI', 'VNOM', 'VNT', 'VSH', 'VSEC',
            'WAFD', 'WAL', 'WBS', 'WCC', 'WD', 'WEN', 'WEX', 'WH', 'WLK', 'WNS',
            'WOLF', 'WOR', 'WPC', 'WPM', 'WSC', 'WSM', 'WSO', 'WTFC', 'WTS', 'WWW',
            'X', 'XPER', 'YETI', 'ZBRA', 'ZD', 'ZEN', 'ZWS'
        ]

    def get_extended_universe(self, include_midcaps: bool = False) -> List[str]:
        """
        Get an extended list of potential dividend stocks.
        Combines S&P 500, dividend aristocrats, and other large-cap dividend payers.

        Args:
            include_midcaps: If True, also includes S&P 400 Mid-Cap constituents

        Returns:
            List of unique ticker symbols
        """
        # Start with S&P 500 (try Wikipedia first, fallback to hardcoded)
        universe = set(self.get_sp500_constituents())
        if len(universe) < 100:
            logger.info("Using hardcoded S&P 500 list (Wikipedia unavailable)")
            universe = set(self.get_hardcoded_sp500())

        # Add S&P 400 Mid-Caps if requested
        if include_midcaps:
            sp400 = self.get_sp400_midcap_constituents()
            if len(sp400) < 50:
                logger.info("Using hardcoded mid-cap list (Wikipedia unavailable)")
                sp400 = self.get_hardcoded_midcaps()
            logger.info(f"Adding {len(sp400)} S&P 400 Mid-Cap stocks to screening pool")
            universe.update(sp400)

        # Add dividend aristocrats
        universe.update(self.get_dividend_aristocrats())

        # Add other well-known large-cap dividend stocks not in S&P 500
        additional_large_caps = [
            # Tech
            'AAPL', 'MSFT', 'AVGO', 'TXN', 'QCOM', 'IBM', 'INTC', 'CSCO', 'ORCL',
            # Healthcare
            'UNH', 'LLY', 'PFE', 'MRK', 'AMGN', 'BMY', 'GILD',
            # Consumer
            'HD', 'COST', 'WMT', 'PG', 'KO', 'PEP', 'PM', 'MO',
            # Financials (non-REITs)
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'USB', 'PNC',
            'TFC', 'MTB', 'FITB', 'KEY', 'RF', 'HBAN', 'CFG', 'ZION',
            # Insurance
            'BRK-B', 'MMC', 'AON', 'AJG', 'TRV', 'ALL', 'MET', 'PRU', 'AIG',
            # Industrials
            'UNP', 'UPS', 'HON', 'RTX', 'LMT', 'GE', 'MMM', 'CAT', 'DE',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PSX', 'VLO', 'MPC',
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC',
            # Communications
            'VZ', 'T', 'CMCSA', 'TMUS',
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'NUE', 'DD'
        ]
        universe.update(additional_large_caps)

        # Remove known REITs and MLPs
        universe = universe - KNOWN_REITS - KNOWN_MLPS

        return sorted(list(universe))

    def check_dividend_history(self, ticker: yf.Ticker) -> Dict:
        """
        Check dividend history for a stock.

        Args:
            ticker: yfinance Ticker object

        Returns:
            Dictionary with dividend analysis
        """
        try:
            # Get dividend history (last 6 years to check 5 consecutive years)
            dividends = ticker.dividends

            if dividends.empty:
                return {'has_dividends': False, 'consecutive_years': 0}

            # Group dividends by year
            dividends_df = dividends.to_frame()
            dividends_df.index = pd.to_datetime(dividends_df.index)
            dividends_df['year'] = dividends_df.index.year

            yearly_divs = dividends_df.groupby('year')['Dividends'].sum()

            # Check for consecutive years of dividend payments
            current_year = datetime.now().year
            consecutive_years = 0

            for year in range(current_year - 1, current_year - 10, -1):
                if year in yearly_divs.index and yearly_divs[year] > 0:
                    consecutive_years += 1
                else:
                    break

            return {
                'has_dividends': True,
                'consecutive_years': consecutive_years,
                'latest_annual_dividend': yearly_divs.iloc[-1] if not yearly_divs.empty else 0,
                'dividend_history': yearly_divs.to_dict()
            }

        except Exception as e:
            logger.debug(f"Error checking dividends: {e}")
            return {'has_dividends': False, 'consecutive_years': 0}

    def screen_stock(self, symbol: str) -> Optional[Dict]:
        """
        Screen a single stock against all criteria.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock info if it passes, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or 'marketCap' not in info:
                return None

            # Check market cap
            market_cap = info.get('marketCap', 0)
            if market_cap < self.market_cap_min:
                return None

            # Check if it's a REIT or MLP
            quote_type = info.get('quoteType', '')
            sector = info.get('sector', '')
            industry = info.get('industry', '')

            if quote_type == 'REIT' or 'REIT' in str(industry).upper():
                return None
            if 'MLP' in str(industry).upper() or 'PARTNERSHIP' in str(industry).upper():
                return None
            if symbol in KNOWN_REITS or symbol in KNOWN_MLPS:
                return None

            # Check average volume
            avg_volume = info.get('averageVolume', 0)
            current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
            avg_dollar_volume = avg_volume * current_price

            if avg_dollar_volume < self.avg_daily_volume_min:
                return None

            # Check dividend history
            div_info = self.check_dividend_history(ticker)

            if not div_info['has_dividends']:
                return None
            if div_info['consecutive_years'] < self.min_dividend_years:
                return None

            # Calculate current yield
            dividend_yield = info.get('dividendYield', 0) or 0
            trailing_annual_dividend = info.get('trailingAnnualDividendRate', 0) or 0

            # Get sector info (GICS-like classification)
            gics_sector = info.get('sector', 'Unknown')
            gics_industry = info.get('industry', 'Unknown')

            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'market_cap': market_cap,
                'avg_dollar_volume': avg_dollar_volume,
                'avg_volume': avg_volume,
                'current_price': current_price,
                'dividend_yield': dividend_yield,
                'trailing_annual_dividend': trailing_annual_dividend,
                'consecutive_dividend_years': div_info['consecutive_years'],
                'sector': gics_sector,
                'industry': gics_industry,
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD')
            }

        except Exception as e:
            logger.debug(f"Error screening {symbol}: {e}")
            return None

    def screen_universe(
        self,
        symbols: Optional[List[str]] = None,
        max_workers: int = 10,
        save: bool = True,
        include_midcaps: bool = False
    ) -> pd.DataFrame:
        """
        Screen all stocks in the universe.

        Args:
            symbols: List of symbols to screen (default: extended universe)
            max_workers: Number of parallel workers
            save: Whether to save results
            include_midcaps: If True, includes S&P 400 Mid-Caps in screening

        Returns:
            DataFrame with qualifying stocks
        """
        if symbols is None:
            symbols = self.get_extended_universe(include_midcaps=include_midcaps)

        logger.info(f"Screening {len(symbols)} stocks...")

        qualifying_stocks = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.screen_stock, symbol): symbol
                for symbol in symbols
            }

            for future in tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Screening stocks"):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        qualifying_stocks.append(result)
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")

        # Create DataFrame
        df = pd.DataFrame(qualifying_stocks)

        if not df.empty:
            # Sort by market cap
            df = df.sort_values('market_cap', ascending=False).reset_index(drop=True)

            logger.info(f"Found {len(df)} qualifying stocks")

            # Log sector distribution
            sector_counts = df['sector'].value_counts()
            logger.info(f"Sector distribution:\n{sector_counts}")

            if save:
                self.save_universe(df)

        return df

    def save_universe(self, df: pd.DataFrame, filename: str = "universe.parquet"):
        """Save universe to parquet file."""
        filepath = self.data_dir / filename
        df.to_parquet(filepath, index=False)

        # Also save as CSV for easy inspection
        csv_path = self.data_dir / filename.replace('.parquet', '.csv')
        df.to_csv(csv_path, index=False)

        # Save metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_stocks': len(df),
            'market_cap_min': self.market_cap_min,
            'min_dividend_years': self.min_dividend_years,
            'avg_daily_volume_min': self.avg_daily_volume_min,
            'sectors': df['sector'].value_counts().to_dict()
        }

        with open(self.data_dir / 'universe_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved universe to {filepath}")

    def load_universe(self, filename: str = "universe.parquet") -> pd.DataFrame:
        """Load universe from parquet file."""
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Universe file not found: {filepath}")

        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} stocks from universe")

        return df

    def get_symbols(self, filename: str = "universe.parquet") -> List[str]:
        """Get list of symbols from saved universe."""
        df = self.load_universe(filename)
        return df['symbol'].tolist()


if __name__ == "__main__":
    # Test universe screening
    logging.basicConfig(level=logging.INFO)

    screener = UniverseScreener()
    universe = screener.screen_universe()
    print(f"\nQualifying stocks: {len(universe)}")
    print(universe.head(20))
