"""
Universe Screening Module (STEP 1 of Strategy)

Identifies dividend-paying US stocks meeting the strategy criteria:
- Market cap > $1B
- Minimum 3 consecutive years of dividend payments
- Average daily volume > $1M
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
        market_cap_min: float = 1_000_000_000,  # $1B
        min_dividend_years: int = 3,
        avg_daily_volume_min: float = 1_000_000,  # $1M
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
            logger.info(f"Fetched {len(symbols)} S&P 500 constituents from Wikipedia")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching S&P 500 list: {e}")
            logger.info("Using hardcoded S&P 500 list as fallback")
            return self._get_sp500_fallback()

    def _get_sp500_fallback(self) -> List[str]:
        """Hardcoded S&P 500 list as fallback when Wikipedia fetch fails."""
        # S&P 500 constituents as of late 2024 (cleaned for yfinance format)
        return [
            # Technology
            'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE', 'CSCO', 'ACN',
            'IBM', 'INTC', 'INTU', 'QCOM', 'TXN', 'AMAT', 'NOW', 'ADI', 'LRCX', 'MU',
            'KLAC', 'APH', 'SNPS', 'CDNS', 'MSI', 'ROP', 'FTNT', 'NXPI', 'MCHP', 'TEL',
            'HPQ', 'HPE', 'KEYS', 'ON', 'FSLR', 'GLW', 'TYL', 'ZBRA', 'TRMB', 'TER',
            'SWKS', 'JBL', 'NTAP', 'WDC', 'AKAM', 'JNPR', 'GEN', 'FFIV', 'EPAM', 'QRVO',
            # Healthcare
            'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'DHR', 'AMGN',
            'ISRG', 'ELV', 'BMY', 'MDT', 'GILD', 'SYK', 'VRTX', 'CVS', 'CI', 'REGN',
            'BSX', 'ZTS', 'BDX', 'HCA', 'MCK', 'EW', 'A', 'IQV', 'GEHC', 'IDXX',
            'HUM', 'CAH', 'BAX', 'BIIB', 'CNC', 'DXCM', 'MTD', 'RMD', 'HOLX', 'TFX',
            'MOH', 'ALGN', 'LH', 'WAT', 'VTRS', 'CRL', 'TECH', 'XRAY', 'HSIC', 'DGX',
            'RVTY', 'BIO', 'INCY', 'CTLT', 'DVA',
            # Financials
            'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'C',
            'AXP', 'PGR', 'BLK', 'SCHW', 'MMC', 'CB', 'ICE', 'CME', 'USB', 'PNC',
            'AON', 'MCO', 'TFC', 'AJG', 'MET', 'AFL', 'TRV', 'AIG', 'PRU', 'ALL',
            'MSCI', 'MTB', 'FIS', 'COF', 'NDAQ', 'FITB', 'BK', 'STT', 'TROW', 'DFS',
            'HBAN', 'RF', 'WTW', 'KEY', 'CINF', 'NTRS', 'CFG', 'L', 'BRO', 'RJF',
            'CBOE', 'GL', 'EG', 'FDS', 'AIZ', 'RE', 'JKHY', 'BEN', 'LNC', 'ZION',
            'IVZ', 'MKTX',
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'BKNG', 'SBUX', 'TJX', 'CMG',
            'ORLY', 'MAR', 'AZO', 'DHI', 'GM', 'ROST', 'YUM', 'F', 'LEN', 'HLT',
            'GRMN', 'EBAY', 'APTV', 'GPC', 'ULTA', 'TSCO', 'DRI', 'PHM', 'KMX', 'POOL',
            'BBY', 'DECK', 'NVR', 'RCL', 'CCL', 'LKQ', 'BWA', 'TPR', 'WYNN', 'LVS',
            'MGM', 'EXPE', 'HAS', 'CZR', 'NCLH', 'MHK', 'ETSY', 'RL', 'WHR', 'BBWI',
            # Consumer Staples
            'PG', 'COST', 'KO', 'PEP', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
            'GIS', 'EL', 'STZ', 'SYY', 'ADM', 'KHC', 'MNST', 'HSY', 'KDP', 'TGT',
            'DG', 'DLTR', 'KR', 'CAG', 'CLX', 'MKC', 'TSN', 'SJM', 'HRL', 'CPB',
            'K', 'CHD', 'KVUE', 'TAP', 'BG', 'LW', 'BF-B',
            # Industrials
            'CAT', 'GE', 'HON', 'UNP', 'RTX', 'DE', 'UPS', 'ETN', 'BA', 'LMT',
            'ADP', 'GD', 'ITW', 'NOC', 'WM', 'MMM', 'EMR', 'CSX', 'NSC', 'JCI',
            'PH', 'TT', 'CTAS', 'CMI', 'FDX', 'CARR', 'PCAR', 'RSG', 'FAST', 'CPRT',
            'ODFL', 'AME', 'GWW', 'ROK', 'VRSK', 'DAL', 'OTIS', 'PWR', 'XYL', 'URI',
            'TDG', 'HWM', 'IR', 'PAYX', 'WAB', 'EFX', 'DOV', 'HUBB', 'UAL', 'IEX',
            'GPN', 'PNR', 'EXPD', 'SWK', 'MAS', 'JBHT', 'LUV', 'BR', 'J', 'TXT',
            'LDOS', 'SNA', 'NDSN', 'BALL', 'AAL', 'AOS', 'CHRW', 'ALLE', 'GNRC', 'PAYC',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB',
            'KMI', 'HES', 'DVN', 'BKR', 'HAL', 'FANG', 'CTRA', 'OKE', 'TRGP', 'EQT',
            'APA', 'MRO',
            # Utilities
            'NEE', 'SO', 'DUK', 'SRE', 'AEP', 'D', 'PCG', 'EXC', 'XEL', 'ED',
            'WEC', 'EIX', 'AWK', 'PPL', 'DTE', 'ES', 'ETR', 'AEE', 'FE', 'CMS',
            'CEG', 'ATO', 'EVRG', 'CNP', 'NRG', 'NI', 'LNT', 'PNW',
            # Materials
            'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DOW',
            'DD', 'PPG', 'CTVA', 'IFF', 'ALB', 'CE', 'FMC', 'PKG', 'CF', 'IP',
            'LYB', 'MOS', 'EMN', 'AVY', 'BALL', 'SEE', 'AMCR', 'WRK',
            # Communication Services
            'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR',
            'EA', 'WBD', 'OMC', 'TTWO', 'LYV', 'MTCH', 'IPG', 'NWSA', 'NWS', 'FOXA',
            'FOX', 'PARA',
            # Real Estate (excluding most REITs, keeping some diversified)
            'AMT', 'PLD', 'EQIX', 'CCI', 'PSA', 'WELL', 'SBAC', 'DLR', 'SPG', 'CBRE',
            'CSGP', 'EXR', 'AVB', 'WY', 'ARE', 'MAA', 'EQR', 'VTR', 'UDR', 'IRM',
            'ESS', 'INVH', 'HST', 'KIM', 'REG', 'BXP', 'CPT', 'DOC', 'FRT',
        ]

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

    def get_extended_universe(self) -> List[str]:
        """
        Get an extended list of potential dividend stocks.
        Combines S&P 500, dividend aristocrats, and other large-cap dividend payers.

        Returns:
            List of unique ticker symbols
        """
        # Start with S&P 500
        universe = set(self.get_sp500_constituents())

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
        save: bool = True
    ) -> pd.DataFrame:
        """
        Screen all stocks in the universe.

        Args:
            symbols: List of symbols to screen (default: extended universe)
            max_workers: Number of parallel workers
            save: Whether to save results

        Returns:
            DataFrame with qualifying stocks
        """
        if symbols is None:
            symbols = self.get_extended_universe()

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
