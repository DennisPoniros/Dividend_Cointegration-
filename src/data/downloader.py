"""
Data Downloader Module

Downloads and stores historical data locally using:
- Alpaca Markets API (primary source for price data)
- yfinance (fallback and dividend data)

Data stored includes:
- OHLCV price data (504+ trading days for cointegration)
- Dividend history (5+ years)
- Fundamental data snapshots
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

# Check if Alpaca is available
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockTradesRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Using yfinance only.")


class DataDownloader:
    """Downloads and stores market data locally."""

    def __init__(
        self,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        data_dir: str = "data",
        use_alpaca: bool = True
    ):
        """
        Initialize data downloader.

        Args:
            alpaca_api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            alpaca_secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
            data_dir: Root directory for data storage
            use_alpaca: Whether to use Alpaca (falls back to yfinance if unavailable)
        """
        self.data_dir = Path(data_dir)
        self.prices_dir = self.data_dir / "prices"
        self.dividends_dir = self.data_dir / "dividends"
        self.fundamentals_dir = self.data_dir / "fundamentals"
        self.reference_dir = self.data_dir / "reference"

        # Create directories
        for d in [self.prices_dir, self.dividends_dir, self.fundamentals_dir, self.reference_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize Alpaca client if available and keys provided
        self.alpaca_client = None
        self.use_alpaca = use_alpaca and ALPACA_AVAILABLE

        if self.use_alpaca:
            api_key = alpaca_api_key or os.getenv('ALPACA_API_KEY')
            secret_key = alpaca_secret_key or os.getenv('ALPACA_SECRET_KEY')

            if api_key and secret_key:
                try:
                    self.alpaca_client = StockHistoricalDataClient(api_key, secret_key)
                    logger.info("Alpaca client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Alpaca client: {e}")
                    self.alpaca_client = None
            else:
                logger.info("Alpaca API keys not provided. Using yfinance only.")
                self.use_alpaca = False

    def download_price_data_alpaca(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Download price data using Alpaca.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpaca_client:
            return None

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )

            bars = self.alpaca_client.get_stock_bars(request)

            if symbol not in bars.data or not bars.data[symbol]:
                return None

            # Convert to DataFrame
            data = []
            for bar in bars.data[symbol]:
                data.append({
                    'date': bar.timestamp.date(),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap if hasattr(bar, 'vwap') else None,
                    'trade_count': bar.trade_count if hasattr(bar, 'trade_count') else None
                })

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            return df

        except Exception as e:
            logger.debug(f"Alpaca error for {symbol}: {e}")
            return None

    def download_price_data_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Download price data using yfinance.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            # auto_adjust=True adjusts BOTH prices AND dividends for stock splits.
            # E.g., AAPL's pre-4:1-split $0.82 dividend is stored as $0.205.
            # This ensures dividend yields are calculated correctly against
            # split-adjusted prices without any additional adjustment.
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                return None

            # Standardize column names
            df.columns = df.columns.str.lower()

            # Rename columns to match our schema
            column_map = {
                'stock splits': 'stock_splits',
            }
            df = df.rename(columns=column_map)

            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    return None

            # Select and order columns
            cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
            if 'dividends' in df.columns:
                cols_to_keep.append('dividends')
            if 'stock_splits' in df.columns:
                cols_to_keep.append('stock_splits')

            df = df[cols_to_keep]
            df.index = pd.to_datetime(df.index.date)
            df.index.name = 'date'

            return df

        except Exception as e:
            logger.warning(f"yfinance error for {symbol}: {e}")
            return None

    def download_price_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_days: int = 3780  # 15 years for extended backtesting
    ) -> Optional[pd.DataFrame]:
        """
        Download price data using best available source.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (default: lookback_days ago)
            end_date: End date (default: today)
            lookback_days: Days to look back if start_date not specified (default: 15 years)

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=lookback_days)

        df = None

        # Try Alpaca first
        if self.use_alpaca and self.alpaca_client:
            df = self.download_price_data_alpaca(symbol, start_date, end_date)
            if df is not None and len(df) > 0:
                df['source'] = 'alpaca'
                return df

        # Fall back to yfinance
        df = self.download_price_data_yfinance(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            df['source'] = 'yfinance'

        return df

    def download_dividend_data(
        self,
        symbol: str,
        years: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Download dividend history using yfinance.

        Args:
            symbol: Stock ticker symbol
            years: Years of history to fetch (default: 10)

        Returns:
            DataFrame with dividend data
        """
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends

            if dividends.empty:
                return None

            # Filter to requested years
            # Convert index to timezone-naive to avoid comparison issues
            if dividends.index.tz is not None:
                dividends.index = dividends.index.tz_localize(None)
            start_date = datetime.now() - timedelta(days=years * 365)
            dividends = dividends[dividends.index >= start_date]

            df = dividends.to_frame()
            df.columns = ['dividend']
            df.index.name = 'date'
            df.index = pd.to_datetime(df.index.date)

            # Add year column for analysis
            df['year'] = df.index.year

            return df

        except Exception as e:
            logger.warning(f"Dividend error for {symbol}: {e}")
            return None

    def download_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Download fundamental data snapshot using yfinance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with fundamental data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            # Extract relevant fundamentals
            fundamentals = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),
                'trailing_annual_dividend': info.get('trailingAnnualDividendRate'),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'fifty_day_average': info.get('fiftyDayAverage'),
                'two_hundred_day_average': info.get('twoHundredDayAverage'),
                'average_volume': info.get('averageVolume'),
                'average_volume_10d': info.get('averageVolume10days'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'revenue': info.get('totalRevenue'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
                'earnings_growth': info.get('earningsGrowth'),
                'revenue_growth': info.get('revenueGrowth'),
                'snapshot_date': datetime.now().isoformat()
            }

            return fundamentals

        except Exception as e:
            logger.warning(f"Fundamentals error for {symbol}: {e}")
            return None

    def download_single_stock(
        self,
        symbol: str,
        include_dividends: bool = True,
        include_fundamentals: bool = True
    ) -> Dict[str, bool]:
        """
        Download all data for a single stock.

        Args:
            symbol: Stock ticker symbol
            include_dividends: Whether to download dividend data
            include_fundamentals: Whether to download fundamental data

        Returns:
            Dictionary indicating success/failure for each data type
        """
        results = {'symbol': symbol, 'prices': False, 'dividends': False, 'fundamentals': False}

        # Download price data
        prices = self.download_price_data(symbol)
        if prices is not None and len(prices) >= 252:  # At least 1 year of data
            prices.to_parquet(self.prices_dir / f"{symbol}.parquet")
            results['prices'] = True

        # Download dividend data
        if include_dividends:
            dividends = self.download_dividend_data(symbol)
            if dividends is not None and len(dividends) > 0:
                dividends.to_parquet(self.dividends_dir / f"{symbol}.parquet")
                results['dividends'] = True

        # Download fundamentals
        if include_fundamentals:
            fundamentals = self.download_fundamentals(symbol)
            if fundamentals is not None:
                with open(self.fundamentals_dir / f"{symbol}.json", 'w') as f:
                    json.dump(fundamentals, f, indent=2, default=str)
                results['fundamentals'] = True

        return results

    def download_universe(
        self,
        symbols: List[str],
        max_workers: int = 5,
        include_dividends: bool = True,
        include_fundamentals: bool = True,
        delay: float = 0.5  # Rate limiting (increased to avoid Yahoo rate limits)
    ) -> pd.DataFrame:
        """
        Download data for all symbols in universe.

        Args:
            symbols: List of stock symbols
            max_workers: Number of parallel workers
            include_dividends: Whether to download dividend data
            include_fundamentals: Whether to download fundamental data
            delay: Delay between requests (rate limiting)

        Returns:
            DataFrame with download results
        """
        logger.info(f"Downloading data for {len(symbols)} symbols...")

        results = []
        consecutive_failures = 0
        max_consecutive_failures = 5  # If 5 in a row fail, likely rate limited

        # Use single-threaded for yfinance to avoid rate limiting
        for symbol in tqdm(symbols, desc="Downloading"):
            try:
                result = self.download_single_stock(
                    symbol,
                    include_dividends=include_dividends,
                    include_fundamentals=include_fundamentals
                )
                results.append(result)

                # Track consecutive failures for rate limit detection
                if not result['prices']:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Detected {consecutive_failures} consecutive failures - possible rate limit. Waiting 60s...")
                        time.sleep(60)
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0

                time.sleep(delay)  # Rate limiting
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'prices': False,
                    'dividends': False,
                    'fundamentals': False
                })
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"Detected {consecutive_failures} consecutive failures - possible rate limit. Waiting 60s...")
                    time.sleep(60)
                    consecutive_failures = 0

        # Create summary DataFrame
        df = pd.DataFrame(results)

        # Log summary
        logger.info(f"Download complete:")
        logger.info(f"  Prices: {df['prices'].sum()}/{len(df)} successful")
        logger.info(f"  Dividends: {df['dividends'].sum()}/{len(df)} successful")
        logger.info(f"  Fundamentals: {df['fundamentals'].sum()}/{len(df)} successful")

        # Save download log
        df.to_csv(self.data_dir / 'download_log.csv', index=False)

        # Save metadata
        metadata = {
            'last_download': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'prices_downloaded': int(df['prices'].sum()),
            'dividends_downloaded': int(df['dividends'].sum()),
            'fundamentals_downloaded': int(df['fundamentals'].sum()),
            'symbols_with_complete_data': int((df['prices'] & df['dividends']).sum())
        }

        with open(self.data_dir / 'download_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return df

    def download_reference_data(self) -> Dict[str, bool]:
        """
        Download reference/benchmark data (SPY, sector ETFs, etc.).

        Returns:
            Dictionary indicating success for each reference
        """
        reference_symbols = {
            'SPY': 'S&P 500 ETF (market hedge)',
            'QQQ': 'Nasdaq 100 ETF',
            'IWM': 'Russell 2000 ETF',
            'VTI': 'Total Stock Market ETF',
            'XLF': 'Financial Sector ETF',
            'XLK': 'Technology Sector ETF',
            'XLE': 'Energy Sector ETF',
            'XLU': 'Utilities Sector ETF',
            'XLI': 'Industrial Sector ETF',
            'XLV': 'Healthcare Sector ETF',
            'XLP': 'Consumer Staples ETF',
            'XLY': 'Consumer Discretionary ETF',
            'XLB': 'Materials Sector ETF',
            'XLRE': 'Real Estate Sector ETF',
            'XLC': 'Communication Services ETF',
            'VIG': 'Dividend Appreciation ETF',
            'SCHD': 'Dividend Value ETF',
            'DVY': 'Dividend Select ETF',
            'VYM': 'High Dividend Yield ETF',
            '^VIX': 'VIX Volatility Index',
            '^TNX': '10-Year Treasury Yield'
        }

        logger.info("Downloading reference data...")

        results = {}
        for symbol, description in tqdm(reference_symbols.items(), desc="Reference data"):
            try:
                # Handle special symbols
                clean_symbol = symbol.replace('^', '')

                prices = self.download_price_data_yfinance(
                    symbol,
                    datetime.now() - timedelta(days=756),
                    datetime.now()
                )

                if prices is not None and len(prices) > 0:
                    prices.to_parquet(self.reference_dir / f"{clean_symbol}.parquet")
                    results[symbol] = True
                    logger.debug(f"Downloaded {symbol}: {description}")
                else:
                    results[symbol] = False
                    logger.warning(f"No data for {symbol}")

                time.sleep(0.3)

            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                results[symbol] = False

        # Save reference metadata
        with open(self.reference_dir / 'reference_metadata.json', 'w') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'symbols': reference_symbols,
                'download_results': results
            }, f, indent=2)

        return results

    def get_downloaded_symbols(self) -> List[str]:
        """Get list of symbols with downloaded price data."""
        return [
            f.stem for f in self.prices_dir.glob('*.parquet')
        ]

    def verify_data_completeness(
        self,
        min_days: int = 504
    ) -> pd.DataFrame:
        """
        Verify that downloaded data meets minimum requirements.

        Args:
            min_days: Minimum trading days required

        Returns:
            DataFrame with verification results
        """
        results = []

        for parquet_file in self.prices_dir.glob('*.parquet'):
            symbol = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                trading_days = len(df)
                has_min_data = trading_days >= min_days

                # Check for data quality
                null_pct = df[['open', 'high', 'low', 'close', 'volume']].isnull().mean().mean()

                results.append({
                    'symbol': symbol,
                    'trading_days': trading_days,
                    'has_min_data': has_min_data,
                    'null_pct': null_pct,
                    'start_date': df.index.min().strftime('%Y-%m-%d'),
                    'end_date': df.index.max().strftime('%Y-%m-%d')
                })

            except Exception as e:
                logger.error(f"Error verifying {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'trading_days': 0,
                    'has_min_data': False,
                    'null_pct': 1.0,
                    'start_date': None,
                    'end_date': None
                })

        df = pd.DataFrame(results)

        # Log summary
        complete = df['has_min_data'].sum()
        logger.info(f"Data completeness: {complete}/{len(df)} symbols have {min_days}+ days")

        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test downloader
    downloader = DataDownloader()

    # Download a test symbol
    result = downloader.download_single_stock('AAPL')
    print(f"AAPL download result: {result}")
