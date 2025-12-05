"""
Data Loader Module

Provides efficient access to locally stored market data for:
- Backtesting
- Strategy development
- Signal generation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from functools import lru_cache

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and provides access to locally stored market data."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.

        Args:
            data_dir: Root directory for data storage

        Note on stock splits and dividends:
            When yfinance downloads data with auto_adjust=True (our default),
            BOTH prices AND dividends are automatically adjusted for stock splits.
            For example, AAPL's pre-split $0.82 dividend (Q3 2020) is stored as
            $0.205 to match the 4:1 split-adjusted prices. No additional adjustment
            is needed.
        """
        self.data_dir = Path(data_dir)
        self.prices_dir = self.data_dir / "prices"
        self.dividends_dir = self.data_dir / "dividends"
        self.fundamentals_dir = self.data_dir / "fundamentals"
        self.universe_dir = self.data_dir / "universe"
        self.reference_dir = self.data_dir / "reference"

        # Cache for frequently accessed data
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._dividend_cache: Dict[str, pd.DataFrame] = {}
        self._fundamentals_cache: Dict[str, Dict] = {}

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available price data."""
        if not self.prices_dir.exists():
            return []
        return sorted([f.stem for f in self.prices_dir.glob('*.parquet')])

    def get_universe(self) -> pd.DataFrame:
        """Load the screened universe of dividend stocks."""
        universe_file = self.universe_dir / "universe.parquet"
        if not universe_file.exists():
            raise FileNotFoundError(
                f"Universe file not found: {universe_file}. "
                "Run universe screening first with: python -m src.data.download_all"
            )
        return pd.read_parquet(universe_file)

    def get_universe_symbols(self) -> List[str]:
        """Get list of symbols in the screened universe."""
        try:
            df = self.get_universe()
            return df['symbol'].tolist()
        except FileNotFoundError:
            # Fall back to available price data
            return self.get_available_symbols()

    def load_price_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load price data for a single symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        if use_cache and symbol in self._price_cache:
            df = self._price_cache[symbol]
        else:
            # Load from file
            price_file = self.prices_dir / f"{symbol}.parquet"
            if not price_file.exists():
                logger.warning(f"Price data not found for {symbol}")
                return None

            df = pd.read_parquet(price_file)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Cache it
            if use_cache:
                self._price_cache[symbol] = df

        # Filter by date range
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]

        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]

        return df

    def load_prices_multi(
        self,
        symbols: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        column: str = 'close',
        align: bool = True
    ) -> pd.DataFrame:
        """
        Load price data for multiple symbols into a single DataFrame.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            column: Price column to use ('open', 'high', 'low', 'close')
            align: Whether to align all series (drop rows with any NaN)

        Returns:
            DataFrame with symbols as columns
        """
        result = pd.DataFrame()

        for symbol in symbols:
            df = self.load_price_data(symbol, start_date, end_date)
            if df is not None and column in df.columns:
                result[symbol] = df[column]

        if align and not result.empty:
            result = result.dropna()

        return result

    def load_returns(
        self,
        symbols: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        method: str = 'simple'  # 'simple' or 'log'
    ) -> pd.DataFrame:
        """
        Load returns for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date
            end_date: End date
            method: Return calculation method ('simple' or 'log')

        Returns:
            DataFrame with daily returns
        """
        prices = self.load_prices_multi(symbols, start_date, end_date)

        if prices.empty:
            return prices

        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns.dropna()

    def load_dividend_data(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load dividend data for a single symbol.

        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data

        Returns:
            DataFrame with dividend data
        """
        # Check cache
        if use_cache and symbol in self._dividend_cache:
            return self._dividend_cache[symbol]

        # Load from file
        div_file = self.dividends_dir / f"{symbol}.parquet"
        if not div_file.exists():
            logger.debug(f"Dividend data not found for {symbol}")
            return None

        df = pd.read_parquet(div_file)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Cache it
        if use_cache:
            self._dividend_cache[symbol] = df

        return df

    def load_fundamentals(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Load fundamental data for a single symbol.

        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data

        Returns:
            Dictionary with fundamental data
        """
        # Check cache
        if use_cache and symbol in self._fundamentals_cache:
            return self._fundamentals_cache[symbol]

        # Load from file
        fund_file = self.fundamentals_dir / f"{symbol}.json"
        if not fund_file.exists():
            logger.debug(f"Fundamental data not found for {symbol}")
            return None

        with open(fund_file, 'r') as f:
            data = json.load(f)

        # Cache it
        if use_cache:
            self._fundamentals_cache[symbol] = data

        return data

    def load_fundamentals_df(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load fundamentals for multiple symbols as a DataFrame.

        Args:
            symbols: List of symbols (default: all available)

        Returns:
            DataFrame with fundamentals
        """
        if symbols is None:
            symbols = [f.stem for f in self.fundamentals_dir.glob('*.json')]

        data = []
        for symbol in symbols:
            fund = self.load_fundamentals(symbol)
            if fund is not None:
                data.append(fund)

        return pd.DataFrame(data)

    def load_reference_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load reference/benchmark data (e.g., SPY, sector ETFs).

        Args:
            symbol: Reference symbol (without ^ prefix)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data
        """
        ref_file = self.reference_dir / f"{symbol}.parquet"
        if not ref_file.exists():
            logger.warning(f"Reference data not found for {symbol}")
            return None

        df = pd.read_parquet(ref_file)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Filter by date range
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]

        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]

        return df

    def get_spy(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> Optional[pd.DataFrame]:
        """Convenience method to load SPY data from the 10-year CSV file."""
        # Use the correct 10-year SPY close prices CSV
        spy_csv_path = self.data_dir / "spy_10y_close.csv"

        if spy_csv_path.exists():
            # Parse the CSV (skip first 2 header rows)
            df = pd.read_csv(spy_csv_path, skiprows=2)
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()

            # Filter by date range
            if start_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                df = df[df.index >= start_date]

            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                df = df[df.index <= end_date]

            return df
        else:
            # Fall back to parquet if CSV not available
            logger.warning("spy_10y_close.csv not found, falling back to SPY.parquet")
            return self.load_reference_data('SPY', start_date, end_date)

    def calculate_dividend_metrics(self, symbol: str) -> Optional[Dict]:
        """
        Calculate dividend metrics for screening.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with dividend metrics
        """
        dividends = self.load_dividend_data(symbol)
        prices = self.load_price_data(symbol)

        if dividends is None or prices is None:
            return None

        try:
            # Group by year
            annual_divs = dividends.groupby('year')['dividend'].sum()

            if len(annual_divs) < 3:
                return None

            # Calculate YoY growth
            yoy_growth = annual_divs.pct_change().dropna()
            growth_std = yoy_growth.std()
            growth_mean = yoy_growth.mean()

            # Current yield
            current_price = prices['close'].iloc[-1]
            latest_annual_div = annual_divs.iloc[-1]
            current_yield = latest_annual_div / current_price

            # Load fundamentals for payout ratio
            fund = self.load_fundamentals(symbol)
            payout_ratio = fund.get('payout_ratio') if fund else None

            return {
                'symbol': symbol,
                'growth_std': growth_std,
                'growth_mean': growth_mean,
                'current_yield': current_yield,
                'payout_ratio': payout_ratio,
                'annual_dividend': latest_annual_div,
                'years_of_data': len(annual_divs)
            }

        except Exception as e:
            logger.error(f"Error calculating dividend metrics for {symbol}: {e}")
            return None

    def get_sector_stocks(self, sector: str) -> List[str]:
        """
        Get list of stocks in a specific sector.

        Args:
            sector: Sector name

        Returns:
            List of symbols in that sector
        """
        try:
            universe = self.get_universe()
            sector_stocks = universe[universe['sector'] == sector]['symbol'].tolist()
            return sector_stocks
        except FileNotFoundError:
            # Fall back to fundamentals
            fund_df = self.load_fundamentals_df()
            if 'sector' in fund_df.columns:
                return fund_df[fund_df['sector'] == sector]['symbol'].tolist()
            return []

    def get_sectors(self) -> List[str]:
        """Get list of unique sectors."""
        try:
            universe = self.get_universe()
            return universe['sector'].unique().tolist()
        except FileNotFoundError:
            fund_df = self.load_fundamentals_df()
            if 'sector' in fund_df.columns:
                return fund_df['sector'].unique().tolist()
            return []

    def get_data_coverage(self) -> pd.DataFrame:
        """
        Get summary of data coverage for all symbols.

        Returns:
            DataFrame with coverage info
        """
        symbols = self.get_available_symbols()

        data = []
        for symbol in symbols:
            prices = self.load_price_data(symbol, use_cache=False)
            dividends = self.load_dividend_data(symbol, use_cache=False)
            fundamentals = self.load_fundamentals(symbol, use_cache=False)

            row = {
                'symbol': symbol,
                'has_prices': prices is not None,
                'price_days': len(prices) if prices is not None else 0,
                'price_start': prices.index.min() if prices is not None else None,
                'price_end': prices.index.max() if prices is not None else None,
                'has_dividends': dividends is not None,
                'dividend_count': len(dividends) if dividends is not None else 0,
                'has_fundamentals': fundamentals is not None
            }
            data.append(row)

        return pd.DataFrame(data)

    def clear_cache(self):
        """Clear all cached data."""
        self._price_cache.clear()
        self._dividend_cache.clear()
        self._fundamentals_cache.clear()
        logger.info("Cache cleared")

    def get_aligned_data_for_basket(
        self,
        symbols: List[str],
        min_days: int = 504
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Get aligned price data for a basket of symbols.
        Useful for cointegration analysis.

        Args:
            symbols: List of stock symbols
            min_days: Minimum required trading days

        Returns:
            Tuple of (aligned price DataFrame, list of valid symbols)
        """
        prices = self.load_prices_multi(symbols, align=True)

        if len(prices) < min_days:
            logger.warning(f"Insufficient aligned data: {len(prices)} < {min_days} days")
            return pd.DataFrame(), []

        # Use only the last min_days
        prices = prices.tail(min_days)

        valid_symbols = prices.columns.tolist()

        return prices, valid_symbols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test loader
    loader = DataLoader()

    print("Available symbols:", loader.get_available_symbols()[:10])

    # Test loading data
    if loader.get_available_symbols():
        symbol = loader.get_available_symbols()[0]
        prices = loader.load_price_data(symbol)
        print(f"\n{symbol} price data:")
        print(prices.tail())
