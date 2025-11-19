"""
Sub-Basket Formation Module (STEP 2)

Implements HYBRID SECTOR-CORRELATION CLUSTERING:
1. Initial Sector Grouping (GICS Level 3)
2. Within-Sector Correlation Clustering (ρ ≥ 0.85)
3. Dividend Trait Screening (secondary filter)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import yaml
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


class BasketFormer:
    """Forms sub-baskets using sector-correlation clustering."""

    def __init__(self, config_path: str = "config/strategy_params.yaml"):
        """
        Initialize basket former.

        Args:
            config_path: Path to strategy parameters
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.basket_config = self.config['basket_formation']
        self.baskets = []

    def assign_sectors(self, universe_symbols: List[str], ibkr) -> Dict[str, str]:
        """
        Assign GICS Level 3 sectors to each stock.

        Args:
            universe_symbols: List of stock symbols
            ibkr: Connected IBKR instance

        Returns:
            Dictionary mapping symbols to GICS sectors
        """
        sector_map = {}

        for symbol in universe_symbols:
            try:
                contract = ibkr.get_contract(symbol)
                details = ibkr.ib.reqContractDetails(contract)

                if details:
                    # IBKR provides category/industry classification
                    # Map this to GICS-like sectors
                    detail = details[0]
                    sector = detail.category if detail.category else 'Unknown'
                    sector_map[symbol] = sector

                ibkr.ib.sleep(0.1)

            except Exception as e:
                logger.error(f"Error getting sector for {symbol}: {e}")
                sector_map[symbol] = 'Unknown'

        return sector_map

    def calculate_rolling_correlation(
        self,
        price_data: Dict[str, pd.DataFrame],
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            window: Rolling window in days (default: 252 = 1 year)

        Returns:
            Correlation matrix
        """
        # Align all price series
        symbols = list(price_data.keys())

        # Create aligned DataFrame
        price_df = pd.DataFrame()
        for symbol in symbols:
            if 'close' in price_data[symbol].columns:
                price_df[symbol] = price_data[symbol]['close']

        # Calculate returns
        returns = price_df.pct_change().dropna()

        # Calculate rolling correlation (using most recent window)
        if len(returns) >= window:
            recent_returns = returns.tail(window)
            corr_matrix = recent_returns.corr()
        else:
            logger.warning(f"Insufficient data for {window}-day correlation. Using all available data.")
            corr_matrix = returns.corr()

        return corr_matrix

    def cluster_by_correlation(
        self,
        symbols: List[str],
        corr_matrix: pd.DataFrame,
        threshold: float = 0.85,
        min_size: int = 6,
        max_size: int = 10
    ) -> List[List[str]]:
        """
        Cluster stocks by correlation using hierarchical clustering.

        Args:
            symbols: List of stock symbols
            corr_matrix: Correlation matrix
            threshold: Minimum correlation threshold (default: 0.85)
            min_size: Minimum cluster size
            max_size: Maximum cluster size

        Returns:
            List of clusters (each cluster is a list of symbols)
        """
        # Convert correlation to distance
        # Distance = 1 - correlation
        distance_matrix = 1 - corr_matrix.abs()

        # Ensure valid distance matrix
        distance_matrix = distance_matrix.fillna(1.0)
        distance_matrix = np.clip(distance_matrix.values, 0, 2)

        # Convert to condensed distance matrix for scipy
        condensed_dist = squareform(distance_matrix, checks=False)

        # Hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='average')

        # Form clusters at threshold
        cluster_distance = 1 - threshold
        labels = fcluster(linkage_matrix, cluster_distance, criterion='distance')

        # Group symbols by cluster
        clusters_dict = {}
        for symbol, label in zip(symbols, labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(symbol)

        # Filter clusters by size
        valid_clusters = [
            cluster for cluster in clusters_dict.values()
            if min_size <= len(cluster) <= max_size
        ]

        logger.info(f"Formed {len(valid_clusters)} valid correlation clusters from {len(symbols)} stocks")

        return valid_clusters

    def calculate_dividend_metrics(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        ibkr
    ) -> Optional[Dict]:
        """
        Calculate dividend metrics for screening.

        Args:
            symbol: Stock symbol
            price_data: Historical price data
            ibkr: Connected IBKR instance

        Returns:
            Dictionary with dividend metrics, or None if unavailable
        """
        try:
            # Get dividend history
            div_history = ibkr.get_dividend_history(symbol, years=5)

            if div_history.empty:
                return None

            # Calculate YoY growth
            div_history['year'] = pd.to_datetime(div_history.index).year
            annual_divs = div_history.groupby('year')['dividend'].sum()

            if len(annual_divs) < 3:
                return None

            yoy_growth = annual_divs.pct_change().dropna()
            growth_std = yoy_growth.std()

            # Calculate current yield
            current_price = price_data['close'].iloc[-1]
            recent_annual_div = annual_divs.iloc[-1]
            current_yield = recent_annual_div / current_price

            # Estimate payout ratio (simplified - would need earnings data)
            # For now, use a placeholder
            payout_ratio = None  # Would need to fetch earnings data

            return {
                'symbol': symbol,
                'growth_std': growth_std,
                'current_yield': current_yield,
                'payout_ratio': payout_ratio,
                'annual_dividend': recent_annual_div
            }

        except Exception as e:
            logger.error(f"Error calculating dividend metrics for {symbol}: {e}")
            return None

    def screen_dividend_traits(
        self,
        cluster: List[str],
        price_data: Dict[str, pd.DataFrame],
        ibkr
    ) -> List[str]:
        """
        Screen cluster by dividend traits.

        Criteria:
        - Dividend growth consistency: std of YoY growth < 15%
        - Payout ratio: 30-70%
        - Yield range: 1.5-6%

        Args:
            cluster: List of stock symbols in cluster
            price_data: Dictionary mapping symbols to price DataFrames
            ibkr: Connected IBKR instance

        Returns:
            List of symbols passing screening
        """
        passed_symbols = []

        for symbol in cluster:
            metrics = self.calculate_dividend_metrics(
                symbol,
                price_data[symbol],
                ibkr
            )

            if metrics is None:
                continue

            # Apply screening criteria
            growth_std_max = self.basket_config['dividend_screening']['growth_std_max']
            payout_min = self.basket_config['dividend_screening']['payout_ratio_min']
            payout_max = self.basket_config['dividend_screening']['payout_ratio_max']
            yield_min = self.basket_config['dividend_screening']['yield_min']
            yield_max = self.basket_config['dividend_screening']['yield_max']

            # Check growth consistency
            if metrics['growth_std'] > growth_std_max:
                continue

            # Check yield range
            if not (yield_min <= metrics['current_yield'] <= yield_max):
                continue

            # Check payout ratio (if available)
            if metrics['payout_ratio'] is not None:
                if not (payout_min <= metrics['payout_ratio'] <= payout_max):
                    continue

            # Passed all screens
            passed_symbols.append(symbol)

        return passed_symbols

    def form_baskets(
        self,
        universe_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        ibkr
    ) -> List[Dict]:
        """
        Form sub-baskets using complete methodology.

        Steps:
        1. Sector grouping
        2. Within-sector correlation clustering
        3. Dividend trait screening

        Args:
            universe_symbols: List of symbols in universe
            price_data: Dictionary mapping symbols to price DataFrames
            ibkr: Connected IBKR instance

        Returns:
            List of basket dictionaries
        """
        logger.info("Starting basket formation...")

        # Step 1: Sector grouping
        logger.info("Step 1: Assigning sectors...")
        sector_map = self.assign_sectors(universe_symbols, ibkr)

        # Group by sector
        sectors = {}
        for symbol, sector in sector_map.items():
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)

        # Filter sectors by minimum size
        min_sector_size = self.basket_config['min_stocks_per_sector']
        valid_sectors = {
            sector: symbols for sector, symbols in sectors.items()
            if len(symbols) >= min_sector_size
        }

        logger.info(f"Found {len(valid_sectors)} sectors with >= {min_sector_size} stocks")

        all_baskets = []

        # Step 2 & 3: For each sector, do correlation clustering and dividend screening
        for sector, symbols in valid_sectors.items():
            logger.info(f"Processing sector: {sector} ({len(symbols)} stocks)")

            # Get price data for this sector
            sector_price_data = {s: price_data[s] for s in symbols if s in price_data}

            if len(sector_price_data) < self.basket_config['correlation']['cluster_size_min']:
                logger.warning(f"Insufficient price data for sector {sector}")
                continue

            # Calculate correlation matrix
            corr_matrix = self.calculate_rolling_correlation(
                sector_price_data,
                window=self.basket_config['correlation']['window_days']
            )

            # Cluster by correlation
            clusters = self.cluster_by_correlation(
                list(sector_price_data.keys()),
                corr_matrix,
                threshold=self.basket_config['correlation']['threshold'],
                min_size=self.basket_config['correlation']['cluster_size_min'],
                max_size=self.basket_config['correlation']['cluster_size_max']
            )

            # Screen each cluster by dividend traits
            for cluster in clusters:
                screened_cluster = self.screen_dividend_traits(
                    cluster,
                    sector_price_data,
                    ibkr
                )

                # Only keep clusters that still meet size requirements after screening
                min_size = self.basket_config['correlation']['cluster_size_min']
                max_size = self.basket_config['correlation']['cluster_size_max']

                if min_size <= len(screened_cluster) <= max_size:
                    basket = {
                        'sector': sector,
                        'symbols': screened_cluster,
                        'size': len(screened_cluster),
                        'avg_correlation': corr_matrix.loc[screened_cluster, screened_cluster].mean().mean()
                    }
                    all_baskets.append(basket)
                    logger.info(f"Created basket in {sector}: {len(screened_cluster)} stocks")

        self.baskets = all_baskets
        logger.info(f"Basket formation complete: {len(self.baskets)} baskets created")

        return self.baskets

    def get_basket_price_data(
        self,
        basket_idx: int,
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get aligned price data for a specific basket.

        Args:
            basket_idx: Index of basket
            price_data: Dictionary mapping symbols to price DataFrames

        Returns:
            DataFrame with aligned close prices
        """
        if basket_idx >= len(self.baskets):
            raise ValueError(f"Basket index {basket_idx} out of range")

        basket = self.baskets[basket_idx]
        symbols = basket['symbols']

        # Create aligned price DataFrame
        price_df = pd.DataFrame()
        for symbol in symbols:
            if symbol in price_data and 'close' in price_data[symbol].columns:
                price_df[symbol] = price_data[symbol]['close']

        # Drop rows with any NaN
        price_df = price_df.dropna()

        return price_df

    def save_baskets(self, filepath: str):
        """Save baskets to CSV."""
        if not self.baskets:
            logger.warning("No baskets to save")
            return

        # Flatten baskets for CSV
        rows = []
        for i, basket in enumerate(self.baskets):
            rows.append({
                'basket_id': i,
                'sector': basket['sector'],
                'size': basket['size'],
                'symbols': ','.join(basket['symbols']),
                'avg_correlation': basket['avg_correlation']
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self.baskets)} baskets to {filepath}")

    def load_baskets(self, filepath: str):
        """Load baskets from CSV."""
        df = pd.read_csv(filepath)

        self.baskets = []
        for _, row in df.iterrows():
            basket = {
                'sector': row['sector'],
                'symbols': row['symbols'].split(','),
                'size': row['size'],
                'avg_correlation': row['avg_correlation']
            }
            self.baskets.append(basket)

        logger.info(f"Loaded {len(self.baskets)} baskets from {filepath}")
