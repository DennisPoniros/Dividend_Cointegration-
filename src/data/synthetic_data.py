"""
Synthetic Data Generator for Backtesting

Generates realistic price data with known cointegration relationships.
This allows testing the strategy on data where we KNOW the ground truth.

Key features:
- Ornstein-Uhlenbeck mean-reverting spreads
- Configurable half-life and volatility
- Realistic price dynamics with trends and volatility clustering
- No forward-looking bias in data generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic price data with known cointegration relationships.

    The generated data has:
    1. Realistic price dynamics (random walk with drift)
    2. Known cointegration relationships (mean-reverting spreads)
    3. Configurable parameters for testing different scenarios
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def generate_ornstein_uhlenbeck(
        self,
        n_days: int,
        theta: float = 0.1,  # Mean reversion speed
        mu: float = 0.0,      # Long-term mean
        sigma: float = 0.02,  # Volatility
        x0: float = 0.0       # Initial value
    ) -> np.ndarray:
        """
        Generate Ornstein-Uhlenbeck process.

        dx = theta * (mu - x) * dt + sigma * dW

        Half-life = ln(2) / theta

        Args:
            n_days: Number of days
            theta: Mean reversion speed (higher = faster reversion)
            mu: Long-term mean
            sigma: Volatility
            x0: Initial value

        Returns:
            Array of OU process values
        """
        dt = 1.0  # Daily
        x = np.zeros(n_days)
        x[0] = x0

        for t in range(1, n_days):
            dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * self.rng.randn()
            x[t] = x[t-1] + dx

        return x

    def generate_cointegrated_pair(
        self,
        n_days: int,
        base_price: float = 100.0,
        drift: float = 0.0001,      # Daily drift
        volatility: float = 0.02,    # Daily volatility
        half_life: float = 15.0,     # Half-life in days
        spread_volatility: float = 0.01,
        hedge_ratio: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a cointegrated pair of price series.

        Stock A follows a random walk.
        Stock B = hedge_ratio * A + mean_reverting_spread

        Args:
            n_days: Number of days
            base_price: Starting price
            drift: Daily drift (annualized ~2.5% for 0.0001)
            volatility: Daily volatility
            half_life: Half-life of mean reversion (days)
            spread_volatility: Volatility of the spread
            hedge_ratio: Cointegration hedge ratio

        Returns:
            Tuple of (prices_A, prices_B)
        """
        # Convert half-life to theta
        theta = np.log(2) / half_life

        # Generate random walk for stock A
        log_returns_a = self.rng.normal(drift, volatility, n_days)
        log_prices_a = np.cumsum(log_returns_a)
        prices_a = base_price * np.exp(log_prices_a)

        # Generate mean-reverting spread
        spread = self.generate_ornstein_uhlenbeck(
            n_days,
            theta=theta,
            mu=0.0,
            sigma=spread_volatility,
            x0=0.0
        )

        # Stock B = hedge_ratio * A + spread
        prices_b = hedge_ratio * prices_a + spread * base_price

        return prices_a, prices_b

    def generate_cointegrated_basket(
        self,
        n_days: int,
        n_stocks: int = 4,
        base_prices: Optional[List[float]] = None,
        sector_drift: float = 0.0001,
        stock_volatility: float = 0.02,
        half_life: float = 15.0,
        spread_volatility: float = 0.008,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Generate a basket of cointegrated stocks.

        Creates n_stocks where the weighted combination mean-reverts.

        Args:
            n_days: Number of days
            n_stocks: Number of stocks in basket
            base_prices: Starting prices for each stock
            sector_drift: Common sector drift
            stock_volatility: Individual stock volatility
            half_life: Half-life of cointegration spread
            spread_volatility: Volatility of the spread
            weights: Cointegration weights (if None, uses equal weights)

        Returns:
            Tuple of (price_dict, weights)
        """
        if base_prices is None:
            base_prices = [100.0 + i * 20 for i in range(n_stocks)]

        if weights is None:
            # Create weights that sum to zero (market neutral)
            weights = np.array([1.0] + [-1.0 / (n_stocks - 1)] * (n_stocks - 1))

        weights = weights / np.sum(np.abs(weights))

        # Convert half-life to theta
        theta = np.log(2) / half_life

        # Generate common sector factor
        sector_returns = self.rng.normal(sector_drift, stock_volatility * 0.5, n_days)
        sector_cumret = np.cumsum(sector_returns)

        # Generate mean-reverting spread
        spread = self.generate_ornstein_uhlenbeck(
            n_days,
            theta=theta,
            mu=0.0,
            sigma=spread_volatility,
            x0=0.0
        )

        # Generate individual stock returns with idiosyncratic component
        prices = {}
        symbols = [f"STOCK_{chr(65+i)}" for i in range(n_stocks)]

        for i, (symbol, base_price, weight) in enumerate(zip(symbols, base_prices, weights)):
            # Idiosyncratic returns
            idio_returns = self.rng.normal(0, stock_volatility * 0.5, n_days)
            idio_cumret = np.cumsum(idio_returns)

            # Total log price = sector + idiosyncratic + weight * spread
            log_prices = sector_cumret + idio_cumret + weight * spread

            prices[symbol] = base_price * np.exp(log_prices)

        return prices, weights

    def generate_multiple_baskets(
        self,
        n_days: int,
        n_baskets: int = 3,
        stocks_per_basket: int = 4,
        half_life_range: Tuple[float, float] = (10.0, 25.0),
        spread_vol_range: Tuple[float, float] = (0.005, 0.015)
    ) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
        """
        Generate multiple baskets of cointegrated stocks.

        Args:
            n_days: Number of days of data
            n_baskets: Number of baskets
            stocks_per_basket: Stocks per basket
            half_life_range: Range for half-life values
            spread_vol_range: Range for spread volatilities

        Returns:
            Tuple of (price_data_dict, baskets_metadata)
        """
        all_prices = {}
        baskets = []
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')  # Business days

        for b in range(n_baskets):
            half_life = self.rng.uniform(*half_life_range)
            spread_vol = self.rng.uniform(*spread_vol_range)

            prices, weights = self.generate_cointegrated_basket(
                n_days=n_days,
                n_stocks=stocks_per_basket,
                half_life=half_life,
                spread_volatility=spread_vol
            )

            # Rename symbols to be unique across baskets
            symbols = []
            for old_symbol, price_array in prices.items():
                new_symbol = f"BASKET{b}_{old_symbol}"
                symbols.append(new_symbol)

                # Create DataFrame with OHLCV
                noise = 1 + self.rng.uniform(-0.005, 0.005, n_days)
                all_prices[new_symbol] = pd.DataFrame({
                    'open': price_array * (1 + self.rng.uniform(-0.002, 0.002, n_days)),
                    'high': price_array * (1 + np.abs(self.rng.normal(0, 0.01, n_days))),
                    'low': price_array * (1 - np.abs(self.rng.normal(0, 0.01, n_days))),
                    'close': price_array,
                    'volume': self.rng.uniform(1e6, 10e6, n_days)
                }, index=dates)

            baskets.append({
                'basket_id': b,
                'symbols': symbols,
                'sector': f'Sector_{b}',
                'size': len(symbols),
                'avg_correlation': 0.85,
                'true_half_life': half_life,
                'true_weights': weights
            })

        logger.info(f"Generated {n_baskets} baskets with {stocks_per_basket} stocks each")
        logger.info(f"Total symbols: {len(all_prices)}")

        return all_prices, baskets

    def generate_realistic_universe(
        self,
        n_days: int = 756,  # 3 years
        n_cointegrated_baskets: int = 5,
        n_random_stocks: int = 20,
        stocks_per_basket: int = 4,
        half_life_range: Tuple[float, float] = (12.0, 25.0),
        spread_vol_range: Tuple[float, float] = (0.006, 0.012)
    ) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
        """
        Generate a realistic universe with both cointegrated and random stocks.

        This simulates a real market where only some stocks are cointegrated.

        Args:
            n_days: Number of trading days
            n_cointegrated_baskets: Number of cointegrated baskets
            n_random_stocks: Number of random (non-cointegrated) stocks
            stocks_per_basket: Stocks per cointegrated basket
            half_life_range: Range of half-lives for cointegrated pairs
            spread_vol_range: Range of spread volatilities

        Returns:
            Tuple of (price_data, baskets_info)
        """
        # Generate cointegrated baskets
        coint_prices, baskets = self.generate_multiple_baskets(
            n_days=n_days,
            n_baskets=n_cointegrated_baskets,
            stocks_per_basket=stocks_per_basket,
            half_life_range=half_life_range,
            spread_vol_range=spread_vol_range
        )

        # Generate random stocks (not cointegrated)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

        for i in range(n_random_stocks):
            symbol = f"RANDOM_{i}"
            base_price = 50 + self.rng.uniform(0, 150)
            drift = self.rng.uniform(-0.0001, 0.0003)
            volatility = self.rng.uniform(0.015, 0.04)

            log_returns = self.rng.normal(drift, volatility, n_days)
            log_prices = np.cumsum(log_returns)
            prices = base_price * np.exp(log_prices)

            coint_prices[symbol] = pd.DataFrame({
                'open': prices * (1 + self.rng.uniform(-0.002, 0.002, n_days)),
                'high': prices * (1 + np.abs(self.rng.normal(0, 0.01, n_days))),
                'low': prices * (1 - np.abs(self.rng.normal(0, 0.01, n_days))),
                'close': prices,
                'volume': self.rng.uniform(1e6, 10e6, n_days)
            }, index=dates)

        # Add random stocks to baskets (as noise)
        for i in range(n_random_stocks // 5):
            baskets.append({
                'basket_id': n_cointegrated_baskets + i,
                'symbols': [f"RANDOM_{j}" for j in range(i*5, min((i+1)*5, n_random_stocks))],
                'sector': f'Random_Sector_{i}',
                'size': min(5, n_random_stocks - i*5),
                'avg_correlation': 0.3,
                'true_half_life': None,
                'true_weights': None
            })

        logger.info(f"Generated universe: {len(coint_prices)} stocks, {len(baskets)} baskets")
        logger.info(f"  - Cointegrated: {n_cointegrated_baskets} baskets ({n_cointegrated_baskets * stocks_per_basket} stocks)")
        logger.info(f"  - Random: {n_random_stocks} stocks")

        return coint_prices, baskets


def create_test_data(
    n_days: int = 756,
    n_baskets: int = 5,
    seed: int = 42
) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """
    Convenience function to create test data.

    Args:
        n_days: Number of trading days (default 3 years)
        n_baskets: Number of cointegrated baskets
        seed: Random seed

    Returns:
        Tuple of (price_data, baskets)
    """
    generator = SyntheticDataGenerator(seed=seed)
    return generator.generate_realistic_universe(
        n_days=n_days,
        n_cointegrated_baskets=n_baskets,
        n_random_stocks=15,
        stocks_per_basket=4,
        half_life_range=(12.0, 22.0),
        spread_vol_range=(0.006, 0.010)
    )
