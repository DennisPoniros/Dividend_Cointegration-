"""
Cointegration Vector Selection Module (STEP 3)

Implements rigorous cointegration testing:
1. Johansen Test (Primary) - trace statistic, eigenvalues
2. Engle-Granger confirmation
3. Validation Battery (half-life, Hurst exponent, out-of-sample)
4. Stability Test (quarterly splits)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import yaml
import logging

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class CointegrationTester:
    """Tests baskets for cointegration and validates relationships."""

    def __init__(self, config_path: str = "config/strategy_params.yaml"):
        """
        Initialize cointegration tester.

        Args:
            config_path: Path to strategy parameters
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.coint_config = self.config['cointegration']
        self.vectors = []

    def test_johansen(
        self,
        price_data: pd.DataFrame,
        lag_order: int = 2,
        det_order: int = 0
    ) -> Optional[Dict]:
        """
        Perform Johansen cointegration test.

        Args:
            price_data: DataFrame with aligned prices for basket
            lag_order: Lag order (default: 2 from config)
            det_order: Deterministic term (0=constant, -1=no constant, 1=linear trend)

        Returns:
            Dictionary with test results, or None if test fails
        """
        try:
            # Johansen test
            result = coint_johansen(price_data, det_order, lag_order)

            # Get trace statistic for rank=1
            trace_stat = result.lr1[0]  # First trace statistic
            critical_value = result.cvt[0, 1]  # 5% critical value

            # Calculate p-value approximation
            # Note: exact p-values require simulation; using approximate conversion
            # p-value < 0.01 corresponds roughly to trace_stat > cv_1pct
            cv_1pct = result.cvt[0, 0]  # 1% critical value
            passes_pvalue = trace_stat > cv_1pct

            # Get largest eigenvalue
            eigenvalue = result.eig[0]

            # Check acceptance criteria
            johansen_config = self.coint_config['johansen']

            if not passes_pvalue:
                logger.debug("Johansen test failed: p-value >= 0.01")
                return None

            if eigenvalue < johansen_config['eigenvalue_min']:
                logger.debug(f"Johansen test failed: eigenvalue {eigenvalue:.4f} < {johansen_config['eigenvalue_min']}")
                return None

            # Get eigenvector (cointegration weights)
            eigenvector = result.evec[:, 0]  # First eigenvector

            return {
                'trace_statistic': trace_stat,
                'critical_value_1pct': cv_1pct,
                'eigenvalue': eigenvalue,
                'eigenvector': eigenvector,
                'rank': 1
            }

        except Exception as e:
            logger.error(f"Error in Johansen test: {e}")
            return None

    def calculate_spread(
        self,
        price_data: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """
        Calculate cointegration spread.

        Args:
            price_data: DataFrame with aligned prices
            weights: Cointegration weights (eigenvector)

        Returns:
            Series with spread values
        """
        # Normalize weights
        weights = weights / np.sum(np.abs(weights))

        # Calculate spread
        spread = (price_data * weights).sum(axis=1)

        return spread

    def test_engle_granger(
        self,
        spread: pd.Series,
        threshold: float = -3.5
    ) -> bool:
        """
        Perform Engle-Granger test on spread.

        Args:
            spread: Cointegration spread
            threshold: ADF statistic threshold (default: -3.5)

        Returns:
            True if passes, False otherwise
        """
        try:
            # ADF test on spread
            adf_result = adfuller(spread, maxlag=None, regression='c', autolag='AIC')

            adf_stat = adf_result[0]
            passes = adf_stat < threshold

            logger.debug(f"Engle-Granger ADF: {adf_stat:.4f}, threshold: {threshold}, passes: {passes}")

            return passes

        except Exception as e:
            logger.error(f"Error in Engle-Granger test: {e}")
            return False

    def calculate_half_life(
        self,
        spread: pd.Series
    ) -> Optional[float]:
        """
        Calculate half-life of mean reversion.

        Uses Ornstein-Uhlenbeck process: dy = -λ(y - μ)dt + σdW
        Half-life = ln(2) / λ

        Args:
            spread: Cointegration spread

        Returns:
            Half-life in days, or None if calculation fails
        """
        try:
            # Calculate lagged spread and changes
            spread_lag = spread.shift(1)
            spread_diff = spread.diff()

            # Remove NaN
            df = pd.DataFrame({
                'spread': spread,
                'spread_lag': spread_lag,
                'spread_diff': spread_diff
            }).dropna()

            # Regression: Δy = α + β*y(t-1) + ε
            # where β = -λ
            X = df['spread_lag'].values
            y = df['spread_diff'].values

            # Add constant
            X_with_const = np.column_stack([np.ones(len(X)), X])

            # OLS
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            lambda_param = -beta[1]

            if lambda_param <= 0:
                logger.warning("Lambda parameter <= 0, mean reversion not detected")
                return None

            half_life = np.log(2) / lambda_param

            return half_life

        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return None

    def calculate_hurst_exponent(
        self,
        spread: pd.Series
    ) -> Optional[float]:
        """
        Calculate Hurst exponent using R/S analysis.

        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending

        Args:
            spread: Cointegration spread

        Returns:
            Hurst exponent, or None if calculation fails
        """
        try:
            # Remove NaN
            series = spread.dropna().values

            # Number of different lag values to test
            lags = range(2, min(100, len(series) // 2))

            # Calculate R/S for different lags
            tau = []
            rs = []

            for lag in lags:
                # Split into subseries
                n = len(series) // lag

                if n < 2:
                    break

                # Calculate R/S for this lag
                rs_lag = []

                for i in range(lag):
                    subset = series[i * n:(i + 1) * n]

                    # Mean
                    m = np.mean(subset)

                    # Centered series
                    y = subset - m

                    # Cumulative sum
                    z = np.cumsum(y)

                    # Range
                    r = np.max(z) - np.min(z)

                    # Standard deviation
                    s = np.std(subset, ddof=1)

                    if s > 0:
                        rs_lag.append(r / s)

                if rs_lag:
                    tau.append(lag)
                    rs.append(np.mean(rs_lag))

            # Fit log(R/S) = log(c) + H * log(tau)
            if len(tau) < 2:
                return None

            log_tau = np.log(tau)
            log_rs = np.log(rs)

            # Linear regression
            coeffs = np.polyfit(log_tau, log_rs, 1)
            hurst = coeffs[0]

            return hurst

        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return None

    def test_out_of_sample(
        self,
        price_data: pd.DataFrame,
        weights: np.ndarray,
        oos_days: int = 126
    ) -> bool:
        """
        Test cointegration on out-of-sample data.

        Args:
            price_data: Full price DataFrame
            weights: Cointegration weights from in-sample
            oos_days: Number of out-of-sample days

        Returns:
            True if passes, False otherwise
        """
        try:
            if len(price_data) < oos_days:
                logger.warning("Insufficient data for out-of-sample test")
                return False

            # Split data
            oos_data = price_data.tail(oos_days)

            # Calculate spread on OOS data
            spread = self.calculate_spread(oos_data, weights)

            # Test for mean reversion
            passes = self.test_engle_granger(
                spread,
                threshold=self.coint_config['validation']['engle_granger_adf_threshold']
            )

            logger.debug(f"Out-of-sample test: {'PASS' if passes else 'FAIL'}")

            return passes

        except Exception as e:
            logger.error(f"Error in out-of-sample test: {e}")
            return False

    def test_stability(
        self,
        price_data: pd.DataFrame,
        weights: np.ndarray
    ) -> bool:
        """
        Test stability across quarterly splits.

        Args:
            price_data: Full price DataFrame (should be 504 days)
            weights: Cointegration weights

        Returns:
            True if stable, False otherwise
        """
        try:
            n_quarters = self.coint_config['stability']['quarters']
            total_days = len(price_data)
            days_per_quarter = total_days // n_quarters

            half_lives = []

            for q in range(n_quarters):
                start_idx = q * days_per_quarter
                end_idx = (q + 1) * days_per_quarter if q < n_quarters - 1 else total_days

                quarter_data = price_data.iloc[start_idx:end_idx]

                # Calculate spread
                spread = self.calculate_spread(quarter_data, weights)

                # Test cointegration
                if not self.test_engle_granger(spread):
                    logger.debug(f"Quarter {q + 1} failed Engle-Granger test")
                    return False

                # Calculate half-life
                hl = self.calculate_half_life(spread)
                if hl is None:
                    return False

                half_lives.append(hl)

            # Check coefficient of variation
            half_lives = np.array(half_lives)
            cv = np.std(half_lives) / np.mean(half_lives)

            max_cv = self.coint_config['stability']['half_life_cv_max']
            passes = cv < max_cv

            logger.debug(f"Stability test: CV={cv:.4f}, max={max_cv}, passes: {passes}")

            return passes

        except Exception as e:
            logger.error(f"Error in stability test: {e}")
            return False

    def validate_vector(
        self,
        price_data: pd.DataFrame,
        weights: np.ndarray
    ) -> Optional[Dict]:
        """
        Run complete validation battery.

        Args:
            price_data: Price DataFrame
            weights: Cointegration weights

        Returns:
            Dictionary with validation results, or None if fails
        """
        # Calculate spread
        spread = self.calculate_spread(price_data, weights)

        # 1. Engle-Granger confirmation
        eg_threshold = self.coint_config['validation']['engle_granger_adf_threshold']
        if not self.test_engle_granger(spread, eg_threshold):
            logger.debug("Failed Engle-Granger confirmation")
            return None

        # 2. Half-life check
        half_life = self.calculate_half_life(spread)
        if half_life is None:
            logger.debug("Failed to calculate half-life")
            return None

        hl_min = self.coint_config['validation']['half_life_min']
        hl_max = self.coint_config['validation']['half_life_max']

        if not (hl_min <= half_life <= hl_max):
            logger.debug(f"Half-life {half_life:.2f} outside bounds [{hl_min}, {hl_max}]")
            return None

        # 3. Hurst exponent check
        hurst = self.calculate_hurst_exponent(spread)
        if hurst is None:
            logger.debug("Failed to calculate Hurst exponent")
            return None

        hurst_max = self.coint_config['validation']['hurst_max']
        if hurst >= hurst_max:
            logger.debug(f"Hurst exponent {hurst:.4f} >= {hurst_max}")
            return None

        # 4. Out-of-sample test
        oos_days = self.coint_config['validation']['out_of_sample_days']
        if not self.test_out_of_sample(price_data, weights, oos_days):
            logger.debug("Failed out-of-sample test")
            return None

        # 5. Stability test
        if not self.test_stability(price_data, weights):
            logger.debug("Failed stability test")
            return None

        # All validations passed
        return {
            'half_life': half_life,
            'hurst': hurst,
            'spread_mean': spread.mean(),
            'spread_std': spread.std()
        }

    def test_basket(
        self,
        price_data: pd.DataFrame,
        basket_info: Dict
    ) -> List[Dict]:
        """
        Test all combinations in a basket for cointegration.

        Args:
            price_data: Aligned price data for basket
            basket_info: Basket metadata

        Returns:
            List of valid cointegration vectors
        """
        symbols = price_data.columns.tolist()
        vectors = []

        # Test combinations of 3-6 stocks
        for n in range(3, min(7, len(symbols) + 1)):
            for combo in combinations(symbols, n):
                combo_data = price_data[list(combo)]

                # Ensure sufficient data
                testing_days = self.coint_config['testing_period_days']
                if len(combo_data) < testing_days:
                    continue

                # Use most recent testing_days
                test_data = combo_data.tail(testing_days)

                # Johansen test
                johansen_result = self.test_johansen(
                    test_data,
                    lag_order=self.coint_config['johansen']['lag_order']
                )

                if johansen_result is None:
                    continue

                # Validation battery
                validation = self.validate_vector(
                    test_data,
                    johansen_result['eigenvector']
                )

                if validation is None:
                    continue

                # Passed all tests!
                vector = {
                    'symbols': list(combo),
                    'weights': johansen_result['eigenvector'],
                    'eigenvalue': johansen_result['eigenvalue'],
                    'trace_stat': johansen_result['trace_statistic'],
                    'half_life': validation['half_life'],
                    'hurst': validation['hurst'],
                    'spread_mean': validation['spread_mean'],
                    'spread_std': validation['spread_std'],
                    'basket_sector': basket_info['sector']
                }

                vectors.append(vector)
                logger.info(f"Valid vector found: {combo}, half-life: {validation['half_life']:.2f}")

        return vectors

    def test_all_baskets(
        self,
        baskets: List[Dict],
        price_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        Test all baskets for cointegration.

        Args:
            baskets: List of basket dictionaries
            price_data: Dictionary mapping symbols to price DataFrames

        Returns:
            List of all valid cointegration vectors
        """
        all_vectors = []

        for i, basket in enumerate(baskets):
            logger.info(f"Testing basket {i + 1}/{len(baskets)}: {basket['sector']}")

            # Get aligned price data for basket
            symbols = basket['symbols']
            basket_prices = pd.DataFrame()

            for symbol in symbols:
                if symbol in price_data and 'close' in price_data[symbol].columns:
                    basket_prices[symbol] = price_data[symbol]['close']

            basket_prices = basket_prices.dropna()

            if basket_prices.empty:
                logger.warning(f"No valid price data for basket {i}")
                continue

            # Test basket
            vectors = self.test_basket(basket_prices, basket)
            all_vectors.extend(vectors)

        self.vectors = all_vectors
        logger.info(f"Cointegration testing complete: {len(all_vectors)} valid vectors found")

        return all_vectors

    def save_vectors(self, filepath: str):
        """Save cointegration vectors to file."""
        if not self.vectors:
            logger.warning("No vectors to save")
            return

        # Convert to DataFrame
        rows = []
        for i, vec in enumerate(self.vectors):
            rows.append({
                'vector_id': i,
                'symbols': ','.join(vec['symbols']),
                'weights': ','.join([f"{w:.6f}" for w in vec['weights']]),
                'eigenvalue': vec['eigenvalue'],
                'half_life': vec['half_life'],
                'hurst': vec['hurst'],
                'sector': vec['basket_sector']
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self.vectors)} vectors to {filepath}")

    def load_vectors(self, filepath: str):
        """Load cointegration vectors from file."""
        df = pd.read_csv(filepath)

        self.vectors = []
        for _, row in df.iterrows():
            vector = {
                'symbols': row['symbols'].split(','),
                'weights': np.array([float(w) for w in row['weights'].split(',')]),
                'eigenvalue': row['eigenvalue'],
                'half_life': row['half_life'],
                'hurst': row['hurst'],
                'basket_sector': row['sector']
            }
            self.vectors.append(vector)

        logger.info(f"Loaded {len(self.vectors)} vectors from {filepath}")
