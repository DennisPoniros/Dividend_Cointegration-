"""
Data validation and normalization utilities.

Ensures data quality and handles normalization steps.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates and normalizes data for the strategy."""

    @staticmethod
    def validate_price_data(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate price data DataFrame.

        Args:
            df: Price DataFrame
            symbol: Stock symbol for logging

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for NaN values
        if df.isnull().any().any():
            nan_counts = df.isnull().sum()
            issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

        # Check for negative prices
        if 'close' in df.columns and (df['close'] <= 0).any():
            issues.append("Negative or zero prices found")

        # Check for logical price relationships
        if all(col in df.columns for col in ['high', 'low', 'close']):
            if (df['high'] < df['low']).any():
                issues.append("High prices less than low prices")

            if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
                issues.append("Close prices outside high-low range")

        # Check for duplicate dates
        if df.index.duplicated().any():
            issues.append("Duplicate dates found")

        # Check for gaps larger than 5 days (excluding weekends)
        if len(df) > 1:
            date_diffs = df.index.to_series().diff().dt.days
            large_gaps = date_diffs[date_diffs > 5]
            if len(large_gaps) > 0:
                issues.append(f"Large gaps in data: {len(large_gaps)} gaps > 5 days")

        is_valid = len(issues) == 0

        if not is_valid:
            logger.warning(f"Validation issues for {symbol}: {', '.join(issues)}")

        return is_valid, issues

    @staticmethod
    def normalize_returns(returns: pd.Series, method: str = 'log') -> pd.Series:
        """
        Normalize returns.

        Args:
            returns: Return series
            method: Normalization method ('log', 'simple', 'z-score')

        Returns:
            Normalized returns
        """
        if method == 'log':
            # Log returns (more appropriate for cointegration)
            return np.log(1 + returns)

        elif method == 'simple':
            # Simple returns (already in this form typically)
            return returns

        elif method == 'z-score':
            # Z-score normalization
            return (returns - returns.mean()) / returns.std()

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def handle_missing_data(
        df: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Handle missing data.

        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing data ('forward_fill', 'drop', 'interpolate')

        Returns:
            DataFrame with missing data handled
        """
        if method == 'forward_fill':
            return df.fillna(method='ffill').fillna(method='bfill')

        elif method == 'drop':
            return df.dropna()

        elif method == 'interpolate':
            return df.interpolate(method='linear')

        else:
            raise ValueError(f"Unknown missing data method: {method}")

    @staticmethod
    def align_data(
        data_dict: Dict[str, pd.DataFrame],
        method: str = 'inner'
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple DataFrames to common dates.

        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            method: Alignment method ('inner', 'outer')

        Returns:
            Dictionary with aligned DataFrames
        """
        if not data_dict:
            return {}

        # Get all dates
        if method == 'inner':
            # Only keep dates common to all series
            common_dates = set(list(data_dict.values())[0].index)

            for df in data_dict.values():
                common_dates &= set(df.index)

            common_dates = sorted(common_dates)

        elif method == 'outer':
            # Keep all dates, fill missing
            common_dates = set()

            for df in data_dict.values():
                common_dates |= set(df.index)

            common_dates = sorted(common_dates)

        else:
            raise ValueError(f"Unknown alignment method: {method}")

        # Reindex all DataFrames
        aligned = {}
        for symbol, df in data_dict.items():
            aligned[symbol] = df.reindex(common_dates)

            if method == 'outer':
                aligned[symbol] = DataValidator.handle_missing_data(
                    aligned[symbol],
                    'forward_fill'
                )

        return aligned

    @staticmethod
    def check_stationarity(series: pd.Series) -> Dict:
        """
        Check if series is stationary using ADF test.

        Args:
            series: Time series to test

        Returns:
            Dictionary with stationarity test results
        """
        from statsmodels.tsa.stattools import adfuller

        try:
            result = adfuller(series.dropna(), maxlag=None, regression='c', autolag='AIC')

            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }

        except Exception as e:
            logger.error(f"Error checking stationarity: {e}")
            return {
                'error': str(e),
                'is_stationary': False
            }

    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in series.

        Args:
            series: Time series
            method: Detection method ('iqr', 'z-score')
            threshold: Threshold for outlier detection

        Returns:
            Boolean series indicating outliers
        """
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outliers = (series < lower_bound) | (series > upper_bound)

        elif method == 'z-score':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        if outliers.sum() > 0:
            logger.info(f"Detected {outliers.sum()} outliers ({outliers.mean()*100:.2f}%)")

        return outliers

    @staticmethod
    def validate_cointegration_vector(vector: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a cointegration vector.

        Args:
            vector: Cointegration vector dictionary

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check required fields
        required_fields = ['symbols', 'weights', 'half_life', 'hurst']

        for field in required_fields:
            if field not in vector:
                issues.append(f"Missing field: {field}")

        # Check weights
        if 'weights' in vector:
            weights = np.array(vector['weights'])

            if len(weights) == 0:
                issues.append("Empty weights array")

            if np.any(np.isnan(weights)):
                issues.append("NaN values in weights")

            if np.all(weights == 0):
                issues.append("All weights are zero")

        # Check half-life
        if 'half_life' in vector:
            hl = vector['half_life']

            if hl <= 0:
                issues.append(f"Invalid half-life: {hl}")

            if hl > 100:
                issues.append(f"Suspiciously large half-life: {hl}")

        # Check Hurst exponent
        if 'hurst' in vector:
            hurst = vector['hurst']

            if not (0 <= hurst <= 1):
                issues.append(f"Invalid Hurst exponent: {hurst}")

        is_valid = len(issues) == 0

        if not is_valid:
            logger.warning(f"Vector validation issues: {', '.join(issues)}")

        return is_valid, issues
