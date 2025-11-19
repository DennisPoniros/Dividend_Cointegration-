"""
Signal Generation Module (STEP 5)

Implements z-score based entry/exit signals:
- Entry: |z-score| ≥ 1.5σ
- Exit: |z-score| ≤ 0.5σ
- Rolling window: 42 days (2× target half-life)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals based on z-scores of cointegration spreads."""

    def __init__(self, config_path: str = "config/strategy_params.yaml"):
        """
        Initialize signal generator.

        Args:
            config_path: Path to strategy parameters
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.signal_config = self.config['signals']
        self.active_positions = {}

    def calculate_spread(
        self,
        price_data: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """
        Calculate cointegration spread.

        Args:
            price_data: DataFrame with prices for symbols in vector
            weights: Cointegration weights

        Returns:
            Series with spread values
        """
        # Normalize weights
        weights = weights / np.sum(np.abs(weights))

        # Calculate spread
        spread = (price_data * weights).sum(axis=1)

        return spread

    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling z-score of spread.

        Args:
            spread: Cointegration spread
            window: Rolling window (default: from config)

        Returns:
            Series with z-scores
        """
        if window is None:
            window = self.signal_config['calculation']['rolling_window']

        # Calculate rolling statistics
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        # Calculate z-score
        zscore = (spread - rolling_mean) / rolling_std

        return zscore

    def generate_entry_signals(
        self,
        zscore: pd.Series
    ) -> pd.DataFrame:
        """
        Generate entry signals based on z-score thresholds.

        Args:
            zscore: Z-score series

        Returns:
            DataFrame with entry signals (1=long, -1=short, 0=none)
        """
        long_threshold = self.signal_config['entry']['long_zscore']
        short_threshold = self.signal_config['entry']['short_zscore']

        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        signals['signal'] = 0

        # Long entry: z-score <= -1.5σ (spread is low, expect reversion up)
        signals.loc[zscore <= long_threshold, 'signal'] = 1

        # Short entry: z-score >= +1.5σ (spread is high, expect reversion down)
        signals.loc[zscore >= short_threshold, 'signal'] = -1

        return signals

    def generate_exit_signals(
        self,
        zscore: pd.Series,
        position: int
    ) -> pd.Series:
        """
        Generate exit signals for existing position.

        Args:
            zscore: Z-score series
            position: Current position (1=long, -1=short)

        Returns:
            Series with exit signals (True=exit, False=hold)
        """
        long_exit = self.signal_config['exit']['long_zscore']
        short_exit = self.signal_config['exit']['short_zscore']

        exits = pd.Series(False, index=zscore.index)

        if position == 1:  # Long position
            # Exit when z-score >= -0.5σ (spread has reverted)
            exits = zscore >= long_exit

        elif position == -1:  # Short position
            # Exit when z-score <= +0.5σ (spread has reverted)
            exits = zscore <= short_exit

        return exits

    def update_vector_signals(
        self,
        vector_id: int,
        price_data: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict:
        """
        Update signals for a specific cointegration vector.

        Args:
            vector_id: Unique identifier for vector
            price_data: Current price data for symbols in vector
            weights: Cointegration weights

        Returns:
            Dictionary with current signal information
        """
        # Calculate spread
        spread = self.calculate_spread(price_data, weights)

        # Calculate z-score
        zscore = self.calculate_zscore(spread)

        # Get current z-score
        current_zscore = zscore.iloc[-1]

        # Check if we have an active position
        is_active = vector_id in self.active_positions

        signal_info = {
            'vector_id': vector_id,
            'timestamp': price_data.index[-1],
            'spread': spread.iloc[-1],
            'zscore': current_zscore,
            'position': 0,
            'action': 'NONE'
        }

        if is_active:
            # Check for exit
            position = self.active_positions[vector_id]['position']
            exit_signal = self.generate_exit_signals(zscore, position).iloc[-1]

            if exit_signal:
                signal_info['action'] = 'EXIT'
                signal_info['position'] = 0
                del self.active_positions[vector_id]
            else:
                signal_info['position'] = position
                signal_info['action'] = 'HOLD'

        else:
            # Check for entry
            entry_signals = self.generate_entry_signals(zscore)
            current_signal = entry_signals['signal'].iloc[-1]

            if current_signal != 0:
                signal_info['action'] = 'ENTER_LONG' if current_signal == 1 else 'ENTER_SHORT'
                signal_info['position'] = current_signal

                # Track position
                self.active_positions[vector_id] = {
                    'position': current_signal,
                    'entry_zscore': current_zscore,
                    'entry_date': price_data.index[-1],
                    'entry_spread': spread.iloc[-1]
                }

        return signal_info

    def generate_all_signals(
        self,
        vectors: List[Dict],
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate signals for all cointegration vectors.

        Args:
            vectors: List of cointegration vector dictionaries
            price_data: Dictionary mapping symbols to price DataFrames

        Returns:
            DataFrame with all current signals
        """
        all_signals = []

        for i, vector in enumerate(vectors):
            try:
                # Get aligned price data for this vector
                symbols = vector['symbols']
                vector_prices = pd.DataFrame()

                for symbol in symbols:
                    if symbol in price_data and 'close' in price_data[symbol].columns:
                        vector_prices[symbol] = price_data[symbol]['close']

                vector_prices = vector_prices.dropna()

                if vector_prices.empty or len(vector_prices) < self.signal_config['calculation']['rolling_window']:
                    continue

                # Update signals
                signal_info = self.update_vector_signals(
                    i,
                    vector_prices,
                    vector['weights']
                )

                signal_info['symbols'] = ','.join(symbols)
                all_signals.append(signal_info)

            except Exception as e:
                logger.error(f"Error generating signals for vector {i}: {e}")
                continue

        return pd.DataFrame(all_signals)

    def get_position_details(self, vector_id: int) -> Optional[Dict]:
        """
        Get details of an active position.

        Args:
            vector_id: Vector identifier

        Returns:
            Position details, or None if no active position
        """
        return self.active_positions.get(vector_id)

    def get_all_positions(self) -> Dict:
        """
        Get all active positions.

        Returns:
            Dictionary of active positions
        """
        return self.active_positions.copy()

    def close_position(self, vector_id: int):
        """
        Manually close a position.

        Args:
            vector_id: Vector identifier
        """
        if vector_id in self.active_positions:
            del self.active_positions[vector_id]
            logger.info(f"Closed position for vector {vector_id}")

    def close_all_positions(self):
        """Close all active positions."""
        count = len(self.active_positions)
        self.active_positions.clear()
        logger.info(f"Closed {count} positions")
