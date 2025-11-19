"""
Portfolio Construction Module (STEP 4)

Implements:
- Equal risk contribution across vectors
- Kelly criterion position sizing (quarter-Kelly)
- Market neutrality monitoring and beta hedging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import logging

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages portfolio construction and position sizing."""

    def __init__(self, config_path: str = "config/strategy_params.yaml"):
        """
        Initialize portfolio manager.

        Args:
            config_path: Path to strategy parameters
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.portfolio_config = self.config['portfolio']
        self.total_capital = 0
        self.positions = {}
        self.hedge_position = None

    def set_capital(self, capital: float):
        """
        Set total portfolio capital.

        Args:
            capital: Total capital in USD
        """
        self.total_capital = capital
        logger.info(f"Portfolio capital set to ${capital:,.2f}")

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion fraction.

        Kelly % = W - [(1 - W) / R]
        where W = win rate, R = avg_win / avg_loss

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)

        Returns:
            Kelly fraction (capped at configured max)
        """
        if avg_loss == 0:
            return 0

        r = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / r)

        # Apply fractional Kelly (quarter-Kelly)
        kelly_fraction = kelly_pct * self.portfolio_config['kelly_fraction']

        # Cap at reasonable bounds
        kelly_fraction = np.clip(kelly_fraction, 0, self.portfolio_config['max_position_per_vector'])

        return kelly_fraction

    def calculate_position_size(
        self,
        vector_info: Dict,
        signal_info: Dict,
        historical_performance: Optional[Dict] = None
    ) -> float:
        """
        Calculate position size for a vector.

        Size = capital × kelly_fraction × (1 / num_active_vectors)

        Args:
            vector_info: Cointegration vector information
            signal_info: Current signal information
            historical_performance: Optional historical stats for Kelly calculation

        Returns:
            Position size in USD
        """
        if self.total_capital == 0:
            logger.warning("Total capital not set")
            return 0

        # If no historical performance, use conservative default
        if historical_performance is None:
            kelly_frac = self.portfolio_config['kelly_fraction']
        else:
            kelly_frac = self.calculate_kelly_fraction(
                historical_performance.get('win_rate', 0.5),
                historical_performance.get('avg_win', 0.02),
                historical_performance.get('avg_loss', 0.01)
            )

        # Scale by z-score magnitude (higher z-score = higher conviction)
        zscore = abs(signal_info.get('zscore', 1.5))
        zscore_scale = min(zscore / 1.5, 2.0)  # Cap at 2x for z-score >= 3.0

        # Calculate base position size
        position_size = self.total_capital * kelly_frac * zscore_scale

        # Apply maximum position limit
        max_position = self.total_capital * self.portfolio_config['max_position_per_vector']
        position_size = min(position_size, max_position)

        return position_size

    def calculate_basket_volatility(
        self,
        price_data: pd.DataFrame,
        weights: np.ndarray,
        window: int = 42
    ) -> float:
        """
        Calculate volatility of basket spread.

        Args:
            price_data: Price data for symbols in basket
            weights: Cointegration weights
            window: Rolling window for volatility

        Returns:
            Annualized volatility
        """
        # Calculate spread
        spread = (price_data * weights).sum(axis=1)

        # Calculate returns
        returns = spread.pct_change().dropna()

        # Calculate volatility
        if len(returns) < window:
            vol = returns.std()
        else:
            vol = returns.tail(window).std()

        # Annualize (assuming daily data)
        annual_vol = vol * np.sqrt(252)

        return annual_vol

    def allocate_positions(
        self,
        signals: pd.DataFrame,
        vectors: List[Dict],
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[int, Dict]:
        """
        Allocate capital across all active signals.

        Args:
            signals: DataFrame with current signals
            vectors: List of cointegration vectors
            price_data: Price data dictionary

        Returns:
            Dictionary mapping vector_id to position allocation
        """
        allocations = {}

        # Filter to entry signals only
        entry_signals = signals[signals['action'].isin(['ENTER_LONG', 'ENTER_SHORT'])]

        if entry_signals.empty:
            return allocations

        # Calculate total risk budget
        max_total_exposure = self.total_capital * self.portfolio_config['max_total_exposure']

        # Equal risk contribution approach
        num_vectors = len(entry_signals)
        risk_per_vector = max_total_exposure / num_vectors

        for _, signal in entry_signals.iterrows():
            vector_id = int(signal['vector_id'])
            vector = vectors[vector_id]

            # Get price data for vector
            symbols = vector['symbols']
            vector_prices = pd.DataFrame()

            for symbol in symbols:
                if symbol in price_data and 'close' in price_data[symbol].columns:
                    vector_prices[symbol] = price_data[symbol]['close']

            if vector_prices.empty:
                continue

            # Calculate volatility
            volatility = self.calculate_basket_volatility(
                vector_prices,
                vector['weights']
            )

            # Position size based on volatility targeting
            # Size = risk_budget / volatility
            if volatility > 0:
                position_size = risk_per_vector / volatility
            else:
                position_size = risk_per_vector

            # Apply limits
            max_position = self.total_capital * self.portfolio_config['max_position_per_vector']
            position_size = min(position_size, max_position)

            # Calculate individual stock positions based on weights
            weights = vector['weights']
            normalized_weights = weights / np.sum(np.abs(weights))

            stock_positions = {}
            for symbol, weight in zip(symbols, normalized_weights):
                stock_positions[symbol] = {
                    'weight': weight,
                    'notional': position_size * weight,
                    'current_price': vector_prices[symbol].iloc[-1]
                }

            allocations[vector_id] = {
                'total_notional': position_size,
                'direction': 1 if signal['action'] == 'ENTER_LONG' else -1,
                'positions': stock_positions,
                'volatility': volatility,
                'zscore': signal['zscore']
            }

        return allocations

    def calculate_portfolio_beta(
        self,
        positions: Dict[int, Dict],
        market_data: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        window: int = 60
    ) -> float:
        """
        Calculate portfolio beta relative to market.

        Args:
            positions: Current position allocations
            market_data: Market index price data (e.g., SPY)
            price_data: Individual stock price data
            window: Rolling window for beta calculation

        Returns:
            Portfolio beta
        """
        # Calculate weighted average beta of all positions
        total_notional = sum(p['total_notional'] for p in positions.values())

        if total_notional == 0:
            return 0

        portfolio_beta = 0

        for vector_id, allocation in positions.items():
            # Calculate beta for each stock in the position
            for symbol, stock_pos in allocation['positions'].items():
                if symbol not in price_data:
                    continue

                # Get stock returns
                stock_prices = price_data[symbol]['close']
                stock_returns = stock_prices.pct_change().dropna()

                # Get market returns
                market_returns = market_data['close'].pct_change().dropna()

                # Align
                aligned = pd.DataFrame({
                    'stock': stock_returns,
                    'market': market_returns
                }).dropna()

                if len(aligned) < window:
                    continue

                # Calculate beta
                covariance = aligned['stock'].tail(window).cov(aligned['market'].tail(window))
                market_variance = aligned['market'].tail(window).var()

                if market_variance > 0:
                    beta = covariance / market_variance
                else:
                    beta = 0

                # Weight by position size
                weight = abs(stock_pos['notional']) / total_notional
                portfolio_beta += beta * weight * np.sign(stock_pos['weight'])

        return portfolio_beta

    def calculate_hedge_size(
        self,
        portfolio_beta: float,
        total_exposure: float
    ) -> float:
        """
        Calculate hedge size to neutralize beta.

        Args:
            portfolio_beta: Current portfolio beta
            total_exposure: Total portfolio exposure

        Returns:
            Notional hedge size (negative = short)
        """
        # Hedge size = -beta × exposure
        hedge_size = -portfolio_beta * total_exposure

        return hedge_size

    def rebalance_for_neutrality(
        self,
        positions: Dict[int, Dict],
        market_data: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame]
    ) -> Optional[Dict]:
        """
        Check beta and rebalance if needed.

        Args:
            positions: Current position allocations
            market_data: Market index data
            price_data: Individual stock price data

        Returns:
            Hedge position if needed, None otherwise
        """
        # Calculate portfolio beta
        beta = self.calculate_portfolio_beta(positions, market_data, price_data)

        beta_threshold = self.portfolio_config['market_neutrality']['beta_threshold']

        logger.info(f"Portfolio beta: {beta:.4f}, threshold: ±{beta_threshold}")

        if abs(beta) > beta_threshold:
            # Calculate hedge
            total_exposure = sum(p['total_notional'] for p in positions.values())
            hedge_size = self.calculate_hedge_size(beta, total_exposure)

            hedge_instrument = self.portfolio_config['market_neutrality']['hedge_instrument']

            hedge = {
                'instrument': hedge_instrument,
                'notional': hedge_size,
                'current_beta': beta,
                'target_beta': 0
            }

            logger.info(f"Beta hedge required: {hedge_instrument} notional ${hedge_size:,.2f}")

            return hedge

        return None

    def get_portfolio_summary(self, positions: Dict[int, Dict]) -> Dict:
        """
        Get summary statistics of current portfolio.

        Args:
            positions: Current position allocations

        Returns:
            Dictionary with portfolio statistics
        """
        if not positions:
            return {
                'total_exposure': 0,
                'num_vectors': 0,
                'num_stocks': 0,
                'avg_volatility': 0,
                'exposure_pct': 0
            }

        total_exposure = sum(abs(p['total_notional']) for p in positions.values())
        num_vectors = len(positions)

        # Count unique stocks
        all_stocks = set()
        volatilities = []

        for allocation in positions.values():
            all_stocks.update(allocation['positions'].keys())
            volatilities.append(allocation['volatility'])

        summary = {
            'total_exposure': total_exposure,
            'num_vectors': num_vectors,
            'num_stocks': len(all_stocks),
            'avg_volatility': np.mean(volatilities),
            'exposure_pct': (total_exposure / self.total_capital) if self.total_capital > 0 else 0,
            'largest_position': max(abs(p['total_notional']) for p in positions.values()),
            'smallest_position': min(abs(p['total_notional']) for p in positions.values())
        }

        return summary
