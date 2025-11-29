"""
Walk-Forward Backtesting Engine

Implements proper out-of-sample testing to eliminate forward-looking bias:
1. Train on historical data (in-sample period)
2. Trade on future data (out-of-sample period)
3. Periodically retrain as new data becomes available

This is the ONLY valid way to backtest a cointegration strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yaml
import logging
from copy import deepcopy

from ..strategy.cointegration import CointegrationTester
from ..strategy.signals import SignalGenerator
from ..strategy.portfolio import PortfolioManager
from ..risk.risk_management import RiskManager
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine.

    Key principle: NEVER use future data to make trading decisions.

    Timeline:
    |-------- Training Window --------|---- Trading Window ----|
    |     (discover cointegration)    |  (trade out-of-sample) |

    Then slide forward and repeat.
    """

    def __init__(self, config_path: str = "config/strategy_params.yaml"):
        """
        Initialize walk-forward backtest engine.

        Args:
            config_path: Path to strategy parameters
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.backtest_config = self.config['backtesting']

        # Walk-forward specific parameters
        self.training_window = 252  # 1 year of training data
        self.trading_window = 63    # 3 months of trading before retraining
        self.min_training_data = 126  # Minimum data for initial vectors

        # Initialize components
        self.coint_tester = CointegrationTester(config_path)
        self.signal_generator = SignalGenerator(config_path)
        self.portfolio_manager = PortfolioManager(config_path)
        self.risk_manager = RiskManager(config_path)

        # Tracking
        self.trades = []
        self.equity_curve = []

    def calculate_commission(self, notional: float, num_shares: float) -> float:
        """Calculate commission costs."""
        per_share_cost = self.backtest_config['commission_per_share']
        return abs(num_shares) * per_share_cost

    def calculate_slippage(self, notional: float) -> float:
        """Calculate slippage costs."""
        slippage_bps = self.backtest_config['slippage_bps']
        return abs(notional) * (slippage_bps / 10000)

    def discover_vectors_on_training_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        training_end_date: datetime,
        baskets: List[Dict]
    ) -> List[Dict]:
        """
        Discover cointegration vectors using ONLY data up to training_end_date.

        This is the key to eliminating forward-looking bias.

        Args:
            price_data: Full price data
            training_end_date: Last date of training period
            baskets: List of stock baskets to test

        Returns:
            List of valid cointegration vectors
        """
        # Filter price data to training period ONLY
        training_data = {}
        for symbol, df in price_data.items():
            mask = df.index <= training_end_date
            if mask.sum() >= self.min_training_data:
                training_data[symbol] = df[mask].copy()

        if not training_data:
            logger.warning("Insufficient training data")
            return []

        # Discover vectors on training data only
        vectors = self.coint_tester.test_all_baskets(baskets, training_data)

        logger.info(f"Discovered {len(vectors)} vectors using data up to {training_end_date}")

        return vectors

    def calculate_spread(
        self,
        price_data: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """Calculate cointegration spread."""
        weights = weights / np.sum(np.abs(weights))
        spread = (price_data * weights).sum(axis=1)
        return spread

    def run_walk_forward_backtest(
        self,
        baskets: List[Dict],
        price_data: Dict[str, pd.DataFrame],
        initial_capital: float = 1_000_000,
        training_window: int = 252,
        trading_window: int = 63,
        retraining_frequency: int = 63
    ) -> Dict:
        """
        Run complete walk-forward backtest.

        Timeline example with 3 years of data:
        - Days 1-252: Training period 1 (discover vectors)
        - Days 253-315: Trading period 1 (trade with vectors from period 1)
        - Days 253-504: Training period 2 (re-discover vectors)
        - Days 505-567: Trading period 2 (trade with vectors from period 2)
        - And so on...

        Args:
            baskets: List of stock baskets
            price_data: Dictionary mapping symbols to price DataFrames
            initial_capital: Starting capital
            training_window: Days of data for training
            trading_window: Days of trading before retraining
            retraining_frequency: How often to retrain (in trading days)

        Returns:
            Dictionary with backtest results
        """
        logger.info("="*60)
        logger.info("WALK-FORWARD BACKTEST (Non-Forward-Looking)")
        logger.info("="*60)
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Training window: {training_window} days")
        logger.info(f"Trading window: {trading_window} days")
        logger.info(f"Retraining frequency: {retraining_frequency} days")

        self.training_window = training_window
        self.trading_window = trading_window

        # Set portfolio capital
        self.portfolio_manager.set_capital(initial_capital)

        # Get all dates from price data
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        if len(all_dates) < training_window + trading_window:
            logger.error("Insufficient data for walk-forward backtest")
            return {'error': 'Insufficient data'}

        logger.info(f"Data range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")

        # Initialize tracking
        current_capital = initial_capital
        self.trades = []
        self.equity_curve = []
        active_positions = {}
        current_vectors = []

        # Determine walk-forward windows
        training_start_idx = 0
        training_end_idx = training_window - 1
        trading_start_idx = training_window

        days_since_retrain = 0

        # Main walk-forward loop
        trading_idx = trading_start_idx

        while trading_idx < len(all_dates):
            # Check if we need to retrain
            if days_since_retrain == 0 or days_since_retrain >= retraining_frequency:
                # Retrain on all data up to current point
                training_end_date = all_dates[trading_idx - 1]

                logger.info(f"\n--- Retraining on data up to {training_end_date} ---")

                # Close all positions before retraining (conservative approach)
                for vector_id in list(active_positions.keys()):
                    pos = active_positions[vector_id]
                    current_prices = self._get_current_prices(
                        pos['symbols'],
                        price_data,
                        all_dates[trading_idx - 1]
                    )
                    self._close_position(
                        vector_id,
                        pos,
                        current_prices,
                        all_dates[trading_idx - 1],
                        "Retraining"
                    )
                active_positions.clear()

                # Discover new vectors using only past data
                current_vectors = self.discover_vectors_on_training_data(
                    price_data,
                    training_end_date,
                    baskets
                )

                # Reset signal generator state
                self.signal_generator.active_positions = {}

                days_since_retrain = 0

                if not current_vectors:
                    logger.warning("No valid vectors found, skipping trading period")
                    trading_idx += trading_window
                    continue

            # Trade for this day
            date = all_dates[trading_idx]

            # Check if trading is allowed
            if not self.risk_manager.is_trading_allowed(date):
                self.equity_curve.append({
                    'date': date,
                    'equity': current_capital,
                    'num_positions': len(active_positions)
                })
                trading_idx += 1
                days_since_retrain += 1
                continue

            # Process each vector
            daily_signals = []

            for i, vector in enumerate(current_vectors):
                try:
                    symbols = vector['symbols']
                    vector_prices = pd.DataFrame()

                    for symbol in symbols:
                        if symbol in price_data:
                            # Get data up to current date ONLY
                            symbol_data = price_data[symbol][price_data[symbol].index <= date]
                            if not symbol_data.empty:
                                vector_prices[symbol] = symbol_data['close']

                    if vector_prices.empty or len(vector_prices) < 42:
                        continue

                    # Calculate spread and z-score
                    spread = self.calculate_spread(vector_prices, vector['weights'])
                    zscore_series = self.signal_generator.calculate_zscore(spread)

                    if zscore_series.empty or pd.isna(zscore_series.iloc[-1]):
                        continue

                    current_zscore = zscore_series.iloc[-1]

                    # Check risk stops if position is active
                    if i in active_positions:
                        position_pnl = self._calculate_position_pnl(
                            i,
                            active_positions[i],
                            vector_prices.iloc[-1]
                        )

                        should_exit, reasons = self.risk_manager.check_position_stops(
                            i,
                            spread,
                            current_zscore,
                            date,
                            position_pnl
                        )

                        if should_exit:
                            self._close_position(
                                i,
                                active_positions[i],
                                vector_prices.iloc[-1],
                                date,
                                f"Risk stop: {', '.join(reasons)}"
                            )
                            del active_positions[i]
                            continue

                    # Generate signal
                    signal_info = self.signal_generator.update_vector_signals(
                        i,
                        vector_prices,
                        vector['weights']
                    )

                    signal_info['vector'] = vector
                    daily_signals.append(signal_info)

                except Exception as e:
                    logger.debug(f"Error processing vector {i} on {date}: {e}")
                    continue

            # Process signals
            for signal in daily_signals:
                vector_id = signal['vector_id']
                action = signal['action']
                vector = signal['vector']

                if action in ['ENTER_LONG', 'ENTER_SHORT']:
                    if vector_id not in active_positions:
                        self._enter_position(
                            vector_id,
                            vector,
                            signal,
                            price_data,
                            date,
                            active_positions
                        )

                elif action == 'EXIT':
                    if vector_id in active_positions:
                        current_prices = self._get_current_prices(
                            vector['symbols'],
                            price_data,
                            date
                        )
                        self._close_position(
                            vector_id,
                            active_positions[vector_id],
                            current_prices,
                            date,
                            "Signal exit"
                        )
                        del active_positions[vector_id]

            # Calculate daily equity
            daily_pnl = sum(
                self._calculate_position_pnl(
                    vid,
                    pos,
                    self._get_current_prices(pos['symbols'], price_data, date)
                )
                for vid, pos in active_positions.items()
            )

            current_equity = current_capital + daily_pnl

            # Check portfolio-level drawdown
            action = self.risk_manager.check_portfolio_drawdown(current_equity)
            if action == 'exit_all':
                for vector_id in list(active_positions.keys()):
                    pos = active_positions[vector_id]
                    current_prices = self._get_current_prices(pos['symbols'], price_data, date)
                    self._close_position(vector_id, pos, current_prices, date, "Portfolio drawdown")
                active_positions.clear()
                self.risk_manager.apply_drawdown_action(action, date)

            elif action == 'cut_50pct':
                for pos in active_positions.values():
                    for symbol in pos['positions']:
                        pos['positions'][symbol]['quantity'] *= 0.5
                self.risk_manager.apply_drawdown_action(action, date)

            # Track equity
            self.equity_curve.append({
                'date': date,
                'equity': current_equity,
                'num_positions': len(active_positions),
                'daily_pnl': daily_pnl
            })

            trading_idx += 1
            days_since_retrain += 1

        # Compile results
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        trades_df = pd.DataFrame(self.trades)

        # Calculate metrics
        if not equity_df.empty:
            metrics = PerformanceMetrics.calculate_all_metrics(
                equity_df['equity'],
                trades_df if not trades_df.empty else None
            )
        else:
            metrics = {}

        results = {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics,
            'final_capital': current_equity if 'current_equity' in dir() else initial_capital,
            'total_return': (current_equity / initial_capital) - 1 if 'current_equity' in dir() else 0
        }

        logger.info("\n" + "="*60)
        logger.info("WALK-FORWARD BACKTEST COMPLETE")
        logger.info("="*60)
        if metrics:
            logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")

        return results

    def _enter_position(
        self,
        vector_id: int,
        vector: Dict,
        signal: Dict,
        price_data: Dict[str, pd.DataFrame],
        date: datetime,
        active_positions: Dict
    ):
        """Enter a new position."""
        allocation = self.portfolio_manager.calculate_position_size(vector, signal)

        current_prices = self._get_current_prices(vector['symbols'], price_data, date)

        weights = vector['weights']
        normalized_weights = weights / np.sum(np.abs(weights))

        position = {
            'entry_date': date,
            'entry_zscore': signal['zscore'],
            'symbols': vector['symbols'],
            'positions': {},
            'direction': 1 if signal['action'] == 'ENTER_LONG' else -1
        }

        total_cost = 0

        for symbol, weight in zip(vector['symbols'], normalized_weights):
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            notional = allocation * weight * position['direction']
            quantity = notional / price

            commission = self.calculate_commission(notional, abs(quantity))
            slippage = self.calculate_slippage(notional)

            trade = {
                'timestamp': date,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'notional': notional,
                'commission': commission,
                'slippage': slippage,
                'total_cost': commission + slippage
            }
            self.trades.append(trade)

            total_cost += trade['total_cost']

            position['positions'][symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'weight': weight
            }

        position['total_cost'] = total_cost
        active_positions[vector_id] = position

        self.risk_manager.register_position_entry(vector_id, date, allocation)

        logger.debug(f"Entered position {vector_id} on {date}: {signal['action']}, z-score: {signal['zscore']:.2f}")

    def _close_position(
        self,
        vector_id: int,
        position: Dict,
        current_prices,
        date: datetime,
        reason: str
    ):
        """Close an existing position."""
        total_pnl = 0

        for symbol, stock_pos in position['positions'].items():
            if isinstance(current_prices, pd.Series):
                if symbol not in current_prices:
                    continue
                current_price = current_prices[symbol]
            else:
                if symbol not in current_prices:
                    continue
                current_price = current_prices[symbol]

            quantity = stock_pos['quantity']
            entry_price = stock_pos['entry_price']

            commission = self.calculate_commission(quantity * current_price, abs(quantity))
            slippage = self.calculate_slippage(quantity * current_price)

            trade = {
                'timestamp': date,
                'symbol': symbol,
                'quantity': -quantity,
                'price': current_price,
                'notional': -quantity * current_price,
                'commission': commission,
                'slippage': slippage,
                'total_cost': commission + slippage
            }
            self.trades.append(trade)

            pnl = (current_price - entry_price) * quantity - trade['total_cost']
            total_pnl += pnl

        self.trades.append({
            'timestamp': date,
            'vector_id': vector_id,
            'type': 'CLOSE_POSITION',
            'pnl': total_pnl,
            'reason': reason,
            'entry_date': position['entry_date'],
            'hold_days': (date - position['entry_date']).days
        })

        self.risk_manager.unregister_position(vector_id)

        logger.debug(f"Closed position {vector_id} on {date}: P&L ${total_pnl:,.2f}, reason: {reason}")

    def _calculate_position_pnl(
        self,
        vector_id: int,
        position: Dict,
        current_prices
    ) -> float:
        """Calculate current P&L for a position."""
        total_pnl = 0

        for symbol, stock_pos in position['positions'].items():
            if isinstance(current_prices, pd.Series):
                if symbol not in current_prices:
                    continue
                current_price = current_prices[symbol]
            else:
                if symbol not in current_prices:
                    continue
                current_price = current_prices[symbol]

            entry_price = stock_pos['entry_price']
            quantity = stock_pos['quantity']

            pnl = (current_price - entry_price) * quantity
            total_pnl += pnl

        return total_pnl

    def _get_current_prices(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> pd.Series:
        """Get current prices for symbols."""
        prices = {}

        for symbol in symbols:
            if symbol in price_data:
                symbol_data = price_data[symbol][price_data[symbol].index <= date]
                if not symbol_data.empty:
                    prices[symbol] = symbol_data['close'].iloc[-1]

        return pd.Series(prices)
