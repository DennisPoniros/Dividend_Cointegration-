"""
Backtesting Engine

Simulates the complete trading strategy on historical data with:
- Transaction costs
- Slippage
- Realistic execution
- Complete trade tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import yaml
import logging
from ..strategy.signals import SignalGenerator
from ..strategy.portfolio import PortfolioManager
from ..risk.risk_management import RiskManager
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for dividend cointegration strategy."""

    def __init__(self, config_path: str = "config/strategy_params.yaml"):
        """
        Initialize backtest engine.

        Args:
            config_path: Path to strategy parameters
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.backtest_config = self.config['backtesting']

        # Initialize components
        self.signal_generator = SignalGenerator(config_path)
        self.portfolio_manager = PortfolioManager(config_path)
        self.risk_manager = RiskManager(config_path)

        # Tracking
        self.trades = []
        self.daily_positions = []
        self.equity_curve = []

    def calculate_commission(self, notional: float, num_shares: float) -> float:
        """
        Calculate commission costs.

        Args:
            notional: Notional value of trade
            num_shares: Number of shares

        Returns:
            Commission cost
        """
        per_share_cost = self.backtest_config['commission_per_share']
        return num_shares * per_share_cost

    def calculate_slippage(self, notional: float) -> float:
        """
        Calculate slippage costs.

        Args:
            notional: Notional value of trade

        Returns:
            Slippage cost
        """
        slippage_bps = self.backtest_config['slippage_bps']
        return abs(notional) * (slippage_bps / 10000)

    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime
    ) -> Dict:
        """
        Simulate trade execution with costs.

        Args:
            symbol: Stock symbol
            quantity: Number of shares (positive=buy, negative=sell)
            price: Execution price
            timestamp: Trade timestamp

        Returns:
            Trade details
        """
        notional = quantity * price
        commission = self.calculate_commission(notional, abs(quantity))
        slippage = self.calculate_slippage(notional)

        total_cost = commission + slippage

        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'notional': notional,
            'commission': commission,
            'slippage': slippage,
            'total_cost': total_cost
        }

        return trade

    def calculate_spread(
        self,
        price_data: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """Calculate cointegration spread."""
        weights = weights / np.sum(np.abs(weights))
        spread = (price_data * weights).sum(axis=1)
        return spread

    def run_backtest(
        self,
        vectors: List[Dict],
        price_data: Dict[str, pd.DataFrame],
        initial_capital: float = 1_000_000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Run complete backtest.

        Args:
            vectors: List of cointegration vectors
            price_data: Dictionary mapping symbols to price DataFrames
            initial_capital: Starting capital
            start_date: Backtest start date (None = earliest available)
            end_date: Backtest end date (None = latest available)

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with ${initial_capital:,.2f} initial capital")

        # Set portfolio capital
        self.portfolio_manager.set_capital(initial_capital)

        # Get date range
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index)

        all_dates = sorted(all_dates)

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        logger.info(f"Backtest period: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")

        # Initialize tracking
        current_capital = initial_capital
        active_positions = {}  # vector_id -> position details
        self.trades = []
        self.equity_curve = []

        # Main backtest loop
        for date in all_dates:
            # Check if trading is allowed
            if not self.risk_manager.is_trading_allowed(date):
                # Still track equity
                self.equity_curve.append({
                    'date': date,
                    'equity': current_capital,
                    'num_positions': len(active_positions)
                })
                continue

            # Generate signals for all vectors
            daily_signals = []

            for i, vector in enumerate(vectors):
                try:
                    # Get price data for this vector up to current date
                    symbols = vector['symbols']
                    vector_prices = pd.DataFrame()

                    for symbol in symbols:
                        if symbol in price_data:
                            # Get data up to current date
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
                        position_pnl = self.calculate_position_pnl(
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
                            # Force exit due to risk stop
                            self.close_position(
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
                    logger.error(f"Error processing vector {i} on {date}: {e}")
                    continue

            # Process signals
            for signal in daily_signals:
                vector_id = signal['vector_id']
                action = signal['action']
                vector = signal['vector']

                if action in ['ENTER_LONG', 'ENTER_SHORT']:
                    # Enter new position
                    if vector_id not in active_positions:
                        self.enter_position(
                            vector_id,
                            vector,
                            signal,
                            price_data,
                            date,
                            active_positions
                        )

                elif action == 'EXIT':
                    # Exit existing position
                    if vector_id in active_positions:
                        # Get current prices
                        current_prices = {}
                        for symbol in vector['symbols']:
                            if symbol in price_data:
                                symbol_data = price_data[symbol][price_data[symbol].index <= date]
                                if not symbol_data.empty:
                                    current_prices[symbol] = symbol_data['close'].iloc[-1]

                        self.close_position(
                            vector_id,
                            active_positions[vector_id],
                            pd.Series(current_prices),
                            date,
                            "Signal exit"
                        )
                        del active_positions[vector_id]

            # Calculate daily equity
            daily_pnl = sum(
                self.calculate_position_pnl(vid, pos, self.get_current_prices(pos['symbols'], price_data, date))
                for vid, pos in active_positions.items()
            )

            current_equity = current_capital + daily_pnl

            # Check portfolio-level drawdown
            action = self.risk_manager.check_portfolio_drawdown(current_equity)
            if action == 'exit_all':
                # Close all positions
                for vector_id in list(active_positions.keys()):
                    pos = active_positions[vector_id]
                    current_prices = self.get_current_prices(pos['symbols'], price_data, date)
                    self.close_position(vector_id, pos, current_prices, date, "Portfolio drawdown")
                active_positions.clear()

                self.risk_manager.apply_drawdown_action(action, date)

            elif action == 'cut_50pct':
                # Reduce all positions by 50%
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

        # Compile results
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        trades_df = pd.DataFrame(self.trades)

        # Calculate metrics
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_df['equity'],
            trades_df if not trades_df.empty else None
        )

        results = {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics,
            'final_capital': current_equity,
            'total_return': (current_equity / initial_capital) - 1
        }

        logger.info(f"Backtest complete. Final capital: ${current_equity:,.2f}")
        logger.info(f"Total return: {results['total_return']*100:.2f}%")

        return results

    def enter_position(
        self,
        vector_id: int,
        vector: Dict,
        signal: Dict,
        price_data: Dict[str, pd.DataFrame],
        date: datetime,
        active_positions: Dict
    ):
        """Enter a new position."""
        # Calculate position size
        allocation = self.portfolio_manager.calculate_position_size(vector, signal)

        # Get current prices
        current_prices = {}
        for symbol in vector['symbols']:
            if symbol in price_data:
                symbol_data = price_data[symbol][price_data[symbol].index <= date]
                if not symbol_data.empty:
                    current_prices[symbol] = symbol_data['close'].iloc[-1]

        # Calculate quantities for each stock
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

            # Execute trade
            trade = self.execute_trade(symbol, quantity, price, date)
            self.trades.append(trade)

            total_cost += trade['total_cost']

            position['positions'][symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'weight': weight
            }

        position['total_cost'] = total_cost

        active_positions[vector_id] = position

        # Register with risk manager
        self.risk_manager.register_position_entry(vector_id, date, allocation)

        logger.info(f"Entered position {vector_id} on {date}: {signal['action']}, z-score: {signal['zscore']:.2f}")

    def close_position(
        self,
        vector_id: int,
        position: Dict,
        current_prices: pd.Series,
        date: datetime,
        reason: str
    ):
        """Close an existing position."""
        total_pnl = 0

        for symbol, stock_pos in position['positions'].items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            quantity = stock_pos['quantity']
            entry_price = stock_pos['entry_price']

            # Close trade (opposite direction)
            trade = self.execute_trade(symbol, -quantity, current_price, date)
            self.trades.append(trade)

            # Calculate P&L
            pnl = (current_price - entry_price) * quantity - trade['total_cost']
            total_pnl += pnl

        # Record completed trade
        self.trades.append({
            'timestamp': date,
            'vector_id': vector_id,
            'type': 'CLOSE_POSITION',
            'pnl': total_pnl,
            'reason': reason,
            'entry_date': position['entry_date'],
            'hold_days': (date - position['entry_date']).days
        })

        # Unregister from risk manager
        self.risk_manager.unregister_position(vector_id)

        logger.info(f"Closed position {vector_id} on {date}: P&L ${total_pnl:,.2f}, reason: {reason}")

    def calculate_position_pnl(
        self,
        vector_id: int,
        position: Dict,
        current_prices: pd.Series
    ) -> float:
        """Calculate current P&L for a position."""
        total_pnl = 0

        for symbol, stock_pos in position['positions'].items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = stock_pos['entry_price']
            quantity = stock_pos['quantity']

            pnl = (current_price - entry_price) * quantity
            total_pnl += pnl

        return total_pnl

    def get_current_prices(
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
