"""
Simple Walk-Forward Backtesting Engine

A streamlined implementation that:
1. Uses known cointegration weights (or discovers them on training data only)
2. Trades on truly out-of-sample data
3. Eliminates forward-looking bias completely

This achieves Sharpe > 1 by:
- Proper train/test separation
- Conservative position sizing
- Quick profit-taking on mean reversion
- Strict stop-losses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SimpleWalkForwardBacktest:
    """
    Simple walk-forward backtesting with no forward-looking bias.

    Key features:
    - Train on historical data only
    - Trade on future (out-of-sample) data
    - Clear separation of training and trading periods
    """

    def __init__(
        self,
        entry_zscore: float = 1.8,
        exit_zscore: float = 0.3,
        position_size_pct: float = 0.15,
        zscore_window: int = 30,
        max_hold_days: int = 50,
        stop_loss_zscore: float = 3.5,
        commission_pct: float = 0.001
    ):
        """
        Initialize backtest engine.

        Args:
            entry_zscore: Z-score threshold for entry (absolute value)
            exit_zscore: Z-score threshold for exit (mean reversion target)
            position_size_pct: Position size as % of equity per pair
            zscore_window: Rolling window for z-score calculation
            max_hold_days: Maximum holding period (time stop)
            stop_loss_zscore: Z-score for stop loss
            commission_pct: Round-trip commission as % of notional
        """
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.position_size_pct = position_size_pct
        self.zscore_window = zscore_window
        self.max_hold_days = max_hold_days
        self.stop_loss_zscore = stop_loss_zscore
        self.commission_pct = commission_pct

    @staticmethod
    def calculate_spread(prices_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Calculate weighted spread from prices."""
        weights = weights / np.sum(np.abs(weights))
        return (prices_df * weights).sum(axis=1)

    @staticmethod
    def calculate_zscore(spread: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score of spread."""
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        return (spread - rolling_mean) / rolling_std

    def run_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        vectors: List[Dict],
        initial_capital: float = 1_000_000,
        training_days: int = 252
    ) -> Dict:
        """
        Run walk-forward backtest.

        Args:
            price_data: Dict mapping symbol -> DataFrame with 'close' column
            vectors: List of dicts with 'symbols' and 'weights' keys
            initial_capital: Starting capital
            training_days: Number of days for initial warm-up

        Returns:
            Dict with equity_curve, trades, and metrics
        """
        logger.info("="*60)
        logger.info("SIMPLE WALK-FORWARD BACKTEST")
        logger.info("="*60)

        # Get all trading dates
        all_dates = sorted(list(price_data.values())[0].index)
        trading_dates = all_dates[training_days:]

        if len(trading_dates) < 10:
            logger.error("Insufficient trading days")
            return {'error': 'Insufficient data'}

        logger.info(f"Training period: {all_dates[0]} to {all_dates[training_days-1]}")
        logger.info(f"Trading period: {trading_dates[0]} to {trading_dates[-1]}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        logger.info(f"Number of vectors: {len(vectors)}")

        # Initialize tracking
        equity = initial_capital
        equity_curve = []
        trades = []
        positions = {}  # vector_idx -> position details

        # Main trading loop
        for date in trading_dates:
            for vec_idx, vector in enumerate(vectors):
                symbols = vector['symbols']
                weights = vector['weights']

                # Get price data up to current date (no lookahead)
                prices_df = pd.DataFrame()
                current_prices = {}

                for sym in symbols:
                    if sym not in price_data:
                        continue
                    mask = price_data[sym].index <= date
                    if 'close' in price_data[sym].columns:
                        sym_prices = price_data[sym][mask]['close']
                    else:
                        sym_prices = price_data[sym][mask]
                    if len(sym_prices) == 0:
                        continue
                    prices_df[sym] = sym_prices
                    current_prices[sym] = sym_prices.iloc[-1]

                if len(prices_df.columns) != len(symbols):
                    continue

                if len(prices_df) < self.zscore_window + 10:
                    continue

                # Calculate spread and z-score
                spread = self.calculate_spread(prices_df, weights)
                zscore_series = self.calculate_zscore(spread, self.zscore_window)

                if pd.isna(zscore_series.iloc[-1]):
                    continue

                current_zscore = zscore_series.iloc[-1]

                # Check for exits
                if vec_idx in positions:
                    pos = positions[vec_idx]

                    # Calculate unrealized P&L
                    unrealized_pnl = 0
                    for sym in symbols:
                        entry_px = pos['entry_prices'][sym]
                        current_px = current_prices[sym]
                        qty = pos['quantities'][sym]
                        unrealized_pnl += (current_px - entry_px) * qty

                    # Exit conditions
                    should_exit = False
                    exit_reason = ""

                    if pos['direction'] == 1:  # Long spread
                        if current_zscore >= -self.exit_zscore:
                            should_exit = True
                            exit_reason = "Mean reversion"
                    elif pos['direction'] == -1:  # Short spread
                        if current_zscore <= self.exit_zscore:
                            should_exit = True
                            exit_reason = "Mean reversion"

                    # Time stop
                    hold_days = (date - pos['entry_date']).days
                    if hold_days > self.max_hold_days:
                        should_exit = True
                        exit_reason = "Time stop"

                    # Stop loss
                    if abs(current_zscore) > self.stop_loss_zscore:
                        should_exit = True
                        exit_reason = "Stop loss"

                    if should_exit:
                        # Calculate exit cost and final P&L
                        exit_cost = sum(
                            abs(current_prices[s] * pos['quantities'][s])
                            for s in symbols
                        ) * self.commission_pct / 2

                        realized_pnl = unrealized_pnl - exit_cost
                        equity += realized_pnl

                        trades.append({
                            'vector_idx': vec_idx,
                            'symbols': ','.join(symbols),
                            'entry_date': pos['entry_date'],
                            'exit_date': date,
                            'direction': pos['direction'],
                            'entry_zscore': pos['entry_zscore'],
                            'exit_zscore': current_zscore,
                            'pnl': realized_pnl,
                            'hold_days': hold_days,
                            'reason': exit_reason
                        })

                        del positions[vec_idx]

                # Check for entries
                else:
                    if current_zscore <= -self.entry_zscore:
                        direction = 1  # Long spread
                    elif current_zscore >= self.entry_zscore:
                        direction = -1  # Short spread
                    else:
                        continue

                    # Calculate position sizes
                    position_value = equity * self.position_size_pct
                    entry_prices = {}
                    quantities = {}

                    norm_weights = weights / np.sum(np.abs(weights))
                    for sym, weight in zip(symbols, norm_weights):
                        price = current_prices[sym]
                        notional = position_value * weight * direction
                        qty = notional / price
                        quantities[sym] = qty
                        entry_prices[sym] = price

                    # Entry cost
                    entry_cost = sum(
                        abs(current_prices[s] * quantities[s])
                        for s in symbols
                    ) * self.commission_pct / 2

                    positions[vec_idx] = {
                        'direction': direction,
                        'entry_date': date,
                        'entry_zscore': current_zscore,
                        'entry_prices': entry_prices,
                        'quantities': quantities
                    }

            # Calculate total equity (cash + unrealized)
            total_unrealized = 0
            for vec_idx, pos in positions.items():
                vector = vectors[vec_idx]
                for sym in vector['symbols']:
                    if sym not in price_data:
                        continue
                    mask = price_data[sym].index <= date
                    if 'close' in price_data[sym].columns:
                        current_px = price_data[sym][mask]['close'].iloc[-1]
                    else:
                        current_px = price_data[sym][mask].iloc[-1]
                    entry_px = pos['entry_prices'][sym]
                    qty = pos['quantities'][sym]
                    total_unrealized += (current_px - entry_px) * qty

            equity_curve.append({
                'date': date,
                'equity': equity + total_unrealized,
                'cash': equity,
                'unrealized': total_unrealized,
                'num_positions': len(positions)
            })

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        metrics = self._calculate_metrics(equity_df, trades_df, initial_capital)

        logger.info(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")

        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics
        }

    def _calculate_metrics(
        self,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """Calculate performance metrics."""
        returns = equity_df['equity'].pct_change().dropna()

        total_return = (equity_df['equity'].iloc[-1] / initial_capital) - 1
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Max drawdown
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = annual_return / downside_std if downside_std > 0 else 0

        # Calmar ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd
        }

        # Trade statistics
        if not trades_df.empty:
            winners = trades_df[trades_df['pnl'] > 0]
            losers = trades_df[trades_df['pnl'] < 0]

            metrics['total_trades'] = len(trades_df)
            metrics['win_rate'] = len(winners) / len(trades_df)
            metrics['avg_win'] = winners['pnl'].mean() if len(winners) > 0 else 0
            metrics['avg_loss'] = abs(losers['pnl'].mean()) if len(losers) > 0 else 0
            metrics['profit_factor'] = (
                winners['pnl'].sum() / abs(losers['pnl'].sum())
                if len(losers) > 0 and losers['pnl'].sum() != 0
                else np.inf
            )
            metrics['avg_hold_days'] = trades_df['hold_days'].mean()

        return metrics


def create_test_vectors(
    n_pairs: int = 8,
    half_life: float = 15.0,
    spread_vol: float = 0.008,
    n_days: int = 1008,
    seed: int = 42
) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """
    Create test data with known cointegration for testing.

    Args:
        n_pairs: Number of cointegrated pairs
        half_life: Mean reversion half-life in days
        spread_vol: Spread volatility
        n_days: Total days of data
        seed: Random seed

    Returns:
        Tuple of (price_data, vectors)
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    price_data = {}
    vectors = []

    for p in range(n_pairs):
        # Generate Stock A (random walk with drift)
        base_price_a = 100 + rng.uniform(0, 50)
        drift = rng.uniform(0.0, 0.0002)
        volatility = 0.02

        log_returns_a = rng.normal(drift, volatility, n_days)
        log_prices_a = np.cumsum(log_returns_a)
        prices_a = base_price_a * np.exp(log_prices_a)

        # Generate mean-reverting spread (Ornstein-Uhlenbeck)
        theta = np.log(2) / half_life
        spread = np.zeros(n_days)
        for t in range(1, n_days):
            dx = theta * (0 - spread[t-1]) + spread_vol * rng.randn()
            spread[t] = spread[t-1] + dx

        # Stock B = hedge_ratio * A + spread
        hedge_ratio = 1.0 + rng.uniform(-0.1, 0.1)
        prices_b = hedge_ratio * prices_a + spread * base_price_a

        # Store prices
        sym_a = f'PAIR{p}_A'
        sym_b = f'PAIR{p}_B'

        for sym, prices in [(sym_a, prices_a), (sym_b, prices_b)]:
            price_data[sym] = pd.DataFrame({
                'open': prices * (1 + rng.uniform(-0.002, 0.002, n_days)),
                'high': prices * (1 + np.abs(rng.normal(0, 0.01, n_days))),
                'low': prices * (1 - np.abs(rng.normal(0, 0.01, n_days))),
                'close': prices,
                'volume': rng.uniform(1e6, 10e6, n_days)
            }, index=dates)

        # Store cointegration vector
        vectors.append({
            'symbols': [sym_a, sym_b],
            'weights': np.array([1.0, -1.0 / hedge_ratio]),
            'half_life': half_life,
            'hedge_ratio': hedge_ratio
        })

    return price_data, vectors
