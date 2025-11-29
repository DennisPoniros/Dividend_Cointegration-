#!/usr/bin/env python3
"""
Dividend Momentum Strategy - Non-Forward Looking Backtest

Strategy:
1. Rank dividend stocks by momentum (returns over past N days)
2. Go long top quintile, short bottom quintile
3. Rebalance monthly
4. No forward-looking bias - uses only historical data

This is a proven factor strategy that should deliver positive Sharpe.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from src.data.loader import DataLoader


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class MomentumStrategy:
    """
    Cross-sectional momentum strategy on dividend stocks.
    No forward-looking bias.
    """

    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Strategy parameters
        self.MOMENTUM_LOOKBACK = 252     # 12-month momentum (skip last month)
        self.SKIP_RECENT = 21            # Skip most recent month (avoid reversal)
        self.REBALANCE_FREQ = 21         # Monthly rebalance
        self.NUM_LONG = 20               # Top 20 stocks
        self.NUM_SHORT = 20              # Bottom 20 stocks
        self.COMMISSION_BPS = 5          # Transaction costs

        self.price_data = {}
        self.returns_data = {}

    def load_data(self, symbols):
        """Load price data for all symbols."""
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > self.MOMENTUM_LOOKBACK + 50:
                self.price_data[symbol] = df['close']
                self.returns_data[symbol] = df['close'].pct_change()
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

    def calculate_momentum(self, end_date):
        """
        Calculate momentum for all stocks using only data up to end_date.
        Momentum = returns from (t - LOOKBACK - SKIP) to (t - SKIP)
        """
        momentum_scores = {}

        for symbol, prices in self.price_data.items():
            prices_hist = prices[prices.index <= end_date]

            if len(prices_hist) < self.MOMENTUM_LOOKBACK + self.SKIP_RECENT + 10:
                continue

            # Get price at start and end of momentum window
            end_idx = len(prices_hist) - self.SKIP_RECENT - 1
            start_idx = end_idx - self.MOMENTUM_LOOKBACK

            if start_idx < 0 or end_idx >= len(prices_hist):
                continue

            start_price = prices_hist.iloc[start_idx]
            end_price = prices_hist.iloc[end_idx]

            if start_price > 0:
                momentum = (end_price / start_price) - 1
                momentum_scores[symbol] = momentum

        return momentum_scores

    def run(self):
        """Run the momentum strategy backtest."""
        # Get all trading dates
        all_dates = set()
        for series in self.price_data.values():
            all_dates.update(series.index)
        all_dates = sorted(all_dates)

        # Start after we have enough history
        start_idx = self.MOMENTUM_LOOKBACK + self.SKIP_RECENT + 50
        if start_idx >= len(all_dates):
            return None

        trading_dates = all_dates[start_idx:]
        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

        # State
        capital = self.initial_capital
        positions = {}  # symbol -> shares
        equity_curve = []
        last_rebalance = None
        trades = []

        for i, date in enumerate(trading_dates):
            # Calculate daily P&L
            daily_pnl = 0
            for symbol, shares in positions.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    prev_date_idx = list(self.price_data[symbol].index).index(date) - 1
                    if prev_date_idx >= 0:
                        prev_date = self.price_data[symbol].index[prev_date_idx]
                        if prev_date in self.price_data[symbol].index:
                            prev_price = self.price_data[symbol].loc[prev_date]
                            curr_price = self.price_data[symbol].loc[date]
                            daily_pnl += shares * (curr_price - prev_price)

            capital += daily_pnl

            # Rebalance check
            should_rebalance = False
            if last_rebalance is None:
                should_rebalance = True
            elif (date - last_rebalance).days >= self.REBALANCE_FREQ:
                should_rebalance = True

            if should_rebalance:
                # Calculate momentum rankings using only historical data
                momentum_scores = self.calculate_momentum(date)

                if len(momentum_scores) < self.NUM_LONG + self.NUM_SHORT:
                    continue

                # Rank stocks by momentum
                sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
                long_stocks = [s[0] for s in sorted_stocks[:self.NUM_LONG]]
                short_stocks = [s[0] for s in sorted_stocks[-self.NUM_SHORT:]]

                # Close old positions and open new ones
                old_positions = set(positions.keys())
                new_positions = set(long_stocks + short_stocks)

                # Calculate turnover cost
                closed_positions = old_positions - new_positions
                opened_positions = new_positions - old_positions

                for symbol in closed_positions:
                    if symbol in positions and symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date]
                        notional = abs(positions[symbol] * price)
                        cost = notional * self.COMMISSION_BPS / 10000
                        capital -= cost

                # Set up new positions
                positions = {}
                position_size = capital / (self.NUM_LONG + self.NUM_SHORT)

                for symbol in long_stocks:
                    if symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date]
                        shares = position_size / price  # Long
                        positions[symbol] = shares

                        if symbol in opened_positions:
                            cost = position_size * self.COMMISSION_BPS / 10000
                            capital -= cost

                for symbol in short_stocks:
                    if symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date]
                        shares = -position_size / price  # Short
                        positions[symbol] = shares

                        if symbol in opened_positions:
                            cost = position_size * self.COMMISSION_BPS / 10000
                            capital -= cost

                last_rebalance = date

                if i % 100 == 0:
                    self.logger.info(f"  {date.date()}: Rebalanced - {len(long_stocks)} long, {len(short_stocks)} short")

            # Track equity
            position_value = 0
            for symbol, shares in positions.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[date]
                    position_value += shares * price

            equity = capital + position_value

            equity_curve.append({
                'date': date,
                'equity': equity,
                'capital': capital,
                'position_value': position_value,
                'num_positions': len(positions)
            })

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        returns = equity_df['equity'].pct_change().dropna()

        rf_daily = 0.02 / 252
        excess_returns = returns - rf_daily

        metrics = {
            'total_return': (equity_df['equity'].iloc[-1] / self.initial_capital) - 1,
            'annual_return': returns.mean() * 252,
            'annual_vol': returns.std() * np.sqrt(252),
            'sharpe': excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'sortino': excess_returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
            'max_drawdown': ((equity_df['equity'] / equity_df['equity'].expanding().max()) - 1).min(),
            'calmar': (returns.mean() * 252) / abs(((equity_df['equity'] / equity_df['equity'].expanding().max()) - 1).min()) if abs(((equity_df['equity'] / equity_df['equity'].expanding().max()) - 1).min()) > 0 else 0
        }

        return {
            'equity_curve': equity_df,
            'metrics': metrics
        }


def print_results(results):
    """Print formatted results."""
    if not results:
        print("No results")
        return

    metrics = results['metrics']

    print("\n" + "="*70)
    print("DIVIDEND MOMENTUM STRATEGY - BACKTEST RESULTS")
    print("(No Forward-Looking Bias)")
    print("="*70)

    print(f"\n{'PERFORMANCE':-^50}")
    print(f"  Total Return:         {metrics['total_return']*100:>12.2f}%")
    print(f"  Annual Return:        {metrics['annual_return']*100:>12.2f}%")
    print(f"  Annual Volatility:    {metrics['annual_vol']*100:>12.2f}%")
    print(f"  Sharpe Ratio:         {metrics['sharpe']:>12.2f}")
    print(f"  Sortino Ratio:        {metrics['sortino']:>12.2f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown']*100:>12.2f}%")
    print(f"  Calmar Ratio:         {metrics['calmar']:>12.2f}")

    print("\n" + "="*70)

    if metrics['sharpe'] >= 1.0:
        print("SUCCESS: Sharpe Ratio >= 1.0!")
    else:
        print(f"Current Sharpe: {metrics['sharpe']:.2f}")


def main():
    logger = setup_logging()

    print("\n" + "="*70)
    print("DIVIDEND MOMENTUM STRATEGY BACKTEST")
    print("(No Forward-Looking Bias)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols")

    strategy = MomentumStrategy(loader)
    strategy.load_data(symbols)

    print("\nRunning momentum backtest...")
    results = strategy.run()

    if results:
        print_results(results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
