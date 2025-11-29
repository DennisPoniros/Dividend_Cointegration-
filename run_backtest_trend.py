#!/usr/bin/env python3
"""
Trend-Following Dividend Strategy

Strategy:
1. Go long dividend stocks that are above their 200-day moving average
2. Stay in cash when stock is below 200-day MA
3. Use momentum to rank stocks above their MA
4. This provides crash protection and captures uptrends
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


class TrendFollowingStrategy:
    """
    Trend-following strategy on dividend stocks.
    Only holds stocks above their 200-day MA.
    """

    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Strategy parameters
        self.TREND_MA = 200            # 200-day moving average filter
        self.MOMENTUM_LOOKBACK = 63    # 3-month momentum for ranking
        self.REBALANCE_FREQ = 21       # Monthly rebalance
        self.NUM_HOLDINGS = 15         # Concentrated portfolio
        self.COMMISSION_BPS = 5

        self.price_data = {}

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > self.TREND_MA + 100:
                self.price_data[symbol] = df['close']
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

    def is_above_trend(self, symbol, date):
        """Check if stock is above its 200-day MA."""
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < self.TREND_MA + 5:
            return False

        current_price = prices.iloc[-1]
        ma = prices.iloc[-self.TREND_MA:].mean()

        return current_price > ma

    def calculate_momentum(self, symbol, date):
        """Calculate momentum for stocks above trend."""
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < self.MOMENTUM_LOOKBACK + 10:
            return None

        end_price = prices.iloc[-1]
        start_price = prices.iloc[-self.MOMENTUM_LOOKBACK - 1]

        if start_price <= 0:
            return None

        return (end_price / start_price) - 1

    def run(self):
        all_dates = set()
        for series in self.price_data.values():
            all_dates.update(series.index)
        all_dates = sorted(all_dates)

        start_idx = self.TREND_MA + 50
        trading_dates = all_dates[start_idx:]
        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()}")

        cash = self.initial_capital
        positions = {}
        equity_curve = []
        last_rebalance = None

        for i, date in enumerate(trading_dates):
            # Calculate current equity
            position_value = 0
            for symbol, shares in positions.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[date]
                    position_value += shares * price

            total_equity = cash + position_value

            # Rebalance check
            should_rebalance = last_rebalance is None or (date - last_rebalance).days >= self.REBALANCE_FREQ

            if should_rebalance:
                # Find stocks above trend with positive momentum
                candidates = []
                for symbol in self.price_data.keys():
                    if date not in self.price_data[symbol].index:
                        continue

                    if not self.is_above_trend(symbol, date):
                        continue

                    momentum = self.calculate_momentum(symbol, date)
                    if momentum is not None and momentum > 0:  # Only positive momentum
                        candidates.append((symbol, momentum))

                # Sort by momentum, take top N
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_stocks = [c[0] for c in candidates[:self.NUM_HOLDINGS]]

                # Liquidate old positions
                for symbol, shares in positions.items():
                    if symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date]
                        value = shares * price
                        cash += value
                        cash -= abs(value) * self.COMMISSION_BPS / 10000

                positions = {}

                # Open new positions if we have candidates
                if top_stocks:
                    available = cash
                    weight = 1.0 / len(top_stocks)  # Equal weight

                    for symbol in top_stocks:
                        if symbol in self.price_data and date in self.price_data[symbol].index:
                            price = self.price_data[symbol].loc[date]
                            target_value = available * weight * 0.98  # 2% buffer
                            shares = target_value / price
                            positions[symbol] = shares
                            cash -= target_value
                            cash -= target_value * self.COMMISSION_BPS / 10000

                last_rebalance = date

                if i % 100 == 0:
                    pos_val = sum(positions[s] * self.price_data[s].loc[date] for s in positions if s in self.price_data and date in self.price_data[s].index)
                    pct_invested = (pos_val / total_equity * 100) if total_equity > 0 else 0
                    self.logger.info(f"  {date.date()}: {len(positions)} stocks, {pct_invested:.0f}% invested")

            # Recalculate equity
            position_value = 0
            for symbol, shares in positions.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[date]
                    position_value += shares * price
            total_equity = cash + position_value

            equity_curve.append({'date': date, 'equity': total_equity})

        # Metrics
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
        }

        return {'equity_curve': equity_df, 'metrics': metrics}


def print_results(results):
    if not results:
        print("No results")
        return

    metrics = results['metrics']

    print("\n" + "="*70)
    print("TREND-FOLLOWING DIVIDEND STRATEGY - RESULTS")
    print("(No Forward-Looking Bias)")
    print("="*70)

    print(f"\n{'PERFORMANCE':-^50}")
    print(f"  Total Return:         {metrics['total_return']*100:>12.2f}%")
    print(f"  Annual Return:        {metrics['annual_return']*100:>12.2f}%")
    print(f"  Annual Volatility:    {metrics['annual_vol']*100:>12.2f}%")
    print(f"  Sharpe Ratio:         {metrics['sharpe']:>12.2f}")
    print(f"  Sortino Ratio:        {metrics['sortino']:>12.2f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown']*100:>12.2f}%")

    print("\n" + "="*70)

    if metrics['sharpe'] >= 1.0:
        print("SUCCESS: Sharpe Ratio >= 1.0!")
    else:
        print(f"Current Sharpe: {metrics['sharpe']:.2f}")


def main():
    logger = setup_logging()

    print("\n" + "="*70)
    print("TREND-FOLLOWING DIVIDEND STRATEGY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols")

    strategy = TrendFollowingStrategy(loader)
    strategy.load_data(symbols)

    print("\nRunning backtest...")
    results = strategy.run()

    if results:
        print_results(results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
