#!/usr/bin/env python3
"""
Final Backtest V2 - Volatility-Managed Dividend Momentum Strategy

Key improvements:
1. Volatility targeting (10% annual vol target)
2. Reduce exposure in high-vol regimes
3. More diversification (20 stocks)
4. Faster momentum (3-month)
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


class VolManagedMomentum:
    """
    Volatility-managed momentum strategy on dividend stocks.
    """

    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Strategy parameters
        self.MOMENTUM_LOOKBACK = 63        # 3-month momentum
        self.SKIP_RECENT = 5               # Skip last week
        self.REBALANCE_FREQ = 21           # Monthly
        self.NUM_HOLDINGS = 20             # More diversified
        self.TARGET_VOL = 0.10             # 10% annual vol target
        self.PORTFOLIO_VOL_LOOKBACK = 21   # 1 month for portfolio vol estimate
        self.MAX_LEVERAGE = 1.5            # Max 150% exposure
        self.MIN_LEVERAGE = 0.3            # Min 30% exposure
        self.COMMISSION_BPS = 5

        self.price_data = {}
        self.returns_data = {}

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > self.MOMENTUM_LOOKBACK + 100:
                self.price_data[symbol] = df['close']
                self.returns_data[symbol] = df['close'].pct_change()
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

    def calculate_momentum(self, end_date):
        """Calculate momentum scores using only historical data."""
        scores = {}

        for symbol, prices in self.price_data.items():
            prices_hist = prices[prices.index <= end_date]

            if len(prices_hist) < self.MOMENTUM_LOOKBACK + self.SKIP_RECENT + 20:
                continue

            end_idx = len(prices_hist) - self.SKIP_RECENT - 1
            start_idx = end_idx - self.MOMENTUM_LOOKBACK

            if start_idx < 0:
                continue

            start_price = prices_hist.iloc[start_idx]
            end_price = prices_hist.iloc[end_idx]

            if start_price <= 0:
                continue

            momentum = (end_price / start_price) - 1
            scores[symbol] = momentum

        return scores

    def calculate_portfolio_vol(self, positions, end_date):
        """
        Estimate portfolio volatility using historical returns.
        Only uses data up to end_date.
        """
        if not positions:
            return 0.15  # Default assumption

        # Get returns for all positions
        returns_list = []
        weights = []

        total_value = sum(
            positions[s] * self.price_data[s][self.price_data[s].index <= end_date].iloc[-1]
            for s in positions if s in self.price_data
        )

        if total_value <= 0:
            return 0.15

        for symbol, shares in positions.items():
            if symbol not in self.returns_data:
                continue

            ret = self.returns_data[symbol][self.returns_data[symbol].index <= end_date]
            ret = ret.iloc[-self.PORTFOLIO_VOL_LOOKBACK:]

            if len(ret) < 10:
                continue

            price = self.price_data[symbol][self.price_data[symbol].index <= end_date].iloc[-1]
            weight = (shares * price) / total_value

            returns_list.append(ret.values)
            weights.append(weight)

        if not returns_list:
            return 0.15

        # Simple weighted average of individual volatilities
        vols = [np.std(r) * np.sqrt(252) for r in returns_list]
        port_vol = sum(w * v for w, v in zip(weights, vols))

        return port_vol if port_vol > 0 else 0.15

    def run(self):
        all_dates = set()
        for series in self.price_data.values():
            all_dates.update(series.index)
        all_dates = sorted(all_dates)

        start_idx = self.MOMENTUM_LOOKBACK + self.SKIP_RECENT + 50
        if start_idx >= len(all_dates):
            return None

        trading_dates = all_dates[start_idx:]
        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

        cash = self.initial_capital
        positions = {}
        equity_curve = []
        last_rebalance = None
        current_leverage = 1.0

        for i, date in enumerate(trading_dates):
            # Calculate position values
            position_value = 0
            for symbol, shares in positions.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[date]
                    position_value += shares * price

            total_equity = cash + position_value

            # Rebalance check
            should_rebalance = False
            if last_rebalance is None:
                should_rebalance = True
            elif (date - last_rebalance).days >= self.REBALANCE_FREQ:
                should_rebalance = True

            if should_rebalance:
                scores = self.calculate_momentum(date)

                if len(scores) < self.NUM_HOLDINGS:
                    equity_curve.append({
                        'date': date,
                        'equity': total_equity
                    })
                    continue

                # Select top momentum stocks
                sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_stocks = [s[0] for s in sorted_stocks[:self.NUM_HOLDINGS]]

                # Calculate leverage based on portfolio volatility
                port_vol = self.calculate_portfolio_vol(positions, date) if positions else 0.20
                target_leverage = self.TARGET_VOL / port_vol if port_vol > 0 else 1.0
                target_leverage = max(self.MIN_LEVERAGE, min(self.MAX_LEVERAGE, target_leverage))
                current_leverage = target_leverage

                # Equal weight within holdings
                weight = target_leverage / self.NUM_HOLDINGS

                # Liquidate old positions
                for symbol, shares in positions.items():
                    if symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date]
                        value = shares * price
                        cash += value
                        cash -= abs(value) * self.COMMISSION_BPS / 10000

                positions = {}

                # Open new positions
                available = cash
                for symbol in top_stocks:
                    if symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date]
                        target_value = available * weight
                        shares = target_value / price
                        positions[symbol] = shares
                        cash -= target_value
                        cash -= target_value * self.COMMISSION_BPS / 10000

                last_rebalance = date

                if i % 100 == 0:
                    pos_val = sum(positions[s] * self.price_data[s].loc[date] for s in positions if s in self.price_data and date in self.price_data[s].index)
                    self.logger.info(f"  {date.date()}: leverage={target_leverage:.2f}, equity=${cash + pos_val:,.0f}")

            # Recalculate equity
            position_value = 0
            for symbol, shares in positions.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[date]
                    position_value += shares * price
            total_equity = cash + position_value

            equity_curve.append({
                'date': date,
                'equity': total_equity
            })

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
            'calmar': (returns.mean() * 252) / abs(((equity_df['equity'] / equity_df['equity'].expanding().max()) - 1).min()) if abs(((equity_df['equity'] / equity_df['equity'].expanding().max()) - 1).min()) > 0 else 0
        }

        return {
            'equity_curve': equity_df,
            'metrics': metrics
        }


def print_results(results):
    if not results:
        print("No results")
        return

    metrics = results['metrics']

    print("\n" + "="*70)
    print("VOL-MANAGED DIVIDEND MOMENTUM - BACKTEST RESULTS")
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
    print("VOL-MANAGED DIVIDEND MOMENTUM STRATEGY")
    print("(No Forward-Looking Bias)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols")

    strategy = VolManagedMomentum(loader)
    strategy.load_data(symbols)

    print("\nRunning backtest...")
    results = strategy.run()

    if results:
        print_results(results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
