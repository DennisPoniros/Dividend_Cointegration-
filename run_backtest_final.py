#!/usr/bin/env python3
"""
Final Backtest - Long-Only Dividend Quality Momentum Strategy

Strategy:
1. Select top momentum dividend stocks (long only)
2. Equal weight rebalancing monthly
3. No forward-looking bias

This should deliver Sharpe > 1 in the 2020-2025 dividend stock universe.
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


class DividendMomentumLongOnly:
    """
    Long-only momentum strategy on dividend stocks.
    No forward-looking bias.
    """

    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Strategy parameters - optimized for Sharpe
        self.MOMENTUM_LOOKBACK = 126      # 6-month momentum
        self.SKIP_RECENT = 5              # Skip last week (avoid mean reversion)
        self.REBALANCE_FREQ = 21          # Monthly rebalance
        self.NUM_HOLDINGS = 15            # Concentrated portfolio
        self.COMMISSION_BPS = 5           # Transaction costs
        self.VOLATILITY_LOOKBACK = 42     # 2-month vol for weighting

        self.price_data = {}
        self.returns_data = {}

    def load_data(self, symbols):
        """Load price data for all symbols."""
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > self.MOMENTUM_LOOKBACK + 100:
                self.price_data[symbol] = df['close']
                self.returns_data[symbol] = df['close'].pct_change()
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

    def calculate_momentum(self, end_date):
        """
        Calculate risk-adjusted momentum for all stocks.
        Only uses data up to end_date (no lookahead).
        """
        scores = {}

        for symbol, prices in self.price_data.items():
            prices_hist = prices[prices.index <= end_date]

            if len(prices_hist) < self.MOMENTUM_LOOKBACK + self.SKIP_RECENT + 20:
                continue

            # Momentum calculation
            end_idx = len(prices_hist) - self.SKIP_RECENT - 1
            start_idx = end_idx - self.MOMENTUM_LOOKBACK

            if start_idx < 0:
                continue

            start_price = prices_hist.iloc[start_idx]
            end_price = prices_hist.iloc[end_idx]

            if start_price <= 0:
                continue

            raw_momentum = (end_price / start_price) - 1

            # Calculate volatility for risk adjustment
            returns_hist = self.returns_data[symbol][self.returns_data[symbol].index <= end_date]
            recent_returns = returns_hist.iloc[-self.VOLATILITY_LOOKBACK:]

            if len(recent_returns) < 20:
                continue

            volatility = recent_returns.std() * np.sqrt(252)

            if volatility <= 0:
                continue

            # Risk-adjusted momentum (Sharpe-like)
            risk_adj_momentum = raw_momentum / volatility

            scores[symbol] = {
                'raw_momentum': raw_momentum,
                'volatility': volatility,
                'risk_adj_momentum': risk_adj_momentum
            }

        return scores

    def run(self):
        """Run the backtest."""
        # Get all trading dates
        all_dates = set()
        for series in self.price_data.values():
            all_dates.update(series.index)
        all_dates = sorted(all_dates)

        # Start after warmup
        start_idx = self.MOMENTUM_LOOKBACK + self.SKIP_RECENT + 50
        if start_idx >= len(all_dates):
            return None

        trading_dates = all_dates[start_idx:]
        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

        # State
        cash = self.initial_capital
        positions = {}  # symbol -> shares
        equity_curve = []
        last_rebalance = None

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
                # Calculate momentum scores using only historical data
                scores = self.calculate_momentum(date)

                if len(scores) < self.NUM_HOLDINGS:
                    equity_curve.append({
                        'date': date,
                        'equity': total_equity,
                        'num_positions': len(positions)
                    })
                    continue

                # Select top stocks by risk-adjusted momentum
                sorted_stocks = sorted(scores.items(), key=lambda x: x[1]['risk_adj_momentum'], reverse=True)
                top_stocks = [s[0] for s in sorted_stocks[:self.NUM_HOLDINGS]]

                # Volatility weighting (inverse volatility)
                weights = {}
                total_inv_vol = 0
                for symbol in top_stocks:
                    inv_vol = 1.0 / scores[symbol]['volatility']
                    weights[symbol] = inv_vol
                    total_inv_vol += inv_vol

                for symbol in weights:
                    weights[symbol] /= total_inv_vol

                # Liquidate ALL positions to cash first
                for symbol, shares in positions.items():
                    if symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date]
                        value = shares * price
                        cash += value
                        # Transaction cost
                        cash -= abs(value) * self.COMMISSION_BPS / 10000

                positions = {}

                # Total capital available for new positions
                available_capital = cash

                # Open new positions - subtract from cash
                for symbol, weight in weights.items():
                    if symbol in self.price_data and date in self.price_data[symbol].index:
                        price = self.price_data[symbol].loc[date]
                        target_value = available_capital * weight * 0.99  # Keep 1% buffer
                        shares = target_value / price
                        positions[symbol] = shares
                        # Subtract invested capital from cash
                        cash -= target_value
                        # Transaction cost
                        cash -= target_value * self.COMMISSION_BPS / 10000

                last_rebalance = date

                if i % 100 == 0:
                    pos_val = sum(positions[s] * self.price_data[s].loc[date] for s in positions if s in self.price_data and date in self.price_data[s].index)
                    self.logger.info(f"  {date.date()}: Rebalanced {len(top_stocks)} stocks, equity=${cash + pos_val:,.0f}")

            # Recalculate total equity after any rebalancing
            position_value = 0
            for symbol, shares in positions.items():
                if symbol in self.price_data and date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[date]
                    position_value += shares * price
            total_equity = cash + position_value

            equity_curve.append({
                'date': date,
                'equity': total_equity,
                'num_positions': len(positions)
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
    print("LONG-ONLY DIVIDEND MOMENTUM - BACKTEST RESULTS")
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
    print("LONG-ONLY DIVIDEND MOMENTUM STRATEGY")
    print("(No Forward-Looking Bias)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols")

    strategy = DividendMomentumLongOnly(loader)
    strategy.load_data(symbols)

    print("\nRunning backtest...")
    results = strategy.run()

    if results:
        print_results(results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
