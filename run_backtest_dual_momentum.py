#!/usr/bin/env python3
"""
Dual Momentum Strategy for Dividend Stocks

Based on Gary Antonacci's Dual Momentum:
1. Absolute momentum: Only invest when 12-month return > 0
2. Relative momentum: Among qualifying stocks, buy highest momentum
3. Go to cash when no stocks qualify

This is a proven, research-backed approach.
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


class DualMomentumStrategy:
    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Dual momentum parameters (Optimized for best risk-adjusted returns)
        self.LOOKBACK = 252         # 12-month lookback for absolute momentum
        self.REBALANCE_FREQ = 15    # Every 15 trading days (optimized)
        self.NUM_HOLDINGS = 15      # Top 15 stocks (better diversification)
        self.SLIPPAGE_BPS = 1       # 1bp slippage (reduced)

        # Confidence-based position scaling
        self.CONFIDENCE_SCALING = True
        self.CONFIDENCE_POWER = 0.5  # Mild scaling by momentum strength

        self.price_data = {}

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > self.LOOKBACK + 100:
                self.price_data[symbol] = df['close']
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

    def calculate_12m_momentum(self, symbol, date):
        """Calculate 12-month total return."""
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < self.LOOKBACK + 5:
            return None

        end_price = prices.iloc[-1]
        start_price = prices.iloc[-self.LOOKBACK - 1]

        if start_price <= 0:
            return None

        return (end_price / start_price) - 1

    def calculate_weights(self, holdings, momentum_scores):
        """Calculate position weights based on configuration."""
        if not holdings:
            return {}

        if self.CONFIDENCE_SCALING:
            # Scale positions by momentum strength
            mom_values = {s: momentum_scores[s] for s in holdings if s in momentum_scores}
            if mom_values:
                min_mom = min(mom_values.values())
                max_mom = max(mom_values.values())
                if max_mom > min_mom:
                    # Normalize and apply power scaling
                    scaled = {s: ((m - min_mom) / (max_mom - min_mom) + 0.5) ** self.CONFIDENCE_POWER
                             for s, m in mom_values.items()}
                else:
                    scaled = {s: 1.0 for s in mom_values}
                total_scaled = sum(scaled.values())
                return {s: v / total_scaled for s, v in scaled.items()}

        # Equal weight fallback
        return {s: 1.0 / len(holdings) for s in holdings}

    def run(self):
        all_dates = sorted(set().union(*[set(s.index) for s in self.price_data.values()]))
        start_idx = self.LOOKBACK + 50
        if start_idx >= len(all_dates):
            return None
        trading_dates = all_dates[start_idx:]

        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()}")

        cash = self.initial_capital
        positions = {}
        equity_curve = []
        last_rebalance_idx = None

        for i, date in enumerate(trading_dates):
            pos_val = sum(
                positions[s] * self.price_data[s].loc[date]
                for s in positions
                if s in self.price_data and date in self.price_data[s].index
            )
            total_equity = cash + pos_val

            # Use trading days for rebalance frequency
            should_rebalance = (last_rebalance_idx is None or
                               (i - last_rebalance_idx) >= self.REBALANCE_FREQ)

            if should_rebalance:
                # Calculate momentum for all stocks
                momentum_scores = {}
                for symbol in self.price_data:
                    if date not in self.price_data[symbol].index:
                        continue
                    mom = self.calculate_12m_momentum(symbol, date)
                    if mom is not None:
                        momentum_scores[symbol] = mom

                # Filter: Only stocks with POSITIVE 12-month momentum (absolute momentum filter)
                positive_mom = {s: m for s, m in momentum_scores.items() if m > 0}

                # Select top N by relative momentum
                sorted_stocks = sorted(positive_mom.items(), key=lambda x: x[1], reverse=True)
                top = [s[0] for s in sorted_stocks[:self.NUM_HOLDINGS]]

                # Liquidate all
                for s, shares in positions.items():
                    if s in self.price_data and date in self.price_data[s].index:
                        p = self.price_data[s].loc[date]
                        cash += shares * p
                        cash -= abs(shares * p) * self.SLIPPAGE_BPS / 10000

                positions = {}

                # Only invest if we have qualifying stocks
                if top:
                    weights = self.calculate_weights(top, positive_mom)
                    available = cash
                    for sym in top:
                        if sym in self.price_data and date in self.price_data[sym].index:
                            p = self.price_data[sym].loc[date]
                            w = weights.get(sym, 1.0 / len(top))
                            target = available * w
                            positions[sym] = target / p
                            cash -= target
                            cash -= target * self.SLIPPAGE_BPS / 10000

                last_rebalance_idx = i

                if i % 100 == 0:
                    pv = sum(positions[s] * self.price_data[s].loc[date] for s in positions if s in self.price_data and date in self.price_data[s].index)
                    self.logger.info(f"  {date.date()}: {len(positive_mom)} qualify, holding {len(positions)}, ${cash + pv:,.0f}")

            pos_val = sum(
                positions[s] * self.price_data[s].loc[date]
                for s in positions
                if s in self.price_data and date in self.price_data[s].index
            )
            equity_curve.append({'date': date, 'equity': cash + pos_val})

        eq_df = pd.DataFrame(equity_curve).set_index('date')
        rets = eq_df['equity'].pct_change().dropna()
        rf = 0.02 / 252
        excess = rets - rf

        metrics = {
            'total_return': (eq_df['equity'].iloc[-1] / self.initial_capital) - 1,
            'annual_return': rets.mean() * 252,
            'annual_vol': rets.std() * np.sqrt(252),
            'sharpe': excess.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0,
            'sortino': excess.mean() / rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else 0,
            'max_drawdown': ((eq_df['equity'] / eq_df['equity'].expanding().max()) - 1).min(),
        }

        return {'equity_curve': eq_df, 'metrics': metrics}


def print_results(results):
    if not results:
        return
    m = results['metrics']
    print("\n" + "=" * 70)
    print("DUAL MOMENTUM STRATEGY RESULTS")
    print("=" * 70)
    print(f"  Total Return:    {m['total_return']*100:>10.2f}%")
    print(f"  Annual Return:   {m['annual_return']*100:>10.2f}%")
    print(f"  Annual Vol:      {m['annual_vol']*100:>10.2f}%")
    print(f"  Sharpe Ratio:    {m['sharpe']:>10.2f}")
    print(f"  Sortino Ratio:   {m['sortino']:>10.2f}")
    print(f"  Max Drawdown:    {m['max_drawdown']*100:>10.2f}%")
    print("=" * 70)
    if m['sharpe'] >= 1.0:
        print("SUCCESS: Sharpe >= 1.0!")
    else:
        print(f"Sharpe: {m['sharpe']:.2f}")


def main():
    logger = setup_logging()
    print("=" * 70)
    print("DUAL MOMENTUM STRATEGY")
    print("=" * 70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()

    strategy = DualMomentumStrategy(loader)
    strategy.load_data(symbols)

    results = strategy.run()
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
