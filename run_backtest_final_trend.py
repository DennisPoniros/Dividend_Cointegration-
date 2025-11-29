#!/usr/bin/env python3
"""
Final Trend-Following Strategy - Parameter Optimization

Testing different MA lengths and momentum lookbacks to maximize Sharpe.
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
        level=logging.WARNING,  # Reduce logging for parameter sweep
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class TrendStrategy:
    def __init__(self, loader, initial_capital=1_000_000,
                 trend_ma=200, momentum_lookback=63, rebalance_freq=21, num_holdings=15):
        self.loader = loader
        self.initial_capital = initial_capital

        self.TREND_MA = trend_ma
        self.MOMENTUM_LOOKBACK = momentum_lookback
        self.REBALANCE_FREQ = rebalance_freq
        self.NUM_HOLDINGS = num_holdings
        self.COMMISSION_BPS = 5

        self.price_data = {}

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > max(self.TREND_MA, self.MOMENTUM_LOOKBACK) + 100:
                self.price_data[symbol] = df['close']

    def is_above_trend(self, symbol, date):
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < self.TREND_MA + 5:
            return False
        current_price = prices.iloc[-1]
        ma = prices.iloc[-self.TREND_MA:].mean()
        return current_price > ma

    def calculate_momentum(self, symbol, date):
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < self.MOMENTUM_LOOKBACK + 10:
            return None
        end_price = prices.iloc[-1]
        start_price = prices.iloc[-self.MOMENTUM_LOOKBACK - 1]
        if start_price <= 0:
            return None
        return (end_price / start_price) - 1

    def run(self):
        all_dates = sorted(set().union(*[set(s.index) for s in self.price_data.values()]))
        start_idx = max(self.TREND_MA, self.MOMENTUM_LOOKBACK) + 50
        if start_idx >= len(all_dates):
            return None
        trading_dates = all_dates[start_idx:]

        cash = self.initial_capital
        positions = {}
        equity_curve = []
        last_rebalance = None

        for date in trading_dates:
            pos_val = sum(
                positions[s] * self.price_data[s].loc[date]
                for s in positions
                if s in self.price_data and date in self.price_data[s].index
            )
            total_equity = cash + pos_val

            should_rebalance = last_rebalance is None or (date - last_rebalance).days >= self.REBALANCE_FREQ

            if should_rebalance:
                candidates = []
                for symbol in self.price_data:
                    if date not in self.price_data[symbol].index:
                        continue
                    if not self.is_above_trend(symbol, date):
                        continue
                    mom = self.calculate_momentum(symbol, date)
                    if mom is not None and mom > 0:
                        candidates.append((symbol, mom))

                candidates.sort(key=lambda x: x[1], reverse=True)
                top = [c[0] for c in candidates[:self.NUM_HOLDINGS]]

                # Liquidate
                for s, shares in positions.items():
                    if s in self.price_data and date in self.price_data[s].index:
                        p = self.price_data[s].loc[date]
                        cash += shares * p
                        cash -= abs(shares * p) * self.COMMISSION_BPS / 10000

                positions = {}

                if top:
                    w = 1.0 / len(top)
                    available = cash
                    for sym in top:
                        if sym in self.price_data and date in self.price_data[sym].index:
                            p = self.price_data[sym].loc[date]
                            target = available * w * 0.98
                            positions[sym] = target / p
                            cash -= target
                            cash -= target * self.COMMISSION_BPS / 10000

                last_rebalance = date

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

        return {
            'total_return': (eq_df['equity'].iloc[-1] / self.initial_capital) - 1,
            'annual_return': rets.mean() * 252,
            'annual_vol': rets.std() * np.sqrt(252),
            'sharpe': excess.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0,
            'max_drawdown': ((eq_df['equity'] / eq_df['equity'].expanding().max()) - 1).min(),
        }


def main():
    logger = setup_logging()
    print("=" * 70)
    print("PARAMETER OPTIMIZATION - TREND-FOLLOWING STRATEGY")
    print("=" * 70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()
    print(f"Found {len(symbols)} symbols\n")

    # Parameter grid
    ma_values = [50, 100, 150, 200]
    momentum_values = [21, 42, 63, 126]
    holdings_values = [10, 15, 20, 25]
    rebalance_values = [5, 10, 21]

    best_sharpe = -999
    best_params = None
    best_result = None

    results = []

    # Simplified grid search - test most promising combinations
    test_params = [
        (50, 21, 15, 5),
        (50, 42, 15, 10),
        (50, 63, 20, 10),
        (100, 42, 15, 10),
        (100, 63, 15, 21),
        (100, 63, 20, 10),
        (150, 42, 15, 10),
        (150, 63, 20, 21),
        (200, 63, 15, 21),
        (200, 126, 20, 21),
    ]

    for ma, mom, hold, rebal in test_params:
        strategy = TrendStrategy(
            loader,
            trend_ma=ma,
            momentum_lookback=mom,
            rebalance_freq=rebal,
            num_holdings=hold
        )
        strategy.load_data(symbols)
        result = strategy.run()

        if result:
            print(f"MA={ma:3d}, Mom={mom:3d}, N={hold:2d}, Rebal={rebal:2d} -> "
                  f"Sharpe={result['sharpe']:.2f}, Return={result['annual_return']*100:.1f}%, "
                  f"Vol={result['annual_vol']*100:.1f}%, DD={result['max_drawdown']*100:.1f}%")

            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_params = (ma, mom, hold, rebal)
                best_result = result

    print("\n" + "=" * 70)
    print("BEST RESULT")
    print("=" * 70)
    if best_result:
        print(f"Parameters: MA={best_params[0]}, Momentum={best_params[1]}, "
              f"Holdings={best_params[2]}, Rebalance={best_params[3]}")
        print(f"  Total Return:    {best_result['total_return']*100:.2f}%")
        print(f"  Annual Return:   {best_result['annual_return']*100:.2f}%")
        print(f"  Annual Vol:      {best_result['annual_vol']*100:.2f}%")
        print(f"  Sharpe Ratio:    {best_result['sharpe']:.2f}")
        print(f"  Max Drawdown:    {best_result['max_drawdown']*100:.2f}%")
        print("=" * 70)
        if best_result['sharpe'] >= 1.0:
            print("SUCCESS: Sharpe >= 1.0!")
        else:
            print(f"Best Sharpe: {best_result['sharpe']:.2f}")


if __name__ == "__main__":
    main()
