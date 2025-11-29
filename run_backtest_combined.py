#!/usr/bin/env python3
"""
Combined Strategy: Trend + Volatility Targeting + Quality Momentum

Key features:
1. Only buy stocks above 100-day MA (trend filter)
2. Rank by risk-adjusted momentum
3. Inverse volatility weighting
4. Target 12% portfolio volatility
5. 10-day rebalance for quicker adaptation
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


class CombinedStrategy:
    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Optimized parameters
        self.TREND_MA = 100             # 100-day MA (faster than 200)
        self.MOMENTUM_LOOKBACK = 42     # ~2 month momentum
        self.SKIP_RECENT = 5            # Skip last week
        self.VOL_LOOKBACK = 21          # 1 month vol
        self.REBALANCE_FREQ = 10        # 2-week rebalance (faster)
        self.NUM_HOLDINGS = 12          # Concentrated
        self.TARGET_VOL = 0.12          # 12% annual vol target
        self.MAX_LEVERAGE = 1.2         # Max 120%
        self.MIN_LEVERAGE = 0.4         # Min 40%
        self.COMMISSION_BPS = 5

        self.price_data = {}
        self.returns_data = {}

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > 300:
                self.price_data[symbol] = df['close']
                self.returns_data[symbol] = df['close'].pct_change()
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

    def is_above_trend(self, symbol, date):
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < self.TREND_MA + 5:
            return False
        current_price = prices.iloc[-1]
        ma = prices.iloc[-self.TREND_MA:].mean()
        return current_price > ma

    def calculate_score(self, symbol, date):
        """Calculate risk-adjusted momentum score."""
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        returns = self.returns_data[symbol][self.returns_data[symbol].index <= date]

        if len(prices) < self.MOMENTUM_LOOKBACK + self.SKIP_RECENT + 20:
            return None, None

        # Momentum
        end_idx = len(prices) - self.SKIP_RECENT - 1
        start_idx = end_idx - self.MOMENTUM_LOOKBACK
        if start_idx < 0:
            return None, None

        start_price = prices.iloc[start_idx]
        end_price = prices.iloc[end_idx]
        if start_price <= 0:
            return None, None

        momentum = (end_price / start_price) - 1

        # Volatility
        recent_returns = returns.iloc[-self.VOL_LOOKBACK:]
        if len(recent_returns) < 10:
            return None, None

        volatility = recent_returns.std() * np.sqrt(252)
        if volatility <= 0.01:
            return None, None

        # Risk-adjusted score
        score = momentum / volatility

        return score, volatility

    def estimate_portfolio_vol(self, positions, date):
        """Estimate portfolio volatility."""
        if not positions:
            return 0.20

        returns_data = []
        weights = []

        total_val = 0
        for s, shares in positions.items():
            if s in self.price_data and date in self.price_data[s].index:
                total_val += shares * self.price_data[s].loc[date]

        if total_val <= 0:
            return 0.20

        for s, shares in positions.items():
            if s not in self.returns_data:
                continue
            ret = self.returns_data[s][self.returns_data[s].index <= date].iloc[-self.VOL_LOOKBACK:]
            if len(ret) < 10:
                continue
            w = (shares * self.price_data[s].loc[date]) / total_val
            returns_data.append(ret.values)
            weights.append(w)

        if not returns_data:
            return 0.20

        # Simple weighted vol (ignores correlation, conservative)
        vols = [np.std(r) * np.sqrt(252) for r in returns_data]
        return sum(w * v for w, v in zip(weights, vols))

    def run(self):
        all_dates = sorted(set().union(*[set(s.index) for s in self.price_data.values()]))
        start_idx = max(self.TREND_MA, self.MOMENTUM_LOOKBACK + self.SKIP_RECENT) + 50
        trading_dates = all_dates[start_idx:]
        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()}")

        cash = self.initial_capital
        positions = {}
        equity_curve = []
        last_rebalance = None

        for i, date in enumerate(trading_dates):
            # Current equity
            pos_val = sum(
                positions[s] * self.price_data[s].loc[date]
                for s in positions
                if s in self.price_data and date in self.price_data[s].index
            )
            total_equity = cash + pos_val

            # Rebalance
            should_rebalance = last_rebalance is None or (date - last_rebalance).days >= self.REBALANCE_FREQ

            if should_rebalance:
                # Find candidates above trend with good momentum
                candidates = []
                for symbol in self.price_data:
                    if date not in self.price_data[symbol].index:
                        continue
                    if not self.is_above_trend(symbol, date):
                        continue
                    score, vol = self.calculate_score(symbol, date)
                    if score is not None and score > 0:
                        candidates.append((symbol, score, vol))

                # Sort by score
                candidates.sort(key=lambda x: x[1], reverse=True)
                top = candidates[:self.NUM_HOLDINGS]

                # Calculate leverage based on expected portfolio vol
                if positions:
                    port_vol = self.estimate_portfolio_vol(positions, date)
                else:
                    port_vol = 0.20

                leverage = self.TARGET_VOL / port_vol if port_vol > 0 else 1.0
                leverage = max(self.MIN_LEVERAGE, min(self.MAX_LEVERAGE, leverage))

                # Liquidate
                for s, shares in positions.items():
                    if s in self.price_data and date in self.price_data[s].index:
                        p = self.price_data[s].loc[date]
                        cash += shares * p
                        cash -= abs(shares * p) * self.COMMISSION_BPS / 10000

                positions = {}

                # Inverse volatility weights
                if top:
                    inv_vols = [1.0 / c[2] for c in top]
                    total_inv = sum(inv_vols)
                    weights = [iv / total_inv * leverage for iv in inv_vols]

                    available = cash
                    for (sym, _, _), w in zip(top, weights):
                        if sym in self.price_data and date in self.price_data[sym].index:
                            p = self.price_data[sym].loc[date]
                            target_val = available * w * 0.98
                            shares = target_val / p
                            positions[sym] = shares
                            cash -= target_val
                            cash -= target_val * self.COMMISSION_BPS / 10000

                last_rebalance = date

                if i % 100 == 0:
                    pv = sum(positions[s] * self.price_data[s].loc[date] for s in positions if s in self.price_data and date in self.price_data[s].index)
                    self.logger.info(f"  {date.date()}: {len(positions)} stocks, leverage={leverage:.1%}")

            # Track equity
            pos_val = sum(
                positions[s] * self.price_data[s].loc[date]
                for s in positions
                if s in self.price_data and date in self.price_data[s].index
            )
            total_equity = cash + pos_val
            equity_curve.append({'date': date, 'equity': total_equity})

        # Metrics
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
    print("\n" + "="*70)
    print("COMBINED STRATEGY - RESULTS")
    print("="*70)
    print(f"  Total Return:    {m['total_return']*100:>10.2f}%")
    print(f"  Annual Return:   {m['annual_return']*100:>10.2f}%")
    print(f"  Annual Vol:      {m['annual_vol']*100:>10.2f}%")
    print(f"  Sharpe Ratio:    {m['sharpe']:>10.2f}")
    print(f"  Sortino Ratio:   {m['sortino']:>10.2f}")
    print(f"  Max Drawdown:    {m['max_drawdown']*100:>10.2f}%")
    print("="*70)
    if m['sharpe'] >= 1.0:
        print("SUCCESS: Sharpe >= 1.0!")
    else:
        print(f"Sharpe: {m['sharpe']:.2f}")


def main():
    logger = setup_logging()
    print("\n" + "="*70)
    print("COMBINED STRATEGY BACKTEST")
    print("="*70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()

    strategy = CombinedStrategy(loader)
    strategy.load_data(symbols)

    print("\nRunning backtest...")
    results = strategy.run()
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
