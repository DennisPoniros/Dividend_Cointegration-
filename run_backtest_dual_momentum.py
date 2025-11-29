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

        # Dual momentum parameters
        self.LOOKBACK = 252         # 12-month lookback for absolute momentum
        self.SKIP_RECENT = 0        # Don't skip any days
        self.REBALANCE_FREQ = 14    # Bi-weekly (optimized for lower downside vol)
        self.NUM_HOLDINGS = 12      # Top 12 stocks (more diversification)
        self.COMMISSION_BPS = 5
        self.USE_RISK_PARITY = False  # Equal weight
        self.USE_QUALITY_FILTER = True  # Require momentum > 0.75 * volatility
        self.QUALITY_MULT = 0.75

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

    def calculate_downside_vol(self, symbol, date, lookback=63):
        """Calculate annualized downside volatility (std of negative returns)."""
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < lookback + 5:
            return None

        returns = prices.pct_change().dropna().iloc[-lookback:]
        neg_returns = returns[returns < 0]

        if len(neg_returns) < 5:
            return None

        return neg_returns.std() * np.sqrt(252)

    def calculate_volatility(self, symbol, date, lookback=63):
        """Calculate annualized volatility for quality filter."""
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < lookback + 5:
            return None

        returns = prices.pct_change().dropna().iloc[-lookback:]
        return returns.std() * np.sqrt(252)

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
        last_rebalance = None

        for i, date in enumerate(trading_dates):
            pos_val = sum(
                positions[s] * self.price_data[s].loc[date]
                for s in positions
                if s in self.price_data and date in self.price_data[s].index
            )
            total_equity = cash + pos_val

            should_rebalance = last_rebalance is None or (date - last_rebalance).days >= self.REBALANCE_FREQ

            if should_rebalance:
                # Calculate momentum and volatility for all stocks
                momentum_scores = {}
                vol_scores = {}
                for symbol in self.price_data:
                    if date not in self.price_data[symbol].index:
                        continue
                    mom = self.calculate_12m_momentum(symbol, date)
                    vol = self.calculate_volatility(symbol, date)
                    if mom is not None:
                        momentum_scores[symbol] = mom
                    if vol is not None:
                        vol_scores[symbol] = vol

                # Filter: Only stocks with POSITIVE 12-month momentum (absolute momentum filter)
                # Plus quality filter: momentum > 0.75 * volatility
                if self.USE_QUALITY_FILTER:
                    positive_mom = {s: m for s, m in momentum_scores.items()
                                   if m > 0 and m > self.QUALITY_MULT * vol_scores.get(s, 999)}
                else:
                    positive_mom = {s: m for s, m in momentum_scores.items() if m > 0}

                # Select top N by relative momentum
                sorted_stocks = sorted(positive_mom.items(), key=lambda x: x[1], reverse=True)
                top = [s[0] for s in sorted_stocks[:self.NUM_HOLDINGS]]

                # Liquidate all
                for s, shares in positions.items():
                    if s in self.price_data and date in self.price_data[s].index:
                        p = self.price_data[s].loc[date]
                        cash += shares * p
                        cash -= abs(shares * p) * self.COMMISSION_BPS / 10000

                positions = {}

                # Only invest if we have qualifying stocks
                if top:
                    # Calculate weights
                    if self.USE_RISK_PARITY:
                        # Risk parity: inverse downside volatility weighting
                        inv_vols = {}
                        for sym in top:
                            dnvol = self.calculate_downside_vol(sym, date)
                            if dnvol and dnvol > 0:
                                inv_vols[sym] = 1.0 / dnvol
                            else:
                                inv_vols[sym] = 1.0  # fallback to equal weight
                        total_inv_vol = sum(inv_vols.values())
                        weights = {s: inv_vols[s] / total_inv_vol for s in top}
                    else:
                        # Equal weight
                        weights = {s: 1.0 / len(top) for s in top}

                    available = cash
                    for sym in top:
                        if sym in self.price_data and date in self.price_data[sym].index:
                            p = self.price_data[sym].loc[date]
                            target = available * weights[sym] * 0.98
                            positions[sym] = target / p
                            cash -= target
                            cash -= target * self.COMMISSION_BPS / 10000

                last_rebalance = date

                if i % 100 == 0:
                    invested = len(positions) > 0
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
