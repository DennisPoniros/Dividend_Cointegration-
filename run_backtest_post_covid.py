#!/usr/bin/env python3
"""
Post-COVID Trend-Following Strategy

Key changes:
1. Start after COVID crash (June 2020)
2. Add market filter (SPY above 200-day MA)
3. Go to cash when market is bearish
4. More aggressive when market is bullish
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


class PostCovidStrategy:
    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Parameters
        self.TREND_MA = 150             # 150-day MA
        self.MOMENTUM_LOOKBACK = 42     # 2-month momentum
        self.REBALANCE_FREQ = 14        # Bi-weekly
        self.NUM_HOLDINGS = 12          # Concentrated
        self.COMMISSION_BPS = 5
        self.START_DATE = pd.Timestamp('2020-07-01')  # Post-COVID

        self.price_data = {}
        self.market_data = None  # Will use average of all stocks as market proxy

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > 300:
                self.price_data[symbol] = df['close']
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

        # Calculate market proxy (equal-weight average of all stocks)
        all_prices = pd.DataFrame(self.price_data)
        self.market_data = all_prices.mean(axis=1)

    def is_market_bullish(self, date):
        """Check if market is above its 200-day MA."""
        market = self.market_data[self.market_data.index <= date]
        if len(market) < 200:
            return True  # Default to bullish

        current = market.iloc[-1]
        ma = market.iloc[-200:].mean()
        return current > ma

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

        # Filter to start after COVID
        trading_dates = [d for d in all_dates if d >= self.START_DATE]
        if len(trading_dates) < 100:
            return None

        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

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
                # Check market regime
                market_bullish = self.is_market_bullish(date)

                if not market_bullish:
                    # Go to cash in bear market
                    for s, shares in positions.items():
                        if s in self.price_data and date in self.price_data[s].index:
                            p = self.price_data[s].loc[date]
                            cash += shares * p
                            cash -= abs(shares * p) * self.COMMISSION_BPS / 10000
                    positions = {}
                    last_rebalance = date
                else:
                    # Normal strategy in bull market
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

                if i % 100 == 0:
                    pv = sum(positions[s] * self.price_data[s].loc[date] for s in positions if s in self.price_data and date in self.price_data[s].index)
                    mode = "BULL" if market_bullish else "BEAR"
                    self.logger.info(f"  {date.date()}: {mode}, {len(positions)} stocks, ${cash + pv:,.0f}")

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
    print("POST-COVID STRATEGY RESULTS")
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
    print("POST-COVID TREND-FOLLOWING STRATEGY")
    print("=" * 70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()

    strategy = PostCovidStrategy(loader)
    strategy.load_data(symbols)

    results = strategy.run()
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
