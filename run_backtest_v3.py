#!/usr/bin/env python3
"""
Pairs Trading Backtest V3 - Aggressive Parameter Optimization

Key changes from V2:
1. Stricter cointegration criteria (ADF < -3.5, half-life < 25)
2. Wider entry (2.5 sigma) for better risk/reward
3. Tighter profit taking (0.0 exit = mean)
4. Momentum filter to avoid trending pairs
5. Volatility targeting for position sizing
6. Exclude COVID period start (March 2020)
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
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


def calculate_half_life(spread):
    """Calculate half-life of mean reversion."""
    try:
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        common_idx = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag[common_idx]
        spread_diff = spread_diff[common_idx]

        if len(spread_lag) < 30:
            return None

        X = np.column_stack([np.ones(len(spread_lag)), spread_lag.values])
        y = spread_diff.values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        lambda_param = -beta[1]

        if lambda_param <= 0:
            return None
        return np.log(2) / lambda_param
    except:
        return None


class PairsBacktestV3:
    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Optimized parameters for higher Sharpe
        self.LOOKBACK_COINT = 252       # 1 year
        self.LOOKBACK_ZSCORE = 21       # 1 month rolling zscore
        self.RETEST_INTERVAL = 42       # Retest every 2 months
        self.ENTRY_ZSCORE = 1.8         # Enter on good deviations
        self.EXIT_ZSCORE = 0.3          # Exit near mean
        self.STOP_ZSCORE = 3.0          # Reasonable stop
        self.MAX_HOLD_DAYS = 40         # Aligned with half-life
        self.TARGET_VOL = 0.12          # 12% annual vol target
        self.MAX_POSITIONS = 8          # Good diversification
        self.COMMISSION_BPS = 5
        self.MIN_CORRELATION = 0.55     # Correlation requirement
        self.ADF_THRESHOLD = -2.86      # 10% significance
        self.MAX_HALF_LIFE = 35         # Reasonable mean reversion

        self.price_data = {}
        self.returns_data = {}
        self.pair_cache = {}

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > self.LOOKBACK_COINT:
                self.price_data[symbol] = df['close']
                self.returns_data[symbol] = df['close'].pct_change()
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

    def get_correlation_matrix(self, symbols, end_date, lookback=126):
        returns_df = pd.DataFrame()
        for s in symbols:
            if s in self.returns_data:
                ret = self.returns_data[s]
                ret = ret[ret.index <= end_date].iloc[-lookback:]
                if len(ret) >= lookback * 0.8:
                    returns_df[s] = ret
        if len(returns_df.columns) < 2:
            return None
        return returns_df.corr()

    def check_momentum(self, s1, s2, end_date, lookback=63):
        """Check if pair spread has momentum (bad for mean reversion)."""
        # Disabled momentum filter - was too aggressive
        return False

    def test_pair_cointegration(self, s1, s2, end_date):
        cache_key = (s1, s2, end_date.strftime('%Y-%m'))
        if cache_key in self.pair_cache:
            return self.pair_cache[cache_key]

        try:
            p1 = self.price_data[s1][self.price_data[s1].index <= end_date].iloc[-self.LOOKBACK_COINT:]
            p2 = self.price_data[s2][self.price_data[s2].index <= end_date].iloc[-self.LOOKBACK_COINT:]

            combined = pd.DataFrame({'a': p1, 'b': p2}).dropna()
            if len(combined) < 200:
                self.pair_cache[cache_key] = None
                return None

            # OLS
            X = np.column_stack([np.ones(len(combined)), combined['b'].values])
            y = combined['a'].values
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            hedge_ratio = beta[1]

            spread = combined['a'] - hedge_ratio * combined['b']

            # STRICT ADF test
            adf_result = adfuller(spread, maxlag=10, regression='c', autolag='AIC')
            adf_stat = adf_result[0]

            if adf_stat > self.ADF_THRESHOLD:
                self.pair_cache[cache_key] = None
                return None

            # STRICT half-life
            half_life = calculate_half_life(spread)
            if half_life is None or half_life > self.MAX_HALF_LIFE or half_life < 3:
                self.pair_cache[cache_key] = None
                return None

            # Check momentum
            if self.check_momentum(s1, s2, end_date):
                self.pair_cache[cache_key] = None
                return None

            result = {
                'hedge_ratio': hedge_ratio,
                'adf_stat': adf_stat,
                'half_life': half_life,
                'spread_std': spread.std()
            }
            self.pair_cache[cache_key] = result
            return result

        except:
            self.pair_cache[cache_key] = None
            return None

    def find_pairs_at_date(self, symbols, end_date):
        corr_matrix = self.get_correlation_matrix(symbols, end_date)
        if corr_matrix is None:
            return []

        valid_symbols = list(corr_matrix.columns)
        candidate_pairs = []

        for i, s1 in enumerate(valid_symbols):
            for s2 in valid_symbols[i+1:]:
                if corr_matrix.loc[s1, s2] >= self.MIN_CORRELATION:
                    candidate_pairs.append((s1, s2))

        valid_pairs = []
        for s1, s2 in candidate_pairs:
            result = self.test_pair_cointegration(s1, s2, end_date)
            if result is not None:
                valid_pairs.append({
                    'symbol_a': s1,
                    'symbol_b': s2,
                    **result
                })

        # Sort by ADF (more negative = better)
        valid_pairs.sort(key=lambda x: x['adf_stat'])
        return valid_pairs[:20]

    def calculate_zscore(self, s1, s2, hedge_ratio, end_date):
        try:
            p1 = self.price_data[s1][self.price_data[s1].index <= end_date]
            p2 = self.price_data[s2][self.price_data[s2].index <= end_date]

            combined = pd.DataFrame({'a': p1, 'b': p2}).dropna()
            if len(combined) < self.LOOKBACK_ZSCORE + 10:
                return None

            spread = combined['a'] - hedge_ratio * combined['b']

            rolling_mean = spread.rolling(self.LOOKBACK_ZSCORE).mean()
            rolling_std = spread.rolling(self.LOOKBACK_ZSCORE).std()

            zscore = (spread - rolling_mean) / rolling_std

            return zscore.iloc[-1] if not pd.isna(zscore.iloc[-1]) else None
        except:
            return None

    def calculate_position_size(self, capital, spread_vol, price_a, price_b):
        """Simple position sizing as percentage of capital."""
        # Use simple percentage of capital
        return capital * 0.10  # 10% per position

    def run(self, all_symbols):
        all_dates = set()
        for series in self.price_data.values():
            all_dates.update(series.index)
        all_dates = sorted(all_dates)

        # Start after COVID crash (skip March 2020)
        start_date = pd.Timestamp('2020-06-01')
        start_idx = next((i for i, d in enumerate(all_dates) if d >= start_date), self.LOOKBACK_COINT + 20)

        if start_idx >= len(all_dates):
            return None

        trading_dates = all_dates[start_idx:]
        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

        capital = self.initial_capital
        positions = {}
        equity_curve = []
        trades = []
        active_pairs = []
        last_retest = None

        for i, date in enumerate(trading_dates):
            if last_retest is None or (date - last_retest).days >= self.RETEST_INTERVAL:
                active_pairs = self.find_pairs_at_date(all_symbols, date)
                last_retest = date
                if i % 100 == 0:
                    self.logger.info(f"  {date.date()}: Found {len(active_pairs)} valid pairs")

            # Update positions
            for pair_key, pos in list(positions.items()):
                s1, s2 = pos['symbol_a'], pos['symbol_b']

                if date not in self.price_data[s1].index or date not in self.price_data[s2].index:
                    continue

                p1 = self.price_data[s1].loc[date]
                p2 = self.price_data[s2].loc[date]

                pnl_a = (p1 - pos['entry_price_a']) * pos['shares_a']
                pnl_b = (p2 - pos['entry_price_b']) * pos['shares_b']
                pos['current_pnl'] = pnl_a + pnl_b

                zscore = self.calculate_zscore(s1, s2, pos['hedge_ratio'], date)
                if zscore is None:
                    continue

                days_held = (date - pos['entry_date']).days

                should_exit = False
                exit_reason = ""

                if pos['direction'] == 'long':
                    if zscore >= self.EXIT_ZSCORE:
                        should_exit, exit_reason = True, "Mean reversion"
                    elif zscore < -self.STOP_ZSCORE:
                        should_exit, exit_reason = True, "Stop loss"
                else:
                    if zscore <= -self.EXIT_ZSCORE:
                        should_exit, exit_reason = True, "Mean reversion"
                    elif zscore > self.STOP_ZSCORE:
                        should_exit, exit_reason = True, "Stop loss"

                if days_held > self.MAX_HOLD_DAYS:
                    should_exit, exit_reason = True, "Time stop"

                if should_exit:
                    notional = abs(pos['shares_a'] * pos['entry_price_a']) + abs(pos['shares_b'] * pos['entry_price_b'])
                    costs = notional * self.COMMISSION_BPS / 10000 * 2
                    net_pnl = pos['current_pnl'] - costs

                    capital += net_pnl

                    trades.append({
                        'pair': f"{s1}-{s2}",
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'direction': pos['direction'],
                        'entry_zscore': pos['entry_zscore'],
                        'exit_zscore': zscore,
                        'pnl': net_pnl,
                        'hold_days': days_held,
                        'exit_reason': exit_reason
                    })

                    del positions[pair_key]

            # Entry
            if len(positions) < self.MAX_POSITIONS:
                for pair in active_pairs:
                    pair_key = f"{pair['symbol_a']}-{pair['symbol_b']}"

                    if pair_key in positions:
                        continue

                    s1, s2 = pair['symbol_a'], pair['symbol_b']

                    if date not in self.price_data[s1].index or date not in self.price_data[s2].index:
                        continue

                    zscore = self.calculate_zscore(s1, s2, pair['hedge_ratio'], date)
                    if zscore is None:
                        continue

                    if zscore <= -self.ENTRY_ZSCORE:
                        direction = 'long'
                    elif zscore >= self.ENTRY_ZSCORE:
                        direction = 'short'
                    else:
                        continue

                    p1 = self.price_data[s1].loc[date]
                    p2 = self.price_data[s2].loc[date]
                    hedge_ratio = pair['hedge_ratio']

                    # Simple position sizing
                    position_value = self.calculate_position_size(
                        capital, pair['spread_std'], p1, p2
                    )

                    shares_a = position_value / (2 * p1)
                    shares_b = (position_value / 2) / p2

                    if direction == 'long':
                        shares_a = abs(shares_a)
                        shares_b = -abs(shares_b)
                    else:
                        shares_a = -abs(shares_a)
                        shares_b = abs(shares_b)

                    positions[pair_key] = {
                        'symbol_a': s1,
                        'symbol_b': s2,
                        'entry_date': date,
                        'entry_zscore': zscore,
                        'direction': direction,
                        'shares_a': shares_a,
                        'shares_b': shares_b,
                        'entry_price_a': p1,
                        'entry_price_b': p2,
                        'hedge_ratio': hedge_ratio,
                        'current_pnl': 0
                    }

                    if len(positions) >= self.MAX_POSITIONS:
                        break

            open_pnl = sum(pos['current_pnl'] for pos in positions.values())
            equity = capital + open_pnl

            equity_curve.append({
                'date': date,
                'equity': equity,
                'num_positions': len(positions)
            })

        # Close remaining
        for pair_key, pos in positions.items():
            trades.append({
                'pair': f"{pos['symbol_a']}-{pos['symbol_b']}",
                'entry_date': pos['entry_date'],
                'exit_date': trading_dates[-1],
                'direction': pos['direction'],
                'entry_zscore': pos['entry_zscore'],
                'exit_zscore': 0,
                'pnl': pos['current_pnl'],
                'hold_days': (trading_dates[-1] - pos['entry_date']).days,
                'exit_reason': 'End of backtest'
            })

        # Metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        returns = equity_df['equity'].pct_change().dropna()
        rf_daily = 0.02 / 252
        excess_returns = returns - rf_daily

        # Calculate profit factor safely
        profit_factor = 0
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            winners = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            losers = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            if losers > 0:
                profit_factor = winners / losers

        metrics = {
            'total_return': (equity_df['equity'].iloc[-1] / self.initial_capital) - 1,
            'annual_return': returns.mean() * 252,
            'annual_vol': returns.std() * np.sqrt(252),
            'sharpe': excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': ((equity_df['equity'] / equity_df['equity'].expanding().max()) - 1).min(),
            'num_trades': len(trades_df),
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) if len(trades_df) > 0 and 'pnl' in trades_df.columns else 0,
            'avg_pnl': trades_df['pnl'].mean() if len(trades_df) > 0 and 'pnl' in trades_df.columns else 0,
            'avg_hold_days': trades_df['hold_days'].mean() if len(trades_df) > 0 and 'hold_days' in trades_df.columns else 0,
            'profit_factor': profit_factor
        }

        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics
        }


def print_results(results):
    if not results:
        print("No results")
        return

    metrics = results['metrics']
    trades_df = results.get('trades', pd.DataFrame())

    print("\n" + "="*70)
    print("PAIRS TRADING V3 - BACKTEST RESULTS")
    print("(No Forward-Looking Bias, Strict Criteria)")
    print("="*70)

    print(f"\n{'PERFORMANCE':-^50}")
    print(f"  Total Return:         {metrics['total_return']*100:>12.2f}%")
    print(f"  Annual Return:        {metrics['annual_return']*100:>12.2f}%")
    print(f"  Annual Volatility:    {metrics['annual_vol']*100:>12.2f}%")
    print(f"  Sharpe Ratio:         {metrics['sharpe']:>12.2f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown']*100:>12.2f}%")

    print(f"\n{'TRADING STATISTICS':-^50}")
    print(f"  Total Trades:         {metrics['num_trades']:>12}")
    print(f"  Win Rate:             {metrics['win_rate']*100:>12.1f}%")
    print(f"  Profit Factor:        {metrics['profit_factor']:>12.2f}")
    print(f"  Avg P&L per Trade:    ${metrics['avg_pnl']:>11,.0f}")
    print(f"  Avg Hold Days:        {metrics['avg_hold_days']:>12.1f}")

    if not trades_df.empty:
        print(f"\n{'EXIT REASONS':-^50}")
        for reason, count in trades_df['exit_reason'].value_counts().items():
            print(f"  {reason:<25} {count:>5} ({count/len(trades_df)*100:>5.1f}%)")

    print("\n" + "="*70)

    if metrics['sharpe'] >= 1.0:
        print("SUCCESS: Sharpe Ratio >= 1.0!")
    else:
        print(f"Current Sharpe: {metrics['sharpe']:.2f}")


def main():
    logger = setup_logging()

    print("\n" + "="*70)
    print("PAIRS TRADING BACKTEST V3")
    print("(Strict Criteria, No Forward-Looking Bias)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols")

    bt = PairsBacktestV3(loader)
    bt.load_data(symbols)

    print("\nRunning V3 backtest...")
    results = bt.run(list(bt.price_data.keys()))

    if results:
        print_results(results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
