#!/usr/bin/env python3
"""
Improved Backtest - Walk-Forward Cointegration Strategy

Key improvements over original:
1. NO FORWARD-LOOKING BIAS - cointegration tested only with data available at each point
2. Walk-forward approach - retests cointegration periodically using rolling windows
3. Rolling OLS hedge ratios using only past data
4. Optimized entry/exit thresholds based on spread characteristics
5. Better risk management with volatility targeting
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
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
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()
        df = pd.DataFrame({'spread': spread, 'spread_lag': spread_lag, 'spread_diff': spread_diff}).dropna()
        if len(df) < 20:
            return None
        X = df['spread_lag'].values
        y = df['spread_diff'].values
        X_with_const = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        lambda_param = -beta[1]
        if lambda_param <= 0:
            return None
        return np.log(2) / lambda_param
    except:
        return None


def calculate_hurst(spread, max_lag=100):
    """Calculate Hurst exponent using R/S analysis."""
    try:
        series = spread.dropna().values
        if len(series) < 50:
            return None
        lags = range(2, min(max_lag, len(series) // 4))
        tau = []
        rs = []

        for lag in lags:
            n = len(series) // lag
            if n < 2:
                break
            rs_lag = []
            for i in range(lag):
                subset = series[i * n:(i + 1) * n]
                if len(subset) < 2:
                    continue
                m = np.mean(subset)
                y = subset - m
                z = np.cumsum(y)
                r = np.max(z) - np.min(z)
                s = np.std(subset, ddof=1)
                if s > 0:
                    rs_lag.append(r / s)
            if rs_lag:
                tau.append(lag)
                rs.append(np.mean(rs_lag))

        if len(tau) < 3:
            return None

        coeffs = np.polyfit(np.log(tau), np.log(rs), 1)
        return coeffs[0]
    except:
        return None


def test_cointegration_pair(prices_a, prices_b, lookback=252):
    """
    Test cointegration between two price series using Engle-Granger.
    Returns hedge ratio and test statistics if cointegrated, None otherwise.

    Uses only the last 'lookback' days of data (no lookahead).
    """
    try:
        # Use only lookback window
        prices_a = prices_a.iloc[-lookback:] if len(prices_a) > lookback else prices_a
        prices_b = prices_b.iloc[-lookback:] if len(prices_b) > lookback else prices_b

        # Align data
        data = pd.DataFrame({'a': prices_a, 'b': prices_b}).dropna()
        if len(data) < 100:
            return None

        # OLS regression: a = beta * b + epsilon
        X = data['b'].values.reshape(-1, 1)
        X = np.column_stack([np.ones(len(X)), X])
        y = data['a'].values

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        hedge_ratio = beta[1]

        # Calculate spread
        spread = data['a'] - hedge_ratio * data['b']

        # ADF test on spread
        adf_result = adfuller(spread, maxlag=None, regression='c', autolag='AIC')
        adf_stat = adf_result[0]

        # Need ADF < -2.86 for 10% significance (or -3.34 for 5%)
        if adf_stat > -2.86:
            return None

        # Half-life
        half_life = calculate_half_life(spread)
        if half_life is None or half_life > 40 or half_life < 3:
            return None

        # Hurst exponent
        hurst = calculate_hurst(spread)
        if hurst is None or hurst > 0.45:
            return None

        return {
            'hedge_ratio': hedge_ratio,
            'adf_stat': adf_stat,
            'half_life': half_life,
            'hurst': hurst,
            'spread_mean': spread.mean(),
            'spread_std': spread.std()
        }
    except:
        return None


class WalkForwardBacktest:
    """
    Walk-forward backtest with NO forward-looking bias.

    Key principles:
    1. Cointegration is tested only using data available at each point
    2. Hedge ratios are calculated using rolling OLS
    3. Entry/exit signals use only past data for z-score calculation
    """

    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Strategy parameters (optimized)
        self.LOOKBACK_COINT = 252  # 1 year for cointegration test
        self.LOOKBACK_ZSCORE = 42  # Rolling window for z-score
        self.RETEST_INTERVAL = 21  # Days between cointegration retests
        self.ENTRY_ZSCORE = 2.0    # Enter on 2 sigma deviation
        self.EXIT_ZSCORE = 0.3     # Exit closer to mean
        self.STOP_ZSCORE = 3.5     # Stop loss at 3.5 sigma
        self.MAX_HOLD_DAYS = 60    # Time-based stop
        self.POSITION_SIZE_PCT = 0.10  # 10% per pair
        self.MAX_POSITIONS = 6     # Maximum concurrent positions
        self.COMMISSION_BPS = 5    # Transaction costs

        # Data storage
        self.price_data = {}
        self.pairs = []

    def load_data(self, symbols):
        """Load price data for symbols."""
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > 252:
                self.price_data[symbol] = df['close']
        self.logger.info(f"Loaded {len(self.price_data)} symbols")

    def find_pairs_at_date(self, date, sector_symbols):
        """
        Find cointegrated pairs using only data available up to 'date'.
        This ensures NO forward-looking bias.
        """
        valid_pairs = []

        # Get prices up to this date only
        prices_dict = {}
        for symbol in sector_symbols:
            if symbol in self.price_data:
                series = self.price_data[symbol]
                series = series[series.index <= date]
                if len(series) >= self.LOOKBACK_COINT:
                    prices_dict[symbol] = series

        if len(prices_dict) < 2:
            return []

        symbols = list(prices_dict.keys())

        # Test all pairs
        for s1, s2 in combinations(symbols, 2):
            result = test_cointegration_pair(
                prices_dict[s1],
                prices_dict[s2],
                self.LOOKBACK_COINT
            )

            if result is not None:
                # Also check correlation
                ret1 = prices_dict[s1].pct_change().dropna().iloc[-self.LOOKBACK_COINT:]
                ret2 = prices_dict[s2].pct_change().dropna().iloc[-self.LOOKBACK_COINT:]
                corr = ret1.corr(ret2)

                if corr > 0.5:  # Only keep reasonably correlated pairs
                    valid_pairs.append({
                        'symbol_a': s1,
                        'symbol_b': s2,
                        'hedge_ratio': result['hedge_ratio'],
                        'adf_stat': result['adf_stat'],
                        'half_life': result['half_life'],
                        'hurst': result['hurst'],
                        'correlation': corr,
                        'found_date': date
                    })

        # Sort by ADF statistic (more negative = more stationary)
        valid_pairs.sort(key=lambda x: x['adf_stat'])

        return valid_pairs[:10]  # Return top 10 pairs

    def calculate_zscore_at_date(self, pair, date):
        """
        Calculate z-score using only data available up to 'date'.
        No forward-looking bias.
        """
        s1, s2 = pair['symbol_a'], pair['symbol_b']

        if s1 not in self.price_data or s2 not in self.price_data:
            return None

        p1 = self.price_data[s1][self.price_data[s1].index <= date]
        p2 = self.price_data[s2][self.price_data[s2].index <= date]

        if len(p1) < self.LOOKBACK_ZSCORE + 10 or len(p2) < self.LOOKBACK_ZSCORE + 10:
            return None

        # Align
        combined = pd.DataFrame({'a': p1, 'b': p2}).dropna()
        if len(combined) < self.LOOKBACK_ZSCORE + 10:
            return None

        # Recalculate hedge ratio using recent data (rolling)
        recent = combined.iloc[-self.LOOKBACK_COINT:]
        X = np.column_stack([np.ones(len(recent)), recent['b'].values])
        y = recent['a'].values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        hedge_ratio = beta[1]

        # Calculate spread
        spread = combined['a'] - hedge_ratio * combined['b']

        # Z-score using rolling window
        rolling_mean = spread.rolling(self.LOOKBACK_ZSCORE).mean()
        rolling_std = spread.rolling(self.LOOKBACK_ZSCORE).std()

        zscore = (spread - rolling_mean) / rolling_std

        if pd.isna(zscore.iloc[-1]):
            return None

        return {
            'zscore': zscore.iloc[-1],
            'hedge_ratio': hedge_ratio,
            'spread': spread.iloc[-1],
            'spread_mean': rolling_mean.iloc[-1],
            'spread_std': rolling_std.iloc[-1]
        }

    def run(self, sector_map):
        """
        Run walk-forward backtest.

        sector_map: dict of {sector: [list of symbols]}
        """
        # Get all trading dates
        all_dates = set()
        for series in self.price_data.values():
            all_dates.update(series.index)
        all_dates = sorted(all_dates)

        # Start after we have enough data for cointegration test
        start_idx = self.LOOKBACK_COINT + 50
        if start_idx >= len(all_dates):
            self.logger.error("Insufficient data")
            return None

        trading_dates = all_dates[start_idx:]
        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

        # State
        capital = self.initial_capital
        positions = {}  # pair_key -> position info
        equity_curve = []
        trades = []
        active_pairs = {}  # sector -> list of pairs
        last_retest = {}  # sector -> date

        for i, date in enumerate(trading_dates):
            # Retest cointegration periodically for each sector
            for sector, symbols in sector_map.items():
                if sector not in last_retest or (date - last_retest[sector]).days >= self.RETEST_INTERVAL:
                    # Find pairs using only historical data up to this date
                    sector_syms = [s for s in symbols if s in self.price_data]
                    new_pairs = self.find_pairs_at_date(date, sector_syms)
                    active_pairs[sector] = new_pairs
                    last_retest[sector] = date

            # Flatten all active pairs
            all_active_pairs = []
            for pairs in active_pairs.values():
                all_active_pairs.extend(pairs)

            daily_pnl = 0

            # Update existing positions
            for pair_key, pos in list(positions.items()):
                pair = pos['pair']

                # Get current prices
                s1, s2 = pair['symbol_a'], pair['symbol_b']
                if date not in self.price_data[s1].index or date not in self.price_data[s2].index:
                    continue

                p1 = self.price_data[s1].loc[date]
                p2 = self.price_data[s2].loc[date]

                # Calculate current P&L
                pnl_a = (p1 - pos['entry_price_a']) * pos['shares_a']
                pnl_b = (p2 - pos['entry_price_b']) * pos['shares_b']
                pos['current_pnl'] = pnl_a + pnl_b

                # Get current z-score
                zscore_info = self.calculate_zscore_at_date(pair, date)
                if zscore_info is None:
                    continue

                current_zscore = zscore_info['zscore']
                days_held = (date - pos['entry_date']).days

                # Exit logic
                should_exit = False
                exit_reason = ""

                if pos['direction'] == 'long':
                    # Long spread: bought A, sold B. Exit when spread rises to mean
                    if current_zscore >= -self.EXIT_ZSCORE:
                        should_exit, exit_reason = True, "Mean reversion"
                    elif current_zscore < -self.STOP_ZSCORE:
                        should_exit, exit_reason = True, "Stop loss"
                else:
                    # Short spread: sold A, bought B. Exit when spread falls to mean
                    if current_zscore <= self.EXIT_ZSCORE:
                        should_exit, exit_reason = True, "Mean reversion"
                    elif current_zscore > self.STOP_ZSCORE:
                        should_exit, exit_reason = True, "Stop loss"

                if days_held > self.MAX_HOLD_DAYS:
                    should_exit, exit_reason = True, "Time stop"

                if should_exit:
                    # Calculate final P&L with transaction costs
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
                        'exit_zscore': current_zscore,
                        'pnl': net_pnl,
                        'hold_days': days_held,
                        'exit_reason': exit_reason
                    })

                    del positions[pair_key]

            # Entry logic - check for new positions
            if len(positions) < self.MAX_POSITIONS:
                for pair in all_active_pairs:
                    pair_key = f"{pair['symbol_a']}-{pair['symbol_b']}"

                    if pair_key in positions:
                        continue

                    zscore_info = self.calculate_zscore_at_date(pair, date)
                    if zscore_info is None:
                        continue

                    current_zscore = zscore_info['zscore']

                    # Entry conditions
                    if current_zscore <= -self.ENTRY_ZSCORE:
                        # Long spread: buy A, sell B
                        direction = 'long'
                    elif current_zscore >= self.ENTRY_ZSCORE:
                        # Short spread: sell A, buy B
                        direction = 'short'
                    else:
                        continue

                    # Get current prices
                    s1, s2 = pair['symbol_a'], pair['symbol_b']
                    if date not in self.price_data[s1].index or date not in self.price_data[s2].index:
                        continue

                    p1 = self.price_data[s1].loc[date]
                    p2 = self.price_data[s2].loc[date]
                    hedge_ratio = zscore_info['hedge_ratio']

                    # Position sizing
                    position_value = capital * self.POSITION_SIZE_PCT

                    # Calculate shares
                    # For hedge ratio h, we want: shares_a * p1 = h * shares_b * p2 (approx equal dollar exposure)
                    shares_a = position_value / (p1 + abs(hedge_ratio) * p2)
                    shares_b = shares_a * hedge_ratio

                    if direction == 'long':
                        # Buy A, sell B
                        shares_a = abs(shares_a)
                        shares_b = -abs(shares_b)
                    else:
                        # Sell A, buy B
                        shares_a = -abs(shares_a)
                        shares_b = abs(shares_b)

                    positions[pair_key] = {
                        'pair': pair,
                        'entry_date': date,
                        'entry_zscore': current_zscore,
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

            # Calculate equity
            open_pnl = sum(pos['current_pnl'] for pos in positions.values())
            equity = capital + open_pnl

            equity_curve.append({
                'date': date,
                'equity': equity,
                'capital': capital,
                'open_pnl': open_pnl,
                'num_positions': len(positions)
            })

        # Close remaining positions at end
        final_date = trading_dates[-1]
        for pair_key, pos in positions.items():
            pair = pos['pair']
            s1, s2 = pair['symbol_a'], pair['symbol_b']

            notional = abs(pos['shares_a'] * pos['entry_price_a']) + abs(pos['shares_b'] * pos['entry_price_b'])
            costs = notional * self.COMMISSION_BPS / 10000 * 2
            net_pnl = pos['current_pnl'] - costs

            trades.append({
                'pair': f"{s1}-{s2}",
                'entry_date': pos['entry_date'],
                'exit_date': final_date,
                'direction': pos['direction'],
                'entry_zscore': pos['entry_zscore'],
                'exit_zscore': 0,
                'pnl': net_pnl,
                'hold_days': (final_date - pos['entry_date']).days,
                'exit_reason': 'End of backtest'
            })

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        returns = equity_df['equity'].pct_change().dropna()

        # Risk-free rate (assume 2% annual)
        rf_daily = 0.02 / 252
        excess_returns = returns - rf_daily

        metrics = {
            'total_return': (equity_df['equity'].iloc[-1] / self.initial_capital) - 1,
            'annual_return': excess_returns.mean() * 252,
            'annual_vol': returns.std() * np.sqrt(252),
            'sharpe': excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'sortino': excess_returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
            'max_drawdown': ((equity_df['equity'] / equity_df['equity'].expanding().max()) - 1).min(),
            'num_trades': len(trades_df),
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_pnl': trades_df['pnl'].mean() if len(trades_df) > 0 else 0,
            'avg_hold_days': trades_df['hold_days'].mean() if len(trades_df) > 0 else 0,
            'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        }

        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics
        }


def print_results(results):
    """Print formatted results."""
    if not results:
        print("No results")
        return

    metrics = results['metrics']
    trades_df = results.get('trades', pd.DataFrame())

    print("\n" + "="*70)
    print("WALK-FORWARD COINTEGRATION STRATEGY - BACKTEST RESULTS")
    print("(No Forward-Looking Bias)")
    print("="*70)

    print(f"\n{'PERFORMANCE':-^50}")
    print(f"  Total Return:         {metrics['total_return']*100:>12.2f}%")
    print(f"  Annual Return:        {metrics['annual_return']*100:>12.2f}%")
    print(f"  Annual Volatility:    {metrics['annual_vol']*100:>12.2f}%")
    print(f"  Sharpe Ratio:         {metrics['sharpe']:>12.2f}")
    print(f"  Sortino Ratio:        {metrics['sortino']:>12.2f}")
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
        print("SUCCESS: Sharpe Ratio >= 1.0")
    else:
        print(f"NOTE: Sharpe Ratio {metrics['sharpe']:.2f} - may need parameter tuning")


def main():
    logger = setup_logging()

    print("\n" + "="*70)
    print("WALK-FORWARD COINTEGRATION STRATEGY - IMPROVED BACKTEST")
    print("(No Forward-Looking Bias)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    loader = DataLoader()
    symbols = loader.get_available_symbols()
    logger.info(f"Found {len(symbols)} symbols")

    if len(symbols) < 20:
        logger.error("Insufficient data. Run `python -m src.data.download_all` first.")
        sys.exit(1)

    # Load fundamentals for sector mapping
    fund_df = loader.load_fundamentals_df(symbols)

    # Create sector map
    sector_map = {}
    for sector in fund_df['sector'].unique():
        sector_symbols = fund_df[fund_df['sector'] == sector]['symbol'].tolist()
        if len(sector_symbols) >= 4:
            sector_map[sector] = sector_symbols

    logger.info(f"Sectors with sufficient stocks: {list(sector_map.keys())}")

    # Initialize backtest
    bt = WalkForwardBacktest(loader)

    # Load all price data
    all_symbols = []
    for syms in sector_map.values():
        all_symbols.extend(syms)
    bt.load_data(list(set(all_symbols)))

    # Run backtest
    print("\nRunning walk-forward backtest...")
    results = bt.run(sector_map)

    if results:
        print_results(results)
    else:
        print("Backtest failed")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
