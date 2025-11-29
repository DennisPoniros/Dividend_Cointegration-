#!/usr/bin/env python3
"""
Practical Backtest v2 - Uses Engle-Granger approach for finding tradeable spreads.

The Johansen test is very strict and may reject spreads that are actually tradeable.
This version uses a more practical approach:
1. Find highly correlated pairs/groups within sectors
2. Use Engle-Granger ADF on the spread
3. Validate with half-life and Hurst exponent
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
import logging

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

        if len(tau) < 2:
            return None

        coeffs = np.polyfit(np.log(tau), np.log(rs), 1)
        return coeffs[0]
    except:
        return None


def find_tradeable_spreads(loader, min_adf=-2.86, max_half_life=45, max_hurst=0.5):
    """
    Find tradeable spreads using Engle-Granger approach.

    Uses more relaxed criteria than strict Johansen:
    - ADF < -2.86 (10% significance for 4 variables)
    - Half-life < 45 days
    - Hurst < 0.5 (mean reverting)
    """
    logger = logging.getLogger(__name__)

    symbols = loader.get_available_symbols()
    fund_df = loader.load_fundamentals_df(symbols)

    vectors = []

    # Process each sector
    for sector in fund_df['sector'].unique():
        sector_symbols = fund_df[fund_df['sector'] == sector]['symbol'].tolist()

        if len(sector_symbols) < 4:
            continue

        # Load aligned prices
        prices = loader.load_prices_multi(sector_symbols, align=True)

        if len(prices) < 252 or len(prices.columns) < 4:
            continue

        logger.info(f"Scanning {sector}: {len(prices.columns)} stocks, {len(prices)} days")

        # Test combinations of 4-5 stocks
        test_symbols = prices.columns.tolist()[:15]  # Limit to prevent combinatorial explosion

        for size in [4, 5]:
            for combo in combinations(test_symbols, size):
                data = prices[list(combo)].dropna()

                if len(data) < 252:
                    continue

                # Calculate returns correlation
                returns = data.pct_change().dropna()
                corr_matrix = returns.corr()
                avg_corr = corr_matrix.values[np.triu_indices(len(combo), 1)].mean()

                if avg_corr < 0.5:  # Skip low correlation groups
                    continue

                # Use equal weights for simplicity (or could optimize)
                # For pairs trading, we'd use regression; for baskets, equal weight is common
                weights = np.ones(len(combo)) / len(combo)

                # Make it a spread (long-short)
                weights[len(combo)//2:] *= -1

                spread = (data * weights).sum(axis=1)

                # ADF test
                try:
                    adf_result = adfuller(spread, maxlag=None, regression='c', autolag='AIC')
                    adf_stat = adf_result[0]
                    adf_pvalue = adf_result[1]
                except:
                    continue

                if adf_stat > min_adf:
                    continue

                # Half-life
                half_life = calculate_half_life(spread)
                if half_life is None or half_life > max_half_life or half_life < 3:
                    continue

                # Hurst exponent
                hurst = calculate_hurst(spread)
                if hurst is None or hurst > max_hurst:
                    continue

                vector = {
                    'symbols': list(combo),
                    'weights': weights,
                    'sector': sector,
                    'adf_stat': adf_stat,
                    'adf_pvalue': adf_pvalue,
                    'half_life': half_life,
                    'hurst': hurst,
                    'avg_correlation': avg_corr
                }
                vectors.append(vector)
                logger.info(f"  Found vector: {combo}, ADF={adf_stat:.2f}, HL={half_life:.1f}, H={hurst:.2f}")

    logger.info(f"Total tradeable spreads found: {len(vectors)}")
    return vectors


def run_backtest(vectors, loader, initial_capital=1_000_000):
    """Run backtest on found vectors."""
    logger = logging.getLogger(__name__)

    if not vectors:
        logger.error("No vectors to backtest")
        return {}

    # Config
    ENTRY_ZSCORE = 1.5
    EXIT_ZSCORE = 0.5
    MAX_ZSCORE = 3.0
    ROLLING_WINDOW = 42
    MAX_HOLD_DAYS = 70
    POSITION_SIZE_PCT = 0.08
    COMMISSION_BPS = 10

    # Get all symbols
    all_symbols = set()
    for vec in vectors:
        all_symbols.update(vec['symbols'])

    # Load price data
    price_data = {}
    for symbol in all_symbols:
        df = loader.load_price_data(symbol)
        if df is not None:
            price_data[symbol] = df

    # Get date range
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)[ROLLING_WINDOW + 10:]

    logger.info(f"Backtest: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

    # Initialize
    capital = initial_capital
    positions = {}
    equity_curve = []
    trades = []

    for date in all_dates:
        daily_pnl = 0

        # Update P&L for open positions
        for vec_id, pos in list(positions.items()):
            vec = vectors[vec_id]
            current_value = 0
            entry_value = 0

            for i, symbol in enumerate(vec['symbols']):
                if symbol in price_data and date in price_data[symbol].index:
                    current_price = price_data[symbol].loc[date, 'close']
                    shares = pos['shares'][i]
                    current_value += shares * current_price
                    entry_value += shares * pos['entry_prices'][i]

            pos['current_pnl'] = current_value - entry_value
            daily_pnl += pos['current_pnl']

        # Process each vector
        for vec_id, vec in enumerate(vectors):
            symbols = vec['symbols']
            weights = vec['weights']

            # Get prices up to current date
            vec_prices = pd.DataFrame()
            has_all = True

            for symbol in symbols:
                if symbol not in price_data:
                    has_all = False
                    break
                symbol_df = price_data[symbol][price_data[symbol].index <= date]
                if len(symbol_df) < ROLLING_WINDOW + 5:
                    has_all = False
                    break
                vec_prices[symbol] = symbol_df['close']

            if not has_all or len(vec_prices.dropna()) < ROLLING_WINDOW + 5:
                continue

            vec_prices = vec_prices.dropna()

            # Calculate spread and z-score
            spread = (vec_prices * weights).sum(axis=1)
            rolling_mean = spread.rolling(ROLLING_WINDOW).mean()
            rolling_std = spread.rolling(ROLLING_WINDOW).std()
            zscore = (spread - rolling_mean) / rolling_std

            current_zscore = zscore.iloc[-1]
            if pd.isna(current_zscore):
                continue

            # Manage position
            if vec_id in positions:
                pos = positions[vec_id]
                days_held = (date - pos['entry_date']).days

                should_exit = False
                exit_reason = ""

                if pos['direction'] == 1 and current_zscore >= -EXIT_ZSCORE:
                    should_exit, exit_reason = True, "Mean reversion"
                elif pos['direction'] == -1 and current_zscore <= EXIT_ZSCORE:
                    should_exit, exit_reason = True, "Mean reversion"
                elif abs(current_zscore) > MAX_ZSCORE:
                    should_exit, exit_reason = True, "Stop loss"
                elif days_held > MAX_HOLD_DAYS:
                    should_exit, exit_reason = True, "Time stop"

                if should_exit:
                    # Close position
                    net_pnl = pos['current_pnl']
                    # Subtract transaction costs
                    notional = sum(abs(s * p) for s, p in zip(pos['shares'], pos['entry_prices']))
                    costs = notional * COMMISSION_BPS / 10000 * 2  # Entry + exit
                    net_pnl -= costs

                    capital += net_pnl

                    trades.append({
                        'vector_id': vec_id,
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                        'entry_zscore': pos['entry_zscore'],
                        'exit_zscore': current_zscore,
                        'pnl': net_pnl,
                        'hold_days': days_held,
                        'exit_reason': exit_reason
                    })

                    del positions[vec_id]

            else:
                # Check for entry
                if abs(current_zscore) >= ENTRY_ZSCORE:
                    position_size = capital * POSITION_SIZE_PCT
                    direction = 1 if current_zscore <= -ENTRY_ZSCORE else -1

                    # Calculate shares
                    current_prices = [vec_prices[s].iloc[-1] for s in symbols]
                    notionals = [position_size * abs(w) for w in weights]
                    shares = [n / p * np.sign(w) * direction for n, p, w in zip(notionals, current_prices, weights)]

                    positions[vec_id] = {
                        'entry_date': date,
                        'entry_zscore': current_zscore,
                        'direction': direction,
                        'shares': shares,
                        'entry_prices': current_prices,
                        'current_pnl': 0
                    }

        # Track equity
        open_pnl = sum(pos['current_pnl'] for pos in positions.values())
        equity = capital + open_pnl

        equity_curve.append({
            'date': date,
            'equity': equity,
            'num_positions': len(positions)
        })

    # Close remaining positions
    for vec_id, pos in positions.items():
        net_pnl = pos['current_pnl']
        notional = sum(abs(s * p) for s, p in zip(pos['shares'], pos['entry_prices']))
        costs = notional * COMMISSION_BPS / 10000 * 2
        capital += net_pnl - costs

        trades.append({
            'vector_id': vec_id,
            'entry_date': pos['entry_date'],
            'exit_date': all_dates[-1],
            'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
            'entry_zscore': pos['entry_zscore'],
            'exit_zscore': 0,
            'pnl': net_pnl - costs,
            'hold_days': (all_dates[-1] - pos['entry_date']).days,
            'exit_reason': 'End of backtest'
        })

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    returns = equity_df['equity'].pct_change().dropna()

    metrics = {
        'total_return': (equity_df['equity'].iloc[-1] / initial_capital) - 1,
        'annual_return': returns.mean() * 252,
        'annual_vol': returns.std() * np.sqrt(252),
        'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'max_drawdown': ((equity_df['equity'] / equity_df['equity'].expanding().max()) - 1).min(),
        'num_trades': len(trades_df),
        'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0,
        'avg_pnl': trades_df['pnl'].mean() if len(trades_df) > 0 else 0,
        'avg_hold_days': trades_df['hold_days'].mean() if len(trades_df) > 0 else 0
    }

    return {
        'equity_curve': equity_df,
        'trades': trades_df,
        'metrics': metrics,
        'vectors': vectors
    }


def print_results(results):
    """Print formatted results."""
    if not results:
        print("No results")
        return

    metrics = results['metrics']
    trades_df = results.get('trades', pd.DataFrame())
    vectors = results.get('vectors', [])

    print("\n" + "="*70)
    print("DIVIDEND COINTEGRATION STRATEGY - BACKTEST RESULTS")
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
    print(f"  Avg P&L per Trade:    ${metrics['avg_pnl']:>11,.0f}")
    print(f"  Avg Hold Days:        {metrics['avg_hold_days']:>12.1f}")

    if not trades_df.empty:
        print(f"\n{'EXIT REASONS':-^50}")
        for reason, count in trades_df['exit_reason'].value_counts().items():
            print(f"  {reason:<25} {count:>5} ({count/len(trades_df)*100:>5.1f}%)")

    print(f"\n{'VECTORS BY SECTOR':-^50}")
    sector_counts = {}
    for v in vectors:
        sector_counts[v['sector']] = sector_counts.get(v['sector'], 0) + 1
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {sector:<30} {count:>5}")

    print("\n" + "="*70)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-"*70)

    if metrics['sharpe'] < 0.5:
        print("1. LOW SHARPE: Consider tighter entry thresholds or better vector selection")
    elif metrics['sharpe'] > 1.0:
        print("1. GOOD SHARPE: Strategy shows promise, consider increasing allocation")

    if abs(metrics['max_drawdown']) > 0.15:
        print("2. HIGH DRAWDOWN: Consider reducing position sizes or adding more vectors")

    if metrics['win_rate'] < 0.45:
        print("3. LOW WIN RATE: Consider faster exits or stricter vector criteria")

    if metrics['avg_hold_days'] > 40:
        print("4. LONG HOLDS: Consider tighter half-life requirements")

    print("5. DATA: Update quarterly with `python -m src.data.download_all`")
    print("6. DIVERSIFICATION: Add more sectors with liquid dividend stocks")


def main():
    logger = setup_logging()

    print("\n" + "="*70)
    print("DIVIDEND COINTEGRATION STRATEGY - PRACTICAL BACKTEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    loader = DataLoader()
    available = loader.get_available_symbols()
    logger.info(f"Found {len(available)} symbols with price data")

    if len(available) < 20:
        logger.error("Insufficient data. Run `python -m src.data.download_all` first.")
        sys.exit(1)

    # Step 1: Find tradeable spreads
    print("\n[1/2] Finding tradeable spreads (Engle-Granger approach)...")
    vectors = find_tradeable_spreads(
        loader,
        min_adf=-2.86,  # 10% significance
        max_half_life=45,
        max_hurst=0.5
    )

    if not vectors:
        logger.error("No tradeable spreads found")
        sys.exit(1)

    # Step 2: Run backtest
    print("\n[2/2] Running backtest...")
    results = run_backtest(vectors, loader)

    # Print results
    print_results(results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
