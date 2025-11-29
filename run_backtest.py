#!/usr/bin/env python3
"""
Run Complete Strategy Backtest

This script runs the full dividend cointegration strategy on local data:
1. Load universe and price data from local storage
2. Form sub-baskets using sector-correlation clustering
3. Find cointegration vectors
4. Run backtest simulation
5. Report performance metrics

Usage:
    python run_backtest.py
    python run_backtest.py --capital 500000
    python run_backtest.py --start-date 2022-01-01
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.strategy.cointegration import CointegrationTester
from src.strategy.signals import SignalGenerator
from src.strategy.portfolio import PortfolioManager
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceMetrics


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def form_sector_baskets(
    loader: DataLoader,
    min_stocks_per_sector: int = 8,
    correlation_threshold: float = 0.7,
    min_basket_size: int = 4,
    max_basket_size: int = 8
) -> List[Dict]:
    """
    Form baskets using sector grouping and correlation clustering.

    Simplified version that works with local data.
    """
    logger = logging.getLogger(__name__)

    # Get universe
    try:
        universe = loader.get_universe()
    except FileNotFoundError:
        logger.warning("Universe file not found, using available symbols")
        symbols = loader.get_available_symbols()
        fundamentals = loader.load_fundamentals_df(symbols)
        universe = fundamentals

    if universe.empty:
        logger.error("No universe data available")
        return []

    # Group by sector
    sectors = universe.groupby('sector')['symbol'].apply(list).to_dict()

    logger.info(f"Found {len(sectors)} sectors")
    for sector, symbols in sectors.items():
        logger.info(f"  {sector}: {len(symbols)} stocks")

    baskets = []

    for sector, symbols in sectors.items():
        if len(symbols) < min_stocks_per_sector:
            logger.debug(f"Skipping {sector}: only {len(symbols)} stocks")
            continue

        # Load prices for this sector
        prices = loader.load_prices_multi(symbols, align=True)

        if prices.empty or len(prices) < 252:
            logger.debug(f"Insufficient price data for {sector}")
            continue

        # Calculate correlation matrix
        returns = prices.pct_change().dropna()
        corr_matrix = returns.tail(252).corr()

        # Find highly correlated groups
        used = set()

        for symbol in symbols:
            if symbol not in corr_matrix.columns or symbol in used:
                continue

            # Find stocks correlated with this one
            correlations = corr_matrix[symbol].drop(symbol)
            high_corr = correlations[correlations >= correlation_threshold]

            cluster = [symbol] + [s for s in high_corr.index if s not in used]

            if min_basket_size <= len(cluster) <= max_basket_size:
                # Calculate average correlation within cluster
                cluster_corr = corr_matrix.loc[cluster, cluster]
                avg_corr = cluster_corr.values[np.triu_indices(len(cluster), 1)].mean()

                basket = {
                    'sector': sector,
                    'symbols': cluster[:max_basket_size],
                    'size': len(cluster[:max_basket_size]),
                    'avg_correlation': avg_corr
                }
                baskets.append(basket)
                used.update(cluster)

                logger.info(f"Created basket: {sector} - {len(basket['symbols'])} stocks, avg_corr={avg_corr:.3f}")

    logger.info(f"Total baskets formed: {len(baskets)}")
    return baskets


def find_cointegration_vectors(
    baskets: List[Dict],
    loader: DataLoader,
    testing_days: int = 504
) -> List[Dict]:
    """Find cointegration vectors in baskets."""
    logger = logging.getLogger(__name__)

    tester = CointegrationTester()
    all_vectors = []

    for i, basket in enumerate(baskets):
        logger.info(f"Testing basket {i+1}/{len(baskets)}: {basket['sector']} ({basket['size']} stocks)")

        # Load aligned prices
        prices, valid_symbols = loader.get_aligned_data_for_basket(
            basket['symbols'],
            min_days=testing_days
        )

        if prices.empty or len(valid_symbols) < 4:
            logger.debug(f"Insufficient data for basket {i}")
            continue

        # Test this basket
        basket_info = {
            'sector': basket['sector'],
            'symbols': valid_symbols
        }

        vectors = tester.test_basket(prices, basket_info)
        all_vectors.extend(vectors)

        if vectors:
            logger.info(f"  Found {len(vectors)} cointegration vectors")

    logger.info(f"Total cointegration vectors found: {len(all_vectors)}")
    return all_vectors


def run_simplified_backtest(
    vectors: List[Dict],
    loader: DataLoader,
    initial_capital: float = 1_000_000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """
    Run a simplified backtest of the cointegration strategy.

    This is a cleaner implementation that avoids some complexity in the full engine.
    """
    logger = logging.getLogger(__name__)

    if not vectors:
        logger.error("No cointegration vectors to backtest")
        return {}

    logger.info(f"Running backtest with {len(vectors)} vectors, ${initial_capital:,.0f} capital")

    # Configuration
    ENTRY_ZSCORE = 1.5
    EXIT_ZSCORE = 0.5
    MAX_ZSCORE = 3.0
    ROLLING_WINDOW = 42
    MAX_HOLD_DAYS = 70
    KELLY_FRACTION = 0.25
    MAX_POSITION_PCT = 0.08
    COMMISSION_PER_SHARE = 0.005
    SLIPPAGE_BPS = 5

    # Get all required symbols
    all_symbols = set()
    for vec in vectors:
        all_symbols.update(vec['symbols'])

    # Load all price data
    price_data = {}
    for symbol in all_symbols:
        df = loader.load_price_data(symbol)
        if df is not None:
            price_data[symbol] = df

    # Get date range
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)

    all_dates = sorted(all_dates)

    if start_date:
        start_dt = pd.to_datetime(start_date)
        all_dates = [d for d in all_dates if d >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date)
        all_dates = [d for d in all_dates if d <= end_dt]

    # Need enough history for rolling calculations
    all_dates = all_dates[ROLLING_WINDOW + 10:]

    logger.info(f"Backtest period: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

    # Initialize tracking
    capital = initial_capital
    positions = {}  # vector_id -> {entry_date, entry_zscore, direction, shares, entry_prices}

    equity_curve = []
    all_trades = []

    # Main loop
    for date_idx, date in enumerate(all_dates):
        daily_pnl = 0

        # Calculate current P&L for all positions
        for vec_id, pos in list(positions.items()):
            vec = vectors[vec_id]
            current_value = 0
            entry_value = 0

            for symbol, shares in pos['shares'].items():
                if symbol in price_data and date in price_data[symbol].index:
                    current_price = price_data[symbol].loc[date, 'close']
                    entry_price = pos['entry_prices'][symbol]
                    current_value += shares * current_price
                    entry_value += shares * entry_price

            pos['current_pnl'] = (current_value - entry_value) * pos['direction']
            daily_pnl += pos['current_pnl']

        # Process each vector
        for vec_id, vec in enumerate(vectors):
            symbols = vec['symbols']
            weights = vec['weights']

            # Get prices up to current date
            vec_prices = pd.DataFrame()
            has_all_prices = True

            for symbol in symbols:
                if symbol not in price_data:
                    has_all_prices = False
                    break

                symbol_df = price_data[symbol][price_data[symbol].index <= date]
                if len(symbol_df) < ROLLING_WINDOW + 5:
                    has_all_prices = False
                    break

                vec_prices[symbol] = symbol_df['close']

            if not has_all_prices or vec_prices.empty:
                continue

            vec_prices = vec_prices.dropna()

            if len(vec_prices) < ROLLING_WINDOW + 5:
                continue

            # Calculate spread and z-score
            normalized_weights = weights / np.sum(np.abs(weights))
            spread = (vec_prices * normalized_weights).sum(axis=1)

            rolling_mean = spread.rolling(ROLLING_WINDOW).mean()
            rolling_std = spread.rolling(ROLLING_WINDOW).std()
            zscore = (spread - rolling_mean) / rolling_std

            current_zscore = zscore.iloc[-1]

            if pd.isna(current_zscore):
                continue

            # Check if we have a position
            if vec_id in positions:
                pos = positions[vec_id]
                days_held = (date - pos['entry_date']).days

                # Check exit conditions
                should_exit = False
                exit_reason = ""

                # Z-score exit
                if pos['direction'] == 1 and current_zscore >= -EXIT_ZSCORE:
                    should_exit = True
                    exit_reason = "Z-score mean reversion"
                elif pos['direction'] == -1 and current_zscore <= EXIT_ZSCORE:
                    should_exit = True
                    exit_reason = "Z-score mean reversion"

                # Stop loss
                if abs(current_zscore) > MAX_ZSCORE:
                    should_exit = True
                    exit_reason = "Stop loss (z-score breach)"

                # Time stop
                if days_held > MAX_HOLD_DAYS:
                    should_exit = True
                    exit_reason = "Time stop"

                if should_exit:
                    # Close position
                    exit_pnl = 0
                    exit_costs = 0

                    for symbol, shares in pos['shares'].items():
                        if symbol in price_data and date in price_data[symbol].index:
                            exit_price = price_data[symbol].loc[date, 'close']
                            entry_price = pos['entry_prices'][symbol]

                            trade_pnl = (exit_price - entry_price) * shares * pos['direction']
                            trade_cost = abs(shares) * COMMISSION_PER_SHARE + abs(shares * exit_price) * SLIPPAGE_BPS / 10000

                            exit_pnl += trade_pnl
                            exit_costs += trade_cost

                    net_pnl = exit_pnl - exit_costs - pos['entry_costs']
                    capital += net_pnl

                    all_trades.append({
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
                    logger.debug(f"Closed {vec_id}: {exit_reason}, PnL=${net_pnl:,.2f}")

            else:
                # Check entry conditions
                if abs(current_zscore) >= ENTRY_ZSCORE:
                    # Calculate position size
                    position_size = min(
                        capital * KELLY_FRACTION,
                        capital * MAX_POSITION_PCT
                    )

                    # Don't enter if we don't have enough capital
                    if position_size < 1000:
                        continue

                    direction = 1 if current_zscore <= -ENTRY_ZSCORE else -1

                    # Calculate shares for each stock
                    shares = {}
                    entry_prices = {}
                    entry_costs = 0

                    for symbol, weight in zip(symbols, normalized_weights):
                        if symbol in price_data and date in price_data[symbol].index:
                            price = price_data[symbol].loc[date, 'close']
                            notional = position_size * abs(weight)
                            num_shares = notional / price

                            shares[symbol] = num_shares * np.sign(weight)
                            entry_prices[symbol] = price
                            entry_costs += abs(num_shares) * COMMISSION_PER_SHARE + notional * SLIPPAGE_BPS / 10000

                    if shares:
                        positions[vec_id] = {
                            'entry_date': date,
                            'entry_zscore': current_zscore,
                            'direction': direction,
                            'shares': shares,
                            'entry_prices': entry_prices,
                            'entry_costs': entry_costs,
                            'current_pnl': 0
                        }

                        logger.debug(f"Entered {vec_id}: {'LONG' if direction == 1 else 'SHORT'}, z={current_zscore:.2f}")

        # Track equity
        open_pnl = sum(pos['current_pnl'] for pos in positions.values())
        equity = capital + open_pnl

        equity_curve.append({
            'date': date,
            'equity': equity,
            'capital': capital,
            'open_pnl': open_pnl,
            'num_positions': len(positions)
        })

    # Close any remaining positions at the end
    for vec_id, pos in list(positions.items()):
        final_pnl = pos['current_pnl'] - pos['entry_costs']
        capital += final_pnl

        all_trades.append({
            'vector_id': vec_id,
            'entry_date': pos['entry_date'],
            'exit_date': all_dates[-1],
            'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
            'entry_zscore': pos['entry_zscore'],
            'exit_zscore': 0,
            'pnl': final_pnl,
            'hold_days': (all_dates[-1] - pos['entry_date']).days,
            'exit_reason': 'End of backtest'
        })

    # Create results
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    trades_df = pd.DataFrame(all_trades)

    # Calculate metrics
    if not equity_df.empty:
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_df['equity'],
            trades_df if not trades_df.empty else None
        )
    else:
        metrics = {}

    results = {
        'equity_curve': equity_df,
        'trades': trades_df,
        'metrics': metrics,
        'final_capital': equity_df['equity'].iloc[-1] if not equity_df.empty else initial_capital,
        'total_return': (equity_df['equity'].iloc[-1] / initial_capital - 1) if not equity_df.empty else 0,
        'num_vectors_used': len(vectors),
        'total_trades': len(trades_df)
    }

    return results


def print_results(results: Dict, vectors: List[Dict]):
    """Print formatted backtest results."""
    print("\n" + "="*70)
    print("DIVIDEND COINTEGRATION STRATEGY - BACKTEST RESULTS")
    print("="*70)

    if not results:
        print("No results to display")
        return

    metrics = results.get('metrics', {})
    trades_df = results.get('trades', pd.DataFrame())

    print(f"\n{'RETURN METRICS':-^50}")
    print(f"  Initial Capital:      ${1_000_000:>15,.2f}")
    print(f"  Final Capital:        ${results.get('final_capital', 0):>15,.2f}")
    print(f"  Total Return:         {results.get('total_return', 0)*100:>15.2f}%")
    print(f"  Annual Return:        {metrics.get('annual_return', 0)*100:>15.2f}%")
    print(f"  Annual Volatility:    {metrics.get('annual_volatility', 0)*100:>15.2f}%")

    print(f"\n{'RISK-ADJUSTED METRICS':-^50}")
    print(f"  Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):>15.3f}")
    print(f"  Sortino Ratio:        {metrics.get('sortino_ratio', 0):>15.3f}")
    print(f"  Calmar Ratio:         {metrics.get('calmar_ratio', 0):>15.3f}")

    print(f"\n{'DRAWDOWN METRICS':-^50}")
    print(f"  Max Drawdown:         {metrics.get('max_drawdown', 0)*100:>15.2f}%")
    print(f"  Peak Date:            {metrics.get('peak_date', 'N/A')}")
    print(f"  Valley Date:          {metrics.get('valley_date', 'N/A')}")

    if not trades_df.empty:
        print(f"\n{'TRADE STATISTICS':-^50}")
        print(f"  Total Trades:         {len(trades_df):>15}")
        print(f"  Win Rate:             {metrics.get('win_rate', 0)*100:>15.2f}%")
        print(f"  Avg Win:              ${metrics.get('avg_win', 0):>15,.2f}")
        print(f"  Avg Loss:             ${metrics.get('avg_loss', 0):>15,.2f}")
        print(f"  Profit Factor:        {metrics.get('profit_factor', 0):>15.2f}")

        avg_hold = trades_df['hold_days'].mean()
        print(f"  Avg Hold Days:        {avg_hold:>15.1f}")

        # Exit reasons breakdown
        print(f"\n{'EXIT REASONS':-^50}")
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = count / len(trades_df) * 100
            print(f"  {reason:<30} {count:>5} ({pct:>5.1f}%)")

    print(f"\n{'STRATEGY INFO':-^50}")
    print(f"  Cointegration Vectors: {results.get('num_vectors_used', 0):>14}")
    print(f"  Unique Stocks:         {len(set(s for v in vectors for s in v['symbols'])):>14}")

    # Sector breakdown
    sectors = {}
    for vec in vectors:
        sector = vec.get('basket_sector', 'Unknown')
        sectors[sector] = sectors.get(sector, 0) + 1

    print(f"\n{'VECTORS BY SECTOR':-^50}")
    for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"  {sector:<35} {count:>5}")

    print("\n" + "="*70)


def generate_recommendations(results: Dict, vectors: List[Dict]) -> List[str]:
    """Generate recommendations based on backtest results."""
    recommendations = []
    metrics = results.get('metrics', {})
    trades_df = results.get('trades', pd.DataFrame())

    sharpe = metrics.get('sharpe_ratio', 0)
    max_dd = abs(metrics.get('max_drawdown', 0))
    win_rate = metrics.get('win_rate', 0)
    annual_return = metrics.get('annual_return', 0)

    # Performance assessment
    if sharpe < 0.5:
        recommendations.append(
            "LOW SHARPE RATIO: Consider tightening entry thresholds (e.g., 1.75σ instead of 1.5σ) "
            "or relaxing correlation requirements to find more robust vectors."
        )
    elif sharpe > 1.5:
        recommendations.append(
            "STRONG SHARPE: Strategy shows good risk-adjusted returns. Consider slightly "
            "increasing position sizes or Kelly fraction."
        )

    if max_dd > 0.15:
        recommendations.append(
            "HIGH DRAWDOWN: Max drawdown exceeds 15% target. Consider reducing position sizes, "
            "tightening stop losses, or increasing diversification across vectors."
        )

    if win_rate < 0.50:
        recommendations.append(
            "LOW WIN RATE: Win rate below 50%. This is acceptable if average win >> average loss, "
            "but consider tightening exit thresholds for faster exits."
        )

    if not trades_df.empty:
        avg_hold = trades_df['hold_days'].mean()
        if avg_hold > 40:
            recommendations.append(
                f"LONG HOLD TIMES: Average {avg_hold:.0f} days exceeds target 25-35 days. "
                "Consider more aggressive exit thresholds or tighter half-life requirements."
            )

        # Check time stops
        time_stops = (trades_df['exit_reason'] == 'Time stop').sum()
        time_stop_pct = time_stops / len(trades_df) * 100
        if time_stop_pct > 20:
            recommendations.append(
                f"HIGH TIME STOP RATE: {time_stop_pct:.0f}% of trades hit time stop. "
                "The cointegration relationships may be weakening faster than expected."
            )

    if len(vectors) < 10:
        recommendations.append(
            "FEW VECTORS: Only found {len(vectors)} cointegration vectors. Consider relaxing "
            "correlation threshold or half-life bounds to find more tradeable relationships."
        )

    if annual_return < 0.08:
        recommendations.append(
            "LOW RETURNS: Annual return below 8% target. Consider increasing position sizing "
            "or finding additional uncorrelated vectors."
        )

    # Data recommendations
    recommendations.append(
        "DATA REFRESH: Run `python -m src.data.download_all` quarterly to update universe "
        "and price data, aligned with earnings seasons."
    )

    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Run dividend cointegration strategy backtest')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Initial capital')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save-results', type=str, help='Save results to CSV')
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    print("\n" + "="*70)
    print("DIVIDEND COINTEGRATION STRATEGY BACKTEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Load data
    logger.info("Loading local data...")
    loader = DataLoader()

    available = loader.get_available_symbols()
    logger.info(f"Found {len(available)} symbols with price data")

    if len(available) < 20:
        logger.error("Insufficient data. Run `python -m src.data.download_all` first.")
        sys.exit(1)

    # Step 1: Form baskets
    print("\n[1/4] Forming sector-correlation baskets...")
    baskets = form_sector_baskets(
        loader,
        min_stocks_per_sector=6,
        correlation_threshold=0.65,  # Relaxed for more baskets
        min_basket_size=4,
        max_basket_size=6
    )

    if not baskets:
        logger.error("No baskets formed. Check data quality.")
        sys.exit(1)

    # Step 2: Find cointegration vectors
    print("\n[2/4] Finding cointegration vectors...")
    vectors = find_cointegration_vectors(baskets, loader, testing_days=504)

    if not vectors:
        logger.warning("No cointegration vectors found with strict criteria. Relaxing...")
        # Try with relaxed criteria
        vectors = find_cointegration_vectors(baskets, loader, testing_days=378)

    if not vectors:
        logger.error("No cointegration vectors found. Strategy cannot run.")
        sys.exit(1)

    # Step 3: Run backtest
    print("\n[3/4] Running backtest...")
    results = run_simplified_backtest(
        vectors,
        loader,
        initial_capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Step 4: Report results
    print("\n[4/4] Generating report...")
    print_results(results, vectors)

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    recommendations = generate_recommendations(results, vectors)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    # Save results if requested
    if args.save_results:
        equity_df = results.get('equity_curve')
        trades_df = results.get('trades')

        if equity_df is not None:
            equity_df.to_csv(f"{args.save_results}_equity.csv")
        if trades_df is not None and not trades_df.empty:
            trades_df.to_csv(f"{args.save_results}_trades.csv", index=False)

        print(f"\nResults saved to {args.save_results}_*.csv")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
