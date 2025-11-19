"""
Example Workflow Script

Demonstrates the complete strategy workflow from universe construction
through backtesting. This is a reference implementation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_workflow():
    """
    Complete example workflow for the dividend cointegration strategy.

    Note: This requires IBKR connection and market data subscriptions.
    For a working example, you'll need to provide actual stock symbols.
    """

    from src.data.ibkr_connector import IBKRConnector
    from src.data.universe import UniverseBuilder
    from src.strategy.basket_formation import BasketFormer
    from src.strategy.cointegration import CointegrationTester
    from src.strategy.signals import SignalGenerator
    from src.strategy.portfolio import PortfolioManager
    from src.risk.risk_management import RiskManager
    from src.backtest.engine import BacktestEngine
    from src.backtest.metrics import PerformanceMetrics

    logger.info("="*60)
    logger.info("DIVIDEND COINTEGRATION STRATEGY - EXAMPLE WORKFLOW")
    logger.info("="*60)

    # ========================================================================
    # STEP 1: CONNECT TO IBKR
    # ========================================================================

    logger.info("\nSTEP 1: Connecting to IBKR...")

    try:
        ibkr = IBKRConnector(config_path="config/ibkr_config.yaml")

        if not ibkr.connect():
            logger.error("Failed to connect to IBKR. Please check your configuration.")
            logger.info("Tip: Make sure TWS or IB Gateway is running and API is enabled.")
            return

        logger.info("✓ Connected to IBKR successfully")

    except Exception as e:
        logger.error(f"IBKR connection error: {e}")
        logger.info("Note: You can still explore the code structure without IBKR connection.")
        logger.info("Consider loading pre-existing CSVs for testing.")
        return

    # ========================================================================
    # STEP 2: BUILD UNIVERSE
    # ========================================================================

    logger.info("\nSTEP 2: Building investment universe...")

    # NOTE: In production, you would provide a comprehensive list of US stocks
    # For example, all constituents of S&P 1500, Russell 3000, etc.
    # Here we use a small sample for demonstration

    candidate_symbols = [
        # Large cap dividend stocks (example - replace with full universe)
        'AAPL', 'MSFT', 'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'VZ',
        'T', 'XOM', 'CVX', 'BAC', 'JPM', 'WFC', 'USB', 'PNC', 'C', 'GS',
        'MMM', 'CAT', 'DE', 'HON', 'UPS', 'FDX', 'RTX', 'LMT', 'BA', 'GD'
    ]

    logger.info(f"Screening {len(candidate_symbols)} candidate stocks...")

    universe_builder = UniverseBuilder()

    # Option 1: Build from scratch (requires IBKR data)
    # universe_df = universe_builder.build_universe(ibkr, candidate_symbols)

    # Option 2: Load from CSV (for testing)
    logger.info("Tip: For faster testing, you can load a pre-built universe CSV")
    logger.info("Example: universe_df = universe_builder.load_universe('data/universe.csv')")

    # For this example, we'll create a mock universe
    logger.warning("Using mock universe for demonstration. Replace with actual data!")

    universe_df = pd.DataFrame({
        'symbol': candidate_symbols[:20],  # Use first 20 for demo
        'market_cap': np.random.uniform(10e9, 500e9, 20),
        'screen_date': datetime.now()
    })

    logger.info(f"✓ Universe constructed: {len(universe_df)} stocks qualified")

    # ========================================================================
    # STEP 3: GET PRICE DATA
    # ========================================================================

    logger.info("\nSTEP 3: Fetching historical price data...")

    universe_symbols = universe_df['symbol'].tolist()

    # Fetch 2 years of daily data
    logger.info(f"Fetching 2 years of data for {len(universe_symbols)} stocks...")
    logger.info("This may take several minutes...")

    # Option 1: Fetch from IBKR
    # price_data = ibkr.get_multiple_historical_data(
    #     universe_symbols,
    #     duration="2 Y",
    #     bar_size="1 day"
    # )

    # Option 2: Load from saved data
    logger.info("Tip: Save and reuse price data to avoid repeated API calls")
    logger.warning("Using mock price data for demonstration. Replace with actual data!")

    # Mock price data
    price_data = {}
    dates = pd.date_range(end=datetime.now(), periods=504, freq='D')

    for symbol in universe_symbols:
        price_data[symbol] = pd.DataFrame({
            'open': np.random.uniform(100, 200, 504),
            'high': np.random.uniform(100, 200, 504),
            'low': np.random.uniform(100, 200, 504),
            'close': np.random.uniform(100, 200, 504),
            'volume': np.random.uniform(1e6, 10e6, 504)
        }, index=dates)

    logger.info(f"✓ Price data fetched for {len(price_data)} stocks")

    # ========================================================================
    # STEP 4: FORM BASKETS
    # ========================================================================

    logger.info("\nSTEP 4: Forming correlation-based baskets...")

    basket_former = BasketFormer()

    baskets = basket_former.form_baskets(
        universe_symbols,
        price_data,
        ibkr
    )

    logger.info(f"✓ Formed {len(baskets)} baskets")

    for i, basket in enumerate(baskets[:3]):  # Show first 3
        logger.info(f"  Basket {i}: {basket['sector']}, "
                   f"{basket['size']} stocks, "
                   f"avg corr: {basket['avg_correlation']:.3f}")

    # ========================================================================
    # STEP 5: TEST COINTEGRATION
    # ========================================================================

    logger.info("\nSTEP 5: Testing for cointegration...")

    coint_tester = CointegrationTester()

    vectors = coint_tester.test_all_baskets(baskets, price_data)

    logger.info(f"✓ Found {len(vectors)} valid cointegration vectors")

    for i, vector in enumerate(vectors[:3]):  # Show first 3
        logger.info(f"  Vector {i}: {len(vector['symbols'])} stocks, "
                   f"half-life: {vector['half_life']:.2f} days, "
                   f"Hurst: {vector['hurst']:.3f}")

    if len(vectors) == 0:
        logger.warning("No valid cointegration vectors found!")
        logger.info("Tip: Try adjusting parameters in config/strategy_params.yaml")
        logger.info("     - Increase correlation threshold")
        logger.info("     - Relax half-life bounds")
        logger.info("     - Use longer testing period")
        return

    # ========================================================================
    # STEP 6: RUN BACKTEST
    # ========================================================================

    logger.info("\nSTEP 6: Running backtest...")

    backtest = BacktestEngine()

    initial_capital = 1_000_000
    logger.info(f"Initial capital: ${initial_capital:,}")

    results = backtest.run_backtest(
        vectors,
        price_data,
        initial_capital=initial_capital
    )

    logger.info("✓ Backtest complete")

    # ========================================================================
    # STEP 7: ANALYZE RESULTS
    # ========================================================================

    logger.info("\nSTEP 7: Performance Analysis")
    logger.info("="*60)

    metrics = results['metrics']

    # Print performance summary
    PerformanceMetrics.print_summary(metrics)

    # Additional insights
    logger.info("\nTrade Statistics:")
    if not results['trades'].empty:
        trades_df = results['trades']
        completed_trades = trades_df[trades_df['type'] == 'CLOSE_POSITION']

        if not completed_trades.empty:
            logger.info(f"  Total trades: {len(completed_trades)}")
            logger.info(f"  Avg hold time: {completed_trades['hold_days'].mean():.1f} days")
            logger.info(f"  Best trade: ${completed_trades['pnl'].max():,.2f}")
            logger.info(f"  Worst trade: ${completed_trades['pnl'].min():,.2f}")

    # ========================================================================
    # STEP 8: SAVE RESULTS
    # ========================================================================

    logger.info("\nSTEP 8: Saving results...")

    # Save equity curve
    results['equity_curve'].to_csv('backtest_equity_curve.csv')
    logger.info("✓ Saved equity curve to: backtest_equity_curve.csv")

    # Save trades
    if not results['trades'].empty:
        results['trades'].to_csv('backtest_trades.csv', index=False)
        logger.info("✓ Saved trades to: backtest_trades.csv")

    # Save vectors
    coint_tester.save_vectors('cointegration_vectors.csv')
    logger.info("✓ Saved vectors to: cointegration_vectors.csv")

    # Save baskets
    basket_former.save_baskets('baskets.csv')
    logger.info("✓ Saved baskets to: baskets.csv")

    # ========================================================================
    # CLEANUP
    # ========================================================================

    logger.info("\nCleaning up...")
    ibkr.disconnect()
    logger.info("✓ Disconnected from IBKR")

    logger.info("\n" + "="*60)
    logger.info("WORKFLOW COMPLETE")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Review the saved CSV files")
    logger.info("2. Load results in the Streamlit dashboard for visualization")
    logger.info("3. Adjust parameters in config/strategy_params.yaml")
    logger.info("4. Run again with different settings")
    logger.info("5. When satisfied, proceed to paper trading")
    logger.info("\nTo view results in dashboard:")
    logger.info("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    try:
        example_workflow()
    except KeyboardInterrupt:
        logger.info("\nWorkflow interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
