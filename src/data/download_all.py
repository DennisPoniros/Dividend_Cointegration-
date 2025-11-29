#!/usr/bin/env python3
"""
Data Download Script

Downloads all data needed for the Dividend Cointegration Strategy:
1. Screens universe for qualifying dividend stocks
2. Downloads price history (10 years for robust cointegration analysis)
3. Downloads dividend history (10 years)
4. Downloads fundamental snapshots
5. Downloads reference data (SPY, sector ETFs)

Usage:
    # Full download (universe screening + all data)
    python -m src.data.download_all

    # Only screen universe
    python -m src.data.download_all --universe-only

    # Only download data (assumes universe already exists)
    python -m src.data.download_all --data-only

    # Specify custom symbols
    python -m src.data.download_all --symbols AAPL,MSFT,JNJ

    # Use Alpaca API (requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars)
    python -m src.data.download_all --use-alpaca

Environment Variables:
    ALPACA_API_KEY      - Alpaca API key (optional)
    ALPACA_SECRET_KEY   - Alpaca secret key (optional)
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.universe import UniverseScreener
from src.data.downloader import DataDownloader
from src.data.loader import DataLoader


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data/download.log')
        ]
    )


def screen_universe(args) -> list:
    """Screen for qualifying dividend stocks."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 1: Universe Screening")
    logger.info("=" * 60)

    screener = UniverseScreener(
        market_cap_min=args.market_cap_min,
        min_dividend_years=args.min_dividend_years,
        avg_daily_volume_min=args.min_dollar_volume
    )

    universe_df = screener.screen_universe(
        max_workers=args.workers,
        save=True
    )

    if universe_df.empty:
        logger.error("No stocks passed screening!")
        return []

    logger.info(f"\nUniverse screening complete:")
    logger.info(f"  Total qualifying stocks: {len(universe_df)}")
    logger.info(f"  Sectors represented: {universe_df['sector'].nunique()}")
    logger.info(f"\nTop 10 by market cap:")
    print(universe_df[['symbol', 'name', 'sector', 'market_cap', 'dividend_yield']].head(10).to_string())

    return universe_df['symbol'].tolist()


def download_data(symbols: list, args):
    """Download all data for the given symbols."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 2: Downloading Market Data")
    logger.info("=" * 60)

    downloader = DataDownloader(
        alpaca_api_key=os.getenv('ALPACA_API_KEY'),
        alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY'),
        use_alpaca=args.use_alpaca
    )

    # Download stock data
    logger.info(f"\nDownloading data for {len(symbols)} symbols...")
    results = downloader.download_universe(
        symbols=symbols,
        max_workers=args.workers,
        include_dividends=True,
        include_fundamentals=True,
        delay=args.delay
    )

    # Download reference data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Downloading Reference Data")
    logger.info("=" * 60)
    ref_results = downloader.download_reference_data()

    # Verify data completeness
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Verifying Data Completeness")
    logger.info("=" * 60)
    verification = downloader.verify_data_completeness(min_days=504)

    complete_count = verification['has_min_data'].sum()
    logger.info(f"\nData verification complete:")
    logger.info(f"  Symbols with 504+ days: {complete_count}/{len(verification)}")

    # Save verification results
    verification.to_csv('data/data_verification.csv', index=False)

    return results, verification


def main():
    parser = argparse.ArgumentParser(
        description='Download data for Dividend Cointegration Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--universe-only',
        action='store_true',
        help='Only screen universe, do not download data'
    )
    mode_group.add_argument(
        '--data-only',
        action='store_true',
        help='Only download data (assumes universe already exists)'
    )
    mode_group.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of specific symbols to download'
    )

    # Universe screening parameters
    parser.add_argument(
        '--market-cap-min',
        type=float,
        default=5_000_000_000,
        help='Minimum market cap in dollars (default: 5B)'
    )
    parser.add_argument(
        '--min-dividend-years',
        type=int,
        default=5,
        help='Minimum consecutive years of dividend payments (default: 5)'
    )
    parser.add_argument(
        '--min-dollar-volume',
        type=float,
        default=10_000_000,
        help='Minimum average daily dollar volume (default: 10M)'
    )

    # Download parameters
    parser.add_argument(
        '--use-alpaca',
        action='store_true',
        help='Use Alpaca API for price data (requires env vars)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.3,
        help='Delay between API requests in seconds (default: 0.3)'
    )

    # Output options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Dividend Cointegration Strategy - Data Download")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Determine symbols to download
    symbols = []

    if args.symbols:
        # Use provided symbols
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        logger.info(f"Using {len(symbols)} provided symbols")

    elif args.data_only:
        # Load existing universe
        try:
            loader = DataLoader()
            symbols = loader.get_universe_symbols()
            logger.info(f"Loaded {len(symbols)} symbols from existing universe")
        except FileNotFoundError:
            logger.error("Universe file not found. Run without --data-only first.")
            sys.exit(1)

    else:
        # Screen universe
        symbols = screen_universe(args)

    if args.universe_only:
        logger.info("\n--universe-only specified. Skipping data download.")
        logger.info(f"\nUniverse saved to: data/universe/universe.parquet")
        return

    if not symbols:
        logger.error("No symbols to download!")
        sys.exit(1)

    # Download data
    results, verification = download_data(symbols, args)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)

    complete_symbols = verification[verification['has_min_data']]['symbol'].tolist()

    logger.info(f"\nSummary:")
    logger.info(f"  Total symbols requested: {len(symbols)}")
    logger.info(f"  Symbols with complete data (504+ days): {len(complete_symbols)}")
    logger.info(f"\nData saved to:")
    logger.info(f"  Prices:       data/prices/")
    logger.info(f"  Dividends:    data/dividends/")
    logger.info(f"  Fundamentals: data/fundamentals/")
    logger.info(f"  Reference:    data/reference/")
    logger.info(f"  Universe:     data/universe/")
    logger.info(f"\nVerification:   data/data_verification.csv")
    logger.info(f"Download log:   data/download_log.csv")

    # Save list of complete symbols for easy access
    with open('data/complete_symbols.txt', 'w') as f:
        f.write('\n'.join(complete_symbols))
    logger.info(f"\nComplete symbols list saved to: data/complete_symbols.txt")

    logger.info(f"\nFinished at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
