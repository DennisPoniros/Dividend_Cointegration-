#!/usr/bin/env python3
"""Debug script to understand why cointegration vectors aren't being found."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).parent))
from src.data.loader import DataLoader

def calculate_half_life(spread):
    """Calculate half-life of mean reversion."""
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

def test_pair_cointegration(prices_df, combo):
    """Test a specific combination for cointegration."""
    data = prices_df[list(combo)].dropna()
    if len(data) < 252:
        return None

    results = {'combo': combo}

    # Johansen test
    try:
        joh = coint_johansen(data, 0, 2)
        trace_stat = joh.lr1[0]
        cv_5pct = joh.cvt[0, 1]
        cv_1pct = joh.cvt[0, 0]
        eigenvalue = joh.eig[0]
        eigenvector = joh.evec[:, 0]

        results['trace_stat'] = trace_stat
        results['cv_5pct'] = cv_5pct
        results['cv_1pct'] = cv_1pct
        results['eigenvalue'] = eigenvalue
        results['passes_5pct'] = trace_stat > cv_5pct
        results['passes_1pct'] = trace_stat > cv_1pct

        # Calculate spread
        weights = eigenvector / np.sum(np.abs(eigenvector))
        spread = (data * weights).sum(axis=1)

        # Engle-Granger
        adf = adfuller(spread, maxlag=None, regression='c', autolag='AIC')
        results['adf_stat'] = adf[0]
        results['adf_pvalue'] = adf[1]
        results['passes_adf'] = adf[0] < -3.5

        # Half-life
        hl = calculate_half_life(spread)
        results['half_life'] = hl
        results['passes_hl'] = hl is not None and 5 <= hl <= 35

        return results
    except Exception as e:
        return {'combo': combo, 'error': str(e)}

def main():
    loader = DataLoader()
    symbols = loader.get_available_symbols()

    print(f"Total symbols: {len(symbols)}")

    # Load fundamentals to get sectors
    fund_df = loader.load_fundamentals_df(symbols)
    print(f"\nSectors: {fund_df['sector'].value_counts().to_dict()}")

    # Pick a sector with enough stocks
    sector = 'Financial Services'
    sector_symbols = fund_df[fund_df['sector'] == sector]['symbol'].tolist()
    print(f"\n{sector}: {len(sector_symbols)} stocks")

    # Load prices
    prices = loader.load_prices_multi(sector_symbols[:12], align=True)
    print(f"Aligned price data: {len(prices)} days, {len(prices.columns)} stocks")

    if len(prices) < 252:
        print("Not enough data!")
        return

    # Test all 4-stock combinations from first 8 stocks
    test_symbols = prices.columns[:8].tolist()
    print(f"\nTesting combinations of: {test_symbols}")

    results = []
    for combo in combinations(test_symbols, 4):
        result = test_pair_cointegration(prices, combo)
        if result:
            results.append(result)

    # Summarize
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("COINTEGRATION TEST RESULTS SUMMARY")
    print("="*80)

    if 'passes_5pct' in df.columns:
        print(f"\nJohansen (5%): {df['passes_5pct'].sum()}/{len(df)} pass")
        print(f"Johansen (1%): {df['passes_1pct'].sum()}/{len(df)} pass")
        print(f"Engle-Granger (ADF < -3.5): {df['passes_adf'].sum()}/{len(df)} pass")
        print(f"Half-life (5-35 days): {df['passes_hl'].sum()}/{len(df)} pass")

        # Show best candidates
        print("\n" + "-"*80)
        print("TOP 10 BY TRACE STATISTIC:")
        print("-"*80)
        top = df.nlargest(10, 'trace_stat')
        for _, row in top.iterrows():
            print(f"  {row['combo']}")
            print(f"    Trace: {row['trace_stat']:.2f} (5%CV: {row['cv_5pct']:.2f}, 1%CV: {row['cv_1pct']:.2f})")
            print(f"    Eigenvalue: {row['eigenvalue']:.4f}")
            print(f"    ADF: {row['adf_stat']:.2f} (p={row['adf_pvalue']:.4f})")
            print(f"    Half-life: {row['half_life']:.1f} days" if row['half_life'] else "    Half-life: N/A")
            print()

        # Find any that pass relaxed criteria
        relaxed = df[(df['passes_5pct']) & (df['adf_stat'] < -2.5)]
        print(f"\nRelaxed criteria (5% Johansen + ADF < -2.5): {len(relaxed)} pass")

        if len(relaxed) > 0:
            print("\nRelaxed candidates:")
            for _, row in relaxed.head(5).iterrows():
                print(f"  {row['combo']}: trace={row['trace_stat']:.2f}, adf={row['adf_stat']:.2f}, hl={row['half_life']:.1f if row['half_life'] else 'N/A'}")

if __name__ == "__main__":
    main()
