#!/usr/bin/env python3
"""
Compare strategy results with and without dividend collection.

This script runs the dual momentum strategy twice:
1. Without dividend collection (original behavior)
2. With dividend collection (new behavior)

And compares the results to show the impact of dividend income.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from src.data.loader import DataLoader
from run_backtest_dual_momentum import DualMomentumStrategy


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise for comparison
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_comparison():
    logger = setup_logging()

    print("=" * 70)
    print("DIVIDEND IMPACT ANALYSIS")
    print("=" * 70)
    print("\nLoading data...")

    loader = DataLoader()
    symbols = loader.get_available_symbols()

    # Run WITHOUT dividend collection
    print("\n[1/2] Running strategy WITHOUT dividend collection...")
    strategy_no_div = DualMomentumStrategy(loader)
    strategy_no_div.COLLECT_DIVIDENDS = False
    strategy_no_div.load_data(symbols)
    results_no_div = strategy_no_div.run()

    # Run WITH dividend collection
    print("[2/2] Running strategy WITH dividend collection...")
    strategy_with_div = DualMomentumStrategy(loader)
    strategy_with_div.COLLECT_DIVIDENDS = True
    strategy_with_div.load_data(symbols)
    results_with_div = strategy_with_div.run()

    if not results_no_div or not results_with_div:
        print("Error: Could not run backtests")
        return

    m_no_div = results_no_div['metrics']
    m_with_div = results_with_div['metrics']

    # Calculate differences
    return_diff = (m_with_div['total_return'] - m_no_div['total_return']) * 100
    annual_return_diff = (m_with_div['annual_return'] - m_no_div['annual_return']) * 100
    sharpe_diff = m_with_div['sharpe'] - m_no_div['sharpe']

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\n{:<30} {:>18} {:>18}".format("Metric", "Without Dividends", "With Dividends"))
    print("-" * 70)

    print("{:<30} {:>17.2f}% {:>17.2f}%".format(
        "Total Return",
        m_no_div['total_return'] * 100,
        m_with_div['total_return'] * 100
    ))
    print("{:<30} {:>17.2f}% {:>17.2f}%".format(
        "Annual Return",
        m_no_div['annual_return'] * 100,
        m_with_div['annual_return'] * 100
    ))
    print("{:<30} {:>17.2f}% {:>17.2f}%".format(
        "Annual Volatility",
        m_no_div['annual_vol'] * 100,
        m_with_div['annual_vol'] * 100
    ))
    print("{:<30} {:>18.2f} {:>18.2f}".format(
        "Sharpe Ratio",
        m_no_div['sharpe'],
        m_with_div['sharpe']
    ))
    print("{:<30} {:>18.2f} {:>18.2f}".format(
        "Sortino Ratio",
        m_no_div['sortino'],
        m_with_div['sortino']
    ))
    print("{:<30} {:>17.2f}% {:>17.2f}%".format(
        "Max Drawdown",
        m_no_div['max_drawdown'] * 100,
        m_with_div['max_drawdown'] * 100
    ))

    print("\n" + "-" * 70)
    print("DIVIDEND CONTRIBUTION")
    print("-" * 70)
    print(f"  Total Dividends Collected:   ${m_with_div['total_dividends']:>15,.0f}")
    print(f"  Annual Dividend Income:      ${m_with_div['annual_dividends']:>15,.0f}")
    print(f"  Dividend % of Initial Cap:   {m_with_div['dividend_contribution']*100:>15.2f}%")

    print("\n" + "-" * 70)
    print("IMPACT SUMMARY")
    print("-" * 70)
    print(f"  Additional Total Return:     {return_diff:>+15.2f}%")
    print(f"  Additional Annual Return:    {annual_return_diff:>+15.2f}%")
    print(f"  Sharpe Improvement:          {sharpe_diff:>+15.2f}")

    # Final portfolio values
    final_no_div = results_no_div['equity_curve']['equity'].iloc[-1]
    final_with_div = results_with_div['equity_curve']['equity'].iloc[-1]
    additional_value = final_with_div - final_no_div

    print("\n" + "-" * 70)
    print("FINAL PORTFOLIO VALUE")
    print("-" * 70)
    print(f"  Without Dividends:           ${final_no_div:>15,.0f}")
    print(f"  With Dividends:              ${final_with_div:>15,.0f}")
    print(f"  Additional Value:            ${additional_value:>+15,.0f}")

    print("\n" + "=" * 70)

    return {
        'without_dividends': results_no_div,
        'with_dividends': results_with_div,
        'impact': {
            'return_diff_pct': return_diff,
            'annual_return_diff_pct': annual_return_diff,
            'sharpe_diff': sharpe_diff,
            'additional_value': additional_value,
            'total_dividends': m_with_div['total_dividends']
        }
    }


if __name__ == "__main__":
    run_comparison()
