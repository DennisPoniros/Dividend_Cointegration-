#!/usr/bin/env python3
"""
Compare strategy results with different dividend handling modes.

This script runs the dual momentum strategy three ways:
1. Without dividend collection (original behavior)
2. With dividend reinvestment (dividends go into trading cash)
3. With dividends held as cash earning risk-free rate

And compares the results to show the impact of each approach.
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

    print("=" * 80)
    print("DIVIDEND IMPACT ANALYSIS")
    print("=" * 80)
    print("\nLoading data...")

    loader = DataLoader()
    symbols = loader.get_available_symbols()

    # Run WITHOUT dividend collection
    print("\n[1/3] Running strategy WITHOUT dividend collection...")
    strategy_no_div = DualMomentumStrategy(loader)
    strategy_no_div.COLLECT_DIVIDENDS = False
    strategy_no_div.load_data(symbols)
    results_no_div = strategy_no_div.run()

    # Run WITH dividend reinvestment
    print("[2/3] Running strategy WITH dividend REINVESTMENT...")
    strategy_reinvest = DualMomentumStrategy(loader)
    strategy_reinvest.COLLECT_DIVIDENDS = True
    strategy_reinvest.DIVIDEND_MODE = 'reinvest'
    strategy_reinvest.load_data(symbols)
    results_reinvest = strategy_reinvest.run()

    # Run WITH dividends as cash earning risk-free rate
    print("[3/3] Running strategy WITH dividends as CASH (earning risk-free)...")
    strategy_cash = DualMomentumStrategy(loader)
    strategy_cash.COLLECT_DIVIDENDS = True
    strategy_cash.DIVIDEND_MODE = 'cash'
    strategy_cash.load_data(symbols)
    results_cash = strategy_cash.run()

    if not results_no_div or not results_reinvest or not results_cash:
        print("Error: Could not run backtests")
        return

    m_no_div = results_no_div['metrics']
    m_reinvest = results_reinvest['metrics']
    m_cash = results_cash['metrics']

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print("\n{:<25} {:>18} {:>18} {:>18}".format(
        "Metric", "No Dividends", "Reinvested", "Cash @ Rf"))
    print("-" * 80)

    print("{:<25} {:>17.2f}% {:>17.2f}% {:>17.2f}%".format(
        "Total Return",
        m_no_div['total_return'] * 100,
        m_reinvest['total_return'] * 100,
        m_cash['total_return'] * 100
    ))
    print("{:<25} {:>17.2f}% {:>17.2f}% {:>17.2f}%".format(
        "Annual Return",
        m_no_div['annual_return'] * 100,
        m_reinvest['annual_return'] * 100,
        m_cash['annual_return'] * 100
    ))
    print("{:<25} {:>17.2f}% {:>17.2f}% {:>17.2f}%".format(
        "Annual Volatility",
        m_no_div['annual_vol'] * 100,
        m_reinvest['annual_vol'] * 100,
        m_cash['annual_vol'] * 100
    ))
    print("{:<25} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Sharpe Ratio",
        m_no_div['sharpe'],
        m_reinvest['sharpe'],
        m_cash['sharpe']
    ))
    print("{:<25} {:>18.2f} {:>18.2f} {:>18.2f}".format(
        "Sortino Ratio",
        m_no_div['sortino'],
        m_reinvest['sortino'],
        m_cash['sortino']
    ))
    print("{:<25} {:>17.2f}% {:>17.2f}% {:>17.2f}%".format(
        "Max Drawdown",
        m_no_div['max_drawdown'] * 100,
        m_reinvest['max_drawdown'] * 100,
        m_cash['max_drawdown'] * 100
    ))

    print("\n" + "-" * 80)
    print("DIVIDEND & COST DETAILS")
    print("-" * 80)
    print(f"  Total Dividends Collected (Reinvest):  ${m_reinvest['total_dividends']:>15,.0f}")
    print(f"  Total Dividends Collected (Cash):      ${m_cash['total_dividends']:>15,.0f}")
    print(f"  Dividend Cash Account (with interest): ${m_cash['dividend_cash_account']:>15,.0f}")
    print(f"  Interest Earned on Div Cash:           ${m_cash['dividend_cash_account'] - m_cash['total_dividends']:>15,.0f}")
    print(f"  Total Borrowing Cost (all scenarios):  ${m_reinvest['total_borrowing_cost']:>15,.0f}")

    print("\n" + "-" * 80)
    print("IMPACT SUMMARY (vs No Dividends)")
    print("-" * 80)

    # Calculate differences
    reinvest_return_diff = (m_reinvest['total_return'] - m_no_div['total_return']) * 100
    cash_return_diff = (m_cash['total_return'] - m_no_div['total_return']) * 100
    reinvest_sharpe_diff = m_reinvest['sharpe'] - m_no_div['sharpe']
    cash_sharpe_diff = m_cash['sharpe'] - m_no_div['sharpe']

    print(f"  Reinvest - Additional Total Return:   {reinvest_return_diff:>+15.2f}%")
    print(f"  Cash@Rf  - Additional Total Return:   {cash_return_diff:>+15.2f}%")
    print(f"  Reinvest - Sharpe Improvement:        {reinvest_sharpe_diff:>+15.2f}")
    print(f"  Cash@Rf  - Sharpe Improvement:        {cash_sharpe_diff:>+15.2f}")

    print("\n" + "-" * 80)
    print("REINVEST vs CASH COMPARISON")
    print("-" * 80)

    reinvest_vs_cash = (m_reinvest['total_return'] - m_cash['total_return']) * 100
    print(f"  Reinvest outperforms Cash@Rf by:      {reinvest_vs_cash:>+15.2f}%")

    # Final portfolio values
    final_no_div = results_no_div['equity_curve']['equity'].iloc[-1]
    final_reinvest = results_reinvest['equity_curve']['equity'].iloc[-1]
    final_cash = results_cash['equity_curve']['equity'].iloc[-1]

    print("\n" + "-" * 80)
    print("FINAL PORTFOLIO VALUE")
    print("-" * 80)
    print(f"  Without Dividends:                    ${final_no_div:>15,.0f}")
    print(f"  With Reinvestment:                    ${final_reinvest:>15,.0f}")
    print(f"  With Cash @ Risk-Free:                ${final_cash:>15,.0f}")
    print(f"  Reinvest Advantage over No Div:       ${final_reinvest - final_no_div:>+15,.0f}")
    print(f"  Reinvest Advantage over Cash@Rf:      ${final_reinvest - final_cash:>+15,.0f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
  - 'No Dividends': Baseline - dividends are ignored entirely
  - 'Reinvested': Dividends add to trading cash, get deployed in leveraged strategy
  - 'Cash @ Rf': Dividends held in separate account earning risk-free rate

  The difference between Reinvest and Cash@Rf shows the value of reinvesting
  dividends into the leveraged momentum strategy vs. just holding them safely.
""")

    print("=" * 80)

    return {
        'no_dividends': results_no_div,
        'reinvested': results_reinvest,
        'cash_rf': results_cash,
    }


if __name__ == "__main__":
    run_comparison()
