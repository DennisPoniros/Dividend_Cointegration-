"""
Performance Metrics Module

Calculates comprehensive performance statistics:
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Profit factor
- etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculates trading performance metrics."""

    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve."""
        return equity_curve.pct_change().fillna(0)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (252 for daily)

        Returns:
            Sharpe ratio
        """
        if returns.std() == 0:
            return 0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

        return sharpe

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sortino ratio.

        Uses downside deviation instead of total volatility.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)

        # Calculate downside deviation
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        downside_std = downside_returns.std()
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

        return sortino

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Equity curve series

        Returns:
            Dictionary with max drawdown, start date, end date, duration
        """
        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Find maximum drawdown
        max_dd = drawdown.min()

        # Find the date of maximum drawdown
        max_dd_date = drawdown.idxmin()

        # Find the peak before the drawdown
        peak_date = equity_curve[:max_dd_date].idxmax()

        # Find recovery date (if recovered)
        recovery_date = None
        if max_dd_date < equity_curve.index[-1]:
            peak_value = equity_curve[peak_date]
            post_dd = equity_curve[max_dd_date:]
            recovered = post_dd[post_dd >= peak_value]
            if len(recovered) > 0:
                recovery_date = recovered.index[0]

        # Calculate duration
        if recovery_date:
            duration = (recovery_date - peak_date).days
        else:
            duration = (equity_curve.index[-1] - peak_date).days

        return {
            'max_drawdown': max_dd,
            'peak_date': peak_date,
            'valley_date': max_dd_date,
            'recovery_date': recovery_date,
            'duration_days': duration
        }

    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> Dict:
        """
        Calculate win rate and related statistics.

        Args:
            trades: DataFrame with trade results (must have 'pnl' column)

        Returns:
            Dictionary with win rate statistics
        """
        if trades.empty or 'pnl' not in trades.columns:
            return {
                'win_rate': 0,
                'num_winners': 0,
                'num_losers': 0,
                'avg_win': 0,
                'avg_loss': 0
            }

        winners = trades[trades['pnl'] > 0]
        losers = trades[trades['pnl'] < 0]

        num_winners = len(winners)
        num_losers = len(losers)
        total_trades = len(trades)

        win_rate = num_winners / total_trades if total_trades > 0 else 0
        avg_win = winners['pnl'].mean() if num_winners > 0 else 0
        avg_loss = abs(losers['pnl'].mean()) if num_losers > 0 else 0

        return {
            'win_rate': win_rate,
            'num_winners': num_winners,
            'num_losers': num_losers,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades
        }

    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades: DataFrame with trade results

        Returns:
            Profit factor
        """
        if trades.empty or 'pnl' not in trades.columns:
            return 0

        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0

        return gross_profit / gross_loss

    @staticmethod
    def calculate_calmar_ratio(
        returns: pd.Series,
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).

        Args:
            returns: Series of returns
            equity_curve: Equity curve
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * periods_per_year
        max_dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        max_dd = abs(max_dd_info['max_drawdown'])

        if max_dd == 0:
            return 0

        return annual_return / max_dd

    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> Dict:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: Equity curve series
            trades: Optional DataFrame with individual trades
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Dictionary with all metrics
        """
        returns = PerformanceMetrics.calculate_returns(equity_curve)

        metrics = {
            # Return metrics
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
            'annual_return': returns.mean() * periods_per_year,
            'annual_volatility': returns.std() * np.sqrt(periods_per_year),

            # Risk-adjusted metrics
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(
                returns, equity_curve, periods_per_year
            ),

            # Drawdown metrics
            **PerformanceMetrics.calculate_max_drawdown(equity_curve),

            # Additional stats
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
        }

        # Trade-specific metrics
        if trades is not None and not trades.empty:
            win_rate_stats = PerformanceMetrics.calculate_win_rate(trades)
            metrics.update(win_rate_stats)
            metrics['profit_factor'] = PerformanceMetrics.calculate_profit_factor(trades)

        return metrics

    @staticmethod
    def print_summary(metrics: Dict):
        """
        Print formatted performance summary.

        Args:
            metrics: Dictionary of performance metrics
        """
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        print("\nReturn Metrics:")
        print(f"  Total Return:        {metrics['total_return']*100:>10.2f}%")
        print(f"  Annual Return:       {metrics['annual_return']*100:>10.2f}%")
        print(f"  Annual Volatility:   {metrics['annual_volatility']*100:>10.2f}%")

        print("\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.3f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.3f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.3f}")

        print("\nDrawdown Metrics:")
        print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>10.2f}%")
        print(f"  Peak Date:           {metrics['peak_date']}")
        print(f"  Valley Date:         {metrics['valley_date']}")
        print(f"  Recovery Date:       {metrics['recovery_date']}")
        print(f"  Duration:            {metrics['duration_days']:>10} days")

        if 'win_rate' in metrics:
            print("\nTrade Statistics:")
            print(f"  Total Trades:        {metrics['total_trades']:>10}")
            print(f"  Win Rate:            {metrics['win_rate']*100:>10.2f}%")
            print(f"  Avg Win:             ${metrics['avg_win']:>10,.2f}")
            print(f"  Avg Loss:            ${metrics['avg_loss']:>10,.2f}")
            print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")

        print("\n" + "="*60 + "\n")
