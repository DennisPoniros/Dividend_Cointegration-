#!/usr/bin/env python3
"""
Extended Analysis for Dual Momentum Strategy

Additional plots and reports:
1. Yearly performance breakdown
2. Sector exposure over time
3. Position weight distribution (inverse yield impact)
4. Regime analysis (bull/bear markets)
5. Seasonal patterns
6. Holding period vs P&L analysis
7. Concentration risk analysis
8. Win/loss streak analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from src.data.loader import DataLoader

OUTPUT_DIR = Path('/home/user/Dividend_Cointegration-/analysis_output')

# Sector mapping for universe stocks
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'CSCO': 'Technology', 'INTC': 'Technology',
    'ORCL': 'Technology', 'IBM': 'Technology', 'AVGO': 'Technology', 'TXN': 'Technology',
    'QCOM': 'Technology', 'ADP': 'Technology',
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'LLY': 'Healthcare', 'PFE': 'Healthcare',
    'MRK': 'Healthcare', 'AMGN': 'Healthcare', 'BMY': 'Healthcare', 'GILD': 'Healthcare',
    'MDT': 'Healthcare', 'ABT': 'Healthcare', 'ABBV': 'Healthcare', 'BDX': 'Healthcare',
    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'BLK': 'Financials', 'C': 'Financials', 'SCHW': 'Financials',
    'USB': 'Financials', 'PNC': 'Financials', 'MTB': 'Financials', 'TFC': 'Financials',
    'RF': 'Financials', 'FITB': 'Financials', 'HBAN': 'Financials', 'CFG': 'Financials',
    'ZION': 'Financials', 'KEY': 'Financials', 'AIG': 'Financials', 'MET': 'Financials',
    'PRU': 'Financials', 'ALL': 'Financials', 'CB': 'Financials', 'TRV': 'Financials',
    'AFL': 'Financials', 'CINF': 'Financials', 'BEN': 'Financials', 'TROW': 'Financials',
    'SPGI': 'Financials', 'MMC': 'Financials', 'AON': 'Financials', 'AJG': 'Financials',
    'BRO': 'Financials',
    # Industrials
    'CAT': 'Industrials', 'HON': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials',
    'MMM': 'Industrials', 'GE': 'Industrials', 'UNP': 'Industrials', 'UPS': 'Industrials',
    'ITW': 'Industrials', 'DOV': 'Industrials', 'FAST': 'Industrials', 'EMR': 'Industrials',
    'ROP': 'Industrials', 'GD': 'Industrials', 'DE': 'Industrials', 'CTAS': 'Industrials',
    'PNR': 'Industrials', 'SWK': 'Industrials', 'EXPD': 'Industrials', 'AOS': 'Industrials',
    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'CLX': 'Consumer Staples', 'KMB': 'Consumer Staples', 'SJM': 'Consumer Staples',
    'HRL': 'Consumer Staples', 'MKC': 'Consumer Staples', 'CL': 'Consumer Staples',
    'SYY': 'Consumer Staples', 'GPC': 'Consumer Staples', 'MO': 'Consumer Staples',
    'PM': 'Consumer Staples', 'COST': 'Consumer Staples', 'WMT': 'Consumer Staples',
    # Consumer Discretionary
    'HD': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy', 'OXY': 'Energy',
    'SLB': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'MPC': 'Energy',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',
    'ED': 'Utilities', 'WEC': 'Utilities', 'ATO': 'Utilities',
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials', 'NEM': 'Materials',
    'FCX': 'Materials', 'NUE': 'Materials', 'DD': 'Materials', 'ALB': 'Materials',
    'PPG': 'Materials', 'SHW': 'Materials', 'AMCR': 'Materials',
    # Communication Services
    'VZ': 'Communication', 'T': 'Communication', 'CMCSA': 'Communication',
    # Real Estate / Other
    'WST': 'Healthcare',
}


def load_results():
    """Load backtest results from CSV files."""
    trades_df = pd.read_csv(OUTPUT_DIR / 'trades.csv', parse_dates=['entry_date', 'exit_date'])
    equity_df = pd.read_csv(OUTPUT_DIR / 'daily_equity.csv', parse_dates=['date'], index_col='date')
    rebalance_df = pd.read_csv(OUTPUT_DIR / 'rebalance_events.csv', parse_dates=['date'])
    return trades_df, equity_df, rebalance_df


def plot_yearly_performance(equity_df, trades_df):
    """Create yearly performance breakdown table and plot."""
    print("\nGenerating yearly performance analysis...")

    # Calculate yearly metrics
    equity_df = equity_df.copy()
    equity_df['year'] = equity_df.index.year
    equity_df['returns'] = equity_df['equity'].pct_change()

    yearly_stats = []
    years = sorted(equity_df['year'].unique())

    for year in years:
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) < 20:
            continue

        year_trades = trades_df[trades_df['exit_date'].dt.year == year]

        returns = year_data['returns'].dropna()
        start_equity = year_data['equity'].iloc[0]
        end_equity = year_data['equity'].iloc[-1]

        # Calculate metrics
        total_return = (end_equity / start_equity) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252 - 0.02) / annual_vol if annual_vol > 0 else 0

        # Max drawdown for year
        rolling_max = year_data['equity'].expanding().max()
        drawdown = (year_data['equity'] / rolling_max) - 1
        max_dd = drawdown.min()

        # Trade stats
        num_trades = len(year_trades)
        win_rate = (year_trades['pnl_net'] > 0).mean() * 100 if num_trades > 0 else 0

        yearly_stats.append({
            'Year': year,
            'Return': total_return * 100,
            'Volatility': annual_vol * 100,
            'Sharpe': sharpe,
            'Max DD': max_dd * 100,
            'Trades': num_trades,
            'Win Rate': win_rate
        })

    yearly_df = pd.DataFrame(yearly_stats)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Annual returns bar chart
    ax1 = axes[0, 0]
    colors = ['green' if r > 0 else 'red' for r in yearly_df['Return']]
    ax1.bar(yearly_df['Year'].astype(str), yearly_df['Return'], color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Annual Returns')
    for i, (year, ret) in enumerate(zip(yearly_df['Year'], yearly_df['Return'])):
        ax1.annotate(f'{ret:.1f}%', (i, ret), ha='center',
                    va='bottom' if ret > 0 else 'top', fontsize=9)

    # 2. Sharpe ratio by year
    ax2 = axes[0, 1]
    ax2.bar(yearly_df['Year'].astype(str), yearly_df['Sharpe'], color='steelblue', alpha=0.7)
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='Sharpe = 1.0')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Annual Sharpe Ratio')
    ax2.legend()

    # 3. Max drawdown by year
    ax3 = axes[1, 0]
    ax3.bar(yearly_df['Year'].astype(str), yearly_df['Max DD'], color='darkred', alpha=0.7)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Annual Maximum Drawdown')

    # 4. Win rate by year
    ax4 = axes[1, 1]
    ax4.bar(yearly_df['Year'].astype(str), yearly_df['Win Rate'], color='teal', alpha=0.7)
    ax4.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Annual Win Rate')
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'yearly_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: yearly_performance.png")

    # Print table
    print("\n" + "=" * 80)
    print("YEARLY PERFORMANCE BREAKDOWN")
    print("=" * 80)
    print(yearly_df.to_string(index=False, float_format=lambda x: f'{x:.1f}'))

    return yearly_df


def plot_sector_exposure(rebalance_df):
    """Plot sector exposure over time."""
    print("\nGenerating sector exposure analysis...")

    sector_data = []

    for _, row in rebalance_df.iterrows():
        date = row['date']
        holdings = eval(row['holdings']) if isinstance(row['holdings'], str) else row['holdings']

        sector_counts = {}
        for symbol in holdings:
            sector = SECTOR_MAP.get(symbol, 'Other')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Normalize to percentages
        total = sum(sector_counts.values())
        for sector, count in sector_counts.items():
            sector_data.append({
                'date': date,
                'sector': sector,
                'weight': count / total * 100 if total > 0 else 0
            })

    sector_df = pd.DataFrame(sector_data)
    pivot_df = sector_df.pivot_table(index='date', columns='sector', values='weight', fill_value=0)

    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab20.colors
    pivot_df.plot.area(ax=ax, stacked=True, alpha=0.8, color=colors[:len(pivot_df.columns)])

    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Weight (%)')
    ax.set_title('Sector Exposure Over Time')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sector_exposure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sector_exposure.png")

    # Print average sector weights
    avg_weights = pivot_df.mean().sort_values(ascending=False)
    print("\nAverage Sector Weights:")
    for sector, weight in avg_weights.items():
        print(f"  {sector:<20} {weight:>6.1f}%")

    return pivot_df


def plot_weight_distribution(trades_df, loader):
    """Analyze impact of inverse yield weighting on position sizes."""
    print("\nGenerating weight distribution analysis...")

    # Load dividend data to calculate yields
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of trade sizes (notional)
    ax1 = axes[0, 0]
    trade_sizes = trades_df['notional'] / 1e6  # In millions
    ax1.hist(trade_sizes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(trade_sizes.mean(), color='red', linestyle='--', label=f'Mean: ${trade_sizes.mean():.2f}M')
    ax1.axvline(trade_sizes.median(), color='green', linestyle='--', label=f'Median: ${trade_sizes.median():.2f}M')
    ax1.set_xlabel('Position Size ($M)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Position Sizes')
    ax1.legend()

    # 2. Position size by symbol (average)
    ax2 = axes[0, 1]
    avg_by_symbol = trades_df.groupby('symbol')['notional'].mean().sort_values(ascending=False).head(20) / 1e6
    ax2.barh(range(len(avg_by_symbol)), avg_by_symbol.values, color='teal', alpha=0.7)
    ax2.set_yticks(range(len(avg_by_symbol)))
    ax2.set_yticklabels(avg_by_symbol.index)
    ax2.set_xlabel('Average Position Size ($M)')
    ax2.set_title('Top 20 Stocks by Average Position Size')
    ax2.invert_yaxis()

    # 3. Position size over time
    ax3 = axes[1, 0]
    trades_df['exit_month'] = trades_df['exit_date'].dt.to_period('M')
    monthly_avg = trades_df.groupby('exit_month')['notional'].mean() / 1e6
    monthly_avg.index = monthly_avg.index.to_timestamp()
    ax3.plot(monthly_avg.index, monthly_avg.values, color='purple', linewidth=1.5)
    ax3.fill_between(monthly_avg.index, monthly_avg.values, alpha=0.3, color='purple')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Average Position Size ($M)')
    ax3.set_title('Average Position Size Over Time')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 4. Position size vs return
    ax4 = axes[1, 1]
    ax4.scatter(trades_df['notional'] / 1e6, trades_df['return_pct'],
                alpha=0.3, s=10, c='steelblue')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Position Size ($M)')
    ax4.set_ylabel('Return (%)')
    ax4.set_title('Position Size vs Return')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'weight_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: weight_distribution.png")


def plot_regime_analysis(equity_df):
    """Analyze performance in different market regimes."""
    print("\nGenerating regime analysis...")

    # Load SPY for regime detection
    spy = pd.read_parquet('data/reference/SPY.parquet')
    spy.index = pd.to_datetime(spy.index)

    # Align dates
    common_dates = equity_df.index.intersection(spy.index)
    equity_aligned = equity_df.loc[common_dates]
    spy_aligned = spy.loc[common_dates]

    # Calculate returns
    strat_returns = equity_aligned['equity'].pct_change()
    spy_returns = spy_aligned['close'].pct_change()

    # Define regimes based on SPY 60-day momentum
    spy_mom = spy_aligned['close'].pct_change(60)

    # Classify regimes
    regimes = pd.Series(index=common_dates, dtype=str)
    regimes[spy_mom > 0.10] = 'Strong Bull'
    regimes[(spy_mom > 0) & (spy_mom <= 0.10)] = 'Mild Bull'
    regimes[(spy_mom <= 0) & (spy_mom > -0.10)] = 'Mild Bear'
    regimes[spy_mom <= -0.10] = 'Strong Bear'
    regimes = regimes.dropna()

    # Calculate performance by regime
    regime_stats = []
    for regime in ['Strong Bull', 'Mild Bull', 'Mild Bear', 'Strong Bear']:
        regime_dates = regimes[regimes == regime].index
        if len(regime_dates) < 10:
            continue

        strat_ret = strat_returns.loc[strat_returns.index.isin(regime_dates)]
        spy_ret = spy_returns.loc[spy_returns.index.isin(regime_dates)]

        regime_stats.append({
            'Regime': regime,
            'Days': len(regime_dates),
            'Strategy Return (Ann.)': strat_ret.mean() * 252 * 100,
            'SPY Return (Ann.)': spy_ret.mean() * 252 * 100,
            'Strategy Vol': strat_ret.std() * np.sqrt(252) * 100,
            'Strategy Sharpe': (strat_ret.mean() * 252 - 0.02) / (strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0,
            'Outperformance': (strat_ret.mean() - spy_ret.mean()) * 252 * 100
        })

    regime_df = pd.DataFrame(regime_stats)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Returns by regime
    ax1 = axes[0, 0]
    x = range(len(regime_df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], regime_df['Strategy Return (Ann.)'], width,
            label='Strategy', color='steelblue', alpha=0.8)
    ax1.bar([i + width/2 for i in x], regime_df['SPY Return (Ann.)'], width,
            label='S&P 500', color='gray', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_df['Regime'], rotation=15)
    ax1.set_ylabel('Annualized Return (%)')
    ax1.set_title('Annualized Returns by Market Regime')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 2. Outperformance by regime
    ax2 = axes[0, 1]
    colors = ['green' if o > 0 else 'red' for o in regime_df['Outperformance']]
    ax2.bar(regime_df['Regime'], regime_df['Outperformance'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Outperformance vs SPY (%)')
    ax2.set_title('Strategy Outperformance by Regime')
    ax2.tick_params(axis='x', rotation=15)

    # 3. Sharpe by regime
    ax3 = axes[1, 0]
    ax3.bar(regime_df['Regime'], regime_df['Strategy Sharpe'], color='teal', alpha=0.7)
    ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='Sharpe = 1.0')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Strategy Sharpe Ratio by Regime')
    ax3.tick_params(axis='x', rotation=15)
    ax3.legend()

    # 4. Days in each regime
    ax4 = axes[1, 1]
    ax4.pie(regime_df['Days'], labels=regime_df['Regime'], autopct='%1.1f%%',
            colors=plt.cm.RdYlGn([0.2, 0.4, 0.6, 0.8]))
    ax4.set_title('Time Spent in Each Regime')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'regime_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: regime_analysis.png")

    # Print table
    print("\n" + "=" * 80)
    print("REGIME ANALYSIS")
    print("=" * 80)
    print(regime_df.to_string(index=False, float_format=lambda x: f'{x:.1f}'))

    return regime_df


def plot_seasonal_patterns(trades_df, equity_df):
    """Analyze seasonal patterns in returns."""
    print("\nGenerating seasonal patterns analysis...")

    equity_df = equity_df.copy()
    equity_df['returns'] = equity_df['equity'].pct_change()
    equity_df['month'] = equity_df.index.month
    equity_df['weekday'] = equity_df.index.weekday

    # Monthly average returns
    monthly_avg = equity_df.groupby('month')['returns'].mean() * 252 * 100

    # Day of week average returns
    weekday_avg = equity_df.groupby('weekday')['returns'].mean() * 252 * 100
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Monthly returns
    ax1 = axes[0, 0]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = ['green' if r > 0 else 'red' for r in monthly_avg.values]
    ax1.bar(months, monthly_avg.values, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Avg Annualized Return (%)')
    ax1.set_title('Average Returns by Month')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Day of week returns
    ax2 = axes[0, 1]
    colors = ['green' if r > 0 else 'red' for r in weekday_avg.values]
    ax2.bar(weekday_names, weekday_avg.values, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Avg Annualized Return (%)')
    ax2.set_title('Average Returns by Day of Week')

    # 3. Win rate by month (from trades)
    trades_df['exit_month'] = trades_df['exit_date'].dt.month
    monthly_winrate = trades_df.groupby('exit_month').apply(
        lambda x: (x['pnl_net'] > 0).mean() * 100
    )
    ax3 = axes[1, 0]
    ax3.bar(months, monthly_winrate.values, color='teal', alpha=0.7)
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Trade Win Rate by Month')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 100)

    # 4. Average trade P&L by month
    monthly_pnl = trades_df.groupby('exit_month')['pnl_net'].mean() / 1000
    ax4 = axes[1, 1]
    colors = ['green' if p > 0 else 'red' for p in monthly_pnl.values]
    ax4.bar(months, monthly_pnl.values, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('Avg Trade P&L ($K)')
    ax4.set_title('Average Trade P&L by Month')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'seasonal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: seasonal_patterns.png")


def plot_holding_analysis(trades_df):
    """Analyze relationship between holding period and P&L."""
    print("\nGenerating holding period analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Holding period distribution
    ax1 = axes[0, 0]
    ax1.hist(trades_df['hold_days'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(trades_df['hold_days'].mean(), color='red', linestyle='--',
                label=f'Mean: {trades_df["hold_days"].mean():.1f} days')
    ax1.set_xlabel('Holding Period (days)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Holding Periods')
    ax1.legend()

    # 2. Holding period vs return
    ax2 = axes[0, 1]
    ax2.scatter(trades_df['hold_days'], trades_df['return_pct'], alpha=0.3, s=10, c='steelblue')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Holding Period (days)')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Holding Period vs Return')

    # 3. Average return by holding period bucket
    ax3 = axes[1, 0]
    trades_df['hold_bucket'] = pd.cut(trades_df['hold_days'], bins=[0, 7, 14, 21, 28, 100],
                                       labels=['0-7', '8-14', '15-21', '22-28', '28+'])
    bucket_avg = trades_df.groupby('hold_bucket')['return_pct'].mean()
    colors = ['green' if r > 0 else 'red' for r in bucket_avg.values]
    ax3.bar(bucket_avg.index.astype(str), bucket_avg.values, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Holding Period (days)')
    ax3.set_ylabel('Average Return (%)')
    ax3.set_title('Average Return by Holding Period')

    # 4. Win rate by holding period
    ax4 = axes[1, 1]
    bucket_winrate = trades_df.groupby('hold_bucket').apply(
        lambda x: (x['pnl_net'] > 0).mean() * 100
    )
    ax4.bar(bucket_winrate.index.astype(str), bucket_winrate.values, color='teal', alpha=0.7)
    ax4.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax4.set_xlabel('Holding Period (days)')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Win Rate by Holding Period')
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'holding_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: holding_analysis.png")


def plot_concentration_analysis(trades_df):
    """Analyze portfolio concentration and top contributors."""
    print("\nGenerating concentration analysis...")

    # Calculate cumulative P&L contribution by symbol
    symbol_pnl = trades_df.groupby('symbol')['pnl_net'].sum().sort_values(ascending=False)
    total_pnl = symbol_pnl.sum()

    # Cumulative contribution
    cumulative = symbol_pnl.cumsum() / total_pnl * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Top 20 contributors
    ax1 = axes[0, 0]
    top20 = symbol_pnl.head(20) / 1e6
    colors = ['green' if p > 0 else 'red' for p in top20.values]
    ax1.barh(range(len(top20)), top20.values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top20)))
    ax1.set_yticklabels(top20.index)
    ax1.set_xlabel('Total P&L ($M)')
    ax1.set_title('Top 20 P&L Contributors')
    ax1.invert_yaxis()

    # 2. Bottom 20 contributors
    ax2 = axes[0, 1]
    bottom20 = symbol_pnl.tail(20) / 1e6
    colors = ['green' if p > 0 else 'red' for p in bottom20.values]
    ax2.barh(range(len(bottom20)), bottom20.values, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(bottom20)))
    ax2.set_yticklabels(bottom20.index)
    ax2.set_xlabel('Total P&L ($M)')
    ax2.set_title('Bottom 20 P&L Contributors')
    ax2.invert_yaxis()

    # 3. Cumulative P&L contribution curve
    ax3 = axes[1, 0]
    ax3.plot(range(1, len(cumulative) + 1), cumulative.values, color='steelblue', linewidth=2)
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50%')
    ax3.axhline(y=80, color='gray', linestyle='--', linewidth=1, label='80%')
    ax3.set_xlabel('Number of Stocks')
    ax3.set_ylabel('Cumulative P&L Contribution (%)')
    ax3.set_title('P&L Concentration Curve')

    # Find how many stocks contribute 50% and 80%
    n_50 = (cumulative <= 50).sum() + 1
    n_80 = (cumulative <= 80).sum() + 1
    ax3.axvline(n_50, color='green', linestyle=':', label=f'{n_50} stocks = 50%')
    ax3.axvline(n_80, color='orange', linestyle=':', label=f'{n_80} stocks = 80%')
    ax3.legend()

    # 4. P&L distribution across all stocks
    ax4 = axes[1, 1]
    ax4.hist(symbol_pnl.values / 1e6, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Total P&L per Stock ($M)')
    ax4.set_ylabel('Number of Stocks')
    ax4.set_title('Distribution of Stock-Level P&L')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'concentration_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: concentration_analysis.png")

    # Print summary
    print(f"\nConcentration Summary:")
    print(f"  Top {n_50} stocks contribute 50% of P&L")
    print(f"  Top {n_80} stocks contribute 80% of P&L")
    print(f"  {(symbol_pnl > 0).sum()} stocks were profitable")
    print(f"  {(symbol_pnl < 0).sum()} stocks were unprofitable")


def plot_streak_analysis(trades_df):
    """Analyze winning and losing streaks."""
    print("\nGenerating streak analysis...")

    # Sort trades by date
    trades_sorted = trades_df.sort_values('exit_date')

    # Calculate win/loss for each trade
    wins = (trades_sorted['pnl_net'] > 0).astype(int)

    # Find streaks
    streak_change = wins.diff().fillna(0) != 0
    streak_id = streak_change.cumsum()

    streaks = []
    for sid in streak_id.unique():
        streak_trades = trades_sorted[streak_id == sid]
        is_win = streak_trades['pnl_net'].iloc[0] > 0
        length = len(streak_trades)
        total_pnl = streak_trades['pnl_net'].sum()
        streaks.append({
            'type': 'Win' if is_win else 'Loss',
            'length': length,
            'total_pnl': total_pnl
        })

    streaks_df = pd.DataFrame(streaks)
    win_streaks = streaks_df[streaks_df['type'] == 'Win']['length']
    loss_streaks = streaks_df[streaks_df['type'] == 'Loss']['length']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Win streak distribution
    ax1 = axes[0, 0]
    ax1.hist(win_streaks, bins=range(1, win_streaks.max() + 2), color='green', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Streak Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Winning Streak Distribution (Max: {win_streaks.max()})')

    # 2. Loss streak distribution
    ax2 = axes[0, 1]
    ax2.hist(loss_streaks, bins=range(1, loss_streaks.max() + 2), color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Streak Length')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Losing Streak Distribution (Max: {loss_streaks.max()})')

    # 3. Rolling win rate (50-trade window)
    ax3 = axes[1, 0]
    rolling_winrate = wins.rolling(50).mean() * 100
    ax3.plot(range(len(rolling_winrate)), rolling_winrate.values, color='steelblue', linewidth=1)
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax3.fill_between(range(len(rolling_winrate)), 50, rolling_winrate.values,
                     where=rolling_winrate.values >= 50, color='green', alpha=0.3)
    ax3.fill_between(range(len(rolling_winrate)), 50, rolling_winrate.values,
                     where=rolling_winrate.values < 50, color='red', alpha=0.3)
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('50-Trade Rolling Win Rate (%)')
    ax3.set_title('Rolling Win Rate Over Time')
    ax3.set_ylim(0, 100)

    # 4. Cumulative P&L
    ax4 = axes[1, 1]
    cumulative_pnl = trades_sorted['pnl_net'].cumsum() / 1e6
    ax4.plot(range(len(cumulative_pnl)), cumulative_pnl.values, color='steelblue', linewidth=1.5)
    ax4.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values, alpha=0.3, color='steelblue')
    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Cumulative P&L ($M)')
    ax4.set_title('Cumulative P&L by Trade')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'streak_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: streak_analysis.png")

    # Print summary
    print(f"\nStreak Summary:")
    print(f"  Longest winning streak: {win_streaks.max()} trades")
    print(f"  Longest losing streak: {loss_streaks.max()} trades")
    print(f"  Average winning streak: {win_streaks.mean():.1f} trades")
    print(f"  Average losing streak: {loss_streaks.mean():.1f} trades")


def main():
    print("=" * 80)
    print("EXTENDED BACKTEST ANALYSIS")
    print("=" * 80)

    # Load results
    trades_df, equity_df, rebalance_df = load_results()
    loader = DataLoader()

    # Generate all analyses
    yearly_df = plot_yearly_performance(equity_df, trades_df)
    sector_df = plot_sector_exposure(rebalance_df)
    plot_weight_distribution(trades_df, loader)
    regime_df = plot_regime_analysis(equity_df)
    plot_seasonal_patterns(trades_df, equity_df)
    plot_holding_analysis(trades_df)
    plot_concentration_analysis(trades_df)
    plot_streak_analysis(trades_df)

    print("\n" + "=" * 80)
    print("EXTENDED ANALYSIS COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
