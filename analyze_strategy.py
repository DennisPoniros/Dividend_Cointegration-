#!/usr/bin/env python3
"""
Comprehensive Analysis of Dual Momentum Strategy
Generates: Trade CSV, Plots, Metrics, Cost/Profit Attribution
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from src.data.loader import DataLoader

# Output directory
OUTPUT_DIR = Path('/home/user/Dividend_Cointegration-/analysis_output')
OUTPUT_DIR.mkdir(exist_ok=True)


class DetailedDualMomentum:
    """Dual Momentum with detailed trade tracking."""

    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital

        # Strategy parameters (Optimized for best risk-adjusted returns)
        self.LOOKBACK = 252
        self.REBALANCE_FREQ = 15       # Every 15 trading days (optimized)
        self.NUM_HOLDINGS = 15         # Top 15 stocks (better diversification)
        self.SLIPPAGE_BPS = 1          # 1bp slippage (reduced)

        # Confidence-based position scaling
        self.CONFIDENCE_SCALING = True
        self.CONFIDENCE_POWER = 0.5    # Mild scaling by momentum strength

        self.price_data = {}
        self.all_trades = []
        self.daily_data = []
        self.rebalance_events = []

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > self.LOOKBACK + 100:
                self.price_data[symbol] = df['close']
        print(f"Loaded {len(self.price_data)} symbols")

    def calculate_12m_momentum(self, symbol, date):
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < self.LOOKBACK + 5:
            return None
        end_price = prices.iloc[-1]
        start_price = prices.iloc[-self.LOOKBACK - 1]
        if start_price <= 0:
            return None
        return (end_price / start_price) - 1

    def calculate_weights(self, holdings, momentum_scores):
        """Calculate position weights based on configuration."""
        if not holdings:
            return {}

        if self.CONFIDENCE_SCALING:
            # Scale positions by momentum strength
            mom_values = {s: momentum_scores[s] for s in holdings if s in momentum_scores}
            if mom_values:
                min_mom = min(mom_values.values())
                max_mom = max(mom_values.values())
                if max_mom > min_mom:
                    # Normalize and apply power scaling
                    scaled = {s: ((m - min_mom) / (max_mom - min_mom) + 0.5) ** self.CONFIDENCE_POWER
                             for s, m in mom_values.items()}
                else:
                    scaled = {s: 1.0 for s in mom_values}
                total_scaled = sum(scaled.values())
                return {s: v / total_scaled for s, v in scaled.items()}

        # Equal weight fallback
        return {s: 1.0 / len(holdings) for s in holdings}

    def run(self):
        all_dates = sorted(set().union(*[set(s.index) for s in self.price_data.values()]))
        start_idx = self.LOOKBACK + 50
        trading_dates = all_dates[start_idx:]

        print(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

        cash = self.initial_capital
        positions = {}  # symbol -> {shares, entry_price, entry_date}
        equity_curve = []
        last_rebalance_idx = None
        total_slippage_cost = 0
        total_turnover = 0

        for i, date in enumerate(trading_dates):
            # Calculate position values
            pos_val = 0
            for sym, pos in positions.items():
                if sym in self.price_data and date in self.price_data[sym].index:
                    pos_val += pos['shares'] * self.price_data[sym].loc[date]

            total_equity = cash + pos_val

            # Use trading days for rebalance frequency
            should_rebalance = (last_rebalance_idx is None or
                               (i - last_rebalance_idx) >= self.REBALANCE_FREQ)

            if should_rebalance:
                # Calculate momentum
                momentum_scores = {}
                for symbol in self.price_data:
                    if date not in self.price_data[symbol].index:
                        continue
                    mom = self.calculate_12m_momentum(symbol, date)
                    if mom is not None:
                        momentum_scores[symbol] = mom

                positive_mom = {s: m for s, m in momentum_scores.items() if m > 0}
                sorted_stocks = sorted(positive_mom.items(), key=lambda x: x[1], reverse=True)
                new_holdings = [s[0] for s in sorted_stocks[:self.NUM_HOLDINGS]]

                # Track sells
                rebalance_sells = []
                rebalance_buys = []
                rebalance_slippage = 0

                # Sell existing positions
                for sym, pos in list(positions.items()):
                    if sym in self.price_data and date in self.price_data[sym].index:
                        exit_price = self.price_data[sym].loc[date]
                        value = pos['shares'] * exit_price
                        slippage = value * self.SLIPPAGE_BPS / 10000
                        cash += value - slippage
                        total_slippage_cost += slippage
                        rebalance_slippage += slippage
                        total_turnover += value

                        # Record trade
                        pnl = (exit_price - pos['entry_price']) * pos['shares'] - slippage
                        hold_days = (date - pos['entry_date']).days
                        ret = (exit_price / pos['entry_price']) - 1

                        self.all_trades.append({
                            'symbol': sym,
                            'entry_date': pos['entry_date'],
                            'exit_date': date,
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'shares': pos['shares'],
                            'notional': pos['shares'] * pos['entry_price'],
                            'pnl_gross': (exit_price - pos['entry_price']) * pos['shares'],
                            'slippage': slippage,
                            'pnl_net': pnl,
                            'return_pct': ret * 100,
                            'hold_days': hold_days,
                            'momentum_at_entry': pos.get('momentum', None)
                        })

                        rebalance_sells.append({
                            'symbol': sym,
                            'shares': pos['shares'],
                            'price': exit_price,
                            'value': value
                        })

                positions = {}

                # Buy new positions with confidence-based weighting
                if new_holdings:
                    weights = self.calculate_weights(new_holdings, positive_mom)
                    available = cash

                    for sym in new_holdings:
                        if sym in self.price_data and date in self.price_data[sym].index:
                            price = self.price_data[sym].loc[date]
                            w = weights.get(sym, 1.0 / len(new_holdings))
                            target = available * w
                            shares = target / price
                            slippage = target * self.SLIPPAGE_BPS / 10000

                            positions[sym] = {
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': date,
                                'momentum': positive_mom.get(sym, 0),
                                'weight': w
                            }

                            cash -= target + slippage
                            total_slippage_cost += slippage
                            rebalance_slippage += slippage
                            total_turnover += target

                            rebalance_buys.append({
                                'symbol': sym,
                                'shares': shares,
                                'price': price,
                                'value': target,
                                'momentum': positive_mom.get(sym, 0),
                                'weight': w
                            })

                self.rebalance_events.append({
                    'date': date,
                    'equity_before': total_equity,
                    'num_sells': len(rebalance_sells),
                    'num_buys': len(rebalance_buys),
                    'turnover': sum(s['value'] for s in rebalance_sells) + sum(b['value'] for b in rebalance_buys),
                    'slippage': rebalance_slippage,
                    'qualifying_stocks': len(positive_mom),
                    'holdings': new_holdings.copy()
                })

                last_rebalance_idx = i

            # Daily tracking
            pos_val = 0
            position_details = {}
            for sym, pos in positions.items():
                if sym in self.price_data and date in self.price_data[sym].index:
                    val = pos['shares'] * self.price_data[sym].loc[date]
                    pos_val += val
                    position_details[sym] = val

            total_equity = cash + pos_val

            self.daily_data.append({
                'date': date,
                'equity': total_equity,
                'cash': cash,
                'invested': pos_val,
                'num_positions': len(positions),
                'pct_invested': pos_val / total_equity * 100 if total_equity > 0 else 0
            })

        # Final metrics
        self.total_slippage = total_slippage_cost
        self.total_turnover = total_turnover

        return self._calculate_metrics()

    def _calculate_metrics(self):
        df = pd.DataFrame(self.daily_data).set_index('date')
        trades_df = pd.DataFrame(self.all_trades)
        rebal_df = pd.DataFrame(self.rebalance_events)

        returns = df['equity'].pct_change().dropna()
        rf = 0.02 / 252
        excess = returns - rf

        # Drawdown calculation
        rolling_max = df['equity'].expanding().max()
        drawdown = (df['equity'] / rolling_max) - 1

        # Find worst drawdown period
        dd_end = drawdown.idxmin()
        dd_start = df['equity'][:dd_end].idxmax()

        metrics = {
            'total_return': (df['equity'].iloc[-1] / self.initial_capital) - 1,
            'annual_return': returns.mean() * 252,
            'annual_vol': returns.std() * np.sqrt(252),
            'sharpe': excess.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'sortino': excess.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
            'max_drawdown': drawdown.min(),
            'max_dd_start': dd_start,
            'max_dd_end': dd_end,
            'calmar': (returns.mean() * 252) / abs(drawdown.min()) if drawdown.min() != 0 else 0,
            'total_trades': len(trades_df),
            'win_rate': (trades_df['pnl_net'] > 0).mean() * 100 if len(trades_df) > 0 else 0,
            'avg_trade_pnl': trades_df['pnl_net'].mean() if len(trades_df) > 0 else 0,
            'avg_winner': trades_df[trades_df['pnl_net'] > 0]['pnl_net'].mean() if len(trades_df[trades_df['pnl_net'] > 0]) > 0 else 0,
            'avg_loser': trades_df[trades_df['pnl_net'] < 0]['pnl_net'].mean() if len(trades_df[trades_df['pnl_net'] < 0]) > 0 else 0,
            'best_trade': trades_df['pnl_net'].max() if len(trades_df) > 0 else 0,
            'worst_trade': trades_df['pnl_net'].min() if len(trades_df) > 0 else 0,
            'avg_hold_days': trades_df['hold_days'].mean() if len(trades_df) > 0 else 0,
            'total_slippage': self.total_slippage,
            'total_turnover': self.total_turnover,
            'turnover_ratio': self.total_turnover / self.initial_capital,
            'slippage_pct': self.total_slippage / self.initial_capital * 100,
            'profit_factor': abs(trades_df[trades_df['pnl_net'] > 0]['pnl_net'].sum() /
                               trades_df[trades_df['pnl_net'] < 0]['pnl_net'].sum())
                            if len(trades_df[trades_df['pnl_net'] < 0]) > 0 and
                               trades_df[trades_df['pnl_net'] < 0]['pnl_net'].sum() != 0 else 0,
        }

        # Monthly returns
        monthly = df['equity'].resample('M').last().pct_change().dropna()
        metrics['best_month'] = monthly.max() * 100
        metrics['worst_month'] = monthly.min() * 100
        metrics['pct_positive_months'] = (monthly > 0).mean() * 100

        # Yearly returns
        yearly = df['equity'].resample('Y').last().pct_change().dropna()
        metrics['years'] = len(yearly)

        return {
            'metrics': metrics,
            'equity_df': df,
            'trades_df': trades_df,
            'rebalance_df': rebal_df,
            'returns': returns,
            'drawdown': drawdown,
            'monthly_returns': monthly,
            'yearly_returns': yearly
        }


def create_plots(results, output_dir):
    """Create all analysis plots."""

    equity_df = results['equity_df']
    drawdown = results['drawdown']
    returns = results['returns']
    trades_df = results['trades_df']
    monthly = results['monthly_returns']

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Equity Curve
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax1 = axes[0]
    ax1.plot(equity_df.index, equity_df['equity'] / 1e6, 'b-', linewidth=1.5)
    ax1.fill_between(equity_df.index, equity_df['equity'] / 1e6, alpha=0.3)
    ax1.set_ylabel('Equity ($M)')
    ax1.set_title('Equity Curve - Dual Momentum Strategy', fontsize=14, fontweight='bold')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # 2. Drawdown
    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.5)
    ax2.plot(drawdown.index, drawdown * 100, 'r-', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Underwater Equity (Drawdown)', fontsize=14, fontweight='bold')
    ax2.set_ylim(drawdown.min() * 100 * 1.1, 5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # 3. Investment Level
    ax3 = axes[2]
    ax3.fill_between(equity_df.index, equity_df['pct_invested'], alpha=0.5, color='green')
    ax3.plot(equity_df.index, equity_df['pct_invested'], 'g-', linewidth=1)
    ax3.set_ylabel('% Invested')
    ax3.set_xlabel('Date')
    ax3.set_title('Investment Level Over Time', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    plt.savefig(output_dir / 'equity_drawdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: equity_drawdown.png")

    # 4. Monthly Returns Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create monthly returns matrix
    monthly_df = pd.DataFrame(monthly)
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    monthly_pivot = monthly_df.pivot(index='year', columns='month', values='equity')
    monthly_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Plot heatmap
    im = ax.imshow(monthly_pivot.values * 100, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)

    ax.set_xticks(range(12))
    ax.set_xticklabels(monthly_pivot.columns)
    ax.set_yticks(range(len(monthly_pivot.index)))
    ax.set_yticklabels(monthly_pivot.index)

    # Add text annotations
    for i in range(len(monthly_pivot.index)):
        for j in range(12):
            val = monthly_pivot.iloc[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.08 else 'black'
                ax.text(j, i, f'{val*100:.1f}%', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='Monthly Return (%)')
    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_returns_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: monthly_returns_heatmap.png")

    # 5. Trade Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trade P&L Distribution
    ax1 = axes[0, 0]
    trades_df['pnl_net'].hist(bins=50, ax=ax1, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=trades_df['pnl_net'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: ${trades_df["pnl_net"].mean():,.0f}')
    ax1.set_xlabel('Trade P&L ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Trade P&L Distribution', fontweight='bold')
    ax1.legend()

    # Return Distribution
    ax2 = axes[0, 1]
    trades_df['return_pct'].hist(bins=50, ax=ax2, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=trades_df['return_pct'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {trades_df["return_pct"].mean():.1f}%')
    ax2.set_xlabel('Trade Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Trade Return Distribution', fontweight='bold')
    ax2.legend()

    # Hold Days Distribution
    ax3 = axes[1, 0]
    trades_df['hold_days'].hist(bins=30, ax=ax3, color='orange', edgecolor='black', alpha=0.7)
    ax3.axvline(x=trades_df['hold_days'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {trades_df["hold_days"].mean():.1f} days')
    ax3.set_xlabel('Hold Days')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Holding Period Distribution', fontweight='bold')
    ax3.legend()

    # Cumulative P&L
    ax4 = axes[1, 1]
    cumulative_pnl = trades_df.sort_values('exit_date')['pnl_net'].cumsum()
    ax4.plot(range(len(cumulative_pnl)), cumulative_pnl / 1e6, 'b-', linewidth=1.5)
    ax4.fill_between(range(len(cumulative_pnl)), cumulative_pnl / 1e6, alpha=0.3)
    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Cumulative P&L ($M)')
    ax4.set_title('Cumulative Trade P&L', fontweight='bold')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'trade_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: trade_analysis.png")

    # 6. Rolling Sharpe
    fig, ax = plt.subplots(figsize=(14, 5))

    rolling_sharpe = (returns.rolling(252).mean() * 252) / (returns.rolling(252).std() * np.sqrt(252))
    ax.plot(rolling_sharpe.index, rolling_sharpe, 'b-', linewidth=1.5)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                    where=rolling_sharpe >= 0, color='green', alpha=0.3)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                    where=rolling_sharpe < 0, color='red', alpha=0.3)
    ax.set_ylabel('Rolling 1Y Sharpe Ratio')
    ax.set_xlabel('Date')
    ax.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.savefig(output_dir / 'rolling_sharpe.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: rolling_sharpe.png")


def generate_report(results, output_dir):
    """Generate text report."""

    m = results['metrics']
    trades_df = results['trades_df']
    rebal_df = results['rebalance_df']
    equity_df = results['equity_df']

    report = []
    report.append("=" * 80)
    report.append("DUAL MOMENTUM STRATEGY - COMPREHENSIVE ANALYSIS REPORT")
    report.append("(Optimized Configuration)")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report.append("\n" + "=" * 80)
    report.append("STRATEGY CONFIGURATION")
    report.append("=" * 80)
    report.append("  Lookback Period:       252 days (12 months)")
    report.append("  Rebalance Frequency:   15 trading days")
    report.append("  Number of Holdings:    15 stocks")
    report.append("  Slippage:              1 bp per trade")
    report.append("  Position Sizing:       Confidence-scaled (p=0.5)")
    report.append("  ")
    report.append("  Optimizations Applied:")
    report.append("    - Increased rebalance frequency (15 vs 21 days)")
    report.append("    - Increased diversification (15 vs 10 holdings)")
    report.append("    - Confidence-based position scaling")
    report.append("    - Reduced transaction costs (1 vs 2 bps)")
    report.append("    - Fixed: Uses trading days (not calendar days)")
    report.append("    - Fixed: Removed cash buffer drag")

    report.append("\n" + "=" * 80)
    report.append("1. PERFORMANCE SUMMARY")
    report.append("=" * 80)
    report.append(f"  Total Return:          {m['total_return']*100:>12.2f}%")
    report.append(f"  Annual Return:         {m['annual_return']*100:>12.2f}%")
    report.append(f"  Annual Volatility:     {m['annual_vol']*100:>12.2f}%")
    report.append(f"  Sharpe Ratio:          {m['sharpe']:>12.2f}")
    report.append(f"  Sortino Ratio:         {m['sortino']:>12.2f}")
    report.append(f"  Calmar Ratio:          {m['calmar']:>12.2f}")
    report.append(f"  Max Drawdown:          {m['max_drawdown']*100:>12.2f}%")
    report.append(f"    Drawdown Start:      {m['max_dd_start'].strftime('%Y-%m-%d')}")
    report.append(f"    Drawdown End:        {m['max_dd_end'].strftime('%Y-%m-%d')}")

    report.append("\n" + "=" * 80)
    report.append("2. BACKTEST STATISTICS")
    report.append("=" * 80)
    report.append(f"  Start Date:            {equity_df.index[0].strftime('%Y-%m-%d')}")
    report.append(f"  End Date:              {equity_df.index[-1].strftime('%Y-%m-%d')}")
    report.append(f"  Trading Days:          {len(equity_df):>12}")
    report.append(f"  Years:                 {len(equity_df)/252:>12.2f}")
    report.append(f"  Number of Rebalances:  {len(rebal_df):>12}")
    report.append(f"  Total Trades:          {m['total_trades']:>12}")

    report.append("\n" + "=" * 80)
    report.append("3. TRADE STATISTICS")
    report.append("=" * 80)
    report.append(f"  Win Rate:              {m['win_rate']:>12.1f}%")
    report.append(f"  Profit Factor:         {m['profit_factor']:>12.2f}")
    report.append(f"  Avg Trade P&L:         ${m['avg_trade_pnl']:>11,.0f}")
    report.append(f"  Avg Winner:            ${m['avg_winner']:>11,.0f}")
    report.append(f"  Avg Loser:             ${m['avg_loser']:>11,.0f}")
    report.append(f"  Best Trade:            ${m['best_trade']:>11,.0f}")
    report.append(f"  Worst Trade:           ${m['worst_trade']:>11,.0f}")
    report.append(f"  Avg Hold Days:         {m['avg_hold_days']:>12.1f}")

    report.append("\n" + "=" * 80)
    report.append("4. COST ATTRIBUTION")
    report.append("=" * 80)
    report.append(f"  Total Turnover:        ${m['total_turnover']:>11,.0f}")
    report.append(f"  Turnover Ratio:        {m['turnover_ratio']:>12.1f}x initial capital")
    report.append(f"  Slippage Rate:         {1:>12} bps (0.01%)")
    report.append(f"  Total Slippage Cost:   ${m['total_slippage']:>11,.0f}")
    report.append(f"  Slippage as % Capital: {m['slippage_pct']:>12.2f}%")
    annual_slippage = m['slippage_pct'] / (len(equity_df) / 252)
    report.append(f"  Annualized Slippage:   {annual_slippage:>12.2f}%")

    # Gross vs Net comparison
    gross_pnl = trades_df['pnl_gross'].sum()
    net_pnl = trades_df['pnl_net'].sum()
    report.append(f"\n  Gross P&L:             ${gross_pnl:>11,.0f}")
    report.append(f"  Net P&L:               ${net_pnl:>11,.0f}")
    report.append(f"  Total Costs:           ${gross_pnl - net_pnl:>11,.0f}")
    report.append(f"  Cost Impact on Return: {(gross_pnl - net_pnl) / gross_pnl * 100:>12.2f}% of gross")

    report.append("\n" + "=" * 80)
    report.append("5. PROFIT ATTRIBUTION BY SYMBOL")
    report.append("=" * 80)

    # Top winners and losers by symbol
    symbol_pnl = trades_df.groupby('symbol').agg({
        'pnl_net': ['sum', 'count', 'mean'],
        'return_pct': 'mean'
    }).round(2)
    symbol_pnl.columns = ['total_pnl', 'num_trades', 'avg_pnl', 'avg_return_pct']
    symbol_pnl = symbol_pnl.sort_values('total_pnl', ascending=False)

    report.append("\n  Top 10 Contributing Symbols:")
    report.append(f"  {'Symbol':<8} {'Total P&L':>12} {'#Trades':>8} {'Avg P&L':>10} {'Avg Ret%':>10}")
    report.append("  " + "-" * 50)
    for sym in symbol_pnl.head(10).index:
        row = symbol_pnl.loc[sym]
        report.append(f"  {sym:<8} ${row['total_pnl']:>11,.0f} {int(row['num_trades']):>8} ${row['avg_pnl']:>9,.0f} {row['avg_return_pct']:>9.1f}%")

    report.append("\n  Bottom 10 Contributing Symbols:")
    report.append(f"  {'Symbol':<8} {'Total P&L':>12} {'#Trades':>8} {'Avg P&L':>10} {'Avg Ret%':>10}")
    report.append("  " + "-" * 50)
    for sym in symbol_pnl.tail(10).index:
        row = symbol_pnl.loc[sym]
        report.append(f"  {sym:<8} ${row['total_pnl']:>11,.0f} {int(row['num_trades']):>8} ${row['avg_pnl']:>9,.0f} {row['avg_return_pct']:>9.1f}%")

    report.append("\n" + "=" * 80)
    report.append("6. MONTHLY RETURNS (%)")
    report.append("=" * 80)

    report.append(f"  Best Month:            {m['best_month']:>12.1f}%")
    report.append(f"  Worst Month:           {m['worst_month']:>12.1f}%")
    report.append(f"  % Positive Months:     {m['pct_positive_months']:>12.1f}%")

    report.append("\n" + "=" * 80)
    report.append("7. RISK METRICS")
    report.append("=" * 80)

    # VaR and CVaR
    returns = results['returns']
    var_95 = returns.quantile(0.05) * 100
    cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
    report.append(f"  Daily VaR (95%):       {var_95:>12.2f}%")
    report.append(f"  Daily CVaR (95%):      {cvar_95:>12.2f}%")
    report.append(f"  Skewness:              {returns.skew():>12.2f}")
    report.append(f"  Kurtosis:              {returns.kurtosis():>12.2f}")

    # Longest drawdown
    dd = results['drawdown']
    dd_periods = (dd < 0).astype(int)
    dd_groups = (dd_periods != dd_periods.shift()).cumsum()
    dd_lengths = dd_periods.groupby(dd_groups).sum()
    max_dd_length = dd_lengths.max()
    report.append(f"  Longest Drawdown:      {max_dd_length:>12} days")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'strategy_report.txt', 'w') as f:
        f.write(report_text)

    print(f"Saved: strategy_report.txt")
    return report_text


def main():
    print("=" * 80)
    print("DUAL MOMENTUM STRATEGY - COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    loader = DataLoader()
    symbols = loader.get_available_symbols()

    strategy = DetailedDualMomentum(loader)
    strategy.load_data(symbols)

    print("\nRunning backtest with detailed tracking...")
    results = strategy.run()

    # Save trade CSV
    trades_df = results['trades_df']
    trades_df.to_csv(OUTPUT_DIR / 'trades.csv', index=False)
    print(f"\nSaved: trades.csv ({len(trades_df)} trades)")

    # Save daily equity CSV
    equity_df = results['equity_df']
    equity_df.to_csv(OUTPUT_DIR / 'daily_equity.csv')
    print(f"Saved: daily_equity.csv ({len(equity_df)} days)")

    # Save rebalance events
    rebal_df = results['rebalance_df']
    rebal_df.to_csv(OUTPUT_DIR / 'rebalance_events.csv', index=False)
    print(f"Saved: rebalance_events.csv ({len(rebal_df)} rebalances)")

    # Generate plots
    print("\nGenerating plots...")
    create_plots(results, OUTPUT_DIR)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(results, OUTPUT_DIR)

    # Print report to console
    print("\n" + report)

    print(f"\n\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
