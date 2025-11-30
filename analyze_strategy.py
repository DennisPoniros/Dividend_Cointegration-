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

        # Strategy parameters
        self.LOOKBACK = 252
        self.REBALANCE_FREQ = 14  # Bi-weekly (optimized for lower downside vol)
        self.NUM_HOLDINGS = 12  # Top 12 stocks (more diversification)
        self.SLIPPAGE_BPS = 2  # Realistic slippage only (no commission)
        self.USE_RISK_PARITY = False  # Equal weight
        self.USE_QUALITY_FILTER = True  # Require momentum > 0.75 * volatility
        self.QUALITY_MULT = 0.75
        self.LEVERAGE = 2.0  # 1.0 = no leverage, 2.0 = 2x leverage

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

    def calculate_downside_vol(self, symbol, date, lookback=63):
        """Calculate annualized downside volatility (std of negative returns)."""
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < lookback + 5:
            return None
        returns = prices.pct_change().dropna().iloc[-lookback:]
        neg_returns = returns[returns < 0]
        if len(neg_returns) < 5:
            return None
        return neg_returns.std() * np.sqrt(252)

    def calculate_volatility(self, symbol, date, lookback=63):
        """Calculate annualized volatility for quality filter."""
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(prices) < lookback + 5:
            return None
        returns = prices.pct_change().dropna().iloc[-lookback:]
        return returns.std() * np.sqrt(252)

    def run(self):
        all_dates = sorted(set().union(*[set(s.index) for s in self.price_data.values()]))
        start_idx = self.LOOKBACK + 50
        trading_dates = all_dates[start_idx:]

        print(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

        cash = self.initial_capital
        margin_debt = 0  # Track borrowed amount for leverage
        positions = {}  # symbol -> {shares, entry_price, entry_date}
        equity_curve = []
        last_rebalance = None
        total_slippage_cost = 0
        total_turnover = 0

        for i, date in enumerate(trading_dates):
            # Calculate position values
            pos_val = 0
            for sym, pos in positions.items():
                if sym in self.price_data and date in self.price_data[sym].index:
                    pos_val += pos['shares'] * self.price_data[sym].loc[date]

            total_equity = cash + pos_val - margin_debt

            should_rebalance = last_rebalance is None or (date - last_rebalance).days >= self.REBALANCE_FREQ

            if should_rebalance:
                # Calculate momentum and volatility
                momentum_scores = {}
                vol_scores = {}
                for symbol in self.price_data:
                    if date not in self.price_data[symbol].index:
                        continue
                    mom = self.calculate_12m_momentum(symbol, date)
                    vol = self.calculate_volatility(symbol, date)
                    if mom is not None:
                        momentum_scores[symbol] = mom
                    if vol is not None:
                        vol_scores[symbol] = vol

                # Filter: positive momentum + quality filter
                if self.USE_QUALITY_FILTER:
                    positive_mom = {s: m for s, m in momentum_scores.items()
                                   if m > 0 and m > self.QUALITY_MULT * vol_scores.get(s, 999)}
                else:
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

                # Pay off margin debt after selling all positions
                cash -= margin_debt
                margin_debt = 0
                positions = {}

                # Buy new positions
                if new_holdings:
                    # Calculate weights
                    if self.USE_RISK_PARITY:
                        # Risk parity: inverse downside volatility weighting
                        inv_vols = {}
                        for sym in new_holdings:
                            dnvol = self.calculate_downside_vol(sym, date)
                            if dnvol and dnvol > 0:
                                inv_vols[sym] = 1.0 / dnvol
                            else:
                                inv_vols[sym] = 1.0  # fallback to equal weight
                        total_inv_vol = sum(inv_vols.values())
                        weights = {s: inv_vols[s] / total_inv_vol for s in new_holdings}
                    else:
                        # Equal weight
                        weights = {s: 1.0 / len(new_holdings) for s in new_holdings}

                    available = cash
                    # With leverage: invest LEVERAGE times our equity
                    leveraged_amount = available * 0.98 * self.LEVERAGE
                    equity_spent = available * 0.98  # What we actually spend from cash

                    for sym in new_holdings:
                        if sym in self.price_data and date in self.price_data[sym].index:
                            price = self.price_data[sym].loc[date]
                            target = leveraged_amount * weights[sym]
                            shares = target / price
                            slippage = target * self.SLIPPAGE_BPS / 10000

                            positions[sym] = {
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': date,
                                'momentum': positive_mom.get(sym, 0)
                            }

                            # We only spend our equity portion from cash
                            cash -= equity_spent * weights[sym] + slippage
                            total_slippage_cost += slippage
                            rebalance_slippage += slippage
                            total_turnover += target

                            rebalance_buys.append({
                                'symbol': sym,
                                'shares': shares,
                                'price': price,
                                'value': target,
                                'momentum': positive_mom.get(sym, 0)
                            })

                    # Track borrowed amount
                    margin_debt = leveraged_amount - equity_spent

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

                last_rebalance = date

            # Daily tracking
            pos_val = 0
            position_details = {}
            for sym, pos in positions.items():
                if sym in self.price_data and date in self.price_data[sym].index:
                    val = pos['shares'] * self.price_data[sym].loc[date]
                    pos_val += val
                    position_details[sym] = val

            total_equity = cash + pos_val - margin_debt

            self.daily_data.append({
                'date': date,
                'equity': total_equity,
                'cash': cash,
                'invested': pos_val,
                'margin_debt': margin_debt,
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

    # 6. Rolling Sharpe (legacy - keeping for backwards compatibility)
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


def calculate_rolling_omega(returns, window=252, threshold=0):
    """
    Calculate rolling Omega ratio.
    Omega = sum of returns above threshold / abs(sum of returns below threshold)
    """
    def omega_ratio(r):
        above = r[r > threshold].sum()
        below = abs(r[r <= threshold].sum())
        if below == 0:
            return np.nan
        return above / below

    return returns.rolling(window).apply(omega_ratio, raw=False)


def calculate_rolling_calmar(returns, window=252):
    """
    Calculate rolling Calmar ratio.
    Calmar = Annualized Return / Max Drawdown
    """
    def calmar_ratio(r):
        ann_ret = r.mean() * 252
        # Calculate max drawdown for this window
        cumulative = (1 + r).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        max_dd = abs(drawdown.min())
        if max_dd == 0:
            return np.nan
        return ann_ret / max_dd

    return returns.rolling(window).apply(calmar_ratio, raw=False)


def calculate_rolling_sortino(returns, window=252, rf=0.02):
    """
    Calculate rolling Sortino ratio.
    Sortino = (Return - Rf) / Downside Deviation
    """
    rf_daily = rf / 252

    def sortino_ratio(r):
        excess = r - rf_daily
        mean_excess = excess.mean() * 252
        downside = r[r < 0]
        if len(downside) < 2:
            return np.nan
        downside_dev = downside.std() * np.sqrt(252)
        if downside_dev == 0:
            return np.nan
        return mean_excess / downside_dev

    return returns.rolling(window).apply(sortino_ratio, raw=False)


def calculate_upside_downside_beta(strategy_returns, benchmark_returns):
    """
    Calculate upside and downside beta.
    Upside beta: beta when benchmark is positive
    Downside beta: beta when benchmark is negative
    """
    # Align the returns
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 30:
        return None, None, None

    # Overall beta
    cov = np.cov(aligned['strategy'], aligned['benchmark'])[0, 1]
    var_bench = np.var(aligned['benchmark'])
    overall_beta = cov / var_bench if var_bench > 0 else np.nan

    # Upside beta (when benchmark is positive)
    up_days = aligned[aligned['benchmark'] > 0]
    if len(up_days) > 10:
        cov_up = np.cov(up_days['strategy'], up_days['benchmark'])[0, 1]
        var_up = np.var(up_days['benchmark'])
        upside_beta = cov_up / var_up if var_up > 0 else np.nan
    else:
        upside_beta = np.nan

    # Downside beta (when benchmark is negative)
    down_days = aligned[aligned['benchmark'] < 0]
    if len(down_days) > 10:
        cov_down = np.cov(down_days['strategy'], down_days['benchmark'])[0, 1]
        var_down = np.var(down_days['benchmark'])
        downside_beta = cov_down / var_down if var_down > 0 else np.nan
    else:
        downside_beta = np.nan

    return overall_beta, upside_beta, downside_beta


def calculate_capture_ratios(strategy_returns, benchmark_returns):
    """
    Calculate upside and downside capture ratios.
    Upside capture: strategy return in up markets / benchmark return in up markets
    Downside capture: strategy return in down markets / benchmark return in down markets
    """
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 30:
        return None, None

    # Upside capture
    up_days = aligned[aligned['benchmark'] > 0]
    if len(up_days) > 10:
        upside_capture = (up_days['strategy'].mean() / up_days['benchmark'].mean()) * 100
    else:
        upside_capture = np.nan

    # Downside capture
    down_days = aligned[aligned['benchmark'] < 0]
    if len(down_days) > 10:
        downside_capture = (down_days['strategy'].mean() / down_days['benchmark'].mean()) * 100
    else:
        downside_capture = np.nan

    return upside_capture, downside_capture


def create_rolling_metrics_plot(results, output_dir, window=252):
    """Create combined rolling metrics plot (Sharpe, Sortino, Omega, Calmar)."""

    returns = results['returns']

    # Calculate all rolling metrics
    rolling_sharpe = (returns.rolling(window).mean() * 252) / (returns.rolling(window).std() * np.sqrt(252))
    rolling_sortino = calculate_rolling_sortino(returns, window)
    rolling_omega = calculate_rolling_omega(returns, window)
    rolling_calmar = calculate_rolling_calmar(returns, window)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # 1. Rolling Sharpe
    ax1 = axes[0]
    ax1.plot(rolling_sharpe.index, rolling_sharpe, 'b-', linewidth=1.5, label='Rolling Sharpe')
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Target = 1.0')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                     where=rolling_sharpe >= 0, color='green', alpha=0.2)
    ax1.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                     where=rolling_sharpe < 0, color='red', alpha=0.2)
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title(f'Rolling {window}-Day Risk-Adjusted Performance Metrics', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_ylim(-2, 4)

    # 2. Rolling Sortino
    ax2 = axes[1]
    ax2.plot(rolling_sortino.index, rolling_sortino, 'purple', linewidth=1.5, label='Rolling Sortino')
    ax2.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='Target = 1.5')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(rolling_sortino.index, rolling_sortino, 0,
                     where=rolling_sortino >= 0, color='purple', alpha=0.2)
    ax2.fill_between(rolling_sortino.index, rolling_sortino, 0,
                     where=rolling_sortino < 0, color='red', alpha=0.2)
    ax2.set_ylabel('Sortino Ratio')
    ax2.legend(loc='upper left')
    ax2.set_ylim(-3, 6)

    # 3. Rolling Omega
    ax3 = axes[2]
    ax3.plot(rolling_omega.index, rolling_omega, 'darkorange', linewidth=1.5, label='Rolling Omega')
    ax3.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Breakeven = 1.0')
    ax3.fill_between(rolling_omega.index, rolling_omega, 1,
                     where=rolling_omega >= 1, color='green', alpha=0.2)
    ax3.fill_between(rolling_omega.index, rolling_omega, 1,
                     where=rolling_omega < 1, color='red', alpha=0.2)
    ax3.set_ylabel('Omega Ratio')
    ax3.legend(loc='upper left')
    ax3.set_ylim(0, 4)

    # 4. Rolling Calmar
    ax4 = axes[3]
    ax4.plot(rolling_calmar.index, rolling_calmar, 'teal', linewidth=1.5, label='Rolling Calmar')
    ax4.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Target = 1.0')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.fill_between(rolling_calmar.index, rolling_calmar, 0,
                     where=rolling_calmar >= 0, color='teal', alpha=0.2)
    ax4.fill_between(rolling_calmar.index, rolling_calmar, 0,
                     where=rolling_calmar < 0, color='red', alpha=0.2)
    ax4.set_ylabel('Calmar Ratio')
    ax4.set_xlabel('Date')
    ax4.legend(loc='upper left')
    ax4.set_ylim(-2, 6)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    plt.savefig(output_dir / 'rolling_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: rolling_metrics.png")

    return {
        'rolling_sharpe': rolling_sharpe,
        'rolling_sortino': rolling_sortino,
        'rolling_omega': rolling_omega,
        'rolling_calmar': rolling_calmar
    }


def create_benchmark_comparison_plot(results, benchmark_returns, output_dir):
    """Create strategy vs benchmark comparison plot with performance metrics."""

    strategy_returns = results['returns']
    equity_df = results['equity_df']

    # Align returns
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 30:
        print("Warning: Not enough aligned data for benchmark comparison")
        return None

    # Calculate cumulative returns
    strat_cumret = (1 + aligned['strategy']).cumprod()
    bench_cumret = (1 + aligned['benchmark']).cumprod()

    # Calculate betas and capture ratios
    overall_beta, upside_beta, downside_beta = calculate_upside_downside_beta(
        aligned['strategy'], aligned['benchmark']
    )
    upside_capture, downside_capture = calculate_capture_ratios(
        aligned['strategy'], aligned['benchmark']
    )

    # Calculate correlation
    correlation = aligned['strategy'].corr(aligned['benchmark'])

    # Calculate alpha (Jensen's alpha)
    rf_daily = 0.02 / 252
    strat_excess = aligned['strategy'].mean() - rf_daily
    bench_excess = aligned['benchmark'].mean() - rf_daily
    alpha = (strat_excess - overall_beta * bench_excess) * 252  # Annualized

    # Calculate Information Ratio
    active_returns = aligned['strategy'] - aligned['benchmark']
    tracking_error = active_returns.std() * np.sqrt(252)
    info_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else np.nan

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    fig = plt.figure(figsize=(16, 12))

    # Create grid spec for complex layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # 1. Cumulative returns comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(strat_cumret.index, strat_cumret, 'b-', linewidth=2, label='Strategy')
    ax1.plot(bench_cumret.index, bench_cumret, 'gray', linewidth=2, alpha=0.8, label='S&P 500 (SPY)')
    ax1.set_ylabel('Cumulative Return (Growth of $1)')
    ax1.set_title('Strategy vs S&P 500 Benchmark', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Add final values annotation
    strat_final = strat_cumret.iloc[-1]
    bench_final = bench_cumret.iloc[-1]
    ax1.annotate(f'Strategy: {strat_final:.2f}x',
                xy=(strat_cumret.index[-1], strat_final),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color='blue')
    ax1.annotate(f'S&P 500: {bench_final:.2f}x',
                xy=(bench_cumret.index[-1], bench_final),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color='gray')

    # 2. Relative performance (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    relative_perf = strat_cumret / bench_cumret
    ax2.plot(relative_perf.index, relative_perf, 'green', linewidth=1.5)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax2.fill_between(relative_perf.index, relative_perf, 1,
                     where=relative_perf >= 1, color='green', alpha=0.3)
    ax2.fill_between(relative_perf.index, relative_perf, 1,
                     where=relative_perf < 1, color='red', alpha=0.3)
    ax2.set_ylabel('Strategy / Benchmark')
    ax2.set_title('Relative Performance (Strategy ÷ S&P 500)', fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 3. Rolling correlation (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    rolling_corr = aligned['strategy'].rolling(63).corr(aligned['benchmark'])
    ax3.plot(rolling_corr.index, rolling_corr, 'purple', linewidth=1.5)
    ax3.axhline(y=correlation, color='red', linestyle='--', alpha=0.7,
                label=f'Avg: {correlation:.2f}')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Correlation')
    ax3.set_title('Rolling 63-Day Correlation with S&P 500', fontweight='bold')
    ax3.set_ylim(-1, 1)
    ax3.legend(loc='lower right')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 4. Scatter plot with regression (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])

    # Color points by benchmark return
    up_mask = aligned['benchmark'] > 0
    ax4.scatter(aligned.loc[up_mask, 'benchmark'] * 100,
               aligned.loc[up_mask, 'strategy'] * 100,
               c='green', alpha=0.4, s=20, label=f'Up days (β={upside_beta:.2f})')
    ax4.scatter(aligned.loc[~up_mask, 'benchmark'] * 100,
               aligned.loc[~up_mask, 'strategy'] * 100,
               c='red', alpha=0.4, s=20, label=f'Down days (β={downside_beta:.2f})')

    # Add regression lines
    x_range = np.linspace(aligned['benchmark'].min(), aligned['benchmark'].max(), 100) * 100
    ax4.plot(x_range, overall_beta * x_range, 'b-', linewidth=2,
             label=f'Overall β={overall_beta:.2f}')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel('S&P 500 Daily Return (%)')
    ax4.set_ylabel('Strategy Daily Return (%)')
    ax4.set_title('Return Scatter: Strategy vs S&P 500', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)

    # 5. Key metrics panel (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Create metrics text box
    metrics_text = [
        "═══════════════════════════════════",
        "     BENCHMARK COMPARISON METRICS      ",
        "═══════════════════════════════════",
        "",
        f"  Overall Beta:        {overall_beta:.3f}",
        f"  Upside Beta:         {upside_beta:.3f}",
        f"  Downside Beta:       {downside_beta:.3f}",
        "",
        f"  Upside Capture:      {upside_capture:.1f}%",
        f"  Downside Capture:    {downside_capture:.1f}%",
        f"  Capture Ratio:       {upside_capture/downside_capture:.2f}x" if downside_capture else "  Capture Ratio:       N/A",
        "",
        f"  Correlation:         {correlation:.3f}",
        f"  Alpha (annualized):  {alpha*100:.2f}%",
        f"  Information Ratio:   {info_ratio:.2f}",
        f"  Tracking Error:      {tracking_error*100:.1f}%",
        "",
        "═══════════════════════════════════",
        f"  Strategy Total:      {(strat_final-1)*100:.1f}%",
        f"  Benchmark Total:     {(bench_final-1)*100:.1f}%",
        f"  Outperformance:      {(strat_final/bench_final-1)*100:.1f}%",
        "═══════════════════════════════════",
    ]

    ax5.text(0.1, 0.95, '\n'.join(metrics_text), transform=ax5.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: benchmark_comparison.png")

    return {
        'overall_beta': overall_beta,
        'upside_beta': upside_beta,
        'downside_beta': downside_beta,
        'upside_capture': upside_capture,
        'downside_capture': downside_capture,
        'correlation': correlation,
        'alpha': alpha,
        'info_ratio': info_ratio,
        'tracking_error': tracking_error
    }


def run_slippage_sensitivity(loader, symbols, slippage_range, benchmark_returns=None):
    """
    Run backtest with different slippage values to analyze sensitivity.
    Returns dict with slippage -> metrics mapping.
    """
    results_by_slippage = {}

    for slippage_bps in slippage_range:
        print(f"  Running backtest with {slippage_bps} bps slippage...", end=" ")

        # Create strategy with specific slippage
        strategy = DetailedDualMomentum(loader)
        strategy.SLIPPAGE_BPS = slippage_bps
        strategy.load_data(symbols)
        results = strategy.run()

        # Extract key metrics
        m = results['metrics']
        results_by_slippage[slippage_bps] = {
            'sharpe': m['sharpe'],
            'sortino': m['sortino'],
            'total_return': m['total_return'],
            'annual_return': m['annual_return'],
            'max_drawdown': m['max_drawdown'],
            'total_slippage': m['total_slippage'],
            'slippage_pct': m['slippage_pct']
        }
        print(f"Sharpe: {m['sharpe']:.2f}")

    return results_by_slippage


def create_slippage_sensitivity_plot(loader, symbols, benchmark_returns, output_dir):
    """Create slippage sensitivity analysis plot."""

    slippage_range = list(range(1, 11))  # 1 to 10 bps

    print("\nRunning slippage sensitivity analysis...")
    sensitivity_results = run_slippage_sensitivity(loader, symbols, slippage_range, benchmark_returns)

    # Calculate S&P 500 Sharpe ratio
    if benchmark_returns is not None and len(benchmark_returns) > 252:
        rf_daily = 0.02 / 252
        bench_excess = benchmark_returns - rf_daily
        spy_sharpe = bench_excess.mean() / benchmark_returns.std() * np.sqrt(252)
        print(f"  S&P 500 Sharpe: {spy_sharpe:.2f}")
    else:
        spy_sharpe = None

    # Extract data for plotting
    slippages = list(sensitivity_results.keys())
    sharpes = [sensitivity_results[s]['sharpe'] for s in slippages]
    sortinos = [sensitivity_results[s]['sortino'] for s in slippages]
    annual_returns = [sensitivity_results[s]['annual_return'] * 100 for s in slippages]
    slippage_costs = [sensitivity_results[s]['slippage_pct'] for s in slippages]

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sharpe Ratio vs Slippage (main plot)
    ax1 = axes[0, 0]
    ax1.plot(slippages, sharpes, 'bo-', markersize=10, linewidth=2, label='Strategy Sharpe')
    ax1.scatter(slippages, sharpes, s=100, c='blue', zorder=5)

    # Add S&P 500 reference line
    if spy_sharpe is not None:
        ax1.axhline(y=spy_sharpe, color='gray', linestyle='--', linewidth=2,
                   label=f'S&P 500 Sharpe ({spy_sharpe:.2f})')
        ax1.fill_between(slippages, spy_sharpe, sharpes,
                        where=[s > spy_sharpe for s in sharpes],
                        color='green', alpha=0.2, label='Outperformance')
        ax1.fill_between(slippages, spy_sharpe, sharpes,
                        where=[s <= spy_sharpe for s in sharpes],
                        color='red', alpha=0.2)

    # Add value labels
    for i, (slip, sharpe) in enumerate(zip(slippages, sharpes)):
        ax1.annotate(f'{sharpe:.2f}', (slip, sharpe), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Slippage (bps)', fontsize=11)
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_title('Sharpe Ratio vs Transaction Costs', fontsize=14, fontweight='bold')
    ax1.set_xticks(slippages)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max(sharpes) * 1.2)

    # 2. Sortino Ratio vs Slippage
    ax2 = axes[0, 1]
    ax2.plot(slippages, sortinos, 'mo-', markersize=10, linewidth=2, label='Strategy Sortino')
    ax2.scatter(slippages, sortinos, s=100, c='purple', zorder=5)

    # Add value labels
    for slip, sortino in zip(slippages, sortinos):
        ax2.annotate(f'{sortino:.2f}', (slip, sortino), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Slippage (bps)', fontsize=11)
    ax2.set_ylabel('Sortino Ratio', fontsize=11)
    ax2.set_title('Sortino Ratio vs Transaction Costs', fontsize=14, fontweight='bold')
    ax2.set_xticks(slippages)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, max(sortinos) * 1.2)

    # 3. Annual Return vs Slippage
    ax3 = axes[1, 0]
    ax3.plot(slippages, annual_returns, 'go-', markersize=10, linewidth=2, label='Annual Return')
    ax3.scatter(slippages, annual_returns, s=100, c='green', zorder=5)

    # Add value labels
    for slip, ret in zip(slippages, annual_returns):
        ax3.annotate(f'{ret:.1f}%', (slip, ret), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    ax3.set_xlabel('Slippage (bps)', fontsize=11)
    ax3.set_ylabel('Annual Return (%)', fontsize=11)
    ax3.set_title('Annual Return vs Transaction Costs', fontsize=14, fontweight='bold')
    ax3.set_xticks(slippages)
    ax3.legend(loc='upper right')

    # 4. Cumulative Slippage Cost vs Slippage Rate
    ax4 = axes[1, 1]
    ax4.bar(slippages, slippage_costs, color='coral', edgecolor='darkred', alpha=0.7)

    # Add value labels on bars
    for slip, cost in zip(slippages, slippage_costs):
        ax4.annotate(f'{cost:.1f}%', (slip, cost), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=9, fontweight='bold')

    ax4.set_xlabel('Slippage (bps)', fontsize=11)
    ax4.set_ylabel('Total Slippage Cost (% of Capital)', fontsize=11)
    ax4.set_title('Cumulative Slippage Cost Over Backtest', fontsize=14, fontweight='bold')
    ax4.set_xticks(slippages)

    plt.tight_layout()
    plt.savefig(output_dir / 'slippage_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: slippage_sensitivity.png")

    return sensitivity_results, spy_sharpe


def generate_report(results, output_dir, benchmark_metrics=None):
    """Generate text report."""

    m = results['metrics']
    trades_df = results['trades_df']
    rebal_df = results['rebalance_df']
    equity_df = results['equity_df']

    report = []
    report.append("=" * 80)
    report.append("DUAL MOMENTUM STRATEGY - COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
    report.append(f"  Slippage Rate:         {2:>12} bps (0.02%)")
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

    # Benchmark comparison section
    if benchmark_metrics is not None:
        bm = benchmark_metrics
        report.append("\n" + "=" * 80)
        report.append("8. BENCHMARK COMPARISON (vs S&P 500)")
        report.append("=" * 80)
        report.append(f"  Overall Beta:          {bm['overall_beta']:>12.3f}")
        report.append(f"  Upside Beta:           {bm['upside_beta']:>12.3f}")
        report.append(f"  Downside Beta:         {bm['downside_beta']:>12.3f}")
        report.append(f"  Beta Asymmetry:        {bm['upside_beta'] - bm['downside_beta']:>12.3f} (upside - downside)")
        report.append("")
        report.append(f"  Upside Capture:        {bm['upside_capture']:>12.1f}%")
        report.append(f"  Downside Capture:      {bm['downside_capture']:>12.1f}%")
        if bm['downside_capture'] and bm['downside_capture'] != 0:
            capture_ratio = bm['upside_capture'] / bm['downside_capture']
            report.append(f"  Capture Ratio:         {capture_ratio:>12.2f}x")
        report.append("")
        report.append(f"  Correlation:           {bm['correlation']:>12.3f}")
        report.append(f"  Alpha (annualized):    {bm['alpha']*100:>12.2f}%")
        report.append(f"  Information Ratio:     {bm['info_ratio']:>12.2f}")
        report.append(f"  Tracking Error:        {bm['tracking_error']*100:>12.1f}%")

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

    # Generate rolling metrics plot
    print("\nGenerating rolling metrics plot...")
    create_rolling_metrics_plot(results, OUTPUT_DIR)

    # Load benchmark data and generate comparison
    print("\nLoading S&P 500 benchmark data...")
    spy_data = loader.get_spy()
    benchmark_metrics = None

    if spy_data is not None and len(spy_data) > 0:
        benchmark_returns = spy_data['close'].pct_change().dropna()
        print(f"Loaded SPY data: {len(benchmark_returns)} days")

        print("Generating benchmark comparison plot...")
        benchmark_metrics = create_benchmark_comparison_plot(results, benchmark_returns, OUTPUT_DIR)

        if benchmark_metrics:
            print("\n" + "=" * 50)
            print("BENCHMARK COMPARISON SUMMARY (vs S&P 500)")
            print("=" * 50)
            print(f"  Overall Beta:      {benchmark_metrics['overall_beta']:.3f}")
            print(f"  Upside Beta:       {benchmark_metrics['upside_beta']:.3f}")
            print(f"  Downside Beta:     {benchmark_metrics['downside_beta']:.3f}")
            print(f"  Upside Capture:    {benchmark_metrics['upside_capture']:.1f}%")
            print(f"  Downside Capture:  {benchmark_metrics['downside_capture']:.1f}%")
            print(f"  Alpha:             {benchmark_metrics['alpha']*100:.2f}%")
            print("=" * 50)
        # Slippage sensitivity analysis
        print("\n" + "=" * 50)
        print("SLIPPAGE SENSITIVITY ANALYSIS")
        print("=" * 50)
        sensitivity_results, spy_sharpe = create_slippage_sensitivity_plot(
            loader, symbols, benchmark_returns, OUTPUT_DIR
        )

        # Print summary table
        print("\nSlippage Sensitivity Summary:")
        print(f"{'Slippage (bps)':<15} {'Sharpe':<10} {'Sortino':<10} {'Annual Ret':<12}")
        print("-" * 47)
        for slip, data in sensitivity_results.items():
            print(f"{slip:<15} {data['sharpe']:<10.2f} {data['sortino']:<10.2f} {data['annual_return']*100:<12.1f}%")
        if spy_sharpe:
            print(f"\nS&P 500 Sharpe: {spy_sharpe:.2f}")
        print("=" * 50)

    else:
        print("Warning: Could not load S&P 500 benchmark data")
        benchmark_returns = None

    # Generate report
    print("\nGenerating report...")
    report = generate_report(results, OUTPUT_DIR, benchmark_metrics)

    # Print report to console
    print("\n" + report)

    print(f"\n\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
