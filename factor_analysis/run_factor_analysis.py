#!/usr/bin/env python3
"""
Comprehensive Factor Analysis for 2x Leverage Dual Momentum Strategy
=====================================================================
This script performs extensive factor analysis on the daily equity data
and individual stock returns for the 132 ticker universe.

Analysis includes:
- Market Factor (Beta to SPY)
- Sector Factor Exposures
- Momentum Factor Analysis
- Volatility Factor
- Size Factor (Market Cap)
- Value/Dividend Yield Factor
- Principal Component Analysis (PCA)
- Factor Correlation Matrix
- Rolling Factor Exposures
- Risk Attribution
- Return Decomposition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PRICES_DIR = DATA_DIR / "prices"
REFERENCE_DIR = DATA_DIR / "reference"
UNIVERSE_FILE = DATA_DIR / "universe" / "universe.csv"
EQUITY_FILE = BASE_DIR / "analysis_output" / "2x_leverage" / "daily_equity.csv"
TRADES_FILE = BASE_DIR / "analysis_output" / "2x_leverage" / "trades.csv"
OUTPUT_DIR = Path(__file__).parent
PLOTS_DIR = OUTPUT_DIR / "plots"
DATA_OUT_DIR = OUTPUT_DIR / "data"
REPORTS_DIR = OUTPUT_DIR / "reports"


def load_all_data():
    """Load all required data for factor analysis."""
    print("Loading data...")

    # Load universe info
    universe = pd.read_csv(UNIVERSE_FILE)
    symbols = universe['symbol'].tolist()
    print(f"  Universe: {len(symbols)} symbols")

    # Load daily equity curve
    equity = pd.read_csv(EQUITY_FILE, parse_dates=['date'])
    equity.set_index('date', inplace=True)
    print(f"  Equity curve: {len(equity)} days from {equity.index.min()} to {equity.index.max()}")

    # Load trades
    trades = pd.read_csv(TRADES_FILE, parse_dates=['entry_date', 'exit_date'])
    print(f"  Trades: {len(trades)} total trades")

    # Load SPY for market factor
    spy = pd.read_parquet(REFERENCE_DIR / "SPY.parquet")
    spy = spy[['close']].rename(columns={'close': 'SPY'})
    print(f"  SPY: {len(spy)} days")

    # Load sector ETFs
    sector_etfs = ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLC', 'XLRE']
    sector_data = {}
    for etf in sector_etfs:
        try:
            df = pd.read_parquet(REFERENCE_DIR / f"{etf}.parquet")
            sector_data[etf] = df['close']
        except:
            print(f"  Warning: Could not load {etf}")
    sector_df = pd.DataFrame(sector_data)
    print(f"  Sector ETFs: {len(sector_data)} loaded")

    # Load all stock prices
    all_prices = {}
    for symbol in symbols:
        try:
            df = pd.read_parquet(PRICES_DIR / f"{symbol}.parquet")
            all_prices[symbol] = df['close']
        except:
            print(f"  Warning: Could not load {symbol}")
    prices_df = pd.DataFrame(all_prices)
    print(f"  Stock prices: {len(all_prices)} symbols loaded")

    return {
        'universe': universe,
        'symbols': symbols,
        'equity': equity,
        'trades': trades,
        'spy': spy,
        'sector_df': sector_df,
        'prices_df': prices_df
    }


def calculate_returns(prices_df, periods=1):
    """Calculate returns for given period."""
    return prices_df.pct_change(periods).dropna()


def calculate_strategy_returns(equity):
    """Calculate strategy returns from equity curve."""
    returns = equity['equity'].pct_change().dropna()
    return returns


def analyze_market_factor(strategy_returns, spy_returns, window=252):
    """Analyze market factor (beta) exposure."""
    print("\n=== Market Factor Analysis ===")

    # Align data
    aligned = pd.concat([strategy_returns, spy_returns], axis=1, join='inner')
    aligned.columns = ['Strategy', 'SPY']
    aligned = aligned.dropna()

    # Full period regression
    X = aligned['SPY'].values.reshape(-1, 1)
    y = aligned['Strategy'].values

    reg = LinearRegression()
    reg.fit(X, y)

    beta = reg.coef_[0]
    alpha_daily = reg.intercept_
    alpha_annual = alpha_daily * 252
    r_squared = reg.score(X, y)

    # Calculate residuals
    predicted = reg.predict(X)
    residuals = y - predicted
    tracking_error = np.std(residuals) * np.sqrt(252)
    information_ratio = alpha_annual / tracking_error if tracking_error > 0 else 0

    print(f"  Beta to SPY: {beta:.3f}")
    print(f"  Alpha (annualized): {alpha_annual*100:.2f}%")
    print(f"  R-squared: {r_squared:.3f}")
    print(f"  Tracking Error: {tracking_error*100:.2f}%")
    print(f"  Information Ratio: {information_ratio:.3f}")

    # Rolling beta
    rolling_beta = aligned['Strategy'].rolling(window).cov(aligned['SPY']) / aligned['SPY'].rolling(window).var()

    return {
        'beta': beta,
        'alpha_annual': alpha_annual,
        'r_squared': r_squared,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'aligned_returns': aligned,
        'rolling_beta': rolling_beta,
        'residuals': residuals
    }


def analyze_sector_factors(strategy_returns, sector_returns, universe):
    """Analyze sector factor exposures."""
    print("\n=== Sector Factor Analysis ===")

    # Map sectors to ETFs
    sector_to_etf = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financial Services': 'XLF',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Utilities': 'XLU',
        'Basic Materials': 'XLB',
        'Communication Services': 'XLC',
        'Real Estate': 'XLRE'
    }

    # Get sector composition of universe
    sector_counts = universe['sector'].value_counts()
    sector_weights = sector_counts / sector_counts.sum()

    print("  Universe Sector Composition:")
    for sector, weight in sector_weights.items():
        print(f"    {sector}: {weight*100:.1f}%")

    # Calculate sector factor exposures via regression
    sector_calc_returns = sector_returns.pct_change().dropna()

    # Align with strategy returns
    aligned = pd.concat([strategy_returns, sector_calc_returns], axis=1, join='inner').dropna()

    # Multi-factor regression
    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values

    reg = LinearRegression()
    reg.fit(X, y)

    sector_exposures = pd.Series(reg.coef_, index=aligned.columns[1:])

    print("\n  Sector Factor Exposures (Betas):")
    for sector, exposure in sector_exposures.sort_values(ascending=False).items():
        print(f"    {sector}: {exposure:.3f}")

    return {
        'sector_weights': sector_weights,
        'sector_exposures': sector_exposures,
        'r_squared': reg.score(X, y)
    }


def analyze_momentum_factor(trades, prices_df):
    """Analyze momentum factor characteristics."""
    print("\n=== Momentum Factor Analysis ===")

    # Analyze momentum at entry
    momentum_stats = trades['momentum_at_entry'].describe()
    print(f"  Momentum at Entry Statistics:")
    print(f"    Mean: {momentum_stats['mean']:.3f}")
    print(f"    Std: {momentum_stats['std']:.3f}")
    print(f"    Min: {momentum_stats['min']:.3f}")
    print(f"    Max: {momentum_stats['max']:.3f}")

    # Momentum vs returns relationship
    correlation = trades['momentum_at_entry'].corr(trades['return_pct'])
    print(f"\n  Momentum-Return Correlation: {correlation:.3f}")

    # Momentum quintile analysis
    trades['momentum_quintile'] = pd.qcut(trades['momentum_at_entry'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    quintile_returns = trades.groupby('momentum_quintile')['return_pct'].agg(['mean', 'std', 'count'])
    quintile_returns['sharpe'] = quintile_returns['mean'] / quintile_returns['std'] if quintile_returns['std'].any() else 0

    print("\n  Returns by Momentum Quintile:")
    print(quintile_returns.to_string())

    # Calculate 12-month momentum for all stocks
    momentum_252d = prices_df.pct_change(252).iloc[-1].dropna()

    return {
        'momentum_stats': momentum_stats,
        'momentum_return_corr': correlation,
        'quintile_returns': quintile_returns,
        'current_momentum': momentum_252d
    }


def analyze_volatility_factor(strategy_returns, prices_df, equity):
    """Analyze volatility factor."""
    print("\n=== Volatility Factor Analysis ===")

    # Strategy volatility
    daily_vol = strategy_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    print(f"  Strategy Daily Volatility: {daily_vol*100:.2f}%")
    print(f"  Strategy Annual Volatility: {annual_vol*100:.2f}%")

    # Rolling volatility (21-day and 63-day)
    rolling_vol_21 = strategy_returns.rolling(21).std() * np.sqrt(252)
    rolling_vol_63 = strategy_returns.rolling(63).std() * np.sqrt(252)

    # Volatility regimes
    vol_median = rolling_vol_63.median()
    high_vol_days = (rolling_vol_63 > vol_median).sum()
    low_vol_days = (rolling_vol_63 <= vol_median).sum()

    print(f"\n  Volatility Regimes:")
    print(f"    Median 63-day Vol: {vol_median*100:.2f}%")
    print(f"    High Vol Days: {high_vol_days}")
    print(f"    Low Vol Days: {low_vol_days}")

    # Returns in different volatility regimes
    high_vol_mask = rolling_vol_63 > vol_median
    high_vol_returns = strategy_returns[high_vol_mask].mean() * 252
    low_vol_returns = strategy_returns[~high_vol_mask].mean() * 252

    print(f"\n  Annualized Returns by Volatility Regime:")
    print(f"    High Volatility: {high_vol_returns*100:.2f}%")
    print(f"    Low Volatility: {low_vol_returns*100:.2f}%")

    # Calculate stock volatilities
    stock_vols = prices_df.pct_change().std() * np.sqrt(252)

    return {
        'daily_vol': daily_vol,
        'annual_vol': annual_vol,
        'rolling_vol_21': rolling_vol_21,
        'rolling_vol_63': rolling_vol_63,
        'vol_median': vol_median,
        'high_vol_returns': high_vol_returns,
        'low_vol_returns': low_vol_returns,
        'stock_volatilities': stock_vols
    }


def analyze_size_factor(trades, universe):
    """Analyze size factor (market cap) exposure."""
    print("\n=== Size Factor Analysis ===")

    # Market cap distribution
    mcap = universe.set_index('symbol')['market_cap']
    log_mcap = np.log10(mcap)

    print(f"  Universe Market Cap Distribution:")
    print(f"    Mean: ${mcap.mean()/1e9:.1f}B")
    print(f"    Median: ${mcap.median()/1e9:.1f}B")
    print(f"    Min: ${mcap.min()/1e9:.1f}B")
    print(f"    Max: ${mcap.max()/1e9:.1f}B")

    # Market cap of traded stocks
    traded_symbols = trades['symbol'].unique()
    traded_mcap = mcap[mcap.index.isin(traded_symbols)]

    print(f"\n  Traded Stocks Market Cap:")
    print(f"    Mean: ${traded_mcap.mean()/1e9:.1f}B")
    print(f"    Median: ${traded_mcap.median()/1e9:.1f}B")

    # Size quintiles
    universe['size_quintile'] = pd.qcut(universe['market_cap'], 5, labels=['Small', 'Q2', 'Q3', 'Q4', 'Large'])
    size_dist = universe['size_quintile'].value_counts()

    # Count trades by size
    trades_with_size = trades.merge(universe[['symbol', 'size_quintile']], on='symbol', how='left')
    trade_counts_by_size = trades_with_size['size_quintile'].value_counts()

    print(f"\n  Trade Counts by Size Quintile:")
    for q in ['Small', 'Q2', 'Q3', 'Q4', 'Large']:
        count = trade_counts_by_size.get(q, 0)
        print(f"    {q}: {count} trades")

    return {
        'market_caps': mcap,
        'traded_mcap': traded_mcap,
        'trade_counts_by_size': trade_counts_by_size
    }


def analyze_dividend_yield_factor(trades, universe):
    """Analyze dividend yield factor."""
    print("\n=== Dividend Yield Factor Analysis ===")

    div_yield = universe.set_index('symbol')['dividend_yield']

    print(f"  Universe Dividend Yield Distribution:")
    print(f"    Mean: {div_yield.mean():.2f}%")
    print(f"    Median: {div_yield.median():.2f}%")
    print(f"    Min: {div_yield.min():.2f}%")
    print(f"    Max: {div_yield.max():.2f}%")

    # Dividend yield of traded stocks
    traded_symbols = trades['symbol'].unique()
    traded_yield = div_yield[div_yield.index.isin(traded_symbols)]

    print(f"\n  Traded Stocks Dividend Yield:")
    print(f"    Mean: {traded_yield.mean():.2f}%")
    print(f"    Median: {traded_yield.median():.2f}%")

    # Yield quintiles
    universe['yield_quintile'] = pd.qcut(universe['dividend_yield'], 5, labels=['Low', 'Q2', 'Q3', 'Q4', 'High'], duplicates='drop')

    # Trades by yield quintile
    trades_with_yield = trades.merge(universe[['symbol', 'yield_quintile']], on='symbol', how='left')
    trade_counts_by_yield = trades_with_yield.groupby('yield_quintile')['return_pct'].agg(['mean', 'count'])

    print(f"\n  Trade Performance by Yield Quintile:")
    print(trade_counts_by_yield.to_string())

    return {
        'dividend_yields': div_yield,
        'traded_yield': traded_yield,
        'trade_counts_by_yield': trade_counts_by_yield
    }


def perform_pca_analysis(prices_df, n_components=10):
    """Perform Principal Component Analysis on stock returns."""
    print("\n=== Principal Component Analysis ===")

    # Calculate returns
    returns = prices_df.pct_change().dropna()

    # Remove stocks with missing data
    returns = returns.dropna(axis=1)
    print(f"  Stocks with complete data: {returns.shape[1]}")

    # Standardize returns
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)

    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(returns_scaled)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"\n  Explained Variance by Component:")
    for i in range(min(5, n_components)):
        print(f"    PC{i+1}: {explained_var[i]*100:.2f}% (Cumulative: {cumulative_var[i]*100:.2f}%)")

    # Factor loadings for top stocks
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=returns.columns
    )

    print(f"\n  Top 5 Stocks by PC1 Loading:")
    top_pc1 = loadings['PC1'].abs().nlargest(5)
    for stock, loading in top_pc1.items():
        print(f"    {stock}: {loadings.loc[stock, 'PC1']:.3f}")

    return {
        'pca': pca,
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'loadings': loadings,
        'n_components': n_components
    }


def analyze_factor_correlations(strategy_returns, spy, sector_df, prices_df):
    """Analyze correlations between different factors."""
    print("\n=== Factor Correlation Analysis ===")

    # Calculate all returns
    spy_ret = spy.pct_change().dropna()
    sector_ret = sector_df.pct_change().dropna()

    # Combine factors
    factors = pd.concat([spy_ret, sector_ret], axis=1)
    factors.columns = ['Market'] + list(sector_df.columns)

    # Align with strategy
    aligned = pd.concat([strategy_returns, factors], axis=1, join='inner').dropna()
    aligned.columns = ['Strategy'] + list(factors.columns)

    # Correlation matrix
    corr_matrix = aligned.corr()

    print("  Strategy Correlations with Factors:")
    strategy_corr = corr_matrix['Strategy'].drop('Strategy').sort_values(ascending=False)
    for factor, corr in strategy_corr.items():
        print(f"    {factor}: {corr:.3f}")

    return {
        'correlation_matrix': corr_matrix,
        'strategy_correlations': strategy_corr
    }


def analyze_risk_attribution(strategy_returns, market_results):
    """Perform risk attribution analysis."""
    print("\n=== Risk Attribution ===")

    beta = market_results['beta']
    residuals = market_results['residuals']
    aligned = market_results['aligned_returns']

    # Total variance
    total_var = strategy_returns.var()

    # Systematic variance (beta^2 * market_var)
    market_var = aligned['SPY'].var()
    systematic_var = (beta ** 2) * market_var

    # Idiosyncratic variance
    idio_var = np.var(residuals)

    # Proportions
    systematic_pct = systematic_var / total_var
    idio_pct = idio_var / total_var

    print(f"  Total Risk (Daily Std): {np.sqrt(total_var)*100:.3f}%")
    print(f"  Systematic Risk: {systematic_pct*100:.1f}%")
    print(f"  Idiosyncratic Risk: {idio_pct*100:.1f}%")

    # Annualized
    total_annual = np.sqrt(total_var * 252) * 100
    systematic_annual = np.sqrt(systematic_var * 252) * 100
    idio_annual = np.sqrt(idio_var * 252) * 100

    print(f"\n  Annualized Risk Decomposition:")
    print(f"    Total: {total_annual:.2f}%")
    print(f"    Systematic: {systematic_annual:.2f}%")
    print(f"    Idiosyncratic: {idio_annual:.2f}%")

    return {
        'total_var': total_var,
        'systematic_var': systematic_var,
        'idiosyncratic_var': idio_var,
        'systematic_pct': systematic_pct,
        'idiosyncratic_pct': idio_pct
    }


def analyze_return_decomposition(equity, market_results, trades):
    """Decompose returns into components."""
    print("\n=== Return Decomposition ===")

    # Total return
    total_return = (equity['equity'].iloc[-1] / equity['equity'].iloc[0]) - 1
    n_years = len(equity) / 252
    annual_return = (1 + total_return) ** (1/n_years) - 1

    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Annualized Return: {annual_return*100:.2f}%")
    print(f"  Investment Period: {n_years:.2f} years")

    # Alpha contribution
    alpha_contribution = market_results['alpha_annual'] * n_years

    # Beta contribution (approximate)
    aligned = market_results['aligned_returns']
    market_total_return = (1 + aligned['SPY']).prod() - 1
    beta_contribution = market_results['beta'] * market_total_return

    print(f"\n  Return Attribution (Approximate):")
    print(f"    Beta Contribution: {beta_contribution*100:.2f}%")
    print(f"    Alpha Contribution: {alpha_contribution*100:.2f}%")

    # Trade analysis
    winning_trades = trades[trades['pnl_net'] > 0]
    losing_trades = trades[trades['pnl_net'] <= 0]

    win_rate = len(winning_trades) / len(trades)
    avg_win = winning_trades['return_pct'].mean()
    avg_loss = losing_trades['return_pct'].mean()
    profit_factor = abs(winning_trades['pnl_net'].sum() / losing_trades['pnl_net'].sum()) if losing_trades['pnl_net'].sum() != 0 else float('inf')

    print(f"\n  Trade Statistics:")
    print(f"    Win Rate: {win_rate*100:.1f}%")
    print(f"    Avg Win: {avg_win:.2f}%")
    print(f"    Avg Loss: {avg_loss:.2f}%")
    print(f"    Profit Factor: {profit_factor:.2f}")

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'alpha_contribution': alpha_contribution,
        'beta_contribution': beta_contribution,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


def analyze_stock_participation(trades, universe):
    """Analyze which stocks participated in the strategy."""
    print("\n=== Stock Participation Analysis ===")

    # Unique stocks traded
    traded_symbols = trades['symbol'].unique()
    n_traded = len(traded_symbols)
    n_universe = len(universe)

    print(f"  Stocks Traded: {n_traded} / {n_universe} ({n_traded/n_universe*100:.1f}%)")

    # Top traded stocks
    trade_counts = trades['symbol'].value_counts()
    print(f"\n  Top 10 Most Traded Stocks:")
    for symbol, count in trade_counts.head(10).items():
        print(f"    {symbol}: {count} trades")

    # Best performing stocks
    stock_pnl = trades.groupby('symbol')['pnl_net'].sum().sort_values(ascending=False)
    print(f"\n  Top 10 by Total P&L:")
    for symbol, pnl in stock_pnl.head(10).items():
        print(f"    {symbol}: ${pnl:,.0f}")

    # Worst performing
    print(f"\n  Bottom 10 by Total P&L:")
    for symbol, pnl in stock_pnl.tail(10).items():
        print(f"    {symbol}: ${pnl:,.0f}")

    # Sector breakdown of trades
    trades_with_sector = trades.merge(universe[['symbol', 'sector']], on='symbol', how='left')
    sector_stats = trades_with_sector.groupby('sector').agg({
        'symbol': 'count',
        'pnl_net': 'sum',
        'return_pct': 'mean'
    }).rename(columns={'symbol': 'trade_count'})

    print(f"\n  Trade Stats by Sector:")
    print(sector_stats.sort_values('pnl_net', ascending=False).to_string())

    return {
        'traded_symbols': traded_symbols,
        'trade_counts': trade_counts,
        'stock_pnl': stock_pnl,
        'sector_stats': sector_stats
    }


# ============== PLOTTING FUNCTIONS ==============

def plot_market_factor(market_results, save_path):
    """Plot market factor analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    aligned = market_results['aligned_returns']

    # 1. Scatter plot
    ax = axes[0, 0]
    ax.scatter(aligned['SPY'] * 100, aligned['Strategy'] * 100, alpha=0.3, s=10)

    # Regression line
    x_line = np.linspace(aligned['SPY'].min(), aligned['SPY'].max(), 100)
    y_line = market_results['beta'] * x_line + market_results['alpha_annual'] / 252
    ax.plot(x_line * 100, y_line * 100, 'r-', linewidth=2,
            label=f'Beta={market_results["beta"]:.2f}, Alpha={market_results["alpha_annual"]*100:.1f}%/yr')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('SPY Daily Return (%)')
    ax.set_ylabel('Strategy Daily Return (%)')
    ax.set_title('Strategy vs Market Returns')
    ax.legend()

    # 2. Rolling beta
    ax = axes[0, 1]
    rolling_beta = market_results['rolling_beta'].dropna()
    ax.plot(rolling_beta.index, rolling_beta.values, linewidth=1)
    ax.axhline(market_results['beta'], color='r', linestyle='--', label=f'Full Period Beta: {market_results["beta"]:.2f}')
    ax.axhline(1, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(2, color='gray', linestyle=':', alpha=0.7)
    ax.fill_between(rolling_beta.index, 1, rolling_beta.values,
                    where=rolling_beta.values > 1, alpha=0.3, color='green')
    ax.fill_between(rolling_beta.index, 1, rolling_beta.values,
                    where=rolling_beta.values < 1, alpha=0.3, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling 252-day Beta')
    ax.set_title('Rolling Market Beta (252-day)')
    ax.legend()

    # 3. Residuals distribution
    ax = axes[1, 0]
    residuals = market_results['residuals'] * 100
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Fit normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
    ax.axvline(mu, color='green', linestyle='--', label=f'Mean: {mu:.3f}%')
    ax.set_xlabel('Residual Return (%)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Residuals (Alpha)')
    ax.legend()

    # 4. Cumulative residuals
    ax = axes[1, 1]
    cum_residuals = np.cumsum(market_results['residuals']) * 100
    ax.plot(aligned.index, cum_residuals, linewidth=1)
    ax.fill_between(aligned.index, 0, cum_residuals, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Alpha (%)')
    ax.set_title('Cumulative Alpha Generation')

    plt.tight_layout()
    plt.savefig(save_path / 'market_factor_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: market_factor_analysis.png")


def plot_sector_exposure(sector_results, universe, save_path):
    """Plot sector exposure analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Universe sector composition
    ax = axes[0]
    sector_weights = sector_results['sector_weights']
    colors = plt.cm.Set3(np.linspace(0, 1, len(sector_weights)))
    wedges, texts, autotexts = ax.pie(sector_weights.values, labels=sector_weights.index,
                                       autopct='%1.1f%%', colors=colors, pctdistance=0.8)
    ax.set_title('Universe Sector Composition')

    # 2. Sector factor exposures
    ax = axes[1]
    exposures = sector_results['sector_exposures'].sort_values()
    colors = ['red' if x < 0 else 'green' for x in exposures.values]
    bars = ax.barh(exposures.index, exposures.values, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Factor Exposure (Beta)')
    ax.set_title('Sector Factor Exposures')

    plt.tight_layout()
    plt.savefig(save_path / 'sector_exposure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sector_exposure.png")


def plot_momentum_analysis(momentum_results, trades, save_path):
    """Plot momentum factor analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Momentum distribution at entry
    ax = axes[0, 0]
    ax.hist(trades['momentum_at_entry'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(trades['momentum_at_entry'].mean(), color='red', linestyle='--',
               label=f'Mean: {trades["momentum_at_entry"].mean():.2f}')
    ax.axvline(trades['momentum_at_entry'].median(), color='green', linestyle='--',
               label=f'Median: {trades["momentum_at_entry"].median():.2f}')
    ax.set_xlabel('12-Month Momentum at Entry')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Momentum at Trade Entry')
    ax.legend()

    # 2. Momentum vs Return scatter
    ax = axes[0, 1]
    ax.scatter(trades['momentum_at_entry'], trades['return_pct'], alpha=0.3, s=10)

    # Regression line
    z = np.polyfit(trades['momentum_at_entry'], trades['return_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(trades['momentum_at_entry'].min(), trades['momentum_at_entry'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Momentum at Entry')
    ax.set_ylabel('Trade Return (%)')
    ax.set_title(f'Momentum vs Trade Return (corr: {momentum_results["momentum_return_corr"]:.3f})')

    # 3. Returns by momentum quintile
    ax = axes[1, 0]
    quintile_ret = momentum_results['quintile_returns']['mean']
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(quintile_ret)))
    bars = ax.bar(quintile_ret.index, quintile_ret.values, color=colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Momentum Quintile')
    ax.set_ylabel('Average Return (%)')
    ax.set_title('Average Trade Return by Momentum Quintile')

    # 4. Current momentum distribution
    ax = axes[1, 1]
    current_mom = momentum_results['current_momentum']
    ax.hist(current_mom * 100, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(current_mom.mean() * 100, color='red', linestyle='--',
               label=f'Mean: {current_mom.mean()*100:.1f}%')
    ax.set_xlabel('12-Month Momentum (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Current 12-Month Momentum Distribution (All 132 Stocks)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path / 'momentum_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: momentum_analysis.png")


def plot_volatility_analysis(vol_results, strategy_returns, save_path):
    """Plot volatility analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rolling volatility
    ax = axes[0, 0]
    ax.plot(vol_results['rolling_vol_21'].index, vol_results['rolling_vol_21'].values * 100,
            alpha=0.5, label='21-day', linewidth=1)
    ax.plot(vol_results['rolling_vol_63'].index, vol_results['rolling_vol_63'].values * 100,
            label='63-day', linewidth=1.5)
    ax.axhline(vol_results['annual_vol'] * 100, color='red', linestyle='--',
               label=f'Full Period: {vol_results["annual_vol"]*100:.1f}%')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility (%)')
    ax.set_title('Rolling Volatility')
    ax.legend()

    # 2. Returns distribution
    ax = axes[0, 1]
    returns_pct = strategy_returns * 100
    ax.hist(returns_pct, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Normal fit
    mu, sigma = returns_pct.mean(), returns_pct.std()
    x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')

    # Stats
    skew = stats.skew(returns_pct)
    kurt = stats.kurtosis(returns_pct)
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Density')
    ax.set_title(f'Return Distribution (Skew: {skew:.2f}, Kurtosis: {kurt:.2f})')
    ax.legend()

    # 3. Volatility vs Returns scatter (monthly)
    ax = axes[1, 0]
    monthly_ret = strategy_returns.resample('ME').apply(lambda x: (1+x).prod()-1) * 100
    monthly_vol = strategy_returns.resample('ME').std() * np.sqrt(21) * 100
    ax.scatter(monthly_vol, monthly_ret, alpha=0.7, s=50)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(monthly_vol.mean(), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Monthly Volatility (%)')
    ax.set_ylabel('Monthly Return (%)')
    ax.set_title('Monthly Return vs Volatility')

    # 4. Stock volatility distribution
    ax = axes[1, 1]
    stock_vols = vol_results['stock_volatilities'] * 100
    ax.hist(stock_vols.dropna(), bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(stock_vols.mean(), color='red', linestyle='--',
               label=f'Mean: {stock_vols.mean():.1f}%')
    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Volatility Distribution (132 Universe Stocks)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path / 'volatility_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: volatility_analysis.png")


def plot_size_factor(size_results, universe, save_path):
    """Plot size factor analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Market cap distribution
    ax = axes[0]
    mcap_billions = size_results['market_caps'] / 1e9
    ax.hist(mcap_billions, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(mcap_billions.mean(), color='red', linestyle='--',
               label=f'Mean: ${mcap_billions.mean():.0f}B')
    ax.axvline(mcap_billions.median(), color='green', linestyle='--',
               label=f'Median: ${mcap_billions.median():.0f}B')
    ax.set_xlabel('Market Cap ($B)')
    ax.set_ylabel('Frequency')
    ax.set_title('Market Cap Distribution (132 Stocks)')
    ax.legend()

    # 2. Trade counts by size
    ax = axes[1]
    trade_counts = size_results['trade_counts_by_size']
    size_order = ['Small', 'Q2', 'Q3', 'Q4', 'Large']
    counts = [trade_counts.get(s, 0) for s in size_order]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(size_order)))
    ax.bar(size_order, counts, color=colors, edgecolor='black')
    ax.set_xlabel('Size Quintile')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trade Frequency by Size Quintile')

    plt.tight_layout()
    plt.savefig(save_path / 'size_factor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: size_factor.png")


def plot_dividend_yield_factor(yield_results, save_path):
    """Plot dividend yield factor analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Dividend yield distribution
    ax = axes[0]
    div_yields = yield_results['dividend_yields']
    ax.hist(div_yields, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(div_yields.mean(), color='red', linestyle='--',
               label=f'Mean: {div_yields.mean():.2f}%')
    ax.axvline(div_yields.median(), color='green', linestyle='--',
               label=f'Median: {div_yields.median():.2f}%')
    ax.set_xlabel('Dividend Yield (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Dividend Yield Distribution (132 Stocks)')
    ax.legend()

    # 2. Returns by yield quintile
    ax = axes[1]
    yield_returns = yield_results['trade_counts_by_yield']
    if len(yield_returns) > 0:
        ax.bar(yield_returns.index.astype(str), yield_returns['mean'],
               color='steelblue', edgecolor='black', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Dividend Yield Quintile')
        ax.set_ylabel('Average Trade Return (%)')
        ax.set_title('Average Trade Return by Dividend Yield Quintile')

    plt.tight_layout()
    plt.savefig(save_path / 'dividend_yield_factor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dividend_yield_factor.png")


def plot_pca_analysis(pca_results, save_path):
    """Plot PCA analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n = pca_results['n_components']
    explained = pca_results['explained_variance']
    cumulative = pca_results['cumulative_variance']
    loadings = pca_results['loadings']

    # 1. Explained variance
    ax = axes[0, 0]
    ax.bar(range(1, n+1), explained * 100, alpha=0.7, label='Individual', edgecolor='black')
    ax.plot(range(1, n+1), cumulative * 100, 'ro-', linewidth=2, label='Cumulative')
    ax.axhline(80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Explained Variance')
    ax.legend()
    ax.set_xticks(range(1, n+1))

    # 2. PC1 vs PC2 loadings
    ax = axes[0, 1]
    ax.scatter(loadings['PC1'], loadings['PC2'], alpha=0.5, s=30)

    # Label top stocks
    top_pc1 = loadings['PC1'].abs().nlargest(5).index
    for stock in top_pc1:
        ax.annotate(stock, (loadings.loc[stock, 'PC1'], loadings.loc[stock, 'PC2']),
                   fontsize=8, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('PC1 Loading')
    ax.set_ylabel('PC2 Loading')
    ax.set_title('Stock Loadings on PC1 vs PC2')

    # 3. Top loadings for PC1
    ax = axes[1, 0]
    top_10_pc1 = loadings['PC1'].abs().nlargest(10)
    colors = ['green' if loadings.loc[s, 'PC1'] > 0 else 'red' for s in top_10_pc1.index]
    ax.barh([loadings.loc[s, 'PC1'] for s in top_10_pc1.index],
            range(len(top_10_pc1)), color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_10_pc1)))
    ax.set_yticklabels(top_10_pc1.index)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('PC1 Loading')
    ax.set_title('Top 10 Stocks by PC1 Loading (Market Factor)')

    # 4. Top loadings for PC2
    ax = axes[1, 1]
    top_10_pc2 = loadings['PC2'].abs().nlargest(10)
    colors = ['green' if loadings.loc[s, 'PC2'] > 0 else 'red' for s in top_10_pc2.index]
    ax.barh([loadings.loc[s, 'PC2'] for s in top_10_pc2.index],
            range(len(top_10_pc2)), color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_10_pc2)))
    ax.set_yticklabels(top_10_pc2.index)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('PC2 Loading')
    ax.set_title('Top 10 Stocks by PC2 Loading')

    plt.tight_layout()
    plt.savefig(save_path / 'pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pca_analysis.png")


def plot_factor_correlations(corr_results, save_path):
    """Plot factor correlation matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))

    corr_matrix = corr_results['correlation_matrix']

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('Factor Correlation Matrix')

    plt.tight_layout()
    plt.savefig(save_path / 'factor_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: factor_correlations.png")


def plot_risk_attribution(risk_results, save_path):
    """Plot risk attribution pie chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Risk decomposition pie
    ax = axes[0]
    labels = ['Systematic\n(Market)', 'Idiosyncratic\n(Alpha)']
    sizes = [risk_results['systematic_pct'], risk_results['idiosyncratic_pct']]
    colors = ['steelblue', 'coral']
    explode = (0.05, 0.05)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title('Variance Decomposition')

    # 2. Risk bar chart
    ax = axes[1]
    risk_types = ['Total', 'Systematic', 'Idiosyncratic']
    annual_risks = [
        np.sqrt(risk_results['total_var'] * 252) * 100,
        np.sqrt(risk_results['systematic_var'] * 252) * 100,
        np.sqrt(risk_results['idiosyncratic_var'] * 252) * 100
    ]
    colors = ['gray', 'steelblue', 'coral']
    ax.bar(risk_types, annual_risks, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Annualized Volatility (%)')
    ax.set_title('Risk Decomposition (Annualized)')

    for i, v in enumerate(annual_risks):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path / 'risk_attribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: risk_attribution.png")


def plot_stock_participation(participation_results, save_path):
    """Plot stock participation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Top traded stocks
    ax = axes[0, 0]
    top_traded = participation_results['trade_counts'].head(15)
    ax.barh(range(len(top_traded)), top_traded.values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_traded)))
    ax.set_yticklabels(top_traded.index)
    ax.set_xlabel('Number of Trades')
    ax.set_title('Top 15 Most Traded Stocks')
    ax.invert_yaxis()

    # 2. Top P&L contributors
    ax = axes[0, 1]
    top_pnl = participation_results['stock_pnl'].head(15)
    colors = ['green' if x > 0 else 'red' for x in top_pnl.values]
    ax.barh(range(len(top_pnl)), top_pnl.values / 1000, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_pnl)))
    ax.set_yticklabels(top_pnl.index)
    ax.set_xlabel('Total P&L ($K)')
    ax.set_title('Top 15 P&L Contributors')
    ax.invert_yaxis()

    # 3. Worst P&L contributors
    ax = axes[1, 0]
    bottom_pnl = participation_results['stock_pnl'].tail(15)
    colors = ['green' if x > 0 else 'red' for x in bottom_pnl.values]
    ax.barh(range(len(bottom_pnl)), bottom_pnl.values / 1000, color=colors, alpha=0.7)
    ax.set_yticks(range(len(bottom_pnl)))
    ax.set_yticklabels(bottom_pnl.index)
    ax.set_xlabel('Total P&L ($K)')
    ax.set_title('Bottom 15 P&L Contributors')
    ax.invert_yaxis()

    # 4. Sector stats
    ax = axes[1, 1]
    sector_stats = participation_results['sector_stats'].sort_values('pnl_net', ascending=True)
    colors = ['green' if x > 0 else 'red' for x in sector_stats['pnl_net'].values]
    ax.barh(range(len(sector_stats)), sector_stats['pnl_net'].values / 1000,
            color=colors, alpha=0.7)
    ax.set_yticks(range(len(sector_stats)))
    ax.set_yticklabels(sector_stats.index)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Total P&L ($K)')
    ax.set_title('P&L by Sector')

    plt.tight_layout()
    plt.savefig(save_path / 'stock_participation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: stock_participation.png")


def plot_equity_and_drawdown(equity, save_path):
    """Plot equity curve and drawdown."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Calculate drawdown
    equity_vals = equity['equity']
    rolling_max = equity_vals.cummax()
    drawdown = (equity_vals - rolling_max) / rolling_max * 100

    # 1. Equity curve
    ax = axes[0]
    ax.plot(equity.index, equity_vals / 1e6, linewidth=1.5, color='steelblue')
    ax.fill_between(equity.index, 1, equity_vals / 1e6, alpha=0.3)
    ax.set_ylabel('Portfolio Value ($M)')
    ax.set_title('2x Leverage Equity Curve')
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax.legend()

    # 2. Drawdown
    ax = axes[1]
    ax.fill_between(equity.index, 0, drawdown, color='red', alpha=0.5)
    ax.plot(equity.index, drawdown, linewidth=1, color='darkred')
    ax.axhline(-10, color='orange', linestyle='--', alpha=0.7, label='-10%')
    ax.axhline(-20, color='red', linestyle='--', alpha=0.7, label='-20%')
    ax.axhline(-30, color='darkred', linestyle='--', alpha=0.7, label='-30%')
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Date')
    ax.set_title('Drawdown')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path / 'equity_drawdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: equity_drawdown.png")


def plot_monthly_heatmap(strategy_returns, save_path):
    """Plot monthly returns heatmap."""
    # Calculate monthly returns
    monthly = strategy_returns.resample('ME').apply(lambda x: (1+x).prod()-1) * 100

    # Create pivot table
    monthly_df = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Return': monthly.values
    })

    pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                vmin=-20, vmax=20, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Monthly Return (%)'})
    ax.set_title('Monthly Returns Heatmap (%)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')

    plt.tight_layout()
    plt.savefig(save_path / 'monthly_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: monthly_heatmap.png")


def plot_rolling_metrics(strategy_returns, spy_returns, save_path):
    """Plot rolling performance metrics."""
    # Align returns
    aligned = pd.concat([strategy_returns, spy_returns], axis=1, join='inner').dropna()
    aligned.columns = ['Strategy', 'SPY']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1. Rolling Sharpe ratio (assuming 0 risk-free rate)
    ax = axes[0]
    rolling_sharpe = (aligned['Strategy'].rolling(252).mean() /
                      aligned['Strategy'].rolling(252).std()) * np.sqrt(252)
    spy_sharpe = (aligned['SPY'].rolling(252).mean() /
                  aligned['SPY'].rolling(252).std()) * np.sqrt(252)
    ax.plot(rolling_sharpe.index, rolling_sharpe, label='Strategy', linewidth=1.5)
    ax.plot(spy_sharpe.index, spy_sharpe, label='SPY', linewidth=1, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(1, color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('Rolling 252-day Sharpe')
    ax.set_title('Rolling Sharpe Ratio')
    ax.legend()

    # 2. Rolling returns
    ax = axes[1]
    rolling_ret_strat = aligned['Strategy'].rolling(252).apply(lambda x: (1+x).prod()-1) * 100
    rolling_ret_spy = aligned['SPY'].rolling(252).apply(lambda x: (1+x).prod()-1) * 100
    ax.plot(rolling_ret_strat.index, rolling_ret_strat, label='Strategy', linewidth=1.5)
    ax.plot(rolling_ret_spy.index, rolling_ret_spy, label='SPY', linewidth=1, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(rolling_ret_strat.index, rolling_ret_spy, rolling_ret_strat,
                    where=rolling_ret_strat > rolling_ret_spy, alpha=0.3, color='green')
    ax.fill_between(rolling_ret_strat.index, rolling_ret_spy, rolling_ret_strat,
                    where=rolling_ret_strat <= rolling_ret_spy, alpha=0.3, color='red')
    ax.set_ylabel('Rolling 252-day Return (%)')
    ax.set_title('Rolling 1-Year Returns')
    ax.legend()

    # 3. Rolling excess return
    ax = axes[2]
    excess = rolling_ret_strat - rolling_ret_spy
    ax.fill_between(excess.index, 0, excess, where=excess > 0, alpha=0.5, color='green', label='Outperformance')
    ax.fill_between(excess.index, 0, excess, where=excess <= 0, alpha=0.5, color='red', label='Underperformance')
    ax.plot(excess.index, excess, color='black', linewidth=0.5)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel('Excess Return (%)')
    ax.set_xlabel('Date')
    ax.set_title('Rolling 1-Year Excess Return vs SPY')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path / 'rolling_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: rolling_metrics.png")


def plot_trade_analysis(trades, save_path):
    """Plot trade analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Return distribution
    ax = axes[0, 0]
    ax.hist(trades['return_pct'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linewidth=1)
    ax.axvline(trades['return_pct'].mean(), color='green', linestyle='--',
               label=f'Mean: {trades["return_pct"].mean():.2f}%')
    ax.set_xlabel('Trade Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Trade Return Distribution')
    ax.legend()

    # 2. P&L distribution
    ax = axes[0, 1]
    pnl_k = trades['pnl_net'] / 1000
    ax.hist(pnl_k, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linewidth=1)
    ax.axvline(pnl_k.mean(), color='green', linestyle='--',
               label=f'Mean: ${pnl_k.mean():.1f}K')
    ax.set_xlabel('Trade P&L ($K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Trade P&L Distribution')
    ax.legend()

    # 3. Holding period vs return
    ax = axes[1, 0]
    ax.scatter(trades['hold_days'], trades['return_pct'], alpha=0.3, s=20)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Holding Period (Days)')
    ax.set_ylabel('Return (%)')
    ax.set_title('Holding Period vs Return')

    # 4. Cumulative P&L over time
    ax = axes[1, 1]
    trades_sorted = trades.sort_values('exit_date')
    cum_pnl = trades_sorted['pnl_net'].cumsum() / 1e6
    ax.plot(trades_sorted['exit_date'], cum_pnl, linewidth=1.5)
    ax.fill_between(trades_sorted['exit_date'], 0, cum_pnl, alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L ($M)')
    ax.set_title('Cumulative Trade P&L')

    plt.tight_layout()
    plt.savefig(save_path / 'trade_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: trade_analysis.png")


def save_factor_data(results, save_path):
    """Save factor analysis data to CSV files."""
    print("\nSaving data files...")

    # Market factor
    market_df = pd.DataFrame({
        'Metric': ['Beta', 'Alpha (Annual)', 'R-Squared', 'Tracking Error', 'Information Ratio'],
        'Value': [
            results['market']['beta'],
            results['market']['alpha_annual'],
            results['market']['r_squared'],
            results['market']['tracking_error'],
            results['market']['information_ratio']
        ]
    })
    market_df.to_csv(save_path / 'market_factor_metrics.csv', index=False)

    # Sector exposures
    sector_df = pd.DataFrame({
        'Sector_ETF': results['sector']['sector_exposures'].index,
        'Exposure': results['sector']['sector_exposures'].values
    })
    sector_df.to_csv(save_path / 'sector_exposures.csv', index=False)

    # Risk attribution
    risk_df = pd.DataFrame({
        'Component': ['Total', 'Systematic', 'Idiosyncratic'],
        'Variance': [
            results['risk']['total_var'],
            results['risk']['systematic_var'],
            results['risk']['idiosyncratic_var']
        ],
        'Percentage': [
            1.0,
            results['risk']['systematic_pct'],
            results['risk']['idiosyncratic_pct']
        ]
    })
    risk_df.to_csv(save_path / 'risk_attribution.csv', index=False)

    # PCA loadings
    results['pca']['loadings'].to_csv(save_path / 'pca_loadings.csv')

    # Stock participation
    stock_pnl = pd.DataFrame({
        'Symbol': results['participation']['stock_pnl'].index,
        'Total_PnL': results['participation']['stock_pnl'].values
    })
    stock_pnl.to_csv(save_path / 'stock_pnl.csv', index=False)

    # Factor correlations
    results['correlations']['correlation_matrix'].to_csv(save_path / 'factor_correlations.csv')

    print("  Data files saved successfully")


def main():
    """Main function to run factor analysis."""
    print("=" * 70)
    print("COMPREHENSIVE FACTOR ANALYSIS - 2x LEVERAGE DUAL MOMENTUM STRATEGY")
    print("=" * 70)

    # Create output directories
    PLOTS_DIR.mkdir(exist_ok=True)
    DATA_OUT_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # Load data
    data = load_all_data()

    # Calculate returns
    strategy_returns = calculate_strategy_returns(data['equity'])
    spy_returns = data['spy'].pct_change().dropna()['SPY']

    # Run all analyses
    results = {}

    results['market'] = analyze_market_factor(strategy_returns, spy_returns)
    results['sector'] = analyze_sector_factors(strategy_returns, data['sector_df'], data['universe'])
    results['momentum'] = analyze_momentum_factor(data['trades'], data['prices_df'])
    results['volatility'] = analyze_volatility_factor(strategy_returns, data['prices_df'], data['equity'])
    results['size'] = analyze_size_factor(data['trades'], data['universe'])
    results['dividend'] = analyze_dividend_yield_factor(data['trades'], data['universe'])
    results['pca'] = perform_pca_analysis(data['prices_df'])
    results['correlations'] = analyze_factor_correlations(strategy_returns, data['spy'], data['sector_df'], data['prices_df'])
    results['risk'] = analyze_risk_attribution(strategy_returns, results['market'])
    results['returns'] = analyze_return_decomposition(data['equity'], results['market'], data['trades'])
    results['participation'] = analyze_stock_participation(data['trades'], data['universe'])

    # Generate plots
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    plot_equity_and_drawdown(data['equity'], PLOTS_DIR)
    plot_market_factor(results['market'], PLOTS_DIR)
    plot_sector_exposure(results['sector'], data['universe'], PLOTS_DIR)
    plot_momentum_analysis(results['momentum'], data['trades'], PLOTS_DIR)
    plot_volatility_analysis(results['volatility'], strategy_returns, PLOTS_DIR)
    plot_size_factor(results['size'], data['universe'], PLOTS_DIR)
    plot_dividend_yield_factor(results['dividend'], PLOTS_DIR)
    plot_pca_analysis(results['pca'], PLOTS_DIR)
    plot_factor_correlations(results['correlations'], PLOTS_DIR)
    plot_risk_attribution(results['risk'], PLOTS_DIR)
    plot_stock_participation(results['participation'], PLOTS_DIR)
    plot_monthly_heatmap(strategy_returns, PLOTS_DIR)
    plot_rolling_metrics(strategy_returns, spy_returns, PLOTS_DIR)
    plot_trade_analysis(data['trades'], PLOTS_DIR)

    # Save data files
    save_factor_data(results, DATA_OUT_DIR)

    print("\n" + "=" * 70)
    print("FACTOR ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Plots: {PLOTS_DIR}")
    print(f"  - Data: {DATA_OUT_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")

    return results


if __name__ == "__main__":
    results = main()
