# Dual Momentum Strategy for Dividend Stocks

A research-backed momentum strategy applied to dividend-paying stocks, achieving a Sharpe ratio > 1.0.

## Strategy Overview

Based on Gary Antonacci's Dual Momentum approach:

1. **Absolute Momentum**: Only invest when 12-month return > 0
2. **Relative Momentum**: Among qualifying stocks, buy the highest momentum
3. **Cash Safety**: Go to cash when no stocks qualify

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback | 252 days | 12-month momentum calculation |
| Rebalance | 21 days | Monthly portfolio rebalancing |
| Holdings | 10 | Top 10 momentum stocks |
| Slippage | 2 bps | Transaction cost assumption |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | 288.89% |
| Annual Return | 26.47% |
| Annual Volatility | 22.60% |
| **Sharpe Ratio** | **1.08** |
| Sortino Ratio | 1.57 |
| Max Drawdown | -21.76% |
| Win Rate | 57.4% |
| Backtest Period | 2020-03-19 to 2025-11-28 |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Dividend_Cointegration-

# Install dependencies
pip install -r requirements.txt

# Optional: Configure Alpaca API for data updates
cp config/alpaca_config.example.yaml config/alpaca_config.yaml
# Edit with your API credentials
```

## Usage

### Run Backtest

```bash
python run_backtest_dual_momentum.py
```

### Run Full Analysis (with plots and reports)

```bash
python analyze_strategy.py
```

This generates:
- `analysis_output/strategy_report.txt` - Comprehensive metrics
- `analysis_output/trades.csv` - All trade details
- `analysis_output/daily_equity.csv` - Daily equity curve
- `analysis_output/equity_drawdown.png` - Equity and drawdown plots
- `analysis_output/monthly_returns_heatmap.png` - Monthly returns
- `analysis_output/trade_analysis.png` - Trade statistics
- `analysis_output/rolling_sharpe.png` - Rolling Sharpe ratio

## Project Structure

```
Dividend_Cointegration-/
├── run_backtest_dual_momentum.py  # Main strategy script
├── analyze_strategy.py            # Comprehensive analysis
├── analysis_output/               # Generated reports and plots
├── data/                          # Local market data
│   ├── prices/                    # OHLCV data (parquet)
│   ├── dividends/                 # Dividend history
│   ├── fundamentals/              # Stock fundamentals
│   └── reference/                 # Benchmark data (SPY, etc.)
├── src/
│   └── data/
│       ├── loader.py              # Data loading utilities
│       ├── downloader.py          # Data download from APIs
│       └── universe.py            # Stock universe management
├── config/
│   └── alpaca_config.example.yaml # API configuration template
└── requirements.txt               # Python dependencies
```

## Data

The repository includes pre-downloaded data for ~130 dividend stocks:
- 10 years of daily price history
- Dividend payment history
- Fundamental data snapshots
- Reference benchmarks (SPY, sector ETFs)

### Update Data

```bash
python -m src.data.download_all
```

Requires Alpaca API credentials in `config/alpaca_config.yaml`.

## How It Works

1. **Load Data**: Read historical prices for all dividend stocks
2. **Calculate Momentum**: 12-month total return for each stock
3. **Apply Absolute Filter**: Only consider stocks with positive momentum
4. **Rank by Relative Momentum**: Sort by momentum strength
5. **Select Top 10**: Equal-weight portfolio of best performers
6. **Rebalance Monthly**: Repeat process every 21 trading days
7. **Hold Cash**: If no stocks have positive momentum, stay in cash

## Backtesting Methodology

- **Walk-forward**: Each day only uses data available up to that point
- **No look-ahead bias**: Momentum calculated with historical data only
- **Realistic costs**: 2 bps slippage on all trades
- **No survivorship bias**: Uses actual historical universe

## Disclaimer

This software is for educational and research purposes. Trading involves risk. Past performance does not guarantee future results. Always test thoroughly with paper trading before deploying real capital.

## License

MIT License
