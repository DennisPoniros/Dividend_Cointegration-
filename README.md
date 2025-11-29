# Dual Momentum Strategy for Dividend Stocks

A momentum strategy applied to dividend-paying stocks, based on Gary Antonacci's Dual Momentum research.

## Strategy Overview

The strategy combines two momentum filters:

1. **Absolute Momentum**: Only invest when 12-month return > 0
2. **Relative Momentum**: Rank qualifying stocks by momentum, buy the top performers
3. **Quality Filter**: Require momentum > 0.75 × volatility (reduces false signals)
4. **Cash Safety**: Hold cash when no stocks qualify

### Current Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback | 252 days | 12-month momentum calculation |
| Rebalance | 14 days | Bi-weekly portfolio rebalancing |
| Holdings | 12 | Top 12 momentum stocks |
| Leverage | 2.0x | Leveraged exposure |
| Slippage | 2 bps | Transaction cost assumption |
| Quality Filter | momentum > 0.75 × vol | Filters low signal-to-noise stocks |

## Backtest Results

| Metric | Value |
|--------|-------|
| Total Return | 1,708.80% |
| Annual Return | 60.65% |
| Annual Volatility | 44.03% |
| Sharpe Ratio | 1.33 |
| Sortino Ratio | 2.00 |
| Calmar Ratio | 1.85 |
| Max Drawdown | -32.81% |
| Win Rate | 57.1% |
| Profit Factor | 1.46 |
| Backtest Period | 2020-03-19 to 2025-11-28 |

## Important Caveats

**Backtesting Limitations:**
- The backtest starts near the March 2020 COVID bottom—one of the best starting points for momentum strategies in recent history
- Results reflect 2x leverage during an exceptional bull market
- High turnover (4,242x capital) means transaction costs significantly impact results
- No out-of-sample validation or adverse regime testing (extended bear/sideways markets)
- Momentum strategies historically underperform in mean-reverting, choppy markets

**Risk Considerations:**
- 2x leverage amplifies both gains and losses
- 44% annual volatility is high; expect significant swings
- Max drawdown of -32.81% occurred over just 37 days
- Real-world execution costs may exceed modeled assumptions

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
- Historical daily price data
- Dividend payment history
- Fundamental data snapshots
- Reference benchmarks (SPY)

### Update Data

```bash
python -m src.data.download_all
```

Requires Alpaca API credentials in `config/alpaca_config.yaml`.

## How It Works

1. **Load Data**: Read historical prices for all dividend stocks
2. **Calculate Momentum**: 12-month total return for each stock
3. **Calculate Volatility**: 63-day rolling volatility for quality filter
4. **Apply Filters**: Positive momentum AND momentum > 0.75 × volatility
5. **Rank & Select**: Top 12 by momentum strength, equal-weighted
6. **Apply Leverage**: 2x leveraged exposure to selected positions
7. **Rebalance**: Repeat every 14 trading days
8. **Cash Hedge**: If nothing qualifies, hold cash

## Backtesting Methodology

- **Walk-forward**: Each day only uses data available up to that point
- **No look-ahead bias**: Momentum calculated with historical data only
- **Transaction costs**: 2 bps slippage on all trades
- **Margin tracking**: Leverage and margin debt properly accounted for

## Disclaimer

This software is for educational and research purposes only. The backtested results shown here are hypothetical and do not represent actual trading. Trading involves substantial risk of loss. Past performance, especially in backtests, does not guarantee future results. The favorable backtest period (starting near COVID lows) likely overstates expected performance. Always conduct thorough due diligence and consider paper trading before deploying real capital.

## License

MIT License
