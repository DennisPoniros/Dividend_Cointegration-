# Dividend Cointegration Trading Strategy

A sophisticated statistical arbitrage system that identifies and trades cointegrated baskets of dividend-paying stocks.

## Overview

This strategy implements a complete dividend cointegration trading system with:

- **Rigorous statistical testing** (Johansen, Engle-Granger, validation battery)
- **Multi-layer risk management** (position-level stops, portfolio circuit breakers)
- **Interactive dashboard** for monitoring and parameter tuning
- **Comprehensive backtesting** with realistic transaction costs
- **IBKR integration** for real market data

## Strategy Outline

### Step 1: Universe Construction
- Market cap > $1B
- Minimum 5 consecutive years of dividend payments
- Average daily volume > $10M
- Excludes REITs, MLPs, special dividends only
- **Update frequency:** Quarterly

### Step 2: Sub-Basket Formation (Hybrid Sector-Correlation Clustering)
1. Initial Sector Grouping (GICS Level 3, minimum 15 stocks per sector)
2. Within-Sector Correlation Clustering (ρ ≥ 0.85, 252-day rolling)
3. Dividend Trait Screening:
   - Dividend growth std < 15%
   - Payout ratio: 30-70%
   - Yield range: 1.5-6%

**Output:** ~20-30 sub-baskets of 6-10 stocks each

### Step 3: Cointegration Vector Selection
**Testing Period:** 504 trading days (2 years)

For each sub-basket:
- **Johansen Test** (primary)
  - Trace statistic p-value < 0.01
  - Rank = 1
  - Largest eigenvalue > 0.15

- **Validation Battery** (all must pass):
  - Engle-Granger ADF < -3.5
  - Half-life: 5-35 trading days
  - Hurst exponent < 0.45
  - Out-of-sample test (last 126 days)

- **Stability Test:**
  - Cointegrated in all 4 quarters
  - Half-life coefficient of variation < 0.4

**Expected Output:** 10-20 robust cointegration vectors

### Step 4: Portfolio Construction
- Equal risk contribution across vectors
- Position sizing: σ_basket × z-score × Kelly_fractional
- Kelly fraction: 0.25 (quarter-Kelly)
- Maximum per vector: 8% of portfolio
- Maximum total exposure: 60% of portfolio
- Market neutrality: |beta| < 0.15

### Step 5: Signal Generation
**Entry Signals:**
- Long entry: z-score ≤ -1.5σ
- Short entry: z-score ≥ +1.5σ

**Exit Signals:**
- Long exit: z-score ≥ -0.5σ OR time stop
- Short exit: z-score ≤ +0.5σ OR time stop

**Calculation:**
- z-score = (current_spread - rolling_mean) / rolling_std
- Rolling window: 42 days

### Step 6: Risk Management (Multi-Layer Hard Stops)

**Immediate Exit Triggers (ANY condition):**
- Z-score exceeds ±3.0σ
- Position held > 70 days
- Rolling ADF p-value > 0.15 for 5 consecutive days
- Half-life exceeds 50 days in 21-day rolling window
- Position drawdown > 4% of allocated capital

**Portfolio-Level Circuit Breakers:**
- 10% drawdown: Cut all position sizes by 50%
- 15% drawdown: Exit all positions, halt trading 20 days
- Daily VaR breach (95%): No new positions that day

### Step 7: Retraining Schedule
**Weekly (Every Monday):**
- Recalculate hedge ratios (rolling 42 days)
- Update z-scores and rolling statistics
- Check half-life on all active positions

**Monthly:**
- Retest cointegration for active vectors

**Quarterly:**
- Rebuild universe
- Reform baskets
- Full cointegration testing

### Step 8: Execution
- Limit orders only, 30-minute max execution window
- For positions > $50K: TWAP over 15 minutes
- Maximum 5% of ADV per stock
- Trading hours: Enter 10:00-15:00 ET, Exit 09:45-15:30 ET

## Installation

### Prerequisites
- Python 3.9+
- Interactive Brokers TWS or IB Gateway
- IBKR account with market data subscriptions

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Dividend_Cointegration-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure IBKR:
```bash
cp config/ibkr_config.example.yaml config/ibkr_config.yaml
# Edit ibkr_config.yaml with your IBKR credentials
```

## Usage

### Interactive Dashboard

Launch the Streamlit dashboard for full interactive control:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- **Universe & Data:** Build and view the stock universe
- **Basket Formation:** Analyze sector-correlation clusters
- **Cointegration Analysis:** View test results and vector statistics
- **Live Signals:** Monitor current trading signals
- **Backtesting:** Run and analyze historical performance
- **Risk Monitor:** Track active positions and risk metrics

### Command Line Usage

```python
from src.data.ibkr_connector import IBKRConnector
from src.data.universe import UniverseBuilder
from src.strategy.basket_formation import BasketFormer
from src.strategy.cointegration import CointegrationTester
from src.backtest.engine import BacktestEngine

# Connect to IBKR
with IBKRConnector() as ibkr:
    # Build universe
    universe_builder = UniverseBuilder()
    universe_df = universe_builder.build_universe(ibkr, symbols=['AAPL', 'MSFT', ...])

    # Form baskets
    basket_former = BasketFormer()
    baskets = basket_former.form_baskets(
        universe_df['symbol'].tolist(),
        price_data,
        ibkr
    )

    # Test cointegration
    coint_tester = CointegrationTester()
    vectors = coint_tester.test_all_baskets(baskets, price_data)

    # Run backtest
    backtest = BacktestEngine()
    results = backtest.run_backtest(
        vectors,
        price_data,
        initial_capital=1_000_000
    )
```

## Project Structure

```
Dividend_Cointegration-/
├── config/                      # Configuration files
│   ├── strategy_params.yaml     # Strategy parameters (tunable)
│   └── ibkr_config.yaml        # IBKR connection config
├── src/                        # Source code
│   ├── data/                   # Data acquisition
│   │   ├── ibkr_connector.py   # IBKR API wrapper
│   │   └── universe.py         # Universe construction
│   ├── strategy/               # Strategy modules
│   │   ├── basket_formation.py # Basket clustering
│   │   ├── cointegration.py    # Cointegration testing
│   │   ├── signals.py          # Signal generation
│   │   └── portfolio.py        # Portfolio management
│   ├── risk/                   # Risk management
│   │   └── risk_management.py  # Risk stops and circuit breakers
│   ├── backtest/              # Backtesting
│   │   ├── engine.py          # Backtest engine
│   │   └── metrics.py         # Performance metrics
│   └── utils/                 # Utilities
│       └── validators.py      # Data validation
├── dashboard/                 # Streamlit dashboard
│   └── app.py                # Main dashboard application
├── tests/                    # Unit tests
└── requirements.txt          # Python dependencies
```

## Configuration

All strategy parameters are configurable via `config/strategy_params.yaml`:

```yaml
universe:
  market_cap_min: 1_000_000_000
  min_dividend_years: 5
  avg_daily_volume_min: 10_000_000

basket_formation:
  correlation:
    threshold: 0.85
    cluster_size_min: 6
    cluster_size_max: 10

cointegration:
  johansen:
    pvalue_threshold: 0.01
    eigenvalue_min: 0.15
  validation:
    half_life_max: 35
    hurst_max: 0.45

signals:
  entry:
    long_zscore: -1.5
    short_zscore: 1.5
  exit:
    long_zscore: -0.5
    short_zscore: 0.5

risk_management:
  position_stops:
    zscore_max: 3.0
    time_stop_days: 70
    position_drawdown_max: 0.04
  portfolio_stops:
    drawdown_10pct_action: cut_50pct
    drawdown_15pct_action: exit_all

portfolio:
  kelly_fraction: 0.25
  max_position_per_vector: 0.08
  max_total_exposure: 0.60
```

## Key Features

### Statistical Rigor
- **Johansen cointegration test** with conservative thresholds
- **Engle-Granger confirmation** for robustness
- **Half-life and Hurst exponent** validation
- **Quarterly stability testing** to ensure persistent relationships

### Risk Management
- **Multi-layer stops** at position and portfolio levels
- **Circuit breakers** for severe drawdowns
- **Market neutrality** monitoring with beta hedging
- **VaR-based** position limits

### Execution
- **Realistic transaction costs** (commission + slippage)
- **TWAP algorithm** for large orders
- **Volume limits** (max 5% of ADV)
- **Optimal trading hours** to avoid volatility

### Dashboard Features
- **Real-time monitoring** of signals and positions
- **Interactive parameter tuning** without code changes
- **Comprehensive visualizations:**
  - Correlation matrices
  - Cointegration spreads
  - Equity curves
  - Drawdown charts
  - Trade tables with P&L breakdown
- **Backtest analytics** with multiple metrics

## Performance Metrics

The strategy tracks:
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Win rate
- Profit factor
- Average win/loss
- Trade duration statistics

## Data Requirements

### Required Data (from IBKR):
- Historical daily OHLCV prices (minimum 2 years)
- Dividend history (minimum 5 years)
- Market capitalization
- Volume data
- Sector classification
- Real-time quotes (for live trading)

### Optional Data:
- Earnings data (for payout ratio calculation)
- Corporate actions
- Stock splits/dividends adjustments

## Important Notes

### NO SYNTHETIC DATA
This is a **production-grade prototype**. All data must come from real sources:
- IBKR API for market data
- No simulated or random data
- All backtests use actual historical prices

### Data Quality
- Automatic validation of price data
- Missing data handling (forward fill with limits)
- Outlier detection and reporting
- Stationarity testing

### Limitations
- Requires IBKR market data subscriptions
- Dividend history access may be limited via IBKR API
- Some fundamental data may require supplementary sources
- Cointegration can break down during market regime changes

## Development Roadmap

- [ ] Integration with alternative data providers
- [ ] Enhanced dividend data fetching
- [ ] Real-time alerting system
- [ ] Machine learning for regime detection
- [ ] Options overlay strategies
- [ ] Multi-currency support

## License

[Specify your license]

## Disclaimer

This software is for educational and research purposes. Trading involves risk. Past performance does not guarantee future results. Always test thoroughly with paper trading before deploying real capital.

## Support

For issues and questions:
- Open an issue on GitHub
- Review the dashboard tutorials
- Check IBKR API documentation: https://ibkr.github.io/tws-api/

---

**Built with:**
- Python 3.9+
- Streamlit (dashboard)
- statsmodels (cointegration testing)
- ib_insync (IBKR integration)
- plotly (visualization)
