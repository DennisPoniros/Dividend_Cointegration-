# Quick Start Guide

## Prerequisites

1. **Python 3.9+** installed
2. **Interactive Brokers** account with TWS or IB Gateway
3. **Market data subscriptions** (US Equities Level 1 minimum)

## Step 1: Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure IBKR Connection

1. Copy the example config:
```bash
cp config/ibkr_config.example.yaml config/ibkr_config.yaml
```

2. Edit `config/ibkr_config.yaml` with your settings:
```yaml
connection:
  host: "127.0.0.1"
  port: 7497  # 7497 for TWS paper, 7496 for live, 4002 for Gateway paper
  client_id: 1

account:
  account_id: "YOUR_ACCOUNT_ID"
```

3. Start TWS or IB Gateway and enable API connections:
   - In TWS: **Global Configuration → API → Settings**
   - Enable ActiveX and Socket Clients
   - Add trusted IP: `127.0.0.1`
   - Set Socket port to match config (7497 for paper)

## Step 3: Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Step 4: Build Your First Strategy

### Option A: Using the Dashboard (Recommended)

1. **Universe & Data Tab:**
   - Click "Connect to IBKR" in sidebar
   - Upload a universe CSV or build from scratch
   - Example universe CSV format:
     ```csv
     symbol,market_cap,screen_date
     AAPL,2800000000000,2024-01-01
     MSFT,2500000000000,2024-01-01
     ```

2. **Basket Formation Tab:**
   - Click "Form Baskets"
   - Review correlation clusters
   - Examine basket details

3. **Cointegration Analysis Tab:**
   - Click "Test Cointegration"
   - Review validation metrics
   - Analyze individual vectors

4. **Backtesting Tab:**
   - Set initial capital
   - Select date range
   - Click "Run Backtest"
   - Analyze results

### Option B: Using Python Scripts

```python
from src.data.ibkr_connector import IBKRConnector
from src.data.universe import UniverseBuilder
from src.strategy.basket_formation import BasketFormer
from src.strategy.cointegration import CointegrationTester
from src.backtest.engine import BacktestEngine

# 1. Connect to IBKR
ibkr = IBKRConnector()
ibkr.connect()

# 2. Build universe (requires a list of candidate symbols)
universe_builder = UniverseBuilder()
# You'll need to provide a comprehensive list of US stocks
candidate_symbols = ['AAPL', 'MSFT', 'JNJ', 'PG', ...]
universe = universe_builder.build_universe(ibkr, candidate_symbols)

# 3. Get price data
universe_symbols = universe['symbol'].tolist()
price_data = ibkr.get_multiple_historical_data(
    universe_symbols,
    duration="2 Y",
    bar_size="1 day"
)

# 4. Form baskets
basket_former = BasketFormer()
baskets = basket_former.form_baskets(
    universe_symbols,
    price_data,
    ibkr
)

# 5. Test cointegration
coint_tester = CointegrationTester()
vectors = coint_tester.test_all_baskets(baskets, price_data)

# 6. Run backtest
backtest = BacktestEngine()
results = backtest.run_backtest(
    vectors,
    price_data,
    initial_capital=1_000_000
)

# 7. View results
from src.backtest.metrics import PerformanceMetrics
PerformanceMetrics.print_summary(results['metrics'])

# 8. Disconnect
ibkr.disconnect()
```

## Step 5: Customize Parameters

Edit `config/strategy_params.yaml` to adjust:

```yaml
# Example: Make entry/exit more conservative
signals:
  entry:
    long_zscore: -2.0    # More extreme entry (was -1.5)
    short_zscore: 2.0
  exit:
    long_zscore: 0.0     # Exit at mean (was -0.5)
    short_zscore: 0.0

# Example: Tighter risk management
risk_management:
  position_stops:
    zscore_max: 2.5      # Exit earlier (was 3.0)
    time_stop_days: 50   # Shorter holding period (was 70)
```

Or use the dashboard's interactive parameter controls in the sidebar.

## Common Issues

### "Connection refused" when connecting to IBKR

**Solution:**
- Ensure TWS/Gateway is running
- Check API is enabled in TWS settings
- Verify port number matches config
- Check firewall settings

### "No market data" errors

**Solution:**
- Verify you have market data subscriptions
- Check if market is open (delayed data available 24/7)
- Ensure symbols are valid US equities

### Insufficient historical data

**Solution:**
- Strategy requires 2 years (504 days) of data
- Newer stocks may not qualify
- Some stocks may have data gaps

### Dividend data not available

**Solution:**
- IBKR's dividend API has limitations
- Consider supplementing with yfinance:
  ```python
  import yfinance as yf
  ticker = yf.Ticker("AAPL")
  dividends = ticker.dividends
  ```

## Next Steps

1. **Paper Trading:**
   - Test with IBKR paper account first
   - Monitor for 1-2 months before live trading

2. **Risk Management:**
   - Start with small capital allocation
   - Set strict position limits
   - Monitor drawdowns closely

3. **Optimization:**
   - Run backtests with different parameters
   - Analyze which sectors perform best
   - Consider transaction cost sensitivity

4. **Live Deployment:**
   - Switch to live IBKR account
   - Enable real-time data subscriptions
   - Set up monitoring and alerts

## Recommended Workflow

**Daily:**
- Check Live Signals tab
- Monitor active positions
- Review risk metrics

**Weekly:**
- Rebalance if needed
- Update hedge ratios
- Check for risk stop triggers

**Monthly:**
- Retest cointegration for active vectors
- Review performance metrics
- Adjust parameters if needed

**Quarterly:**
- Rebuild universe
- Reform baskets
- Full cointegration testing
- Comprehensive backtest

## Resources

- **IBKR API Docs:** https://ibkr.github.io/tws-api/
- **Streamlit Docs:** https://docs.streamlit.io/
- **Cointegration Theory:**
  - Engle & Granger (1987) "Co-integration and Error Correction"
  - Johansen (1991) "Estimation and Hypothesis Testing"
  - Avellaneda & Lee (2010) "Statistical Arbitrage in the U.S. Equities Market"

## Support

If you encounter issues:
1. Check the logs in the dashboard
2. Review IBKR API error codes
3. Verify data quality (universe/basket CSVs)
4. Test with smaller datasets first

## Safety Checklist

Before live trading:
- [ ] Tested extensively in paper trading
- [ ] Understand all risk stops
- [ ] Set appropriate position limits
- [ ] Have risk budget defined
- [ ] Monitoring system in place
- [ ] Emergency exit plan ready
- [ ] Reviewed all transaction costs
- [ ] Verified data quality

**Remember: This is a sophisticated strategy. Start small and scale gradually.**
