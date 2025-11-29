# Dividend Cointegration Trading Strategy

A systematic mean-reversion strategy exploiting cointegration relationships among dividend-paying stocks.

---

## STEP 1: Universe Construction (FIXED CRITERIA)

### Basket Creation
**ALL dividend-paying US stocks with:**
- Market cap > $5B (liquidity requirement)
- Minimum 5 consecutive years of dividend payments
- Average daily volume > $10M
- **Exclude:** REITs, MLPs, special dividends only

**Timeline:** Update quarterly (synchronized with earnings seasons)

---

## STEP 2: Sub-Basket Formation (HYBRID SECTOR-CORRELATION CLUSTERING)

### Implementation (in this exact order):

1. **Initial Sector Grouping (GICS Level 3)**
   - Minimum 15 stocks per sector-group
   - Create 8-12 distinct sector pools

2. **Within-Sector Correlation Clustering**
   - Rolling 252-day Pearson correlation
   - Threshold: ρ ≥ 0.85 (stricter than literature)
   - Form sub-clusters of 6-10 stocks per sector

3. **Dividend Trait Screening (Secondary Filter)**
   - Dividend growth consistency: Standard deviation of YoY growth < 15%
   - Payout ratio: 30-70% (excludes unsustainable high payers and stingy accumulators)
   - Yield range: 1.5-6% (excludes potential dividend cuts and distressed situations)

**Result:** ~20-30 sub-baskets of 6-10 stocks each

**Alpha Mechanism:** The consistency filter IS the alpha—stable dividend growers have mean-reverting fundamentals

---

## STEP 3: Cointegration Vector Selection (SPECIFIC PROTOCOL)

**Testing Period:** 504 trading days (2 years)

### For each sub-basket:

#### Johansen Test (Primary)
- Test all combinations of 4-6 stocks (FIXED range)
- Lag order: K=2 (based on daily data convention)
- Specification: Constant term, no trend
- **Acceptance criteria:**
  - Trace statistic p-value < 0.01 (conservative)
  - Rank = 1 (single strongest vector)
  - Largest eigenvalue > 0.15

#### Validation Battery (ALL must pass):
- Engle-Granger confirmation: ADF on residuals < -3.5
- Half-life: 15-35 trading days (HARD bounds)
- Hurst exponent: < 0.45 (strong mean reversion)
- Out-of-sample test: Last 126 days shows p-value < 0.05

#### Stability Test:
- Split 504 days into 4 quarters
- Calculate cointegration in each quarter
- **Requirement:** Cointegrated in ALL 4 quarters (coefficient of variation of half-life < 0.4)

**Vector Weighting:** Use eigenvector from largest eigenvalue, normalized

**Expected Output:** 10-20 robust cointegration vectors (from 20-30 candidate baskets)

---

## STEP 4: Portfolio Construction (EXACT ALLOCATION)

### Capital Allocation
- Equal risk contribution across selected vectors
- Position sizing per vector: σ_basket × z-score × Kelly_fractional
- **Kelly fraction:** 0.25 (quarter-Kelly, non-negotiable)
- Maximum per vector: 8% of portfolio
- Maximum total cointegration exposure: 60% of portfolio

### Market Neutrality
- Calculate portfolio beta weekly
- Rebalance if |beta| > 0.15
- Use SPY hedge if persistent beta drift

---

## STEP 5: Signal Generation (FIXED THRESHOLDS)

### Trading Rules (Avellaneda-Lee Modified)

#### Entry Signals
- **Long entry:** z-score ≤ -1.5σ
- **Short entry:** z-score ≥ +1.5σ
- *(More conservative than literature's -1.25/+1.25)*

#### Exit Signals
- **Long exit:** z-score ≥ -0.5σ OR time stop
- **Short exit:** z-score ≤ +0.5σ OR time stop

#### Calculation
```
z-score = (current_spread - rolling_mean) / rolling_std
Rolling window: 42 days (2× target average half-life of 21 days)
Recalculate daily
```

---

## STEP 6: Risk Management (MULTI-LAYER HARD STOPS)

### Immediate Exit Triggers (ANY condition):
- z-score exceeds ±3.0σ (relationship breakdown)
- Position held > 70 days (2× maximum half-life)
- Rolling ADF p-value > 0.15 for 5 consecutive days
- Half-life exceeds 50 days in 21-day rolling window
- Position drawdown > 4% of allocated capital

### Portfolio-Level Circuit Breakers
- **10% drawdown:** Cut all position sizes by 50%
- **15% drawdown:** Exit all positions, halt trading 20 days
- **Daily VaR breach (95%):** No new positions that day

### Stop-Loss Per Trade
- Entry at 1.5σ → Stop at 3.0σ
- Represents 1.5σ adverse movement

---

## STEP 7: Retraining Schedule (FIXED CALENDAR)

### Weekly (Every Monday)
- Recalculate hedge ratios (rolling 42 days)
- Update z-scores and rolling statistics
- Check half-life on all active positions

### Monthly (First trading day)
- Re-run Johansen test on all active vectors (rolling 504 days)
- Check if p-value still < 0.05
- If failed: close position immediately, remove from tradeable universe for 63 days

### Quarterly (Post-earnings season)
- Full universe refresh
- Re-run entire basket formation process
- Discover new cointegration vectors
- **Grandfather rule:** Keep existing positions if they still pass monthly tests

### Annual (January)
- Full strategy review
- Optimize thresholds using prior year data (walk-forward)
- Adjust Kelly fraction if Sharpe ratio changed > 30%

---

## STEP 8: Execution (SPECIFIC PROTOCOLS)

### Order Types
- Limit orders only
- Max execution window: 30 minutes
- Cancel and retry with market order if unfilled

### Execution Algorithm
- For positions < $50K: Direct limit orders
- For positions > $50K: TWAP over 15 minutes
- Maximum 5% of ADV per stock

### Trading Hours
- **Enter positions:** 10:00-15:00 ET (avoid open/close volatility)
- **Exit positions:** 09:45-15:30 ET (allow closing volatility exits if needed)

---

## STEP 9: Advanced Enhancements (PHASE 2 - AFTER 6 MONTHS LIVE)

### HMM Regime Detection
- 3-state model (low/medium/high volatility)
- Adjustment: No new entries in high-volatility regime
- Tighten stops to 2.5σ in high-volatility regime

### Reinforcement Learning Position Sizing
- Train DDPG agent on historical data
- Replace fixed Kelly with RL-optimized sizing
- Constraint: RL cannot exceed 2× Kelly recommendation

### Dividend Event Overlay
- Adjust hedge ratios 2 days before ex-dividend date
- Reduce position by (dividend_yield / spread_volatility) coefficient
- Resume normal hedge ratio on ex-date

---

## KEY DECISIONS LOCKED IN

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Basket size | 4-6 stocks | Narrower is better for stability |
| Testing lookback | 504 days | 2 years exactly |
| Half-life range | 15-35 days | Tight bounds for tradeable mean reversion |
| Entry threshold | ±1.5σ | Conservative vs literature |
| Rebalancing | Quarterly | Synchronized with dividends/earnings |
| Kelly fraction | 0.25 | Fixed, not adaptive initially |
| Max holding period | 70 days | Hard stop |
| Primary test | Johansen | Engle-Granger as secondary confirmation |

---

## EXPECTED PERFORMANCE (REALISTIC POST-COST)

| Metric | Target |
|--------|--------|
| Annual return | 12-15% |
| Sharpe ratio | 1.2-1.5 |
| Maximum drawdown | 18% |
| Win rate | 55-60% |
| Average holding period | 25-35 days |
| Turnover | ~15× annually |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download market data (first time only)
python -m src.data.download_all

# 3. Run backtest
python run_backtest.py
```

---

## Repository Structure

```
├── config/                 # Strategy parameters (YAML)
├── data/                   # Local market data storage
│   ├── prices/            # OHLCV data (parquet)
│   ├── dividends/         # Dividend history
│   ├── fundamentals/      # Company snapshots
│   └── reference/         # SPY, sector ETFs
├── src/
│   ├── data/              # Data download & loading
│   ├── strategy/          # Cointegration, signals, portfolio
│   ├── risk/              # Risk management
│   └── backtest/          # Backtesting engine
├── run_backtest.py        # Main backtest script
└── STRATEGY.md            # This file
```
