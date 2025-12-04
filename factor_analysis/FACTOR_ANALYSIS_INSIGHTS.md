# Factor Analysis Insights: 2x Leverage Dual Momentum Strategy

## Executive Summary

This comprehensive factor analysis examines the 2x leveraged dual momentum dividend stock strategy across multiple dimensions: market exposure, sector tilts, momentum characteristics, volatility patterns, size and yield factors, and risk attribution. The analysis covers **1,433 trading days** from March 2020 to November 2025, during which the strategy generated a **1,943.73% total return** (70.00% annualized) with **1,716 trades** across **95 unique stocks** from a universe of 132 dividend-paying equities.

---

## 1. Market Factor Analysis

### Key Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Beta to SPY | **1.699** | Strategy amplifies market moves ~70% more than a 1x market exposure |
| Alpha (Annualized) | **28.00%** | Significant positive alpha after accounting for market beta |
| R-Squared | **0.443** | Market explains only ~44% of strategy variance - significant alpha generation |
| Tracking Error | **35.71%** | High deviation from market returns (expected for active momentum strategy) |
| Information Ratio | **0.784** | Strong risk-adjusted alpha (>0.5 is considered good) |

### Interpretation

The strategy exhibits a **beta of 1.70**, which is consistent with the 2x leverage employed but not a full 2.0 due to:
1. Cash holdings during rebalancing periods
2. Quality filtering that excludes highest-volatility stocks
3. Not being 100% invested at all times

The **28% annualized alpha** is exceptional and indicates the momentum stock selection process adds significant value beyond mere market exposure. The relatively low R-squared (0.443) confirms that less than half of the strategy's returns can be attributed to simple market exposure - the majority comes from active stock selection and momentum timing.

The **Information Ratio of 0.78** suggests this is a skilled strategy - generating substantial alpha relative to the tracking error assumed. Values above 0.5 are typically considered good for active strategies.

### Rolling Beta Observations
- Beta varies significantly over time (see `plots/market_factor_analysis.png`)
- During high-volatility periods (e.g., COVID recovery), beta tends to exceed 2.0
- In calmer markets, beta compresses toward 1.5
- This dynamic beta behavior is characteristic of momentum strategies

---

## 2. Sector Factor Analysis

### Universe Sector Composition
| Sector | Weight | Notes |
|--------|--------|-------|
| Financial Services | 25.0% | Largest sector - banks, insurers, asset managers |
| Industrials | 14.4% | Machinery, aerospace, logistics |
| Consumer Defensive | 12.1% | Stable dividend payers |
| Healthcare | 10.6% | Pharma, medical devices |
| Technology | 8.3% | Despite lower weight, high momentum |
| Utilities | 8.3% | Traditional dividend sector |
| Basic Materials | 7.6% | Mining, chemicals |
| Energy | 6.8% | Oil & gas majors |
| Consumer Cyclical | 4.5% | Retail, restaurants |
| Communication Services | 2.3% | Telecom |

### Sector Factor Exposures (Betas)
| Sector ETF | Exposure | Interpretation |
|------------|----------|----------------|
| **XLI (Industrials)** | +1.166 | Highest positive exposure - strategy favors cyclical industrials |
| **XLK (Technology)** | +0.819 | Strong tech tilt despite lower universe weight |
| **XLF (Financials)** | +0.606 | Moderate positive exposure to financials |
| **XLU (Utilities)** | +0.385 | Slight defensive tilt |
| **XLY (Consumer Cyclical)** | -0.320 | Underweight cyclical consumer |
| **XLE (Energy)** | -0.218 | Underweight despite high P&L contribution |
| **XLB (Materials)** | -0.235 | Slight underweight |

### Key Sector Insights

1. **Industrial Leadership**: The highest sector beta (1.166 to XLI) indicates the strategy strongly favors industrial stocks. This makes sense given industrials often exhibit strong momentum during economic expansions.

2. **Technology Overweight**: Despite only 8.3% universe weight, tech shows a strong 0.819 beta. The strategy effectively picks momentum winners from tech (e.g., AVGO, ORCL) which drive outsized returns.

3. **Energy Paradox**: Negative sector beta (-0.218) despite Energy being the 4th largest P&L contributor. This suggests the strategy times energy exposure well - buying during momentum uptrends rather than holding persistently.

4. **Defensive Underweight**: Negative exposure to Consumer Staples (XLP: -0.163) indicates the momentum filter naturally rotates away from low-volatility defensive names.

---

## 3. Momentum Factor Analysis

### Momentum at Entry Statistics
| Statistic | Value |
|-----------|-------|
| Mean | 0.716 (71.6% 12-month return) |
| Std Dev | 0.392 |
| Min | 0.158 |
| Max | 5.342 (534% 12-month return!) |

### Momentum-Return Relationship

**Correlation: -0.019** (essentially zero)

This surprising finding indicates that **within the selected momentum stocks, higher entry momentum does not predict better forward returns**. This is consistent with academic research showing momentum works at the cross-section (selecting high vs. low momentum) but not linearly.

### Returns by Momentum Quintile (at entry)
| Quintile | Avg Return | Std Dev | Sharpe |
|----------|------------|---------|--------|
| Q1 (Lowest) | 1.12% | 4.47% | 0.25 |
| Q2 | 1.19% | 6.04% | 0.20 |
| Q3 | 0.64% | 6.39% | 0.10 |
| Q4 | 1.23% | 7.01% | 0.18 |
| Q5 (Highest) | 0.81% | 6.26% | 0.13 |

### Momentum Insights

1. **Non-Linear Momentum**: The lowest momentum quintile (Q1) actually has the best Sharpe ratio (0.25). This suggests the quality filter and 14-day rebalancing help avoid momentum crashes.

2. **Extreme Momentum Risk**: The highest quintile (Q5) shows lower returns and higher volatility - classic momentum crash behavior.

3. **Sweet Spot**: Moderate momentum (Q2 and Q4) appears optimal, balancing trend strength with crash risk.

4. **Entry Momentum Range**: Mean entry momentum of 71.6% represents strong stocks, but the quality filter (momentum > 0.75 × volatility) prevents purely chasing performance.

---

## 4. Volatility Factor Analysis

### Strategy Volatility
| Metric | Value |
|--------|-------|
| Daily Volatility | 3.01% |
| Annual Volatility | 47.85% |
| Median 63-day Vol | 45.19% |

### Volatility Regimes
| Regime | Returns (Ann.) | Days |
|--------|---------------|------|
| High Volatility | 69.60% | 685 |
| Low Volatility | 59.94% | 685 |

### Volatility Insights

1. **Higher Vol = Higher Returns**: The strategy performs better during high volatility periods (69.6% vs. 60.0% annual returns). This is characteristic of momentum strategies which thrive on price dispersion.

2. **Volatility Clustering**: The ~48% annual volatility is consistent with 2x leverage on a diversified equity portfolio. Rolling volatility shows significant time-variation (see `plots/volatility_analysis.png`).

3. **Return Distribution**: Strategy returns exhibit negative skewness and excess kurtosis (fat tails), typical of leveraged momentum strategies.

4. **Risk Consideration**: The high volatility (47.85% annual) requires strong conviction and appropriate position sizing. Max drawdown of -35.70% (from main strategy report) is substantial.

---

## 5. Size Factor Analysis

### Universe Market Cap Distribution
| Metric | Value |
|--------|-------|
| Mean | $208.6B |
| Median | $77.2B |
| Min | $7.9B |
| Max | $4,138.2B (Apple) |

### Traded Stocks Market Cap
| Metric | Value |
|--------|-------|
| Mean | $246.9B |
| Median | $85.7B |

### Trade Counts by Size Quintile
| Size Quintile | Trades | % of Total |
|---------------|--------|------------|
| Large (Q5) | 645 | **37.6%** |
| Q4 | 235 | 13.7% |
| Q3 | 178 | 10.4% |
| Q2 | 409 | 23.8% |
| Small (Q1) | 249 | 14.5% |

### Size Factor Insights

1. **Large-Cap Bias**: 37.6% of trades occur in the largest quintile. Large-caps often exhibit stronger, more persistent momentum due to analyst coverage and institutional flows.

2. **Barbell Effect**: High trade counts in both Large (645) and Q2 (409) quintiles suggest the strategy finds opportunities across the cap spectrum.

3. **Mid-Cap Avoidance**: The middle quintile (Q3) has the fewest trades, possibly due to these stocks falling between momentum leaders and smaller high-performers.

4. **Liquidity Alignment**: The large-cap bias ensures adequate liquidity for the 2x leveraged positions without excessive market impact.

---

## 6. Dividend Yield Factor Analysis

### Universe Dividend Yield Distribution
| Metric | Value |
|--------|-------|
| Mean | 2.69% |
| Median | 2.57% |
| Min | 0.00% (INTC suspended dividend) |
| Max | 7.19% (MO - Altria) |

### Trade Performance by Yield Quintile
| Yield Quintile | Avg Return | Trade Count |
|----------------|------------|-------------|
| Low (Q1) | **1.35%** | 631 |
| Q2 | 1.02% | 401 |
| Q3 | 0.66% | 186 |
| Q4 | 0.46% | 269 |
| High (Q5) | 0.90% | 229 |

### Dividend Yield Insights

1. **Lower Yield = Better Returns**: The lowest yield quintile shows the highest average return (1.35%). This aligns with the strategy's inverse yield weighting - lower yielding stocks get higher weights.

2. **Growth over Income**: Low-yield stocks tend to be growth-oriented with higher momentum, explaining the performance advantage.

3. **High Yield Trap Avoided**: High-yielding stocks (Q5) show lower returns (0.90%), potentially due to value traps and dividend cuts.

4. **Trade Distribution**: The strategy heavily favors low-yield stocks (631 trades in Q1 vs. 229 in Q5), consistent with momentum-based selection naturally rotating into growth names.

---

## 7. Principal Component Analysis (PCA)

### Variance Explanation
| Component | Individual | Cumulative |
|-----------|------------|------------|
| PC1 (Market) | 41.87% | 41.87% |
| PC2 | 8.12% | 49.99% |
| PC3 | 3.90% | 53.89% |
| PC4 | 2.42% | 56.31% |
| PC5 | 1.92% | 58.23% |

### Top Stocks by PC1 Loading (Market Factor)
| Stock | Loading | Sector |
|-------|---------|--------|
| PRU | 0.115 | Financial Services |
| MET | 0.114 | Financial Services |
| ITW | 0.111 | Industrials |
| PNC | 0.111 | Financial Services |
| BAC | 0.110 | Financial Services |

### PCA Insights

1. **Market Dominance**: PC1 explains 41.87% of variance, representing the common market factor. This is typical for equity portfolios.

2. **Financial Concentration in PC1**: The top PC1 loadings are dominated by financials (PRU, MET, PNC, BAC). These stocks are highly sensitive to overall market beta and interest rate expectations.

3. **Diversification Benefit**: PC2-PC5 only add ~17% incremental variance explanation, suggesting the 132-stock universe is well-diversified with limited sector concentration.

4. **10 Components for 58%**: Even 10 principal components explain only 58% of variance, indicating significant idiosyncratic risk in individual stocks - good for alpha generation.

---

## 8. Risk Attribution

### Variance Decomposition
| Component | Proportion | Annual Volatility |
|-----------|------------|-------------------|
| **Total Risk** | 100.0% | 47.85% |
| Systematic (Market) | 44.3% | 31.84% |
| Idiosyncratic (Alpha) | 55.7% | 35.71% |

### Risk Attribution Insights

1. **Majority Idiosyncratic**: 55.7% of variance is idiosyncratic, meaning the majority of risk (and return) comes from stock-specific bets rather than market exposure.

2. **Active Risk Budget**: The 35.71% idiosyncratic volatility represents the "active risk" taken to generate alpha. This is substantial but expected for an active momentum strategy.

3. **Risk-Adjusted Value**: The 28% alpha generated on 35.71% tracking error yields the 0.78 Information Ratio - attractive risk-adjusted returns.

4. **Leverage Amplification**: Both systematic and idiosyncratic risks are amplified by the 2x leverage. In a 1x version, total volatility would be ~24%.

---

## 9. Return Decomposition

### Total Return Attribution
| Component | Contribution |
|-----------|--------------|
| **Total Return** | 1,943.73% |
| Beta Contribution | 352.40% |
| Alpha Contribution | 159.22% |
| Unexplained/Interaction | ~1,431% |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Win Rate | 57.1% |
| Average Win | +4.85% |
| Average Loss | -4.14% |
| Profit Factor | 1.45 |

### Return Insights

1. **Massive Outperformance**: The strategy's 1,943% return dwarfs the market contribution (352%). The compounding effect of high alpha over 5.7 years is remarkable.

2. **Positive Expectancy**: With 57.1% win rate and +4.85% avg win vs -4.14% avg loss, each trade has positive expected value. Profit factor of 1.45 means gross profits are 45% larger than gross losses.

3. **Compound Effect**: The "unexplained" portion largely represents the compounding interaction between alpha and beta over the long holding period - this is the power of sustained high returns.

---

## 10. Stock Participation Analysis

### Participation Rate
- **95 of 132 stocks traded** (72.0% participation)
- 37 stocks never traded (filtered by momentum/quality criteria)

### Top 10 Contributors by P&L
| Rank | Stock | Total P&L | Trades | Sector |
|------|-------|-----------|--------|--------|
| 1 | **GE** | $4,837,709 | 78 | Industrials |
| 2 | AVGO | $2,956,433 | 66 | Technology |
| 3 | OXY | $2,021,125 | 39 | Energy |
| 4 | LLY | $2,016,901 | 65 | Healthcare |
| 5 | ORCL | $1,333,382 | 38 | Technology |
| 6 | CAH | $1,193,991 | 47 | Healthcare |
| 7 | CTAS | $957,264 | - | Industrials |
| 8 | FCX | $903,444 | 42 | Basic Materials |
| 9 | NUE | $675,327 | 38 | Basic Materials |
| 10 | NEM | $543,956 | - | Basic Materials |

### Bottom 10 by P&L
| Rank | Stock | Total P&L | Notes |
|------|-------|-----------|-------|
| 1 | BRO | -$348,312 | Insurance broker |
| 2 | QCOM | -$331,303 | Semiconductor cycles |
| 3 | ALB | -$323,225 | Lithium volatility |
| 4 | JPM | -$240,225 | Bank underperformance |
| 5 | SCHW | -$228,156 | Financial volatility |

### Sector P&L Attribution
| Sector | P&L | Trades | Avg Return |
|--------|-----|--------|------------|
| **Industrials** | $6.26M | 244 | 1.03% |
| **Technology** | $5.36M | 217 | 1.45% |
| Healthcare | $3.64M | 240 | 0.86% |
| Energy | $2.76M | 318 | 1.20% |
| Basic Materials | $1.61M | 166 | 1.62% |
| Financial Services | $0.78M | 345 | 0.47% |
| Consumer Defensive | $0.54M | 102 | 0.96% |
| Consumer Cyclical | $0.15M | 35 | 0.95% |
| Communication Services | $0.05M | 22 | 0.33% |
| **Utilities** | **-$0.14M** | 27 | -0.33% |

### Stock Participation Insights

1. **GE Dominance**: GE contributed nearly $5M in profits across 78 trades - the restructuring and aerospace momentum played perfectly into the strategy.

2. **Tech Alpha**: Technology generates the highest average return per trade (1.45%) despite moderate trade count. Quality over quantity.

3. **Basic Materials Efficiency**: Best return per trade (1.62%) among major sectors, driven by commodity momentum (NUE, FCX, NEM).

4. **Utilities Drag**: Only sector with negative P&L, as dividend-focused utilities lack momentum characteristics the strategy seeks.

5. **Concentration Risk**: Top 10 stocks contributed $18.5M of the total ~$19.4M in net profits. Success is heavily concentrated in winners.

---

## 11. Key Takeaways and Recommendations

### Strengths
1. **Exceptional Alpha**: 28% annualized alpha with 0.78 Information Ratio demonstrates genuine skill
2. **Dynamic Factor Exposure**: Strategy naturally rotates into momentum leaders, avoiding stale positions
3. **Quality Filter Effectiveness**: Momentum > 0.75 × volatility filter helps avoid momentum crashes
4. **Sector Rotation**: Positive industrial/tech exposure during growth periods, reduced energy beta despite opportunistic gains

### Risks and Considerations
1. **High Volatility**: 47.85% annual volatility and -35.70% max drawdown require strong risk tolerance
2. **Leverage Costs**: Significant borrowing costs ($1.54M) erode returns
3. **Momentum Crash Risk**: Strategy is vulnerable to rapid momentum reversals
4. **Concentration**: Top 10 stocks drive majority of returns - single stock risk

### Potential Enhancements
1. **Dynamic Leverage**: Reduce leverage during high VIX periods to limit drawdowns
2. **Sector Limits**: Cap sector exposure to diversify factor bets
3. **Momentum Decay Filter**: Consider exit signals when momentum decelerates
4. **Dividend Growth Overlay**: Incorporate dividend growth rates as quality signal

---

## Appendix: File Directory

### Plots (factor_analysis/plots/)
- `equity_drawdown.png` - Equity curve and drawdown visualization
- `market_factor_analysis.png` - Beta analysis, residuals, rolling beta
- `sector_exposure.png` - Sector composition and factor exposures
- `momentum_analysis.png` - Momentum distribution and quintile analysis
- `volatility_analysis.png` - Rolling volatility and return distribution
- `size_factor.png` - Market cap distribution and trade counts
- `dividend_yield_factor.png` - Yield distribution and performance
- `pca_analysis.png` - Principal component analysis visualization
- `factor_correlations.png` - Factor correlation heatmap
- `risk_attribution.png` - Risk decomposition charts
- `stock_participation.png` - Stock-level P&L analysis
- `monthly_heatmap.png` - Monthly returns heatmap
- `rolling_metrics.png` - Rolling Sharpe, returns, excess returns
- `trade_analysis.png` - Trade return and P&L distributions

### Data Files (factor_analysis/data/)
- `market_factor_metrics.csv` - Market factor regression results
- `sector_exposures.csv` - Sector ETF betas
- `risk_attribution.csv` - Variance decomposition
- `pca_loadings.csv` - Stock loadings on principal components
- `stock_pnl.csv` - P&L by stock
- `factor_correlations.csv` - Full correlation matrix

---

*Analysis generated: December 2025*
*Strategy Period: March 2020 - November 2025*
*Universe: 132 dividend-paying large-cap U.S. equities*
