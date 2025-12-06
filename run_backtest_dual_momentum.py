#!/usr/bin/env python3
"""
Dual Momentum Strategy for Dividend Stocks

Based on Gary Antonacci's Dual Momentum:
1. Absolute momentum: Only invest when 12-month return > 0
2. Relative momentum: Among qualifying stocks, buy highest momentum
3. Inverse yield weighting: Lower yield = higher weight (momentum confirmation)
4. Go to cash when no stocks qualify

This is a proven, research-backed approach enhanced with dividend signals.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from src.data.loader import DataLoader


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class DualMomentumStrategy:
    def __init__(self, loader, initial_capital=1_000_000):
        self.loader = loader
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Dual momentum parameters
        self.LOOKBACK = 252         # 12-month lookback for absolute momentum
        self.SKIP_RECENT = 0        # Don't skip any days
        self.REBALANCE_FREQ = 21    # Monthly
        self.NUM_HOLDINGS = 10      # Top 10 stocks
        self.SLIPPAGE_BPS = 5  # Commission/slippage
        self.USE_RISK_PARITY = False  # Equal weight
        self.USE_QUALITY_FILTER = False  # No quality filter
        self.QUALITY_MULT = 0.75
        self.LEVERAGE = 1.0  # 1.0 = no leverage
        self.USE_INVERSE_YIELD_WEIGHT = False  # Equal weight

        self.price_data = {}
        self.dividend_data = {}
        self.dividend_events = {}  # Track dividend amounts by symbol and date
        self.risk_free_rates = {}  # Daily risk-free rates from TNX
        self.DEFAULT_RISK_FREE_RATE = 0.02  # Default 2% annual rate

        # Dividend handling
        self.COLLECT_DIVIDENDS = True  # Whether to collect dividends on held positions
        self.DIVIDEND_MODE = 'reinvest'  # 'reinvest' or 'cash' (earn risk-free rate)
        self.total_dividends_collected = 0  # Track total dividends for reporting
        self.dividend_cash_account = 0  # Separate account for dividends earning risk-free
        self.total_borrowing_cost = 0  # Track borrowing costs

    def load_data(self, symbols):
        for symbol in symbols:
            df = self.loader.load_price_data(symbol)
            if df is not None and len(df) > self.LOOKBACK + 100:
                self.price_data[symbol] = df['close']
                if 'dividends' in df.columns:
                    self.dividend_data[symbol] = df['dividends']
                    # Build dividend events lookup: {date: amount} for non-zero dividends
                    div_series = df['dividends']
                    div_events = div_series[div_series > 0]
                    if len(div_events) > 0:
                        self.dividend_events[symbol] = div_events.to_dict()
        self.logger.info(f"Loaded {len(self.price_data)} symbols with price data")
        self.logger.info(f"Loaded {len(self.dividend_data)} symbols with dividend data")
        total_div_events = sum(len(v) for v in self.dividend_events.values())
        self.logger.info(f"Loaded {total_div_events} dividend events across {len(self.dividend_events)} symbols")

        # Load risk-free rate data
        self._load_risk_free_rates()

    def _load_risk_free_rates(self):
        """Load 10-Year Treasury rates for borrowing cost calculation."""
        tnx_data = self.loader.load_reference_data('TNX')
        if tnx_data is not None and 'close' in tnx_data.columns:
            for date, row in tnx_data.iterrows():
                self.risk_free_rates[date] = row['close'] / 100.0
            self.logger.info(f"Loaded {len(self.risk_free_rates)} days of risk-free rate data (TNX)")
        else:
            self.logger.warning(f"No TNX data available, using default rate of {self.DEFAULT_RISK_FREE_RATE*100:.1f}%")

    def get_risk_free_rate(self, date):
        """Get the risk-free rate for a given date."""
        if date in self.risk_free_rates:
            return self.risk_free_rates[date]
        available_dates = [d for d in self.risk_free_rates.keys() if d <= date]
        if available_dates:
            return self.risk_free_rates[max(available_dates)]
        return self.DEFAULT_RISK_FREE_RATE

    def calculate_12m_momentum(self, symbol, date):
        """Calculate 12-month total return."""
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

    def calculate_dividend_yield(self, symbol, date):
        """Calculate trailing 12-month dividend yield."""
        if symbol not in self.dividend_data:
            return 0.02  # Default 2% yield if no dividend data
        divs = self.dividend_data[symbol][self.dividend_data[symbol].index <= date]
        prices = self.price_data[symbol][self.price_data[symbol].index <= date]
        if len(divs) < self.LOOKBACK or len(prices) < 1:
            return 0.02
        annual_div = divs.iloc[-self.LOOKBACK:].sum()
        current_price = prices.iloc[-1]
        if current_price <= 0:
            return 0.02
        div_yield = annual_div / current_price
        return max(div_yield, 0.005)  # Floor at 0.5% to avoid division issues

    def calculate_inverse_yield_weights(self, symbols, date):
        """Calculate weights inversely proportional to dividend yield.

        Lower yield = price ran up = momentum confirmation = higher weight.
        """
        inverse_yields = {}
        for sym in symbols:
            div_yield = self.calculate_dividend_yield(sym, date)
            inverse_yields[sym] = 1.0 / div_yield
        total = sum(inverse_yields.values())
        return {s: inverse_yields[s] / total for s in symbols}

    def collect_dividends(self, positions, date):
        """
        Collect dividends for positions held on ex-dividend date.

        Args:
            positions: Dict of {symbol: shares}
            date: Current date

        Returns:
            Tuple of (dividend to add to trading cash, dividend to add to separate account)
        """
        if not self.COLLECT_DIVIDENDS:
            return 0.0, 0.0

        dividend_income = 0.0
        for symbol, shares in positions.items():
            if symbol in self.dividend_events:
                # Check if there's a dividend on this date
                if date in self.dividend_events[symbol]:
                    div_per_share = self.dividend_events[symbol][date]
                    div_amount = shares * div_per_share
                    dividend_income += div_amount
                    self.total_dividends_collected += div_amount

        if self.DIVIDEND_MODE == 'reinvest':
            # All dividends go back into trading cash (reinvested)
            return dividend_income, 0.0
        else:  # 'cash' mode
            # Dividends go to separate account earning risk-free rate
            return 0.0, dividend_income

    def run(self, start_date=None):
        all_dates = sorted(set().union(*[set(s.index) for s in self.price_data.values()]))
        start_idx = self.LOOKBACK + 50
        if start_idx >= len(all_dates):
            return None
        trading_dates = all_dates[start_idx:]

        # Filter by start_date if provided
        if start_date:
            trading_dates = [d for d in trading_dates if d >= pd.Timestamp(start_date)]

        self.logger.info(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()}")

        # Reset tracking
        self.total_dividends_collected = 0
        self.dividend_cash_account = 0
        self.total_borrowing_cost = 0

        cash = self.initial_capital
        margin_debt = 0  # Track borrowed amount for leverage
        positions = {}
        equity_curve = []
        last_rebalance = None

        for i, date in enumerate(trading_dates):
            # Collect dividends for held positions BEFORE calculating equity
            div_to_cash, div_to_account = self.collect_dividends(positions, date)
            cash += div_to_cash
            self.dividend_cash_account += div_to_account

            # Accrue interest on dividend cash account (if using 'cash' mode)
            if self.dividend_cash_account > 0:
                rf_rate = self.get_risk_free_rate(date)
                daily_rate = rf_rate / 252
                interest = self.dividend_cash_account * daily_rate
                self.dividend_cash_account += interest

            # Calculate daily borrowing cost on margin debt
            if margin_debt > 0:
                rf_rate = self.get_risk_free_rate(date)
                daily_rate = rf_rate / 252
                daily_borrowing_cost = margin_debt * daily_rate
                cash -= daily_borrowing_cost
                self.total_borrowing_cost += daily_borrowing_cost

            # Handle delistings: if a position no longer has data, liquidate at last known price
            delisted = []
            for s in list(positions.keys()):
                if s not in self.price_data or date not in self.price_data[s].index:
                    # Stock delisted or no data - sell at last available price
                    available_dates = self.price_data[s].index[self.price_data[s].index < date]
                    if len(available_dates) > 0:
                        last_price = self.price_data[s].loc[available_dates[-1]]
                        cash += positions[s] * last_price
                        self.logger.warning(f"  {date.date()}: {s} delisted, liquidated at ${last_price:.2f}")
                    else:
                        self.logger.warning(f"  {date.date()}: {s} delisted with no price data, position lost")
                    delisted.append(s)
            for s in delisted:
                del positions[s]

            pos_val = sum(
                positions[s] * self.price_data[s].loc[date]
                for s in positions
                if s in self.price_data and date in self.price_data[s].index
            )
            # Total equity includes dividend cash account
            total_equity = cash + pos_val - margin_debt + self.dividend_cash_account

            should_rebalance = last_rebalance is None or (date - last_rebalance).days >= self.REBALANCE_FREQ

            if should_rebalance:
                # Calculate momentum and volatility for all stocks
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

                # Filter: Only stocks with POSITIVE 12-month momentum (absolute momentum filter)
                # Plus quality filter: momentum > 0.75 * volatility
                if self.USE_QUALITY_FILTER:
                    positive_mom = {s: m for s, m in momentum_scores.items()
                                   if m > 0 and m > self.QUALITY_MULT * vol_scores.get(s, 999)}
                else:
                    positive_mom = {s: m for s, m in momentum_scores.items() if m > 0}

                # Select top N by relative momentum
                sorted_stocks = sorted(positive_mom.items(), key=lambda x: x[1], reverse=True)
                top = [s[0] for s in sorted_stocks[:self.NUM_HOLDINGS]]

                # Liquidate all
                for s, shares in positions.items():
                    if s in self.price_data and date in self.price_data[s].index:
                        p = self.price_data[s].loc[date]
                        cash += shares * p
                        cash -= abs(shares * p) * self.SLIPPAGE_BPS / 10000

                # Pay off margin debt after selling all positions
                cash -= margin_debt
                margin_debt = 0
                positions = {}

                # Only invest if we have qualifying stocks
                if top:
                    # Calculate weights
                    if self.USE_RISK_PARITY:
                        # Risk parity: inverse downside volatility weighting
                        inv_vols = {}
                        for sym in top:
                            dnvol = self.calculate_downside_vol(sym, date)
                            if dnvol and dnvol > 0:
                                inv_vols[sym] = 1.0 / dnvol
                            else:
                                inv_vols[sym] = 1.0  # fallback to equal weight
                        total_inv_vol = sum(inv_vols.values())
                        weights = {s: inv_vols[s] / total_inv_vol for s in top}
                    elif self.USE_INVERSE_YIELD_WEIGHT:
                        # Inverse yield weighting: lower yield = higher weight
                        # Lower yield means price ran up = momentum confirmation
                        weights = self.calculate_inverse_yield_weights(top, date)
                    else:
                        # Equal weight
                        weights = {s: 1.0 / len(top) for s in top}

                    available = cash
                    # With leverage: invest LEVERAGE times our equity
                    leveraged_amount = available * 0.98 * self.LEVERAGE
                    equity_spent = available * 0.98  # What we actually spend from cash

                    for sym in top:
                        if sym in self.price_data and date in self.price_data[sym].index:
                            p = self.price_data[sym].loc[date]
                            target = leveraged_amount * weights[sym]
                            positions[sym] = target / p
                            # We only spend our equity portion from cash
                            cash -= equity_spent * weights[sym]
                            cash -= target * self.SLIPPAGE_BPS / 10000

                    # Track borrowed amount
                    margin_debt = leveraged_amount - equity_spent

                last_rebalance = date

                if i % 100 == 0:
                    invested = len(positions) > 0
                    pv = sum(positions[s] * self.price_data[s].loc[date] for s in positions if s in self.price_data and date in self.price_data[s].index)
                    self.logger.info(f"  {date.date()}: {len(positive_mom)} qualify, holding {len(positions)}, ${cash + pv - margin_debt:,.0f}")

            pos_val = sum(
                positions[s] * self.price_data[s].loc[date]
                for s in positions
                if s in self.price_data and date in self.price_data[s].index
            )
            equity_curve.append({'date': date, 'equity': cash + pos_val - margin_debt + self.dividend_cash_account})

        eq_df = pd.DataFrame(equity_curve).set_index('date')
        rets = eq_df['equity'].pct_change().dropna()
        rf = 0.02 / 252
        excess = rets - rf

        # Calculate number of years for dividend metrics
        years = len(trading_dates) / 252

        metrics = {
            'total_return': (eq_df['equity'].iloc[-1] / self.initial_capital) - 1,
            'annual_return': rets.mean() * 252,
            'annual_vol': rets.std() * np.sqrt(252),
            'sharpe': excess.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0,
            'sortino': excess.mean() / rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else 0,
            'max_drawdown': ((eq_df['equity'] / eq_df['equity'].expanding().max()) - 1).min(),
            'total_dividends': self.total_dividends_collected,
            'annual_dividends': self.total_dividends_collected / years if years > 0 else 0,
            'dividend_contribution': self.total_dividends_collected / self.initial_capital,
            'dividend_cash_account': self.dividend_cash_account,
            'total_borrowing_cost': self.total_borrowing_cost,
            'dividend_mode': self.DIVIDEND_MODE,
        }

        return {'equity_curve': eq_df, 'metrics': metrics}


def print_results(results):
    if not results:
        return
    m = results['metrics']
    print("\n" + "=" * 70)
    print("DUAL MOMENTUM STRATEGY RESULTS")
    print("=" * 70)
    print(f"  Total Return:    {m['total_return']*100:>10.2f}%")
    print(f"  Annual Return:   {m['annual_return']*100:>10.2f}%")
    print(f"  Annual Vol:      {m['annual_vol']*100:>10.2f}%")
    print(f"  Sharpe Ratio:    {m['sharpe']:>10.2f}")
    print(f"  Sortino Ratio:   {m['sortino']:>10.2f}")
    print(f"  Max Drawdown:    {m['max_drawdown']*100:>10.2f}%")
    print("-" * 70)
    print("DIVIDEND INCOME")
    print("-" * 70)
    print(f"  Total Dividends: ${m['total_dividends']:>13,.0f}")
    print(f"  Annual Dividends:${m['annual_dividends']:>13,.0f}")
    print(f"  Div Contribution:{m['dividend_contribution']*100:>10.2f}%  (of initial capital)")
    print("=" * 70)
    if m['sharpe'] >= 1.0:
        print("SUCCESS: Sharpe >= 1.0!")
    else:
        print(f"Sharpe: {m['sharpe']:.2f}")


def main():
    logger = setup_logging()
    print("=" * 70)
    print("DUAL MOMENTUM STRATEGY")
    print("=" * 70)

    loader = DataLoader()
    # Use universe CSV to filter symbols (not all available price data)
    symbols = loader.get_universe_symbols()

    strategy = DualMomentumStrategy(loader)
    strategy.load_data(symbols)

    # Run full period (no start date filter)
    results = strategy.run()
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
