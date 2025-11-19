"""
Risk Management Module (STEP 6)

Implements multi-layer hard stops:
1. Position-level stops (z-score, time, ADF, half-life, drawdown)
2. Portfolio-level circuit breakers (drawdown, VaR)
3. Trade-level stop losses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yaml
import logging
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages all risk controls and circuit breakers."""

    def __init__(self, config_path: str = "config/strategy_params.yaml"):
        """
        Initialize risk manager.

        Args:
            config_path: Path to strategy parameters
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.risk_config = self.config['risk_management']
        self.position_entry_dates = {}
        self.position_entry_capital = {}
        self.portfolio_high_water_mark = 0
        self.trading_halted = False
        self.halt_until = None

    def check_zscore_breach(self, zscore: float) -> bool:
        """
        Check if z-score exceeds maximum threshold.

        Args:
            zscore: Current z-score

        Returns:
            True if breached, False otherwise
        """
        max_zscore = self.risk_config['position_stops']['zscore_max']
        breached = abs(zscore) > max_zscore

        if breached:
            logger.warning(f"Z-score breach: {zscore:.2f} > ±{max_zscore}")

        return breached

    def check_time_stop(
        self,
        vector_id: int,
        current_date: datetime
    ) -> bool:
        """
        Check if position has exceeded time stop.

        Args:
            vector_id: Vector identifier
            current_date: Current date

        Returns:
            True if time stop breached, False otherwise
        """
        if vector_id not in self.position_entry_dates:
            return False

        entry_date = self.position_entry_dates[vector_id]
        days_held = (current_date - entry_date).days

        max_days = self.risk_config['position_stops']['time_stop_days']
        breached = days_held > max_days

        if breached:
            logger.warning(f"Time stop breach for vector {vector_id}: {days_held} days > {max_days}")

        return breached

    def check_adf_breach(
        self,
        spread: pd.Series,
        window: int = 21
    ) -> bool:
        """
        Check if ADF p-value has breached for consecutive days.

        Args:
            spread: Cointegration spread
            window: Rolling window for ADF test

        Returns:
            True if breached, False otherwise
        """
        pvalue_threshold = self.risk_config['position_stops']['adf_pvalue_breach']
        breach_days = self.risk_config['position_stops']['adf_breach_days']

        if len(spread) < window + breach_days:
            return False

        # Test ADF for last N days
        consecutive_breaches = 0

        for i in range(breach_days):
            test_data = spread.iloc[-(window + i):(-i if i > 0 else None)]

            try:
                adf_result = adfuller(test_data, maxlag=None, regression='c', autolag='AIC')
                pvalue = adf_result[1]

                if pvalue > pvalue_threshold:
                    consecutive_breaches += 1
                else:
                    break

            except Exception as e:
                logger.error(f"Error in ADF test: {e}")
                return False

        breached = consecutive_breaches >= breach_days

        if breached:
            logger.warning(f"ADF breach: p-value > {pvalue_threshold} for {breach_days} consecutive days")

        return breached

    def calculate_half_life(self, spread: pd.Series) -> Optional[float]:
        """
        Calculate half-life of mean reversion.

        Args:
            spread: Cointegration spread

        Returns:
            Half-life in days, or None if calculation fails
        """
        try:
            spread_lag = spread.shift(1)
            spread_diff = spread.diff()

            df = pd.DataFrame({
                'spread': spread,
                'spread_lag': spread_lag,
                'spread_diff': spread_diff
            }).dropna()

            X = df['spread_lag'].values
            y = df['spread_diff'].values
            X_with_const = np.column_stack([np.ones(len(X)), X])

            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            lambda_param = -beta[1]

            if lambda_param <= 0:
                return None

            half_life = np.log(2) / lambda_param
            return half_life

        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return None

    def check_half_life_breach(
        self,
        spread: pd.Series
    ) -> bool:
        """
        Check if half-life has exceeded threshold in rolling window.

        Args:
            spread: Cointegration spread

        Returns:
            True if breached, False otherwise
        """
        window = self.risk_config['position_stops']['half_life_window']
        max_half_life = self.risk_config['position_stops']['half_life_breach']

        if len(spread) < window:
            return False

        # Calculate half-life on rolling window
        recent_spread = spread.tail(window)
        half_life = self.calculate_half_life(recent_spread)

        if half_life is None:
            logger.warning("Half-life calculation failed")
            return True  # Conservative: treat as breach

        breached = half_life > max_half_life

        if breached:
            logger.warning(f"Half-life breach: {half_life:.2f} > {max_half_life}")

        return breached

    def check_position_drawdown(
        self,
        vector_id: int,
        current_pnl: float
    ) -> bool:
        """
        Check if position drawdown exceeds threshold.

        Args:
            vector_id: Vector identifier
            current_pnl: Current P&L for position

        Returns:
            True if breached, False otherwise
        """
        if vector_id not in self.position_entry_capital:
            return False

        allocated_capital = self.position_entry_capital[vector_id]
        drawdown_pct = current_pnl / allocated_capital

        max_drawdown = self.risk_config['position_stops']['position_drawdown_max']
        breached = drawdown_pct < -max_drawdown

        if breached:
            logger.warning(
                f"Position drawdown breach for vector {vector_id}: "
                f"{drawdown_pct*100:.2f}% < -{max_drawdown*100:.2f}%"
            )

        return breached

    def check_position_stops(
        self,
        vector_id: int,
        spread: pd.Series,
        zscore: float,
        current_date: datetime,
        current_pnl: float
    ) -> Tuple[bool, List[str]]:
        """
        Check all position-level stops.

        Args:
            vector_id: Vector identifier
            spread: Cointegration spread
            zscore: Current z-score
            current_date: Current date
            current_pnl: Current position P&L

        Returns:
            Tuple of (should_exit, list of reasons)
        """
        reasons = []

        # Check z-score breach
        if self.check_zscore_breach(zscore):
            reasons.append("Z-score breach")

        # Check time stop
        if self.check_time_stop(vector_id, current_date):
            reasons.append("Time stop")

        # Check ADF breach
        if self.check_adf_breach(spread):
            reasons.append("ADF p-value breach")

        # Check half-life breach
        if self.check_half_life_breach(spread):
            reasons.append("Half-life breach")

        # Check position drawdown
        if self.check_position_drawdown(vector_id, current_pnl):
            reasons.append("Position drawdown")

        should_exit = len(reasons) > 0

        if should_exit:
            logger.warning(f"Position stop triggered for vector {vector_id}: {', '.join(reasons)}")

        return should_exit, reasons

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Series of returns
            confidence: Confidence level (default: 95%)

        Returns:
            VaR value
        """
        if returns.empty:
            return 0

        var = returns.quantile(1 - confidence)
        return var

    def check_var_breach(
        self,
        daily_return: float,
        historical_returns: pd.Series
    ) -> bool:
        """
        Check if daily VaR is breached.

        Args:
            daily_return: Today's return
            historical_returns: Historical returns for VaR calculation

        Returns:
            True if breached, False otherwise
        """
        confidence = self.risk_config['portfolio_stops']['var_confidence']
        var = self.calculate_var(historical_returns, confidence)

        breached = daily_return < var

        if breached:
            logger.warning(f"VaR breach: {daily_return*100:.2f}% < {var*100:.2f}%")

        return breached

    def check_portfolio_drawdown(
        self,
        current_portfolio_value: float
    ) -> Optional[str]:
        """
        Check portfolio-level drawdown triggers.

        Args:
            current_portfolio_value: Current total portfolio value

        Returns:
            Action to take ('cut_50pct', 'exit_all', or None)
        """
        if current_portfolio_value > self.portfolio_high_water_mark:
            self.portfolio_high_water_mark = current_portfolio_value

        if self.portfolio_high_water_mark == 0:
            return None

        drawdown = (current_portfolio_value - self.portfolio_high_water_mark) / self.portfolio_high_water_mark

        # Check 15% drawdown (most severe)
        if drawdown <= -0.15:
            logger.critical(f"15% portfolio drawdown! Current drawdown: {drawdown*100:.2f}%")
            return 'exit_all'

        # Check 10% drawdown
        if drawdown <= -0.10:
            logger.error(f"10% portfolio drawdown! Current drawdown: {drawdown*100:.2f}%")
            return 'cut_50pct'

        return None

    def apply_drawdown_action(
        self,
        action: str,
        current_date: datetime
    ):
        """
        Apply portfolio drawdown action.

        Args:
            action: Action to take ('cut_50pct' or 'exit_all')
            current_date: Current date
        """
        if action == 'exit_all':
            self.trading_halted = True
            halt_days = self.risk_config['portfolio_stops']['halt_days']
            self.halt_until = current_date + timedelta(days=halt_days)

            logger.critical(
                f"TRADING HALTED until {self.halt_until.strftime('%Y-%m-%d')} "
                f"({halt_days} days)"
            )

        elif action == 'cut_50pct':
            logger.error("Cutting all position sizes by 50%")

    def is_trading_allowed(self, current_date: datetime) -> bool:
        """
        Check if trading is currently allowed.

        Args:
            current_date: Current date

        Returns:
            True if trading allowed, False otherwise
        """
        if not self.trading_halted:
            return True

        if current_date >= self.halt_until:
            self.trading_halted = False
            self.halt_until = None
            logger.info("Trading halt lifted")
            return True

        logger.warning(f"Trading halted until {self.halt_until.strftime('%Y-%m-%d')}")
        return False

    def register_position_entry(
        self,
        vector_id: int,
        entry_date: datetime,
        allocated_capital: float
    ):
        """
        Register a new position for tracking.

        Args:
            vector_id: Vector identifier
            entry_date: Entry date
            allocated_capital: Capital allocated to position
        """
        self.position_entry_dates[vector_id] = entry_date
        self.position_entry_capital[vector_id] = allocated_capital

        logger.info(f"Registered position {vector_id}: ${allocated_capital:,.2f} on {entry_date.strftime('%Y-%m-%d')}")

    def unregister_position(self, vector_id: int):
        """
        Remove position from tracking.

        Args:
            vector_id: Vector identifier
        """
        if vector_id in self.position_entry_dates:
            del self.position_entry_dates[vector_id]

        if vector_id in self.position_entry_capital:
            del self.position_entry_capital[vector_id]

        logger.info(f"Unregistered position {vector_id}")

    def get_risk_summary(self) -> Dict:
        """
        Get summary of current risk status.

        Returns:
            Dictionary with risk metrics
        """
        summary = {
            'num_active_positions': len(self.position_entry_dates),
            'trading_halted': self.trading_halted,
            'halt_until': self.halt_until.strftime('%Y-%m-%d') if self.halt_until else None,
            'portfolio_hwm': self.portfolio_high_water_mark,
            'total_allocated': sum(self.position_entry_capital.values()) if self.position_entry_capital else 0
        }

        return summary
