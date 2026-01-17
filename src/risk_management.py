"""
Risk management module.
Handles position sizing, stop-loss, and take-profit calculations.
"""

import logging
from typing import Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk parameters for trades."""

    def __init__(self, config: dict):
        """
        Initialize the risk manager.

        Args:
            config: Configuration dictionary with risk management settings
        """
        self.config = config
        self.risk_pct = config['risk_management']['risk_per_trade_pct']
        self.account_size = config['risk_management']['account_size']
        self.tp1_rr = config['risk_management']['tp1_rr']
        self.tp2_rr = config['risk_management']['tp2_rr']
        self.tp1_exit_pct = config['risk_management']['tp1_exit_pct']
        self.tp2_exit_pct = config['risk_management']['tp2_exit_pct']
        self.use_ema200_exit = config['risk_management']['use_ema200_hard_exit']

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_equity: float = None
    ) -> int:
        """
        Calculate position size based on risk percentage.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            account_equity: Current account equity (uses initial if None)

        Returns:
            Number of shares/contracts to trade
        """
        if account_equity is None:
            account_equity = self.account_size

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            logger.warning("Risk per share is zero, cannot calculate position size")
            return 0

        # Calculate dollar risk
        dollar_risk = account_equity * (self.risk_pct / 100)

        # Calculate position size
        position_size = int(dollar_risk / risk_per_share)

        return max(position_size, 1)  # At least 1 share

    def calculate_take_profits(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str
    ) -> Tuple[float, float]:
        """
        Calculate take profit levels based on risk-reward ratios.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 'LONG' or 'SHORT'

        Returns:
            Tuple of (TP1, TP2) prices
        """
        risk = abs(entry_price - stop_loss)

        if direction.upper() == 'LONG':
            tp1 = entry_price + (risk * self.tp1_rr)
            tp2 = entry_price + (risk * self.tp2_rr)
        else:  # SHORT
            tp1 = entry_price - (risk * self.tp1_rr)
            tp2 = entry_price - (risk * self.tp2_rr)

        return tp1, tp2

    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        direction: str,
        lookback: int = 5
    ) -> float:
        """
        Calculate stop loss based on recent swing levels.

        Args:
            df: DataFrame with price data
            direction: 'LONG' or 'SHORT'
            lookback: Number of bars to look back for swing levels

        Returns:
            Stop loss price
        """
        if df.empty or len(df) < lookback:
            logger.warning("Insufficient data for stop loss calculation")
            return None

        recent_data = df.iloc[-lookback:]

        if direction.upper() == 'LONG':
            # For longs, stop below recent swing low
            stop_loss = recent_data['low'].min()
        else:  # SHORT
            # For shorts, stop above recent swing high
            stop_loss = recent_data['high'].max()

        return stop_loss

    def check_ema200_exit(
        self,
        current_price: float,
        ema200: float,
        direction: str
    ) -> bool:
        """
        Check if price has crossed EMA200 against the trade direction.

        Args:
            current_price: Current price
            ema200: EMA200 value
            direction: 'LONG' or 'SHORT'

        Returns:
            True if should exit
        """
        if not self.use_ema200_exit:
            return False

        if direction.upper() == 'LONG':
            # Exit long if price crosses below EMA200
            return current_price < ema200
        else:  # SHORT
            # Exit short if price crosses above EMA200
            return current_price > ema200

    def update_account_equity(self, pnl: float) -> float:
        """
        Update account equity based on profit/loss.

        Args:
            pnl: Profit or loss from trade

        Returns:
            New account equity
        """
        self.account_size += pnl
        return self.account_size

    def get_risk_metrics(self) -> Dict:
        """
        Get current risk management metrics.

        Returns:
            Dictionary of risk metrics
        """
        return {
            'account_size': self.account_size,
            'risk_per_trade_pct': self.risk_pct,
            'risk_per_trade_dollars': self.account_size * (self.risk_pct / 100),
            'tp1_rr': self.tp1_rr,
            'tp2_rr': self.tp2_rr,
            'tp1_exit_pct': self.tp1_exit_pct,
            'tp2_exit_pct': self.tp2_exit_pct
        }

    def validate_trade(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        direction: str
    ) -> Tuple[bool, str]:
        """
        Validate trade parameters.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit
            take_profit_2: Second take profit
            direction: 'LONG' or 'SHORT'

        Returns:
            Tuple of (is_valid, reason)
        """
        if direction.upper() == 'LONG':
            # For longs: SL < Entry < TP1 < TP2
            if not (stop_loss < entry_price < take_profit_1 < take_profit_2):
                return False, "Invalid price levels for LONG trade"

        else:  # SHORT
            # For shorts: TP2 < TP1 < Entry < SL
            if not (take_profit_2 < take_profit_1 < entry_price < stop_loss):
                return False, "Invalid price levels for SHORT trade"

        # Check minimum risk-reward
        risk = abs(entry_price - stop_loss)
        reward1 = abs(take_profit_1 - entry_price)
        reward2 = abs(take_profit_2 - entry_price)

        actual_rr1 = reward1 / risk if risk > 0 else 0
        actual_rr2 = reward2 / risk if risk > 0 else 0

        if actual_rr1 < self.tp1_rr - 0.01:  # Allow small tolerance
            return False, f"TP1 risk-reward {actual_rr1:.2f} less than required {self.tp1_rr}"

        if actual_rr2 < self.tp2_rr - 0.01:
            return False, f"TP2 risk-reward {actual_rr2:.2f} less than required {self.tp2_rr}"

        return True, "Trade parameters valid"

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        position_size: int,
        direction: str
    ) -> float:
        """
        Calculate profit/loss for a trade.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            position_size: Number of shares/contracts
            direction: 'LONG' or 'SHORT'

        Returns:
            Profit or loss in dollars
        """
        if direction.upper() == 'LONG':
            pnl = (exit_price - entry_price) * position_size
        else:  # SHORT
            pnl = (entry_price - exit_price) * position_size

        return pnl
