"""
Signal generation module.
Identifies bullish and bearish entry opportunities based on configured filters.
"""

import logging
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class Signal:
    """Represents a trading signal."""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        date: str,
        rationale: str
    ):
        """
        Initialize a trading signal.

        Args:
            symbol: Trading symbol
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit target (1:1 RR)
            take_profit_2: Second take profit target (1:2 RR)
            date: Signal date
            rationale: Reason for the signal
        """
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit_1 = take_profit_1
        self.take_profit_2 = take_profit_2
        self.date = date
        self.rationale = rationale

    def to_dict(self) -> Dict:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'tp1': self.take_profit_1,
            'tp2': self.take_profit_2,
            'date': self.date,
            'rationale': self.rationale
        }

    def __str__(self) -> str:
        """String representation of signal."""
        return (
            f"{self.symbol} {self.direction} @ {self.entry_price:.2f} "
            f"| SL: {self.stop_loss:.2f} | TP1: {self.take_profit_1:.2f} "
            f"| TP2: {self.take_profit_2:.2f}"
        )


class SignalGenerator:
    """Generates trading signals based on technical conditions."""

    def __init__(self, config: dict):
        """
        Initialize the signal generator.

        Args:
            config: Configuration dictionary with entry filter settings
        """
        self.config = config
        self.filters = config['entry_filters']
        self.ema_short = config['indicators']['ema_short']
        self.ema_long = config['indicators']['ema_long']

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        max_signals: Optional[int] = None
    ) -> List[Signal]:
        """
        Generate signals from a DataFrame with indicators.

        Args:
            df: DataFrame with OHLCV and indicator data
            symbol: Trading symbol
            max_signals: Maximum number of signals to return

        Returns:
            List of Signal objects
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}, skipping signal generation")
            return []

        signals = []

        # Get the last row (most recent bar)
        # In backtesting, we'll iterate through all rows
        last_idx = df.index[-1]
        last_row = df.loc[last_idx]

        # Check for bullish signal
        if self._check_bullish_conditions(df, last_idx):
            signal = self._create_bullish_signal(last_row, symbol, last_idx)
            if signal:
                signals.append(signal)

        # Check for bearish signal
        if self._check_bearish_conditions(df, last_idx):
            signal = self._create_bearish_signal(last_row, symbol, last_idx)
            if signal:
                signals.append(signal)

        # Limit number of signals
        if max_signals and len(signals) > max_signals:
            signals = signals[:max_signals]

        return signals

    def _check_bullish_conditions(self, df: pd.DataFrame, idx) -> bool:
        """
        Check if all bullish entry conditions are met.

        Conditions:
        1. Close > EMA9 (momentum)
        2. Close > VWAP (institutional level)
        3. Close > EMA200 (uptrend)
        4. High > High[1] (internal breakout)
        5. Bullish FVG present in recent bars

        Args:
            df: DataFrame with indicators
            idx: Index to check

        Returns:
            True if all conditions are met
        """
        try:
            row = df.loc[idx]

            # Get filter requirements
            require_ema200 = self.filters.get('require_ema200_trend', True)
            require_ema9 = self.filters.get('require_ema9_alignment', True)
            require_vwap = self.filters.get('require_vwap_alignment', True)
            require_breakout = self.filters.get('require_internal_breakout', True)
            require_fvg = self.filters.get('require_fvg', True)

            conditions = []

            # EMA9 alignment
            if require_ema9:
                ema9_col = f'ema{self.ema_short}'
                if ema9_col in row and pd.notna(row[ema9_col]):
                    conditions.append(row['close'] > row[ema9_col])
                else:
                    return False

            # VWAP alignment
            if require_vwap:
                if 'vwap' in row and pd.notna(row['vwap']):
                    conditions.append(row['close'] > row['vwap'])
                else:
                    return False

            # EMA200 trend
            if require_ema200:
                ema200_col = f'ema{self.ema_long}'
                if ema200_col in row and pd.notna(row[ema200_col]):
                    conditions.append(row['close'] > row[ema200_col])
                else:
                    return False

            # Internal breakout
            if require_breakout:
                if 'bullish_breakout' in row:
                    conditions.append(row['bullish_breakout'] == True)
                else:
                    return False

            # Fair Value Gap
            if require_fvg:
                if 'has_recent_bullish_fvg' in row:
                    conditions.append(row['has_recent_bullish_fvg'] == True)
                else:
                    return False

            return all(conditions)

        except Exception as e:
            logger.error(f"Error checking bullish conditions: {e}")
            return False

    def _check_bearish_conditions(self, df: pd.DataFrame, idx) -> bool:
        """
        Check if all bearish entry conditions are met.

        Conditions (opposite of bullish):
        1. Close < EMA9
        2. Close < VWAP
        3. Close < EMA200 (downtrend)
        4. Low < Low[1] (internal breakout)
        5. Bearish FVG present in recent bars

        Args:
            df: DataFrame with indicators
            idx: Index to check

        Returns:
            True if all conditions are met
        """
        try:
            row = df.loc[idx]

            # Get filter requirements
            require_ema200 = self.filters.get('require_ema200_trend', True)
            require_ema9 = self.filters.get('require_ema9_alignment', True)
            require_vwap = self.filters.get('require_vwap_alignment', True)
            require_breakout = self.filters.get('require_internal_breakout', True)
            require_fvg = self.filters.get('require_fvg', True)

            conditions = []

            # EMA9 alignment
            if require_ema9:
                ema9_col = f'ema{self.ema_short}'
                if ema9_col in row and pd.notna(row[ema9_col]):
                    conditions.append(row['close'] < row[ema9_col])
                else:
                    return False

            # VWAP alignment
            if require_vwap:
                if 'vwap' in row and pd.notna(row['vwap']):
                    conditions.append(row['close'] < row['vwap'])
                else:
                    return False

            # EMA200 trend
            if require_ema200:
                ema200_col = f'ema{self.ema_long}'
                if ema200_col in row and pd.notna(row[ema200_col]):
                    conditions.append(row['close'] < row[ema200_col])
                else:
                    return False

            # Internal breakout
            if require_breakout:
                if 'bearish_breakout' in row:
                    conditions.append(row['bearish_breakout'] == True)
                else:
                    return False

            # Fair Value Gap
            if require_fvg:
                if 'has_recent_bearish_fvg' in row:
                    conditions.append(row['has_recent_bearish_fvg'] == True)
                else:
                    return False

            return all(conditions)

        except Exception as e:
            logger.error(f"Error checking bearish conditions: {e}")
            return False

    def _create_bullish_signal(self, row, symbol: str, date) -> Optional[Signal]:
        """Create a bullish signal with risk management levels."""
        try:
            entry_price = row['close']
            stop_loss = row['swing_low'] if pd.notna(row.get('swing_low')) else row['low']

            # Calculate risk
            risk = entry_price - stop_loss

            if risk <= 0:
                logger.warning(f"Invalid risk for bullish signal on {symbol}: {risk}")
                return None

            # Calculate targets
            tp1 = entry_price + risk  # 1:1 RR
            tp2 = entry_price + (2 * risk)  # 1:2 RR

            # Build rationale
            rationale = self._build_rationale(row, 'LONG')

            return Signal(
                symbol=symbol,
                direction='LONG',
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                date=str(date),
                rationale=rationale
            )

        except Exception as e:
            logger.error(f"Error creating bullish signal: {e}")
            return None

    def _create_bearish_signal(self, row, symbol: str, date) -> Optional[Signal]:
        """Create a bearish signal with risk management levels."""
        try:
            entry_price = row['close']
            stop_loss = row['swing_high'] if pd.notna(row.get('swing_high')) else row['high']

            # Calculate risk
            risk = stop_loss - entry_price

            if risk <= 0:
                logger.warning(f"Invalid risk for bearish signal on {symbol}: {risk}")
                return None

            # Calculate targets
            tp1 = entry_price - risk  # 1:1 RR
            tp2 = entry_price - (2 * risk)  # 1:2 RR

            # Build rationale
            rationale = self._build_rationale(row, 'SHORT')

            return Signal(
                symbol=symbol,
                direction='SHORT',
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                date=str(date),
                rationale=rationale
            )

        except Exception as e:
            logger.error(f"Error creating bearish signal: {e}")
            return None

    def _build_rationale(self, row, direction: str) -> str:
        """Build a rationale string explaining why the signal was generated."""
        reasons = []

        ema9_col = f'ema{self.ema_short}'
        ema200_col = f'ema{self.ema_long}'

        if direction == 'LONG':
            if row['close'] > row.get(ema200_col, 0):
                reasons.append("Uptrend (>EMA200)")
            if row['close'] > row.get(ema9_col, 0):
                reasons.append("Momentum (>EMA9)")
            if row['close'] > row.get('vwap', 0):
                reasons.append("Above VWAP")
            if row.get('bullish_breakout'):
                reasons.append("Bullish breakout")
            if row.get('has_recent_bullish_fvg'):
                reasons.append("Bullish FVG present")
        else:  # SHORT
            if row['close'] < row.get(ema200_col, float('inf')):
                reasons.append("Downtrend (<EMA200)")
            if row['close'] < row.get(ema9_col, float('inf')):
                reasons.append("Momentum (<EMA9)")
            if row['close'] < row.get('vwap', float('inf')):
                reasons.append("Below VWAP")
            if row.get('bearish_breakout'):
                reasons.append("Bearish breakout")
            if row.get('has_recent_bearish_fvg'):
                reasons.append("Bearish FVG present")

        return " | ".join(reasons) if reasons else "Signal conditions met"
