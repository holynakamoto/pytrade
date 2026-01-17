"""
Technical indicators module.
Calculates EMA, VWAP, Fair Value Gaps, and detects internal breakouts.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for trading signals."""

    def __init__(self, config: dict):
        """
        Initialize the technical indicators calculator.

        Args:
            config: Configuration dictionary with indicator parameters
        """
        self.config = config
        self.ema_short = config['indicators']['ema_short']
        self.ema_long = config['indicators']['ema_long']
        self.fvg_lookback = config['indicators']['fvg_lookback']
        self.swing_lookback = config['indicators']['swing_lookback']

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators on a DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        if df.empty:
            return df

        df = df.copy()

        # EMAs
        df = self.calculate_ema(df, self.ema_short, f'ema{self.ema_short}')
        df = self.calculate_ema(df, self.ema_long, f'ema{self.ema_long}')

        # VWAP
        df = self.calculate_vwap(df)

        # Internal Breakout signals
        df = self.detect_internal_breakout(df)

        # Fair Value Gaps
        df = self.detect_fair_value_gaps(df)

        # Swing highs/lows for stop loss
        df = self.calculate_swing_levels(df)

        return df

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int, column_name: str) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average.

        Args:
            df: DataFrame with price data
            period: EMA period
            column_name: Name for the new column

        Returns:
            DataFrame with EMA column added
        """
        df[column_name] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price (session-based).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with VWAP column added
        """
        # Typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Cumulative numerator and denominator
        cum_tp_vol = (typical_price * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()

        # Calculate VWAP with zero-volume protection
        df['vwap'] = cum_tp_vol.div(cum_vol).where(cum_vol > 0, pd.NA)

        return df

    @staticmethod
    def detect_internal_breakout(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect internal bar breakouts.

        Bullish: Current high > Previous high
        Bearish: Current low < Previous low

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with breakout columns added
        """
        # Bullish internal breakout: high > high[1]
        df['bullish_breakout'] = df['high'] > df['high'].shift(1)

        # Bearish internal breakout: low < low[1]
        df['bearish_breakout'] = df['low'] < df['low'].shift(1)

        return df

    def detect_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG).

        Bullish FVG: Gap between bar[2].low and bar[0].high (bar[1] creates the gap)
        Bearish FVG: Gap between bar[2].high and bar[0].low

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with FVG columns added
        """
        # Bullish FVG: low[0] > high[2] (current low above high from 2 bars ago)
        # This creates an unfilled gap
        df['bullish_fvg'] = df['low'] > df['high'].shift(2)

        # Bearish FVG: high[0] < low[2] (current high below low from 2 bars ago)
        df['bearish_fvg'] = df['high'] < df['low'].shift(2)

        # Check if FVG exists in recent lookback period
        df['has_recent_bullish_fvg'] = (
            df['bullish_fvg'].rolling(window=self.fvg_lookback).sum() > 0
        )

        df['has_recent_bearish_fvg'] = (
            df['bearish_fvg'].rolling(window=self.fvg_lookback).sum() > 0
        )

        return df

    def calculate_swing_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate swing highs and lows for stop-loss placement.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with swing level columns added
        """
        # Swing low: lowest low in the lookback period
        df['swing_low'] = df['low'].rolling(window=self.swing_lookback).min()

        # Swing high: highest high in the lookback period
        df['swing_high'] = df['high'].rolling(window=self.swing_lookback).max()

        return df

    @staticmethod
    def get_trend_direction(df: pd.DataFrame, ema_column: str = 'ema200') -> pd.DataFrame:
        """
        Determine overall trend based on price vs EMA.

        Args:
            df: DataFrame with price and EMA data
            ema_column: Name of the EMA column to use

        Returns:
            DataFrame with trend column added
        """
        df['trend'] = np.where(df['close'] > df[ema_column], 'up', 'down')
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (for future use in position sizing).

        Args:
            df: DataFrame with OHLCV data
            period: ATR period

        Returns:
            DataFrame with ATR column added
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        df['atr'] = true_range.rolling(period).mean()

        return df

    def validate_indicators(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required indicators have been calculated.

        Args:
            df: DataFrame with indicators

        Returns:
            True if all indicators are present
        """
        required_columns = [
            f'ema{self.ema_short}',
            f'ema{self.ema_long}',
            'vwap',
            'bullish_breakout',
            'bearish_breakout',
            'has_recent_bullish_fvg',
            'has_recent_bearish_fvg',
            'swing_low',
            'swing_high'
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            logger.error(f"Missing required indicator columns: {missing}")
            return False

        return True
