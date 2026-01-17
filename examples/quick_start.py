#!/usr/bin/env python3
"""
Quick start example for the Daily Breakout FVG Trading System.
This script demonstrates basic usage without requiring full configuration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
from datetime import datetime, timedelta
import pandas as pd

from data import DataFetcher
from indicators import TechnicalIndicators
from signals import SignalGenerator
from risk_management import RiskManager


def quick_start_example():
    """
    Quick start example: Generate signals for a single stock.
    """
    print("="*60)
    print("Daily Breakout FVG Trading System - Quick Start")
    print("="*60)

    # Simple configuration
    config = {
        'data_source': {
            'primary': 'yahoo',
            'cache_days': 300,
            'polygon_api_key': '',
            'alphavantage_api_key': ''
        },
        'indicators': {
            'ema_short': 9,
            'ema_long': 200,
            'vwap_session': 'daily',
            'fvg_lookback': 5,
            'swing_lookback': 5
        },
        'entry_filters': {
            'require_ema200_trend': True,
            'require_ema9_alignment': True,
            'require_vwap_alignment': True,
            'require_internal_breakout': True,
            'require_fvg': True
        },
        'risk_management': {
            'risk_per_trade_pct': 1.0,
            'account_size': 10000,
            'tp1_rr': 1.0,
            'tp2_rr': 2.0,
            'tp1_exit_pct': 50,
            'tp2_exit_pct': 100,
            'use_ema200_hard_exit': True
        }
    }

    # Test symbol
    symbol = 'SPY'
    print(f"\nAnalyzing {symbol}...\n")

    # Initialize components
    data_fetcher = DataFetcher(config)
    indicator_calc = TechnicalIndicators(config)
    signal_gen = SignalGenerator(config)
    risk_mgr = RiskManager(config)

    # Fetch data
    print("Fetching data...")
    df = data_fetcher.fetch_data(symbol)

    if df.empty:
        print(f"Failed to fetch data for {symbol}")
        return

    print(f"Fetched {len(df)} bars")

    # Calculate indicators
    print("Calculating indicators...")
    df = indicator_calc.calculate_all_indicators(df)

    # Validate
    if not indicator_calc.validate_indicators(df):
        print("Indicator validation failed")
        return

    # Generate signals
    print("Generating signals...")
    signals = signal_gen.generate_signals(df, symbol)

    # Display results
    if signals:
        print(f"\n{'='*60}")
        print(f"SIGNALS GENERATED: {len(signals)}")
        print(f"{'='*60}\n")

        for sig in signals:
            print(f"Symbol: {sig.symbol}")
            print(f"Direction: {sig.direction}")
            print(f"Entry: ${sig.entry_price:.2f}")
            print(f"Stop Loss: ${sig.stop_loss:.2f}")
            print(f"Take Profit 1 (1:1): ${sig.take_profit_1:.2f}")
            print(f"Take Profit 2 (1:2): ${sig.take_profit_2:.2f}")
            print(f"Rationale: {sig.rationale}")
            print(f"Date: {sig.date}")

            # Calculate position size
            position_size = risk_mgr.calculate_position_size(
                sig.entry_price,
                sig.stop_loss
            )
            risk_dollars = abs(sig.entry_price - sig.stop_loss) * position_size

            print(f"\nRisk Management:")
            print(f"  Position Size: {position_size} shares")
            print(f"  Risk per Share: ${abs(sig.entry_price - sig.stop_loss):.2f}")
            print(f"  Total Risk: ${risk_dollars:.2f}")
            print(f"  Risk %: {(risk_dollars / config['risk_management']['account_size']) * 100:.2f}%")
            print("\n" + "-"*60 + "\n")
    else:
        print("\nNo signals generated. Conditions not met.")

    # Show latest market data
    print("\nLatest Market Data:")
    print("-"*60)
    latest = df.iloc[-1]
    ema_short_col = f"ema{config['indicators']['ema_short']}"
    ema_long_col = f"ema{config['indicators']['ema_long']}"
    print(f"Close: ${latest['close']:.2f}")
    print(f"EMA9: ${latest[ema_short_col]:.2f}")
    print(f"EMA200: ${latest[ema_long_col]:.2f}")
    print(f"VWAP: ${latest['vwap']:.2f}")
    print(f"Bullish Breakout: {latest['bullish_breakout']}")
    print(f"Bearish Breakout: {latest['bearish_breakout']}")
    print(f"Recent Bullish FVG: {latest['has_recent_bullish_fvg']}")
    print(f"Recent Bearish FVG: {latest['has_recent_bearish_fvg']}")
    print("="*60)


if __name__ == '__main__':
    try:
        quick_start_example()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
