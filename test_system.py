#!/usr/bin/env python3
"""
Simple test of the trading system with synthetic data.
Tests core functionality without requiring external data sources.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators import TechnicalIndicators
from signals import SignalGenerator
from risk_management import RiskManager


def generate_test_data(bars=300):
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=bars, freq='D')

    # Generate uptrending price data
    base_price = 100
    trend = np.linspace(0, 50, bars)  # Uptrend
    noise = np.random.randn(bars) * 2
    close_prices = base_price + trend + noise

    # Generate OHLCV
    data = {
        'open': close_prices + np.random.randn(bars) * 0.5,
        'high': close_prices + np.abs(np.random.randn(bars)) * 2,
        'low': close_prices - np.abs(np.random.randn(bars)) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, bars)
    }

    df = pd.DataFrame(data, index=dates)

    # Ensure OHLC relationships are valid
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    return df


def test_system():
    """Test the trading system with synthetic data."""
    print("="*60)
    print("Testing Daily Breakout FVG Trading System")
    print("="*60)

    # Configuration
    config = {
        'indicators': {
            'ema_short': 9,
            'ema_long': 200,
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

    # Generate test data
    print("\n1. Generating synthetic data...")
    df = generate_test_data(300)
    print(f"   Generated {len(df)} bars of OHLCV data")

    # Test indicators
    print("\n2. Testing indicators module...")
    indicator_calc = TechnicalIndicators(config)
    df = indicator_calc.calculate_all_indicators(df)

    if indicator_calc.validate_indicators(df):
        print("   ✓ All indicators calculated successfully")
        print(f"   - EMA9: {df['ema9'].iloc[-1]:.2f}")
        print(f"   - EMA200: {df['ema200'].iloc[-1]:.2f}")
        print(f"   - VWAP: {df['vwap'].iloc[-1]:.2f}")
        print(f"   - Recent Bullish FVG: {df['has_recent_bullish_fvg'].iloc[-1]}")
    else:
        print("   ✗ Indicator validation failed")
        return False

    # Test signal generation
    print("\n3. Testing signal generation...")
    signal_gen = SignalGenerator(config)
    signals = signal_gen.generate_signals(df, 'TEST', max_signals=5)

    print(f"   Generated {len(signals)} signal(s)")

    if signals:
        for i, sig in enumerate(signals, 1):
            print(f"\n   Signal {i}:")
            print(f"   - Direction: {sig.direction}")
            print(f"   - Entry: ${sig.entry_price:.2f}")
            print(f"   - Stop Loss: ${sig.stop_loss:.2f}")
            print(f"   - TP1: ${sig.take_profit_1:.2f}")
            print(f"   - TP2: ${sig.take_profit_2:.2f}")
            print(f"   - Rationale: {sig.rationale}")
    else:
        print("   (No signals - conditions not met, which is expected with synthetic data)")

    # Test risk management
    print("\n4. Testing risk management...")
    risk_mgr = RiskManager(config)

    # Test with example trade
    entry = 150.0
    stop_loss = 147.0
    direction = 'LONG'

    position_size = risk_mgr.calculate_position_size(entry, stop_loss)
    tp1, tp2 = risk_mgr.calculate_take_profits(entry, stop_loss, direction)
    is_valid, reason = risk_mgr.validate_trade(entry, stop_loss, tp1, tp2, direction)

    print(f"   Example LONG trade:")
    print(f"   - Entry: ${entry:.2f}")
    print(f"   - Stop Loss: ${stop_loss:.2f}")
    print(f"   - Position Size: {position_size} shares")
    print(f"   - TP1 (1:1): ${tp1:.2f}")
    print(f"   - TP2 (1:2): ${tp2:.2f}")
    print(f"   - Valid: {is_valid} - {reason}")

    # Test PnL calculation
    exit_price = tp1
    pnl = risk_mgr.calculate_pnl(entry, exit_price, position_size, direction)
    print(f"   - PnL at TP1: ${pnl:.2f}")

    print("\n" + "="*60)
    print("✓ All core components tested successfully!")
    print("="*60)

    return True


if __name__ == '__main__':
    try:
        success = test_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
