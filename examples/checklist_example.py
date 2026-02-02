#!/usr/bin/env python3
"""
Trading Strategy Checklist Example.
Demonstrates how to use the checklist module for trade validation.

Checklist Items:
- Pre-Market Analysis
- Key Level
- Internal Breakout Indicator
- Wait for Candle to Close
- If FVG - take it!!
- Base Entry: .50
- TP1: .65 (30%)
- TP2: .75 (50%)
- TP3: .90 (1:1)
- If ARROW indicator, wait for candle close + FVG
- EXIT: If Int. OB invalidated
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from strategy_checklist import (
    TradingChecklist,
    ChecklistValidator,
    TradeSetup,
    ChecklistStatus
)
from data import DataFetcher
from indicators import TechnicalIndicators


def example_manual_checklist():
    """
    Example: Manual checklist usage for trade validation.
    """
    print("=" * 60)
    print("MANUAL CHECKLIST EXAMPLE")
    print("=" * 60)

    # Create checklist with default items
    checklist = TradingChecklist()

    # Start a new trade setup
    setup = checklist.start_new_trade(
        symbol="SPY",
        direction="LONG",
        key_level=450.00
    )

    # Simulate completing pre-market analysis
    checklist.check_item('pre_market_analysis', True,
                        "Reviewed overnight action, identified key levels")

    # Confirm key level
    checklist.check_item('key_level', True,
                        "Major support at $450.00")

    # Calculate trade levels based on swing range
    swing_high = 455.00
    swing_low = 445.00
    checklist.calculate_trade_levels(swing_high, swing_low)

    # Check entry conditions
    checklist.check_item('internal_breakout', True,
                        "High > High[1] confirmed")

    checklist.check_item('candle_close', True,
                        "15-min candle closed")

    checklist.check_item('fvg_present', True,
                        "Bullish FVG identified - TAKE IT!")

    # Skip ARROW indicator (not present)
    checklist.skip_item('arrow_indicator', "No arrow signal today")

    # Print the checklist
    print(checklist.print_checklist())

    # Check if ready
    ready, missing = checklist.is_ready_to_trade()
    if ready:
        print("*** TRADE VALIDATED - Ready to execute! ***")
        print(f"\nTrade Details:")
        print(f"  Entry: ${setup.entry_price:.2f}")
        print(f"  Stop Loss: ${setup.stop_loss:.2f}")
        print(f"  Risk: ${setup.risk_amount:.2f}")
        for tp in setup.take_profits:
            print(f"  {tp.name}: ${tp.price:.2f} (Exit {tp.exit_pct}%)")
    else:
        print(f"*** NOT READY - Missing: {', '.join(missing)} ***")


def example_automated_validation():
    """
    Example: Automated checklist validation using indicator data.
    """
    print("\n" + "=" * 60)
    print("AUTOMATED VALIDATION EXAMPLE")
    print("=" * 60)

    # Configuration
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
        'strategy_checklist': {
            'enabled': True,
            'entry_level': 0.50,
            'take_profit_levels': {
                'tp1': {'level': 0.65, 'exit_pct': 30},
                'tp2': {'level': 0.75, 'exit_pct': 50},
                'tp3': {'level': 0.90, 'exit_pct': 20},
            },
            'require_fvg': True,
            'require_internal_breakout': True,
            'require_candle_close': True,
        }
    }

    symbol = 'SPY'
    print(f"\nFetching data for {symbol}...")

    # Fetch and process data
    data_fetcher = DataFetcher(config)
    indicator_calc = TechnicalIndicators(config)

    df = data_fetcher.fetch_data(symbol)
    if df.empty:
        print("Failed to fetch data")
        return

    df = indicator_calc.calculate_all_indicators(df)
    print(f"Processed {len(df)} bars with indicators")

    # Create checklist and validator
    checklist = TradingChecklist(config)
    validator = ChecklistValidator(config)

    # Get latest bar
    latest_idx = df.index[-1]
    latest = df.loc[latest_idx]

    # Determine direction based on trend
    direction = 'LONG' if latest['close'] > latest[f'ema{config["indicators"]["ema_long"]}'] else 'SHORT'

    # Start trade setup
    key_level = latest['close']
    setup = checklist.start_new_trade(symbol, direction, key_level)

    # Auto-validate conditions from indicator data
    print(f"\nValidating {direction} setup...")
    results = checklist.validate_from_dataframe(df, latest_idx)

    # Calculate levels using swing high/low
    swing_high = latest.get('swing_high', latest['high'])
    swing_low = latest.get('swing_low', latest['low'])
    checklist.calculate_trade_levels(swing_high, swing_low)

    # Mark pre-market items
    checklist.check_item('pre_market_analysis', True, "Auto-analysis")
    checklist.check_item('key_level', True, f"${key_level:.2f}")

    # Validate entry using ChecklistValidator
    is_valid, reason = validator.validate_entry(df, latest_idx, direction)
    print(f"\nEntry Validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"Reason: {reason}")

    # Print checklist
    print(checklist.print_checklist())

    # Show market conditions
    print("\nCurrent Market Conditions:")
    print(f"  Close: ${latest['close']:.2f}")
    print(f"  EMA9: ${latest[f'ema{config[\"indicators\"][\"ema_short\"]}']:.2f}")
    print(f"  EMA200: ${latest[f'ema{config[\"indicators\"][\"ema_long\"]}']:.2f}")
    print(f"  Bullish Breakout: {latest.get('bullish_breakout', False)}")
    print(f"  Bearish Breakout: {latest.get('bearish_breakout', False)}")
    print(f"  Bullish FVG: {latest.get('has_recent_bullish_fvg', False)}")
    print(f"  Bearish FVG: {latest.get('has_recent_bearish_fvg', False)}")


def example_tp_levels():
    """
    Example: Demonstrate take profit level calculations.
    """
    print("\n" + "=" * 60)
    print("TAKE PROFIT LEVELS EXAMPLE")
    print("=" * 60)

    # Scenario: LONG trade
    print("\n--- LONG Trade Example ---")
    print("Swing Low: $100.00")
    print("Swing High: $110.00")
    print("Range: $10.00")

    setup = TradeSetup(symbol="TEST", direction="LONG", key_level=100.00)
    setup.calculate_levels(swing_high=110.00, swing_low=100.00)

    print(f"\nCalculated Levels:")
    print(f"  Entry (.50): ${setup.entry_price:.2f}")
    print(f"  Stop Loss: ${setup.stop_loss:.2f}")
    for tp in setup.take_profits:
        print(f"  {tp.name} (.{int(tp.level*100)}): ${tp.price:.2f} - Exit {tp.exit_pct}%")

    print(f"\n  Risk per share: ${setup.risk_amount:.2f}")

    # Scenario: SHORT trade
    print("\n--- SHORT Trade Example ---")
    print("Swing High: $110.00")
    print("Swing Low: $100.00")
    print("Range: $10.00")

    setup_short = TradeSetup(symbol="TEST", direction="SHORT", key_level=110.00)
    setup_short.calculate_levels(swing_high=110.00, swing_low=100.00)

    print(f"\nCalculated Levels:")
    print(f"  Entry (.50): ${setup_short.entry_price:.2f}")
    print(f"  Stop Loss: ${setup_short.stop_loss:.2f}")
    for tp in setup_short.take_profits:
        print(f"  {tp.name} (.{int(tp.level*100)}): ${tp.price:.2f} - Exit {tp.exit_pct}%")

    print(f"\n  Risk per share: ${setup_short.risk_amount:.2f}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# TRADING STRATEGY CHECKLIST EXAMPLES")
    print("#" * 60)

    # Run examples
    example_manual_checklist()
    example_tp_levels()

    # Automated example requires network access
    try:
        example_automated_validation()
    except Exception as e:
        print(f"\nAutomated validation skipped: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
