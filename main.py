#!/usr/bin/env python3
"""
Daily Breakout FVG Trading System
Main orchestration script for generating daily trading signals and running backtests.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import DataFetcher
from indicators import TechnicalIndicators
from signals import SignalGenerator
from risk_management import RiskManager
from backtest import Backtester


def setup_logging(config: dict):
    """Set up logging configuration."""
    log_level = config['output']['log_level']
    log_path = config['output']['log_path']

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_daily_signals(config: dict, logger) -> pd.DataFrame:
    """
    Generate daily trading signals for configured symbols.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        DataFrame with daily signals
    """
    logger.info("Starting daily signal generation...")

    # Initialize components
    data_fetcher = DataFetcher(config)
    indicator_calc = TechnicalIndicators(config)
    signal_gen = SignalGenerator(config)
    risk_mgr = RiskManager(config)

    # Fetch data for all symbols
    symbols = config['symbols']
    logger.info(f"Fetching data for {len(symbols)} symbols...")

    data_dict = data_fetcher.fetch_multiple(symbols)

    if not data_dict:
        logger.error("No data fetched for any symbols")
        return pd.DataFrame()

    logger.info(f"Successfully fetched data for {len(data_dict)} symbols")

    # Generate signals
    all_signals = []

    for symbol, df in data_dict.items():
        logger.info(f"Processing {symbol}...")

        # Calculate indicators
        df_with_indicators = indicator_calc.calculate_all_indicators(df)

        # Validate indicators
        if not indicator_calc.validate_indicators(df_with_indicators):
            logger.warning(f"Indicator validation failed for {symbol}, skipping")
            continue

        # Generate signals
        signals = signal_gen.generate_signals(
            df_with_indicators,
            symbol,
            max_signals=config['targets']['max_signals_per_day']
        )

        if signals:
            logger.info(f"Generated {len(signals)} signal(s) for {symbol}")
            for signal in signals:
                # Validate trade parameters
                is_valid, reason = risk_mgr.validate_trade(
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit_1,
                    signal.take_profit_2,
                    signal.direction
                )

                if is_valid:
                    all_signals.append(signal.to_dict())
                else:
                    logger.warning(f"Invalid signal for {symbol}: {reason}")

    # Convert to DataFrame
    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        logger.info(f"\nGenerated {len(signals_df)} total signals")
        return signals_df
    else:
        logger.info("No signals generated")
        return pd.DataFrame()


def run_backtest(config: dict, logger):
    """
    Run backtest on historical data.

    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Starting backtest...")

    # Initialize components
    data_fetcher = DataFetcher(config)
    indicator_calc = TechnicalIndicators(config)
    signal_gen = SignalGenerator(config)
    risk_mgr = RiskManager(config)
    backtester = Backtester(config, risk_mgr, signal_gen)

    # Fetch historical data
    symbols = config['symbols']
    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']

    logger.info(f"Fetching historical data from {start_date} to {end_date}...")

    data_dict = data_fetcher.fetch_multiple(symbols, start_date, end_date)

    if not data_dict:
        logger.error("No data fetched for backtesting")
        return

    logger.info(f"Fetched data for {len(data_dict)} symbols")

    # Calculate indicators for all symbols
    logger.info("Calculating indicators...")
    for symbol in data_dict.keys():
        data_dict[symbol] = indicator_calc.calculate_all_indicators(data_dict[symbol])

    # Run backtest
    results = backtester.run_backtest(data_dict, verbose=True)

    # Print results
    backtester.print_results(results)

    # Save trade history
    if results.trades:
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_date': t.entry_date,
                'entry_price': t.entry_price,
                'exit_date': t.exit_date,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason
            }
            for t in results.trades
        ])

        output_path = './output/backtest_trades.csv'
        trades_df.to_csv(output_path, index=False)
        logger.info(f"\nTrade history saved to {output_path}")

    return results


def display_signals(signals_df: pd.DataFrame, config: dict):
    """Display signals in a formatted table."""
    if signals_df.empty:
        print("\nNo signals generated for today.\n")
        return

    # Format for display
    display_df = signals_df.copy()
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
    display_df['stop_loss'] = display_df['stop_loss'].apply(lambda x: f"${x:.2f}")
    display_df['tp1'] = display_df['tp1'].apply(lambda x: f"${x:.2f}")
    display_df['tp2'] = display_df['tp2'].apply(lambda x: f"${x:.2f}")

    # Print table
    print("\n" + "="*100)
    print("DAILY TRADING SIGNALS - " + datetime.now().strftime('%Y-%m-%d'))
    print("="*100)
    print(tabulate(
        display_df,
        headers='keys',
        tablefmt='grid',
        showindex=False
    ))
    print("="*100 + "\n")

    # Save to CSV if configured
    if config['output']['csv_output']:
        csv_path = config['output']['csv_path']
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        signals_df.to_csv(csv_path, index=False)
        print(f"Signals saved to {csv_path}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Daily Breakout FVG Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate daily signals:
    python main.py --mode signals

  Run backtest:
    python main.py --mode backtest

  Run both:
    python main.py --mode both
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['signals', 'backtest', 'both'],
        default='signals',
        help='Operation mode: signals (daily), backtest, or both'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(config)

    logger.info("="*60)
    logger.info("Daily Breakout FVG Trading System v1.0")
    logger.info("="*60)

    # Execute based on mode
    try:
        if args.mode in ['signals', 'both']:
            signals_df = generate_daily_signals(config, logger)

            if config['output']['console_output']:
                display_signals(signals_df, config)

        if args.mode in ['backtest', 'both']:
            run_backtest(config, logger)

        logger.info("Execution completed successfully")

    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
