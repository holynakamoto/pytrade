# Daily Breakout FVG Trading System

A Python-based algorithmic trading system that generates daily trading signals using bullish/bearish internal breakouts, Fair Value Gaps (FVG), EMAs, and VWAP, with disciplined 1:1 and 1:2 risk-reward management.

## Features

- **Multi-Source Data Fetching**: Yahoo Finance (free), Polygon.io, Alpha Vantage
- **Technical Indicators**: EMA9, EMA200, VWAP, Internal Breakouts, Fair Value Gaps
- **Signal Generation**: Rule-based bullish/bearish entry signals with clear rationale
- **Risk Management**: Dynamic stop-loss, 1:1 and 1:2 take-profit targets
- **Backtesting**: Historical performance analysis with key metrics
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Output Formats**: Console display, CSV export, logging

## Strategy Overview

### Entry Conditions

**Bullish (LONG) Signal**:
1. Close > EMA9 (short-term momentum)
2. Close > VWAP (institutional reference)
3. Close > EMA200 (overall uptrend)
4. High > High[1] (internal breakout)
5. Bullish FVG present in recent bars

**Bearish (SHORT) Signal**:
1. Close < EMA9
2. Close < VWAP
3. Close < EMA200 (overall downtrend)
4. Low < Low[1] (internal breakout)
5. Bearish FVG present in recent bars

### Risk Management

- **Stop Loss**: Below recent swing low (long) / Above recent swing high (short)
- **Take Profit 1**: 1:1 risk-reward (50% position exit)
- **Take Profit 2**: 1:2 risk-reward (remaining 50% exit)
- **Hard Exit**: Close position if price crosses EMA200 against trade direction
- **Position Sizing**: 1% risk per trade (configurable)

## Installation

### Prerequisites

- Python 3.10+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pytrade
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
   - Edit `config.yaml` to customize symbols, risk parameters, and data sources
   - Set API keys via environment variables (optional for Polygon/Alpha Vantage):
     ```bash
     export POLYGON_API_KEY="your_key_here"
     export ALPHAVANTAGE_API_KEY="your_key_here"
     ```

## Usage

### Generate Daily Signals

Run the system to get today's trading signals:

```bash
python main.py --mode signals
```

This will:
- Fetch latest data for configured symbols
- Calculate technical indicators
- Generate signals based on entry conditions
- Display signals in formatted table
- Save signals to CSV (if configured)

### Run Backtest

Test the strategy on historical data:

```bash
python main.py --mode backtest
```

This will:
- Fetch historical data for the configured period
- Simulate trades based on the strategy
- Calculate performance metrics (Sharpe ratio, profit factor, win rate, etc.)
- Display detailed results
- Save trade history to CSV

### Run Both

Generate signals and backtest in one run:

```bash
python main.py --mode both
```

### Custom Configuration

Use a custom config file:

```bash
python main.py --mode signals --config my_config.yaml
```

## Configuration

Key settings in `config.yaml`:

### Symbols
```yaml
symbols:
  - "SPY"
  - "AAPL"
  - "MSFT"
  # Add more symbols...
```

### Indicators
```yaml
indicators:
  ema_short: 9
  ema_long: 200
  fvg_lookback: 5
  swing_lookback: 5
```

### Risk Management
```yaml
risk_management:
  risk_per_trade_pct: 1.0
  account_size: 10000
  tp1_rr: 1.0
  tp2_rr: 2.0
  use_ema200_hard_exit: true
```

### Backtesting
```yaml
backtest:
  start_date: "2020-01-01"
  end_date: "2025-01-17"
```

## Project Structure

```text
pytrade/
├── main.py                 # Main orchestration script
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── data.py            # Data fetching module
│   ├── indicators.py      # Technical indicators
│   ├── signals.py         # Signal generation
│   ├── risk_management.py # Risk/position sizing
│   └── backtest.py        # Backtesting engine
├── output/                # Generated signals and results
└── logs/                  # Application logs
```

## Output

### Daily Signals

Signals are displayed in a formatted table with:
- Symbol
- Direction (LONG/SHORT)
- Entry Price
- Stop Loss
- Take Profit 1 (1:1 RR)
- Take Profit 2 (1:2 RR)
- Date
- Rationale (why the signal was generated)

Example:
```text
Symbol | Direction | Entry   | SL      | TP1     | TP2     | Rationale
SPY    | LONG     | $450.25 | $447.50 | $453.00 | $455.75 | Uptrend | Momentum | FVG
```

### Backtest Results

Performance metrics include:
- Total Return ($ and %)
- Win Rate
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Average Win/Loss
- Trade History

## Performance Targets (MVP)

- **Sharpe Ratio**: > 1.0
- **Profit Factor**: > 1.2
- **Win Rate**: > 45%
- **Max Signals per Day**: 5

## Automation

### Linux/Mac (Cron)

Run daily at 4:30 PM (after market close):

```bash
crontab -e
```

Add:
```text
30 16 * * 1-5 cd /path/to/pytrade && /usr/bin/python3 main.py --mode signals
```

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create new task
3. Set trigger: Daily at 4:30 PM, weekdays only
4. Set action: Run `python main.py --mode signals`
5. Set start in: `C:\path\to\pytrade`

## Limitations & Future Enhancements

### Current Limitations (MVP)
- Daily timeframe only (no intraday)
- No live order execution
- Simple FVG detection (2-3 bar pattern)
- No transaction costs in backtest
- No portfolio-level risk management

### Planned Enhancements
- **v1.1**: ATR-based position sizing
- **v1.2**: Broker integration (Alpaca, IBKR)
- **v2.0**: Multi-timeframe analysis, Telegram alerts
- **v3.0**: Volume profile, order blocks

## Troubleshooting

### No Data Fetched
- Check internet connection
- Verify symbol tickers are valid
- Check API keys if using Polygon/Alpha Vantage
- Review logs in `logs/trading_system.log`

### No Signals Generated
- Ensure sufficient historical data (200+ bars for EMA200)
- Check if entry filters are too restrictive in `config.yaml`
- Verify market conditions align with strategy

### Backtest Errors
- Ensure date range has sufficient data
- Check that symbols have data for the full period
- Increase `cache_days` in config if needed

## Contributing

This is a personal trading tool. For questions or issues:
1. Check the logs first
2. Review the configuration
3. Test with a single symbol to isolate issues

## Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Trading involves risk of loss
- Test thoroughly with paper trading before live use
- The authors assume no liability for trading losses

## License

MIT License - See LICENSE file for details

## Credits

Built by Nick, retail trader in Denver
Version 1.0 (MVP)
Date: 2026-01-17
