"""
Backtesting engine module.
Simulates trading strategy on historical data and calculates performance metrics.
"""

import logging
from typing import List, Dict
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    direction: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    stop_loss: float
    tp1: float
    tp2: float
    position_size: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResults:
    """Backtesting results and performance metrics."""
    trades: List[Trade] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_bars_held: float = 0.0
    initial_capital: float = 0.0
    final_capital: float = 0.0


class Backtester:
    """Backtesting engine for the trading strategy."""

    def __init__(self, config: dict, risk_manager, signal_generator):
        """
        Initialize the backtester.

        Args:
            config: Configuration dictionary
            risk_manager: RiskManager instance
            signal_generator: SignalGenerator instance
        """
        self.config = config
        self.risk_manager = risk_manager
        self.signal_generator = signal_generator

        self.start_date = config['backtest']['start_date']
        self.end_date = config['backtest']['end_date']
        self.initial_capital = config['risk_management']['account_size']

        self.ema_short = config['indicators']['ema_short']
        self.ema_long = config['indicators']['ema_long']

    def run_backtest(
        self,
        data_dict: Dict[str, pd.DataFrame],
        verbose: bool = True
    ) -> BacktestResults:
        """
        Run backtest on multiple symbols.

        Args:
            data_dict: Dictionary mapping symbols to DataFrames with indicators
            verbose: Whether to print progress

        Returns:
            BacktestResults object
        """
        all_trades = []
        equity_curve = [self.initial_capital]
        current_equity = self.initial_capital

        for symbol, df in data_dict.items():
            if df.empty:
                continue

            if verbose:
                logger.info(f"Backtesting {symbol}...")

            # Filter by backtest date range
            df_backtest = df[(df.index >= self.start_date) & (df.index <= self.end_date)].copy()

            if df_backtest.empty:
                logger.warning(f"No data for {symbol} in backtest period")
                continue

            # Run backtest on this symbol
            symbol_trades = self._backtest_symbol(symbol, df_backtest, current_equity)
            all_trades.extend(symbol_trades)

            # Update equity
            for trade in symbol_trades:
                current_equity += trade.pnl
                equity_curve.append(current_equity)

        # Calculate performance metrics
        results = self._calculate_metrics(all_trades, equity_curve)
        results.initial_capital = self.initial_capital
        results.final_capital = current_equity

        return results

    def _backtest_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_equity: float
    ) -> List[Trade]:
        """
        Backtest a single symbol.

        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV and indicators
            current_equity: Current account equity

        Returns:
            List of Trade objects
        """
        trades = []
        in_position = False
        position = None

        # Iterate through bars
        for i in range(len(df)):
            current_idx = df.index[i]
            current_bar = df.iloc[i]

            # Skip if we don't have enough data for indicators
            if pd.isna(current_bar.get(f'ema{self.ema_long}')):
                continue

            # If not in position, check for entry signals
            if not in_position:
                # Check for bullish signal
                if self._check_bullish_conditions(df, i):
                    position = self._enter_long_position(
                        symbol, current_bar, current_idx, current_equity
                    )
                    if position:
                        in_position = True

                # Check for bearish signal
                elif self._check_bearish_conditions(df, i):
                    position = self._enter_short_position(
                        symbol, current_bar, current_idx, current_equity
                    )
                    if position:
                        in_position = True

            # If in position, check for exit conditions
            elif in_position:
                # Check for exit
                exit_trade = self._check_exit_conditions(
                    position, current_bar, current_idx, df, i
                )

                if exit_trade:
                    trades.append(exit_trade)
                    in_position = False
                    position = None
                    current_equity += exit_trade.pnl

        # Close any open position at the end
        if in_position and position:
            final_bar = df.iloc[-1]
            final_idx = df.index[-1]
            exit_trade = self._force_exit_position(position, final_bar, final_idx)
            trades.append(exit_trade)

        return trades

    def _check_bullish_conditions(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if bullish entry conditions are met at index."""
        if idx >= len(df):
            return False
        return self.signal_generator._check_bullish_conditions(df, df.index[idx])

    def _check_bearish_conditions(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if bearish entry conditions are met at index."""
        if idx >= len(df):
            return False
        return self.signal_generator._check_bearish_conditions(df, df.index[idx])

    def _enter_long_position(self, symbol: str, bar, date, equity: float) -> Dict:
        """Enter a long position."""
        entry_price = bar['close']
        stop_loss = bar['swing_low'] if pd.notna(bar.get('swing_low')) else bar['low']
        tp1, tp2 = self.risk_manager.calculate_take_profits(entry_price, stop_loss, 'LONG')

        position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, equity)

        if position_size == 0:
            return None

        return {
            'symbol': symbol,
            'direction': 'LONG',
            'entry_date': str(date),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'position_size': position_size,
            'partial_exit': False
        }

    def _enter_short_position(self, symbol: str, bar, date, equity: float) -> Dict:
        """Enter a short position."""
        entry_price = bar['close']
        stop_loss = bar['swing_high'] if pd.notna(bar.get('swing_high')) else bar['high']
        tp1, tp2 = self.risk_manager.calculate_take_profits(entry_price, stop_loss, 'SHORT')

        position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, equity)

        if position_size == 0:
            return None

        return {
            'symbol': symbol,
            'direction': 'SHORT',
            'entry_date': str(date),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'position_size': position_size,
            'partial_exit': False
        }

    def _check_exit_conditions(
        self,
        position: Dict,
        current_bar,
        current_date,
        df: pd.DataFrame,
        idx: int
    ) -> Trade:
        """Check if any exit conditions are met."""
        direction = position['direction']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        tp1 = position['tp1']
        tp2 = position['tp2']
        partial_exit = position['partial_exit']

        high = current_bar['high']
        low = current_bar['low']
        close = current_bar['close']

        exit_price = None
        exit_reason = None

        if direction == 'LONG':
            # Check stop loss
            if low <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'Stop Loss'

            # Check TP2
            elif high >= tp2:
                exit_price = tp2
                exit_reason = 'TP2 (1:2 RR)'

            # Check TP1 (in real trading would be partial exit)
            elif high >= tp1 and not partial_exit:
                # For simplicity, we'll exit at TP1 as well
                # In production, you'd reduce position size here
                exit_price = tp1
                exit_reason = 'TP1 (1:1 RR)'

            # Check EMA200 exit
            elif self.risk_manager.use_ema200_exit:
                ema200 = current_bar.get(f'ema{self.ema_long}')
                if pd.notna(ema200) and close < ema200:
                    exit_price = close
                    exit_reason = 'EMA200 Cross'

        else:  # SHORT
            # Check stop loss
            if high >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'Stop Loss'

            # Check TP2
            elif low <= tp2:
                exit_price = tp2
                exit_reason = 'TP2 (1:2 RR)'

            # Check TP1
            elif low <= tp1 and not partial_exit:
                exit_price = tp1
                exit_reason = 'TP1 (1:1 RR)'

            # Check EMA200 exit
            elif self.risk_manager.use_ema200_exit:
                ema200 = current_bar.get(f'ema{self.ema_long}')
                if pd.notna(ema200) and close > ema200:
                    exit_price = close
                    exit_reason = 'EMA200 Cross'

        # If exit triggered, create trade
        if exit_price is not None:
            return self._create_trade_record(
                position, exit_price, current_date, exit_reason, idx
            )

        return None

    def _force_exit_position(self, position: Dict, final_bar, final_date) -> Trade:
        """Force exit position at end of backtest."""
        exit_price = final_bar['close']
        return self._create_trade_record(
            position, exit_price, final_date, 'End of Backtest', 0
        )

    def _create_trade_record(
        self,
        position: Dict,
        exit_price: float,
        exit_date,
        exit_reason: str,
        bars_held: int
    ) -> Trade:
        """Create a Trade record."""
        entry_price = position['entry_price']
        position_size = position['position_size']
        direction = position['direction']

        pnl = self.risk_manager.calculate_pnl(
            entry_price, exit_price, position_size, direction
        )

        pnl_pct = (pnl / (entry_price * position_size)) * 100

        return Trade(
            symbol=position['symbol'],
            direction=direction,
            entry_date=position['entry_date'],
            entry_price=entry_price,
            exit_date=str(exit_date),
            exit_price=exit_price,
            stop_loss=position['stop_loss'],
            tp1=position['tp1'],
            tp2=position['tp2'],
            position_size=position_size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            bars_held=bars_held
        )

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float]
    ) -> BacktestResults:
        """Calculate performance metrics from trades."""
        results = BacktestResults()
        results.trades = trades

        if not trades:
            logger.warning("No trades to analyze")
            return results

        results.total_trades = len(trades)

        # Win/Loss stats
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        results.winning_trades = len(wins)
        results.losing_trades = len(losses)
        results.win_rate = results.winning_trades / results.total_trades if results.total_trades > 0 else 0

        # PnL stats
        results.total_pnl = sum(t.pnl for t in trades)
        results.total_return_pct = (results.total_pnl / self.initial_capital) * 100

        if wins:
            results.avg_win = sum(t.pnl for t in wins) / len(wins)
            results.largest_win = max(t.pnl for t in wins)

        if losses:
            results.avg_loss = sum(t.pnl for t in losses) / len(losses)
            results.largest_loss = min(t.pnl for t in losses)

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            results.sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Max drawdown
        if len(equity_curve) > 1:
            peak = equity_curve[0]
            max_dd = 0
            max_dd_pct = 0

            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                dd_pct = (dd / peak) * 100 if peak > 0 else 0

                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct

            results.max_drawdown = max_dd
            results.max_drawdown_pct = max_dd_pct

        # Average bars held
        results.avg_bars_held = sum(t.bars_held for t in trades) / len(trades) if trades else 0

        return results

    def print_results(self, results: BacktestResults):
        """Print backtest results in a formatted way."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${results.initial_capital:,.2f}")
        print(f"Final Capital: ${results.final_capital:,.2f}")
        print(f"Total Return: ${results.total_pnl:,.2f} ({results.total_return_pct:.2f}%)")
        print()
        print(f"Total Trades: {results.total_trades}")
        print(f"Winning Trades: {results.winning_trades}")
        print(f"Losing Trades: {results.losing_trades}")
        print(f"Win Rate: {results.win_rate*100:.2f}%")
        print()
        print(f"Average Win: ${results.avg_win:,.2f}")
        print(f"Average Loss: ${results.avg_loss:,.2f}")
        print(f"Largest Win: ${results.largest_win:,.2f}")
        print(f"Largest Loss: ${results.largest_loss:,.2f}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print()
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.2f}%)")
        print(f"Avg Bars Held: {results.avg_bars_held:.1f}")
        print("="*60)

        # Performance vs targets
        print("\nPERFORMANCE VS TARGETS")
        print("-"*60)
        targets = self.config['targets']
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f} (Target: {targets['min_sharpe_ratio']})")
        print(f"Profit Factor: {results.profit_factor:.2f} (Target: {targets['min_profit_factor']})")
        print(f"Win Rate: {results.win_rate:.2%} (Target: {targets['min_win_rate']:.0%})")
        print("="*60 + "\n")
