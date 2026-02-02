"""
Trading Strategy Checklist Module.
Implements a pre-trade checklist for manual and automated trading validation.

Strategy Rules:
- Pre-Market Analysis required
- Key Level identification
- Internal Breakout confirmation
- Wait for Candle Close
- FVG (Fair Value Gap) entry trigger
- Fibonacci-based take profit levels (.50, .65, .75, .90)
- ARROW indicator with candle close + FVG confirmation
- Exit on Internal Order Block invalidation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class ChecklistStatus(Enum):
    """Status of a checklist item."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChecklistItem:
    """Represents a single checklist item."""
    name: str
    description: str
    status: ChecklistStatus = ChecklistStatus.PENDING
    notes: str = ""
    timestamp: Optional[datetime] = None

    def confirm(self, notes: str = "") -> None:
        """Mark item as confirmed."""
        self.status = ChecklistStatus.CONFIRMED
        self.notes = notes
        self.timestamp = datetime.now()

    def fail(self, reason: str = "") -> None:
        """Mark item as failed."""
        self.status = ChecklistStatus.FAILED
        self.notes = reason
        self.timestamp = datetime.now()

    def skip(self, reason: str = "") -> None:
        """Mark item as skipped."""
        self.status = ChecklistStatus.SKIPPED
        self.notes = reason
        self.timestamp = datetime.now()

    def reset(self) -> None:
        """Reset item to pending."""
        self.status = ChecklistStatus.PENDING
        self.notes = ""
        self.timestamp = None

    def is_complete(self) -> bool:
        """Check if item has been addressed (not pending)."""
        return self.status != ChecklistStatus.PENDING

    def is_passed(self) -> bool:
        """Check if item passed (confirmed or skipped)."""
        return self.status in (ChecklistStatus.CONFIRMED, ChecklistStatus.SKIPPED)


@dataclass
class TakeProfitLevel:
    """Represents a take profit level with exit percentage."""
    name: str
    level: float  # Fibonacci/percentage level (e.g., 0.65)
    exit_pct: float  # Percentage of position to exit (e.g., 30)
    price: Optional[float] = None  # Calculated price
    hit: bool = False
    hit_time: Optional[datetime] = None


@dataclass
class TradeSetup:
    """Represents a complete trade setup with all levels."""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    key_level: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: List[TakeProfitLevel] = field(default_factory=list)
    risk_amount: Optional[float] = None
    position_size: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)

    def calculate_levels(self, swing_high: float, swing_low: float) -> None:
        """
        Calculate entry and take profit levels based on swing range.

        Uses Fibonacci-style levels:
        - Entry: .50 (50% retracement)
        - TP1: .65 (30% exit)
        - TP2: .75 (50% exit)
        - TP3: .90 (remaining 20% - 1:1 R:R)
        """
        range_size = swing_high - swing_low

        if self.direction == 'LONG':
            # For longs: entry near low, targets toward high
            self.entry_price = swing_low + (range_size * 0.50)
            self.stop_loss = swing_low - (range_size * 0.10)  # Below swing low

            self.take_profits = [
                TakeProfitLevel("TP1", 0.65, 30, swing_low + (range_size * 0.65)),
                TakeProfitLevel("TP2", 0.75, 50, swing_low + (range_size * 0.75)),
                TakeProfitLevel("TP3", 0.90, 20, swing_low + (range_size * 0.90)),
            ]
        else:  # SHORT
            # For shorts: entry near high, targets toward low
            self.entry_price = swing_high - (range_size * 0.50)
            self.stop_loss = swing_high + (range_size * 0.10)  # Above swing high

            self.take_profits = [
                TakeProfitLevel("TP1", 0.65, 30, swing_high - (range_size * 0.65)),
                TakeProfitLevel("TP2", 0.75, 50, swing_high - (range_size * 0.75)),
                TakeProfitLevel("TP3", 0.90, 20, swing_high - (range_size * 0.90)),
            ]

        # Calculate risk
        if self.entry_price and self.stop_loss:
            self.risk_amount = abs(self.entry_price - self.stop_loss)


class TradingChecklist:
    """
    Pre-trade checklist for validating trade setups.

    Implements the following checklist items:
    1. Pre-Market Analysis
    2. Key Level Identification
    3. Internal Breakout Indicator
    4. Candle Close Confirmation
    5. FVG (Fair Value Gap) Present
    6. Base Entry at .50 level
    7. Take Profit Levels Set (TP1: .65, TP2: .75, TP3: .90)
    8. ARROW Indicator (optional)
    9. Exit Conditions Defined
    """

    DEFAULT_CHECKLIST = [
        ("pre_market_analysis", "Pre-Market Analysis", "Complete pre-market analysis (levels, news, overnight action)"),
        ("key_level", "Key Level", "Identify key support/resistance level for trade"),
        ("internal_breakout", "Internal Breakout Indicator", "Confirm internal breakout (High > High[1] for longs)"),
        ("candle_close", "Wait for Candle Close", "Wait for candle to close before entry"),
        ("fvg_present", "FVG Present", "Fair Value Gap identified - take the trade!"),
        ("base_entry", "Base Entry: .50", "Entry at 50% retracement level"),
        ("tp1_set", "TP1: .65 (30%)", "First take profit at .65 level - exit 30%"),
        ("tp2_set", "TP2: .75 (50%)", "Second take profit at .75 level - exit 50%"),
        ("tp3_set", "TP3: .90 (1:1)", "Third take profit at .90 level - remaining position"),
        ("arrow_indicator", "ARROW Indicator", "If ARROW indicator present, wait for candle close + FVG"),
        ("exit_defined", "Exit Condition", "EXIT: If Internal Order Block invalidated"),
    ]

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the trading checklist.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.items: Dict[str, ChecklistItem] = {}
        self.setup: Optional[TradeSetup] = None
        self._initialize_checklist()

    def _initialize_checklist(self) -> None:
        """Initialize checklist items from default or config."""
        checklist_config = self.config.get('checklist', {})
        custom_items = checklist_config.get('items', self.DEFAULT_CHECKLIST)

        for item_id, name, description in custom_items:
            self.items[item_id] = ChecklistItem(name=name, description=description)

    def reset(self) -> None:
        """Reset all checklist items to pending."""
        for item in self.items.values():
            item.reset()
        self.setup = None

    def start_new_trade(self, symbol: str, direction: str, key_level: float) -> TradeSetup:
        """
        Start a new trade setup with the checklist.

        Args:
            symbol: Trading symbol
            direction: 'LONG' or 'SHORT'
            key_level: Key support/resistance level

        Returns:
            TradeSetup object
        """
        self.reset()
        self.setup = TradeSetup(
            symbol=symbol,
            direction=direction,
            key_level=key_level
        )
        return self.setup

    def check_item(self, item_id: str, passed: bool = True, notes: str = "") -> bool:
        """
        Check off a checklist item.

        Args:
            item_id: ID of the checklist item
            passed: Whether the check passed
            notes: Optional notes

        Returns:
            True if item was found and updated
        """
        if item_id not in self.items:
            logger.warning(f"Unknown checklist item: {item_id}")
            return False

        item = self.items[item_id]
        if passed:
            item.confirm(notes)
        else:
            item.fail(notes)

        return True

    def skip_item(self, item_id: str, reason: str = "") -> bool:
        """Skip a checklist item (mark as not applicable)."""
        if item_id not in self.items:
            return False
        self.items[item_id].skip(reason)
        return True

    def validate_from_dataframe(self, df: pd.DataFrame, idx) -> Dict[str, bool]:
        """
        Automatically validate checklist items from indicator data.

        Args:
            df: DataFrame with indicator data
            idx: Index of the bar to check

        Returns:
            Dictionary of item_id -> passed status
        """
        if df.empty or idx not in df.index:
            logger.warning("Invalid DataFrame or index for checklist validation")
            return {}

        row = df.loc[idx]
        results = {}

        # Check Internal Breakout
        if self.setup and self.setup.direction == 'LONG':
            if 'bullish_breakout' in row:
                passed = bool(row['bullish_breakout'])
                self.check_item('internal_breakout', passed,
                              "Bullish breakout detected" if passed else "No bullish breakout")
                results['internal_breakout'] = passed
        elif self.setup and self.setup.direction == 'SHORT':
            if 'bearish_breakout' in row:
                passed = bool(row['bearish_breakout'])
                self.check_item('internal_breakout', passed,
                              "Bearish breakout detected" if passed else "No bearish breakout")
                results['internal_breakout'] = passed

        # Check FVG
        if self.setup and self.setup.direction == 'LONG':
            if 'has_recent_bullish_fvg' in row:
                passed = bool(row['has_recent_bullish_fvg'])
                self.check_item('fvg_present', passed,
                              "Bullish FVG present - TAKE IT!" if passed else "No bullish FVG")
                results['fvg_present'] = passed
        elif self.setup and self.setup.direction == 'SHORT':
            if 'has_recent_bearish_fvg' in row:
                passed = bool(row['has_recent_bearish_fvg'])
                self.check_item('fvg_present', passed,
                              "Bearish FVG present - TAKE IT!" if passed else "No bearish FVG")
                results['fvg_present'] = passed

        # Candle close is always confirmed when we have complete bar data
        self.check_item('candle_close', True, "Candle closed")
        results['candle_close'] = True

        return results

    def calculate_trade_levels(self, swing_high: float, swing_low: float) -> None:
        """
        Calculate and set all trade levels (.50, .65, .75, .90).

        Args:
            swing_high: Recent swing high
            swing_low: Recent swing low
        """
        if not self.setup:
            logger.warning("No trade setup initialized")
            return

        self.setup.calculate_levels(swing_high, swing_low)

        # Mark the level items as confirmed
        if self.setup.entry_price:
            self.check_item('base_entry', True,
                          f"Entry at .50 level: ${self.setup.entry_price:.2f}")

        if self.setup.take_profits:
            for i, tp in enumerate(self.setup.take_profits, 1):
                item_id = f"tp{i}_set"
                if item_id in self.items:
                    self.check_item(item_id, True,
                                  f"{tp.name} at ${tp.price:.2f} ({tp.exit_pct}% exit)")

        # Exit condition defined
        self.check_item('exit_defined', True,
                       f"Stop loss at ${self.setup.stop_loss:.2f} / Exit if Int. OB invalidated")

    def is_ready_to_trade(self) -> Tuple[bool, List[str]]:
        """
        Check if all required items are confirmed.

        Returns:
            Tuple of (ready, list of missing items)
        """
        required_items = [
            'key_level',
            'internal_breakout',
            'candle_close',
            'fvg_present',
            'base_entry',
            'exit_defined'
        ]

        missing = []
        for item_id in required_items:
            if item_id in self.items:
                if not self.items[item_id].is_passed():
                    missing.append(self.items[item_id].name)

        return len(missing) == 0, missing

    def get_status(self) -> Dict[str, any]:
        """Get current checklist status."""
        confirmed = sum(1 for item in self.items.values() if item.status == ChecklistStatus.CONFIRMED)
        failed = sum(1 for item in self.items.values() if item.status == ChecklistStatus.FAILED)
        pending = sum(1 for item in self.items.values() if item.status == ChecklistStatus.PENDING)

        return {
            'total': len(self.items),
            'confirmed': confirmed,
            'failed': failed,
            'pending': pending,
            'ready': self.is_ready_to_trade()[0],
            'setup': self.setup
        }

    def print_checklist(self) -> str:
        """Print formatted checklist status."""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("TRADING STRATEGY CHECKLIST")
        lines.append("=" * 60)

        if self.setup:
            lines.append(f"Symbol: {self.setup.symbol} | Direction: {self.setup.direction}")
            lines.append(f"Key Level: ${self.setup.key_level:.2f}")
            lines.append("-" * 60)

        status_symbols = {
            ChecklistStatus.PENDING: "[ ]",
            ChecklistStatus.CONFIRMED: "[X]",
            ChecklistStatus.FAILED: "[!]",
            ChecklistStatus.SKIPPED: "[-]"
        }

        for item_id, item in self.items.items():
            symbol = status_symbols[item.status]
            line = f"{symbol} {item.name}"
            if item.notes:
                line += f" - {item.notes}"
            lines.append(line)

        lines.append("-" * 60)

        ready, missing = self.is_ready_to_trade()
        if ready:
            lines.append("STATUS: READY TO TRADE")
        else:
            lines.append(f"STATUS: NOT READY - Missing: {', '.join(missing)}")

        if self.setup and self.setup.take_profits:
            lines.append("\nTAKE PROFIT LEVELS:")
            for tp in self.setup.take_profits:
                if tp.price:
                    lines.append(f"  {tp.name}: ${tp.price:.2f} (Exit {tp.exit_pct}%)")
            if self.setup.stop_loss:
                lines.append(f"  Stop Loss: ${self.setup.stop_loss:.2f}")

        lines.append("=" * 60 + "\n")

        return "\n".join(lines)


class ChecklistValidator:
    """
    Validates trade setups against the checklist criteria.
    Can be used for both manual and automated trading.
    """

    def __init__(self, config: dict):
        """
        Initialize the validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.checklist_config = config.get('strategy_checklist', {})

        # Take profit configuration
        self.tp_levels = self.checklist_config.get('take_profit_levels', {
            'tp1': {'level': 0.65, 'exit_pct': 30},
            'tp2': {'level': 0.75, 'exit_pct': 50},
            'tp3': {'level': 0.90, 'exit_pct': 20},
        })

        # Entry configuration
        self.entry_level = self.checklist_config.get('entry_level', 0.50)

        # Required checks
        self.require_fvg = self.checklist_config.get('require_fvg', True)
        self.require_internal_breakout = self.checklist_config.get('require_internal_breakout', True)
        self.require_candle_close = self.checklist_config.get('require_candle_close', True)
        self.require_arrow_confirmation = self.checklist_config.get('require_arrow_confirmation', False)

    def validate_entry(self, df: pd.DataFrame, idx, direction: str) -> Tuple[bool, str]:
        """
        Validate if entry conditions are met.

        Args:
            df: DataFrame with indicator data
            idx: Bar index to check
            direction: 'LONG' or 'SHORT'

        Returns:
            Tuple of (valid, reason)
        """
        if df.empty or idx not in df.index:
            return False, "Invalid data"

        row = df.loc[idx]
        reasons = []

        # Check Internal Breakout
        if self.require_internal_breakout:
            if direction == 'LONG':
                if not row.get('bullish_breakout', False):
                    return False, "No bullish internal breakout"
                reasons.append("Internal breakout confirmed")
            else:
                if not row.get('bearish_breakout', False):
                    return False, "No bearish internal breakout"
                reasons.append("Internal breakout confirmed")

        # Check FVG
        if self.require_fvg:
            if direction == 'LONG':
                if not row.get('has_recent_bullish_fvg', False):
                    return False, "No bullish FVG - wait for setup"
                reasons.append("FVG present - TAKE IT!")
            else:
                if not row.get('has_recent_bearish_fvg', False):
                    return False, "No bearish FVG - wait for setup"
                reasons.append("FVG present - TAKE IT!")

        return True, " | ".join(reasons)

    def check_exit_conditions(self, df: pd.DataFrame, idx, trade_setup: TradeSetup) -> Tuple[bool, str]:
        """
        Check if exit conditions are met (Internal OB invalidated).

        Args:
            df: DataFrame with indicator data
            idx: Current bar index
            trade_setup: The active trade setup

        Returns:
            Tuple of (should_exit, reason)
        """
        if df.empty or idx not in df.index:
            return False, ""

        row = df.loc[idx]

        # Check if Internal Order Block is invalidated
        # For longs: if we get a bearish breakout, the setup is invalidated
        # For shorts: if we get a bullish breakout, the setup is invalidated
        if trade_setup.direction == 'LONG':
            if row.get('bearish_breakout', False):
                return True, "EXIT: Internal OB invalidated (bearish breakout)"
        else:
            if row.get('bullish_breakout', False):
                return True, "EXIT: Internal OB invalidated (bullish breakout)"

        # Check stop loss
        if trade_setup.stop_loss:
            if trade_setup.direction == 'LONG' and row['low'] <= trade_setup.stop_loss:
                return True, f"EXIT: Stop loss hit at ${trade_setup.stop_loss:.2f}"
            elif trade_setup.direction == 'SHORT' and row['high'] >= trade_setup.stop_loss:
                return True, f"EXIT: Stop loss hit at ${trade_setup.stop_loss:.2f}"

        return False, ""

    def check_take_profits(self, row, trade_setup: TradeSetup) -> List[TakeProfitLevel]:
        """
        Check if any take profit levels were hit.

        Args:
            row: Current bar data
            trade_setup: The active trade setup

        Returns:
            List of TakeProfitLevel objects that were hit
        """
        hit_levels = []

        for tp in trade_setup.take_profits:
            if tp.hit:
                continue

            if trade_setup.direction == 'LONG':
                if row['high'] >= tp.price:
                    tp.hit = True
                    tp.hit_time = datetime.now()
                    hit_levels.append(tp)
            else:  # SHORT
                if row['low'] <= tp.price:
                    tp.hit = True
                    tp.hit_time = datetime.now()
                    hit_levels.append(tp)

        return hit_levels


def create_checklist_from_config(config: dict) -> TradingChecklist:
    """
    Factory function to create a TradingChecklist from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured TradingChecklist instance
    """
    return TradingChecklist(config)
