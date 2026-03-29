"""
Base strategy class — all strategies inherit from this.
Provides common interface for signal generation, entry, and exit logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    signal: Signal
    symbol: str
    entry_price: float
    stop_loss: float
    target_price: float
    strategy_name: str
    confidence: float  # 0.0 to 1.0
    reason: str


@dataclass
class TradeResult:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_percent: float
    entry_time: str
    exit_time: str
    exit_reason: str  # "stop_loss", "target", "trailing_stop", "square_off"
    strategy_name: str


class BaseStrategy(ABC):
    """All strategies must implement these methods."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.is_active = config.get("enabled", True)

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int,
    ) -> Optional[TradeSignal]:
        """
        Look at data up to current_idx and decide: BUY, SELL, or HOLD.
        Must NOT look ahead (no future data leakage).

        Args:
            df: DataFrame with OHLCV + indicators
            current_idx: Current candle index (only use data[:current_idx+1])

        Returns:
            TradeSignal if entry conditions met, None otherwise
        """
        pass

    @abstractmethod
    def should_exit(
        self,
        df: pd.DataFrame,
        current_idx: int,
        entry_price: float,
        side: str,
    ) -> tuple[bool, str]:
        """
        Check if an open position should be exited (beyond stop loss).
        Returns (should_exit: bool, reason: str).
        """
        pass

    def pre_market_filter(
        self,
        symbol: str,
        gap_percent: float,
        avg_volume: float,
        vix: float,
    ) -> tuple[bool, str]:
        """
        Pre-market check: should this strategy trade this symbol today?
        Default implementation checks basic filters.
        Override in subclass for strategy-specific filters.
        """
        if avg_volume < self.config.get("min_avg_volume", 1_000_000):
            return False, f"Low volume: {avg_volume:,.0f}"

        if vix > self.config.get("vix_skip_threshold", 25):
            return False, f"VIX too high: {vix:.1f}"

        return True, "OK"
