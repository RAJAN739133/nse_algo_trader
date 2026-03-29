"""Pullback Filter — Detects false breakouts on gap-up days.

On gap-up days, many breakouts are traps. This filter checks:
1. Upper wick > body on recent candles (weak buying)
2. Volume declining during breakout (no conviction)
3. Weak break = price barely crossed ORB high

If 2+ signals fire, skip the breakout and wait for pullback re-entry.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PullbackFilter:
    """Detect false breakouts and suggest pullback re-entry."""

    def __init__(self, config=None):
        self.min_gap_pct = 1.5  # only activate on gap-up > 1.5%
        self.wick_threshold = 3  # need 3+ candles with upper wick > body
        self.volume_decline = 0.7  # volume must drop 30%+
        self.weak_break_pct = 0.002  # barely above ORB high

    def is_false_breakout(self, candles_1m, orb_high, gap_pct):
        """
        Check if current breakout is likely false.
        Returns: (is_false: bool, signals: list, pullback_entry: float)
        """
        if gap_pct < self.min_gap_pct:
            return False, [], 0  # Only check on significant gap-ups

        if len(candles_1m) < 15:
            return False, [], 0

        recent = candles_1m.iloc[-5:]
        signals = []

        # Signal 1: Upper wicks dominate (buyers exhausted)
        upper_wick = recent["high"] - recent[["close", "open"]].max(axis=1)
        body = abs(recent["close"] - recent["open"])
        weak_candles = (upper_wick > body).sum()
        if weak_candles >= self.wick_threshold:
            signals.append(f"Upper wicks: {weak_candles}/5 candles show exhaustion")

        # Signal 2: Volume declining during breakout
        if len(candles_1m) > 20:
            pre_break_vol = candles_1m.iloc[-20:-5]["volume"].mean()
            break_vol = recent["volume"].mean()
            if pre_break_vol > 0 and break_vol < pre_break_vol * self.volume_decline:
                signals.append(f"Volume declining: {break_vol/pre_break_vol:.0%} of pre-breakout")

        # Signal 3: Weak break (barely above ORB high)
        latest_price = recent.iloc[-1]["close"]
        break_strength = (latest_price - orb_high) / orb_high
        if 0 < break_strength < self.weak_break_pct:
            signals.append(f"Weak break: only {break_strength:.2%} above ORB high")

        is_false = len(signals) >= 2

        # Calculate pullback re-entry price
        pullback_entry = 0
        if is_false:
            # Wait for price to pull back to ORB high (now support)
            pullback_entry = orb_high * 1.001
            logger.info(f"  FALSE BREAKOUT DETECTED: {'; '.join(signals)}")
            logger.info(f"  Wait for pullback to {pullback_entry:.2f}")

        return is_false, signals, pullback_entry
