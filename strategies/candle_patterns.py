"""
Candlestick Pattern Detection for Intraday Trading
═══════════════════════════════════════════════════════════════

Japanese candlestick patterns for entry/exit confirmation.
All patterns return a score from -100 (bearish) to +100 (bullish).

Patterns Implemented:
- Single: Doji, Hammer, Shooting Star, Marubozu
- Double: Engulfing, Harami, Tweezer
- Triple: Morning Star, Evening Star, Three White Soldiers, Three Black Crows

Usage:
    detector = CandlePatternDetector()
    score = detector.analyze(candles, idx)  # Returns -100 to +100
    patterns = detector.detect_all(candles, idx)  # Returns list of patterns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PatternResult:
    """Result of pattern detection."""
    name: str
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: int   # 1-3 (weak, medium, strong)
    score: int      # -100 to +100
    description: str


class CandlePatternDetector:
    """
    Detects Japanese candlestick patterns for trading signals.
    """
    
    def __init__(self, body_threshold: float = 0.3, shadow_threshold: float = 2.0):
        """
        Args:
            body_threshold: Min body/range ratio for "real body" (default 30%)
            shadow_threshold: Shadow must be X times body for hammer/star patterns
        """
        self.body_threshold = body_threshold
        self.shadow_threshold = shadow_threshold
    
    def _get_candle_parts(self, candle: pd.Series) -> Dict:
        """Extract candle components."""
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        
        body = abs(c - o)
        range_ = h - l
        
        if c >= o:  # Green/bullish candle
            upper_shadow = h - c
            lower_shadow = o - l
            color = "GREEN"
        else:  # Red/bearish candle
            upper_shadow = h - o
            lower_shadow = c - l
            color = "RED"
        
        body_pct = body / range_ if range_ > 0 else 0
        
        return {
            "open": o, "high": h, "low": l, "close": c,
            "body": body, "range": range_,
            "upper_shadow": upper_shadow, "lower_shadow": lower_shadow,
            "body_pct": body_pct, "color": color,
            "mid": (o + c) / 2,
        }
    
    # ═══════════════════════════════════════════════════════════
    # SINGLE CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════
    
    def _is_doji(self, parts: Dict) -> Optional[PatternResult]:
        """
        Doji: Very small body, indicates indecision.
        - Body < 10% of range
        - Signal depends on context (trend exhaustion)
        """
        if parts["body_pct"] < 0.1 and parts["range"] > 0:
            return PatternResult(
                name="Doji",
                direction="NEUTRAL",
                strength=2,
                score=0,
                description="Indecision candle - potential reversal"
            )
        return None
    
    def _is_hammer(self, parts: Dict) -> Optional[PatternResult]:
        """
        Hammer: Small body at top, long lower shadow.
        - Lower shadow >= 2x body
        - Upper shadow very small
        - Bullish reversal at bottom of downtrend
        """
        if parts["range"] == 0:
            return None
        
        if (parts["lower_shadow"] >= parts["body"] * self.shadow_threshold and
            parts["upper_shadow"] < parts["body"] * 0.5 and
            parts["body_pct"] < 0.4):
            return PatternResult(
                name="Hammer",
                direction="BULLISH",
                strength=2,
                score=60,
                description="Bullish reversal - buyers rejected lower prices"
            )
        return None
    
    def _is_inverted_hammer(self, parts: Dict) -> Optional[PatternResult]:
        """
        Inverted Hammer: Small body at bottom, long upper shadow.
        - Upper shadow >= 2x body
        - Lower shadow very small
        - Bullish at bottom of downtrend
        """
        if parts["range"] == 0:
            return None
        
        if (parts["upper_shadow"] >= parts["body"] * self.shadow_threshold and
            parts["lower_shadow"] < parts["body"] * 0.5 and
            parts["body_pct"] < 0.4):
            return PatternResult(
                name="Inverted Hammer",
                direction="BULLISH",
                strength=1,
                score=40,
                description="Potential bullish reversal - needs confirmation"
            )
        return None
    
    def _is_shooting_star(self, parts: Dict, prev_parts: Optional[Dict]) -> Optional[PatternResult]:
        """
        Shooting Star: Like inverted hammer but at TOP of uptrend.
        - Upper shadow >= 2x body
        - Previous candle should be bullish
        """
        if parts["range"] == 0:
            return None
        
        if (parts["upper_shadow"] >= parts["body"] * self.shadow_threshold and
            parts["lower_shadow"] < parts["body"] * 0.5 and
            parts["body_pct"] < 0.4):
            
            # Check if at top of uptrend
            if prev_parts and prev_parts["color"] == "GREEN":
                return PatternResult(
                    name="Shooting Star",
                    direction="BEARISH",
                    strength=2,
                    score=-60,
                    description="Bearish reversal at top - sellers rejected higher prices"
                )
        return None
    
    def _is_marubozu(self, parts: Dict) -> Optional[PatternResult]:
        """
        Marubozu: Strong body with minimal shadows.
        - Body > 80% of range
        - Strong momentum candle
        """
        if parts["body_pct"] > 0.8:
            direction = "BULLISH" if parts["color"] == "GREEN" else "BEARISH"
            score = 70 if direction == "BULLISH" else -70
            return PatternResult(
                name="Marubozu",
                direction=direction,
                strength=3,
                score=score,
                description=f"Strong {direction.lower()} momentum - no shadows"
            )
        return None
    
    # ═══════════════════════════════════════════════════════════
    # DOUBLE CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════
    
    def _is_bullish_engulfing(self, curr: Dict, prev: Dict) -> Optional[PatternResult]:
        """
        Bullish Engulfing: Green candle completely engulfs prior red candle.
        - Current body engulfs previous body
        - Strong reversal signal
        """
        if prev["color"] == "RED" and curr["color"] == "GREEN":
            # Current open below prev close, current close above prev open
            if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
                if curr["body"] > prev["body"]:
                    return PatternResult(
                        name="Bullish Engulfing",
                        direction="BULLISH",
                        strength=3,
                        score=80,
                        description="Strong bullish reversal - buyers took control"
                    )
        return None
    
    def _is_bearish_engulfing(self, curr: Dict, prev: Dict) -> Optional[PatternResult]:
        """
        Bearish Engulfing: Red candle completely engulfs prior green candle.
        """
        if prev["color"] == "GREEN" and curr["color"] == "RED":
            # Current open above prev close, current close below prev open
            if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
                if curr["body"] > prev["body"]:
                    return PatternResult(
                        name="Bearish Engulfing",
                        direction="BEARISH",
                        strength=3,
                        score=-80,
                        description="Strong bearish reversal - sellers took control"
                    )
        return None
    
    def _is_bullish_harami(self, curr: Dict, prev: Dict) -> Optional[PatternResult]:
        """
        Bullish Harami: Small green inside large red (pregnant pattern).
        - Current candle inside previous candle's body
        - Trend reversal signal
        """
        if prev["color"] == "RED" and prev["body_pct"] > 0.5:
            # Current body inside previous body
            curr_high = max(curr["open"], curr["close"])
            curr_low = min(curr["open"], curr["close"])
            prev_high = prev["open"]  # Red candle opens high
            prev_low = prev["close"]  # Red candle closes low
            
            if curr_high <= prev_high and curr_low >= prev_low:
                if curr["body"] < prev["body"] * 0.5:
                    return PatternResult(
                        name="Bullish Harami",
                        direction="BULLISH",
                        strength=2,
                        score=50,
                        description="Potential bullish reversal - momentum slowing"
                    )
        return None
    
    def _is_bearish_harami(self, curr: Dict, prev: Dict) -> Optional[PatternResult]:
        """
        Bearish Harami: Small red inside large green.
        """
        if prev["color"] == "GREEN" and prev["body_pct"] > 0.5:
            curr_high = max(curr["open"], curr["close"])
            curr_low = min(curr["open"], curr["close"])
            prev_high = prev["close"]  # Green candle closes high
            prev_low = prev["open"]    # Green candle opens low
            
            if curr_high <= prev_high and curr_low >= prev_low:
                if curr["body"] < prev["body"] * 0.5:
                    return PatternResult(
                        name="Bearish Harami",
                        direction="BEARISH",
                        strength=2,
                        score=-50,
                        description="Potential bearish reversal - momentum slowing"
                    )
        return None
    
    def _is_tweezer_top(self, curr: Dict, prev: Dict) -> Optional[PatternResult]:
        """
        Tweezer Top: Two candles with same highs at top of uptrend.
        """
        if abs(curr["high"] - prev["high"]) / prev["high"] < 0.001:
            if prev["color"] == "GREEN" and curr["color"] == "RED":
                return PatternResult(
                    name="Tweezer Top",
                    direction="BEARISH",
                    strength=2,
                    score=-55,
                    description="Double top pattern - resistance confirmed"
                )
        return None
    
    def _is_tweezer_bottom(self, curr: Dict, prev: Dict) -> Optional[PatternResult]:
        """
        Tweezer Bottom: Two candles with same lows at bottom of downtrend.
        """
        if abs(curr["low"] - prev["low"]) / prev["low"] < 0.001:
            if prev["color"] == "RED" and curr["color"] == "GREEN":
                return PatternResult(
                    name="Tweezer Bottom",
                    direction="BULLISH",
                    strength=2,
                    score=55,
                    description="Double bottom pattern - support confirmed"
                )
        return None
    
    # ═══════════════════════════════════════════════════════════
    # TRIPLE CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════
    
    def _is_morning_star(self, candles: List[Dict]) -> Optional[PatternResult]:
        """
        Morning Star: Large red, small doji/body, large green.
        - Strong bullish reversal pattern
        """
        if len(candles) < 3:
            return None
        
        first, second, third = candles[-3], candles[-2], candles[-1]
        
        # First: Large red candle
        if first["color"] != "RED" or first["body_pct"] < 0.5:
            return None
        
        # Second: Small body (doji or small real body)
        if second["body_pct"] > 0.3:
            return None
        
        # Third: Large green candle that closes above midpoint of first
        if third["color"] != "GREEN" or third["body_pct"] < 0.5:
            return None
        
        if third["close"] > first["mid"]:
            return PatternResult(
                name="Morning Star",
                direction="BULLISH",
                strength=3,
                score=85,
                description="Strong bullish reversal - trend change likely"
            )
        return None
    
    def _is_evening_star(self, candles: List[Dict]) -> Optional[PatternResult]:
        """
        Evening Star: Large green, small doji/body, large red.
        - Strong bearish reversal pattern
        """
        if len(candles) < 3:
            return None
        
        first, second, third = candles[-3], candles[-2], candles[-1]
        
        # First: Large green candle
        if first["color"] != "GREEN" or first["body_pct"] < 0.5:
            return None
        
        # Second: Small body
        if second["body_pct"] > 0.3:
            return None
        
        # Third: Large red candle that closes below midpoint of first
        if third["color"] != "RED" or third["body_pct"] < 0.5:
            return None
        
        if third["close"] < first["mid"]:
            return PatternResult(
                name="Evening Star",
                direction="BEARISH",
                strength=3,
                score=-85,
                description="Strong bearish reversal - trend change likely"
            )
        return None
    
    def _is_three_white_soldiers(self, candles: List[Dict]) -> Optional[PatternResult]:
        """
        Three White Soldiers: Three consecutive green candles with higher closes.
        - Strong bullish continuation/reversal
        """
        if len(candles) < 3:
            return None
        
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # All green with decent bodies
        if not all(c["color"] == "GREEN" and c["body_pct"] > 0.4 for c in [c1, c2, c3]):
            return None
        
        # Each closes higher than previous
        if c2["close"] > c1["close"] and c3["close"] > c2["close"]:
            # Each opens within body of previous
            if c2["open"] > c1["open"] and c3["open"] > c2["open"]:
                return PatternResult(
                    name="Three White Soldiers",
                    direction="BULLISH",
                    strength=3,
                    score=90,
                    description="Very strong bullish signal - sustained buying"
                )
        return None
    
    def _is_three_black_crows(self, candles: List[Dict]) -> Optional[PatternResult]:
        """
        Three Black Crows: Three consecutive red candles with lower closes.
        - Strong bearish continuation/reversal
        """
        if len(candles) < 3:
            return None
        
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # All red with decent bodies
        if not all(c["color"] == "RED" and c["body_pct"] > 0.4 for c in [c1, c2, c3]):
            return None
        
        # Each closes lower than previous
        if c2["close"] < c1["close"] and c3["close"] < c2["close"]:
            return PatternResult(
                name="Three Black Crows",
                direction="BEARISH",
                strength=3,
                score=-90,
                description="Very strong bearish signal - sustained selling"
            )
        return None
    
    # ═══════════════════════════════════════════════════════════
    # PUBLIC INTERFACE
    # ═══════════════════════════════════════════════════════════
    
    def detect_all(self, candles: pd.DataFrame, idx: int) -> List[PatternResult]:
        """
        Detect all candlestick patterns at given index.
        
        Args:
            candles: DataFrame with OHLC columns
            idx: Current candle index
        
        Returns:
            List of detected patterns
        """
        if idx < 0 or idx >= len(candles):
            return []
        
        patterns = []
        
        curr = self._get_candle_parts(candles.iloc[idx])
        prev = self._get_candle_parts(candles.iloc[idx - 1]) if idx >= 1 else None
        
        # Single candle patterns
        if p := self._is_doji(curr):
            patterns.append(p)
        if p := self._is_hammer(curr):
            patterns.append(p)
        if p := self._is_inverted_hammer(curr):
            patterns.append(p)
        if prev and (p := self._is_shooting_star(curr, prev)):
            patterns.append(p)
        if p := self._is_marubozu(curr):
            patterns.append(p)
        
        # Double candle patterns
        if prev:
            if p := self._is_bullish_engulfing(curr, prev):
                patterns.append(p)
            if p := self._is_bearish_engulfing(curr, prev):
                patterns.append(p)
            if p := self._is_bullish_harami(curr, prev):
                patterns.append(p)
            if p := self._is_bearish_harami(curr, prev):
                patterns.append(p)
            if p := self._is_tweezer_top(curr, prev):
                patterns.append(p)
            if p := self._is_tweezer_bottom(curr, prev):
                patterns.append(p)
        
        # Triple candle patterns
        if idx >= 2:
            parts_list = [
                self._get_candle_parts(candles.iloc[idx - 2]),
                self._get_candle_parts(candles.iloc[idx - 1]),
                curr
            ]
            if p := self._is_morning_star(parts_list):
                patterns.append(p)
            if p := self._is_evening_star(parts_list):
                patterns.append(p)
            if p := self._is_three_white_soldiers(parts_list):
                patterns.append(p)
            if p := self._is_three_black_crows(parts_list):
                patterns.append(p)
        
        return patterns
    
    def analyze(self, candles: pd.DataFrame, idx: int) -> int:
        """
        Get composite pattern score at given index.
        
        Returns:
            Score from -100 (very bearish) to +100 (very bullish)
        """
        patterns = self.detect_all(candles, idx)
        
        if not patterns:
            return 0
        
        # Weight by strength and take average
        total_score = 0
        total_weight = 0
        
        for p in patterns:
            weight = p.strength
            total_score += p.score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return int(total_score / total_weight)
    
    def get_signal(self, candles: pd.DataFrame, idx: int, threshold: int = 40) -> Optional[str]:
        """
        Get trading signal based on patterns.
        
        Args:
            candles: OHLC DataFrame
            idx: Current index
            threshold: Minimum score for signal (default 40)
        
        Returns:
            "LONG", "SHORT", or None
        """
        score = self.analyze(candles, idx)
        
        if score >= threshold:
            return "LONG"
        elif score <= -threshold:
            return "SHORT"
        return None
    
    def confirm_direction(self, candles: pd.DataFrame, idx: int, direction: str) -> Tuple[bool, int]:
        """
        Check if patterns confirm a given direction.
        
        Returns:
            (is_confirmed, confidence_boost)
        """
        score = self.analyze(candles, idx)
        
        if direction == "LONG":
            if score > 20:
                return True, min(30, score // 2)
            elif score < -40:
                return False, 0  # Contradicting
        elif direction == "SHORT":
            if score < -20:
                return True, min(30, abs(score) // 2)
            elif score > 40:
                return False, 0  # Contradicting
        
        return True, 0  # Neutral - doesn't contradict
