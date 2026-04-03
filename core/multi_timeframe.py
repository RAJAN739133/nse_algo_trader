"""
Multi-Timeframe Analysis - Confluence across multiple timeframes.

Features:
- 1-minute, 5-minute, 15-minute, hourly candle aggregation
- Timeframe alignment detection
- Higher timeframe context for entries
- Multi-timeframe indicator calculation
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"


class TrendDirection(Enum):
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class TimeframeAnalysis:
    """Analysis for a single timeframe."""
    timeframe: Timeframe
    trend: TrendDirection
    trend_strength: float  # 0-100
    
    # Key levels
    support: float
    resistance: float
    pivot: float
    
    # Indicators
    ema_fast: float
    ema_slow: float
    rsi: float
    macd: float
    macd_signal: float
    atr: float
    
    # Signals
    ema_bullish: bool
    rsi_oversold: bool
    rsi_overbought: bool
    macd_bullish: bool
    
    # Price position
    above_ema_fast: bool
    above_ema_slow: bool
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultiTimeframeSignal:
    """Combined signal across multiple timeframes."""
    symbol: str
    
    # Alignment
    timeframes_aligned: int  # Number of timeframes agreeing
    total_timeframes: int
    alignment_score: float  # 0-100
    
    # Direction
    consensus_direction: TrendDirection
    primary_tf_trend: TrendDirection    # Your trading timeframe
    higher_tf_trend: TrendDirection     # Higher timeframe context
    
    # Individual analyses
    analyses: Dict[Timeframe, TimeframeAnalysis]
    
    # Entry signal
    signal_strength: str  # "strong", "moderate", "weak", "none"
    entry_bias: str       # "BUY", "SELL", "HOLD"
    
    # Key levels from higher timeframes
    htf_support: float
    htf_resistance: float
    
    timestamp: datetime = field(default_factory=datetime.now)


class CandleAggregator:
    """
    Aggregates lower timeframe candles into higher timeframes.
    Maintains candle buffers for multiple timeframes.
    """
    
    def __init__(
        self,
        base_timeframe: Timeframe = Timeframe.M1,
        target_timeframes: List[Timeframe] = None
    ):
        self.base_timeframe = base_timeframe
        self.target_timeframes = target_timeframes or [
            Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1
        ]
        
        # Conversion ratios (from 1-minute candles)
        self._tf_minutes = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.M30: 30,
            Timeframe.H1: 60,
            Timeframe.D1: 375  # NSE trading hours
        }
        
        # Candle buffers
        self._base_buffer: deque = deque(maxlen=1000)  # 1-min candles
        self._candle_buffers: Dict[Timeframe, deque] = {
            tf: deque(maxlen=200) for tf in self.target_timeframes
        }
        
        # Current forming candles
        self._forming_candles: Dict[Timeframe, Dict] = {}
    
    def add_candle(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int
    ):
        """
        Add a base timeframe candle and update all higher timeframes.
        """
        candle = {
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }
        
        self._base_buffer.append(candle)
        
        # Update each higher timeframe
        for tf in self.target_timeframes:
            self._update_timeframe(tf, candle)
    
    def _update_timeframe(self, tf: Timeframe, base_candle: Dict):
        """Update a specific timeframe with new base candle."""
        tf_minutes = self._tf_minutes[tf]
        base_minutes = self._tf_minutes[self.base_timeframe]
        candles_needed = tf_minutes // base_minutes
        
        timestamp = base_candle["timestamp"]
        
        # Determine which period this belongs to
        period_start = self._get_period_start(timestamp, tf_minutes)
        
        if tf not in self._forming_candles or self._forming_candles[tf].get("period") != period_start:
            # New period - save old candle if exists
            if tf in self._forming_candles and self._forming_candles[tf].get("count", 0) > 0:
                completed = self._forming_candles[tf].copy()
                completed.pop("period", None)
                completed.pop("count", None)
                self._candle_buffers[tf].append(completed)
            
            # Start new candle
            self._forming_candles[tf] = {
                "period": period_start,
                "timestamp": period_start,
                "open": base_candle["open"],
                "high": base_candle["high"],
                "low": base_candle["low"],
                "close": base_candle["close"],
                "volume": base_candle["volume"],
                "count": 1
            }
        else:
            # Update existing candle
            forming = self._forming_candles[tf]
            forming["high"] = max(forming["high"], base_candle["high"])
            forming["low"] = min(forming["low"], base_candle["low"])
            forming["close"] = base_candle["close"]
            forming["volume"] += base_candle["volume"]
            forming["count"] += 1
    
    def _get_period_start(self, timestamp: datetime, period_minutes: int) -> datetime:
        """Get the start of the period containing this timestamp."""
        # Align to period boundaries
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        period_start_minutes = (minutes_since_midnight // period_minutes) * period_minutes
        
        return timestamp.replace(
            hour=period_start_minutes // 60,
            minute=period_start_minutes % 60,
            second=0,
            microsecond=0
        )
    
    def get_candles(self, tf: Timeframe, n: int = 50) -> pd.DataFrame:
        """
        Get the last n candles for a timeframe.
        Includes the currently forming candle.
        """
        candles = list(self._candle_buffers[tf])[-n:]
        
        # Add forming candle
        if tf in self._forming_candles and self._forming_candles[tf].get("count", 0) > 0:
            forming = self._forming_candles[tf].copy()
            forming.pop("period", None)
            forming.pop("count", None)
            candles.append(forming)
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")
    
    def get_all_timeframes(self, n: int = 50) -> Dict[Timeframe, pd.DataFrame]:
        """Get candles for all timeframes."""
        return {tf: self.get_candles(tf, n) for tf in self.target_timeframes}


class MultiTimeframeAnalyzer:
    """
    Analyzes price action across multiple timeframes for confluence.
    """
    
    def __init__(
        self,
        primary_tf: Timeframe = Timeframe.M5,
        context_tfs: List[Timeframe] = None,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        atr_period: int = 14
    ):
        self.primary_tf = primary_tf
        self.context_tfs = context_tfs or [Timeframe.M15, Timeframe.H1]
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        
        self.aggregator = CandleAggregator(
            base_timeframe=Timeframe.M1,
            target_timeframes=[primary_tf] + context_tfs
        )
    
    def add_candle(self, *args, **kwargs):
        """Pass-through to aggregator."""
        self.aggregator.add_candle(*args, **kwargs)
    
    def analyze_timeframe(self, tf: Timeframe, df: pd.DataFrame) -> Optional[TimeframeAnalysis]:
        """Analyze a single timeframe."""
        if df is None or len(df) < 50:
            return None
        
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # EMAs
        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)
        
        # RSI
        rsi = self._rsi(close, self.rsi_period)
        
        # MACD
        macd, signal = self._macd(close)
        
        # ATR
        atr = self._atr(high, low, close, self.atr_period)
        
        # Support/Resistance (simple pivot points)
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        pivot = (recent_high + recent_low + close[-1]) / 3
        
        # Trend detection
        trend, strength = self._detect_trend(close, ema_fast, ema_slow)
        
        return TimeframeAnalysis(
            timeframe=tf,
            trend=trend,
            trend_strength=strength,
            support=recent_low,
            resistance=recent_high,
            pivot=pivot,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi=rsi,
            macd=macd,
            macd_signal=signal,
            atr=atr,
            ema_bullish=ema_fast > ema_slow,
            rsi_oversold=rsi < 30,
            rsi_overbought=rsi > 70,
            macd_bullish=macd > signal,
            above_ema_fast=close[-1] > ema_fast,
            above_ema_slow=close[-1] > ema_slow
        )
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0
        
        multiplier = 2 / (period + 1)
        ema = data[:period].mean()
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _rsi(self, data: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(data) < period + 1:
            return 50  # Neutral
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _macd(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD and signal line."""
        if len(data) < 26:
            return 0, 0
        
        ema12 = self._ema(data, 12)
        ema26 = self._ema(data, 26)
        macd = ema12 - ema26
        
        # Simplified signal (would need full MACD line for proper signal)
        signal = macd * 0.9  # Approximate
        
        return macd, signal
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate ATR."""
        if len(high) < period + 1:
            return (high[-1] - low[-1]) if len(high) > 0 else 0
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        return np.mean(tr[-period:])
    
    def _detect_trend(
        self,
        close: np.ndarray,
        ema_fast: float,
        ema_slow: float
    ) -> Tuple[TrendDirection, float]:
        """Detect trend direction and strength."""
        current = close[-1]
        
        # EMA relationship
        ema_diff = (ema_fast - ema_slow) / ema_slow * 100 if ema_slow > 0 else 0
        
        # Price position
        above_fast = current > ema_fast
        above_slow = current > ema_slow
        
        # Recent momentum (last 10 candles)
        momentum = (current - close[-10]) / close[-10] * 100 if len(close) >= 10 else 0
        
        # Determine trend
        if ema_diff > 1 and above_fast and above_slow and momentum > 1:
            trend = TrendDirection.STRONG_UP
            strength = min(100, abs(ema_diff) * 20 + abs(momentum) * 10)
        elif ema_diff > 0.3 and above_slow:
            trend = TrendDirection.UP
            strength = min(80, abs(ema_diff) * 15 + abs(momentum) * 8)
        elif ema_diff < -1 and not above_fast and not above_slow and momentum < -1:
            trend = TrendDirection.STRONG_DOWN
            strength = min(100, abs(ema_diff) * 20 + abs(momentum) * 10)
        elif ema_diff < -0.3 and not above_slow:
            trend = TrendDirection.DOWN
            strength = min(80, abs(ema_diff) * 15 + abs(momentum) * 8)
        else:
            trend = TrendDirection.NEUTRAL
            strength = max(0, 50 - abs(ema_diff) * 20)
        
        return trend, strength
    
    def get_confluence_signal(self, symbol: str) -> MultiTimeframeSignal:
        """
        Get confluence signal across all timeframes.
        
        Returns:
            MultiTimeframeSignal with alignment score and entry bias
        """
        all_dfs = self.aggregator.get_all_timeframes()
        analyses = {}
        
        # Analyze each timeframe
        for tf in [self.primary_tf] + self.context_tfs:
            df = all_dfs.get(tf)
            if df is not None and len(df) > 0:
                analysis = self.analyze_timeframe(tf, df)
                if analysis:
                    analyses[tf] = analysis
        
        if not analyses:
            return self._empty_signal(symbol)
        
        # Calculate alignment
        bullish_count = sum(1 for a in analyses.values() if a.trend in [TrendDirection.UP, TrendDirection.STRONG_UP])
        bearish_count = sum(1 for a in analyses.values() if a.trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN])
        total = len(analyses)
        
        if bullish_count > bearish_count:
            aligned = bullish_count
            consensus = TrendDirection.UP if bullish_count < total else TrendDirection.STRONG_UP
            entry_bias = "BUY"
        elif bearish_count > bullish_count:
            aligned = bearish_count
            consensus = TrendDirection.DOWN if bearish_count < total else TrendDirection.STRONG_DOWN
            entry_bias = "SELL"
        else:
            aligned = 0
            consensus = TrendDirection.NEUTRAL
            entry_bias = "HOLD"
        
        alignment_score = (aligned / total * 100) if total > 0 else 0
        
        # Determine signal strength
        if alignment_score >= 80:
            signal_strength = "strong"
        elif alignment_score >= 60:
            signal_strength = "moderate"
        elif alignment_score >= 40:
            signal_strength = "weak"
        else:
            signal_strength = "none"
        
        # Get primary and higher timeframe trends
        primary_analysis = analyses.get(self.primary_tf)
        primary_trend = primary_analysis.trend if primary_analysis else TrendDirection.NEUTRAL
        
        # Higher TF is the largest context timeframe
        htf = max(self.context_tfs, key=lambda tf: self.aggregator._tf_minutes[tf])
        htf_analysis = analyses.get(htf)
        htf_trend = htf_analysis.trend if htf_analysis else TrendDirection.NEUTRAL
        
        # Key levels from higher timeframes
        htf_support = htf_analysis.support if htf_analysis else 0
        htf_resistance = htf_analysis.resistance if htf_analysis else 0
        
        return MultiTimeframeSignal(
            symbol=symbol,
            timeframes_aligned=aligned,
            total_timeframes=total,
            alignment_score=alignment_score,
            consensus_direction=consensus,
            primary_tf_trend=primary_trend,
            higher_tf_trend=htf_trend,
            analyses=analyses,
            signal_strength=signal_strength,
            entry_bias=entry_bias,
            htf_support=htf_support,
            htf_resistance=htf_resistance
        )
    
    def _empty_signal(self, symbol: str) -> MultiTimeframeSignal:
        """Return empty signal when data insufficient."""
        return MultiTimeframeSignal(
            symbol=symbol,
            timeframes_aligned=0,
            total_timeframes=0,
            alignment_score=0,
            consensus_direction=TrendDirection.NEUTRAL,
            primary_tf_trend=TrendDirection.NEUTRAL,
            higher_tf_trend=TrendDirection.NEUTRAL,
            analyses={},
            signal_strength="none",
            entry_bias="HOLD",
            htf_support=0,
            htf_resistance=0
        )
    
    def should_trade_with_htf(
        self,
        primary_signal: str,  # "BUY" or "SELL"
        require_htf_alignment: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if primary signal aligns with higher timeframe context.
        
        Returns:
            (should_trade, reason)
        """
        signal = self.get_confluence_signal("")
        
        # Check if higher TF supports the trade
        htf_bullish = signal.higher_tf_trend in [TrendDirection.UP, TrendDirection.STRONG_UP]
        htf_bearish = signal.higher_tf_trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]
        htf_neutral = signal.higher_tf_trend == TrendDirection.NEUTRAL
        
        if primary_signal == "BUY":
            if htf_bullish:
                return True, "Higher TF supports long"
            elif htf_neutral and not require_htf_alignment:
                return True, "Higher TF neutral, proceeding with caution"
            elif htf_bearish:
                return False, "Higher TF bearish, avoid long"
            else:
                return not require_htf_alignment, "Higher TF not aligned"
        
        elif primary_signal == "SELL":
            if htf_bearish:
                return True, "Higher TF supports short"
            elif htf_neutral and not require_htf_alignment:
                return True, "Higher TF neutral, proceeding with caution"
            elif htf_bullish:
                return False, "Higher TF bullish, avoid short"
            else:
                return not require_htf_alignment, "Higher TF not aligned"
        
        return False, "Invalid signal"
