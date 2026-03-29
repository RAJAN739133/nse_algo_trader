"""
Opening Range Breakout (ORB) Strategy for NSE Intraday.

Logic:
1. Wait for the first 15-minute candle (9:15 - 9:30) to form
2. Mark the high and low of that candle as the "opening range"
3. BUY if price breaks above ORB high with volume > 1.5× average
4. SELL if price breaks below ORB low with volume > 1.5× average
5. Stop loss = opposite end of the ORB range
6. Target = 1:1.5 or trail with Supertrend

Why this works on NSE:
- The 9:15-9:30 candle captures overnight gap + initial auction
- Breakout from this range has strong follow-through on trending days
- Volume confirmation filters out false breakouts
- Works best on liquid F&O stocks (Nifty 50)
"""

from typing import Optional
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal, TradeSignal


class ORBStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(name="ORB", config=config)
        self.orb_minutes = config.get("orb_period_minutes", 15)
        self.volume_mult = config.get("volume_multiplier", 1.5)
        self.atr_stop_mult = config.get("atr_stop_multiplier", 1.5)
        self.partial_exit_rr = config.get("partial_exit_rr", 1.0)

    def _find_orb_range(self, df: pd.DataFrame, current_idx: int) -> dict | None:
        """
        Find the ORB high/low for the current trading day.
        Uses the first N minutes of candles after market open (9:15).
        """
        if "datetime" in df.columns:
            times = pd.to_datetime(df["datetime"])
        else:
            times = pd.to_datetime(df.index)

        current_time = times.iloc[current_idx]
        current_date = current_time.date()

        # Find candles in the ORB window (9:15 to 9:15 + orb_minutes)
        market_open = pd.Timestamp(f"{current_date} 09:15:00")
        orb_end = market_open + pd.Timedelta(minutes=self.orb_minutes)

        # Can't generate signal during ORB formation period
        if current_time <= orb_end:
            return None

        # Get ORB candles
        mask = (times >= market_open) & (times <= orb_end) & (times.dt.date == current_date)
        day_mask = mask.values[:current_idx + 1]

        if not day_mask.any():
            return None

        orb_data = df.iloc[:current_idx + 1][day_mask]

        if len(orb_data) == 0:
            return None

        return {
            "high": orb_data["high"].max(),
            "low": orb_data["low"].min(),
            "volume": orb_data["volume"].mean(),
            "range": orb_data["high"].max() - orb_data["low"].min(),
        }

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int,
    ) -> Optional[TradeSignal]:
        """Generate ORB breakout signal."""
        if current_idx < 20:  # need some history for indicators
            return None

        orb = self._find_orb_range(df, current_idx)
        if orb is None:
            return None

        row = df.iloc[current_idx]
        symbol = row.get("symbol", "UNKNOWN")
        close = row["close"]
        volume = row["volume"]
        atr = row.get("atr", orb["range"])

        # Volume confirmation
        vol_avg = df["volume"].iloc[max(0, current_idx - 20):current_idx].mean()
        if vol_avg == 0:
            return None
        vol_ratio = volume / vol_avg

        # Skip if volume too low
        if vol_ratio < self.volume_mult:
            return None

        # BUY signal: close above ORB high
        if close > orb["high"]:
            stop = orb["low"] * 0.999  # just below ORB low
            risk = close - stop
            target = close + risk * 1.5  # 1:1.5 RR

            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(stop, 2),
                target_price=round(target, 2),
                strategy_name=self.name,
                confidence=min(1.0, vol_ratio / 3),  # higher vol = higher confidence
                reason=f"ORB breakout UP | Vol {vol_ratio:.1f}x | Range ₹{orb['range']:.2f}",
            )

        # SELL signal: close below ORB low
        if close < orb["low"]:
            stop = orb["high"] * 1.001  # just above ORB high
            risk = stop - close
            target = close - risk * 1.5

            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(stop, 2),
                target_price=round(target, 2),
                strategy_name=self.name,
                confidence=min(1.0, vol_ratio / 3),
                reason=f"ORB breakout DOWN | Vol {vol_ratio:.1f}x | Range ₹{orb['range']:.2f}",
            )

        return None

    def should_exit(
        self,
        df: pd.DataFrame,
        current_idx: int,
        entry_price: float,
        side: str,
    ) -> tuple[bool, str]:
        """Check for strategy-specific exit beyond stop loss."""
        if "datetime" in df.columns:
            current_time = pd.to_datetime(df["datetime"].iloc[current_idx])
        else:
            current_time = pd.to_datetime(df.index[current_idx])

        # Square off by 3:10 PM
        if current_time.hour >= 15 and current_time.minute >= 10:
            return True, "square_off"

        # Skip dead zone exit (optional — reduce exposure in choppy hours)
        if 11 <= current_time.hour < 14:
            row = df.iloc[current_idx]
            rsi = row.get("rsi", 50)
            # If RSI is neutral in dead zone, consider exiting
            if 45 < rsi < 55:
                return True, "dead_zone_neutral"

        return False, ""

    def pre_market_filter(
        self,
        symbol: str,
        gap_percent: float,
        avg_volume: float,
        vix: float,
    ) -> tuple[bool, str]:
        """ORB-specific pre-market filter."""
        allowed, reason = super().pre_market_filter(symbol, gap_percent, avg_volume, vix)
        if not allowed:
            return allowed, reason

        # ORB works best with moderate gaps (0.5-3%)
        # Avoid flat opens (no momentum) and huge gaps (exhaustion)
        if abs(gap_percent) < 0.2:
            return False, f"Gap too small ({gap_percent:.1f}%) — no momentum"

        if abs(gap_percent) > 3.0:
            return False, f"Gap too large ({gap_percent:.1f}%) — exhaustion risk"

        return True, "OK"
