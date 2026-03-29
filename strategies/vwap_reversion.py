"""
VWAP Mean Reversion Strategy for NSE Intraday.

Logic:
1. Calculate VWAP and its standard deviation bands throughout the day
2. BUY when price drops to VWAP - 1σ band (oversold) + overall trend is UP
3. SELL when price rises to VWAP + 1σ band (overbought) + overall trend is DOWN
4. Target = VWAP itself (the mean)
5. Stop = VWAP ± 2σ band

Why this works on NSE:
- VWAP is the benchmark price institutions use for execution
- Price tends to revert to VWAP on range-bound days
- Works best on Nifty 50 large caps with institutional participation
- Higher win rate (~55-65%) but smaller reward per trade than ORB
"""

from typing import Optional
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal, TradeSignal


class VWAPReversionStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(name="VWAP_Reversion", config=config)
        self.entry_band = config.get("entry_band", 1.0)   # enter at ± 1σ
        self.stop_band = config.get("stop_band", 2.0)      # stop at ± 2σ
        self.target_band = config.get("target_band", 0.0)   # target = VWAP
        self.min_vol_pctile = config.get("min_volume_percentile", 60)

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int,
    ) -> Optional[TradeSignal]:
        """Generate VWAP mean reversion signal."""
        if current_idx < 30:  # need enough data for VWAP to stabilize
            return None

        row = df.iloc[current_idx]
        symbol = row.get("symbol", "UNKNOWN")
        close = row["close"]
        vwap = row.get("vwap", None)
        vwap_std = row.get("vwap_std", None)

        if vwap is None or vwap_std is None or pd.isna(vwap) or pd.isna(vwap_std):
            return None

        if vwap_std == 0:
            return None

        # Check time — VWAP reversion works best after 10:00 AM
        if "datetime" in df.columns:
            current_time = pd.to_datetime(df["datetime"].iloc[current_idx])
        else:
            current_time = pd.to_datetime(df.index[current_idx])

        if current_time.hour < 10:
            return None  # too early, VWAP not stable yet

        if current_time.hour >= 15:
            return None  # too late, approaching close

        # Calculate bands
        upper_entry = vwap + self.entry_band * vwap_std
        lower_entry = vwap - self.entry_band * vwap_std
        upper_stop = vwap + self.stop_band * vwap_std
        lower_stop = vwap - self.stop_band * vwap_std

        # Volume check — only trade when volume is above average
        vol_ratio = row.get("vol_ratio", 1.0)
        if pd.isna(vol_ratio) or vol_ratio < 0.8:
            return None

        # Trend filter: use EMA to determine if we should go long or short
        ema = row.get("ema_20", vwap)
        rsi = row.get("rsi", 50)

        # BUY: price at or below lower band + RSI < 40 (oversold)
        if close <= lower_entry and rsi < 40:
            target = vwap + self.target_band * vwap_std

            # Sanity: reward must be worth it
            reward = target - close
            risk = close - lower_stop
            if risk <= 0 or reward / risk < 0.8:
                return None

            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(lower_stop, 2),
                target_price=round(target, 2),
                strategy_name=self.name,
                confidence=min(1.0, (40 - rsi) / 30),
                reason=f"VWAP reversion BUY | RSI={rsi:.0f} | {(close-vwap)/vwap_std:.1f}σ from VWAP",
            )

        # SELL: price at or above upper band + RSI > 60 (overbought)
        if close >= upper_entry and rsi > 60:
            target = vwap - self.target_band * vwap_std

            reward = close - target
            risk = upper_stop - close
            if risk <= 0 or reward / risk < 0.8:
                return None

            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(upper_stop, 2),
                target_price=round(target, 2),
                strategy_name=self.name,
                confidence=min(1.0, (rsi - 60) / 30),
                reason=f"VWAP reversion SELL | RSI={rsi:.0f} | {(close-vwap)/vwap_std:.1f}σ from VWAP",
            )

        return None

    def should_exit(
        self,
        df: pd.DataFrame,
        current_idx: int,
        entry_price: float,
        side: str,
    ) -> tuple[bool, str]:
        """VWAP-specific exit: exit if price crosses VWAP (target reached)."""
        row = df.iloc[current_idx]
        close = row["close"]
        vwap = row.get("vwap", entry_price)

        if "datetime" in df.columns:
            current_time = pd.to_datetime(df["datetime"].iloc[current_idx])
        else:
            current_time = pd.to_datetime(df.index[current_idx])

        # Square off by 3:10 PM
        if current_time.hour >= 15 and current_time.minute >= 10:
            return True, "square_off"

        # Target: price reverted to VWAP
        if side == "BUY" and close >= vwap:
            return True, "target_vwap"

        if side == "SELL" and close <= vwap:
            return True, "target_vwap"

        return False, ""

    def pre_market_filter(
        self,
        symbol: str,
        gap_percent: float,
        avg_volume: float,
        vix: float,
    ) -> tuple[bool, str]:
        """VWAP reversion works best on low-gap, range-bound days."""
        allowed, reason = super().pre_market_filter(symbol, gap_percent, avg_volume, vix)
        if not allowed:
            return allowed, reason

        # VWAP reversion prefers small gaps (range-bound days)
        if abs(gap_percent) > 1.5:
            return False, f"Gap too large ({gap_percent:.1f}%) — trending day, use ORB"

        return True, "OK"
