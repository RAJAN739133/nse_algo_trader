"""ORB v2 — Opening Range Breakout with gap reversal + pullback filter."""
import logging
import pandas as pd
import numpy as np
from strategies.base import Signal, TradeSignal

logger = logging.getLogger(__name__)


class ORBv2Strategy:
    """Enhanced ORB: gap reversal detection, smart dead zone, pullback filter."""
    name = "ORB_v2"

    def __init__(self, config=None):
        self.config = config or {}
        self.orb_period = self.config.get("orb_period_minutes", 15)
        self.vol_mult = self.config.get("volume_multiplier", 1.5)
        self.atr_mult = self.config.get("atr_stop_multiplier", 1.5)

    def pre_market_filter(self, symbol, gap_pct, volume, vix):
        if abs(gap_pct) > 3.0: return False, "Gap too large"
        if vix > 25: return False, "VIX too high"
        return True, "OK"

    def check_signal(self, symbol, candles_15m, candles_1m, prev_close, vix=15):
        """
        Check for ORB v2 signals:
        1. Standard breakout above/below ORB range
        2. Gap reversal: open=day low/high with recovery (our specialty)
        3. Pullback detection: avoid false breakouts on gap-up days
        """
        if len(candles_15m) < 2:
            return None

        first = candles_15m.iloc[0]
        orb_high, orb_low = first["high"], first["low"]
        orb_range = orb_high - orb_low

        if orb_range < 1:
            return None  # too narrow

        latest = candles_1m.iloc[-1] if len(candles_1m) > 0 else first
        price = latest["close"]
        open_price = candles_1m.iloc[0]["open"] if len(candles_1m) > 0 else first["open"]
        gap_pct = (open_price - prev_close) / prev_close * 100 if prev_close else 0

        # === Gap Reversal Detection ===
        if abs(gap_pct) > 0.8:
            # Gap up but open = day low → bullish reversal
            if gap_pct > 0.8 and abs(open_price - candles_1m["low"].min()) < orb_range * 0.1:
                if price > open_price * 1.002:  # confirmed recovery
                    sl = candles_1m["low"].min() - orb_range * 0.3
                    return TradeSignal(
                        signal=Signal.BUY, symbol=symbol, entry_price=price,
                        stop_loss=sl, target_price=price + (price - sl) * 1.5,
                        strategy_name="ORB_v2_GAP_REV", confidence=0.7,
                        reason=f"Gap-up reversal: open={open_price:.2f} was day low, now recovering"
                    )
            # Gap down but open = day high → bearish breakdown signal
            if gap_pct < -0.8 and abs(open_price - candles_1m["high"].max()) < orb_range * 0.1:
                if price < open_price * 0.998:  # confirmed weakness
                    sl = candles_1m["high"].max() + orb_range * 0.3
                    return TradeSignal(
                        signal=Signal.SELL, symbol=symbol, entry_price=price,
                        stop_loss=sl, target_price=price - (sl - price) * 1.5,
                        strategy_name="ORB_v2_GAP_REV", confidence=0.7,
                        reason=f"Gap-down reversal: open={open_price:.2f} was day high, now breaking down"
                    )

        # === Pullback False Breakout Filter ===
        if gap_pct > 1.5 and len(candles_1m) > 15:
            # Check for false breakout signals on gap-up days
            recent = candles_1m.iloc[-5:]
            # Signal 1: Upper wick > body (weak buying)
            upper_wick = recent["high"] - recent[["close","open"]].max(axis=1)
            body = abs(recent["close"] - recent["open"])
            weak_break = (upper_wick > body).sum() >= 3
            # Signal 2: Volume declining on breakout
            vol_declining = recent["volume"].iloc[-1] < recent["volume"].iloc[0] * 0.7
            if weak_break and vol_declining:
                logger.info(f"  {symbol}: Pullback filter triggered — false breakout on gap-up")
                return None  # skip this breakout, wait for pullback

        # === Standard ORB Breakout ===
        if price > orb_high * 1.001:
            atr = orb_range * self.atr_mult
            sl = price - atr
            target = price + atr * 1.5
            return TradeSignal(
                signal=Signal.BUY, symbol=symbol, entry_price=price,
                stop_loss=sl, target_price=target,
                strategy_name="ORB_v2", confidence=0.65,
                reason=f"Breakout above ORB high {orb_high:.2f}"
            )

        # === Standard ORB Breakdown (SHORT) ===
        if price < orb_low * 0.999:
            atr = orb_range * self.atr_mult
            sl = price + atr
            target = price - atr * 1.5
            return TradeSignal(
                signal=Signal.SELL, symbol=symbol, entry_price=price,
                stop_loss=sl, target_price=target,
                strategy_name="ORB_v2", confidence=0.65,
                reason=f"Breakdown below ORB low {orb_low:.2f}"
            )

        return None
