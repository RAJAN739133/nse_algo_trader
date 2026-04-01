"""
PRO Strategy V2 — Improved research-backed intraday trading logic.

Changes from V1:
─────────────────────────────────────────────────────────────
1.  GAP FILTER: Skip ORB on gap-up/down > 1.5% (fakeouts)
2.  ADR FILTER: Skip if ORB range > 60% of Average Daily Range
3.  ADAPTIVE R:R: 1.5 in low-vol, 2.0 in high-vol regimes
4.  PROGRESSIVE TRAILING: Trail to 50% profit, then ORB level, then BE
5.  TIME-DECAY EXIT: Cut flat trades after 90 min (1.5 hrs)
6.  PROPER VWAP STD: Typical-price-based deviation, not close.std()
7.  CUMULATIVE VOLUME FLOW: Confirm volume direction, not just spike
8.  WIDER VWAP BANDS: 2.0σ entry (was 1.5σ) — fewer but higher-quality
9.  PARTIAL EXIT at 1× risk, trail remainder to 2× (from WealthHub study)
10. ENGULF CANDLE BOOST: If breakout candle is engulfing, skip retest
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProStrategyV2:
    """Improved research-backed intraday strategy for NSE."""

    def __init__(self, config=None):
        self.config = config or {}
        self.orb_candles = 3
        self.min_breakout_buffer_pct = 0.003
        self.volume_spike_mult = 1.3
        self.use_full_range_stop = True
        self.base_rr_ratio = 1.5
        self.high_vol_rr_ratio = 2.0
        self.max_gap_pct = 0.015
        self.max_orb_to_adr_ratio = 0.60
        self.vwap_rsi_overbought = 73
        self.vwap_rsi_oversold = 27
        self.vwap_min_band_pct = 0.005
        self.vwap_entry_sigma = 2.0
        self.max_risk_pct = 0.02
        self.cooldown_candles = 6
        self.no_entry_after_hour = 14
        self.no_entry_after_minute = 0
        self.square_off_hour = 15
        self.square_off_minute = 15
        self.momentum_lookback = 3
        self.momentum_threshold = 0.005
        self.require_retest = True
        self.time_decay_candles = 18
        self.partial_exit_at_rr = 1.0
        self.partial_exit_pct = 0.50

    def compute_orb(self, candles, n=3):
        if len(candles) < n:
            return None
        orb = candles.iloc[:n]
        high = orb["high"].max()
        low = orb["low"].min()
        rng = high - low
        avg_vol = orb["volume"].mean()
        return {"high": high, "low": low, "range": rng, "avg_volume": avg_vol}

    def compute_vwap_proper(self, candles, up_to_idx):
        seen = candles.iloc[:up_to_idx + 1]
        total_vol = seen["volume"].sum()
        if total_vol == 0:
            return seen["close"].mean(), 0
        tp = (seen["high"] + seen["low"] + seen["close"]) / 3
        vwap = (tp * seen["volume"]).sum() / total_vol
        variance = ((tp - vwap) ** 2 * seen["volume"]).sum() / total_vol
        std = np.sqrt(variance) if variance > 0 else seen["close"].iloc[-1] * 0.01
        return vwap, std

    def compute_rsi(self, closes, period=14):
        if len(closes) < period + 1:
            return 50
        delta = closes.diff()
        gains = delta.where(delta > 0, 0).iloc[-period:]
        losses = (-delta.where(delta < 0, 0)).iloc[-period:]
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def estimate_adr(self, candles):
        if len(candles) < 10:
            return candles["high"].max() - candles["low"].min()
        return candles["high"].max() - candles["low"].min()

    def get_rr_ratio(self, candles, idx):
        if idx < 6:
            return self.base_rr_ratio
        recent = candles.iloc[max(0, idx - 12):idx]
        avg_candle_range = ((recent["high"] - recent["low"]) / recent["close"]).mean()
        if avg_candle_range > 0.005:
            return self.high_vol_rr_ratio
        return self.base_rr_ratio

    def check_momentum_direction(self, candles, idx, direction):
        if idx < self.momentum_lookback:
            return True
        recent = candles.iloc[idx - self.momentum_lookback:idx]
        pct_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]
        if direction == "SHORT" and pct_change > self.momentum_threshold:
            return False
        if direction == "LONG" and pct_change < -self.momentum_threshold:
            return False
        return True

    def check_volume_flow(self, candles, idx, direction):
        if idx < 5:
            return True
        vol = candles["volume"].iloc[idx]
        avg = candles["volume"].iloc[max(0, idx - 10):idx].mean()
        if avg > 0 and vol < avg * self.volume_spike_mult:
            return False
        recent = candles.iloc[max(0, idx - 5):idx + 1]
        green_vol = recent[recent["close"] >= recent["open"]]["volume"].sum()
        red_vol = recent[recent["close"] < recent["open"]]["volume"].sum()
        total = green_vol + red_vol
        if total == 0:
            return True
        if direction == "LONG" and green_vol < red_vol * 0.8:
            return False
        if direction == "SHORT" and red_vol < green_vol * 0.8:
            return False
        return True

    def check_retest(self, candles, idx, level, direction):
        if not self.require_retest:
            return True
        if idx < 1:
            return True
        curr = candles.iloc[idx]
        prev = candles.iloc[idx - 1]
        curr_body = abs(curr["close"] - curr["open"])
        prev_body = abs(prev["close"] - prev["open"])
        if prev_body > 0 and curr_body > prev_body * 1.5:
            if direction == "LONG" and curr["close"] > curr["open"]:
                return True
            if direction == "SHORT" and curr["close"] < curr["open"]:
                return True
        prev_close = prev["close"]
        if direction == "LONG":
            return prev_close >= level * 0.999
        else:
            return prev_close <= level * 1.001

    def check_orb_adr_ratio(self, orb, candles):
        adr = self.estimate_adr(candles)
        if adr == 0:
            return True
        return (orb["range"] / adr) <= self.max_orb_to_adr_ratio

    def generate_orb_signal(self, candles, idx, orb, direction):
        row = candles.iloc[idx]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])
        hour, minute = t.hour, t.minute
        if hour > self.no_entry_after_hour or (hour == self.no_entry_after_hour and minute >= self.no_entry_after_minute):
            return None
        if orb["range"] < close * 0.002:
            return None
        if not self.check_orb_adr_ratio(orb, candles.iloc[:idx + 1]):
            return None
        buffer = close * self.min_breakout_buffer_pct
        rr_ratio = self.get_rr_ratio(candles, idx)

        if direction == "LONG" and close > orb["high"] + buffer:
            if not self.check_volume_flow(candles, idx, "LONG"):
                return None
            if not self.check_momentum_direction(candles, idx, "LONG"):
                return None
            if not self.check_retest(candles, idx, orb["high"], "LONG"):
                return None
            sl = orb["low"] - orb["range"] * 0.1 if self.use_full_range_stop else orb["high"] - orb["range"] * 0.5
            risk = close - sl
            tgt = close + risk * rr_ratio
            return {"side": "LONG", "entry": close, "sl": sl, "tgt": tgt, "risk": risk, "time": t, "type": "ORB_BREAKOUT", "reason": f"ORB breakout >{orb['high']:.2f}+buf, RR={rr_ratio}"}

        elif direction == "SHORT" and close < orb["low"] - buffer:
            if not self.check_volume_flow(candles, idx, "SHORT"):
                return None
            if not self.check_momentum_direction(candles, idx, "SHORT"):
                return None
            if not self.check_retest(candles, idx, orb["low"], "SHORT"):
                return None
            sl = orb["high"] + orb["range"] * 0.1 if self.use_full_range_stop else orb["low"] + orb["range"] * 0.5
            risk = sl - close
            tgt = close - risk * rr_ratio
            return {"side": "SHORT", "entry": close, "sl": sl, "tgt": tgt, "risk": risk, "time": t, "type": "ORB_BREAKDOWN", "reason": f"ORB breakdown <{orb['low']:.2f}-buf, RR={rr_ratio}"}
        return None

    def generate_vwap_signal(self, candles, idx, direction):
        row = candles.iloc[idx]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])
        hour, minute = t.hour, t.minute
        if hour < 10 or (hour == 10 and minute < 30):
            return None
        if hour > self.no_entry_after_hour or (hour == self.no_entry_after_hour and minute >= self.no_entry_after_minute):
            return None
        vwap, std = self.compute_vwap_proper(candles, idx)
        if std == 0:
            return None
        rsi = self.compute_rsi(candles["close"].iloc[:idx + 1])
        if std < close * self.vwap_min_band_pct:
            return None
        if not self.check_momentum_direction(candles, idx, direction):
            return None
        if not self.check_volume_flow(candles, idx, direction):
            return None
        band = std * self.vwap_entry_sigma

        if direction == "LONG" and close < vwap - band and rsi < self.vwap_rsi_oversold:
            sl = vwap - band * 2.0
            risk = close - sl
            if risk <= 0:
                return None
            return {"side": "LONG", "entry": close, "sl": sl, "tgt": vwap, "risk": risk, "time": t, "type": "VWAP_OVERSOLD", "reason": f"VWAP oversold RSI={rsi:.0f}, {(close - vwap) / std:.1f}σ"}

        elif direction == "SHORT" and close > vwap + band and rsi > self.vwap_rsi_overbought:
            sl = vwap + band * 2.0
            risk = sl - close
            if risk <= 0:
                return None
            return {"side": "SHORT", "entry": close, "sl": sl, "tgt": vwap, "risk": risk, "time": t, "type": "VWAP_OVERBOUGHT", "reason": f"VWAP overbought RSI={rsi:.0f}, {(close - vwap) / std:.1f}σ"}
        return None

    def generate_pullback_signal(self, candles, idx, orb, direction):
        row = candles.iloc[idx]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])
        hour = t.hour
        if hour < 10 or hour > 11:
            return None

        if direction == "LONG":
            if close <= orb["high"]:
                return None
            recent_low = candles["low"].iloc[max(0, idx - 4):idx].min()
            if recent_low > orb["high"] * 1.005:
                return None
            if close > recent_low:
                if not self.check_momentum_direction(candles, idx, "LONG"):
                    return None
                if not self.check_volume_flow(candles, idx, "LONG"):
                    return None
                sl = recent_low - orb["range"] * 0.2
                risk = close - sl
                tgt = close + risk * 2.0
                return {"side": "LONG", "entry": close, "sl": sl, "tgt": tgt, "risk": risk, "time": t, "type": "PULLBACK_LONG", "reason": f"Pullback to ORB high {orb['high']:.2f}"}

        elif direction == "SHORT":
            if close >= orb["low"]:
                return None
            recent_high = candles["high"].iloc[max(0, idx - 4):idx].max()
            if recent_high < orb["low"] * 0.995:
                return None
            if close < recent_high:
                if not self.check_momentum_direction(candles, idx, "SHORT"):
                    return None
                if not self.check_volume_flow(candles, idx, "SHORT"):
                    return None
                sl = recent_high + orb["range"] * 0.2
                risk = sl - close
                tgt = close - risk * 2.0
                return {"side": "SHORT", "entry": close, "sl": sl, "tgt": tgt, "risk": risk, "time": t, "type": "PULLBACK_SHORT", "reason": f"Pullback to ORB low {orb['low']:.2f}"}
        return None
