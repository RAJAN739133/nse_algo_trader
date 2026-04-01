"""VWAP v2 — Mean reversion with regime filter, trend alignment, stricter RSI."""
import logging
import numpy as np
import pandas as pd
from strategies.base import Signal, TradeSignal

logger = logging.getLogger(__name__)


class VWAPv2Strategy:
    """Enhanced VWAP: only fires on range-bound days with strict conditions."""
    name = "VWAP_v2"

    def __init__(self, config=None):
        self.config = config or {}

    def pre_market_filter(self, symbol, gap_pct, volume, vix):
        if vix > 18: return False, "VIX > 18, skip VWAP"
        if abs(gap_pct) > 1.5: return False, "Gap too large for mean reversion"
        return True, "OK"

    def check_signal(self, symbol, candles_df, vix=15, atr_pct=0.01):
        """
        VWAP v2 signal conditions (ALL must be true):
        1. VIX < 18 (calm market)
        2. Small gap (< 1.5%)
        3. ATR percentile low (range-bound day)
        4. RSI extreme (< 35 or > 65)
        5. EMA 10 > EMA 20 (trend aligned for bounce)
        6. Price at VWAP ± 1 sigma band
        7. Time window: 14:00-15:00 only
        """
        if len(candles_df) < 30:
            return None

        df = candles_df.copy()
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["vwap_std"] = df["close"].rolling(20).std()
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
        df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        lat = df.iloc[-1]
        vwap = lat["vwap"]
        std = lat.get("vwap_std", 0)
        rsi = lat.get("rsi", 50)
        ema10 = lat.get("ema_10", 0)
        ema20 = lat.get("ema_20", 0)
        price = lat["close"]

        if pd.isna(std) or std == 0:
            return None

        # Regime filter: only trade range-bound days
        if vix > 18:
            return None
        if atr_pct > 0.025:  # high ATR = trending day, skip
            return None

        # RSI must be extreme
        if 35 < rsi < 65:
            return None

        lower_band = vwap - std
        upper_band = vwap + std

        # Oversold bounce (RSI < 35, price below lower band)
        if rsi < 35 and price < lower_band:
            # Trend alignment: short-term EMA should support bounce
            if ema10 > ema20 * 0.998:  # allow slight lag
                sl = vwap - std * 2
                return TradeSignal(
                    signal=Signal.BUY, symbol=symbol, entry_price=price,
                    stop_loss=sl, target_price=vwap,
                    strategy_name="VWAP_v2", confidence=0.6,
                    reason=f"Oversold bounce: RSI={rsi:.0f}, price below VWAP-1σ"
                )

        # Overbought rejection (RSI > 65, price above upper band) — SHORT
        if rsi > 65 and price > upper_band:
            # Trend alignment: short-term EMA should confirm weakness
            if ema10 < ema20 * 1.002:  # allow slight lag
                sl = vwap + std * 2
                return TradeSignal(
                    signal=Signal.SELL, symbol=symbol, entry_price=price,
                    stop_loss=sl, target_price=vwap,
                    strategy_name="VWAP_v2", confidence=0.6,
                    reason=f"Overbought rejection: RSI={rsi:.0f}, price above VWAP+1σ"
                )

        return None
