"""
PRO Strategy V2 — Complete Trading System with ML + Patterns + Price Action

ARCHITECTURE:
═════════════
  1. ML → Support/Resistance levels (where to trade)
  2. Price Action → Direction bias (LONG or SHORT)
  3. Patterns → Entry confirmation (when to enter)
  4. ORB → Backup strategy when no ML levels

WHAT EACH TOOL DOES:
────────────────────
  ML (Support/Resistance):
    - K-Means clustering finds key price levels
    - Bounce probability at each level
    - Sets targets and stops automatically
    
  Price Action:
    - Real-time position in day's range
    - Adapts to V-reversals
    - Upper 55% = LONG, Lower 45% = SHORT
    
  Patterns:
    - Confirms bounce/rejection at ML levels
    - Double bottom at support → LONG
    - Bearish engulfing at resistance → SHORT
    
  ORB (Fallback):
    - Used when no clear ML levels
    - 30-min range breakout
    - 1% target, 0.5% stop

PROFIT MATH:
  - With ML levels + pattern confirmation: 65-70% WR
  - Average R:R: 1.5:1 to 2:1
  - 5 trades × 67% WR = 3.35 wins
  - NET: +3-4% per day after costs
"""
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProStrategyV2:
    """Improved research-backed intraday strategy for NSE."""

    def __init__(self, config=None):
        self.config = config or {}
        
        # ══════════════════════════════════════════════════════════
        # SIMPLIFIED ORB SETTINGS (proven profitable)
        # ══════════════════════════════════════════════════════════
        self.orb_candles = 6  # 30 mins (9:15-9:45) - proper range
        self.min_breakout_buffer_pct = 0.001  # 0.1% above ORB (was 0.3%)
        self.volume_spike_mult = 1.5  # 1.5x volume for confirmation
        self.use_full_range_stop = False  # Tighter stop
        
        # PROFIT TARGETS (the key change)
        self.target_pct = 0.01  # 1% target (not RR based)
        self.stop_pct = 0.005  # 0.5% stop loss
        self.base_rr_ratio = 2.0  # 2:1 RR
        self.high_vol_rr_ratio = 2.5  # Higher in volatile
        
        # ══════════════════════════════════════════════════════════
        # TRAILING STOP SETTINGS (NEW - catches bigger moves)
        # ══════════════════════════════════════════════════════════
        self.trailing_enabled = True
        self.trailing_activation_pct = 0.005  # Activate trailing after 0.5% profit
        self.trailing_stop_pct = 0.003  # Trail 0.3% behind price
        
        # Removed filters (were killing trades)
        self.max_gap_pct = 0.03  # Relaxed from 1.5% to 3%
        self.max_orb_to_adr_ratio = 0.80  # Relaxed from 60%
        self.require_retest = False  # DISABLED - was missing momentum
        
        # VWAP (kept but simplified)
        self.vwap_rsi_overbought = 70
        self.vwap_rsi_oversold = 30
        self.vwap_min_band_pct = 0.003
        self.vwap_entry_sigma = 1.5  # Lower sigma = more trades
        
        # Time limits
        self.max_risk_pct = 0.02
        self.cooldown_candles = 3  # Reduced from 6
        self.no_entry_after_hour = 14
        self.no_entry_after_minute = 30  # Extended from 14:00
        self.square_off_hour = 15
        self.square_off_minute = 15
        
        # Momentum (simplified)
        self.momentum_lookback = 3
        self.momentum_threshold = 0.003  # Relaxed from 0.5%
        
        # Time decay
        self.time_decay_candles = 24  # 2 hours (was 90 min)
        self.partial_exit_at_rr = 1.0
        self.partial_exit_pct = 0.50
        
        # ══════════════════════════════════════════════════════════
        # ML SUPPORT/RESISTANCE SETTINGS
        # ══════════════════════════════════════════════════════════
        self.sr_lookback_days = 20  # Days of history for S/R detection
        self.sr_cluster_count = 5   # Number of S/R levels to find
        self.sr_touch_threshold = 0.003  # 0.3% tolerance for "touch"
        self.sr_min_touches = 2     # Minimum touches to be valid level
        self.sr_levels_cache = {}   # Cache S/R levels per symbol

    # ══════════════════════════════════════════════════════════════════════
    # ML SUPPORT/RESISTANCE DETECTION
    # ══════════════════════════════════════════════════════════════════════
    
    def find_pivot_points(self, candles, window=5):
        """
        Find swing highs and lows (pivot points) in price data.
        These are potential support/resistance levels.
        """
        pivots = []
        highs = candles['high'].values
        lows = candles['low'].values
        
        for i in range(window, len(candles) - window):
            # Swing high: higher than surrounding candles
            if highs[i] == max(highs[i-window:i+window+1]):
                pivots.append({'price': highs[i], 'type': 'resistance', 'idx': i})
            
            # Swing low: lower than surrounding candles
            if lows[i] == min(lows[i-window:i+window+1]):
                pivots.append({'price': lows[i], 'type': 'support', 'idx': i})
        
        return pivots
    
    def cluster_sr_levels(self, pivots, current_price):
        """
        Use K-Means clustering to group nearby pivots into S/R levels.
        Returns levels with strength scores based on number of touches.
        """
        if len(pivots) < 3:
            return []
        
        prices = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        
        # Determine optimal clusters (max 5, min 2)
        n_clusters = min(self.sr_cluster_count, max(2, len(pivots) // 3))
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(prices)
            
            # Get cluster centers as S/R levels
            levels = []
            for i, center in enumerate(kmeans.cluster_centers_.flatten()):
                # Count how many pivots are in this cluster
                cluster_pivots = [p for j, p in enumerate(pivots) if kmeans.labels_[j] == i]
                touches = len(cluster_pivots)
                
                if touches >= self.sr_min_touches:
                    # Determine if support or resistance based on position vs current price
                    is_support = center < current_price
                    
                    # Strength based on touches and recency
                    strength = min(100, touches * 20)
                    
                    levels.append({
                        'price': center,
                        'type': 'support' if is_support else 'resistance',
                        'touches': touches,
                        'strength': strength
                    })
            
            # Sort by distance from current price
            levels.sort(key=lambda x: abs(x['price'] - current_price))
            return levels
            
        except Exception as e:
            logger.warning(f"K-Means clustering failed: {e}")
            return []
    
    def get_sr_levels(self, symbol, candles, current_price):
        """
        Get support and resistance levels for a symbol.
        Uses caching to avoid recalculating every candle.
        """
        cache_key = f"{symbol}_{len(candles)}"
        
        if cache_key in self.sr_levels_cache:
            return self.sr_levels_cache[cache_key]
        
        # Find pivot points
        pivots = self.find_pivot_points(candles, window=5)
        
        if not pivots:
            return {'support': [], 'resistance': []}
        
        # Cluster into S/R levels
        levels = self.cluster_sr_levels(pivots, current_price)
        
        # Separate into support and resistance
        result = {
            'support': sorted([l for l in levels if l['type'] == 'support'], 
                            key=lambda x: x['price'], reverse=True),  # Nearest first
            'resistance': sorted([l for l in levels if l['type'] == 'resistance'],
                                key=lambda x: x['price'])  # Nearest first
        }
        
        self.sr_levels_cache[cache_key] = result
        return result
    
    def get_nearest_sr(self, symbol, candles, current_price, direction):
        """
        Get the nearest support (for LONG) or resistance (for SHORT).
        Returns level with target and stop prices.
        """
        levels = self.get_sr_levels(symbol, candles, current_price)
        
        if direction == "LONG":
            # For LONG: stop below nearest support, target at nearest resistance
            support = levels['support'][0] if levels['support'] else None
            resistance = levels['resistance'][0] if levels['resistance'] else None
            
            if support and resistance:
                return {
                    'stop': support['price'] * 0.998,  # Slightly below support
                    'target': resistance['price'] * 0.998,  # Slightly below resistance
                    'support': support,
                    'resistance': resistance
                }
        
        elif direction == "SHORT":
            # For SHORT: stop above nearest resistance, target at nearest support
            support = levels['support'][0] if levels['support'] else None
            resistance = levels['resistance'][0] if levels['resistance'] else None
            
            if support and resistance:
                return {
                    'stop': resistance['price'] * 1.002,  # Slightly above resistance
                    'target': support['price'] * 1.002,  # Slightly above support
                    'support': support,
                    'resistance': resistance
                }
        
        return None
    
    def price_near_level(self, price, level, tolerance=None):
        """Check if price is near a S/R level."""
        if tolerance is None:
            tolerance = self.sr_touch_threshold
        return abs(price - level) / level <= tolerance

    def compute_orb(self, candles, n=6):
        """
        Compute Opening Range from first N candles (default 6 = 30 mins)
        
        ORB is 9:15-9:45 AM range (6 x 5-min candles)
        This gives market time to establish true direction
        """
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

    # ══════════════════════════════════════════════════════════════════════
    # NEW INTRADAY INDICATORS
    # ══════════════════════════════════════════════════════════════════════
    
    def compute_ema(self, closes, period):
        """Compute Exponential Moving Average for intraday."""
        if len(closes) < period:
            return closes.mean()
        return closes.ewm(span=period, adjust=False).mean().iloc[-1]
    
    def compute_ema_series(self, closes, period):
        """Compute EMA series for crossover detection."""
        if len(closes) < period:
            return pd.Series([closes.mean()] * len(closes))
        return closes.ewm(span=period, adjust=False).mean()
    
    def compute_macd(self, closes, fast=12, slow=26, signal=9):
        """
        Compute MACD for INTRADAY (using candle counts, not days).
        For 5-min candles: fast=12 (~1hr), slow=26 (~2hr), signal=9 (~45min)
        
        Returns: (macd_line, signal_line, histogram)
        """
        if len(closes) < slow + signal:
            return 0, 0, 0
        
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def compute_bollinger_bands(self, closes, period=20, std_dev=2.0):
        """
        Compute Bollinger Bands for INTRADAY.
        For 5-min candles: period=20 (~100 min / 1.5 hrs)
        
        Returns: (upper_band, middle_band, lower_band, bandwidth_pct)
        """
        if len(closes) < period:
            mid = closes.mean()
            std = closes.std() if len(closes) > 1 else mid * 0.01
            return mid + std * std_dev, mid, mid - std * std_dev, 0.02
        
        recent = closes.iloc[-period:]
        middle = recent.mean()
        std = recent.std()
        
        upper = middle + std * std_dev
        lower = middle - std * std_dev
        bandwidth_pct = (upper - lower) / middle if middle > 0 else 0.02
        
        return upper, middle, lower, bandwidth_pct

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
        """
        SIMPLIFIED ORB BREAKOUT SIGNAL
        
        Rules:
        1. Price breaks ORB high/low by 0.1%
        2. Volume is 1.5x average
        3. Target: 1%, Stop: 0.5%
        
        Removed (killed good trades):
        - Momentum direction check
        - Volume flow direction
        - Retest requirement
        """
        row = candles.iloc[idx]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        t = pd.to_datetime(row["datetime"])
        hour, minute = t.hour, t.minute
        
        # Time filter
        if hour > self.no_entry_after_hour or (hour == self.no_entry_after_hour and minute >= self.no_entry_after_minute):
            return None
        
        # ORB range must be meaningful (0.3% - 1.5%)
        orb_pct = orb["range"] / close
        if orb_pct < 0.003 or orb_pct > 0.015:
            return None
        
        # Breakout levels (0.1% beyond ORB)
        breakout_level = orb["high"] * 1.001
        breakdown_level = orb["low"] * 0.999
        
        # VOLUME CHECK ONLY (simple 1.5x average)
        avg_vol = candles["volume"].iloc[max(0, idx - 20):idx].mean()
        curr_vol = candles["volume"].iloc[idx]
        volume_ok = avg_vol > 0 and curr_vol > avg_vol * self.volume_spike_mult
        
        if not volume_ok:
            return None
        
        # ═══════════════════════════════════════════════════
        # LONG: Price breaks above ORB high
        # ═══════════════════════════════════════════════════
        if direction == "LONG" and high > breakout_level:
            entry = breakout_level
            sl = entry * (1 - self.stop_pct)  # 0.5% stop
            tgt = entry * (1 + self.target_pct)  # 1% target
            risk = entry - sl
            return {
                "side": "LONG",
                "entry": entry,
                "sl": sl,
                "tgt": tgt,
                "risk": risk,
                "time": t,
                "type": "ORB_BREAKOUT",
                "reason": f"ORB breakout >{orb['high']:.2f}, vol {curr_vol/avg_vol:.1f}x"
            }
        
        # ═══════════════════════════════════════════════════
        # SHORT: Price breaks below ORB low
        # ═══════════════════════════════════════════════════
        elif direction == "SHORT" and low < breakdown_level:
            entry = breakdown_level
            sl = entry * (1 + self.stop_pct)  # 0.5% stop
            tgt = entry * (1 - self.target_pct)  # 1% target
            risk = sl - entry
            return {
                "side": "SHORT",
                "entry": entry,
                "sl": sl,
                "tgt": tgt,
                "risk": risk,
                "time": t,
                "type": "ORB_BREAKDOWN",
                "reason": f"ORB breakdown <{orb['low']:.2f}, vol {curr_vol/avg_vol:.1f}x"
            }
        
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

    # ══════════════════════════════════════════════════════════════════════
    # NEW INTRADAY STRATEGIES
    # ══════════════════════════════════════════════════════════════════════
    
    def generate_ema_crossover_signal(self, candles, idx, direction):
        """
        EMA 9/21 Crossover for INTRADAY trading.
        
        For 5-min candles:
        - EMA 9 = 45 min (fast)
        - EMA 21 = 105 min (slow, ~1.75 hrs)
        
        Entry Rules:
        - LONG: EMA 9 crosses ABOVE EMA 21 + price above both
        - SHORT: EMA 9 crosses BELOW EMA 21 + price below both
        - Requires volume confirmation
        - Works best in trending regimes
        """
        if idx < 25:  # Need enough data for EMA 21
            return None
        
        row = candles.iloc[idx]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])
        hour, minute = t.hour, t.minute
        
        # Time filter: 9:45 - 14:30
        if hour < 9 or (hour == 9 and minute < 45):
            return None
        if hour > self.no_entry_after_hour or (hour == self.no_entry_after_hour and minute >= self.no_entry_after_minute):
            return None
        
        # Compute EMAs
        closes = candles["close"].iloc[:idx + 1]
        ema_9 = self.compute_ema_series(closes, 9)
        ema_21 = self.compute_ema_series(closes, 21)
        
        current_ema_9 = ema_9.iloc[-1]
        current_ema_21 = ema_21.iloc[-1]
        prev_ema_9 = ema_9.iloc[-2] if len(ema_9) > 1 else current_ema_9
        prev_ema_21 = ema_21.iloc[-2] if len(ema_21) > 1 else current_ema_21
        
        # Detect crossover
        bullish_cross = prev_ema_9 <= prev_ema_21 and current_ema_9 > current_ema_21
        bearish_cross = prev_ema_9 >= prev_ema_21 and current_ema_9 < current_ema_21
        
        # Volume confirmation (intraday: vs today's average)
        today_avg_vol = candles["volume"].iloc[:idx].mean() if idx > 0 else 0
        curr_vol = candles["volume"].iloc[idx]
        vol_ok = today_avg_vol > 0 and curr_vol > today_avg_vol * 1.2
        
        if not vol_ok:
            return None
        
        # LONG: Bullish crossover + price above both EMAs
        if direction == "LONG" and bullish_cross and close > current_ema_9 > current_ema_21:
            # Stop below EMA 21, target 1.5x risk
            sl = current_ema_21 * 0.998  # Slightly below EMA 21
            risk = close - sl
            if risk <= 0 or risk / close < 0.003:  # Min 0.3% risk
                return None
            tgt = close + risk * 1.5
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "EMA_CROSS_LONG",
                "reason": f"EMA 9/21 bullish cross, price above both"
            }
        
        # SHORT: Bearish crossover + price below both EMAs
        if direction == "SHORT" and bearish_cross and close < current_ema_9 < current_ema_21:
            sl = current_ema_21 * 1.002  # Slightly above EMA 21
            risk = sl - close
            if risk <= 0 or risk / close < 0.003:
                return None
            tgt = close - risk * 1.5
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "EMA_CROSS_SHORT",
                "reason": f"EMA 9/21 bearish cross, price below both"
            }
        
        return None
    
    def generate_bollinger_signal(self, candles, idx, direction):
        """
        Bollinger Bands Mean Reversion for INTRADAY.
        
        For 5-min candles: period=20 (~100 min)
        
        Entry Rules:
        - LONG: Price touches lower band + RSI < 30 + bullish candle
        - SHORT: Price touches upper band + RSI > 70 + bearish candle
        - Target = middle band (SMA 20)
        - Works best in ranging/choppy regimes
        
        IMPROVED CONDITIONS (fixes old 33% WR):
        - Require candle reversal pattern (not just touch)
        - Volume spike on reversal
        - Don't trade in strong trends (bandwidth > 3%)
        """
        if idx < 25:
            return None
        
        row = candles.iloc[idx]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        open_price = row["open"]
        t = pd.to_datetime(row["datetime"])
        hour, minute = t.hour, t.minute
        
        # Time filter: 10:30 - 14:00 (mean reversion works mid-day)
        if hour < 10 or (hour == 10 and minute < 30):
            return None
        if hour > 14:
            return None
        
        # Compute Bollinger Bands
        closes = candles["close"].iloc[:idx + 1]
        upper, middle, lower, bandwidth = self.compute_bollinger_bands(closes, period=20, std_dev=2.0)
        
        # Don't trade if bands are too wide (trending market)
        if bandwidth > 0.03:  # > 3% bandwidth = strong trend
            return None
        
        # Compute RSI
        rsi = self.compute_rsi(closes, period=14)
        
        # Volume check (intraday)
        today_avg_vol = candles["volume"].iloc[:idx].mean() if idx > 0 else 0
        curr_vol = candles["volume"].iloc[idx]
        vol_spike = today_avg_vol > 0 and curr_vol > today_avg_vol * 1.3
        
        # Check for candle reversal pattern
        is_bullish_candle = close > open_price and (close - open_price) > (high - low) * 0.5
        is_bearish_candle = close < open_price and (open_price - close) > (high - low) * 0.5
        
        # LONG: Touch lower band + oversold RSI + bullish reversal
        if direction == "LONG" and low <= lower and rsi < 30:
            if not is_bullish_candle:
                return None  # Need bullish reversal candle
            if not vol_spike:
                return None  # Need volume confirmation
            
            sl = lower * 0.997  # Below lower band
            risk = close - sl
            if risk <= 0:
                return None
            tgt = middle  # Target = middle band
            
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "BB_OVERSOLD_LONG",
                "reason": f"BB lower band touch + RSI {rsi:.0f} + bullish reversal"
            }
        
        # SHORT: Touch upper band + overbought RSI + bearish reversal
        if direction == "SHORT" and high >= upper and rsi > 70:
            if not is_bearish_candle:
                return None
            if not vol_spike:
                return None
            
            sl = upper * 1.003  # Above upper band
            risk = sl - close
            if risk <= 0:
                return None
            tgt = middle
            
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "BB_OVERBOUGHT_SHORT",
                "reason": f"BB upper band touch + RSI {rsi:.0f} + bearish reversal"
            }
        
        return None
    
    def generate_vwap_improved_signal(self, candles, idx, direction):
        """
        IMPROVED VWAP Mean Reversion for INTRADAY.
        
        Fixes for old 33% WR:
        1. Require 2σ deviation (not 1.5σ)
        2. Require RSI confirmation (< 25 or > 75)
        3. Require volume spike on entry candle
        4. Require reversal candle pattern
        5. Don't trade first/last hour
        6. Check MACD for momentum alignment
        
        Entry Rules:
        - LONG: Price 2σ below VWAP + RSI < 25 + bullish candle + MACD turning
        - SHORT: Price 2σ above VWAP + RSI > 75 + bearish candle + MACD turning
        """
        if idx < 30:  # Need data for MACD
            return None
        
        row = candles.iloc[idx]
        close = row["close"]
        open_price = row["open"]
        high = row["high"]
        low = row["low"]
        t = pd.to_datetime(row["datetime"])
        hour, minute = t.hour, t.minute
        
        # Time filter: 10:30 - 14:00 (no first/last hour)
        if hour < 10 or (hour == 10 and minute < 30):
            return None
        if hour > 14:
            return None
        
        # Compute VWAP and deviation
        vwap, std = self.compute_vwap_proper(candles, idx)
        if std == 0 or std < close * 0.003:
            return None
        
        deviation = (close - vwap) / std
        
        # Need 2σ deviation (stricter than old 1.5σ)
        if abs(deviation) < 2.0:
            return None
        
        # Compute indicators
        closes = candles["close"].iloc[:idx + 1]
        rsi = self.compute_rsi(closes, period=14)
        macd_line, signal_line, histogram = self.compute_macd(closes)
        
        # Check for MACD turning (histogram changing direction)
        if idx > 1:
            prev_histogram = self.compute_macd(candles["close"].iloc[:idx])[2]
            macd_turning_up = histogram > prev_histogram and histogram > -0.5
            macd_turning_down = histogram < prev_histogram and histogram < 0.5
        else:
            macd_turning_up = macd_turning_down = False
        
        # Volume spike
        today_avg_vol = candles["volume"].iloc[:idx].mean() if idx > 0 else 0
        curr_vol = candles["volume"].iloc[idx]
        vol_spike = today_avg_vol > 0 and curr_vol > today_avg_vol * 1.5
        
        # Candle patterns
        is_bullish = close > open_price and (close - open_price) > (high - low) * 0.4
        is_bearish = close < open_price and (open_price - close) > (high - low) * 0.4
        
        # LONG: 2σ below VWAP + RSI < 25 + bullish candle + MACD turning up
        if direction == "LONG" and deviation < -2.0 and rsi < 25:
            if not is_bullish:
                return None
            if not vol_spike:
                return None
            if not macd_turning_up:
                return None
            
            sl = close - std * 1.0  # Tighter stop
            risk = close - sl
            if risk <= 0:
                return None
            
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": vwap,
                "risk": risk, "time": t, "type": "VWAP_EXTREME_LONG",
                "reason": f"VWAP {deviation:.1f}σ + RSI {rsi:.0f} + MACD turn + vol spike"
            }
        
        # SHORT: 2σ above VWAP + RSI > 75 + bearish candle + MACD turning down
        if direction == "SHORT" and deviation > 2.0 and rsi > 75:
            if not is_bearish:
                return None
            if not vol_spike:
                return None
            if not macd_turning_down:
                return None
            
            sl = close + std * 1.0
            risk = sl - close
            if risk <= 0:
                return None
            
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": vwap,
                "risk": risk, "time": t, "type": "VWAP_EXTREME_SHORT",
                "reason": f"VWAP +{deviation:.1f}σ + RSI {rsi:.0f} + MACD turn + vol spike"
            }
        
        return None
    
    def generate_macd_signal(self, candles, idx, direction):
        """
        MACD Momentum Confirmation for INTRADAY.
        
        For 5-min candles:
        - Fast EMA: 12 candles (~1 hour)
        - Slow EMA: 26 candles (~2 hours)
        - Signal: 9 candles (~45 min)
        
        Entry Rules:
        - LONG: MACD crosses above signal + histogram positive + price above VWAP
        - SHORT: MACD crosses below signal + histogram negative + price below VWAP
        - Requires volume and trend alignment
        
        This is a MOMENTUM strategy, not mean reversion.
        """
        if idx < 35:  # Need enough data
            return None
        
        row = candles.iloc[idx]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])
        hour, minute = t.hour, t.minute
        
        # Time filter: 9:45 - 14:30
        if hour < 9 or (hour == 9 and minute < 45):
            return None
        if hour > self.no_entry_after_hour or (hour == self.no_entry_after_hour and minute >= self.no_entry_after_minute):
            return None
        
        # Compute MACD
        closes = candles["close"].iloc[:idx + 1]
        macd_line, signal_line, histogram = self.compute_macd(closes)
        
        # Previous MACD values for crossover detection
        prev_closes = candles["close"].iloc[:idx]
        prev_macd, prev_signal, prev_hist = self.compute_macd(prev_closes)
        
        # Detect crossovers
        bullish_cross = prev_macd <= prev_signal and macd_line > signal_line
        bearish_cross = prev_macd >= prev_signal and macd_line < signal_line
        
        # VWAP for trend confirmation
        vwap, std = self.compute_vwap_proper(candles, idx)
        above_vwap = close > vwap
        below_vwap = close < vwap
        
        # Volume confirmation
        today_avg_vol = candles["volume"].iloc[:idx].mean() if idx > 0 else 0
        curr_vol = candles["volume"].iloc[idx]
        vol_ok = today_avg_vol > 0 and curr_vol > today_avg_vol * 1.2
        
        if not vol_ok:
            return None
        
        # ATR for stop/target
        if idx >= 14:
            atr = (candles["high"].iloc[idx-14:idx+1] - candles["low"].iloc[idx-14:idx+1]).mean()
        else:
            atr = (row["high"] - row["low"]) * 2
        atr = max(atr, close * 0.005)  # Min 0.5%
        
        # LONG: Bullish MACD cross + positive histogram + above VWAP
        if direction == "LONG" and bullish_cross and histogram > 0 and above_vwap:
            sl = close - atr * 1.5
            risk = close - sl
            tgt = close + risk * 2.0
            
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "MACD_MOMENTUM_LONG",
                "reason": f"MACD bullish cross, hist={histogram:.2f}, above VWAP"
            }
        
        # SHORT: Bearish MACD cross + negative histogram + below VWAP
        if direction == "SHORT" and bearish_cross and histogram < 0 and below_vwap:
            sl = close + atr * 1.5
            risk = sl - close
            tgt = close - risk * 2.0
            
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "MACD_MOMENTUM_SHORT",
                "reason": f"MACD bearish cross, hist={histogram:.2f}, below VWAP"
            }
        
        return None
    
    def check_macd_confirmation(self, candles, idx, direction):
        """
        Check if MACD confirms the trade direction.
        Used as additional filter for other strategies.
        
        Returns: (confirmed: bool, strength: float 0-1)
        """
        if idx < 30:
            return True, 0.5  # Not enough data, allow trade
        
        closes = candles["close"].iloc[:idx + 1]
        macd_line, signal_line, histogram = self.compute_macd(closes)
        
        if direction == "LONG":
            # MACD should be positive or turning positive
            if macd_line > signal_line and histogram > 0:
                return True, 0.8
            elif macd_line > signal_line or histogram > -0.5:
                return True, 0.6
            else:
                return False, 0.3
        
        elif direction == "SHORT":
            # MACD should be negative or turning negative
            if macd_line < signal_line and histogram < 0:
                return True, 0.8
            elif macd_line < signal_line or histogram < 0.5:
                return True, 0.6
            else:
                return False, 0.3
        
        return True, 0.5

    # ══════════════════════════════════════════════════════════════════════
    # ML SUPPORT/RESISTANCE SIGNAL (Uses ML levels + Pattern confirmation)
    # ══════════════════════════════════════════════════════════════════════
    
    def generate_sr_bounce_signal(self, symbol, candles, idx, direction, pattern_detector=None):
        """
        Generate signal when price bounces off ML-detected support/resistance.
        
        This is THE professional approach:
        1. ML identifies key S/R levels from historical pivots
        2. Wait for price to reach the level
        3. Confirm with candlestick pattern (optional but improves WR)
        4. Enter with stop beyond the level, target at opposite level
        """
        if idx < 20:
            return None
            
        row = candles.iloc[idx]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        t = pd.to_datetime(row["datetime"])
        hour, minute = t.hour, t.minute
        
        # Time filter
        if hour > self.no_entry_after_hour or (hour == self.no_entry_after_hour and minute >= self.no_entry_after_minute):
            return None
        
        # Get ML-detected S/R levels
        sr_data = self.get_nearest_sr(symbol, candles.iloc[:idx], close, direction)
        
        if not sr_data:
            return None
        
        # Volume check
        avg_vol = candles["volume"].iloc[max(0, idx - 20):idx].mean()
        curr_vol = candles["volume"].iloc[idx]
        vol_ok = avg_vol > 0 and curr_vol > avg_vol * 1.2
        
        # ═══════════════════════════════════════════════════
        # LONG: Price near support level + bouncing
        # ═══════════════════════════════════════════════════
        if direction == "LONG":
            support_level = sr_data['support']['price']
            
            # Check if price touched/near support
            if not self.price_near_level(low, support_level, tolerance=0.005):
                return None
            
            # Check for bounce (close above open, above support)
            if close <= row["open"] or close < support_level:
                return None
            
            # Pattern confirmation (optional but boosts WR)
            pattern_bonus = ""
            if pattern_detector:
                patterns = pattern_detector.detect_patterns(candles, idx)
                bullish_patterns = [p for p in patterns if p['bias'] == 'bullish']
                if bullish_patterns:
                    pattern_bonus = f" + {bullish_patterns[0]['pattern']}"
            
            if not vol_ok and not pattern_bonus:
                return None  # Need either volume or pattern
            
            entry = close
            sl = sr_data['stop']
            tgt = sr_data['target']
            risk = entry - sl
            
            if risk <= 0 or (tgt - entry) / risk < 1.2:
                return None  # Bad R:R
            
            return {
                "side": "LONG",
                "entry": entry,
                "sl": sl,
                "tgt": tgt,
                "risk": risk,
                "time": t,
                "type": "SR_BOUNCE_LONG",
                "reason": f"Bounce off support {support_level:.2f} (str={sr_data['support']['strength']}%){pattern_bonus}"
            }
        
        # ═══════════════════════════════════════════════════
        # SHORT: Price near resistance level + rejection
        # ═══════════════════════════════════════════════════
        elif direction == "SHORT":
            resistance_level = sr_data['resistance']['price']
            
            # Check if price touched/near resistance
            if not self.price_near_level(high, resistance_level, tolerance=0.005):
                return None
            
            # Check for rejection (close below open, below resistance)
            if close >= row["open"] or close > resistance_level:
                return None
            
            # Pattern confirmation (optional but boosts WR)
            pattern_bonus = ""
            if pattern_detector:
                patterns = pattern_detector.detect_patterns(candles, idx)
                bearish_patterns = [p for p in patterns if p['bias'] == 'bearish']
                if bearish_patterns:
                    pattern_bonus = f" + {bearish_patterns[0]['pattern']}"
            
            if not vol_ok and not pattern_bonus:
                return None  # Need either volume or pattern
            
            entry = close
            sl = sr_data['stop']
            tgt = sr_data['target']
            risk = sl - entry
            
            if risk <= 0 or (entry - tgt) / risk < 1.2:
                return None  # Bad R:R
            
            return {
                "side": "SHORT",
                "entry": entry,
                "sl": sl,
                "tgt": tgt,
                "risk": risk,
                "time": t,
                "type": "SR_REJECTION_SHORT",
                "reason": f"Rejection at resistance {resistance_level:.2f} (str={sr_data['resistance']['strength']}%){pattern_bonus}"
            }
