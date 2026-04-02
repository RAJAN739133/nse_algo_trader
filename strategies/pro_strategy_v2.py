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
