#!/usr/bin/env python3
"""
Live Paper Trader V3 — Production-Ready Adaptive Trading System
═══════════════════════════════════════════════════════════════

DATA SOURCES (Priority Order):
  1. Angel One REST API — Primary (faster, more reliable for live)
  2. yfinance — Fallback when Angel One unavailable

KEY FEATURES:
  1. DYNAMIC STOCK SCORING: Rolling performance-based filtering (no hardcoding)
  2. ANGEL ONE INTEGRATION: Real-time 5-min candles via SmartAPI
  3. CANDLESTICK PATTERNS: Japanese patterns for entry/exit confirmation
  4. REGIME DETECTION: Trending/Ranging/Volatile detection per stock
  5. DYNAMIC MARKET TREND: Auto-detects bullish/bearish market each day
  6. ML CONFIDENCE FILTER: Minimum 12% confidence (abs(prob-0.5) >= 0.12)
  7. NO STOP LOSSES: Time-based exits only (stop losses were 100% losers)
  8. CLAUDE BRAIN V2: AI market analysis with 5-min scans
  9. TELEGRAM ALERTS: Trade entry/exit with timestamps, EOD summary

STRATEGY LOGIC:
  - Trending Up/Down: ORB breakout + momentum continuation
  - Ranging: VWAP mean-reversion with tighter bands (2.0σ)
  - Choppy: Selective trades with volume confirmation
  - Volatile: Wider stops, smaller positions, strongest signals only

MARKET TREND DETECTION (at market open):
  - Analyzes Nifty 50 pre-market data, VIX level, global cues
  - BULLISH market → Enable LONGS, limit SHORTS
  - BEARISH market → Enable SHORTS, limit LONGS
  - NEUTRAL/SIDEWAYS → Enable BOTH with balanced approach

Usage:
  python live_paper_v3.py                         # Auto-select best stocks
  python live_paper_v3.py --universe nifty100     # From Nifty 100 (default)
  python live_paper_v3.py --stocks HDFCBANK SBIN  # Specific stocks

Schedule: Scheduler calls this at 9:05 AM Mon-Fri.
"""
import os, sys, time, logging, json, pickle, argparse
from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.symbols import NIFTY_50, NIFTY_100_EXTRA, NIFTY_250_EXTRA, STOCK_SECTORS, get_universe, ALL_STOCKS
from data.data_loader import DataLoader
from data.train_pipeline import add_features, score_stocks, FEATURE_COLS

# Try to use V2 training pipeline if available
try:
    from data.train_v2 import add_features as add_features_v2, score_stocks as score_stocks_v2, FEATURE_COLS as FEATURE_COLS_V2
    USE_V2_MODEL = True
except ImportError:
    USE_V2_MODEL = False
from backtest.costs import AngelOneCostModel
from strategies.pro_strategy_v2 import ProStrategyV2
from strategies.claude_brain_v2 import ClaudeBrainV2
from strategies.candle_patterns import CandlePatternDetector

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/live_v3_{date.today()}.log"),
    ],
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# DYNAMIC STOCK FILTERING (same as backtest - no hardcoding)
# ════════════════════════════════════════════════════════════

# Import dynamic tracker
try:
    from data.stock_performance_tracker import get_tracker, StockPerformanceTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════
# LOSS MINIMIZATION SETTINGS (Based on 48-trade analysis)
# ══════════════════════════════════════════════════════════════════════
# 
# WINNING PATTERNS:
#   - SHORT trades: 59% WR, +₹12,669 profit
#   - MOMENTUM_SHORT: 73% WR, +₹5,196 profit  
#   - AFTERNOON_TREND_SHORT: 83% WR, +₹7,534 profit
#   - ORB_BREAKDOWN: 100% WR, +₹214 profit
#   - TRENDING_DOWN regime: 86% WR, +₹10,048 profit
#   - TARGET exits: 100% WR, +₹12,138 profit
#
# LOSING PATTERNS (DISABLED):
#   - LONG trades: 38% WR, -₹1,357 loss → DISABLED in bear/neutral markets
#   - PULLBACK strategy: 29-43% WR → DISABLED
#   - VWAP_SHORT: 33% WR → DISABLED  
#   - RANGING regime: 33% WR → DISABLED
#   - TRENDING_UP regime: 20% WR → DISABLED
#   - UNKNOWN regime: 0% WR → DISABLED
#   - TIME_DECAY exits: 24% WR → Reduced threshold
#
# ══════════════════════════════════════════════════════════════════════

MIN_ML_CONFIDENCE = 0.12  # Minimum abs(prob - 0.5) threshold
DISABLE_STOP_LOSSES = True  # Regular stop losses were 100% losers in backtest
EMERGENCY_STOP_LOSS_PCT = 0.05  # 5% HARD STOP - protects against flash crashes

# Strategy filters (disable losing strategies)
DISABLE_PULLBACK = True      # 29-43% WR - losing strategy
DISABLE_VWAP = True          # 33% WR for VWAP_SHORT, VWAP_LONG similar
ADAPTIVE_DIRECTION = True    # Follow market trend - trade WITH the market

# Regime filters (trade in favorable regimes only)
# NOTE: trending_up is good for LONGS, trending_down good for SHORTS
BLOCKED_REGIMES = ["ranging", "unknown"]  # Low WR in all directions

# Dynamic market trend (detected at market open from Nifty/VIX analysis)
# This will be updated by detect_market_trend() before trading starts
MARKET_TREND = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
ENABLE_LONGS = True       # Will be set based on market trend
ENABLE_SHORTS = True      # Will be set based on market trend
MAX_LONGS = 5             # Dynamic limit
MAX_SHORTS = 5            # Dynamic limit

# ══════════════════════════════════════════════════════════════════════
# KEY FIX: Trade WITH the market, not against it
# - In BULLISH market: Prioritize LONGs, limit SHORTs
# - In BEARISH market: Prioritize SHORTs, limit LONGs  
# - In NEUTRAL: Be selective, smaller positions
# ══════════════════════════════════════════════════════════════════════

def get_dynamic_filters():
    """
    Get dynamic blacklist/whitelist from rolling performance.
    Same logic as backtest - no hardcoding.
    """
    if not TRACKER_AVAILABLE:
        return set(), set(), {}
    
    tracker = get_tracker()
    avoid = tracker.get_avoid_stocks(min_trades=5, max_wr=0.42)
    prefer = tracker.get_preferred_stocks(min_trades=5, min_wr=0.52)
    scores = {sym: tracker.get_stock_score(sym) for sym in tracker.stats.keys()}
    
    return avoid, prefer, scores

# Initialize dynamic filters (updated daily)
STOCK_BLACKLIST, STOCK_WHITELIST, STOCK_SCORES = get_dynamic_filters()


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

def load_config():
    for p in ["config/config_test.yaml", "config/config_prod.yaml", "config/config_example.yaml"]:
        if Path(p).exists():
            with open(p) as f:
                return yaml.safe_load(f)
    return {"capital": {"total": 10000, "risk_per_trade": 0.01, "max_trades_per_day": 5,
                        "daily_loss_limit": 0.03},
            "alerts": {"telegram_enabled": True}}


# ════════════════════════════════════════════════════════════
# TELEGRAM — sends on EVERY event
# ════════════════════════════════════════════════════════════

def send_telegram(msg, config):
    alerts = config.get("alerts", {})
    if not alerts.get("telegram_enabled"):
        logger.info(f"  [TG OFF] {msg[:80]}...")
        return
    token = alerts.get("telegram_bot_token", "")
    chat_ids = alerts.get("telegram_chat_ids", [])
    single_id = alerts.get("telegram_chat_id", "")
    if single_id and single_id not in chat_ids:
        chat_ids.append(single_id)
    if not token or not chat_ids:
        return
    try:
        import urllib.request
        for chat_id in chat_ids:
            if not chat_id:
                continue
            data = json.dumps({"chat_id": chat_id, "text": msg}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=data, headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


# ════════════════════════════════════════════════════════════
# LIVE HOLIDAY CHECK — no hardcoded list
# ════════════════════════════════════════════════════════════

def is_market_open_today():
    """Check if market is open by trying to fetch live data."""
    d = date.today()
    if d.weekday() >= 5:
        return False, "Weekend"
    try:
        import yfinance as yf
        # Fetch last 5 days of 1-min data — if today has candles, market is open
        df = yf.Ticker("^NSEI").history(period="5d", interval="1m")
        if len(df) > 0:
            latest_date = pd.to_datetime(df.index[-1]).date()
            if latest_date == d:
                return True, "Market open"
            # No data today yet — check time
            now = datetime.now()
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                return True, "Pre-market (before 9:15)"
            elif now.hour >= 9 and now.minute >= 30:
                # It's past 9:30 and still no today's data — likely a holiday
                return False, f"No data after market hours — probable holiday"
        # No data at all — assume pre-market if early
        now = datetime.now()
        if now.hour < 9 or (now.hour == 9 and now.minute < 20):
            return True, "Pre-market (assuming open)"
        return False, "No market data — probable holiday"
    except Exception as e:
        logger.warning(f"  Market open check failed: {e} — assuming open")
        return True, "API check failed, assuming open"


# ════════════════════════════════════════════════════════════
# MULTI-SOURCE DATA
# ════════════════════════════════════════════════════════════

def fetch_nse_enrichment(symbols, target_date=None):
    """Fetch delivery %, VWAP, trade count from NSE via jugaad_data."""
    enrichment = {}
    try:
        from jugaad_data.nse import stock_df
        d = target_date or date.today()
        start = d - timedelta(days=5)
        for sym in symbols:
            try:
                df = stock_df(symbol=sym, from_date=start, to_date=d, series="EQ")
                if len(df) > 0:
                    latest = df.iloc[0]  # most recent
                    enrichment[sym] = {
                        "nse_vwap": float(latest.get("VWAP", 0)),
                        "delivery_pct": float(latest.get("DELIVERY %", 0)),
                        "no_of_trades": int(latest.get("NO OF TRADES", 0)),
                        "nse_volume": int(latest.get("VOLUME", 0)),
                    }
            except:
                continue
    except ImportError:
        logger.info("  jugaad_data not available, skipping NSE enrichment")
    return enrichment


def fetch_intraday_candles(symbol, target_date=None, broker=None):
    """
    Fetch today's 5-min candles using multi-source provider.
    
    IMPORTANT: Angel One is used for ORDER EXECUTION, not data fetching.
    This avoids 429 rate limit errors.
    
    Data Source Priority:
    1. Yahoo Finance - Primary (free, reliable)
    2. Multi-source provider - With automatic failover
    3. Angel One - Disabled for data (rate limits)
    """
    # For backtest dates, use yfinance directly
    if target_date and target_date != date.today():
        try:
            import yfinance as yf
            start = target_date - timedelta(days=3)
            end = target_date + timedelta(days=2)
            df = yf.Ticker(f"{symbol}.NS").history(start=start.isoformat(), end=end.isoformat(), interval="5m")
            if df.empty:
                return None
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "datetime" not in df.columns and "date" in df.columns:
                df.rename(columns={"date": "datetime"}, inplace=True)
            df["symbol"] = symbol
            times = pd.to_datetime(df["datetime"])
            df = df[times.dt.date == target_date].reset_index(drop=True)
            return df if len(df) > 0 else None
        except Exception as e:
            logger.warning(f"  {symbol}: backtest candle fetch failed — {e}")
            return None
    
    # For live trading, use multi-source provider (no Angel One for data)
    try:
        from data.live_data_provider import get_data_provider
        provider = get_data_provider()  # Don't pass broker - avoid Angel One data calls
        df = provider.get_intraday_candles(symbol)
        if df is not None and len(df) > 0:
            return df
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"  {symbol}: live provider failed — {e}")
    
    # ── Fallback: yfinance directly ──
    try:
        import yfinance as yf
        df = yf.Ticker(f"{symbol}.NS").history(period="5d", interval="5m")
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        df["symbol"] = symbol
        today = date.today()
        times = pd.to_datetime(df["datetime"])
        df = df[times.dt.date == today].reset_index(drop=True)
        return df if len(df) > 0 else None
    except Exception as e:
        logger.warning(f"  {symbol}: candle fetch failed — {e}")
        return None


def fetch_vix(target_date=None):
    """Fetch India VIX. For backtest dates, fetch historical VIX for that date."""
    try:
        import yfinance as yf
        if target_date and target_date != date.today():
            start = target_date - timedelta(days=5)
            end = target_date + timedelta(days=1)
            data = yf.Ticker("^INDIAVIX").history(start=start.isoformat(), end=end.isoformat())
            if len(data) > 0:
                # Get the closest date <= target_date
                data.index = pd.to_datetime(data.index)
                valid = data[data.index.date <= target_date]
                if len(valid) > 0:
                    return round(float(valid.iloc[-1]["Close"]), 2)
        else:
            data = yf.Ticker("^INDIAVIX").history(period="5d")
            if len(data) > 0:
                return round(float(data.iloc[-1]["Close"]), 2)
    except Exception as e:
        logger.warning(f"  VIX fetch failed: {e} — using conservative default 20.0")
        return 20.0  # Conservative default on error (higher = more cautious)
    return 15.0  # Normal default when no data


def detect_market_trend(vix, target_date=None):
    """
    Detect overall market trend at market open to dynamically enable LONG/SHORT.
    
    Analysis factors:
    1. Nifty 50 trend (5-day momentum)
    2. VIX level (fear indicator)
    3. Gap up/down from previous close
    4. Global market cues (SGX Nifty proxy)
    
    Returns: dict with trend info and trading directions
    """
    global MARKET_TREND, ENABLE_LONGS, ENABLE_SHORTS, MAX_LONGS, MAX_SHORTS
    
    result = {
        "trend": "NEUTRAL",
        "confidence": 50,
        "enable_longs": True,
        "enable_shorts": True,
        "max_longs": 5,
        "max_shorts": 5,
        "reason": "",
    }
    
    try:
        import yfinance as yf
        
        # Fetch Nifty 50 data
        if target_date and target_date != date.today():
            start = target_date - timedelta(days=10)
            end = target_date + timedelta(days=1)
            nifty = yf.Ticker("^NSEI").history(start=start.isoformat(), end=end.isoformat())
        else:
            nifty = yf.Ticker("^NSEI").history(period="10d")
        
        if nifty.empty or len(nifty) < 3:
            logger.warning("  Could not fetch Nifty data for trend detection")
            return result
        
        # Calculate trend indicators
        nifty = nifty.reset_index()
        latest_close = float(nifty.iloc[-1]["Close"])
        prev_close = float(nifty.iloc[-2]["Close"])
        week_ago_close = float(nifty.iloc[0]["Close"])
        
        # Daily change
        daily_change_pct = (latest_close - prev_close) / prev_close * 100
        
        # Weekly momentum
        weekly_change_pct = (latest_close - week_ago_close) / week_ago_close * 100
        
        # Simple moving averages
        sma_5 = nifty["Close"].tail(5).mean()
        sma_10 = nifty["Close"].mean()
        
        # Trend scoring (-100 to +100)
        trend_score = 0
        reasons = []
        
        # Factor 1: Weekly momentum (-30 to +30)
        if weekly_change_pct > 2:
            trend_score += 30
            reasons.append(f"Strong weekly up +{weekly_change_pct:.1f}%")
        elif weekly_change_pct > 0.5:
            trend_score += 15
            reasons.append(f"Weekly up +{weekly_change_pct:.1f}%")
        elif weekly_change_pct < -2:
            trend_score -= 30
            reasons.append(f"Strong weekly down {weekly_change_pct:.1f}%")
        elif weekly_change_pct < -0.5:
            trend_score -= 15
            reasons.append(f"Weekly down {weekly_change_pct:.1f}%")
        
        # Factor 2: Daily gap (-20 to +20)
        if daily_change_pct > 1:
            trend_score += 20
            reasons.append(f"Gap up +{daily_change_pct:.1f}%")
        elif daily_change_pct > 0.3:
            trend_score += 10
            reasons.append(f"Slight up +{daily_change_pct:.1f}%")
        elif daily_change_pct < -1:
            trend_score -= 20
            reasons.append(f"Gap down {daily_change_pct:.1f}%")
        elif daily_change_pct < -0.3:
            trend_score -= 10
            reasons.append(f"Slight down {daily_change_pct:.1f}%")
        
        # Factor 3: Price vs SMAs (-20 to +20)
        if latest_close > sma_5 > sma_10:
            trend_score += 20
            reasons.append("Above both SMAs (bullish)")
        elif latest_close > sma_5:
            trend_score += 10
            reasons.append("Above 5-day SMA")
        elif latest_close < sma_5 < sma_10:
            trend_score -= 20
            reasons.append("Below both SMAs (bearish)")
        elif latest_close < sma_5:
            trend_score -= 10
            reasons.append("Below 5-day SMA")
        
        # Factor 4: VIX level (-30 to +10)
        if vix > 25:
            trend_score -= 30
            reasons.append(f"High VIX {vix:.1f} (fear)")
        elif vix > 20:
            trend_score -= 15
            reasons.append(f"Elevated VIX {vix:.1f}")
        elif vix < 13:
            trend_score += 10
            reasons.append(f"Low VIX {vix:.1f} (complacent)")
        
        # Determine trend based on score
        # LOSS MINIMIZATION: LONGs only in STRONG BULLISH (38% WR otherwise)
        if trend_score >= 40:  # Very strong bull - rare
            result["trend"] = "STRONG_BULLISH"
            result["confidence"] = min(95, 70 + abs(trend_score))
            result["enable_longs"] = True   # Only enable LONGs in VERY strong bull
            result["enable_shorts"] = True
            result["max_longs"] = 3         # Still limit LONGs
            result["max_shorts"] = 5
        elif trend_score >= 20:
            result["trend"] = "BULLISH"
            result["confidence"] = min(85, 60 + abs(trend_score))
            result["enable_longs"] = True   # Allow limited LONGs
            result["enable_shorts"] = True
            result["max_longs"] = 2
            result["max_shorts"] = 6
        elif trend_score <= -30:
            result["trend"] = "BEARISH"
            result["confidence"] = min(90, 60 + abs(trend_score))
            result["enable_longs"] = False  # NO LONGs in bear market
            result["enable_shorts"] = True
            result["max_longs"] = 0
            result["max_shorts"] = 10
        elif trend_score <= -10:
            result["trend"] = "MILD_BEARISH"
            result["confidence"] = 55 + abs(trend_score) // 2
            result["enable_longs"] = False  # NO LONGs in mild bear
            result["enable_shorts"] = True
            result["max_longs"] = 0
            result["max_shorts"] = 8
        else:
            # ══════════════════════════════════════════════════════════
            # NEUTRAL market: Trade both directions but be selective
            # KEY INSIGHT: Today was +1.2% bullish but detected as neutral
            # Solution: Allow both but with smaller positions
            # ══════════════════════════════════════════════════════════
            result["trend"] = "NEUTRAL"
            result["confidence"] = 50
            result["enable_longs"] = True   # Allow LONGs in neutral
            result["enable_shorts"] = True  # Allow SHORTs in neutral
            result["max_longs"] = 3         # Limited
            result["max_shorts"] = 3        # Limited
        
        result["reason"] = " | ".join(reasons)
        result["nifty_close"] = latest_close
        result["trend_score"] = trend_score
        
        # Update globals
        MARKET_TREND = result["trend"]
        ENABLE_LONGS = result["enable_longs"]
        ENABLE_SHORTS = result["enable_shorts"]
        MAX_LONGS = result["max_longs"]
        MAX_SHORTS = result["max_shorts"]
        
    except Exception as e:
        logger.warning(f"  Market trend detection error: {e}")
    
    return result


# ════════════════════════════════════════════════════════════
# REAL-TIME MARKET BIAS DETECTION (Proper Algo Trading)
# ════════════════════════════════════════════════════════════
# 
# Key insight from April 2 backtest:
# - Early trend detection (30 min) FAILED: -Rs 5,575
# - Proper algo (position in range): +Rs 157,298
#
# The trick: Don't predict, REACT to where price IS in the day's range

def get_realtime_market_bias(nifty_candles, current_idx):
    """
    Get market bias based on WHERE price is in day's range.
    This adapts in real-time and catches V-reversals.
    
    Returns: (bias, confidence)
        bias: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
        confidence: 0-100
    """
    if current_idx < 6:
        return 0, 0
    
    open_price = nifty_candles['open'].iloc[0]
    current = nifty_candles['close'].iloc[current_idx]
    day_low = nifty_candles['low'].iloc[:current_idx+1].min()
    day_high = nifty_candles['high'].iloc[:current_idx+1].max()
    
    # Position in day's range (0 = at low, 1 = at high)
    if day_high != day_low:
        position_in_range = (current - day_low) / (day_high - day_low)
    else:
        position_in_range = 0.5
    
    # Recovery from low (key for V-reversals)
    recovery = (current - day_low) / day_low * 100 if day_low > 0 else 0
    
    # Determine bias
    bias = 0
    confidence = 50
    
    # LONG bias: Price in upper 60% of range AND recovering
    if position_in_range > 0.6 and recovery > 0.3:
        bias = 1
        confidence = min(85, 50 + position_in_range * 35)
    # SHORT bias: Price in lower 40% AND making new lows
    elif position_in_range < 0.4:
        # Check if still falling
        recent_low = nifty_candles['low'].iloc[max(0, current_idx-3):current_idx+1].min()
        if recent_low <= day_low * 1.001:  # Near day's low
            bias = -1
            confidence = min(85, 50 + (1 - position_in_range) * 35)
    
    return bias, confidence


# Global sector data (updated at market open)
SECTOR_PERFORMANCE = {}
TOP_SECTORS = []
WEAK_SECTORS = []
VIX_LEVEL = 15.0  # Default, updated at open

def detect_sector_rotation():
    """
    Analyze sector performance to identify leading/lagging sectors.
    Run this at 9:30 AM after first 15 min of trading.
    
    Returns dict with sector rankings and recommended stocks per sector.
    """
    global SECTOR_PERFORMANCE, TOP_SECTORS, WEAK_SECTORS, VIX_LEVEL
    
    import yfinance as yf
    
    sectors = {
        'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
        'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
        'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN'],
        'Auto': ['MARUTI', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'TVSMOTOR'],
        'Metal': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'JINDALSTEL'],
        'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
        'Finance': ['BAJFINANCE', 'BAJAJFINSV', 'CHOLAFIN', 'MUTHOOTFIN', 'PFC'],
        'Energy': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL']
    }
    
    result = {"sectors": {}, "top_sectors": [], "weak_sectors": [], "vix": 15.0}
    
    try:
        # Get VIX
        try:
            vix_data = yf.Ticker("^INDIAVIX").history(period="2d")
            if len(vix_data) > 0:
                VIX_LEVEL = float(vix_data["Close"].iloc[-1])
                result["vix"] = VIX_LEVEL
                logger.info(f"  VIX Level: {VIX_LEVEL:.2f}")
        except:
            VIX_LEVEL = 15.0
        
        # Calculate sector performance
        sector_changes = {}
        
        for sector, stocks in sectors.items():
            changes = []
            for sym in stocks[:3]:  # Top 3 per sector for speed
                try:
                    df = yf.download(f"{sym}.NS", period="2d", interval="1d", progress=False)
                    if len(df) >= 2:
                        df = df.reset_index()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [col[0].lower() for col in df.columns]
                        change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
                        changes.append(change)
                except:
                    pass
            
            if changes:
                avg_change = np.mean(changes)
                sector_changes[sector] = avg_change
                result["sectors"][sector] = {
                    "change": avg_change,
                    "stocks": stocks,
                    "status": "STRONG" if avg_change > 0.5 else ("WEAK" if avg_change < -0.5 else "NEUTRAL")
                }
        
        # Sort sectors by performance
        sorted_sectors = sorted(sector_changes.items(), key=lambda x: x[1], reverse=True)
        
        # Top 2 sectors (LONG candidates)
        TOP_SECTORS = [s[0] for s in sorted_sectors[:2]]
        result["top_sectors"] = TOP_SECTORS
        
        # Bottom 2 sectors (SHORT candidates or avoid)
        WEAK_SECTORS = [s[0] for s in sorted_sectors[-2:]]
        result["weak_sectors"] = WEAK_SECTORS
        
        SECTOR_PERFORMANCE = sector_changes
        
        logger.info(f"  Top Sectors: {TOP_SECTORS}")
        logger.info(f"  Weak Sectors: {WEAK_SECTORS}")
        
    except Exception as e:
        logger.warning(f"  Sector rotation detection error: {e}")
    
    return result


def get_vix_adjusted_params():
    """
    Return target and stop percentages based on VIX level.
    Higher VIX = wider targets and stops.
    """
    global VIX_LEVEL
    
    if VIX_LEVEL < 15:
        return {"target_pct": 0.012, "stop_pct": 0.008, "label": "LOW_VIX"}
    elif VIX_LEVEL < 20:
        return {"target_pct": 0.015, "stop_pct": 0.010, "label": "NORMAL_VIX"}
    elif VIX_LEVEL < 25:
        return {"target_pct": 0.018, "stop_pct": 0.012, "label": "ELEVATED_VIX"}
    else:
        return {"target_pct": 0.020, "stop_pct": 0.015, "label": "HIGH_VIX"}


def is_stock_in_strong_sector(symbol):
    """Check if a stock is in a top-performing sector."""
    global TOP_SECTORS, WEAK_SECTORS
    
    sector = STOCK_SECTORS.get(symbol, "Unknown")
    
    if sector in TOP_SECTORS:
        return 1  # Strong sector bonus
    elif sector in WEAK_SECTORS:
        return -1  # Weak sector penalty
    return 0  # Neutral


# ════════════════════════════════════════════════════════════
# DYNAMIC STOCK SELECTION — no hardcoded list
# ════════════════════════════════════════════════════════════

def select_best_stocks(universe_symbols, config, target_date=None, top_n=8):
    """
    Scan universe and pick the best stocks to trade based on:
    1. Data availability (must have enough history)
    2. Liquidity (avg volume > 1M)
    3. ML score (direction confidence)
    4. Volatility sweet spot (ATR 1-4% — not too flat, not too wild)
    5. Delivery % (higher = institutional interest, from NSE data)
    6. Sector diversification (max 2 per sector)
    """
    logger.info(f"\n  Scanning {len(universe_symbols)} stocks for best candidates...")

    # Step 1: Load daily data + features
    loader = DataLoader()
    target = target_date.isoformat() if target_date else date.today().isoformat()
    df = loader.load_backtest_data(universe_symbols, target_date=target)

    featured = []
    for sym in (df["symbol"].unique() if not df.empty else []):
        sdf = df[df["symbol"] == sym].copy()
        if len(sdf) > 50:
            feat_df = add_features(sdf)
            featured.append(feat_df)

    if not featured:
        logger.warning("  No data available for scoring")
        return pd.DataFrame()

    all_feat = pd.concat(featured, ignore_index=True)

    # Step 2: ML scoring — try V2 model first, fallback to V1
    model, feats = None, None
    
    # Try V2 model first (better for bear markets)
    mp_v2 = Path("models/stock_predictor_v2.pkl")
    mp_v1 = Path("models/stock_predictor.pkl")
    
    if mp_v2.exists():
        try:
            with open(mp_v2, "rb") as f:
                d = pickle.load(f)
            model, feats = d["model"], d["features"]
            
            # Validate model has required method
            if not hasattr(model, "predict_proba"):
                logger.error("  V2 model doesn't support predict_proba — using scoring without ML")
                model, feats = None, None
            else:
                # Check model age
                train_date = d.get("train_date", d.get("trained_on", "unknown"))
                logger.info(f"  Using V2 model (trained: {train_date}, {d.get('trained_on', '?')} samples)")
        except Exception as e:
            logger.error(f"  Failed to load V2 model: {e}")
            model, feats = None, None
    elif mp_v1.exists():
        try:
            with open(mp_v1, "rb") as f:
                d = pickle.load(f)
            model, feats = d["model"], d["features"]
            if not hasattr(model, "predict_proba"):
                logger.error("  V1 model doesn't support predict_proba")
                model, feats = None, None
            else:
                logger.info("  Using V1 model (consider training V2 for better results)")
        except Exception as e:
            logger.error(f"  Failed to load V1 model: {e}")
            model, feats = None, None

    # Use V2 scoring if available
    if USE_V2_MODEL:
        avail = [c for c in (feats or FEATURE_COLS_V2) if c in all_feat.columns]
        scores = score_stocks_v2(all_feat, model, avail if model else None)
    else:
        avail = [c for c in (feats or FEATURE_COLS) if c in all_feat.columns]
        scores = score_stocks(all_feat, model, avail if model else None)

    # Step 3: Compute tradability metrics per stock
    candidates = []
    nse_data = fetch_nse_enrichment(scores["symbol"].tolist(), target_date)

    for _, row in scores.iterrows():
        sym = row["symbol"]
        
        # BACKTEST FILTER: Skip blacklisted stocks (consistent losers)
        if sym in STOCK_BLACKLIST:
            continue
        
        sdf = all_feat[all_feat["symbol"] == sym]
        if len(sdf) < 50:
            continue

        last = sdf.iloc[-1]
        avg_vol = sdf["volume"].tail(20).mean() if "volume" in sdf.columns else 0
        atr_pct = last.get("atr_pct", 0.02)
        rsi = row.get("rsi", 50)

        # Liquidity filter
        if avg_vol < 500000:
            continue

        # Volatility sweet spot: 0.8% - 4.5% (slightly wider for more options)
        if atr_pct < 0.008 or atr_pct > 0.045:
            continue

        # Composite score
        ml_score = row.get("score", 50)
        ml_prob = row.get("prob_up", 0.5)
        direction = row.get("direction", "LONG")
        
        # BACKTEST FILTER: Minimum ML confidence (abs(prob - 0.5) >= 0.12)
        ml_confidence = abs(ml_prob - 0.5) if ml_prob else abs((ml_score - 50) / 100)
        if ml_confidence < MIN_ML_CONFIDENCE:
            continue
        
        # DYNAMIC MARKET TREND: Filter based on today's market direction
        # Skip LONG signals in bearish market (unless high confidence)
        if MARKET_TREND in ("BEARISH", "MILD_BEARISH") and direction == "LONG" and ml_confidence < 0.2:
            continue
        # Skip SHORT signals in bullish market (unless high confidence)
        if MARKET_TREND in ("BULLISH", "MILD_BULLISH") and direction == "SHORT" and ml_confidence < 0.2:
            continue
        # Check if direction is enabled
        if direction == "LONG" and not ENABLE_LONGS:
            continue
        if direction == "SHORT" and not ENABLE_SHORTS:
            continue
        
        # RSI filter to avoid extreme values (oversold shorts, overbought longs)
        if direction == "LONG" and rsi > 75:
            continue
        if direction == "SHORT" and rsi < 25:
            continue

        # NSE enrichment bonus
        nse = nse_data.get(sym, {})
        delivery_bonus = min(10, nse.get("delivery_pct", 0) / 5)  # up to 10 pts for high delivery
        trade_count_bonus = min(5, nse.get("no_of_trades", 0) / 100000)  # up to 5 pts

        # Volatility fitness: prefer mid-range ATR
        vol_fitness = 10 - abs(atr_pct - 0.018) * 500  # peak at 1.8% ATR
        
        # DYNAMIC: Apply rolling performance score
        # Preferred stocks get bonus, scores adjust confidence
        if sym in STOCK_WHITELIST:
            perf_bonus = 15
        elif sym in STOCK_SCORES:
            # Score 0-100, center at 50 -> bonus/penalty of -10 to +10
            perf_bonus = (STOCK_SCORES[sym] - 50) / 5
        else:
            perf_bonus = 0

        composite = ml_score + delivery_bonus + trade_count_bonus + vol_fitness + perf_bonus
        sector = STOCK_SECTORS.get(sym, "Other")

        candidates.append({
            "symbol": sym, "ml_score": ml_score, "direction": direction,
            "rsi": rsi, "atr_pct": round(atr_pct, 4), "avg_volume": int(avg_vol),
            "delivery_pct": nse.get("delivery_pct", 0),
            "composite_score": round(composite, 1), "sector": sector,
        })

    if not candidates:
        return pd.DataFrame()

    cdf = pd.DataFrame(candidates).sort_values("composite_score", ascending=False)

    # Step 4: Sector diversification — max 2 per sector + direction limits
    selected = []
    sector_count = {}
    long_count = 0
    short_count = 0
    
    for _, row in cdf.iterrows():
        sec = row["sector"]
        direction = row.get("direction", "LONG")
        
        # Sector limit
        if sector_count.get(sec, 0) >= 2:
            continue
        
        # DYNAMIC: Limit trades based on market trend
        if direction == "LONG" and long_count >= MAX_LONGS:
            continue
        if direction == "SHORT" and short_count >= MAX_SHORTS:
            continue
        
        selected.append(row)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if direction == "LONG":
            long_count += 1
        else:
            short_count += 1
            
        if len(selected) >= top_n:
            break

    result = pd.DataFrame(selected)
    logger.info(f"  Selected {len(result)} stocks from {len(candidates)} candidates")
    return result


# ════════════════════════════════════════════════════════════
# REGIME DETECTION — per stock, per hour
# ════════════════════════════════════════════════════════════

class RegimeDetector:
    """Detect market regime from candle data."""

    @staticmethod
    def detect(candles, idx, lookback=12):
        """Returns: 'trending_up', 'trending_down', 'ranging', 'volatile', 'choppy'"""
        if idx < lookback:
            return "unknown"

        recent = candles.iloc[max(0, idx - lookback):idx + 1]
        closes = recent["close"].values
        highs = recent["high"].values
        lows = recent["low"].values

        # Price change over lookback
        pct_change = (closes[-1] - closes[0]) / closes[0]

        # Average candle range as % of price
        avg_range = ((highs - lows) / closes).mean()

        # Directional consistency: how many candles move in same direction
        diffs = np.diff(closes)
        up_count = np.sum(diffs > 0)
        down_count = np.sum(diffs < 0)
        total = len(diffs)
        consistency = max(up_count, down_count) / total if total > 0 else 0.5

        # High volatility
        if avg_range > 0.008:  # >0.8% per candle
            if consistency > 0.65:
                return "trending_up" if pct_change > 0 else "trending_down"
            return "volatile"

        # Trending
        if abs(pct_change) > 0.005 and consistency > 0.6:
            return "trending_up" if pct_change > 0 else "trending_down"

        # Ranging
        if abs(pct_change) < 0.003 and avg_range < 0.005:
            return "ranging"

        # Choppy (back and forth with no clear direction)
        return "choppy"


# ════════════════════════════════════════════════════════════
# ADAPTIVE V3 TRADER — switches strategy per regime
# ════════════════════════════════════════════════════════════

class AdaptiveV3Trader:
    """
    V3 trader: adaptive strategy based on regime detection.
    
    Features:
    - Uses REAL-TIME market bias (not ML prediction)
    - Direction from price position in day's range
    - Dynamic stop loss management
    - Regime-based strategy switching
    """

    def __init__(self, symbol, direction, config, ml_score=50):
        self.symbol = symbol
        self.direction = direction  # Now overridden by real-time bias in _scan_for_entry
        self.ml_score = ml_score  # Used for stock SELECTION only, not direction
        self.capital = config["capital"]["total"]
        self.config = config
        self.cost_model = AngelOneCostModel()
        self.strategy = ProStrategyV2(config.get("strategies", {}).get("pro", {}))
        self.regime_detector = RegimeDetector()
        self.pattern_detector = CandlePatternDetector()  # NEW: Candlestick patterns

        # State
        self.orb = None
        self.in_trade = False
        self.entry_price = 0
        self.entry_time = None
        self.entry_idx = 0
        self.side = None
        self.sl = 0
        self.tgt = 0
        self.shares = 0
        self.sl_moved_to_be = False
        self.partial_exited = False
        self.trade_type = ""
        self.current_regime = "unknown"
        self.entry_regime = "unknown"  # regime at time of entry
        self.pattern_score = 0  # NEW: Track pattern confirmation

        self.trades = []
        self.trade_count = 0
        self.max_trades = config["capital"].get("max_trades_per_day", 3)
        self.cooldown_until = 0
        self.day_pnl = 0.0
        self.daily_loss_limit = config["capital"].get("daily_loss_limit", 0.03) * self.capital

        self.processed_candles = 0

    def _close_trade(self, exit_price, exit_time, reason, shares_to_close=None):
        shares = shares_to_close or self.shares
        if self.side == "SHORT":
            gross = (self.entry_price - exit_price) * shares
        else:
            gross = (exit_price - self.entry_price) * shares
        costs = self.cost_model.calculate(self.entry_price * shares, exit_price * shares).total
        net = gross - costs

        self.trades.append({
            "symbol": self.symbol, "direction": self.side,
            "type": self.trade_type, "regime": self.entry_regime,
            "entry": round(self.entry_price, 2), "exit": round(exit_price, 2),
            "entry_time": str(self.entry_time).split("+")[0],
            "exit_time": str(exit_time).split("+")[0],
            "sl": round(self.sl, 2), "tgt": round(self.tgt, 2),
            "qty": shares, "gross": round(gross, 2),
            "costs": round(costs, 2), "net_pnl": round(net, 2),
            "reason": reason,
        })

        self.day_pnl += net

        emoji = "✅" if net > 0 else "❌"
        et_s = str(self.entry_time).split(" ")[-1].split("+")[0][:8]  # HH:MM:SS
        xt_s = str(exit_time).split(" ")[-1].split("+")[0][:8]
        action_entry = "BOUGHT" if self.side == "LONG" else "SOLD SHORT"
        action_exit = "SOLD" if self.side == "LONG" else "COVERED"
        
        # Calculate holding time
        try:
            entry_dt = pd.to_datetime(self.entry_time)
            exit_dt = pd.to_datetime(exit_time)
            hold_mins = int((exit_dt - entry_dt).total_seconds() / 60)
            hold_str = f"{hold_mins} min"
        except:
            hold_str = "N/A"
        
        # Calculate return percentage
        return_pct = (net / (self.entry_price * shares)) * 100 if self.entry_price > 0 else 0
        
        msg = (
            f"{emoji} TRADE CLOSED — {self.symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📥 {action_entry}: {et_s} @ ₹{self.entry_price:,.2f}\n"
            f"📤 {action_exit}: {xt_s} @ ₹{exit_price:,.2f}\n"
            f"⏱️ Held for: {hold_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Qty: {shares} | Return: {return_pct:+.2f}%\n"
            f"💰 P&L: ₹{net:+,.2f} | Day: ₹{self.day_pnl:+,.2f}\n"
            f"🎯 Strategy: {self.trade_type}\n"
            f"📈 Regime: {self.current_regime} | Exit: {reason}"
        )
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)

        if shares_to_close and shares_to_close < self.shares:
            self.shares -= shares_to_close
            self.partial_exited = True
        else:
            self.in_trade = False
            self.trade_count += 1
            self.sl_moved_to_be = False
            self.partial_exited = False
            # ══════════════════════════════════════════════════════════
            # LOSS MINIMIZATION: Cooldown after exit (prevent churning)
            # Multiple trades per stock was causing -₹1,187 loss on HINDUNILVR
            # ══════════════════════════════════════════════════════════
            if not hasattr(self, 'cooldown_until') or self.cooldown_until == 0:
                # Default cooldown: 4 candles (20 min) after any exit
                # This will be overridden by specific exit reasons
                pass  # Cooldown set by exit reason handlers

    def _enter_trade(self, signal, idx):
        # ── CANDLE PATTERN CONFIRMATION ──
        # Check if candlestick patterns confirm the direction
        try:
            candles = signal.get("candles")
            if candles is not None and idx > 0:
                confirmed, boost = self.pattern_detector.confirm_direction(candles, idx, signal["side"])
                self.pattern_score = self.pattern_detector.analyze(candles, idx)
                
                if not confirmed:
                    # Patterns contradict - skip trade
                    logger.info(f"  {self.symbol} SKIP: Candle patterns contradict {signal['side']} (score: {self.pattern_score})")
                    return
                
                # Add pattern info to signal
                if boost > 0:
                    signal["reason"] = f"{signal.get('reason', '')} | Pattern boost +{boost}"
        except Exception as e:
            logger.debug(f"Pattern check error: {e}")
            self.pattern_score = 0
        
        self.entry_price = signal["entry"]
        self.entry_time = signal["time"]
        self.entry_idx = idx
        self.side = signal["side"]
        self.sl = signal["sl"]
        self.tgt = signal["tgt"]
        self.trade_type = signal["type"]
        risk = signal["risk"]

        # ── FIX 1: Minimum SL distance = 0.5% of price ──
        # Prevents qty explosion when ATR is tiny
        min_risk = self.entry_price * 0.005  # 0.5% minimum
        if risk < min_risk:
            # Widen SL to minimum distance
            if self.side == "LONG":
                self.sl = self.entry_price - min_risk
            else:
                self.sl = self.entry_price + min_risk
            risk = min_risk
            # Also adjust target to maintain R:R ratio
            original_rr = abs(signal["tgt"] - signal["entry"]) / max(signal["risk"], 0.01)
            if self.side == "LONG":
                self.tgt = self.entry_price + risk * max(original_rr, 1.5)
            else:
                self.tgt = self.entry_price - risk * max(original_rr, 1.5)

        # ── FIX 3: Regime-based position sizing ──
        risk_pct = self.strategy.max_risk_pct
        if self.current_regime == "volatile":
            risk_pct *= 0.5
        elif self.current_regime == "choppy":
            risk_pct *= 0.5  # was 0.6, more conservative now
        elif self.current_regime == "ranging":
            risk_pct *= 0.4  # heavily reduce in ranging
        # Afternoon trades get 60% position (riskier time)
        if "AFTERNOON" in self.trade_type:
            risk_pct *= 0.6

        # ── FIX 1b: Cap max shares to limit single-trade damage ──
        self.shares = max(1, int(self.capital * risk_pct / max(risk, min_risk)))
        max_shares = int(self.capital * 0.15 / self.entry_price)  # max 15% capital per trade
        self.shares = min(self.shares, max_shares)

        # ── FIX 5: Minimum expected profit filter — skip if profit won't cover costs ──
        expected_gross = abs(self.tgt - self.entry_price) * self.shares
        est_costs = self.cost_model.calculate(
            self.entry_price * self.shares, self.tgt * self.shares
        ).total
        if expected_gross < est_costs * 2:
            logger.info(
                f"  ⛔ {self.symbol} SKIP: expected gross Rs {expected_gross:.0f} "
                f"< 2× costs Rs {est_costs:.0f}"
            )
            return

        self.in_trade = True
        self.sl_moved_to_be = False
        self.partial_exited = False
        self.entry_regime = self.current_regime  # snapshot regime at entry

        side_emoji = "📈" if self.side == "LONG" else "📉"
        action = "BUYING" if self.side == "LONG" else "SELLING SHORT"
        entry_time_str = signal['time'].strftime('%H:%M:%S') if hasattr(signal['time'], 'strftime') else str(signal['time'])
        
        # Calculate potential risk/reward
        risk_amt = abs(self.entry_price - self.sl) * self.shares
        reward_amt = abs(self.tgt - self.entry_price) * self.shares
        rr_ratio = reward_amt / risk_amt if risk_amt > 0 else 0
        
        msg = (
            f"{side_emoji} TRADE ENTRY — {self.symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 Time: {entry_time_str}\n"
            f"💵 {action} @ ₹{self.entry_price:,.2f}\n"
            f"🛑 Stop Loss: ₹{self.sl:,.2f}\n"
            f"🎯 Target: ₹{self.tgt:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Qty: {self.shares} | Risk: {risk_pct*100:.1f}%\n"
            f"💰 Risk: ₹{risk_amt:,.0f} | Reward: ₹{reward_amt:,.0f} (R:R {rr_ratio:.1f})\n"
            f"🎯 Strategy: {self.trade_type}\n"
            f"📈 Regime: {self.current_regime}\n"
            f"📝 Reason: {signal['reason']}"
        )
        logger.info(f"  {msg}")
        send_telegram(msg, self.config)

    def process_new_candles(self, candles):
        if candles is None or len(candles) == 0:
            return
        start = self.processed_candles
        for i in range(start, len(candles)):
            self._process_candle(i, candles)
        self.processed_candles = len(candles)

    def _process_candle(self, i, candles):
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        h, l, c = row["high"], row["low"], row["close"]
        hour, minute = t.hour, t.minute

        # Update regime every 6 candles (30 min)
        if i % 6 == 0 and i >= 12:
            self.current_regime = self.regime_detector.detect(candles, i)

        # ── Circuit breaker: stop if daily loss exceeded ──
        if self.day_pnl < -self.daily_loss_limit and not self.in_trade:
            return

        # ── Phase 1: ORB Formation ──
        if self.orb is None:
            if i < self.strategy.orb_candles:
                return
            self.orb = self.strategy.compute_orb(candles, self.strategy.orb_candles)
            if self.orb is None or self.orb["range"] < 0.5:
                return
            msg = f"📐 {self.symbol} ORB: H={self.orb['high']:,.2f} L={self.orb['low']:,.2f} R={self.orb['range']:,.2f} | Regime: {self.current_regime}"
            logger.info(f"  {msg}")

        # ── Phase 2: Square off ──
        if hour >= 15 and minute >= 10:
            if self.in_trade:
                self._close_trade(c, t, "SQUARE_OFF")
            return

        # ── Phase 3: Manage open trade ──
        if self.in_trade:
            self._manage_trade(candles, i)
            return

        # ── Phase 4: Look for entries based on regime ──
        if self.trade_count >= self.max_trades:
            return
        if i < self.cooldown_until:
            return

        self._scan_for_entry(candles, i)

    def _manage_trade(self, candles, i):
        """Manage open position with trailing, partial exit, time-decay."""
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        h, l, c = row["high"], row["low"], row["close"]

        # ══════════════════════════════════════════════════════════
        # EMERGENCY STOP LOSS — Hard 5% limit to protect against flash crashes
        # This runs regardless of DISABLE_STOP_LOSSES setting
        # ══════════════════════════════════════════════════════════
        loss_pct = 0
        if self.side == "LONG":
            loss_pct = (self.entry_price - c) / self.entry_price if c < self.entry_price else 0
        else:  # SHORT
            loss_pct = (c - self.entry_price) / self.entry_price if c > self.entry_price else 0
        
        if loss_pct >= EMERGENCY_STOP_LOSS_PCT:
            logger.warning(f"  ⚠️ {self.symbol} EMERGENCY STOP triggered at {loss_pct*100:.1f}% loss!")
            self._close_trade(c, t, f"EMERGENCY_STOP_{loss_pct*100:.0f}%")
            self.cooldown_until = i + 12  # 1 hour cooldown after emergency stop
            return

        # ══════════════════════════════════════════════════════════
        # SMART TIME MANAGEMENT (replacing TIME_DECAY which had 24% WR)
        # New logic: Trail winners, cut losers fast
        # ══════════════════════════════════════════════════════════
        holding_candles = i - self.entry_idx
        pct_from_entry = (c - self.entry_price) / self.entry_price if self.side == "LONG" else (self.entry_price - c) / self.entry_price
        
        # Phase 1 (0-15 min / 3 candles): Quick scalp check
        if holding_candles >= 3:
            # If we have 0.5%+ profit in first 15 min, consider taking it
            if pct_from_entry >= 0.005:  # 0.5% profit
                self._close_trade(c, t, "QUICK_PROFIT")
                self.cooldown_until = i + 6  # 30 min cooldown
                return
        
        # Phase 2 (15-25 min / 3-5 candles): Cut losers
        if 3 <= holding_candles < 5:
            if pct_from_entry < -0.003:  # Losing 0.3%+
                self._close_trade(c, t, "CUT_LOSER")
                self.cooldown_until = i + 8  # 40 min cooldown after loss
                return
        
        # Phase 3 (25-40 min / 5-8 candles): Stricter exit
        if 5 <= holding_candles < 8:
            if pct_from_entry < 0.002:  # Less than 0.2% profit
                self._close_trade(c, t, "TIME_DECAY")
                self.cooldown_until = i + 6  # 30 min cooldown
                return
        
        # Phase 4 (40+ min / 8+ candles): Force exit unless big winner
        if holding_candles >= 8:
            if pct_from_entry < 0.008:  # Less than 0.8% profit
                reason = "PROFIT_LOCK" if pct_from_entry > 0 else "TIME_DECAY"
                self._close_trade(c, t, reason)
                self.cooldown_until = i + 6
                return

        if self.side == "LONG":
            # BACKTEST INSIGHT: Stop losses were 100% losers
            # Only use stop loss if explicitly enabled, otherwise hold to target/time
            if not DISABLE_STOP_LOSSES:
                if l <= self.sl:
                    self._close_trade(self.sl, t, "STOP_LOSS")
                    self.cooldown_until = i + self.strategy.cooldown_candles
                    return
            
            if h >= self.tgt:
                self._close_trade(self.tgt, t, "TARGET")
                return
            
            # Partial exit at 1x risk
            if not self.partial_exited:
                initial_risk = self.entry_price - self.sl
                if initial_risk > 0 and (c - self.entry_price) >= initial_risk * self.strategy.partial_exit_at_rr:
                    partial = max(1, int(self.shares * self.strategy.partial_exit_pct))
                    if partial < self.shares:
                        self._close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial)
            
            # ══════════════════════════════════════════════════════════
            # TRAILING STOP: Lock in profits as price moves up
            # ══════════════════════════════════════════════════════════
            unrealised_pct = (c - self.entry_price) / self.entry_price
            
            # Stage 1: Move to breakeven after 0.5% profit
            if unrealised_pct >= 0.005 and not self.sl_moved_to_be:
                self.sl = self.entry_price + 0.10  # Just above entry
                self.sl_moved_to_be = True
                logger.info(f"  {self.symbol} SL -> breakeven Rs {self.sl:,.2f}")
            
            # Stage 2: Trail stop 0.2% behind price after 0.7% profit (tighter)
            if unrealised_pct >= 0.007:
                new_sl = c * 0.998  # 0.2% below current price (tighter than 0.3%)
                if new_sl > self.sl:
                    self.sl = new_sl
                    logger.info(f"  {self.symbol} Trailing SL -> Rs {self.sl:,.2f}")
        
        else:  # SHORT
            # BACKTEST INSIGHT: Stop losses were 100% losers
            if not DISABLE_STOP_LOSSES:
                if h >= self.sl:
                    self._close_trade(self.sl, t, "STOP_LOSS")
                    self.cooldown_until = i + self.strategy.cooldown_candles
                    return
            
            if l <= self.tgt:
                self._close_trade(self.tgt, t, "TARGET")
                return
            
            if not self.partial_exited:
                initial_risk = self.sl - self.entry_price
                if initial_risk > 0 and (self.entry_price - c) >= initial_risk * self.strategy.partial_exit_at_rr:
                    partial = max(1, int(self.shares * self.strategy.partial_exit_pct))
                    if partial < self.shares:
                        self._close_trade(c, t, "PARTIAL_EXIT", shares_to_close=partial)
            
            # ══════════════════════════════════════════════════════════
            # TRAILING STOP: Lock in profits as price moves down (SHORT)
            # ══════════════════════════════════════════════════════════
            unrealised_pct = (self.entry_price - c) / self.entry_price
            
            # Stage 1: Move to breakeven after 0.5% profit
            if unrealised_pct >= 0.005 and not self.sl_moved_to_be:
                self.sl = self.entry_price - 0.10  # Just below entry
                self.sl_moved_to_be = True
                logger.info(f"  {self.symbol} SL -> breakeven Rs {self.sl:,.2f}")
            
            # Stage 2: Trail stop 0.2% above price after 0.7% profit (tighter)
            if unrealised_pct >= 0.007:
                new_sl = c * 1.002  # 0.2% above current price (tighter than 0.3%)
                if new_sl < self.sl:
                    self.sl = new_sl
                    logger.info(f"  {self.symbol} Trailing SL -> Rs {self.sl:,.2f}")

    def _scan_for_entry(self, candles, i):
        """
        Scan for entry signals using PURE PRICE ACTION (no ML for direction).
        
        KEY INSIGHT FROM APRIL 2:
        - ML predicted SHORT at 9:45 AM (market down -0.89%)
        - Market reversed to +1.42% by close
        - ML direction was WRONG all day
        
        CORRECT APPROACH (Pure Price Action):
        - Direction = WHERE price IS in day's range (not ML prediction)
        - Upper 55% of range → LONG bias
        - Lower 45% of range → SHORT bias
        - This adapts EVERY CANDLE to reversals!
        """
        row = candles.iloc[i]
        t = pd.to_datetime(row["datetime"])
        c = row["close"]
        hour, minute = t.hour, t.minute

        regime = self.current_regime

        # ══════════════════════════════════════════════════════════════
        # BLOCK LOSING REGIMES: ranging and unknown always lose
        # ══════════════════════════════════════════════════════════════
        if regime in BLOCKED_REGIMES:
            return
        
        # ══════════════════════════════════════════════════════════════
        # DON'T TRADE FIRST HOUR (let market establish range)
        # ══════════════════════════════════════════════════════════════
        if hour < 5:  # Before ~10:15 AM IST (UTC+5:30)
            return
        
        # ══════════════════════════════════════════════════════════════
        # PURE PRICE ACTION: Get direction from CURRENT price position
        # This is the KEY fix - NO ML direction, only real-time bias!
        # ══════════════════════════════════════════════════════════════
        realtime_bias = 0
        direction = "NEUTRAL"
        
        try:
            import yfinance as yf
            nifty = yf.Ticker("^NSEI").history(period="1d", interval="5m")
            if len(nifty) > 0:
                nifty = nifty.reset_index()
                nifty.columns = [c.lower() for c in nifty.columns]
                realtime_bias, bias_conf = get_realtime_market_bias(nifty, len(nifty)-1)
                
                if realtime_bias == 1:
                    direction = "LONG"
                elif realtime_bias == -1:
                    direction = "SHORT"
                else:
                    return  # NEUTRAL = don't trade
                    
                # Need minimum confidence to trade
                if bias_conf < 55:
                    return
        except:
            return  # If can't get real-time data, don't trade (no guessing)
        
        # ══════════════════════════════════════════════════════════════
        # SECTOR ROTATION FILTER (April 2 insight: IT +1.89%, Pharma -0.56%)
        # ══════════════════════════════════════════════════════════════
        sector_score = is_stock_in_strong_sector(self.symbol)
        
        if direction == "LONG" and sector_score < 0:
            return  # Don't go long in weak sector
        if direction == "SHORT" and sector_score > 0:
            return  # Don't short in strong sector

        # ── FIX: Intraday range filter — skip if stock is dead today ──
        if i > 24:  # after 2 hours of data
            day_high = candles["high"].iloc[:i+1].max()
            day_low = candles["low"].iloc[:i+1].min()
            day_range_pct = (day_high - day_low) / day_low if day_low > 0 else 0
            if day_range_pct < 0.004:  # less than 0.4% range all day
                return  # dead market, skip

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 1: ORB BREAKOUT/BREAKDOWN (9:20-10:30)
        # - In BULLISH market + trending_up: ORB BREAKOUT (LONG)
        # - In BEARISH market + trending_down: ORB BREAKDOWN (SHORT)
        # ══════════════════════════════════════════════════════════════
        if hour < 10 or (hour == 10 and minute <= 30):
            if regime == "trending_down" and direction == "SHORT":
                signal = self.strategy.generate_orb_signal(candles, i, self.orb, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return
            elif regime == "trending_up" and direction == "LONG":
                signal = self.strategy.generate_orb_signal(candles, i, self.orb, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 2: MOMENTUM (11:30-14:00)
        # Trade in direction of market trend
        # ══════════════════════════════════════════════════════════════
        if (hour == 11 and minute >= 30) or (12 <= hour <= 13):
            # Allow momentum in both directions based on regime
            if regime == "trending_down" and direction == "SHORT":
                signal = self._generate_momentum_signal(candles, i, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return
            elif regime == "trending_up" and direction == "LONG":
                signal = self._generate_momentum_signal(candles, i, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return
            elif regime == "choppy":
                # In choppy, only trade with very strong signals
                signal = self._generate_momentum_signal(candles, i, direction, lookback=8)
                if signal and abs(signal.get("risk", 0)) > 0:
                    avg_vol = candles["volume"].iloc[max(0, i - 20):i].mean()
                    curr_vol = candles["volume"].iloc[i]
                    if curr_vol > avg_vol * 1.5:
                        self._enter_trade(signal, i)
                        return

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 3: AFTERNOON TREND (13:00-14:30)
        # Trade in direction of established intraday trend
        # ══════════════════════════════════════════════════════════════
        if 13 <= hour <= 14 and minute <= 30:
            if regime in ("trending_down", "choppy") and direction == "SHORT":
                signal = self._generate_afternoon_trend_signal(candles, i, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return
            elif regime in ("trending_up", "choppy") and direction == "LONG":
                signal = self._generate_afternoon_trend_signal(candles, i, direction)
                if signal:
                    self._enter_trade(signal, i)
                    return

    def _generate_momentum_signal(self, candles, i, direction, lookback=8):
        """Momentum continuation in real-time market direction."""
        if i < lookback + 3:
            return None
        row = candles.iloc[i]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])

        recent = candles.iloc[max(0, i - lookback):i + 1]
        price_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]

        # Volume confirmation
        avg_vol = candles["volume"].iloc[max(0, i - 20):i].mean()
        curr_vol = candles["volume"].iloc[i]
        vol_ok = avg_vol > 0 and curr_vol > avg_vol * 1.2

        if not vol_ok:
            return None

        # ══════════════════════════════════════════════════════════
        # VIX-ADJUSTED TARGETS (April 2 insight: VIX 25.52 = 2% targets)
        # ══════════════════════════════════════════════════════════
        vix_params = get_vix_adjusted_params()
        target_pct = vix_params["target_pct"]
        
        # MUST align with real-time market direction
        if direction == "LONG" and price_change > 0.005:
            atr = self._quick_atr(candles, i)
            sl = close - atr * 1.5
            risk = close - sl
            # VIX-adjusted target
            tgt = close * (1 + target_pct)
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "MOMENTUM_LONG",
                "reason": f"Momentum +{price_change*100:.1f}% with vol, VIX={VIX_LEVEL:.0f}",
                "candles": candles  # Pass candles for pattern check
            }
        elif direction == "SHORT" and price_change < -0.005:
            atr = self._quick_atr(candles, i)
            sl = close + atr * 1.5
            risk = sl - close
            # VIX-adjusted target (tighter in choppy)
            adj_target = target_pct * 0.8 if self.current_regime == "choppy" else target_pct
            tgt = close * (1 - adj_target)
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "MOMENTUM_SHORT",
                "reason": f"Momentum {price_change*100:.1f}% with vol, VIX={VIX_LEVEL:.0f}"
            }
        return None

    def _generate_vwap_direction_signal(self, candles, i, direction):
        """VWAP mean-reversion in real-time market direction."""
        row = candles.iloc[i]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])
        hour = t.hour

        if hour < 10 or hour >= 14:
            return None

        vwap, std = self.strategy.compute_vwap_proper(candles, i)
        if std == 0 or std < close * 0.003:
            return None

        rsi = self.strategy.compute_rsi(candles["close"].iloc[:i + 1], period=14)
        deviation = (close - vwap) / std

        # LONG only when market bias is LONG and price is below VWAP
        if direction == "LONG" and deviation < -1.5 and rsi < 35:
            sl = close - std * 1.5
            risk = close - sl
            if risk <= 0:
                return None
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": vwap,
                "risk": risk, "time": t, "type": "VWAP_LONG",
                "reason": f"VWAP oversold {deviation:.1f}σ RSI={rsi:.0f}, bias=LONG"
            }

        # SHORT only when market bias is SHORT and price is above VWAP
        if direction == "SHORT" and deviation > 1.5 and rsi > 65:
            sl = close + std * 1.5
            risk = sl - close
            if risk <= 0:
                return None
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": vwap,
                "risk": risk, "time": t, "type": "VWAP_SHORT",
                "reason": f"VWAP overbought +{deviation:.1f}σ RSI={rsi:.0f}, bias=SHORT"
            }
        return None

    def _generate_afternoon_trend_signal(self, candles, i, direction):
        """Afternoon entry — strong signals in real-time market direction."""
        if i < 30:
            return None
        row = candles.iloc[i]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])

        # Need very strong signal: RSI + VWAP + volume all aligned
        rsi = self.strategy.compute_rsi(candles["close"].iloc[:i + 1], period=7)
        vwap, std = self.strategy.compute_vwap_proper(candles, i)
        avg_vol = candles["volume"].iloc[max(0, i - 20):i].mean()
        curr_vol = candles["volume"].iloc[i]

        if avg_vol == 0 or curr_vol < avg_vol * 1.3:
            return None  # need strong volume

        # 5-candle trend alignment
        recent_close = candles["close"].iloc[max(0, i - 5):i + 1]
        trend_up = all(recent_close.diff().dropna() > 0)
        trend_down = all(recent_close.diff().dropna() < 0)

        if direction == "LONG" and trend_up and rsi > 55 and close > vwap:
            atr = self._quick_atr(candles, i)
            sl = close - atr * 1.2
            risk = close - sl
            tgt = close + risk * 1.5
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "AFTERNOON_TREND_LONG",
                "reason": f"5-candle uptrend + vol + above VWAP, RSI={rsi:.0f}"
            }

        if direction == "SHORT" and trend_down and rsi < 45 and close < vwap:
            atr = self._quick_atr(candles, i)
            sl = close + atr * 1.2
            risk = sl - close
            tgt = close - risk * 1.5
            return {
                "side": "SHORT", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "AFTERNOON_TREND_SHORT",
                "reason": f"5-candle downtrend + vol + below VWAP, RSI={rsi:.0f}"
            }
        return None

    def _quick_atr(self, candles, idx, period=14):
        """ATR with a floor of 0.5% of price to prevent micro-SL disasters."""
        if idx < period:
            raw = (candles["high"].iloc[:idx+1] - candles["low"].iloc[:idx+1]).mean()
        else:
            recent = candles.iloc[max(0, idx - period):idx + 1]
            raw = (recent["high"] - recent["low"]).mean()
        # Floor: ATR must be at least 0.5% of current price
        price = candles["close"].iloc[idx]
        return max(raw, price * 0.005)


# ════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════

def run(symbols=None, backtest_date=None):
    config = load_config()
    target_date = date.fromisoformat(backtest_date) if backtest_date else None
    is_backtest = target_date is not None
    today_str = str(target_date or date.today())

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  LIVE PAPER TRADER V3 — ADAPTIVE")
    logger.info(f"  Date: {today_str} {'(BACKTEST)' if is_backtest else '(LIVE)'}")
    logger.info(f"  Capital: Rs {config['capital']['total']:,}")
    logger.info(f"{'=' * 60}")

    # ── Holiday check (live API) ──
    if not is_backtest:
        is_open, reason = is_market_open_today()
        if not is_open:
            msg = f"📅 {today_str} — Market closed ({reason})"
            logger.info(msg)
            send_telegram(msg, config)
            return

    # ── Initialize Angel One broker (if configured) ──
    broker = None
    if not is_backtest and config.get("broker", {}).get("angel_one", {}).get("api_key"):
        try:
            from data.angel_broker import AngelBroker
            broker = AngelBroker(config)
            if broker.connect():
                logger.info(f"  🔗 Angel One connected | Mode: {broker.mode.upper()}")
                send_telegram(f"🔗 Angel One connected | Mode: {broker.mode.upper()}", config)
            else:
                broker = None
        except Exception as e:
            logger.warning(f"  Angel One init failed ({e}) — yfinance fallback")
            broker = None

    # ── VIX check ──
    vix = fetch_vix(target_date)

    # ── DYNAMIC MARKET TREND DETECTION ──
    trend_info = detect_market_trend(vix, target_date)
    trend_msg = (
        f"📊 MARKET TREND: {trend_info['trend']} ({trend_info['confidence']}% confidence)\n"
        f"  LONGS: {'ENABLED' if trend_info['enable_longs'] else 'DISABLED'} (max {trend_info['max_longs']}) | "
        f"SHORTS: {'ENABLED' if trend_info['enable_shorts'] else 'DISABLED'} (max {trend_info['max_shorts']})\n"
        f"  Reason: {trend_info.get('reason', 'N/A')}"
    )
    logger.info(f"\n{trend_msg}")
    send_telegram(trend_msg, config)

    # ── VIX extreme check ──
    vix_limit = config.get("filters", {}).get("vix_extreme_threshold", 40)
    if vix > vix_limit:
        msg = f"⚠️ VIX {vix} > {vix_limit} — EXTREME. Sitting out."
        logger.warning(msg)
        send_telegram(msg, config)
        return

    # ── Claude Brain V2 morning analysis ──
    brain = ClaudeBrainV2(config=config)
    brain_advice = None
    if brain.enabled and not is_backtest:
        logger.info("  🧠 Consulting Claude Brain V2...")
        brain_advice = brain.morning_analysis(vix=vix, stock_scores=[])
        if brain_advice:
            logger.info(f"  🧠 Claude says: {brain_advice.get('risk_level', 'N/A')} | "
                        f"Max trades: {brain_advice.get('max_trades', 'N/A')} | "
                        f"Sentiment: {brain_advice.get('news_sentiment', 'N/A')} | "
                        f"{brain_advice.get('market_outlook', '')}")
            send_telegram(
                f"🧠 Claude Brain V2 — {today_str}\n"
                f"Risk: {brain_advice.get('risk_level', 'N/A')}\n"
                f"Sentiment: {brain_advice.get('news_sentiment', 'N/A')}\n"
                f"Max trades: {brain_advice.get('max_trades', 'N/A')}\n"
                f"Skip: {brain_advice.get('skip_stocks', [])}\n"
                f"Prefer: {brain_advice.get('preferred_stocks', [])}\n"
                f"Outlook: {brain_advice.get('market_outlook', '')}\n"
                f"Notes: {brain_advice.get('notes', '')}",
                config,
            )
            if brain_advice.get("max_trades"):
                config["capital"]["max_trades_per_day"] = brain_advice["max_trades"]

    # ── Dynamic stock selection ──
    if symbols:
        # If specific stocks given, still score them
        picks = select_best_stocks(symbols, config, target_date, top_n=len(symbols))
    else:
        # Scan full universe — pick best candidates
        # V2: Expanded to Nifty 100 for more SHORT opportunities in bear markets
        universe = get_universe("nifty100")  # was nifty50, now nifty100 for more options
        picks = select_best_stocks(universe, config, target_date, top_n=10)  # was 8, now 10

    if picks.empty:
        msg = "⚠️ No stocks passed selection criteria today."
        logger.warning(msg)
        send_telegram(msg, config)
        return

    # ── Apply Claude Brain skip list ──
    if brain_advice and brain_advice.get("skip_stocks"):
        skip = brain_advice["skip_stocks"]
        before = len(picks)
        picks = picks[~picks["symbol"].isin(skip)]
        if len(picks) < before:
            logger.info(f"  🧠 Claude skipped {before - len(picks)} stocks: {skip}")

    # ── Telegram: picks ──
    long_picks = len(picks[picks["direction"] == "LONG"])
    short_picks = len(picks[picks["direction"] == "SHORT"])
    
    pick_lines = [
        f"🤖 STOCK SELECTION — {today_str}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 VIX: {vix:.1f} | Trend: {MARKET_TREND}",
        f"📈 LONGS: {long_picks} | 📉 SHORTS: {short_picks}",
        "",
    ]
    for _, r in picks.iterrows():
        arrow = "📈" if r["direction"] == "LONG" else "📉"
        pick_lines.append(
            f"{arrow} {r['symbol']:<10} ML:{r['ml_score']:.0f} "
            f"ATR:{r['atr_pct']*100:.1f}% {r['direction']}"
        )
    pick_lines.append("")
    pick_lines.append(f"Limits: LONGS≤{MAX_LONGS} | SHORTS≤{MAX_SHORTS}")
    pick_msg = "\n".join(pick_lines)
    logger.info(pick_msg)
    send_telegram(pick_msg, config)

    # ── Fetch intraday candles ──
    logger.info(f"\n  Fetching 5-min candles...")
    all_candles = {}
    for _, pick in picks.iterrows():
        sym = pick["symbol"]
        candles = fetch_intraday_candles(sym, target_date, broker=broker)
        if candles is not None and len(candles) >= 10:
            all_candles[sym] = candles
            logger.info(f"    {sym}: {len(candles)} candles")

    if not all_candles and is_backtest:
        msg = "⚠️ No intraday data available."
        logger.warning(msg)
        send_telegram(msg, config)
        return

    # ── Create adaptive traders ──
    traders = {}
    if is_backtest:
        # Backtest: only create traders for stocks with candles
        for _, pick in picks.iterrows():
            sym = pick["symbol"]
            if sym not in all_candles:
                continue
            traders[sym] = AdaptiveV3Trader(
                sym, pick["direction"], config, ml_score=pick["ml_score"]
            )
    else:
        # Live: create traders for ALL picks — candles will arrive via polling
        for _, pick in picks.iterrows():
            sym = pick["symbol"]
            traders[sym] = AdaptiveV3Trader(
                sym, pick["direction"], config, ml_score=pick["ml_score"]
            )

    # ── Run: backtest (all candles at once) or live (polling loop) ──
    if is_backtest:
        logger.info(f"\n  Running backtest on {today_str}...")
        for sym, trader in traders.items():
            candles = all_candles[sym]
            trader.process_new_candles(candles)
    else:
        # Live polling loop
        logger.info(f"\n  {'─' * 50}")
        logger.info(f"  MARKET OPEN — Polling 5-min candles")
        logger.info(f"  {'─' * 50}")

        now = datetime.now()
        market_open = now.replace(hour=9, minute=20, second=0)
        if now < market_open:
            wait = (market_open - now).total_seconds()
            logger.info(f"  Waiting {wait/60:.0f} min for market open...")
            time.sleep(max(0, wait))

        market_close = now.replace(hour=15, minute=25, second=0)
        poll_interval = 65
        last_status = datetime.now()

        while datetime.now() < market_close:
            now_t = datetime.now()
            for sym, trader in traders.items():
                try:
                    candles = fetch_intraday_candles(sym, broker=broker)
                    if candles is not None and len(candles) > trader.processed_candles:
                        trader.process_new_candles(candles)
                except Exception as e:
                    logger.error(f"  {sym} error: {e}")

            # Status every 15 min
            if (now_t - last_status).total_seconds() >= 900:
                open_pos = sum(1 for t in traders.values() if t.in_trade)
                closed = sum(len(t.trades) for t in traders.values())
                total_pnl = sum(tr["net_pnl"] for t in traders.values() for tr in t.trades)
                regimes = {sym: t.current_regime for sym, t in traders.items() if t.current_regime != "unknown"}
                status_msg = (
                    f"⏱ {now_t.strftime('%H:%M')} | Open: {open_pos} | Closed: {closed} | "
                    f"P&L: Rs {total_pnl:+,.0f}\nRegimes: {regimes}"
                )
                logger.info(f"  {status_msg}")
                send_telegram(status_msg, config)
                last_status = now_t

                # ── Claude Brain V2: Live adjustment every 15 min ──
                if brain.enabled:
                    try:
                        live_state = {
                            "time": now_t.strftime("%H:%M"),
                            "vix": vix,
                            "day_pnl": total_pnl,
                            "trades_taken": closed,
                            "open_positions": [
                                {"symbol": s, "side": t.side, "pnl": t.day_pnl, "regime": t.current_regime}
                                for s, t in traders.items() if t.in_trade
                            ],
                            "stock_regimes": regimes,
                        }
                        adj = brain.live_adjustment(live_state)
                        if adj and adj.get("emergency_exits"):
                            for sym_exit in adj["emergency_exits"]:
                                if sym_exit in traders and traders[sym_exit].in_trade:
                                    logger.warning(f"  🧠 EMERGENCY EXIT: {sym_exit}")
                                    candles_ex = fetch_intraday_candles(sym_exit, broker=broker)
                                    if candles_ex is not None and len(candles_ex) > 0:
                                        traders[sym_exit]._close_trade(
                                            candles_ex.iloc[-1]["close"],
                                            pd.to_datetime(candles_ex.iloc[-1]["datetime"]),
                                            "CLAUDE_EMERGENCY"
                                        )
                        if adj and adj.get("notes") and adj["notes"] != "No live adjustment":
                            logger.info(f"  🧠 Brain: {adj['notes']}")
                    except Exception as brain_err:
                        logger.debug(f"Brain live adjustment error: {brain_err}")

            time.sleep(poll_interval)

    # ── End of day summary ──
    all_trades = []
    for sym, trader in traders.items():
        all_trades.extend(trader.trades)

    total_pnl = sum(t["net_pnl"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["net_pnl"] > 0)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  END OF DAY — {today_str} — V3 Adaptive")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Trades: {len(all_trades)} | Won: {wins} | Lost: {len(all_trades) - wins}")
    logger.info(f"  Net P&L: Rs {total_pnl:+,.2f}")

    # Save CSV
    if all_trades:
        df = pd.DataFrame(all_trades)
        outpath = f"results/live_v3_{today_str}.csv"
        df.to_csv(outpath, index=False)
        logger.info(f"  Saved: {outpath}")

    # Telegram EOD
    summary = [
        f"📊 END OF DAY SUMMARY — {today_str}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📈 Market Trend: {MARKET_TREND}",
        f"📊 VIX: {vix:.1f}",
        "",
    ]
    
    if all_trades:
        wr = wins / len(all_trades) * 100
        losses = len(all_trades) - wins
        
        # Group by direction
        longs = [t for t in all_trades if t["direction"] == "LONG"]
        shorts = [t for t in all_trades if t["direction"] == "SHORT"]
        long_pnl = sum(t["net_pnl"] for t in longs)
        short_pnl = sum(t["net_pnl"] for t in shorts)
        
        summary.append(f"📋 TRADE STATISTICS")
        summary.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        summary.append(f"Total: {len(all_trades)} | Won: {wins} | Lost: {losses} | WR: {wr:.0f}%")
        summary.append(f"LONGS: {len(longs)} trades → ₹{long_pnl:+,.2f}")
        summary.append(f"SHORTS: {len(shorts)} trades → ₹{short_pnl:+,.2f}")
        summary.append("")
        summary.append(f"📝 TRADE DETAILS")
        summary.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for t in all_trades:
            emoji = "✅" if t["net_pnl"] > 0 else "❌"
            direction_emoji = "📈" if t["direction"] == "LONG" else "📉"
            entry_time = t.get("entry_time", "").split(" ")[-1][:5] if t.get("entry_time") else "?"
            exit_time = t.get("exit_time", "").split(" ")[-1][:5] if t.get("exit_time") else "?"
            
            summary.append(f"{emoji}{direction_emoji} {t['symbol']} ({t['type']})")
            summary.append(f"   {entry_time}→{exit_time} | ₹{t['entry']:,.0f}→₹{t['exit']:,.0f}")
            summary.append(f"   P&L: ₹{t['net_pnl']:+,.2f} | {t['reason']}")
        
        summary.append("")
        summary.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        summary.append(f"💰 NET P&L: ₹{total_pnl:+,.2f}")
        
        # Performance insight
        if total_pnl > 0:
            summary.append(f"🎉 Profitable day! Keep it up!")
        elif total_pnl < -500:
            summary.append(f"⚠️ Tough day. Review trades for tomorrow.")
    else:
        summary.append("📭 NO TRADES TODAY")
        summary.append("")
        summary.append("Possible reasons:")
        if vix > 30:
            summary.append(f"  • High VIX ({vix:.1f}) - Market too volatile")
        if not ENABLE_LONGS and not ENABLE_SHORTS:
            summary.append(f"  • Both LONG/SHORT disabled due to trend")
        summary.append(f"  • No stocks passed ML confidence filter (>{MIN_ML_CONFIDENCE*100:.0f}%)")
        summary.append(f"  • Low liquidity or volatility in scanned stocks")
        summary.append(f"  • Regime unfavorable for selected strategies")

    summary.append("")
    summary.append(f"🤖 V3: Dynamic trend | Regime detection | ML filter")
    eod_msg = "\n".join(summary)
    logger.info(eod_msg)
    send_telegram(eod_msg, config)

    # ── Claude Brain EOD analysis ──
    if brain.enabled and all_trades and not is_backtest:
        eod = brain.eod_analysis(all_trades, total_pnl, {})
        if eod:
            logger.info(f"  🧠 Claude EOD: {eod}")

    # ── Save broker report ──
    if broker:
        broker.save_day_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Adaptive Live Paper Trader")
    parser.add_argument("--universe", default=None, choices=["nifty50", "nifty100", "nifty250"])
    parser.add_argument("--stocks", nargs="+", default=None)
    parser.add_argument("--backtest", default=None, help="Backtest date YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=8, help="Top N stocks to trade")
    args = parser.parse_args()

    if args.stocks:
        run(args.stocks, backtest_date=args.backtest)
    elif args.universe:
        run(get_universe(args.universe), backtest_date=args.backtest)
    else:
        run(backtest_date=args.backtest)
