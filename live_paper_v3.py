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
DISABLE_VWAP = False         # FIXED: Now uses 2σ + RSI + MACD + volume spike
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


def get_ml_strategy_recommendation(stock_features, regime):
    """
    Use ML features to recommend the best strategy for a stock.
    
    Args:
        stock_features: dict with ML features (atr_pct, adx, rsi, momentum, etc.)
        regime: current market regime (trending_up, trending_down, ranging, etc.)
    
    Returns:
        dict with recommended strategies and confidence
    """
    atr_pct = stock_features.get("atr_pct", 0.02)
    adx = stock_features.get("adx", 25)
    rsi = stock_features.get("rsi", 50)
    ml_confidence = stock_features.get("ml_confidence", 0.0)
    volatility_ratio = stock_features.get("volatility_ratio", 1.0)
    
    recommendations = {
        "primary_strategy": None,
        "secondary_strategy": None,
        "avoid_strategies": [],
        "confidence": 50,
        "reasoning": "",
    }
    
    # High ADX (> 25) + trending regime = momentum/trend strategies
    if adx > 25 and regime in ("trending_up", "trending_down"):
        recommendations["primary_strategy"] = "MACD_MOMENTUM"
        recommendations["secondary_strategy"] = "EMA_CROSSOVER"
        recommendations["avoid_strategies"] = ["VWAP", "BOLLINGER"]
        recommendations["confidence"] = min(85, 60 + adx)
        recommendations["reasoning"] = f"Strong trend (ADX={adx:.0f}), use momentum"
    
    # Low ADX (< 20) + ranging/choppy = mean reversion
    elif adx < 20 and regime in ("ranging", "choppy"):
        recommendations["primary_strategy"] = "VWAP_EXTREME"
        recommendations["secondary_strategy"] = "BOLLINGER"
        recommendations["avoid_strategies"] = ["MACD_MOMENTUM", "EMA_CROSSOVER"]
        recommendations["confidence"] = min(75, 50 + (20 - adx) * 2)
        recommendations["reasoning"] = f"Weak trend (ADX={adx:.0f}), use mean reversion"
    
    # High volatility (ATR > 2.5%) = ORB/momentum in morning only
    elif atr_pct > 0.025:
        recommendations["primary_strategy"] = "ORB_BREAKOUT"
        recommendations["secondary_strategy"] = "MOMENTUM"
        recommendations["avoid_strategies"] = ["AFTERNOON_TREND"]
        recommendations["confidence"] = 65
        recommendations["reasoning"] = f"High volatility (ATR={atr_pct*100:.1f}%), morning momentum"
    
    # RSI extreme + mean reversion setup
    elif rsi < 30 or rsi > 70:
        recommendations["primary_strategy"] = "VWAP_EXTREME"
        recommendations["secondary_strategy"] = "BOLLINGER"
        recommendations["avoid_strategies"] = []
        recommendations["confidence"] = min(80, 50 + abs(rsi - 50))
        recommendations["reasoning"] = f"RSI extreme ({rsi:.0f}), mean reversion setup"
    
    # Default: balanced approach
    else:
        recommendations["primary_strategy"] = "MOMENTUM"
        recommendations["secondary_strategy"] = "EMA_CROSSOVER"
        recommendations["avoid_strategies"] = []
        recommendations["confidence"] = 50 + int(ml_confidence * 100)
        recommendations["reasoning"] = "Balanced setup, multiple strategies"
    
    # Adjust confidence based on ML model confidence
    recommendations["confidence"] = min(95, recommendations["confidence"] + int(ml_confidence * 30))
    
    return recommendations

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
    Detect overall market trend using INTRADAY data (not stale daily data).
    
    KEY CHANGE FOR INTRADAY:
    - Uses today's open as reference (not yesterday's close)
    - Position in today's range determines bias
    - Recent momentum (last 30 min) matters more than weekly trend
    
    Analysis factors:
    1. Intraday change from today's open
    2. Position in today's range (0=low, 1=high)
    3. VIX level (fear indicator)
    4. Recent 30-min momentum
    
    Returns: dict with trend info and trading directions
    """
    global MARKET_TREND, ENABLE_LONGS, ENABLE_SHORTS, MAX_LONGS, MAX_SHORTS
    
    result = {
        "trend": "NEUTRAL",
        "confidence": 50,
        "enable_longs": True,
        "enable_shorts": True,
        "max_longs": 3,
        "max_shorts": 3,
        "reason": "",
    }
    
    try:
        # For backtest, fetch historical intraday data
        if target_date and target_date != date.today():
            import yfinance as yf
            start = target_date
            end = target_date + timedelta(days=1)
            nifty = yf.Ticker("^NSEI").history(start=start.isoformat(), end=end.isoformat(), interval="5m")
            
            if nifty.empty or len(nifty) < 6:
                logger.warning("  Could not fetch Nifty intraday data for trend detection")
                return result
            
            nifty = nifty.reset_index()
            nifty.columns = [c.lower() for c in nifty.columns]
            
            # Calculate intraday metrics
            today_open = float(nifty['open'].iloc[0])
            today_high = float(nifty['high'].max())
            today_low = float(nifty['low'].min())
            current_price = float(nifty['close'].iloc[-1])
            
            intraday_change_pct = (current_price - today_open) / today_open * 100 if today_open > 0 else 0
            range_size = today_high - today_low
            position_in_range = (current_price - today_low) / range_size if range_size > 0 else 0.5
        else:
            # For live trading, use cached Nifty data
            NIFTY_CACHE.refresh(force=True)
            
            if NIFTY_CACHE.candles is None or len(NIFTY_CACHE.candles) < 6:
                logger.warning("  Could not fetch Nifty intraday data for trend detection")
                return result
            
            today_open = NIFTY_CACHE.today_open
            current_price = NIFTY_CACHE.current_price
            intraday_change_pct = NIFTY_CACHE.intraday_change_pct
            position_in_range = NIFTY_CACHE.position_in_range
        
        # Trend scoring based on INTRADAY metrics
        trend_score = 0
        reasons = []
        
        # Factor 1: Intraday change from TODAY'S OPEN (most important!)
        if intraday_change_pct > 1.0:
            trend_score += 40
            reasons.append(f"Strong intraday up +{intraday_change_pct:.1f}%")
        elif intraday_change_pct > 0.3:
            trend_score += 20
            reasons.append(f"Intraday up +{intraday_change_pct:.1f}%")
        elif intraday_change_pct < -1.0:
            trend_score -= 40
            reasons.append(f"Strong intraday down {intraday_change_pct:.1f}%")
        elif intraday_change_pct < -0.3:
            trend_score -= 20
            reasons.append(f"Intraday down {intraday_change_pct:.1f}%")
        else:
            reasons.append(f"Flat intraday {intraday_change_pct:+.1f}%")
        
        # Factor 2: Position in TODAY'S range (0=at low, 1=at high)
        if position_in_range > 0.75:
            trend_score += 25
            reasons.append(f"Near day high ({position_in_range:.0%})")
        elif position_in_range > 0.55:
            trend_score += 10
            reasons.append(f"Upper half ({position_in_range:.0%})")
        elif position_in_range < 0.25:
            trend_score -= 25
            reasons.append(f"Near day low ({position_in_range:.0%})")
        elif position_in_range < 0.45:
            trend_score -= 10
            reasons.append(f"Lower half ({position_in_range:.0%})")
        
        # Factor 3: VIX level (still relevant for position sizing)
        if vix > 25:
            trend_score -= 15
            reasons.append(f"High VIX {vix:.1f}")
        elif vix > 20:
            trend_score -= 5
            reasons.append(f"Elevated VIX {vix:.1f}")
        elif vix < 13:
            trend_score += 5
            reasons.append(f"Low VIX {vix:.1f}")
        
        # Determine trend based on INTRADAY score
        if trend_score >= 50:
            result["trend"] = "STRONG_BULLISH"
            result["confidence"] = min(95, 70 + abs(trend_score) // 2)
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 5
            result["max_shorts"] = 2
        elif trend_score >= 20:
            result["trend"] = "BULLISH"
            result["confidence"] = min(85, 60 + abs(trend_score) // 2)
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 4
            result["max_shorts"] = 3
        elif trend_score <= -50:
            result["trend"] = "BEARISH"
            result["confidence"] = min(95, 70 + abs(trend_score) // 2)
            result["enable_longs"] = False
            result["enable_shorts"] = True
            result["max_longs"] = 0
            result["max_shorts"] = 6
        elif trend_score <= -20:
            result["trend"] = "MILD_BEARISH"
            result["confidence"] = min(80, 55 + abs(trend_score) // 2)
            result["enable_longs"] = True  # Allow 1 LONG in mild bear
            result["enable_shorts"] = True
            result["max_longs"] = 1
            result["max_shorts"] = 5
        else:
            result["trend"] = "NEUTRAL"
            result["confidence"] = 50
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 3
            result["max_shorts"] = 3
        
        result["reason"] = " | ".join(reasons)
        result["nifty_open"] = today_open
        result["nifty_current"] = current_price
        result["intraday_change_pct"] = intraday_change_pct
        result["position_in_range"] = position_in_range
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

def get_realtime_market_bias(nifty_candles=None, current_idx=None):
    """
    Get market bias based on WHERE price is in TODAY's range.
    This adapts in real-time and catches V-reversals.
    
    NOW USES CACHED DATA for efficiency!
    
    Returns: (bias, confidence)
        bias: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
        confidence: 0-100
    """
    # Use cached data if no candles provided
    if nifty_candles is None:
        return NIFTY_CACHE.get_intraday_bias()
    
    # Legacy support: if candles provided, use them
    if current_idx is None:
        current_idx = len(nifty_candles) - 1
    
    if current_idx < 6:
        return 0, 0
    
    open_price = nifty_candles['open'].iloc[0]
    current = nifty_candles['close'].iloc[current_idx]
    day_low = nifty_candles['low'].iloc[:current_idx+1].min()
    day_high = nifty_candles['high'].iloc[:current_idx+1].max()
    
    # Position in TODAY's range (0 = at low, 1 = at high)
    if day_high != day_low:
        position_in_range = (current - day_low) / (day_high - day_low)
    else:
        position_in_range = 0.5
    
    # Intraday change from TODAY'S OPEN (not yesterday's close!)
    intraday_change_pct = (current - open_price) / open_price * 100 if open_price > 0 else 0
    
    # Determine bias
    bias = 0
    confidence = 50
    
    # LONG bias: Price in upper 55% of range
    if position_in_range > 0.55:
        bias = 1
        confidence = min(85, 50 + position_in_range * 35)
        # Boost if intraday is also positive
        if intraday_change_pct > 0.3:
            confidence = min(90, confidence + 5)
    
    # SHORT bias: Price in lower 45% of range
    elif position_in_range < 0.45:
        bias = -1
        confidence = min(85, 50 + (1 - position_in_range) * 35)
        # Boost if intraday is also negative
        if intraday_change_pct < -0.3:
            confidence = min(90, confidence + 5)
    
    # Check for recent momentum (last 3 candles)
    if current_idx >= 3:
        recent_change = (nifty_candles['close'].iloc[current_idx] - 
                        nifty_candles['close'].iloc[current_idx-3]) / nifty_candles['close'].iloc[current_idx-3] * 100
        # Strong recent momentum can flip or strengthen bias
        if recent_change > 0.2 and bias != -1:
            bias = 1
            confidence = min(85, confidence + 5)
        elif recent_change < -0.2 and bias != 1:
            bias = -1
            confidence = min(85, confidence + 5)
    
    return bias, confidence


# Global sector data (updated at market open)
SECTOR_PERFORMANCE = {}
TOP_SECTORS = []
WEAK_SECTORS = []
VIX_LEVEL = 15.0  # Default, updated at open


# ════════════════════════════════════════════════════════════
# NIFTY INTRADAY CACHE — Avoid repeated API calls
# ════════════════════════════════════════════════════════════

class NiftyIntradayCache:
    """
    Cache Nifty 50 intraday data with automatic refresh.
    
    Key insight: For intraday trading, we only care about TODAY's data.
    - Refreshes every 5 minutes (aligned with candle intervals)
    - Provides instant access for trend/bias calculations
    - Reduces API calls from ~750/day to ~75/day
    """
    
    def __init__(self, refresh_interval_sec=300):
        self.refresh_interval = refresh_interval_sec  # 5 min default
        self.last_fetch = None
        self.candles = None
        self.today_open = None
        self.today_high = None
        self.today_low = None
        self.current_price = None
        self.intraday_change_pct = 0.0
        self.position_in_range = 0.5  # 0=at low, 1=at high
        self._lock = False
    
    def _needs_refresh(self):
        if self.candles is None or self.last_fetch is None:
            return True
        elapsed = (datetime.now() - self.last_fetch).total_seconds()
        return elapsed >= self.refresh_interval
    
    def refresh(self, force=False):
        """Fetch fresh Nifty intraday data."""
        if not force and not self._needs_refresh():
            return self.candles
        
        if self._lock:
            return self.candles
        
        self._lock = True
        try:
            import yfinance as yf
            nifty = yf.Ticker("^NSEI").history(period="1d", interval="5m")
            
            if nifty.empty or len(nifty) < 2:
                self._lock = False
                return self.candles
            
            nifty = nifty.reset_index()
            nifty.columns = [c.lower() for c in nifty.columns]
            
            self.candles = nifty
            self.last_fetch = datetime.now()
            
            # Calculate intraday metrics
            self.today_open = float(nifty['open'].iloc[0])
            self.today_high = float(nifty['high'].max())
            self.today_low = float(nifty['low'].min())
            self.current_price = float(nifty['close'].iloc[-1])
            
            # Intraday change from TODAY'S OPEN (not yesterday's close)
            if self.today_open > 0:
                self.intraday_change_pct = (self.current_price - self.today_open) / self.today_open * 100
            
            # Position in today's range (0=at low, 1=at high)
            range_size = self.today_high - self.today_low
            if range_size > 0:
                self.position_in_range = (self.current_price - self.today_low) / range_size
            else:
                self.position_in_range = 0.5
            
            logger.debug(f"NiftyCache refreshed: {len(nifty)} candles, "
                        f"open={self.today_open:.0f}, current={self.current_price:.0f}, "
                        f"change={self.intraday_change_pct:+.2f}%, pos={self.position_in_range:.2f}")
        
        except Exception as e:
            logger.warning(f"NiftyCache refresh failed: {e}")
        finally:
            self._lock = False
        
        return self.candles
    
    def get_candles(self):
        """Get cached candles, refreshing if needed."""
        self.refresh()
        return self.candles
    
    def get_intraday_bias(self):
        """
        Get market bias based on WHERE price is in TODAY's range.
        This is the KEY intraday insight — not historical trend!
        
        Returns: (bias, confidence)
            bias: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
            confidence: 0-100
        """
        self.refresh()
        
        if self.candles is None or len(self.candles) < 6:
            return 0, 0
        
        bias = 0
        confidence = 50
        
        # Position in range determines bias
        # Upper 60% = LONG bias, Lower 40% = SHORT bias
        if self.position_in_range > 0.6:
            bias = 1
            confidence = min(85, 50 + self.position_in_range * 35)
        elif self.position_in_range < 0.4:
            bias = -1
            confidence = min(85, 50 + (1 - self.position_in_range) * 35)
        
        # Boost confidence if intraday change aligns with position
        if bias == 1 and self.intraday_change_pct > 0.3:
            confidence = min(90, confidence + 10)
        elif bias == -1 and self.intraday_change_pct < -0.3:
            confidence = min(90, confidence + 10)
        
        return bias, confidence
    
    def get_intraday_trend(self):
        """
        Detect trend using ONLY intraday data — not stale daily data.
        
        Returns: dict with trend info
        """
        self.refresh()
        
        result = {
            "trend": "NEUTRAL",
            "confidence": 50,
            "enable_longs": True,
            "enable_shorts": True,
            "max_longs": 3,
            "max_shorts": 3,
            "reason": "",
            "intraday_change_pct": self.intraday_change_pct,
            "position_in_range": self.position_in_range,
        }
        
        if self.candles is None or len(self.candles) < 6:
            return result
        
        reasons = []
        
        # Factor 1: Intraday change from open (most important!)
        if self.intraday_change_pct > 1.0:
            result["trend"] = "STRONG_BULLISH"
            result["confidence"] = min(90, 70 + abs(self.intraday_change_pct) * 10)
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 5
            result["max_shorts"] = 2
            reasons.append(f"Strong intraday up +{self.intraday_change_pct:.1f}%")
        elif self.intraday_change_pct > 0.3:
            result["trend"] = "BULLISH"
            result["confidence"] = min(80, 60 + abs(self.intraday_change_pct) * 15)
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 4
            result["max_shorts"] = 3
            reasons.append(f"Intraday up +{self.intraday_change_pct:.1f}%")
        elif self.intraday_change_pct < -1.0:
            result["trend"] = "BEARISH"
            result["confidence"] = min(90, 70 + abs(self.intraday_change_pct) * 10)
            result["enable_longs"] = False
            result["enable_shorts"] = True
            result["max_longs"] = 0
            result["max_shorts"] = 6
            reasons.append(f"Strong intraday down {self.intraday_change_pct:.1f}%")
        elif self.intraday_change_pct < -0.3:
            result["trend"] = "MILD_BEARISH"
            result["confidence"] = min(75, 55 + abs(self.intraday_change_pct) * 15)
            result["enable_longs"] = False
            result["enable_shorts"] = True
            result["max_longs"] = 1
            result["max_shorts"] = 5
            reasons.append(f"Intraday down {self.intraday_change_pct:.1f}%")
        else:
            result["trend"] = "NEUTRAL"
            result["confidence"] = 50
            result["enable_longs"] = True
            result["enable_shorts"] = True
            result["max_longs"] = 3
            result["max_shorts"] = 3
            reasons.append(f"Flat intraday {self.intraday_change_pct:+.1f}%")
        
        # Factor 2: Position in today's range
        if self.position_in_range > 0.75:
            reasons.append("Near day high (bullish)")
            if result["trend"] in ("BULLISH", "STRONG_BULLISH"):
                result["confidence"] = min(95, result["confidence"] + 5)
        elif self.position_in_range < 0.25:
            reasons.append("Near day low (bearish)")
            if result["trend"] in ("BEARISH", "MILD_BEARISH"):
                result["confidence"] = min(95, result["confidence"] + 5)
        
        # Factor 3: Recent momentum (last 6 candles = 30 min)
        if len(self.candles) >= 6:
            recent = self.candles.tail(6)
            recent_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
            if recent_change > 0.2:
                reasons.append(f"30min momentum +{recent_change:.1f}%")
            elif recent_change < -0.2:
                reasons.append(f"30min momentum {recent_change:.1f}%")
        
        result["reason"] = " | ".join(reasons)
        result["nifty_open"] = self.today_open
        result["nifty_current"] = self.current_price
        
        return result


# Global Nifty cache instance
NIFTY_CACHE = NiftyIntradayCache(refresh_interval_sec=300)  # 5 min refresh

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

        # ══════════════════════════════════════════════════════════════
        # ML-ENHANCED FEATURES for intraday trading
        # ══════════════════════════════════════════════════════════════
        
        # ML volatility prediction (ATR expected vs actual)
        expected_atr = last.get("atr_14", atr_pct * last.get("close", 100))
        volatility_ratio = atr_pct / 0.02 if atr_pct > 0 else 1.0  # vs 2% baseline
        
        # ML momentum features
        momentum_5 = last.get("momentum_5", 0)  # 5-day momentum
        momentum_10 = last.get("momentum_10", 0)  # 10-day momentum
        
        # ML trend strength (from features if available)
        adx = last.get("adx", 25)  # Average Directional Index
        trend_strength = "strong" if adx > 25 else "weak"
        
        # Optimal entry time suggestion based on historical patterns
        # (ML could be trained to predict best entry windows)
        if atr_pct > 0.025:
            optimal_window = "morning"  # High vol stocks better in morning
        elif atr_pct < 0.015:
            optimal_window = "afternoon"  # Low vol stocks need time to move
        else:
            optimal_window = "any"
        
        candidates.append({
            "symbol": sym, "ml_score": ml_score, "direction": direction,
            "ml_prob": ml_prob, "ml_confidence": ml_confidence,
            "rsi": rsi, "atr_pct": round(atr_pct, 4), "avg_volume": int(avg_vol),
            "delivery_pct": nse.get("delivery_pct", 0),
            "composite_score": round(composite, 1), "sector": sector,
            # NEW ML-enhanced fields
            "volatility_ratio": round(volatility_ratio, 2),
            "momentum_5": round(momentum_5, 4) if momentum_5 else 0,
            "momentum_10": round(momentum_10, 4) if momentum_10 else 0,
            "adx": round(adx, 1) if adx else 25,
            "trend_strength": trend_strength,
            "optimal_window": optimal_window,
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

    def __init__(self, symbol, direction, config, ml_score=50, claude_brain=None, ml_features=None):
        self.symbol = symbol
        self.direction = direction  # Now overridden by real-time bias in _scan_for_entry
        self.ml_score = ml_score  # Used for stock SELECTION only, not direction
        self.capital = config["capital"]["total"]
        self.config = config
        self.cost_model = AngelOneCostModel()
        self.strategy = ProStrategyV2(config.get("strategies", {}).get("pro", {}))
        self.regime_detector = RegimeDetector()
        self.pattern_detector = CandlePatternDetector()  # Candlestick patterns
        self.claude_brain = claude_brain  # Optional: Claude Brain for entry confirmation
        
        # ══════════════════════════════════════════════════════════════
        # ML-ENHANCED FEATURES for smarter trading decisions
        # ══════════════════════════════════════════════════════════════
        self.ml_features = ml_features or {}
        self.ml_prob = self.ml_features.get("ml_prob", 0.5)
        self.ml_confidence = self.ml_features.get("ml_confidence", 0.0)
        self.volatility_ratio = self.ml_features.get("volatility_ratio", 1.0)
        self.trend_strength = self.ml_features.get("trend_strength", "weak")
        self.optimal_window = self.ml_features.get("optimal_window", "any")
        self.adx = self.ml_features.get("adx", 25)

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
        self.pattern_score = 0  # Track pattern confirmation
        
        # ══════════════════════════════════════════════════════════════
        # INTRADAY-SPECIFIC STATE
        # ══════════════════════════════════════════════════════════════
        self.today_open = None       # Today's opening price
        self.today_high = None       # Today's high so far
        self.today_low = None        # Today's low so far
        self.gap_pct = 0.0           # Gap from previous close
        self.position_in_range = 0.5 # Where price is in today's range (0-1)
        self.intraday_change_pct = 0.0  # Change from today's open

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
        
        # ── CLAUDE BRAIN CONFIRMATION (optional, for high-value trades) ──
        # Only consult Claude for larger position sizes to save API calls
        if self.claude_brain and self.claude_brain.enabled:
            try:
                # Build intraday data for Claude
                intraday_data = {
                    "today_open": self.today_open,
                    "current_price": signal["entry"],
                    "today_high": self.today_high,
                    "today_low": self.today_low,
                    "intraday_change_pct": self.intraday_change_pct,
                    "position_in_range": self.position_in_range,
                    "volume_vs_avg": 1.0,  # Would need to calculate
                    "vwap": signal.get("vwap", signal["entry"]),
                }
                
                # Get Nifty bias from cache
                nifty_bias = "LONG" if NIFTY_CACHE.position_in_range > 0.55 else "SHORT" if NIFTY_CACHE.position_in_range < 0.45 else "NEUTRAL"
                
                # Ask Claude for confirmation
                claude_advice = self.claude_brain.analyze_stock_intraday(
                    self.symbol, 
                    intraday_data, 
                    nifty_bias, 
                    self.current_regime
                )
                
                if claude_advice:
                    # Check if Claude says to skip
                    if not claude_advice.get("take_trade", True):
                        logger.info(f"  {self.symbol} SKIP: Claude says no - {claude_advice.get('reason', 'N/A')}")
                        return
                    
                    # Adjust position size based on Claude's confidence
                    confidence = claude_advice.get("confidence", 50)
                    if confidence < 40:
                        logger.info(f"  {self.symbol} SKIP: Claude confidence too low ({confidence}%)")
                        return
                    
                    # Use Claude's suggested levels if provided and valid
                    if claude_advice.get("stop_loss") and claude_advice["stop_loss"] > 0:
                        signal["sl"] = claude_advice["stop_loss"]
                    if claude_advice.get("target") and claude_advice["target"] > 0:
                        signal["tgt"] = claude_advice["target"]
                    
                    signal["reason"] = f"{signal.get('reason', '')} | Claude: {confidence}%"
                    
            except Exception as e:
                logger.debug(f"Claude entry check error: {e}")
        
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
        
        # ══════════════════════════════════════════════════════════════
        # ML-BASED POSITION SIZING ADJUSTMENT
        # Use ML features to adjust position size
        # ══════════════════════════════════════════════════════════════
        
        # High ML confidence = larger position
        if self.ml_confidence >= 0.20:  # 20%+ confidence
            risk_pct *= 1.2  # 20% larger position
            signal["reason"] = f"{signal.get('reason', '')} | ML conf +20%"
        elif self.ml_confidence >= 0.15:
            risk_pct *= 1.1  # 10% larger
        elif self.ml_confidence < 0.10:
            risk_pct *= 0.8  # 20% smaller for low confidence
        
        # Strong trend (ADX > 30) = larger position in trend direction
        if self.adx > 30 and self.trend_strength == "strong":
            risk_pct *= 1.15
            signal["reason"] = f"{signal.get('reason', '')} | ADX {self.adx:.0f}"
        
        # High volatility ratio = smaller position (risk management)
        if self.volatility_ratio > 1.5:  # 50% more volatile than baseline
            risk_pct *= 0.7
        elif self.volatility_ratio < 0.7:  # 30% less volatile
            risk_pct *= 1.1  # Can take slightly larger position

        # ── FIX 1b: Cap max shares to limit single-trade damage ──
        # INTRADAY MARGIN: Get actual margin from Angel One API (or fallback to 20%)
        self.shares = max(1, int(self.capital * risk_pct / max(risk, min_risk)))
        
        # Use broker's margin calculator if available
        available_margin = self.capital * 0.40  # Use 40% of capital per trade
        if hasattr(self, 'broker') and self.broker and hasattr(self.broker, 'get_margin_required'):
            # Get real margin from Angel One
            margin_for_one = self.broker.get_margin_required(
                self.symbol, 1, self.entry_price, self.side, "INTRADAY"
            )
            if margin_for_one > 0:
                max_shares = int(available_margin / margin_for_one)
                logger.debug(f"  {self.symbol}: Margin/share Rs {margin_for_one:.0f}, max {max_shares} shares")
            else:
                max_shares = int(available_margin / (self.entry_price * 0.20))
        else:
            # Fallback: 20% margin for MIS/intraday
            margin_per_share = self.entry_price * 0.20
            max_shares = int(available_margin / margin_per_share)
        
        self.shares = min(self.shares, max(1, max_shares))

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

        # ══════════════════════════════════════════════════════════════
        # INTRADAY TRACKING: Update today's metrics on every candle
        # ══════════════════════════════════════════════════════════════
        if i == 0:
            # First candle = today's open
            self.today_open = row["open"]
            self.today_high = h
            self.today_low = l
        else:
            # Update today's high/low
            self.today_high = max(self.today_high or h, h)
            self.today_low = min(self.today_low or l, l)
        
        # Calculate intraday metrics
        if self.today_open and self.today_open > 0:
            self.intraday_change_pct = (c - self.today_open) / self.today_open * 100
        
        if self.today_high and self.today_low and self.today_high > self.today_low:
            self.position_in_range = (c - self.today_low) / (self.today_high - self.today_low)
        else:
            self.position_in_range = 0.5

        # Update regime every 6 candles (30 min)
        if i % 6 == 0 and i >= 12:
            self.current_regime = self.regime_detector.detect(candles, i)

        # ── Circuit breaker: stop if daily loss exceeded ──
        if self.day_pnl < -self.daily_loss_limit and not self.in_trade:
            return

        # ══════════════════════════════════════════════════════════════
        # Phase 1: ORB Formation (with INTRADAY gap awareness)
        # ══════════════════════════════════════════════════════════════
        if self.orb is None:
            if i < self.strategy.orb_candles:
                return
            self.orb = self.strategy.compute_orb(candles, self.strategy.orb_candles)
            if self.orb is None or self.orb["range"] < 0.5:
                return
            
            # Add today's open to ORB for gap-adjusted analysis
            self.orb["today_open"] = self.today_open
            self.orb["gap_direction"] = "UP" if c > self.today_open else "DOWN" if c < self.today_open else "FLAT"
            
            msg = (f"📐 {self.symbol} ORB: H={self.orb['high']:,.2f} L={self.orb['low']:,.2f} "
                   f"R={self.orb['range']:,.2f} | Open={self.today_open:,.2f} | "
                   f"Gap={self.orb['gap_direction']} | Regime: {self.current_regime}")
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
        Scan for entry signals using PURE INTRADAY PRICE ACTION.
        
        KEY INSIGHT FOR INTRADAY:
        - Direction = WHERE price IS in TODAY's range (not ML/daily prediction)
        - Upper 55% of range → LONG bias
        - Lower 45% of range → SHORT bias
        - This adapts EVERY CANDLE to reversals!
        - Uses CACHED Nifty data (no repeated API calls)
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
        # ML-BASED TIME WINDOW FILTER
        # Use ML's optimal_window suggestion for entry timing
        # ══════════════════════════════════════════════════════════════
        if hour < 5:  # Before ~10:15 AM IST (UTC+5:30)
            # Exception: Allow morning entries for high-vol stocks
            if self.optimal_window != "morning" or hour < 4:
                return
        
        # Skip afternoon entries for stocks that work better in morning
        if hour >= 8 and self.optimal_window == "morning":
            return  # ~13:30+ IST, skip if stock is morning-optimal
        
        # ══════════════════════════════════════════════════════════════
        # INTRADAY PRICE ACTION: Get direction from CACHED Nifty data
        # Uses NiftyCache — refreshes every 5 min, no repeated API calls!
        # ══════════════════════════════════════════════════════════════
        realtime_bias = 0
        direction = "NEUTRAL"
        
        # Use cached Nifty data (auto-refreshes every 5 min)
        realtime_bias, bias_conf = NIFTY_CACHE.get_intraday_bias()
        
        if realtime_bias == 1:
            direction = "LONG"
        elif realtime_bias == -1:
            direction = "SHORT"
        else:
            return  # NEUTRAL = don't trade
        
        # Need minimum confidence to trade
        if bias_conf < 55:
            return
        
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
        # ML STRATEGY RECOMMENDATION
        # Use ML features to determine best strategy for this stock
        # ══════════════════════════════════════════════════════════════
        ml_rec = get_ml_strategy_recommendation(self.ml_features, regime)
        avoid_strategies = ml_rec.get("avoid_strategies", [])

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 1: ORB BREAKOUT/BREAKDOWN (9:20-10:30)
        # - In BULLISH market + trending_up: ORB BREAKOUT (LONG)
        # - In BEARISH market + trending_down: ORB BREAKDOWN (SHORT)
        # ══════════════════════════════════════════════════════════════
        if hour < 10 or (hour == 10 and minute <= 30):
            if "ORB" not in avoid_strategies:
                if regime == "trending_down" and direction == "SHORT":
                    signal = self.strategy.generate_orb_signal(candles, i, self.orb, direction)
                    if signal:
                        signal["ml_rec"] = ml_rec.get("primary_strategy")
                        self._enter_trade(signal, i)
                        return
                elif regime == "trending_up" and direction == "LONG":
                    signal = self.strategy.generate_orb_signal(candles, i, self.orb, direction)
                    if signal:
                        signal["ml_rec"] = ml_rec.get("primary_strategy")
                        self._enter_trade(signal, i)
                        return

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 2: EMA 9/21 CROSSOVER (9:45-14:30)
        # Works best in trending regimes — ML recommends for strong trends
        # ══════════════════════════════════════════════════════════════
        if (hour == 9 and minute >= 45) or (10 <= hour <= 14):
            if "EMA_CROSSOVER" not in avoid_strategies:
                if regime in ("trending_up", "trending_down"):
                    signal = self.strategy.generate_ema_crossover_signal(candles, i, direction)
                    if signal:
                        # Add MACD confirmation
                        macd_ok, macd_strength = self.strategy.check_macd_confirmation(candles, i, direction)
                        if macd_ok and macd_strength >= 0.6:
                            signal["reason"] = f"{signal['reason']} | MACD {macd_strength:.0%}"
                            # Boost if ML recommends this strategy
                            if ml_rec.get("primary_strategy") == "EMA_CROSSOVER":
                                signal["reason"] = f"{signal['reason']} | ML recommended"
                            self._enter_trade(signal, i)
                            return

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 3: MACD MOMENTUM (9:45-14:30)
        # Pure momentum play — ML recommends for high ADX
        # ══════════════════════════════════════════════════════════════
        if (hour == 9 and minute >= 45) or (10 <= hour <= 14):
            if "MACD_MOMENTUM" not in avoid_strategies:
                if regime in ("trending_up", "trending_down", "volatile"):
                    signal = self.strategy.generate_macd_signal(candles, i, direction)
                    if signal:
                        if ml_rec.get("primary_strategy") == "MACD_MOMENTUM":
                            signal["reason"] = f"{signal['reason']} | ML recommended"
                        self._enter_trade(signal, i)
                        return

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 4: MOMENTUM CONTINUATION (11:30-14:00)
        # Trade in direction of market trend
        # ══════════════════════════════════════════════════════════════
        if (hour == 11 and minute >= 30) or (12 <= hour <= 13):
            if "MOMENTUM" not in avoid_strategies:
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
                        today_avg_vol = candles["volume"].iloc[:i].mean() if i > 0 else 0
                        curr_vol = candles["volume"].iloc[i]
                        if today_avg_vol > 0 and curr_vol > today_avg_vol * 1.5:
                            self._enter_trade(signal, i)
                            return

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 5: IMPROVED VWAP MEAN REVERSION (10:30-14:00)
        # ML recommends for low ADX / ranging markets
        # ══════════════════════════════════════════════════════════════
        if (hour == 10 and minute >= 30) or (11 <= hour <= 14):
            if "VWAP" not in avoid_strategies and "VWAP_EXTREME" not in avoid_strategies:
                if regime in ("ranging", "choppy"):
                    signal = self.strategy.generate_vwap_improved_signal(candles, i, direction)
                    if signal:
                        if ml_rec.get("primary_strategy") == "VWAP_EXTREME":
                            signal["reason"] = f"{signal['reason']} | ML recommended"
                        self._enter_trade(signal, i)
                        return

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 6: BOLLINGER BANDS MEAN REVERSION (10:30-14:00)
        # ML recommends for low ADX / ranging markets
        # ══════════════════════════════════════════════════════════════
        if (hour == 10 and minute >= 30) or (11 <= hour <= 14):
            if "BOLLINGER" not in avoid_strategies:
                if regime in ("ranging", "choppy"):
                    signal = self.strategy.generate_bollinger_signal(candles, i, direction)
                    if signal:
                        if ml_rec.get("secondary_strategy") == "BOLLINGER":
                            signal["reason"] = f"{signal['reason']} | ML recommended"
                        self._enter_trade(signal, i)
                        return

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 7: AFTERNOON TREND (13:00-14:30)
        # Trade in direction of established intraday trend
        # ══════════════════════════════════════════════════════════════
        if 13 <= hour <= 14 and minute <= 30:
            if "AFTERNOON_TREND" not in avoid_strategies:
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

        # ══════════════════════════════════════════════════════════════
        # INTRADAY FIX: Compare volume to TODAY's average only
        # Not 20-candle historical average which could span multiple days
        # ══════════════════════════════════════════════════════════════
        today_avg_vol = candles["volume"].iloc[:i].mean() if i > 0 else 0
        curr_vol = candles["volume"].iloc[i]
        vol_ok = today_avg_vol > 0 and curr_vol > today_avg_vol * 1.3  # Higher threshold for intraday

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
        # ══════════════════════════════════════════════════════════════
        # INTRADAY FIX: Use intraday RSI (7 candles = 35 min, not 7 days)
        # ══════════════════════════════════════════════════════════════
        rsi = self.strategy.compute_rsi(candles["close"].iloc[:i + 1], period=7)
        vwap, std = self.strategy.compute_vwap_proper(candles, i)
        
        # INTRADAY FIX: Compare to TODAY's average volume only
        today_avg_vol = candles["volume"].iloc[:i].mean() if i > 0 else 0
        curr_vol = candles["volume"].iloc[i]

        if today_avg_vol == 0 or curr_vol < today_avg_vol * 1.3:
            return None  # need strong volume vs TODAY's average

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
        """
        ATR for INTRADAY trading.
        
        KEY CHANGE: Uses all candles from TODAY (not arbitrary lookback)
        This ensures ATR reflects today's volatility, not historical.
        """
        # Use all available candles today (up to period limit)
        lookback = min(idx + 1, period)
        if lookback < 3:
            lookback = idx + 1  # Use whatever we have
        
        recent = candles.iloc[max(0, idx - lookback + 1):idx + 1]
        raw = (recent["high"] - recent["low"]).mean()
        
        # Floor: ATR must be at least 0.5% of current price
        price = candles["close"].iloc[idx]
        return max(raw, price * 0.005)
    
    def _get_intraday_support_resistance(self, candles, idx):
        """
        Get support/resistance levels based on TODAY's data only.
        
        For intraday, multi-day S/R is less relevant than:
        - Today's open (psychological level)
        - Today's high/low (range bounds)
        - VWAP (fair value)
        """
        if idx < 6:
            return None, None
        
        # Today's levels
        today_high = candles["high"].iloc[:idx+1].max()
        today_low = candles["low"].iloc[:idx+1].min()
        today_open = candles["open"].iloc[0]
        current = candles["close"].iloc[idx]
        
        # VWAP as dynamic support/resistance
        vwap, _ = self.strategy.compute_vwap_proper(candles, idx)
        
        # Determine support/resistance based on where price is
        if current > vwap:
            # Price above VWAP: VWAP is support, today's high is resistance
            support = max(vwap, today_open) if today_open < current else vwap
            resistance = today_high
        else:
            # Price below VWAP: today's low is support, VWAP is resistance
            support = today_low
            resistance = min(vwap, today_open) if today_open > current else vwap
        
        return support, resistance


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
    # Show intraday-specific trend info
    intraday_change = trend_info.get('intraday_change_pct', 0)
    pos_in_range = trend_info.get('position_in_range', 0.5)
    trend_msg = (
        f"📊 INTRADAY TREND: {trend_info['trend']} ({trend_info['confidence']}% confidence)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Nifty Open: ₹{trend_info.get('nifty_open', 0):,.0f}\n"
        f"📍 Current: ₹{trend_info.get('nifty_current', 0):,.0f} ({intraday_change:+.2f}%)\n"
        f"📊 Position in Range: {pos_in_range:.0%}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 LONGS: {'ENABLED' if trend_info['enable_longs'] else 'DISABLED'} (max {trend_info['max_longs']})\n"
        f"🔴 SHORTS: {'ENABLED' if trend_info['enable_shorts'] else 'DISABLED'} (max {trend_info['max_shorts']})\n"
        f"📝 {trend_info.get('reason', 'N/A')}"
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

    # ── Claude Brain V2 with INTRADAY analysis ──
    brain = ClaudeBrainV2(config=config)
    brain_advice = None
    intraday_brain_advice = None
    
    if brain.enabled and not is_backtest:
        logger.info("  🧠 Consulting Claude Brain V2 (Intraday Mode)...")
        
        # First, get intraday-specific analysis using cached Nifty data
        NIFTY_CACHE.refresh(force=True)
        intraday_metrics = {
            "nifty_open": NIFTY_CACHE.today_open,
            "nifty_current": NIFTY_CACHE.current_price,
            "nifty_high": NIFTY_CACHE.today_high,
            "nifty_low": NIFTY_CACHE.today_low,
            "intraday_change_pct": NIFTY_CACHE.intraday_change_pct,
            "position_in_range": NIFTY_CACHE.position_in_range,
            "vix": vix,
            "time": datetime.now().strftime("%H:%M"),
            "sector_performance": SECTOR_PERFORMANCE,
        }
        
        intraday_brain_advice = brain.analyze_intraday_trend(intraday_metrics)
        if intraday_brain_advice:
            logger.info(f"  🧠 Intraday Analysis: {intraday_brain_advice.get('trend', 'N/A')} | "
                        f"Direction: {intraday_brain_advice.get('direction', 'N/A')} | "
                        f"Confidence: {intraday_brain_advice.get('confidence', 0)}%")
            
            # Use Claude's intraday direction to override global settings
            claude_direction = intraday_brain_advice.get("direction", "NEUTRAL")
            claude_confidence = intraday_brain_advice.get("confidence", 50)
            
            if claude_direction == "LONG" and claude_confidence >= 60:
                ENABLE_LONGS = True
                MAX_LONGS = min(MAX_LONGS + 1, 6)
            elif claude_direction == "SHORT" and claude_confidence >= 60:
                ENABLE_SHORTS = True
                MAX_SHORTS = min(MAX_SHORTS + 1, 6)
        
        # Also get traditional morning analysis for news/sentiment
        brain_advice = brain.morning_analysis(vix=vix, stock_scores=[])
        if brain_advice:
            logger.info(f"  🧠 Morning Brief: {brain_advice.get('risk_level', 'N/A')} | "
                        f"Sentiment: {brain_advice.get('news_sentiment', 'N/A')}")
            
            # Telegram with combined intraday + morning analysis
            intraday_str = ""
            if intraday_brain_advice:
                intraday_str = (
                    f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 INTRADAY ANALYSIS\n"
                    f"Trend: {intraday_brain_advice.get('trend', 'N/A')}\n"
                    f"Direction: {intraday_brain_advice.get('direction', 'N/A')} ({intraday_brain_advice.get('confidence', 0)}%)\n"
                    f"Strategy: {intraday_brain_advice.get('strategy_for_now', 'N/A')}\n"
                    f"Reasoning: {intraday_brain_advice.get('reasoning', 'N/A')}\n"
                )
            
            send_telegram(
                f"🧠 Claude Brain V2 — {today_str}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📰 MORNING BRIEF\n"
                f"Risk: {brain_advice.get('risk_level', 'N/A')}\n"
                f"Sentiment: {brain_advice.get('news_sentiment', 'N/A')}\n"
                f"Max trades: {brain_advice.get('max_trades', 'N/A')}\n"
                f"Outlook: {brain_advice.get('market_outlook', '')}"
                f"{intraday_str}",
                config,
            )
            if brain_advice.get("max_trades"):
                config["capital"]["max_trades_per_day"] = brain_advice["max_trades"]

    # ── Dynamic stock selection (EXPANDED for intraday) ──
    if symbols:
        # If specific stocks given, still score them
        picks = select_best_stocks(symbols, config, target_date, top_n=len(symbols))
    else:
        # ══════════════════════════════════════════════════════════════
        # INTRADAY: Use larger universe for more opportunities
        # - Nifty 250 + FNO liquid stocks = ~300 stocks
        # - More stocks = more chances to find good setups
        # - ML model filters to best 15-20 candidates
        # ══════════════════════════════════════════════════════════════
        universe = get_universe("fno")  # Full universe: Nifty 250 + FNO liquid (~300 stocks)
        picks = select_best_stocks(universe, config, target_date, top_n=15)  # Increased from 10 to 15

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
    
    # ── NEWS-BASED STOCK ADJUSTMENT (Claude Brain V2) ──
    news_adjustment = None
    if brain.enabled and not is_backtest and not picks.empty:
        logger.info("  🧠 Analyzing news for stock adjustments...")
        
        # Get intraday metrics for context
        intraday_metrics = {
            "intraday_change_pct": NIFTY_CACHE.intraday_change_pct,
            "position_in_range": NIFTY_CACHE.position_in_range,
        }
        
        news_adjustment = brain.adjust_stocks_by_news(picks, intraday_metrics)
        
        if news_adjustment:
            # Apply skip list from news
            if news_adjustment.get("skip_stocks"):
                skip_news = news_adjustment["skip_stocks"]
                before = len(picks)
                picks = picks[~picks["symbol"].isin(skip_news)]
                if len(picks) < before:
                    logger.info(f"  📰 News skip: {skip_news}")
            
            # Apply direction flips from news
            if news_adjustment.get("flip_direction"):
                for sym, new_dir in news_adjustment["flip_direction"].items():
                    if sym in picks["symbol"].values:
                        picks.loc[picks["symbol"] == sym, "direction"] = new_dir
                        logger.info(f"  📰 News flip: {sym} → {new_dir}")
            
            # Log boost stocks
            if news_adjustment.get("boost_stocks"):
                logger.info(f"  📰 News boost: {news_adjustment['boost_stocks']}")
            
            # Log news summary
            if news_adjustment.get("news_summary"):
                logger.info(f"  📰 {news_adjustment['news_summary']}")
            
            # Send Telegram with news summary
            if news_adjustment.get("news_summary") and news_adjustment["news_summary"] != "No news analysis available":
                send_telegram(
                    f"📰 NEWS ANALYSIS — {today_str}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Sentiment: {news_adjustment.get('market_sentiment', 'neutral').upper()}\n"
                    f"Skip: {news_adjustment.get('skip_stocks', [])}\n"
                    f"Boost: {news_adjustment.get('boost_stocks', [])}\n"
                    f"Flip: {news_adjustment.get('flip_direction', {})}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{news_adjustment.get('news_summary', '')}",
                    config,
                )

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
        conf_str = f"{r.get('ml_confidence', 0)*100:.0f}%" if r.get('ml_confidence') else "N/A"
        trend_str = "↑" if r.get('trend_strength') == 'strong' else "→"
        pick_lines.append(
            f"{arrow} {r['symbol']:<10} ML:{r['ml_score']:.0f} Conf:{conf_str} "
            f"ATR:{r['atr_pct']*100:.1f}% {trend_str} {r['direction']}"
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

    # ── Create adaptive traders (with Claude Brain + ML features for intraday) ──
    traders = {}
    if is_backtest:
        # Backtest: only create traders for stocks with candles
        for _, pick in picks.iterrows():
            sym = pick["symbol"]
            if sym not in all_candles:
                continue
            # Extract ML features for the trader
            ml_features = {
                "ml_prob": pick.get("ml_prob", 0.5),
                "ml_confidence": pick.get("ml_confidence", 0.0),
                "volatility_ratio": pick.get("volatility_ratio", 1.0),
                "trend_strength": pick.get("trend_strength", "weak"),
                "optimal_window": pick.get("optimal_window", "any"),
                "adx": pick.get("adx", 25),
            }
            traders[sym] = AdaptiveV3Trader(
                sym, pick["direction"], config, ml_score=pick["ml_score"],
                ml_features=ml_features
            )
    else:
        # Live: create traders with Claude Brain + ML features for intraday decisions
        for _, pick in picks.iterrows():
            sym = pick["symbol"]
            ml_features = {
                "ml_prob": pick.get("ml_prob", 0.5),
                "ml_confidence": pick.get("ml_confidence", 0.0),
                "volatility_ratio": pick.get("volatility_ratio", 1.0),
                "trend_strength": pick.get("trend_strength", "weak"),
                "optimal_window": pick.get("optimal_window", "any"),
                "adx": pick.get("adx", 25),
            }
            traders[sym] = AdaptiveV3Trader(
                sym, pick["direction"], config, ml_score=pick["ml_score"],
                claude_brain=brain if brain.enabled else None,
                ml_features=ml_features
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

                # ── Claude Brain V2: Live INTRADAY adjustment every 15 min ──
                if brain.enabled:
                    try:
                        # Refresh Nifty cache for latest intraday data
                        NIFTY_CACHE.refresh()
                        
                        # Build intraday-focused state
                        live_state = {
                            "time": now_t.strftime("%H:%M"),
                            "vix": vix,
                            "day_pnl": total_pnl,
                            "trades_taken": closed,
                            # INTRADAY: Add position context for each trade
                            "open_positions": [
                                {
                                    "symbol": s, 
                                    "side": t.side, 
                                    "pnl": t.day_pnl, 
                                    "regime": t.current_regime,
                                    "entry_price": t.entry_price,
                                    "position_in_range": t.position_in_range,
                                    "intraday_change": t.intraday_change_pct,
                                }
                                for s, t in traders.items() if t.in_trade
                            ],
                            "stock_regimes": regimes,
                            # INTRADAY: Market context
                            "nifty_intraday": {
                                "change_pct": NIFTY_CACHE.intraday_change_pct,
                                "position_in_range": NIFTY_CACHE.position_in_range,
                                "bias": "LONG" if NIFTY_CACHE.position_in_range > 0.55 else "SHORT" if NIFTY_CACHE.position_in_range < 0.45 else "NEUTRAL",
                            },
                        }
                        
                        adj = brain.live_adjustment(live_state)
                        
                        # Handle emergency exits
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
                        
                        # Also get fresh intraday trend analysis
                        intraday_metrics = {
                            "nifty_open": NIFTY_CACHE.today_open,
                            "nifty_current": NIFTY_CACHE.current_price,
                            "nifty_high": NIFTY_CACHE.today_high,
                            "nifty_low": NIFTY_CACHE.today_low,
                            "intraday_change_pct": NIFTY_CACHE.intraday_change_pct,
                            "position_in_range": NIFTY_CACHE.position_in_range,
                            "vix": vix,
                            "time": now_t.strftime("%H:%M"),
                        }
                        intraday_trend = brain.analyze_intraday_trend(intraday_metrics)
                        
                        if intraday_trend and intraday_trend.get("direction") != "NEUTRAL":
                            logger.info(f"  🧠 Intraday Update: {intraday_trend.get('direction')} "
                                       f"({intraday_trend.get('confidence', 0)}%) - "
                                       f"{intraday_trend.get('reasoning', '')[:50]}")
                        
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
