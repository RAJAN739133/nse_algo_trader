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
MARKET_TREND = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL, MILD_BULLISH, MILD_BEARISH
ENABLE_LONGS = True       # Will be set based on market trend
ENABLE_SHORTS = True      # Will be set based on market trend
MAX_LONGS = 5             # Dynamic limit
MAX_SHORTS = 5            # Dynamic limit

# ══════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER - HARDCODED, NON-NEGOTIABLE SAFETY
# ══════════════════════════════════════════════════════════════════════
CIRCUIT_BREAKER_LOSS_PCT = 0.02   # Stop ALL trading after 2% daily loss
CIRCUIT_BREAKER_TRIGGERED = False  # Set True when triggered
DAILY_PNL = 0.0                    # Track daily P&L
DAILY_STARTING_CAPITAL = 0.0       # Capital at start of day

# ══════════════════════════════════════════════════════════════════════
# POSITION SIZING - Adapted for small capital (Rs 10K-50K)
# With tiny capital, we need larger position % per trade
# BUT limit total concurrent positions to control risk
# ══════════════════════════════════════════════════════════════════════
MAX_POSITION_PCT = 0.30            # Max 30% of MARGIN per trade (not capital)
MIN_POSITION_PCT = 0.10            # Min 10% per trade (avoid tiny positions)
MAX_CONCURRENT_POSITIONS = 2       # ONLY 2 positions at a time (risk control!)

# ══════════════════════════════════════════════════════════════════════
# INTRADAY MARGIN LEVERAGE
# Brokers provide 5x margin for MIS/intraday orders
# This is KEY for small capital profitability!
# ══════════════════════════════════════════════════════════════════════
INTRADAY_MARGIN_MULTIPLIER = 5.0   # 5x leverage for MIS orders
# Example: Rs 10,000 capital → Rs 50,000 buying power
# This means with Rs 10K, you can buy stocks worth Rs 50K

# ══════════════════════════════════════════════════════════════════════
# DYNAMIC CAPITAL MANAGER - Position size grows with profits!
# ══════════════════════════════════════════════════════════════════════
class DynamicCapitalManager:
    """
    Tracks actual capital including profits/losses.
    Position sizes automatically scale as capital grows.
    
    Key insight: With fixed 10K allocation, profits stay small.
    With dynamic sizing, a 50K capital uses 7.5K per trade (15%),
    but when it grows to 80K, trades become 12K automatically.
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.realized_pnl = 0.0
        self.peak_capital = initial_capital
        self.drawdown = 0.0
        
    def update_pnl(self, pnl: float):
        """Update capital after a trade closes."""
        self.realized_pnl += pnl
        self.current_capital = self.initial_capital + self.realized_pnl
        
        # Track peak for drawdown calculation
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Calculate drawdown from peak
        if self.peak_capital > 0:
            self.drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        logger.info(f"💰 Capital updated: ₹{self.current_capital:,.0f} "
                   f"(P&L: ₹{self.realized_pnl:+,.0f}, DD: {self.drawdown*100:.1f}%)")
    
    def get_position_size(self, risk_pct: float = 0.20) -> float:
        """
        Get position size based on CURRENT capital WITH MARGIN.
        
        Args:
            risk_pct: Percentage of MARGIN to use per trade (default 20%)
            
        Returns:
            Amount in rupees for this trade (includes margin leverage)
        """
        # Apply intraday margin (5x leverage)
        margin_buying_power = self.current_capital * INTRADAY_MARGIN_MULTIPLIER
        
        # Position value based on margin
        base_size = margin_buying_power * risk_pct
        
        # Minimum viable trade size (to cover brokerage and make meaningful profit)
        # With margin: 10K capital = 50K buying power, min trade = 10K
        min_size = max(10000, self.current_capital)
        
        # Maximum single trade (50% of margin to leave room for SL)
        max_size = margin_buying_power * 0.50
        
        position = max(min_size, min(base_size, max_size))
        
        logger.debug(f"Position size: ₹{position:,.0f} "
                    f"({risk_pct*100:.0f}% of ₹{margin_buying_power:,.0f} margin)")
        
        return position
    
    def get_buying_power(self) -> float:
        """Get total buying power including margin."""
        return self.current_capital * INTRADAY_MARGIN_MULTIPLIER
    
    def get_dynamic_risk_pct(self) -> float:
        """
        Adjust risk percentage based on performance.
        - Winning streak: Can increase slightly
        - Drawdown: Reduce risk to preserve capital
        """
        if self.drawdown > 0.10:  # 10%+ drawdown
            return 0.05  # Reduce to 5% per trade
        elif self.drawdown > 0.05:  # 5-10% drawdown
            return 0.08  # Reduce to 8%
        elif self.realized_pnl > self.initial_capital * 0.10:  # 10%+ profit
            return 0.12  # Can increase to 12%
        else:
            return 0.10  # Default 10%
    
    def should_reduce_exposure(self) -> bool:
        """Check if we should reduce position sizes due to drawdown."""
        return self.drawdown > 0.07  # 7% drawdown threshold
    
    def get_stats(self) -> dict:
        """Get capital statistics."""
        return {
            "initial": self.initial_capital,
            "current": self.current_capital,
            "realized_pnl": self.realized_pnl,
            "return_pct": (self.realized_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0,
            "peak": self.peak_capital,
            "drawdown_pct": self.drawdown * 100,
        }

# Global capital manager instance
CAPITAL_MANAGER = None

def get_capital_manager(initial_capital: float = None) -> DynamicCapitalManager:
    """Get or create the global capital manager."""
    global CAPITAL_MANAGER
    if CAPITAL_MANAGER is None and initial_capital is not None:
        CAPITAL_MANAGER = DynamicCapitalManager(initial_capital)
        logger.info(f"💰 Capital Manager initialized with ₹{initial_capital:,.0f}")
    return CAPITAL_MANAGER

# ══════════════════════════════════════════════════════════════════════
# KEY FIX: Trade WITH the market, not against it
# - In BULLISH market: Prioritize LONGs, limit SHORTs
# - In BEARISH market: DISABLE LONGs completely (not just limit!)
# - In MILD_BEARISH: DISABLE LONGs (conservative)
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
            result["enable_longs"] = False  # DISABLE LONGs completely in bearish
            result["enable_shorts"] = True
            result["max_longs"] = 0         # No LONGs in any bearish market
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
# CORRELATION TRACKING — Avoid correlated positions
# ════════════════════════════════════════════════════════════

class CorrelationTracker:
    """
    Track rolling correlations between stocks to avoid correlated positions.
    Pro traders use this to ensure true diversification.
    """
    def __init__(self, lookback_days=30, correlation_threshold=0.7):
        self.lookback_days = lookback_days
        self.threshold = correlation_threshold
        self.correlation_matrix = None
        self.last_update = None
        self._price_cache = {}
    
    def update_correlations(self, symbols, loader=None):
        """Calculate rolling correlation matrix for given symbols."""
        import yfinance as yf
        
        if loader is None:
            loader = DataLoader()
        
        # Get price data for all symbols
        prices = {}
        for sym in symbols[:50]:  # Limit to top 50 for speed
            try:
                if sym in self._price_cache:
                    prices[sym] = self._price_cache[sym]
                else:
                    df = yf.download(f"{sym}.NS", period=f"{self.lookback_days + 5}d", 
                                    interval="1d", progress=False)
                    if len(df) >= self.lookback_days // 2:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [c[0].lower() for c in df.columns]
                        prices[sym] = df['close'].pct_change().dropna()
                        self._price_cache[sym] = prices[sym]
            except:
                pass
        
        if len(prices) < 2:
            return
        
        # Create DataFrame with aligned dates
        price_df = pd.DataFrame(prices)
        price_df = price_df.dropna(axis=1, thresh=self.lookback_days // 2)
        
        if price_df.shape[1] >= 2:
            self.correlation_matrix = price_df.corr()
            self.last_update = datetime.now()
            logger.debug(f"Correlation matrix updated: {self.correlation_matrix.shape}")
    
    def get_correlation(self, sym1, sym2):
        """Get correlation between two symbols."""
        if self.correlation_matrix is None:
            return 0.0
        
        try:
            if sym1 in self.correlation_matrix.columns and sym2 in self.correlation_matrix.columns:
                return self.correlation_matrix.loc[sym1, sym2]
        except:
            pass
        return 0.0
    
    def is_correlated_with_portfolio(self, symbol, portfolio_symbols):
        """
        Check if symbol is highly correlated with any stock in portfolio.
        Returns (is_correlated, max_correlation, correlated_with)
        """
        if not portfolio_symbols:
            return False, 0.0, None
        
        max_corr = 0.0
        correlated_with = None
        
        for port_sym in portfolio_symbols:
            corr = abs(self.get_correlation(symbol, port_sym))
            if corr > max_corr:
                max_corr = corr
                correlated_with = port_sym
        
        is_correlated = max_corr >= self.threshold
        return is_correlated, max_corr, correlated_with
    
    def get_diversified_subset(self, candidates, max_stocks=10):
        """
        From a list of candidate stocks, select a diversified subset
        where no two stocks have correlation > threshold.
        """
        if self.correlation_matrix is None or len(candidates) == 0:
            return candidates[:max_stocks]
        
        selected = []
        for stock in candidates:
            sym = stock["symbol"] if isinstance(stock, dict) else stock
            
            is_corr, corr_val, corr_with = self.is_correlated_with_portfolio(
                sym, [s["symbol"] if isinstance(s, dict) else s for s in selected]
            )
            
            if not is_corr:
                selected.append(stock)
                if len(selected) >= max_stocks:
                    break
            else:
                logger.debug(f"  Skipping {sym}: {corr_val:.2f} correlation with {corr_with}")
        
        return selected


# Global correlation tracker
CORRELATION_TRACKER = CorrelationTracker(lookback_days=30, correlation_threshold=0.70)


# ════════════════════════════════════════════════════════════
# VOLUME-BASED SLIPPAGE MODEL — Realistic execution
# ════════════════════════════════════════════════════════════

def calculate_dynamic_slippage(symbol, order_size_pct, avg_daily_volume, spread_pct=0.0005):
    """
    Calculate expected slippage based on order size relative to volume.
    
    Pro algo traders use this to estimate execution costs.
    
    Args:
        symbol: Stock symbol
        order_size_pct: Order size as % of ADV (Average Daily Volume)
        avg_daily_volume: Stock's average daily volume
        spread_pct: Typical bid-ask spread (default 0.05%)
    
    Returns:
        Expected slippage as decimal (e.g., 0.001 = 0.1%)
    """
    # Base spread cost (always pay half the spread)
    base_slippage = spread_pct / 2
    
    # Market impact based on participation rate
    # Rule of thumb: sqrt(participation_rate) * impact_factor
    participation_rate = order_size_pct / 100  # e.g., 1% of daily volume
    
    if participation_rate < 0.001:  # < 0.1% of ADV
        impact = 0.0001  # 1 bps
    elif participation_rate < 0.01:  # < 1% of ADV
        impact = 0.0003 * np.sqrt(participation_rate * 100)  # ~3-10 bps
    elif participation_rate < 0.05:  # < 5% of ADV
        impact = 0.001 * np.sqrt(participation_rate * 20)  # ~10-30 bps
    else:  # > 5% of ADV (should avoid)
        impact = 0.003 * participation_rate * 10  # Very high impact
    
    total_slippage = base_slippage + impact
    
    # Cap at reasonable maximum
    return min(total_slippage, 0.01)  # Max 1% slippage


def get_max_order_size(symbol, avg_daily_volume, max_participation=0.02):
    """
    Calculate maximum order size to limit market impact.
    
    Pro rule: Don't exceed 1-2% of daily volume.
    """
    return int(avg_daily_volume * max_participation)


# ════════════════════════════════════════════════════════════
# EARNINGS CALENDAR — Skip stocks with upcoming results
# ════════════════════════════════════════════════════════════

def get_upcoming_earnings(symbols, days_ahead=5):
    """
    Check which stocks have earnings coming up.
    Skip these to avoid event risk.
    
    Returns list of symbols to skip.
    """
    # Note: In production, integrate with a financial calendar API
    # For now, we'll use a static check based on known patterns
    
    skip_stocks = []
    today = date.today()
    
    # Q4 FY26 earnings season: Apr-May 2026
    # Major companies report: TCS (Apr 10), Infy (Apr 14), HDFC Bank (Apr 20)
    # This is a placeholder - integrate with real API for production
    
    # Approximate earnings dates (would come from API in production)
    earnings_calendar = {
        # Format: "SYMBOL": (month, day) for Q4 FY26
        "TCS": (4, 10),
        "INFY": (4, 14),
        "WIPRO": (4, 18),
        "HDFCBANK": (4, 20),
        "ICICIBANK": (4, 22),
        "AXISBANK": (4, 25),
        "RELIANCE": (4, 28),
        "BAJFINANCE": (4, 22),
        # Add more as needed
    }
    
    for sym in symbols:
        if sym in earnings_calendar:
            earn_month, earn_day = earnings_calendar[sym]
            try:
                earn_date = date(today.year, earn_month, earn_day)
                days_to_earnings = (earn_date - today).days
                
                if 0 <= days_to_earnings <= days_ahead:
                    skip_stocks.append(sym)
                    logger.info(f"  ⚠️ Skipping {sym}: Earnings in {days_to_earnings} days")
            except:
                pass
    
    return skip_stocks


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

    # Step 2: ML scoring — try models in order of preference
    model, feats = None, None
    ml_pipeline = None
    
    # Model paths in order of preference
    mp_new = Path("models/trading_model.pkl")  # New ML pipeline model
    mp_v2 = Path("models/stock_predictor_v2.pkl")
    mp_v1 = Path("models/stock_predictor.pkl")
    
    # Try new ML pipeline first (if available)
    if mp_new.exists():
        try:
            from ml.data_pipeline import MLDataPipeline
            ml_pipeline = MLDataPipeline()
            if ml_pipeline.load_model("trading_model"):
                logger.info(f"  Using NEW ML pipeline model (52 features)")
            else:
                ml_pipeline = None
        except Exception as e:
            logger.debug(f"  New ML pipeline not available: {e}")
            ml_pipeline = None
    
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
        
        # ══════════════════════════════════════════════════════════════
        # DYNAMIC CAPITAL: Use current capital (includes profits!)
        # Not the static config value that never changes
        # ══════════════════════════════════════════════════════════════
        initial_capital = config["capital"]["total"]
        cap_mgr = get_capital_manager(initial_capital)
        if cap_mgr:
            self.capital = cap_mgr.current_capital  # Use CURRENT capital with profits!
            logger.debug(f"Using dynamic capital: ₹{self.capital:,.0f} (initial: ₹{initial_capital:,.0f})")
        else:
            self.capital = initial_capital
        
        self.config = config
        self.cost_model = AngelOneCostModel()
        
        # Claude Brain dynamic parameters (can be adjusted during trading)
        self.position_size_factor = config.get("position_size_factor", 1.0)
        self.risk_per_trade = config["capital"].get("risk_per_trade", 0.02)
        
        # ══════════════════════════════════════════════════════════════
        # DYNAMIC PROFIT BOOKING - ML + Claude Driven with Safety Bounds
        # ══════════════════════════════════════════════════════════════
        # PHILOSOPHY: Let ML and Claude optimize, but with guardrails
        # 
        # 1. ML analyzes historical patterns → suggests base values
        # 2. Claude researches current market → adjusts for conditions
        # 3. Hardcoded bounds → prevent dangerous values
        # 4. Circuit breaker → always enforced (non-negotiable)
        # ══════════════════════════════════════════════════════════════
        
        # SAFETY BOUNDS (non-negotiable, based on backtesting)
        # For small capital (Rs 10K-50K), position_pct is higher
        BOUNDS = {
            "quick_profit": (0.005, 0.02),   # 0.5% to 2.0%
            "stop": (0.003, 0.015),          # 0.3% to 1.5%
            "cut_loss": (0.002, 0.008),      # 0.2% to 0.8%
            "target": (0.008, 0.025),        # 0.8% to 2.5%
            "trail_activation": (0.003, 0.015),
            "time_decay": (10, 30),          # 50 min to 150 min
            "position_pct": (0.05, 0.15),    # 5% to 15% per trade (for small capital)
        }
        
        # DEFAULTS (used if ML/Claude don't provide values)
        DEFAULTS = {
            "quick_profit": 0.008,   # 0.8%
            "stop": 0.005,           # 0.5%
            "cut_loss": 0.003,       # 0.3%
            "target": 0.012,         # 1.2%
            "trail_activation": 0.005,
            "time_decay": 16,
            "position_pct": 0.10,    # 10% per trade (for small capital)
        }
        
        # Get values from ML features (if available)
        ml_tweaks = {}
        if ml_features:
            # ML can suggest values based on volatility and trend strength
            volatility = ml_features.get("volatility_ratio", 1.0)
            trend_strength = ml_features.get("trend_strength", "weak")
            
            # Higher volatility → tighter stops, faster profit booking
            if volatility > 1.5:
                ml_tweaks["quick_profit"] = 0.006  # Book faster in volatile market
                ml_tweaks["stop"] = 0.004          # Tighter stop
                ml_tweaks["position_pct"] = 0.02   # Smaller position
            elif volatility < 0.7:
                ml_tweaks["quick_profit"] = 0.012  # Can wait longer in calm market
                ml_tweaks["stop"] = 0.006
                ml_tweaks["position_pct"] = 0.04
            
            # Strong trend → wider targets
            if trend_strength == "strong":
                ml_tweaks["target"] = 0.018
                ml_tweaks["time_decay"] = 24  # Hold longer in strong trend
        
        # Get values from Claude Brain (if available)
        tweaks = config.get("strategy_tweaks", {})
        
        # Merge: Claude > ML > Defaults (with bounds enforced)
        def get_bounded(key, default_key=None):
            dk = default_key or key.replace("_pct", "")
            # Priority: tweaks (Claude) > ml_tweaks (ML) > DEFAULTS
            value = tweaks.get(f"{key}", ml_tweaks.get(dk, DEFAULTS.get(dk, 0)))
            min_v, max_v = BOUNDS.get(dk, (0, 1))
            return max(min_v, min(max_v, value))
        
        self.target_pct = get_bounded("target_pct", "target")
        self.stop_pct = get_bounded("stop_pct", "stop")
        self.quick_profit_pct = get_bounded("quick_profit_pct", "quick_profit")
        self.cut_loss_pct = get_bounded("cut_loss_pct", "cut_loss")
        self.trailing_activation_pct = get_bounded("trailing_activation_pct", "trail_activation")
        self.time_decay_candles = int(get_bounded("time_decay_candles", "time_decay"))
        self.max_position_pct = get_bounded("position_pct", "position_pct")
        self.orb_atr_multiplier = tweaks.get("orb_atr_multiplier", 1.5)
        self.strategy = ProStrategyV2(config.get("strategies", {}).get("pro", {}))
        self.regime_detector = RegimeDetector()
        self.pattern_detector = CandlePatternDetector()  # Candlestick patterns
        self.claude_brain = claude_brain  # Optional: Claude Brain for entry confirmation
        
        # Log the profit booking settings being used
        logger.info(f"  {symbol} Profit Settings: "
                   f"QuickProfit={self.quick_profit_pct*100:.1f}% | "
                   f"Target={self.target_pct*100:.1f}% | "
                   f"Stop={self.stop_pct*100:.1f}% | "
                   f"CutLoss={self.cut_loss_pct*100:.1f}%")
        
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
        
        # Gap trading state
        self.prev_close = None  # Set from daily data before trading starts
        self.gap_pct = 0.0      # Today's gap percentage
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
        
        # Calculate holding time
        try:
            entry_dt = pd.to_datetime(self.entry_time)
            exit_dt = pd.to_datetime(exit_time)
            holding_minutes = int((exit_dt - entry_dt).total_seconds() / 60)
        except:
            holding_minutes = 0

        trade_record = {
            "symbol": self.symbol, "direction": self.side,
            "type": self.trade_type, "regime": self.entry_regime,
            "entry": round(self.entry_price, 2), "exit": round(exit_price, 2),
            "entry_time": str(self.entry_time).split("+")[0],
            "exit_time": str(exit_time).split("+")[0],
            "sl": round(self.sl, 2), "tgt": round(self.tgt, 2),
            "qty": shares, "gross": round(gross, 2),
            "costs": round(costs, 2), "net_pnl": round(net, 2),
            "reason": reason,
            "holding_minutes": holding_minutes,
        }
        self.trades.append(trade_record)
        
        # ══════════════════════════════════════════════════════════════
        # UPDATE DYNAMIC CAPITAL - Position sizes grow with profits!
        # ══════════════════════════════════════════════════════════════
        cap_mgr = get_capital_manager()
        if cap_mgr:
            cap_mgr.update_pnl(net)
            # Update self.capital for next trade's position sizing
            self.capital = cap_mgr.current_capital
            stats = cap_mgr.get_stats()
            logger.info(f"📊 Capital: ₹{stats['current']:,.0f} | "
                       f"Total P&L: ₹{stats['realized_pnl']:+,.0f} ({stats['return_pct']:+.1f}%) | "
                       f"DD: {stats['drawdown_pct']:.1f}%")
        
        # Log to paper trading tracker (only in paper mode)
        if self.config.get("trading", {}).get("mode") == "paper":
            try:
                from paper_trading_tracker import PaperTradingTracker
                tracker = PaperTradingTracker()
                tracker.log_trade({
                    "symbol": self.symbol,
                    "side": self.side,
                    "strategy": self.trade_type,
                    "entry_price": self.entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "gross_pnl": gross,
                    "costs": costs,
                    "net_pnl": net,
                    "holding_minutes": holding_minutes,
                    "exit_reason": reason,
                    "regime": self.entry_regime,
                    "vix": getattr(self, 'vix_at_entry', None),
                })
            except Exception as e:
                logger.debug(f"Paper tracker log failed: {e}")

        self.day_pnl += net
        
        # ══════════════════════════════════════════════════════════════
        # CIRCUIT BREAKER CHECK - Update daily PnL and check threshold
        # ══════════════════════════════════════════════════════════════
        global DAILY_PNL, CIRCUIT_BREAKER_TRIGGERED
        DAILY_PNL += net
        
        # Check if circuit breaker should trigger
        if DAILY_STARTING_CAPITAL > 0:
            daily_loss_pct = -DAILY_PNL / DAILY_STARTING_CAPITAL
            if daily_loss_pct >= CIRCUIT_BREAKER_LOSS_PCT and not CIRCUIT_BREAKER_TRIGGERED:
                CIRCUIT_BREAKER_TRIGGERED = True
                cb_msg = (
                    f"🛑 CIRCUIT BREAKER TRIGGERED!\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📉 Daily Loss: ₹{DAILY_PNL:,.2f} ({daily_loss_pct*100:.1f}%)\n"
                    f"⛔ All new trades STOPPED\n"
                    f"📊 Capital: ₹{DAILY_STARTING_CAPITAL:,.0f}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚠️ Take a break. Review what went wrong."
                )
                logger.warning(f"\n{'='*60}\n{cb_msg}\n{'='*60}")
                send_telegram(cb_msg, self.config)

        emoji = "✅" if net > 0 else "❌"
        et_s = str(self.entry_time).split(" ")[-1].split("+")[0][:8]  # HH:MM:SS
        xt_s = str(exit_time).split(" ")[-1].split("+")[0][:8]
        action_entry = "BOUGHT" if self.side == "LONG" else "SOLD SHORT"
        action_exit = "SOLD" if self.side == "LONG" else "COVERED"
        
        hold_str = f"{holding_minutes} min" if holding_minutes > 0 else "N/A"
        
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
        # ══════════════════════════════════════════════════════════════
        # CIRCUIT BREAKER CHECK - Block all new entries if triggered
        # ══════════════════════════════════════════════════════════════
        if CIRCUIT_BREAKER_TRIGGERED:
            logger.info(f"  ⛔ {self.symbol} BLOCKED: Circuit breaker triggered (daily loss > {CIRCUIT_BREAKER_LOSS_PCT*100:.0f}%)")
            return
        
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

        # ══════════════════════════════════════════════════════════════
        # DYNAMIC POSITION SIZING (Claude Brain + ML + Regime)
        # ══════════════════════════════════════════════════════════════
        
        # Start with Claude Brain's dynamic risk (or config default)
        risk_pct = self.risk_per_trade  # Set by Claude Brain based on VIX/market
        
        # Apply Claude Brain's position size factor (market-adaptive)
        risk_pct *= self.position_size_factor
        
        # Regime-based adjustment
        if self.current_regime == "volatile":
            risk_pct *= 0.5
        elif self.current_regime == "choppy":
            risk_pct *= 0.5
        elif self.current_regime == "ranging":
            risk_pct *= 0.4
        
        # Afternoon trades are riskier
        if "AFTERNOON" in self.trade_type:
            risk_pct *= 0.6
        
        # ML-based position sizing adjustment
        if self.ml_confidence >= 0.20:
            risk_pct *= 1.2
            signal["reason"] = f"{signal.get('reason', '')} | ML conf +20%"
        elif self.ml_confidence >= 0.15:
            risk_pct *= 1.1
        elif self.ml_confidence < 0.10:
            risk_pct *= 0.8
        
        # Strong trend (ADX > 30) = larger position in trend direction
        if self.adx > 30 and self.trend_strength == "strong":
            risk_pct *= 1.15
            signal["reason"] = f"{signal.get('reason', '')} | ADX {self.adx:.0f}"
        
        # High volatility ratio = smaller position (risk management)
        if self.volatility_ratio > 1.5:
            risk_pct *= 0.7
        elif self.volatility_ratio < 0.7:
            risk_pct *= 1.1
        
        # ══════════════════════════════════════════════════════════════
        # DYNAMIC POSITION SIZING WITH INTRADAY MARGIN (5x LEVERAGE)
        # 
        # Key insight for small capital profitability:
        # - Capital: Rs 10,000
        # - Intraday Margin: 5x = Rs 50,000 buying power
        # - Position: 30% of margin = Rs 15,000 worth of shares
        # - Stock at Rs 1,500: 15,000 / 1,500 = 10 shares (not 1!)
        # 
        # This is how real intraday trading works!
        # ══════════════════════════════════════════════════════════════
        cap_mgr = get_capital_manager()
        if cap_mgr:
            current_capital = cap_mgr.current_capital
            dynamic_risk_pct = cap_mgr.get_dynamic_risk_pct()
            risk_pct = min(risk_pct, dynamic_risk_pct, self.max_position_pct)
            
            if cap_mgr.should_reduce_exposure():
                risk_pct *= 0.7
                logger.warning(f"⚠️ Drawdown mode: reducing position to {risk_pct*100:.1f}%")
        else:
            current_capital = self.capital
            risk_pct = min(risk_pct, self.max_position_pct)
        
        risk_pct = max(risk_pct, MIN_POSITION_PCT)
        
        # ══════════════════════════════════════════════════════════════
        # APPLY INTRADAY MARGIN LEVERAGE (5x)
        # This is the KEY fix for small capital!
        # ══════════════════════════════════════════════════════════════
        margin_buying_power = current_capital * INTRADAY_MARGIN_MULTIPLIER
        position_value = margin_buying_power * risk_pct  # Value of shares to buy
        
        # Calculate shares based on MARGIN buying power (not just capital!)
        self.shares = max(1, int(position_value / self.entry_price))
        
        # Safety check: Limit to what margin can actually cover
        # Margin required per share is typically 20% of price for MIS
        margin_per_share = self.entry_price * 0.20  # 20% margin requirement
        max_shares_by_margin = int(current_capital / margin_per_share)
        self.shares = min(self.shares, max(1, max_shares_by_margin))
        
        logger.info(f"📊 Position Sizing: Capital ₹{current_capital:,.0f} × {INTRADAY_MARGIN_MULTIPLIER}x margin "
                   f"= ₹{margin_buying_power:,.0f} buying power")
        logger.info(f"   → {risk_pct*100:.0f}% = ₹{position_value:,.0f} → {self.shares} shares @ ₹{self.entry_price:,.2f}")
        
        # Use broker's margin calculator if available
        # Margin usage is also adjusted by Claude Brain's position_size_factor
        base_margin_pct = 0.40  # Base: 40% of capital per trade
        available_margin = self.capital * base_margin_pct * self.position_size_factor
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
        
        # ══════════════════════════════════════════════════════════
        # PROFIT BOOKING SYSTEM - Based on backtest optimization
        # KEY INSIGHT: Quick profit booking at 1% turns losses into profits!
        # ══════════════════════════════════════════════════════════
        # RULES:
        # 1. Book profit IMMEDIATELY when we hit quick_profit_pct (1%)
        # 2. Cut losers early (0.5%) before they become big losses
        # 3. Don't hold forever hoping for bigger moves
        # ══════════════════════════════════════════════════════════
        
        # PHASE 0: IMMEDIATE PROFIT CHECK (every candle after entry)
        # This is the KEY change - check profit on EVERY candle, not just after 4
        if holding_candles >= 2:  # Give at least 2 candles (10 min) for position to develop
            if pct_from_entry >= self.quick_profit_pct:
                self._close_trade(c, t, "QUICK_PROFIT")
                self.cooldown_until = i + 4  # Short cooldown, look for next opportunity
                return
        
        # PHASE 1: Early profit check - even smaller profits are good
        if holding_candles >= 3:
            if pct_from_entry >= self.quick_profit_pct * 0.7:  # 70% of target = ~0.7%
                self._close_trade(c, t, "EARLY_PROFIT")
                self.cooldown_until = i + 5
                return
        
        # PHASE 2: Cut losers early (before they become big losses)
        if holding_candles >= 4:
            if pct_from_entry < -self.cut_loss_pct:
                self._close_trade(c, t, "CUT_LOSER")
                self.cooldown_until = i + 6
                return
        
        # PHASE 3: Tighter management - exit flat trades
        if holding_candles >= 8:  # 40 minutes
            if pct_from_entry < self.quick_profit_pct * 0.3:  # Less than 0.3% profit
                self._close_trade(c, t, "TIME_DECAY")
                self.cooldown_until = i + 5
                return
        
        # PHASE 4: Force exit at max holding time
        if holding_candles >= self.time_decay_candles:
            reason = "PROFIT_LOCK" if pct_from_entry > 0 else "TIME_DECAY"
            self._close_trade(c, t, reason)
            self.cooldown_until = i + 4
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
            # Uses dynamic trailing_activation_pct from Claude Brain
            # ══════════════════════════════════════════════════════════
            unrealised_pct = (c - self.entry_price) / self.entry_price
            
            # Stage 1: Move to breakeven after activation threshold
            if unrealised_pct >= self.trailing_activation_pct and not self.sl_moved_to_be:
                self.sl = self.entry_price + 0.10  # Just above entry
                self.sl_moved_to_be = True
                logger.info(f"  {self.symbol} SL -> breakeven Rs {self.sl:,.2f}")
            
            # Stage 2: Trail stop behind price after 1.4x activation (dynamic)
            trail_threshold = self.trailing_activation_pct * 1.4
            if unrealised_pct >= trail_threshold:
                trail_distance = self.trailing_activation_pct * 0.4  # Trail 40% of activation
                new_sl = c * (1 - trail_distance)
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
            # Uses dynamic trailing_activation_pct from Claude Brain
            # ══════════════════════════════════════════════════════════
            unrealised_pct = (self.entry_price - c) / self.entry_price
            
            # Stage 1: Move to breakeven after activation threshold
            if unrealised_pct >= self.trailing_activation_pct and not self.sl_moved_to_be:
                self.sl = self.entry_price - 0.10  # Just below entry
                self.sl_moved_to_be = True
                logger.info(f"  {self.symbol} SL -> breakeven Rs {self.sl:,.2f}")
            
            # Stage 2: Trail stop above price after 1.4x activation (dynamic)
            trail_threshold = self.trailing_activation_pct * 1.4
            if unrealised_pct >= trail_threshold:
                trail_distance = self.trailing_activation_pct * 0.4  # Trail 40% of activation
                new_sl = c * (1 + trail_distance)
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
        # ══════════════════════════════════════════════════════════════
        # STRATEGY 0: GAP TRADING (9:15-10:00)
        # First hour gap plays - continuation or fill
        # Uses Claude Brain for gap analysis
        # ══════════════════════════════════════════════════════════════
        if hour == 9 and minute >= 15 and minute <= 45:
            if "GAP" not in avoid_strategies:
                gap_signal = self._generate_gap_signal(candles, i, direction)
                if gap_signal:
                    self._enter_trade(gap_signal, i)
                    return
        
        # ══════════════════════════════════════════════════════════════
        # STRATEGY 1: ORB BREAKOUT (9:45-10:30)
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
        # DYNAMIC TARGETS (Claude Brain or VIX fallback)
        # target_pct and stop_pct come from Claude Brain via config
        # ══════════════════════════════════════════════════════════
        target_pct = self.target_pct  # From Claude Brain
        atr_mult = self.orb_atr_multiplier  # From Claude Brain
        
        # MUST align with real-time market direction
        if direction == "LONG" and price_change > 0.005:
            atr = self._quick_atr(candles, i)
            sl = close - atr * atr_mult
            risk = close - sl
            tgt = close * (1 + target_pct)
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "MOMENTUM_LONG",
                "reason": f"Momentum +{price_change*100:.1f}% with vol, VIX={VIX_LEVEL:.0f}",
                "candles": candles
            }
        elif direction == "SHORT" and price_change < -0.005:
            atr = self._quick_atr(candles, i)
            sl = close + atr * atr_mult
            risk = sl - close
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

        # Use dynamic ATR multiplier from Claude Brain (slightly tighter for afternoon)
        afternoon_atr_mult = self.orb_atr_multiplier * 0.8  # 80% of normal for tighter stops
        
        if direction == "LONG" and trend_up and rsi > 55 and close > vwap:
            atr = self._quick_atr(candles, i)
            sl = close - atr * afternoon_atr_mult
            risk = close - sl
            tgt = close + risk * 1.5
            return {
                "side": "LONG", "entry": close, "sl": sl, "tgt": tgt,
                "risk": risk, "time": t, "type": "AFTERNOON_TREND_LONG",
                "reason": f"5-candle uptrend + vol + above VWAP, RSI={rsi:.0f}"
            }

        if direction == "SHORT" and trend_down and rsi < 45 and close < vwap:
            atr = self._quick_atr(candles, i)
            sl = close + atr * afternoon_atr_mult
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
    
    def _generate_gap_signal(self, candles, i, direction):
        """
        Generate gap trading signal in the first 30 minutes.
        
        Gap Types:
        1. GAP_CONTINUATION: Trade with the gap direction (breakaway/continuation gaps)
        2. GAP_FILL: Fade the gap (common/exhaustion gaps)
        
        Uses Claude Brain for analysis when available.
        """
        if i < 3:
            return None
        
        row = candles.iloc[i]
        close = row["close"]
        t = pd.to_datetime(row["datetime"])
        today_open = candles["open"].iloc[0]
        
        # Calculate gap from previous close (stored at init)
        if not hasattr(self, 'prev_close') or self.prev_close is None:
            # Estimate prev_close from first candle if not stored
            first_open = candles["open"].iloc[0]
            # If gap > 0.3%, consider it a real gap
            self.prev_close = first_open  # Will be set properly in run()
            return None
        
        gap_pct = (today_open - self.prev_close) / self.prev_close
        abs_gap = abs(gap_pct)
        
        # Skip micro gaps
        if abs_gap < 0.005:  # < 0.5%
            return None
        
        # Determine gap direction
        gap_up = gap_pct > 0
        
        # Current price position relative to gap
        filled_pct = 0
        if gap_up:
            # For gap up: filled_pct = how much has price dropped from open
            filled_pct = (today_open - close) / (today_open - self.prev_close) if today_open != self.prev_close else 0
        else:
            # For gap down: filled_pct = how much has price risen from open
            filled_pct = (close - today_open) / (self.prev_close - today_open) if today_open != self.prev_close else 0
        
        # Volume analysis
        avg_vol = candles["volume"].iloc[:i].mean() if i > 0 else candles["volume"].iloc[0]
        curr_vol = candles["volume"].iloc[i]
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
        
        # First 15 min trend
        first_candles = candles.iloc[:min(i+1, 3)]
        first_trend = "UP" if first_candles["close"].iloc[-1] > first_candles["open"].iloc[0] else "DOWN"
        
        signal = None
        
        # ══════════════════════════════════════════════════════════
        # GAP CONTINUATION (Trade WITH the gap)
        # - Large gap (>1%) + Volume + Price holding near open
        # ══════════════════════════════════════════════════════════
        if abs_gap >= 0.01 and vol_ratio >= 1.5 and filled_pct < 0.3:
            if gap_up and direction == "LONG" and first_trend == "UP":
                # Gap up continuation - go LONG
                atr = self._quick_atr(candles, i)
                sl = min(self.prev_close, today_open - atr * self.orb_atr_multiplier)
                target_pct = self.target_pct * 1.5  # Wider target for gap plays
                tgt = close * (1 + target_pct)
                
                signal = {
                    "side": "LONG",
                    "entry": close,
                    "sl": sl,
                    "tgt": tgt,
                    "risk": close - sl,
                    "time": t,
                    "type": "GAP_CONTINUATION_LONG",
                    "reason": f"Gap +{gap_pct*100:.1f}% holding, vol {vol_ratio:.1f}x"
                }
            
            elif not gap_up and direction == "SHORT" and first_trend == "DOWN":
                # Gap down continuation - go SHORT
                atr = self._quick_atr(candles, i)
                sl = max(self.prev_close, today_open + atr * self.orb_atr_multiplier)
                target_pct = self.target_pct * 1.5
                tgt = close * (1 - target_pct)
                
                signal = {
                    "side": "SHORT",
                    "entry": close,
                    "sl": sl,
                    "tgt": tgt,
                    "risk": sl - close,
                    "time": t,
                    "type": "GAP_CONTINUATION_SHORT",
                    "reason": f"Gap {gap_pct*100:.1f}% holding, vol {vol_ratio:.1f}x"
                }
        
        # ══════════════════════════════════════════════════════════
        # GAP FILL (Fade the gap)
        # - Smaller gap (0.5-1.5%) + Weak volume + Already filling
        # ══════════════════════════════════════════════════════════
        elif 0.005 <= abs_gap <= 0.015 and vol_ratio < 1.3 and 0.2 <= filled_pct <= 0.6:
            if gap_up and direction == "SHORT":
                # Fading gap up - go SHORT for gap fill
                atr = self._quick_atr(candles, i)
                sl = today_open + atr * 0.5  # Tight stop above today's open
                tgt = self.prev_close  # Target = gap fill
                
                # Only if gap fill target gives good R:R
                risk = sl - close
                reward = close - tgt
                if reward > risk * 1.5:
                    signal = {
                        "side": "SHORT",
                        "entry": close,
                        "sl": sl,
                        "tgt": tgt,
                        "risk": risk,
                        "time": t,
                        "type": "GAP_FILL_SHORT",
                        "reason": f"Fading +{gap_pct*100:.1f}% gap, {filled_pct*100:.0f}% filled"
                    }
            
            elif not gap_up and direction == "LONG":
                # Fading gap down - go LONG for gap fill
                atr = self._quick_atr(candles, i)
                sl = today_open - atr * 0.5  # Tight stop below today's open
                tgt = self.prev_close  # Target = gap fill
                
                risk = close - sl
                reward = tgt - close
                if reward > risk * 1.5:
                    signal = {
                        "side": "LONG",
                        "entry": close,
                        "sl": sl,
                        "tgt": tgt,
                        "risk": risk,
                        "time": t,
                        "type": "GAP_FILL_LONG",
                        "reason": f"Fading {gap_pct*100:.1f}% gap, {filled_pct*100:.0f}% filled"
                    }
        
        return signal


# ════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════

def run(symbols=None, backtest_date=None):
    global MARKET_TREND, ENABLE_LONGS, ENABLE_SHORTS, MAX_LONGS, MAX_SHORTS
    global CIRCUIT_BREAKER_TRIGGERED, DAILY_PNL, DAILY_STARTING_CAPITAL
    global MIN_ML_CONFIDENCE, BLOCKED_REGIMES
    
    config = load_config()
    target_date = date.fromisoformat(backtest_date) if backtest_date else None
    is_backtest = target_date is not None
    today_str = str(target_date or date.today())
    
    # ══════════════════════════════════════════════════════════════
    # CIRCUIT BREAKER RESET AT START OF DAY
    # ══════════════════════════════════════════════════════════════
    CIRCUIT_BREAKER_TRIGGERED = False
    DAILY_PNL = 0.0
    DAILY_STARTING_CAPITAL = config['capital']['total']

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  LIVE PAPER TRADER V3 — ADAPTIVE")
    logger.info(f"  Date: {today_str} {'(BACKTEST)' if is_backtest else '(LIVE)'}")
    logger.info(f"  Capital: Rs {config['capital']['total']:,}")
    logger.info(f"  Circuit Breaker: Stops at {CIRCUIT_BREAKER_LOSS_PCT * 100:.0f}% daily loss")
    logger.info(f"  Max Position Size: {MAX_POSITION_PCT * 100:.0f}% per trade")
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
            # Apply Claude Brain's dynamic parameters
            if brain_advice.get("max_trades"):
                config["capital"]["max_trades_per_day"] = brain_advice["max_trades"]
                logger.info(f"  🧠 Claude set max_trades = {brain_advice['max_trades']}")
            
            if brain_advice.get("risk_per_trade_pct"):
                config["capital"]["risk_per_trade"] = brain_advice["risk_per_trade_pct"]
                logger.info(f"  🧠 Claude set risk_per_trade = {brain_advice['risk_per_trade_pct']*100:.1f}%")
            
            if brain_advice.get("position_size_factor"):
                config["position_size_factor"] = brain_advice["position_size_factor"]
                logger.info(f"  🧠 Claude set position_size_factor = {brain_advice['position_size_factor']:.1f}x")
            
            # Apply strategy tweaks
            tweaks = brain_advice.get("strategy_tweaks", {})
            if tweaks:
                config["strategy_tweaks"] = tweaks
                logger.info(f"  🧠 Claude set target={tweaks.get('target_pct',0)*100:.1f}%, "
                           f"stop={tweaks.get('stop_pct',0)*100:.1f}%, "
                           f"quick_profit={tweaks.get('quick_profit_pct',0)*100:.1f}%")
            
            # Apply entry filters (direction control)
            filters = brain_advice.get("entry_filters", {})
            if filters:
                # Note: global declarations are at function top
                if "enable_longs" in filters:
                    ENABLE_LONGS = filters["enable_longs"]
                if "enable_shorts" in filters:
                    ENABLE_SHORTS = filters["enable_shorts"]
                if "max_longs" in filters:
                    MAX_LONGS = filters["max_longs"]
                if "max_shorts" in filters:
                    MAX_SHORTS = filters["max_shorts"]
                if "min_ml_confidence" in filters:
                    MIN_ML_CONFIDENCE = filters["min_ml_confidence"]
                if "blocked_regimes" in filters:
                    BLOCKED_REGIMES = filters["blocked_regimes"]
                logger.info(f"  🧠 Claude set LONGS={ENABLE_LONGS}(max {MAX_LONGS}), "
                           f"SHORTS={ENABLE_SHORTS}(max {MAX_SHORTS}), "
                           f"ML_conf={MIN_ML_CONFIDENCE*100:.0f}%")

    # ══════════════════════════════════════════════════════════════
    # GAP TRADING & EVENT ANALYSIS (9:10 AM - After Opening Prices)
    # ══════════════════════════════════════════════════════════════
    gap_signals = []
    event_warnings = {"avoid": [], "caution": [], "opportunities": []}
    
    if not is_backtest:
        try:
            from core.event_calendar import get_event_warnings, get_morning_gap_signals, GapAnalyzer
            
            # Get event warnings (earnings, RBI, etc.)
            if symbols:
                event_warnings = get_event_warnings(symbols)
            else:
                # Get warnings for popular stocks
                popular = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
                          "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC"]
                event_warnings = get_event_warnings(popular)
            
            if event_warnings.get("avoid"):
                logger.info(f"  ⚠️ AVOID (earnings today): {event_warnings['avoid']}")
            if event_warnings.get("caution"):
                logger.info(f"  ⚡ CAUTION (RBI impact): {event_warnings['caution'][:5]}")
            if event_warnings.get("opportunities"):
                logger.info(f"  💰 POST-EARNINGS opportunities: {event_warnings['opportunities']}")
            
            # Analyze overnight gaps after market opens (9:15+)
            # This would be called with real opening data in live trading
            logger.info("  📊 Gap analysis will run after 9:15 AM with opening prices")
            
        except ImportError as e:
            logger.debug(f"Event calendar not available: {e}")
        except Exception as e:
            logger.warning(f"Gap/event analysis failed: {e}")

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
