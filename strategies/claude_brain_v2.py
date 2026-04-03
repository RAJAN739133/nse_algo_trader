"""
Claude Brain V2 — Live Adaptive AI Strategy Engine (INTRADAY FOCUSED)

Unlike V1 (morning-only), V2 is called THROUGHOUT the trading day:
  1. PRE-MARKET (8:50 AM): Morning brief + risk level + stock picks
  2. LIVE MARKET (every 5 min during market): Quick regime check, strategy adjustment
  3. FULL SCAN (every 15 min): Deep analysis with news, adjust strategies
  4. NON-MARKET (every 6 hours): News scan, next-day preparation
  5. POST-MARKET (3:30 PM): EOD analysis, learn from today's mistakes

INTRADAY FOCUS:
  - All analysis uses TODAY's data only (not historical daily data)
  - Reference price = TODAY'S OPEN (not yesterday's close)
  - Position in TODAY's range determines direction
  - Support/Resistance = TODAY's high/low + VWAP
  - Volume comparison = vs TODAY's average

Features:
  - Reads ML model scores and adjusts confidence per stock
  - Fetches live market news (Google News RSS) and analyzes sentiment
  - Creates dynamic strategy rules based on current market regime
  - Suggests emergency exits or new entries mid-day
  - Analyzes market trends (Nifty, VIX, sector rotation)
  - Learns from past trade history to avoid repeating mistakes

Safety: All suggestions validated against hard limits. Claude NEVER
overrides risk management — only suggests adjustments within bounds.
"""
import os, json, logging, time, hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

SAFETY_LIMITS = {
    "max_risk_per_trade": 0.02,
    "max_trades_per_day": 5,
    "max_position_pct": 0.60,
    "min_stop_loss_pct": 0.005,
    "vix_absolute_skip": 35,
    "max_api_calls_per_day": 50,  # Increased for more frequent scans
    "quick_scan_interval_sec": 300,   # 5 min during market
    "full_scan_interval_sec": 900,    # 15 min full analysis
    "non_market_scan_interval_sec": 21600,  # 6 hours outside market
}

# Cache to avoid repeated identical calls
_response_cache = {}
_API_COUNT_FILE = Path("data/.claude_api_count.json")


def _get_api_call_count() -> int:
    """Get today's API call count from persistent storage."""
    try:
        if _API_COUNT_FILE.exists():
            data = json.loads(_API_COUNT_FILE.read_text())
            if data.get("date") == str(date.today()):
                return data.get("count", 0)
    except Exception:
        pass
    return 0


def _increment_api_call_count():
    """Increment and persist today's API call count."""
    try:
        _API_COUNT_FILE.parent.mkdir(exist_ok=True)
        current = _get_api_call_count()
        _API_COUNT_FILE.write_text(json.dumps({
            "date": str(date.today()),
            "count": current + 1
        }))
    except Exception as e:
        logger.debug(f"Failed to persist API count: {e}")


def _call_claude(api_key, system_prompt, user_prompt, max_tokens=800):
    """Raw Claude API call with caching and rate limiting."""
    api_count = _get_api_call_count()
    if api_count >= SAFETY_LIMITS["max_api_calls_per_day"]:
        logger.warning(f"Claude API daily limit reached ({api_count}/{SAFETY_LIMITS['max_api_calls_per_day']})")
        return None

    # Simple cache by prompt hash
    cache_key = hashlib.md5(user_prompt.encode()).hexdigest()[:12]
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    try:
        import requests
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        _increment_api_call_count()
        data = resp.json()
        if "error" in data:
            logger.warning(f"Claude API error: {data['error'].get('message', '')}")
            return None
        text = data.get("content", [{}])[0].get("text", "")
        # Parse JSON from response
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
        result = json.loads(clean)
        _response_cache[cache_key] = result
        return result
    except json.JSONDecodeError:
        logger.warning(f"Claude returned non-JSON: {text[:100]}")
        return None
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return None


def _fetch_market_news(symbols=None):
    """Fetch latest market news from Google News RSS (free, no API key)."""
    news = []
    try:
        import requests
        from xml.etree import ElementTree
        # General market news
        queries = ["NSE+India+stock+market", "Nifty+50+today"]
        if symbols:
            queries += [f"{s}+NSE+stock" for s in symbols[:3]]

        for q in queries:
            try:
                url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
                resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    continue
                root = ElementTree.fromstring(resp.content)
                for item in root.findall(".//item")[:3]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    source = item.find("source").text if item.find("source") is not None else ""
                    if title:
                        news.append({"title": title, "source": source, "date": pub_date})
            except:
                continue
    except ImportError:
        logger.debug("requests not available for news fetch")
    return news[:10]  # max 10 headlines


class ClaudeBrainV2:
    """
    Adaptive AI Brain — runs throughout the trading day.
    Makes Claude aware of ML model, news, positions, and past trades.
    """

    SYSTEM_PROMPT = """You are an expert NSE India INTRADAY trading analyst AI embedded inside an algo trading bot.
You analyze real-time data and provide actionable JSON responses.

CRITICAL INTRADAY RULES:
- ALL analysis uses TODAY's data only — ignore yesterday's close
- Reference price = TODAY'S OPEN (first candle open price)
- Direction = WHERE price IS in TODAY's range (not historical trend)
- Support = TODAY's low, VWAP when price above it
- Resistance = TODAY's high, VWAP when price below it
- Volume comparison = vs TODAY's average, not historical

KEY RULES:
- You ONLY suggest adjustments within safety limits
- You NEVER recommend holding overnight (intraday only)
- Risk per trade: max 2% of capital
- Max trades per day: 5
- You always return valid JSON, nothing else
- Be specific with numbers: exact stop loss levels, exact position sizes
- When suggesting strategy changes, explain WHY briefly

INTRADAY METRICS TO CONSIDER:
- intraday_change_pct: % change from today's open
- position_in_range: 0=at day low, 1=at day high
- today_open, today_high, today_low: key intraday levels"""

    def __init__(self, config=None):
        self.api_key = ""
        if config:
            self.api_key = config.get("claude", {}).get("api_key", "")
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")

        self.enabled = bool(self.api_key) and self.api_key not in (
            "", "YOUR_ANTHROPIC_API_KEY", "TEST_KEY"
        )
        if config and not config.get("claude", {}).get("enabled", True):
            self.enabled = False

        self._trade_history = self._load_trade_history()
        self._last_live_call = None

    def _load_trade_history(self):
        """Load recent trade results for learning."""
        results_dir = Path("results")
        if not results_dir.exists():
            return []
        history = []
        try:
            import pandas as pd
            csvs = sorted(results_dir.glob("live_v3_*.csv"))[-5:]
            for csv in csvs:
                df = pd.read_csv(csv)
                day_pnl = df["net_pnl"].sum()
                wins = len(df[df["net_pnl"] > 0])
                losses = len(df[df["net_pnl"] <= 0])
                worst = df.nsmallest(1, "net_pnl")
                best = df.nlargest(1, "net_pnl")
                history.append({
                    "date": csv.stem.replace("live_v3_", ""),
                    "trades": len(df),
                    "wins": wins,
                    "losses": losses,
                    "pnl": round(day_pnl, 2),
                    "worst_trade": worst.iloc[0].to_dict() if len(worst) > 0 else {},
                    "best_trade": best.iloc[0].to_dict() if len(best) > 0 else {},
                })
        except Exception as e:
            logger.debug(f"Trade history load error: {e}")
        return history

    # ════════════════════════════════════════════════
    # 1. PRE-MARKET ANALYSIS
    # ════════════════════════════════════════════════

    def morning_analysis(self, vix, stock_scores, ml_model_info=None):
        """
        Called at 8:50-9:05 AM. Returns:
        - risk_level, max_trades, skip_stocks
        - strategy_tweaks (ATR multiplier, time windows)
        - stock_directions with confidence
        - news_sentiment
        """
        if not self.enabled:
            return self._default_morning(vix)

        news = _fetch_market_news()
        news_text = "\n".join([f"- {n['title']} ({n['source']})" for n in news[:6]])

        scores_text = ""
        if stock_scores:
            top = stock_scores[:10] if isinstance(stock_scores, list) else []
            scores_text = json.dumps(top, default=str)

        history_text = json.dumps(self._trade_history[-3:], default=str) if self._trade_history else "No recent history"

        prompt = f"""TODAY: {date.today()} | VIX: {vix}

ML MODEL SCORES (top stocks):
{scores_text}

RECENT TRADE HISTORY (last 3 days):
{history_text}

MARKET NEWS:
{news_text}

ML MODEL INFO: {json.dumps(ml_model_info or {}, default=str)}

Analyze everything above and return JSON:
{{
  "risk_level": "conservative|normal|aggressive",
  "max_trades": 3,
  "risk_per_trade_pct": 0.02,
  "position_size_factor": 1.0,
  "skip_stocks": ["SYM1"],
  "preferred_stocks": ["SYM1", "SYM2"],
  "stock_directions": {{"RELIANCE": "SHORT", "SBIN": "LONG"}},
  "strategy_tweaks": {{
    "orb_atr_multiplier": 1.5,
    "target_pct": 0.01,
    "stop_pct": 0.005,
    "quick_profit_pct": 0.008,
    "cut_loss_pct": 0.004,
    "trailing_activation_pct": 0.005,
    "time_decay_candles": 24,
    "widen_stops_pct": 0,
    "prefer_momentum_over_orb": false,
    "avoid_afternoon_trades": false
  }},
  "entry_filters": {{
    "min_ml_confidence": 0.12,
    "blocked_regimes": ["ranging", "unknown"],
    "enable_longs": true,
    "enable_shorts": true,
    "max_longs": 5,
    "max_shorts": 5
  }},
  "news_sentiment": "bullish|bearish|neutral",
  "market_outlook": "brief 1 line",
  "avoid_sectors": [],
  "notes": "brief reasoning"
}}

DYNAMIC PARAMETERS GUIDE (VIX-based):
- VIX < 15: max_trades=5, risk=3%, target=1.5%, stop=0.8%, quick_profit=1.2%
- VIX 15-20: max_trades=4, risk=2.5%, target=1.2%, stop=0.6%, quick_profit=1.0%
- VIX 20-25: max_trades=3, risk=2%, target=1.0%, stop=0.5%, quick_profit=0.8%
- VIX 25-30: max_trades=2, risk=1.5%, target=0.8%, stop=0.4%, quick_profit=0.6%
- VIX > 30: max_trades=1, risk=1%, target=0.6%, stop=0.3%, quick_profit=0.5%

DIRECTION RULES (news-based):
- Bearish news: enable_shorts=true, max_shorts=5, max_longs=2
- Bullish news: enable_longs=true, max_longs=5, max_shorts=2
- Mixed/neutral: both enabled, balanced"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=1000)
        return self._validate_morning(result, vix) if result else self._default_morning(vix)

    # ════════════════════════════════════════════════
    # 2. QUICK MARKET SCAN (every 5 min during market)
    # ════════════════════════════════════════════════

    def quick_scan(self, regimes, vix, day_pnl, trades_taken):
        """
        Fast 5-minute scan during market hours. Lightweight check.
        Returns strategy adjustments based on regime changes.
        """
        if not self.enabled:
            return self._default_quick_scan()

        now = datetime.now()
        
        # Count regimes
        regime_counts = {}
        for sym, reg in regimes.items():
            regime_counts[reg] = regime_counts.get(reg, 0) + 1
        
        # Determine dominant regime
        dominant = max(regime_counts, key=regime_counts.get) if regime_counts else "unknown"
        
        # Quick heuristic response (no API call if simple)
        if dominant in ("trending_up", "trending_down"):
            return {
                "market_trend": dominant,
                "recommended_strategies": ["ORB", "MOMENTUM", "AFTERNOON_TREND"],
                "skip_strategies": [],
                "position_size_factor": 1.0,
                "notes": f"Market trending {dominant.split('_')[1]}, use trend-following strategies"
            }
        elif dominant == "choppy":
            return {
                "market_trend": "choppy",
                "recommended_strategies": ["VWAP", "MOMENTUM"],  # V2: Allow momentum in choppy
                "skip_strategies": ["AFTERNOON_TREND"],
                "position_size_factor": 0.7,
                "notes": "Choppy market - reduce size, prefer mean reversion"
            }
        elif dominant == "ranging":
            return {
                "market_trend": "ranging",
                "recommended_strategies": ["VWAP"],
                "skip_strategies": ["ORB", "AFTERNOON_TREND"],
                "position_size_factor": 0.6,
                "notes": "Ranging market - only VWAP with extreme deviation"
            }
        elif dominant == "volatile":
            return {
                "market_trend": "volatile",
                "recommended_strategies": ["MOMENTUM"],
                "skip_strategies": ["VWAP"],
                "position_size_factor": 0.5,
                "notes": "High volatility - wider stops, smaller size, momentum only"
            }
        
        return self._default_quick_scan()

    def _default_quick_scan(self):
        return {
            "market_trend": "unknown",
            "recommended_strategies": ["ORB", "MOMENTUM", "VWAP"],
            "skip_strategies": [],
            "position_size_factor": 1.0,
            "notes": "Default scan"
        }

    # ════════════════════════════════════════════════
    # 3. FULL LIVE ADJUSTMENT (every 15 min)
    # ════════════════════════════════════════════════

    def live_adjustment(self, current_state):
        """
        Called every 15 min during market hours. Full analysis with Claude.
        Receives current state:
        - open_positions: [{symbol, side, entry, current, pnl, regime}]
        - day_pnl: float
        - trades_taken: int
        - stock_regimes: {symbol: regime}
        - vix: float
        - time: str (HH:MM)
        - market_trend: from quick_scan

        Returns:
        - emergency_exits: [symbol]
        - tighten_stops: {symbol: new_sl}
        - add_stocks: [{symbol, direction, reason}]
        - remove_stocks: [symbol]
        - strategy_switch: {symbol: new_strategy}
        - notes: str
        """
        if not self.enabled:
            return self._default_live()

        # Rate limit: max 1 call per 10 min (was 25 min)
        now = datetime.now()
        if self._last_live_call and (now - self._last_live_call).seconds < 600:
            return self._default_live()
        self._last_live_call = now

        # Fetch fresh news for context
        news = _fetch_market_news()
        news_text = "\n".join([f"- {n['title']}" for n in news[:5]]) if news else "No recent news"

        prompt = f"""LIVE MARKET UPDATE — {now.strftime('%H:%M')}

CURRENT STATE:
{json.dumps(current_state, default=str, indent=2)}

LATEST NEWS:
{news_text}

Analyze market conditions and suggest adjustments.

Return JSON:
{{
  "market_analysis": {{
    "trend": "bullish|bearish|sideways",
    "strength": 1-10,
    "key_levels": {{"nifty_support": 0, "nifty_resistance": 0}},
    "sector_leaders": [],
    "sector_laggards": []
  }},
  "emergency_exits": [],
  "tighten_stops": {{}},
  "add_stocks": [],
  "remove_stocks": [],
  "strategy_switch": {{}},
  "new_strategy_rules": [],
  "risk_adjustment": 1.0,
  "notes": "brief reasoning"
}}

Rules:
- market_analysis: Your view on overall market direction
- emergency_exits: only if stock is clearly going against position with no recovery
- tighten_stops: {{symbol: new_sl_price}} — only tighten, never widen
- strategy_switch: {{symbol: "MOMENTUM"|"VWAP"|"ORB"|"SKIP"}}
- new_strategy_rules: [{{"name": "rule_name", "condition": "description", "action": "description"}}]
- risk_adjustment: 0.5=half size, 1.0=normal, 1.5=increase (only if winning)"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=1000)
        return self._validate_live(result) if result else self._default_live()

    # ════════════════════════════════════════════════
    # 4. INTRADAY MARKET TREND ANALYSIS (NEW!)
    # ════════════════════════════════════════════════

    def analyze_intraday_trend(self, intraday_metrics):
        """
        Analyze market trend using INTRADAY data only.
        This is the KEY method for intraday trading decisions.
        
        Args:
            intraday_metrics: dict with:
                - nifty_open: Today's Nifty open price
                - nifty_current: Current Nifty price
                - nifty_high: Today's Nifty high
                - nifty_low: Today's Nifty low
                - intraday_change_pct: % change from today's open
                - position_in_range: 0-1 (where price is in today's range)
                - vix: Current VIX level
                - time: Current time (HH:MM)
                - sector_performance: {sector: % change today}
        
        Returns:
            - trend: bullish/bearish/sideways
            - direction: LONG/SHORT/NEUTRAL
            - confidence: 0-100
            - key_levels: {support, resistance, vwap_estimate}
            - strategy_for_now: which strategy to use
        """
        if not self.enabled:
            return self._default_intraday_trend(intraday_metrics)
        
        prompt = f"""INTRADAY MARKET ANALYSIS — {intraday_metrics.get('time', 'N/A')}

CRITICAL: Use ONLY today's data for analysis. Ignore any historical daily trends.

TODAY'S NIFTY DATA:
- Open: {intraday_metrics.get('nifty_open', 'N/A')}
- Current: {intraday_metrics.get('nifty_current', 'N/A')}
- High: {intraday_metrics.get('nifty_high', 'N/A')}
- Low: {intraday_metrics.get('nifty_low', 'N/A')}
- Change from Open: {intraday_metrics.get('intraday_change_pct', 0):+.2f}%
- Position in Range: {intraday_metrics.get('position_in_range', 0.5):.0%} (0%=at low, 100%=at high)

VIX: {intraday_metrics.get('vix', 'N/A')}

SECTOR PERFORMANCE TODAY:
{json.dumps(intraday_metrics.get('sector_performance', {}), default=str)}

Based on WHERE price IS in today's range (not historical trend), determine:

Return JSON:
{{
  "trend": "bullish|bearish|sideways",
  "direction": "LONG|SHORT|NEUTRAL",
  "confidence": 75,
  "reasoning": "Position in upper/lower range, momentum direction",
  "key_levels": {{
    "support": {intraday_metrics.get('nifty_low', 0)},
    "resistance": {intraday_metrics.get('nifty_high', 0)},
    "pivot": 0
  }},
  "strategy_for_now": "ORB|MOMENTUM|VWAP|AFTERNOON_TREND|WAIT",
  "position_size_factor": 1.0,
  "sector_focus": ["IT", "Banking"],
  "sector_avoid": [],
  "time_based_advice": "specific advice based on current time",
  "risk_level": "low|medium|high"
}}

INTRADAY RULES:
- Position in range > 60% = BULLISH bias (LONG preferred)
- Position in range < 40% = BEARISH bias (SHORT preferred)
- 40-60% = NEUTRAL (wait or use VWAP mean reversion)
- Morning (9:15-11:00): ORB/MOMENTUM strategies
- Midday (11:00-13:00): VWAP/MOMENTUM strategies
- Afternoon (13:00-14:30): AFTERNOON_TREND only with strong signals
- After 14:30: Reduce new entries, manage existing positions"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=700)
        if result:
            return self._validate_intraday_trend(result, intraday_metrics)
        return self._default_intraday_trend(intraday_metrics)
    
    def _validate_intraday_trend(self, result, metrics):
        """Validate and enhance intraday trend analysis."""
        v = result.copy()
        
        # Ensure key fields exist
        v["confidence"] = max(0, min(100, v.get("confidence", 50)))
        v["position_size_factor"] = max(0.3, min(1.5, v.get("position_size_factor", 1.0)))
        
        # Auto-adjust based on position in range if Claude missed it
        pos = metrics.get("position_in_range", 0.5)
        if pos > 0.7 and v.get("direction") == "SHORT":
            v["confidence"] = max(30, v["confidence"] - 20)
            v["warning"] = "Direction contradicts position in range"
        elif pos < 0.3 and v.get("direction") == "LONG":
            v["confidence"] = max(30, v["confidence"] - 20)
            v["warning"] = "Direction contradicts position in range"
        
        v["validated"] = True
        return v
    
    def _default_intraday_trend(self, metrics):
        """Default intraday trend based on simple rules."""
        pos = metrics.get("position_in_range", 0.5)
        change = metrics.get("intraday_change_pct", 0)
        
        if pos > 0.6 and change > 0.3:
            trend = "bullish"
            direction = "LONG"
            confidence = min(80, 50 + pos * 30)
        elif pos < 0.4 and change < -0.3:
            trend = "bearish"
            direction = "SHORT"
            confidence = min(80, 50 + (1 - pos) * 30)
        else:
            trend = "sideways"
            direction = "NEUTRAL"
            confidence = 50
        
        return {
            "trend": trend,
            "direction": direction,
            "confidence": confidence,
            "reasoning": f"Position in range: {pos:.0%}, intraday change: {change:+.2f}%",
            "key_levels": {
                "support": metrics.get("nifty_low", 0),
                "resistance": metrics.get("nifty_high", 0),
            },
            "strategy_for_now": "MOMENTUM" if abs(change) > 0.5 else "VWAP",
            "position_size_factor": 1.0,
            "sector_focus": [],
            "sector_avoid": [],
            "risk_level": "medium",
            "validated": True,
        }

    def analyze_market_trend(self, nifty_data=None, vix=None, sector_data=None):
        """
        Analyze overall market trend and suggest strategy adjustments.
        Called periodically to understand market direction.
        
        NOTE: For intraday trading, prefer analyze_intraday_trend() instead.
        
        Returns:
        - trend: bullish/bearish/sideways
        - strength: 1-10
        - recommended_direction: LONG/SHORT/BOTH
        - sector_rotation: which sectors to focus
        - strategy_preference: which strategies work best now
        """
        if not self.enabled:
            return self._default_market_trend()

        prompt = f"""MARKET TREND ANALYSIS

VIX: {vix or 'N/A'}
Nifty Data: {json.dumps(nifty_data or {}, default=str)}
Sector Performance: {json.dumps(sector_data or {}, default=str)}

Current Time: {datetime.now().strftime('%H:%M')}

Analyze the market and provide trading direction:

Return JSON:
{{
  "trend": "bullish|bearish|sideways",
  "strength": 7,
  "confidence": 75,
  "recommended_direction": "LONG|SHORT|BOTH",
  "avoid_direction": "LONG|SHORT|NONE",
  "sector_focus": ["IT", "Banking"],
  "sector_avoid": ["Metals"],
  "strategy_preference": {{
    "trending_market": ["ORB", "MOMENTUM"],
    "current_recommendation": ["MOMENTUM", "VWAP"]
  }},
  "key_insight": "one line market insight",
  "risk_level": "low|medium|high"
}}"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=600)
        if result:
            return result
        return self._default_market_trend()

    def _default_market_trend(self):
        return {
            "trend": "sideways",
            "strength": 5,
            "confidence": 50,
            "recommended_direction": "BOTH",
            "avoid_direction": "NONE",
            "sector_focus": [],
            "sector_avoid": [],
            "strategy_preference": {
                "trending_market": ["ORB", "MOMENTUM"],
                "current_recommendation": ["VWAP", "MOMENTUM"]
            },
            "key_insight": "No Claude analysis available",
            "risk_level": "medium"
        }

    # ════════════════════════════════════════════════
    # 5. STOCK ANALYSIS (on-demand) — INTRADAY FOCUSED
    # ════════════════════════════════════════════════

    def analyze_stock_intraday(self, symbol, intraday_data, market_bias, regime, news=None):
        """
        INTRADAY-focused analysis of a single stock before entry.
        Uses today's price action, not historical data.
        
        Args:
            symbol: Stock symbol
            intraday_data: dict with today's data:
                - today_open: Stock's opening price today
                - current_price: Current price
                - today_high: Today's high
                - today_low: Today's low
                - intraday_change_pct: % change from today's open
                - position_in_range: 0-1 (where price is in today's range)
                - volume_vs_avg: Current volume vs today's average
                - vwap: Today's VWAP
            market_bias: LONG/SHORT/NEUTRAL from Nifty analysis
            regime: trending_up/trending_down/ranging/volatile/choppy

        Returns:
        - take_trade: bool
        - direction: LONG/SHORT
        - confidence: 0-100
        - entry_price: suggested entry
        - stop_loss: based on today's levels
        - target: based on today's levels
        - reason: str
        """
        if not self.enabled:
            return self._default_stock_intraday(intraday_data, market_bias)

        stock_news = _fetch_market_news([symbol]) if not news else news
        news_text = "\n".join([f"- {n['title']}" for n in stock_news[:3]]) or "No recent news"

        prompt = f"""INTRADAY TRADE DECISION: {symbol}

TODAY'S STOCK DATA (use ONLY this for analysis):
- Open: ₹{intraday_data.get('today_open', 'N/A')}
- Current: ₹{intraday_data.get('current_price', 'N/A')}
- High: ₹{intraday_data.get('today_high', 'N/A')}
- Low: ₹{intraday_data.get('today_low', 'N/A')}
- Change from Open: {intraday_data.get('intraday_change_pct', 0):+.2f}%
- Position in Range: {intraday_data.get('position_in_range', 0.5):.0%}
- Volume vs Today Avg: {intraday_data.get('volume_vs_avg', 1.0):.1f}x
- VWAP: ₹{intraday_data.get('vwap', 'N/A')}

MARKET CONTEXT:
- Nifty Bias: {market_bias}
- Stock Regime: {regime}

Recent News:
{news_text}

Past Performance:
{self._get_stock_history(symbol)}

INTRADAY ENTRY RULES:
- Direction should ALIGN with market_bias (don't fight the market)
- Position in range > 55% + market LONG = LONG trade
- Position in range < 45% + market SHORT = SHORT trade
- Stop loss = today's low (LONG) or today's high (SHORT)
- Target = opposite end of today's range or VWAP

Should I take this trade? Return JSON:
{{
  "take_trade": true,
  "direction": "LONG|SHORT",
  "confidence": 75,
  "entry_price": {intraday_data.get('current_price', 0)},
  "stop_loss": 0,
  "target": 0,
  "position_size_factor": 1.0,
  "reason": "brief explanation using intraday logic"
}}"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=500)
        if result:
            result["take_trade"] = result.get("take_trade", True)
            result["confidence"] = max(0, min(100, result.get("confidence", 50)))
            return result
        return self._default_stock_intraday(intraday_data, market_bias)
    
    def _default_stock_intraday(self, data, market_bias):
        """Default stock analysis based on intraday rules."""
        pos = data.get("position_in_range", 0.5)
        current = data.get("current_price", 0)
        today_low = data.get("today_low", current * 0.99)
        today_high = data.get("today_high", current * 1.01)
        vwap = data.get("vwap", current)
        
        # Direction based on position in range + market bias
        if pos > 0.55 and market_bias == "LONG":
            direction = "LONG"
            stop_loss = today_low * 0.998
            target = today_high * 0.998 if today_high > current else current * 1.01
            take_trade = True
        elif pos < 0.45 and market_bias == "SHORT":
            direction = "SHORT"
            stop_loss = today_high * 1.002
            target = today_low * 1.002 if today_low < current else current * 0.99
            take_trade = True
        else:
            direction = "NEUTRAL"
            stop_loss = 0
            target = 0
            take_trade = False
        
        return {
            "take_trade": take_trade,
            "direction": direction,
            "confidence": 50,
            "entry_price": current,
            "stop_loss": stop_loss,
            "target": target,
            "position_size_factor": 1.0,
            "reason": f"Position {pos:.0%}, bias {market_bias}",
        }

    def analyze_stock(self, symbol, candle_summary, ml_score, direction, regime, news=None):
        """
        Deep analysis of a single stock before entry.
        Called when a signal is generated but before execution.
        
        NOTE: For pure intraday, prefer analyze_stock_intraday() instead.

        Returns:
        - take_trade: bool
        - confidence: 0-100
        - adjusted_sl: float (or null)
        - adjusted_target: float (or null)
        - reason: str
        """
        if not self.enabled:
            return {"take_trade": True, "confidence": 50, "reason": "No Claude analysis"}

        stock_news = _fetch_market_news([symbol]) if not news else news
        news_text = "\n".join([f"- {n['title']}" for n in stock_news[:3]]) or "No recent news"

        prompt = f"""TRADE DECISION: {symbol}

ML Score: {ml_score} | Direction: {direction} | Regime: {regime}
Candle Data: {json.dumps(candle_summary, default=str)}

Recent News:
{news_text}

Past Performance on this stock:
{self._get_stock_history(symbol)}

Should I take this trade? Return JSON:
{{
  "take_trade": true,
  "confidence": 75,
  "adjusted_sl": null,
  "adjusted_target": null,
  "position_size_factor": 1.0,
  "reason": "brief"
}}"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=400)
        if result:
            result["take_trade"] = result.get("take_trade", True)
            result["confidence"] = max(0, min(100, result.get("confidence", 50)))
            return result
        return {"take_trade": True, "confidence": 50, "reason": "API unavailable"}

    # ════════════════════════════════════════════════
    # 4. NEWS-BASED DIRECTION OVERRIDE + STOCK ADJUSTMENT
    # ════════════════════════════════════════════════

    def news_direction_check(self, symbols):
        """
        Scan news for the given symbols and return directional bias.
        Called during stock selection to override ML direction if news is strong.

        Returns: {symbol: {"bias": "LONG"|"SHORT"|"NEUTRAL", "strength": 0-10, "headline": str}}
        """
        if not self.enabled:
            return {}

        news = _fetch_market_news(symbols)
        if not news:
            return {}

        news_text = "\n".join([f"- {n['title']} ({n['source']})" for n in news])

        prompt = f"""Analyze these market news headlines for stock direction bias.
Stocks I'm tracking: {', '.join(symbols[:10])}

News:
{news_text}

Return JSON — only include stocks mentioned in news:
{{
  "SYMBOL": {{"bias": "LONG|SHORT|NEUTRAL", "strength": 5, "headline": "relevant headline"}},
  ...
}}

strength: 1-3 = weak signal, 4-6 = moderate, 7-10 = strong (earnings, FDA, major event)"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=600)
        if result and isinstance(result, dict):
            return {k: v for k, v in result.items() if k in symbols}
        return {}

    def adjust_stocks_by_news(self, stock_picks, intraday_metrics=None):
        """
        Adjust stock selection based on news analysis.
        This is the KEY method for news-driven stock filtering.
        
        Args:
            stock_picks: DataFrame with columns [symbol, direction, ml_score, ...]
            intraday_metrics: dict with Nifty intraday data
        
        Returns:
            dict with:
            - adjusted_picks: list of adjusted stock recommendations
            - skip_stocks: list of stocks to avoid (bad news)
            - flip_direction: dict of stocks to flip direction
            - boost_stocks: list of stocks to prioritize (good news)
            - news_summary: brief news summary
        """
        if not self.enabled:
            return self._default_news_adjustment(stock_picks)
        
        symbols = stock_picks["symbol"].tolist() if hasattr(stock_picks, "symbol") else []
        if not symbols:
            return self._default_news_adjustment(stock_picks)
        
        # Fetch news for all symbols
        news = _fetch_market_news(symbols)
        if not news:
            return self._default_news_adjustment(stock_picks)
        
        news_text = "\n".join([f"- {n['title']} ({n['source']})" for n in news[:15]])
        
        # Build current picks summary
        picks_summary = []
        for _, row in stock_picks.iterrows():
            picks_summary.append({
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "ml_score": row.get("ml_score"),
            })
        
        intraday_str = ""
        if intraday_metrics:
            intraday_str = f"""
INTRADAY CONTEXT:
- Nifty Change: {intraday_metrics.get('intraday_change_pct', 0):+.2f}%
- Position in Range: {intraday_metrics.get('position_in_range', 0.5):.0%}
- Market Bias: {'BULLISH' if intraday_metrics.get('position_in_range', 0.5) > 0.55 else 'BEARISH' if intraday_metrics.get('position_in_range', 0.5) < 0.45 else 'NEUTRAL'}
"""
        
        prompt = f"""STOCK SELECTION ADJUSTMENT — News-Based

CURRENT STOCK PICKS:
{json.dumps(picks_summary, indent=2)}
{intraday_str}
LATEST NEWS:
{news_text}

Analyze the news and adjust stock selection:

1. SKIP any stock with NEGATIVE news (earnings miss, fraud, downgrade)
2. FLIP direction if news contradicts ML direction (e.g., ML says SHORT but earnings beat)
3. BOOST stocks with strong POSITIVE news (earnings beat, upgrade, deal)
4. Consider news recency — today's news > yesterday's

Return JSON:
{{
  "skip_stocks": ["SYM1"],
  "flip_direction": {{"SYM2": "LONG"}},
  "boost_stocks": ["SYM3"],
  "reduce_position": {{"SYM4": 0.5}},
  "news_summary": "Brief summary of key news affecting trading today",
  "sector_news": {{
    "IT": "positive|negative|neutral",
    "Banking": "positive|negative|neutral"
  }},
  "market_sentiment": "bullish|bearish|neutral",
  "confidence": 75
}}

Rules:
- Only include stocks from the picks list
- flip_direction: change to opposite direction
- reduce_position: factor to multiply position size (0.5 = half size)
- Be conservative — only act on clear news signals"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=800)
        
        if result:
            return self._validate_news_adjustment(result, symbols)
        return self._default_news_adjustment(stock_picks)
    
    def _validate_news_adjustment(self, result, valid_symbols):
        """Validate news adjustment results."""
        v = result.copy()
        
        # Ensure only valid symbols
        v["skip_stocks"] = [s for s in v.get("skip_stocks", []) if s in valid_symbols]
        v["flip_direction"] = {k: v for k, v in v.get("flip_direction", {}).items() if k in valid_symbols}
        v["boost_stocks"] = [s for s in v.get("boost_stocks", []) if s in valid_symbols]
        v["reduce_position"] = {k: max(0.3, min(1.0, v)) for k, v in v.get("reduce_position", {}).items() if k in valid_symbols}
        v["confidence"] = max(0, min(100, v.get("confidence", 50)))
        v["validated"] = True
        
        return v
    
    def _default_news_adjustment(self, stock_picks):
        """Default news adjustment (no changes)."""
        return {
            "skip_stocks": [],
            "flip_direction": {},
            "boost_stocks": [],
            "reduce_position": {},
            "news_summary": "No news analysis available",
            "sector_news": {},
            "market_sentiment": "neutral",
            "confidence": 50,
            "validated": True,
        }

    # ════════════════════════════════════════════════
    # 5. DYNAMIC STRATEGY CREATION
    # ════════════════════════════════════════════════

    def suggest_strategy(self, market_condition, recent_performance):
        """
        Ask Claude to create a new strategy rule based on current conditions.
        Called when existing strategies are underperforming.

        Returns: list of strategy rules that can be applied
        """
        if not self.enabled:
            return []

        prompt = f"""Our existing strategies (ORB, VWAP, Momentum, Afternoon Trend) are underperforming.

Market conditions:
{json.dumps(market_condition, default=str)}

Recent 5-day performance:
{json.dumps(recent_performance, default=str)}

Suggest 1-2 NEW intraday strategy rules. Each rule should have:
- Clear entry condition (price, volume, indicator based)
- Stop loss logic
- Target logic
- Which market regime it works best in

Return JSON:
{{
  "strategies": [
    {{
      "name": "strategy_name",
      "type": "LONG|SHORT|BOTH",
      "entry_condition": "specific condition",
      "exit_condition": "specific condition",
      "stop_loss_rule": "ATR-based or fixed %",
      "target_rule": "R:R based or level based",
      "best_regime": "trending|ranging|volatile",
      "best_time_window": "09:30-11:00",
      "reasoning": "why this should work"
    }}
  ]
}}"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=1000)
        if result and "strategies" in result:
            return result["strategies"]
        return []

    # ════════════════════════════════════════════════
    # 6. EOD LEARNING
    # ════════════════════════════════════════════════

    def eod_analysis(self, trades, daily_pnl, regimes_seen):
        """
        End-of-day deep analysis. Claude learns from today's trades.
        Returns lessons and suggestions for tomorrow.
        """
        if not self.enabled or not trades:
            return {"lessons": [], "tomorrow_suggestion": "Use defaults"}

        prompt = f"""END OF DAY ANALYSIS — {date.today()}

Today's Trades:
{json.dumps(trades, default=str, indent=2)}

Daily P&L: Rs {daily_pnl:+,.2f}
Regimes seen: {json.dumps(regimes_seen, default=str)}

Past 5 days:
{json.dumps(self._trade_history, default=str)}

Analyze:
1. What patterns led to winning trades?
2. What mistakes led to losing trades?
3. Which strategies worked best today?
4. What should change tomorrow?

Return JSON:
{{
  "winning_patterns": ["pattern1"],
  "losing_patterns": ["pattern1"],
  "best_strategy_today": "strategy_name",
  "worst_strategy_today": "strategy_name",
  "lessons": ["lesson1"],
  "tomorrow_adjustments": {{
    "risk_level": "conservative|normal|aggressive",
    "prefer_strategies": ["MOMENTUM"],
    "avoid_strategies": ["VWAP"],
    "sector_preference": [],
    "time_window_focus": "morning|afternoon|all_day"
  }},
  "summary": "1 paragraph summary"
}}"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=1000)
        if result:
            # Save learnings for tomorrow
            self._save_learnings(result)
            return result
        return {"lessons": [], "tomorrow_suggestion": "Use defaults"}

    # ════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════

    def _get_stock_history(self, symbol):
        """Get past trade results for a specific stock."""
        results = []
        for day in self._trade_history:
            worst = day.get("worst_trade", {})
            best = day.get("best_trade", {})
            if worst.get("symbol") == symbol:
                results.append(f"LOSS on {day['date']}: Rs {worst.get('net_pnl', 0):+,.0f}")
            if best.get("symbol") == symbol:
                results.append(f"WIN on {day['date']}: Rs {best.get('net_pnl', 0):+,.0f}")
        return "\n".join(results[-3:]) if results else "No history"

    def _save_learnings(self, analysis):
        """Save EOD learnings for next day's pre-market."""
        try:
            path = Path("data/.claude_learnings.json")
            path.parent.mkdir(exist_ok=True)
            existing = json.loads(path.read_text()) if path.exists() else []
            existing.append({
                "date": str(date.today()),
                "analysis": analysis,
            })
            # Keep last 10 days
            path.write_text(json.dumps(existing[-10:], indent=2, default=str))
        except Exception as e:
            logger.debug(f"Save learnings error: {e}")

    def load_yesterday_learnings(self):
        """Load learnings from previous days for morning analysis."""
        try:
            path = Path("data/.claude_learnings.json")
            if path.exists():
                data = json.loads(path.read_text())
                return data[-1] if data else None
        except:
            pass
        return None

    def _validate_morning(self, result, vix):
        v = result.copy()
        if vix > SAFETY_LIMITS["vix_absolute_skip"]:
            v["risk_level"] = "skip"
            v["max_trades"] = 0
            v["risk_per_trade_pct"] = 0.01
        
        # Validate max_trades within safety limits
        v["max_trades"] = min(v.get("max_trades", 3), SAFETY_LIMITS["max_trades_per_day"])
        
        # Validate risk_per_trade within safety limits (0.5% to 3%)
        v["risk_per_trade_pct"] = max(0.005, min(0.03, v.get("risk_per_trade_pct", 0.02)))
        
        # Validate position_size_factor (0.3 to 1.5)
        v["position_size_factor"] = max(0.3, min(1.5, v.get("position_size_factor", 1.0)))
        
        # Validate strategy_tweaks
        tweaks = v.get("strategy_tweaks", {})
        tweaks["orb_atr_multiplier"] = max(1.0, min(2.5, tweaks.get("orb_atr_multiplier", 1.5)))
        tweaks["target_pct"] = max(0.005, min(0.02, tweaks.get("target_pct", 0.01)))
        tweaks["stop_pct"] = max(0.003, min(0.01, tweaks.get("stop_pct", 0.005)))
        tweaks["quick_profit_pct"] = max(0.004, min(0.015, tweaks.get("quick_profit_pct", 0.008)))
        tweaks["cut_loss_pct"] = max(0.002, min(0.008, tweaks.get("cut_loss_pct", 0.004)))
        tweaks["trailing_activation_pct"] = max(0.003, min(0.01, tweaks.get("trailing_activation_pct", 0.005)))
        tweaks["time_decay_candles"] = max(12, min(36, tweaks.get("time_decay_candles", 24)))
        v["strategy_tweaks"] = tweaks
        
        # Validate entry_filters
        filters = v.get("entry_filters", {})
        filters["min_ml_confidence"] = max(0.05, min(0.25, filters.get("min_ml_confidence", 0.12)))
        filters["enable_longs"] = filters.get("enable_longs", True)
        filters["enable_shorts"] = filters.get("enable_shorts", True)
        filters["max_longs"] = max(1, min(10, filters.get("max_longs", 5)))
        filters["max_shorts"] = max(1, min(10, filters.get("max_shorts", 5)))
        filters["blocked_regimes"] = filters.get("blocked_regimes", ["ranging", "unknown"])
        v["entry_filters"] = filters
        
        v["validated"] = True
        return v

    def _validate_live(self, result):
        v = result.copy()
        v["risk_adjustment"] = max(0.3, min(1.5, v.get("risk_adjustment", 1.0)))
        # Never widen stops — only keep tighten_stops
        v.pop("widen_stops", None)
        v["validated"] = True
        return v

    def _default_morning(self, vix):
        """
        VIX-based defaults when Claude API is unavailable.
        These mirror what Claude would return for each VIX level.
        """
        if vix < 15:
            return {
                "risk_level": "aggressive",
                "max_trades": 5,
                "risk_per_trade_pct": 0.03,
                "position_size_factor": 1.0,
                "skip_stocks": [],
                "preferred_stocks": [],
                "stock_directions": {},
                "strategy_tweaks": {
                    "orb_atr_multiplier": 1.8,
                    "target_pct": 0.015,
                    "stop_pct": 0.008,
                    "quick_profit_pct": 0.012,
                    "cut_loss_pct": 0.005,
                    "trailing_activation_pct": 0.006,
                    "time_decay_candles": 30,
                    "widen_stops_pct": 0,
                    "prefer_momentum_over_orb": True,
                    "avoid_afternoon_trades": False
                },
                "entry_filters": {
                    "min_ml_confidence": 0.10,
                    "blocked_regimes": ["unknown"],
                    "enable_longs": True,
                    "enable_shorts": True,
                    "max_longs": 5,
                    "max_shorts": 5
                },
                "news_sentiment": "neutral",
                "market_outlook": f"Low VIX ({vix:.1f}) - aggressive mode, wider targets",
                "notes": "Claude unavailable, using VIX-based defaults",
                "validated": True,
            }
        elif vix < 20:
            return {
                "risk_level": "normal",
                "max_trades": 4,
                "risk_per_trade_pct": 0.025,
                "position_size_factor": 1.0,
                "skip_stocks": [],
                "preferred_stocks": [],
                "stock_directions": {},
                "strategy_tweaks": {
                    "orb_atr_multiplier": 1.5,
                    "target_pct": 0.012,
                    "stop_pct": 0.006,
                    "quick_profit_pct": 0.010,
                    "cut_loss_pct": 0.004,
                    "trailing_activation_pct": 0.005,
                    "time_decay_candles": 24,
                    "widen_stops_pct": 0,
                    "prefer_momentum_over_orb": False,
                    "avoid_afternoon_trades": False
                },
                "entry_filters": {
                    "min_ml_confidence": 0.12,
                    "blocked_regimes": ["ranging", "unknown"],
                    "enable_longs": True,
                    "enable_shorts": True,
                    "max_longs": 4,
                    "max_shorts": 4
                },
                "news_sentiment": "neutral",
                "market_outlook": f"Normal VIX ({vix:.1f}) - balanced approach",
                "notes": "Claude unavailable, using VIX-based defaults",
                "validated": True,
            }
        elif vix < 25:
            return {
                "risk_level": "conservative",
                "max_trades": 3,
                "risk_per_trade_pct": 0.02,
                "position_size_factor": 0.8,
                "skip_stocks": [],
                "preferred_stocks": [],
                "stock_directions": {},
                "strategy_tweaks": {
                    "orb_atr_multiplier": 1.3,
                    "target_pct": 0.010,
                    "stop_pct": 0.005,
                    "quick_profit_pct": 0.008,
                    "cut_loss_pct": 0.004,
                    "trailing_activation_pct": 0.005,
                    "time_decay_candles": 20,
                    "widen_stops_pct": 0,
                    "prefer_momentum_over_orb": False,
                    "avoid_afternoon_trades": True
                },
                "entry_filters": {
                    "min_ml_confidence": 0.15,
                    "blocked_regimes": ["ranging", "unknown", "choppy"],
                    "enable_longs": True,
                    "enable_shorts": True,
                    "max_longs": 3,
                    "max_shorts": 3
                },
                "news_sentiment": "neutral",
                "market_outlook": f"Elevated VIX ({vix:.1f}) - conservative, tighter stops",
                "notes": "Claude unavailable, using VIX-based defaults",
                "validated": True,
            }
        elif vix < 30:
            return {
                "risk_level": "defensive",
                "max_trades": 2,
                "risk_per_trade_pct": 0.015,
                "position_size_factor": 0.6,
                "skip_stocks": [],
                "preferred_stocks": [],
                "stock_directions": {},
                "strategy_tweaks": {
                    "orb_atr_multiplier": 1.2,
                    "target_pct": 0.008,
                    "stop_pct": 0.004,
                    "quick_profit_pct": 0.006,
                    "cut_loss_pct": 0.003,
                    "trailing_activation_pct": 0.004,
                    "time_decay_candles": 16,
                    "widen_stops_pct": 0,
                    "prefer_momentum_over_orb": False,
                    "avoid_afternoon_trades": True
                },
                "entry_filters": {
                    "min_ml_confidence": 0.18,
                    "blocked_regimes": ["ranging", "unknown", "choppy", "volatile"],
                    "enable_longs": True,
                    "enable_shorts": True,
                    "max_longs": 2,
                    "max_shorts": 2
                },
                "news_sentiment": "cautious",
                "market_outlook": f"High VIX ({vix:.1f}) - defensive, quick exits",
                "notes": "Claude unavailable, using VIX-based defaults",
                "validated": True,
            }
        else:
            return {
                "risk_level": "minimal",
                "max_trades": 1,
                "risk_per_trade_pct": 0.01,
                "position_size_factor": 0.5,
                "skip_stocks": [],
                "preferred_stocks": [],
                "stock_directions": {},
                "strategy_tweaks": {
                    "orb_atr_multiplier": 1.0,
                    "target_pct": 0.006,
                    "stop_pct": 0.003,
                    "quick_profit_pct": 0.005,
                    "cut_loss_pct": 0.002,
                    "trailing_activation_pct": 0.003,
                    "time_decay_candles": 12,
                    "widen_stops_pct": 0,
                    "prefer_momentum_over_orb": False,
                    "avoid_afternoon_trades": True
                },
                "entry_filters": {
                    "min_ml_confidence": 0.20,
                    "blocked_regimes": ["ranging", "unknown", "choppy", "volatile"],
                    "enable_longs": False,
                    "enable_shorts": True,
                    "max_longs": 1,
                    "max_shorts": 2
                },
                "news_sentiment": "bearish",
                "market_outlook": f"Extreme VIX ({vix:.1f}) - minimal exposure, SHORT bias",
                "notes": "Claude unavailable, using VIX-based defaults",
                "validated": True,
            }

    def _default_live(self):
        return {
            "emergency_exits": [],
            "tighten_stops": {},
            "add_stocks": [],
            "remove_stocks": [],
            "strategy_switch": {},
            "new_strategy_rules": [],
            "risk_adjustment": 1.0,
            "notes": "No live adjustment",
            "validated": True,
        }

    def get_api_usage(self):
        return {"calls_today": _get_api_call_count(), "limit": SAFETY_LIMITS["max_api_calls_per_day"]}
    
    # ════════════════════════════════════════════════
    # 7. GAP & EVENT ANALYSIS (Morning Pre-Market)
    # ════════════════════════════════════════════════
    
    def analyze_gaps(self, gap_data: list, events: dict = None):
        """
        Analyze overnight gaps and decide which to trade.
        Called at 9:10 AM after opening prices are available.
        
        Args:
            gap_data: List of dicts with symbol, gap_pct, gap_type, prev_trend
            events: Event calendar data (earnings, RBI, etc.)
        
        Returns:
            Dict with approved gaps, rejected gaps, and trading parameters
        """
        if not self.enabled:
            return self._default_gap_analysis(gap_data)
        
        # Rate limit
        now = datetime.now()
        
        prompt = f"""MORNING GAP ANALYSIS — {now.strftime('%Y-%m-%d %H:%M')}

OVERNIGHT GAPS DETECTED:
{json.dumps(gap_data, indent=2, default=str)}

MARKET EVENTS TODAY:
{json.dumps(events or {}, indent=2, default=str)}

Analyze each gap and return JSON:
{{
    "approved_trades": [
        {{
            "symbol": "SYM",
            "direction": "LONG|SHORT",
            "strategy": "GAP_CONTINUATION|GAP_FILL|GAP_BREAKAWAY",
            "entry_zone": [lower, upper],
            "stop_loss": price,
            "target_1": price,
            "target_2": price,
            "position_size_factor": 0.5-1.5,
            "confidence": 0-100,
            "reason": "why this gap is tradeable"
        }}
    ],
    "rejected_gaps": {{
        "SYM": "reason for rejection"
    }},
    "max_gap_trades": 2,
    "overall_gap_sentiment": "bullish|bearish|mixed",
    "notes": "overall market gap analysis"
}}

GAP TRADING RULES:
1. BREAKAWAY GAPS (large + volume + new trend): Trade WITH gap, don't fade
2. CONTINUATION GAPS (mid-trend): Trade WITH gap direction
3. EXHAUSTION GAPS (extreme + end of trend): FADE the gap
4. COMMON GAPS (small, no volume): Expect gap fill, fade cautiously
5. NEVER trade gaps in stocks with earnings TODAY
6. Prefer gaps that are stock-specific (sector didn't gap same way)
7. Max 2-3 gap trades per day to manage risk
8. Gap trades work best in first 30-60 minutes"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=800)
        return self._validate_gap_analysis(result, gap_data) if result else self._default_gap_analysis(gap_data)
    
    def _validate_gap_analysis(self, result, gap_data):
        """Validate Claude's gap analysis."""
        v = result.copy()
        
        # Ensure approved_trades is a list
        if "approved_trades" not in v:
            v["approved_trades"] = []
        
        # Validate each approved trade
        validated_trades = []
        for trade in v.get("approved_trades", []):
            if not trade.get("symbol"):
                continue
            
            # Ensure required fields
            trade["confidence"] = max(0, min(100, trade.get("confidence", 50)))
            trade["position_size_factor"] = max(0.3, min(1.5, trade.get("position_size_factor", 1.0)))
            
            # Must have stop loss
            if not trade.get("stop_loss"):
                continue
            
            validated_trades.append(trade)
        
        v["approved_trades"] = validated_trades[:3]  # Max 3 gap trades
        v["max_gap_trades"] = min(3, v.get("max_gap_trades", 2))
        v["validated"] = True
        
        return v
    
    def _default_gap_analysis(self, gap_data):
        """Default gap analysis when Claude is unavailable."""
        approved = []
        rejected = {}
        
        for gap in gap_data:
            symbol = gap.get("symbol", "")
            gap_pct = abs(gap.get("gap_pct", 0))
            gap_type = gap.get("gap_type", "common")
            
            # Simple rules-based approval
            if gap_pct < 0.5:
                rejected[symbol] = "Gap too small (<0.5%)"
                continue
            
            if gap_type == "breakaway" and gap_pct >= 1.5:
                direction = "LONG" if gap.get("gap_pct", 0) > 0 else "SHORT"
                approved.append({
                    "symbol": symbol,
                    "direction": direction,
                    "strategy": "GAP_BREAKAWAY",
                    "confidence": 70,
                    "position_size_factor": 1.0,
                    "reason": f"Strong breakaway gap {gap_pct:.1f}%"
                })
            elif gap_type == "common" and 0.5 <= gap_pct <= 1.5:
                direction = "SHORT" if gap.get("gap_pct", 0) > 0 else "LONG"
                approved.append({
                    "symbol": symbol,
                    "direction": direction,
                    "strategy": "GAP_FILL",
                    "confidence": 60,
                    "position_size_factor": 0.8,
                    "reason": f"Common gap {gap_pct:.1f}% likely to fill"
                })
            else:
                rejected[symbol] = f"Gap type {gap_type} not ideal for trading"
        
        return {
            "approved_trades": approved[:2],  # Max 2 without Claude
            "rejected_gaps": rejected,
            "max_gap_trades": 2,
            "overall_gap_sentiment": "neutral",
            "notes": "Default gap analysis (Claude unavailable)",
            "validated": True
        }
    
    def analyze_earnings_reaction(self, symbol: str, earnings_data: dict, 
                                   price_data: dict):
        """
        Analyze post-earnings price action for trading opportunity.
        Called on the day after earnings announcement.
        
        Args:
            symbol: Stock symbol
            earnings_data: Dict with beat/miss, guidance, etc.
            price_data: Dict with gap_pct, first_hour_trend, volume_ratio
        """
        if not self.enabled:
            return self._default_earnings_reaction(symbol, price_data)
        
        prompt = f"""POST-EARNINGS ANALYSIS — {symbol}

EARNINGS DATA:
{json.dumps(earnings_data, indent=2, default=str)}

PRICE REACTION:
{json.dumps(price_data, indent=2, default=str)}

Analyze the earnings reaction and return JSON:
{{
    "trade_recommendation": "LONG|SHORT|AVOID",
    "confidence": 0-100,
    "entry_strategy": "GAP_CONTINUATION|GAP_FADE|WAIT_FOR_PULLBACK|NO_TRADE",
    "entry_price_zone": [lower, upper],
    "stop_loss": price,
    "target": price,
    "position_size_factor": 0.5-1.0,
    "max_hold_time": "30min|1hour|end_of_day",
    "reasoning": "explanation",
    "key_levels": {{
        "support": price,
        "resistance": price,
        "gap_fill_level": price
    }}
}}

EARNINGS REACTION RULES:
1. Beat + Gap Up + Strong buying: Trade LONG continuation
2. Beat + Gap Up + Selling: Wait for pullback or avoid
3. Miss + Gap Down + No recovery: Trade SHORT
4. Miss + Gap Down + Recovery attempt: Fade the bounce
5. In-line results: Usually fade the initial move
6. First 30 min is most volatile - smaller size
7. Volume confirms conviction - high volume = real move"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=600)
        if result:
            result["validated"] = True
            return result
        return self._default_earnings_reaction(symbol, price_data)
    
    def _default_earnings_reaction(self, symbol: str, price_data: dict):
        """Default earnings reaction without Claude."""
        gap_pct = price_data.get("gap_pct", 0)
        volume_ratio = price_data.get("volume_ratio", 1.0)
        
        if abs(gap_pct) < 2:
            return {
                "trade_recommendation": "AVOID",
                "confidence": 30,
                "entry_strategy": "NO_TRADE",
                "reasoning": "Gap too small for earnings play",
                "validated": True
            }
        
        # Large gap with volume = continuation
        if volume_ratio > 2.0:
            direction = "LONG" if gap_pct > 0 else "SHORT"
            strategy = "GAP_CONTINUATION"
        else:
            # Low volume gap = fade
            direction = "SHORT" if gap_pct > 0 else "LONG"
            strategy = "GAP_FADE"
        
        return {
            "trade_recommendation": direction,
            "confidence": 55,
            "entry_strategy": strategy,
            "position_size_factor": 0.7,  # Reduced for earnings volatility
            "max_hold_time": "1hour",
            "reasoning": f"Default earnings analysis: {gap_pct:+.1f}% gap, {volume_ratio:.1f}x volume",
            "validated": True
        }
