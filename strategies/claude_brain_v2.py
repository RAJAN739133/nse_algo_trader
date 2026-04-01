"""
Claude Brain V2 — Live Adaptive AI Strategy Engine

Unlike V1 (morning-only), V2 is called THROUGHOUT the trading day:
  1. PRE-MARKET (8:50 AM): Morning brief + risk level + stock picks
  2. LIVE MARKET (every 30 min): Adjust strategies, skip/add stocks, change stops
  3. MID-DAY (12:30 PM): Re-evaluate positions, news scan, direction changes
  4. POST-MARKET (3:30 PM): EOD analysis, learn from today's mistakes

Features:
  - Reads ML model scores and adjusts confidence per stock
  - Fetches live market news (Google News RSS) and analyzes sentiment
  - Creates dynamic strategy rules based on current market regime
  - Suggests emergency exits or new entries mid-day
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
    "max_api_calls_per_day": 30,
}

# Cache to avoid repeated identical calls
_response_cache = {}
_api_call_count = 0


def _call_claude(api_key, system_prompt, user_prompt, max_tokens=800):
    """Raw Claude API call with caching and rate limiting."""
    global _api_call_count
    if _api_call_count >= SAFETY_LIMITS["max_api_calls_per_day"]:
        logger.warning("Claude API daily limit reached")
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
        _api_call_count += 1
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

    SYSTEM_PROMPT = """You are an expert NSE India intraday trading analyst AI embedded inside an algo trading bot.
You analyze real-time data and provide actionable JSON responses.

KEY RULES:
- You ONLY suggest adjustments within safety limits
- You NEVER recommend holding overnight (intraday only)
- Risk per trade: max 2% of capital
- Max trades per day: 5
- You always return valid JSON, nothing else
- Be specific with numbers: exact stop loss levels, exact position sizes
- When suggesting strategy changes, explain WHY briefly"""

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
  "max_trades": 2,
  "skip_stocks": ["SYM1"],
  "preferred_stocks": ["SYM1", "SYM2"],
  "stock_directions": {{"RELIANCE": "SHORT", "SBIN": "LONG"}},
  "strategy_tweaks": {{
    "orb_atr_multiplier": 1.5,
    "widen_stops_pct": 0,
    "prefer_momentum_over_orb": false,
    "avoid_afternoon_trades": false,
    "time_decay_minutes": 45
  }},
  "news_sentiment": "bullish|bearish|neutral",
  "market_outlook": "brief 1 line",
  "avoid_sectors": [],
  "notes": "brief reasoning"
}}"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=1000)
        return self._validate_morning(result, vix) if result else self._default_morning(vix)

    # ════════════════════════════════════════════════
    # 2. LIVE MARKET ADJUSTMENT (every 30 min)
    # ════════════════════════════════════════════════

    def live_adjustment(self, current_state):
        """
        Called every 30 min during market hours. Receives current state:
        - open_positions: [{symbol, side, entry, current, pnl, regime}]
        - day_pnl: float
        - trades_taken: int
        - stock_regimes: {symbol: regime}
        - vix: float
        - time: str (HH:MM)

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

        # Rate limit: max 1 call per 25 min
        now = datetime.now()
        if self._last_live_call and (now - self._last_live_call).seconds < 1500:
            return self._default_live()
        self._last_live_call = now

        prompt = f"""LIVE MARKET UPDATE — {now.strftime('%H:%M')}

CURRENT STATE:
{json.dumps(current_state, default=str, indent=2)}

Based on positions, P&L, regimes, and time of day, suggest adjustments.

Return JSON:
{{
  "emergency_exits": [],
  "tighten_stops": {{}},
  "widen_stops": {{}},
  "add_stocks": [],
  "remove_stocks": [],
  "strategy_switch": {{}},
  "new_strategy_rules": [],
  "risk_adjustment": 1.0,
  "notes": "brief reasoning"
}}

Rules:
- emergency_exits: only if stock is clearly going against position with no recovery
- tighten_stops: {{symbol: new_sl_price}} — only tighten, never widen beyond original
- strategy_switch: {{symbol: "MOMENTUM"|"VWAP"|"ORB"|"SKIP"}}
- new_strategy_rules: [{{"name": "rule_name", "condition": "description", "action": "description"}}]
- risk_adjustment: 0.5=half size, 1.0=normal, 1.5=increase (only if winning)"""

        result = _call_claude(self.api_key, self.SYSTEM_PROMPT, prompt, max_tokens=800)
        return self._validate_live(result) if result else self._default_live()

    # ════════════════════════════════════════════════
    # 3. STOCK ANALYSIS (on-demand)
    # ════════════════════════════════════════════════

    def analyze_stock(self, symbol, candle_summary, ml_score, direction, regime, news=None):
        """
        Deep analysis of a single stock before entry.
        Called when a signal is generated but before execution.

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
    # 4. NEWS-BASED DIRECTION OVERRIDE
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
        v["max_trades"] = min(v.get("max_trades", 2), SAFETY_LIMITS["max_trades_per_day"])
        tweaks = v.get("strategy_tweaks", {})
        tweaks["orb_atr_multiplier"] = max(1.0, min(2.5, tweaks.get("orb_atr_multiplier", 1.5)))
        v["strategy_tweaks"] = tweaks
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
        level = "conservative" if vix > 20 else ("aggressive" if vix < 14 else "normal")
        return {
            "risk_level": level,
            "max_trades": 2,
            "skip_stocks": [],
            "preferred_stocks": [],
            "stock_directions": {},
            "strategy_tweaks": {"orb_atr_multiplier": 1.5},
            "news_sentiment": "neutral",
            "market_outlook": f"Default mode, VIX={vix:.1f}",
            "notes": "Claude unavailable, using defaults",
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
        return {"calls_today": _api_call_count, "limit": SAFETY_LIMITS["max_api_calls_per_day"]}
