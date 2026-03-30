"""Claude AI Brain — Adaptive strategy adjustment via API.

Every morning at 8:50 AM, calls Claude API with:
- Current VIX, FII/DII flows, news sentiment
- Recent trade performance (last 5 days)
- Today's stock scores

Claude returns:
- Risk level adjustment (conservative/normal/aggressive)
- Stocks to avoid today
- Strategy parameter tweaks
- End-of-day analysis after market close

Safety: All Claude suggestions are validated against hard limits.
"""
import os, json, logging
from datetime import date

logger = logging.getLogger(__name__)

SAFETY_LIMITS = {
    "max_risk_per_trade": 0.02,      # Never risk > 2%
    "max_trades_per_day": 3,          # Never > 3 trades
    "max_position_pct": 0.60,         # Never > 60% in one stock
    "min_stop_loss_pct": 0.005,       # Stop must be > 0.5%
    "vix_absolute_skip": 30,          # Always skip if VIX > 30
}


class ClaudeBrain:
    """AI-powered strategy adaptation."""

    def __init__(self, api_key=None, config=None):
        # Read API key from: 1) parameter, 2) config file, 3) environment variable
        if api_key:
            self.api_key = api_key
        elif config and config.get("claude", {}).get("api_key"):
            self.api_key = config["claude"]["api_key"]
        else:
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")

        self.enabled = bool(self.api_key) and self.api_key not in (
            "", "YOUR_ANTHROPIC_API_KEY", "PASTE_YOUR_ANTHROPIC_API_KEY_HERE", "TEST_KEY"
        )

        if config and not config.get("claude", {}).get("enabled", True):
            self.enabled = False

    def get_morning_analysis(self, vix, fii_net, recent_trades, stock_scores):
        """Call Claude for morning strategy adjustment."""
        if not self.enabled:
            logger.info("Claude Brain: No API key, using defaults")
            return self._default_response(vix)

        prompt = f"""You are an intraday trading analyst for NSE India.
Today: {date.today()}
VIX: {vix}
FII Net: Rs {fii_net:,.0f} Cr
Recent 5-day performance: {json.dumps(recent_trades[-5:] if recent_trades else [])}
Top stock scores: {json.dumps(stock_scores[:5] if stock_scores else [])}

Return JSON only:
{{"risk_level": "conservative|normal|aggressive",
  "skip_stocks": ["SYM1"],
  "orb_atr_multiplier": 1.5,
  "max_trades": 2,
  "notes": "brief reasoning"}}"""

        try:
            import requests
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            data = resp.json()
            # Check for API errors (billing, auth, etc.)
            if "error" in data:
                err_msg = data["error"].get("message", "Unknown API error")
                logger.warning(f"Claude API error: {err_msg}")
                if "credit" in err_msg.lower() or "billing" in err_msg.lower():
                    logger.warning("  → Add credits at console.anthropic.com/settings/plans")
                return self._default_response(vix)
            text = data.get("content", [{}])[0].get("text", "{}")
            # Parse JSON from response
            result = json.loads(text.strip().strip("```json").strip("```"))
            return self._validate(result, vix)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._default_response(vix)

    def _validate(self, result, vix):
        """Validate Claude's suggestions against safety limits."""
        validated = result.copy()
        # Force conservative if VIX is dangerous
        if vix > SAFETY_LIMITS["vix_absolute_skip"]:
            validated["risk_level"] = "skip"
            validated["max_trades"] = 0
        # Cap trades
        if validated.get("max_trades", 2) > SAFETY_LIMITS["max_trades_per_day"]:
            validated["max_trades"] = SAFETY_LIMITS["max_trades_per_day"]
        # Cap ATR multiplier
        atr = validated.get("orb_atr_multiplier", 1.5)
        validated["orb_atr_multiplier"] = max(1.0, min(2.5, atr))
        validated["validated"] = True
        return validated

    def _default_response(self, vix):
        if vix > 25: level = "conservative"
        elif vix < 14: level = "aggressive"
        else: level = "normal"
        return {
            "risk_level": level,
            "skip_stocks": [],
            "orb_atr_multiplier": 1.5,
            "max_trades": 2,
            "notes": f"Default: VIX={vix:.1f}",
            "validated": True,
        }

    def get_eod_analysis(self, trades, daily_pnl):
        """End-of-day analysis — what went right/wrong."""
        if not self.enabled or not trades:
            return "No Claude analysis (API key not set or no trades)"
        # Could call Claude API here for detailed EOD review
        return f"EOD: {len(trades)} trades, P&L Rs {daily_pnl:+,.2f}"
