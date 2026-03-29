"""Kill Switch — 3 ways to stop trading immediately.

1. KILL file: touch KILL in project root
2. Telegram: /stop command
3. Config: trading.enabled = false

Also supports vacation mode: /vacation 5 (skip 5 days)
"""
import os, json, logging
from pathlib import Path
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)


class KillSwitch:
    """Check all kill conditions before any trade."""

    def __init__(self, config=None):
        self.config = config or {}
        self.state_file = Path("data/.kill_state.json")
        self.state = self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try: return json.loads(self.state_file.read_text())
            except: pass
        return {"killed": False, "vacation_until": None, "reason": ""}

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state))

    def is_killed(self):
        """Check all kill conditions. Returns (killed: bool, reason: str)."""
        # Method 1: KILL file
        if Path("KILL").exists():
            return True, "KILL file found in project root"

        # Method 2: State file (from Telegram /stop)
        if self.state.get("killed"):
            return True, f"Manually stopped: {self.state.get('reason', 'no reason')}"

        # Method 3: Vacation mode
        vac = self.state.get("vacation_until")
        if vac and date.today().isoformat() < vac:
            return True, f"Vacation mode until {vac}"

        # Method 4: Config flag
        if not self.config.get("trading", {}).get("enabled", True):
            return True, "trading.enabled = false in config"

        return False, "OK"

    def stop(self, reason="manual"):
        self.state["killed"] = True
        self.state["reason"] = reason
        self._save_state()
        logger.warning(f"KILL SWITCH ACTIVATED: {reason}")

    def start(self):
        self.state["killed"] = False
        self.state["reason"] = ""
        self._save_state()
        # Also remove KILL file if it exists
        kill_file = Path("KILL")
        if kill_file.exists(): kill_file.unlink()
        logger.info("Kill switch deactivated — trading resumed")

    def vacation(self, days):
        until = (date.today() + timedelta(days=days)).isoformat()
        self.state["vacation_until"] = until
        self._save_state()
        logger.info(f"Vacation mode: no trading until {until}")

    def status(self):
        killed, reason = self.is_killed()
        return {
            "killed": killed,
            "reason": reason,
            "vacation_until": self.state.get("vacation_until"),
        }
