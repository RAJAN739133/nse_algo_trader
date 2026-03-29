"""
Circuit breaker — automatically stops trading when loss limits are hit.

Rules:
- Daily loss limit: 3% of capital → stop all trading for the day
- Weekly loss limit: 7% of capital → stop all trading for the week
- Max trades per day: 2 (prevents overtrading)
- Max consecutive losses: 3 → pause and alert
"""

import logging
from datetime import date, datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DayRecord:
    date: date
    trades: int = 0
    pnl: float = 0.0
    consecutive_losses: int = 0
    is_stopped: bool = False
    stop_reason: str = ""


class CircuitBreaker:
    def __init__(
        self,
        capital: float,
        daily_loss_limit: float = 0.03,   # 3%
        weekly_loss_limit: float = 0.07,   # 7%
        max_trades_per_day: int = 2,
        max_consecutive_losses: int = 3,
    ):
        self.capital = capital
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.max_trades_per_day = max_trades_per_day
        self.max_consecutive_losses = max_consecutive_losses

        self.today: DayRecord = DayRecord(date=date.today())
        self.week_pnl: float = 0.0
        self.week_start: date = date.today()

    def can_trade(self) -> tuple[bool, str]:
        """
        Check all circuit breaker conditions.
        Returns (allowed: bool, reason: str).
        """
        # Reset if new day
        if date.today() != self.today.date:
            self._new_day()

        # Check if already stopped today
        if self.today.is_stopped:
            return False, self.today.stop_reason

        # Check daily trade count
        if self.today.trades >= self.max_trades_per_day:
            reason = f"Max trades ({self.max_trades_per_day}) reached today"
            self.today.is_stopped = True
            self.today.stop_reason = reason
            logger.warning(reason)
            return False, reason

        # Check daily loss limit
        max_daily_loss = self.capital * self.daily_loss_limit
        if abs(self.today.pnl) >= max_daily_loss and self.today.pnl < 0:
            reason = (
                f"Daily loss limit hit: ₹{abs(self.today.pnl):,.0f} "
                f"(limit: ₹{max_daily_loss:,.0f})"
            )
            self.today.is_stopped = True
            self.today.stop_reason = reason
            logger.warning(reason)
            return False, reason

        # Check weekly loss limit
        max_weekly_loss = self.capital * self.weekly_loss_limit
        if abs(self.week_pnl) >= max_weekly_loss and self.week_pnl < 0:
            reason = (
                f"Weekly loss limit hit: ₹{abs(self.week_pnl):,.0f} "
                f"(limit: ₹{max_weekly_loss:,.0f})"
            )
            self.today.is_stopped = True
            self.today.stop_reason = reason
            logger.warning(reason)
            return False, reason

        # Check consecutive losses
        if self.today.consecutive_losses >= self.max_consecutive_losses:
            reason = f"{self.max_consecutive_losses} consecutive losses — pausing"
            self.today.is_stopped = True
            self.today.stop_reason = reason
            logger.warning(reason)
            return False, reason

        return True, "OK"

    def record_trade(self, pnl: float):
        """Record a completed trade's P&L."""
        self.today.trades += 1
        self.today.pnl += pnl
        self.week_pnl += pnl

        if pnl < 0:
            self.today.consecutive_losses += 1
        else:
            self.today.consecutive_losses = 0

        logger.info(
            f"Trade #{self.today.trades}: ₹{pnl:,.2f} | "
            f"Day P&L: ₹{self.today.pnl:,.2f} | "
            f"Week P&L: ₹{self.week_pnl:,.2f}"
        )

    def _new_day(self):
        """Reset daily counters."""
        self.today = DayRecord(date=date.today())

        # Reset weekly if new week (Monday)
        today = date.today()
        if today.weekday() == 0:  # Monday
            self.week_pnl = 0.0
            self.week_start = today
            logger.info("New week — weekly P&L reset")

    def get_status(self) -> dict:
        """Current circuit breaker status."""
        max_daily = self.capital * self.daily_loss_limit
        max_weekly = self.capital * self.weekly_loss_limit
        return {
            "can_trade": self.can_trade()[0],
            "trades_today": self.today.trades,
            "trades_remaining": max(0, self.max_trades_per_day - self.today.trades),
            "day_pnl": self.today.pnl,
            "day_loss_remaining": max_daily - abs(min(0, self.today.pnl)),
            "week_pnl": self.week_pnl,
            "week_loss_remaining": max_weekly - abs(min(0, self.week_pnl)),
            "consecutive_losses": self.today.consecutive_losses,
            "is_stopped": self.today.is_stopped,
            "stop_reason": self.today.stop_reason,
        }
