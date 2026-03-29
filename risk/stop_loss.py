"""
Stop loss management — ATR-based initial stops and Supertrend trailing.

Three stages of a trade's stop loss:
1. INITIAL: ATR-based, placed at entry
2. BREAKEVEN: Moved to entry price when profit = 1× risk
3. TRAILING: Follows Supertrend when profit = 1.5× risk
"""

import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class StopStage(Enum):
    INITIAL = "initial"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"


@dataclass
class StopState:
    price: float
    stage: StopStage
    initial_risk: float  # distance from entry at start


class StopLossManager:
    def __init__(
        self,
        atr_multiplier: float = 1.5,
        breakeven_at_rr: float = 1.0,   # move to BE at 1:1
        trail_at_rr: float = 1.5,        # start trailing at 1.5:1
    ):
        self.atr_multiplier = atr_multiplier
        self.breakeven_at_rr = breakeven_at_rr
        self.trail_at_rr = trail_at_rr

    def initial_stop(
        self,
        entry_price: float,
        atr: float,
        side: str,  # "BUY" or "SELL"
    ) -> StopState:
        """Calculate initial ATR-based stop loss."""
        distance = atr * self.atr_multiplier

        if side == "BUY":
            stop_price = entry_price - distance
        else:
            stop_price = entry_price + distance

        return StopState(
            price=round(stop_price, 2),
            stage=StopStage.INITIAL,
            initial_risk=distance,
        )

    def orb_stop(
        self,
        entry_price: float,
        orb_low: float,
        orb_high: float,
        side: str,
        buffer_pct: float = 0.001,  # 0.1% buffer below ORB range
    ) -> StopState:
        """
        For ORB strategy: stop goes below ORB low (long) or above ORB high (short).
        More natural support/resistance level than pure ATR.
        """
        if side == "BUY":
            stop_price = orb_low * (1 - buffer_pct)
            distance = entry_price - stop_price
        else:
            stop_price = orb_high * (1 + buffer_pct)
            distance = stop_price - entry_price

        return StopState(
            price=round(stop_price, 2),
            stage=StopStage.INITIAL,
            initial_risk=distance,
        )

    def update_stop(
        self,
        current_stop: StopState,
        entry_price: float,
        current_price: float,
        side: str,
        supertrend: float = None,
    ) -> StopState:
        """
        Progress stop through stages based on how far price has moved.

        Args:
            current_stop: Current stop state
            entry_price: Original entry price
            current_price: Latest price
            side: "BUY" or "SELL"
            supertrend: Current Supertrend value (for trailing stage)
        """
        initial_risk = current_stop.initial_risk

        if side == "BUY":
            unrealised_profit = current_price - entry_price
            rr_ratio = unrealised_profit / initial_risk if initial_risk > 0 else 0

            # Stage 3: Trail with Supertrend
            if (
                rr_ratio >= self.trail_at_rr
                and supertrend is not None
                and current_stop.stage != StopStage.TRAILING
            ):
                new_stop = max(current_stop.price, supertrend)
                logger.info(
                    f"TRAILING stop activated at ₹{new_stop:.2f} "
                    f"(RR={rr_ratio:.1f}x)"
                )
                return StopState(new_stop, StopStage.TRAILING, initial_risk)

            # If already trailing, ratchet up with Supertrend
            if current_stop.stage == StopStage.TRAILING and supertrend is not None:
                new_stop = max(current_stop.price, supertrend)
                return StopState(new_stop, StopStage.TRAILING, initial_risk)

            # Stage 2: Move to breakeven
            if (
                rr_ratio >= self.breakeven_at_rr
                and current_stop.stage == StopStage.INITIAL
            ):
                logger.info(
                    f"BREAKEVEN stop at ₹{entry_price:.2f} (RR={rr_ratio:.1f}x)"
                )
                return StopState(entry_price, StopStage.BREAKEVEN, initial_risk)

        else:  # SELL side (mirror logic)
            unrealised_profit = entry_price - current_price
            rr_ratio = unrealised_profit / initial_risk if initial_risk > 0 else 0

            if (
                rr_ratio >= self.trail_at_rr
                and supertrend is not None
                and current_stop.stage != StopStage.TRAILING
            ):
                new_stop = min(current_stop.price, supertrend)
                return StopState(new_stop, StopStage.TRAILING, initial_risk)

            if current_stop.stage == StopStage.TRAILING and supertrend is not None:
                new_stop = min(current_stop.price, supertrend)
                return StopState(new_stop, StopStage.TRAILING, initial_risk)

            if (
                rr_ratio >= self.breakeven_at_rr
                and current_stop.stage == StopStage.INITIAL
            ):
                return StopState(entry_price, StopStage.BREAKEVEN, initial_risk)

        return current_stop  # no change

    def is_stopped_out(
        self,
        stop: StopState,
        current_price: float,
        side: str,
    ) -> bool:
        """Check if current price has hit the stop loss."""
        if side == "BUY":
            return current_price <= stop.price
        else:
            return current_price >= stop.price
