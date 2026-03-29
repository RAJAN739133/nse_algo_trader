"""
Position sizing based on the 1% risk rule.
This is the most important module in the entire system.

Core formula:
    position_size = (capital * risk_percent) / (entry_price - stop_loss_price)

With ₹1L capital and 1% risk:
    max_risk_per_trade = ₹1,000
    If stop is ₹10 away from entry → buy 100 shares
    If stop is ₹5 away from entry → buy 200 shares
    If stop is ₹20 away from entry → buy 50 shares
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    shares: int
    trade_value: float
    risk_amount: float
    risk_percent: float
    stop_distance: float
    leverage_used: float  # how much of MIS margin you're using


class PositionSizer:
    def __init__(
        self,
        capital: float,
        risk_per_trade: float = 0.01,  # 1%
        max_position_pct: float = 0.50,  # max 50% of capital in one trade
        intraday_leverage: float = 5.0,  # MIS gives ~5x on equity
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.intraday_leverage = intraday_leverage

    def calculate(
        self,
        entry_price: float,
        stop_loss_price: float,
        vix: float = 15.0,
    ) -> PositionSize:
        """
        Calculate the correct number of shares to buy/sell.

        Args:
            entry_price: Expected entry price
            stop_loss_price: Stop loss price (ATR-based)
            vix: Current India VIX — reduces size if elevated

        Returns:
            PositionSize with shares, value, risk details
        """
        # Step 1: Calculate stop distance
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0:
            logger.warning("Stop distance is zero — cannot size position")
            return PositionSize(0, 0, 0, 0, 0, 0)

        # Step 2: Adjust risk for VIX
        adjusted_risk = self.risk_per_trade
        if vix > 25:
            # VIX > 25: don't trade at all
            logger.warning(f"VIX={vix:.1f} > 25 — skipping trade")
            return PositionSize(0, 0, 0, 0, stop_distance, 0)
        elif vix > 20:
            # VIX 20-25: halve the risk
            adjusted_risk = self.risk_per_trade * 0.5
            logger.info(f"VIX={vix:.1f} > 20 — reducing risk to {adjusted_risk:.1%}")

        # Step 3: Max rupees at risk
        max_risk_rupees = self.capital * adjusted_risk

        # Step 4: Position size from risk
        shares = int(max_risk_rupees / stop_distance)

        # Step 5: Cap by max position value
        max_trade_value = self.capital * self.max_position_pct * self.intraday_leverage
        max_shares_by_value = int(max_trade_value / entry_price)
        shares = min(shares, max_shares_by_value)

        # Step 6: Ensure at least 1 share (or 0 if risk too high)
        if shares <= 0:
            logger.warning(
                f"Stop too wide: ₹{stop_distance:.2f} per share. "
                f"Need stop within ₹{max_risk_rupees / 1:.2f} to trade 1 share."
            )
            return PositionSize(0, 0, 0, 0, stop_distance, 0)

        trade_value = shares * entry_price
        actual_risk = shares * stop_distance
        actual_risk_pct = actual_risk / self.capital
        leverage_used = trade_value / self.capital

        return PositionSize(
            shares=shares,
            trade_value=trade_value,
            risk_amount=actual_risk,
            risk_percent=actual_risk_pct,
            stop_distance=stop_distance,
            leverage_used=leverage_used,
        )

    def update_capital(self, new_capital: float):
        """Update capital after P&L changes."""
        self.capital = new_capital
        logger.info(f"Capital updated to ₹{new_capital:,.2f}")
