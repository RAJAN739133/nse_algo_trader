"""
Cost models for intraday equity trades (NSE India).
Includes all real-world charges so backtest P&L matches actual trading.

Supports multiple brokers:
- Angel One (default)
- Generic cost model

Charges per trade (as of 2025-2026):
- Brokerage: ₹20 per executed order (flat fee for both Angel One & Zerodha)
- STT: 0.025% on sell side only (intraday equity)
- Exchange txn: 0.00345% (NSE)
- GST: 18% on (brokerage + exchange txn)
- SEBI charges: ₹10 per crore
- Stamp duty: 0.003% on buy side only
"""

from dataclasses import dataclass


@dataclass
class TradeCosts:
    brokerage: float
    stt: float
    exchange_txn: float
    gst: float
    sebi: float
    stamp_duty: float
    total: float
    total_percent: float  # as % of turnover


class AngelOneCostModel:
    """Calculate all charges for an intraday equity trade on Angel One."""

    BROKERAGE_FLAT = 20.0           # ₹20 per order (Angel One flat fee)
    BROKERAGE_PCT = 0.0003          # 0.03% (not used, Angel One is flat ₹20)
    STT_RATE = 0.00025              # 0.025% on sell side
    EXCHANGE_TXN_RATE = 0.0000345   # 0.00345%
    GST_RATE = 0.18                 # 18% on brokerage + exchange
    SEBI_RATE = 0.000001            # ₹10 per crore = 0.0001%
    STAMP_DUTY_RATE = 0.00003       # 0.003% on buy side

    def calculate(
        self,
        buy_value: float,
        sell_value: float,
    ) -> TradeCosts:
        """
        Calculate total cost for a complete round-trip trade.

        Args:
            buy_value: Total value of buy order (price × shares)
            sell_value: Total value of sell order (price × shares)
        """
        # Validate inputs
        if buy_value <= 0 or sell_value <= 0:
            raise ValueError(f"Invalid trade values: buy={buy_value}, sell={sell_value}")
        
        turnover = buy_value + sell_value

        # Brokerage: Angel One charges flat ₹20 per order (not percentage-based)
        brokerage = self.BROKERAGE_FLAT * 2  # ₹20 buy + ₹20 sell = ₹40 round trip

        # STT: only on sell side for intraday
        stt = sell_value * self.STT_RATE

        # Exchange transaction charges
        exchange_txn = turnover * self.EXCHANGE_TXN_RATE

        # GST on brokerage + exchange charges
        gst = (brokerage + exchange_txn) * self.GST_RATE

        # SEBI charges
        sebi = turnover * self.SEBI_RATE

        # Stamp duty on buy side
        stamp_duty = buy_value * self.STAMP_DUTY_RATE

        total = brokerage + stt + exchange_txn + gst + sebi + stamp_duty
        total_pct = (total / turnover * 100) if turnover > 0 else 0

        return TradeCosts(
            brokerage=round(brokerage, 2),
            stt=round(stt, 2),
            exchange_txn=round(exchange_txn, 2),
            gst=round(gst, 2),
            sebi=round(sebi, 2),
            stamp_duty=round(stamp_duty, 2),
            total=round(total, 2),
            total_percent=round(total_pct, 4),
        )

    def total_cost(self, price: float, qty: int, trade_type: str = "intraday") -> float:
        """Quick cost estimate for a single side (buy or sell)."""
        value = price * qty
        result = self.calculate(value, value)
        return result.total / 2  # half for one side

    def estimate_daily_cost(
        self,
        avg_trade_value: float,
        trades_per_day: int = 2,
    ) -> dict:
        """Estimate daily and monthly trading costs."""
        single_trade = self.calculate(avg_trade_value, avg_trade_value)
        daily = single_trade.total * trades_per_day
        monthly = daily * 22  # ~22 trading days

        return {
            "per_trade": single_trade.total,
            "daily": round(daily, 2),
            "monthly": round(monthly, 2),
            "yearly": round(monthly * 12, 2),
            "breakdown": single_trade,
        }


# Alias for backward compatibility
ZerodhaCostModel = AngelOneCostModel
