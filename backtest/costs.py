"""
Zerodha cost model for intraday equity trades.
Includes all real-world charges so backtest P&L matches actual trading.

Charges per trade (as of 2025):
- Brokerage: ₹20 per executed order (or 0.03%, whichever is lower)
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


class ZerodhaCostModel:
    """Calculate all charges for an intraday equity trade on Zerodha."""

    BROKERAGE_FLAT = 20.0           # ₹20 per order
    BROKERAGE_PCT = 0.0003          # 0.03%
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
        turnover = buy_value + sell_value

        # Brokerage: ₹20 per order or 0.03%, whichever is lower
        brok_buy = min(self.BROKERAGE_FLAT, buy_value * self.BROKERAGE_PCT)
        brok_sell = min(self.BROKERAGE_FLAT, sell_value * self.BROKERAGE_PCT)
        brokerage = brok_buy + brok_sell

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
