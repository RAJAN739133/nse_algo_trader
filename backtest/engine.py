"""
Backtesting engine — simulates strategies on historical data.

Features:
- Realistic cost model (Zerodha charges)
- Slippage simulation
- Position sizing with risk management
- Circuit breaker integration
- Detailed trade log and performance metrics
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal, TradeResult
from risk.position_sizer import PositionSizer
from risk.stop_loss import StopLossManager, StopStage
from risk.circuit_breaker import CircuitBreaker
from backtest.costs import ZerodhaCostModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    capital: float = 100_000
    risk_per_trade: float = 0.01
    max_trades_per_day: int = 2
    daily_loss_limit: float = 0.03
    slippage_pct: float = 0.0005  # 0.05% slippage per trade
    commission_model: str = "zerodha"


@dataclass
class BacktestResult:
    trades: list[TradeResult] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    daily_pnl: dict = field(default_factory=dict)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        return np.mean(losses) if losses else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for val in self.equity_curve:
            peak = max(peak, val)
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        return max_dd * 100

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio (assuming 0% risk-free rate)."""
        if not self.trades:
            return 0
        daily_returns = list(self.daily_pnl.values())
        if len(daily_returns) < 2:
            return 0
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        if std_ret == 0:
            return 0
        return (mean_ret / std_ret) * np.sqrt(252)

    def summary(self, initial_capital: float) -> dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": f"{self.win_rate:.1f}%",
            "total_pnl": f"₹{self.total_pnl:,.2f}",
            "return_pct": f"{self.total_pnl / initial_capital * 100:.1f}%",
            "avg_win": f"₹{self.avg_win:,.2f}",
            "avg_loss": f"₹{self.avg_loss:,.2f}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "max_drawdown": f"{self.max_drawdown:.1f}%",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
        }


class BacktestEngine:
    """Run a strategy against historical data with realistic simulation."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.sizer = PositionSizer(
            capital=config.capital,
            risk_per_trade=config.risk_per_trade,
        )
        self.stop_mgr = StopLossManager()
        self.cost_model = ZerodhaCostModel()
        self.circuit_breaker = CircuitBreaker(
            capital=config.capital,
            daily_loss_limit=config.daily_loss_limit,
            max_trades_per_day=config.max_trades_per_day,
        )

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest for a single strategy on a single stock's data.

        Args:
            strategy: Strategy instance (ORB, VWAP, etc.)
            data: DataFrame with OHLCV + indicators, sorted by time
        """
        result = BacktestResult()
        capital = self.config.capital
        result.equity_curve.append(capital)

        in_trade = False
        trade_entry = None
        trade_stop = None
        trade_side = None
        trade_shares = 0
        trade_entry_idx = 0

        current_date = None

        for i in range(len(data)):
            row = data.iloc[i]

            # Track date changes for circuit breaker
            if "datetime" in data.columns:
                row_date = pd.to_datetime(data["datetime"].iloc[i]).date()
            else:
                row_date = pd.to_datetime(data.index[i]).date()

            if row_date != current_date:
                current_date = row_date
                # Reset daily circuit breaker
                self.circuit_breaker._new_day()

            # --- If in a trade, check exits ---
            if in_trade:
                current_price = row["close"]
                supertrend = row.get("supertrend", None)

                # Update trailing stop
                trade_stop = self.stop_mgr.update_stop(
                    trade_stop, trade_entry, current_price, trade_side, supertrend
                )

                # Check stop loss
                if self.stop_mgr.is_stopped_out(trade_stop, current_price, trade_side):
                    exit_price = trade_stop.price
                    exit_price = self._apply_slippage(exit_price, "exit", trade_side)
                    pnl = self._calc_pnl(
                        trade_entry, exit_price, trade_shares, trade_side
                    )
                    result.trades.append(
                        TradeResult(
                            symbol=row.get("symbol", ""),
                            side=trade_side,
                            entry_price=trade_entry,
                            exit_price=exit_price,
                            shares=trade_shares,
                            pnl=pnl,
                            pnl_percent=pnl / capital * 100,
                            entry_time=str(data.iloc[trade_entry_idx].get("datetime", "")),
                            exit_time=str(row.get("datetime", "")),
                            exit_reason=trade_stop.stage.value + "_stop",
                            strategy_name=strategy.name,
                        )
                    )
                    capital += pnl
                    self.circuit_breaker.record_trade(pnl)
                    result.equity_curve.append(capital)
                    result.daily_pnl[str(row_date)] = (
                        result.daily_pnl.get(str(row_date), 0) + pnl
                    )
                    in_trade = False
                    continue

                # Check strategy-specific exit
                should_exit, reason = strategy.should_exit(
                    data, i, trade_entry, trade_side
                )
                if should_exit:
                    exit_price = self._apply_slippage(current_price, "exit", trade_side)
                    pnl = self._calc_pnl(
                        trade_entry, exit_price, trade_shares, trade_side
                    )
                    result.trades.append(
                        TradeResult(
                            symbol=row.get("symbol", ""),
                            side=trade_side,
                            entry_price=trade_entry,
                            exit_price=exit_price,
                            shares=trade_shares,
                            pnl=pnl,
                            pnl_percent=pnl / capital * 100,
                            entry_time=str(data.iloc[trade_entry_idx].get("datetime", "")),
                            exit_time=str(row.get("datetime", "")),
                            exit_reason=reason,
                            strategy_name=strategy.name,
                        )
                    )
                    capital += pnl
                    self.circuit_breaker.record_trade(pnl)
                    result.equity_curve.append(capital)
                    result.daily_pnl[str(row_date)] = (
                        result.daily_pnl.get(str(row_date), 0) + pnl
                    )
                    in_trade = False
                    continue

            # --- Not in a trade, check for entries ---
            else:
                can_trade, reason = self.circuit_breaker.can_trade()
                if not can_trade:
                    continue

                signal = strategy.generate_signal(data, i)
                if signal is None:
                    continue

                # Size the position
                self.sizer.update_capital(capital)
                pos = self.sizer.calculate(
                    entry_price=signal.entry_price,
                    stop_loss_price=signal.stop_loss,
                )

                if pos.shares == 0:
                    continue

                # Enter trade
                entry_price = self._apply_slippage(
                    signal.entry_price, "entry", signal.signal.value
                )
                trade_entry = entry_price
                trade_side = signal.signal.value
                trade_shares = pos.shares
                trade_entry_idx = i

                # Set initial stop
                atr = row.get("atr", abs(entry_price - signal.stop_loss))
                trade_stop = self.stop_mgr.initial_stop(entry_price, atr, trade_side)
                in_trade = True

        return result

    def _apply_slippage(self, price: float, action: str, side: str) -> float:
        """Simulate slippage — worse fill than expected."""
        slip = price * self.config.slippage_pct
        if (action == "entry" and side == "BUY") or (action == "exit" and side == "SELL"):
            return price + slip  # pay more / receive less
        else:
            return price - slip

    def _calc_pnl(
        self,
        entry: float,
        exit: float,
        shares: int,
        side: str,
    ) -> float:
        """Calculate P&L including transaction costs."""
        if side == "BUY":
            gross_pnl = (exit - entry) * shares
            buy_val = entry * shares
            sell_val = exit * shares
        else:
            gross_pnl = (entry - exit) * shares
            buy_val = exit * shares
            sell_val = entry * shares

        costs = self.cost_model.calculate(buy_val, sell_val)
        return round(gross_pnl - costs.total, 2)
