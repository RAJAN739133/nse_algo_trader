"""
Performance Attribution - Factor and Brinson attribution analysis.

Features:
- Brinson attribution (allocation, selection, interaction)
- Factor attribution (market, size, value, momentum)
- Strategy contribution analysis
- Regime-based performance breakdown
- Win/loss streak analysis
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"


@dataclass
class Trade:
    """Trade record for attribution analysis."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: int
    entry_time: datetime
    exit_time: datetime
    strategy_name: str
    sector: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Market context at trade time
    market_return: float = 0.0
    vix_at_entry: float = 0.0
    regime: MarketRegime = MarketRegime.SIDEWAYS
    
    # Factors
    beta: float = 1.0
    market_cap: str = ""  # "large", "mid", "small"
    momentum_score: float = 0.0
    value_score: float = 0.0


@dataclass
class BrinsonAttribution:
    """Brinson-Fachler attribution results."""
    period_start: date
    period_end: date
    
    # Total return decomposition
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    
    # Attribution effects
    allocation_effect: float      # Sector weight decisions
    selection_effect: float       # Stock selection within sectors
    interaction_effect: float     # Combined effect
    
    # By sector
    sector_allocation: Dict[str, float]
    sector_selection: Dict[str, float]
    sector_contribution: Dict[str, float]


@dataclass
class FactorAttribution:
    """Factor-based attribution results."""
    period_start: date
    period_end: date
    
    # Factor exposures
    market_exposure: float      # Beta
    size_exposure: float        # SMB
    value_exposure: float       # HML
    momentum_exposure: float    # UMD
    
    # Factor contributions to return
    market_contribution: float
    size_contribution: float
    value_contribution: float
    momentum_contribution: float
    
    # Residual (alpha)
    alpha: float
    alpha_t_stat: float
    
    # Total
    total_return: float
    explained_return: float
    unexplained_return: float


@dataclass
class StrategyAttribution:
    """Attribution by trading strategy."""
    strategy_name: str
    
    # Performance
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    
    # Contribution
    contribution_pct: float  # % of total P&L
    trade_count: int
    
    # By regime
    regime_performance: Dict[MarketRegime, Dict]


@dataclass
class StreakAnalysis:
    """Win/loss streak analysis."""
    current_streak: int
    current_streak_type: str  # "win" or "loss"
    max_win_streak: int
    max_loss_streak: int
    avg_win_streak: float
    avg_loss_streak: float
    
    # Streak P&L
    best_streak_pnl: float
    worst_streak_pnl: float
    
    # Recovery
    avg_recovery_trades: float  # Trades to recover from max loss


class PerformanceAttributor:
    """
    Comprehensive performance attribution engine.
    """
    
    def __init__(
        self,
        benchmark_returns: Dict[date, float] = None,
        sector_weights: Dict[str, float] = None,
        factor_returns: Dict[str, Dict[date, float]] = None
    ):
        self.benchmark_returns = benchmark_returns or {}
        self.sector_weights = sector_weights or {}
        self.factor_returns = factor_returns or {}
        
        self._trades: List[Trade] = []
        self._daily_pnl: Dict[date, float] = {}
        self._daily_returns: Dict[date, float] = {}
    
    def add_trade(self, trade: Trade):
        """Add a trade for attribution analysis."""
        # Calculate P&L if not set
        if trade.pnl == 0:
            if trade.side == "BUY":
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.qty
            else:
                trade.pnl = (trade.entry_price - trade.exit_price) * trade.qty
            
            entry_value = trade.entry_price * trade.qty
            trade.pnl_pct = (trade.pnl / entry_value * 100) if entry_value > 0 else 0
        
        self._trades.append(trade)
        
        # Update daily P&L
        trade_date = trade.exit_time.date()
        if trade_date not in self._daily_pnl:
            self._daily_pnl[trade_date] = 0
        self._daily_pnl[trade_date] += trade.pnl
    
    def add_trades(self, trades: List[Trade]):
        """Add multiple trades."""
        for trade in trades:
            self.add_trade(trade)
    
    def brinson_attribution(
        self,
        start_date: date = None,
        end_date: date = None
    ) -> BrinsonAttribution:
        """
        Perform Brinson-Fachler attribution.
        
        Decomposes excess return into:
        - Allocation: Did we overweight the right sectors?
        - Selection: Did we pick the right stocks within sectors?
        - Interaction: Combined effect
        """
        trades = self._filter_trades(start_date, end_date)
        
        if not trades:
            return self._empty_brinson(start_date, end_date)
        
        # Group trades by sector
        sector_trades: Dict[str, List[Trade]] = defaultdict(list)
        for trade in trades:
            sector = trade.sector or "Other"
            sector_trades[sector].append(trade)
        
        # Calculate portfolio sector weights and returns
        total_value = sum(t.entry_price * t.qty for t in trades)
        
        portfolio_sector_weights = {}
        portfolio_sector_returns = {}
        
        for sector, sector_trade_list in sector_trades.items():
            sector_value = sum(t.entry_price * t.qty for t in sector_trade_list)
            sector_pnl = sum(t.pnl for t in sector_trade_list)
            
            portfolio_sector_weights[sector] = sector_value / total_value if total_value > 0 else 0
            portfolio_sector_returns[sector] = sector_pnl / sector_value if sector_value > 0 else 0
        
        # Benchmark sector weights and returns (use defaults or provided)
        benchmark_sector_weights = self.sector_weights or {s: 1/len(sector_trades) for s in sector_trades}
        benchmark_sector_returns = {}  # Would need actual sector index returns
        
        # For simplicity, use average of all trades as "benchmark" per sector
        all_return = sum(t.pnl for t in trades) / total_value if total_value > 0 else 0
        for sector in sector_trades:
            benchmark_sector_returns[sector] = all_return
        
        # Brinson attribution
        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0
        
        sector_allocation = {}
        sector_selection = {}
        sector_contribution = {}
        
        for sector in sector_trades:
            wp = portfolio_sector_weights.get(sector, 0)
            wb = benchmark_sector_weights.get(sector, 0)
            rp = portfolio_sector_returns.get(sector, 0)
            rb = benchmark_sector_returns.get(sector, 0)
            
            # Allocation effect: (wp - wb) * rb
            alloc = (wp - wb) * rb
            allocation_effect += alloc
            sector_allocation[sector] = alloc * 100
            
            # Selection effect: wb * (rp - rb)
            sel = wb * (rp - rb)
            selection_effect += sel
            sector_selection[sector] = sel * 100
            
            # Interaction effect: (wp - wb) * (rp - rb)
            inter = (wp - wb) * (rp - rb)
            interaction_effect += inter
            
            # Total contribution
            sector_contribution[sector] = (alloc + sel + inter) * 100
        
        portfolio_return = sum(t.pnl for t in trades) / total_value * 100 if total_value > 0 else 0
        benchmark_return = sum(self.benchmark_returns.get(d, 0) for d in self._daily_pnl.keys())
        
        return BrinsonAttribution(
            period_start=start_date or min(t.entry_time.date() for t in trades),
            period_end=end_date or max(t.exit_time.date() for t in trades),
            portfolio_return=round(portfolio_return, 2),
            benchmark_return=round(benchmark_return * 100, 2),
            excess_return=round(portfolio_return - benchmark_return * 100, 2),
            allocation_effect=round(allocation_effect * 100, 2),
            selection_effect=round(selection_effect * 100, 2),
            interaction_effect=round(interaction_effect * 100, 2),
            sector_allocation=sector_allocation,
            sector_selection=sector_selection,
            sector_contribution=sector_contribution
        )
    
    def factor_attribution(
        self,
        start_date: date = None,
        end_date: date = None
    ) -> FactorAttribution:
        """
        Perform factor-based attribution using Fama-French style factors.
        
        Factors:
        - Market (beta)
        - Size (SMB - Small minus Big)
        - Value (HML - High minus Low book-to-market)
        - Momentum (UMD - Up minus Down)
        """
        trades = self._filter_trades(start_date, end_date)
        
        if not trades:
            return self._empty_factor(start_date, end_date)
        
        # Calculate portfolio characteristics
        total_value = sum(t.entry_price * t.qty for t in trades)
        
        # Average factor exposures
        avg_beta = np.mean([t.beta for t in trades])
        avg_momentum = np.mean([t.momentum_score for t in trades])
        avg_value = np.mean([t.value_score for t in trades])
        
        # Size exposure (positive = small cap tilt)
        size_scores = []
        for t in trades:
            if t.market_cap == "small":
                size_scores.append(1)
            elif t.market_cap == "large":
                size_scores.append(-1)
            else:
                size_scores.append(0)
        avg_size = np.mean(size_scores) if size_scores else 0
        
        # Get factor returns for period
        period_dates = sorted(self._daily_pnl.keys())
        market_factor_return = sum(self.benchmark_returns.get(d, 0) for d in period_dates)
        
        # Simplified factor returns (would use actual SMB, HML, UMD data)
        smb_return = 0.02  # Placeholder
        hml_return = 0.01  # Placeholder
        umd_return = 0.015  # Placeholder
        
        # Factor contributions
        market_contrib = avg_beta * market_factor_return * 100
        size_contrib = avg_size * smb_return * 100
        value_contrib = avg_value * hml_return * 100
        momentum_contrib = avg_momentum * umd_return * 100
        
        explained = market_contrib + size_contrib + value_contrib + momentum_contrib
        
        # Actual return
        total_pnl = sum(t.pnl for t in trades)
        total_return = total_pnl / total_value * 100 if total_value > 0 else 0
        
        # Alpha (unexplained)
        alpha = total_return - explained
        
        # T-stat (simplified)
        returns = [t.pnl_pct for t in trades]
        alpha_t_stat = alpha / (np.std(returns) / np.sqrt(len(returns))) if returns and np.std(returns) > 0 else 0
        
        return FactorAttribution(
            period_start=start_date or min(t.entry_time.date() for t in trades),
            period_end=end_date or max(t.exit_time.date() for t in trades),
            market_exposure=round(avg_beta, 2),
            size_exposure=round(avg_size, 2),
            value_exposure=round(avg_value, 2),
            momentum_exposure=round(avg_momentum, 2),
            market_contribution=round(market_contrib, 2),
            size_contribution=round(size_contrib, 2),
            value_contribution=round(value_contrib, 2),
            momentum_contribution=round(momentum_contrib, 2),
            alpha=round(alpha, 2),
            alpha_t_stat=round(alpha_t_stat, 2),
            total_return=round(total_return, 2),
            explained_return=round(explained, 2),
            unexplained_return=round(alpha, 2)
        )
    
    def strategy_attribution(self) -> Dict[str, StrategyAttribution]:
        """
        Attribute performance by trading strategy.
        """
        # Group by strategy
        strategy_trades: Dict[str, List[Trade]] = defaultdict(list)
        for trade in self._trades:
            strategy_trades[trade.strategy_name].append(trade)
        
        total_pnl = sum(t.pnl for t in self._trades)
        
        results = {}
        for strategy, trades in strategy_trades.items():
            pnl = sum(t.pnl for t in trades)
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(wins) / len(trades) * 100 if trades else 0
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
            
            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Sharpe (simplified)
            returns = [t.pnl_pct for t in trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
            
            # Regime performance
            regime_perf = {}
            for regime in MarketRegime:
                regime_trades = [t for t in trades if t.regime == regime]
                if regime_trades:
                    regime_perf[regime] = {
                        "trades": len(regime_trades),
                        "pnl": sum(t.pnl for t in regime_trades),
                        "win_rate": len([t for t in regime_trades if t.pnl > 0]) / len(regime_trades) * 100
                    }
            
            results[strategy] = StrategyAttribution(
                strategy_name=strategy,
                total_pnl=round(pnl, 2),
                win_rate=round(win_rate, 1),
                avg_win=round(avg_win, 2),
                avg_loss=round(avg_loss, 2),
                profit_factor=round(profit_factor, 2),
                sharpe_ratio=round(sharpe, 2),
                contribution_pct=round(pnl / total_pnl * 100 if total_pnl != 0 else 0, 1),
                trade_count=len(trades),
                regime_performance=regime_perf
            )
        
        return results
    
    def streak_analysis(self) -> StreakAnalysis:
        """
        Analyze winning and losing streaks.
        """
        if not self._trades:
            return self._empty_streak()
        
        # Sort by exit time
        sorted_trades = sorted(self._trades, key=lambda t: t.exit_time)
        
        # Track streaks
        current_streak = 0
        current_type = None
        max_win = 0
        max_loss = 0
        win_streaks = []
        loss_streaks = []
        
        current_streak_pnl = 0
        best_streak_pnl = 0
        worst_streak_pnl = 0
        
        for trade in sorted_trades:
            is_win = trade.pnl > 0
            
            if current_type is None:
                current_type = "win" if is_win else "loss"
                current_streak = 1
                current_streak_pnl = trade.pnl
            elif (is_win and current_type == "win") or (not is_win and current_type == "loss"):
                current_streak += 1
                current_streak_pnl += trade.pnl
            else:
                # Streak ended
                if current_type == "win":
                    win_streaks.append(current_streak)
                    best_streak_pnl = max(best_streak_pnl, current_streak_pnl)
                else:
                    loss_streaks.append(current_streak)
                    worst_streak_pnl = min(worst_streak_pnl, current_streak_pnl)
                
                # Start new streak
                current_type = "win" if is_win else "loss"
                current_streak = 1
                current_streak_pnl = trade.pnl
        
        # Final streak
        if current_type == "win":
            win_streaks.append(current_streak)
            max_win = max(win_streaks)
        else:
            loss_streaks.append(current_streak)
            max_loss = max(loss_streaks)
        
        # Recovery analysis
        recovery_counts = []
        in_drawdown = False
        drawdown_start = 0
        cumulative = 0
        peak = 0
        
        for i, trade in enumerate(sorted_trades):
            cumulative += trade.pnl
            
            if cumulative > peak:
                if in_drawdown:
                    recovery_counts.append(i - drawdown_start)
                    in_drawdown = False
                peak = cumulative
            elif cumulative < peak and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
        
        return StreakAnalysis(
            current_streak=current_streak,
            current_streak_type=current_type or "",
            max_win_streak=max(win_streaks) if win_streaks else 0,
            max_loss_streak=max(loss_streaks) if loss_streaks else 0,
            avg_win_streak=np.mean(win_streaks) if win_streaks else 0,
            avg_loss_streak=np.mean(loss_streaks) if loss_streaks else 0,
            best_streak_pnl=round(best_streak_pnl, 2),
            worst_streak_pnl=round(worst_streak_pnl, 2),
            avg_recovery_trades=np.mean(recovery_counts) if recovery_counts else 0
        )
    
    def regime_performance(self) -> Dict[MarketRegime, Dict]:
        """
        Analyze performance by market regime.
        """
        results = {}
        
        for regime in MarketRegime:
            trades = [t for t in self._trades if t.regime == regime]
            
            if not trades:
                continue
            
            pnl = sum(t.pnl for t in trades)
            wins = len([t for t in trades if t.pnl > 0])
            
            results[regime] = {
                "total_trades": len(trades),
                "total_pnl": round(pnl, 2),
                "win_rate": round(wins / len(trades) * 100, 1),
                "avg_pnl": round(pnl / len(trades), 2),
                "best_trade": round(max(t.pnl for t in trades), 2),
                "worst_trade": round(min(t.pnl for t in trades), 2)
            }
        
        return results
    
    def calculate_advanced_metrics(self) -> Dict:
        """
        Calculate advanced performance metrics.
        """
        if not self._trades:
            return {}
        
        returns = [t.pnl_pct for t in self._trades]
        pnls = [t.pnl for t in self._trades]
        
        # Basic
        total_return = sum(pnls)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Risk-adjusted
        sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns else 0
        sortino = avg_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar (return / max drawdown)
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win/loss metrics
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        # Drawdown duration
        drawdown_mask = drawdowns > 0
        if any(drawdown_mask):
            drawdown_periods = []
            in_dd = False
            dd_start = 0
            for i, is_dd in enumerate(drawdown_mask):
                if is_dd and not in_dd:
                    in_dd = True
                    dd_start = i
                elif not is_dd and in_dd:
                    in_dd = False
                    drawdown_periods.append(i - dd_start)
            max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        else:
            max_dd_duration = 0
        
        return {
            "total_trades": len(self._trades),
            "total_pnl": round(total_return, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_duration": max_dd_duration,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "best_trade": round(max(pnls), 2) if pnls else 0,
            "worst_trade": round(min(pnls), 2) if pnls else 0
        }
    
    def _filter_trades(
        self,
        start_date: date = None,
        end_date: date = None
    ) -> List[Trade]:
        """Filter trades by date range."""
        trades = self._trades
        
        if start_date:
            trades = [t for t in trades if t.exit_time.date() >= start_date]
        if end_date:
            trades = [t for t in trades if t.exit_time.date() <= end_date]
        
        return trades
    
    def _empty_brinson(self, start: date, end: date) -> BrinsonAttribution:
        return BrinsonAttribution(
            period_start=start or date.today(),
            period_end=end or date.today(),
            portfolio_return=0, benchmark_return=0, excess_return=0,
            allocation_effect=0, selection_effect=0, interaction_effect=0,
            sector_allocation={}, sector_selection={}, sector_contribution={}
        )
    
    def _empty_factor(self, start: date, end: date) -> FactorAttribution:
        return FactorAttribution(
            period_start=start or date.today(),
            period_end=end or date.today(),
            market_exposure=0, size_exposure=0, value_exposure=0, momentum_exposure=0,
            market_contribution=0, size_contribution=0, value_contribution=0, momentum_contribution=0,
            alpha=0, alpha_t_stat=0,
            total_return=0, explained_return=0, unexplained_return=0
        )
    
    def _empty_streak(self) -> StreakAnalysis:
        return StreakAnalysis(
            current_streak=0, current_streak_type="",
            max_win_streak=0, max_loss_streak=0,
            avg_win_streak=0, avg_loss_streak=0,
            best_streak_pnl=0, worst_streak_pnl=0,
            avg_recovery_trades=0
        )
