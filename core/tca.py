"""
Transaction Cost Analysis (TCA) - Pre-trade and post-trade analysis.

Features:
- Pre-trade cost estimation
- Post-trade execution benchmarking
- Arrival price, VWAP, TWAP benchmarks
- Implementation shortfall analysis
- Execution quality scoring
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreTradeEstimate:
    """Pre-trade cost estimation."""
    symbol: str
    side: str
    qty: int
    current_price: float
    
    # Cost estimates (in basis points)
    spread_cost_bps: float
    market_impact_bps: float
    timing_risk_bps: float
    commission_bps: float
    total_cost_bps: float
    
    # Cost estimates (in rupees)
    spread_cost_rs: float
    market_impact_rs: float
    timing_risk_rs: float
    commission_rs: float
    total_cost_rs: float
    
    # Recommendations
    recommended_algo: str
    recommended_duration_minutes: int
    participation_rate: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionBenchmark:
    """Benchmark prices for post-trade analysis."""
    arrival_price: float           # Price at decision time
    vwap: float                    # Volume-weighted average price during execution
    twap: float                    # Time-weighted average price during execution
    open_price: float              # Day's open
    close_price: float             # Day's close (or current for intraday)
    interval_high: float           # High during execution window
    interval_low: float            # Low during execution window


@dataclass
class PostTradeAnalysis:
    """Post-trade execution analysis."""
    order_id: str
    symbol: str
    side: str
    qty: int
    executed_qty: int
    
    # Execution prices
    avg_execution_price: float
    arrival_price: float
    decision_price: float  # Price when trade decision was made
    
    # Benchmarks
    benchmarks: ExecutionBenchmark
    
    # Slippage vs benchmarks (in basis points)
    slippage_vs_arrival_bps: float
    slippage_vs_vwap_bps: float
    slippage_vs_twap_bps: float
    
    # Implementation shortfall
    implementation_shortfall_bps: float
    implementation_shortfall_rs: float
    
    # Breakdown
    delay_cost_bps: float      # Cost of delay between decision and execution start
    trading_cost_bps: float    # Cost during actual trading
    opportunity_cost_bps: float  # Cost of unfilled portion
    
    # Quality metrics
    execution_quality_score: float  # 0-100 score
    fill_rate: float               # Executed / Target
    execution_time_seconds: float
    
    # Market conditions
    market_volatility: float
    market_volume: int
    participation_rate: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TCAReport:
    """Aggregate TCA report for multiple trades."""
    period_start: datetime
    period_end: datetime
    total_trades: int
    total_volume: int
    total_value: float
    
    # Aggregate metrics (bps)
    avg_slippage_vs_arrival: float
    avg_slippage_vs_vwap: float
    avg_implementation_shortfall: float
    
    # Costs
    total_trading_cost_rs: float
    total_trading_cost_bps: float
    
    # Quality
    avg_execution_quality: float
    avg_fill_rate: float
    
    # By category
    by_side: Dict[str, Dict]
    by_symbol: Dict[str, Dict]
    by_algo: Dict[str, Dict]
    
    # Outliers
    worst_trades: List[PostTradeAnalysis]
    best_trades: List[PostTradeAnalysis]


class TransactionCostAnalyzer:
    """
    Comprehensive transaction cost analysis system.
    """
    
    def __init__(
        self,
        commission_per_side_bps: float = 3.0,  # ~3 bps per side
        default_spread_bps: float = 5.0,
        market_impact_model=None,
        slippage_model=None
    ):
        self.commission_per_side_bps = commission_per_side_bps
        self.default_spread_bps = default_spread_bps
        self.market_impact_model = market_impact_model
        self.slippage_model = slippage_model
        
        # Store historical analyses
        self._analyses: List[PostTradeAnalysis] = []
        self._pre_trade_estimates: Dict[str, PreTradeEstimate] = {}
    
    def pre_trade_analysis(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        volatility: float = 0.02,
        avg_daily_volume: int = 1_000_000,
        spread_bps: float = None,
        urgency: float = 0.5
    ) -> PreTradeEstimate:
        """
        Estimate costs before executing a trade.
        
        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            qty: Quantity to trade
            price: Current price
            volatility: Daily volatility
            avg_daily_volume: Average daily volume
            spread_bps: Bid-ask spread in basis points
            urgency: 0 (patient) to 1 (urgent)
        
        Returns:
            PreTradeEstimate with cost breakdown and recommendations
        """
        trade_value = qty * price
        
        # 1. Spread cost (half spread to cross)
        spread = spread_bps if spread_bps else self.default_spread_bps
        spread_cost_bps = spread / 2
        spread_cost_rs = trade_value * spread_cost_bps / 10000
        
        # 2. Market impact
        if self.market_impact_model:
            impact = self.market_impact_model.estimate_impact(
                symbol, qty, price, side, execution_time_minutes=30
            )
            market_impact_bps = impact.total_impact_bps
        else:
            # Simple model: impact proportional to order size relative to ADV
            participation = qty / avg_daily_volume
            market_impact_bps = participation * 100 * 10  # Rough estimate
        
        market_impact_rs = trade_value * market_impact_bps / 10000
        
        # 3. Timing risk (risk of price moving against us)
        # Higher for more volatile stocks and longer execution
        timing_risk_bps = volatility * 10000 * 0.1  # ~10% of daily vol
        timing_risk_bps *= (1 - urgency)  # More patient = more timing risk
        timing_risk_rs = trade_value * timing_risk_bps / 10000
        
        # 4. Commission
        commission_bps = self.commission_per_side_bps * 2  # Round trip
        commission_rs = trade_value * commission_bps / 10000
        
        # Total
        total_bps = spread_cost_bps + market_impact_bps + timing_risk_bps + commission_bps
        total_rs = spread_cost_rs + market_impact_rs + timing_risk_rs + commission_rs
        
        # Recommendations
        participation = qty / avg_daily_volume
        
        if participation < 0.01:
            # Small order: can be aggressive
            recommended_algo = "MARKET" if urgency > 0.7 else "TWAP"
            duration = 5 if urgency > 0.7 else 15
            rec_participation = 0.20
        elif participation < 0.05:
            # Medium order
            recommended_algo = "TWAP" if urgency > 0.5 else "VWAP"
            duration = 15 if urgency > 0.5 else 30
            rec_participation = 0.10
        elif participation < 0.10:
            # Large order
            recommended_algo = "VWAP" if urgency > 0.3 else "IS"
            duration = 30 if urgency > 0.3 else 60
            rec_participation = 0.07
        else:
            # Very large order
            recommended_algo = "IS"
            duration = 60
            rec_participation = 0.05
        
        estimate = PreTradeEstimate(
            symbol=symbol,
            side=side,
            qty=qty,
            current_price=price,
            spread_cost_bps=round(spread_cost_bps, 2),
            market_impact_bps=round(market_impact_bps, 2),
            timing_risk_bps=round(timing_risk_bps, 2),
            commission_bps=round(commission_bps, 2),
            total_cost_bps=round(total_bps, 2),
            spread_cost_rs=round(spread_cost_rs, 2),
            market_impact_rs=round(market_impact_rs, 2),
            timing_risk_rs=round(timing_risk_rs, 2),
            commission_rs=round(commission_rs, 2),
            total_cost_rs=round(total_rs, 2),
            recommended_algo=recommended_algo,
            recommended_duration_minutes=duration,
            participation_rate=rec_participation
        )
        
        # Cache for post-trade comparison
        self._pre_trade_estimates[f"{symbol}_{side}_{datetime.now().strftime('%H%M')}"] = estimate
        
        return estimate
    
    def post_trade_analysis(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        executed_qty: int,
        fills: List[Tuple[int, float, datetime]],  # (qty, price, time)
        decision_time: datetime,
        arrival_price: float,
        benchmarks: ExecutionBenchmark,
        market_volume: int = None
    ) -> PostTradeAnalysis:
        """
        Analyze execution quality after trade completion.
        
        Args:
            order_id: Unique order identifier
            symbol: Stock symbol
            side: "BUY" or "SELL"
            qty: Target quantity
            executed_qty: Actually executed quantity
            fills: List of (quantity, price, timestamp) tuples
            decision_time: When trade decision was made
            arrival_price: Price at decision time
            benchmarks: Benchmark prices for comparison
            market_volume: Total market volume during execution
        
        Returns:
            PostTradeAnalysis with detailed execution breakdown
        """
        if not fills:
            logger.warning(f"No fills for order {order_id}")
            return None
        
        # Calculate average execution price
        total_value = sum(q * p for q, p, _ in fills)
        total_qty = sum(q for q, _, _ in fills)
        avg_exec_price = total_value / total_qty if total_qty > 0 else 0
        
        # Execution time
        first_fill_time = min(t for _, _, t in fills)
        last_fill_time = max(t for _, _, t in fills)
        exec_time_seconds = (last_fill_time - first_fill_time).total_seconds()
        
        # Calculate slippage vs benchmarks
        def calc_slippage(exec_price, benchmark, side):
            if benchmark == 0:
                return 0
            if side == "BUY":
                return (exec_price - benchmark) / benchmark * 10000
            else:
                return (benchmark - exec_price) / benchmark * 10000
        
        slippage_arrival = calc_slippage(avg_exec_price, arrival_price, side)
        slippage_vwap = calc_slippage(avg_exec_price, benchmarks.vwap, side)
        slippage_twap = calc_slippage(avg_exec_price, benchmarks.twap, side)
        
        # Implementation shortfall breakdown
        # IS = Delay Cost + Trading Cost + Opportunity Cost
        
        # Delay cost: price move from decision to first execution
        delay_cost = calc_slippage(benchmarks.arrival_price, arrival_price, side)
        
        # Trading cost: price move during execution
        trading_cost = calc_slippage(avg_exec_price, benchmarks.arrival_price, side)
        
        # Opportunity cost: unfilled portion at final price
        fill_rate = executed_qty / qty if qty > 0 else 1
        if fill_rate < 1:
            unfilled_cost = calc_slippage(benchmarks.close_price, avg_exec_price, side)
            opportunity_cost = unfilled_cost * (1 - fill_rate)
        else:
            opportunity_cost = 0
        
        # Total implementation shortfall
        impl_shortfall_bps = delay_cost + trading_cost + opportunity_cost
        trade_value = executed_qty * avg_exec_price
        impl_shortfall_rs = trade_value * impl_shortfall_bps / 10000
        
        # Execution quality score (0-100)
        # Based on multiple factors:
        # 1. Slippage vs VWAP (40%)
        # 2. Fill rate (30%)
        # 3. Implementation shortfall (30%)
        
        vwap_score = max(0, 100 - abs(slippage_vwap) * 5)  # Lose 5 points per bps
        fill_score = fill_rate * 100
        is_score = max(0, 100 - abs(impl_shortfall_bps) * 3)  # Lose 3 points per bps
        
        quality_score = (vwap_score * 0.4 + fill_score * 0.3 + is_score * 0.3)
        
        # Market volatility during execution
        price_range = benchmarks.interval_high - benchmarks.interval_low
        volatility = (price_range / benchmarks.vwap) if benchmarks.vwap > 0 else 0
        
        # Participation rate
        participation = executed_qty / market_volume if market_volume else 0
        
        analysis = PostTradeAnalysis(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            executed_qty=executed_qty,
            avg_execution_price=round(avg_exec_price, 2),
            arrival_price=round(arrival_price, 2),
            decision_price=round(arrival_price, 2),
            benchmarks=benchmarks,
            slippage_vs_arrival_bps=round(slippage_arrival, 2),
            slippage_vs_vwap_bps=round(slippage_vwap, 2),
            slippage_vs_twap_bps=round(slippage_twap, 2),
            implementation_shortfall_bps=round(impl_shortfall_bps, 2),
            implementation_shortfall_rs=round(impl_shortfall_rs, 2),
            delay_cost_bps=round(delay_cost, 2),
            trading_cost_bps=round(trading_cost, 2),
            opportunity_cost_bps=round(opportunity_cost, 2),
            execution_quality_score=round(quality_score, 1),
            fill_rate=round(fill_rate, 4),
            execution_time_seconds=round(exec_time_seconds, 1),
            market_volatility=round(volatility, 4),
            market_volume=market_volume or 0,
            participation_rate=round(participation, 4)
        )
        
        self._analyses.append(analysis)
        return analysis
    
    def generate_report(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        symbol: str = None
    ) -> TCAReport:
        """
        Generate aggregate TCA report for a period.
        """
        # Filter analyses
        analyses = self._analyses
        
        if start_date:
            analyses = [a for a in analyses if a.timestamp >= start_date]
        if end_date:
            analyses = [a for a in analyses if a.timestamp <= end_date]
        if symbol:
            analyses = [a for a in analyses if a.symbol == symbol]
        
        if not analyses:
            logger.warning("No analyses found for report")
            return None
        
        # Aggregate metrics
        total_volume = sum(a.executed_qty for a in analyses)
        total_value = sum(a.executed_qty * a.avg_execution_price for a in analyses)
        
        avg_slippage_arrival = np.mean([a.slippage_vs_arrival_bps for a in analyses])
        avg_slippage_vwap = np.mean([a.slippage_vs_vwap_bps for a in analyses])
        avg_is = np.mean([a.implementation_shortfall_bps for a in analyses])
        
        total_cost_rs = sum(a.implementation_shortfall_rs for a in analyses)
        total_cost_bps = (total_cost_rs / total_value * 10000) if total_value > 0 else 0
        
        avg_quality = np.mean([a.execution_quality_score for a in analyses])
        avg_fill_rate = np.mean([a.fill_rate for a in analyses])
        
        # By side
        by_side = {}
        for side in ["BUY", "SELL"]:
            side_analyses = [a for a in analyses if a.side == side]
            if side_analyses:
                by_side[side] = {
                    "count": len(side_analyses),
                    "avg_slippage_bps": np.mean([a.slippage_vs_vwap_bps for a in side_analyses]),
                    "avg_quality": np.mean([a.execution_quality_score for a in side_analyses])
                }
        
        # By symbol
        symbols = set(a.symbol for a in analyses)
        by_symbol = {}
        for sym in symbols:
            sym_analyses = [a for a in analyses if a.symbol == sym]
            by_symbol[sym] = {
                "count": len(sym_analyses),
                "total_volume": sum(a.executed_qty for a in sym_analyses),
                "avg_slippage_bps": np.mean([a.slippage_vs_vwap_bps for a in sym_analyses]),
                "avg_quality": np.mean([a.execution_quality_score for a in sym_analyses])
            }
        
        # Worst and best trades
        sorted_by_is = sorted(analyses, key=lambda a: a.implementation_shortfall_bps)
        worst = sorted_by_is[-5:] if len(sorted_by_is) >= 5 else sorted_by_is
        best = sorted_by_is[:5]
        
        report = TCAReport(
            period_start=min(a.timestamp for a in analyses),
            period_end=max(a.timestamp for a in analyses),
            total_trades=len(analyses),
            total_volume=total_volume,
            total_value=total_value,
            avg_slippage_vs_arrival=round(avg_slippage_arrival, 2),
            avg_slippage_vs_vwap=round(avg_slippage_vwap, 2),
            avg_implementation_shortfall=round(avg_is, 2),
            total_trading_cost_rs=round(total_cost_rs, 2),
            total_trading_cost_bps=round(total_cost_bps, 2),
            avg_execution_quality=round(avg_quality, 1),
            avg_fill_rate=round(avg_fill_rate, 4),
            by_side=by_side,
            by_symbol=by_symbol,
            by_algo={},  # TODO: Track by algo type
            worst_trades=list(reversed(worst)),
            best_trades=best
        )
        
        return report
    
    def compare_to_pre_trade(
        self,
        analysis: PostTradeAnalysis
    ) -> Dict:
        """
        Compare actual execution to pre-trade estimate.
        """
        # Find matching pre-trade estimate
        key_pattern = f"{analysis.symbol}_{analysis.side}"
        matching_estimates = [
            (k, v) for k, v in self._pre_trade_estimates.items()
            if k.startswith(key_pattern)
        ]
        
        if not matching_estimates:
            return {"error": "No matching pre-trade estimate found"}
        
        # Use most recent estimate
        _, estimate = sorted(matching_estimates, key=lambda x: x[0])[-1]
        
        # Compare
        actual_cost_bps = analysis.implementation_shortfall_bps
        estimated_cost_bps = estimate.total_cost_bps
        
        return {
            "estimated_cost_bps": estimated_cost_bps,
            "actual_cost_bps": actual_cost_bps,
            "difference_bps": actual_cost_bps - estimated_cost_bps,
            "estimation_accuracy": 1 - abs(actual_cost_bps - estimated_cost_bps) / max(abs(estimated_cost_bps), 1),
            "recommended_algo": estimate.recommended_algo,
            "components": {
                "spread": {"estimated": estimate.spread_cost_bps, "note": "Half spread cost"},
                "impact": {"estimated": estimate.market_impact_bps, "note": "Market impact"},
                "timing": {"estimated": estimate.timing_risk_bps, "note": "Timing/delay risk"},
                "commission": {"estimated": estimate.commission_bps, "note": "Brokerage fees"}
            }
        }


def calculate_vwap(trades: List[Tuple[float, int, datetime]]) -> float:
    """
    Calculate VWAP from trade data.
    trades: List of (price, volume, timestamp)
    """
    if not trades:
        return 0.0
    
    total_value = sum(price * volume for price, volume, _ in trades)
    total_volume = sum(volume for _, volume, _ in trades)
    
    return total_value / total_volume if total_volume > 0 else 0.0


def calculate_twap(trades: List[Tuple[float, int, datetime]]) -> float:
    """
    Calculate TWAP from trade data.
    """
    if not trades:
        return 0.0
    
    return np.mean([price for price, _, _ in trades])
