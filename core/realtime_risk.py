"""
Real-Time Risk Engine - Continuous risk monitoring and stress testing.

Features:
- Real-time aggregate risk calculation
- Stress testing scenarios
- Greeks calculation (for options)
- P&L attribution
- Automatic position adjustment recommendations
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress test scenario."""
    name: str
    description: str
    market_shock_pct: float      # Overall market move
    volatility_shock_pct: float  # VIX/volatility spike
    sector_shocks: Dict[str, float] = field(default_factory=dict)  # Sector-specific moves
    correlation_spike: float = 0.0  # Increase in correlations
    
    def __post_init__(self):
        if not self.sector_shocks:
            self.sector_shocks = {}


@dataclass
class StressTestResult:
    """Results of a stress test."""
    scenario_name: str
    portfolio_pnl: float
    portfolio_pnl_pct: float
    position_impacts: Dict[str, float]  # symbol -> P&L impact
    worst_position: str
    worst_position_loss: float
    var_under_stress: float
    margin_call_risk: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAlert:
    """Real-time risk alert."""
    alert_id: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    category: str  # "VAR", "EXPOSURE", "DRAWDOWN", "CORRELATION", "MARGIN"
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class GreeksSnapshot:
    """Options greeks (placeholder for future options support)."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


# Predefined stress scenarios for Indian markets
PREDEFINED_SCENARIOS = [
    StressScenario(
        name="Market Crash",
        description="2020 COVID-style crash: -10% in a day",
        market_shock_pct=-10.0,
        volatility_shock_pct=100.0,
        sector_shocks={"Banking": -12.0, "IT": -8.0, "Pharma": -5.0},
        correlation_spike=0.3
    ),
    StressScenario(
        name="Flash Crash",
        description="Sudden -5% move with quick recovery",
        market_shock_pct=-5.0,
        volatility_shock_pct=50.0,
        correlation_spike=0.2
    ),
    StressScenario(
        name="Sector Rotation",
        description="IT/Tech selloff, defensive rally",
        market_shock_pct=-2.0,
        volatility_shock_pct=20.0,
        sector_shocks={"IT": -8.0, "Pharma": 3.0, "FMCG": 2.0, "Banking": -4.0}
    ),
    StressScenario(
        name="Rate Hike Shock",
        description="RBI surprise rate hike",
        market_shock_pct=-3.0,
        volatility_shock_pct=30.0,
        sector_shocks={"Banking": -6.0, "RealEstate": -8.0, "Auto": -5.0, "IT": -2.0}
    ),
    StressScenario(
        name="Currency Crisis",
        description="INR depreciation shock",
        market_shock_pct=-4.0,
        volatility_shock_pct=40.0,
        sector_shocks={"IT": 5.0, "Pharma": 3.0, "Oil": -10.0, "FMCG": -4.0}
    ),
    StressScenario(
        name="Bull Rally",
        description="Strong positive momentum",
        market_shock_pct=5.0,
        volatility_shock_pct=-20.0,
        sector_shocks={"Banking": 7.0, "Auto": 6.0, "RealEstate": 8.0}
    ),
    StressScenario(
        name="Volatility Spike",
        description="VIX doubles, market flat",
        market_shock_pct=0.0,
        volatility_shock_pct=100.0,
        correlation_spike=0.4
    ),
]


class RealTimeRiskEngine:
    """
    Continuous risk monitoring engine.
    
    Features:
    - Sub-second risk updates
    - Automatic alert generation
    - Stress testing
    - Position recommendations
    """
    
    def __init__(
        self,
        portfolio_risk_engine,
        update_interval_ms: int = 1000,
        var_confidence: float = 0.95,
        max_alerts_per_hour: int = 10
    ):
        self.portfolio_engine = portfolio_risk_engine
        self.update_interval_ms = update_interval_ms
        self.var_confidence = var_confidence
        self.max_alerts_per_hour = max_alerts_per_hour
        
        # Risk state
        self._current_var: float = 0.0
        self._current_exposure: float = 0.0
        self._current_pnl: float = 0.0
        
        # Alerts
        self._alerts: deque = deque(maxlen=100)
        self._alert_counter = 0
        self._alerts_this_hour: deque = deque(maxlen=self.max_alerts_per_hour)
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        self._update_callbacks: List[Callable] = []
        
        # Monitoring thread
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Historical tracking for intraday analysis
        self._intraday_pnl: deque = deque(maxlen=500)  # ~8 hours of minute data
        self._intraday_var: deque = deque(maxlen=500)
        
        # Stress test results cache
        self._stress_results: Dict[str, StressTestResult] = {}
    
    def start(self):
        """Start real-time monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Real-time risk engine started (interval: {self.update_interval_ms}ms)")
    
    def stop(self):
        """Stop real-time monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Real-time risk engine stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._update_risk_metrics()
                self._check_for_alerts()
                
                # Notify callbacks
                for callback in self._update_callbacks:
                    try:
                        callback(self.get_snapshot())
                    except Exception as e:
                        logger.error(f"Update callback error: {e}")
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
            
            time.sleep(self.update_interval_ms / 1000)
    
    def _update_risk_metrics(self):
        """Update current risk metrics."""
        with self._lock:
            metrics = self.portfolio_engine.calculate_metrics()
            
            self._current_var = metrics.var_95
            self._current_exposure = metrics.gross_exposure
            
            # Calculate current P&L
            positions = self.portfolio_engine.positions
            self._current_pnl = sum(p.pnl for p in positions.values())
            
            # Record intraday history
            self._intraday_pnl.append((datetime.now(), self._current_pnl))
            self._intraday_var.append((datetime.now(), self._current_var))
    
    def _check_for_alerts(self):
        """Check for risk limit breaches and generate alerts."""
        # Rate limit alerts
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        self._alerts_this_hour = deque(
            [t for t in self._alerts_this_hour if t > hour_ago],
            maxlen=self.max_alerts_per_hour
        )
        
        if len(self._alerts_this_hour) >= self.max_alerts_per_hour:
            return
        
        # Check limits
        breaches = self.portfolio_engine.check_limits()
        
        for limit_name, status, current, threshold in breaches:
            severity = "CRITICAL" if status == "BREACH" else "WARNING"
            
            # Determine category
            category = "EXPOSURE"
            if "VAR" in limit_name.upper():
                category = "VAR"
            elif "DRAWDOWN" in limit_name.upper():
                category = "DRAWDOWN"
            elif "CORRELATION" in limit_name.upper():
                category = "CORRELATION"
            
            self._create_alert(
                severity=severity,
                category=category,
                message=f"{limit_name} {status}: {current:.2f} vs limit {threshold:.2f}",
                current_value=current,
                threshold=threshold
            )
    
    def _create_alert(
        self,
        severity: str,
        category: str,
        message: str,
        current_value: float,
        threshold: float
    ):
        """Create and dispatch a risk alert."""
        self._alert_counter += 1
        alert = RiskAlert(
            alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._alert_counter}",
            severity=severity,
            category=category,
            message=message,
            current_value=current_value,
            threshold=threshold
        )
        
        with self._lock:
            self._alerts.append(alert)
            self._alerts_this_hour.append(datetime.now())
        
        logger.warning(f"[{severity}] {category}: {message}")
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def run_stress_test(self, scenario: StressScenario) -> StressTestResult:
        """
        Run a stress test scenario.
        
        Calculates portfolio impact under the given stress conditions.
        """
        positions = dict(self.portfolio_engine.positions)
        
        if not positions:
            return StressTestResult(
                scenario_name=scenario.name,
                portfolio_pnl=0,
                portfolio_pnl_pct=0,
                position_impacts={},
                worst_position="",
                worst_position_loss=0,
                var_under_stress=0,
                margin_call_risk=False
            )
        
        position_impacts = {}
        total_impact = 0.0
        
        for symbol, pos in positions.items():
            # Get sector-specific shock or use market shock
            sector = pos.sector or "Other"
            shock_pct = scenario.sector_shocks.get(sector, scenario.market_shock_pct)
            
            # Adjust for beta
            position_shock = shock_pct * pos.beta
            
            # Calculate impact
            if pos.side == "LONG":
                impact = pos.market_value * (position_shock / 100)
            else:
                # Shorts benefit from market decline
                impact = -pos.market_value * (position_shock / 100)
            
            position_impacts[symbol] = impact
            total_impact += impact
        
        # Find worst position
        worst_symbol = min(position_impacts.keys(), key=lambda s: position_impacts[s])
        worst_loss = position_impacts[worst_symbol]
        
        # Calculate portfolio value and P&L percentage
        portfolio_value = self.portfolio_engine.capital + self._current_pnl
        pnl_pct = (total_impact / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # Estimate VaR under stress (simplified: scale by volatility shock)
        var_multiplier = 1 + (scenario.volatility_shock_pct / 100)
        var_under_stress = self._current_var * var_multiplier
        
        # Check margin call risk
        # Assume margin call if loss > 30% of capital
        margin_call_risk = abs(total_impact) > (self.portfolio_engine.capital * 0.30)
        
        result = StressTestResult(
            scenario_name=scenario.name,
            portfolio_pnl=total_impact,
            portfolio_pnl_pct=pnl_pct,
            position_impacts=position_impacts,
            worst_position=worst_symbol,
            worst_position_loss=worst_loss,
            var_under_stress=var_under_stress,
            margin_call_risk=margin_call_risk
        )
        
        self._stress_results[scenario.name] = result
        return result
    
    def run_all_stress_tests(self) -> Dict[str, StressTestResult]:
        """Run all predefined stress scenarios."""
        results = {}
        for scenario in PREDEFINED_SCENARIOS:
            results[scenario.name] = self.run_stress_test(scenario)
        return results
    
    def custom_stress_test(
        self,
        name: str,
        market_shock_pct: float,
        volatility_shock_pct: float = 0.0,
        sector_shocks: Dict[str, float] = None
    ) -> StressTestResult:
        """Run a custom stress test."""
        scenario = StressScenario(
            name=name,
            description=f"Custom: {market_shock_pct}% market shock",
            market_shock_pct=market_shock_pct,
            volatility_shock_pct=volatility_shock_pct,
            sector_shocks=sector_shocks or {}
        )
        return self.run_stress_test(scenario)
    
    def get_position_recommendations(self) -> List[Dict]:
        """
        Generate position adjustment recommendations based on risk analysis.
        """
        recommendations = []
        
        with self._lock:
            metrics = self.portfolio_engine.calculate_metrics()
            breaches = self.portfolio_engine.check_limits()
            positions = dict(self.portfolio_engine.positions)
        
        # Check for concentration issues
        if metrics.largest_position_pct > 8:  # Warning at 8%, limit typically 10%
            # Find the largest position
            largest = max(positions.values(), key=lambda p: p.market_value)
            excess_pct = metrics.largest_position_pct - 8
            reduce_qty = int(largest.qty * (excess_pct / metrics.largest_position_pct))
            
            recommendations.append({
                "type": "REDUCE",
                "symbol": largest.symbol,
                "current_qty": largest.qty,
                "recommended_qty": largest.qty - reduce_qty,
                "reason": f"Position concentration {metrics.largest_position_pct:.1f}% exceeds warning threshold",
                "priority": "HIGH"
            })
        
        # Check for correlation issues
        high_corr_pairs = self.portfolio_engine.correlation_tracker.get_highly_correlated_pairs(0.7)
        for sym1, sym2, corr in high_corr_pairs[:3]:  # Top 3 correlated pairs
            if sym1 in positions and sym2 in positions:
                # Recommend reducing the smaller position
                pos1, pos2 = positions[sym1], positions[sym2]
                smaller = pos1 if pos1.market_value < pos2.market_value else pos2
                
                recommendations.append({
                    "type": "HEDGE_OR_REDUCE",
                    "symbol": smaller.symbol,
                    "correlated_with": pos1.symbol if smaller == pos2 else pos2.symbol,
                    "correlation": corr,
                    "reason": f"High correlation ({corr:.2f}) increases portfolio risk",
                    "priority": "MEDIUM"
                })
        
        # Check sector concentration
        for sector, exposure_pct in metrics.sector_exposures.items():
            if exposure_pct > 25:  # Warning at 25%, limit typically 30%
                sector_positions = [p for p in positions.values() if p.sector == sector]
                if sector_positions:
                    # Recommend reducing largest in sector
                    largest_in_sector = max(sector_positions, key=lambda p: p.market_value)
                    recommendations.append({
                        "type": "REDUCE",
                        "symbol": largest_in_sector.symbol,
                        "reason": f"{sector} sector exposure {exposure_pct:.1f}% exceeds warning threshold",
                        "priority": "MEDIUM"
                    })
        
        # Drawdown-based recommendations
        if metrics.current_drawdown > 5:  # 5% drawdown warning
            recommendations.append({
                "type": "REDUCE_EXPOSURE",
                "reason": f"Portfolio drawdown {metrics.current_drawdown:.1f}% - consider reducing overall exposure",
                "recommended_action": "Reduce position sizes by 20-30%",
                "priority": "HIGH"
            })
        
        return recommendations
    
    def get_intraday_pnl_series(self) -> List[Tuple[datetime, float]]:
        """Get intraday P&L time series."""
        return list(self._intraday_pnl)
    
    def get_intraday_var_series(self) -> List[Tuple[datetime, float]]:
        """Get intraday VaR time series."""
        return list(self._intraday_var)
    
    def get_alerts(self, unacknowledged_only: bool = False) -> List[RiskAlert]:
        """Get recent alerts."""
        with self._lock:
            alerts = list(self._alerts)
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    break
    
    def get_snapshot(self) -> Dict:
        """Get current risk snapshot."""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "current_pnl": self._current_pnl,
                "current_var": self._current_var,
                "current_exposure": self._current_exposure,
                "active_alerts": len([a for a in self._alerts if not a.acknowledged]),
                "stress_test_results": {
                    name: {
                        "pnl": r.portfolio_pnl,
                        "pnl_pct": r.portfolio_pnl_pct,
                        "margin_call_risk": r.margin_call_risk
                    }
                    for name, r in self._stress_results.items()
                }
            }
    
    def on_alert(self, callback: Callable):
        """Register callback for risk alerts."""
        self._alert_callbacks.append(callback)
    
    def on_update(self, callback: Callable):
        """Register callback for risk updates."""
        self._update_callbacks.append(callback)


class GreeksCalculator:
    """
    Greeks calculator for options positions.
    Placeholder for future options trading support.
    """
    
    @staticmethod
    def black_scholes_greeks(
        spot: float,
        strike: float,
        time_to_expiry: float,  # Years
        risk_free_rate: float,
        volatility: float,
        option_type: str = "call"
    ) -> GreeksSnapshot:
        """
        Calculate Black-Scholes greeks.
        """
        from scipy.stats import norm
        
        if time_to_expiry <= 0 or volatility <= 0:
            return GreeksSnapshot()
        
        sqrt_t = np.sqrt(time_to_expiry)
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        
        if option_type.lower() == "call":
            delta = norm.cdf(d1)
            theta = (-spot * norm.pdf(d1) * volatility / (2 * sqrt_t) - 
                    risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = (-spot * norm.pdf(d1) * volatility / (2 * sqrt_t) + 
                    risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
        
        gamma = norm.pdf(d1) / (spot * volatility * sqrt_t)
        vega = spot * norm.pdf(d1) * sqrt_t / 100  # Per 1% move
        rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
        
        return GreeksSnapshot(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )
