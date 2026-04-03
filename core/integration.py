"""
Integration Module - Combines all institutional-grade components.

This module provides a unified interface to use all the new features together:
- OMS for order management
- Execution algorithms
- Portfolio risk
- Real-time risk monitoring
- Event calendar
- TCA
- Analytics
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from core.oms import (
    OrderManagementSystem, Order, OrderSide, OrderType, 
    OrderState, OrderPriority, TimeInForce
)
from core.execution_algos import (
    ExecutionAlgoManager, ExecutionAlgoType, TWAPAlgo, VWAPAlgo
)
from core.market_impact import MarketImpactModel, SlippageModel
from core.portfolio_risk import PortfolioRiskEngine, RiskLimits, CorrelationTracker
from core.realtime_risk import RealTimeRiskEngine, PREDEFINED_SCENARIOS
from core.tca import TransactionCostAnalyzer, PreTradeEstimate
from core.order_types import ComplexOrderManager, BracketOrder, TrailingStopOrder
from core.event_calendar import EventCalendar, EventType
from core.multi_timeframe import MultiTimeframeAnalyzer, Timeframe
from core.analytics import MonteCarloSimulator, AdvancedMetrics
from core.database import get_database, TradeRecord
from core.optimization import ParameterOptimizer, Parameter, BacktestObjective

logger = logging.getLogger(__name__)


class InstitutionalTradingSystem:
    """
    Unified trading system integrating all institutional-grade components.
    
    Usage:
        system = InstitutionalTradingSystem(capital=1_000_000)
        system.initialize()
        
        # Check if trade is allowed
        can_trade, reason = system.pre_trade_check("RELIANCE", "BUY", 100, 2450.0)
        
        # Create and submit order
        if can_trade:
            order = system.submit_order("RELIANCE", "BUY", 100, 2450.0)
        
        # Monitor risk
        risk_status = system.get_risk_status()
    """
    
    def __init__(
        self,
        capital: float,
        config: Dict = None,
        sector_map: Dict[str, str] = None,
        broker_adapter=None
    ):
        self.capital = capital
        self.config = config or {}
        self.sector_map = sector_map or {}
        self.broker_adapter = broker_adapter
        
        # Core components (initialized in initialize())
        self.oms: Optional[OrderManagementSystem] = None
        self.execution_manager: Optional[ExecutionAlgoManager] = None
        self.market_impact: Optional[MarketImpactModel] = None
        self.slippage_model: Optional[SlippageModel] = None
        self.portfolio_risk: Optional[PortfolioRiskEngine] = None
        self.realtime_risk: Optional[RealTimeRiskEngine] = None
        self.tca: Optional[TransactionCostAnalyzer] = None
        self.complex_orders: Optional[ComplexOrderManager] = None
        self.calendar: Optional[EventCalendar] = None
        self.database = None
        
        # Analyzers (created on demand)
        self._mtf_analyzers: Dict[str, MultiTimeframeAnalyzer] = {}
        
        self._initialized = False
    
    def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return
        
        logger.info("Initializing Institutional Trading System...")
        
        # Risk limits
        limits = RiskLimits(
            max_var_95_pct=self.config.get("max_var_pct", 2.0),
            max_gross_exposure_pct=self.config.get("max_gross_exposure", 200.0),
            max_net_exposure_pct=self.config.get("max_net_exposure", 100.0),
            max_single_position_pct=self.config.get("max_position_pct", 10.0),
            max_sector_exposure_pct=self.config.get("max_sector_pct", 30.0),
            max_correlation=self.config.get("max_correlation", 0.7),
            max_drawdown_pct=self.config.get("max_drawdown", 10.0)
        )
        
        # Initialize components
        self.oms = OrderManagementSystem(
            broker_adapter=self.broker_adapter,
            max_orders_per_day=self.config.get("max_orders_per_day", 100)
        )
        
        self.market_impact = MarketImpactModel()
        self.slippage_model = SlippageModel()
        
        self.portfolio_risk = PortfolioRiskEngine(
            capital=self.capital,
            limits=limits,
            sector_map=self.sector_map
        )
        
        self.realtime_risk = RealTimeRiskEngine(
            portfolio_risk_engine=self.portfolio_risk
        )
        
        self.tca = TransactionCostAnalyzer(
            market_impact_model=self.market_impact,
            slippage_model=self.slippage_model
        )
        
        self.execution_manager = ExecutionAlgoManager(
            oms=self.oms,
            price_fetcher=self._get_price
        )
        
        self.complex_orders = ComplexOrderManager(
            broker_adapter=self.broker_adapter,
            price_fetcher=self._get_price
        )
        
        self.calendar = EventCalendar()
        
        # Database
        self.database = get_database(
            backend_type=self.config.get("database_type", "sqlite"),
            db_path=self.config.get("db_path", "data/trading.db")
        )
        self.database.connect()
        
        # Start monitoring
        self.realtime_risk.start()
        self.complex_orders.start_monitoring()
        
        # Register callbacks
        self.oms.on_fill(self._on_fill)
        self.realtime_risk.on_alert(self._on_risk_alert)
        
        self._initialized = True
        logger.info("Institutional Trading System initialized successfully")
    
    def shutdown(self):
        """Shutdown all components."""
        if not self._initialized:
            return
        
        logger.info("Shutting down Institutional Trading System...")
        
        self.realtime_risk.stop()
        self.complex_orders.stop_monitoring()
        
        if self.database:
            self.database.disconnect()
        
        self._initialized = False
    
    def _get_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        if self.broker_adapter:
            return self.broker_adapter.get_ltp(symbol) or 0.0
        return 0.0
    
    def _on_fill(self, order: Order, fill):
        """Handle order fill."""
        logger.info(f"Fill: {order.order_id} {fill.qty}@{fill.price}")
        
        # Update portfolio risk
        side = "LONG" if order.side == OrderSide.BUY else "SHORT"
        self.portfolio_risk.update_position(
            symbol=order.symbol,
            qty=order.qty,
            entry_price=fill.price,
            current_price=fill.price,
            side=side
        )
    
    def _on_risk_alert(self, alert):
        """Handle risk alert."""
        logger.warning(f"Risk Alert [{alert.severity}]: {alert.message}")
    
    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float
    ) -> Tuple[bool, str]:
        """
        Comprehensive pre-trade check.
        
        Checks:
        1. Event calendar (earnings, corporate actions)
        2. Portfolio risk limits
        3. Correlation with existing positions
        4. Market impact estimation
        5. Pre-trade cost analysis
        
        Returns:
            (can_trade, reason)
        """
        checks = []
        
        # 1. Event calendar check
        avoid, event_reason = self.calendar.should_avoid_trading(symbol)
        if avoid:
            return False, f"Event risk: {event_reason}"
        checks.append("✓ No blocking events")
        
        # 2. Market holiday check
        if not self.calendar.is_trading_day(date.today()):
            return False, "Market holiday"
        checks.append("✓ Trading day")
        
        # 3. Portfolio risk check
        position_side = "LONG" if side == "BUY" else "SHORT"
        can_add, risk_reason = self.portfolio_risk.can_add_position(
            symbol, qty, price, position_side
        )
        if not can_add:
            return False, f"Risk limit: {risk_reason}"
        checks.append("✓ Within risk limits")
        
        # 4. Check real-time risk status
        risk_status = self.realtime_risk.get_snapshot()
        if risk_status.get("active_alerts", 0) > 0:
            unack_alerts = self.realtime_risk.get_alerts(unacknowledged_only=True)
            critical = [a for a in unack_alerts if a.severity == "CRITICAL"]
            if critical:
                return False, f"Critical risk alert: {critical[0].message}"
        checks.append("✓ No critical alerts")
        
        # 5. Pre-trade cost analysis
        estimate = self.tca.pre_trade_analysis(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price
        )
        
        # Warn if costs are high
        if estimate.total_cost_bps > 50:  # 0.5% cost
            checks.append(f"⚠ High trading cost: {estimate.total_cost_bps:.1f}bps")
        else:
            checks.append(f"✓ Trading cost: {estimate.total_cost_bps:.1f}bps")
        
        logger.info(f"Pre-trade check for {symbol}: PASSED\n" + "\n".join(checks))
        return True, "All checks passed"
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float = None,
        order_type: str = "MARKET",
        stop_loss: float = None,
        take_profit: float = None,
        strategy_name: str = "",
        use_algo: str = None
    ) -> Optional[Order]:
        """
        Submit an order through the OMS.
        
        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            qty: Quantity
            price: Limit price (None for market)
            order_type: "MARKET", "LIMIT", "STOP"
            stop_loss: Stop loss price (creates bracket if set)
            take_profit: Take profit price (creates bracket if set)
            strategy_name: Name of strategy
            use_algo: Execution algorithm ("TWAP", "VWAP", "IS", None)
        
        Returns:
            Order object or None if failed
        """
        # Pre-trade check
        current_price = price or self._get_price(symbol)
        can_trade, reason = self.pre_trade_check(symbol, side, qty, current_price)
        
        if not can_trade:
            logger.warning(f"Order rejected: {reason}")
            return None
        
        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        
        # Determine order type
        if order_type == "MARKET":
            ot = OrderType.MARKET
        elif order_type == "LIMIT":
            ot = OrderType.LIMIT
        elif order_type == "STOP":
            ot = OrderType.STOP
        else:
            ot = OrderType.MARKET
        
        # If bracket order (SL and TP both set)
        if stop_loss and take_profit:
            bracket = self.complex_orders.create_bracket(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=price,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit
            )
            self.complex_orders.submit_bracket(bracket.bracket_id)
            logger.info(f"Bracket order created: {bracket.bracket_id}")
            return bracket.entry_leg
        
        # If using execution algorithm
        if use_algo:
            algo_type = {
                "TWAP": ExecutionAlgoType.TWAP,
                "VWAP": ExecutionAlgoType.VWAP,
                "IS": ExecutionAlgoType.IMPLEMENTATION_SHORTFALL,
                "ICEBERG": ExecutionAlgoType.ICEBERG
            }.get(use_algo.upper())
            
            if algo_type:
                algo = self.execution_manager.create_algo(
                    algo_type=algo_type,
                    symbol=symbol,
                    side=side,
                    qty=qty
                )
                self.execution_manager.start_algo(algo.algo_id)
                logger.info(f"Execution algo started: {algo.algo_id}")
                return None  # Algo handles order creation
        
        # Standard order
        order = self.oms.create_order(
            symbol=symbol,
            side=order_side,
            qty=qty,
            order_type=ot,
            limit_price=price,
            strategy_name=strategy_name
        )
        
        if order:
            self.oms.submit_order(order.order_id)
        
        return order
    
    def create_trailing_stop(
        self,
        symbol: str,
        qty: int,
        side: str = "SELL",  # SELL for long position
        trail_percent: float = None,
        trail_amount: float = None,
        activation_price: float = None
    ) -> TrailingStopOrder:
        """Create a trailing stop order."""
        return self.complex_orders.create_trailing_stop(
            symbol=symbol,
            side=side,
            qty=qty,
            trail_percent=trail_percent,
            trail_amount=trail_amount,
            activation_price=activation_price
        )
    
    def get_risk_status(self) -> Dict:
        """Get comprehensive risk status."""
        return {
            "portfolio": self.portfolio_risk.get_status(),
            "realtime": self.realtime_risk.get_snapshot(),
            "oms": self.oms.get_status(),
            "complex_orders": self.complex_orders.get_status()
        }
    
    def run_stress_tests(self) -> Dict:
        """Run all predefined stress tests."""
        return self.realtime_risk.run_all_stress_tests()
    
    def get_recommendations(self) -> List[Dict]:
        """Get position adjustment recommendations."""
        return self.realtime_risk.get_position_recommendations()
    
    def get_mtf_signal(self, symbol: str) -> Dict:
        """Get multi-timeframe signal for a symbol."""
        if symbol not in self._mtf_analyzers:
            self._mtf_analyzers[symbol] = MultiTimeframeAnalyzer(
                primary_tf=Timeframe.M5,
                context_tfs=[Timeframe.M15, Timeframe.H1]
            )
        
        signal = self._mtf_analyzers[symbol].get_confluence_signal(symbol)
        
        return {
            "symbol": symbol,
            "alignment_score": signal.alignment_score,
            "entry_bias": signal.entry_bias,
            "signal_strength": signal.signal_strength,
            "primary_trend": signal.primary_tf_trend.value,
            "higher_tf_trend": signal.higher_tf_trend.value,
            "htf_support": signal.htf_support,
            "htf_resistance": signal.htf_resistance
        }
    
    def analyze_performance(
        self,
        returns: List[float],
        run_monte_carlo: bool = True
    ) -> Dict:
        """
        Comprehensive performance analysis.
        
        Args:
            returns: List of daily returns (as decimals, e.g., 0.01 for 1%)
            run_monte_carlo: Whether to run Monte Carlo simulation
        
        Returns:
            Dictionary with metrics, Monte Carlo results (if run), etc.
        """
        metrics = AdvancedMetrics(returns)
        full_metrics = metrics.full_metrics()
        
        result = {
            "metrics": {
                "total_return": full_metrics.total_return,
                "annualized_return": full_metrics.annualized_return,
                "sharpe_ratio": full_metrics.sharpe_ratio,
                "sortino_ratio": full_metrics.sortino_ratio,
                "calmar_ratio": full_metrics.calmar_ratio,
                "max_drawdown": full_metrics.max_drawdown,
                "max_drawdown_duration": full_metrics.max_drawdown_duration_days,
                "win_rate": full_metrics.win_rate,
                "profit_factor": full_metrics.profit_factor,
                "var_95": full_metrics.var_95,
                "cvar_95": full_metrics.cvar_95
            }
        }
        
        if run_monte_carlo and len(returns) >= 20:
            simulator = MonteCarloSimulator(returns, self.capital)
            mc_result = simulator.run_simulation(
                num_simulations=5000,
                time_horizon_days=252
            )
            
            result["monte_carlo"] = {
                "expected_return": mc_result.expected_return,
                "return_5th_percentile": mc_result.return_5th_percentile,
                "return_95th_percentile": mc_result.return_95th_percentile,
                "expected_max_drawdown": mc_result.expected_max_drawdown,
                "prob_positive_return": mc_result.prob_positive_return,
                "prob_loss_gt_20pct": mc_result.prob_loss_greater_than_20pct
            }
        
        return result
    
    def optimize_parameters(
        self,
        parameters: List[Parameter],
        backtest_runner,
        n_trials: int = 100
    ) -> Dict:
        """
        Run parameter optimization.
        
        Args:
            parameters: List of Parameter objects defining search space
            backtest_runner: Function that takes params dict, returns metric
            n_trials: Number of optimization trials
        
        Returns:
            Best parameters and optimization history
        """
        objective = BacktestObjective(
            backtest_runner=backtest_runner,
            metric="sharpe_ratio",
            direction="maximize"
        )
        
        optimizer = ParameterOptimizer(parameters, objective)
        result = optimizer.optimize_bayesian(n_trials=n_trials)
        
        return {
            "best_params": result.best_params,
            "best_value": result.best_value,
            "n_trials": result.n_trials,
            "optimization_time": result.optimization_time_seconds
        }


# Factory function for quick setup
def create_trading_system(
    capital: float,
    config: Dict = None,
    sector_map: Dict = None
) -> InstitutionalTradingSystem:
    """
    Create and initialize trading system.
    
    Args:
        capital: Starting capital
        config: Optional configuration dictionary
        sector_map: Symbol to sector mapping
    
    Returns:
        Initialized InstitutionalTradingSystem
    """
    system = InstitutionalTradingSystem(
        capital=capital,
        config=config,
        sector_map=sector_map
    )
    system.initialize()
    return system
