"""
Portfolio Risk Engine - Institutional-grade portfolio risk management.

Features:
- Value at Risk (VaR) - Parametric, Historical, Monte Carlo
- Expected Shortfall (CVaR)
- Correlation matrix tracking
- Factor exposure monitoring
- Sector/concentration limits
- Gross/Net exposure management
- Greeks for options (if applicable)
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Portfolio position with risk attributes."""
    symbol: str
    qty: int
    entry_price: float
    current_price: float
    side: str  # "LONG" or "SHORT"
    sector: str = ""
    beta: float = 1.0
    
    @property
    def market_value(self) -> float:
        return abs(self.qty) * self.current_price
    
    @property
    def notional_value(self) -> float:
        if self.side == "LONG":
            return self.qty * self.current_price
        else:
            return -self.qty * self.current_price
    
    @property
    def pnl(self) -> float:
        if self.side == "LONG":
            return self.qty * (self.current_price - self.entry_price)
        else:
            return self.qty * (self.entry_price - self.current_price)
    
    @property
    def pnl_percent(self) -> float:
        entry_value = abs(self.qty) * self.entry_price
        return (self.pnl / entry_value * 100) if entry_value > 0 else 0


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for portfolio."""
    # Value at Risk
    var_95: float        # 95% VaR (1-day)
    var_99: float        # 99% VaR (1-day)
    cvar_95: float       # Expected Shortfall at 95%
    cvar_99: float       # Expected Shortfall at 99%
    
    # Exposure
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    
    # Concentration
    largest_position_pct: float
    top_5_concentration: float
    herfindahl_index: float  # Concentration measure
    
    # Sector exposure
    sector_exposures: Dict[str, float]
    max_sector_exposure: float
    
    # Beta
    portfolio_beta: float
    beta_adjusted_exposure: float
    
    # Correlation
    avg_correlation: float
    max_pairwise_correlation: float
    
    # Drawdown
    current_drawdown: float
    max_drawdown: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_var_95_pct: float = 2.0       # Max 2% daily VaR
    max_gross_exposure_pct: float = 200.0  # Max 200% gross
    max_net_exposure_pct: float = 100.0    # Max 100% net
    max_single_position_pct: float = 10.0  # Max 10% in single position
    max_sector_exposure_pct: float = 30.0  # Max 30% per sector
    max_correlation: float = 0.7           # Max pairwise correlation
    max_drawdown_pct: float = 10.0         # Max 10% drawdown
    min_positions_for_var: int = 2         # Min positions for VaR calc


class CorrelationTracker:
    """
    Tracks rolling correlations between positions.
    Uses exponentially weighted correlation for recency bias.
    """
    
    def __init__(self, lookback_days: int = 60, halflife_days: int = 20):
        self.lookback_days = lookback_days
        self.halflife_days = halflife_days
        
        # Store returns history: symbol -> deque of daily returns
        self._returns_history: Dict[str, deque] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        self._symbols: List[str] = []
        self._last_update: Optional[datetime] = None
    
    def add_return(self, symbol: str, daily_return: float):
        """Add a daily return observation."""
        if symbol not in self._returns_history:
            self._returns_history[symbol] = deque(maxlen=self.lookback_days)
        self._returns_history[symbol].append(daily_return)
    
    def update_from_prices(self, prices: Dict[str, List[float]]):
        """
        Update returns from price history.
        prices: {symbol: [list of daily prices]}
        """
        for symbol, price_list in prices.items():
            if len(price_list) < 2:
                continue
            
            if symbol not in self._returns_history:
                self._returns_history[symbol] = deque(maxlen=self.lookback_days)
            
            # Calculate returns
            for i in range(1, len(price_list)):
                ret = (price_list[i] - price_list[i-1]) / price_list[i-1]
                self._returns_history[symbol].append(ret)
    
    def compute_correlation_matrix(self) -> Optional[np.ndarray]:
        """
        Compute exponentially weighted correlation matrix.
        Returns None if insufficient data.
        """
        # Filter symbols with enough data
        min_obs = max(10, self.lookback_days // 3)
        valid_symbols = [s for s, r in self._returns_history.items() if len(r) >= min_obs]
        
        if len(valid_symbols) < 2:
            return None
        
        self._symbols = valid_symbols
        n = len(valid_symbols)
        
        # Create aligned returns matrix
        min_len = min(len(self._returns_history[s]) for s in valid_symbols)
        returns_matrix = np.zeros((min_len, n))
        
        for i, symbol in enumerate(valid_symbols):
            returns = list(self._returns_history[symbol])[-min_len:]
            returns_matrix[:, i] = returns
        
        # Calculate exponential weights
        weights = np.array([2 ** (-i / self.halflife_days) for i in range(min_len - 1, -1, -1)])
        weights = weights / weights.sum()
        
        # Weighted correlation
        weighted_returns = returns_matrix * np.sqrt(weights[:, np.newaxis])
        
        # Center the data
        weighted_means = np.average(returns_matrix, axis=0, weights=weights)
        centered = weighted_returns - weighted_means * np.sqrt(weights[:, np.newaxis])
        
        # Correlation = cov / (std * std)
        cov_matrix = centered.T @ centered
        std = np.sqrt(np.diag(cov_matrix))
        std[std == 0] = 1  # Avoid division by zero
        
        corr_matrix = cov_matrix / np.outer(std, std)
        np.fill_diagonal(corr_matrix, 1.0)
        
        self._correlation_matrix = corr_matrix
        self._last_update = datetime.now()
        
        return corr_matrix
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get correlation between two symbols."""
        if self._correlation_matrix is None:
            self.compute_correlation_matrix()
        
        if self._correlation_matrix is None:
            return None
        
        try:
            i1 = self._symbols.index(symbol1)
            i2 = self._symbols.index(symbol2)
            return float(self._correlation_matrix[i1, i2])
        except (ValueError, IndexError):
            return None
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Get pairs with correlation above threshold."""
        if self._correlation_matrix is None:
            self.compute_correlation_matrix()
        
        if self._correlation_matrix is None:
            return []
        
        pairs = []
        n = len(self._symbols)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self._correlation_matrix[i, j]
                if abs(corr) >= threshold:
                    pairs.append((self._symbols[i], self._symbols[j], corr))
        
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def avg_correlation(self) -> float:
        """Calculate average pairwise correlation."""
        if self._correlation_matrix is None:
            self.compute_correlation_matrix()
        
        if self._correlation_matrix is None or len(self._symbols) < 2:
            return 0.0
        
        # Get upper triangle (excluding diagonal)
        n = len(self._symbols)
        upper_tri = self._correlation_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(np.abs(upper_tri)))


class VaRCalculator:
    """
    Value at Risk calculator supporting multiple methods:
    - Parametric (variance-covariance)
    - Historical simulation
    - Monte Carlo simulation
    """
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
    
    def parametric_var(
        self,
        portfolio_value: float,
        returns: List[float],
        confidence: float = 0.95
    ) -> float:
        """
        Parametric VaR assuming normal distribution.
        VaR = -μ + σ * z_α
        """
        if len(returns) < 5:
            return 0.0
        
        returns_array = np.array(returns)
        mean = np.mean(returns_array)
        std = np.std(returns_array)
        
        z_score = stats.norm.ppf(1 - confidence)
        var_pct = -(mean + z_score * std)
        
        return portfolio_value * max(0, var_pct)
    
    def historical_var(
        self,
        portfolio_value: float,
        returns: List[float],
        confidence: float = 0.95
    ) -> float:
        """
        Historical VaR - uses actual return distribution.
        More accurate for non-normal returns.
        """
        if len(returns) < 10:
            return 0.0
        
        returns_array = np.array(returns)
        percentile = (1 - confidence) * 100
        var_pct = -np.percentile(returns_array, percentile)
        
        return portfolio_value * max(0, var_pct)
    
    def monte_carlo_var(
        self,
        portfolio_value: float,
        mean_return: float,
        volatility: float,
        confidence: float = 0.95,
        simulations: int = 10000
    ) -> float:
        """
        Monte Carlo VaR - simulates future returns.
        """
        # Simulate returns
        simulated_returns = np.random.normal(mean_return, volatility, simulations)
        
        # Calculate portfolio values
        simulated_values = portfolio_value * (1 + simulated_returns)
        
        # VaR is the percentile loss
        var_value = portfolio_value - np.percentile(simulated_values, (1 - confidence) * 100)
        
        return max(0, var_value)
    
    def expected_shortfall(
        self,
        portfolio_value: float,
        returns: List[float],
        confidence: float = 0.95
    ) -> float:
        """
        Expected Shortfall (CVaR) - average loss beyond VaR.
        More coherent risk measure than VaR.
        """
        if len(returns) < 10:
            return 0.0
        
        returns_array = np.array(returns)
        var_pct = -np.percentile(returns_array, (1 - confidence) * 100)
        
        # ES is the average of returns worse than VaR
        tail_returns = returns_array[returns_array < -var_pct]
        
        if len(tail_returns) == 0:
            return self.historical_var(portfolio_value, returns, confidence)
        
        es_pct = -np.mean(tail_returns)
        return portfolio_value * max(0, es_pct)


class PortfolioRiskEngine:
    """
    Central portfolio risk management engine.
    
    Provides:
    - Real-time risk metrics calculation
    - Limit monitoring and alerts
    - Position-level and portfolio-level risk
    - Integration with correlation tracking
    """
    
    def __init__(
        self,
        capital: float,
        limits: RiskLimits = None,
        sector_map: Dict[str, str] = None,
        beta_map: Dict[str, float] = None
    ):
        self.capital = capital
        self.limits = limits or RiskLimits()
        self.sector_map = sector_map or {}
        self.beta_map = beta_map or {}
        
        self.positions: Dict[str, Position] = {}
        self.correlation_tracker = CorrelationTracker()
        self.var_calculator = VaRCalculator()
        
        # Historical tracking
        self._portfolio_values: deque = deque(maxlen=252)  # ~1 year of daily values
        self._daily_returns: deque = deque(maxlen=252)
        self._peak_value = capital
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Alert callbacks
        self._alert_callbacks: List = []
    
    def update_position(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        current_price: float,
        side: str
    ):
        """Update or add a position."""
        with self._lock:
            sector = self.sector_map.get(symbol, "Other")
            beta = self.beta_map.get(symbol, 1.0)
            
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                entry_price=entry_price,
                current_price=current_price,
                side=side,
                sector=sector,
                beta=beta
            )
    
    def remove_position(self, symbol: str):
        """Remove a closed position."""
        with self._lock:
            self.positions.pop(symbol, None)
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].current_price = price
    
    def record_daily_close(self, portfolio_value: float):
        """Record end-of-day portfolio value for risk calculations."""
        with self._lock:
            self._portfolio_values.append(portfolio_value)
            
            if len(self._portfolio_values) >= 2:
                prev_value = self._portfolio_values[-2]
                daily_return = (portfolio_value - prev_value) / prev_value
                self._daily_returns.append(daily_return)
            
            # Update drawdown tracking
            if portfolio_value > self._peak_value:
                self._peak_value = portfolio_value
            
            self._current_drawdown = (self._peak_value - portfolio_value) / self._peak_value
            self._max_drawdown = max(self._max_drawdown, self._current_drawdown)
    
    def calculate_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        with self._lock:
            positions = list(self.positions.values())
        
        if not positions:
            return self._empty_metrics()
        
        # Calculate exposures
        long_exposure = sum(p.market_value for p in positions if p.side == "LONG")
        short_exposure = sum(p.market_value for p in positions if p.side == "SHORT")
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        # Concentration metrics
        market_values = [p.market_value for p in positions]
        total_value = sum(market_values)
        
        if total_value > 0:
            position_pcts = [mv / total_value for mv in market_values]
            largest_position_pct = max(position_pcts) * 100
            top_5_pcts = sorted(position_pcts, reverse=True)[:5]
            top_5_concentration = sum(top_5_pcts) * 100
            herfindahl = sum(pct ** 2 for pct in position_pcts)
        else:
            largest_position_pct = 0
            top_5_concentration = 0
            herfindahl = 0
        
        # Sector exposures
        sector_exposures = {}
        for pos in positions:
            sector = pos.sector or "Other"
            if sector not in sector_exposures:
                sector_exposures[sector] = 0
            sector_exposures[sector] += pos.market_value
        
        if total_value > 0:
            sector_exposures = {s: (v / total_value * 100) for s, v in sector_exposures.items()}
        
        max_sector = max(sector_exposures.values()) if sector_exposures else 0
        
        # Beta calculations
        weighted_beta = sum(p.beta * p.market_value for p in positions)
        portfolio_beta = weighted_beta / total_value if total_value > 0 else 1.0
        beta_adjusted = net_exposure * portfolio_beta
        
        # Correlation metrics
        self.correlation_tracker.compute_correlation_matrix()
        avg_corr = self.correlation_tracker.avg_correlation()
        high_corr_pairs = self.correlation_tracker.get_highly_correlated_pairs(0.5)
        max_corr = high_corr_pairs[0][2] if high_corr_pairs else 0
        
        # VaR calculations
        returns = list(self._daily_returns)
        portfolio_value = self.capital + sum(p.pnl for p in positions)
        
        var_95 = self.var_calculator.historical_var(portfolio_value, returns, 0.95)
        var_99 = self.var_calculator.historical_var(portfolio_value, returns, 0.99)
        cvar_95 = self.var_calculator.expected_shortfall(portfolio_value, returns, 0.95)
        cvar_99 = self.var_calculator.expected_shortfall(portfolio_value, returns, 0.99)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            largest_position_pct=largest_position_pct,
            top_5_concentration=top_5_concentration,
            herfindahl_index=herfindahl,
            sector_exposures=sector_exposures,
            max_sector_exposure=max_sector,
            portfolio_beta=portfolio_beta,
            beta_adjusted_exposure=beta_adjusted,
            avg_correlation=avg_corr,
            max_pairwise_correlation=max_corr,
            current_drawdown=self._current_drawdown * 100,
            max_drawdown=self._max_drawdown * 100
        )
    
    def _empty_metrics(self) -> RiskMetrics:
        """Return empty metrics when no positions."""
        return RiskMetrics(
            var_95=0, var_99=0, cvar_95=0, cvar_99=0,
            gross_exposure=0, net_exposure=0, long_exposure=0, short_exposure=0,
            largest_position_pct=0, top_5_concentration=0, herfindahl_index=0,
            sector_exposures={}, max_sector_exposure=0,
            portfolio_beta=1.0, beta_adjusted_exposure=0,
            avg_correlation=0, max_pairwise_correlation=0,
            current_drawdown=self._current_drawdown * 100,
            max_drawdown=self._max_drawdown * 100
        )
    
    def check_limits(self) -> List[Tuple[str, str, float, float]]:
        """
        Check all risk limits.
        Returns list of (limit_name, status, current_value, limit_value) for breaches.
        """
        metrics = self.calculate_metrics()
        breaches = []
        
        # VaR limit
        var_pct = (metrics.var_95 / self.capital * 100) if self.capital > 0 else 0
        if var_pct > self.limits.max_var_95_pct:
            breaches.append(("VaR95", "BREACH", var_pct, self.limits.max_var_95_pct))
        
        # Gross exposure
        gross_pct = (metrics.gross_exposure / self.capital * 100) if self.capital > 0 else 0
        if gross_pct > self.limits.max_gross_exposure_pct:
            breaches.append(("GrossExposure", "BREACH", gross_pct, self.limits.max_gross_exposure_pct))
        
        # Net exposure
        net_pct = abs(metrics.net_exposure / self.capital * 100) if self.capital > 0 else 0
        if net_pct > self.limits.max_net_exposure_pct:
            breaches.append(("NetExposure", "BREACH", net_pct, self.limits.max_net_exposure_pct))
        
        # Single position concentration
        if metrics.largest_position_pct > self.limits.max_single_position_pct:
            breaches.append(("SinglePosition", "BREACH", 
                           metrics.largest_position_pct, self.limits.max_single_position_pct))
        
        # Sector exposure
        if metrics.max_sector_exposure > self.limits.max_sector_exposure_pct:
            breaches.append(("SectorExposure", "BREACH",
                           metrics.max_sector_exposure, self.limits.max_sector_exposure_pct))
        
        # Correlation
        if metrics.max_pairwise_correlation > self.limits.max_correlation:
            breaches.append(("Correlation", "WARNING",
                           metrics.max_pairwise_correlation, self.limits.max_correlation))
        
        # Drawdown
        if metrics.current_drawdown > self.limits.max_drawdown_pct:
            breaches.append(("Drawdown", "BREACH",
                           metrics.current_drawdown, self.limits.max_drawdown_pct))
        
        # Trigger alerts for breaches
        for breach in breaches:
            self._trigger_alert(breach)
        
        return breaches
    
    def can_add_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        side: str
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be added within limits.
        Returns (allowed, reason).
        """
        position_value = qty * price
        
        with self._lock:
            current_gross = sum(p.market_value for p in self.positions.values())
            current_net = sum(p.notional_value for p in self.positions.values())
            total_value = current_gross + position_value
        
        # Check gross exposure
        new_gross_pct = (current_gross + position_value) / self.capital * 100
        if new_gross_pct > self.limits.max_gross_exposure_pct:
            return False, f"Would exceed gross exposure limit ({new_gross_pct:.1f}% > {self.limits.max_gross_exposure_pct}%)"
        
        # Check single position size
        position_pct = position_value / self.capital * 100
        if position_pct > self.limits.max_single_position_pct:
            return False, f"Position too large ({position_pct:.1f}% > {self.limits.max_single_position_pct}%)"
        
        # Check sector exposure
        sector = self.sector_map.get(symbol, "Other")
        with self._lock:
            sector_value = sum(p.market_value for p in self.positions.values() if p.sector == sector)
        new_sector_pct = (sector_value + position_value) / self.capital * 100
        if new_sector_pct > self.limits.max_sector_exposure_pct:
            return False, f"Would exceed sector limit for {sector} ({new_sector_pct:.1f}%)"
        
        # Check correlation with existing positions
        high_corr_symbols = []
        for existing in self.positions.values():
            corr = self.correlation_tracker.get_correlation(symbol, existing.symbol)
            if corr is not None and abs(corr) > self.limits.max_correlation:
                high_corr_symbols.append((existing.symbol, corr))
        
        if high_corr_symbols:
            return False, f"High correlation with existing positions: {high_corr_symbols}"
        
        return True, "OK"
    
    def position_size_for_risk(
        self,
        symbol: str,
        price: float,
        stop_loss_pct: float,
        max_risk_pct: float = 1.0
    ) -> int:
        """
        Calculate position size to risk a specific percentage of capital.
        Incorporates portfolio-level constraints.
        """
        # Risk-based sizing
        max_risk_amount = self.capital * (max_risk_pct / 100)
        risk_per_share = price * (stop_loss_pct / 100)
        
        if risk_per_share <= 0:
            return 0
        
        qty_from_risk = int(max_risk_amount / risk_per_share)
        
        # Limit by single position constraint
        max_position_value = self.capital * (self.limits.max_single_position_pct / 100)
        qty_from_position = int(max_position_value / price)
        
        # Limit by sector constraint
        sector = self.sector_map.get(symbol, "Other")
        with self._lock:
            sector_value = sum(p.market_value for p in self.positions.values() if p.sector == sector)
        max_sector_value = self.capital * (self.limits.max_sector_exposure_pct / 100)
        remaining_sector = max_sector_value - sector_value
        qty_from_sector = int(remaining_sector / price) if remaining_sector > 0 else 0
        
        # Take minimum
        return max(0, min(qty_from_risk, qty_from_position, qty_from_sector))
    
    def _trigger_alert(self, breach: Tuple[str, str, float, float]):
        """Trigger alert callbacks for limit breaches."""
        for callback in self._alert_callbacks:
            try:
                callback(breach)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def on_limit_breach(self, callback):
        """Register callback for limit breaches."""
        self._alert_callbacks.append(callback)
    
    def get_status(self) -> dict:
        """Get comprehensive risk status."""
        metrics = self.calculate_metrics()
        breaches = self.check_limits()
        
        return {
            "capital": self.capital,
            "positions": len(self.positions),
            "gross_exposure": metrics.gross_exposure,
            "net_exposure": metrics.net_exposure,
            "var_95": metrics.var_95,
            "var_95_pct": metrics.var_95 / self.capital * 100 if self.capital > 0 else 0,
            "current_drawdown": metrics.current_drawdown,
            "max_drawdown": metrics.max_drawdown,
            "portfolio_beta": metrics.portfolio_beta,
            "avg_correlation": metrics.avg_correlation,
            "limit_breaches": len(breaches),
            "breaches": breaches,
            "status": "OK" if not breaches else "LIMIT_BREACH"
        }
