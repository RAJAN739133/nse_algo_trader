"""
Advanced Analytics - Monte Carlo simulation, additional risk metrics.

Features:
- Monte Carlo simulation for drawdown and return projections
- Sortino ratio, Calmar ratio
- Maximum drawdown duration
- Rolling performance metrics
- Regime detection
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    simulations: int
    time_horizon_days: int
    
    # Return distribution
    expected_return: float
    median_return: float
    return_5th_percentile: float
    return_95th_percentile: float
    return_std: float
    
    # Drawdown distribution
    expected_max_drawdown: float
    median_max_drawdown: float
    max_drawdown_95th: float  # 95th percentile (worst case)
    
    # Probability metrics
    prob_positive_return: float
    prob_loss_greater_than_10pct: float
    prob_loss_greater_than_20pct: float
    
    # Sharpe distribution
    expected_sharpe: float
    sharpe_5th_percentile: float
    sharpe_95th_percentile: float
    
    # Raw simulation paths (for plotting)
    return_paths: Optional[np.ndarray] = None


@dataclass
class RiskMetricsSuite:
    """Comprehensive risk metrics."""
    # Return-based
    total_return: float
    annualized_return: float
    cagr: float
    
    # Volatility
    daily_volatility: float
    annualized_volatility: float
    downside_volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    avg_drawdown_duration_days: int
    current_drawdown: float
    
    # Tail risk
    var_95: float
    var_99: float
    cvar_95: float
    skewness: float
    kurtosis: float
    
    # Win/loss
    win_rate: float
    profit_factor: float
    avg_win_to_loss: float
    expectancy: float


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio analysis.
    Uses bootstrap resampling of historical returns.
    """
    
    def __init__(
        self,
        historical_returns: List[float],
        initial_capital: float = 100000,
        seed: int = 42
    ):
        self.returns = np.array(historical_returns)
        self.initial_capital = initial_capital
        self.rng = np.random.default_rng(seed)
        
        # Calculate return statistics
        self.mean_return = np.mean(self.returns)
        self.std_return = np.std(self.returns)
    
    def simulate_paths(
        self,
        num_simulations: int = 10000,
        time_horizon_days: int = 252,
        method: str = "bootstrap"
    ) -> np.ndarray:
        """
        Generate Monte Carlo simulation paths.
        
        Args:
            num_simulations: Number of simulation paths
            time_horizon_days: Number of days to simulate
            method: "bootstrap" (resample historical) or "parametric" (normal dist)
        
        Returns:
            Array of shape (num_simulations, time_horizon_days) with cumulative returns
        """
        paths = np.zeros((num_simulations, time_horizon_days))
        
        if method == "bootstrap":
            # Bootstrap resampling from historical returns
            for i in range(num_simulations):
                sampled_returns = self.rng.choice(self.returns, size=time_horizon_days, replace=True)
                paths[i] = np.cumprod(1 + sampled_returns) - 1
        
        elif method == "parametric":
            # Assume normal distribution
            for i in range(num_simulations):
                daily_returns = self.rng.normal(self.mean_return, self.std_return, time_horizon_days)
                paths[i] = np.cumprod(1 + daily_returns) - 1
        
        elif method == "block_bootstrap":
            # Block bootstrap to preserve autocorrelation
            block_size = 5
            num_blocks = time_horizon_days // block_size + 1
            
            for i in range(num_simulations):
                # Sample blocks
                blocks = []
                for _ in range(num_blocks):
                    start_idx = self.rng.integers(0, len(self.returns) - block_size)
                    blocks.extend(self.returns[start_idx:start_idx + block_size])
                
                sampled_returns = np.array(blocks[:time_horizon_days])
                paths[i] = np.cumprod(1 + sampled_returns) - 1
        
        return paths
    
    def run_simulation(
        self,
        num_simulations: int = 10000,
        time_horizon_days: int = 252,
        method: str = "bootstrap",
        store_paths: bool = False
    ) -> MonteCarloResult:
        """
        Run full Monte Carlo simulation and analyze results.
        
        Args:
            num_simulations: Number of simulation paths
            time_horizon_days: Number of days to simulate
            method: Simulation method
            store_paths: Whether to store all paths in result
        
        Returns:
            MonteCarloResult with statistics
        """
        paths = self.simulate_paths(num_simulations, time_horizon_days, method)
        
        # Final returns
        final_returns = paths[:, -1]
        
        # Calculate drawdowns for each path
        max_drawdowns = []
        for path in paths:
            equity_curve = self.initial_capital * (1 + path)
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (running_max - equity_curve) / running_max
            max_drawdowns.append(np.max(drawdowns))
        
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate Sharpe ratios for each path
        sharpes = []
        for path in paths:
            # Get daily returns from cumulative
            daily_rets = np.diff(np.concatenate([[0], path])) / (np.concatenate([[1], 1 + path[:-1]]))
            if np.std(daily_rets) > 0:
                sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
            else:
                sharpe = 0
            sharpes.append(sharpe)
        
        sharpes = np.array(sharpes)
        
        return MonteCarloResult(
            simulations=num_simulations,
            time_horizon_days=time_horizon_days,
            expected_return=np.mean(final_returns) * 100,
            median_return=np.median(final_returns) * 100,
            return_5th_percentile=np.percentile(final_returns, 5) * 100,
            return_95th_percentile=np.percentile(final_returns, 95) * 100,
            return_std=np.std(final_returns) * 100,
            expected_max_drawdown=np.mean(max_drawdowns) * 100,
            median_max_drawdown=np.median(max_drawdowns) * 100,
            max_drawdown_95th=np.percentile(max_drawdowns, 95) * 100,
            prob_positive_return=np.mean(final_returns > 0) * 100,
            prob_loss_greater_than_10pct=np.mean(final_returns < -0.10) * 100,
            prob_loss_greater_than_20pct=np.mean(final_returns < -0.20) * 100,
            expected_sharpe=np.mean(sharpes),
            sharpe_5th_percentile=np.percentile(sharpes, 5),
            sharpe_95th_percentile=np.percentile(sharpes, 95),
            return_paths=paths if store_paths else None
        )
    
    def confidence_intervals(
        self,
        time_horizons: List[int] = None,
        confidence_levels: List[float] = None,
        num_simulations: int = 10000
    ) -> Dict[int, Dict[float, Tuple[float, float]]]:
        """
        Calculate confidence intervals for returns at different time horizons.
        
        Returns:
            Dict mapping time_horizon -> {confidence_level -> (lower, upper)}
        """
        if time_horizons is None:
            time_horizons = [21, 63, 126, 252]  # 1, 3, 6, 12 months
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        max_horizon = max(time_horizons)
        paths = self.simulate_paths(num_simulations, max_horizon)
        
        results = {}
        for horizon in time_horizons:
            horizon_returns = paths[:, horizon - 1]
            results[horizon] = {}
            
            for conf in confidence_levels:
                alpha = (1 - conf) / 2
                lower = np.percentile(horizon_returns, alpha * 100) * 100
                upper = np.percentile(horizon_returns, (1 - alpha) * 100) * 100
                results[horizon][conf] = (lower, upper)
        
        return results


class AdvancedMetrics:
    """
    Calculate advanced risk and performance metrics.
    """
    
    def __init__(
        self,
        returns: List[float],
        benchmark_returns: List[float] = None,
        risk_free_rate: float = 0.05,  # 5% annual
        trading_days_per_year: int = 252
    ):
        self.returns = np.array(returns)
        self.benchmark = np.array(benchmark_returns) if benchmark_returns else None
        self.rf_rate = risk_free_rate / trading_days_per_year
        self.trading_days = trading_days_per_year
        
        # Precompute basics
        self.n_obs = len(self.returns)
        self.total_return = np.prod(1 + self.returns) - 1
        self.mean_return = np.mean(self.returns)
        self.std_return = np.std(self.returns)
    
    def sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = self.returns - self.rf_rate
        if np.std(excess_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days)
    
    def sortino_ratio(self, mar: float = 0) -> float:
        """
        Calculate Sortino ratio.
        Uses downside deviation instead of total volatility.
        
        Args:
            mar: Minimum acceptable return (default 0)
        """
        excess_returns = self.returns - mar
        downside_returns = self.returns[self.returns < mar]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return float('inf')
        
        return np.mean(excess_returns) / downside_std * np.sqrt(self.trading_days)
    
    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.
        Annualized return / Maximum drawdown.
        """
        ann_return = (1 + self.total_return) ** (self.trading_days / self.n_obs) - 1
        max_dd = self.max_drawdown()
        
        if max_dd == 0:
            return float('inf')
        
        return ann_return / max_dd
    
    def omega_ratio(self, threshold: float = 0) -> float:
        """
        Calculate Omega ratio.
        Sum of returns above threshold / Sum of returns below threshold.
        """
        above = self.returns[self.returns > threshold] - threshold
        below = threshold - self.returns[self.returns <= threshold]
        
        if np.sum(below) == 0:
            return float('inf')
        
        return np.sum(above) / np.sum(below)
    
    def information_ratio(self) -> float:
        """
        Calculate Information ratio.
        Active return / Tracking error.
        """
        if self.benchmark is None:
            return 0
        
        active_returns = self.returns - self.benchmark
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return 0
        
        return np.mean(active_returns) / tracking_error * np.sqrt(self.trading_days)
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return np.max(drawdowns)
    
    def max_drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in days."""
        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        
        in_drawdown = cumulative < running_max
        
        if not any(in_drawdown):
            return 0
        
        # Find drawdown periods
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def average_drawdown(self) -> Tuple[float, float]:
        """Calculate average drawdown and average duration."""
        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        
        # Find drawdown periods
        dd_values = []
        dd_durations = []
        current_dd = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdowns):
            if dd > 0:
                current_dd = max(current_dd, dd)
                current_duration += 1
            elif current_duration > 0:
                dd_values.append(current_dd)
                dd_durations.append(current_duration)
                current_dd = 0
                current_duration = 0
        
        if current_duration > 0:
            dd_values.append(current_dd)
            dd_durations.append(current_duration)
        
        avg_dd = np.mean(dd_values) if dd_values else 0
        avg_duration = np.mean(dd_durations) if dd_durations else 0
        
        return avg_dd, avg_duration
    
    def var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return -np.percentile(self.returns, (1 - confidence) * 100)
    
    def cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = self.var(confidence)
        tail = self.returns[self.returns <= -var]
        return -np.mean(tail) if len(tail) > 0 else var
    
    def skewness(self) -> float:
        """Calculate return distribution skewness."""
        return stats.skew(self.returns)
    
    def kurtosis(self) -> float:
        """Calculate return distribution kurtosis (excess)."""
        return stats.kurtosis(self.returns)
    
    def full_metrics(self) -> RiskMetricsSuite:
        """Calculate all metrics."""
        avg_dd, avg_dd_duration = self.average_drawdown()
        
        # Win/loss metrics
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns <= 0]
        
        win_rate = len(wins) / len(self.returns) * 100 if len(self.returns) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        gross_profit = np.sum(wins)
        gross_loss = abs(np.sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win_to_loss = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        # Current drawdown
        cumulative = np.cumprod(1 + self.returns)
        peak = np.max(cumulative)
        current_dd = (peak - cumulative[-1]) / peak if len(cumulative) > 0 else 0
        
        # Annualized metrics
        ann_return = (1 + self.total_return) ** (self.trading_days / self.n_obs) - 1
        ann_vol = self.std_return * np.sqrt(self.trading_days)
        
        # Downside volatility
        negative_returns = self.returns[self.returns < 0]
        downside_vol = np.std(negative_returns) * np.sqrt(self.trading_days) if len(negative_returns) > 0 else 0
        
        # CAGR
        years = self.n_obs / self.trading_days
        cagr = (1 + self.total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return RiskMetricsSuite(
            total_return=self.total_return * 100,
            annualized_return=ann_return * 100,
            cagr=cagr * 100,
            daily_volatility=self.std_return * 100,
            annualized_volatility=ann_vol * 100,
            downside_volatility=downside_vol * 100,
            sharpe_ratio=self.sharpe_ratio(),
            sortino_ratio=self.sortino_ratio(),
            calmar_ratio=self.calmar_ratio(),
            omega_ratio=self.omega_ratio(),
            information_ratio=self.information_ratio(),
            max_drawdown=self.max_drawdown() * 100,
            max_drawdown_duration_days=self.max_drawdown_duration(),
            avg_drawdown=avg_dd * 100,
            avg_drawdown_duration_days=int(avg_duration),
            current_drawdown=current_dd * 100,
            var_95=self.var(0.95) * 100,
            var_99=self.var(0.99) * 100,
            cvar_95=self.cvar(0.95) * 100,
            skewness=self.skewness(),
            kurtosis=self.kurtosis(),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win_to_loss=avg_win_to_loss,
            expectancy=expectancy * 100
        )


class RollingMetrics:
    """
    Calculate rolling performance metrics.
    """
    
    def __init__(self, returns: List[float], window: int = 21):
        self.returns = np.array(returns)
        self.window = window
    
    def rolling_sharpe(self) -> np.ndarray:
        """Calculate rolling Sharpe ratio."""
        n = len(self.returns)
        if n < self.window:
            return np.array([])
        
        sharpes = []
        for i in range(self.window, n + 1):
            window_returns = self.returns[i - self.window:i]
            if np.std(window_returns) > 0:
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
            else:
                sharpe = 0
            sharpes.append(sharpe)
        
        return np.array(sharpes)
    
    def rolling_volatility(self) -> np.ndarray:
        """Calculate rolling annualized volatility."""
        n = len(self.returns)
        if n < self.window:
            return np.array([])
        
        vols = []
        for i in range(self.window, n + 1):
            window_returns = self.returns[i - self.window:i]
            vol = np.std(window_returns) * np.sqrt(252)
            vols.append(vol)
        
        return np.array(vols) * 100
    
    def rolling_beta(self, benchmark: List[float]) -> np.ndarray:
        """Calculate rolling beta vs benchmark."""
        benchmark = np.array(benchmark)
        n = len(self.returns)
        
        if n < self.window or len(benchmark) != n:
            return np.array([])
        
        betas = []
        for i in range(self.window, n + 1):
            port_ret = self.returns[i - self.window:i]
            bench_ret = benchmark[i - self.window:i]
            
            cov = np.cov(port_ret, bench_ret)[0, 1]
            var = np.var(bench_ret)
            
            beta = cov / var if var > 0 else 1
            betas.append(beta)
        
        return np.array(betas)
    
    def rolling_max_drawdown(self) -> np.ndarray:
        """Calculate rolling maximum drawdown."""
        n = len(self.returns)
        if n < self.window:
            return np.array([])
        
        mdd = []
        for i in range(self.window, n + 1):
            window_returns = self.returns[i - self.window:i]
            cumulative = np.cumprod(1 + window_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / running_max
            mdd.append(np.max(drawdowns))
        
        return np.array(mdd) * 100


class RegimeDetector:
    """
    Detect market regimes for conditional analysis.
    """
    
    @staticmethod
    def detect_regime(
        returns: List[float],
        volatility_threshold: float = 0.02,
        trend_window: int = 20
    ) -> List[str]:
        """
        Detect market regime for each observation.
        
        Regimes:
        - bull: Positive trend, normal volatility
        - bear: Negative trend, normal volatility
        - high_vol: High volatility regardless of trend
        - sideways: Low volatility, no clear trend
        """
        returns = np.array(returns)
        n = len(returns)
        regimes = []
        
        for i in range(n):
            # Calculate local metrics
            start_idx = max(0, i - trend_window)
            local_returns = returns[start_idx:i + 1]
            
            if len(local_returns) < 5:
                regimes.append("sideways")
                continue
            
            # Volatility
            vol = np.std(local_returns) * np.sqrt(252)
            
            # Trend
            cumret = np.prod(1 + local_returns) - 1
            daily_trend = cumret / len(local_returns)
            
            # Classify
            if vol > volatility_threshold * np.sqrt(252):
                regimes.append("high_vol")
            elif daily_trend > 0.001:
                regimes.append("bull")
            elif daily_trend < -0.001:
                regimes.append("bear")
            else:
                regimes.append("sideways")
        
        return regimes
