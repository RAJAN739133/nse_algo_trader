"""
Market Impact Model - Almgren-Chriss and extensions.

Models how orders affect market prices:
- Temporary impact (during execution)
- Permanent impact (persists after execution)
- Optimal execution trajectories
- Order sizing based on ADV
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImpactEstimate:
    """Estimated market impact for an order."""
    temporary_impact_bps: float    # Basis points
    permanent_impact_bps: float
    total_impact_bps: float
    estimated_slippage_rupees: float
    optimal_execution_time_minutes: float
    participation_rate: float
    confidence: float  # 0-1, lower if inputs are uncertain


@dataclass  
class OptimalTrajectory:
    """Optimal execution trajectory from Almgren-Chriss."""
    times: List[float]           # Time points (0 to T)
    holdings: List[int]          # Shares remaining at each time
    trading_rates: List[int]     # Shares to trade per period
    expected_cost_bps: float     # Expected implementation shortfall
    cost_variance_bps: float     # Variance of cost


class MarketImpactModel:
    """
    Almgren-Chriss market impact model with NSE-specific calibration.
    
    Reference: Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
    
    Total Cost = Risk (variance of price) + Impact (expected cost)
    
    Parameters:
    - sigma: Daily volatility
    - eta: Temporary impact coefficient (price impact per share/time)
    - gamma: Permanent impact coefficient (price shift per share)
    - lambda: Risk aversion parameter
    """
    
    # Default parameters calibrated for NSE large-caps
    DEFAULT_SIGMA = 0.02       # 2% daily volatility
    DEFAULT_ETA = 2.5e-7       # Temporary impact coefficient
    DEFAULT_GAMMA = 2.5e-8     # Permanent impact coefficient  
    DEFAULT_LAMBDA = 1e-6      # Risk aversion
    
    def __init__(
        self,
        sigma: float = None,
        eta: float = None,
        gamma: float = None,
        risk_aversion: float = None
    ):
        self.sigma = sigma or self.DEFAULT_SIGMA
        self.eta = eta or self.DEFAULT_ETA
        self.gamma = gamma or self.DEFAULT_GAMMA
        self.risk_aversion = risk_aversion or self.DEFAULT_LAMBDA
        
        # Stock-specific overrides
        self._stock_params: Dict[str, Dict] = {}
    
    def set_stock_params(
        self,
        symbol: str,
        avg_daily_volume: int,
        volatility: float,
        avg_spread_bps: float = 5.0,
        avg_trade_size: int = 100
    ):
        """
        Set stock-specific parameters for better impact estimation.
        
        These parameters are used to calibrate the model:
        - Higher volume = lower impact
        - Higher volatility = higher temporary impact
        - Wider spread = higher transaction costs
        """
        # Calibrate eta (temporary impact) based on spread and volume
        # Intuition: Low liquidity stocks have higher temporary impact
        eta_calibrated = self.DEFAULT_ETA * (1_000_000 / max(avg_daily_volume, 100_000))
        eta_calibrated *= (1 + avg_spread_bps / 100)  # Adjust for spread
        
        # Calibrate gamma (permanent impact) based on volume
        gamma_calibrated = self.DEFAULT_GAMMA * (1_000_000 / max(avg_daily_volume, 100_000))
        
        self._stock_params[symbol] = {
            "adv": avg_daily_volume,
            "volatility": volatility,
            "spread_bps": avg_spread_bps,
            "avg_trade_size": avg_trade_size,
            "eta": eta_calibrated,
            "gamma": gamma_calibrated,
            "sigma": volatility
        }
        
        logger.debug(f"Set impact params for {symbol}: ADV={avg_daily_volume:,}, "
                    f"vol={volatility:.2%}, spread={avg_spread_bps}bps")
    
    def _get_params(self, symbol: str) -> Tuple[float, float, float]:
        """Get (sigma, eta, gamma) for a symbol."""
        if symbol in self._stock_params:
            p = self._stock_params[symbol]
            return p["sigma"], p["eta"], p["gamma"]
        return self.sigma, self.eta, self.gamma
    
    def estimate_impact(
        self,
        symbol: str,
        qty: int,
        price: float,
        side: str = "BUY",
        execution_time_minutes: float = 30.0
    ) -> ImpactEstimate:
        """
        Estimate market impact for an order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            price: Current price
            side: "BUY" or "SELL"
            execution_time_minutes: Planned execution duration
        
        Returns:
            ImpactEstimate with temporary and permanent impact estimates
        """
        sigma, eta, gamma = self._get_params(symbol)
        
        # Get ADV for participation rate
        params = self._stock_params.get(symbol, {})
        adv = params.get("adv", 1_000_000)
        
        # Convert execution time to trading day fraction (6.25 hours = 375 minutes)
        T = execution_time_minutes / 375.0
        
        # Participation rate (our volume / expected market volume during period)
        # Assume volume is uniformly distributed
        expected_market_volume = adv * T
        participation_rate = qty / expected_market_volume if expected_market_volume > 0 else 1.0
        
        # Temporary impact: I_temp = eta * (X/T) where X=qty, T=time
        # This is the "instantaneous" price move that reverses after trading
        trading_rate = qty / T if T > 0 else qty
        temp_impact = eta * trading_rate * price
        temp_impact_bps = (temp_impact / price) * 10000
        
        # Permanent impact: I_perm = gamma * X
        # This is the permanent price shift
        perm_impact = gamma * qty * price
        perm_impact_bps = (perm_impact / price) * 10000
        
        # Total impact
        total_impact_bps = temp_impact_bps + perm_impact_bps
        
        # Convert to rupees
        trade_value = qty * price
        slippage_rupees = trade_value * (total_impact_bps / 10000)
        
        # Sign convention: BUY orders have positive impact (worse price)
        if side == "SELL":
            temp_impact_bps = -temp_impact_bps
            perm_impact_bps = -perm_impact_bps
            total_impact_bps = -total_impact_bps
        
        # Optimal execution time (minimize total cost)
        optimal_time = self._optimal_execution_time(symbol, qty, price)
        
        # Confidence based on data availability
        confidence = 0.8 if symbol in self._stock_params else 0.5
        
        return ImpactEstimate(
            temporary_impact_bps=round(temp_impact_bps, 2),
            permanent_impact_bps=round(perm_impact_bps, 2),
            total_impact_bps=round(total_impact_bps, 2),
            estimated_slippage_rupees=round(slippage_rupees, 2),
            optimal_execution_time_minutes=round(optimal_time, 1),
            participation_rate=round(participation_rate, 4),
            confidence=confidence
        )
    
    def _optimal_execution_time(
        self,
        symbol: str,
        qty: int,
        price: float
    ) -> float:
        """
        Calculate optimal execution time using Almgren-Chriss.
        Balances urgency (risk) vs market impact.
        
        Optimal T* = sqrt(lambda * sigma^2 * X^2 / (2 * eta))
        """
        sigma, eta, gamma = self._get_params(symbol)
        
        if eta <= 0:
            return 30.0  # Default to 30 minutes
        
        # Calculate optimal time (in trading day fraction)
        numerator = self.risk_aversion * (sigma ** 2) * (qty ** 2)
        denominator = 2 * eta
        
        T_optimal = math.sqrt(numerator / denominator) if denominator > 0 else 0.1
        
        # Convert to minutes (capped at 2 hours, minimum 5 minutes)
        T_minutes = T_optimal * 375
        T_minutes = max(5, min(120, T_minutes))
        
        return T_minutes
    
    def optimal_trajectory(
        self,
        symbol: str,
        qty: int,
        price: float,
        execution_time_minutes: float = None,
        num_periods: int = 10
    ) -> OptimalTrajectory:
        """
        Calculate the optimal execution trajectory.
        
        Returns the Almgren-Chriss optimal trading schedule that minimizes
        expected cost + risk aversion * variance.
        
        The trajectory follows: x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
        where kappa = sqrt(lambda*sigma^2/(2*eta))
        """
        sigma, eta, gamma = self._get_params(symbol)
        
        if execution_time_minutes is None:
            execution_time_minutes = self._optimal_execution_time(symbol, qty, price)
        
        T = execution_time_minutes / 375.0  # Convert to day fraction
        dt = T / num_periods
        
        # Calculate kappa (urgency parameter)
        if eta > 0:
            kappa = math.sqrt(self.risk_aversion * (sigma ** 2) / (2 * eta))
        else:
            kappa = 1.0
        
        # Generate trajectory
        times = []
        holdings = []
        trading_rates = []
        
        for i in range(num_periods + 1):
            t = i * dt
            times.append(t * 375)  # Convert back to minutes
            
            # Optimal holdings at time t
            if kappa * T > 0.001:  # Avoid numerical issues
                x_t = qty * math.sinh(kappa * (T - t)) / math.sinh(kappa * T)
            else:
                # Linear decay for small kappa
                x_t = qty * (1 - t / T)
            
            holdings.append(int(x_t))
            
            # Trading rate is derivative of holdings
            if i > 0:
                rate = holdings[i - 1] - holdings[i]
                trading_rates.append(max(0, rate))
        
        # Calculate expected cost (implementation shortfall)
        # E[Cost] = 0.5 * gamma * X^2 + eta * sum(n_k^2 / dt)
        perm_cost = 0.5 * gamma * qty * qty * price
        temp_cost = eta * sum(r ** 2 for r in trading_rates) / dt * price if dt > 0 else 0
        
        total_cost = perm_cost + temp_cost
        expected_cost_bps = (total_cost / (qty * price)) * 10000 if qty > 0 else 0
        
        # Variance of cost
        # Var[Cost] = sigma^2 * sum(x_k^2 * dt)
        variance = (sigma ** 2) * sum(h ** 2 for h in holdings) * dt * (price ** 2)
        cost_variance_bps = math.sqrt(variance) / (qty * price) * 10000 if qty > 0 else 0
        
        return OptimalTrajectory(
            times=times,
            holdings=holdings,
            trading_rates=trading_rates,
            expected_cost_bps=round(expected_cost_bps, 2),
            cost_variance_bps=round(cost_variance_bps, 2)
        )
    
    def max_order_size(
        self,
        symbol: str,
        price: float,
        max_impact_bps: float = 10.0,
        execution_time_minutes: float = 30.0
    ) -> int:
        """
        Calculate maximum order size for given impact constraint.
        
        Inverts the impact formula to find max qty that keeps
        total impact below threshold.
        """
        sigma, eta, gamma = self._get_params(symbol)
        
        T = execution_time_minutes / 375.0
        
        # From: impact_bps = (eta * X/T + gamma * X) * 10000 / price
        # Solve for X: X = (impact * price / 10000) / (eta/T + gamma)
        
        denominator = (eta / T if T > 0 else eta) + gamma
        if denominator <= 0:
            return 10000  # Default max
        
        max_impact_price = max_impact_bps * price / 10000
        max_qty = int(max_impact_price / denominator)
        
        return max(1, max_qty)
    
    def recommended_participation_rate(
        self,
        symbol: str,
        qty: int,
        urgency: float = 0.5  # 0=very patient, 1=very urgent
    ) -> float:
        """
        Recommend a participation rate based on order size and urgency.
        
        Args:
            symbol: Stock symbol
            qty: Order quantity
            urgency: 0 (patient) to 1 (urgent)
        
        Returns:
            Recommended participation rate (0.01 to 0.25)
        """
        params = self._stock_params.get(symbol, {})
        adv = params.get("adv", 1_000_000)
        
        # Base participation: smaller orders can have higher participation
        order_pct_of_adv = qty / adv
        
        if order_pct_of_adv < 0.01:
            # Very small order (<1% ADV): can be more aggressive
            base_rate = 0.15
        elif order_pct_of_adv < 0.05:
            # Small order (1-5% ADV)
            base_rate = 0.10
        elif order_pct_of_adv < 0.10:
            # Medium order (5-10% ADV)
            base_rate = 0.07
        else:
            # Large order (>10% ADV): be cautious
            base_rate = 0.05
        
        # Adjust for urgency
        # High urgency = higher participation (accept more impact)
        rate = base_rate * (0.5 + urgency)
        
        # Clamp to reasonable range
        return max(0.01, min(0.25, rate))


class SlippageModel:
    """
    Advanced slippage model considering:
    - Order size relative to volume
    - Bid-ask spread
    - Volatility
    - Time of day
    """
    
    def __init__(self, base_slippage_bps: float = 5.0):
        self.base_slippage_bps = base_slippage_bps
        self._stock_data: Dict[str, Dict] = {}
    
    def set_stock_data(
        self,
        symbol: str,
        avg_daily_volume: int,
        avg_spread_bps: float,
        volatility: float
    ):
        """Set stock-specific data for slippage estimation."""
        self._stock_data[symbol] = {
            "adv": avg_daily_volume,
            "spread_bps": avg_spread_bps,
            "volatility": volatility
        }
    
    def estimate_slippage(
        self,
        symbol: str,
        qty: int,
        price: float,
        side: str = "BUY",
        time_of_day: str = "mid"  # "open", "mid", "close"
    ) -> float:
        """
        Estimate slippage in basis points.
        
        Components:
        1. Half spread (crossing bid-ask)
        2. Size impact (larger orders = more slippage)
        3. Volatility adjustment
        4. Time of day adjustment
        """
        data = self._stock_data.get(symbol, {})
        adv = data.get("adv", 1_000_000)
        spread_bps = data.get("spread_bps", 5.0)
        volatility = data.get("volatility", 0.02)
        
        # 1. Half spread (always pay half the spread to cross)
        spread_cost = spread_bps / 2
        
        # 2. Size impact: larger % of ADV = more slippage
        size_pct = qty / adv
        if size_pct < 0.001:
            size_impact = 0
        elif size_pct < 0.01:
            size_impact = size_pct * 100  # 0-1 bps
        elif size_pct < 0.05:
            size_impact = 1 + (size_pct - 0.01) * 200  # 1-9 bps
        else:
            size_impact = 9 + (size_pct - 0.05) * 500  # 9+ bps
        
        # 3. Volatility adjustment: higher vol = more slippage
        vol_multiplier = volatility / 0.02  # Normalized to 2% base
        
        # 4. Time of day: higher slippage at open/close
        time_multiplier = {
            "open": 1.5,   # High volatility at open
            "mid": 1.0,    # Normal
            "close": 1.3   # Some volatility at close
        }.get(time_of_day, 1.0)
        
        # Combine components
        total_slippage = (spread_cost + size_impact) * vol_multiplier * time_multiplier
        
        # Add base slippage
        total_slippage += self.base_slippage_bps
        
        # Sign: BUY = positive (pay more), SELL = positive (receive less)
        return round(total_slippage, 2)
    
    def estimate_slippage_rupees(
        self,
        symbol: str,
        qty: int,
        price: float,
        side: str = "BUY",
        time_of_day: str = "mid"
    ) -> float:
        """Estimate slippage in rupees."""
        slippage_bps = self.estimate_slippage(symbol, qty, price, side, time_of_day)
        trade_value = qty * price
        return round(trade_value * slippage_bps / 10000, 2)
