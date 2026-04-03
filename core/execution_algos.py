"""
Execution Algorithms - Institutional-grade order execution strategies.

Features:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)  
- Implementation Shortfall
- Iceberg/Hidden orders
- Adaptive execution with participation rate control
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionAlgoType(Enum):
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "is"
    ICEBERG = "iceberg"
    POV = "pov"  # Percentage of Volume


@dataclass
class ExecutionSlice:
    """Represents a single slice of a parent order."""
    slice_id: str
    parent_algo_id: str
    target_qty: int
    executed_qty: int = 0
    avg_price: float = 0.0
    scheduled_time: datetime = None
    executed_time: datetime = None
    status: str = "pending"  # pending, executing, complete, cancelled


@dataclass
class AlgoStats:
    """Statistics for execution algorithm performance."""
    total_qty: int
    executed_qty: int
    remaining_qty: int
    avg_execution_price: float
    arrival_price: float  # Price at algo start
    vwap_benchmark: float  # Market VWAP during execution
    twap_benchmark: float  # Time-weighted average price
    slippage_bps: float   # Basis points vs arrival
    participation_rate: float
    execution_time_seconds: float
    num_slices: int
    num_fills: int


class BaseExecutionAlgo(ABC):
    """Base class for execution algorithms."""
    
    def __init__(
        self,
        algo_id: str,
        symbol: str,
        side: str,
        total_qty: int,
        start_time: datetime,
        end_time: datetime,
        on_slice_ready: Callable = None,
        price_fetcher: Callable = None,
        volume_fetcher: Callable = None
    ):
        self.algo_id = algo_id
        self.symbol = symbol
        self.side = side
        self.total_qty = total_qty
        self.start_time = start_time
        self.end_time = end_time
        self.on_slice_ready = on_slice_ready
        self.price_fetcher = price_fetcher
        self.volume_fetcher = volume_fetcher
        
        self.arrival_price = 0.0
        self.executed_qty = 0
        self.executed_value = 0.0
        self.slices: List[ExecutionSlice] = []
        self.fills: List[Tuple[int, float, datetime]] = []  # (qty, price, time)
        
        self.is_active = False
        self.is_complete = False
        self.is_cancelled = False
        
        self._lock = threading.Lock()
        self._execution_thread = None
    
    @property
    def remaining_qty(self) -> int:
        return self.total_qty - self.executed_qty
    
    @property
    def avg_execution_price(self) -> float:
        if self.executed_qty == 0:
            return 0.0
        return self.executed_value / self.executed_qty
    
    @abstractmethod
    def generate_schedule(self) -> List[ExecutionSlice]:
        """Generate execution schedule based on algo logic."""
        pass
    
    def start(self):
        """Start the execution algorithm."""
        if self.is_active:
            logger.warning(f"Algo {self.algo_id} already running")
            return
        
        # Capture arrival price
        if self.price_fetcher:
            self.arrival_price = self.price_fetcher(self.symbol)
        
        # Generate schedule
        self.slices = self.generate_schedule()
        
        self.is_active = True
        logger.info(f"Starting {self.__class__.__name__} for {self.symbol}: "
                   f"{self.total_qty} shares in {len(self.slices)} slices")
        
        # Start execution thread
        self._execution_thread = threading.Thread(target=self._run, daemon=True)
        self._execution_thread.start()
    
    def _run(self):
        """Main execution loop."""
        while self.is_active and not self.is_complete and not self.is_cancelled:
            now = datetime.now()
            
            # Check if past end time
            if now >= self.end_time:
                logger.info(f"Algo {self.algo_id} reached end time")
                self._handle_completion()
                break
            
            # Find next slice to execute
            for slice_obj in self.slices:
                if slice_obj.status == "pending" and slice_obj.scheduled_time <= now:
                    self._execute_slice(slice_obj)
            
            # Check if all done
            if self.executed_qty >= self.total_qty:
                self._handle_completion()
                break
            
            time.sleep(1)  # Check every second
    
    def _execute_slice(self, slice_obj: ExecutionSlice):
        """Execute a single slice."""
        slice_obj.status = "executing"
        
        # Adjust quantity if we're near the end
        actual_qty = min(slice_obj.target_qty, self.remaining_qty)
        
        if actual_qty <= 0:
            slice_obj.status = "complete"
            return
        
        # Notify callback to place order
        if self.on_slice_ready:
            try:
                self.on_slice_ready(
                    algo_id=self.algo_id,
                    symbol=self.symbol,
                    side=self.side,
                    qty=actual_qty,
                    slice_id=slice_obj.slice_id
                )
                slice_obj.executed_time = datetime.now()
                logger.debug(f"Slice {slice_obj.slice_id}: {actual_qty} shares queued")
            except Exception as e:
                logger.error(f"Slice execution failed: {e}")
                slice_obj.status = "pending"  # Retry later
                return
        
        slice_obj.status = "complete"
    
    def record_fill(self, qty: int, price: float):
        """Record a fill from the market."""
        with self._lock:
            self.fills.append((qty, price, datetime.now()))
            self.executed_qty += qty
            self.executed_value += qty * price
    
    def cancel(self):
        """Cancel the algorithm."""
        self.is_cancelled = True
        self.is_active = False
        logger.info(f"Algo {self.algo_id} cancelled. Executed: {self.executed_qty}/{self.total_qty}")
    
    def _handle_completion(self):
        """Handle algorithm completion."""
        self.is_complete = True
        self.is_active = False
    
    def get_stats(self) -> AlgoStats:
        """Get execution statistics."""
        vwap = self._calculate_vwap()
        twap = self._calculate_twap()
        
        slippage = 0.0
        if self.arrival_price > 0 and self.avg_execution_price > 0:
            if self.side == "BUY":
                slippage = (self.avg_execution_price - self.arrival_price) / self.arrival_price * 10000
            else:
                slippage = (self.arrival_price - self.avg_execution_price) / self.arrival_price * 10000
        
        exec_time = 0.0
        if self.fills:
            exec_time = (self.fills[-1][2] - self.start_time).total_seconds()
        
        return AlgoStats(
            total_qty=self.total_qty,
            executed_qty=self.executed_qty,
            remaining_qty=self.remaining_qty,
            avg_execution_price=self.avg_execution_price,
            arrival_price=self.arrival_price,
            vwap_benchmark=vwap,
            twap_benchmark=twap,
            slippage_bps=slippage,
            participation_rate=self._calculate_participation_rate(),
            execution_time_seconds=exec_time,
            num_slices=len(self.slices),
            num_fills=len(self.fills)
        )
    
    def _calculate_vwap(self) -> float:
        """Calculate VWAP from fills."""
        if not self.fills:
            return 0.0
        total_value = sum(qty * price for qty, price, _ in self.fills)
        total_qty = sum(qty for qty, _, _ in self.fills)
        return total_value / total_qty if total_qty > 0 else 0.0
    
    def _calculate_twap(self) -> float:
        """Calculate TWAP from fills."""
        if not self.fills:
            return 0.0
        return sum(price for _, price, _ in self.fills) / len(self.fills)
    
    def _calculate_participation_rate(self) -> float:
        """Calculate participation rate (our volume / market volume)."""
        if self.volume_fetcher:
            market_volume = self.volume_fetcher(self.symbol)
            if market_volume > 0:
                return self.executed_qty / market_volume
        return 0.0


class TWAPAlgo(BaseExecutionAlgo):
    """
    Time-Weighted Average Price execution.
    Splits order into equal-sized slices over the execution window.
    """
    
    def __init__(
        self,
        algo_id: str,
        symbol: str,
        side: str,
        total_qty: int,
        start_time: datetime,
        end_time: datetime,
        num_slices: int = 10,
        randomize: bool = True,  # Add randomness to slice timing
        **kwargs
    ):
        super().__init__(algo_id, symbol, side, total_qty, start_time, end_time, **kwargs)
        self.num_slices = num_slices
        self.randomize = randomize
    
    def generate_schedule(self) -> List[ExecutionSlice]:
        """Generate TWAP schedule with equal time intervals."""
        slices = []
        duration = (self.end_time - self.start_time).total_seconds()
        interval = duration / self.num_slices
        base_qty = self.total_qty // self.num_slices
        remainder = self.total_qty % self.num_slices
        
        for i in range(self.num_slices):
            # Add remainder to last slice
            qty = base_qty + (1 if i < remainder else 0)
            
            # Calculate slice time
            offset_seconds = interval * i
            if self.randomize and i > 0:
                # Add ±20% randomness to timing (not first slice)
                jitter = interval * 0.2 * (np.random.random() * 2 - 1)
                offset_seconds += jitter
            
            scheduled_time = self.start_time + timedelta(seconds=offset_seconds)
            
            slices.append(ExecutionSlice(
                slice_id=f"{self.algo_id}_S{i+1:02d}",
                parent_algo_id=self.algo_id,
                target_qty=qty,
                scheduled_time=scheduled_time
            ))
        
        return slices


class VWAPAlgo(BaseExecutionAlgo):
    """
    Volume-Weighted Average Price execution.
    Distributes order according to historical volume profile.
    """
    
    def __init__(
        self,
        algo_id: str,
        symbol: str,
        side: str,
        total_qty: int,
        start_time: datetime,
        end_time: datetime,
        volume_profile: List[float] = None,  # Historical volume by time bucket
        participation_limit: float = 0.15,   # Max 15% of volume
        **kwargs
    ):
        super().__init__(algo_id, symbol, side, total_qty, start_time, end_time, **kwargs)
        self.volume_profile = volume_profile or self._default_volume_profile()
        self.participation_limit = participation_limit
    
    def _default_volume_profile(self) -> List[float]:
        """
        Default NSE intraday volume profile (9:15 AM to 3:30 PM).
        Higher volume at open, lunch dip, and close.
        """
        # 30-minute buckets (13 buckets for full day)
        return [
            0.15,  # 9:15-9:45  - High opening volume
            0.10,  # 9:45-10:15
            0.08,  # 10:15-10:45
            0.07,  # 10:45-11:15
            0.06,  # 11:15-11:45
            0.05,  # 11:45-12:15 - Lunch lull
            0.05,  # 12:15-12:45
            0.06,  # 12:45-1:15
            0.07,  # 1:15-1:45
            0.08,  # 1:45-2:15
            0.08,  # 2:15-2:45
            0.10,  # 2:45-3:15  - Pre-close pickup
            0.05,  # 3:15-3:30  - Final 15 min
        ]
    
    def generate_schedule(self) -> List[ExecutionSlice]:
        """Generate VWAP schedule based on volume profile."""
        slices = []
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Normalize profile to match our time window
        num_buckets = len(self.volume_profile)
        bucket_duration = duration / num_buckets
        
        # Calculate quantity per bucket based on volume weights
        total_weight = sum(self.volume_profile)
        
        for i, weight in enumerate(self.volume_profile):
            # Allocate qty proportional to expected volume
            qty = int(self.total_qty * (weight / total_weight))
            
            if qty <= 0:
                continue
            
            # Schedule at start of bucket
            scheduled_time = self.start_time + timedelta(seconds=bucket_duration * i)
            
            slices.append(ExecutionSlice(
                slice_id=f"{self.algo_id}_V{i+1:02d}",
                parent_algo_id=self.algo_id,
                target_qty=qty,
                scheduled_time=scheduled_time
            ))
        
        # Ensure we allocate full quantity
        allocated = sum(s.target_qty for s in slices)
        if allocated < self.total_qty and slices:
            slices[-1].target_qty += (self.total_qty - allocated)
        
        return slices


class ImplementationShortfallAlgo(BaseExecutionAlgo):
    """
    Implementation Shortfall (Arrival Price) execution.
    Balances urgency vs market impact, front-loads execution.
    Uses Almgren-Chriss optimal trajectory.
    """
    
    def __init__(
        self,
        algo_id: str,
        symbol: str,
        side: str,
        total_qty: int,
        start_time: datetime,
        end_time: datetime,
        urgency: float = 0.5,  # 0=passive, 1=aggressive
        volatility: float = 0.02,  # Daily volatility
        avg_daily_volume: int = 1000000,
        **kwargs
    ):
        super().__init__(algo_id, symbol, side, total_qty, start_time, end_time, **kwargs)
        self.urgency = urgency
        self.volatility = volatility
        self.avg_daily_volume = avg_daily_volume
    
    def generate_schedule(self) -> List[ExecutionSlice]:
        """
        Generate IS schedule using simplified Almgren-Chriss model.
        More urgent = faster execution (front-loaded).
        """
        slices = []
        num_slices = max(5, min(20, self.total_qty // 100))  # 5-20 slices
        
        duration = (self.end_time - self.start_time).total_seconds()
        interval = duration / num_slices
        
        # Calculate exponential decay rate based on urgency
        # Higher urgency = faster decay = more front-loaded
        decay_rate = 1 + self.urgency * 2  # Range: 1 to 3
        
        # Generate weights using exponential decay
        weights = []
        for i in range(num_slices):
            t = i / num_slices
            weight = np.exp(-decay_rate * t)
            weights.append(weight)
        
        total_weight = sum(weights)
        
        for i, weight in enumerate(weights):
            qty = int(self.total_qty * (weight / total_weight))
            if qty <= 0:
                continue
            
            scheduled_time = self.start_time + timedelta(seconds=interval * i)
            
            slices.append(ExecutionSlice(
                slice_id=f"{self.algo_id}_IS{i+1:02d}",
                parent_algo_id=self.algo_id,
                target_qty=qty,
                scheduled_time=scheduled_time
            ))
        
        # Ensure full allocation
        allocated = sum(s.target_qty for s in slices)
        if allocated < self.total_qty and slices:
            slices[0].target_qty += (self.total_qty - allocated)  # Add to first slice
        
        return slices


class IcebergAlgo(BaseExecutionAlgo):
    """
    Iceberg/Hidden order execution.
    Shows only a small visible quantity while hiding the full size.
    """
    
    def __init__(
        self,
        algo_id: str,
        symbol: str,
        side: str,
        total_qty: int,
        start_time: datetime,
        end_time: datetime,
        visible_qty: int = None,  # Quantity to show
        min_visible_pct: float = 0.05,  # Minimum 5% visible
        max_visible_pct: float = 0.15,  # Maximum 15% visible
        randomize_size: bool = True,    # Vary visible size
        **kwargs
    ):
        super().__init__(algo_id, symbol, side, total_qty, start_time, end_time, **kwargs)
        
        if visible_qty:
            self.visible_qty = visible_qty
        else:
            # Default to 10% visible
            self.visible_qty = max(1, int(total_qty * 0.10))
        
        self.min_visible_pct = min_visible_pct
        self.max_visible_pct = max_visible_pct
        self.randomize_size = randomize_size
    
    def generate_schedule(self) -> List[ExecutionSlice]:
        """Generate iceberg schedule - continuous refill slices."""
        slices = []
        remaining = self.total_qty
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Estimate number of slices needed
        num_slices = max(1, self.total_qty // self.visible_qty)
        interval = duration / num_slices if num_slices > 1 else duration
        
        slice_num = 0
        current_time = self.start_time
        
        while remaining > 0:
            # Calculate slice size
            if self.randomize_size:
                # Vary between min and max visible percentage
                pct = self.min_visible_pct + np.random.random() * (self.max_visible_pct - self.min_visible_pct)
                qty = max(1, int(self.total_qty * pct))
            else:
                qty = self.visible_qty
            
            qty = min(qty, remaining)
            
            slices.append(ExecutionSlice(
                slice_id=f"{self.algo_id}_ICE{slice_num+1:02d}",
                parent_algo_id=self.algo_id,
                target_qty=qty,
                scheduled_time=current_time
            ))
            
            remaining -= qty
            current_time += timedelta(seconds=interval)
            slice_num += 1
            
            # Safety limit
            if slice_num > 100:
                if slices:
                    slices[-1].target_qty += remaining
                break
        
        return slices


class POVAlgo(BaseExecutionAlgo):
    """
    Percentage of Volume (POV) execution.
    Executes as a fixed percentage of market volume.
    """
    
    def __init__(
        self,
        algo_id: str,
        symbol: str,
        side: str,
        total_qty: int,
        start_time: datetime,
        end_time: datetime,
        target_participation: float = 0.10,  # 10% of volume
        max_participation: float = 0.20,     # Never exceed 20%
        **kwargs
    ):
        super().__init__(algo_id, symbol, side, total_qty, start_time, end_time, **kwargs)
        self.target_participation = target_participation
        self.max_participation = max_participation
        self._last_market_volume = 0
        self._last_check_time = None
    
    def generate_schedule(self) -> List[ExecutionSlice]:
        """
        POV generates slices dynamically based on market volume.
        Initial schedule is just a placeholder.
        """
        # POV is reactive - we create one initial slice and adapt
        return [ExecutionSlice(
            slice_id=f"{self.algo_id}_POV_INIT",
            parent_algo_id=self.algo_id,
            target_qty=0,  # Will be set dynamically
            scheduled_time=self.start_time
        )]
    
    def _run(self):
        """Override run to implement volume-reactive execution."""
        check_interval = 30  # Check every 30 seconds
        
        while self.is_active and not self.is_complete and not self.is_cancelled:
            now = datetime.now()
            
            if now >= self.end_time:
                self._handle_completion()
                break
            
            # Get current market volume
            if self.volume_fetcher:
                current_volume = self.volume_fetcher(self.symbol)
                volume_delta = current_volume - self._last_market_volume
                
                if volume_delta > 0:
                    # Calculate our share of this volume
                    target_qty = int(volume_delta * self.target_participation)
                    target_qty = min(target_qty, self.remaining_qty)
                    
                    # Cap at max participation
                    max_qty = int(volume_delta * self.max_participation)
                    target_qty = min(target_qty, max_qty)
                    
                    if target_qty > 0 and self.on_slice_ready:
                        slice_obj = ExecutionSlice(
                            slice_id=f"{self.algo_id}_POV_{len(self.slices)}",
                            parent_algo_id=self.algo_id,
                            target_qty=target_qty,
                            scheduled_time=now
                        )
                        self.slices.append(slice_obj)
                        self._execute_slice(slice_obj)
                    
                    self._last_market_volume = current_volume
            
            if self.executed_qty >= self.total_qty:
                self._handle_completion()
                break
            
            time.sleep(check_interval)


class ExecutionAlgoManager:
    """
    Manager for running multiple execution algorithms.
    Provides unified interface for algo creation, monitoring, and reporting.
    """
    
    def __init__(self, oms=None, price_fetcher=None, volume_fetcher=None):
        self.oms = oms
        self.price_fetcher = price_fetcher
        self.volume_fetcher = volume_fetcher
        
        self._algos: Dict[str, BaseExecutionAlgo] = {}
        self._lock = threading.Lock()
    
    def create_algo(
        self,
        algo_type: ExecutionAlgoType,
        symbol: str,
        side: str,
        qty: int,
        start_time: datetime = None,
        end_time: datetime = None,
        **kwargs
    ) -> Optional[BaseExecutionAlgo]:
        """Create and register a new execution algorithm."""
        
        if start_time is None:
            start_time = datetime.now()
        if end_time is None:
            # Default: execute over 30 minutes
            end_time = start_time + timedelta(minutes=30)
        
        algo_id = f"{algo_type.value}_{symbol}_{datetime.now().strftime('%H%M%S')}"
        
        # Common kwargs
        common = {
            "algo_id": algo_id,
            "symbol": symbol,
            "side": side,
            "total_qty": qty,
            "start_time": start_time,
            "end_time": end_time,
            "on_slice_ready": self._on_slice_ready,
            "price_fetcher": self.price_fetcher,
            "volume_fetcher": self.volume_fetcher,
        }
        
        algo = None
        if algo_type == ExecutionAlgoType.TWAP:
            algo = TWAPAlgo(**common, **kwargs)
        elif algo_type == ExecutionAlgoType.VWAP:
            algo = VWAPAlgo(**common, **kwargs)
        elif algo_type == ExecutionAlgoType.IMPLEMENTATION_SHORTFALL:
            algo = ImplementationShortfallAlgo(**common, **kwargs)
        elif algo_type == ExecutionAlgoType.ICEBERG:
            algo = IcebergAlgo(**common, **kwargs)
        elif algo_type == ExecutionAlgoType.POV:
            algo = POVAlgo(**common, **kwargs)
        
        if algo:
            with self._lock:
                self._algos[algo_id] = algo
            logger.info(f"Created {algo_type.value} algo: {algo_id}")
        
        return algo
    
    def _on_slice_ready(self, algo_id: str, symbol: str, side: str, qty: int, slice_id: str):
        """Callback when algo slice needs execution."""
        if self.oms:
            from core.oms import OrderSide, OrderType
            
            order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
            order = self.oms.create_order(
                symbol=symbol,
                side=order_side,
                qty=qty,
                order_type=OrderType.MARKET,
                strategy_name=f"ALGO:{algo_id}"
            )
            if order:
                self.oms.submit_order(order.order_id)
                logger.debug(f"Slice {slice_id} submitted as order {order.order_id}")
    
    def start_algo(self, algo_id: str) -> bool:
        """Start an algorithm."""
        algo = self._algos.get(algo_id)
        if not algo:
            logger.error(f"Algo not found: {algo_id}")
            return False
        algo.start()
        return True
    
    def cancel_algo(self, algo_id: str) -> bool:
        """Cancel an algorithm."""
        algo = self._algos.get(algo_id)
        if not algo:
            return False
        algo.cancel()
        return True
    
    def record_fill(self, algo_id: str, qty: int, price: float):
        """Record a fill for an algorithm."""
        algo = self._algos.get(algo_id)
        if algo:
            algo.record_fill(qty, price)
    
    def get_algo_stats(self, algo_id: str) -> Optional[AlgoStats]:
        """Get statistics for an algorithm."""
        algo = self._algos.get(algo_id)
        if algo:
            return algo.get_stats()
        return None
    
    def get_all_stats(self) -> Dict[str, AlgoStats]:
        """Get statistics for all algorithms."""
        return {aid: algo.get_stats() for aid, algo in self._algos.items()}
    
    def get_active_algos(self) -> List[str]:
        """Get list of active algorithm IDs."""
        return [aid for aid, algo in self._algos.items() if algo.is_active]
