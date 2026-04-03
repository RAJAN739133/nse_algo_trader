"""
Order Management System (OMS) - Institutional-grade order lifecycle management.

Features:
- Order state machine with full lifecycle tracking
- Order queuing with priority handling
- Partial fill management
- Parent/child order relationships (for brackets)
- Amendment and cancellation support
- Audit trail for compliance
"""

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Order lifecycle states following FIX protocol semantics."""
    PENDING_NEW = "pending_new"          # Created, not yet submitted
    NEW = "new"                          # Accepted by broker
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELLED = "cancelled"
    PENDING_AMEND = "pending_amend"
    AMENDED = "amended"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    DAY = "DAY"           # Valid for the trading day
    IOC = "IOC"           # Immediate or Cancel
    FOK = "FOK"           # Fill or Kill
    GTD = "GTD"           # Good Till Date
    GTC = "GTC"           # Good Till Cancelled


class OrderPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4  # For stop-loss orders


@dataclass
class Fill:
    """Represents a single fill (execution) event."""
    fill_id: str
    order_id: str
    timestamp: datetime
    qty: int
    price: float
    commission: float = 0.0
    exchange_order_id: str = ""
    
    def __post_init__(self):
        if not self.fill_id:
            self.fill_id = str(uuid.uuid4())[:8]


@dataclass
class Order:
    """Comprehensive order object with full state tracking."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: int
    
    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_amount: Optional[float] = None  # For trailing stops
    trailing_percent: Optional[float] = None
    
    # Time in force
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_time: Optional[datetime] = None
    
    # State tracking
    state: OrderState = OrderState.PENDING_NEW
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    remaining_qty: int = 0
    
    # Fills history
    fills: List[Fill] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Parent/Child relationships for bracket orders
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    
    # Routing
    exchange: str = "NSE"
    product_type: str = "INTRADAY"
    broker_order_id: str = ""
    
    # Priority for queue
    priority: OrderPriority = OrderPriority.NORMAL
    
    # Strategy context
    strategy_name: str = ""
    signal_id: str = ""
    
    # Audit
    state_history: List[Dict] = field(default_factory=list)
    rejection_reason: str = ""
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.remaining_qty = self.qty - self.filled_qty
        self._record_state_change("CREATED", self.state.value)
    
    def _record_state_change(self, from_state: str, to_state: str, reason: str = ""):
        """Record state transition for audit trail."""
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason
        })
        self.last_updated = datetime.now()
    
    def transition_to(self, new_state: OrderState, reason: str = "") -> bool:
        """
        Transition order to new state with validation.
        Returns True if transition successful.
        """
        valid_transitions = {
            OrderState.PENDING_NEW: [OrderState.NEW, OrderState.REJECTED],
            OrderState.NEW: [OrderState.PARTIALLY_FILLED, OrderState.FILLED, 
                           OrderState.PENDING_CANCEL, OrderState.PENDING_AMEND,
                           OrderState.EXPIRED, OrderState.REJECTED],
            OrderState.PARTIALLY_FILLED: [OrderState.FILLED, OrderState.PENDING_CANCEL,
                                         OrderState.PENDING_AMEND, OrderState.CANCELLED],
            OrderState.PENDING_CANCEL: [OrderState.CANCELLED, OrderState.FILLED,
                                       OrderState.PARTIALLY_FILLED],
            OrderState.PENDING_AMEND: [OrderState.AMENDED, OrderState.REJECTED,
                                      OrderState.NEW],
            OrderState.AMENDED: [OrderState.NEW, OrderState.PARTIALLY_FILLED,
                                OrderState.FILLED, OrderState.PENDING_CANCEL],
        }
        
        if new_state not in valid_transitions.get(self.state, []):
            logger.warning(f"Invalid state transition: {self.state.value} -> {new_state.value}")
            return False
        
        old_state = self.state
        self._record_state_change(old_state.value, new_state.value, reason)
        self.state = new_state
        
        if new_state == OrderState.REJECTED:
            self.rejection_reason = reason
            
        logger.debug(f"Order {self.order_id}: {old_state.value} -> {new_state.value}")
        return True
    
    def add_fill(self, fill: Fill):
        """Process a fill event."""
        self.fills.append(fill)
        
        # Update fill quantities
        old_filled = self.filled_qty
        self.filled_qty += fill.qty
        self.remaining_qty = self.qty - self.filled_qty
        
        # Calculate weighted average price
        if self.filled_qty > 0:
            total_value = (self.avg_fill_price * old_filled) + (fill.price * fill.qty)
            self.avg_fill_price = total_value / self.filled_qty
        
        # Update state
        if self.filled_qty >= self.qty:
            self.transition_to(OrderState.FILLED, f"Filled: {fill.qty}@{fill.price}")
        elif self.filled_qty > 0:
            self.transition_to(OrderState.PARTIALLY_FILLED, 
                             f"Partial: {self.filled_qty}/{self.qty}")
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (can receive fills)."""
        return self.state in [OrderState.NEW, OrderState.PARTIALLY_FILLED,
                             OrderState.PENDING_CANCEL, OrderState.PENDING_AMEND]
    
    @property
    def is_complete(self) -> bool:
        """Check if order is in terminal state."""
        return self.state in [OrderState.FILLED, OrderState.CANCELLED,
                             OrderState.REJECTED, OrderState.EXPIRED]
    
    def to_dict(self) -> dict:
        """Serialize order to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "qty": self.qty,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "state": self.state.value,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "remaining_qty": self.remaining_qty,
            "created_at": self.created_at.isoformat(),
            "broker_order_id": self.broker_order_id,
            "strategy_name": self.strategy_name,
            "fills": [{"qty": f.qty, "price": f.price, "time": f.timestamp.isoformat()} 
                     for f in self.fills],
            "state_history": self.state_history,
        }


class OrderQueue:
    """Priority queue for order management."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues: Dict[OrderPriority, deque] = {
            p: deque() for p in OrderPriority
        }
        self._lock = threading.Lock()
    
    def enqueue(self, order: Order) -> bool:
        """Add order to queue. Returns False if queue full."""
        with self._lock:
            total_size = sum(len(q) for q in self._queues.values())
            if total_size >= self.max_size:
                logger.warning(f"Order queue full ({total_size} orders)")
                return False
            self._queues[order.priority].append(order)
            return True
    
    def dequeue(self) -> Optional[Order]:
        """Get highest priority order."""
        with self._lock:
            # Check queues from highest to lowest priority
            for priority in reversed(OrderPriority):
                if self._queues[priority]:
                    return self._queues[priority].popleft()
        return None
    
    def peek(self) -> Optional[Order]:
        """View next order without removing."""
        with self._lock:
            for priority in reversed(OrderPriority):
                if self._queues[priority]:
                    return self._queues[priority][0]
        return None
    
    def size(self) -> int:
        """Total orders in queue."""
        with self._lock:
            return sum(len(q) for q in self._queues.values())
    
    def clear(self):
        """Clear all queues."""
        with self._lock:
            for q in self._queues.values():
                q.clear()


class OrderManagementSystem:
    """
    Central order management system with full lifecycle support.
    
    Features:
    - Order creation, submission, amendment, cancellation
    - Order queuing with priority
    - Partial fill handling
    - Bracket order management
    - Audit trail
    """
    
    def __init__(
        self,
        broker_adapter=None,
        max_orders_per_day: int = 100,
        max_pending_orders: int = 50,
        audit_file: str = "logs/oms_audit.jsonl"
    ):
        self.broker_adapter = broker_adapter
        self.max_orders_per_day = max_orders_per_day
        self.max_pending_orders = max_pending_orders
        self.audit_file = Path(audit_file)
        
        # Order storage
        self._orders: Dict[str, Order] = {}
        self._orders_by_symbol: Dict[str, List[str]] = {}
        self._broker_to_oms: Dict[str, str] = {}  # broker_order_id -> order_id
        
        # Queues
        self._submission_queue = OrderQueue()
        
        # Statistics
        self._orders_today = 0
        self._last_reset_date = datetime.now().date()
        
        # Callbacks
        self._on_fill_callbacks: List[Callable] = []
        self._on_state_change_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Ensure audit directory exists
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _reset_daily_counters(self):
        """Reset counters if new trading day."""
        today = datetime.now().date()
        if today != self._last_reset_date:
            self._orders_today = 0
            self._last_reset_date = today
    
    def _audit_log(self, event_type: str, order: Order, details: dict = None):
        """Write to audit log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "qty": order.qty,
            "state": order.state.value,
            "details": details or {}
        }
        try:
            with open(self.audit_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float = None,
        stop_price: float = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        priority: OrderPriority = OrderPriority.NORMAL,
        strategy_name: str = "",
        parent_order_id: str = None,
        **kwargs
    ) -> Optional[Order]:
        """
        Create a new order (does not submit yet).
        Returns Order object or None if creation fails.
        """
        self._reset_daily_counters()
        
        # Validate daily limits
        if self._orders_today >= self.max_orders_per_day:
            logger.warning(f"Daily order limit reached ({self.max_orders_per_day})")
            return None
        
        # Validate pending order limits
        pending = sum(1 for o in self._orders.values() if o.is_active)
        if pending >= self.max_pending_orders:
            logger.warning(f"Max pending orders reached ({self.max_pending_orders})")
            return None
        
        # Validate order parameters
        if order_type == OrderType.LIMIT and limit_price is None:
            logger.error("Limit order requires limit_price")
            return None
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            logger.error("Stop order requires stop_price")
            return None
        
        order = Order(
            order_id="",  # Will be auto-generated
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            priority=priority,
            strategy_name=strategy_name,
            parent_order_id=parent_order_id,
            **kwargs
        )
        
        with self._lock:
            self._orders[order.order_id] = order
            if symbol not in self._orders_by_symbol:
                self._orders_by_symbol[symbol] = []
            self._orders_by_symbol[symbol].append(order.order_id)
            self._orders_today += 1
        
        self._audit_log("ORDER_CREATED", order)
        logger.info(f"Order created: {order.order_id} {side.value} {qty} {symbol}")
        
        return order
    
    def submit_order(self, order_id: str) -> bool:
        """Submit order to broker."""
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False
        
        if order.state != OrderState.PENDING_NEW:
            logger.error(f"Cannot submit order in state {order.state.value}")
            return False
        
        # Add to submission queue
        if not self._submission_queue.enqueue(order):
            order.transition_to(OrderState.REJECTED, "Queue full")
            return False
        
        order.submitted_at = datetime.now()
        
        # If we have a broker adapter, submit immediately
        if self.broker_adapter:
            return self._execute_submission(order)
        else:
            # Simulate immediate acceptance for paper trading
            order.transition_to(OrderState.NEW, "Accepted (paper)")
            self._audit_log("ORDER_SUBMITTED", order)
            return True
    
    def _execute_submission(self, order: Order) -> bool:
        """Execute order submission to broker."""
        try:
            # Convert to broker format
            broker_params = {
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": order.qty,
                "order_type": order.order_type.value,
                "price": order.limit_price or 0.0,
                "trigger_price": order.stop_price or 0.0,
                "product": order.product_type,
            }
            
            result = self.broker_adapter.place_order(**broker_params)
            
            if result and hasattr(result, 'order_id'):
                order.broker_order_id = result.order_id
                self._broker_to_oms[result.order_id] = order.order_id
                
                if result.status == "COMPLETE":
                    # Immediate fill (market order)
                    fill = Fill(
                        fill_id="",
                        order_id=order.order_id,
                        timestamp=datetime.now(),
                        qty=order.qty,
                        price=result.fill_price
                    )
                    order.add_fill(fill)
                    order.transition_to(OrderState.FILLED, "Market fill")
                else:
                    order.transition_to(OrderState.NEW, "Accepted by broker")
                
                self._audit_log("ORDER_SUBMITTED", order, {"broker_id": result.order_id})
                return True
            else:
                order.transition_to(OrderState.REJECTED, 
                                   result.rejection_reason if result else "Broker rejection")
                self._audit_log("ORDER_REJECTED", order)
                return False
                
        except Exception as e:
            order.transition_to(OrderState.REJECTED, str(e))
            self._audit_log("ORDER_ERROR", order, {"error": str(e)})
            logger.error(f"Order submission failed: {e}")
            return False
    
    def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """Request order cancellation."""
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False
        
        if not order.is_active:
            logger.warning(f"Cannot cancel order in state {order.state.value}")
            return False
        
        order.transition_to(OrderState.PENDING_CANCEL, reason)
        
        if self.broker_adapter and order.broker_order_id:
            try:
                self.broker_adapter.cancel_order(order.broker_order_id)
                order.transition_to(OrderState.CANCELLED, reason)
                self._audit_log("ORDER_CANCELLED", order, {"reason": reason})
                return True
            except Exception as e:
                logger.error(f"Cancel failed: {e}")
                return False
        else:
            # Paper trading: immediate cancel
            order.transition_to(OrderState.CANCELLED, reason)
            self._audit_log("ORDER_CANCELLED", order, {"reason": reason})
            return True
    
    def amend_order(
        self,
        order_id: str,
        new_qty: int = None,
        new_limit_price: float = None,
        new_stop_price: float = None
    ) -> bool:
        """Amend an active order."""
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False
        
        if order.state not in [OrderState.NEW, OrderState.PARTIALLY_FILLED]:
            logger.warning(f"Cannot amend order in state {order.state.value}")
            return False
        
        old_values = {
            "qty": order.qty,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price
        }
        
        order.transition_to(OrderState.PENDING_AMEND, "Amendment requested")
        
        # Apply amendments
        if new_qty is not None and new_qty > order.filled_qty:
            order.qty = new_qty
            order.remaining_qty = new_qty - order.filled_qty
        if new_limit_price is not None:
            order.limit_price = new_limit_price
        if new_stop_price is not None:
            order.stop_price = new_stop_price
        
        order.transition_to(OrderState.AMENDED, "Amendment applied")
        order.transition_to(OrderState.NEW, "Amendment complete")
        
        self._audit_log("ORDER_AMENDED", order, {
            "old": old_values,
            "new": {"qty": order.qty, "limit_price": order.limit_price, 
                   "stop_price": order.stop_price}
        })
        
        return True
    
    def process_fill(self, broker_order_id: str, fill_qty: int, fill_price: float,
                     commission: float = 0.0) -> bool:
        """Process an incoming fill from broker."""
        order_id = self._broker_to_oms.get(broker_order_id)
        if not order_id:
            # Try direct order_id match
            order_id = broker_order_id
        
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"Fill received for unknown order: {broker_order_id}")
            return False
        
        fill = Fill(
            fill_id="",
            order_id=order_id,
            timestamp=datetime.now(),
            qty=fill_qty,
            price=fill_price,
            commission=commission,
            exchange_order_id=broker_order_id
        )
        
        order.add_fill(fill)
        self._audit_log("ORDER_FILLED", order, {
            "fill_qty": fill_qty,
            "fill_price": fill_price,
            "commission": commission
        })
        
        # Notify callbacks
        for callback in self._on_fill_callbacks:
            try:
                callback(order, fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")
        
        return True
    
    def create_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        order_type: OrderType = OrderType.LIMIT,
        strategy_name: str = ""
    ) -> Optional[Dict[str, Order]]:
        """
        Create a bracket order (entry + stop loss + take profit).
        Returns dict with 'entry', 'stop_loss', 'take_profit' orders.
        """
        # Entry order
        entry_order = self.create_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=entry_price,
            strategy_name=strategy_name,
            priority=OrderPriority.HIGH
        )
        
        if not entry_order:
            return None
        
        # Stop loss (opposite side)
        sl_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        sl_order = self.create_order(
            symbol=symbol,
            side=sl_side,
            qty=qty,
            order_type=OrderType.STOP,
            stop_price=stop_loss,
            parent_order_id=entry_order.order_id,
            strategy_name=strategy_name,
            priority=OrderPriority.CRITICAL  # Stop loss is critical
        )
        
        # Take profit (opposite side)
        tp_order = self.create_order(
            symbol=symbol,
            side=sl_side,
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=take_profit,
            parent_order_id=entry_order.order_id,
            strategy_name=strategy_name,
            priority=OrderPriority.NORMAL
        )
        
        if sl_order:
            entry_order.child_order_ids.append(sl_order.order_id)
        if tp_order:
            entry_order.child_order_ids.append(tp_order.order_id)
        
        self._audit_log("BRACKET_CREATED", entry_order, {
            "stop_loss_id": sl_order.order_id if sl_order else None,
            "take_profit_id": tp_order.order_id if tp_order else None
        })
        
        return {
            "entry": entry_order,
            "stop_loss": sl_order,
            "take_profit": tp_order
        }
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        order_ids = self._orders_by_symbol.get(symbol, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active (pending/partial) orders."""
        return [o for o in self._orders.values() if o.is_active]
    
    def get_filled_orders_today(self) -> List[Order]:
        """Get all filled orders from today."""
        today = datetime.now().date()
        return [o for o in self._orders.values() 
                if o.state == OrderState.FILLED and o.created_at.date() == today]
    
    def on_fill(self, callback: Callable):
        """Register callback for fill events."""
        self._on_fill_callbacks.append(callback)
    
    def on_state_change(self, callback: Callable):
        """Register callback for state change events."""
        self._on_state_change_callbacks.append(callback)
    
    def get_status(self) -> dict:
        """Get OMS status summary."""
        active = [o for o in self._orders.values() if o.is_active]
        filled = [o for o in self._orders.values() if o.state == OrderState.FILLED]
        cancelled = [o for o in self._orders.values() if o.state == OrderState.CANCELLED]
        rejected = [o for o in self._orders.values() if o.state == OrderState.REJECTED]
        
        return {
            "total_orders": len(self._orders),
            "orders_today": self._orders_today,
            "active_orders": len(active),
            "filled_orders": len(filled),
            "cancelled_orders": len(cancelled),
            "rejected_orders": len(rejected),
            "queue_size": self._submission_queue.size(),
            "symbols_traded": list(self._orders_by_symbol.keys()),
        }
    
    def export_orders(self, filepath: str = None) -> str:
        """Export all orders to JSON file."""
        if not filepath:
            filepath = f"results/oms_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "export_time": datetime.now().isoformat(),
            "status": self.get_status(),
            "orders": [o.to_dict() for o in self._orders.values()]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Orders exported to {filepath}")
        return filepath
