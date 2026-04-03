"""
Enhanced Order Types - Bracket, stop-limit, trailing stops, and more.

Features:
- Bracket orders (entry + SL + target)
- Stop-limit orders
- Native trailing stops
- OCO (One-Cancels-Other) orders
- Peg orders
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    TRIGGERED = "triggered"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class OrderLeg:
    """Single leg of a complex order."""
    leg_id: str
    order_type: str  # "MARKET", "LIMIT", "STOP", "STOP_LIMIT"
    side: str  # "BUY" or "SELL"
    qty: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    filled_qty: int = 0
    broker_order_id: str = ""
    
    def __post_init__(self):
        if not self.leg_id:
            self.leg_id = f"LEG_{uuid.uuid4().hex[:8]}"


@dataclass
class BracketOrder:
    """
    Bracket order: Entry + Stop Loss + Take Profit.
    All three orders are linked - filling entry activates SL and TP,
    and filling either SL or TP cancels the other.
    """
    bracket_id: str
    symbol: str
    side: str  # "BUY" or "SELL" for entry
    qty: int
    
    # Entry
    entry_type: str = "LIMIT"  # "MARKET" or "LIMIT"
    entry_price: Optional[float] = None
    
    # Stop Loss
    stop_loss_price: float = 0.0
    stop_loss_type: str = "STOP"  # "STOP" or "STOP_LIMIT"
    stop_loss_limit: Optional[float] = None  # For stop-limit
    
    # Take Profit
    take_profit_price: float = 0.0
    take_profit_type: str = "LIMIT"
    
    # State
    status: OrderStatus = OrderStatus.PENDING
    entry_leg: Optional[OrderLeg] = None
    sl_leg: Optional[OrderLeg] = None
    tp_leg: Optional[OrderLeg] = None
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0
    
    def __post_init__(self):
        if not self.bracket_id:
            self.bracket_id = f"BKT_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Create entry leg
        exit_side = "SELL" if self.side == "BUY" else "BUY"
        
        self.entry_leg = OrderLeg(
            leg_id=f"{self.bracket_id}_ENTRY",
            order_type=self.entry_type,
            side=self.side,
            qty=self.qty,
            limit_price=self.entry_price
        )
        
        # Create SL leg
        self.sl_leg = OrderLeg(
            leg_id=f"{self.bracket_id}_SL",
            order_type=self.stop_loss_type,
            side=exit_side,
            qty=self.qty,
            stop_price=self.stop_loss_price,
            limit_price=self.stop_loss_limit
        )
        
        # Create TP leg
        self.tp_leg = OrderLeg(
            leg_id=f"{self.bracket_id}_TP",
            order_type=self.take_profit_type,
            side=exit_side,
            qty=self.qty,
            limit_price=self.take_profit_price
        )


@dataclass
class TrailingStopOrder:
    """
    Trailing stop order that follows the price.
    
    For LONG positions:
    - Stop trails below market by trail_amount or trail_percent
    - Stop only moves up, never down
    
    For SHORT positions:
    - Stop trails above market by trail_amount or trail_percent
    - Stop only moves down, never up
    """
    order_id: str
    symbol: str
    side: str  # "SELL" for long positions, "BUY" for short
    qty: int
    
    # Trailing parameters (use one)
    trail_amount: Optional[float] = None    # Fixed rupee amount
    trail_percent: Optional[float] = None   # Percentage of price
    
    # Activation
    activation_price: Optional[float] = None  # Start trailing when this price hit
    
    # Current state
    status: OrderStatus = OrderStatus.PENDING
    current_stop_price: float = 0.0
    highest_price: float = 0.0   # For sell trailing stop
    lowest_price: float = float('inf')  # For buy trailing stop
    
    # Fills
    fill_price: float = 0.0
    filled_qty: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"TRAIL_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        if not self.trail_amount and not self.trail_percent:
            raise ValueError("Must specify either trail_amount or trail_percent")
    
    def update_stop(self, current_price: float) -> Tuple[bool, float]:
        """
        Update the trailing stop based on current price.
        Returns (triggered, new_stop_price).
        """
        if self.status not in [OrderStatus.ACTIVE, OrderStatus.PENDING]:
            return False, self.current_stop_price
        
        # Check activation
        if self.status == OrderStatus.PENDING:
            if self.activation_price:
                if self.side == "SELL" and current_price >= self.activation_price:
                    self.status = OrderStatus.ACTIVE
                    self.highest_price = current_price
                elif self.side == "BUY" and current_price <= self.activation_price:
                    self.status = OrderStatus.ACTIVE
                    self.lowest_price = current_price
                else:
                    return False, 0.0
            else:
                # No activation price - start immediately
                self.status = OrderStatus.ACTIVE
                if self.side == "SELL":
                    self.highest_price = current_price
                else:
                    self.lowest_price = current_price
        
        # Calculate trail distance
        if self.trail_amount:
            trail_distance = self.trail_amount
        else:
            trail_distance = current_price * (self.trail_percent / 100)
        
        # Update stop based on side
        if self.side == "SELL":
            # Trailing stop for long position
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.current_stop_price = current_price - trail_distance
            
            # Check if triggered
            if current_price <= self.current_stop_price:
                self.status = OrderStatus.TRIGGERED
                return True, self.current_stop_price
        else:
            # Trailing stop for short position
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                self.current_stop_price = current_price + trail_distance
            
            # Check if triggered
            if current_price >= self.current_stop_price:
                self.status = OrderStatus.TRIGGERED
                return True, self.current_stop_price
        
        return False, self.current_stop_price


@dataclass
class OCOOrder:
    """
    One-Cancels-Other order.
    Two orders linked together - when one fills, the other is cancelled.
    Common use: Stop loss and take profit simultaneously.
    """
    oco_id: str
    symbol: str
    
    order_a: OrderLeg
    order_b: OrderLeg
    
    status: OrderStatus = OrderStatus.PENDING
    filled_order: str = ""  # "A" or "B"
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.oco_id:
            self.oco_id = f"OCO_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"


@dataclass
class StopLimitOrder:
    """
    Stop-Limit order: Triggered at stop price, then becomes limit order.
    Provides price protection but may not fill.
    """
    order_id: str
    symbol: str
    side: str
    qty: int
    stop_price: float      # Trigger price
    limit_price: float     # Limit price after trigger
    
    status: OrderStatus = OrderStatus.PENDING
    triggered_at: Optional[datetime] = None
    fill_price: float = 0.0
    filled_qty: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"STPLMT_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    def check_trigger(self, current_price: float) -> bool:
        """Check if stop price is triggered."""
        if self.status != OrderStatus.PENDING:
            return False
        
        if self.side == "BUY":
            # Buy stop: trigger when price >= stop_price
            triggered = current_price >= self.stop_price
        else:
            # Sell stop: trigger when price <= stop_price
            triggered = current_price <= self.stop_price
        
        if triggered:
            self.status = OrderStatus.TRIGGERED
            self.triggered_at = datetime.now()
        
        return triggered


class ComplexOrderManager:
    """
    Manages complex order types and their lifecycle.
    """
    
    def __init__(
        self,
        broker_adapter=None,
        price_fetcher: Callable = None,
        update_interval_ms: int = 500
    ):
        self.broker_adapter = broker_adapter
        self.price_fetcher = price_fetcher
        self.update_interval_ms = update_interval_ms
        
        # Order storage
        self._brackets: Dict[str, BracketOrder] = {}
        self._trailing_stops: Dict[str, TrailingStopOrder] = {}
        self._oco_orders: Dict[str, OCOOrder] = {}
        self._stop_limits: Dict[str, StopLimitOrder] = {}
        
        # Callbacks
        self._on_fill_callbacks: List[Callable] = []
        self._on_trigger_callbacks: List[Callable] = []
        
        # Monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def create_bracket(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float = None,
        stop_loss_price: float = None,
        take_profit_price: float = None,
        entry_type: str = "LIMIT",
        stop_loss_type: str = "STOP"
    ) -> BracketOrder:
        """
        Create a bracket order.
        
        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL" for entry direction
            qty: Quantity
            entry_price: Entry limit price (None for market)
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            entry_type: "MARKET" or "LIMIT"
            stop_loss_type: "STOP" or "STOP_LIMIT"
        
        Returns:
            BracketOrder object
        """
        bracket = BracketOrder(
            bracket_id="",
            symbol=symbol,
            side=side,
            qty=qty,
            entry_type=entry_type,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            stop_loss_type=stop_loss_type,
            take_profit_price=take_profit_price
        )
        
        with self._lock:
            self._brackets[bracket.bracket_id] = bracket
        
        logger.info(f"Created bracket order {bracket.bracket_id}: "
                   f"{side} {qty} {symbol} @ {entry_price}, "
                   f"SL: {stop_loss_price}, TP: {take_profit_price}")
        
        return bracket
    
    def create_trailing_stop(
        self,
        symbol: str,
        side: str,
        qty: int,
        trail_amount: float = None,
        trail_percent: float = None,
        activation_price: float = None
    ) -> TrailingStopOrder:
        """
        Create a trailing stop order.
        
        Args:
            symbol: Stock symbol
            side: "SELL" for long positions, "BUY" for shorts
            qty: Quantity
            trail_amount: Fixed trailing distance in rupees
            trail_percent: Trailing distance as percentage
            activation_price: Price at which trailing begins
        
        Returns:
            TrailingStopOrder object
        """
        order = TrailingStopOrder(
            order_id="",
            symbol=symbol,
            side=side,
            qty=qty,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            activation_price=activation_price
        )
        
        with self._lock:
            self._trailing_stops[order.order_id] = order
        
        trail_desc = f"₹{trail_amount}" if trail_amount else f"{trail_percent}%"
        logger.info(f"Created trailing stop {order.order_id}: "
                   f"{side} {qty} {symbol}, trail: {trail_desc}")
        
        return order
    
    def create_oco(
        self,
        symbol: str,
        order_a_type: str,
        order_a_side: str,
        order_a_qty: int,
        order_a_price: float,
        order_b_type: str,
        order_b_side: str,
        order_b_qty: int,
        order_b_price: float
    ) -> OCOOrder:
        """
        Create an OCO (One-Cancels-Other) order.
        
        Typically used with stop loss and take profit.
        """
        order_a = OrderLeg(
            leg_id="",
            order_type=order_a_type,
            side=order_a_side,
            qty=order_a_qty,
            limit_price=order_a_price if order_a_type == "LIMIT" else None,
            stop_price=order_a_price if order_a_type == "STOP" else None
        )
        
        order_b = OrderLeg(
            leg_id="",
            order_type=order_b_type,
            side=order_b_side,
            qty=order_b_qty,
            limit_price=order_b_price if order_b_type == "LIMIT" else None,
            stop_price=order_b_price if order_b_type == "STOP" else None
        )
        
        oco = OCOOrder(
            oco_id="",
            symbol=symbol,
            order_a=order_a,
            order_b=order_b
        )
        
        with self._lock:
            self._oco_orders[oco.oco_id] = oco
        
        logger.info(f"Created OCO order {oco.oco_id}")
        return oco
    
    def create_stop_limit(
        self,
        symbol: str,
        side: str,
        qty: int,
        stop_price: float,
        limit_price: float
    ) -> StopLimitOrder:
        """Create a stop-limit order."""
        order = StopLimitOrder(
            order_id="",
            symbol=symbol,
            side=side,
            qty=qty,
            stop_price=stop_price,
            limit_price=limit_price
        )
        
        with self._lock:
            self._stop_limits[order.order_id] = order
        
        logger.info(f"Created stop-limit {order.order_id}: "
                   f"{side} {qty} {symbol}, stop: {stop_price}, limit: {limit_price}")
        
        return order
    
    def submit_bracket(self, bracket_id: str) -> bool:
        """Submit bracket order to broker."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            logger.error(f"Bracket not found: {bracket_id}")
            return False
        
        # Submit entry order
        if self.broker_adapter:
            result = self.broker_adapter.place_order(
                symbol=bracket.symbol,
                side=bracket.entry_leg.side,
                qty=bracket.entry_leg.qty,
                order_type=bracket.entry_leg.order_type,
                price=bracket.entry_leg.limit_price or 0.0
            )
            
            if result and result.status in ["COMPLETE", "PLACED"]:
                bracket.entry_leg.broker_order_id = result.order_id
                bracket.entry_leg.status = OrderStatus.ACTIVE
                bracket.status = OrderStatus.ACTIVE
                
                if result.status == "COMPLETE":
                    self._handle_bracket_entry_fill(bracket, result.fill_price)
                
                return True
            else:
                bracket.status = OrderStatus.REJECTED
                return False
        else:
            # Paper mode: simulate immediate fill
            bracket.entry_leg.status = OrderStatus.FILLED
            bracket.status = OrderStatus.ACTIVE
            if self.price_fetcher:
                price = self.price_fetcher(bracket.symbol)
                self._handle_bracket_entry_fill(bracket, price)
            return True
    
    def _handle_bracket_entry_fill(self, bracket: BracketOrder, fill_price: float):
        """Handle bracket entry fill - activate SL and TP."""
        bracket.entry_leg.fill_price = fill_price
        bracket.entry_leg.filled_qty = bracket.qty
        bracket.entry_leg.status = OrderStatus.FILLED
        bracket.filled_at = datetime.now()
        
        # Activate SL and TP legs
        bracket.sl_leg.status = OrderStatus.ACTIVE
        bracket.tp_leg.status = OrderStatus.ACTIVE
        
        if self.broker_adapter:
            # Place SL order
            sl_result = self.broker_adapter.place_order(
                symbol=bracket.symbol,
                side=bracket.sl_leg.side,
                qty=bracket.sl_leg.qty,
                order_type="SL",
                trigger_price=bracket.sl_leg.stop_price,
                price=bracket.sl_leg.limit_price or 0.0
            )
            if sl_result:
                bracket.sl_leg.broker_order_id = sl_result.order_id
            
            # Place TP order
            tp_result = self.broker_adapter.place_order(
                symbol=bracket.symbol,
                side=bracket.tp_leg.side,
                qty=bracket.tp_leg.qty,
                order_type="LIMIT",
                price=bracket.tp_leg.limit_price
            )
            if tp_result:
                bracket.tp_leg.broker_order_id = tp_result.order_id
        
        logger.info(f"Bracket {bracket.bracket_id} entry filled @ {fill_price}, "
                   f"SL and TP activated")
    
    def start_monitoring(self):
        """Start price monitoring for trailing stops and triggers."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Complex order monitoring started")
    
    def stop_monitoring(self):
        """Stop price monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Complex order monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_all_orders()
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
            
            time.sleep(self.update_interval_ms / 1000)
    
    def _check_all_orders(self):
        """Check all active orders for triggers."""
        if not self.price_fetcher:
            return
        
        # Get all symbols we're tracking
        symbols = set()
        with self._lock:
            for b in self._brackets.values():
                if b.status == OrderStatus.ACTIVE:
                    symbols.add(b.symbol)
            for t in self._trailing_stops.values():
                if t.status in [OrderStatus.PENDING, OrderStatus.ACTIVE]:
                    symbols.add(t.symbol)
            for s in self._stop_limits.values():
                if s.status == OrderStatus.PENDING:
                    symbols.add(s.symbol)
        
        # Get current prices
        prices = {}
        for symbol in symbols:
            try:
                prices[symbol] = self.price_fetcher(symbol)
            except Exception:
                pass
        
        # Check trailing stops
        with self._lock:
            for order_id, order in list(self._trailing_stops.items()):
                if order.symbol in prices:
                    triggered, stop_price = order.update_stop(prices[order.symbol])
                    if triggered:
                        self._handle_trailing_trigger(order, prices[order.symbol])
        
        # Check stop-limits
        with self._lock:
            for order_id, order in list(self._stop_limits.items()):
                if order.symbol in prices and order.status == OrderStatus.PENDING:
                    if order.check_trigger(prices[order.symbol]):
                        self._handle_stop_limit_trigger(order)
        
        # Check brackets for SL/TP hits
        with self._lock:
            for bracket_id, bracket in list(self._brackets.items()):
                if bracket.status == OrderStatus.ACTIVE and bracket.symbol in prices:
                    self._check_bracket_exit(bracket, prices[bracket.symbol])
    
    def _handle_trailing_trigger(self, order: TrailingStopOrder, current_price: float):
        """Handle trailing stop trigger."""
        logger.info(f"Trailing stop {order.order_id} triggered @ {order.current_stop_price}")
        
        if self.broker_adapter:
            result = self.broker_adapter.place_order(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                order_type="MARKET"
            )
            if result:
                order.fill_price = result.fill_price
                order.filled_qty = order.qty
                order.status = OrderStatus.FILLED
        else:
            order.fill_price = current_price
            order.filled_qty = order.qty
            order.status = OrderStatus.FILLED
        
        for callback in self._on_trigger_callbacks:
            try:
                callback("TRAILING_STOP", order)
            except Exception:
                pass
    
    def _handle_stop_limit_trigger(self, order: StopLimitOrder):
        """Handle stop-limit trigger - place limit order."""
        logger.info(f"Stop-limit {order.order_id} triggered, placing limit @ {order.limit_price}")
        
        if self.broker_adapter:
            result = self.broker_adapter.place_order(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                order_type="LIMIT",
                price=order.limit_price
            )
            if result:
                order.status = OrderStatus.ACTIVE
        else:
            order.status = OrderStatus.ACTIVE
        
        for callback in self._on_trigger_callbacks:
            try:
                callback("STOP_LIMIT", order)
            except Exception:
                pass
    
    def _check_bracket_exit(self, bracket: BracketOrder, current_price: float):
        """Check if bracket SL or TP is hit."""
        entry_price = bracket.entry_leg.fill_price
        
        if bracket.side == "BUY":
            # Long position
            if current_price <= bracket.stop_loss_price:
                self._handle_bracket_exit(bracket, "SL", current_price)
            elif current_price >= bracket.take_profit_price:
                self._handle_bracket_exit(bracket, "TP", current_price)
        else:
            # Short position
            if current_price >= bracket.stop_loss_price:
                self._handle_bracket_exit(bracket, "SL", current_price)
            elif current_price <= bracket.take_profit_price:
                self._handle_bracket_exit(bracket, "TP", current_price)
    
    def _handle_bracket_exit(self, bracket: BracketOrder, exit_type: str, price: float):
        """Handle bracket exit (SL or TP hit)."""
        entry_price = bracket.entry_leg.fill_price
        
        if exit_type == "SL":
            bracket.sl_leg.fill_price = price
            bracket.sl_leg.filled_qty = bracket.qty
            bracket.sl_leg.status = OrderStatus.FILLED
            bracket.tp_leg.status = OrderStatus.CANCELLED
            bracket.exit_reason = "stop_loss"
        else:
            bracket.tp_leg.fill_price = price
            bracket.tp_leg.filled_qty = bracket.qty
            bracket.tp_leg.status = OrderStatus.FILLED
            bracket.sl_leg.status = OrderStatus.CANCELLED
            bracket.exit_reason = "take_profit"
        
        # Calculate P&L
        if bracket.side == "BUY":
            bracket.pnl = (price - entry_price) * bracket.qty
        else:
            bracket.pnl = (entry_price - price) * bracket.qty
        
        bracket.status = OrderStatus.FILLED
        
        logger.info(f"Bracket {bracket.bracket_id} exited via {exit_type} @ {price}, "
                   f"P&L: ₹{bracket.pnl:,.2f}")
        
        for callback in self._on_fill_callbacks:
            try:
                callback(bracket)
            except Exception:
                pass
    
    def cancel_bracket(self, bracket_id: str) -> bool:
        """Cancel a bracket order."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            return False
        
        if bracket.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        # Cancel all active legs
        for leg in [bracket.entry_leg, bracket.sl_leg, bracket.tp_leg]:
            if leg.status == OrderStatus.ACTIVE and leg.broker_order_id:
                if self.broker_adapter:
                    try:
                        self.broker_adapter.cancel_order(leg.broker_order_id)
                    except Exception:
                        pass
            leg.status = OrderStatus.CANCELLED
        
        bracket.status = OrderStatus.CANCELLED
        logger.info(f"Bracket {bracket_id} cancelled")
        return True
    
    def cancel_trailing_stop(self, order_id: str) -> bool:
        """Cancel a trailing stop order."""
        order = self._trailing_stops.get(order_id)
        if not order:
            return False
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Trailing stop {order_id} cancelled")
        return True
    
    def get_active_brackets(self) -> List[BracketOrder]:
        """Get all active bracket orders."""
        with self._lock:
            return [b for b in self._brackets.values() if b.status == OrderStatus.ACTIVE]
    
    def get_active_trailing_stops(self) -> List[TrailingStopOrder]:
        """Get all active trailing stop orders."""
        with self._lock:
            return [t for t in self._trailing_stops.values() 
                   if t.status in [OrderStatus.PENDING, OrderStatus.ACTIVE]]
    
    def on_fill(self, callback: Callable):
        """Register callback for fill events."""
        self._on_fill_callbacks.append(callback)
    
    def on_trigger(self, callback: Callable):
        """Register callback for trigger events."""
        self._on_trigger_callbacks.append(callback)
    
    def get_status(self) -> dict:
        """Get status of all complex orders."""
        with self._lock:
            return {
                "brackets": {
                    "total": len(self._brackets),
                    "active": len([b for b in self._brackets.values() 
                                  if b.status == OrderStatus.ACTIVE]),
                    "filled": len([b for b in self._brackets.values() 
                                  if b.status == OrderStatus.FILLED])
                },
                "trailing_stops": {
                    "total": len(self._trailing_stops),
                    "active": len([t for t in self._trailing_stops.values() 
                                  if t.status in [OrderStatus.PENDING, OrderStatus.ACTIVE]])
                },
                "oco_orders": {
                    "total": len(self._oco_orders),
                    "active": len([o for o in self._oco_orders.values() 
                                  if o.status == OrderStatus.ACTIVE])
                },
                "stop_limits": {
                    "total": len(self._stop_limits),
                    "pending": len([s for s in self._stop_limits.values() 
                                   if s.status == OrderStatus.PENDING])
                }
            }
