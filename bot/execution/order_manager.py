"""
Order Manager
=============
Central order management with tracking, retries, and idempotency.

Features:
1. Order tracking with unique IDs
2. Retry logic with exponential backoff
3. Idempotency using order cache
4. Order state machine
5. Paper/Live mode abstraction
"""

import uuid
import time
import logging
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import OrderedDict

from ..core.mode import TradingMode, ModeManager
from ..strategies.signal import Signal, SignalType


logger = logging.getLogger("bot.execution.order_manager")


class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING = "pending"          # Order created, not yet submitted
    SUBMITTED = "submitted"      # Sent to broker
    OPEN = "open"                # Accepted by exchange
    PARTIAL = "partial"          # Partially filled
    FILLED = "filled"            # Completely filled
    CANCELLED = "cancelled"      # Cancelled by user/system
    REJECTED = "rejected"        # Rejected by broker/exchange
    EXPIRED = "expired"          # Validity expired
    FAILED = "failed"            # Failed due to error


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"          # Stop-loss
    SL_M = "SL-M"      # Stop-loss market


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    """Product type"""
    INTRADAY = "INTRADAY"
    CNC = "CNC"          # Cash & Carry (delivery)
    MARGIN = "MARGIN"


@dataclass
class Order:
    """Order data structure"""
    # Identification
    order_id: str = ""                    # Internal unique ID
    broker_order_id: str = ""             # Broker-assigned ID
    
    # Symbol
    symbol: str = ""
    exchange: str = "NSE"
    token: str = ""
    
    # Order details
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.INTRADAY
    quantity: int = 0
    price: float = 0.0
    trigger_price: float = 0.0
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    
    # Timestamps
    created_at: str = ""
    submitted_at: str = ""
    filled_at: str = ""
    
    # Retry tracking
    attempts: int = 0
    last_error: str = ""
    
    # Metadata
    strategy: str = ""
    signal_id: str = ""
    parent_order_id: str = ""     # For linked orders (SL/target)
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state"""
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        }
    
    @property
    def is_active(self) -> bool:
        """Check if order is active"""
        return self.status in {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.OPEN,
            OrderStatus.PARTIAL,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'broker_order_id': self.broker_order_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'product': self.product.value,
            'quantity': self.quantity,
            'price': self.price,
            'trigger_price': self.trigger_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'created_at': self.created_at,
            'attempts': self.attempts,
            'strategy': self.strategy,
        }


class LRUCache:
    """Simple LRU cache for idempotency"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def has(self, key: str) -> bool:
        return key in self.cache


class OrderManager:
    """
    Central order management system.
    
    Handles:
    - Order creation from signals
    - Order submission with retries
    - Order tracking and updates
    - Idempotency to prevent duplicates
    - Paper/Live mode switching
    
    Usage:
        manager = OrderManager(
            mode_manager=mode_manager,
            executor=paper_executor,  # or live_executor
        )
        
        order = manager.create_order(signal)
        result = manager.submit_order(order)
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAYS = [0.5, 1.0, 2.0]  # Exponential backoff
    
    def __init__(
        self,
        mode_manager: ModeManager,
        paper_executor: Optional[Callable[[Order], Order]] = None,
        live_executor: Optional[Callable[[Order], Order]] = None,
        on_order_update: Optional[Callable[[Order], None]] = None,
    ):
        """
        Initialize order manager.
        
        Args:
            mode_manager: Mode manager for paper/live switching
            paper_executor: Executor for paper mode
            live_executor: Executor for live mode
            on_order_update: Callback for order status changes
        """
        self.mode_manager = mode_manager
        self.paper_executor = paper_executor
        self.live_executor = live_executor
        self.on_order_update = on_order_update
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        
        # Idempotency cache
        self.idempotency_cache = LRUCache(max_size=1000)
        
        # Statistics
        self.stats = {
            'orders_created': 0,
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_failed': 0,
            'retries': 0,
        }
    
    @property
    def current_executor(self) -> Optional[Callable[[Order], Order]]:
        """Get executor based on current mode"""
        if self.mode_manager.current_mode == TradingMode.LIVE:
            return self.live_executor
        return self.paper_executor
    
    def create_order_from_signal(self, signal: Signal) -> Optional[Order]:
        """
        Create an order from a trading signal.
        
        Args:
            signal: Trading signal with type, symbol, price, etc.
            
        Returns:
            Order object or None if signal is invalid
        """
        # Determine order side
        if signal.signal_type in {SignalType.BUY, SignalType.EXIT_SHORT}:
            side = OrderSide.BUY
        elif signal.signal_type in {SignalType.SELL, SignalType.EXIT_LONG}:
            side = OrderSide.SELL
        else:
            logger.warning(f"Cannot create order for signal type: {signal.signal_type}")
            return None
        
        # Get quantity from signal metadata
        quantity = signal.metadata.get('quantity', 0)
        if quantity <= 0:
            logger.warning(f"Invalid quantity in signal: {quantity}")
            return None
        
        order = Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,  # Default to market for speed
            product=ProductType.INTRADAY,
            quantity=quantity,
            price=signal.price,
            strategy=signal.metadata.get('strategy', ''),
            signal_id=signal.signal_id,
        )
        
        self.orders[order.order_id] = order
        self.stats['orders_created'] += 1
        
        logger.info(f"Created order {order.order_id}: {order.side.value} {order.quantity} {order.symbol}")
        
        return order
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
        trigger_price: float = 0.0,
        strategy: str = "",
    ) -> Order:
        """
        Create an order directly.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Number of shares
            order_type: MARKET, LIMIT, SL, SL-M
            price: Limit price (for LIMIT orders)
            trigger_price: Trigger price (for SL orders)
            strategy: Strategy name for tracking
            
        Returns:
            Order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            product=ProductType.INTRADAY,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            strategy=strategy,
        )
        
        self.orders[order.order_id] = order
        self.stats['orders_created'] += 1
        
        return order
    
    def submit_order(
        self,
        order: Order,
        idempotency_key: Optional[str] = None,
    ) -> Order:
        """
        Submit an order with retry logic.
        
        Args:
            order: Order to submit
            idempotency_key: Optional key for deduplication
            
        Returns:
            Updated order with status
        """
        # Check idempotency
        if idempotency_key:
            cached = self.idempotency_cache.get(idempotency_key)
            if cached:
                logger.info(f"Duplicate order detected: {idempotency_key}")
                return cached
        
        executor = self.current_executor
        if not executor:
            logger.error("No executor available")
            order.status = OrderStatus.FAILED
            order.last_error = "No executor configured"
            return order
        
        # Retry loop
        last_error = ""
        for attempt in range(self.MAX_RETRIES):
            order.attempts = attempt + 1
            
            try:
                order.submitted_at = datetime.now().isoformat()
                order.status = OrderStatus.SUBMITTED
                
                # Execute order
                result = executor(order)
                
                # Update order from result
                order.status = result.status
                order.broker_order_id = result.broker_order_id
                order.filled_quantity = result.filled_quantity
                order.average_price = result.average_price
                
                if result.status == OrderStatus.FILLED:
                    order.filled_at = datetime.now().isoformat()
                    self.stats['orders_filled'] += 1
                    logger.info(f"Order filled: {order.order_id} @ ₹{order.average_price}")
                    break
                elif result.status in {OrderStatus.REJECTED, OrderStatus.CANCELLED}:
                    order.last_error = result.last_error
                    logger.warning(f"Order {result.status.value}: {order.last_error}")
                    break
                
            except Exception as e:
                last_error = str(e)
                order.last_error = last_error
                logger.error(f"Order attempt {attempt + 1} failed: {e}")
                
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAYS[attempt]
                    self.stats['retries'] += 1
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
        
        # Final status check
        if order.attempts >= self.MAX_RETRIES and order.status != OrderStatus.FILLED:
            order.status = OrderStatus.FAILED
            self.stats['orders_failed'] += 1
            logger.error(f"Order failed after {self.MAX_RETRIES} attempts: {last_error}")
        
        # Update tracking
        if order.is_active:
            self.active_orders[order.order_id] = order
        else:
            self.active_orders.pop(order.order_id, None)
        
        # Cache for idempotency
        if idempotency_key:
            self.idempotency_cache.set(idempotency_key, order)
        
        self.stats['orders_submitted'] += 1
        
        # Callback
        if self.on_order_update:
            try:
                self.on_order_update(order)
            except Exception as e:
                logger.error(f"on_order_update callback failed: {e}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled, False otherwise
        """
        order = self.orders.get(order_id)
        if not order:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        if not order.is_active:
            logger.warning(f"Cannot cancel terminal order: {order.status}")
            return False
        
        order.status = OrderStatus.CANCELLED
        self.active_orders.pop(order_id, None)
        
        if self.on_order_update:
            self.on_order_update(order)
        
        logger.info(f"Order cancelled: {order_id}")
        return True
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all active orders.
        
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        for order_id in list(self.active_orders.keys()):
            if self.cancel_order(order_id):
                cancelled += 1
        
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        return [o for o in self.orders.values() if o.symbol == symbol]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get order statistics"""
        return self.stats.copy()
