"""
Order Validator
===============
Validate orders before execution.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from datetime import datetime

from ..core.constants import Limits, MarketHours
from ..utils.time_utils import get_ist_now, is_trading_hours, can_open_new_trades


class ValidationError(Exception):
    """Order validation failed"""
    pass


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    INTRADAY = "INTRADAY"  # MIS
    CNC = "CNC"            # Delivery
    NRML = "NRML"          # Normal (F&O)


@dataclass
class Order:
    """Order representation"""
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.INTRADAY
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    tag: str = ""


@dataclass
class ValidationResult:
    """Result of order validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __str__(self) -> str:
        if self.is_valid:
            if self.warnings:
                return f"✅ Valid (with warnings: {', '.join(self.warnings)})"
            return "✅ Valid"
        return f"❌ Invalid: {', '.join(self.errors)}"


class OrderValidator:
    """
    Validate orders against various rules before execution.
    
    Checks:
    - Market hours
    - Symbol validity
    - Price reasonableness
    - Quantity limits
    - Risk limits (via RiskManager)
    """
    
    def __init__(
        self,
        capital: float,
        max_order_value: float = Limits.MAX_ORDER_VALUE,
        min_order_qty: int = Limits.MIN_ORDER_QTY,
        max_slippage_pct: float = Limits.MAX_SLIPPAGE_PCT,
        allowed_symbols: Optional[List[str]] = None,
    ):
        """
        Initialize order validator.
        
        Args:
            capital: Current trading capital
            max_order_value: Maximum value per order
            min_order_qty: Minimum quantity per order
            max_slippage_pct: Maximum acceptable slippage
            allowed_symbols: Whitelist of tradeable symbols (None = all allowed)
        """
        self.capital = capital
        self.max_order_value = max_order_value
        self.min_order_qty = min_order_qty
        self.max_slippage_pct = max_slippage_pct
        self.allowed_symbols = allowed_symbols
    
    def update_capital(self, new_capital: float):
        """Update capital for validation"""
        self.capital = new_capital
    
    def validate(
        self,
        order: Order,
        current_price: Optional[float] = None,
        check_market_hours: bool = True,
    ) -> ValidationResult:
        """
        Validate an order.
        
        Args:
            order: Order to validate
            current_price: Current market price for slippage check
            check_market_hours: Whether to check if market is open
        
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check market hours
        if check_market_hours:
            now = get_ist_now()
            
            if not is_trading_hours():
                errors.append(f"Outside trading hours (current: {now.strftime('%H:%M')})")
            
            if order.side == OrderSide.BUY and not can_open_new_trades():
                errors.append("No new BUY orders after 2:30 PM")
        
        # Check symbol
        if not order.symbol or len(order.symbol) < 2:
            errors.append("Invalid symbol")
        
        if self.allowed_symbols and order.symbol not in self.allowed_symbols:
            errors.append(f"Symbol {order.symbol} not in allowed list")
        
        # Check quantity
        if order.quantity < self.min_order_qty:
            errors.append(f"Quantity {order.quantity} below minimum {self.min_order_qty}")
        
        if order.quantity > 10000:  # Sanity check
            warnings.append(f"Large quantity: {order.quantity}")
        
        # Check price
        if order.price <= 0:
            errors.append(f"Invalid price: {order.price}")
        
        if order.price < 1:
            warnings.append(f"Low price: ₹{order.price}")
        
        if order.price > 100000:
            warnings.append(f"High price: ₹{order.price}")
        
        # Check order value
        order_value = order.quantity * order.price
        
        if order_value > self.max_order_value:
            errors.append(
                f"Order value ₹{order_value:,.0f} exceeds max ₹{self.max_order_value:,.0f}"
            )
        
        if order_value > self.capital * 0.5:
            warnings.append(f"Order value is {order_value/self.capital*100:.1f}% of capital")
        
        # Check slippage (if current price provided)
        if current_price and current_price > 0:
            slippage_pct = abs(order.price - current_price) / current_price
            
            if slippage_pct > self.max_slippage_pct:
                errors.append(
                    f"Price deviation {slippage_pct*100:.2f}% exceeds max {self.max_slippage_pct*100:.2f}%"
                )
            elif slippage_pct > self.max_slippage_pct / 2:
                warnings.append(f"Price deviation: {slippage_pct*100:.2f}%")
        
        # Check stop loss order specifics
        if order.order_type in (OrderType.SL, OrderType.SL_M):
            if order.trigger_price is None:
                errors.append("Stop loss order requires trigger price")
            elif order.side == OrderSide.BUY and order.trigger_price < order.price:
                warnings.append("Buy SL trigger below limit price")
            elif order.side == OrderSide.SELL and order.trigger_price > order.price:
                warnings.append("Sell SL trigger above limit price")
        
        # Check limit order price reasonableness
        if order.order_type == OrderType.LIMIT and current_price:
            deviation = abs(order.price - current_price) / current_price
            if deviation > 0.05:  # 5%
                warnings.append(f"Limit price {deviation*100:.1f}% away from market")
        
        # Check stop loss and target consistency
        if order.stop_loss and order.target:
            if order.side == OrderSide.BUY:
                if order.stop_loss >= order.price:
                    errors.append("Buy stop loss must be below entry price")
                if order.target <= order.price:
                    errors.append("Buy target must be above entry price")
            else:
                if order.stop_loss <= order.price:
                    errors.append("Sell stop loss must be above entry price")
                if order.target >= order.price:
                    errors.append("Sell target must be below entry price")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def validate_batch(self, orders: List[Order]) -> List[ValidationResult]:
        """Validate multiple orders"""
        return [self.validate(order) for order in orders]
    
    def pre_execution_check(
        self,
        order: Order,
        current_price: float,
        available_margin: float,
    ) -> ValidationResult:
        """
        Final validation just before execution.
        
        More strict checks for real-time execution.
        
        Args:
            order: Order to validate
            current_price: Latest market price
            available_margin: Available margin for trading
        
        Returns:
            ValidationResult
        """
        result = self.validate(order, current_price, check_market_hours=True)
        
        errors = list(result.errors)
        warnings = list(result.warnings)
        
        # Check available margin
        order_value = order.quantity * order.price
        margin_required = order_value * 0.20  # ~20% margin for intraday
        
        if order.product == ProductType.INTRADAY:
            if margin_required > available_margin:
                errors.append(
                    f"Insufficient margin: need ₹{margin_required:,.0f}, have ₹{available_margin:,.0f}"
                )
        else:
            if order_value > available_margin:
                errors.append(
                    f"Insufficient funds: need ₹{order_value:,.0f}, have ₹{available_margin:,.0f}"
                )
        
        # Stricter slippage check
        slippage_pct = abs(order.price - current_price) / current_price
        if slippage_pct > 0.002:  # 0.2%
            warnings.append(f"Pre-execution slippage: {slippage_pct*100:.3f}%")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


# ============================================================================
# UNIT TESTS
# ============================================================================

def _test_validator():
    """Test order validator"""
    print("Testing order validator...")
    
    validator = OrderValidator(
        capital=100_000,
        max_order_value=50_000,
        allowed_symbols=["RELIANCE", "TCS", "INFY"],
    )
    
    # Valid order
    order1 = Order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        quantity=10,
        price=2500.0,
        order_type=OrderType.MARKET,
    )
    
    result1 = validator.validate(order1, current_price=2498.0, check_market_hours=False)
    print(f"Order 1: {result1}")
    assert result1.is_valid, "Valid order should pass"
    
    # Invalid order - exceeds max value
    order2 = Order(
        symbol="TCS",
        side=OrderSide.BUY,
        quantity=100,
        price=3500.0,  # 350,000 order value
        order_type=OrderType.MARKET,
    )
    
    result2 = validator.validate(order2, check_market_hours=False)
    print(f"Order 2: {result2}")
    assert not result2.is_valid, "Order exceeding max value should fail"
    
    # Invalid symbol
    order3 = Order(
        symbol="UNKNOWN",
        side=OrderSide.BUY,
        quantity=10,
        price=100.0,
        order_type=OrderType.MARKET,
    )
    
    result3 = validator.validate(order3, check_market_hours=False)
    print(f"Order 3: {result3}")
    assert not result3.is_valid, "Unknown symbol should fail"
    
    # Order with warnings
    order4 = Order(
        symbol="INFY",
        side=OrderSide.BUY,
        quantity=100,
        price=1500.0,  # 150,000 value but under capital
        order_type=OrderType.LIMIT,
    )
    
    validator_lenient = OrderValidator(capital=200_000, max_order_value=200_000)
    result4 = validator_lenient.validate(order4, current_price=1400.0, check_market_hours=False)
    print(f"Order 4: {result4}")
    assert result4.is_valid, "Order should be valid"
    assert len(result4.warnings) > 0, "Should have warnings about price deviation"
    
    print("\n✅ All validator tests passed!")


if __name__ == "__main__":
    _test_validator()
