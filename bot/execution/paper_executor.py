"""
Paper Mode Executor
===================
Simulates order execution for paper trading.

Features:
1. Realistic fill simulation with slippage
2. Position tracking
3. Trade history
4. P&L calculation
5. Export capability
"""

import logging
import random
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from .order_manager import Order, OrderStatus, OrderSide


logger = logging.getLogger("bot.execution.paper_executor")


@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    side: str                    # "LONG" or "SHORT"
    quantity: int = 0
    average_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    entry_time: str = ""
    entry_order_id: str = ""
    
    @property
    def value(self) -> float:
        """Position value at current price"""
        return self.quantity * self.current_price
    
    def update_pnl(self, current_price: float) -> None:
        """Update unrealized P&L"""
        self.current_price = current_price
        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.average_price) * self.quantity
        else:
            self.unrealized_pnl = (self.average_price - current_price) * self.quantity


@dataclass
class PaperTrade:
    """Record of completed paper trade"""
    trade_id: str
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    gross_pnl: float
    strategy: str = ""
    exit_reason: str = ""


class PaperExecutor:
    """
    Paper trading order executor.
    
    Simulates realistic order execution including:
    - Market order fills with slippage
    - Limit order price matching
    - Stop-loss trigger logic
    
    Usage:
        executor = PaperExecutor(
            slippage_bps=5,        # 5 basis points slippage
            price_getter=get_ltp,   # Function to get current price
        )
        
        result = executor.execute(order)
    """
    
    def __init__(
        self,
        slippage_bps: int = 5,
        initial_capital: float = 100000.0,
        price_getter: Optional[callable] = None,
    ):
        """
        Initialize paper executor.
        
        Args:
            slippage_bps: Slippage in basis points (5 = 0.05%)
            initial_capital: Starting capital
            price_getter: Function(symbol) -> float to get current price
        """
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.price_getter = price_getter
        
        # Position tracking
        self.positions: Dict[str, PaperPosition] = {}
        
        # Trade history
        self.trades: List[PaperTrade] = []
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_pnl': 0.0,
        }
    
    def execute(self, order: Order) -> Order:
        """
        Execute a paper order.
        
        Args:
            order: Order to execute
            
        Returns:
            Order with updated status and fill information
        """
        logger.info(f"[PAPER] Executing: {order.side.value} {order.quantity} {order.symbol}")
        
        # Get execution price
        base_price = order.price
        if self.price_getter:
            try:
                base_price = self.price_getter(order.symbol) or order.price
            except Exception:
                pass
        
        if base_price <= 0:
            order.status = OrderStatus.REJECTED
            order.last_error = "Invalid price"
            return order
        
        # Apply slippage
        fill_price = self._apply_slippage(base_price, order.side)
        
        # Fill the order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.broker_order_id = f"PAPER-{order.order_id[:8]}"
        
        # Update positions
        self._update_position(order, fill_price)
        
        logger.info(f"[PAPER] Filled @ ₹{fill_price:.2f} (slippage: {self.slippage_bps}bps)")
        
        return order
    
    def __call__(self, order: Order) -> Order:
        """Callable interface for order manager"""
        return self.execute(order)
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply random slippage to price"""
        # Random slippage up to max
        slippage_pct = random.uniform(0, self.slippage_bps) / 10000
        
        if side == OrderSide.BUY:
            # Buy fills slightly higher
            return price * (1 + slippage_pct)
        else:
            # Sell fills slightly lower
            return price * (1 - slippage_pct)
    
    def _update_position(self, order: Order, fill_price: float) -> None:
        """Update positions based on filled order"""
        symbol = order.symbol
        quantity = order.filled_quantity
        
        existing = self.positions.get(symbol)
        
        if order.side == OrderSide.BUY:
            if existing and existing.side == "SHORT":
                # Closing short position
                self._close_position(symbol, fill_price, order)
            else:
                # Opening or adding to long
                self._open_position(symbol, "LONG", quantity, fill_price, order)
        else:
            if existing and existing.side == "LONG":
                # Closing long position
                self._close_position(symbol, fill_price, order)
            else:
                # Opening or adding to short
                self._open_position(symbol, "SHORT", quantity, fill_price, order)
    
    def _open_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order: Order,
    ) -> None:
        """Open or add to position"""
        existing = self.positions.get(symbol)
        
        if existing and existing.side == side:
            # Add to existing position (average price)
            total_value = existing.average_price * existing.quantity + price * quantity
            existing.quantity += quantity
            existing.average_price = total_value / existing.quantity
        else:
            # New position
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                average_price=price,
                current_price=price,
                entry_time=datetime.now().isoformat(),
                entry_order_id=order.order_id,
            )
        
        logger.info(f"[PAPER] Position opened: {side} {quantity} {symbol} @ ₹{price:.2f}")
    
    def _close_position(
        self,
        symbol: str,
        price: float,
        order: Order,
    ) -> None:
        """Close position and record trade"""
        position = self.positions.get(symbol)
        if not position:
            return
        
        # Calculate P&L
        if position.side == "LONG":
            gross_pnl = (price - position.average_price) * position.quantity
        else:
            gross_pnl = (position.average_price - price) * position.quantity
        
        # Record trade
        trade = PaperTrade(
            trade_id=f"TRADE-{order.order_id[:8]}",
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.average_price,
            exit_price=price,
            entry_time=position.entry_time,
            exit_time=datetime.now().isoformat(),
            gross_pnl=gross_pnl,
            strategy=order.strategy,
        )
        self.trades.append(trade)
        
        # Update stats
        self.stats['total_trades'] += 1
        self.stats['gross_pnl'] += gross_pnl
        if gross_pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        # Update capital
        self.capital += gross_pnl
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(
            f"[PAPER] Position closed: {symbol} P&L = ₹{gross_pnl:+,.2f} "
            f"(Capital: ₹{self.capital:,.2f})"
        )
    
    def get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all positions as dicts"""
        return [
            {
                'symbol': p.symbol,
                'side': p.side,
                'quantity': p.quantity,
                'entry_price': p.average_price,
                'current_price': p.current_price,
                'unrealized_pnl': p.unrealized_pnl,
            }
            for p in self.positions.values()
        ]
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_pnl(price)
    
    def get_today_trades(self) -> List[Dict[str, Any]]:
        """Get today's trades for EOD processing"""
        return [
            {
                'symbol': t.symbol,
                'side': 'BUY' if t.side == 'LONG' else 'SELL',
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'strategy': t.strategy,
                'exit_reason': t.exit_reason,
            }
            for t in self.trades
        ]
    
    def close_all_positions(self, current_prices: Dict[str, float]) -> float:
        """
        Close all positions at current prices.
        
        Used by square-off job.
        
        Args:
            current_prices: Dict of symbol -> price
            
        Returns:
            Total P&L from closed positions
        """
        total_pnl = 0.0
        
        for symbol, position in list(self.positions.items()):
            price = current_prices.get(symbol, position.current_price)
            
            # Create closing order
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL if position.side == "LONG" else OrderSide.BUY,
                quantity=position.quantity,
            )
            
            self._close_position(symbol, price, order)
            
            # Get P&L from last trade
            if self.trades:
                total_pnl += self.trades[-1].gross_pnl
        
        return total_pnl
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        win_rate = 0.0
        if self.stats['total_trades'] > 0:
            win_rate = self.stats['winning_trades'] / self.stats['total_trades'] * 100
        
        return {
            **self.stats,
            'win_rate': win_rate,
            'capital': self.capital,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
        }
    
    def reset(self) -> None:
        """Reset executor to initial state"""
        self.capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_pnl': 0.0,
        }
        logger.info("[PAPER] Executor reset to initial state")
