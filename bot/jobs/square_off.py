"""
Square Off Job
==============
Job that runs at 3:10 PM IST to close all positions.

Tasks:
1. Get all open positions
2. Place market orders to close each position
3. Verify all positions are closed
4. Log square-off results
"""

import logging
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field

from ..strategies.signal import Signal, SignalType, ExitReason


logger = logging.getLogger("bot.jobs.square_off")


@dataclass
class SquareOffResult:
    """Result of square-off operation"""
    timestamp: str = ""
    positions_closed: int = 0
    positions_failed: int = 0
    total_pnl: float = 0.0
    
    details: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return self.positions_failed == 0


class SquareOffJob:
    """
    End-of-day square-off job.
    
    Closes all open intraday positions at 3:10 PM.
    This is a critical job that MUST succeed.
    
    Retry logic:
    - Try market orders first
    - If market order fails, retry up to 3 times
    - Log all failures for manual intervention
    
    Usage:
        job = SquareOffJob(
            position_getter=get_open_positions,
            order_executor=execute_order,
        )
        result = job.run()
    """
    
    def __init__(
        self,
        position_getter: Callable[[], List[Dict[str, Any]]],
        order_executor: Callable[[Signal], bool],
        max_retries: int = 3,
        on_complete: Optional[Callable[[SquareOffResult], None]] = None,
    ):
        """
        Initialize square-off job.
        
        Args:
            position_getter: Function that returns list of open positions
                Each position should have: symbol, side, quantity, entry_price, current_price
            order_executor: Function to execute a close order
            max_retries: Maximum retries for failed orders
            on_complete: Callback when square-off is complete
        """
        self.get_positions = position_getter
        self.execute_order = order_executor
        self.max_retries = max_retries
        self.on_complete = on_complete
    
    def run(self) -> SquareOffResult:
        """
        Run the square-off process.
        
        Returns:
            SquareOffResult with details of all closed positions
        """
        logger.info("=" * 60)
        logger.info("STARTING END-OF-DAY SQUARE OFF")
        logger.info("=" * 60)
        
        result = SquareOffResult(
            timestamp=datetime.now().isoformat(),
        )
        
        # Get all open positions
        try:
            positions = self.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            result.errors.append(f"Failed to get positions: {e}")
            return result
        
        if not positions:
            logger.info("No open positions to close")
            return result
        
        logger.info(f"Found {len(positions)} open positions to close")
        
        # Close each position
        total_pnl = 0.0
        
        for position in positions:
            symbol = position.get('symbol', 'UNKNOWN')
            side = position.get('side', 'BUY')
            quantity = position.get('quantity', 0)
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', entry_price)
            
            logger.info(f"Closing position: {symbol} {side} {quantity} @ {entry_price}")
            
            # Create exit signal
            if side == 'BUY':
                exit_signal = Signal(
                    signal_type=SignalType.EXIT_LONG,
                    symbol=symbol,
                    price=current_price,
                    exit_reason=ExitReason.TIME_EXIT,
                    reason="EOD Square Off",
                    metadata={'quantity': quantity},
                )
            else:
                exit_signal = Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    symbol=symbol,
                    price=current_price,
                    exit_reason=ExitReason.TIME_EXIT,
                    reason="EOD Square Off",
                    metadata={'quantity': quantity},
                )
            
            # Execute with retries
            closed = False
            for attempt in range(1, self.max_retries + 1):
                try:
                    success = self.execute_order(exit_signal)
                    if success:
                        closed = True
                        break
                    else:
                        logger.warning(f"Attempt {attempt} failed for {symbol}")
                except Exception as e:
                    logger.error(f"Attempt {attempt} error for {symbol}: {e}")
                
                if attempt < self.max_retries:
                    import time
                    time.sleep(1)  # Brief pause before retry
            
            # Calculate P&L
            if side == 'BUY':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            # Record result
            detail = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'closed': closed,
            }
            result.details.append(detail)
            
            if closed:
                result.positions_closed += 1
                total_pnl += pnl
                logger.info(f"  ✅ Closed {symbol}: P&L = ₹{pnl:+,.2f}")
            else:
                result.positions_failed += 1
                result.errors.append(f"Failed to close {symbol} after {self.max_retries} attempts")
                logger.error(f"  ❌ FAILED to close {symbol}!")
        
        result.total_pnl = total_pnl
        
        # Log summary
        logger.info("=" * 60)
        logger.info("SQUARE OFF COMPLETE")
        logger.info(f"  Positions closed: {result.positions_closed}")
        logger.info(f"  Positions failed: {result.positions_failed}")
        logger.info(f"  Total P&L: ₹{result.total_pnl:+,.2f}")
        logger.info("=" * 60)
        
        # Alert on failures
        if result.positions_failed > 0:
            logger.critical(
                f"ALERT: {result.positions_failed} positions failed to close! "
                "Manual intervention required!"
            )
        
        # Callback
        if self.on_complete:
            try:
                self.on_complete(result)
            except Exception as e:
                logger.error(f"on_complete callback failed: {e}")
        
        return result
    
    def force_close_all(self) -> SquareOffResult:
        """
        Force close all positions immediately.
        
        Same as run() but with higher urgency logging.
        Called by kill switch.
        """
        logger.warning("🚨 FORCE CLOSING ALL POSITIONS")
        return self.run()
