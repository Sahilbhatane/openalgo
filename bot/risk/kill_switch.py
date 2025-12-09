"""
Kill Switch
===========
Emergency stop functionality to immediately close all positions.
"""

import json
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict
from enum import Enum


class KillSwitchTrigger(Enum):
    """Reasons for kill switch activation"""
    MANUAL = "MANUAL"                    # User triggered
    MAX_DRAWDOWN = "MAX_DRAWDOWN"        # Drawdown limit hit
    DAILY_LOSS = "DAILY_LOSS"            # Daily loss limit hit
    CONNECTION_LOST = "CONNECTION_LOST"  # Lost broker connection
    DATA_STALE = "DATA_STALE"            # No market data
    API_ERROR = "API_ERROR"              # Repeated API errors
    MARKET_HALT = "MARKET_HALT"          # Market trading halt
    SYSTEM_ERROR = "SYSTEM_ERROR"        # System/application error


@dataclass
class KillSwitchState:
    """Kill switch state"""
    is_active: bool = False
    triggered_at: Optional[str] = None
    trigger_reason: Optional[str] = None
    trigger_type: Optional[str] = None
    positions_closed: int = 0
    orders_cancelled: int = 0
    pnl_at_trigger: float = 0.0
    notes: str = ""


@dataclass
class Position:
    """Simple position representation for kill switch"""
    symbol: str
    quantity: int
    side: str  # "BUY" or "SELL"
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


class KillSwitch:
    """
    Emergency kill switch to close all positions immediately.
    
    Features:
    - Manual activation
    - Automatic triggers from risk manager
    - Closes all open positions
    - Cancels all pending orders
    - Prevents new orders until reset
    - Logs all actions for audit
    
    Usage:
        kill_switch = KillSwitch(...)
        
        # Register position closer (from execution module)
        kill_switch.register_position_closer(close_all_positions)
        kill_switch.register_order_canceller(cancel_all_orders)
        
        # Trigger
        kill_switch.activate(KillSwitchTrigger.MANUAL, "User requested")
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        on_activate: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize kill switch.
        
        Args:
            data_dir: Directory for state persistence
            on_activate: Callback when activated
        """
        self.data_dir = data_dir or Path("bot_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "kill_switch_state.json"
        
        self._lock = threading.Lock()
        self._state = self._load_state()
        self._on_activate = on_activate
        
        # Callbacks for position/order management
        self._position_closers: List[Callable[[], int]] = []
        self._order_cancellers: List[Callable[[], int]] = []
        self._notification_handlers: List[Callable[[str, str], None]] = []
    
    def _load_state(self) -> KillSwitchState:
        """Load state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    return KillSwitchState(**data)
            except (json.JSONDecodeError, TypeError):
                pass
        return KillSwitchState()
    
    def _save_state(self):
        """Persist state to file"""
        with open(self.state_file, "w") as f:
            json.dump(vars(self._state), f, indent=2)
    
    @property
    def is_active(self) -> bool:
        """Check if kill switch is active"""
        return self._state.is_active
    
    @property
    def state(self) -> KillSwitchState:
        """Get current state"""
        return self._state
    
    def register_position_closer(self, closer: Callable[[], int]):
        """
        Register a function to close all positions.
        
        The function should:
        - Close all open positions
        - Return the number of positions closed
        """
        self._position_closers.append(closer)
    
    def register_order_canceller(self, canceller: Callable[[], int]):
        """
        Register a function to cancel all pending orders.
        
        The function should:
        - Cancel all pending orders
        - Return the number of orders cancelled
        """
        self._order_cancellers.append(canceller)
    
    def register_notification_handler(self, handler: Callable[[str, str], None]):
        """
        Register a notification handler.
        
        Args:
            handler: Function(title, message) to send notifications
        """
        self._notification_handlers.append(handler)
    
    def _notify(self, title: str, message: str):
        """Send notifications to all handlers"""
        for handler in self._notification_handlers:
            try:
                handler(title, message)
            except Exception:
                pass  # Don't let notification errors stop kill switch
    
    def activate(
        self,
        trigger: KillSwitchTrigger,
        reason: str,
        pnl: float = 0.0,
    ) -> KillSwitchState:
        """
        Activate the kill switch.
        
        This will:
        1. Close all open positions
        2. Cancel all pending orders
        3. Prevent new orders
        4. Log the activation
        5. Send notifications
        
        Args:
            trigger: What triggered the kill switch
            reason: Human-readable reason
            pnl: P&L at time of trigger
        
        Returns:
            Updated state
        """
        with self._lock:
            if self._state.is_active:
                return self._state  # Already active
            
            self._state.is_active = True
            self._state.triggered_at = datetime.now().isoformat()
            self._state.trigger_type = trigger.value
            self._state.trigger_reason = reason
            self._state.pnl_at_trigger = pnl
            
            # Cancel all orders first
            orders_cancelled = 0
            for canceller in self._order_cancellers:
                try:
                    orders_cancelled += canceller()
                except Exception as e:
                    self._state.notes += f"Order cancel error: {e}\n"
            
            self._state.orders_cancelled = orders_cancelled
            
            # Close all positions
            positions_closed = 0
            for closer in self._position_closers:
                try:
                    positions_closed += closer()
                except Exception as e:
                    self._state.notes += f"Position close error: {e}\n"
            
            self._state.positions_closed = positions_closed
            
            # Save state
            self._save_state()
            
            # Notify
            self._notify(
                "🚨 KILL SWITCH ACTIVATED",
                f"Reason: {reason}\n"
                f"Trigger: {trigger.value}\n"
                f"Positions closed: {positions_closed}\n"
                f"Orders cancelled: {orders_cancelled}\n"
                f"P&L at trigger: ₹{pnl:,.2f}"
            )
            
            # Call on_activate callback
            if self._on_activate:
                try:
                    self._on_activate()
                except Exception:
                    pass
            
            return self._state
    
    def deactivate(self, confirmation: str = "") -> tuple[bool, str]:
        """
        Deactivate the kill switch.
        
        Args:
            confirmation: Must be "CONFIRM_RESUME" to proceed
        
        Returns:
            Tuple of (success, message)
        """
        if confirmation != "CONFIRM_RESUME":
            return False, "Must provide confirmation='CONFIRM_RESUME'"
        
        with self._lock:
            if not self._state.is_active:
                return True, "Kill switch was not active"
            
            # Archive current state
            self._archive_activation()
            
            # Reset state
            self._state = KillSwitchState()
            self._save_state()
            
            self._notify(
                "✅ Kill Switch Deactivated",
                "Trading can resume. Please review the cause before continuing."
            )
            
            return True, "Kill switch deactivated"
    
    def _archive_activation(self):
        """Archive the activation for audit trail"""
        if not self._state.triggered_at:
            return
        
        archive_file = self.data_dir / "kill_switch_history.json"
        
        history = []
        if archive_file.exists():
            try:
                with open(archive_file) as f:
                    history = json.load(f)
            except (json.JSONDecodeError, TypeError):
                pass
        
        history.append(vars(self._state))
        
        with open(archive_file, "w") as f:
            json.dump(history, f, indent=2)
    
    def check_and_trigger(
        self,
        current_pnl: float,
        max_loss_pct: float,
        capital: float,
    ) -> bool:
        """
        Check if kill switch should be triggered based on P&L.
        
        Args:
            current_pnl: Current P&L
            max_loss_pct: Maximum allowed loss percentage
            capital: Starting capital
        
        Returns:
            True if triggered
        """
        if self._state.is_active:
            return True
        
        loss_pct = abs(min(0, current_pnl)) / capital
        
        if loss_pct >= max_loss_pct:
            self.activate(
                trigger=KillSwitchTrigger.DAILY_LOSS,
                reason=f"Daily loss {loss_pct*100:.1f}% >= {max_loss_pct*100}% limit",
                pnl=current_pnl,
            )
            return True
        
        return False
    
    def arm(self):
        """Arm the kill switch (ready to activate)"""
        # Currently just ensures state file exists
        self._save_state()
    
    def get_status(self) -> str:
        """Get human-readable status"""
        s = self._state
        
        if not s.is_active:
            return """
╔══════════════════════════════════════════════════════════════╗
║                    KILL SWITCH STATUS                        ║
╠══════════════════════════════════════════════════════════════╣
║  Status: ✅ INACTIVE (Armed and ready)                       ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    KILL SWITCH STATUS                        ║
╠══════════════════════════════════════════════════════════════╣
║  Status: 🚨 ACTIVE - ALL TRADING HALTED                      ║
╠══════════════════════════════════════════════════════════════╣
║  Triggered At:    {s.triggered_at or 'N/A':<40}║
║  Trigger Type:    {s.trigger_type or 'N/A':<40}║
║  Reason:          {(s.trigger_reason or 'N/A')[:40]:<40}║
╠══════════════════════════════════════════════════════════════╣
║  Actions Taken:                                              ║
║    Positions Closed: {s.positions_closed:>6}                                  ║
║    Orders Cancelled: {s.orders_cancelled:>6}                                  ║
║    P&L at Trigger:   ₹{s.pnl_at_trigger:>10,.2f}                           ║
╠══════════════════════════════════════════════════════════════╣
║  To resume trading, call:                                    ║
║    kill_switch.deactivate(confirmation="CONFIRM_RESUME")     ║
╚══════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# UNIT TESTS
# ============================================================================

def _test_kill_switch():
    """Test kill switch"""
    print("Testing kill switch...")
    
    import tempfile
    import shutil
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create kill switch
        ks = KillSwitch(data_dir=temp_dir)
        
        # Check initial state
        assert not ks.is_active, "Should be inactive initially"
        print(ks.get_status())
        
        # Register mock closers
        def mock_close_positions():
            print("  Mock: Closing all positions...")
            return 3
        
        def mock_cancel_orders():
            print("  Mock: Cancelling all orders...")
            return 2
        
        ks.register_position_closer(mock_close_positions)
        ks.register_order_canceller(mock_cancel_orders)
        
        # Activate
        print("\nActivating kill switch...")
        state = ks.activate(
            trigger=KillSwitchTrigger.MANUAL,
            reason="Test activation",
            pnl=-5000,
        )
        
        assert ks.is_active, "Should be active"
        assert state.positions_closed == 3, "Should have closed 3 positions"
        assert state.orders_cancelled == 2, "Should have cancelled 2 orders"
        
        print(ks.get_status())
        
        # Try to activate again (should be no-op)
        state2 = ks.activate(KillSwitchTrigger.MAX_DRAWDOWN, "Second trigger")
        assert state2.trigger_type == "MANUAL", "Should keep first trigger"
        
        # Try to deactivate without confirmation
        success, msg = ks.deactivate()
        assert not success, "Should fail without confirmation"
        
        # Deactivate with confirmation
        success, msg = ks.deactivate(confirmation="CONFIRM_RESUME")
        assert success, f"Should succeed: {msg}"
        assert not ks.is_active, "Should be inactive"
        
        print("\nAfter deactivation:")
        print(ks.get_status())
        
        # Test automatic trigger
        ks2 = KillSwitch(data_dir=temp_dir)
        ks2.register_position_closer(mock_close_positions)
        
        triggered = ks2.check_and_trigger(
            current_pnl=-6000,
            max_loss_pct=0.05,
            capital=100000,
        )
        assert triggered, "Should trigger at 6% loss"
        assert ks2.is_active, "Should be active"
        
        print("\n✅ All kill switch tests passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    _test_kill_switch()
