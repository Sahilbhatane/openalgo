"""
Risk Manager
============
Central risk management with daily limits, drawdown tracking, and exposure control.
"""

import json
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable
from enum import Enum
import threading


class RiskLevel(Enum):
    """Risk alert levels"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class RiskLimits:
    """Configurable risk limits"""
    # Daily limits
    max_daily_loss_pct: float = 0.05      # 5% max daily loss
    max_daily_loss_amount: float = 5000.0  # Absolute max daily loss
    max_daily_profit_pct: float = 0.10     # 10% daily profit (stop greedy trading)
    
    # Per-trade limits
    max_loss_per_trade_pct: float = 0.01   # 1% max loss per trade
    max_position_size_pct: float = 0.30    # 30% max in single position
    
    # Drawdown limits
    max_drawdown_pct: float = 0.10         # 10% max drawdown from peak
    
    # Exposure limits
    max_open_positions: int = 3
    max_total_exposure_pct: float = 0.90   # 90% max capital at risk
    
    # Order limits
    max_orders_per_day: int = 20           # Circuit breaker
    max_order_value: float = 50_000.0      # Max single order
    
    # Warning thresholds (trigger alerts before limits)
    warning_daily_loss_pct: float = 0.03   # Warn at 3% daily loss
    warning_drawdown_pct: float = 0.05     # Warn at 5% drawdown


@dataclass
class RiskState:
    """Current risk state for the day"""
    date: str = ""
    starting_capital: float = 0.0
    current_capital: float = 0.0
    peak_capital: float = 0.0
    
    # Daily P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    
    # Drawdown
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_today: float = 0.0
    
    # Positions
    open_positions: int = 0
    total_exposure: float = 0.0
    exposure_pct: float = 0.0
    
    # Orders
    orders_today: int = 0
    
    # Status
    risk_level: str = "NORMAL"
    is_trading_allowed: bool = True
    halt_reason: str = ""
    
    # Alerts
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "RiskState":
        data.pop('alerts', None)  # Handle separately
        return cls(**data, alerts=[])


class RiskManager:
    """
    Central risk management system.
    
    Tracks:
    - Daily P&L limits
    - Drawdown from peak
    - Position exposure
    - Order count limits
    
    Features:
    - Real-time risk monitoring
    - Automatic trading halt when limits breached
    - Kill switch integration
    - Persistent state across restarts
    """
    
    def __init__(
        self,
        capital: float,
        limits: Optional[RiskLimits] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize risk manager.
        
        Args:
            capital: Starting capital
            limits: Risk limits (uses defaults if not provided)
            data_dir: Directory for persisting state
        """
        self.capital = capital
        self.limits = limits or RiskLimits()
        self.data_dir = data_dir or Path("bot_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._state = self._load_or_create_state()
        self._callbacks: List[Callable[[RiskLevel, str], None]] = []
        
    def _load_or_create_state(self) -> RiskState:
        """Load today's state or create new"""
        today = date.today().isoformat()
        state_file = self.data_dir / f"risk_state_{today}.json"
        
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    state = RiskState.from_dict(data)
                    if state.date == today:
                        return state
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Create new state for today
        return RiskState(
            date=today,
            starting_capital=self.capital,
            current_capital=self.capital,
            peak_capital=self.capital,
        )
    
    def _save_state(self):
        """Persist current state"""
        today = date.today().isoformat()
        state_file = self.data_dir / f"risk_state_{today}.json"
        
        with open(state_file, "w") as f:
            json.dump(self._state.to_dict(), f, indent=2)
    
    def register_callback(self, callback: Callable[[RiskLevel, str], None]):
        """Register callback for risk alerts"""
        self._callbacks.append(callback)
    
    def _notify(self, level: RiskLevel, message: str):
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(level, message)
            except Exception:
                pass  # Don't let callback errors affect risk manager
    
    @property
    def state(self) -> RiskState:
        """Get current risk state"""
        return self._state
    
    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        return self._state.is_trading_allowed
    
    def update_capital(self, current_capital: float, unrealized_pnl: float = 0.0):
        """
        Update capital and recalculate risk metrics.
        
        Args:
            current_capital: Current account value
            unrealized_pnl: Unrealized P&L from open positions
        """
        with self._lock:
            self._state.current_capital = current_capital
            self._state.unrealized_pnl = unrealized_pnl
            
            # Update peak
            if current_capital > self._state.peak_capital:
                self._state.peak_capital = current_capital
            
            # Calculate P&L
            self._state.realized_pnl = current_capital - self._state.starting_capital - unrealized_pnl
            self._state.total_pnl = current_capital - self._state.starting_capital
            self._state.daily_pnl_pct = (self._state.total_pnl / self._state.starting_capital) * 100
            
            # Calculate drawdown
            self._state.current_drawdown = self._state.peak_capital - current_capital
            self._state.current_drawdown_pct = (self._state.current_drawdown / self._state.peak_capital) * 100
            
            if self._state.current_drawdown > self._state.max_drawdown_today:
                self._state.max_drawdown_today = self._state.current_drawdown
            
            # Check limits
            self._check_limits()
            self._save_state()
    
    def update_positions(self, open_positions: int, total_exposure: float):
        """
        Update position count and exposure.
        
        Args:
            open_positions: Number of open positions
            total_exposure: Total value at risk
        """
        with self._lock:
            self._state.open_positions = open_positions
            self._state.total_exposure = total_exposure
            self._state.exposure_pct = (total_exposure / self._state.current_capital) * 100
            
            self._check_limits()
            self._save_state()
    
    def record_order(self):
        """Record an order (for daily limit tracking)"""
        with self._lock:
            self._state.orders_today += 1
            self._check_limits()
            self._save_state()
    
    def _check_limits(self):
        """Check all limits and update risk level"""
        alerts = []
        new_level = RiskLevel.NORMAL
        
        # Check daily loss limit
        daily_loss_pct = abs(min(0, self._state.daily_pnl_pct)) / 100
        
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            new_level = RiskLevel.SHUTDOWN
            alerts.append(f"LIMIT: Daily loss {daily_loss_pct*100:.1f}% >= {self.limits.max_daily_loss_pct*100}% limit")
        elif daily_loss_pct >= self.limits.warning_daily_loss_pct:
            new_level = max(new_level, RiskLevel.WARNING)
            alerts.append(f"WARNING: Daily loss {daily_loss_pct*100:.1f}% approaching limit")
        
        # Check absolute daily loss
        if abs(min(0, self._state.total_pnl)) >= self.limits.max_daily_loss_amount:
            new_level = RiskLevel.SHUTDOWN
            alerts.append(f"LIMIT: Daily loss ₹{abs(self._state.total_pnl):.0f} >= ₹{self.limits.max_daily_loss_amount:.0f}")
        
        # Check drawdown
        dd_pct = self._state.current_drawdown_pct / 100
        
        if dd_pct >= self.limits.max_drawdown_pct:
            new_level = RiskLevel.SHUTDOWN
            alerts.append(f"LIMIT: Drawdown {dd_pct*100:.1f}% >= {self.limits.max_drawdown_pct*100}% limit")
        elif dd_pct >= self.limits.warning_drawdown_pct:
            new_level = max(new_level, RiskLevel.WARNING)
            alerts.append(f"WARNING: Drawdown {dd_pct*100:.1f}% approaching limit")
        
        # Check order count
        if self._state.orders_today >= self.limits.max_orders_per_day:
            new_level = max(new_level, RiskLevel.CRITICAL)
            alerts.append(f"LIMIT: Order count {self._state.orders_today} >= {self.limits.max_orders_per_day}")
        
        # Check exposure
        if self._state.exposure_pct / 100 >= self.limits.max_total_exposure_pct:
            new_level = max(new_level, RiskLevel.WARNING)
            alerts.append(f"WARNING: Exposure {self._state.exposure_pct:.1f}% at limit")
        
        # Check position count
        if self._state.open_positions >= self.limits.max_open_positions:
            new_level = max(new_level, RiskLevel.WARNING)
            alerts.append(f"INFO: Max positions ({self.limits.max_open_positions}) reached")
        
        # Update state
        self._state.risk_level = new_level.value
        self._state.alerts = alerts
        
        # Halt trading if shutdown level
        if new_level == RiskLevel.SHUTDOWN:
            self._state.is_trading_allowed = False
            self._state.halt_reason = alerts[0] if alerts else "Risk limit breached"
            self._notify(new_level, self._state.halt_reason)
        elif new_level in (RiskLevel.WARNING, RiskLevel.CRITICAL):
            for alert in alerts:
                self._notify(new_level, alert)
    
    def can_open_position(
        self,
        position_value: float,
        risk_amount: float,
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Args:
            position_value: Value of the new position
            risk_amount: Maximum risk (stop loss amount)
        
        Returns:
            Tuple of (allowed, reason)
        """
        if not self._state.is_trading_allowed:
            return False, f"Trading halted: {self._state.halt_reason}"
        
        # Check position count
        if self._state.open_positions >= self.limits.max_open_positions:
            return False, f"Max positions ({self.limits.max_open_positions}) reached"
        
        # Check order count
        if self._state.orders_today >= self.limits.max_orders_per_day:
            return False, f"Max orders ({self.limits.max_orders_per_day}) reached today"
        
        # Check position size
        max_position = self._state.current_capital * self.limits.max_position_size_pct
        if position_value > max_position:
            return False, f"Position ₹{position_value:.0f} > max ₹{max_position:.0f}"
        
        # Check exposure
        new_exposure = self._state.total_exposure + position_value
        max_exposure = self._state.current_capital * self.limits.max_total_exposure_pct
        if new_exposure > max_exposure:
            return False, f"Exposure would exceed {self.limits.max_total_exposure_pct*100}%"
        
        # Check per-trade risk
        max_risk = self._state.current_capital * self.limits.max_loss_per_trade_pct
        if risk_amount > max_risk:
            return False, f"Risk ₹{risk_amount:.0f} > max ₹{max_risk:.0f} per trade"
        
        return True, "OK"
    
    def halt_trading(self, reason: str):
        """Manually halt trading"""
        with self._lock:
            self._state.is_trading_allowed = False
            self._state.halt_reason = reason
            self._state.risk_level = RiskLevel.SHUTDOWN.value
            self._save_state()
            self._notify(RiskLevel.SHUTDOWN, reason)
    
    def resume_trading(self, override_checks: bool = False):
        """
        Resume trading after halt.
        
        Args:
            override_checks: Skip risk checks (dangerous)
        """
        with self._lock:
            if not override_checks:
                # Re-check limits before resuming
                self._check_limits()
                if self._state.risk_level == RiskLevel.SHUTDOWN.value:
                    return False, "Risk limits still breached"
            
            self._state.is_trading_allowed = True
            self._state.halt_reason = ""
            self._state.risk_level = RiskLevel.NORMAL.value
            self._save_state()
            return True, "Trading resumed"
    
    def reset_daily_state(self, new_capital: Optional[float] = None):
        """Reset state for new trading day"""
        with self._lock:
            capital = new_capital or self._state.current_capital
            self._state = RiskState(
                date=date.today().isoformat(),
                starting_capital=capital,
                current_capital=capital,
                peak_capital=capital,
            )
            self._save_state()
    
    def get_status_report(self) -> str:
        """Get human-readable status report"""
        s = self._state
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                      RISK STATUS                             ║
╠══════════════════════════════════════════════════════════════╣
║  Status: {s.risk_level:>10}  Trading: {'✅ ALLOWED' if s.is_trading_allowed else '❌ HALTED ':>12}     ║
╠══════════════════════════════════════════════════════════════╣
║  CAPITAL                                                     ║
║    Starting:       ₹{s.starting_capital:>12,.2f}                        ║
║    Current:        ₹{s.current_capital:>12,.2f}                        ║
║    Peak:           ₹{s.peak_capital:>12,.2f}                        ║
╠══════════════════════════════════════════════════════════════╣
║  P&L                                                         ║
║    Realized:       ₹{s.realized_pnl:>+12,.2f}                        ║
║    Unrealized:     ₹{s.unrealized_pnl:>+12,.2f}                        ║
║    Total:          ₹{s.total_pnl:>+12,.2f} ({s.daily_pnl_pct:+.2f}%)              ║
╠══════════════════════════════════════════════════════════════╣
║  DRAWDOWN                                                    ║
║    Current:        ₹{s.current_drawdown:>12,.2f} ({s.current_drawdown_pct:.2f}%)           ║
║    Max Today:      ₹{s.max_drawdown_today:>12,.2f}                        ║
╠══════════════════════════════════════════════════════════════╣
║  EXPOSURE                                                    ║
║    Positions:      {s.open_positions:>12} / {self.limits.max_open_positions}                        ║
║    Exposure:       ₹{s.total_exposure:>12,.2f} ({s.exposure_pct:.1f}%)             ║
║    Orders Today:   {s.orders_today:>12} / {self.limits.max_orders_per_day}                       ║
╚══════════════════════════════════════════════════════════════╝
"""
