"""
Trading Signals
===============
Signal classes for BUY, SELL, HOLD decisions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class SignalType(Enum):
    """Type of trading signal"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT_LONG = "EXIT_LONG"   # Exit existing long position
    EXIT_SHORT = "EXIT_SHORT" # Exit existing short position


class SignalStrength(Enum):
    """Strength/confidence of the signal"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


class ExitReason(Enum):
    """Reason for exit signals"""
    STOP_LOSS = "STOP_LOSS"
    TARGET_HIT = "TARGET_HIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_EXIT = "TIME_EXIT"         # End of day exit
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    RISK_LIMIT = "RISK_LIMIT"
    MANUAL = "MANUAL"


@dataclass
class Signal:
    """
    Trading signal with all relevant information.
    
    Contains:
    - Signal type (BUY/SELL/HOLD)
    - Entry/exit prices
    - Stop loss and target
    - Confidence/strength
    - Reasoning
    
    Usage:
        signal = Signal(
            signal_type=SignalType.BUY,
            symbol="RELIANCE",
            price=2500.0,
            stop_loss=2450.0,
            target=2575.0,
            strength=SignalStrength.STRONG,
            reason="Breakout above opening range high with volume",
        )
    """
    signal_type: SignalType
    symbol: str
    price: float
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Entry parameters
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    
    # Position sizing hints
    suggested_quantity: Optional[int] = None
    position_size_pct: Optional[float] = None  # As % of capital
    
    # Signal metadata
    strength: SignalStrength = SignalStrength.MODERATE
    confidence: float = 0.5  # 0-1 confidence score
    
    # Strategy information
    strategy_name: str = ""
    reason: str = ""
    
    # Exit specific
    exit_reason: Optional[ExitReason] = None
    
    # Additional data
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal"""
        if self.signal_type in (SignalType.BUY, SignalType.SELL):
            if self.stop_loss is None:
                raise ValueError("Entry signals require stop_loss")
        
        # Validate stop loss direction
        if self.signal_type == SignalType.BUY and self.stop_loss:
            if self.stop_loss >= self.price:
                raise ValueError("Buy stop loss must be below entry price")
        
        if self.signal_type == SignalType.SELL and self.stop_loss:
            if self.stop_loss <= self.price:
                raise ValueError("Sell stop loss must be above entry price")
    
    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal"""
        return self.signal_type in (SignalType.BUY, SignalType.SELL)
    
    @property
    def is_exit(self) -> bool:
        """Check if this is an exit signal"""
        return self.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT)
    
    @property
    def is_hold(self) -> bool:
        """Check if this is a hold signal"""
        return self.signal_type == SignalType.HOLD
    
    @property
    def is_long(self) -> bool:
        """Check if this is a long signal"""
        return self.signal_type == SignalType.BUY
    
    @property
    def is_short(self) -> bool:
        """Check if this is a short signal"""
        return self.signal_type == SignalType.SELL
    
    @property
    def risk_per_share(self) -> float:
        """Calculate risk per share"""
        if self.stop_loss is None:
            return 0.0
        return abs(self.price - self.stop_loss)
    
    @property
    def reward_per_share(self) -> float:
        """Calculate reward per share"""
        if self.target is None:
            return 0.0
        return abs(self.target - self.price)
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio"""
        risk = self.risk_per_share
        if risk == 0:
            return 0.0
        return self.reward_per_share / risk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "stop_loss": self.stop_loss,
            "target": self.target,
            "trailing_stop_pct": self.trailing_stop_pct,
            "suggested_quantity": self.suggested_quantity,
            "position_size_pct": self.position_size_pct,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "strategy_name": self.strategy_name,
            "reason": self.reason,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "indicators": self.indicators,
            "metadata": self.metadata,
            "risk_reward_ratio": self.risk_reward_ratio,
        }
    
    @classmethod
    def hold(cls, symbol: str, reason: str = "No setup") -> "Signal":
        """Create a HOLD signal"""
        return cls(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            price=0.0,
            reason=reason,
        )
    
    @classmethod
    def exit_long(
        cls,
        symbol: str,
        price: float,
        reason: ExitReason,
        description: str = "",
    ) -> "Signal":
        """Create an exit signal for long position"""
        return cls(
            signal_type=SignalType.EXIT_LONG,
            symbol=symbol,
            price=price,
            exit_reason=reason,
            reason=description or reason.value,
        )
    
    @classmethod
    def exit_short(
        cls,
        symbol: str,
        price: float,
        reason: ExitReason,
        description: str = "",
    ) -> "Signal":
        """Create an exit signal for short position"""
        return cls(
            signal_type=SignalType.EXIT_SHORT,
            symbol=symbol,
            price=price,
            exit_reason=reason,
            reason=description or reason.value,
        )
    
    def __str__(self) -> str:
        """Human-readable representation"""
        if self.is_hold:
            return f"HOLD {self.symbol}: {self.reason}"
        
        if self.is_exit:
            return f"{self.signal_type.value} {self.symbol} @ {self.price:.2f} ({self.reason})"
        
        sl_str = f"SL:{self.stop_loss:.2f}" if self.stop_loss else ""
        tgt_str = f"TGT:{self.target:.2f}" if self.target else ""
        rr_str = f"RR:1:{self.risk_reward_ratio:.1f}" if self.risk_reward_ratio else ""
        
        return (
            f"{self.signal_type.value} {self.symbol} @ {self.price:.2f} | "
            f"{sl_str} {tgt_str} {rr_str} | "
            f"{self.strength.name} | {self.reason}"
        )


# ============================================================================
# Signal Aggregator
# ============================================================================

class SignalAggregator:
    """
    Aggregate signals from multiple strategies.
    
    Combines signals using voting or weighted averaging.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize aggregator.
        
        Args:
            weights: Strategy name -> weight mapping
        """
        self.weights = weights or {}
    
    def aggregate(self, signals: list[Signal]) -> Optional[Signal]:
        """
        Aggregate multiple signals into one.
        
        Args:
            signals: List of signals from different strategies
        
        Returns:
            Aggregated signal or None if no consensus
        """
        if not signals:
            return None
        
        if len(signals) == 1:
            return signals[0]
        
        # Group by type
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        holds = [s for s in signals if s.signal_type == SignalType.HOLD]
        
        # Simple majority voting
        if len(buys) > len(sells) and len(buys) > len(holds):
            # Average the buy signals
            return self._average_signals(buys, SignalType.BUY)
        elif len(sells) > len(buys) and len(sells) > len(holds):
            return self._average_signals(sells, SignalType.SELL)
        else:
            # No consensus, hold
            return Signal.hold(
                signals[0].symbol,
                f"No consensus: {len(buys)} buys, {len(sells)} sells, {len(holds)} holds"
            )
    
    def _average_signals(self, signals: list[Signal], signal_type: SignalType) -> Signal:
        """Average multiple signals of the same type"""
        if not signals:
            return None
        
        # Weight calculation
        total_weight = 0
        weighted_price = 0
        weighted_sl = 0
        weighted_target = 0
        weighted_confidence = 0
        
        for s in signals:
            weight = self.weights.get(s.strategy_name, 1.0) * s.confidence
            total_weight += weight
            weighted_price += s.price * weight
            if s.stop_loss:
                weighted_sl += s.stop_loss * weight
            if s.target:
                weighted_target += s.target * weight
            weighted_confidence += s.confidence * weight
        
        if total_weight == 0:
            return signals[0]  # Fallback
        
        avg_price = weighted_price / total_weight
        avg_sl = weighted_sl / total_weight if weighted_sl else signals[0].stop_loss
        avg_target = weighted_target / total_weight if weighted_target else signals[0].target
        avg_confidence = weighted_confidence / total_weight
        
        # Combine reasons
        reasons = [s.reason for s in signals if s.reason]
        combined_reason = "; ".join(reasons[:3])  # Limit to 3
        
        return Signal(
            signal_type=signal_type,
            symbol=signals[0].symbol,
            price=avg_price,
            stop_loss=avg_sl,
            target=avg_target,
            confidence=avg_confidence,
            strategy_name="aggregated",
            reason=combined_reason,
            strength=SignalStrength.MODERATE,
        )
