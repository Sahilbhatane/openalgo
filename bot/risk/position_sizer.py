"""
Position Sizer
==============
Calculate optimal position sizes based on risk management rules.
"""

from dataclasses import dataclass
from typing import Optional
import math

from ..utils.charges import calculate_total_charges, Exchange, Segment


@dataclass
class PositionSizeResult:
    """Result of position size calculation"""
    quantity: int
    position_value: float
    risk_amount: float
    risk_per_share: float
    charges: float
    breakeven_move: float
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    
    def __str__(self) -> str:
        return f"""
Position Size Calculation:
  Quantity:       {self.quantity:>10} shares
  Position Value: ₹{self.position_value:>12,.2f}
  Entry Price:    ₹{self.entry_price:>12,.2f}
  Stop Loss:      ₹{self.stop_loss:>12,.2f}
  Target:         ₹{self.target_price:>12,.2f}
  Risk/Share:     ₹{self.risk_per_share:>12,.2f}
  Total Risk:     ₹{self.risk_amount:>12,.2f}
  Charges:        ₹{self.charges:>12,.2f}
  Breakeven Move: ₹{self.breakeven_move:>12,.4f}
  Risk:Reward:    1:{self.risk_reward_ratio:>10.2f}
"""


class PositionSizer:
    """
    Calculate position sizes based on risk management.
    
    Uses fixed fractional position sizing:
    - Risk a fixed percentage of capital per trade
    - Position size = (Capital * Risk%) / Risk per share
    
    Also accounts for:
    - Brokerage and charges
    - Slippage estimates
    - Minimum position requirements
    """
    
    def __init__(
        self,
        capital: float,
        max_risk_per_trade: float = 0.01,  # 1%
        max_position_pct: float = 0.30,     # 30%
        slippage_estimate: float = 0.001,   # 0.1%
        exchange: Exchange = Exchange.NSE,
    ):
        """
        Initialize position sizer.
        
        Args:
            capital: Total trading capital
            max_risk_per_trade: Max risk as fraction of capital
            max_position_pct: Max position size as fraction of capital
            slippage_estimate: Estimated slippage percentage
            exchange: Trading exchange
        """
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.slippage_estimate = slippage_estimate
        self.exchange = exchange
    
    def update_capital(self, new_capital: float):
        """Update capital for position sizing"""
        self.capital = new_capital
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 1.5,
        is_long: bool = True,
    ) -> float:
        """
        Calculate stop loss price using ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            atr_multiplier: Multiplier for ATR
            is_long: True for long positions
        
        Returns:
            Stop loss price
        """
        stop_distance = atr * atr_multiplier
        
        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward: float = 1.5,
        is_long: bool = True,
    ) -> float:
        """
        Calculate target price based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_reward: Risk-reward ratio (target/risk)
            is_long: True for long positions
        
        Returns:
            Target price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward
        
        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward
    
    def calculate_size(
        self,
        entry_price: float,
        stop_loss: float,
        target_price: Optional[float] = None,
        risk_reward: float = 1.5,
        include_charges: bool = True,
        is_long: bool = True,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price (calculated if not provided)
            risk_reward: Risk-reward ratio
            include_charges: Include charges in risk calculation
            is_long: True for long positions
        
        Returns:
            PositionSizeResult with all calculations
        """
        # Calculate target if not provided
        if target_price is None:
            target_price = self.calculate_target(
                entry_price, stop_loss, risk_reward, is_long
            )
        
        # Risk per share (before charges)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            raise ValueError("Stop loss must be different from entry price")
        
        # Add slippage to risk
        slippage_amount = entry_price * self.slippage_estimate
        risk_per_share_with_slippage = risk_per_share + slippage_amount
        
        # Maximum risk amount
        max_risk = self.capital * self.max_risk_per_trade
        
        # Initial quantity calculation
        quantity = int(max_risk / risk_per_share_with_slippage)
        
        if include_charges and quantity > 0:
            # Calculate charges and adjust
            charges = calculate_total_charges(
                buy_price=entry_price,
                sell_price=stop_loss,
                quantity=quantity,
                exchange=self.exchange,
                segment=Segment.EQUITY_INTRADAY,
            )
            
            # Total risk including charges
            total_risk_per_share = risk_per_share_with_slippage + charges.charges_per_share
            
            # Recalculate quantity
            quantity = int(max_risk / total_risk_per_share)
            
            # Recalculate charges with new quantity
            charges = calculate_total_charges(
                buy_price=entry_price,
                sell_price=stop_loss,
                quantity=quantity,
                exchange=self.exchange,
                segment=Segment.EQUITY_INTRADAY,
            )
        else:
            charges = calculate_total_charges(
                buy_price=entry_price,
                sell_price=stop_loss,
                quantity=max(1, quantity),
                exchange=self.exchange,
                segment=Segment.EQUITY_INTRADAY,
            )
        
        # Ensure minimum quantity
        quantity = max(1, quantity)
        
        # Check max position size
        position_value = quantity * entry_price
        max_position = self.capital * self.max_position_pct
        
        if position_value > max_position:
            quantity = int(max_position / entry_price)
            quantity = max(1, quantity)
            position_value = quantity * entry_price
        
        # Final calculations
        risk_amount = quantity * risk_per_share
        
        # Calculate actual risk-reward
        reward_per_share = abs(target_price - entry_price)
        actual_rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        # Breakeven move (to cover charges)
        breakeven_move = charges.charges_per_share if quantity > 0 else 0
        
        return PositionSizeResult(
            quantity=quantity,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_per_share=risk_per_share,
            charges=charges.total_charges,
            breakeven_move=breakeven_move,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_reward_ratio=actual_rr,
        )
    
    def calculate_size_from_atr(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 1.5,
        risk_reward: float = 1.5,
        is_long: bool = True,
        include_charges: bool = True,
    ) -> PositionSizeResult:
        """
        Calculate position size using ATR for stop loss.
        
        This is the recommended method for volatility-based positioning.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            atr_multiplier: Multiplier for stop loss distance
            risk_reward: Target risk-reward ratio
            is_long: True for long positions
            include_charges: Include charges in calculation
        
        Returns:
            PositionSizeResult
        """
        stop_loss = self.calculate_stop_loss(
            entry_price, atr, atr_multiplier, is_long
        )
        
        return self.calculate_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_reward=risk_reward,
            include_charges=include_charges,
            is_long=is_long,
        )
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly Criterion for optimal bet sizing.
        
        Kelly % = W - [(1-W) / R]
        Where:
            W = Win rate
            R = Avg Win / Avg Loss
        
        Note: In practice, use half-Kelly or less for safety.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
        
        Returns:
            Kelly percentage (fraction of capital to risk)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Cap at max risk per trade
        return min(max(0, kelly), self.max_risk_per_trade)


# ============================================================================
# UNIT TESTS
# ============================================================================

def _test_position_sizer():
    """Test position sizer"""
    print("Testing position sizer...")
    
    sizer = PositionSizer(
        capital=100_000,
        max_risk_per_trade=0.01,  # 1%
        max_position_pct=0.30,    # 30%
    )
    
    # Test with explicit stop loss
    result = sizer.calculate_size(
        entry_price=100.0,
        stop_loss=95.0,  # 5% stop
        risk_reward=1.5,
    )
    
    print(result)
    
    assert result.quantity > 0, "Should calculate positive quantity"
    assert result.risk_amount <= 1000, f"Risk should be <= 1% of capital, got {result.risk_amount}"
    assert result.position_value <= 30000, "Position should be <= 30% of capital"
    
    # Test with ATR
    result_atr = sizer.calculate_size_from_atr(
        entry_price=100.0,
        atr=3.0,
        atr_multiplier=1.5,
        risk_reward=1.5,
    )
    
    print("\nATR-based sizing:")
    print(result_atr)
    
    # Verify stop loss is 1.5 * ATR away
    expected_sl = 100.0 - (3.0 * 1.5)
    assert abs(result_atr.stop_loss - expected_sl) < 0.01, f"Stop loss should be {expected_sl}"
    
    # Test Kelly criterion
    kelly = sizer.kelly_criterion(
        win_rate=0.55,
        avg_win=200,
        avg_loss=100,
    )
    print(f"\nKelly criterion: {kelly*100:.2f}% of capital")
    
    print("\n✅ All position sizer tests passed!")


if __name__ == "__main__":
    _test_position_sizer()
