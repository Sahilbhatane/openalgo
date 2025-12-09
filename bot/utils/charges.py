"""
Brokerage and Charges Calculator
================================
Calculate all trading costs: brokerage, STT, exchange charges, GST, stamp duty.

Reference: AngelOne charges as of 2024
- Brokerage: Rs 20 per executed order or 0.03% (whichever is lower) for intraday
- STT: 0.025% on sell side for intraday equity
- Exchange charges: ~0.00322% NSE, ~0.00375% BSE
- GST: 18% on (brokerage + exchange charges)
- SEBI charges: Rs 10 per crore
- Stamp duty: 0.015% on buy side for intraday (Maharashtra rate, varies by state)
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Exchange(Enum):
    """Stock exchange"""
    NSE = "NSE"
    BSE = "BSE"


class Segment(Enum):
    """Trading segment"""
    EQUITY_INTRADAY = "EQUITY_INTRADAY"
    EQUITY_DELIVERY = "EQUITY_DELIVERY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


@dataclass
class ChargesBreakdown:
    """Detailed breakdown of all trading charges"""
    turnover: float  # Total buy + sell value
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    total_charges: float
    charges_per_share: float
    breakeven_points: float  # Price movement needed to break even
    
    def __str__(self) -> str:
        return f"""
Charges Breakdown:
  Turnover:         ₹{self.turnover:>12,.2f}
  Brokerage:        ₹{self.brokerage:>12,.2f}
  STT:              ₹{self.stt:>12,.2f}
  Exchange Charges: ₹{self.exchange_charges:>12,.2f}
  GST:              ₹{self.gst:>12,.2f}
  SEBI Charges:     ₹{self.sebi_charges:>12,.2f}
  Stamp Duty:       ₹{self.stamp_duty:>12,.2f}
  ─────────────────────────────
  Total Charges:    ₹{self.total_charges:>12,.2f}
  Per Share:        ₹{self.charges_per_share:>12,.4f}
  Breakeven:        {self.breakeven_points:>12,.4f} points
"""


# Rate constants (as of 2024)
BROKERAGE_PCT: float = 0.0003  # 0.03%
BROKERAGE_MAX: float = 20.0  # Rs 20 max per order

STT_INTRADAY_SELL: float = 0.00025  # 0.025% on sell side

EXCHANGE_NSE: float = 0.0000322  # 0.00322%
EXCHANGE_BSE: float = 0.0000375  # 0.00375%

GST_RATE: float = 0.18  # 18%

SEBI_RATE: float = 0.000001  # Rs 10 per crore = 0.0001%

STAMP_DUTY_INTRADAY: float = 0.00015  # 0.015% on buy side


def calculate_brokerage(
    buy_value: float,
    sell_value: float,
    segment: Segment = Segment.EQUITY_INTRADAY,
) -> float:
    """
    Calculate brokerage for a complete trade (buy + sell).
    
    AngelOne Equity Intraday: 0.03% or Rs 20, whichever is lower (per order)
    
    Args:
        buy_value: Value of buy order
        sell_value: Value of sell order
        segment: Trading segment
    
    Returns:
        Total brokerage for both legs
    """
    if segment == Segment.EQUITY_INTRADAY:
        buy_brokerage = min(buy_value * BROKERAGE_PCT, BROKERAGE_MAX)
        sell_brokerage = min(sell_value * BROKERAGE_PCT, BROKERAGE_MAX)
        return buy_brokerage + sell_brokerage
    elif segment == Segment.EQUITY_DELIVERY:
        # Zero brokerage for delivery (AngelOne)
        return 0.0
    else:
        # F&O - flat Rs 20 per order
        return BROKERAGE_MAX * 2


def calculate_stt(
    buy_value: float,
    sell_value: float,
    segment: Segment = Segment.EQUITY_INTRADAY,
) -> float:
    """
    Calculate Securities Transaction Tax.
    
    Equity Intraday: 0.025% on sell side only
    Equity Delivery: 0.1% on both sides
    
    Args:
        buy_value: Value of buy order
        sell_value: Value of sell order
        segment: Trading segment
    
    Returns:
        Total STT
    """
    if segment == Segment.EQUITY_INTRADAY:
        return sell_value * STT_INTRADAY_SELL
    elif segment == Segment.EQUITY_DELIVERY:
        return (buy_value + sell_value) * 0.001  # 0.1%
    else:
        return 0.0  # Different for F&O


def calculate_exchange_charges(
    buy_value: float,
    sell_value: float,
    exchange: Exchange = Exchange.NSE,
) -> float:
    """
    Calculate exchange transaction charges.
    
    NSE: 0.00322%
    BSE: 0.00375%
    
    Applied on total turnover (buy + sell)
    """
    turnover = buy_value + sell_value
    rate = EXCHANGE_NSE if exchange == Exchange.NSE else EXCHANGE_BSE
    return turnover * rate


def calculate_gst(brokerage: float, exchange_charges: float) -> float:
    """
    Calculate GST on brokerage and exchange charges.
    
    GST @ 18% on (brokerage + exchange transaction charges)
    """
    return (brokerage + exchange_charges) * GST_RATE


def calculate_sebi_charges(buy_value: float, sell_value: float) -> float:
    """
    Calculate SEBI turnover charges.
    
    Rs 10 per crore of turnover = 0.0001%
    """
    turnover = buy_value + sell_value
    return turnover * SEBI_RATE


def calculate_stamp_duty(
    buy_value: float,
    segment: Segment = Segment.EQUITY_INTRADAY,
) -> float:
    """
    Calculate stamp duty.
    
    Equity Intraday: 0.015% on buy side (Maharashtra rate)
    Note: Rates vary by state. Using Maharashtra as default.
    
    Args:
        buy_value: Value of buy order
        segment: Trading segment
    
    Returns:
        Stamp duty amount
    """
    if segment == Segment.EQUITY_INTRADAY:
        return buy_value * STAMP_DUTY_INTRADAY
    elif segment == Segment.EQUITY_DELIVERY:
        return buy_value * 0.00015  # Same rate
    else:
        return 0.0


def calculate_total_charges(
    buy_price: float,
    sell_price: float,
    quantity: int,
    exchange: Exchange = Exchange.NSE,
    segment: Segment = Segment.EQUITY_INTRADAY,
) -> ChargesBreakdown:
    """
    Calculate complete charges breakdown for a trade.
    
    Args:
        buy_price: Entry price per share
        sell_price: Exit price per share
        quantity: Number of shares
        exchange: NSE or BSE
        segment: Trading segment
    
    Returns:
        ChargesBreakdown with all charges
    
    Example:
        >>> charges = calculate_total_charges(
        ...     buy_price=100.0,
        ...     sell_price=102.0,
        ...     quantity=100
        ... )
        >>> print(f"Total charges: ₹{charges.total_charges:.2f}")
        >>> print(f"Breakeven: {charges.breakeven_points:.4f} points")
    """
    buy_value = buy_price * quantity
    sell_value = sell_price * quantity
    turnover = buy_value + sell_value
    
    brokerage = calculate_brokerage(buy_value, sell_value, segment)
    stt = calculate_stt(buy_value, sell_value, segment)
    exchange_charges = calculate_exchange_charges(buy_value, sell_value, exchange)
    gst = calculate_gst(brokerage, exchange_charges)
    sebi_charges = calculate_sebi_charges(buy_value, sell_value)
    stamp_duty = calculate_stamp_duty(buy_value, segment)
    
    total_charges = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty
    charges_per_share = total_charges / quantity if quantity > 0 else 0
    
    # Breakeven: how many points price must move to cover charges
    # For a BUY trade: sell_price must be at least buy_price + (charges / quantity)
    breakeven_points = charges_per_share
    
    return ChargesBreakdown(
        turnover=turnover,
        brokerage=brokerage,
        stt=stt,
        exchange_charges=exchange_charges,
        gst=gst,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
        total_charges=total_charges,
        charges_per_share=charges_per_share,
        breakeven_points=breakeven_points,
    )


def calculate_breakeven_points(
    price: float,
    quantity: int,
    exchange: Exchange = Exchange.NSE,
    segment: Segment = Segment.EQUITY_INTRADAY,
) -> float:
    """
    Calculate minimum price movement needed to break even.
    
    Useful for:
    - Setting minimum target distance
    - Understanding if a trade is worth taking
    
    Args:
        price: Current/entry price
        quantity: Number of shares
        exchange: NSE or BSE
        segment: Trading segment
    
    Returns:
        Points needed to break even
    
    Example:
        >>> breakeven = calculate_breakeven_points(100.0, 100)
        >>> print(f"Need {breakeven:.4f} point move to break even")
    """
    # Calculate round-trip charges assuming same exit price
    charges = calculate_total_charges(
        buy_price=price,
        sell_price=price,
        quantity=quantity,
        exchange=exchange,
        segment=segment,
    )
    return charges.charges_per_share


def estimate_position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    max_risk_pct: float = 0.01,  # 1% risk
    include_charges: bool = True,
    exchange: Exchange = Exchange.NSE,
) -> tuple[int, float, float]:
    """
    Calculate position size based on risk management.
    
    Given capital and max risk, calculate how many shares to buy
    such that if stop loss is hit, loss doesn't exceed max_risk_pct.
    
    Args:
        capital: Total trading capital
        entry_price: Entry price per share
        stop_loss: Stop loss price
        max_risk_pct: Maximum risk as percentage of capital
        include_charges: Include charges in risk calculation
        exchange: NSE or BSE
    
    Returns:
        Tuple of (quantity, position_value, risk_amount)
    
    Example:
        >>> qty, value, risk = estimate_position_size(
        ...     capital=100000,
        ...     entry_price=100,
        ...     stop_loss=95,
        ...     max_risk_pct=0.01
        ... )
        >>> print(f"Buy {qty} shares worth ₹{value:.2f}, risking ₹{risk:.2f}")
    """
    max_risk_amount = capital * max_risk_pct
    risk_per_share = abs(entry_price - stop_loss)
    
    if risk_per_share <= 0:
        raise ValueError("Stop loss must be different from entry price")
    
    # Initial quantity without charges
    quantity = int(max_risk_amount / risk_per_share)
    
    if include_charges and quantity > 0:
        # Adjust for charges
        # Charges are typically ~0.05-0.1% of turnover for a round trip
        # We need to account for this in our risk calculation
        charges = calculate_total_charges(
            buy_price=entry_price,
            sell_price=stop_loss,
            quantity=quantity,
            exchange=exchange,
        )
        
        # Total risk = (loss from price movement) + charges
        total_risk_per_share = risk_per_share + charges.charges_per_share
        quantity = int(max_risk_amount / total_risk_per_share)
    
    # Ensure minimum quantity
    quantity = max(1, quantity)
    
    position_value = quantity * entry_price
    actual_risk = quantity * risk_per_share
    
    return quantity, position_value, actual_risk


# ============================================================================
# UNIT TESTS
# ============================================================================

def _test_charges():
    """Run basic tests for charges calculation"""
    print("Testing charges calculation...")
    
    # Test case: Buy 100 shares @ 100, sell @ 102
    charges = calculate_total_charges(
        buy_price=100.0,
        sell_price=102.0,
        quantity=100
    )
    
    print(charges)
    
    assert charges.turnover == 20200.0, f"Expected turnover 20200, got {charges.turnover}"
    assert charges.total_charges > 0, "Total charges should be positive"
    assert charges.breakeven_points > 0, "Breakeven should be positive"
    
    # Test position sizing
    qty, value, risk = estimate_position_size(
        capital=100000,
        entry_price=100,
        stop_loss=95,
        max_risk_pct=0.01
    )
    
    print(f"\nPosition sizing test:")
    print(f"  Quantity: {qty}")
    print(f"  Position value: ₹{value:,.2f}")
    print(f"  Risk amount: ₹{risk:,.2f}")
    
    assert qty > 0, "Quantity should be positive"
    assert risk <= 1000, f"Risk should be <= 1000 (1%), got {risk}"
    
    # Test breakeven calculation
    breakeven = calculate_breakeven_points(100.0, 100)
    print(f"\nBreakeven for 100 shares @ ₹100: {breakeven:.4f} points")
    
    print("\n✅ All charges tests passed!")


if __name__ == "__main__":
    _test_charges()
