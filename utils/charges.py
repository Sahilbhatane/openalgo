"""
Trading Charges Calculator for OpenAlgo

This module provides utility functions to calculate various trading charges including:
- Brokerage charges
- GST on brokerage and transaction charges
- Securities Transaction Tax (STT)
- Exchange transaction charges
- API subscription amortization
- Total trade cost analysis
- Breakeven calculations

Author: OpenAlgo Team
Date: December 2025
"""

from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

# ==================== Configuration Constants ====================

# Brokerage rates (can be overridden)
DEFAULT_BROKERAGE_RATE = 0.0003  # 0.03% or ₹20 per executed order (whichever is lower)
DEFAULT_BROKERAGE_CAP = 20.0  # Maximum ₹20 per order

# Tax rates
GST_RATE = 0.18  # 18% GST on brokerage and transaction charges

# STT (Securities Transaction Tax) rates
STT_EQUITY_DELIVERY = 0.001  # 0.1% on both buy and sell
STT_EQUITY_INTRADAY_SELL = 0.00025  # 0.025% on sell side only
STT_EQUITY_OPTIONS_SELL = 0.0005  # 0.05% on sell side (on premium)
STT_FUTURES = 0.00001  # 0.001% on sell side only

# Exchange transaction charges (approximate rates - vary by exchange)
EXCHANGE_TXN_CHARGE_RATE = 0.0000325  # ~0.00325% for NSE equity
EXCHANGE_TXN_CHARGE_FO_RATE = 0.000019  # ~0.0019% for NSE F&O

# SEBI charges
SEBI_CHARGES_RATE = 0.000001  # ₹10 per crore

# Stamp duty
STAMP_DUTY_RATE = 0.00003  # 0.003% or ₹300 per crore on buy side

# Default trading days per month
DEFAULT_TRADING_DAYS_PER_MONTH = 20


# ==================== Core Calculation Functions ====================


def brokerage(
    turnover: float,
    brokerage_rate: float = DEFAULT_BROKERAGE_RATE,
    cap_per_order: float = DEFAULT_BROKERAGE_CAP,
) -> float:
    """
    Calculate brokerage charges

    Brokerage is typically charged as a percentage of turnover or a flat fee per order,
    whichever is lower. For discount brokers, it's often capped at ₹20 per executed order.

    Args:
        turnover: Total turnover value (price × quantity)
        brokerage_rate: Brokerage rate as decimal (default: 0.03%)
        cap_per_order: Maximum brokerage cap per order (default: ₹20)

    Returns:
        float: Brokerage amount in rupees

    Example:
        >>> brokerage(100000)  # ₹1 lakh turnover
        20.0  # Capped at ₹20

        >>> brokerage(50000)  # ₹50k turnover
        15.0  # 0.03% of 50000
    """
    if turnover <= 0:
        return 0.0

    calculated_brokerage = turnover * brokerage_rate
    return min(calculated_brokerage, cap_per_order)


def gst(
    brokerage_amount: float, txn_charges: float, gst_rate: float = GST_RATE
) -> float:
    """
    Calculate GST on brokerage and transaction charges

    GST is levied at 18% on the sum of brokerage and transaction charges.

    Args:
        brokerage_amount: Brokerage charge amount
        txn_charges: Transaction charges (exchange + clearing + SEBI)
        gst_rate: GST rate as decimal (default: 18%)

    Returns:
        float: GST amount in rupees

    Example:
        >>> gst(20.0, 10.0)  # ₹20 brokerage + ₹10 txn charges
        5.4  # 18% of 30
    """
    taxable_amount = brokerage_amount + txn_charges
    return taxable_amount * gst_rate


def stt(
    turnover: float, transaction_type: str = "equity_intraday", side: str = "sell"
) -> float:
    """
    Calculate Securities Transaction Tax (STT)

    STT rates vary based on:
    - Segment (equity delivery, intraday, F&O)
    - Side (buy/sell)

    Args:
        turnover: Total turnover value
        transaction_type: Type of transaction
            - "equity_delivery": Equity delivery trades
            - "equity_intraday": Equity intraday trades
            - "equity_options": Equity options
            - "futures": Futures contracts
        side: Transaction side - "buy" or "sell"

    Returns:
        float: STT amount in rupees

    Example:
        >>> stt(100000, "equity_intraday", "sell")
        25.0  # 0.025% of 100000

        >>> stt(100000, "equity_delivery", "buy")
        100.0  # 0.1% of 100000
    """
    if turnover <= 0:
        return 0.0

    transaction_type = transaction_type.lower()
    side = side.lower()

    if transaction_type == "equity_delivery":
        # STT on both buy and sell for delivery
        return turnover * STT_EQUITY_DELIVERY

    elif transaction_type == "equity_intraday":
        # STT only on sell side for intraday
        if side == "sell":
            return turnover * STT_EQUITY_INTRADAY_SELL
        return 0.0

    elif transaction_type == "equity_options":
        # STT only on sell side for options
        if side == "sell":
            return turnover * STT_EQUITY_OPTIONS_SELL
        return 0.0

    elif transaction_type == "futures":
        # STT only on sell side for futures
        if side == "sell":
            return turnover * STT_FUTURES
        return 0.0

    else:
        # Default to equity intraday sell
        return turnover * STT_EQUITY_INTRADAY_SELL if side == "sell" else 0.0


def exchange_fee(
    turnover: float,
    segment: str = "equity",
    include_sebi: bool = True,
    include_stamp_duty: bool = True,
    side: str = "buy",
) -> float:
    """
    Calculate exchange transaction fees

    Includes:
    - Exchange transaction charges
    - SEBI turnover charges
    - Stamp duty (on buy side)

    Args:
        turnover: Total turnover value
        segment: Trading segment - "equity" or "fo" (futures & options)
        include_sebi: Include SEBI charges (default: True)
        include_stamp_duty: Include stamp duty (default: True)
        side: Transaction side - "buy" or "sell"

    Returns:
        float: Total exchange fees in rupees

    Example:
        >>> exchange_fee(100000, "equity", side="buy")
        6.25  # Exchange + SEBI + Stamp duty
    """
    if turnover <= 0:
        return 0.0

    total_fee = 0.0
    segment = segment.lower()
    side = side.lower()

    # Exchange transaction charges
    if segment == "fo" or segment == "f&o":
        total_fee += turnover * EXCHANGE_TXN_CHARGE_FO_RATE
    else:
        total_fee += turnover * EXCHANGE_TXN_CHARGE_RATE

    # SEBI charges
    if include_sebi:
        total_fee += turnover * SEBI_CHARGES_RATE

    # Stamp duty (only on buy side)
    if include_stamp_duty and side == "buy":
        total_fee += turnover * STAMP_DUTY_RATE

    return total_fee


def api_daily_amort(
    subscription_monthly: float, trading_days: int = DEFAULT_TRADING_DAYS_PER_MONTH
) -> float:
    """
    Calculate daily amortization of monthly API subscription cost

    Args:
        subscription_monthly: Monthly subscription cost
        trading_days: Number of trading days in a month (default: 20)

    Returns:
        float: Daily amortized cost

    Example:
        >>> api_daily_amort(1000, 20)
        50.0  # ₹1000 / 20 days
    """
    if subscription_monthly <= 0 or trading_days <= 0:
        return 0.0

    return subscription_monthly / trading_days


def sebi_charges(turnover: float) -> float:
    """
    Calculate SEBI turnover charges

    SEBI charges ₹10 per crore of turnover.

    Args:
        turnover: Total turnover value

    Returns:
        float: SEBI charges in rupees

    Example:
        >>> sebi_charges(10000000)  # ₹1 crore
        10.0
    """
    return turnover * SEBI_CHARGES_RATE


def stamp_duty(turnover: float) -> float:
    """
    Calculate stamp duty (on buy side only)

    Stamp duty is 0.003% or ₹300 per crore on buy side.

    Args:
        turnover: Total turnover value

    Returns:
        float: Stamp duty in rupees

    Example:
        >>> stamp_duty(100000)
        3.0
    """
    return turnover * STAMP_DUTY_RATE


# ==================== Comprehensive Cost Calculations ====================


def total_trade_cost(
    turnover: float,
    transaction_type: str = "equity_intraday",
    side: str = "buy",
    subscription_monthly: float = 0.0,
    trading_days_per_month: int = DEFAULT_TRADING_DAYS_PER_MONTH,
    brokerage_rate: float = DEFAULT_BROKERAGE_RATE,
    brokerage_cap: float = DEFAULT_BROKERAGE_CAP,
) -> dict:
    """
    Calculate total cost of a trade including all charges

    This function provides a comprehensive breakdown of all trading costs:
    - Brokerage
    - STT
    - Exchange transaction charges
    - SEBI charges
    - Stamp duty (if buy side)
    - GST
    - Amortized API subscription cost

    Args:
        turnover: Total turnover value (price × quantity)
        transaction_type: Type of transaction (equity_delivery, equity_intraday, etc.)
        side: Transaction side - "buy" or "sell"
        subscription_monthly: Monthly API subscription cost (default: 0)
        trading_days_per_month: Trading days per month (default: 20)
        brokerage_rate: Brokerage rate as decimal (default: 0.03%)
        brokerage_cap: Maximum brokerage cap (default: ₹20)

    Returns:
        dict: Detailed breakdown of all costs

    Example:
        >>> result = total_trade_cost(100000, "equity_intraday", "sell", 1000)
        >>> print(f"Total cost: ₹{result['total_cost']:.2f}")
        Total cost: ₹78.65
    """
    if turnover <= 0:
        return {
            "turnover": 0.0,
            "brokerage": 0.0,
            "stt": 0.0,
            "exchange_charges": 0.0,
            "sebi_charges": 0.0,
            "stamp_duty": 0.0,
            "gst": 0.0,
            "api_cost_daily": 0.0,
            "total_cost": 0.0,
            "cost_percentage": 0.0,
        }

    # Calculate individual components
    brokerage_amt = brokerage(turnover, brokerage_rate, brokerage_cap)
    stt_amt = stt(turnover, transaction_type, side)

    # Exchange fees breakdown
    segment = (
        "fo"
        if "option" in transaction_type or "future" in transaction_type
        else "equity"
    )
    exchange_charges = exchange_fee(
        turnover, segment, include_sebi=False, include_stamp_duty=False, side=side
    )
    sebi_amt = sebi_charges(turnover)
    stamp_duty_amt = stamp_duty(turnover) if side.lower() == "buy" else 0.0

    # GST on brokerage and transaction charges
    txn_charges = exchange_charges + sebi_amt
    gst_amt = gst(brokerage_amt, txn_charges)

    # API subscription amortization
    api_cost = api_daily_amort(subscription_monthly, trading_days_per_month)

    # Total cost
    total = (
        brokerage_amt
        + stt_amt
        + exchange_charges
        + sebi_amt
        + stamp_duty_amt
        + gst_amt
        + api_cost
    )

    return {
        "turnover": round(turnover, 2),
        "brokerage": round(brokerage_amt, 2),
        "stt": round(stt_amt, 2),
        "exchange_charges": round(exchange_charges, 2),
        "sebi_charges": round(sebi_amt, 2),
        "stamp_duty": round(stamp_duty_amt, 2),
        "gst": round(gst_amt, 2),
        "api_cost_daily": round(api_cost, 2),
        "total_cost": round(total, 2),
        "cost_percentage": round((total / turnover) * 100, 4),
    }


def round_trip_cost(
    turnover: float,
    transaction_type: str = "equity_intraday",
    subscription_monthly: float = 0.0,
    trading_days_per_month: int = DEFAULT_TRADING_DAYS_PER_MONTH,
    brokerage_rate: float = DEFAULT_BROKERAGE_RATE,
    brokerage_cap: float = DEFAULT_BROKERAGE_CAP,
) -> dict:
    """
    Calculate round-trip cost (buy + sell)

    Args:
        turnover: Turnover for one leg (same for both buy and sell)
        transaction_type: Type of transaction
        subscription_monthly: Monthly API subscription cost
        trading_days_per_month: Trading days per month
        brokerage_rate: Brokerage rate as decimal
        brokerage_cap: Maximum brokerage cap

    Returns:
        dict: Combined costs for buy and sell

    Example:
        >>> result = round_trip_cost(100000, "equity_intraday", 1000)
        >>> print(f"Round trip cost: ₹{result['total_cost']:.2f}")
    """
    # Calculate buy side
    buy_cost = total_trade_cost(
        turnover,
        transaction_type,
        "buy",
        subscription_monthly / 2,  # Split API cost between buy and sell
        trading_days_per_month,
        brokerage_rate,
        brokerage_cap,
    )

    # Calculate sell side
    sell_cost = total_trade_cost(
        turnover,
        transaction_type,
        "sell",
        subscription_monthly / 2,
        trading_days_per_month,
        brokerage_rate,
        brokerage_cap,
    )

    # Combine costs
    return {
        "turnover": round(turnover * 2, 2),  # Total turnover for round trip
        "buy_cost": buy_cost["total_cost"],
        "sell_cost": sell_cost["total_cost"],
        "total_cost": round(buy_cost["total_cost"] + sell_cost["total_cost"], 2),
        "cost_percentage": round(
            ((buy_cost["total_cost"] + sell_cost["total_cost"]) / (turnover * 2)) * 100,
            4,
        ),
        "buy_breakdown": buy_cost,
        "sell_breakdown": sell_cost,
    }


def per_trade_gross_needed(
    balance: float,
    target_net: float,
    trades_per_day: int,
    subscription_daily: float,
    transaction_type: str = "equity_intraday",
    brokerage_rate: float = DEFAULT_BROKERAGE_RATE,
    brokerage_cap: float = DEFAULT_BROKERAGE_CAP,
) -> dict:
    """
    Calculate gross profit needed per trade to achieve target net profit

    This function helps determine the minimum profit per trade needed to:
    1. Cover all trading costs (brokerage, STT, GST, etc.)
    2. Cover daily API subscription costs
    3. Achieve target net profit for the day

    Args:
        balance: Available trading balance
        target_net: Target net profit for the day
        trades_per_day: Number of trades planned per day
        subscription_daily: Daily API subscription cost
        transaction_type: Type of transaction
        brokerage_rate: Brokerage rate as decimal
        brokerage_cap: Maximum brokerage cap

    Returns:
        dict: Breakdown of required gross profit and costs

    Example:
        >>> result = per_trade_gross_needed(100000, 1000, 10, 50)
        >>> print(f"Gross profit needed per trade: ₹{result['gross_per_trade']:.2f}")
    """
    if trades_per_day <= 0:
        return {"error": "trades_per_day must be greater than 0"}

    # Assume average turnover per trade (using full balance / trades)
    avg_turnover_per_trade = balance / trades_per_day

    # Calculate round trip cost for one trade
    trade_cost = round_trip_cost(
        avg_turnover_per_trade,
        transaction_type,
        subscription_daily,
        1,  # Already daily cost
        brokerage_rate,
        brokerage_cap,
    )

    # Total costs per day
    total_trading_costs = trade_cost["total_cost"] * trades_per_day

    # Total gross needed for the day
    total_gross_needed = target_net + total_trading_costs

    # Per trade gross profit needed
    gross_per_trade = total_gross_needed / trades_per_day

    # Breakeven (just to cover costs)
    breakeven_per_trade = total_trading_costs / trades_per_day

    return {
        "balance": round(balance, 2),
        "target_net_daily": round(target_net, 2),
        "trades_per_day": trades_per_day,
        "avg_turnover_per_trade": round(avg_turnover_per_trade, 2),
        "cost_per_trade": round(trade_cost["total_cost"], 2),
        "total_daily_costs": round(total_trading_costs, 2),
        "subscription_daily": round(subscription_daily, 2),
        "breakeven_per_trade": round(breakeven_per_trade, 2),
        "gross_per_trade": round(gross_per_trade, 2),
        "gross_percentage": round((gross_per_trade / avg_turnover_per_trade) * 100, 4),
        "total_gross_needed": round(total_gross_needed, 2),
    }


def breakeven_analysis(
    capital: float,
    subscription_monthly: float,
    trades_per_day: int = 10,
    trading_days_per_month: int = DEFAULT_TRADING_DAYS_PER_MONTH,
    transaction_type: str = "equity_intraday",
) -> dict:
    """
    Perform breakeven analysis for a trading strategy

    Calculates:
    - Monthly costs
    - Required daily profit to breakeven
    - Required profit per trade
    - Win rate needed with different reward:risk ratios

    Args:
        capital: Trading capital
        subscription_monthly: Monthly API/platform subscription
        trades_per_day: Average trades per day
        trading_days_per_month: Trading days per month
        transaction_type: Type of transaction

    Returns:
        dict: Comprehensive breakeven analysis

    Example:
        >>> analysis = breakeven_analysis(500000, 1000, 20, 20)
        >>> print(f"Monthly breakeven: ₹{analysis['monthly_breakeven']:.2f}")
    """
    # Calculate daily subscription cost
    subscription_daily = api_daily_amort(subscription_monthly, trading_days_per_month)

    # Average turnover per trade
    avg_turnover = capital / trades_per_day

    # Calculate cost per round trip
    rt_cost = round_trip_cost(avg_turnover, transaction_type, subscription_daily, 1)

    # Total monthly trading costs
    total_trades_monthly = trades_per_day * trading_days_per_month
    monthly_trading_costs = rt_cost["total_cost"] * total_trades_monthly

    # Total monthly costs
    monthly_total_costs = monthly_trading_costs + subscription_monthly

    # Daily breakeven
    daily_breakeven = monthly_total_costs / trading_days_per_month

    # Per trade breakeven
    per_trade_breakeven = daily_breakeven / trades_per_day

    return {
        "capital": round(capital, 2),
        "subscription_monthly": round(subscription_monthly, 2),
        "trades_per_day": trades_per_day,
        "trading_days_per_month": trading_days_per_month,
        "avg_turnover_per_trade": round(avg_turnover, 2),
        "cost_per_round_trip": round(rt_cost["total_cost"], 2),
        "monthly_trading_costs": round(monthly_trading_costs, 2),
        "monthly_subscription": round(subscription_monthly, 2),
        "monthly_total_costs": round(monthly_total_costs, 2),
        "monthly_breakeven": round(monthly_total_costs, 2),
        "daily_breakeven": round(daily_breakeven, 2),
        "per_trade_breakeven": round(per_trade_breakeven, 2),
        "breakeven_percentage": round((per_trade_breakeven / avg_turnover) * 100, 4),
    }


# ==================== Utility Functions ====================


def format_currency(amount: float) -> str:
    """
    Format amount as Indian currency

    Args:
        amount: Amount to format

    Returns:
        str: Formatted currency string

    Example:
        >>> format_currency(1234.56)
        '₹1,234.56'
    """
    return f"₹{amount:,.2f}"


def print_cost_breakdown(cost_dict: dict, title: str = "Trade Cost Breakdown"):
    """
    Pretty print cost breakdown

    Args:
        cost_dict: Dictionary from total_trade_cost() or round_trip_cost()
        title: Title for the breakdown
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    for key, value in cost_dict.items():
        if (
            isinstance(value, (int, float))
            and key != "cost_percentage"
            and key != "gross_percentage"
        ):
            print(f"{key.replace('_', ' ').title():<30} {format_currency(value):>18}")
        elif key == "cost_percentage" or key == "gross_percentage":
            print(f"{key.replace('_', ' ').title():<30} {value:>17.4f}%")

    print(f"{'=' * 50}\n")


# ==================== Example Usage ====================


if __name__ == "__main__":
    """
    Example usage of the charges module
    """

    print("=" * 60)
    print("OpenAlgo Trading Charges Calculator".center(60))
    print("=" * 60)

    # Example 1: Single trade cost
    print("\n1. Single Intraday Trade (₹1 Lakh turnover)")
    cost = total_trade_cost(
        turnover=100000,
        transaction_type="equity_intraday",
        side="sell",
        subscription_monthly=1000,
    )
    print_cost_breakdown(cost, "Single Trade Cost")

    # Example 2: Round trip cost
    print("\n2. Round Trip (Buy + Sell) - ₹1 Lakh each leg")
    rt = round_trip_cost(100000, "equity_intraday", 1000)
    print_cost_breakdown(rt, "Round Trip Cost")

    # Example 3: Gross profit needed
    print("\n3. Gross Profit Required Per Trade")
    gross = per_trade_gross_needed(
        balance=500000, target_net=2000, trades_per_day=10, subscription_daily=50
    )
    print_cost_breakdown(gross, "Daily Trading Plan")

    # Example 4: Breakeven analysis
    print("\n4. Monthly Breakeven Analysis")
    be = breakeven_analysis(
        capital=500000, subscription_monthly=1000, trades_per_day=20
    )
    print_cost_breakdown(be, "Breakeven Analysis")
