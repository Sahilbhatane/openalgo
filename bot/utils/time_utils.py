"""
Time Utilities
==============
IST timezone handling, market hours checking, trading day validation.
"""

from datetime import datetime, date, time, timedelta
from typing import Optional
import pytz
from functools import lru_cache

# India Standard Time
IST = pytz.timezone("Asia/Kolkata")

# Market timing constants
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_MARKET_OPEN = time(9, 0)
POST_MARKET_CLOSE = time(16, 0)

# Bot's trading window (conservative)
BOT_START = time(9, 20)  # Wait 5 min after open
BOT_SQUARE_OFF = time(15, 10)  # Square off 20 min before close
BOT_NO_NEW_TRADES = time(14, 30)  # No new trades after 2:30 PM


def get_ist_now() -> datetime:
    """Get current time in IST"""
    return datetime.now(IST)


def get_today_date() -> date:
    """Get today's date in IST"""
    return get_ist_now().date()


def get_current_time() -> time:
    """Get current time (time only) in IST"""
    return get_ist_now().time()


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if market is currently open.
    
    Args:
        dt: Datetime to check (default: now in IST)
    
    Returns:
        True if within market hours (9:15 AM - 3:30 PM IST)
    """
    if dt is None:
        dt = get_ist_now()
    
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    
    current_time = dt.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_trading_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if within bot's trading hours.
    
    More conservative than market hours:
    - Start: 9:20 AM (after initial volatility)
    - End: 3:10 PM (before close volatility)
    
    Args:
        dt: Datetime to check (default: now in IST)
    
    Returns:
        True if within bot's trading window
    """
    if dt is None:
        dt = get_ist_now()
    
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    
    current_time = dt.time()
    return BOT_START <= current_time <= BOT_SQUARE_OFF


def can_open_new_trades(dt: Optional[datetime] = None) -> bool:
    """
    Check if new trades can be opened.
    
    Stops opening new trades at 2:30 PM to allow time for exits.
    
    Args:
        dt: Datetime to check (default: now in IST)
    
    Returns:
        True if new trades can be opened
    """
    if dt is None:
        dt = get_ist_now()
    
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    
    current_time = dt.time()
    return BOT_START <= current_time <= BOT_NO_NEW_TRADES


def time_to_market_open(dt: Optional[datetime] = None) -> timedelta:
    """
    Calculate time remaining until market opens.
    
    Args:
        dt: Datetime to check (default: now in IST)
    
    Returns:
        timedelta until market open (negative if already open)
    """
    if dt is None:
        dt = get_ist_now()
    
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    
    today_open = dt.replace(
        hour=MARKET_OPEN.hour,
        minute=MARKET_OPEN.minute,
        second=0,
        microsecond=0
    )
    
    if dt.time() >= MARKET_CLOSE:
        # Market closed, calculate for next day
        next_day = dt + timedelta(days=1)
        next_trading_day = _get_next_trading_day(next_day.date())
        today_open = datetime.combine(next_trading_day, MARKET_OPEN)
        today_open = IST.localize(today_open)
    
    return today_open - dt


def time_to_market_close(dt: Optional[datetime] = None) -> timedelta:
    """
    Calculate time remaining until market closes.
    
    Args:
        dt: Datetime to check (default: now in IST)
    
    Returns:
        timedelta until market close
    """
    if dt is None:
        dt = get_ist_now()
    
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    
    today_close = dt.replace(
        hour=MARKET_CLOSE.hour,
        minute=MARKET_CLOSE.minute,
        second=0,
        microsecond=0
    )
    
    return today_close - dt


def time_to_square_off(dt: Optional[datetime] = None) -> timedelta:
    """
    Calculate time remaining until bot's square off time (3:10 PM).
    
    Args:
        dt: Datetime to check (default: now in IST)
    
    Returns:
        timedelta until square off
    """
    if dt is None:
        dt = get_ist_now()
    
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    
    square_off_time = dt.replace(
        hour=BOT_SQUARE_OFF.hour,
        minute=BOT_SQUARE_OFF.minute,
        second=0,
        microsecond=0
    )
    
    return square_off_time - dt


def is_trading_day(dt: Optional[date] = None) -> bool:
    """
    Check if a date is a trading day (not weekend or holiday).
    
    Args:
        dt: Date to check (default: today in IST)
    
    Returns:
        True if it's a trading day
    """
    if dt is None:
        dt = get_today_date()
    
    # Check weekend
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check holidays (this is a simplified list - in production, 
    # fetch from NSE or maintain a complete list)
    if dt in _get_holidays(dt.year):
        return False
    
    return True


@lru_cache(maxsize=10)
def _get_holidays(year: int) -> set:
    """
    Get NSE holidays for a year.
    
    Note: This is a simplified list. In production, fetch from:
    - NSE website
    - Angel One API
    - Maintain in database
    
    Args:
        year: Year to get holidays for
    
    Returns:
        Set of holiday dates
    """
    # Major NSE holidays (approximate - varies by year)
    # TODO: Replace with API call or database lookup
    holidays = set()
    
    if year == 2024:
        holidays = {
            date(2024, 1, 26),   # Republic Day
            date(2024, 3, 8),    # Mahashivratri
            date(2024, 3, 25),   # Holi
            date(2024, 3, 29),   # Good Friday
            date(2024, 4, 11),   # Id-ul-Fitr
            date(2024, 4, 14),   # Dr. Ambedkar Jayanti
            date(2024, 4, 17),   # Ram Navami
            date(2024, 4, 21),   # Mahavir Jayanti
            date(2024, 5, 1),    # Maharashtra Day
            date(2024, 5, 23),   # Buddha Purnima
            date(2024, 6, 17),   # Bakri Id
            date(2024, 7, 17),   # Muharram
            date(2024, 8, 15),   # Independence Day
            date(2024, 10, 2),   # Gandhi Jayanti
            date(2024, 10, 12),  # Dussehra
            date(2024, 11, 1),   # Diwali Laxmi Pujan
            date(2024, 11, 15),  # Gurunanak Jayanti
            date(2024, 12, 25),  # Christmas
        }
    elif year == 2025:
        holidays = {
            date(2025, 1, 26),   # Republic Day
            date(2025, 2, 26),   # Mahashivratri
            date(2025, 3, 14),   # Holi
            date(2025, 3, 31),   # Id-ul-Fitr
            date(2025, 4, 6),    # Ram Navami
            date(2025, 4, 10),   # Mahavir Jayanti
            date(2025, 4, 14),   # Dr. Ambedkar Jayanti
            date(2025, 4, 18),   # Good Friday
            date(2025, 5, 1),    # Maharashtra Day
            date(2025, 5, 12),   # Buddha Purnima
            date(2025, 6, 7),    # Bakri Id
            date(2025, 8, 15),   # Independence Day
            date(2025, 8, 27),   # Janmashtami
            date(2025, 10, 2),   # Gandhi Jayanti/Dussehra
            date(2025, 10, 21),  # Diwali Laxmi Pujan
            date(2025, 11, 5),   # Gurunanak Jayanti
            date(2025, 12, 25),  # Christmas
        }
    
    return holidays


def _get_next_trading_day(dt: date) -> date:
    """Get the next trading day from a given date"""
    next_day = dt
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day


def get_previous_trading_day(dt: Optional[date] = None) -> date:
    """Get the previous trading day"""
    if dt is None:
        dt = get_today_date()
    
    prev_day = dt - timedelta(days=1)
    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)
    
    return prev_day


def get_next_trading_day(dt: Optional[date] = None) -> date:
    """Get the next trading day"""
    if dt is None:
        dt = get_today_date()
    
    return _get_next_trading_day(dt + timedelta(days=1))


def format_time_remaining(td: timedelta) -> str:
    """
    Format a timedelta as human-readable string.
    
    Args:
        td: timedelta to format
    
    Returns:
        Human-readable string like "2h 30m" or "45m 30s"
    """
    total_seconds = int(td.total_seconds())
    
    if total_seconds < 0:
        return "Already passed"
    
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_market_status() -> str:
    """
    Get current market status as human-readable string.
    
    Returns:
        Status string like "OPEN (2h 15m to close)" or "CLOSED"
    """
    now = get_ist_now()
    
    if not is_trading_day(now.date()):
        next_day = get_next_trading_day(now.date())
        return f"CLOSED (Holiday/Weekend) - Opens {next_day.strftime('%A, %d %b')}"
    
    current_time = now.time()
    
    if current_time < PRE_MARKET_OPEN:
        time_to_open = time_to_market_open(now)
        return f"PRE-MARKET - Opens in {format_time_remaining(time_to_open)}"
    
    if current_time < MARKET_OPEN:
        time_to_open = time_to_market_open(now)
        return f"PRE-MARKET SESSION - Trading in {format_time_remaining(time_to_open)}"
    
    if current_time <= MARKET_CLOSE:
        time_to_close = time_to_market_close(now)
        return f"MARKET OPEN - Closes in {format_time_remaining(time_to_close)}"
    
    if current_time <= POST_MARKET_CLOSE:
        return "POST-MARKET SESSION"
    
    next_day = get_next_trading_day(now.date())
    return f"CLOSED - Opens {next_day.strftime('%A, %d %b')} at 9:15 AM"


# ============================================================================
# UNIT TESTS
# ============================================================================

def _test_time_utils():
    """Run basic tests for time utilities"""
    print("Testing time utilities...")
    
    # Test IST now
    now = get_ist_now()
    print(f"Current IST: {now}")
    assert now.tzinfo is not None, "Should be timezone aware"
    
    # Test market status
    status = get_market_status()
    print(f"Market status: {status}")
    
    # Test trading day
    today = get_today_date()
    is_today_trading = is_trading_day(today)
    print(f"Is today ({today}) a trading day: {is_today_trading}")
    
    # Test next trading day
    next_td = get_next_trading_day(today)
    print(f"Next trading day: {next_td}")
    
    # Test time formatting
    print(f"Format 2h 30m: {format_time_remaining(timedelta(hours=2, minutes=30))}")
    print(f"Format 45m 30s: {format_time_remaining(timedelta(minutes=45, seconds=30))}")
    
    # Test specific dates
    saturday = date(2024, 12, 14)  # A Saturday
    assert not is_trading_day(saturday), "Saturday should not be trading day"
    
    print("\n✅ All time utilities tests passed!")


if __name__ == "__main__":
    _test_time_utils()
