"""
Trading Constants
=================
All magic numbers and configuration constants in one place.
"""

from dataclasses import dataclass
from datetime import time
from typing import Final

# ============================================================================
# DEFAULT CAPITAL AND RISK SETTINGS
# ============================================================================
DEFAULT_CAPITAL: Final[float] = 100_000.0  # INR
DEFAULT_MAX_RISK_PER_TRADE: Final[float] = 0.01  # 1% of capital per trade
DEFAULT_MAX_DAILY_DRAWDOWN: Final[float] = 0.05  # 5% max daily drawdown


@dataclass(frozen=True)
class MarketHours:
    """NSE/BSE Market Hours (IST)"""
    
    # Pre-market session
    PRE_MARKET_OPEN: time = time(9, 0)
    PRE_MARKET_CLOSE: time = time(9, 8)
    
    # Normal trading session
    MARKET_OPEN: time = time(9, 15)
    MARKET_CLOSE: time = time(15, 30)
    
    # Post-market session
    POST_MARKET_OPEN: time = time(15, 40)
    POST_MARKET_CLOSE: time = time(16, 0)
    
    # Our bot's trading window (conservative)
    BOT_START: time = time(9, 20)  # Wait 5 min after open for stability
    BOT_SQUARE_OFF: time = time(15, 10)  # Square off 20 min before close
    BOT_STOP_NEW_TRADES: time = time(14, 30)  # No new trades after 2:30 PM


@dataclass(frozen=True)
class TradingSession:
    """Session timing for scheduled jobs"""
    
    # Morning preparation
    MORNING_LEARNING: time = time(8, 0)  # Fetch data, analyze, create plan
    PRE_MARKET_ANALYSIS: time = time(9, 5)  # Final checks before market open
    
    # Trading session
    MARKET_OPEN_CHECK: time = time(9, 15)  # Verify connection and start
    FIRST_CANDLE_READY: time = time(9, 20)  # First 5-min candle closed
    
    # End of day
    SQUARE_OFF: time = time(15, 10)  # Close all positions
    EOD_REPORT: time = time(15, 35)  # Generate daily report
    
    # Cleanup
    DAILY_CLEANUP: time = time(16, 0)  # Clear caches, backup logs


@dataclass(frozen=True)
class Limits:
    """Trading and risk limits"""
    
    # Position limits
    MAX_POSITIONS: int = 3  # Maximum concurrent positions
    MAX_POSITION_SIZE_PCT: float = 0.30  # Max 30% of capital in single position
    MIN_POSITION_SIZE: float = 1000.0  # Minimum position value in INR
    
    # Order limits
    MAX_ORDERS_PER_DAY: int = 20  # Circuit breaker
    MAX_ORDER_VALUE: float = 50_000.0  # Max single order value
    MIN_ORDER_QTY: int = 1  # Minimum quantity
    
    # Risk limits
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    MAX_SINGLE_TRADE_LOSS: float = 0.01  # 1% max loss per trade
    MAX_DRAWDOWN_PCT: float = 0.10  # 10% max drawdown from peak
    
    # Slippage assumptions
    EXPECTED_SLIPPAGE_PCT: float = 0.0010  # 0.1% expected slippage
    MAX_SLIPPAGE_PCT: float = 0.0050  # 0.5% max acceptable slippage
    
    # Retry limits
    MAX_ORDER_RETRIES: int = 3
    MAX_API_RETRIES: int = 5


@dataclass(frozen=True)
class Timeouts:
    """API and connection timeouts in seconds"""
    
    API_REQUEST: int = 30
    WEBSOCKET_CONNECT: int = 10
    WEBSOCKET_READ: int = 60
    ORDER_CONFIRMATION: int = 5
    DATA_STALE_THRESHOLD: int = 3  # Warn if no data for 3 seconds


# ============================================================================
# STRATEGY CONSTANTS
# ============================================================================

@dataclass(frozen=True)
class VolatilityBreakoutParams:
    """Parameters for the volatility breakout strategy"""
    
    # Opening range
    OPENING_RANGE_MINUTES: int = 15  # First 15 minutes to define range
    
    # ATR settings
    ATR_PERIOD: int = 14  # ATR lookback period
    ATR_MULTIPLIER_SL: float = 1.5  # Stop loss = ATR * multiplier
    
    # VWAP confirmation
    REQUIRE_VWAP_CONFIRMATION: bool = True  # Price must be on right side of VWAP
    
    # Target settings
    RISK_REWARD_RATIO: float = 1.5  # Target = 1.5 * risk
    
    # Volume filter
    VOLUME_SURGE_MULTIPLIER: float = 1.5  # Volume must be 1.5x average
    
    # Breakout threshold
    BREAKOUT_BUFFER_PCT: float = 0.001  # 0.1% buffer above/below range


@dataclass(frozen=True)
class IndicatorParams:
    """Default indicator parameters"""
    
    # Moving averages
    EMA_SHORT: int = 9
    EMA_MEDIUM: int = 21
    EMA_LONG: int = 50
    
    # RSI
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30
    
    # ATR
    ATR_PERIOD: int = 14
    
    # VWAP (calculated from day start, no period needed)
    
    # Bollinger Bands
    BB_PERIOD: int = 20
    BB_STD_DEV: float = 2.0


# ============================================================================
# ANGEL ONE API LIMITS (from SmartAPI documentation)
# ============================================================================

@dataclass(frozen=True)
class AngelOneLimits:
    """AngelOne SmartAPI rate limits and constraints"""
    
    # Rate limits
    ORDERS_PER_SECOND: int = 10
    HISTORICAL_REQUESTS_PER_MINUTE: int = 60
    WEBSOCKET_MAX_TOKENS: int = 3000  # Max 3000 tokens per connection
    
    # Historical data limits
    MAX_CANDLES_1MIN: int = 30 * 1440  # 30 days of 1-min candles
    MAX_CANDLES_5MIN: int = 90 * 288  # 90 days of 5-min candles
    MAX_CANDLES_15MIN: int = 180 * 96  # 180 days of 15-min candles
    MAX_CANDLES_1DAY: int = 2000  # 2000 daily candles
    
    # Session validity
    SESSION_VALID_HOURS: int = 6  # Re-login after 6 hours


# ============================================================================
# CHARGES AND TAXES (as of 2024)
# ============================================================================

@dataclass(frozen=True)
class BrokerageCharges:
    """AngelOne brokerage and regulatory charges"""
    
    # Brokerage (AngelOne is typically 0 for delivery, flat for intraday)
    EQUITY_INTRADAY_PCT: float = 0.0003  # 0.03% or Rs 20 max
    EQUITY_INTRADAY_MAX: float = 20.0  # Max Rs 20 per order
    
    # STT (Securities Transaction Tax)
    STT_INTRADAY_SELL: float = 0.00025  # 0.025% on sell side only
    
    # Exchange Transaction Charges
    NSE_TXN_CHARGE: float = 0.0000322  # 0.00322% 
    BSE_TXN_CHARGE: float = 0.0000375  # 0.00375%
    
    # GST (on brokerage + transaction charges)
    GST_RATE: float = 0.18  # 18% GST
    
    # SEBI Charges
    SEBI_CHARGE: float = 0.000001  # Rs 10 per crore
    
    # Stamp Duty (on buy side only)
    STAMP_DUTY_RATE: float = 0.00015  # 0.015% for equity intraday
    
    # DP Charges (not applicable for intraday)
    DP_CHARGE: float = 0.0


# ============================================================================
# FILE AND PATH CONSTANTS
# ============================================================================

# Relative to project root
DATA_DIR: Final[str] = "bot_data"
LOGS_DIR: Final[str] = "bot_data/logs"
REPORTS_DIR: Final[str] = "bot_data/reports"
PLANS_DIR: Final[str] = "bot_data/plans"
TRADES_DIR: Final[str] = "bot_data/trades"
MODELS_DIR: Final[str] = "bot_data/models"

# File names
TODAY_PLAN_FILE: Final[str] = "today_plan.json"
TRADE_LOG_FILE: Final[str] = "trades.csv"
EQUITY_CURVE_FILE: Final[str] = "equity_curve.json"


# ============================================================================
# SYMBOLS AND INSTRUMENTS
# ============================================================================

# Default watchlist for morning learning
DEFAULT_WATCHLIST: Final[list[str]] = [
    # Nifty 50 top movers (add more as needed)
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "ITC",
    "SBIN",
    "BHARTIARTL",
    "KOTAKBANK",
    "BAJFINANCE",
    "LT",
    "AXISBANK",
    "ASIANPAINT",
    "MARUTI",
    "TITAN",
    "SUNPHARMA",
    "ULTRACEMCO",
    "WIPRO",
    "HCLTECH",
]

# Index tokens for reference
NIFTY_50_TOKEN: Final[str] = "99926000"
NIFTY_BANK_TOKEN: Final[str] = "99926009"
INDIA_VIX_TOKEN: Final[str] = "99926016"
