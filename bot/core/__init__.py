# Bot Core Module
# ===============
# Central configuration, mode management, and constants

from .config import Config
from .constants import (
    MarketHours,
    TradingSession,
    Limits,
    Timeouts,
    DEFAULT_CAPITAL,
    DEFAULT_MAX_RISK_PER_TRADE,
)
from .mode import TradingMode, ModeManager
from .logging_config import setup_logging, get_logger

__all__ = [
    "Config",
    "MarketHours",
    "TradingSession",
    "Limits",
    "Timeouts",
    "DEFAULT_CAPITAL",
    "DEFAULT_MAX_RISK_PER_TRADE",
    "TradingMode",
    "ModeManager",
    "setup_logging",
    "get_logger",
]
