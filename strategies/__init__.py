"""
OpenAlgo Strategies Module

This package contains trading strategy implementations for the intraday trading bot.

Modules:
    - base_strategy: Abstract base class for all strategies
    - vol_breakout: Volatility breakout strategy with VWAP confirmation
"""

from .base_strategy import BaseStrategy, Signal, SignalAction
from .vol_breakout import VolatilityBreakoutStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalAction",
    "VolatilityBreakoutStrategy",
]
