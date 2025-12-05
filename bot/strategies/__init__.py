# Bot Strategies Module
# =====================
# Trading strategies and signal generation

from .signal import Signal, SignalType, SignalStrength
from .base_strategy import BaseStrategy, StrategyConfig
from .volatility_breakout import VolatilityBreakoutStrategy, BreakoutParams

__all__ = [
    "Signal",
    "SignalType",
    "SignalStrength",
    "BaseStrategy",
    "StrategyConfig",
    "VolatilityBreakoutStrategy",
    "BreakoutParams",
]
