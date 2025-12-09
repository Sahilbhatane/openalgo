"""
Base Strategy
=============
Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List, Any
import pandas as pd

from .signal import Signal, SignalType


@dataclass
class StrategyConfig:
    """Base configuration for strategies"""
    name: str = "BaseStrategy"
    enabled: bool = True
    
    # Timeframe
    timeframe: str = "5min"  # 1min, 5min, 15min, 1hour
    
    # Trading window
    start_time: str = "09:20"  # Start generating signals
    end_time: str = "14:30"    # Stop generating new entry signals
    
    # Position management
    max_positions: int = 1
    allow_multiple_entries: bool = False
    
    # Risk parameters
    max_risk_per_trade: float = 0.01  # 1% risk per trade
    default_risk_reward: float = 1.5
    
    # Filters
    min_volume_multiplier: float = 1.0  # Relative to average
    min_price: float = 50.0
    max_price: float = 10000.0


@dataclass
class StrategyState:
    """Runtime state of a strategy"""
    date: str = ""
    signals_generated: int = 0
    trades_taken: int = 0
    current_positions: List[str] = field(default_factory=list)
    last_signal_time: Optional[str] = None
    is_active: bool = True
    notes: str = ""


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement:
    - analyze(): Generate signals from market data
    - should_exit(): Check if existing position should be closed
    
    Lifecycle:
    1. Initialize strategy with config
    2. Call reset_daily() at start of each day
    3. Call analyze() for each new candle/tick
    4. Call should_exit() to manage open positions
    5. Call on_trade() when trade is executed
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config or StrategyConfig()
        self.state = StrategyState()
        self._historical_data: Dict[str, pd.DataFrame] = {}
        self._today_data: Dict[str, pd.DataFrame] = {}
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def is_enabled(self) -> bool:
        return self.config.enabled
    
    def reset_daily(self):
        """Reset strategy state for new trading day"""
        today = date.today().isoformat()
        if self.state.date != today:
            self.state = StrategyState(date=today)
            self._today_data = {}
    
    def update_data(self, symbol: str, data: pd.DataFrame):
        """
        Update market data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame
        """
        self._today_data[symbol] = data
    
    def set_historical_data(self, symbol: str, data: pd.DataFrame):
        """
        Set historical data for a symbol (for indicators).
        
        Args:
            symbol: Trading symbol
            data: Historical OHLCV DataFrame
        """
        self._historical_data[symbol] = data
    
    def get_combined_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get combined historical and today's data"""
        hist = self._historical_data.get(symbol)
        today = self._today_data.get(symbol)
        
        if hist is None and today is None:
            return None
        
        if hist is None:
            return today
        
        if today is None:
            return hist
        
        # Combine, avoiding duplicates
        combined = pd.concat([hist, today])
        combined = combined[~combined.index.duplicated(keep='last')]
        return combined.sort_index()
    
    @abstractmethod
    def analyze(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
    ) -> Signal:
        """
        Analyze market data and generate a signal.
        
        This is the main strategy logic. Called for each new candle/tick.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            timestamp: Current time
        
        Returns:
            Signal (BUY, SELL, or HOLD)
        """
        pass
    
    @abstractmethod
    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_side: str,  # "BUY" or "SELL"
        entry_time: datetime,
        current_time: datetime,
    ) -> Optional[Signal]:
        """
        Check if an existing position should be closed.
        
        Called for each open position to check exit conditions.
        
        Args:
            symbol: Trading symbol
            entry_price: Position entry price
            current_price: Current market price
            position_side: "BUY" for long, "SELL" for short
            entry_time: When position was opened
            current_time: Current time
        
        Returns:
            Exit signal if position should be closed, None otherwise
        """
        pass
    
    def on_trade(self, signal: Signal, fill_price: float, quantity: int):
        """
        Called when a trade is executed.
        
        Override to update strategy state after trade.
        
        Args:
            signal: The signal that was executed
            fill_price: Actual fill price
            quantity: Quantity traded
        """
        self.state.trades_taken += 1
        
        if signal.is_entry:
            self.state.current_positions.append(signal.symbol)
        elif signal.is_exit:
            if signal.symbol in self.state.current_positions:
                self.state.current_positions.remove(signal.symbol)
    
    def can_trade(self, symbol: str, timestamp: datetime) -> tuple[bool, str]:
        """
        Check if strategy can generate a new entry signal.
        
        Args:
            symbol: Trading symbol
            timestamp: Current time
        
        Returns:
            Tuple of (can_trade, reason)
        """
        if not self.is_enabled:
            return False, "Strategy disabled"
        
        if not self.state.is_active:
            return False, "Strategy inactive for today"
        
        # Check time window
        current_time = timestamp.strftime("%H:%M")
        if current_time < self.config.start_time:
            return False, f"Before trading window ({self.config.start_time})"
        
        if current_time > self.config.end_time:
            return False, f"After trading window ({self.config.end_time})"
        
        # Check position limits
        if len(self.state.current_positions) >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"
        
        # Check if already in position for this symbol
        if symbol in self.state.current_positions and not self.config.allow_multiple_entries:
            return False, f"Already in position for {symbol}"
        
        return True, "OK"
    
    def validate_signal(self, signal: Signal) -> tuple[bool, str]:
        """
        Validate a signal before returning it.
        
        Override to add custom validation.
        
        Args:
            signal: Signal to validate
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if signal.is_hold:
            return True, "OK"
        
        if signal.is_entry:
            # Check risk-reward
            if signal.risk_reward_ratio < 1.0:
                return False, f"Risk-reward {signal.risk_reward_ratio:.2f} < 1.0"
            
            # Check stop loss distance
            risk_pct = signal.risk_per_share / signal.price
            if risk_pct > 0.05:  # More than 5% stop
                return False, f"Stop loss too wide: {risk_pct*100:.1f}%"
            
            if risk_pct < 0.002:  # Less than 0.2% stop
                return False, f"Stop loss too tight: {risk_pct*100:.2f}%"
        
        return True, "OK"
    
    def get_status(self) -> str:
        """Get human-readable status"""
        return f"""
Strategy: {self.name}
  Enabled: {self.is_enabled}
  Date: {self.state.date}
  Signals: {self.state.signals_generated}
  Trades: {self.state.trades_taken}
  Positions: {self.state.current_positions}
  Last Signal: {self.state.last_signal_time}
"""


class CompositeStrategy(BaseStrategy):
    """
    Combine multiple strategies with weighted voting.
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: Optional[Dict[str, float]] = None,
        require_consensus: bool = True,
    ):
        """
        Initialize composite strategy.
        
        Args:
            strategies: List of strategies to combine
            weights: Strategy name -> weight mapping
            require_consensus: Require majority agreement
        """
        super().__init__(StrategyConfig(name="CompositeStrategy"))
        self.strategies = strategies
        self.weights = weights or {s.name: 1.0 for s in strategies}
        self.require_consensus = require_consensus
    
    def reset_daily(self):
        super().reset_daily()
        for strategy in self.strategies:
            strategy.reset_daily()
    
    def update_data(self, symbol: str, data: pd.DataFrame):
        super().update_data(symbol, data)
        for strategy in self.strategies:
            strategy.update_data(symbol, data)
    
    def analyze(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
    ) -> Signal:
        """Combine signals from all strategies"""
        signals = []
        
        for strategy in self.strategies:
            if strategy.is_enabled:
                signal = strategy.analyze(symbol, current_price, timestamp)
                signals.append(signal)
        
        if not signals:
            return Signal.hold(symbol, "No active strategies")
        
        # Count signal types
        buys = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sells = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        
        # Require majority for consensus
        threshold = len(signals) / 2 if self.require_consensus else 1
        
        if buys > threshold and buys > sells:
            # Use the strongest buy signal
            buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
            best = max(buy_signals, key=lambda s: s.confidence)
            best.strategy_name = f"composite({','.join(s.strategy_name for s in buy_signals)})"
            return best
        
        if sells > threshold and sells > buys:
            sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
            best = max(sell_signals, key=lambda s: s.confidence)
            best.strategy_name = f"composite({','.join(s.strategy_name for s in sell_signals)})"
            return best
        
        return Signal.hold(symbol, f"No consensus: {buys} buy, {sells} sell")
    
    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_side: str,
        entry_time: datetime,
        current_time: datetime,
    ) -> Optional[Signal]:
        """Check exit conditions from all strategies"""
        for strategy in self.strategies:
            exit_signal = strategy.should_exit(
                symbol, entry_price, current_price,
                position_side, entry_time, current_time
            )
            if exit_signal:
                return exit_signal
        return None
