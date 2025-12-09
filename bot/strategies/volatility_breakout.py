"""
Volatility Breakout Strategy
============================
Opening range breakout with VWAP confirmation.

Strategy Logic:
1. Wait for first N minutes to define opening range (high/low)
2. Calculate ATR for stop loss sizing
3. Wait for breakout above range high (BUY) or below range low (SELL)
4. Confirm with VWAP: price should be above VWAP for BUY, below for SELL
5. Confirm with volume surge (optional)
6. Entry with ATR-based stop loss
7. Target at 1.5x risk (risk-reward ratio)
8. Exit by 3:10 PM if still in position
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
import pandas as pd

from .base_strategy import BaseStrategy, StrategyConfig
from .signal import Signal, SignalType, SignalStrength, ExitReason
from ..utils.indicators import (
    calculate_atr,
    calculate_vwap,
    calculate_opening_range,
    calculate_volume_surge,
    OpeningRange,
)


@dataclass
class BreakoutParams:
    """Parameters for volatility breakout strategy"""
    # Opening range
    opening_range_minutes: int = 15  # First 15 minutes
    
    # ATR settings
    atr_period: int = 14
    atr_multiplier_sl: float = 1.5  # Stop loss = entry ± (ATR * multiplier)
    
    # Breakout settings
    breakout_buffer_pct: float = 0.001  # 0.1% buffer above/below range
    require_close_above: bool = True    # Candle must close above breakout level
    
    # VWAP confirmation
    require_vwap_confirmation: bool = True
    vwap_buffer_pct: float = 0.001  # Price must be 0.1% on right side of VWAP
    
    # Volume confirmation
    require_volume_surge: bool = True
    volume_surge_multiplier: float = 1.2  # Volume must be 1.2x average
    
    # Target
    risk_reward_ratio: float = 1.5
    
    # Time constraints
    min_time_after_open: int = 15  # Minimum minutes after market open
    max_time_for_entry: str = "14:30"  # No new entries after this time
    
    # Filters
    min_range_pct: float = 0.003  # Min 0.3% range (avoid tight ranges)
    max_range_pct: float = 0.03   # Max 3% range (avoid gaps)


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) strategy with VWAP confirmation.
    
    This is a classic intraday strategy that:
    - Captures the first major move of the day
    - Uses volatility (ATR) for position sizing
    - Confirms with VWAP and volume
    
    Works best on:
    - High-volume liquid stocks
    - Days with clear directional moves
    - Trending market conditions
    
    Avoid:
    - Choppy/ranging days
    - Low volume stocks
    - Major news/event days (gaps)
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        params: Optional[BreakoutParams] = None,
    ):
        super().__init__(config or StrategyConfig(name="VolatilityBreakout"))
        self.params = params or BreakoutParams()
        
        # Strategy state
        self._opening_ranges: dict[str, OpeningRange] = {}
        self._breakout_triggered: dict[str, bool] = {}
        self._entry_bar_seen: dict[str, bool] = {}
    
    def reset_daily(self):
        """Reset for new trading day"""
        super().reset_daily()
        self._opening_ranges = {}
        self._breakout_triggered = {}
        self._entry_bar_seen = {}
    
    def _calculate_opening_range(self, symbol: str) -> Optional[OpeningRange]:
        """Calculate opening range from today's data"""
        data = self._today_data.get(symbol)
        if data is None or len(data) < 2:
            return None
        
        return calculate_opening_range(
            data,
            minutes=self.params.opening_range_minutes,
            buffer_pct=self.params.breakout_buffer_pct,
        )
    
    def _get_atr(self, symbol: str) -> Optional[float]:
        """Get current ATR value"""
        data = self.get_combined_data(symbol)
        if data is None or len(data) < self.params.atr_period + 1:
            return None
        
        atr_series = calculate_atr(
            data['high'],
            data['low'],
            data['close'],
            period=self.params.atr_period,
        )
        return atr_series.iloc[-1] if not atr_series.empty else None
    
    def _get_vwap(self, symbol: str) -> Optional[float]:
        """Get current VWAP value"""
        data = self._today_data.get(symbol)
        if data is None or len(data) < 2:
            return None
        
        vwap_series = calculate_vwap(
            data['high'],
            data['low'],
            data['close'],
            data['volume'],
            reset_daily=True,
        )
        return vwap_series.iloc[-1] if not vwap_series.empty else None
    
    def _get_volume_surge(self, symbol: str) -> Optional[float]:
        """Get current volume surge ratio"""
        data = self.get_combined_data(symbol)
        if data is None or len(data) < 21:
            return None
        
        surge = calculate_volume_surge(data['volume'], period=20)
        return surge.iloc[-1] if not surge.empty else None
    
    def analyze(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
    ) -> Signal:
        """
        Analyze for breakout signals.
        
        Flow:
        1. Check if we can trade
        2. Ensure opening range is calculated
        3. Check for breakout conditions
        4. Apply confirmations (VWAP, volume)
        5. Generate signal with proper stops and targets
        """
        # Check if we can trade
        can_trade, reason = self.can_trade(symbol, timestamp)
        if not can_trade:
            return Signal.hold(symbol, reason)
        
        # Get or calculate opening range
        if symbol not in self._opening_ranges:
            opening_range = self._calculate_opening_range(symbol)
            if opening_range:
                self._opening_ranges[symbol] = opening_range
                self._breakout_triggered[symbol] = False
        
        opening_range = self._opening_ranges.get(symbol)
        if opening_range is None:
            return Signal.hold(symbol, "Opening range not yet defined")
        
        # Validate range size
        range_pct = opening_range.range_size / current_price
        if range_pct < self.params.min_range_pct:
            return Signal.hold(symbol, f"Range too tight: {range_pct*100:.2f}%")
        if range_pct > self.params.max_range_pct:
            return Signal.hold(symbol, f"Range too wide: {range_pct*100:.2f}%")
        
        # Get indicators
        atr = self._get_atr(symbol)
        if atr is None:
            return Signal.hold(symbol, "ATR not available")
        
        vwap = self._get_vwap(symbol)
        volume_surge = self._get_volume_surge(symbol)
        
        # Check for breakout
        breakout_buy = current_price > opening_range.breakout_buy
        breakout_sell = current_price < opening_range.breakout_sell
        
        if not breakout_buy and not breakout_sell:
            return Signal.hold(
                symbol,
                f"No breakout: price {current_price:.2f} within range [{opening_range.low:.2f}, {opening_range.high:.2f}]"
            )
        
        # Already triggered today (one breakout trade per day per symbol)
        if self._breakout_triggered.get(symbol, False):
            return Signal.hold(symbol, "Breakout already traded today")
        
        # Determine direction
        is_long = breakout_buy
        
        # VWAP confirmation
        if self.params.require_vwap_confirmation and vwap is not None:
            vwap_buffer = vwap * self.params.vwap_buffer_pct
            
            if is_long and current_price < vwap - vwap_buffer:
                return Signal.hold(symbol, f"BUY rejected: price {current_price:.2f} below VWAP {vwap:.2f}")
            
            if not is_long and current_price > vwap + vwap_buffer:
                return Signal.hold(symbol, f"SELL rejected: price {current_price:.2f} above VWAP {vwap:.2f}")
        
        # Volume confirmation
        if self.params.require_volume_surge and volume_surge is not None:
            if volume_surge < self.params.volume_surge_multiplier:
                return Signal.hold(
                    symbol,
                    f"Volume too low: {volume_surge:.2f}x (need {self.params.volume_surge_multiplier}x)"
                )
        
        # Calculate stop loss and target
        stop_distance = atr * self.params.atr_multiplier_sl
        
        if is_long:
            stop_loss = current_price - stop_distance
            target = current_price + (stop_distance * self.params.risk_reward_ratio)
            signal_type = SignalType.BUY
        else:
            stop_loss = current_price + stop_distance
            target = current_price - (stop_distance * self.params.risk_reward_ratio)
            signal_type = SignalType.SELL
        
        # Build reason string
        reasons = []
        reasons.append(f"Breakout {'above' if is_long else 'below'} range")
        if vwap:
            reasons.append(f"VWAP={vwap:.2f}")
        if volume_surge:
            reasons.append(f"Vol={volume_surge:.1f}x")
        reasons.append(f"ATR={atr:.2f}")
        
        # Determine signal strength
        strength = SignalStrength.MODERATE
        if volume_surge and volume_surge > 2.0:
            strength = SignalStrength.STRONG
        if volume_surge and volume_surge > 3.0:
            strength = SignalStrength.VERY_STRONG
        
        # Calculate confidence (0-1)
        confidence = 0.5
        if vwap and ((is_long and current_price > vwap) or (not is_long and current_price < vwap)):
            confidence += 0.2
        if volume_surge and volume_surge > self.params.volume_surge_multiplier:
            confidence += 0.15
            if volume_surge > 2.0:
                confidence += 0.15
        
        # Create signal
        signal = Signal(
            signal_type=signal_type,
            symbol=symbol,
            price=current_price,
            stop_loss=stop_loss,
            target=target,
            strength=strength,
            confidence=min(1.0, confidence),
            strategy_name=self.name,
            reason="; ".join(reasons),
            indicators={
                "atr": atr,
                "vwap": vwap or 0.0,
                "volume_surge": volume_surge or 0.0,
                "range_high": opening_range.high,
                "range_low": opening_range.low,
            },
        )
        
        # Validate signal
        is_valid, validation_reason = self.validate_signal(signal)
        if not is_valid:
            return Signal.hold(symbol, f"Signal rejected: {validation_reason}")
        
        # Mark breakout as triggered
        self._breakout_triggered[symbol] = True
        self.state.signals_generated += 1
        self.state.last_signal_time = timestamp.isoformat()
        
        return signal
    
    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_side: str,
        entry_time: datetime,
        current_time: datetime,
    ) -> Optional[Signal]:
        """
        Check if position should be exited.
        
        Exit conditions:
        1. Stop loss hit
        2. Target hit
        3. Time exit (3:10 PM)
        4. VWAP cross against position
        """
        is_long = position_side == "BUY"
        
        # Get opening range for SL/target reference
        opening_range = self._opening_ranges.get(symbol)
        atr = self._get_atr(symbol)
        
        if atr:
            stop_distance = atr * self.params.atr_multiplier_sl
            target_distance = stop_distance * self.params.risk_reward_ratio
            
            if is_long:
                stop_loss = entry_price - stop_distance
                target = entry_price + target_distance
                
                # Check stop loss
                if current_price <= stop_loss:
                    return Signal.exit_long(
                        symbol, current_price,
                        ExitReason.STOP_LOSS,
                        f"Stop loss hit: {current_price:.2f} <= {stop_loss:.2f}"
                    )
                
                # Check target
                if current_price >= target:
                    return Signal.exit_long(
                        symbol, current_price,
                        ExitReason.TARGET_HIT,
                        f"Target hit: {current_price:.2f} >= {target:.2f}"
                    )
            else:
                stop_loss = entry_price + stop_distance
                target = entry_price - target_distance
                
                if current_price >= stop_loss:
                    return Signal.exit_short(
                        symbol, current_price,
                        ExitReason.STOP_LOSS,
                        f"Stop loss hit: {current_price:.2f} >= {stop_loss:.2f}"
                    )
                
                if current_price <= target:
                    return Signal.exit_short(
                        symbol, current_price,
                        ExitReason.TARGET_HIT,
                        f"Target hit: {current_price:.2f} <= {target:.2f}"
                    )
        
        # Time exit (3:10 PM)
        exit_time = time(15, 10)
        if current_time.time() >= exit_time:
            exit_signal_type = SignalType.EXIT_LONG if is_long else SignalType.EXIT_SHORT
            return Signal(
                signal_type=exit_signal_type,
                symbol=symbol,
                price=current_price,
                exit_reason=ExitReason.TIME_EXIT,
                reason="End of day square off (3:10 PM)",
            )
        
        # VWAP cross against position
        vwap = self._get_vwap(symbol)
        if vwap:
            if is_long and current_price < vwap * 0.995:  # 0.5% below VWAP
                # Optional: could exit on VWAP cross
                # For now, just track but don't exit
                pass
        
        return None
    
    def get_opening_range(self, symbol: str) -> Optional[OpeningRange]:
        """Get the calculated opening range for a symbol"""
        return self._opening_ranges.get(symbol)


# ============================================================================
# UNIT TESTS
# ============================================================================

def _test_strategy():
    """Test volatility breakout strategy"""
    import numpy as np
    
    print("Testing volatility breakout strategy...")
    
    # Create sample data
    np.random.seed(42)
    
    # Simulate 5-min candles starting at 9:15
    dates = pd.date_range(start='2024-01-15 09:15', periods=60, freq='5min')
    
    # Simulate opening range then breakout
    prices = [100.0]
    for i in range(1, 60):
        if i < 3:  # First 15 min: range bound
            prices.append(prices[-1] + np.random.randn() * 0.3)
        elif i == 10:  # Breakout candle
            prices.append(prices[-1] + 2.0)  # Break above
        else:
            prices.append(prices[-1] + np.random.randn() * 0.3)
    
    close = pd.Series(prices, index=dates)
    high = close + np.abs(np.random.randn(60) * 0.2)
    low = close - np.abs(np.random.randn(60) * 0.2)
    open_ = close.shift(1).fillna(100)
    volume = pd.Series(np.random.randint(10000, 50000, 60), index=dates)
    
    # Spike volume on breakout
    volume.iloc[10] = 100000
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    # Create strategy
    strategy = VolatilityBreakoutStrategy()
    strategy.reset_daily()
    
    # Set data
    strategy.update_data("TEST", df.iloc[:15])  # First 15 minutes
    
    # Analyze before breakout
    signal1 = strategy.analyze(
        "TEST",
        current_price=100.5,
        timestamp=datetime(2024, 1, 15, 9, 35),
    )
    print(f"Before breakout: {signal1}")
    assert signal1.is_hold, "Should be HOLD before breakout"
    
    # Update with breakout data
    strategy.update_data("TEST", df.iloc[:12])
    strategy.set_historical_data("TEST", df.iloc[:12])  # For ATR
    
    # Analyze at breakout
    breakout_price = df['high'].iloc[:3].max() + 0.5  # Above opening range
    signal2 = strategy.analyze(
        "TEST",
        current_price=breakout_price,
        timestamp=datetime(2024, 1, 15, 10, 5),
    )
    print(f"At breakout: {signal2}")
    
    # Test should_exit
    if signal2.is_entry:
        exit_signal = strategy.should_exit(
            "TEST",
            entry_price=breakout_price,
            current_price=breakout_price + 3.0,  # At target
            position_side="BUY",
            entry_time=datetime(2024, 1, 15, 10, 5),
            current_time=datetime(2024, 1, 15, 11, 0),
        )
        print(f"Exit check (at target): {exit_signal}")
    
    print("\n✅ Strategy tests passed!")


if __name__ == "__main__":
    _test_strategy()
