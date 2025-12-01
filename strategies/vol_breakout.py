"""
Volatility Breakout Strategy for OpenAlgo Intraday Trading Bot

This module implements a volatility breakout strategy with VWAP confirmation.
It generates BUY signals when price breaks above the opening range high with
volume confirmation and optional VWAP filter.

Strategy Logic:
    - Entry: Price breaks above first N minutes high (breakout)
    - Volume: Current volume > min_volume_factor × average volume
    - VWAP: Price > VWAP (optional confirmation)
    - Stop-Loss: Entry - ATR × atr_mult_sl
    - Target: Entry + RR × (Entry - Stop-Loss)

Author: OpenAlgo Team
Date: December 2025
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Add project root for imports
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.base_strategy import BaseStrategy, Signal, SignalAction, StrategyConfig
from utils.logging import get_logger

logger = get_logger(__name__)


# ==================== Configuration ====================


@dataclass
class VolBreakoutConfig(StrategyConfig):
    """
    Configuration for Volatility Breakout Strategy.

    Extends base StrategyConfig with strategy-specific parameters.
    """

    # Opening range parameters
    entry_window_minutes: int = 15  # First N minutes to define opening range

    # Volume filter
    min_volume_factor: float = 1.2  # Volume must be > this × avg volume

    # Stop-loss parameters
    atr_mult_sl: float = 1.5  # ATR multiplier for stop-loss
    atr_period: int = 14  # ATR calculation period

    # Risk-Reward ratio
    rr: float = 1.5  # Risk:Reward ratio for target calculation

    # VWAP confirmation
    vwap_confirm: bool = True  # Require price > VWAP for longs

    # Trade management
    max_hold_minutes: int = 180  # Max holding time (3 hours)
    cooldown_minutes: int = 5  # Cooldown between signals

    # Additional filters
    min_atr_pct: float = 0.01  # Min ATR as % of price (1%)
    max_spread_pct: float = 0.005  # Max bid-ask spread allowed (0.5%)

    def __post_init__(self):
        """Set strategy name"""
        self.name = "VolatilityBreakout"


# ==================== Position Sizing ====================


def compute_qty(
    capital: float,
    entry_price: float,
    stop_loss: float,
    risk_per_trade: float = 0.02,
    max_position_pct: float = 0.10,
    min_qty: int = 1,
    subscription_daily: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute position size based on risk management and trading costs.

    This function calculates the optimal quantity considering:
    1. Risk per trade (% of capital at risk)
    2. Maximum position size (% of capital)
    3. Trading costs (brokerage, taxes, fees)

    Args:
        capital: Total trading capital
        entry_price: Expected entry price
        stop_loss: Stop-loss price
        risk_per_trade: Max risk per trade as decimal (default 2%)
        max_position_pct: Max position as % of capital (default 10%)
        min_qty: Minimum quantity (default 1)
        subscription_daily: Daily API subscription cost (for cost adjustment)

    Returns:
        dict with:
            - qty: Suggested quantity
            - position_value: Total position value
            - risk_amount: Amount at risk
            - cost_estimate: Estimated round-trip costs

    Example:
        >>> result = compute_qty(100000, 500, 480, 0.02)
        >>> print(f"Buy {result['qty']} shares")
    """
    if entry_price <= 0 or stop_loss <= 0:
        return {
            "qty": 0,
            "position_value": 0,
            "risk_amount": 0,
            "cost_estimate": 0,
            "error": "Invalid prices",
        }

    # Risk per share
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0:
        risk_per_share = entry_price * 0.02  # Default 2% if SL equals entry

    # Max risk amount
    max_risk_amount = capital * risk_per_trade

    # Calculate qty based on risk
    qty_by_risk = int(max_risk_amount / risk_per_share)

    # Calculate qty based on max position size
    max_position_value = capital * max_position_pct
    qty_by_position = int(max_position_value / entry_price)

    # Take the minimum of both constraints
    qty = max(min_qty, min(qty_by_risk, qty_by_position))

    # Calculate position metrics
    position_value = qty * entry_price
    risk_amount = qty * risk_per_share

    # Estimate trading costs (simplified)
    # Using approximate cost of 0.05% per leg for intraday
    cost_pct = 0.0005  # 0.05%
    cost_estimate = position_value * cost_pct * 2  # Round trip

    # Add subscription cost if provided
    if subscription_daily > 0:
        cost_estimate += subscription_daily / 10  # Assume 10 trades per day

    return {
        "qty": qty,
        "position_value": round(position_value, 2),
        "risk_amount": round(risk_amount, 2),
        "cost_estimate": round(cost_estimate, 2),
        "risk_per_share": round(risk_per_share, 2),
        "qty_by_risk": qty_by_risk,
        "qty_by_position": qty_by_position,
    }


# ==================== Indicator Calculator ====================


class IndicatorCalculator:
    """
    Technical indicator calculator for the breakout strategy.
    """

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR value"""
        if df.empty or len(df) < period:
            return 0.0

        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()

        return float(atr.iloc[-1]) if len(atr) > 0 else 0.0

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> float:
        """Calculate current VWAP value"""
        if df.empty:
            return 0.0

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()

        vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.inf)
        return float(vwap.iloc[-1]) if len(vwap) > 0 else 0.0

    @staticmethod
    def calculate_avg_volume(df: pd.DataFrame, period: int = 20) -> float:
        """Calculate average volume"""
        if df.empty:
            return 0.0

        return float(df["volume"].tail(period).mean())

    @staticmethod
    def get_opening_range(df: pd.DataFrame, minutes: int = 15) -> Dict[str, float]:
        """
        Get opening range high and low from first N minutes.

        Args:
            df: Intraday OHLCV DataFrame with datetime index
            minutes: Number of minutes for opening range

        Returns:
            Dict with high, low of opening range
        """
        if df.empty:
            return {"high": 0.0, "low": 0.0, "range": 0.0}

        # Get today's data only
        if hasattr(df.index, "date"):
            today = df.index[-1].date()
            today_data = df[df.index.date == today]
        else:
            today_data = df

        if today_data.empty:
            return {"high": 0.0, "low": 0.0, "range": 0.0}

        # Get first N minutes
        first_bar_time = today_data.index[0]
        cutoff_time = first_bar_time + timedelta(minutes=minutes)

        opening_range_data = today_data[today_data.index <= cutoff_time]

        if opening_range_data.empty:
            return {"high": 0.0, "low": 0.0, "range": 0.0}

        high = float(opening_range_data["high"].max())
        low = float(opening_range_data["low"].min())

        return {"high": high, "low": low, "range": high - low}


# ==================== Volatility Breakout Strategy ====================


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy with VWAP Confirmation.

    This strategy generates BUY signals when:
    1. Price breaks above the high of the first N minutes (opening range)
    2. Current volume > min_volume_factor × average volume
    3. Price > VWAP (if vwap_confirm is enabled)

    Stop-loss is set at entry_price - ATR × atr_mult_sl
    Target is set at entry_price + RR × risk_amount

    Features:
    - Cooldown timer to prevent signal spam
    - Position sizing using risk management
    - Logging for all decisions
    - Trade time limit (max hold minutes)

    Example:
        config = {
            "entry_window_minutes": 15,
            "min_volume_factor": 1.2,
            "atr_mult_sl": 1.5,
            "rr": 1.5,
            "vwap_confirm": True,
            "capital": 100000
        }
        strategy = VolatilityBreakoutStrategy(config)
        signal = strategy.generate_signal(live_bar, history, indicators, plan_entry)
    """

    def __init__(self, config: Union[Dict[str, Any], VolBreakoutConfig]):
        """
        Initialize the Volatility Breakout Strategy.

        Args:
            config: Strategy configuration (dict or VolBreakoutConfig)
        """
        # Convert dict to VolBreakoutConfig if needed
        if isinstance(config, dict):
            vol_config = VolBreakoutConfig(**config)
        else:
            vol_config = config

        super().__init__(vol_config)

        # Strategy-specific config
        self.entry_window_minutes = vol_config.entry_window_minutes
        self.min_volume_factor = vol_config.min_volume_factor
        self.atr_mult_sl = vol_config.atr_mult_sl
        self.atr_period = vol_config.atr_period
        self.rr = vol_config.rr
        self.vwap_confirm = vol_config.vwap_confirm
        self.max_hold_minutes = vol_config.max_hold_minutes
        self.cooldown_minutes = vol_config.cooldown_minutes
        self.min_atr_pct = vol_config.min_atr_pct
        self.max_spread_pct = vol_config.max_spread_pct

        # Internal state
        self._last_signal_time: Optional[datetime] = None
        self._opening_range_cache: Dict[str, Dict[str, float]] = {}
        self._indicator_calc = IndicatorCalculator()

        logger.info(
            f"VolatilityBreakoutStrategy initialized: "
            f"entry_window={self.entry_window_minutes}m, "
            f"min_vol_factor={self.min_volume_factor}, "
            f"atr_mult={self.atr_mult_sl}, "
            f"rr={self.rr}, "
            f"vwap_confirm={self.vwap_confirm}"
        )

    def generate_signal(
        self,
        live_bar: Dict[str, Any],
        history: pd.DataFrame,
        indicators: Dict[str, Any],
        plan_entry: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        Generate trading signal based on volatility breakout conditions.

        Args:
            live_bar: Current bar with keys:
                - open, high, low, close, volume
                - timestamp (datetime or str)
                - symbol
            history: Historical OHLCV DataFrame (intraday, e.g., 5m bars)
            indicators: Pre-computed indicators (optional, will compute if missing):
                - atr: ATR value
                - vwap: VWAP value
                - avg_volume: Average volume
            plan_entry: Optional entry from morning plan:
                - entry_zones: [[low, high], ...]
                - stop_loss: float
                - targets: [float, ...]
                - suggested_qty: int

        Returns:
            Signal object with action, prices, qty, reason
        """
        symbol = live_bar.get("symbol", "UNKNOWN")
        timestamp = live_bar.get("timestamp", datetime.now())

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        current_price = float(live_bar.get("close", 0))
        current_volume = float(live_bar.get("volume", 0))
        current_high = float(live_bar.get("high", current_price))

        # === Cooldown Check ===
        if not self._check_cooldown(timestamp):
            reason = f"Cooldown active (wait {self.cooldown_minutes}m between signals)"
            logger.debug(f"[{symbol}] {reason}")
            return Signal.hold(reason)

        # === Max Trades Check (before trading window check) ===
        if self.trades_today >= self.config.max_trades_per_day:
            reason = f"Max trades reached ({self.config.max_trades_per_day}/day)"
            logger.debug(f"[{symbol}] {reason}")
            return Signal.hold(reason)

        # === Trading Window Check ===
        if not self._is_within_trading_window(timestamp):
            reason = f"Outside trading window ({self.config.trading_start_time} - {self.config.trading_end_time})"
            logger.debug(f"[{symbol}] {reason}")
            return Signal.hold(reason)

        # === Get or Calculate Indicators ===
        atr = indicators.get("atr") or self._indicator_calc.calculate_atr(
            history, self.atr_period
        )
        vwap = indicators.get("vwap") or self._indicator_calc.calculate_vwap(history)
        avg_volume = indicators.get(
            "avg_volume"
        ) or self._indicator_calc.calculate_avg_volume(history)

        # === Get Opening Range ===
        opening_range = self._get_opening_range(history, symbol)
        breakout_level = opening_range["high"]

        if breakout_level <= 0:
            reason = "Opening range not yet established"
            logger.debug(f"[{symbol}] {reason}")
            return Signal.hold(reason)

        # === Validate ATR ===
        if atr <= 0 or (atr / current_price) < self.min_atr_pct:
            reason = f"ATR too low ({atr:.2f}, {atr/current_price*100:.2f}% of price)"
            logger.debug(f"[{symbol}] {reason}")
            return Signal.hold(reason)

        # === Check Breakout Conditions ===
        conditions = self._evaluate_conditions(
            current_price=current_price,
            current_high=current_high,
            current_volume=current_volume,
            breakout_level=breakout_level,
            avg_volume=avg_volume,
            vwap=vwap,
        )

        # Log condition evaluation
        logger.debug(
            f"[{symbol}] Conditions: "
            f"price={current_price:.2f}, "
            f"breakout_level={breakout_level:.2f}, "
            f"volume_ratio={conditions['volume_ratio']:.2f}, "
            f"vwap={vwap:.2f}, "
            f"breakout={conditions['is_breakout']}, "
            f"vol_ok={conditions['volume_ok']}, "
            f"vwap_ok={conditions['vwap_ok']}"
        )

        # === Generate Signal ===
        if conditions["all_conditions_met"]:
            # Calculate entry, stop-loss, and targets
            entry_price = current_price
            stop_loss = entry_price - (atr * self.atr_mult_sl)
            risk = entry_price - stop_loss
            target = entry_price + (risk * self.rr)

            # Use plan entry if available
            if plan_entry:
                if plan_entry.get("stop_loss"):
                    stop_loss = float(plan_entry["stop_loss"])
                if plan_entry.get("targets"):
                    target = (
                        float(plan_entry["targets"][0])
                        if plan_entry["targets"]
                        else target
                    )

            # Calculate position size
            qty_result = compute_qty(
                capital=self.config.capital,
                entry_price=entry_price,
                stop_loss=stop_loss,
                risk_per_trade=self.config.risk_per_trade,
                max_position_pct=self.config.max_position_size,
            )

            qty = (
                plan_entry.get("suggested_qty", qty_result["qty"])
                if plan_entry
                else qty_result["qty"]
            )

            # Build reason string
            reason = (
                f"Breakout above {breakout_level:.2f} (ORB High) | "
                f"Volume {conditions['volume_ratio']:.1f}x avg | "
                f"Price > VWAP ({vwap:.2f})"
                if self.vwap_confirm
                else ""
            )

            # Update cooldown timer
            self._last_signal_time = timestamp

            logger.info(
                f"[{symbol}] 🟢 BUY SIGNAL: "
                f"entry={entry_price:.2f}, "
                f"sl={stop_loss:.2f}, "
                f"target={target:.2f}, "
                f"qty={qty}, "
                f"risk={risk:.2f}"
            )

            return Signal.buy(
                entry_price=entry_price,
                stop_loss=stop_loss,
                targets=[target, entry_price + (risk * self.rr * 1.5)],  # T1 and T2
                qty=qty,
                reason=reason,
                confidence=conditions["confidence"],
                atr=atr,
                vwap=vwap,
                volume_ratio=conditions["volume_ratio"],
                breakout_level=breakout_level,
            )

        # No breakout - return HOLD
        reasons = []
        if not conditions["is_breakout"]:
            reasons.append(
                f"No breakout (price {current_price:.2f} < {breakout_level:.2f})"
            )
        if not conditions["volume_ok"]:
            reasons.append(
                f"Low volume ({conditions['volume_ratio']:.1f}x < {self.min_volume_factor}x)"
            )
        if not conditions["vwap_ok"]:
            reasons.append(f"Below VWAP ({current_price:.2f} < {vwap:.2f})")

        reason = " | ".join(reasons) if reasons else "Conditions not met"
        logger.debug(f"[{symbol}] HOLD: {reason}")

        return Signal.hold(reason)

    def _check_cooldown(self, current_time: datetime) -> bool:
        """
        Check if cooldown period has passed since last signal.

        Args:
            current_time: Current timestamp

        Returns:
            True if can generate new signal
        """
        if self._last_signal_time is None:
            return True

        elapsed = (current_time - self._last_signal_time).total_seconds() / 60
        return elapsed >= self.cooldown_minutes

    def _get_opening_range(
        self, history: pd.DataFrame, symbol: str
    ) -> Dict[str, float]:
        """
        Get or compute opening range for the symbol.

        Caches the opening range after initial calculation to avoid recomputation.

        Args:
            history: Historical OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Dict with high, low, range
        """
        # Check cache first (for current trading day)
        today = datetime.now().date()
        cache_key = f"{symbol}_{today}"

        if cache_key in self._opening_range_cache:
            return self._opening_range_cache[cache_key]

        # Calculate opening range
        opening_range = self._indicator_calc.get_opening_range(
            history, self.entry_window_minutes
        )

        # Cache if valid
        if opening_range["high"] > 0:
            self._opening_range_cache[cache_key] = opening_range

        return opening_range

    def _is_within_trading_window(self, timestamp: datetime) -> bool:
        """
        Check if the given timestamp is within the configured trading window.

        Args:
            timestamp: Datetime to check

        Returns:
            True if within trading window
        """
        time_str = timestamp.strftime("%H:%M")
        return (
            self.config.trading_start_time <= time_str <= self.config.trading_end_time
        )

    def _evaluate_conditions(
        self,
        current_price: float,
        current_high: float,
        current_volume: float,
        breakout_level: float,
        avg_volume: float,
        vwap: float,
    ) -> Dict[str, Any]:
        """
        Evaluate all breakout conditions.

        Args:
            current_price: Current close price
            current_high: Current bar high
            current_volume: Current bar volume
            breakout_level: Opening range high (breakout level)
            avg_volume: Average volume
            vwap: Current VWAP

        Returns:
            Dict with condition results and confidence
        """
        # Condition 1: Price breakout
        is_breakout = current_price > breakout_level

        # Condition 2: Volume confirmation
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        volume_ok = volume_ratio >= self.min_volume_factor

        # Condition 3: VWAP confirmation (optional)
        vwap_ok = current_price > vwap if self.vwap_confirm else True

        # All conditions met?
        all_met = is_breakout and volume_ok and vwap_ok

        # Calculate confidence (0.0 to 1.0)
        confidence = 0.0
        if all_met:
            # Base confidence
            confidence = 0.5

            # Bonus for strong breakout (price > 1% above breakout level)
            if current_price > breakout_level * 1.01:
                confidence += 0.1

            # Bonus for high volume (> 2x avg)
            if volume_ratio > 2.0:
                confidence += 0.15
            elif volume_ratio > 1.5:
                confidence += 0.1

            # Bonus for price above VWAP
            if current_price > vwap:
                confidence += 0.1

            # Cap at 1.0
            confidence = min(1.0, confidence)

        return {
            "is_breakout": is_breakout,
            "volume_ok": volume_ok,
            "vwap_ok": vwap_ok,
            "volume_ratio": volume_ratio,
            "all_conditions_met": all_met,
            "confidence": confidence,
        }

    def clear_cache(self) -> None:
        """Clear the opening range cache (call at start of new day)"""
        self._opening_range_cache.clear()
        logger.debug("Opening range cache cleared")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics and cache"""
        super().reset_daily_stats()
        self.clear_cache()
        self._last_signal_time = None


# ==================== Example Usage ====================

if __name__ == "__main__":
    """Example usage and quick test"""

    print("=" * 60)
    print("Volatility Breakout Strategy - Test Run")
    print("=" * 60)

    # Create strategy with config
    config = {
        "capital": 100000,
        "entry_window_minutes": 15,
        "min_volume_factor": 1.2,
        "atr_mult_sl": 1.5,
        "rr": 1.5,
        "vwap_confirm": True,
        "max_trades_per_day": 5,
        "risk_per_trade": 0.02,
    }

    strategy = VolatilityBreakoutStrategy(config)
    print(f"\nStrategy: {strategy}")
    print(f"Config: {strategy.config.to_dict()}")

    # Simulate a breakout scenario
    print("\n--- Simulating Breakout Scenario ---")

    # Create sample history with opening range
    dates = pd.date_range(start="2025-12-01 09:15", periods=30, freq="5min")

    np.random.seed(42)
    base_price = 500

    # First 3 bars establish opening range (09:15 - 09:30)
    opening_high = 505
    opening_low = 498

    history = pd.DataFrame(
        {
            "open": [500, 502, 504]
            + [503 + np.random.uniform(-2, 8) for _ in range(27)],
            "high": [503, 505, 505]
            + [504 + np.random.uniform(0, 10) for _ in range(27)],
            "low": [498, 500, 501]
            + [500 + np.random.uniform(-3, 2) for _ in range(27)],
            "close": [502, 504, 503]
            + [503 + np.random.uniform(-1, 9) for _ in range(27)],
            "volume": [100000, 120000, 110000]
            + [int(100000 * np.random.uniform(0.8, 1.5)) for _ in range(27)],
        },
        index=dates,
    )

    # Breakout bar
    live_bar = {
        "symbol": "SBIN",
        "timestamp": datetime(2025, 12, 1, 10, 0),
        "open": 506,
        "high": 512,
        "low": 505,
        "close": 510,  # Breakout above 505
        "volume": 180000,  # High volume
    }

    # Pre-computed indicators
    indicators = {"atr": 8.5, "vwap": 503.5, "avg_volume": 110000}

    # Generate signal
    signal = strategy.generate_signal(live_bar, history, indicators)

    print(f"\nLive Bar: {live_bar}")
    print(f"Indicators: {indicators}")
    print(f"\nSignal: {signal.to_dict()}")

    # Test HOLD scenario (no breakout)
    print("\n--- Simulating No Breakout Scenario ---")

    live_bar_no_breakout = {
        "symbol": "SBIN",
        "timestamp": datetime(2025, 12, 1, 10, 5),
        "open": 502,
        "high": 504,
        "low": 501,
        "close": 503,  # Below breakout level
        "volume": 90000,  # Low volume
    }

    # Reset cooldown for test
    strategy._last_signal_time = None

    signal_hold = strategy.generate_signal(live_bar_no_breakout, history, indicators)

    print(f"\nLive Bar: {live_bar_no_breakout}")
    print(f"Signal: {signal_hold.to_dict()}")

    print("\n" + "=" * 60)
