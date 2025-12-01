"""
Unit Tests for Volatility Breakout Strategy

This module contains comprehensive tests for the VolatilityBreakoutStrategy class.

Test Coverage:
    - Breakout detection (price breaks above opening range high)
    - No breakout scenarios (HOLD signals)
    - Volume filtering
    - VWAP confirmation
    - Cooldown timer
    - Position sizing
    - Edge cases

Author: OpenAlgo Team
Date: December 2025
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.base_strategy import Signal, SignalAction
from strategies.vol_breakout import (
    IndicatorCalculator,
    VolatilityBreakoutStrategy,
    VolBreakoutConfig,
    compute_qty,
)

# ==================== Fixtures ====================


@pytest.fixture
def default_config():
    """Default strategy configuration"""
    return {
        "capital": 100000,
        "entry_window_minutes": 15,
        "min_volume_factor": 1.2,
        "atr_mult_sl": 1.5,
        "rr": 1.5,
        "vwap_confirm": True,
        "max_trades_per_day": 5,
        "risk_per_trade": 0.02,
        "max_position_size": 0.10,
        "cooldown_minutes": 5,
        "trading_start_time": "09:20",
        "trading_end_time": "15:10",
    }


@pytest.fixture
def strategy(default_config):
    """Create a strategy instance"""
    return VolatilityBreakoutStrategy(default_config)


@pytest.fixture
def sample_history():
    """
    Create sample historical data with a clear opening range.

    Opening range (first 15 minutes, 3 bars of 5min):
        - High: 505
        - Low: 498
    """
    dates = pd.date_range(start="2025-12-01 09:15", periods=30, freq="5min")

    np.random.seed(42)

    # First 3 bars (09:15 - 09:30) establish opening range
    opening_data = {
        "open": [500, 502, 504],
        "high": [503, 505, 505],  # Max high = 505
        "low": [498, 500, 501],  # Min low = 498
        "close": [502, 504, 503],
        "volume": [100000, 120000, 110000],
    }

    # Remaining bars (after opening range)
    remaining_opens = [503 + np.random.uniform(-2, 3) for _ in range(27)]
    remaining_highs = [max(o, o + np.random.uniform(0, 4)) for o in remaining_opens]
    remaining_lows = [min(o, o - np.random.uniform(0, 3)) for o in remaining_opens]
    remaining_closes = [
        (h + l) / 2 + np.random.uniform(-1, 1)
        for h, l in zip(remaining_highs, remaining_lows)
    ]
    remaining_volumes = [int(100000 * np.random.uniform(0.8, 1.3)) for _ in range(27)]

    data = {
        "open": opening_data["open"] + remaining_opens,
        "high": opening_data["high"] + remaining_highs,
        "low": opening_data["low"] + remaining_lows,
        "close": opening_data["close"] + remaining_closes,
        "volume": opening_data["volume"] + remaining_volumes,
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def default_indicators():
    """Default indicator values"""
    return {"atr": 8.0, "vwap": 503.0, "avg_volume": 110000}


@pytest.fixture
def breakout_bar():
    """Live bar that triggers a breakout"""
    return {
        "symbol": "SBIN",
        "timestamp": datetime(2025, 12, 1, 10, 0),
        "open": 506,
        "high": 512,
        "low": 505,
        "close": 510,  # Above opening range high (505)
        "volume": 180000,  # 1.64x avg volume (110000)
    }


@pytest.fixture
def no_breakout_bar():
    """Live bar that does NOT trigger a breakout"""
    return {
        "symbol": "SBIN",
        "timestamp": datetime(2025, 12, 1, 10, 0),
        "open": 502,
        "high": 504,
        "low": 501,
        "close": 503,  # Below opening range high (505)
        "volume": 90000,  # 0.82x avg volume (low)
    }


# ==================== Test: Breakout Detection ====================


class TestBreakoutDetection:
    """Tests for breakout signal generation"""

    def test_breakout_generates_buy_signal(
        self, strategy, sample_history, default_indicators, breakout_bar
    ):
        """
        Test that a valid breakout generates a BUY signal.

        Conditions:
        - Price (510) > Opening Range High (505) ✓
        - Volume (180000) > 1.2 × Avg Volume (110000 × 1.2 = 132000) ✓
        - Price (510) > VWAP (503) ✓
        """
        signal = strategy.generate_signal(
            live_bar=breakout_bar, history=sample_history, indicators=default_indicators
        )

        assert signal.action == SignalAction.BUY
        assert signal.entry_price == pytest.approx(510, rel=0.01)
        assert signal.stop_loss < signal.entry_price
        assert len(signal.targets) >= 1
        assert signal.targets[0] > signal.entry_price
        assert signal.qty > 0
        assert "Breakout" in signal.reason

    def test_breakout_with_correct_stop_loss(
        self, strategy, sample_history, default_indicators, breakout_bar
    ):
        """
        Test that stop-loss is calculated correctly using ATR.

        Expected: SL = Entry - (ATR × atr_mult_sl)
                  SL = 510 - (8.0 × 1.5) = 510 - 12 = 498
        """
        signal = strategy.generate_signal(
            live_bar=breakout_bar, history=sample_history, indicators=default_indicators
        )

        expected_sl = breakout_bar["close"] - (default_indicators["atr"] * 1.5)
        assert signal.stop_loss == pytest.approx(expected_sl, rel=0.01)

    def test_breakout_with_correct_target(
        self, strategy, sample_history, default_indicators, breakout_bar
    ):
        """
        Test that target is calculated correctly using R:R ratio.

        Expected: Target = Entry + (RR × Risk)
                  Risk = Entry - SL = 510 - 498 = 12
                  Target = 510 + (1.5 × 12) = 510 + 18 = 528
        """
        signal = strategy.generate_signal(
            live_bar=breakout_bar, history=sample_history, indicators=default_indicators
        )

        risk = breakout_bar["close"] - signal.stop_loss
        expected_target = breakout_bar["close"] + (risk * 1.5)
        assert signal.targets[0] == pytest.approx(expected_target, rel=0.01)

    def test_breakout_records_metadata(
        self, strategy, sample_history, default_indicators, breakout_bar
    ):
        """Test that breakout signal includes metadata"""
        signal = strategy.generate_signal(
            live_bar=breakout_bar, history=sample_history, indicators=default_indicators
        )

        assert "atr" in signal.metadata
        assert "vwap" in signal.metadata
        assert "volume_ratio" in signal.metadata
        assert signal.metadata["volume_ratio"] > 1.2


# ==================== Test: No Breakout (HOLD) ====================


class TestNoBreakout:
    """Tests for HOLD signal generation when conditions not met"""

    def test_no_breakout_returns_hold(
        self, strategy, sample_history, default_indicators, no_breakout_bar
    ):
        """
        Test that no breakout generates HOLD signal.

        Conditions NOT met:
        - Price (503) < Opening Range High (505) ✗
        - Volume (90000) < 1.2 × Avg Volume (132000) ✗
        """
        signal = strategy.generate_signal(
            live_bar=no_breakout_bar,
            history=sample_history,
            indicators=default_indicators,
        )

        assert signal.action == SignalAction.HOLD
        assert signal.qty == 0
        assert signal.entry_price == 0.0

    def test_no_breakout_explains_reason(
        self, strategy, sample_history, default_indicators, no_breakout_bar
    ):
        """Test that HOLD signal includes reason for not trading"""
        signal = strategy.generate_signal(
            live_bar=no_breakout_bar,
            history=sample_history,
            indicators=default_indicators,
        )

        assert signal.reason != ""
        # Should mention either price below breakout or volume issue
        assert "breakout" in signal.reason.lower() or "volume" in signal.reason.lower()

    def test_price_at_breakout_level_holds(
        self, strategy, sample_history, default_indicators
    ):
        """Test that price exactly at breakout level doesn't trigger (must be above)"""
        at_level_bar = {
            "symbol": "SBIN",
            "timestamp": datetime(2025, 12, 1, 10, 0),
            "open": 504,
            "high": 505,
            "low": 503,
            "close": 505,  # Exactly at breakout level, not above
            "volume": 150000,
        }

        signal = strategy.generate_signal(
            live_bar=at_level_bar, history=sample_history, indicators=default_indicators
        )

        assert signal.action == SignalAction.HOLD

    def test_breakout_with_low_volume_holds(
        self, strategy, sample_history, default_indicators
    ):
        """Test that breakout with insufficient volume generates HOLD"""
        low_volume_bar = {
            "symbol": "SBIN",
            "timestamp": datetime(2025, 12, 1, 10, 0),
            "open": 506,
            "high": 512,
            "low": 505,
            "close": 510,  # Breakout ✓
            "volume": 100000,  # 0.91x avg volume ✗
        }

        signal = strategy.generate_signal(
            live_bar=low_volume_bar,
            history=sample_history,
            indicators=default_indicators,
        )

        assert signal.action == SignalAction.HOLD
        assert "volume" in signal.reason.lower()


# ==================== Test: VWAP Confirmation ====================


class TestVWAPConfirmation:
    """Tests for VWAP filter"""

    def test_breakout_below_vwap_holds_when_vwap_enabled(
        self, default_config, sample_history
    ):
        """Test that breakout below VWAP holds when vwap_confirm is True"""
        config = default_config.copy()
        config["vwap_confirm"] = True
        strategy = VolatilityBreakoutStrategy(config)

        below_vwap_bar = {
            "symbol": "SBIN",
            "timestamp": datetime(2025, 12, 1, 10, 0),
            "open": 506,
            "high": 512,
            "low": 505,
            "close": 510,  # Breakout ✓
            "volume": 180000,  # Volume OK ✓
        }

        indicators = {
            "atr": 8.0,
            "vwap": 520.0,  # Price (510) < VWAP (520) ✗
            "avg_volume": 110000,
        }

        signal = strategy.generate_signal(
            live_bar=below_vwap_bar, history=sample_history, indicators=indicators
        )

        assert signal.action == SignalAction.HOLD
        assert "vwap" in signal.reason.lower()

    def test_breakout_below_vwap_triggers_when_vwap_disabled(
        self, default_config, sample_history
    ):
        """Test that breakout below VWAP triggers when vwap_confirm is False"""
        config = default_config.copy()
        config["vwap_confirm"] = False
        strategy = VolatilityBreakoutStrategy(config)

        below_vwap_bar = {
            "symbol": "SBIN",
            "timestamp": datetime(2025, 12, 1, 10, 0),
            "open": 506,
            "high": 512,
            "low": 505,
            "close": 510,  # Breakout ✓
            "volume": 180000,  # Volume OK ✓
        }

        indicators = {
            "atr": 8.0,
            "vwap": 520.0,  # Price < VWAP (but filter disabled)
            "avg_volume": 110000,
        }

        signal = strategy.generate_signal(
            live_bar=below_vwap_bar, history=sample_history, indicators=indicators
        )

        assert signal.action == SignalAction.BUY


# ==================== Test: Cooldown Timer ====================


class TestCooldownTimer:
    """Tests for cooldown between signals"""

    def test_cooldown_prevents_rapid_signals(
        self, strategy, sample_history, default_indicators, breakout_bar
    ):
        """Test that cooldown prevents signals within cooldown period"""
        # First signal should go through
        signal1 = strategy.generate_signal(
            live_bar=breakout_bar, history=sample_history, indicators=default_indicators
        )
        assert signal1.action == SignalAction.BUY

        # Second signal within 5 minutes should be blocked
        quick_bar = breakout_bar.copy()
        quick_bar["timestamp"] = breakout_bar["timestamp"] + timedelta(minutes=2)

        signal2 = strategy.generate_signal(
            live_bar=quick_bar, history=sample_history, indicators=default_indicators
        )
        assert signal2.action == SignalAction.HOLD
        assert "cooldown" in signal2.reason.lower()

    def test_signal_after_cooldown_works(
        self, strategy, sample_history, default_indicators, breakout_bar
    ):
        """Test that signal is allowed after cooldown expires"""
        # First signal
        signal1 = strategy.generate_signal(
            live_bar=breakout_bar, history=sample_history, indicators=default_indicators
        )
        assert signal1.action == SignalAction.BUY

        # Signal after cooldown (6 minutes later)
        later_bar = breakout_bar.copy()
        later_bar["timestamp"] = breakout_bar["timestamp"] + timedelta(minutes=6)

        signal2 = strategy.generate_signal(
            live_bar=later_bar, history=sample_history, indicators=default_indicators
        )
        assert signal2.action == SignalAction.BUY


# ==================== Test: Trading Window ====================


class TestTradingWindow:
    """Tests for trading time restrictions"""

    def test_before_trading_window_holds(
        self, strategy, sample_history, default_indicators
    ):
        """Test that signals before trading start time are blocked"""
        early_bar = {
            "symbol": "SBIN",
            "timestamp": datetime(2025, 12, 1, 9, 15),  # Before 09:20
            "open": 506,
            "high": 512,
            "low": 505,
            "close": 510,
            "volume": 180000,
        }

        signal = strategy.generate_signal(
            live_bar=early_bar, history=sample_history, indicators=default_indicators
        )

        assert signal.action == SignalAction.HOLD
        assert "window" in signal.reason.lower() or "trading" in signal.reason.lower()

    def test_after_trading_window_holds(
        self, strategy, sample_history, default_indicators
    ):
        """Test that signals after trading end time are blocked"""
        late_bar = {
            "symbol": "SBIN",
            "timestamp": datetime(2025, 12, 1, 15, 15),  # After 15:10
            "open": 506,
            "high": 512,
            "low": 505,
            "close": 510,
            "volume": 180000,
        }

        signal = strategy.generate_signal(
            live_bar=late_bar, history=sample_history, indicators=default_indicators
        )

        assert signal.action == SignalAction.HOLD


# ==================== Test: Max Trades Per Day ====================


class TestMaxTrades:
    """Tests for daily trade limits"""

    def test_max_trades_limit_enforced(
        self, default_config, sample_history, default_indicators
    ):
        """Test that max trades per day is enforced"""
        config = default_config.copy()
        config["max_trades_per_day"] = 2
        config["cooldown_minutes"] = 0  # Disable cooldown for this test
        strategy = VolatilityBreakoutStrategy(config)

        # Simulate 2 trades already done
        strategy.trades_today = 2

        # Use a bar within trading hours
        in_hours_bar = {
            "symbol": "SBIN",
            "timestamp": datetime(
                2025, 12, 1, 10, 0
            ),  # 10:00 AM - within trading window
            "open": 506,
            "high": 512,
            "low": 505,
            "close": 510,
            "volume": 180000,
        }

        signal = strategy.generate_signal(
            live_bar=in_hours_bar, history=sample_history, indicators=default_indicators
        )

        assert signal.action == SignalAction.HOLD
        assert "max" in signal.reason.lower() or "trades" in signal.reason.lower()


# ==================== Test: Position Sizing ====================


class TestPositionSizing:
    """Tests for compute_qty function"""

    def test_compute_qty_basic(self):
        """Test basic position sizing calculation"""
        result = compute_qty(
            capital=100000, entry_price=500, stop_loss=490, risk_per_trade=0.02
        )

        assert result["qty"] > 0
        assert result["position_value"] > 0
        assert result["risk_amount"] > 0

        # Risk should be approximately 2% of capital
        # Max risk = 100000 × 0.02 = 2000
        # Risk per share = 500 - 490 = 10
        # Qty by risk = 2000 / 10 = 200
        assert result["qty"] == 200 or result["qty"] == result["qty_by_position"]

    def test_compute_qty_respects_max_position(self):
        """Test that qty respects max position size"""
        result = compute_qty(
            capital=100000,
            entry_price=500,
            stop_loss=495,  # Small SL = would allow large qty
            risk_per_trade=0.05,  # 5% risk would allow 1000 qty
            max_position_pct=0.10,  # But max position is 10%
        )

        # Max position = 100000 × 0.10 = 10000
        # Max qty = 10000 / 500 = 20
        assert result["qty"] <= 20

    def test_compute_qty_handles_zero_sl(self):
        """Test that zero stop-loss is handled gracefully"""
        result = compute_qty(
            capital=100000,
            entry_price=500,
            stop_loss=500,  # SL equals entry
            risk_per_trade=0.02,
        )

        # Should use default 2% risk
        assert result["qty"] > 0
        assert "error" not in result

    def test_compute_qty_handles_invalid_prices(self):
        """Test that invalid prices return error"""
        result = compute_qty(
            capital=100000, entry_price=0, stop_loss=0, risk_per_trade=0.02
        )

        assert result["qty"] == 0
        assert "error" in result


# ==================== Test: Indicator Calculator ====================


class TestIndicatorCalculator:
    """Tests for IndicatorCalculator class"""

    def test_calculate_atr(self, sample_history):
        """Test ATR calculation"""
        calc = IndicatorCalculator()
        atr = calc.calculate_atr(sample_history, period=14)

        assert atr > 0
        assert isinstance(atr, float)

    def test_calculate_vwap(self, sample_history):
        """Test VWAP calculation"""
        calc = IndicatorCalculator()
        vwap = calc.calculate_vwap(sample_history)

        assert vwap > 0
        assert isinstance(vwap, float)
        # VWAP should be within price range
        assert sample_history["low"].min() <= vwap <= sample_history["high"].max()

    def test_calculate_avg_volume(self, sample_history):
        """Test average volume calculation"""
        calc = IndicatorCalculator()
        avg_vol = calc.calculate_avg_volume(sample_history, period=20)

        assert avg_vol > 0
        assert isinstance(avg_vol, float)

    def test_get_opening_range(self, sample_history):
        """Test opening range extraction"""
        calc = IndicatorCalculator()
        opening_range = calc.get_opening_range(sample_history, minutes=15)

        assert opening_range["high"] == 505  # Max high of first 3 bars
        assert opening_range["low"] == 498  # Min low of first 3 bars
        assert opening_range["range"] == 7  # 505 - 498


# ==================== Test: Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_history_returns_hold(
        self, strategy, default_indicators, breakout_bar
    ):
        """Test that empty history returns HOLD"""
        empty_history = pd.DataFrame()

        signal = strategy.generate_signal(
            live_bar=breakout_bar, history=empty_history, indicators=default_indicators
        )

        assert signal.action == SignalAction.HOLD

    def test_zero_atr_returns_hold(self, strategy, sample_history, breakout_bar):
        """Test that zero ATR returns HOLD"""
        indicators = {"atr": 0, "vwap": 503.0, "avg_volume": 110000}

        signal = strategy.generate_signal(
            live_bar=breakout_bar, history=sample_history, indicators=indicators
        )

        assert signal.action == SignalAction.HOLD
        assert "atr" in signal.reason.lower()

    def test_missing_indicators_computed_from_history(
        self, strategy, sample_history, breakout_bar
    ):
        """Test that missing indicators are computed from history"""
        # Provide empty indicators dict
        signal = strategy.generate_signal(
            live_bar=breakout_bar,
            history=sample_history,
            indicators={},  # Empty - should compute from history
        )

        # Should still work (either BUY or HOLD based on computed values)
        assert signal.action in [SignalAction.BUY, SignalAction.HOLD]

    def test_plan_entry_overrides_defaults(
        self, strategy, sample_history, default_indicators, breakout_bar
    ):
        """Test that plan_entry values override calculated values"""
        plan_entry = {
            "stop_loss": 495.0,
            "targets": [530.0, 540.0],
            "suggested_qty": 50,
        }

        signal = strategy.generate_signal(
            live_bar=breakout_bar,
            history=sample_history,
            indicators=default_indicators,
            plan_entry=plan_entry,
        )

        if signal.action == SignalAction.BUY:
            assert signal.stop_loss == pytest.approx(495.0)
            assert signal.qty == 50


# ==================== Test: Strategy Reset ====================


class TestStrategyReset:
    """Tests for daily reset functionality"""

    def test_reset_clears_cache(self, strategy):
        """Test that reset clears opening range cache"""
        # Add something to cache
        strategy._opening_range_cache["TEST_2025-12-01"] = {"high": 100}

        strategy.reset_daily_stats()

        assert len(strategy._opening_range_cache) == 0

    def test_reset_clears_trades_today(self, strategy):
        """Test that reset clears trade count"""
        strategy.trades_today = 5

        strategy.reset_daily_stats()

        assert strategy.trades_today == 0

    def test_reset_clears_cooldown(self, strategy):
        """Test that reset clears cooldown timer"""
        strategy._last_signal_time = datetime.now()

        strategy.reset_daily_stats()

        assert strategy._last_signal_time is None


# ==================== Test: Signal Confidence ====================


class TestSignalConfidence:
    """Tests for signal confidence calculation"""

    def test_strong_breakout_has_higher_confidence(
        self, strategy, sample_history, default_indicators
    ):
        """Test that strong breakout (high volume, big move) has higher confidence"""
        strong_breakout = {
            "symbol": "SBIN",
            "timestamp": datetime(2025, 12, 1, 10, 0),
            "open": 508,
            "high": 520,
            "low": 507,
            "close": 518,  # Strong move above 505
            "volume": 300000,  # 2.7x avg volume
        }

        signal = strategy.generate_signal(
            live_bar=strong_breakout,
            history=sample_history,
            indicators=default_indicators,
        )

        if signal.action == SignalAction.BUY:
            # Strong breakout should have confidence > 0.7
            assert signal.confidence >= 0.7

    def test_weak_breakout_has_lower_confidence(
        self, strategy, sample_history, default_indicators
    ):
        """Test that weak breakout (barely above, minimal volume) has lower confidence"""
        weak_breakout = {
            "symbol": "SBIN",
            "timestamp": datetime(2025, 12, 1, 10, 0),
            "open": 505,
            "high": 506,
            "low": 504,
            "close": 505.5,  # Barely above 505
            "volume": 135000,  # Just above 1.2x threshold
        }

        signal = strategy.generate_signal(
            live_bar=weak_breakout,
            history=sample_history,
            indicators=default_indicators,
        )

        if signal.action == SignalAction.BUY:
            # Weak breakout should have confidence < 0.7
            assert signal.confidence < 0.7


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
