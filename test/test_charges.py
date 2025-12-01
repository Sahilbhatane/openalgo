"""
Unit Tests for Trading Charges Calculator (utils/charges.py)

Tests cover all charge calculation functions with various scenarios and edge cases.

Author: OpenAlgo Team
Date: December 2025
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from utils.charges import (
    api_daily_amort,
    breakeven_analysis,
    brokerage,
    exchange_fee,
    gst,
    per_trade_gross_needed,
    round_trip_cost,
    sebi_charges,
    stamp_duty,
    stt,
    total_trade_cost,
)

# ==================== Test Data ====================

# Sample turnover values for testing
TURNOVER_SMALL = 10000  # ₹10k
TURNOVER_MEDIUM = 50000  # ₹50k
TURNOVER_LARGE = 100000  # ₹1 lakh
TURNOVER_XLARGE = 500000  # ₹5 lakh
TURNOVER_XXLARGE = 1000000  # ₹10 lakh


# ==================== Brokerage Tests ====================


class TestBrokerage:
    """Test brokerage calculation"""

    def test_brokerage_under_cap(self):
        """Test brokerage when calculated amount is under cap"""
        # ₹10k turnover: 0.03% = ₹3 (under ₹20 cap)
        result = brokerage(TURNOVER_SMALL)
        assert result == pytest.approx(3.0, rel=0.01)

    def test_brokerage_at_cap(self):
        """Test brokerage when it hits the cap"""
        # ₹1 lakh turnover: 0.03% = ₹30, but capped at ₹20
        result = brokerage(TURNOVER_LARGE)
        assert result == 20.0

    def test_brokerage_well_over_cap(self):
        """Test brokerage for large turnover"""
        # ₹10 lakh turnover: 0.03% = ₹300, but capped at ₹20
        result = brokerage(TURNOVER_XXLARGE)
        assert result == 20.0

    def test_brokerage_zero_turnover(self):
        """Test brokerage with zero turnover"""
        result = brokerage(0)
        assert result == 0.0

    def test_brokerage_custom_rate(self):
        """Test brokerage with custom rate"""
        # 0.05% with ₹50 cap
        result = brokerage(TURNOVER_MEDIUM, brokerage_rate=0.0005, cap_per_order=50)
        assert result == 25.0

    def test_brokerage_negative_turnover(self):
        """Test brokerage with negative turnover"""
        result = brokerage(-1000)
        assert result == 0.0


# ==================== GST Tests ====================


class TestGST:
    """Test GST calculation"""

    def test_gst_basic(self):
        """Test basic GST calculation"""
        # GST on ₹20 brokerage + ₹10 txn charges = ₹30 * 18% = ₹5.40
        result = gst(20.0, 10.0)
        assert result == pytest.approx(5.4, rel=0.01)

    def test_gst_zero_charges(self):
        """Test GST with zero charges"""
        result = gst(0.0, 0.0)
        assert result == 0.0

    def test_gst_only_brokerage(self):
        """Test GST with only brokerage"""
        result = gst(100.0, 0.0)
        assert result == pytest.approx(18.0, rel=0.01)

    def test_gst_only_txn_charges(self):
        """Test GST with only transaction charges"""
        result = gst(0.0, 50.0)
        assert result == pytest.approx(9.0, rel=0.01)

    def test_gst_custom_rate(self):
        """Test GST with custom rate"""
        # 12% GST
        result = gst(100.0, 50.0, gst_rate=0.12)
        assert result == pytest.approx(18.0, rel=0.01)


# ==================== STT Tests ====================


class TestSTT:
    """Test Securities Transaction Tax calculation"""

    def test_stt_equity_delivery_buy(self):
        """Test STT for equity delivery buy"""
        # 0.1% on ₹1 lakh = ₹100
        result = stt(TURNOVER_LARGE, "equity_delivery", "buy")
        assert result == pytest.approx(100.0, rel=0.01)

    def test_stt_equity_delivery_sell(self):
        """Test STT for equity delivery sell"""
        # 0.1% on ₹1 lakh = ₹100
        result = stt(TURNOVER_LARGE, "equity_delivery", "sell")
        assert result == pytest.approx(100.0, rel=0.01)

    def test_stt_equity_intraday_buy(self):
        """Test STT for equity intraday buy"""
        # No STT on buy side for intraday
        result = stt(TURNOVER_LARGE, "equity_intraday", "buy")
        assert result == 0.0

    def test_stt_equity_intraday_sell(self):
        """Test STT for equity intraday sell"""
        # 0.025% on ₹1 lakh = ₹25
        result = stt(TURNOVER_LARGE, "equity_intraday", "sell")
        assert result == pytest.approx(25.0, rel=0.01)

    def test_stt_equity_options_sell(self):
        """Test STT for equity options sell"""
        # 0.05% on ₹1 lakh = ₹50
        result = stt(TURNOVER_LARGE, "equity_options", "sell")
        assert result == pytest.approx(50.0, rel=0.01)

    def test_stt_futures_sell(self):
        """Test STT for futures sell"""
        # 0.001% on ₹1 lakh = ₹1
        result = stt(TURNOVER_LARGE, "futures", "sell")
        assert result == pytest.approx(1.0, rel=0.01)

    def test_stt_zero_turnover(self):
        """Test STT with zero turnover"""
        result = stt(0, "equity_intraday", "sell")
        assert result == 0.0


# ==================== Exchange Fee Tests ====================


class TestExchangeFee:
    """Test exchange transaction fee calculation"""

    def test_exchange_fee_equity_buy(self):
        """Test exchange fee for equity buy"""
        result = exchange_fee(TURNOVER_LARGE, "equity", side="buy")
        # Should include exchange charges + SEBI + stamp duty
        assert result > 0
        assert result < TURNOVER_LARGE * 0.001  # Sanity check

    def test_exchange_fee_equity_sell(self):
        """Test exchange fee for equity sell"""
        result = exchange_fee(TURNOVER_LARGE, "equity", side="sell")
        # Should include exchange charges + SEBI (no stamp duty on sell)
        assert result > 0
        assert result < TURNOVER_LARGE * 0.001

    def test_exchange_fee_fo(self):
        """Test exchange fee for F&O"""
        result = exchange_fee(TURNOVER_LARGE, "fo", side="buy")
        assert result > 0

    def test_exchange_fee_without_sebi(self):
        """Test exchange fee excluding SEBI charges"""
        result = exchange_fee(TURNOVER_LARGE, "equity", include_sebi=False)
        result_with_sebi = exchange_fee(TURNOVER_LARGE, "equity", include_sebi=True)
        assert result < result_with_sebi

    def test_exchange_fee_without_stamp_duty(self):
        """Test exchange fee excluding stamp duty"""
        result = exchange_fee(
            TURNOVER_LARGE, "equity", include_stamp_duty=False, side="buy"
        )
        result_with_stamp = exchange_fee(
            TURNOVER_LARGE, "equity", include_stamp_duty=True, side="buy"
        )
        assert result < result_with_stamp


# ==================== API Daily Amortization Tests ====================


class TestAPIDailyAmort:
    """Test API subscription daily amortization"""

    def test_api_daily_amort_basic(self):
        """Test basic daily amortization"""
        # ₹1000 per month / 20 days = ₹50 per day
        result = api_daily_amort(1000, 20)
        assert result == 50.0

    def test_api_daily_amort_custom_days(self):
        """Test with custom trading days"""
        # ₹1500 per month / 25 days = ₹60 per day
        result = api_daily_amort(1500, 25)
        assert result == 60.0

    def test_api_daily_amort_zero_subscription(self):
        """Test with zero subscription"""
        result = api_daily_amort(0, 20)
        assert result == 0.0

    def test_api_daily_amort_zero_days(self):
        """Test with zero trading days"""
        result = api_daily_amort(1000, 0)
        assert result == 0.0


# ==================== SEBI Charges Tests ====================


class TestSEBICharges:
    """Test SEBI turnover charges"""

    def test_sebi_charges_one_crore(self):
        """Test SEBI charges for ₹1 crore turnover"""
        # ₹10 per crore
        result = sebi_charges(10000000)
        assert result == pytest.approx(10.0, rel=0.01)

    def test_sebi_charges_one_lakh(self):
        """Test SEBI charges for ₹1 lakh turnover"""
        result = sebi_charges(TURNOVER_LARGE)
        assert result == pytest.approx(0.1, rel=0.01)


# ==================== Stamp Duty Tests ====================


class TestStampDuty:
    """Test stamp duty calculation"""

    def test_stamp_duty_one_lakh(self):
        """Test stamp duty for ₹1 lakh"""
        # 0.003% of ₹1 lakh = ₹3
        result = stamp_duty(TURNOVER_LARGE)
        assert result == pytest.approx(3.0, rel=0.01)

    def test_stamp_duty_one_crore(self):
        """Test stamp duty for ₹1 crore"""
        # 0.003% of ₹1 crore = ₹300
        result = stamp_duty(10000000)
        assert result == pytest.approx(300.0, rel=0.01)


# ==================== Total Trade Cost Tests ====================


class TestTotalTradeCost:
    """Test comprehensive trade cost calculation"""

    def test_total_trade_cost_intraday_buy(self):
        """Test total cost for intraday buy"""
        result = total_trade_cost(
            TURNOVER_LARGE, "equity_intraday", "buy", subscription_monthly=1000
        )

        assert "turnover" in result
        assert "brokerage" in result
        assert "stt" in result
        assert "total_cost" in result
        assert result["turnover"] == TURNOVER_LARGE
        assert result["total_cost"] > 0
        assert result["stt"] == 0.0  # No STT on buy for intraday

    def test_total_trade_cost_intraday_sell(self):
        """Test total cost for intraday sell"""
        result = total_trade_cost(
            TURNOVER_LARGE, "equity_intraday", "sell", subscription_monthly=1000
        )

        assert result["stt"] > 0  # STT on sell for intraday
        assert result["stamp_duty"] == 0.0  # No stamp duty on sell
        assert result["total_cost"] > 0

    def test_total_trade_cost_delivery(self):
        """Test total cost for delivery trade"""
        result = total_trade_cost(
            TURNOVER_LARGE, "equity_delivery", "buy", subscription_monthly=1000
        )

        assert result["stt"] > 0  # STT on both buy and sell for delivery
        assert result["total_cost"] > 0

    def test_total_trade_cost_zero_turnover(self):
        """Test total cost with zero turnover"""
        result = total_trade_cost(0, "equity_intraday", "buy")

        assert result["total_cost"] == 0.0
        assert result["brokerage"] == 0.0

    def test_total_trade_cost_without_subscription(self):
        """Test total cost without API subscription"""
        result = total_trade_cost(TURNOVER_LARGE, "equity_intraday", "sell")

        assert result["api_cost_daily"] == 0.0
        assert result["total_cost"] > 0

    def test_total_trade_cost_structure(self):
        """Test that result has all expected keys"""
        result = total_trade_cost(TURNOVER_LARGE, "equity_intraday", "sell")

        expected_keys = [
            "turnover",
            "brokerage",
            "stt",
            "exchange_charges",
            "sebi_charges",
            "stamp_duty",
            "gst",
            "api_cost_daily",
            "total_cost",
            "cost_percentage",
        ]

        for key in expected_keys:
            assert key in result


# ==================== Round Trip Cost Tests ====================


class TestRoundTripCost:
    """Test round trip (buy + sell) cost calculation"""

    def test_round_trip_cost_basic(self):
        """Test basic round trip cost"""
        result = round_trip_cost(TURNOVER_LARGE, "equity_intraday", 1000)

        assert "buy_cost" in result
        assert "sell_cost" in result
        assert "total_cost" in result
        assert result["total_cost"] == result["buy_cost"] + result["sell_cost"]
        assert result["turnover"] == TURNOVER_LARGE * 2

    def test_round_trip_cost_includes_both_legs(self):
        """Test that round trip includes costs from both buy and sell"""
        result = round_trip_cost(TURNOVER_LARGE, "equity_intraday", 1000)

        # Buy side should have stamp duty, sell side should have STT
        assert result["buy_breakdown"]["stamp_duty"] > 0
        assert result["sell_breakdown"]["stt"] > 0

    def test_round_trip_cost_delivery(self):
        """Test round trip for delivery trades"""
        result = round_trip_cost(TURNOVER_LARGE, "equity_delivery", 1000)

        # Both sides should have STT for delivery
        assert result["buy_breakdown"]["stt"] > 0
        assert result["sell_breakdown"]["stt"] > 0


# ==================== Per Trade Gross Needed Tests ====================


class TestPerTradeGrossNeeded:
    """Test calculation of required gross profit per trade"""

    def test_per_trade_gross_basic(self):
        """Test basic gross profit calculation"""
        result = per_trade_gross_needed(
            balance=100000, target_net=1000, trades_per_day=10, subscription_daily=50
        )

        assert "gross_per_trade" in result
        assert "breakeven_per_trade" in result
        assert "cost_per_trade" in result
        assert result["gross_per_trade"] > result["breakeven_per_trade"]

    def test_per_trade_gross_with_large_target(self):
        """Test with large net profit target"""
        result = per_trade_gross_needed(
            balance=500000, target_net=5000, trades_per_day=20, subscription_daily=100
        )

        assert result["gross_per_trade"] > 0
        assert result["total_gross_needed"] > result["target_net_daily"]

    def test_per_trade_gross_zero_trades(self):
        """Test with zero trades per day"""
        result = per_trade_gross_needed(
            balance=100000, target_net=1000, trades_per_day=0, subscription_daily=50
        )

        assert "error" in result

    def test_per_trade_gross_structure(self):
        """Test that result has all expected keys"""
        result = per_trade_gross_needed(
            balance=100000, target_net=1000, trades_per_day=10, subscription_daily=50
        )

        expected_keys = [
            "balance",
            "target_net_daily",
            "trades_per_day",
            "avg_turnover_per_trade",
            "cost_per_trade",
            "total_daily_costs",
            "subscription_daily",
            "breakeven_per_trade",
            "gross_per_trade",
            "gross_percentage",
            "total_gross_needed",
        ]

        for key in expected_keys:
            assert key in result


# ==================== Breakeven Analysis Tests ====================


class TestBreakevenAnalysis:
    """Test breakeven analysis calculation"""

    def test_breakeven_analysis_basic(self):
        """Test basic breakeven analysis"""
        result = breakeven_analysis(
            capital=500000, subscription_monthly=1000, trades_per_day=10
        )

        assert "monthly_breakeven" in result
        assert "daily_breakeven" in result
        assert "per_trade_breakeven" in result
        assert result["monthly_breakeven"] > 0

    def test_breakeven_analysis_structure(self):
        """Test that result has all expected keys"""
        result = breakeven_analysis(500000, 1000)

        expected_keys = [
            "capital",
            "subscription_monthly",
            "trades_per_day",
            "trading_days_per_month",
            "avg_turnover_per_trade",
            "cost_per_round_trip",
            "monthly_trading_costs",
            "monthly_subscription",
            "monthly_total_costs",
            "monthly_breakeven",
            "daily_breakeven",
            "per_trade_breakeven",
            "breakeven_percentage",
        ]

        for key in expected_keys:
            assert key in result

    def test_breakeven_analysis_high_frequency(self):
        """Test breakeven for high frequency trading"""
        result = breakeven_analysis(
            capital=1000000, subscription_monthly=2000, trades_per_day=50
        )

        assert result["per_trade_breakeven"] > 0
        assert result["monthly_breakeven"] > result["subscription_monthly"]


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for realistic scenarios"""

    def test_realistic_day_trader_scenario(self):
        """Test realistic day trader scenario"""
        # Day trader with ₹5 lakh capital, 20 trades/day, ₹1000/month subscription
        capital = 500000
        trades_per_day = 20
        subscription_monthly = 1000

        # Calculate breakeven
        be = breakeven_analysis(capital, subscription_monthly, trades_per_day)

        assert be["daily_breakeven"] > 0
        assert be["per_trade_breakeven"] > 0

        # Check if breakeven is reasonable (should be < 1% of turnover per trade)
        assert be["breakeven_percentage"] < 1.0

    def test_realistic_swing_trader_scenario(self):
        """Test realistic swing trader scenario"""
        # Swing trader with ₹10 lakh capital, 5 trades/day, ₹2000/month subscription
        capital = 1000000
        trades_per_day = 5
        subscription_monthly = 2000

        be = breakeven_analysis(capital, subscription_monthly, trades_per_day)

        assert be["monthly_breakeven"] > subscription_monthly
        assert be["per_trade_breakeven"] > 0

    def test_cost_comparison_different_turnovers(self):
        """Test cost comparison across different turnover sizes"""
        turnovers = [10000, 50000, 100000, 500000, 1000000]
        results = []

        for turnover in turnovers:
            cost = total_trade_cost(turnover, "equity_intraday", "sell")
            results.append(
                {
                    "turnover": turnover,
                    "cost": cost["total_cost"],
                    "percentage": cost["cost_percentage"],
                }
            )

        # Cost percentage should generally decrease with higher turnover
        # (due to brokerage cap)
        assert results[0]["percentage"] > results[-1]["percentage"]


# ==================== Edge Cases Tests ====================


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_small_turnover(self):
        """Test with very small turnover"""
        result = total_trade_cost(100, "equity_intraday", "sell")
        assert result["total_cost"] >= 0

    def test_very_large_turnover(self):
        """Test with very large turnover"""
        result = total_trade_cost(10000000, "equity_intraday", "sell")
        assert result["total_cost"] > 0
        assert result["brokerage"] == 20.0  # Should hit cap

    def test_zero_subscription(self):
        """Test with zero subscription"""
        result = total_trade_cost(TURNOVER_LARGE, "equity_intraday", "sell", 0)
        assert result["api_cost_daily"] == 0.0

    def test_negative_values_protection(self):
        """Test that functions handle negative values gracefully"""
        assert brokerage(-1000) == 0.0
        assert stt(-1000, "equity_intraday", "sell") == 0.0


# ==================== Performance Tests ====================


class TestPerformance:
    """Test calculation performance"""

    def test_bulk_calculation_performance(self):
        """Test performance with bulk calculations"""
        import time

        start = time.time()

        # Calculate costs for 1000 trades
        for _ in range(1000):
            total_trade_cost(TURNOVER_LARGE, "equity_intraday", "sell", 1000)

        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0


# ==================== Run Tests ====================


if __name__ == "__main__":
    """
    Run all tests with pytest

    Usage:
        python test_charges.py
        or
        pytest test_charges.py -v
    """
    pytest.main([__file__, "-v", "--tb=short"])
