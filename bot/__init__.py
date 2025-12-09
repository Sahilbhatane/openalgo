# OpenAlgo Intraday Trading Bot
# ================================
# A fully autonomous intraday trading system built on OpenAlgo framework

"""
PROJECT STRUCTURE
=================

bot/
├── __init__.py              # Bot package initialization
├── main.py                  # Main entry point - starts all services
│
├── core/                    # Core configuration and utilities
│   ├── __init__.py
│   ├── config.py           # Central configuration (PAPER/LIVE mode, etc.)
│   ├── constants.py        # Trading constants, market hours, etc.
│   ├── mode.py             # Mode management (PAPER_MODE, LIVE_MODE)
│   └── logging_config.py   # Centralized logging setup
│
├── data/                    # Already created - data fetching and validation
│   ├── __init__.py
│   ├── historical_client.py
│   ├── market_feed.py
│   └── validators.py
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── charges.py          # Brokerage, STT, GST, stamp duty calculation
│   ├── indicators.py       # Technical indicators (ATR, RSI, VWAP, EMA, etc.)
│   ├── metrics.py          # Performance metrics (Sharpe, Sortino, etc.)
│   └── time_utils.py       # IST time handling, market hours checks
│
├── risk/                    # Risk management
│   ├── __init__.py
│   ├── risk_manager.py     # Max drawdown, exposure limits, daily limits
│   ├── position_sizer.py   # Calculate position size based on risk
│   ├── order_validator.py  # Validate orders before execution
│   └── kill_switch.py      # Emergency stop - square off all positions
│
├── strategies/              # Trading strategies
│   ├── __init__.py
│   ├── base_strategy.py    # Abstract base class for all strategies
│   ├── volatility_breakout.py  # Main strategy: volatility breakout + VWAP
│   └── signal.py           # Signal classes (BUY, SELL, HOLD)
│
├── jobs/                    # Scheduled jobs
│   ├── __init__.py
│   ├── scheduler.py        # APScheduler setup for all jobs
│   ├── morning_learning.py # 08:00 IST - research and plan generation
│   ├── market_open.py      # 09:15 IST - start trading
│   ├── square_off.py       # 15:10 IST - close all positions
│   └── end_of_day.py       # 15:35 IST - generate daily report
│
├── execution/               # Order execution
│   ├── __init__.py
│   ├── order_manager.py    # Order lifecycle management
│   ├── paper_executor.py   # Paper trading execution (simulation)
│   └── live_executor.py    # Live trading execution (via OpenAlgo)
│
├── reports/                 # Reporting and analytics
│   ├── __init__.py
│   ├── daily_report.py     # Daily trade report generation
│   ├── performance.py      # Equity curve and performance tracking
│   └── export.py           # CSV/HTML/JSON export
│
├── quant/                   # Advanced quantitative analysis
│   ├── __init__.py
│   ├── monte_carlo.py      # Monte Carlo simulation (30,000+ paths)
│   └── advanced_analysis.py # Market regime, Hurst exponent, VaR
│
└── tests/                   # Unit tests
    ├── __init__.py
    ├── test_charges.py
    ├── test_risk_manager.py
    ├── test_strategies.py
    └── test_validators.py


USAGE
=====
    # Start the bot in paper mode (default)
    python -m bot.main
    
    # With debug logging
    python -m bot.main --debug


HUMAN STEPS REQUIRED BEFORE RUNNING
====================================
1. Set TRADING_MODE='PAPER' in .env (default, safe)
2. Paper trade for 3-4 weeks minimum
3. Review daily reports and fix any issues
4. Only after consistent profitability, manually change to LIVE mode
5. Start LIVE mode with 10% of intended capital
6. Gradually increase capital over weeks

NEVER:
- Set TRADING_MODE='LIVE' without 3-4 weeks paper testing
- Run without reviewing daily reports
- Trade more than you can afford to lose
- Ignore kill switch triggers
"""

__version__ = "1.0.0"
__author__ = "OpenAlgo Trading Bot"

# Core imports
from .core.config import Config
from .core.mode import TradingMode, ModeManager
from .core.constants import TradingSession

# Quant imports
from .quant.monte_carlo import MonteCarloSimulator, MonteCarloConfig, MonteCarloResult
from .quant.advanced_analysis import AdvancedQuantAnalysis, MarketRegime, VolatilityRegime

# Main entry
from .main import TradingBot

__all__ = [
    "TradingBot",
    "Config",
    "TradingMode",
    "ModeManager",
    "TradingSession",
    # Quant
    "MonteCarloSimulator",
    "MonteCarloConfig",
    "MonteCarloResult",
    "AdvancedQuantAnalysis",
    "MarketRegime",
    "VolatilityRegime",
]

