"""
Morning Learning Job for OpenAlgo Intraday Trading Bot

This module implements the pre-market analysis job that runs at 08:00 IST daily.
It analyzes symbols from a watchlist, computes technical indicators, fetches
news sentiment, and generates a trading plan (today_plan.json).

Features:
    - Loads symbols from config/watchlist.json
    - Fetches historical OHLC data (1m, 5m, 1d intervals)
    - Computes indicators: ATR(14), RSI(14), VWAP gap, momentum, volume
    - Fetches 48-hour news headlines (with fallback to price-only scoring)
    - Computes composite score: liquidity × volatility × momentum × (1 + sentiment)
    - Ranks and selects top-K symbols
    - Generates entry zones, stop-loss, targets, and position sizing
    - Uses charges module for realistic cost calculations

Usage:
    # As CLI
    python -m jobs.morning_learning --test          # Test mode with sample symbols
    python -m jobs.morning_learning                  # Full watchlist analysis
    python -m jobs.morning_learning --symbols TCS,INFY,RELIANCE  # Custom symbols
    python -m jobs.morning_learning --top-k 5       # Select top 5 symbols

    # As async function
    import asyncio
    from jobs.morning_learning import run
    asyncio.run(run(test_mode=True))

Author: OpenAlgo Team
Date: December 2025
"""

import argparse
import asyncio
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.charges import (
    DEFAULT_BROKERAGE_CAP,
    DEFAULT_BROKERAGE_RATE,
    per_trade_gross_needed,
    round_trip_cost,
    total_trade_cost,
)
from utils.logging import get_logger

logger = get_logger(__name__)


# ==================== Configuration Constants ====================

# Default paths
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
WATCHLIST_PATH = CONFIG_DIR / "watchlist.json"
OUTPUT_PATH = DATA_DIR / "today_plan.json"

# Test mode symbols
TEST_SYMBOLS = ["TCS", "RELIANCE", "INFY"]

# Default parameters
DEFAULT_TOP_K = 10
DEFAULT_ATR_PERIOD = 14
DEFAULT_RSI_PERIOD = 14
DEFAULT_VOLUME_PERIOD = 20
DEFAULT_NEWS_HOURS = 48

# Risk parameters
DEFAULT_RISK_PER_TRADE = 0.02  # 2% risk per trade
DEFAULT_MAX_POSITION_SIZE = 0.10  # Max 10% of capital per position
DEFAULT_ATR_SL_MULTIPLIER = 2.0  # Stop-loss at 2x ATR
DEFAULT_TARGET_RR_RATIO = 1.5  # 1.5:1 reward-risk ratio

# Trading days and costs
TRADING_DAYS_PER_MONTH = 20
API_SUBSCRIPTION_MONTHLY = 0.0  # Set via env or config


# ==================== Data Classes ====================


@dataclass
class IndicatorData:
    """Container for computed technical indicators"""

    atr_14: float = 0.0
    rsi_14: float = 50.0
    vwap: float = 0.0
    vwap_gap_pct: float = 0.0  # % distance from VWAP
    momentum_5m: float = 0.0  # 5-min momentum
    momentum_15m: float = 0.0  # 15-min momentum
    momentum_30m: float = 0.0  # 30-min momentum
    avg_volume_20: float = 0.0
    current_price: float = 0.0
    high_of_day: float = 0.0
    low_of_day: float = 0.0
    prev_close: float = 0.0


@dataclass
class SentimentData:
    """Container for sentiment analysis results"""

    score: float = 0.0  # -1 (very bearish) to +1 (very bullish)
    headline_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    news_available: bool = False
    top_headlines: List[str] = field(default_factory=list)


@dataclass
class SymbolAnalysis:
    """Complete analysis for a single symbol"""

    symbol: str
    exchange: str = "NSE"

    # Computed scores
    liquidity_score: float = 0.0
    volatility_score: float = 0.0
    momentum_score: float = 0.0
    sentiment_bias: float = 0.0
    composite_score: float = 0.0

    # Indicators
    indicators: IndicatorData = field(default_factory=IndicatorData)
    sentiment: SentimentData = field(default_factory=SentimentData)

    # Trading plan
    entry_zones: List[List[float]] = field(default_factory=list)  # [[low, high], ...]
    stop_loss: float = 0.0
    targets: List[float] = field(default_factory=list)
    suggested_qty: int = 0

    # Meta
    timestamp: str = ""
    data_quality: str = "good"  # good, partial, failed
    errors: List[str] = field(default_factory=list)


@dataclass
class TradingPlan:
    """Complete trading plan for the day"""

    date: str
    generated_at: str
    mode: str  # "full" or "test"
    symbols_analyzed: int
    symbols_selected: int
    total_capital: float
    risk_per_trade: float
    symbols: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Market Data Client Interface ====================


class MarketDataClient:
    """
    Abstract market data client interface.

    In production, this should be replaced with the actual broker's data API.
    Currently supports Groww broker via OpenAlgo's broker integration.

    TODO: Replace placeholder methods with actual API calls:
        - get_historical_data() -> Use broker.groww.api.data.BrokerData.get_history()
        - get_ltp() -> Use broker.groww.api.data.get_quotes()
    """

    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize market data client.

        Args:
            auth_token: Authentication token for broker API.
                       Falls back to GROWW_AUTH_TOKEN env variable.
        """
        self.auth_token = auth_token or os.getenv("GROWW_AUTH_TOKEN", "")
        self._broker_data = None

        # Try to initialize actual broker client
        try:
            from broker.groww.api.data import BrokerData

            if self.auth_token:
                self._broker_data = BrokerData(self.auth_token)
                logger.info("Initialized Groww BrokerData client")
            else:
                logger.warning("No auth token provided, using simulated data")
        except ImportError:
            logger.warning("Could not import Groww BrokerData, using simulated data")
        except Exception as e:
            logger.error(f"Error initializing BrokerData: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1d",
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "TCS")
            exchange: Exchange code ("NSE", "BSE")
            interval: Timeframe - "1m", "5m", "15m", "1h", "1d"
            days_back: Number of days of historical data

        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: datetime

        TODO: Replace with actual API call:
            return await self._fetch_from_groww(symbol, exchange, interval, days_back)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Try actual broker API first
        if self._broker_data:
            try:
                df = self._broker_data.get_history(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=self._map_interval(interval),
                    start_time=start_date.strftime("%Y-%m-%d"),
                    end_time=end_date.strftime("%Y-%m-%d"),
                )
                if df is not None and not df.empty:
                    logger.info(
                        f"Fetched {len(df)} candles for {symbol} from Groww API"
                    )
                    return self._normalize_dataframe(df)
            except Exception as e:
                logger.warning(
                    f"Groww API failed for {symbol}: {e}, using simulated data"
                )

        # Fallback: Generate simulated data for testing
        return self._generate_simulated_data(symbol, interval, days_back)

    async def get_ltp(self, symbol: str, exchange: str = "NSE") -> float:
        """
        Get last traded price for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange code

        Returns:
            Last traded price as float

        TODO: Replace with actual API call:
            return await self._fetch_ltp_from_groww(symbol, exchange)
        """
        # Try actual broker API
        if self._broker_data:
            try:
                # Use quotes endpoint for LTP
                # This is a placeholder - actual implementation depends on Groww's quote API
                df = await self.get_historical_data(symbol, exchange, "1m", 1)
                if not df.empty:
                    return float(df["close"].iloc[-1])
            except Exception as e:
                logger.warning(f"Could not fetch LTP for {symbol}: {e}")

        # Fallback: Generate simulated price
        return self._get_simulated_ltp(symbol)

    def _map_interval(self, interval: str) -> str:
        """Map standard interval to Groww API format"""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",  # May need to compute from 5m
            "30m": "30m",  # May need to compute from 5m
            "1h": "1h",
            "1d": "D",
            "1D": "D",
        }
        return mapping.get(interval, "D")

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns to standard format"""
        # Ensure lowercase column names
        df.columns = [c.lower() for c in df.columns]

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = 0.0

        return df[required]

    def _generate_simulated_data(
        self, symbol: str, interval: str, days_back: int
    ) -> pd.DataFrame:
        """
        Generate simulated OHLCV data for testing.
        Uses symbol hash for reproducible pseudo-random data.
        """
        np.random.seed(hash(symbol) % (2**32))

        # Determine number of bars based on interval
        bars_per_day = {
            "1m": 375,  # 9:15 to 15:30 = 375 minutes
            "5m": 75,
            "15m": 25,
            "30m": 13,
            "1h": 6,
            "1d": 1,
        }
        n_bars = bars_per_day.get(interval, 1) * days_back

        # Base price from symbol hash (₹100 - ₹5000 range)
        base_price = 100 + (hash(symbol) % 4900)

        # Generate price series with random walk
        returns = np.random.normal(0.0001, 0.02, n_bars)
        prices = base_price * np.cumprod(1 + returns)

        # Generate OHLCV
        data = []
        for i, close in enumerate(prices):
            volatility = abs(np.random.normal(0, 0.01))
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = low + (high - low) * np.random.random()
            volume = int(np.random.exponential(100000))

            data.append(
                {
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": volume,
                }
            )

        # Create DataFrame with datetime index
        end_time = datetime.now()
        if interval == "1d":
            dates = pd.date_range(end=end_time, periods=n_bars, freq="B")
        elif interval in ["1m", "5m", "15m", "30m"]:
            # Use trading hours only (use 'min' instead of deprecated 'T')
            dates = pd.date_range(
                end=end_time, periods=n_bars, freq=interval.replace("m", "min")
            )
        else:
            dates = pd.date_range(end=end_time, periods=n_bars, freq="h")

        df = pd.DataFrame(data, index=dates[: len(data)])
        logger.debug(f"Generated {len(df)} simulated bars for {symbol}")
        return df

    def _get_simulated_ltp(self, symbol: str) -> float:
        """Generate simulated LTP based on symbol"""
        np.random.seed(hash(symbol) % (2**32))
        base = 100 + (hash(symbol) % 4900)
        return round(base * (1 + np.random.normal(0, 0.02)), 2)


# ==================== News/Sentiment Client ====================


class NewsClient:
    """
    News and sentiment data client.

    Fetches news headlines from various sources and computes sentiment scores
    using rule-based analysis.

    TODO: Integrate with actual news APIs:
        - NewsAPI (newsapi.org)
        - Alpha Vantage News
        - Google News RSS
        - Economic Times API
        - MoneyControl API
    """

    # Sentiment lexicon for rule-based analysis
    POSITIVE_WORDS = {
        "bullish",
        "surge",
        "gain",
        "profit",
        "growth",
        "rally",
        "soar",
        "jump",
        "rise",
        "boost",
        "strong",
        "upgrade",
        "outperform",
        "beat",
        "exceed",
        "positive",
        "optimistic",
        "recovery",
        "breakout",
        "high",
        "record",
        "momentum",
        "buy",
        "accumulate",
        "up",
        "green",
        "bull",
    }

    NEGATIVE_WORDS = {
        "bearish",
        "fall",
        "loss",
        "decline",
        "crash",
        "plunge",
        "drop",
        "sink",
        "tumble",
        "weak",
        "downgrade",
        "underperform",
        "miss",
        "disappoint",
        "negative",
        "pessimistic",
        "recession",
        "breakdown",
        "low",
        "sell",
        "avoid",
        "down",
        "red",
        "bear",
        "risk",
        "concern",
        "warning",
        "cut",
        "slash",
        "trouble",
        "crisis",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news client.

        Args:
            api_key: API key for news service. Falls back to NEWS_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "")
        self._has_api_access = bool(self.api_key)

        if not self._has_api_access:
            logger.warning(
                "NEWS_API_KEY not set. News fetching disabled, using price-only scoring."
            )

    async def fetch_headlines(
        self, symbol: str, hours_back: int = 48, max_headlines: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch news headlines for a symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            hours_back: Hours of news to fetch (default 48)
            max_headlines: Maximum headlines to return

        Returns:
            List of dicts with keys: headline, source, published_at, url

        TODO: Replace with actual API implementation:
            if self._has_api_access:
                return await self._fetch_from_newsapi(symbol, hours_back)
        """
        if not self._has_api_access:
            return []

        # Placeholder for actual NewsAPI integration
        # ========================================
        # IMPLEMENTATION REQUIRED:
        #
        # import httpx
        #
        # async with httpx.AsyncClient() as client:
        #     response = await client.get(
        #         "https://newsapi.org/v2/everything",
        #         params={
        #             "q": symbol,
        #             "from": (datetime.now() - timedelta(hours=hours_back)).isoformat(),
        #             "sortBy": "relevancy",
        #             "language": "en",
        #             "pageSize": max_headlines,
        #             "apiKey": self.api_key
        #         }
        #     )
        #     data = response.json()
        #     return [
        #         {
        #             "headline": article["title"],
        #             "source": article["source"]["name"],
        #             "published_at": article["publishedAt"],
        #             "url": article["url"]
        #         }
        #         for article in data.get("articles", [])
        #     ]
        # ========================================

        logger.debug(f"News API call for {symbol} - implement actual API integration")
        return []

    def compute_sentiment(self, headlines: List[str]) -> SentimentData:
        """
        Compute sentiment score from headlines using rule-based analysis.

        Args:
            headlines: List of headline strings

        Returns:
            SentimentData with score from -1 (bearish) to +1 (bullish)
        """
        if not headlines:
            return SentimentData(news_available=False)

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for headline in headlines:
            words = set(headline.lower().split())

            pos_matches = len(words & self.POSITIVE_WORDS)
            neg_matches = len(words & self.NEGATIVE_WORDS)

            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1
            else:
                neutral_count += 1

        total = len(headlines)

        # Calculate weighted sentiment score
        if positive_count + negative_count > 0:
            # Score from -1 to +1
            raw_score = (positive_count - negative_count) / (
                positive_count + negative_count
            )
            # Weight by coverage (more headlines = more confidence)
            confidence = min(1.0, total / 10)  # Full confidence at 10+ headlines
            score = raw_score * confidence
        else:
            score = 0.0

        return SentimentData(
            score=round(score, 4),
            headline_count=total,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            news_available=True,
            top_headlines=headlines[:5],  # Keep top 5 headlines
        )


# ==================== Technical Indicator Calculator ====================


class IndicatorCalculator:
    """
    Technical indicator calculator for trading analysis.

    Computes various indicators from OHLCV data:
        - ATR (Average True Range)
        - RSI (Relative Strength Index)
        - VWAP (Volume Weighted Average Price)
        - Momentum (multi-timeframe)
        - Volume analysis
    """

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.

        ATR measures market volatility by decomposing the entire range
        of an asset price for a given period.

        Args:
            df: DataFrame with high, low, close columns
            period: Lookback period (default 14)

        Returns:
            Series of ATR values
        """
        if df.empty or len(df) < period:
            return pd.Series([0.0])

        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()

        return atr

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions.

        Args:
            df: DataFrame with close column
            period: Lookback period (default 14)

        Returns:
            Series of RSI values (0-100)
        """
        if df.empty or len(df) < period:
            return pd.Series([50.0])

        delta = df["close"].diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price.

        VWAP is the ratio of value traded to total volume traded.
        Used as a trading benchmark.

        Args:
            df: DataFrame with high, low, close, volume columns

        Returns:
            Series of VWAP values
        """
        if df.empty:
            return pd.Series([0.0])

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()

        vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.inf)

        return vwap.fillna(df["close"])

    @staticmethod
    def calculate_momentum(df: pd.DataFrame, periods: int) -> float:
        """
        Calculate price momentum over N periods.

        Momentum = (Current Price - Price N periods ago) / Price N periods ago

        Args:
            df: DataFrame with close column
            periods: Number of periods for momentum calculation

        Returns:
            Momentum as percentage change
        """
        if df.empty or len(df) < periods + 1:
            return 0.0

        current = df["close"].iloc[-1]
        past = df["close"].iloc[-periods - 1]

        if past == 0:
            return 0.0

        return round((current - past) / past, 4)

    @staticmethod
    def calculate_average_volume(df: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate average volume over N periods.

        Args:
            df: DataFrame with volume column
            period: Lookback period (default 20)

        Returns:
            Average volume
        """
        if df.empty or len(df) < period:
            return df["volume"].mean() if not df.empty else 0.0

        return df["volume"].tail(period).mean()


# ==================== Position Sizing Calculator ====================


class PositionSizer:
    """
    Position sizing calculator using risk management principles.

    Calculates position size based on:
        - Account balance
        - Risk per trade
        - Stop-loss distance (ATR-based)
        - Trading costs (using charges module)
    """

    def __init__(
        self,
        capital: float,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
        atr_sl_multiplier: float = DEFAULT_ATR_SL_MULTIPLIER,
        subscription_monthly: float = API_SUBSCRIPTION_MONTHLY,
    ):
        """
        Initialize position sizer.

        Args:
            capital: Total trading capital
            risk_per_trade: Max risk per trade as decimal (default 0.02 = 2%)
            max_position_size: Max position as fraction of capital (default 0.10)
            atr_sl_multiplier: ATR multiplier for stop-loss (default 2.0)
            subscription_monthly: Monthly API subscription cost
        """
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.atr_sl_multiplier = atr_sl_multiplier
        self.subscription_monthly = subscription_monthly

    def calculate_position_size(
        self, entry_price: float, atr: float, stop_loss: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size and trading parameters.

        Args:
            entry_price: Expected entry price
            atr: Average True Range (14-period)
            stop_loss: Optional custom stop-loss price

        Returns:
            Dict with qty, stop_loss, targets, costs
        """
        if entry_price <= 0 or atr <= 0:
            return {
                "suggested_qty": 0,
                "stop_loss": 0,
                "targets": [],
                "risk_amount": 0,
                "position_value": 0,
                "costs": {},
            }

        # Calculate stop-loss if not provided
        if stop_loss is None:
            stop_loss = entry_price - (atr * self.atr_sl_multiplier)

        # Risk per share
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            risk_per_share = atr  # Fallback

        # Max risk amount
        max_risk_amount = self.capital * self.risk_per_trade

        # Calculate quantity based on risk
        qty_by_risk = int(max_risk_amount / risk_per_share)

        # Calculate quantity based on max position size
        max_position_value = self.capital * self.max_position_size
        qty_by_position = int(max_position_value / entry_price)

        # Take minimum of both constraints
        suggested_qty = max(1, min(qty_by_risk, qty_by_position))

        # Calculate position value and costs
        position_value = suggested_qty * entry_price

        # Calculate round-trip costs using charges module
        costs = round_trip_cost(
            turnover=position_value,
            transaction_type="equity_intraday",
            subscription_monthly=self.subscription_monthly / TRADING_DAYS_PER_MONTH,
        )

        # Calculate targets (1.5R, 2R)
        targets = [
            round(entry_price + (risk_per_share * 1.5), 2),  # 1.5:1 R:R
            round(entry_price + (risk_per_share * 2.0), 2),  # 2:1 R:R
        ]

        return {
            "suggested_qty": suggested_qty,
            "stop_loss": round(stop_loss, 2),
            "targets": targets,
            "risk_amount": round(risk_per_share * suggested_qty, 2),
            "position_value": round(position_value, 2),
            "costs": costs,
        }


# ==================== Morning Learning Engine ====================


class MorningLearning:
    """
    Main engine for morning pre-market analysis.

    Orchestrates:
        1. Loading watchlist symbols
        2. Fetching historical data for each symbol
        3. Computing technical indicators
        4. Fetching and analyzing news sentiment
        5. Computing composite scores
        6. Ranking and selecting top-K symbols
        7. Generating trading plan with entry/exit levels
    """

    def __init__(
        self,
        market_client: Optional[MarketDataClient] = None,
        news_client: Optional[NewsClient] = None,
        watchlist_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        capital: Optional[float] = None,
        top_k: int = DEFAULT_TOP_K,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    ):
        """
        Initialize MorningLearning engine.

        Args:
            market_client: Market data client (default: auto-create)
            news_client: News client (default: auto-create)
            watchlist_path: Path to watchlist JSON
            output_path: Path for output today_plan.json
            capital: Trading capital (default: from env ACCOUNT_BALANCE)
            top_k: Number of top symbols to select
            risk_per_trade: Risk per trade as decimal
        """
        self.market_client = market_client or MarketDataClient()
        self.news_client = news_client or NewsClient()

        self.watchlist_path = watchlist_path or WATCHLIST_PATH
        self.output_path = output_path or OUTPUT_PATH

        # Capital from env or default
        self.capital = capital or float(os.getenv("ACCOUNT_BALANCE", "1000000"))
        self.top_k = top_k
        self.risk_per_trade = risk_per_trade

        # Initialize calculators
        self.indicator_calc = IndicatorCalculator()
        self.position_sizer = PositionSizer(
            capital=self.capital, risk_per_trade=risk_per_trade
        )

        logger.info(
            f"MorningLearning initialized with capital=₹{self.capital:,.2f}, top_k={top_k}"
        )

    def load_watchlist(self) -> List[Dict[str, str]]:
        """
        Load symbols from watchlist configuration file.

        Returns:
            List of dicts with symbol and exchange keys
        """
        if not self.watchlist_path.exists():
            logger.warning(
                f"Watchlist not found at {self.watchlist_path}, using defaults"
            )
            return [{"symbol": s, "exchange": "NSE"} for s in TEST_SYMBOLS]

        try:
            with open(self.watchlist_path, "r") as f:
                data = json.load(f)

            # Handle different formats
            if isinstance(data, list):
                # List of symbols or dicts
                if data and isinstance(data[0], str):
                    return [{"symbol": s, "exchange": "NSE"} for s in data]
                return data
            elif isinstance(data, dict):
                # Dict with 'symbols' key
                symbols = data.get("symbols", [])
                if symbols and isinstance(symbols[0], str):
                    return [{"symbol": s, "exchange": "NSE"} for s in symbols]
                return symbols

            return []

        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            return []

    async def analyze_symbol(
        self, symbol: str, exchange: str = "NSE"
    ) -> SymbolAnalysis:
        """
        Perform complete analysis for a single symbol.

        Args:
            symbol: Stock symbol
            exchange: Exchange code

        Returns:
            SymbolAnalysis with all computed data
        """
        analysis = SymbolAnalysis(
            symbol=symbol, exchange=exchange, timestamp=datetime.now().isoformat()
        )

        try:
            # Fetch historical data for different timeframes
            logger.info(f"Analyzing {symbol}...")

            # Daily data for ATR, RSI, trend
            df_daily = await self.market_client.get_historical_data(
                symbol, exchange, "1d", days_back=30
            )

            # 5-minute data for intraday analysis
            df_5m = await self.market_client.get_historical_data(
                symbol, exchange, "5m", days_back=5
            )

            # 1-minute data for precise entry
            df_1m = await self.market_client.get_historical_data(
                symbol, exchange, "1m", days_back=2
            )

            if df_daily.empty:
                analysis.data_quality = "failed"
                analysis.errors.append("No daily data available")
                return analysis

            # Compute indicators
            analysis.indicators = self._compute_indicators(df_daily, df_5m, df_1m)

            # Fetch news and compute sentiment
            try:
                headlines_data = await self.news_client.fetch_headlines(
                    symbol, hours_back=DEFAULT_NEWS_HOURS
                )
                headlines = [
                    h.get("headline", "") for h in headlines_data if h.get("headline")
                ]
                analysis.sentiment = self.news_client.compute_sentiment(headlines)
            except Exception as e:
                logger.warning(f"News fetch failed for {symbol}: {e}")
                analysis.sentiment = SentimentData(news_available=False)

            # Compute component scores
            analysis.liquidity_score = self._compute_liquidity_score(
                analysis.indicators
            )
            analysis.volatility_score = self._compute_volatility_score(
                analysis.indicators
            )
            analysis.momentum_score = self._compute_momentum_score(analysis.indicators)
            analysis.sentiment_bias = (
                analysis.sentiment.score if analysis.sentiment.news_available else 0.0
            )

            # Composite score
            analysis.composite_score = self._compute_composite_score(
                analysis.liquidity_score,
                analysis.volatility_score,
                analysis.momentum_score,
                analysis.sentiment_bias,
            )

            # Generate trading plan
            self._generate_trading_plan(analysis)

            analysis.data_quality = "good"
            logger.info(f"{symbol}: score={analysis.composite_score:.4f}")

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            traceback.print_exc()
            analysis.data_quality = "failed"
            analysis.errors.append(str(e))

        return analysis

    def _compute_indicators(
        self, df_daily: pd.DataFrame, df_5m: pd.DataFrame, df_1m: pd.DataFrame
    ) -> IndicatorData:
        """Compute all technical indicators from data"""
        indicators = IndicatorData()

        # From daily data
        if not df_daily.empty:
            atr_series = self.indicator_calc.calculate_atr(df_daily, DEFAULT_ATR_PERIOD)
            rsi_series = self.indicator_calc.calculate_rsi(df_daily, DEFAULT_RSI_PERIOD)

            indicators.atr_14 = float(atr_series.iloc[-1])
            indicators.rsi_14 = float(rsi_series.iloc[-1])
            indicators.avg_volume_20 = self.indicator_calc.calculate_average_volume(
                df_daily, DEFAULT_VOLUME_PERIOD
            )
            indicators.prev_close = float(df_daily["close"].iloc[-1])

        # From 5-minute data
        if not df_5m.empty:
            vwap_series = self.indicator_calc.calculate_vwap(df_5m)
            indicators.vwap = float(vwap_series.iloc[-1])
            indicators.current_price = float(df_5m["close"].iloc[-1])

            # VWAP gap
            if indicators.vwap > 0:
                indicators.vwap_gap_pct = round(
                    (indicators.current_price - indicators.vwap) / indicators.vwap, 4
                )

            # Multi-timeframe momentum (using 5m bars)
            indicators.momentum_5m = self.indicator_calc.calculate_momentum(
                df_5m, 1
            )  # 5m
            indicators.momentum_15m = self.indicator_calc.calculate_momentum(
                df_5m, 3
            )  # 15m
            indicators.momentum_30m = self.indicator_calc.calculate_momentum(
                df_5m, 6
            )  # 30m

            # Today's high/low
            today = df_5m.index[-1].date() if hasattr(df_5m.index[-1], "date") else None
            if today:
                today_data = (
                    df_5m[df_5m.index.date == today]
                    if hasattr(df_5m.index, "date")
                    else df_5m.tail(75)
                )
            else:
                today_data = df_5m.tail(75)  # Approximate today's data

            if not today_data.empty:
                indicators.high_of_day = float(today_data["high"].max())
                indicators.low_of_day = float(today_data["low"].min())

        # From 1-minute data (for precise current price)
        if not df_1m.empty:
            indicators.current_price = float(df_1m["close"].iloc[-1])

        return indicators

    def _compute_liquidity_score(self, indicators: IndicatorData) -> float:
        """
        Compute liquidity score based on average volume.

        Higher volume = higher liquidity = better for trading.
        Score normalized to 0-1 range.
        """
        # Normalize volume (log scale, assuming 1M volume = 1.0)
        if indicators.avg_volume_20 <= 0:
            return 0.0

        score = math.log10(indicators.avg_volume_20 + 1) / 6  # log10(1M) ≈ 6
        return min(1.0, max(0.0, score))

    def _compute_volatility_score(self, indicators: IndicatorData) -> float:
        """
        Compute volatility score based on ATR.

        Moderate volatility is ideal:
        - Too low = not enough movement
        - Too high = too risky

        Score peaks around 2-3% ATR/price ratio.
        """
        if indicators.current_price <= 0 or indicators.atr_14 <= 0:
            return 0.0

        atr_pct = indicators.atr_14 / indicators.current_price

        # Bell curve peaking at ~2.5% ATR
        optimal_atr = 0.025
        score = math.exp(-((atr_pct - optimal_atr) ** 2) / (2 * 0.015**2))

        return min(1.0, max(0.0, score))

    def _compute_momentum_score(self, indicators: IndicatorData) -> float:
        """
        Compute momentum score combining multiple timeframes.

        Positive momentum preferred, weighted by timeframe.
        """
        # Weighted average of momentum across timeframes
        weights = {"5m": 0.2, "15m": 0.3, "30m": 0.5}

        weighted_momentum = (
            indicators.momentum_5m * weights["5m"]
            + indicators.momentum_15m * weights["15m"]
            + indicators.momentum_30m * weights["30m"]
        )

        # Normalize to 0-1 (assuming -5% to +5% range)
        normalized = (weighted_momentum + 0.05) / 0.10

        # Apply sigmoid for smooth 0-1 range
        score = 1 / (1 + math.exp(-10 * (normalized - 0.5)))

        return min(1.0, max(0.0, score))

    def _compute_composite_score(
        self, liquidity: float, volatility: float, momentum: float, sentiment: float
    ) -> float:
        """
        Compute composite score combining all factors.

        Formula: liquidity × volatility × momentum × (1 + sentiment_bias)

        - Sentiment bias ranges from -1 to +1, so multiplier is 0 to 2
        - All components are 0-1, so final score is 0-2
        """
        # Clamp sentiment to reasonable range
        sentiment_multiplier = 1 + max(-0.5, min(0.5, sentiment))

        composite = liquidity * volatility * momentum * sentiment_multiplier

        return round(composite, 6)

    def _generate_trading_plan(self, analysis: SymbolAnalysis) -> None:
        """
        Generate entry zones, stop-loss, targets, and position size.

        Updates analysis in-place with trading plan data.
        """
        indicators = analysis.indicators

        if indicators.current_price <= 0 or indicators.atr_14 <= 0:
            return

        price = indicators.current_price
        atr = indicators.atr_14

        # Entry zones based on price action
        # Zone 1: Breakout above day high
        # Zone 2: Pullback to VWAP
        entry_zones = []

        # Breakout entry zone (above day high)
        if indicators.high_of_day > 0:
            entry_zones.append(
                [
                    round(indicators.high_of_day, 2),
                    round(indicators.high_of_day + atr * 0.3, 2),
                ]
            )

        # VWAP pullback zone
        if indicators.vwap > 0:
            entry_zones.append(
                [
                    round(indicators.vwap - atr * 0.2, 2),
                    round(indicators.vwap + atr * 0.2, 2),
                ]
            )

        analysis.entry_zones = entry_zones

        # Calculate position size and levels
        position_data = self.position_sizer.calculate_position_size(
            entry_price=price, atr=atr
        )

        analysis.stop_loss = position_data["stop_loss"]
        analysis.targets = position_data["targets"]
        analysis.suggested_qty = position_data["suggested_qty"]

    async def run_analysis(
        self, symbols: Optional[List[Dict[str, str]]] = None, test_mode: bool = False
    ) -> TradingPlan:
        """
        Run complete morning analysis.

        Args:
            symbols: Optional list of symbols to analyze (overrides watchlist)
            test_mode: Use test symbols if True

        Returns:
            TradingPlan with selected symbols and trading parameters
        """
        start_time = datetime.now()
        logger.info(
            f"Starting morning analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Load symbols
        if test_mode:
            symbols = [{"symbol": s, "exchange": "NSE"} for s in TEST_SYMBOLS]
            logger.info(f"Test mode: analyzing {TEST_SYMBOLS}")
        elif symbols is None:
            symbols = self.load_watchlist()
            logger.info(f"Loaded {len(symbols)} symbols from watchlist")

        if not symbols:
            logger.error("No symbols to analyze")
            return TradingPlan(
                date=start_time.strftime("%Y-%m-%d"),
                generated_at=start_time.isoformat(),
                mode="test" if test_mode else "full",
                symbols_analyzed=0,
                symbols_selected=0,
                total_capital=self.capital,
                risk_per_trade=self.risk_per_trade,
            )

        # Analyze all symbols
        analyses: List[SymbolAnalysis] = []

        for item in symbols:
            symbol = item.get("symbol", item) if isinstance(item, dict) else item
            exchange = item.get("exchange", "NSE") if isinstance(item, dict) else "NSE"

            analysis = await self.analyze_symbol(symbol, exchange)
            analyses.append(analysis)

        # Filter successful analyses and sort by score
        valid_analyses = [a for a in analyses if a.data_quality != "failed"]
        valid_analyses.sort(key=lambda x: x.composite_score, reverse=True)

        # Select top-K
        selected = valid_analyses[: self.top_k]

        # Create trading plan
        plan = TradingPlan(
            date=start_time.strftime("%Y-%m-%d"),
            generated_at=start_time.isoformat(),
            mode="test" if test_mode else "full",
            symbols_analyzed=len(analyses),
            symbols_selected=len(selected),
            total_capital=self.capital,
            risk_per_trade=self.risk_per_trade,
            symbols=[self._analysis_to_dict(a) for a in selected],
            metadata={
                "execution_time_seconds": (datetime.now() - start_time).total_seconds(),
                "news_api_available": self.news_client._has_api_access,
                "data_source": (
                    "groww" if self.market_client._broker_data else "simulated"
                ),
            },
        )

        # Save to file
        self._save_plan(plan)

        logger.info(
            f"Analysis complete: {len(selected)}/{len(analyses)} symbols selected "
            f"in {plan.metadata['execution_time_seconds']:.2f}s"
        )

        return plan

    def _analysis_to_dict(self, analysis: SymbolAnalysis) -> Dict[str, Any]:
        """Convert SymbolAnalysis to output dict format"""
        return {
            "symbol": analysis.symbol,
            "exchange": analysis.exchange,
            "score": analysis.composite_score,
            "entry_zones": analysis.entry_zones,
            "stop_loss": analysis.stop_loss,
            "targets": analysis.targets,
            "suggested_qty": analysis.suggested_qty,
            "indicators": {
                "atr_14": analysis.indicators.atr_14,
                "rsi_14": analysis.indicators.rsi_14,
                "vwap": analysis.indicators.vwap,
                "vwap_gap_pct": analysis.indicators.vwap_gap_pct,
                "momentum_5m": analysis.indicators.momentum_5m,
                "momentum_15m": analysis.indicators.momentum_15m,
                "momentum_30m": analysis.indicators.momentum_30m,
                "avg_volume_20": analysis.indicators.avg_volume_20,
                "current_price": analysis.indicators.current_price,
            },
            "sentiment": {
                "score": analysis.sentiment.score,
                "headline_count": analysis.sentiment.headline_count,
                "news_available": analysis.sentiment.news_available,
            },
            "scores": {
                "liquidity": analysis.liquidity_score,
                "volatility": analysis.volatility_score,
                "momentum": analysis.momentum_score,
                "sentiment_bias": analysis.sentiment_bias,
            },
            "data_quality": analysis.data_quality,
            "timestamp": analysis.timestamp,
        }

    def _save_plan(self, plan: TradingPlan) -> None:
        """Save trading plan to JSON file"""
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "date": plan.date,
            "generated_at": plan.generated_at,
            "mode": plan.mode,
            "symbols_analyzed": plan.symbols_analyzed,
            "symbols_selected": plan.symbols_selected,
            "total_capital": plan.total_capital,
            "risk_per_trade": plan.risk_per_trade,
            "symbols": plan.symbols,
            "metadata": plan.metadata,
        }

        with open(self.output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Trading plan saved to {self.output_path}")


# ==================== Main Entry Point ====================


async def run(
    test_mode: bool = False,
    symbols: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    capital: Optional[float] = None,
) -> TradingPlan:
    """
    Main async entry point for morning learning job.

    This function should be called at 08:00 IST daily by the scheduler.

    Args:
        test_mode: Run with test symbols ['TCS', 'RELIANCE', 'INFY']
        symbols: Optional custom symbol list (overrides watchlist and test_mode)
        top_k: Number of top symbols to select
        capital: Trading capital (default from env)

    Returns:
        TradingPlan object with analysis results

    Example:
        import asyncio
        from jobs.morning_learning import run

        # Test mode
        plan = asyncio.run(run(test_mode=True))
        print(f"Selected {len(plan.symbols)} symbols")

        # Production mode
        plan = asyncio.run(run())
    """
    logger.info("=" * 60)
    logger.info("MORNING LEARNING JOB STARTING")
    logger.info("=" * 60)

    # Initialize engine
    engine = MorningLearning(top_k=top_k, capital=capital)

    # Convert symbols if provided
    symbol_list = None
    if symbols:
        symbol_list = [{"symbol": s, "exchange": "NSE"} for s in symbols]

    # Run analysis
    plan = await engine.run_analysis(symbols=symbol_list, test_mode=test_mode)

    logger.info("=" * 60)
    logger.info("MORNING LEARNING JOB COMPLETE")
    logger.info("=" * 60)

    return plan


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Morning Learning Job - Pre-market symbol analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test mode with sample symbols
    python -m jobs.morning_learning --test

    # Full watchlist analysis
    python -m jobs.morning_learning

    # Custom symbols
    python -m jobs.morning_learning --symbols TCS,INFY,RELIANCE,HDFCBANK

    # Select top 5 symbols
    python -m jobs.morning_learning --top-k 5

    # Custom capital
    python -m jobs.morning_learning --capital 500000
        """,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with sample symbols [TCS, RELIANCE, INFY]",
    )

    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to analyze (e.g., TCS,INFY,RELIANCE)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top symbols to select (default: {DEFAULT_TOP_K})",
    )

    parser.add_argument(
        "--capital",
        type=float,
        help="Trading capital in INR (default: from env ACCOUNT_BALANCE or 1000000)",
    )

    parser.add_argument(
        "--output", type=str, help=f"Output file path (default: {OUTPUT_PATH})"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Run async job
    try:
        plan = asyncio.run(
            run(
                test_mode=args.test,
                symbols=symbols,
                top_k=args.top_k,
                capital=args.capital,
            )
        )

        # Print summary
        print("\n" + "=" * 60)
        print("MORNING LEARNING RESULTS")
        print("=" * 60)
        print(f"Date: {plan.date}")
        print(f"Symbols Analyzed: {plan.symbols_analyzed}")
        print(f"Symbols Selected: {plan.symbols_selected}")
        print(f"Capital: ₹{plan.total_capital:,.2f}")
        print(f"Risk/Trade: {plan.risk_per_trade * 100:.1f}%")
        print("\nTop Symbols:")
        print("-" * 60)

        for i, sym in enumerate(plan.symbols, 1):
            print(
                f"{i}. {sym['symbol']:12} | Score: {sym['score']:.6f} | "
                f"Qty: {sym['suggested_qty']:4} | SL: ₹{sym['stop_loss']:.2f}"
            )

        print("-" * 60)
        print(f"Plan saved to: {OUTPUT_PATH}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nAborted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running morning learning: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
