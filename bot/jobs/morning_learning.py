"""
Morning Learning Job
====================
Pre-market analysis job that runs at 8:00 AM IST.

Tasks:
1. Fetch previous day's data for watchlist
2. Calculate technical indicators
3. Score each symbol for trading potential
4. Create today's trading plan (today_plan.json)
"""

import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import pandas as pd

from ..core.constants import DEFAULT_WATCHLIST
from ..utils.indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_vwap,
    calculate_ema,
    calculate_volume_surge,
)


logger = logging.getLogger("bot.jobs.morning")


@dataclass
class SymbolScore:
    """
    Scoring for a symbol's trading potential today.
    
    Higher scores = more suitable for trading today.
    """
    symbol: str
    score: float = 0.0  # 0-100 score
    
    # Price action
    prev_close: float = 0.0
    prev_high: float = 0.0
    prev_low: float = 0.0
    prev_volume: float = 0.0
    
    # Indicators
    atr: float = 0.0
    atr_pct: float = 0.0  # ATR as % of price
    rsi: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    
    # Scoring components
    volatility_score: float = 0.0
    trend_score: float = 0.0
    volume_score: float = 0.0
    momentum_score: float = 0.0
    
    # Notes
    trend: str = ""  # "BULLISH", "BEARISH", "NEUTRAL"
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TodayPlan:
    """
    Trading plan for today.
    
    Created by morning learning job, consumed by trading strategies.
    """
    date: str = ""
    created_at: str = ""
    
    # Symbols to trade (sorted by score)
    top_symbol: str = ""  # Best single candidate (top of top_symbols)
    top_symbols: List[str] = field(default_factory=list)
    symbol_scores: List[SymbolScore] = field(default_factory=list)
    
    # Market analysis
    market_trend: str = "NEUTRAL"  # Based on index analysis
    nifty_trend: str = "NEUTRAL"
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH
    
    # Recommendations
    max_positions: int = 3
    position_size_pct: float = 0.30  # Suggested position size
    
    # Notes
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['symbol_scores'] = [s.to_dict() if hasattr(s, 'to_dict') else s 
                                   for s in self.symbol_scores]
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "TodayPlan":
        scores = [SymbolScore(**s) if isinstance(s, dict) else s 
                  for s in data.get('symbol_scores', [])]
        data['symbol_scores'] = scores
        return cls(**data)


class MorningLearningJob:
    """
    Morning learning/analysis job.
    
    Run at 8:00 AM IST to:
    1. Analyze previous day's data
    2. Score symbols for trading potential
    3. Create trading plan for today
    
    Usage:
        job = MorningLearningJob(data_fetcher, data_dir)
        plan = job.run()
    """
    
    def __init__(
        self,
        data_fetcher,  # HistoricalClient or similar
        data_dir: Optional[Path] = None,
        watchlist: Optional[List[str]] = None,
        top_n: int = 10,
        universe_source: Optional[str] = None,
        universe_limit: Optional[int] = None,
    ):
        """
        Initialize morning learning job.
        
        Args:
            data_fetcher: Object with get_ohlc(symbol, ...) method
            data_dir: Directory for saving plans
            watchlist: Symbols to analyze
            top_n: Number of top symbols to include in plan
            universe_source: Where to build the symbol universe from.
                - "watchlist" (default): uses watchlist or DEFAULT_WATCHLIST
                - "instruments_db": uses local SymToken database table
                - "instruments_api": calls OpenAlgo /api/v1/instruments endpoint
            universe_limit: Maximum symbols to analyze (safety cap). Defaults to env MORNING_UNIVERSE_LIMIT or 250.
        """
        self.data_fetcher = data_fetcher
        self.data_dir = data_dir or Path("bot_data/plans")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # If watchlist is explicitly provided, always use it.
        self._explicit_watchlist = watchlist
        self.universe_source = (universe_source or os.getenv("MORNING_UNIVERSE_SOURCE", "watchlist")).strip().lower()
        self.universe_limit = universe_limit or int(os.getenv("MORNING_UNIVERSE_LIMIT", "250"))
        self.top_n = top_n
    
    def run(self) -> TodayPlan:
        """
        Run the morning learning analysis.
        
        Returns:
            TodayPlan for today's trading
        """
        logger.info("Starting morning learning job...")
        
        today = date.today()
        
        universe = self._get_universe()
        analyzed_symbols = universe[: max(self.universe_limit, 0)]

        # Score each symbol
        symbol_scores = []
        for symbol in analyzed_symbols:
            try:
                score = self._analyze_symbol(symbol)
                if score:
                    symbol_scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
        
        # Sort by score
        symbol_scores.sort(key=lambda s: s.score, reverse=True)
        
        # Get top symbols
        top_symbols = [s.symbol for s in symbol_scores[:self.top_n]]
        top_symbol = top_symbols[0] if top_symbols else ""
        
        # Analyze market trend
        market_trend = self._analyze_market_trend()
        volatility_regime = self._analyze_volatility_regime(symbol_scores)
        
        # Create plan
        plan = TodayPlan(
            date=today.isoformat(),
            created_at=datetime.now().isoformat(),
            top_symbol=top_symbol,
            top_symbols=top_symbols,
            symbol_scores=symbol_scores[:self.top_n],
            market_trend=market_trend,
            nifty_trend=market_trend,  # Same for now
            volatility_regime=volatility_regime,
            max_positions=3 if volatility_regime == "NORMAL" else 2,
            position_size_pct=0.30 if volatility_regime == "LOW" else 0.20,
            notes=[
                f"Universe source: {self._describe_universe_source()}",
                f"Analyzed {len(analyzed_symbols)} symbols",
                f"Top scorers: {', '.join(top_symbols[:5])}",
                f"Market trend: {market_trend}",
                f"Volatility: {volatility_regime}",
            ],
        )
        
        # Save plan
        self._save_plan(plan)
        
        logger.info(f"Morning learning complete. Top symbols: {top_symbols[:5]}")
        
        return plan

    def _describe_universe_source(self) -> str:
        if self._explicit_watchlist is not None:
            return "explicit_watchlist"
        return self.universe_source

    def _get_universe(self) -> List[str]:
        """Build the list of symbols to analyze."""
        if self._explicit_watchlist is not None:
            return self._normalize_symbols(self._explicit_watchlist)

        source = self.universe_source
        if source in {"watchlist", "default"}:
            return self._normalize_symbols(DEFAULT_WATCHLIST)
        if source in {"instruments_db", "db"}:
            symbols = self._fetch_universe_from_instruments_db()
            return symbols if symbols else self._normalize_symbols(DEFAULT_WATCHLIST)
        if source in {"instruments_api", "api"}:
            symbols = self._fetch_universe_from_instruments_api()
            return symbols if symbols else self._normalize_symbols(DEFAULT_WATCHLIST)

        logger.warning(f"Unknown MORNING_UNIVERSE_SOURCE='{source}', falling back to DEFAULT_WATCHLIST")
        return self._normalize_symbols(DEFAULT_WATCHLIST)

    def _normalize_symbols(self, symbols: List[str]) -> List[str]:
        """Upper-case, de-dupe, and filter obviously invalid symbols."""
        seen: set[str] = set()
        normalized: List[str] = []

        # Allow common NSE symbol chars (e.g. M&M, L&TFH, BAJAJ-AUTO)
        pattern = re.compile(r"^[A-Z0-9&._-]+$")

        for raw in symbols or []:
            if raw is None:
                continue
            sym = str(raw).strip().upper()
            if not sym:
                continue
            if not pattern.match(sym):
                continue
            if sym in seen:
                continue
            seen.add(sym)
            normalized.append(sym)

        return normalized

    def _fetch_universe_from_instruments_db(self) -> List[str]:
        """Fetch symbols from local SymToken database table."""
        try:
            from database.symbol import db_session, SymToken

            exchange = os.getenv("MORNING_UNIVERSE_EXCHANGE", "NSE").strip().upper()
            instrumenttype = os.getenv("MORNING_UNIVERSE_INSTRUMENTTYPE", "").strip().upper()
            limit = max(self.universe_limit, 0)

            query = db_session.query(SymToken.symbol)
            if exchange:
                query = query.filter(SymToken.exchange == exchange)
            if instrumenttype:
                query = query.filter(SymToken.instrumenttype == instrumenttype)

            # Deterministic ordering to make runs stable
            query = query.order_by(SymToken.symbol.asc()).distinct().limit(limit)
            rows = query.all()

            symbols = [row[0] for row in rows if row and row[0]]
            return self._normalize_symbols(symbols)
        except Exception as e:
            logger.warning(f"Failed to build universe from instruments DB: {e}")
            return []

    def _fetch_universe_from_instruments_api(self) -> List[str]:
        """Fetch symbols via OpenAlgo instruments REST API."""
        try:
            import requests

            base_url = os.getenv("OPENALGO_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
            api_key = (os.getenv("OPENALGO_API_KEY") or os.getenv("OPENALGO_APIKEY") or "").strip()
            if not api_key:
                logger.warning("OPENALGO_API_KEY not set; cannot call instruments API")
                return []

            exchange = os.getenv("MORNING_UNIVERSE_EXCHANGE", "NSE").strip().upper()
            limit = max(self.universe_limit, 0)

            url = f"{base_url}/api/v1/instruments"
            resp = requests.get(
                url,
                params={"apikey": api_key, "exchange": exchange, "format": "json"},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()

            data = payload.get("data", []) if isinstance(payload, dict) else []
            symbols = [row.get("symbol") for row in data if isinstance(row, dict)]
            normalized = self._normalize_symbols(symbols)
            return normalized[:limit]
        except Exception as e:
            logger.warning(f"Failed to build universe from instruments API: {e}")
            return []
    
    def _analyze_symbol(self, symbol: str) -> Optional[SymbolScore]:
        """
        Analyze a single symbol and return its score.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            SymbolScore or None if analysis fails
        """
        # Fetch last 30 days of daily data
        try:
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=60)
            
            df = self.data_fetcher.get_ohlc(
                symbol=symbol,
                interval="ONE_DAY",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
            )
            
            if df is None or len(df) < 20:
                return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {e}")
            return None
        
        # Calculate indicators
        atr = calculate_atr(df['high'], df['low'], df['close'], period=14)
        rsi = calculate_rsi(df['close'], period=14)
        ema_20 = calculate_ema(df['close'], period=20)
        ema_50 = calculate_ema(df['close'], period=50)
        vol_surge = calculate_volume_surge(df['volume'], period=20)
        
        # Get latest values
        last_close = df['close'].iloc[-1]
        last_high = df['high'].iloc[-1]
        last_low = df['low'].iloc[-1]
        last_volume = df['volume'].iloc[-1]
        last_atr = atr.iloc[-1] if not atr.empty else 0
        last_rsi = rsi.iloc[-1] if not rsi.empty else 50
        last_ema_20 = ema_20.iloc[-1] if not ema_20.empty else last_close
        last_ema_50 = ema_50.iloc[-1] if not ema_50.empty else last_close
        last_vol_surge = vol_surge.iloc[-1] if not vol_surge.empty else 1.0
        
        # Calculate scores
        atr_pct = (last_atr / last_close * 100) if last_close > 0 else 0
        
        # Volatility score (want 1-3% ATR)
        if 1.0 <= atr_pct <= 3.0:
            volatility_score = 25
        elif 0.5 <= atr_pct < 1.0 or 3.0 < atr_pct <= 5.0:
            volatility_score = 15
        else:
            volatility_score = 5
        
        # Trend score
        if last_close > last_ema_20 > last_ema_50:
            trend = "BULLISH"
            trend_score = 25
        elif last_close < last_ema_20 < last_ema_50:
            trend = "BEARISH"
            trend_score = 20  # Bearish is tradeable but slightly less preferred
        else:
            trend = "NEUTRAL"
            trend_score = 10
        
        # Volume score
        if last_vol_surge > 1.5:
            volume_score = 25
        elif last_vol_surge > 1.0:
            volume_score = 15
        else:
            volume_score = 5
        
        # Momentum score (RSI not extreme)
        if 40 <= last_rsi <= 60:
            momentum_score = 25  # Room to move
        elif 30 <= last_rsi < 40 or 60 < last_rsi <= 70:
            momentum_score = 20
        else:
            momentum_score = 10  # Overbought/oversold
        
        total_score = volatility_score + trend_score + volume_score + momentum_score
        
        notes = []
        if atr_pct > 3.0:
            notes.append("High volatility")
        if atr_pct < 1.0:
            notes.append("Low volatility")
        if last_rsi > 70:
            notes.append("Overbought")
        if last_rsi < 30:
            notes.append("Oversold")
        if last_vol_surge > 2.0:
            notes.append("Volume surge")
        
        return SymbolScore(
            symbol=symbol,
            score=total_score,
            prev_close=last_close,
            prev_high=last_high,
            prev_low=last_low,
            prev_volume=last_volume,
            atr=last_atr,
            atr_pct=atr_pct,
            rsi=last_rsi,
            ema_20=last_ema_20,
            ema_50=last_ema_50,
            volatility_score=volatility_score,
            trend_score=trend_score,
            volume_score=volume_score,
            momentum_score=momentum_score,
            trend=trend,
            notes=notes,
        )
    
    def _analyze_market_trend(self) -> str:
        """Analyze overall market trend using Nifty 50"""
        try:
            # Try to analyze NIFTY
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=30)
            
            df = self.data_fetcher.get_ohlc(
                symbol="NIFTY",
                interval="ONE_DAY",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
            )
            
            if df is None or len(df) < 10:
                return "NEUTRAL"
            
            ema_10 = calculate_ema(df['close'], period=10)
            ema_20 = calculate_ema(df['close'], period=20)
            
            last_close = df['close'].iloc[-1]
            last_ema_10 = ema_10.iloc[-1]
            last_ema_20 = ema_20.iloc[-1]
            
            if last_close > last_ema_10 > last_ema_20:
                return "BULLISH"
            elif last_close < last_ema_10 < last_ema_20:
                return "BEARISH"
            
        except Exception:
            pass
        
        return "NEUTRAL"
    
    def _analyze_volatility_regime(self, scores: List[SymbolScore]) -> str:
        """Determine volatility regime from symbol scores"""
        if not scores:
            return "NORMAL"
        
        avg_atr_pct = sum(s.atr_pct for s in scores) / len(scores)
        
        if avg_atr_pct > 3.0:
            return "HIGH"
        elif avg_atr_pct < 1.0:
            return "LOW"
        
        return "NORMAL"
    
    def _save_plan(self, plan: TodayPlan):
        """Save plan to file"""
        filename = f"today_plan_{plan.date}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
        
        # Also save as 'today_plan.json' for easy access
        latest = self.data_dir / "today_plan.json"
        with open(latest, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
        
        logger.info(f"Saved trading plan to {filepath}")
    
    def load_today_plan(self) -> Optional[TodayPlan]:
        """Load today's plan if it exists"""
        filepath = self.data_dir / "today_plan.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            plan = TodayPlan.from_dict(data)
            
            # Check if plan is for today
            if plan.date == date.today().isoformat():
                return plan
            
        except Exception as e:
            logger.warning(f"Failed to load plan: {e}")
        
        return None
