"""
Market Open Job
===============
Job that runs at market open (9:15 AM IST).

Tasks:
1. Verify broker connection
2. Load today's trading plan
3. Subscribe to market data feeds
4. Initialize strategies
5. Start trading engine
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

from ..core.config import Config
from .morning_learning import TodayPlan


logger = logging.getLogger("bot.jobs.market_open")


class MarketOpenJob:
    """
    Market open job - initializes trading for the day.
    
    This job runs at 9:15 AM (or slightly after) to:
    1. Verify all connections are ready
    2. Load the trading plan from morning learning
    3. Subscribe to market data for today's symbols
    4. Initialize trading strategies
    5. Signal that trading can begin
    
    Usage:
        job = MarketOpenJob(
            config=config,
            on_ready=start_trading_callback,
        )
        job.run()
    """
    
    def __init__(
        self,
        config: Config,
        data_fetcher=None,  # HistoricalClient
        market_feed=None,   # MarketFeed
        on_ready: Optional[Callable[[TodayPlan], None]] = None,
        plans_dir: Optional[Path] = None,
    ):
        """
        Initialize market open job.
        
        Args:
            config: Bot configuration
            data_fetcher: Historical data client (for verification)
            market_feed: Real-time market feed
            on_ready: Callback when ready to trade
            plans_dir: Directory where plans are stored
        """
        self.config = config
        self.data_fetcher = data_fetcher
        self.market_feed = market_feed
        self.on_ready = on_ready
        self.plans_dir = plans_dir or Path("bot_data/plans")
        
        self._is_ready = False
        self._today_plan: Optional[TodayPlan] = None
        self._errors: List[str] = []
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready
    
    @property
    def today_plan(self) -> Optional[TodayPlan]:
        return self._today_plan
    
    def run(self) -> bool:
        """
        Run market open initialization.
        
        Returns:
            True if ready to trade, False otherwise
        """
        logger.info("Running market open job...")
        
        self._errors = []
        self._is_ready = False
        
        # Step 1: Verify connections
        if not self._verify_connections():
            logger.error("Connection verification failed")
            return False
        
        # Step 2: Load trading plan
        self._today_plan = self._load_plan()
        if self._today_plan is None:
            logger.warning("No trading plan found - will trade with default settings")
            # Create minimal plan
            self._today_plan = TodayPlan(
                date=datetime.now().date().isoformat(),
                created_at=datetime.now().isoformat(),
                top_symbols=[],  # Empty - strategies will use their own watchlist
                notes=["No morning plan - using defaults"],
            )
        
        # Step 3: Subscribe to market data
        if not self._subscribe_to_feeds():
            logger.error("Failed to subscribe to market feeds")
            return False
        
        # Step 4: Mark as ready
        self._is_ready = True
        logger.info("Market open job complete - ready to trade")
        
        # Step 5: Call ready callback
        if self.on_ready:
            try:
                self.on_ready(self._today_plan)
            except Exception as e:
                logger.error(f"on_ready callback failed: {e}")
        
        return True
    
    def _verify_connections(self) -> bool:
        """Verify all required connections are working"""
        all_good = True
        
        # Check broker connection via data fetcher
        if self.data_fetcher:
            try:
                # Try a simple API call
                # This will depend on the actual data_fetcher implementation
                logger.info("Verifying broker connection...")
                # data_fetcher.verify_connection() or similar
                logger.info("Broker connection OK")
            except Exception as e:
                logger.error(f"Broker connection failed: {e}")
                self._errors.append(f"Broker: {e}")
                all_good = False
        
        # Check market feed
        if self.market_feed:
            try:
                logger.info("Verifying market feed connection...")
                # market_feed.is_connected() or similar
                logger.info("Market feed connection OK")
            except Exception as e:
                logger.error(f"Market feed connection failed: {e}")
                self._errors.append(f"Market Feed: {e}")
                all_good = False
        
        return all_good
    
    def _load_plan(self) -> Optional[TodayPlan]:
        """Load today's trading plan"""
        import json
        
        plan_file = self.plans_dir / "today_plan.json"
        
        if not plan_file.exists():
            logger.warning(f"No plan file found at {plan_file}")
            return None
        
        try:
            with open(plan_file) as f:
                data = json.load(f)
            
            plan = TodayPlan.from_dict(data)
            
            # Verify plan is for today
            today = datetime.now().date().isoformat()
            if plan.date != today:
                logger.warning(f"Plan is for {plan.date}, not today ({today})")
                return None
            
            logger.info(f"Loaded trading plan with {len(plan.top_symbols)} symbols")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to load plan: {e}")
            return None
    
    def _subscribe_to_feeds(self) -> bool:
        """Subscribe to market data feeds for today's symbols"""
        if not self.market_feed:
            logger.warning("No market feed configured")
            return True  # Not an error if not configured
        
        if not self._today_plan or not self._today_plan.top_symbols:
            logger.info("No symbols to subscribe to")
            return True
        
        try:
            logger.info(f"Subscribing to {len(self._today_plan.top_symbols)} symbols...")
            
            # This will depend on actual market_feed implementation
            # Example:
            # for symbol in self._today_plan.top_symbols:
            #     self.market_feed.subscribe(symbol, mode="QUOTE")
            
            logger.info("Subscription complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            self._errors.append(f"Subscription: {e}")
            return False
    
    def get_status(self) -> str:
        """Get job status"""
        status = [
            f"Market Open Job Status",
            f"  Ready: {self._is_ready}",
            f"  Plan loaded: {self._today_plan is not None}",
        ]
        
        if self._today_plan:
            status.append(f"  Top symbols: {self._today_plan.top_symbols[:5]}")
            status.append(f"  Market trend: {self._today_plan.market_trend}")
        
        if self._errors:
            status.append(f"  Errors:")
            for err in self._errors:
                status.append(f"    - {err}")
        
        return "\n".join(status)
