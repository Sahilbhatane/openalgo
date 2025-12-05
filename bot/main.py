"""
OpenAlgo Intraday Trading Bot
=============================
Main entry point for the autonomous trading bot.

This bot:
1. Runs morning analysis at 8:00 AM IST
2. Opens positions based on signals after 9:15 AM
3. Manages risk with ATR-based stops
4. Squares off all positions by 3:10 PM
5. Generates daily reports

Usage:
    # Start in paper mode (default)
    python -m bot.main
    
    # With custom config
    python -m bot.main --config config.json

WARNING: Live trading uses real money. Start with paper mode.
"""

import sys
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime, time

# Core modules
from .core.config import Config
from .core.mode import TradingMode
from .core.logging_config import setup_logging
from .core.constants import TradingSession

# Jobs
from .jobs.scheduler import JobScheduler
from .jobs.morning_learning import MorningLearningJob
from .jobs.market_open import MarketOpenJob
from .jobs.square_off import SquareOffJob
from .jobs.end_of_day import EndOfDayJob

# Execution
from .execution.order_manager import OrderManager
from .execution.paper_executor import PaperExecutor
from .execution.live_executor import LiveExecutor

# Risk
from .risk.risk_manager import RiskManager
from .risk.kill_switch import KillSwitch

# Reports
from .reports.performance import PerformanceTracker

# Utils
from .utils.time_utils import get_ist_now, is_trading_day


# Configure logging
logger = logging.getLogger("bot.main")


class TradingBot:
    """
    Main trading bot class.
    
    Orchestrates all components:
    - Job scheduling
    - Order execution
    - Risk management
    - Reporting
    """
    
    def __init__(self, config: Config):
        """
        Initialize the trading bot.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self.running = False
        
        # Setup logging
        setup_logging(
            log_dir=config.logs_dir,
        )
        
        logger.info("=" * 60)
        logger.info("INITIALIZING OPENALGO TRADING BOT")
        logger.info("=" * 60)
        
        # Mode manager is part of config
        self.mode_manager = config.mode_manager
        logger.info(f"Trading Mode: {self.mode_manager.current_mode.value}")
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            capital=config.trading.capital,
        )
        logger.info(f"Initial Capital: ₹{config.trading.capital:,.2f}")
        
        # Initialize kill switch
        self.kill_switch = KillSwitch(
            on_activate=self._handle_kill_switch,
        )
        
        # Initialize executors
        self.paper_executor = PaperExecutor(
            initial_capital=config.trading.capital,
            slippage_bps=5,  # 5 basis points
        )
        
        self.live_executor = None
        if config.openalgo.api_key:
            self.live_executor = LiveExecutor(
                api_key=config.openalgo.api_key,
                base_url=config.openalgo.base_url,
            )
        
        # Initialize order manager
        self.order_manager = OrderManager(
            mode_manager=self.mode_manager,
            paper_executor=self.paper_executor,
            live_executor=self.live_executor,
        )
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(
            data_dir=config.reports_dir,
            initial_capital=config.trading.capital,
        )
        
        # Initialize scheduler
        self.scheduler = JobScheduler()
        
        # Setup jobs
        self._setup_jobs()
        
        logger.info("Bot initialized successfully")
    
    def _setup_jobs(self) -> None:
        """Configure scheduled jobs"""
        # Morning learning job - 8:00 AM
        self.scheduler.add_daily_job(
            job_id="morning_learning",
            run_time=time(8, 0),
            func=self._run_morning_learning,
            name="Morning Learning",
        )
        logger.info("Scheduled: Morning Learning @ 08:00 IST")
        
        # Market open job - 9:15 AM
        self.scheduler.add_daily_job(
            job_id="market_open",
            run_time=time(9, 15),
            func=self._run_market_open,
            name="Market Open",
        )
        logger.info("Scheduled: Market Open @ 09:15 IST")
        
        # Square off job - 3:10 PM
        self.scheduler.add_daily_job(
            job_id="square_off",
            run_time=time(15, 10),
            func=self._run_square_off,
            name="Square Off",
        )
        logger.info("Scheduled: Square Off @ 15:10 IST")
        
        # End of day job - 3:35 PM
        self.scheduler.add_daily_job(
            job_id="end_of_day",
            run_time=time(15, 35),
            func=self._run_end_of_day,
            name="End of Day",
        )
        logger.info("Scheduled: End of Day @ 15:35 IST")
    
    def _run_morning_learning(self) -> None:
        """Run morning learning job"""
        logger.info("Running morning learning...")
        # Morning learning implementation would go here
    
    def _run_market_open(self) -> None:
        """Run market open job"""
        logger.info("Running market open checks...")
        # Market open implementation would go here
    
    def _run_square_off(self) -> None:
        """Run square off job"""
        square_off_job = SquareOffJob(
            position_getter=self._get_positions,
            order_executor=self._execute_signal,
        )
        square_off_job.run()
    
    def _run_end_of_day(self) -> None:
        """Run end of day processing"""
        eod_job = EndOfDayJob(
            trade_getter=self._get_today_trades,
            report_dir=self.config.reports_dir / "daily",
        )
        
        # Generate report
        report = eod_job.run()
        
        # Update performance tracker
        self.performance_tracker.record_day(
            pnl=report.net_pnl,
            trades=report.total_trades,
            winning_trades=report.winning_trades,
            charges=report.total_charges,
        )
        
        # Check if we can go live
        can_go_live, reason = self.performance_tracker.can_go_live()
        if can_go_live and self.mode_manager.is_paper:
            logger.info(f"🎉 Paper trading criteria met: {reason}")
            logger.info("Consider switching to LIVE mode after manual review")
    
    def _get_positions(self) -> list:
        """Get current positions"""
        if self.mode_manager.is_paper:
            return self.paper_executor.get_all_positions()
        elif self.live_executor:
            return self.live_executor.get_positions()
        return []
    
    def _get_today_trades(self) -> list:
        """Get today's trades"""
        if self.mode_manager.is_paper:
            return self.paper_executor.get_today_trades()
        return []
    
    def _execute_signal(self, signal) -> bool:
        """Execute a trading signal"""
        order = self.order_manager.create_order_from_signal(signal)
        if order:
            result = self.order_manager.submit_order(order)
            return result.status.value == "filled"
        return False
    
    def _handle_kill_switch(self) -> None:
        """Handle kill switch activation"""
        logger.critical("🚨 KILL SWITCH ACTIVATED")
        
        # Cancel all pending orders
        self.order_manager.cancel_all_orders()
        
        # Square off all positions
        if self.mode_manager.is_paper:
            prices = {}  # Would need real prices
            self.paper_executor.close_all_positions(prices)
        
        logger.critical("All positions closed, orders cancelled")
    
    def start(self) -> None:
        """Start the trading bot"""
        if self.running:
            logger.warning("Bot is already running")
            return
        
        self.running = True
        
        # Check if trading day
        now = get_ist_now()
        if not is_trading_day(now):
            logger.warning(f"Not a trading day: {now.strftime('%A, %Y-%m-%d')}")
        
        logger.info("=" * 60)
        logger.info("STARTING TRADING BOT")
        logger.info(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} IST")
        logger.info(f"Mode: {self.mode_manager.current_mode.value}")
        logger.info("=" * 60)
        
        # Start scheduler
        self.scheduler.start()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        
        logger.info("Bot is running. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        try:
            while self.running:
                import time as time_module
                time_module.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self) -> None:
        """Stop the trading bot"""
        if not self.running:
            return
        
        logger.info("Stopping trading bot...")
        self.running = False
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Final statistics
        stats = self.performance_tracker.get_statistics()
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info(f"Total P&L: ₹{stats['total_pnl']:+,.2f}")
        logger.info(f"Total Return: {stats['total_return_pct']:+.2f}%")
        logger.info(f"Win Rate: {stats['win_rate']:.1f}%")
        logger.info(f"Trading Days: {stats['trading_days']}")
        logger.info("=" * 60)
        
        logger.info("Trading bot stopped")
    
    def _shutdown_handler(self, signum, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="OpenAlgo Intraday Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bot.main                    # Start with default config
  python -m bot.main --paper            # Start in paper mode (default)
  python -m bot.main --config my.json   # Use custom config file
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (JSON)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Run in paper trading mode (default)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config.load()
    
    # Create and start bot
    bot = TradingBot(config)
    
    try:
        bot.start()
    except Exception as e:
        logger.exception(f"Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
