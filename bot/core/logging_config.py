"""
Logging Configuration
=====================
Centralized logging setup for the trading bot.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


# Default log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log levels for different components
COMPONENT_LOG_LEVELS = {
    "bot": logging.INFO,
    "bot.core": logging.INFO,
    "bot.risk": logging.INFO,
    "bot.execution": logging.INFO,
    "bot.strategies": logging.INFO,
    "bot.jobs": logging.INFO,
    "bot.data": logging.DEBUG,  # More verbose for data issues
}


class ColorFormatter(logging.Formatter):
    """Colored log formatter for console output"""
    
    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Set up logging for the trading bot.
    
    Args:
        log_dir: Directory for log files
        console_level: Logging level for console output
        file_level: Logging level for file output
        max_bytes: Max size of each log file
        backup_count: Number of backup files to keep
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "bot_data" / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger for bot
    root_logger = logging.getLogger("bot")
    root_logger.setLevel(logging.DEBUG)  # Capture all, handlers will filter
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    if sys.platform != "win32" or os.getenv("TERM"):
        console_handler.setFormatter(ColorFormatter(LOG_FORMAT, LOG_DATE_FORMAT))
    else:
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)
    
    # Main log file (rotating by size)
    today = datetime.now().strftime("%Y-%m-%d")
    main_log_file = log_dir / f"bot_{today}.log"
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)
    
    # Trade log (separate file for trades only)
    trade_log_file = log_dir / f"trades_{today}.log"
    trade_handler = RotatingFileHandler(
        trade_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    trade_handler.setLevel(logging.INFO)
    trade_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    trade_handler.addFilter(lambda record: "TRADE" in record.getMessage())
    
    trade_logger = logging.getLogger("bot.trades")
    trade_logger.addHandler(trade_handler)
    
    # Error log (separate file for errors only)
    error_log_file = log_dir / f"errors_{today}.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(error_handler)
    
    # Set levels for specific components
    for component, level in COMPONENT_LOG_LEVELS.items():
        logging.getLogger(component).setLevel(level)
    
    # Suppress noisy third-party loggers
    for noisy_logger in ["urllib3", "websockets", "asyncio"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    root_logger.info("Logging initialized: %s", log_dir)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a component.
    
    Args:
        name: Component name (e.g., "risk", "execution", "strategies")
    
    Returns:
        Logger instance with proper hierarchy
    """
    if not name.startswith("bot"):
        name = f"bot.{name}"
    return logging.getLogger(name)


class TradeLogger:
    """
    Specialized logger for trade events.
    
    Logs to both main log and separate trade log file.
    """
    
    def __init__(self):
        self.logger = get_logger("trades")
    
    def log_signal(self, symbol: str, signal: str, price: float, reason: str):
        """Log a trading signal"""
        self.logger.info(
            "SIGNAL | %s | %s @ %.2f | %s",
            symbol, signal, price, reason
        )
    
    def log_order_placed(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        order_id: str,
    ):
        """Log an order placement"""
        self.logger.info(
            "TRADE:ORDER_PLACED | %s | %s %d @ %.2f | order_id=%s",
            symbol, side, qty, price, order_id
        )
    
    def log_order_filled(
        self,
        symbol: str,
        side: str,
        qty: int,
        fill_price: float,
        order_id: str,
    ):
        """Log an order fill"""
        self.logger.info(
            "TRADE:ORDER_FILLED | %s | %s %d @ %.2f | order_id=%s",
            symbol, side, qty, fill_price, order_id
        )
    
    def log_order_rejected(
        self,
        symbol: str,
        side: str,
        qty: int,
        reason: str,
        order_id: str = "",
    ):
        """Log an order rejection"""
        self.logger.warning(
            "TRADE:ORDER_REJECTED | %s | %s %d | reason=%s | order_id=%s",
            symbol, side, qty, reason, order_id
        )
    
    def log_position_opened(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_loss: float,
        target: float,
    ):
        """Log a new position"""
        self.logger.info(
            "TRADE:POSITION_OPENED | %s | %s %d @ %.2f | SL=%.2f | TGT=%.2f",
            symbol, side, qty, entry_price, stop_loss, target
        )
    
    def log_position_closed(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str,
    ):
        """Log a closed position"""
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        self.logger.info(
            "TRADE:POSITION_CLOSED | %s | %s %d | entry=%.2f exit=%.2f | PnL=%s | %s",
            symbol, side, qty, entry_price, exit_price, pnl_str, reason
        )
    
    def log_stop_loss_hit(self, symbol: str, price: float, loss: float):
        """Log a stop loss trigger"""
        self.logger.warning(
            "TRADE:STOP_LOSS_HIT | %s @ %.2f | loss=%.2f",
            symbol, price, loss
        )
    
    def log_target_hit(self, symbol: str, price: float, profit: float):
        """Log a target price hit"""
        self.logger.info(
            "TRADE:TARGET_HIT | %s @ %.2f | profit=%.2f",
            symbol, price, profit
        )
    
    def log_square_off(self, symbol: str, price: float, pnl: float, reason: str):
        """Log a square-off (forced exit)"""
        self.logger.info(
            "TRADE:SQUARE_OFF | %s @ %.2f | PnL=%.2f | %s",
            symbol, price, pnl, reason
        )


# Global trade logger instance
trade_logger = TradeLogger()
