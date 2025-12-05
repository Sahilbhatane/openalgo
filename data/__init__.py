# Data module for OpenAlgo Intraday Trading Bot
# Contains historical data client, market feed, and validators

from .historical_client import HistoricalClient
from .market_feed import MarketFeed
from .validators import (
    validate_ohlc,
    validate_tick,
    validate_ticks,
    ValidationError,
    OHLCValidationError,
    TickValidationError,
    DataCorruptionError,
    ValidationResult,
    ValidationIssue,
    Severity,
    convert_angelone_prices,
    check_data_freshness,
)

__all__ = [
    # Clients
    'HistoricalClient',
    'MarketFeed',
    # Validators
    'validate_ohlc',
    'validate_tick',
    'validate_ticks',
    # Exceptions
    'ValidationError',
    'OHLCValidationError',
    'TickValidationError',
    'DataCorruptionError',
    # Result types
    'ValidationResult',
    'ValidationIssue',
    'Severity',
    # Utilities
    'convert_angelone_prices',
    'check_data_freshness',
]
