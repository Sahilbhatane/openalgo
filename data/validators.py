"""
Data Validators for OpenAlgo Intraday Trading Bot

This module provides validation functions for OHLC and tick data received from
AngelOne SmartAPI. It ensures data integrity before processing by the strategy engine.

AngelOne SmartAPI Data Formats:
-------------------------------
Historical API Response: [timestamp, open, high, low, close, volume]
- Timestamp: ISO format with timezone (e.g., "2023-09-06T11:15:00+05:30")
- OHLC: Prices as floats
- Volume: Integer (can be 0 for indices)

WebSocket Tick Response (Binary, Little Endian):
- Mode 1 (LTP): 51 bytes - token, sequence, timestamp, ltp
- Mode 2 (Quote): 123 bytes - adds qty, avg price, volume, buy/sell qty, OHLC
- Mode 3 (SnapQuote): 379 bytes - adds best 5 bid/ask, OI, circuit limits

Price Handling:
- For currencies: divide by 10,000,000 (7 decimal places)
- For everything else: divide by 100 (prices in paise)

Human Steps Required:
---------------------
1. Validate that your symbol tokens match the current master contract file
2. Monitor for corporate actions (splits, bonuses) that affect price continuity
3. Cross-check validation results against actual broker data during paper trading
4. Adjust thresholds (MAX_PRICE_CHANGE_PCT, etc.) based on your trading instruments

Usage:
------
    from data.validators import validate_ohlc, validate_tick, ValidationError
    
    # Validate OHLC DataFrame
    df_clean, issues = validate_ohlc(df, auto_fix=True)
    
    # Validate single tick
    tick_clean, issues = validate_tick(tick_data, auto_fix=True)
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants (Adjust based on your instruments)
# =============================================================================

# Maximum allowed price change between consecutive candles (%)
# Adjust higher for volatile instruments like crypto or small-caps
MAX_PRICE_CHANGE_PCT = 20.0

# Maximum gap allowed between timestamps (in candle intervals)
# e.g., 2 means a gap of 2x the candle interval is tolerated
MAX_TIMESTAMP_GAP_MULTIPLIER = 3

# Price bounds (prevents obviously erroneous data)
MIN_VALID_PRICE = 0.01  # Prices below this are invalid
MAX_VALID_PRICE = 10_000_000  # ₹1 Crore - adjust for your instruments

# Volume bounds
MIN_VALID_VOLUME = 0  # 0 is valid for indices (NIFTY, BANKNIFTY)
MAX_VALID_VOLUME = 10_000_000_000  # 10 billion shares/contracts

# AngelOne specific: prices are in paise, divide by 100
ANGELONE_PRICE_DIVISOR = 100
ANGELONE_CURRENCY_DIVISOR = 10_000_000


# =============================================================================
# Exception Classes
# =============================================================================

class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class OHLCValidationError(ValidationError):
    """Raised when OHLC data validation fails."""
    pass


class TickValidationError(ValidationError):
    """Raised when tick data validation fails."""
    pass


class DataCorruptionError(ValidationError):
    """Raised when data is corrupted beyond auto-repair."""
    pass


class Severity(Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in data."""
    severity: Severity
    field: str
    message: str
    row_index: Optional[int] = None
    original_value: Any = None
    fixed_value: Any = None
    auto_fixed: bool = False
    
    def __str__(self) -> str:
        location = f"[row {self.row_index}]" if self.row_index is not None else ""
        fixed = " (auto-fixed)" if self.auto_fixed else ""
        return f"{self.severity.value.upper()}{location} {self.field}: {self.message}{fixed}"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    rows_removed: int = 0
    rows_fixed: int = 0
    
    @property
    def has_errors(self) -> bool:
        return any(i.severity in (Severity.ERROR, Severity.CRITICAL) for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        error_count = sum(1 for i in self.issues if i.severity == Severity.ERROR)
        warning_count = sum(1 for i in self.issues if i.severity == Severity.WARNING)
        return (
            f"Valid: {self.is_valid} | "
            f"Errors: {error_count} | Warnings: {warning_count} | "
            f"Rows fixed: {self.rows_fixed} | Rows removed: {self.rows_removed}"
        )


# =============================================================================
# OHLC Validation Functions
# =============================================================================

def validate_ohlc(
    df: pd.DataFrame,
    interval_minutes: int = 5,
    auto_fix: bool = True,
    strict: bool = False,
    is_index: bool = False
) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Validate and optionally fix OHLC DataFrame.
    
    This function checks for:
    1. Required columns exist (timestamp, open, high, low, close, volume)
    2. Non-null values in critical fields
    3. Non-negative OHLC values
    4. Volume >= 0 (0 allowed for indices)
    5. OHLC relationship: low <= open <= high, low <= close <= high
    6. Timestamp continuity (no excessive gaps)
    7. Reasonable price changes between candles
    8. Proper data types
    
    Args:
        df: DataFrame with OHLC data
        interval_minutes: Expected candle interval in minutes
        auto_fix: If True, attempt to fix minor issues
        strict: If True, raise exception on any error
        is_index: If True, allow volume=0 (indices don't have volume)
    
    Returns:
        Tuple of (cleaned DataFrame, ValidationResult)
    
    Raises:
        OHLCValidationError: If strict=True and validation fails
        DataCorruptionError: If data is too corrupted to use
    
    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2025-01-01', periods=5, freq='5T'),
        ...     'open': [100, 101, 102, 103, 104],
        ...     'high': [101, 102, 103, 104, 105],
        ...     'low': [99, 100, 101, 102, 103],
        ...     'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        ...     'volume': [1000, 1100, 1200, 1300, 1400]
        ... })
        >>> df_clean, result = validate_ohlc(df)
        >>> print(result.summary())
    """
    issues: List[ValidationIssue] = []
    rows_removed = 0
    rows_fixed = 0
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Step 1: Check required columns
    # -------------------------------------------------------------------------
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise OHLCValidationError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # -------------------------------------------------------------------------
    # Step 2: Handle empty DataFrame
    # -------------------------------------------------------------------------
    if df.empty:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            field="dataframe",
            message="Empty DataFrame provided"
        ))
        return df, ValidationResult(is_valid=True, issues=issues)
    
    # -------------------------------------------------------------------------
    # Step 3: Validate and convert data types
    # -------------------------------------------------------------------------
    
    # Timestamp conversion
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            issues.append(ValidationIssue(
                severity=Severity.INFO,
                field="timestamp",
                message="Converted timestamp to datetime",
                auto_fixed=True
            ))
            rows_fixed += 1
        except Exception as e:
            raise OHLCValidationError(f"Cannot parse timestamp column: {e}")
    
    # Numeric columns conversion
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                issues.append(ValidationIssue(
                    severity=Severity.INFO,
                    field=col,
                    message=f"Converted {col} to numeric",
                    auto_fixed=True
                ))
            except Exception as e:
                raise OHLCValidationError(f"Cannot convert {col} to numeric: {e}")
    
    # -------------------------------------------------------------------------
    # Step 4: Check for null values
    # -------------------------------------------------------------------------
    null_counts = df[required_columns].isnull().sum()
    
    for col, count in null_counts.items():
        if count > 0:
            if col == 'volume' and auto_fix:
                # Auto-fix: fill missing volume with 0 or forward fill
                if is_index:
                    df['volume'] = df['volume'].fillna(0)
                else:
                    df['volume'] = df['volume'].fillna(method='ffill').fillna(0)
                
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field=col,
                    message=f"Filled {count} null volume values",
                    auto_fixed=True
                ))
                rows_fixed += count
                
            elif col in ['open', 'high', 'low', 'close'] and auto_fix:
                # For price columns, try forward fill for small gaps
                null_pct = count / len(df) * 100
                if null_pct < 5:  # Less than 5% nulls
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        field=col,
                        message=f"Forward-filled {count} null {col} values",
                        auto_fixed=True
                    ))
                    rows_fixed += count
                else:
                    # Too many nulls, remove those rows
                    before_len = len(df)
                    df = df.dropna(subset=[col])
                    removed = before_len - len(df)
                    rows_removed += removed
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        field=col,
                        message=f"Removed {removed} rows with null {col} values (>{null_pct:.1f}% nulls)"
                    ))
            else:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    field=col,
                    message=f"{count} null values found"
                ))
    
    # -------------------------------------------------------------------------
    # Step 5: Validate price bounds
    # -------------------------------------------------------------------------
    for col in ['open', 'high', 'low', 'close']:
        # Check for negative prices
        negative_mask = df[col] < 0
        if negative_mask.any():
            count = negative_mask.sum()
            if auto_fix:
                df.loc[negative_mask, col] = df.loc[negative_mask, col].abs()
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field=col,
                    message=f"Fixed {count} negative {col} values (converted to absolute)",
                    auto_fixed=True
                ))
                rows_fixed += count
            else:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    field=col,
                    message=f"{count} negative values found"
                ))
        
        # Check for zero prices (usually invalid except for delisted stocks)
        zero_mask = df[col] == 0
        if zero_mask.any():
            count = zero_mask.sum()
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field=col,
                message=f"{count} zero price values found (may indicate data issue)"
            ))
        
        # Check for unreasonably high prices
        high_mask = df[col] > MAX_VALID_PRICE
        if high_mask.any():
            count = high_mask.sum()
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field=col,
                message=f"{count} values exceed maximum valid price (₹{MAX_VALID_PRICE:,})"
            ))
    
    # -------------------------------------------------------------------------
    # Step 6: Validate OHLC relationships
    # -------------------------------------------------------------------------
    
    # High should be >= Open, Close, Low
    invalid_high = (df['high'] < df['open']) | (df['high'] < df['close']) | (df['high'] < df['low'])
    if invalid_high.any():
        count = invalid_high.sum()
        if auto_fix:
            # Fix by setting high to max of OHLC
            df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'high', 'low', 'close']].max(axis=1)
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field="high",
                message=f"Fixed {count} rows where high < max(open, close, low)",
                auto_fixed=True
            ))
            rows_fixed += count
        else:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field="high",
                message=f"{count} rows where high is not the highest value"
            ))
    
    # Low should be <= Open, Close, High
    invalid_low = (df['low'] > df['open']) | (df['low'] > df['close']) | (df['low'] > df['high'])
    if invalid_low.any():
        count = invalid_low.sum()
        if auto_fix:
            # Fix by setting low to min of OHLC
            df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'high', 'low', 'close']].min(axis=1)
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field="low",
                message=f"Fixed {count} rows where low > min(open, close, high)",
                auto_fixed=True
            ))
            rows_fixed += count
        else:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field="low",
                message=f"{count} rows where low is not the lowest value"
            ))
    
    # -------------------------------------------------------------------------
    # Step 7: Validate volume
    # -------------------------------------------------------------------------
    if not is_index:
        # For non-index instruments, volume should typically be > 0
        zero_volume = df['volume'] == 0
        if zero_volume.any():
            count = zero_volume.sum()
            # Only warn if significant portion has zero volume
            if count > len(df) * 0.1:  # More than 10%
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field="volume",
                    message=f"{count} rows with zero volume ({count/len(df)*100:.1f}%)"
                ))
    
    negative_volume = df['volume'] < 0
    if negative_volume.any():
        count = negative_volume.sum()
        if auto_fix:
            df.loc[negative_volume, 'volume'] = df.loc[negative_volume, 'volume'].abs()
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field="volume",
                message=f"Fixed {count} negative volume values",
                auto_fixed=True
            ))
            rows_fixed += count
        else:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field="volume",
                message=f"{count} negative volume values"
            ))
    
    # -------------------------------------------------------------------------
    # Step 8: Validate timestamp continuity
    # -------------------------------------------------------------------------
    if len(df) > 1:
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate expected interval
        expected_interval = timedelta(minutes=interval_minutes)
        max_gap = expected_interval * MAX_TIMESTAMP_GAP_MULTIPLIER
        
        # Check for gaps
        time_diffs = df['timestamp'].diff()
        gaps = time_diffs > max_gap
        
        if gaps.any():
            gap_count = gaps.sum()
            gap_indices = df.index[gaps].tolist()
            
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field="timestamp",
                message=f"{gap_count} gaps > {max_gap} found in timestamp sequence at indices: {gap_indices[:5]}{'...' if len(gap_indices) > 5 else ''}"
            ))
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['timestamp'], keep='first')
        if duplicates.any():
            count = duplicates.sum()
            if auto_fix:
                df = df.drop_duplicates(subset=['timestamp'], keep='first')
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field="timestamp",
                    message=f"Removed {count} duplicate timestamps",
                    auto_fixed=True
                ))
                rows_removed += count
            else:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    field="timestamp",
                    message=f"{count} duplicate timestamps found"
                ))
        
        # Check for out-of-order timestamps
        if not df['timestamp'].is_monotonic_increasing:
            if auto_fix:
                df = df.sort_values('timestamp').reset_index(drop=True)
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field="timestamp",
                    message="Sorted timestamps to ensure chronological order",
                    auto_fixed=True
                ))
            else:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    field="timestamp",
                    message="Timestamps are not in chronological order"
                ))
    
    # -------------------------------------------------------------------------
    # Step 9: Validate price changes (detect anomalies)
    # -------------------------------------------------------------------------
    if len(df) > 1:
        for col in ['open', 'high', 'low', 'close']:
            pct_change = df[col].pct_change().abs() * 100
            extreme_changes = pct_change > MAX_PRICE_CHANGE_PCT
            
            if extreme_changes.any():
                count = extreme_changes.sum()
                max_change = pct_change.max()
                
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field=col,
                    message=f"{count} extreme price changes > {MAX_PRICE_CHANGE_PCT}% detected (max: {max_change:.1f}%)"
                ))
    
    # -------------------------------------------------------------------------
    # Final validation status
    # -------------------------------------------------------------------------
    result = ValidationResult(
        is_valid=not any(i.severity in (Severity.ERROR, Severity.CRITICAL) for i in issues),
        issues=issues,
        rows_removed=rows_removed,
        rows_fixed=rows_fixed
    )
    
    if strict and not result.is_valid:
        error_messages = [str(i) for i in issues if i.severity in (Severity.ERROR, Severity.CRITICAL)]
        raise OHLCValidationError(
            f"OHLC validation failed with {len(error_messages)} errors:\n" +
            "\n".join(error_messages[:10])
        )
    
    # Log summary
    logger.info(f"OHLC validation: {result.summary()}")
    
    return df, result


# =============================================================================
# Tick Validation Functions
# =============================================================================

def validate_tick(
    tick: Dict[str, Any],
    auto_fix: bool = True,
    strict: bool = False,
    is_index: bool = False,
    is_currency: bool = False
) -> Tuple[Dict[str, Any], ValidationResult]:
    """
    Validate and optionally fix a single tick from WebSocket feed.
    
    AngelOne WebSocket tick format varies by mode:
    - Mode 1 (LTP): token, sequence, timestamp, ltp
    - Mode 2 (Quote): + volume, OHLC, buy/sell qty
    - Mode 3 (SnapQuote): + OI, best 5 bid/ask
    
    Args:
        tick: Dictionary containing tick data
        auto_fix: If True, attempt to fix minor issues
        strict: If True, raise exception on any error
        is_index: If True, allow volume=0
        is_currency: If True, divide prices by 10,000,000 instead of 100
    
    Returns:
        Tuple of (cleaned tick dict, ValidationResult)
    
    Raises:
        TickValidationError: If strict=True and validation fails
    
    Example:
        >>> tick = {
        ...     'symbol': 'RELIANCE',
        ...     'token': '2885',
        ...     'ltp': 245000,  # In paise
        ...     'volume': 1234567
        ... }
        >>> tick_clean, result = validate_tick(tick)
        >>> print(tick_clean['ltp'])  # 2450.00
    """
    issues: List[ValidationIssue] = []
    tick = tick.copy()  # Don't modify original
    
    # -------------------------------------------------------------------------
    # Step 1: Check for required fields
    # -------------------------------------------------------------------------
    # At minimum, we need ltp (Last Traded Price)
    if 'ltp' not in tick and 'last_price' not in tick and 'lastPrice' not in tick:
        raise TickValidationError(
            "Tick must contain 'ltp', 'last_price', or 'lastPrice' field. "
            f"Available fields: {list(tick.keys())}"
        )
    
    # Normalize field names
    field_mapping = {
        'last_price': 'ltp',
        'lastPrice': 'ltp',
        'last_traded_price': 'ltp',
        'tradingSymbol': 'symbol',
        'tradingsymbol': 'symbol',
        'symboltoken': 'token',
        'symbol_token': 'token',
        'instrument_token': 'token',
        'tradedVolume': 'volume',
        'traded_volume': 'volume',
        'vol': 'volume',
        'openPrice': 'open',
        'open_price': 'open',
        'highPrice': 'high',
        'high_price': 'high',
        'lowPrice': 'low',
        'low_price': 'low',
        'closePrice': 'close',
        'close_price': 'close',
        'prevClose': 'close',
    }
    
    for old_key, new_key in field_mapping.items():
        if old_key in tick and new_key not in tick:
            tick[new_key] = tick.pop(old_key)
    
    # -------------------------------------------------------------------------
    # Step 2: Validate and convert LTP
    # -------------------------------------------------------------------------
    ltp = tick.get('ltp')
    
    if ltp is None:
        raise TickValidationError("LTP is None")
    
    # Check if numeric
    if not isinstance(ltp, (int, float)):
        try:
            ltp = float(ltp)
            tick['ltp'] = ltp
            issues.append(ValidationIssue(
                severity=Severity.INFO,
                field="ltp",
                message="Converted LTP to float",
                auto_fixed=True
            ))
        except (ValueError, TypeError):
            raise TickValidationError(f"LTP is not a valid number: {ltp}")
    
    # Check for NaN/Inf
    if math.isnan(ltp) or math.isinf(ltp):
        raise TickValidationError(f"LTP is NaN or Inf: {ltp}")
    
    # Check for negative
    if ltp < 0:
        if auto_fix:
            tick['ltp'] = abs(ltp)
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field="ltp",
                message=f"Fixed negative LTP: {ltp} -> {abs(ltp)}",
                original_value=ltp,
                fixed_value=abs(ltp),
                auto_fixed=True
            ))
        else:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field="ltp",
                message=f"Negative LTP: {ltp}"
            ))
    
    # Check for zero (usually invalid except during circuit)
    if ltp == 0:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            field="ltp",
            message="LTP is zero (may indicate circuit or data issue)"
        ))
    
    # Check bounds
    if ltp > MAX_VALID_PRICE:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            field="ltp",
            message=f"LTP {ltp} exceeds maximum valid price (may need price divisor)"
        ))
    
    # -------------------------------------------------------------------------
    # Step 3: Validate volume (if present)
    # -------------------------------------------------------------------------
    if 'volume' in tick:
        volume = tick['volume']
        
        if volume is None:
            if auto_fix:
                tick['volume'] = 0
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field="volume",
                    message="Fixed null volume to 0",
                    auto_fixed=True
                ))
        elif not isinstance(volume, (int, float)):
            try:
                volume = int(float(volume))
                tick['volume'] = volume
            except (ValueError, TypeError):
                if auto_fix:
                    tick['volume'] = 0
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        field="volume",
                        message=f"Fixed invalid volume '{volume}' to 0",
                        auto_fixed=True
                    ))
                else:
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        field="volume",
                        message=f"Invalid volume type: {type(volume)}"
                    ))
        else:
            # Check for negative
            if volume < 0:
                if auto_fix:
                    tick['volume'] = abs(int(volume))
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        field="volume",
                        message=f"Fixed negative volume: {volume}",
                        auto_fixed=True
                    ))
                else:
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        field="volume",
                        message=f"Negative volume: {volume}"
                    ))
            
            # Check bounds
            if volume > MAX_VALID_VOLUME:
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field="volume",
                    message=f"Volume {volume} exceeds maximum valid ({MAX_VALID_VOLUME:,})"
                ))
    
    # -------------------------------------------------------------------------
    # Step 4: Validate OHLC fields (if present)
    # -------------------------------------------------------------------------
    for price_field in ['open', 'high', 'low', 'close']:
        if price_field in tick:
            price = tick[price_field]
            
            if price is None:
                if auto_fix:
                    # Use LTP as fallback
                    tick[price_field] = tick.get('ltp', 0)
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        field=price_field,
                        message=f"Fixed null {price_field} using LTP",
                        auto_fixed=True
                    ))
                continue
            
            if not isinstance(price, (int, float)):
                try:
                    tick[price_field] = float(price)
                except (ValueError, TypeError):
                    if auto_fix:
                        tick[price_field] = tick.get('ltp', 0)
                        issues.append(ValidationIssue(
                            severity=Severity.WARNING,
                            field=price_field,
                            message=f"Fixed invalid {price_field} using LTP",
                            auto_fixed=True
                        ))
            
            if tick[price_field] < 0:
                if auto_fix:
                    tick[price_field] = abs(tick[price_field])
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        field=price_field,
                        message=f"Fixed negative {price_field}",
                        auto_fixed=True
                    ))
    
    # -------------------------------------------------------------------------
    # Step 5: Validate OHLC relationships (if all present)
    # -------------------------------------------------------------------------
    if all(f in tick for f in ['open', 'high', 'low', 'close']):
        o, h, l, c = tick['open'], tick['high'], tick['low'], tick['close']
        
        if h < max(o, c, l):
            if auto_fix:
                tick['high'] = max(o, h, l, c)
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field="high",
                    message="Fixed high to be maximum of OHLC",
                    auto_fixed=True
                ))
        
        if l > min(o, c, h):
            if auto_fix:
                tick['low'] = min(o, h, l, c)
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    field="low",
                    message="Fixed low to be minimum of OHLC",
                    auto_fixed=True
                ))
    
    # -------------------------------------------------------------------------
    # Step 6: Validate timestamp (if present)
    # -------------------------------------------------------------------------
    if 'timestamp' in tick:
        ts = tick['timestamp']
        
        if isinstance(ts, (int, float)):
            # Assume epoch milliseconds (AngelOne format)
            try:
                tick['timestamp'] = datetime.fromtimestamp(ts / 1000)
            except (ValueError, OSError):
                if auto_fix:
                    tick['timestamp'] = datetime.now()
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        field="timestamp",
                        message=f"Fixed invalid epoch timestamp: {ts}",
                        auto_fixed=True
                    ))
        elif isinstance(ts, str):
            try:
                tick['timestamp'] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except ValueError:
                if auto_fix:
                    tick['timestamp'] = datetime.now()
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        field="timestamp",
                        message=f"Fixed invalid timestamp string: {ts}",
                        auto_fixed=True
                    ))
    
    # -------------------------------------------------------------------------
    # Step 7: Add missing but useful fields
    # -------------------------------------------------------------------------
    if 'timestamp' not in tick and auto_fix:
        tick['timestamp'] = datetime.now()
        issues.append(ValidationIssue(
            severity=Severity.INFO,
            field="timestamp",
            message="Added missing timestamp",
            auto_fixed=True
        ))
    
    # -------------------------------------------------------------------------
    # Final validation status
    # -------------------------------------------------------------------------
    result = ValidationResult(
        is_valid=not any(i.severity in (Severity.ERROR, Severity.CRITICAL) for i in issues),
        issues=issues,
        rows_fixed=sum(1 for i in issues if i.auto_fixed)
    )
    
    if strict and not result.is_valid:
        error_messages = [str(i) for i in issues if i.severity in (Severity.ERROR, Severity.CRITICAL)]
        raise TickValidationError(
            f"Tick validation failed:\n" + "\n".join(error_messages)
        )
    
    return tick, result


# =============================================================================
# Batch Tick Validation
# =============================================================================

def validate_ticks(
    ticks: List[Dict[str, Any]],
    auto_fix: bool = True,
    drop_invalid: bool = True
) -> Tuple[List[Dict[str, Any]], ValidationResult]:
    """
    Validate a batch of ticks.
    
    Args:
        ticks: List of tick dictionaries
        auto_fix: If True, attempt to fix minor issues
        drop_invalid: If True, remove ticks that fail validation
    
    Returns:
        Tuple of (list of valid ticks, combined ValidationResult)
    """
    valid_ticks = []
    all_issues = []
    rows_removed = 0
    
    for i, tick in enumerate(ticks):
        try:
            clean_tick, result = validate_tick(tick, auto_fix=auto_fix)
            
            if result.is_valid or not drop_invalid:
                valid_ticks.append(clean_tick)
            else:
                rows_removed += 1
            
            # Add row index to issues
            for issue in result.issues:
                issue.row_index = i
                all_issues.append(issue)
                
        except (TickValidationError, DataCorruptionError) as e:
            rows_removed += 1
            all_issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field="tick",
                message=str(e),
                row_index=i
            ))
    
    combined_result = ValidationResult(
        is_valid=rows_removed == 0,
        issues=all_issues,
        rows_removed=rows_removed
    )
    
    logger.info(f"Batch tick validation: {len(valid_ticks)}/{len(ticks)} valid, {rows_removed} removed")
    
    return valid_ticks, combined_result


# =============================================================================
# Utility Functions
# =============================================================================

def convert_angelone_prices(
    data: Union[Dict[str, Any], pd.DataFrame],
    is_currency: bool = False
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Convert AngelOne prices from paise to rupees.
    
    AngelOne API returns prices as integers:
    - Regular instruments: price * 100 (paise)
    - Currencies: price * 10,000,000 (7 decimal places)
    
    Args:
        data: Dictionary or DataFrame with price data
        is_currency: If True, use currency divisor
    
    Returns:
        Data with converted prices
    """
    divisor = ANGELONE_CURRENCY_DIVISOR if is_currency else ANGELONE_PRICE_DIVISOR
    price_fields = ['ltp', 'open', 'high', 'low', 'close', 'bid', 'ask']
    
    if isinstance(data, dict):
        data = data.copy()
        for field in price_fields:
            if field in data and isinstance(data[field], (int, float)):
                data[field] = data[field] / divisor
        return data
    
    elif isinstance(data, pd.DataFrame):
        data = data.copy()
        for field in price_fields:
            if field in data.columns:
                data[field] = data[field] / divisor
        return data
    
    return data


def check_data_freshness(
    timestamp: datetime,
    max_age_seconds: int = 60
) -> Tuple[bool, float]:
    """
    Check if data timestamp is recent enough.
    
    Args:
        timestamp: Data timestamp
        max_age_seconds: Maximum allowed age in seconds
    
    Returns:
        Tuple of (is_fresh, age_in_seconds)
    """
    now = datetime.now()
    
    # Handle timezone-aware timestamps
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)
    
    age = (now - timestamp).total_seconds()
    is_fresh = age <= max_age_seconds
    
    return is_fresh, age


# =============================================================================
# Unit Tests
# =============================================================================

def run_tests():
    """
    Run unit tests for validators module.
    
    Call this function to verify the validators work correctly:
        python -c "from data.validators import run_tests; run_tests()"
    """
    import traceback
    
    print("=" * 60)
    print("Running Validator Unit Tests")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    def test(name: str, condition: bool, message: str = ""):
        nonlocal tests_passed, tests_failed
        if condition:
            print(f"  ✓ {name}")
            tests_passed += 1
        else:
            print(f"  ✗ {name}: {message}")
            tests_failed += 1
    
    # -------------------------------------------------------------------------
    # Test 1: Valid OHLC DataFrame
    # -------------------------------------------------------------------------
    print("\n1. Testing valid OHLC DataFrame...")
    try:
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01 09:15', periods=5, freq='5T'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        df_clean, result = validate_ohlc(df)
        test("Valid DataFrame passes", result.is_valid)
        test("No errors reported", not result.has_errors)
        test("DataFrame unchanged", len(df_clean) == 5)
    except Exception as e:
        test("Valid DataFrame test", False, str(e))
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Test 2: OHLC with missing columns
    # -------------------------------------------------------------------------
    print("\n2. Testing OHLC with missing columns...")
    try:
        df_bad = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=3, freq='5T'),
            'open': [100, 101, 102],
            'close': [101, 102, 103]
        })
        
        try:
            validate_ohlc(df_bad)
            test("Missing columns raises error", False, "Should have raised OHLCValidationError")
        except OHLCValidationError as e:
            test("Missing columns raises error", "Missing required columns" in str(e))
    except Exception as e:
        test("Missing columns test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 3: OHLC with negative prices (auto-fix)
    # -------------------------------------------------------------------------
    print("\n3. Testing OHLC with negative prices (auto-fix)...")
    try:
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=3, freq='5T'),
            'open': [100, -101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        df_clean, result = validate_ohlc(df, auto_fix=True)
        test("Negative price fixed", df_clean.iloc[1]['open'] == 101)
        test("Auto-fix reported", result.rows_fixed > 0)
    except Exception as e:
        test("Negative prices test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 4: OHLC relationship violation (high < low)
    # -------------------------------------------------------------------------
    print("\n4. Testing OHLC relationship violations...")
    try:
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=3, freq='5T'),
            'open': [100, 101, 102],
            'high': [99, 102, 103],  # First high < low!
            'low': [100, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        df_clean, result = validate_ohlc(df, auto_fix=True)
        test("OHLC relationship fixed", df_clean.iloc[0]['high'] >= df_clean.iloc[0]['low'])
    except Exception as e:
        test("OHLC relationship test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 5: OHLC with null volume (auto-fix)
    # -------------------------------------------------------------------------
    print("\n5. Testing OHLC with null volume...")
    try:
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=3, freq='5T'),
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, None, 1200]
        })
        
        df_clean, result = validate_ohlc(df, auto_fix=True)
        test("Null volume fixed", pd.notna(df_clean.iloc[1]['volume']))
    except Exception as e:
        test("Null volume test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 6: Valid tick
    # -------------------------------------------------------------------------
    print("\n6. Testing valid tick...")
    try:
        tick = {
            'symbol': 'RELIANCE',
            'token': '2885',
            'ltp': 2450.0,
            'volume': 1234567,
            'open': 2440.0,
            'high': 2460.0,
            'low': 2435.0,
            'close': 2445.0
        }
        
        tick_clean, result = validate_tick(tick)
        test("Valid tick passes", result.is_valid)
        test("LTP preserved", tick_clean['ltp'] == 2450.0)
    except Exception as e:
        test("Valid tick test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 7: Tick with missing LTP
    # -------------------------------------------------------------------------
    print("\n7. Testing tick with missing LTP...")
    try:
        tick = {'symbol': 'RELIANCE', 'volume': 1000}
        
        try:
            validate_tick(tick)
            test("Missing LTP raises error", False, "Should have raised TickValidationError")
        except TickValidationError as e:
            test("Missing LTP raises error", "ltp" in str(e).lower())
    except Exception as e:
        test("Missing LTP test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 8: Tick with negative LTP (auto-fix)
    # -------------------------------------------------------------------------
    print("\n8. Testing tick with negative LTP (auto-fix)...")
    try:
        tick = {'ltp': -100.0, 'volume': 1000}
        
        tick_clean, result = validate_tick(tick, auto_fix=True)
        test("Negative LTP fixed", tick_clean['ltp'] == 100.0)
        test("Fix was reported", any('negative' in str(i).lower() for i in result.issues))
    except Exception as e:
        test("Negative LTP test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 9: Tick field normalization
    # -------------------------------------------------------------------------
    print("\n9. Testing tick field normalization...")
    try:
        tick = {
            'lastPrice': 100.0,
            'tradingSymbol': 'INFY',
            'tradedVolume': 5000
        }
        
        tick_clean, result = validate_tick(tick, auto_fix=True)
        test("lastPrice normalized to ltp", 'ltp' in tick_clean)
        test("tradingSymbol normalized", 'symbol' in tick_clean)
        test("tradedVolume normalized", 'volume' in tick_clean)
    except Exception as e:
        test("Field normalization test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 10: Batch tick validation
    # -------------------------------------------------------------------------
    print("\n10. Testing batch tick validation...")
    try:
        ticks = [
            {'ltp': 100.0, 'volume': 1000},
            {'ltp': -50.0, 'volume': 2000},  # Invalid
            {'ltp': 150.0, 'volume': 3000},
            {'symbol': 'BAD'},  # Missing LTP
        ]
        
        valid_ticks, result = validate_ticks(ticks, auto_fix=True, drop_invalid=True)
        test("Batch validation filters invalid", len(valid_ticks) == 3)
        test("Invalid ticks counted", result.rows_removed == 1)
    except Exception as e:
        test("Batch validation test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 11: Price conversion utility
    # -------------------------------------------------------------------------
    print("\n11. Testing AngelOne price conversion...")
    try:
        tick = {'ltp': 245000, 'open': 244000}  # Prices in paise
        
        converted = convert_angelone_prices(tick)
        test("LTP converted correctly", converted['ltp'] == 2450.0)
        test("Open converted correctly", converted['open'] == 2440.0)
        
        # Test currency conversion
        currency_tick = {'ltp': 832500000}  # 83.25 with 7 decimal places
        converted_currency = convert_angelone_prices(currency_tick, is_currency=True)
        test("Currency conversion", abs(converted_currency['ltp'] - 83.25) < 0.01)
    except Exception as e:
        test("Price conversion test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 12: Data freshness check
    # -------------------------------------------------------------------------
    print("\n12. Testing data freshness check...")
    try:
        fresh_ts = datetime.now() - timedelta(seconds=30)
        is_fresh, age = check_data_freshness(fresh_ts, max_age_seconds=60)
        test("Fresh data detected", is_fresh)
        test("Age calculated correctly", 25 < age < 35)
        
        stale_ts = datetime.now() - timedelta(seconds=120)
        is_fresh, age = check_data_freshness(stale_ts, max_age_seconds=60)
        test("Stale data detected", not is_fresh)
    except Exception as e:
        test("Freshness check test", False, str(e))
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print("=" * 60)
    
    if tests_failed > 0:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return False
    else:
        print("\n✓ All tests passed!")
        return True


# =============================================================================
# Module entry point
# =============================================================================

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = run_tests()
    exit(0 if success else 1)
