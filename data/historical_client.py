"""
Historical Data Client for OpenAlgo Intraday Trading Bot

This module provides a robust client for fetching historical OHLC data
from the AngelOne/OpenAlgo Historical Data API.

Configuration:
    Set the following environment variables in your .env file:
    - HISTORICAL_API_KEY: Your historical data API key
    - HISTORICAL_API_SECRET: Your historical data API secret

Example Usage:
    from data.historical_client import HistoricalClient
    
    client = HistoricalClient()
    df = client.get_ohlc("RELIANCE", "5minute", start_date="2025-12-01", end_date="2025-12-05")
    print(df)
"""

import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from functools import wraps

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)


class HistoricalDataError(Exception):
    """Base exception for historical data errors."""
    pass


class DataIncompleteError(HistoricalDataError):
    """Raised when fetched data is incomplete or missing expected fields."""
    pass


class DataMalformedError(HistoricalDataError):
    """Raised when data format is invalid or cannot be parsed."""
    pass


class APIConnectionError(HistoricalDataError):
    """Raised when connection to API fails after retries."""
    pass


class SimpleCache:
    """
    Simple in-memory cache with TTL (Time To Live) support.
    
    Caches OHLC data to avoid repeated API calls for the same data.
    Cache entries expire after the specified TTL.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache with TTL.
        
        Args:
            ttl_seconds: Time to live for cache entries in seconds (default: 5 minutes)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        key_string = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if it exists and hasn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self._ttl:
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return entry['data'].copy()
            else:
                # Entry expired, remove it
                del self._cache[key]
                logger.debug(f"Cache expired for key: {key[:8]}...")
        return None
    
    def set(self, key: str, data: pd.DataFrame) -> None:
        """
        Store data in cache.
        
        Args:
            key: Cache key
            data: DataFrame to cache
        """
        self._cache[key] = {
            'data': data.copy(),
            'timestamp': time.time()
        }
        logger.debug(f"Cached data for key: {key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


def with_cache(func: Callable) -> Callable:
    """Decorator to enable caching for methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Generate cache key
        cache_key = self._cache._generate_key(func.__name__, *args, **kwargs)
        
        # Check cache first
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Call original function
        result = func(self, *args, **kwargs)
        
        # Cache the result
        if result is not None and not result.empty:
            self._cache.set(cache_key, result)
        
        return result
    return wrapper


class HistoricalClient:
    """
    Client for fetching historical OHLC data from AngelOne/OpenAlgo API.
    
    This client provides:
    - Automatic retry logic with exponential backoff
    - Request timeouts to prevent hanging
    - Response validation and error handling
    - Simple caching to avoid repeated lookups
    - Clean pandas DataFrame output
    
    Attributes:
        base_url: Base URL for the historical data API
        api_key: API key for authentication
        api_secret: API secret for authentication
    
    Example:
        >>> client = HistoricalClient()
        >>> df = client.get_ohlc("RELIANCE", "5minute")
        >>> print(df.columns)
        Index(['timestamp', 'open', 'high', 'low', 'close', 'volume'], dtype='object')
    """
    
    # =========================================================================
    # CONFIGURATION: Update these endpoints with your actual API URLs
    # =========================================================================
    
    # Base URL for the Historical Data API
    # TODO: Replace with your actual AngelOne/OpenAlgo historical data endpoint
    DEFAULT_BASE_URL = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/historical/v1"
    
    # Alternative endpoints (uncomment and modify as needed):
    # DEFAULT_BASE_URL = "https://your-custom-api.com/historical"
    # DEFAULT_BASE_URL = "http://127.0.0.1:5000/api/historical"  # Local OpenAlgo
    
    # API endpoints (relative to base_url)
    ENDPOINT_CANDLE = "/getCandleData"  # Single symbol OHLC data
    
    # Supported intervals
    VALID_INTERVALS = [
        "ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "TEN_MINUTE",
        "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY"
    ]
    
    # Interval mapping (user-friendly to API format)
    INTERVAL_MAP = {
        "1m": "ONE_MINUTE", "1minute": "ONE_MINUTE", "1min": "ONE_MINUTE",
        "3m": "THREE_MINUTE", "3minute": "THREE_MINUTE", "3min": "THREE_MINUTE",
        "5m": "FIVE_MINUTE", "5minute": "FIVE_MINUTE", "5min": "FIVE_MINUTE",
        "10m": "TEN_MINUTE", "10minute": "TEN_MINUTE", "10min": "TEN_MINUTE",
        "15m": "FIFTEEN_MINUTE", "15minute": "FIFTEEN_MINUTE", "15min": "FIFTEEN_MINUTE",
        "30m": "THIRTY_MINUTE", "30minute": "THIRTY_MINUTE", "30min": "THIRTY_MINUTE",
        "1h": "ONE_HOUR", "1hour": "ONE_HOUR", "60minute": "ONE_HOUR",
        "1d": "ONE_DAY", "1day": "ONE_DAY", "daily": "ONE_DAY",
    }
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        cache_ttl: int = 300
    ):
        """
        Initialize the Historical Data Client.
        
        Args:
            base_url: Base URL for the API. If None, uses DEFAULT_BASE_URL
            api_key: API key. If None, reads from HISTORICAL_API_KEY env var
            api_secret: API secret. If None, reads from HISTORICAL_API_SECRET env var
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
            cache_ttl: Cache time-to-live in seconds (default: 300)
        
        Raises:
            ValueError: If API credentials are not provided or found in environment
        """
        self.base_url = (base_url or os.getenv('HISTORICAL_BASE_URL') or self.DEFAULT_BASE_URL).rstrip('/')
        self.api_key = api_key or os.getenv('HISTORICAL_API_KEY')
        self.api_secret = api_secret or os.getenv('HISTORICAL_API_SECRET')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Validate credentials
        if not self.api_key:
            raise ValueError(
                "API key is required. Set HISTORICAL_API_KEY environment variable "
                "or pass api_key parameter."
            )
        if not self.api_secret:
            raise ValueError(
                "API secret is required. Set HISTORICAL_API_SECRET environment variable "
                "or pass api_secret parameter."
            )
        
        # Initialize cache
        self._cache = SimpleCache(ttl_seconds=cache_ttl)
        
        # Initialize session with retry logic
        self._session = self._create_session()
        
        # Auth token (populated on first request if using JWT auth)
        self._auth_token: Optional[str] = None
        
        logger.info(f"HistoricalClient initialized with base_url: {self.base_url}")
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic.
        
        Returns:
            Configured requests.Session instance
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers including authentication.
        
        Returns:
            Dictionary of HTTP headers
        
        Note:
            TODO: Modify this method based on your API's authentication scheme.
            Common patterns:
            - API Key in header: {'X-API-Key': self.api_key}
            - Bearer token: {'Authorization': f'Bearer {self._auth_token}'}
            - Basic auth: handled separately via session.auth
        """
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-PrivateKey': self.api_key,
        }
        
        # Add auth token if available (for JWT-based auth)
        if self._auth_token:
            headers['Authorization'] = f'Bearer {self._auth_token}'
        
        return headers
    
    def _normalize_interval(self, interval: str) -> str:
        """
        Convert user-friendly interval to API format.
        
        Args:
            interval: Interval string (e.g., "5m", "5minute", "FIVE_MINUTE")
            
        Returns:
            API-compatible interval string
            
        Raises:
            ValueError: If interval is not recognized
        """
        interval_upper = interval.upper()
        interval_lower = interval.lower()
        
        # Check if already in API format
        if interval_upper in self.VALID_INTERVALS:
            return interval_upper
        
        # Try to map from user-friendly format
        if interval_lower in self.INTERVAL_MAP:
            return self.INTERVAL_MAP[interval_lower]
        
        raise ValueError(
            f"Invalid interval: {interval}. "
            f"Valid options: {', '.join(self.VALID_INTERVALS)} "
            f"or shortcuts: {', '.join(self.INTERVAL_MAP.keys())}"
        )
    
    def _parse_datetime(self, dt_input: Any) -> datetime:
        """
        Parse various datetime inputs into datetime object.
        
        Args:
            dt_input: datetime object, string, or None
            
        Returns:
            datetime object
        """
        if dt_input is None:
            return datetime.now()
        
        if isinstance(dt_input, datetime):
            return dt_input
        
        if isinstance(dt_input, str):
            # Try common formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%Y/%m/%d",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(dt_input, fmt)
                except ValueError:
                    continue
            
            raise ValueError(f"Could not parse datetime: {dt_input}")
        
        raise TypeError(f"Expected datetime or string, got {type(dt_input)}")
    
    def _validate_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate API response and extract candle data.
        
        Args:
            data: Raw API response dictionary
            
        Returns:
            List of candle data dictionaries
            
        Raises:
            DataIncompleteError: If required fields are missing
            DataMalformedError: If data structure is invalid
        """
        # Check for API error response
        if data.get('status') == False or data.get('success') == False:
            error_msg = data.get('message') or data.get('error') or 'Unknown API error'
            raise HistoricalDataError(f"API error: {error_msg}")
        
        # Extract candle data - adjust based on your API response structure
        # TODO: Modify these keys based on your actual API response format
        candles = data.get('data') or data.get('candles') or data.get('result')
        
        if candles is None:
            raise DataIncompleteError(
                "Response missing candle data. "
                f"Available keys: {list(data.keys())}"
            )
        
        if not isinstance(candles, list):
            raise DataMalformedError(
                f"Expected list of candles, got {type(candles)}"
            )
        
        return candles
    
    def _candles_to_dataframe(self, candles: List[Any]) -> pd.DataFrame:
        """
        Convert raw candle data to clean pandas DataFrame.
        
        Args:
            candles: List of candle data (list of lists or list of dicts)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            DataMalformedError: If candle data cannot be parsed
        """
        if not candles:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        try:
            # Determine data format
            first_candle = candles[0]
            
            if isinstance(first_candle, list):
                # Format: [[timestamp, open, high, low, close, volume], ...]
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            elif isinstance(first_candle, dict):
                # Format: [{'timestamp': ..., 'open': ..., ...}, ...]
                df = pd.DataFrame(candles)
                
                # Normalize column names (handle various naming conventions)
                column_mapping = {
                    'time': 'timestamp', 'datetime': 'timestamp', 'date': 'timestamp',
                    't': 'timestamp', 'ts': 'timestamp',
                    'o': 'open', 'Open': 'open',
                    'h': 'high', 'High': 'high',
                    'l': 'low', 'Low': 'low',
                    'c': 'close', 'Close': 'close',
                    'v': 'volume', 'Volume': 'volume', 'vol': 'volume',
                }
                df = df.rename(columns=column_mapping)
            
            else:
                raise DataMalformedError(f"Unknown candle format: {type(first_candle)}")
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise DataIncompleteError(f"Missing required columns: {missing}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure numeric types for OHLCV
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Select and order columns
            df = df[required_columns]
            
            # Check for NaN values
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Data contains {nan_count} NaN values")
            
            return df
            
        except (KeyError, IndexError) as e:
            raise DataMalformedError(f"Failed to parse candle data: {e}")
    
    @with_cache
    def get_ohlc(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        exchange: str = "NSE",
        symbol_token: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLC (Open, High, Low, Close) data for a single symbol.
        
        Args:
            symbol: Trading symbol (e.g., "RELIANCE", "NIFTY", "BANKNIFTY")
            interval: Candle interval (e.g., "5minute", "15minute", "1hour")
            start_date: Start date for data (default: 7 days ago)
            end_date: End date for data (default: now)
            exchange: Exchange code (default: "NSE")
            symbol_token: Symbol token if known (speeds up lookup)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            HistoricalDataError: If API request fails
            DataIncompleteError: If data is incomplete
            DataMalformedError: If data format is invalid
            
        Example:
            >>> df = client.get_ohlc("RELIANCE", "5minute", "2025-12-01", "2025-12-05")
            >>> print(df.head())
                        timestamp    open    high     low   close   volume
            0 2025-12-01 09:15:00  2450.0  2455.0  2448.0  2453.0  1234567
            1 2025-12-01 09:20:00  2453.0  2458.0  2451.0  2456.0   987654
        """
        # Parse dates
        end_dt = self._parse_datetime(end_date)
        start_dt = self._parse_datetime(start_date) if start_date else (end_dt - timedelta(days=7))
        
        # Normalize interval
        api_interval = self._normalize_interval(interval)
        
        # Format dates for API
        # TODO: Adjust date format based on your API requirements
        from_date = start_dt.strftime("%Y-%m-%d %H:%M")
        to_date = end_dt.strftime("%Y-%m-%d %H:%M")
        
        # Build request payload
        # TODO: Modify this payload based on your API's request format
        payload = {
            "exchange": exchange,
            "symboltoken": symbol_token or symbol,  # Some APIs require token, others use symbol
            "interval": api_interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        
        logger.info(f"Fetching OHLC: {symbol} {api_interval} from {from_date} to {to_date}")
        
        try:
            response = self._session.post(
                f"{self.base_url}{self.ENDPOINT_CANDLE}",
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            raise APIConnectionError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise APIConnectionError(f"Connection failed: {e}")
        except requests.exceptions.HTTPError as e:
            raise HistoricalDataError(f"HTTP error: {e}")
        
        try:
            data = response.json()
        except ValueError:
            raise DataMalformedError("Invalid JSON response from API")
        
        # Validate and extract candles
        candles = self._validate_response(data)
        
        # Convert to DataFrame
        df = self._candles_to_dataframe(candles)
        
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        
        return df
    
    def get_multiple(
        self,
        symbols: List[str],
        interval: str,
        days_back: int = 7,
        exchange: str = "NSE"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLC data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            interval: Candle interval
            days_back: Number of days of historical data (default: 7)
            exchange: Exchange code (default: "NSE")
            
        Returns:
            Dictionary mapping symbol to DataFrame
            
        Example:
            >>> data = client.get_multiple(["RELIANCE", "TCS", "INFY"], "15minute", days_back=5)
            >>> for symbol, df in data.items():
            ...     print(f"{symbol}: {len(df)} candles")
            RELIANCE: 150 candles
            TCS: 150 candles
            INFY: 150 candles
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        results: Dict[str, pd.DataFrame] = {}
        errors: List[str] = []
        
        for symbol in symbols:
            try:
                df = self.get_ohlc(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    exchange=exchange
                )
                results[symbol] = df
                
            except HistoricalDataError as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                errors.append(f"{symbol}: {e}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed symbols
        
        if errors:
            logger.warning(f"Errors occurred for {len(errors)} symbols: {errors}")
        
        logger.info(f"Fetched data for {len(results)} symbols, {len(errors)} errors")
        
        return results
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'size': self._cache.size(),
            'ttl_seconds': self._cache._ttl,
        }
    
    def close(self) -> None:
        """Close the client session and clean up resources."""
        self._session.close()
        self._cache.clear()
        logger.info("HistoricalClient closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_client: Optional[HistoricalClient] = None


def get_client() -> HistoricalClient:
    """
    Get or create the default HistoricalClient instance.
    
    Returns:
        HistoricalClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = HistoricalClient()
    return _default_client


def get_ohlc(symbol: str, interval: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to fetch OHLC data using the default client.
    
    Args:
        symbol: Trading symbol
        interval: Candle interval
        **kwargs: Additional arguments passed to HistoricalClient.get_ohlc()
        
    Returns:
        DataFrame with OHLC data
    """
    return get_client().get_ohlc(symbol, interval, **kwargs)


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    print("=" * 60)
    print("Historical Data Client - Example Usage")
    print("=" * 60)
    
    try:
        # Create client
        client = HistoricalClient()
        
        # Fetch single symbol
        print("\n1. Fetching single symbol...")
        df = client.get_ohlc("RELIANCE", "5minute", days_back=1)
        print(f"   Fetched {len(df)} candles")
        if not df.empty:
            print(f"   Columns: {list(df.columns)}")
            print(f"   First row:\n{df.head(1)}")
        
        # Fetch multiple symbols
        print("\n2. Fetching multiple symbols...")
        data = client.get_multiple(["RELIANCE", "TCS"], "15minute", days_back=1)
        for symbol, symbol_df in data.items():
            print(f"   {symbol}: {len(symbol_df)} candles")
        
        # Cache stats
        print(f"\n3. Cache stats: {client.get_cache_stats()}")
        
        # Cleanup
        client.close()
        print("\n4. Client closed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
