"""
Market Feed Client for OpenAlgo Intraday Trading Bot

This module provides a high-speed WebSocket-based market data feed
for real-time tick data from AngelOne/OpenAlgo.

Configuration:
    Set the following environment variables in your .env file:
    - BROKER_API_KEY_MARKET: Your market feed API key
    - BROKER_API_SECRET_MARKET: Your market feed API secret

Features:
    - WebSocket-based real-time tick streaming
    - Automatic reconnection with exponential backoff
    - Heartbeat monitoring for connection health
    - Async event loop integration
    - Thread-safe callback dispatch

Example Usage:
    import asyncio
    from data.market_feed import MarketFeed
    
    def on_tick(tick):
        print(f"Tick: {tick['symbol']} @ {tick['ltp']}")
    
    feed = MarketFeed()
    feed.on_tick(on_tick)
    
    async def main():
        await feed.connect()
        await feed.subscribe(["RELIANCE", "TCS", "INFY"])
        # Feed runs until disconnected
    
    asyncio.run(main())
"""

import os
import json
import time
import asyncio
import threading
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    raise ImportError(
        "websockets library is required. Install with: pip install websockets"
    )

# Configure logging
logger = logging.getLogger(__name__)


class FeedState(Enum):
    """Market feed connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class FeedError(Exception):
    """Base exception for market feed errors."""
    pass


class ConnectionError(FeedError):
    """Raised when connection fails."""
    pass


class SubscriptionError(FeedError):
    """Raised when subscription fails."""
    pass


class FeedTimeoutError(FeedError):
    """Raised when feed becomes unresponsive."""
    pass


@dataclass
class Tick:
    """
    Represents a single market tick.
    
    Attributes:
        symbol: Trading symbol
        token: Symbol token/code
        ltp: Last traded price
        open: Open price
        high: High price
        low: Low price
        close: Previous close price
        volume: Total traded volume
        bid: Best bid price
        ask: Best ask price
        bid_qty: Best bid quantity
        ask_qty: Best ask quantity
        timestamp: Tick timestamp
        exchange: Exchange code
    """
    symbol: str
    token: str
    ltp: float
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = "NSE"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tick to dictionary."""
        return {
            'symbol': self.symbol,
            'token': self.token,
            'ltp': self.ltp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_qty': self.bid_qty,
            'ask_qty': self.ask_qty,
            'timestamp': self.timestamp.isoformat(),
            'exchange': self.exchange,
        }


class MarketFeed:
    """
    High-speed WebSocket market data feed for AngelOne/OpenAlgo.
    
    This class provides:
    - Real-time tick streaming via WebSocket
    - Automatic reconnection with exponential backoff
    - Heartbeat monitoring (raises warning if no data for >3 seconds)
    - Async event loop integration
    - Thread-safe callback dispatch
    
    Attributes:
        api_key: Market feed API key
        api_secret: Market feed API secret
        state: Current connection state
    
    Example:
        >>> feed = MarketFeed()
        >>> feed.on_tick(lambda t: print(f"{t['symbol']}: {t['ltp']}"))
        >>> await feed.connect()
        >>> await feed.subscribe(["RELIANCE", "NIFTY50"])
    """
    
    # =========================================================================
    # CONFIGURATION: Update these endpoints with your actual API URLs
    # =========================================================================
    
    # WebSocket URL for the Market Feed API
    # TODO: Replace with your actual AngelOne/OpenAlgo WebSocket endpoint
    DEFAULT_WS_URL = "wss://smartapisocket.angelone.in/smart-stream"
    
    # Alternative endpoints (uncomment and modify as needed):
    # DEFAULT_WS_URL = "wss://your-custom-feed.com/stream"
    # DEFAULT_WS_URL = "ws://127.0.0.1:8765"  # Local OpenAlgo WebSocket
    
    # REST API URL for authentication (if needed)
    DEFAULT_REST_URL = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/user/v1"
    
    # Heartbeat configuration
    HEARTBEAT_INTERVAL = 30  # Send heartbeat every 30 seconds
    FEED_TIMEOUT = 3  # Warn if no data for 3 seconds
    
    # Reconnection configuration
    MAX_RECONNECT_ATTEMPTS = 10
    INITIAL_RECONNECT_DELAY = 1  # seconds
    MAX_RECONNECT_DELAY = 60  # seconds
    
    def __init__(
        self,
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        auth_token: Optional[str] = None,
        feed_token: Optional[str] = None
    ):
        """
        Initialize the Market Feed client.
        
        Args:
            ws_url: WebSocket URL. If None, uses DEFAULT_WS_URL
            api_key: API key. If None, reads from BROKER_API_KEY_MARKET env var
            api_secret: API secret. If None, reads from BROKER_API_SECRET_MARKET env var
            auth_token: Pre-authenticated JWT token (optional)
            feed_token: Feed-specific token (optional, some APIs require this)
        """
        self.ws_url = ws_url or os.getenv('MARKET_FEED_WS_URL') or self.DEFAULT_WS_URL
        self.api_key = api_key or os.getenv('BROKER_API_KEY_MARKET')
        self.api_secret = api_secret or os.getenv('BROKER_API_SECRET_MARKET')
        self.auth_token = auth_token
        self.feed_token = feed_token
        
        # Connection state
        self._state = FeedState.DISCONNECTED
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._reconnect_attempts = 0
        
        # Subscriptions
        self._subscribed_symbols: Set[str] = set()
        self._symbol_tokens: Dict[str, str] = {}  # symbol -> token mapping
        
        # Callbacks
        self._tick_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._state_callbacks: List[Callable[[FeedState], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        
        # Timing
        self._last_tick_time: float = 0
        self._last_heartbeat_time: float = 0
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Message queue for processing
        self._message_queue: deque = deque(maxlen=10000)
        
        logger.info(f"MarketFeed initialized with ws_url: {self.ws_url}")
    
    @property
    def state(self) -> FeedState:
        """Get current connection state."""
        return self._state
    
    @state.setter
    def state(self, new_state: FeedState):
        """Set connection state and notify callbacks."""
        old_state = self._state
        self._state = new_state
        
        if old_state != new_state:
            logger.info(f"Feed state changed: {old_state.value} -> {new_state.value}")
            for callback in self._state_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if feed is connected."""
        return self._state == FeedState.CONNECTED and self._websocket is not None
    
    def on_tick(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function for tick updates.
        
        The callback will be called with a dictionary containing tick data:
        {
            'symbol': str,
            'token': str,
            'ltp': float,
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': int,
            'bid': float,
            'ask': float,
            'timestamp': str (ISO format),
            'exchange': str
        }
        
        Args:
            callback: Function to call on each tick
            
        Example:
            >>> def handle_tick(tick):
            ...     print(f"{tick['symbol']}: LTP={tick['ltp']}")
            >>> feed.on_tick(handle_tick)
        """
        if callback not in self._tick_callbacks:
            self._tick_callbacks.append(callback)
            logger.debug(f"Registered tick callback: {callback.__name__}")
    
    def on_state_change(self, callback: Callable[[FeedState], None]) -> None:
        """Register a callback for state changes."""
        if callback not in self._state_callbacks:
            self._state_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register a callback for errors."""
        if callback not in self._error_callbacks:
            self._error_callbacks.append(callback)
    
    def _notify_error(self, error: Exception) -> None:
        """Notify all error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for WebSocket connection.
        
        Returns:
            Dictionary of headers
            
        Note:
            TODO: Modify based on your API's authentication requirements.
        """
        headers = {
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': 'CLIENT_LOCAL_IP',
            'X-ClientPublicIP': 'CLIENT_PUBLIC_IP',
            'X-MACAddress': 'MAC_ADDRESS',
            'X-PrivateKey': self.api_key or '',
        }
        
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
        if self.feed_token:
            headers['X-FeedToken'] = self.feed_token
        
        return headers
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection to the market feed.
        
        This method:
        1. Authenticates with the feed server
        2. Establishes WebSocket connection
        3. Starts background tasks for receiving data and heartbeat
        
        Raises:
            ConnectionError: If connection fails after retries
        """
        if self.is_connected:
            logger.warning("Already connected")
            return
        
        self.state = FeedState.CONNECTING
        
        try:
            # Build connection URL with auth params if needed
            # TODO: Modify based on your API's connection method
            connect_url = self.ws_url
            
            # Some APIs require token in URL
            if self.auth_token:
                connect_url = f"{self.ws_url}?token={self.auth_token}"
            
            logger.info(f"Connecting to WebSocket: {connect_url[:50]}...")
            
            # Create WebSocket connection
            self._websocket = await websockets.connect(
                connect_url,
                extra_headers=self._get_auth_headers(),
                ping_interval=self.HEARTBEAT_INTERVAL,
                ping_timeout=10,
                close_timeout=5,
                max_size=10 * 1024 * 1024,  # 10MB max message size
            )
            
            self.state = FeedState.CONNECTED
            self._reconnect_attempts = 0
            self._last_tick_time = time.time()
            
            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            
            logger.info("WebSocket connected successfully")
            
            # Send authentication message if required
            await self._send_auth_message()
            
        except Exception as e:
            self.state = FeedState.ERROR
            logger.error(f"Connection failed: {e}")
            raise ConnectionError(f"Failed to connect: {e}")
    
    async def _send_auth_message(self) -> None:
        """
        Send authentication message after connection.
        
        TODO: Modify based on your API's authentication protocol.
        Some APIs require an auth message after WebSocket connection.
        """
        if not self._websocket:
            return
        
        # Example auth message structure
        # TODO: Replace with your actual auth message format
        auth_message = {
            "action": "auth",
            "api_key": self.api_key,
            "token": self.auth_token,
            "feed_token": self.feed_token,
        }
        
        # Only send if we have credentials
        if self.api_key or self.auth_token:
            try:
                await self._websocket.send(json.dumps(auth_message))
                logger.debug("Sent authentication message")
            except Exception as e:
                logger.warning(f"Failed to send auth message: {e}")
    
    async def disconnect(self) -> None:
        """
        Disconnect from the market feed.
        
        Cleanly closes the WebSocket connection and stops all background tasks.
        """
        logger.info("Disconnecting from market feed...")
        
        self.state = FeedState.DISCONNECTED
        
        # Cancel background tasks
        for task in [self._receive_task, self._heartbeat_task, self._watchdog_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None
        
        # Clear subscriptions
        self._subscribed_symbols.clear()
        
        logger.info("Disconnected from market feed")
    
    async def subscribe(
        self,
        symbols: List[str],
        exchange: str = "NSE",
        mode: str = "FULL"
    ) -> None:
        """
        Subscribe to market data for specified symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            exchange: Exchange code (default: "NSE")
            mode: Subscription mode - "LTP", "QUOTE", or "FULL" (default: "FULL")
                  - LTP: Last traded price only
                  - QUOTE: LTP + OHLC + volume
                  - FULL: All available data including depth
        
        Raises:
            SubscriptionError: If subscription fails
            
        Example:
            >>> await feed.subscribe(["RELIANCE", "TCS", "INFY"])
            >>> await feed.subscribe(["NIFTY 50"], mode="LTP")
        """
        if not self.is_connected:
            raise SubscriptionError("Not connected. Call connect() first.")
        
        if not symbols:
            logger.warning("No symbols provided for subscription")
            return
        
        # Build subscription message
        # TODO: Modify based on your API's subscription format
        # This is a common format used by many Indian brokers
        
        # Mode mapping
        mode_map = {
            "LTP": 1,
            "QUOTE": 2,
            "FULL": 3,
        }
        mode_code = mode_map.get(mode.upper(), 3)
        
        # Build token list
        # TODO: You may need to convert symbols to tokens using a master contract
        tokens = []
        for symbol in symbols:
            # If symbol is already a token (numeric), use it directly
            if symbol.isdigit():
                tokens.append(symbol)
            else:
                # TODO: Look up token from symbol using master contract
                # For now, use symbol as-is (some APIs accept symbol names)
                tokens.append(symbol)
                self._symbol_tokens[symbol] = symbol
        
        # Subscription message format (AngelOne SmartStream format)
        # TODO: Adjust based on your actual API
        subscribe_message = {
            "correlationID": f"sub_{int(time.time() * 1000)}",
            "action": 1,  # 1 = subscribe, 0 = unsubscribe
            "params": {
                "mode": mode_code,
                "tokenList": [
                    {
                        "exchangeType": 1 if exchange == "NSE" else 2,  # 1=NSE, 2=BSE
                        "tokens": tokens
                    }
                ]
            }
        }
        
        try:
            await self._websocket.send(json.dumps(subscribe_message))
            
            # Track subscriptions
            self._subscribed_symbols.update(symbols)
            
            logger.info(f"Subscribed to {len(symbols)} symbols: {symbols[:5]}...")
            
        except Exception as e:
            raise SubscriptionError(f"Failed to subscribe: {e}")
    
    async def unsubscribe(self, symbols: List[str], exchange: str = "NSE") -> None:
        """
        Unsubscribe from market data for specified symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            exchange: Exchange code (default: "NSE")
        """
        if not self.is_connected:
            return
        
        # TODO: Adjust unsubscribe message format
        unsubscribe_message = {
            "correlationID": f"unsub_{int(time.time() * 1000)}",
            "action": 0,  # 0 = unsubscribe
            "params": {
                "mode": 3,
                "tokenList": [
                    {
                        "exchangeType": 1 if exchange == "NSE" else 2,
                        "tokens": symbols
                    }
                ]
            }
        }
        
        try:
            await self._websocket.send(json.dumps(unsubscribe_message))
            
            # Update subscriptions
            self._subscribed_symbols -= set(symbols)
            
            logger.info(f"Unsubscribed from {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
    
    async def _receive_loop(self) -> None:
        """
        Main loop for receiving and processing WebSocket messages.
        
        Runs continuously until disconnected, processing incoming ticks
        and dispatching to registered callbacks.
        """
        logger.debug("Starting receive loop")
        
        while self.is_connected:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=self.FEED_TIMEOUT + 5
                )
                
                # Update last tick time
                self._last_tick_time = time.time()
                
                # Process message
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                # No message received within timeout, check connection
                logger.debug("Receive timeout, checking connection...")
                continue
                
            except ConnectionClosed as e:
                logger.warning(f"WebSocket closed: {e}")
                await self._handle_disconnect()
                break
                
            except WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                await self._handle_disconnect()
                break
                
            except asyncio.CancelledError:
                logger.debug("Receive loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self._notify_error(e)
        
        logger.debug("Receive loop ended")
    
    async def _process_message(self, message: Any) -> None:
        """
        Process a received WebSocket message.
        
        Args:
            message: Raw message (bytes or string)
        """
        try:
            # Parse message
            if isinstance(message, bytes):
                # Binary message - may need special parsing
                # TODO: Implement binary parsing if your feed uses binary format
                data = self._parse_binary_message(message)
            else:
                # JSON message
                data = json.loads(message)
            
            if data is None:
                return
            
            # Handle different message types
            # TODO: Adjust based on your API's message format
            
            # Check for error messages
            if isinstance(data, dict) and data.get('status') == 'error':
                logger.warning(f"Feed error: {data.get('message')}")
                return
            
            # Check for tick data
            if isinstance(data, dict) and 'ltp' in data:
                # Single tick
                tick = self._parse_tick(data)
                if tick:
                    self._dispatch_tick(tick.to_dict())
                    
            elif isinstance(data, list):
                # Multiple ticks
                for item in data:
                    tick = self._parse_tick(item)
                    if tick:
                        self._dispatch_tick(tick.to_dict())
                        
            elif isinstance(data, dict) and 'data' in data:
                # Wrapped response
                tick_data = data.get('data')
                if isinstance(tick_data, list):
                    for item in tick_data:
                        tick = self._parse_tick(item)
                        if tick:
                            self._dispatch_tick(tick.to_dict())
                elif isinstance(tick_data, dict):
                    tick = self._parse_tick(tick_data)
                    if tick:
                        self._dispatch_tick(tick.to_dict())
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON message: {e}")
            # Skip but log invalid messages
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def _parse_binary_message(self, message: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse binary WebSocket message.
        
        Args:
            message: Raw binary message
            
        Returns:
            Parsed data dictionary or None
            
        Note:
            TODO: Implement based on your feed's binary format.
            Many feeds use packed binary format for speed.
        """
        # Placeholder - implement based on your actual binary format
        # Common formats include:
        # - Protobuf
        # - MessagePack
        # - Custom packed structs
        
        try:
            # Try JSON first (some "binary" messages are just JSON)
            return json.loads(message.decode('utf-8'))
        except:
            pass
        
        # TODO: Add your binary parsing logic here
        # Example for packed struct:
        # import struct
        # token, ltp, volume = struct.unpack('>IQQ', message[:20])
        # return {'token': token, 'ltp': ltp / 100, 'volume': volume}
        
        logger.debug(f"Received binary message of {len(message)} bytes")
        return None
    
    def _parse_tick(self, data: Dict[str, Any]) -> Optional[Tick]:
        """
        Parse raw tick data into Tick object.
        
        Args:
            data: Raw tick data dictionary
            
        Returns:
            Tick object or None if parsing fails
        """
        try:
            # Extract fields with various possible key names
            # TODO: Adjust based on your API's tick format
            
            symbol = (
                data.get('symbol') or 
                data.get('tradingsymbol') or 
                data.get('name') or 
                str(data.get('token', ''))
            )
            
            token = str(
                data.get('token') or 
                data.get('symboltoken') or 
                data.get('instrument_token') or 
                symbol
            )
            
            ltp = float(
                data.get('ltp') or 
                data.get('last_price') or 
                data.get('lastPrice') or 
                0
            )
            
            # Handle optional fields
            tick = Tick(
                symbol=symbol,
                token=token,
                ltp=ltp,
                open=float(data.get('open') or data.get('openPrice') or 0),
                high=float(data.get('high') or data.get('highPrice') or 0),
                low=float(data.get('low') or data.get('lowPrice') or 0),
                close=float(data.get('close') or data.get('closePrice') or 0),
                volume=int(data.get('volume') or data.get('tradedVolume') or 0),
                bid=float(data.get('bid') or data.get('bestBid') or 0),
                ask=float(data.get('ask') or data.get('bestAsk') or 0),
                bid_qty=int(data.get('bid_qty') or data.get('bidQty') or 0),
                ask_qty=int(data.get('ask_qty') or data.get('askQty') or 0),
                exchange=data.get('exchange') or data.get('exchangeType') or 'NSE',
            )
            
            # Parse timestamp if available
            timestamp_str = data.get('timestamp') or data.get('exchange_timestamp')
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, (int, float)):
                        tick.timestamp = datetime.fromtimestamp(timestamp_str / 1000)
                    else:
                        tick.timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    pass
            
            return tick
            
        except Exception as e:
            logger.debug(f"Failed to parse tick: {e}")
            return None
    
    def _dispatch_tick(self, tick_data: Dict[str, Any]) -> None:
        """
        Dispatch tick to all registered callbacks.
        
        Args:
            tick_data: Tick data dictionary
        """
        for callback in self._tick_callbacks:
            try:
                callback(tick_data)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """
        Send periodic heartbeat messages to keep connection alive.
        """
        logger.debug("Starting heartbeat loop")
        
        while self.is_connected:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                
                if self._websocket:
                    # Send heartbeat
                    # TODO: Adjust heartbeat message format
                    heartbeat = {"action": "heartbeat", "timestamp": int(time.time() * 1000)}
                    await self._websocket.send(json.dumps(heartbeat))
                    self._last_heartbeat_time = time.time()
                    logger.debug("Heartbeat sent")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
        
        logger.debug("Heartbeat loop ended")
    
    async def _watchdog_loop(self) -> None:
        """
        Monitor feed health and raise warning if feed becomes unresponsive.
        """
        logger.debug("Starting watchdog loop")
        
        while self.is_connected:
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Check time since last tick
                time_since_tick = time.time() - self._last_tick_time
                
                if time_since_tick > self.FEED_TIMEOUT:
                    warning_msg = f"No tick data received for {time_since_tick:.1f} seconds"
                    logger.warning(warning_msg)
                    
                    # Notify error callbacks
                    self._notify_error(FeedTimeoutError(warning_msg))
                    
                    # If really stale (>30 seconds), attempt reconnect
                    if time_since_tick > 30 and self._subscribed_symbols:
                        logger.warning("Feed appears dead, attempting reconnect...")
                        await self._handle_disconnect()
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
        
        logger.debug("Watchdog loop ended")
    
    async def _handle_disconnect(self) -> None:
        """Handle unexpected disconnection with reconnection logic."""
        if self._state == FeedState.RECONNECTING:
            return  # Already reconnecting
        
        self.state = FeedState.RECONNECTING
        
        # Close existing connection
        if self._websocket:
            try:
                await self._websocket.close()
            except:
                pass
            self._websocket = None
        
        # Attempt reconnection with exponential backoff
        while self._reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            
            # Calculate delay with exponential backoff
            delay = min(
                self.INITIAL_RECONNECT_DELAY * (2 ** (self._reconnect_attempts - 1)),
                self.MAX_RECONNECT_DELAY
            )
            
            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS} "
                f"in {delay:.1f}s..."
            )
            
            await asyncio.sleep(delay)
            
            try:
                await self.connect()
                
                # Resubscribe to previously subscribed symbols
                if self._subscribed_symbols:
                    await self.subscribe(list(self._subscribed_symbols))
                
                logger.info("Reconnected successfully")
                return
                
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
        
        # Max attempts reached
        self.state = FeedState.ERROR
        error = ConnectionError(f"Failed to reconnect after {self.MAX_RECONNECT_ATTEMPTS} attempts")
        self._notify_error(error)
        raise error
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        return list(self._subscribed_symbols)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get feed statistics.
        
        Returns:
            Dictionary with feed statistics
        """
        return {
            'state': self._state.value,
            'subscribed_symbols': len(self._subscribed_symbols),
            'callbacks_registered': len(self._tick_callbacks),
            'reconnect_attempts': self._reconnect_attempts,
            'last_tick_age_seconds': time.time() - self._last_tick_time if self._last_tick_time else None,
        }


# =============================================================================
# Synchronous wrapper for non-async code
# =============================================================================

class MarketFeedSync:
    """
    Synchronous wrapper for MarketFeed.
    
    Runs the async feed in a background thread for use in synchronous code.
    
    Example:
        >>> feed = MarketFeedSync()
        >>> feed.on_tick(lambda t: print(f"{t['symbol']}: {t['ltp']}"))
        >>> feed.connect()
        >>> feed.subscribe(["RELIANCE", "TCS"])
        >>> # ... later
        >>> feed.disconnect()
    """
    
    def __init__(self, **kwargs):
        """Initialize with same parameters as MarketFeed."""
        self._feed = MarketFeed(**kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def _run_loop(self):
        """Run event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()
    
    def on_tick(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register tick callback."""
        self._feed.on_tick(callback)
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register error callback."""
        self._feed.on_error(callback)
    
    def connect(self) -> None:
        """Connect to feed (blocking)."""
        if self._running:
            return
        
        # Start background thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        # Wait for loop to start
        while self._loop is None:
            time.sleep(0.01)
        
        # Connect asynchronously
        future = asyncio.run_coroutine_threadsafe(
            self._feed.connect(),
            self._loop
        )
        future.result(timeout=30)  # Wait for connection
        
        self._running = True
    
    def subscribe(self, symbols: List[str], **kwargs) -> None:
        """Subscribe to symbols (blocking)."""
        if not self._running or not self._loop:
            raise RuntimeError("Not connected")
        
        future = asyncio.run_coroutine_threadsafe(
            self._feed.subscribe(symbols, **kwargs),
            self._loop
        )
        future.result(timeout=10)
    
    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols (blocking)."""
        if not self._running or not self._loop:
            return
        
        future = asyncio.run_coroutine_threadsafe(
            self._feed.unsubscribe(symbols),
            self._loop
        )
        future.result(timeout=10)
    
    def disconnect(self) -> None:
        """Disconnect from feed (blocking)."""
        if not self._running or not self._loop:
            return
        
        # Disconnect
        future = asyncio.run_coroutine_threadsafe(
            self._feed.disconnect(),
            self._loop
        )
        future.result(timeout=10)
        
        # Stop loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        
        self._running = False
        self._loop = None
        self._thread = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._running and self._feed.is_connected


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Market Feed Client - Example Usage")
    print("=" * 60)
    
    # Example tick handler
    tick_count = 0
    
    def handle_tick(tick: Dict[str, Any]):
        global tick_count
        tick_count += 1
        print(f"[{tick_count}] {tick['symbol']}: LTP={tick['ltp']}, Vol={tick['volume']}")
    
    def handle_error(error: Exception):
        print(f"ERROR: {error}")
    
    async def main():
        # Create feed
        feed = MarketFeed()
        
        # Register callbacks
        feed.on_tick(handle_tick)
        feed.on_error(handle_error)
        
        try:
            # Connect
            print("\n1. Connecting...")
            await feed.connect()
            print(f"   State: {feed.state.value}")
            
            # Subscribe
            print("\n2. Subscribing to symbols...")
            await feed.subscribe(["RELIANCE", "TCS", "INFY"])
            print(f"   Subscribed: {feed.get_subscribed_symbols()}")
            
            # Run for a bit
            print("\n3. Listening for ticks (10 seconds)...")
            await asyncio.sleep(10)
            
            # Stats
            print(f"\n4. Stats: {feed.get_stats()}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Disconnect
            print("\n5. Disconnecting...")
            await feed.disconnect()
            print("   Done")
    
    # Run example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
