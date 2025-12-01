"""
Groww Broker Adapter for OpenAlgo

This adapter provides integration with Groww Trade API including:
- Authentication and session management
- Order placement and management
- Position and holdings retrieval
- Market data (LTP, candles)
- Paper trading simulation with configurable slippage and partial fills

Author: OpenAlgo Team
Date: December 2025
"""

import json
import os
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from utils.httpx_client import get_httpx_client
from utils.logging import get_logger

logger = get_logger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""

    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class TransactionType(Enum):
    """Transaction type enumeration"""

    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    """Product type enumeration"""

    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"
    MTF = "MTF"


class GrowwAdapter:
    """
    Groww Broker Adapter with Paper Trading Simulation

    This adapter interfaces with Groww Trade API and provides paper trading
    capabilities for testing strategies without real money.

    Environment Variables Required:
        GROWW_API_KEY: Groww API key
        GROWW_API_SECRET: Groww API secret
        PAPER_MODE: Enable paper trading (true/false)
        PAPER_SLIPPAGE: Slippage percentage for paper trades (default: 0.1)
        PAPER_PARTIAL_FILL_PROB: Probability of partial fills (default: 0.2)
    """

    # Groww API Base URLs (Placeholder - Update with actual endpoints)
    BASE_URL = "https://api.groww.in/v1"
    AUTH_URL = f"{BASE_URL}/auth"
    ORDERS_URL = f"{BASE_URL}/orders"
    POSITIONS_URL = f"{BASE_URL}/positions"
    HOLDINGS_URL = f"{BASE_URL}/holdings"
    FUNDS_URL = f"{BASE_URL}/funds"
    MARKET_DATA_URL = f"{BASE_URL}/market"

    def __init__(self):
        """Initialize Groww Adapter with configuration from environment"""

        # Load configuration from environment
        self.api_key = os.getenv("GROWW_API_KEY", "")
        self.api_secret = os.getenv("GROWW_API_SECRET", "")
        self.paper_mode = os.getenv("PAPER_MODE", "false").lower() == "true"

        # Paper trading configuration
        self.paper_slippage = float(os.getenv("PAPER_SLIPPAGE", "0.1"))  # 0.1% default
        self.paper_partial_fill_prob = float(
            os.getenv("PAPER_PARTIAL_FILL_PROB", "0.2")
        )  # 20% default
        self.paper_initial_balance = float(
            os.getenv("PAPER_INITIAL_BALANCE", "1000000")
        )  # 10 lakh default

        # Session management
        self.access_token = None
        self.token_expiry = None
        self.session_data = {}

        # Paper trading state
        if self.paper_mode:
            self._initialize_paper_trading()

        # HTTP client with connection pooling
        self.client = get_httpx_client()

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1  # seconds

        logger.info(f"GrowwAdapter initialized - Paper Mode: {self.paper_mode}")

    def _initialize_paper_trading(self):
        """Initialize paper trading state"""
        self.paper_orders = {}
        self.paper_positions = {}
        self.paper_balance = self.paper_initial_balance
        self.paper_used_margin = 0.0
        self.paper_order_counter = 1

        logger.info(
            f"Paper trading initialized with balance: ₹{self.paper_balance:,.2f}"
        )

    # ==================== Authentication ====================

    def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with Groww API and obtain access token

        Returns:
            dict: Authentication response with status and token info

        Example:
            {
                "status": "success",
                "access_token": "eyJhbGc...",
                "expires_in": 3600,
                "message": "Authentication successful"
            }
        """
        if self.paper_mode:
            return self._authenticate_paper()

        try:
            # Prepare authentication payload
            payload = {"api_key": self.api_key, "api_secret": self.api_secret}

            response = self._make_request(
                method="POST", url=self.AUTH_URL, data=payload, auth_required=False
            )

            if response.get("status") == "success":
                self.access_token = response.get("access_token")
                expires_in = response.get("expires_in", 3600)
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

                logger.info("Successfully authenticated with Groww API")
                return {
                    "status": "success",
                    "access_token": self.access_token,
                    "expires_in": expires_in,
                    "message": "Authentication successful",
                }
            else:
                error_msg = response.get("message", "Authentication failed")
                logger.error(f"Authentication failed: {error_msg}")
                return {"status": "error", "message": error_msg}

        except Exception as e:
            logger.exception(f"Authentication error: {e}")
            return {"status": "error", "message": str(e)}

    def _authenticate_paper(self) -> Dict[str, Any]:
        """Paper trading authentication (always succeeds)"""
        self.access_token = "PAPER_TOKEN_" + str(int(time.time()))
        self.token_expiry = datetime.now() + timedelta(hours=24)

        logger.info("Paper trading authentication successful")
        return {
            "status": "success",
            "access_token": self.access_token,
            "expires_in": 86400,
            "message": "Paper trading authentication successful",
        }

    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid token"""
        if not self.access_token:
            return False
        if self.token_expiry and datetime.now() >= self.token_expiry:
            return False
        return True

    # ==================== Balance & Funds ====================

    def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance and margin information

        Returns:
            dict: Balance details with available cash, used margin, etc.

        Example:
            {
                "status": "success",
                "data": {
                    "availablecash": "950000.00",
                    "collateral": "0.00",
                    "m2munrealized": "5000.00",
                    "m2mrealized": "2000.00",
                    "utiliseddebits": "50000.00"
                }
            }
        """
        if self.paper_mode:
            return self._get_balance_paper()

        try:
            response = self._make_request(method="GET", url=self.FUNDS_URL)

            if response.get("status") == "success":
                # Transform Groww API response to standard format
                data = response.get("data", {})
                balance_data = {
                    "availablecash": f"{data.get('available_balance', 0):.2f}",
                    "collateral": f"{data.get('collateral', 0):.2f}",
                    "m2munrealized": f"{data.get('unrealized_pnl', 0):.2f}",
                    "m2mrealized": f"{data.get('realized_pnl', 0):.2f}",
                    "utiliseddebits": f"{data.get('used_margin', 0):.2f}",
                }

                return {"status": "success", "data": balance_data}
            else:
                return {
                    "status": "error",
                    "message": response.get("message", "Failed to fetch balance"),
                }

        except Exception as e:
            logger.exception(f"Error fetching balance: {e}")
            return {"status": "error", "message": str(e)}

    def _get_balance_paper(self) -> Dict[str, Any]:
        """Get paper trading balance"""
        available_cash = self.paper_balance - self.paper_used_margin

        # Calculate unrealized P&L from open positions
        unrealized_pnl = sum(
            pos.get("unrealized_pnl", 0) for pos in self.paper_positions.values()
        )

        balance_data = {
            "availablecash": f"{available_cash:.2f}",
            "collateral": "0.00",
            "m2munrealized": f"{unrealized_pnl:.2f}",
            "m2mrealized": "0.00",
            "utiliseddebits": f"{self.paper_used_margin:.2f}",
        }

        return {"status": "success", "data": balance_data}

    # ==================== Market Data ====================

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Get Last Traded Price for a symbol

        Args:
            symbol: Trading symbol (e.g., 'SBIN', 'NIFTY')
            exchange: Exchange name (NSE, BSE, NFO, etc.)

        Returns:
            dict: LTP data

        Example:
            {
                "status": "success",
                "data": {
                    "symbol": "SBIN",
                    "ltp": 625.50,
                    "exchange": "NSE"
                }
            }
        """
        if self.paper_mode:
            return self._get_ltp_paper(symbol, exchange)

        try:
            endpoint = f"{self.MARKET_DATA_URL}/quote/{exchange}/{symbol}"

            response = self._make_request(method="GET", url=endpoint)

            if response.get("status") == "success":
                data = response.get("data", {})
                return {
                    "status": "success",
                    "data": {
                        "symbol": symbol,
                        "ltp": data.get("last_price", 0),
                        "exchange": exchange,
                        "timestamp": data.get("timestamp"),
                    },
                }
            else:
                return {
                    "status": "error",
                    "message": response.get("message", "Failed to fetch LTP"),
                }

        except Exception as e:
            logger.exception(f"Error fetching LTP for {symbol}: {e}")
            return {"status": "error", "message": str(e)}

    def _get_ltp_paper(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Generate simulated LTP for paper trading"""
        # Simulate realistic price based on symbol
        # In production, you'd use historical data or live market data
        base_price = hash(symbol) % 10000 / 10  # Simple hash-based price
        noise = random.uniform(-5, 5)
        ltp = max(1.0, base_price + noise)

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "ltp": round(ltp, 2),
                "exchange": exchange,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def get_candles(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1minute",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get historical candle data

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Candle interval (1minute, 5minute, 15minute, 1day, etc.)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            dict: Historical candle data

        Example:
            {
                "status": "success",
                "data": {
                    "candles": [
                        [timestamp, open, high, low, close, volume],
                        ...
                    ]
                }
            }
        """
        if self.paper_mode:
            return self._get_candles_paper(symbol, exchange, interval)

        try:
            # Set default dates if not provided
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            endpoint = f"{self.MARKET_DATA_URL}/historical/{exchange}/{symbol}"
            params = {"interval": interval, "from": from_date, "to": to_date}

            response = self._make_request(method="GET", url=endpoint, params=params)

            if response.get("status") == "success":
                return {"status": "success", "data": response.get("data", {})}
            else:
                return {
                    "status": "error",
                    "message": response.get("message", "Failed to fetch candles"),
                }

        except Exception as e:
            logger.exception(f"Error fetching candles for {symbol}: {e}")
            return {"status": "error", "message": str(e)}

    def _get_candles_paper(
        self, symbol: str, exchange: str, interval: str
    ) -> Dict[str, Any]:
        """Generate simulated candle data for paper trading"""
        # Generate simple simulated candles
        candles = []
        base_price = hash(symbol) % 10000 / 10

        for i in range(100):
            timestamp = int((datetime.now() - timedelta(minutes=100 - i)).timestamp())
            open_price = base_price + random.uniform(-10, 10)
            close_price = open_price + random.uniform(-5, 5)
            high_price = max(open_price, close_price) + random.uniform(0, 2)
            low_price = min(open_price, close_price) - random.uniform(0, 2)
            volume = random.randint(1000, 100000)

            candles.append(
                [
                    timestamp,
                    round(open_price, 2),
                    round(high_price, 2),
                    round(low_price, 2),
                    round(close_price, 2),
                    volume,
                ]
            )

        return {"status": "success", "data": {"candles": candles}}

    # ==================== Order Management ====================

    def place_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        product: str = "INTRADAY",
        validity: str = "DAY",
        disclosed_quantity: int = 0,
        trigger_price: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Place a new order

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            transaction_type: BUY or SELL
            quantity: Order quantity
            order_type: MARKET, LIMIT, SL, SL-M
            price: Limit price (for LIMIT orders)
            product: INTRADAY, DELIVERY, MTF
            validity: DAY, IOC
            disclosed_quantity: Disclosed quantity for iceberg orders
            trigger_price: Trigger price for SL orders

        Returns:
            dict: Order placement response with order ID

        Example:
            {
                "status": "success",
                "data": {
                    "order_id": "240101000001",
                    "message": "Order placed successfully"
                }
            }
        """
        if self.paper_mode:
            return self._place_order_paper(
                symbol, exchange, transaction_type, quantity, order_type, price, product
            )

        try:
            # Prepare order payload
            payload = {
                "symbol": symbol,
                "exchange": exchange,
                "transaction_type": transaction_type.upper(),
                "quantity": quantity,
                "order_type": order_type.upper(),
                "product": product.upper(),
                "validity": validity.upper(),
            }

            if order_type.upper() in ["LIMIT", "SL"]:
                payload["price"] = price

            if order_type.upper() in ["SL", "SL-M"]:
                payload["trigger_price"] = trigger_price

            if disclosed_quantity > 0:
                payload["disclosed_quantity"] = disclosed_quantity

            response = self._make_request(
                method="POST", url=self.ORDERS_URL, data=payload
            )

            if response.get("status") == "success":
                order_id = response.get("data", {}).get("order_id")
                logger.info(f"Order placed successfully: {order_id}")
                return {
                    "status": "success",
                    "data": {
                        "order_id": order_id,
                        "message": "Order placed successfully",
                    },
                }
            else:
                error_msg = response.get("message", "Order placement failed")
                logger.error(f"Order placement failed: {error_msg}")
                return {"status": "error", "message": error_msg}

        except Exception as e:
            logger.exception(f"Error placing order: {e}")
            return {"status": "error", "message": str(e)}

    def _place_order_paper(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float,
        product: str,
    ) -> Dict[str, Any]:
        """Place a simulated paper trading order"""

        # Generate order ID
        order_id = (
            f"PAPER{datetime.now().strftime('%y%m%d')}{self.paper_order_counter:06d}"
        )
        self.paper_order_counter += 1

        # Get current LTP
        ltp_response = self._get_ltp_paper(symbol, exchange)
        ltp = ltp_response["data"]["ltp"]

        # Calculate execution price with slippage
        slippage_multiplier = 1 + (self.paper_slippage / 100)
        if transaction_type.upper() == "BUY":
            execution_price = ltp * slippage_multiplier
        else:
            execution_price = ltp / slippage_multiplier

        # Determine if order should be partially filled
        is_partial_fill = random.random() < self.paper_partial_fill_prob
        filled_quantity = quantity
        if is_partial_fill and order_type.upper() == "LIMIT":
            filled_quantity = random.randint(int(quantity * 0.3), quantity)

        # Calculate order value and margin requirement
        order_value = execution_price * filled_quantity
        margin_required = order_value * 0.2 if product == "INTRADAY" else order_value

        # Check if sufficient balance
        available = self.paper_balance - self.paper_used_margin
        if transaction_type.upper() == "BUY" and available < margin_required:
            return {
                "status": "error",
                "message": f"Insufficient funds. Required: ₹{margin_required:.2f}, Available: ₹{available:.2f}",
            }

        # Store order details
        order_status = (
            OrderStatus.COMPLETE.value
            if filled_quantity == quantity
            else OrderStatus.OPEN.value
        )

        self.paper_orders[order_id] = {
            "order_id": order_id,
            "symbol": symbol,
            "exchange": exchange,
            "transaction_type": transaction_type.upper(),
            "quantity": quantity,
            "filled_quantity": filled_quantity,
            "pending_quantity": quantity - filled_quantity,
            "price": execution_price,
            "order_type": order_type.upper(),
            "product": product.upper(),
            "status": order_status,
            "timestamp": datetime.now().isoformat(),
            "message": (
                "Order executed" if filled_quantity == quantity else "Partial fill"
            ),
        }

        # Update positions and margin
        if filled_quantity > 0:
            self._update_paper_position(
                symbol,
                exchange,
                transaction_type,
                filled_quantity,
                execution_price,
                product,
            )

        logger.info(
            f"Paper order placed: {order_id} - {symbol} {transaction_type} {filled_quantity}/{quantity}"
        )

        return {
            "status": "success",
            "data": {
                "order_id": order_id,
                "message": f"Paper order placed - {order_status}",
                "filled_quantity": filled_quantity,
                "execution_price": round(execution_price, 2),
            },
        }

    def _update_paper_position(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        product: str,
    ):
        """Update paper trading positions"""
        position_key = f"{symbol}_{exchange}_{product}"

        if position_key not in self.paper_positions:
            self.paper_positions[position_key] = {
                "symbol": symbol,
                "exchange": exchange,
                "product": product,
                "quantity": 0,
                "average_price": 0.0,
                "unrealized_pnl": 0.0,
            }

        position = self.paper_positions[position_key]

        if transaction_type.upper() == "BUY":
            total_cost = (position["quantity"] * position["average_price"]) + (
                quantity * price
            )
            position["quantity"] += quantity
            position["average_price"] = (
                total_cost / position["quantity"] if position["quantity"] > 0 else 0
            )

            # Update used margin
            margin_value = quantity * price * (0.2 if product == "INTRADAY" else 1.0)
            self.paper_used_margin += margin_value

        else:  # SELL
            position["quantity"] -= quantity

            # Release margin
            margin_value = quantity * price * (0.2 if product == "INTRADAY" else 1.0)
            self.paper_used_margin = max(0, self.paper_used_margin - margin_value)

            # Calculate realized P&L for sell
            realized_pnl = (price - position["average_price"]) * quantity
            self.paper_balance += realized_pnl

        # Remove position if quantity is zero
        if position["quantity"] == 0:
            del self.paper_positions[position_key]

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order

        Args:
            order_id: Order ID to cancel

        Returns:
            dict: Cancellation response

        Example:
            {
                "status": "success",
                "data": {
                    "order_id": "240101000001",
                    "message": "Order cancelled successfully"
                }
            }
        """
        if self.paper_mode:
            return self._cancel_order_paper(order_id)

        try:
            endpoint = f"{self.ORDERS_URL}/{order_id}"

            response = self._make_request(method="DELETE", url=endpoint)

            if response.get("status") == "success":
                logger.info(f"Order cancelled successfully: {order_id}")
                return {
                    "status": "success",
                    "data": {
                        "order_id": order_id,
                        "message": "Order cancelled successfully",
                    },
                }
            else:
                return {
                    "status": "error",
                    "message": response.get("message", "Order cancellation failed"),
                }

        except Exception as e:
            logger.exception(f"Error cancelling order {order_id}: {e}")
            return {"status": "error", "message": str(e)}

    def _cancel_order_paper(self, order_id: str) -> Dict[str, Any]:
        """Cancel a paper trading order"""
        if order_id not in self.paper_orders:
            return {"status": "error", "message": f"Order not found: {order_id}"}

        order = self.paper_orders[order_id]

        if order["status"] in [OrderStatus.COMPLETE.value, OrderStatus.CANCELLED.value]:
            return {
                "status": "error",
                "message": f"Cannot cancel order with status: {order['status']}",
            }

        order["status"] = OrderStatus.CANCELLED.value
        order["message"] = "Order cancelled"

        logger.info(f"Paper order cancelled: {order_id}")

        return {
            "status": "success",
            "data": {
                "order_id": order_id,
                "message": "Paper order cancelled successfully",
            },
        }

    # ==================== Positions & Holdings ====================

    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions

        Returns:
            dict: Positions data

        Example:
            {
                "status": "success",
                "data": [
                    {
                        "symbol": "SBIN",
                        "exchange": "NSE",
                        "product": "INTRADAY",
                        "quantity": 100,
                        "average_price": 625.50,
                        "ltp": 627.00,
                        "pnl": 150.00
                    }
                ]
            }
        """
        if self.paper_mode:
            return self._get_positions_paper()

        try:
            response = self._make_request(method="GET", url=self.POSITIONS_URL)

            if response.get("status") == "success":
                return {"status": "success", "data": response.get("data", [])}
            else:
                return {
                    "status": "error",
                    "message": response.get("message", "Failed to fetch positions"),
                }

        except Exception as e:
            logger.exception(f"Error fetching positions: {e}")
            return {"status": "error", "message": str(e)}

    def _get_positions_paper(self) -> Dict[str, Any]:
        """Get paper trading positions"""
        positions_list = []

        for position in self.paper_positions.values():
            # Get current LTP for unrealized P&L
            ltp_response = self._get_ltp_paper(position["symbol"], position["exchange"])
            ltp = ltp_response["data"]["ltp"]

            unrealized_pnl = (ltp - position["average_price"]) * position["quantity"]

            positions_list.append(
                {
                    "symbol": position["symbol"],
                    "exchange": position["exchange"],
                    "product": position["product"],
                    "quantity": position["quantity"],
                    "average_price": round(position["average_price"], 2),
                    "ltp": ltp,
                    "pnl": round(unrealized_pnl, 2),
                }
            )

        return {"status": "success", "data": positions_list}

    # ==================== HTTP Helper Methods ====================

    def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        auth_required: bool = True,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Groww API with retry logic

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            url: Full URL endpoint
            data: Request payload
            params: URL parameters
            auth_required: Whether authentication is required

        Returns:
            dict: API response
        """
        # Check authentication if required
        if auth_required and not self.is_authenticated():
            logger.warning("Not authenticated, attempting to authenticate...")
            auth_result = self.authenticate()
            if auth_result.get("status") != "success":
                return {"status": "error", "message": "Authentication required"}

        # Prepare headers
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if auth_required and self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.client.get(url, headers=headers, params=params)
                elif method.upper() == "POST":
                    response = self.client.post(url, headers=headers, json=data)
                elif method.upper() == "DELETE":
                    response = self.client.delete(url, headers=headers)
                else:
                    response = self.client.request(
                        method, url, headers=headers, json=data
                    )

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {e}"
                )

                if e.response.status_code == 401:
                    # Token expired, re-authenticate
                    logger.info("Token expired, re-authenticating...")
                    self.access_token = None
                    self.authenticate()
                    continue

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return {
                        "status": "error",
                        "message": f"HTTP {e.response.status_code}: {str(e)}",
                    }

            except Exception as e:
                logger.warning(
                    f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return {"status": "error", "message": str(e)}

        return {"status": "error", "message": "Max retries exceeded"}

    # ==================== Utility Methods ====================

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of a specific order

        Args:
            order_id: Order ID to check

        Returns:
            dict: Order status details
        """
        if self.paper_mode:
            if order_id in self.paper_orders:
                return {"status": "success", "data": self.paper_orders[order_id]}
            else:
                return {"status": "error", "message": "Order not found"}

        try:
            endpoint = f"{self.ORDERS_URL}/{order_id}"
            response = self._make_request(method="GET", url=endpoint)
            return response
        except Exception as e:
            logger.exception(f"Error fetching order status: {e}")
            return {"status": "error", "message": str(e)}

    def get_paper_trading_summary(self) -> Dict[str, Any]:
        """
        Get paper trading summary (only available in paper mode)

        Returns:
            dict: Paper trading statistics
        """
        if not self.paper_mode:
            return {"status": "error", "message": "Paper trading not enabled"}

        total_pnl = sum(
            pos.get("unrealized_pnl", 0) for pos in self.paper_positions.values()
        )

        return {
            "status": "success",
            "data": {
                "initial_balance": self.paper_initial_balance,
                "current_balance": self.paper_balance,
                "used_margin": self.paper_used_margin,
                "available_balance": self.paper_balance - self.paper_used_margin,
                "total_unrealized_pnl": round(total_pnl, 2),
                "total_orders": len(self.paper_orders),
                "open_positions": len(self.paper_positions),
                "net_pnl": round(
                    self.paper_balance - self.paper_initial_balance + total_pnl, 2
                ),
            },
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    """
    Example usage of GrowwAdapter

    Set environment variables before running:
        export GROWW_API_KEY="your_api_key"
        export GROWW_API_SECRET="your_api_secret"
        export PAPER_MODE="true"
    """

    # Initialize adapter
    adapter = GrowwAdapter()

    # Authenticate
    auth_result = adapter.authenticate()
    print("Authentication:", auth_result)

    # Get balance
    balance = adapter.get_balance()
    print("\nBalance:", json.dumps(balance, indent=2))

    # Get LTP
    ltp = adapter.get_ltp("SBIN", "NSE")
    print("\nLTP:", json.dumps(ltp, indent=2))

    # Place order
    order_result = adapter.place_order(
        symbol="SBIN",
        exchange="NSE",
        transaction_type="BUY",
        quantity=10,
        order_type="MARKET",
        product="INTRADAY",
    )
    print("\nOrder Result:", json.dumps(order_result, indent=2))

    # Get positions
    positions = adapter.get_positions()
    print("\nPositions:", json.dumps(positions, indent=2))

    # Paper trading summary
    if adapter.paper_mode:
        summary = adapter.get_paper_trading_summary()
        print("\nPaper Trading Summary:", json.dumps(summary, indent=2))
