"""
Live Mode Executor
==================
Executes orders against the real broker API.

Features:
1. Integration with OpenAlgo API
2. Order status polling
3. Error handling and recovery
4. Audit logging
"""

import time
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any

from .order_manager import Order, OrderStatus, OrderSide, OrderType


logger = logging.getLogger("bot.execution.live_executor")


class LiveExecutor:
    """
    Live trading order executor.
    
    Uses OpenAlgo's REST API to place real orders.
    
    IMPORTANT: This executes REAL trades with REAL money!
    
    Usage:
        executor = LiveExecutor(
            api_key="your-openalgo-api-key",
            base_url="http://127.0.0.1:5000",
        )
        
        result = executor.execute(order)
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://127.0.0.1:5000",
        timeout: int = 30,
        max_poll_attempts: int = 10,
        poll_interval: float = 0.5,
    ):
        """
        Initialize live executor.
        
        Args:
            api_key: OpenAlgo API key
            base_url: OpenAlgo server URL
            timeout: HTTP request timeout
            max_poll_attempts: Max attempts to poll order status
            poll_interval: Seconds between status polls
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_poll_attempts = max_poll_attempts
        self.poll_interval = poll_interval
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
        })
    
    def execute(self, order: Order) -> Order:
        """
        Execute order via OpenAlgo API.
        
        Args:
            order: Order to execute
            
        Returns:
            Order with updated status
        """
        logger.warning(f"[LIVE] Executing REAL order: {order.side.value} {order.quantity} {order.symbol}")
        
        try:
            # Map order to API format
            payload = self._build_payload(order)
            
            # Place order
            response = self._place_order(payload)
            
            if response.get('status') == 'success':
                order.broker_order_id = str(response.get('orderid', ''))
                order.status = OrderStatus.SUBMITTED
                logger.info(f"[LIVE] Order submitted: {order.broker_order_id}")
                
                # Poll for fill status
                self._poll_order_status(order)
            else:
                order.status = OrderStatus.REJECTED
                order.last_error = response.get('message', 'Unknown error')
                logger.error(f"[LIVE] Order rejected: {order.last_error}")
            
        except requests.RequestException as e:
            order.status = OrderStatus.FAILED
            order.last_error = f"API error: {str(e)}"
            logger.error(f"[LIVE] API error: {e}")
        
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.last_error = f"Execution error: {str(e)}"
            logger.error(f"[LIVE] Execution error: {e}")
        
        return order
    
    def __call__(self, order: Order) -> Order:
        """Callable interface for order manager"""
        return self.execute(order)
    
    def _build_payload(self, order: Order) -> Dict[str, Any]:
        """Build API request payload from order"""
        # Map order type
        pricetype = "MARKET"
        if order.order_type == OrderType.LIMIT:
            pricetype = "LIMIT"
        elif order.order_type == OrderType.SL:
            pricetype = "SL"
        elif order.order_type == OrderType.SL_M:
            pricetype = "SL-M"
        
        return {
            'apikey': self.api_key,
            'strategy': order.strategy or 'bot',
            'exchange': order.exchange,
            'symbol': order.symbol,
            'action': order.side.value,
            'product': 'MIS',  # Intraday
            'pricetype': pricetype,
            'quantity': str(order.quantity),
            'price': str(order.price) if order.price > 0 else '0',
            'trigger_price': str(order.trigger_price) if order.trigger_price > 0 else '0',
            'disclosed_quantity': '0',
        }
    
    def _place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Place order via API"""
        url = f"{self.base_url}/api/v1/placeorder"
        
        logger.debug(f"[LIVE] POST {url}")
        
        response = self.session.post(
            url,
            json=payload,
            timeout=self.timeout,
        )
        
        return response.json()
    
    def _poll_order_status(self, order: Order) -> None:
        """Poll for order status until terminal state"""
        for attempt in range(self.max_poll_attempts):
            time.sleep(self.poll_interval)
            
            try:
                status = self._get_order_status(order.broker_order_id)
                
                if status.get('status') == 'complete':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = int(status.get('filled_quantity', order.quantity))
                    order.average_price = float(status.get('average_price', order.price))
                    logger.info(f"[LIVE] Order filled @ ₹{order.average_price}")
                    return
                
                elif status.get('status') in ['rejected', 'cancelled']:
                    order.status = OrderStatus.REJECTED
                    order.last_error = status.get('message', 'Order rejected')
                    return
                
            except Exception as e:
                logger.warning(f"[LIVE] Status poll failed: {e}")
        
        # If we get here, assume order is still open
        order.status = OrderStatus.OPEN
        logger.warning(f"[LIVE] Order status unknown after {self.max_poll_attempts} polls")
    
    def _get_order_status(self, broker_order_id: str) -> Dict[str, Any]:
        """Get order status from API"""
        url = f"{self.base_url}/api/v1/orderstatus"
        
        response = self.session.post(
            url,
            json={
                'apikey': self.api_key,
                'orderid': broker_order_id,
                'strategy': 'bot',
            },
            timeout=self.timeout,
        )
        
        return response.json()
    
    def cancel_order(self, order: Order) -> bool:
        """Cancel an open order"""
        if not order.broker_order_id:
            logger.warning("[LIVE] Cannot cancel order without broker ID")
            return False
        
        try:
            url = f"{self.base_url}/api/v1/cancelorder"
            
            response = self.session.post(
                url,
                json={
                    'apikey': self.api_key,
                    'orderid': order.broker_order_id,
                    'strategy': 'bot',
                },
                timeout=self.timeout,
            )
            
            result = response.json()
            
            if result.get('status') == 'success':
                order.status = OrderStatus.CANCELLED
                logger.info(f"[LIVE] Order cancelled: {order.broker_order_id}")
                return True
            else:
                logger.warning(f"[LIVE] Cancel failed: {result.get('message')}")
                return False
            
        except Exception as e:
            logger.error(f"[LIVE] Cancel error: {e}")
            return False
    
    def modify_order(
        self,
        order: Order,
        new_price: Optional[float] = None,
        new_quantity: Optional[int] = None,
    ) -> bool:
        """Modify an open order"""
        if not order.broker_order_id:
            logger.warning("[LIVE] Cannot modify order without broker ID")
            return False
        
        try:
            url = f"{self.base_url}/api/v1/modifyorder"
            
            payload = {
                'apikey': self.api_key,
                'orderid': order.broker_order_id,
                'strategy': 'bot',
                'exchange': order.exchange,
                'symbol': order.symbol,
            }
            
            if new_price is not None:
                payload['price'] = str(new_price)
            if new_quantity is not None:
                payload['quantity'] = str(new_quantity)
            
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            
            result = response.json()
            
            if result.get('status') == 'success':
                if new_price is not None:
                    order.price = new_price
                if new_quantity is not None:
                    order.quantity = new_quantity
                logger.info(f"[LIVE] Order modified: {order.broker_order_id}")
                return True
            else:
                logger.warning(f"[LIVE] Modify failed: {result.get('message')}")
                return False
            
        except Exception as e:
            logger.error(f"[LIVE] Modify error: {e}")
            return False
    
    def get_positions(self) -> list:
        """Get current positions from broker"""
        try:
            url = f"{self.base_url}/api/v1/positionbook"
            
            response = self.session.post(
                url,
                json={
                    'apikey': self.api_key,
                },
                timeout=self.timeout,
            )
            
            result = response.json()
            
            if result.get('status') == 'success':
                return result.get('data', [])
            else:
                logger.warning(f"[LIVE] Get positions failed: {result.get('message')}")
                return []
            
        except Exception as e:
            logger.error(f"[LIVE] Get positions error: {e}")
            return []
    
    def get_order_book(self) -> list:
        """Get order book from broker"""
        try:
            url = f"{self.base_url}/api/v1/orderbook"
            
            response = self.session.post(
                url,
                json={
                    'apikey': self.api_key,
                },
                timeout=self.timeout,
            )
            
            result = response.json()
            
            if result.get('status') == 'success':
                return result.get('data', [])
            else:
                logger.warning(f"[LIVE] Get order book failed: {result.get('message')}")
                return []
            
        except Exception as e:
            logger.error(f"[LIVE] Get order book error: {e}")
            return []
