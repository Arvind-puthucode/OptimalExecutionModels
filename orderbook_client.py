import json
import asyncio
import logging
import time
import threading
import requests
from typing import Dict, List, Callable, Optional, Any, Tuple

# Try to import websockets, but provide a fallback if not available
try:
    import websockets  # type: ignore
    from websockets.exceptions import ConnectionClosed  # type: ignore
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets package not available. Will use HTTP polling as fallback.")


class OrderBookClient:
    """
    Client that connects to order book data stream.
    Provides both WebSocket and fallback HTTP polling methods.
    """
    
    # Singleton instance
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OrderBookClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, websocket_url=None, http_url=None):
        # Only initialize once
        if OrderBookClient._initialized:
            return
        
        # URLs
        self.websocket_url = websocket_url or "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
        self.http_url = http_url or "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=1000"
        
        # State
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}  # price -> size
        self.last_update_id = 0
        self.last_update_time = 0
        
        # HTTP polling
        self.use_websocket = WEBSOCKETS_AVAILABLE
        self.polling_interval = 5  # seconds
        self.polling_thread = None
        self.keep_polling = False
        
        # Callbacks
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Setup logging
        self.logger = logging.getLogger('OrderBookClient')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Lock for thread safety when updating order book
        self.order_book_lock = threading.Lock()
        
        OrderBookClient._initialized = True
    
    @classmethod
    def get_instance(cls, websocket_url=None, http_url=None):
        """
        Get or create the singleton instance of the OrderBookClient.
        """
        if cls._instance is None:
            cls(websocket_url, http_url)
        return cls._instance
    
    def reset(self):
        """Reset the client state without destroying the singleton instance."""
        with self.order_book_lock:
            self.callbacks = []
            self.bids = {}
            self.asks = {}
            self.last_update_id = 0
            self.last_update_time = 0
        
        # Stop polling if active
        if self.polling_thread and self.polling_thread.is_alive():
            self.keep_polling = False
            self.polling_thread.join(timeout=1.0)
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback function that will be called with each order book update."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a registered callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_order_book_snapshot(self) -> Dict[str, Any]:
        """Get the current order book state."""
        with self.order_book_lock:
            return {
                'bids': sorted(([price, size] for price, size in self.bids.items()), reverse=True),
                'asks': sorted(([price, size] for price, size in self.asks.items())),
                'last_update_id': self.last_update_id,
                'last_update_time': self.last_update_time
            }
    
    def get_best_bid_ask(self) -> Dict[str, float]:
        """Get the best bid and ask prices."""
        with self.order_book_lock:
            best_bid = max(self.bids.keys()) if self.bids else None
            best_ask = min(self.asks.keys()) if self.asks else None
            best_bid_size = self.bids.get(best_bid, 0) if best_bid else 0
            best_ask_size = self.asks.get(best_ask, 0) if best_ask else 0
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'best_bid_size': best_bid_size,
                'best_ask_size': best_ask_size,
                'spread': best_ask - best_bid if best_bid and best_ask else None
            }
    
    def get_bids_asks(self) -> Tuple[Dict[float, float], Dict[float, float]]:
        """Get the current bids and asks."""
        with self.order_book_lock:
            return self.bids.copy(), self.asks.copy()
    
    def _process_orderbook_update(self, data: Dict[str, Any]) -> None:
        """Process an order book update."""
        try:
            # Update order book state in a thread-safe way
            with self.order_book_lock:
                # Update order book state
                if 'bids' in data:
                    for price_str, size_str in data['bids']:
                        price = float(price_str)
                        size = float(size_str)
                        if size == 0:
                            if price in self.bids:
                                del self.bids[price]
                        else:
                            self.bids[price] = size
                
                if 'asks' in data:
                    for price_str, size_str in data['asks']:
                        price = float(price_str)
                        size = float(size_str)
                        if size == 0:
                            if price in self.asks:
                                del self.asks[price]
                        else:
                            self.asks[price] = size
                
                # Update metadata
                if 'update_id' in data:
                    self.last_update_id = data['update_id']
                self.last_update_time = time.time()
            
            # Get snapshot for callbacks
            snapshot = self.get_order_book_snapshot()
            
            # Call registered callbacks with the snapshot
            for callback in self.callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing order book update: {e}")
    
    def _poll_http_endpoint(self):
        """Poll HTTP endpoint for order book data."""
        self.keep_polling = True
        
        while self.keep_polling:
            try:
                # Make HTTP request to get order book data
                response = requests.get(self.http_url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Convert to format expected by _process_orderbook_update
                    processed_data = {
                        'bids': [[str(x[0]), str(x[1])] for x in data.get('bids', [])[:100]],
                        'asks': [[str(x[0]), str(x[1])] for x in data.get('asks', [])[:100]],
                        'update_id': data.get('lastUpdateId', 0)
                    }
                    
                    # Process the data
                    self._process_orderbook_update(processed_data)
                    self.logger.info(f"Polled order book: {len(processed_data['bids'])} bids, {len(processed_data['asks'])} asks")
                else:
                    self.logger.warning(f"HTTP polling failed: {response.status_code}")
            
            except Exception as e:
                self.logger.error(f"Error polling HTTP endpoint: {e}")
            
            # Sleep before next poll
            time.sleep(self.polling_interval)
    
    def start_polling(self):
        """Start HTTP polling for order book data."""
        if self.polling_thread and self.polling_thread.is_alive():
            self.logger.info("Polling already active")
            return True
            
        try:
            # Start polling in a background thread
            self.polling_thread = threading.Thread(
                target=self._poll_http_endpoint,
                daemon=True,
                name="OrderBookPoller"
            )
            self.polling_thread.start()
            self.logger.info("Started HTTP polling for order book data")
            return True
        except Exception as e:
            self.logger.error(f"Error starting polling: {e}")
            return False
    
    def stop_polling(self):
        """Stop HTTP polling."""
        if self.polling_thread and self.polling_thread.is_alive():
            self.keep_polling = False
            self.logger.info("Stopping HTTP polling")
    
    def has_data(self):
        """Check if the order book has data."""
        with self.order_book_lock:
            return len(self.bids) > 0 and len(self.asks) > 0
    
    def is_connected(self):
        """Check if client is receiving order book updates."""
        # If we've received an update in the last 10 seconds and have data, consider connected
        return (time.time() - self.last_update_time) < 10 and self.has_data()
    
    def connect(self):
        """
        Connect to the order book data source.
        Uses HTTP polling as it's more reliable with Streamlit.
        """
        return self.start_polling()
        
    def disconnect(self):
        """Disconnect from the data source."""
        self.stop_polling()


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create client instance
    client = OrderBookClient.get_instance()
    
    # Define callback
    def on_update(data):
        best = client.get_best_bid_ask()
        print(f"Best Bid: {best['best_bid']}, Best Ask: {best['best_ask']}, Spread: {best['spread']}")
    
    # Register callback
    client.add_callback(on_update)
    
    # Connect to data source
    client.connect()
    
    try:
        # Run for 30 seconds
        time.sleep(30)
    finally:
        # Disconnect
        client.disconnect()
