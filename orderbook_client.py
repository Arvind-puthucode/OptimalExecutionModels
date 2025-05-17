import json
import asyncio
import logging
import websockets
from websockets.exceptions import ConnectionClosed
import time
from typing import Dict, List, Callable, Optional, Any


class OrderBookClient:
    """
    WebSocket client that connects to an order book data stream,
    processes the incoming messages, and maintains the current order book state.
    """
    
    def __init__(self, url: str, max_reconnect_attempts: int = 10, reconnect_delay: int = 5):
        self.url = url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.reconnect_attempts = 0
        self.running = False
        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Order book state
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}  # price -> size
        self.last_update_id = 0
        self.last_update_time = 0
        
        # Setup logging
        self.logger = logging.getLogger('OrderBookClient')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback function that will be called with each order book update."""
        self.callbacks.append(callback)
    
    def get_order_book_snapshot(self) -> Dict[str, Any]:
        """Get the current order book state."""
        return {
            'bids': sorted(([price, size] for price, size in self.bids.items()), reverse=True),
            'asks': sorted(([price, size] for price, size in self.asks.items())),
            'last_update_id': self.last_update_id,
            'last_update_time': self.last_update_time
        }
    
    def get_best_bid_ask(self) -> Dict[str, float]:
        """Get the best bid and ask prices."""
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
    
    def _process_orderbook_update(self, data: Dict[str, Any]) -> None:
        """Process an order book update."""
        try:
            # Update order book state based on the message format
            # Note: This implementation assumes a specific message format
            # You may need to adjust this based on the actual data structure

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
            
            # Call registered callbacks
            for callback in self.callbacks:
                try:
                    callback(self.get_order_book_snapshot())
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing order book update: {e}")
    
    async def connect(self) -> None:
        """Connect to the WebSocket and start processing messages."""
        self.running = True
        self.reconnect_attempts = 0
        
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.logger.info(f"Connecting to {self.url}")
                async with websockets.connect(self.url) as websocket:
                    self.connection = websocket
                    self.reconnect_attempts = 0  # Reset on successful connection
                    self.logger.info("Connected to the order book stream")
                    
                    # Process messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            self._process_orderbook_update(data)
                        except json.JSONDecodeError:
                            self.logger.error(f"Failed to parse message: {message}")
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
            
            except ConnectionClosed as e:
                self.logger.warning(f"WebSocket connection closed: {e}")
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
            
            # Reconnection logic
            if self.running:
                self.reconnect_attempts += 1
                wait_time = self.reconnect_delay * self.reconnect_attempts
                self.logger.info(f"Reconnecting in {wait_time} seconds (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                await asyncio.sleep(wait_time)
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached. Giving up.")
            self.running = False
    
    def disconnect(self) -> None:
        """Stop the client and close the WebSocket connection."""
        self.logger.info("Disconnecting from the order book stream")
        self.running = False
        if self.connection:
            asyncio.create_task(self.connection.close())


async def main():
    """Example usage of the OrderBookClient."""
    # Initialize the client
    url = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
    client = OrderBookClient(url)
    
    # Define a callback to handle order book updates
    def on_orderbook_update(order_book):
        best = client.get_best_bid_ask()
        print(f"Best Bid: {best['best_bid']}, Best Ask: {best['best_ask']}, Spread: {best['spread']}")
    
    # Register the callback
    client.add_callback(on_orderbook_update)
    
    # Start the client in a separate task
    client_task = asyncio.create_task(client.connect())
    
    try:
        # Run for a while
        await asyncio.sleep(60)
    finally:
        # Clean up
        client.disconnect()
        await client_task


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 