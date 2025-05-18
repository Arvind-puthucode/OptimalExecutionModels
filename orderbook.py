from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time


class OrderBook:
    """
    Class to maintain an L2 order book with bids and asks.
    Maintains sorted bids (high to low) and asks (low to high).
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}  # price -> quantity
        self.last_update_id: int = 0
        self.last_update_time: float = 0
    
    def update(self, data: Dict[str, Any]) -> None:
        """
        Update the order book with new data.
        
        Args:
            data: Dictionary containing the order book update with 'bids' and 'asks'
                 as lists of [price_string, quantity_string]
        """
        if 'bids' in data:
            self._update_price_level(data['bids'], self.bids)
            
        if 'asks' in data:
            self._update_price_level(data['asks'], self.asks)
            
        # Update metadata
        if 'update_id' in data:
            self.last_update_id = data['update_id']
        self.last_update_time = time.time()
    
    def _update_price_level(self, price_levels: List[List[str]], book_side: Dict[float, float]) -> None:
        """
        Update a side of the order book (bids or asks).
        
        Args:
            price_levels: List of [price_string, quantity_string] entries
            book_side: The side of the book to update (self.bids or self.asks)
        """
        for price_str, quantity_str in price_levels:
            price = float(price_str)
            quantity = float(quantity_str)
            
            if quantity == 0:
                # Remove the price level if quantity is 0
                if price in book_side:
                    del book_side[price]
            else:
                # Update or add the price level
                book_side[price] = quantity
    
    def get_bids(self, depth: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Get the bids sorted from highest to lowest price.
        
        Args:
            depth: Optional number of price levels to return
            
        Returns:
            List of (price, quantity) tuples sorted by price (high to low)
        """
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        if depth is not None:
            return sorted_bids[:depth]
        return sorted_bids
    
    def get_asks(self, depth: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Get the asks sorted from lowest to highest price.
        
        Args:
            depth: Optional number of price levels to return
            
        Returns:
            List of (price, quantity) tuples sorted by price (low to high)
        """
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
        if depth is not None:
            return sorted_asks[:depth]
        return sorted_asks
    
    def get_spread(self) -> Optional[float]:
        """
        Calculate the spread between the best bid and best ask.
        
        Returns:
            The spread or None if either best bid or best ask is not available
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
            
        return best_ask[0] - best_bid[0]
    
    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """
        Get the highest bid price and quantity.
        
        Returns:
            Tuple of (price, quantity) or None if no bids exist
        """
        if not self.bids:
            return None
        best_price = max(self.bids.keys())
        return (best_price, self.bids[best_price])
    
    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """
        Get the lowest ask price and quantity.
        
        Returns:
            Tuple of (price, quantity) or None if no asks exist
        """
        if not self.asks:
            return None
        best_price = min(self.asks.keys())
        return (best_price, self.asks[best_price])
    
    def get_book_depth(self) -> Dict[str, int]:
        """
        Get the number of price levels on each side of the book.
        
        Returns:
            Dictionary with 'bids' and 'asks' counts
        """
        return {
            'bids': len(self.bids),
            'asks': len(self.asks)
        }
    
    def get_volume_at_price(self, price: float) -> float:
        """
        Get the volume/quantity at a specific price level.
        
        Args:
            price: The price level to check
            
        Returns:
            The quantity at the given price or 0 if the price level doesn't exist
        """
        if price in self.bids:
            return self.bids[price]
        elif price in self.asks:
            return self.asks[price]
        return 0
    
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of the current order book state.
        
        Returns:
            Dictionary containing the current order book state
        """
        return {
            'symbol': self.symbol,
            'bids': self.get_bids(),
            'asks': self.get_asks(),
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'spread': self.get_spread(),
            'depth': self.get_book_depth(),
            'last_update_id': self.last_update_id,
            'last_update_time': self.last_update_time
        }
    
    def get_price_levels(self, num_levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get a specified number of price levels from both sides of the book.
        
        Args:
            num_levels: Number of price levels to return from each side
            
        Returns:
            Dictionary with top bid and ask levels
        """
        return {
            'bids': self.get_bids(num_levels),
            'asks': self.get_asks(num_levels)
        }
    
    def calculate_market_impact(self, side: str, quantity: float) -> Optional[float]:
        """
        Calculate the estimated price impact of a market order.
        
        Args:
            side: 'buy' or 'sell'
            quantity: The quantity to execute
            
        Returns:
            The estimated average execution price or None if not enough liquidity
        """
        remaining = quantity
        total_cost = 0.0
        
        if side.lower() == 'buy':
            price_levels = self.get_asks()
        elif side.lower() == 'sell':
            price_levels = self.get_bids()
        else:
            raise ValueError("Side must be 'buy' or 'sell'")
            
        for price, available_qty in price_levels:
            if remaining <= 0:
                break
                
            executed = min(remaining, available_qty)
            total_cost += executed * price
            remaining -= executed
            
        if remaining > 0:
            # Not enough liquidity
            return None
            
        return total_cost / quantity
    
    def get_depth(self, side: str, depth: Optional[int] = None) -> float:
        """
        Calculate the total quantity available on one side of the book up to a specified depth.
        
        Args:
            side: 'bid' or 'ask'
            depth: Optional number of price levels to include
            
        Returns:
            Total quantity available on the specified side
        """
        if side.lower() in ['bid', 'bids']:
            price_levels = self.get_bids(depth)
        elif side.lower() in ['ask', 'asks']:
            price_levels = self.get_asks(depth)
        else:
            raise ValueError("Side must be 'bid' or 'ask'")
            
        return sum(qty for _, qty in price_levels)


# Example usage
if __name__ == "__main__":
    # Create a new order book
    book = OrderBook("BTC-USDT")
    
    # Sample data (typically would come from the WebSocket client)
    sample_data = {
        'bids': [['50000.0', '1.5'], ['49900.0', '2.3'], ['49800.0', '5.0']],
        'asks': [['50100.0', '1.0'], ['50200.0', '3.2'], ['50300.0', '2.5']]
    }
    
    # Update the order book
    book.update(sample_data)
    
    # Get the best bid and ask
    print(f"Best Bid: {book.get_best_bid()}")
    print(f"Best Ask: {book.get_best_ask()}")
    print(f"Spread: {book.get_spread()}")
    
    # Get the top 2 levels
    levels = book.get_price_levels(2)
    print(f"Top 2 Bids: {levels['bids']}")
    print(f"Top 2 Asks: {levels['asks']}")
    
    # Calculate market impact
    buy_impact = book.calculate_market_impact('buy', 2.0)
    print(f"Market impact of buying 2.0: {buy_impact}") 