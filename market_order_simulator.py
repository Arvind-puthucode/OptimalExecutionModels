from typing import Dict, List, Tuple, Optional, Any
import time
from orderbook import OrderBook


def simulate_market_order(
    order_book: OrderBook,
    side: str,
    quantity_usd: float,
    price_impact_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Simulate executing a market order by walking the order book.
    
    Args:
        order_book: The current order book state
        side: 'buy' or 'sell'
        quantity_usd: Amount in USD to trade
        price_impact_factor: Multiplier for the price impact
    
    Returns:
        Dictionary with execution details including average price and slippage
    """
    start_time = time.time()
    
    # Get reference prices
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    
    if not best_bid or not best_ask:
        return {
            "success": False,
            "error": "Insufficient liquidity in order book"
        }
    
    mid_price = (best_bid[0] + best_ask[0]) / 2
    
    # Set expected price and book side to walk based on order side
    if side.lower() == 'buy':
        expected_price = best_ask[0]  # Best ask price
        price_levels = order_book.get_asks()  # Walk the asks
    elif side.lower() == 'sell':
        expected_price = best_bid[0]  # Best bid price
        price_levels = order_book.get_bids()  # Walk the bids
    else:
        return {
            "success": False,
            "error": f"Invalid order side: {side}"
        }
    
    # Set initial values
    remaining_usd = quantity_usd
    executed_base_quantity = 0.0  # BTC, ETH, etc.
    total_cost = 0.0  # In USD
    levels_walked = 0
    
    # Walk the book
    execution_detail = []
    for price, quantity in price_levels:
        # Apply price impact factor to make simulation more realistic
        adjusted_price = price * price_impact_factor if side.lower() == 'buy' else price / price_impact_factor
        
        # Calculate how much we can execute at this level
        level_value_usd = quantity * adjusted_price
        executed_usd = min(remaining_usd, level_value_usd)
        executed_quantity = executed_usd / adjusted_price
        
        # Record this execution
        execution_detail.append({
            "price": adjusted_price,
            "quantity": executed_quantity,
            "value_usd": executed_usd
        })
        
        # Update counters
        executed_base_quantity += executed_quantity
        total_cost += executed_usd
        remaining_usd -= executed_usd
        levels_walked += 1
        
        # If order is filled, break
        if remaining_usd <= 0.001:  # Account for small floating-point errors
            remaining_usd = 0
            break
    
    # Calculate results
    if executed_base_quantity == 0:
        return {
            "success": False,
            "error": "No liquidity available to execute order"
        }
    
    avg_execution_price = total_cost / executed_base_quantity
    
    # Calculate slippage as percentage difference from expected price
    slippage_pct = ((avg_execution_price - expected_price) / expected_price) * 100
    if side.lower() == 'sell':
        slippage_pct = -slippage_pct  # For sell orders, negative means worse execution
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return {
        "success": True,
        "side": side,
        "quantity_usd": quantity_usd,
        "filled_usd": total_cost,
        "unfilled_usd": remaining_usd,
        "executed_quantity": executed_base_quantity,
        "avg_execution_price": avg_execution_price,
        "expected_price": expected_price,
        "mid_price": mid_price,
        "slippage_pct": slippage_pct,
        "slippage_usd": abs(avg_execution_price - expected_price) * executed_base_quantity,
        "levels_walked": levels_walked,
        "execution_time_ms": execution_time,
        "execution_detail": execution_detail,
        "is_complete": remaining_usd == 0
    }


def estimate_trading_fee(order_value: float, fee_percentage: float) -> float:
    """
    Calculate the trading fee for an order.
    
    Args:
        order_value: Total value of the order in USD
        fee_percentage: Fee percentage (e.g., 0.0010 for 0.10%)
    
    Returns:
        Fee amount in USD
    """
    return order_value * fee_percentage


def calculate_order_metrics(
    simulation_result: Dict[str, Any],
    fee_percentage: float
) -> Dict[str, Any]:
    """
    Calculate comprehensive order metrics from simulation results.
    
    Args:
        simulation_result: Result from simulate_market_order
        fee_percentage: Fee percentage for the order
        
    Returns:
        Dictionary with all order metrics
    """
    if not simulation_result["success"]:
        return simulation_result
    
    # Extract values from simulation result
    order_value = simulation_result["filled_usd"]
    slippage_usd = simulation_result["slippage_usd"]
    
    # Calculate fee
    fee_usd = estimate_trading_fee(order_value, fee_percentage)
    
    # Calculate total cost including fees
    total_cost = order_value + fee_usd
    
    # Calculate market impact as a percentage of the order value
    market_impact_pct = (slippage_usd / order_value) * 100 if order_value > 0 else 0
    
    # Combine metrics
    return {
        **simulation_result,
        "fee_percentage": fee_percentage,
        "fee_usd": fee_usd,
        "market_impact_pct": market_impact_pct,
        "total_cost": total_cost,
        "net_quantity": simulation_result["executed_quantity"],
        "effective_price": total_cost / simulation_result["executed_quantity"] if simulation_result["executed_quantity"] > 0 else 0
    }


# Example usage
if __name__ == "__main__":
    # Create a sample order book
    from orderbook import OrderBook
    
    # Initialize order book
    book = OrderBook("BTC-USDT")
    
    # Sample data
    sample_data = {
        'bids': [['50000.0', '1.5'], ['49900.0', '2.3'], ['49800.0', '5.0'], ['49700.0', '7.0']],
        'asks': [['50100.0', '1.0'], ['50200.0', '3.2'], ['50300.0', '2.5'], ['50400.0', '4.0']]
    }
    
    # Update the order book
    book.update(sample_data)
    
    # Simulate a buy order
    buy_result = simulate_market_order(book, 'buy', 100000.0)
    
    # Calculate metrics with a 0.10% fee
    buy_metrics = calculate_order_metrics(buy_result, 0.001)
    
    # Print results
    print(f"Buy Order Simulation:")
    print(f"  Quantity: ${buy_metrics['quantity_usd']:.2f}")
    print(f"  Executed: {buy_metrics['executed_quantity']:.6f} BTC")
    print(f"  Avg Price: ${buy_metrics['avg_execution_price']:.2f}")
    print(f"  Expected Price: ${buy_metrics['expected_price']:.2f}")
    print(f"  Slippage: {buy_metrics['slippage_pct']:.4f}%")
    print(f"  Fee: ${buy_metrics['fee_usd']:.2f}")
    print(f"  Total Cost: ${buy_metrics['total_cost']:.2f}")
    print(f"  Levels Walked: {buy_metrics['levels_walked']}")
    
    # Simulate a sell order
    sell_result = simulate_market_order(book, 'sell', 100000.0)
    
    # Calculate metrics with a 0.10% fee
    sell_metrics = calculate_order_metrics(sell_result, 0.001)
    
    # Print results
    print(f"\nSell Order Simulation:")
    print(f"  Quantity: ${sell_metrics['quantity_usd']:.2f}")
    print(f"  Executed: {sell_metrics['executed_quantity']:.6f} BTC")
    print(f"  Avg Price: ${sell_metrics['avg_execution_price']:.2f}")
    print(f"  Expected Price: ${sell_metrics['expected_price']:.2f}")
    print(f"  Slippage: {sell_metrics['slippage_pct']:.4f}%")
    print(f"  Fee: ${sell_metrics['fee_usd']:.2f}")
    print(f"  Total Cost: ${sell_metrics['total_cost']:.2f}")
    print(f"  Levels Walked: {sell_metrics['levels_walked']}") 