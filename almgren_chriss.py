"""
Implementation of the Almgren-Chriss market impact model.

This module provides functions to calculate market impact using the Almgren-Chriss model,
which considers both temporary and permanent price impacts of trades.

The Almgren-Chriss model divides price impact into two components:
1. Temporary impact: Immediate price changes that disappear after the trade (price bounces back)
2. Permanent impact: Lasting price changes that persist after the trade

The key parameters in the model are:
- σ (sigma): Market volatility
- γ (gamma): Temporary impact factor
- η (eta): Permanent impact factor

The total price impact is calculated as: (η + 0.5γ) * X
Where X is the order size.

References:
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
  Journal of Risk, 3, 5-40.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
from orderbook import OrderBook


def calculate_market_depth(order_book: OrderBook, depth_levels: int = 5) -> Dict[str, float]:
    """
    Calculate market depth metrics from the order book.
    
    Args:
        order_book: Current order book state
        depth_levels: Number of levels to consider for depth calculation
        
    Returns:
        Dictionary with market depth metrics
    """
    bids = order_book.get_bids(depth_levels)
    asks = order_book.get_asks(depth_levels)
    
    # Calculate total volume at each side
    bid_volume = sum(qty for _, qty in bids)
    ask_volume = sum(qty for _, qty in asks)
    
    # Calculate weighted average price
    bid_value = sum(price * qty for price, qty in bids)
    ask_value = sum(price * qty for price, qty in asks)
    
    bid_vwap = bid_value / bid_volume if bid_volume > 0 else 0
    ask_vwap = ask_value / ask_volume if ask_volume > 0 else 0
    
    # Get best bid and ask
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    
    if not best_bid or not best_ask:
        return {
            "bid_volume": 0,
            "ask_volume": 0,
            "bid_vwap": 0,
            "ask_vwap": 0,
            "mid_price": 0,
            "spread": 0,
            "depth_imbalance": 0,
            "relative_depth": 0
        }
    
    mid_price = (best_bid[0] + best_ask[0]) / 2
    spread = best_ask[0] - best_bid[0]
    
    # Calculate depth imbalance (negative means more selling pressure, positive means more buying pressure)
    depth_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
    
    # Calculate relative depth (average volume per unit of price)
    price_range_bids = max(bid[0] for bid in bids) - min(bid[0] for bid in bids) if bids else 0
    price_range_asks = max(ask[0] for ask in asks) - min(ask[0] for ask in asks) if asks else 0
    
    relative_depth = ((bid_volume / price_range_bids if price_range_bids > 0 else 0) + 
                     (ask_volume / price_range_asks if price_range_asks > 0 else 0)) / 2
    
    return {
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "bid_vwap": bid_vwap,
        "ask_vwap": ask_vwap,
        "mid_price": mid_price,
        "spread": spread,
        "depth_imbalance": depth_imbalance,
        "relative_depth": relative_depth
    }


def estimate_temporary_impact(depth: Dict[str, float], volatility: float, 
                              market_cap: Optional[float] = None) -> float:
    """
    Estimate temporary impact factor (gamma) based on market depth and volatility.
    
    Args:
        depth: Market depth metrics from calculate_market_depth
        volatility: Market volatility factor (annualized)
        market_cap: Optional market capitalization for scaling
        
    Returns:
        Temporary impact factor (gamma)
    """
    # Base impact factor derived from spread and depth
    base_factor = depth["spread"] / (depth["bid_volume"] + depth["ask_volume"]) * 2
    
    # Adjust for volatility (higher volatility = higher impact)
    volatility_adjustment = math.sqrt(volatility)
    
    # Adjust for market cap if available (lower cap = higher impact)
    cap_adjustment = 1.0
    if market_cap is not None and market_cap > 0:
        cap_adjustment = math.pow(10, 12) / market_cap
        cap_adjustment = min(cap_adjustment, 5.0)  # Cap the adjustment
    
    # Depth imbalance affects impact (more imbalance = higher impact)
    imbalance_factor = 1 + abs(depth["depth_imbalance"])
    
    # Combine factors
    gamma = base_factor * volatility_adjustment * cap_adjustment * imbalance_factor
    
    # Ensure reasonable bounds
    gamma = max(gamma, 1e-6)  # Minimum impact
    gamma = min(gamma, 0.001)  # Maximum impact (0.1% per unit)
    
    return gamma


def estimate_permanent_impact(depth: Dict[str, float], market_resilience: float = 0.1) -> float:
    """
    Estimate permanent impact factor (eta) based on market depth and resilience.
    
    Args:
        depth: Market depth metrics from calculate_market_depth
        market_resilience: How quickly the market recovers from impact (0-1)
        
    Returns:
        Permanent impact factor (eta)
    """
    # In Almgren-Chriss model, permanent impact is typically smaller than temporary
    # and depends on how quickly the market recovers
    
    # Calculate base impact from spread and relative depth
    base_impact = depth["spread"] / depth["relative_depth"] if depth["relative_depth"] > 0 else 0
    
    # Adjust for market resilience (higher resilience = lower permanent impact)
    resilience_factor = 1 - market_resilience
    
    # Permanent impact is typically a fraction of the temporary impact
    eta = base_impact * resilience_factor * 0.3  # 30% of base impact
    
    # Ensure reasonable bounds
    eta = max(eta, 1e-7)  # Minimum impact
    eta = min(eta, 0.0005)  # Maximum impact (0.05% per unit)
    
    return eta


def calculate_almgren_chriss_impact(
    order_book: OrderBook, 
    side: str, 
    quantity: float, 
    volatility: float,
    market_resilience: float = 0.1,
    market_cap: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate market impact using Almgren-Chriss model.
    
    The Almgren-Chriss model splits market impact into temporary and permanent components:
    - Temporary impact: Immediate price reaction that disappears after the trade
    - Permanent impact: Lasting price change that persists in the market
    
    The model uses the formula: Total Impact = permanent_impact + 0.5 * temporary_impact
    
    Parameters:
    - order_book: Current order book state
    - side: 'buy' or 'sell'
    - quantity: Order size in base currency
    - volatility: Market volatility factor (annualized, e.g., 0.3 for 30%)
    - market_resilience: How quickly the market recovers (0-1)
    - market_cap: Optional market capitalization for scaling
    
    Returns:
    - Dictionary with impact components and parameters
    """
    # Calculate market depth
    depth = calculate_market_depth(order_book)
    
    # Get mid price
    mid_price = depth["mid_price"]
    
    # Estimate temporary impact factor (gamma)
    gamma = estimate_temporary_impact(depth, volatility, market_cap)
    
    # Estimate permanent impact factor (eta)
    eta = estimate_permanent_impact(depth, market_resilience)
    
    # Calculate impacts
    # For sell orders, impact is negative (price decreases)
    direction = 1 if side.lower() == 'buy' else -1
    
    temporary_impact = direction * gamma * quantity
    permanent_impact = direction * eta * quantity
    
    # Calculate total impact using Almgren-Chriss formula
    total_impact = permanent_impact + 0.5 * temporary_impact
    
    # Calculate impact in price units
    impact_price = mid_price * (1 + total_impact)
    
    # Calculate price improvement/slippage
    if side.lower() == 'buy':
        expected_price = order_book.get_best_ask()[0] if order_book.get_best_ask() else mid_price
        slippage = impact_price - expected_price
    else:  # sell
        expected_price = order_book.get_best_bid()[0] if order_book.get_best_bid() else mid_price
        slippage = expected_price - impact_price
    
    return {
        "mid_price": mid_price,
        "gamma": gamma,
        "eta": eta,
        "temporary_impact": temporary_impact,
        "permanent_impact": permanent_impact,
        "total_impact": total_impact,
        "expected_price": expected_price,
        "impact_price": impact_price,
        "slippage": slippage,
        "slippage_bps": (slippage / expected_price) * 10000 if expected_price > 0 else 0,
        "side": side,
        "quantity": quantity,
        "volatility": volatility,
        "market_resilience": market_resilience
    }


def simulate_market_order_with_impact(
    order_book: OrderBook,
    side: str,
    quantity_usd: float,
    volatility: float,
    market_resilience: float = 0.1,
    market_cap: Optional[float] = None,
    fee_percentage: float = 0.001
) -> Dict[str, Any]:
    """
    Simulate a market order execution with Almgren-Chriss impact model.
    
    Args:
        order_book: Current order book state
        side: 'buy' or 'sell'
        quantity_usd: Amount in USD to trade
        volatility: Market volatility (annualized)
        market_resilience: Market recovery rate (0-1)
        market_cap: Optional market capitalization
        fee_percentage: Trading fee percentage
        
    Returns:
        Dictionary with execution details including impact metrics
    """
    # Get reference prices
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    
    if not best_bid or not best_ask:
        return {
            "success": False,
            "error": "Insufficient liquidity in order book"
        }
    
    mid_price = (best_bid[0] + best_ask[0]) / 2
    
    # Estimate base quantity (approximate conversion from USD to base currency)
    if side.lower() == 'buy':
        base_price = best_ask[0]
    else:
        base_price = best_bid[0]
    
    estimated_quantity = quantity_usd / base_price
    
    # Calculate market impact using Almgren-Chriss model
    impact_result = calculate_almgren_chriss_impact(
        order_book, 
        side, 
        estimated_quantity, 
        volatility,
        market_resilience,
        market_cap
    )
    
    # Adjust price based on impact
    adjusted_price = impact_result["impact_price"]
    
    # Calculate executed quantity and cost
    executed_quantity = quantity_usd / adjusted_price
    total_cost = quantity_usd
    
    # Calculate fee
    fee_usd = total_cost * fee_percentage
    
    # Prepare result dictionary
    result = {
        "success": True,
        "side": side,
        "quantity_usd": quantity_usd,
        "filled_usd": total_cost,
        "unfilled_usd": 0,  # Assuming full fill
        "executed_quantity": executed_quantity,
        "avg_execution_price": adjusted_price,
        "expected_price": impact_result["expected_price"],
        "mid_price": mid_price,
        "slippage_pct": (impact_result["slippage"] / impact_result["expected_price"]) * 100 if impact_result["expected_price"] > 0 else 0,
        "slippage_usd": impact_result["slippage"] * executed_quantity,
        "fee_percentage": fee_percentage,
        "fee_usd": fee_usd,
        "total_cost": total_cost + fee_usd if side.lower() == 'buy' else total_cost - fee_usd,
        "is_complete": True,
        "impact_model": "almgren_chriss",
        "impact_metrics": impact_result
    }
    
    return result


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
    
    # Define parameters
    volatility = 0.5  # 50% annualized volatility
    market_resilience = 0.2  # Market recovery rate
    market_cap = 1e9  # $1 billion market cap
    
    # Calculate market impact for a buy order of 2 BTC
    impact = calculate_almgren_chriss_impact(book, 'buy', 2.0, volatility, market_resilience, market_cap)
    
    # Print impact details
    print(f"Almgren-Chriss Market Impact:")
    print(f"  Temporary Impact Factor (gamma): {impact['gamma']:.8f}")
    print(f"  Permanent Impact Factor (eta): {impact['eta']:.8f}")
    print(f"  Temporary Impact: {impact['temporary_impact']:.6f}")
    print(f"  Permanent Impact: {impact['permanent_impact']:.6f}")
    print(f"  Total Impact: {impact['total_impact']:.6f}")
    print(f"  Expected Price: ${impact['expected_price']:.2f}")
    print(f"  Impact Price: ${impact['impact_price']:.2f}")
    print(f"  Slippage: ${impact['slippage']:.2f}")
    print(f"  Slippage (bps): {impact['slippage_bps']:.2f}")
    
    # Simulate a market order using the impact model
    order_result = simulate_market_order_with_impact(
        book, 'buy', 100000.0, volatility, market_resilience, market_cap
    )
    
    # Print order simulation results
    print(f"\nMarket Order Simulation with Almgren-Chriss:")
    print(f"  Quantity: ${order_result['quantity_usd']:.2f}")
    print(f"  Executed: {order_result['executed_quantity']:.6f} BTC")
    print(f"  Avg Price: ${order_result['avg_execution_price']:.2f}")
    print(f"  Expected Price: ${order_result['expected_price']:.2f}")
    print(f"  Slippage: {order_result['slippage_pct']:.4f}%")
    print(f"  Fee: ${order_result['fee_usd']:.2f}")
    print(f"  Total Cost: ${order_result['total_cost']:.2f}") 