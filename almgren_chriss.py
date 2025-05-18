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

# Check if calibration module exists
try:
    from almgren_chriss_calibration import AlmgrenChrissCalibrator
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False


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
    # Base impact factor derived from spread and depth - using a much smaller baseline
    # Crypto markets have higher liquidity and faster recovery than traditional markets
    # So impact factors need to be scaled down significantly
    
    # Make sure we don't divide by zero
    total_volume = max(depth["bid_volume"] + depth["ask_volume"], 1.0)
    
    # Base factor is spread relative to depth, but scaled down significantly
    base_factor = (depth["spread"] / total_volume) * 0.01
    
    # Adjust for volatility (higher volatility = higher impact)
    # Cap volatility to a reasonable value
    capped_volatility = min(volatility, 0.5)
    volatility_adjustment = math.sqrt(capped_volatility)
    
    # Adjust for market cap if available (lower cap = higher impact)
    cap_adjustment = 1.0
    if market_cap is not None and market_cap > 0:
        # For crypto, market cap impact is less pronounced than traditional markets
        cap_adjustment = math.pow(10, 10) / market_cap
        cap_adjustment = min(cap_adjustment, 2.0)  # Cap the adjustment
    
    # Depth imbalance affects impact (more imbalance = higher impact)
    # But with a lighter effect for crypto markets
    imbalance_factor = 1 + 0.5 * abs(depth["depth_imbalance"])
    
    # Combine factors
    gamma = base_factor * volatility_adjustment * cap_adjustment * imbalance_factor
    
    # Ensure reasonable bounds for crypto markets
    gamma = max(gamma, 1e-7)  # Minimum impact 
    gamma = min(gamma, 0.0001)  # Maximum impact (0.01% per unit)
    
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
    # For crypto markets, permanent impact is much smaller than in traditional markets
    # due to high liquidity and rapid recovery, especially on major trading venues
    
    # Make sure relative_depth is not zero
    rel_depth = max(depth["relative_depth"], 0.1)
    
    # Calculate base impact from spread and relative depth, but scaled down significantly
    base_impact = (depth["spread"] / rel_depth) * 0.005
    
    # Adjust for market resilience (higher resilience = lower permanent impact)
    # Crypto markets are generally more resilient (recover faster)
    resilience_factor = 1 - min(market_resilience, 0.95)  # Cap at 0.95 to ensure some minimum impact
    
    # Permanent impact is a smaller fraction of temporary impact for crypto
    eta = base_impact * resilience_factor * 0.1  # Only 10% of base impact
    
    # Ensure reasonable bounds for crypto markets
    eta = max(eta, 5e-8)  # Minimum impact
    eta = min(eta, 0.00005)  # Maximum impact (0.005% per unit)
    
    return eta


def calculate_almgren_chriss_impact(
    order_book: OrderBook, 
    side: str, 
    quantity: float, 
    volatility: float,
    market_resilience: float = 0.1,
    market_cap: Optional[float] = None,
    use_calibration: bool = True
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
    - use_calibration: Whether to use calibrated parameters if available
    
    Returns:
    - Dictionary with impact components and parameters
    """
    # Calculate market depth
    depth = calculate_market_depth(order_book)
    
    # Get mid price
    mid_price = depth["mid_price"]
    
    # Get calibrated parameters if available and requested
    calibration_params = {}
    if use_calibration and CALIBRATION_AVAILABLE:
        try:
            calibrator = AlmgrenChrissCalibrator()
            calibration_params = calibrator.get_market_parameters(order_book.symbol, volatility)
        except Exception as e:
            print(f"Error getting calibrated parameters: {e}")
    
    # Estimate temporary impact factor (gamma)
    gamma = estimate_temporary_impact(depth, volatility, market_cap)
    
    # Estimate permanent impact factor (eta)
    eta = estimate_permanent_impact(depth, market_resilience)
    
    # Apply calibration scaling if available
    if calibration_params:
        gamma_scale = calibration_params.get('gamma_scale', 1.0)
        eta_scale = calibration_params.get('eta_scale', 1.0)
        volatility_adjustment = calibration_params.get('volatility_adjustment', 1.0)
        
        gamma *= gamma_scale * volatility_adjustment
        eta *= eta_scale
    
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
        "market_resilience": market_resilience,
        "calibrated": bool(calibration_params) if use_calibration and CALIBRATION_AVAILABLE else False
    }


def simulate_market_order_with_impact(
    order_book: OrderBook,
    side: str,
    quantity_usd: float,
    volatility: float,
    market_resilience: float = 0.1,
    market_cap: Optional[float] = None,
    fee_percentage: float = 0.001,
    use_calibration: bool = True
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
        use_calibration: Whether to use calibrated parameters
        
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
        market_cap,
        use_calibration
    )
    
    # Adjust price based on impact
    adjusted_price = impact_result["impact_price"]
    
    # Calculate executed quantity and cost
    executed_quantity = quantity_usd / adjusted_price
    total_cost = quantity_usd
    
    # Apply impact scaling based on market conditions
    # For crypto, scale down the impact based on order size
    # Large orders have proportionally less impact in highly liquid markets
    impact_scale = 1.0
    
    # Normalize quantity as a percentage of market depth
    depth = impact_result.get("mid_price", 0) * (order_book.get_depth('bid', 10) + order_book.get_depth('ask', 10))
    if depth > 0:
        order_size_pct = quantity_usd / depth
        
        # Apply scaling formula based on empirical observations
        # Smaller impact for small orders, gradually increasing for larger orders but capped
        if order_size_pct < 0.01:  # Very small order
            impact_scale = 0.3  # Reduce impact by 70%
        elif order_size_pct < 0.05:  # Small order
            impact_scale = 0.5  # Reduce impact by 50%
        elif order_size_pct < 0.10:  # Medium order
            impact_scale = 0.7  # Reduce impact by 30%
        elif order_size_pct < 0.25:  # Large order
            impact_scale = 0.9  # Reduce impact by 10%
        # Else use full impact for very large orders (>25% of visible depth)
    
    # Adjust impact based on scaling
    slippage = impact_result["slippage"] * impact_scale
    slippage_bps = impact_result["slippage_bps"] * impact_scale
    
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
        "slippage_pct": (slippage / impact_result["expected_price"]) * 100 if impact_result["expected_price"] > 0 else 0,
        "slippage_usd": slippage * executed_quantity,
        "slippage_bps": slippage_bps,
        "fee_percentage": fee_percentage,
        "fee_usd": fee_usd,
        "total_cost": total_cost + fee_usd if side.lower() == 'buy' else total_cost - fee_usd,
        "is_complete": True,
        "impact_model": "almgren_chriss",
        "calibrated": impact_result.get("calibrated", False),
        "impact_metrics": impact_result,
        "impact_scale": impact_scale,
        "order_size_relative": order_size_pct if 'order_size_pct' in locals() else 0.0
    }
    
    return result


def generate_optimal_execution_schedule(
    total_quantity: float,
    time_horizon: float,  # in hours
    eta: float,           # permanent impact parameter
    gamma: float,         # temporary impact parameter
    sigma: float,         # volatility
    lambda_risk: float,   # risk aversion factor (0 = risk-neutral)
    num_intervals: int = 10
) -> Dict[str, Any]:
    """
    Generate an optimal execution schedule using the Almgren-Chriss framework.
    
    This function computes the optimal trading trajectory that minimizes the 
    combination of market impact and timing risk for large orders.
    
    Args:
        total_quantity: Total order size to execute
        time_horizon: Time horizon in hours
        eta: Permanent impact parameter
        gamma: Temporary impact parameter
        sigma: Market volatility
        lambda_risk: Risk aversion factor (higher = more risk-averse)
        num_intervals: Number of discrete time intervals for the schedule
        
    Returns:
        Dictionary with execution schedule details
    """
    # Convert time horizon to seconds
    T = time_horizon * 3600
    
    # Initialize variables
    kappa = 0.0
    use_linear = False
    expected_cost_bps = 0.0
    expected_risk_bps = 0.0
    
    try:
        # Hard cap all parameters to extremely conservative values to avoid overflow
        capped_eta = min(eta, 0.00001)
        capped_gamma = min(gamma, 0.00001)
        capped_sigma = min(sigma, 0.3)
        
        # Risk aversion needs special handling
        if lambda_risk <= 0.0:
            # Risk-neutral case - use linear schedule
            use_linear = True
            capped_lambda = 0.0
        else:
            # For risk-averse case, cap lambda to a reasonable value
            capped_lambda = min(lambda_risk, 0.2)
        
        if use_linear or capped_gamma <= 0 or capped_eta <= 0:
            use_linear = True
        else:
            # Calculate kappa with stable values
            # The formula is: kappa = sqrt(lambda * sigma^2 / (gamma + eta))
            denominator = capped_gamma + capped_eta
            
            if denominator > 0:
                kappa_value = math.sqrt((capped_lambda * capped_sigma**2) / denominator)
                kappa = min(kappa_value, 0.00005)  # Cap kappa to a very small value
                
                # If kappa is too small or too large, fall back to linear
                if kappa < 1e-8 or kappa > 0.00005 or math.isnan(kappa):
                    use_linear = True
            else:
                use_linear = True
    except (ValueError, OverflowError, ZeroDivisionError):
        # Any numerical issues, default to linear schedule
        use_linear = True
    
    # Generate time points (in hours)
    time_points = np.linspace(0, time_horizon, num_intervals + 1)
    
    # Calculate expected cost and risk components
    initial_price = 100.0  # Arbitrary reference price for cost calculation
    
    if use_linear:
        # Linear trajectory (risk-neutral case)
        inventory = np.linspace(total_quantity, 0, num_intervals + 1)
        trading_rates = []
        
        # Calculate trading rates from inventory changes
        for i in range(len(inventory) - 1):
            trading_rates.append(inventory[i] - inventory[i+1])
        
        # For linear schedule, expected cost is simpler
        permanent_cost = 0.5 * capped_eta * total_quantity**2
        temporary_cost = capped_gamma * total_quantity**2 / num_intervals
        
        # Risk is zero for instantaneous execution
        expected_cost_bps = (permanent_cost + temporary_cost) * 10000 / (initial_price * total_quantity)
        expected_risk_bps = 0.0
    else:
        try:
            # Optimal trajectory based on hyperbolic cosine formula
            inventory = []
            
            # Safe version of the cosh calculation
            for t in time_points:
                t_sec = t * 3600  # Convert hours to seconds
                remain_time = T - t_sec
                
                if t_sec >= T:
                    # At final time, inventory is 0
                    remaining = 0
                else:
                    try:
                        # Use a scaled version of the formula to avoid overflow
                        scale_factor = 0.0001  # Small enough to avoid overflow
                        scaled_kappa = kappa * scale_factor
                        
                        num = math.cosh(scaled_kappa * remain_time)
                        denom = math.cosh(scaled_kappa * T)
                        
                        if denom > 0 and not math.isnan(num) and not math.isinf(num):
                            remaining = total_quantity * (num / denom)
                        else:
                            # Fall back to linear for this point
                            remaining = total_quantity * (1 - t/time_horizon)
                    except (OverflowError, ValueError):
                        # Linear fallback for any calculation issues
                        remaining = total_quantity * (1 - t/time_horizon)
                
                inventory.append(remaining)
            
            inventory = np.array(inventory)
            
            # Calculate trading rates from inventory changes
            trading_rates = []
            for i in range(len(inventory) - 1):
                trading_rates.append(inventory[i] - inventory[i+1])
            
            # Calculate expected costs (with reasonable limitations)
            # Impact cost with permanent and temporary components
            permanent_cost = 0.5 * capped_eta * total_quantity**2
            
            # For temporary cost, use the formula based on trading rates
            temp_cost_sum = 0
            for rate in trading_rates:
                temp_cost_sum += capped_gamma * rate**2
            
            temporary_cost = temp_cost_sum
            
            # Risk cost (simplified to avoid overflow)
            risk_cost = 0
            if capped_lambda > 0 and capped_sigma > 0:
                for i in range(len(trading_rates)):
                    t = time_points[i+1] - time_points[i]  # Time interval in hours
                    avg_inventory = (inventory[i] + inventory[i+1]) / 2
                    risk_cost += capped_lambda * (capped_sigma**2) * (avg_inventory**2) * t
            
            # Calculate costs in basis points (limited to reasonable values)
            expected_cost_bps = min((permanent_cost + temporary_cost) * 10000 / (initial_price * total_quantity), 50.0)
            expected_risk_bps = min(risk_cost * 10000 / (initial_price * total_quantity), 20.0)
            
        except Exception as e:
            # Any error, fall back to linear
            print(f"Error calculating optimal schedule: {e}")
            inventory = np.linspace(total_quantity, 0, num_intervals + 1)
            trading_rates = [(inventory[i] - inventory[i+1]) for i in range(len(inventory) - 1)]
            
            # Simple cost estimate for linear
            expected_cost_bps = (capped_eta + capped_gamma/num_intervals) * total_quantity * 10000 / initial_price
            expected_risk_bps = 0.0
    
    # Return the schedule and cost estimates
    return {
        "inventory": inventory.tolist(),
        "time_points": time_points.tolist(),
        "trading_rates": trading_rates,
        "expected_cost_bps": expected_cost_bps,
        "expected_risk_bps": expected_risk_bps,
        "expected_total_bps": expected_cost_bps + expected_risk_bps,
        "kappa": kappa,
        "used_linear": use_linear,
        "params": {
            "eta": eta,
            "gamma": gamma,
            "sigma": sigma,
            "lambda": lambda_risk,
            "capped_eta": capped_eta,
            "capped_gamma": capped_gamma,
            "capped_sigma": capped_sigma,
            "capped_lambda": capped_lambda if 'capped_lambda' in locals() else 0.0
        }
    }


def compare_execution_strategies(
    order_book: OrderBook,
    side: str,
    quantity: float,
    time_horizon: float,
    volatility: float,
    market_resilience: float = 0.1,
    lambda_risk: float = 1.0,
    num_strategies: int = 3
) -> Dict[str, Any]:
    """
    Compare different execution strategies for a large order.
    
    This function compares:
    1. Immediate execution (market order)
    2. Linear time-weighted average price (TWAP)
    3. Optimal Almgren-Chriss execution
    
    Args:
        order_book: Current order book state
        side: 'buy' or 'sell'
        quantity: Order size to execute
        time_horizon: Time horizon in hours
        volatility: Market volatility
        market_resilience: Market resilience factor
        lambda_risk: Risk aversion parameter
        num_strategies: Number of strategies to compare
        
    Returns:
        Dictionary with strategy comparison details
    """
    # Calculate market depth
    depth = calculate_market_depth(order_book)
    
    # Get mid price
    mid_price = depth["mid_price"]
    
    # Estimate impact parameters
    gamma = estimate_temporary_impact(depth, volatility)
    eta = estimate_permanent_impact(depth, market_resilience)
    
    # Direction multiplier
    direction = 1 if side.lower() == 'buy' else -1
    
    # Strategy 1: Immediate execution
    immediate_impact = calculate_almgren_chriss_impact(
        order_book, side, quantity, volatility, market_resilience
    )
    
    # Strategy 2: Linear TWAP
    num_intervals = 10
    interval_size = quantity / num_intervals
    twap_impact_total = 0
    twap_trades = []
    
    # Simulate TWAP execution
    for i in range(num_intervals):
        interval_impact = calculate_almgren_chriss_impact(
            order_book, side, interval_size, volatility, market_resilience
        )
        twap_impact_total += interval_impact["total_impact"] * interval_size
        twap_trades.append({
            "size": interval_size,
            "slippage_bps": interval_impact.get("slippage_bps", 0),
            "expected_price": mid_price * (1 + direction * interval_impact["total_impact"])
        })
    
    twap_impact_bps = (twap_impact_total / (quantity * mid_price)) * 10000 if quantity > 0 and mid_price > 0 else 0
    
    # Strategy 3: Optimal Almgren-Chriss
    ac_schedule = generate_optimal_execution_schedule(
        quantity, time_horizon, eta, gamma, volatility, lambda_risk, num_intervals
    )
    
    # Calculate trade details for optimal schedule
    ac_trades = []
    ac_impact_total = 0
    
    # Use trading_rates instead of trades (which was removed in the updated implementation)
    if "trading_rates" in ac_schedule:
        for i, trade_size in enumerate(ac_schedule["trading_rates"]):
            # For each trade in the optimal schedule
            if trade_size > 0:  # Only process non-zero trades
                trade_impact = calculate_almgren_chriss_impact(
                    order_book, side, trade_size, volatility, market_resilience
                )
                ac_impact_total += trade_impact["total_impact"] * trade_size
                ac_trades.append({
                    "size": trade_size,
                    "slippage_bps": trade_impact.get("slippage_bps", 0),
                    "expected_price": mid_price * (1 + direction * trade_impact["total_impact"])
                })
    
    # Calculate total impact in basis points
    ac_impact_bps = 0
    if quantity > 0 and mid_price > 0:
        ac_impact_bps = (ac_impact_total / (quantity * mid_price)) * 10000
    
    # Return comparison results
    return {
        "immediate_execution": {
            "expected_cost_bps": min(immediate_impact.get("slippage_bps", 0), 100.0),  # Cap at 100 bps for reasonable display
            "expected_price": immediate_impact.get("impact_price", mid_price),
            "execution_time": 0
        },
        "twap_execution": {
            "expected_cost_bps": min(twap_impact_bps, 50.0),  # Cap at 50 bps for reasonable display
            "trades": twap_trades,
            "execution_time": time_horizon
        },
        "optimal_execution": {
            "expected_cost_bps": min(ac_impact_bps, 50.0),  # Cap at 50 bps for reasonable display
            "expected_cost_model_bps": ac_schedule["expected_cost_bps"],
            "trades": ac_trades,
            "schedule": {
                "time_points": ac_schedule["time_points"],
                "trading_rates": ac_schedule["trading_rates"],
                "inventory": ac_schedule["inventory"]
            },
            "execution_time": time_horizon,
            "kappa": ac_schedule["kappa"]
        },
        "parameters": {
            "gamma": gamma,
            "eta": eta,
            "volatility": volatility,
            "lambda_risk": lambda_risk
        }
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