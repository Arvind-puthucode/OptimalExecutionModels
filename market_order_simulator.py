from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
from orderbook import OrderBook
from regression_models import SlippageModel, simulate_market_order_with_prediction
# Import the enhanced regression models
try:
    from enhanced_regression_models import EnhancedSlippageModel, TimeSeriesSlippageModel, simulate_market_order_with_enhanced_ml
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    ENHANCED_MODELS_AVAILABLE = False
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import pandas as pd


class MakerTakerClassifier:
    """
    Classifier for predicting maker/taker proportion of orders.
    
    For market orders, this is always 100% taker.
    For limit orders, this uses a logistic regression model to predict
    the probability of an order being a taker or maker based on market conditions.
    """
    
    def __init__(self, model_path="maker_taker_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load existing model if available."""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def _save_model(self):
        """Save the current model."""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names
            }
            try:
                joblib.dump(model_data, self.model_path)
            except Exception as e:
                print(f"Error saving model: {e}")
    
    def train(self, X, y, test_size=0.2):
        """
        Train a logistic regression model to predict maker/taker classification.
        
        Args:
            X: Features dataframe
            y: Target values (1 for taker, 0 for maker)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        from sklearn.model_selection import train_test_split
        
        if len(X) < 10:
            raise ValueError("Not enough data for training (minimum 10 records needed)")
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        
        self._save_model()
        
        return {
            "model_type": "Logistic Regression",
            "samples_count": len(X),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": accuracy
        }
    
    def predict_maker_taker(self, order_type, order_price, order_book, quantity_usd, volatility):
        """
        Predict maker/taker proportions for an order.
        
        Args:
            order_type: Type of order ('market', 'limit', etc.)
            order_price: Price of the order (for limit orders)
            order_book: Current order book state
            quantity_usd: Order size in USD
            volatility: Market volatility
            
        Returns:
            Dictionary with maker/taker proportions and prediction details
        """
        # For market orders, it's always 100% taker
        if order_type.lower() == 'market':
            return {
                "taker_proportion": 1.0,
                "maker_proportion": 0.0,
                "prediction_method": "rule-based"
            }
        
        # For limit orders, use the model to predict taker proportion
        if self.model is None:
            # If no model available, use simple heuristics
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            
            if not best_bid or not best_ask:
                return {
                    "taker_proportion": 0.5,
                    "maker_proportion": 0.5,
                    "prediction_method": "fallback-heuristic"
                }
            
            mid_price = (best_bid[0] + best_ask[0]) / 2
            
            # If limit buy above current ask or limit sell below current bid, likely to be taker
            if (order_price >= best_ask[0]) or (order_price <= best_bid[0]):
                taker_proportion = 0.9  # Mostly taker
            # If closer to mid price, more likely to be part taker
            elif abs(order_price - mid_price) / mid_price < 0.001:
                taker_proportion = 0.5  # Mix of maker/taker
            else:
                taker_proportion = 0.1  # Mostly maker
            
            return {
                "taker_proportion": taker_proportion,
                "maker_proportion": 1.0 - taker_proportion,
                "prediction_method": "heuristic"
            }
        
        # Calculate features for model prediction
        features = self._calculate_prediction_features(order_type, order_price, order_book, quantity_usd, volatility)
        
        # Create dataframe with features matching the trained model
        pred_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        for feature, value in features.items():
            if feature in self.feature_names:
                pred_df[feature] = value
        
        # Predict probability of being a taker
        taker_probability = self.model.predict_proba(pred_df)[0][1]
        
        return {
            "taker_proportion": taker_probability,
            "maker_proportion": 1.0 - taker_probability,
            "prediction_method": "model"
        }
    
    def _calculate_prediction_features(self, order_type, order_price, order_book, quantity_usd, volatility):
        """
        Calculate features needed for maker/taker prediction.
        
        Args:
            order_type: Type of order
            order_price: Price of the order
            order_book: Current order book state
            quantity_usd: Order size in USD
            volatility: Market volatility
            
        Returns:
            Dictionary of features for prediction
        """
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        if not best_bid or not best_ask:
            return {
                'price_to_mid': 0,
                'price_to_best': 0,
                'order_size_relative': 0,
                'volatility': volatility,
                'spread_relative': 0
            }
        
        mid_price = (best_bid[0] + best_ask[0]) / 2
        spread = best_ask[0] - best_bid[0]
        spread_relative = spread / mid_price
        
        # Normalize price relative to mid price
        price_to_mid = (order_price - mid_price) / mid_price if mid_price > 0 else 0
        
        # Price relative to best bid/ask
        if order_price >= mid_price:
            # Buy-side logic: compare to best ask
            price_to_best = (order_price - best_ask[0]) / best_ask[0] if best_ask[0] > 0 else 0
        else:
            # Sell-side logic: compare to best bid
            price_to_best = (best_bid[0] - order_price) / best_bid[0] if best_bid[0] > 0 else 0
        
        # Order size relative to market depth (first 5 levels)
        depth = sum(qty for _, qty in order_book.get_bids(5) + order_book.get_asks(5))
        order_size_relative = quantity_usd / (depth * mid_price) if (depth * mid_price) > 0 else 0
        
        return {
            'price_to_mid': price_to_mid,
            'price_to_best': price_to_best,
            'order_size_relative': order_size_relative,
            'volatility': volatility,
            'spread_relative': spread_relative
        }


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
    
    # Add maker/taker classification (market orders are always 100% taker)
    maker_taker_classifier = MakerTakerClassifier()
    maker_taker_info = maker_taker_classifier.predict_maker_taker('market', 0, order_book, quantity_usd, 0)
    
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
        "is_complete": remaining_usd == 0,
        "maker_proportion": maker_taker_info["maker_proportion"],
        "taker_proportion": maker_taker_info["taker_proportion"]
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
    
    # Maker/taker fee calculation if available
    maker_proportion = simulation_result.get("maker_proportion", 0)
    taker_proportion = simulation_result.get("taker_proportion", 1.0)  # Default to 100% taker for market orders
    
    # Calculate weighted fee based on maker/taker proportions
    # This is educational only - we're still using the flat fee_percentage for now
    maker_fee_pct = fee_percentage * 0.7  # Maker fees are typically lower (~70% of taker fees)
    weighted_fee_pct = (maker_proportion * maker_fee_pct) + (taker_proportion * fee_percentage)
    
    # Combine metrics
    return {
        **simulation_result,
        "fee_percentage": fee_percentage,
        "fee_usd": fee_usd,
        "market_impact_pct": market_impact_pct,
        "total_cost": total_cost,
        "net_quantity": simulation_result["executed_quantity"],
        "effective_price": total_cost / simulation_result["executed_quantity"] if simulation_result["executed_quantity"] > 0 else 0,
        "maker_proportion": maker_proportion,
        "taker_proportion": taker_proportion,
        "weighted_fee_pct": weighted_fee_pct
    }


def simulate_market_order_with_ml(
    order_book: OrderBook,
    side: str,
    quantity_usd: float,
    volatility: float,
    fee_percentage: float = 0.001,
    risk_level: str = 'q50'
) -> Dict[str, Any]:
    """
    Simulate a market order execution using machine learning for slippage prediction.
    
    Args:
        order_book: The current order book state
        side: 'buy' or 'sell' 
        quantity_usd: Amount in USD to trade
        volatility: Market volatility
        fee_percentage: Trading fee percentage
        risk_level: Risk level for prediction (q10, q25, q50, q75, q90, mean)
    
    Returns:
        Dictionary with execution details including predicted average price and slippage
    """
    # First try to use enhanced models if available
    if ENHANCED_MODELS_AVAILABLE:
        try:
            # Check if the enhanced model exists and has data
            enhanced_model = EnhancedSlippageModel()
            if enhanced_model.linear_model is not None:
                # If the enhanced model is available, use it with the default "linear" model
                # The other enhanced functions will be called through the app.py when desired
                return simulate_market_order_with_enhanced_ml(
                    order_book,
                    side,
                    quantity_usd,
                    volatility,
                    fee_percentage,
                    "linear",  # Default to linear model
                    risk_level
                )
        except Exception as e:
            # If enhanced model fails, fall back to standard prediction
            print(f"Enhanced model failed, falling back to standard model: {e}")
            pass
    
    # Fall back to standard model
    model = SlippageModel()
    return simulate_market_order_with_prediction(
        order_book,
        side,
        quantity_usd,
        volatility,
        model,
        risk_level,
        fee_percentage
    )


class ModelSelectionStrategy:
    """
    Implements a decision-making framework for selecting the appropriate 
    market impact model based on order characteristics and market conditions.
    
    This strategy selects between:
    1. Simple order book simulation (for small orders)
    2. ML-based slippage prediction (for medium orders)
    3. Almgren-Chriss model (for large orders and execution planning)
    """
    
    def __init__(self):
        # Define thresholds for model selection
        self.small_order_threshold = 0.01  # 1% of avg daily volume
        self.large_order_threshold = 0.05  # 5% of avg daily volume
        self.high_volatility_threshold = 0.4  # 40% annualized vol
        
        # Check if enhanced models are available
        self.has_enhanced_models = ENHANCED_MODELS_AVAILABLE
        
        # Check if Almgren-Chriss model is available
        try:
            import almgren_chriss as ac
            from almgren_chriss_calibration import AlmgrenChrissCalibrator
            self.has_almgren_chriss = True
        except ImportError:
            self.has_almgren_chriss = False
    
    def select_model(self, order_book, side, quantity_usd, volatility, 
                    avg_daily_volume, execution_timeframe=None):
        """
        Select the most appropriate market impact model for the given order.
        
        Args:
            order_book: Current order book state
            side: 'buy' or 'sell'
            quantity_usd: Order size in USD
            volatility: Market volatility
            avg_daily_volume: Average daily trading volume in USD
            execution_timeframe: Optional timeframe for execution in hours
            
        Returns:
            Dictionary with selected model and rationale
        """
        # Calculate order size relative to daily volume
        if avg_daily_volume > 0:
            relative_size = quantity_usd / avg_daily_volume
        else:
            relative_size = 0.1  # Default to medium-sized if volume unknown
        
        # Check execution timeframe
        has_execution_window = execution_timeframe is not None and execution_timeframe > 0
        
        # Calculate order book depth
        bid_depth = sum(qty for _, qty in order_book.get_bids())
        ask_depth = sum(qty for _, qty in order_book.get_asks())
        
        # Check if order size exceeds available liquidity
        if side.lower() == 'buy' and quantity_usd > ask_depth:
            exceeds_orderbook = True
        elif side.lower() == 'sell' and quantity_usd > bid_depth:
            exceeds_orderbook = True
        else:
            exceeds_orderbook = False
        
        # 1. SMALL orders with sufficient liquidity -> Order book simulation
        if relative_size <= self.small_order_threshold and not exceeds_orderbook:
            model_type = "orderbook"
            rationale = "Small order with sufficient liquidity in order book"
        
        # 2. LARGE orders or orders with execution timeframe -> Almgren-Chriss
        elif (relative_size >= self.large_order_threshold or has_execution_window) and self.has_almgren_chriss:
            model_type = "almgren_chriss"
            rationale = "Large order size relative to daily volume" if relative_size >= self.large_order_threshold else "Order with specific execution timeframe"
        
        # 3. MEDIUM orders with enhanced models available -> ML model
        elif self.has_enhanced_models:
            # For high volatility scenarios, prefer Almgren-Chriss if available
            if volatility >= self.high_volatility_threshold and self.has_almgren_chriss:
                model_type = "almgren_chriss"
                rationale = "Medium order in high volatility conditions"
            else:
                model_type = "ml_enhanced"
                rationale = "Medium-sized order, using ML prediction"
        
        # 4. MEDIUM orders without enhanced models -> Basic ML or order book
        else:
            if exceeds_orderbook and self.has_almgren_chriss:
                model_type = "almgren_chriss"
                rationale = "Order exceeds visible liquidity in order book"
            else:
                model_type = "ml_basic"
                rationale = "Medium-sized order, using basic ML prediction"
        
        # Prepare result with additional context for the decision
        result = {
            "selected_model": model_type,
            "rationale": rationale,
            "order_metrics": {
                "relative_size": relative_size,
                "exceeds_orderbook": exceeds_orderbook,
                "high_volatility": volatility >= self.high_volatility_threshold,
                "has_execution_window": has_execution_window
            }
        }
        
        return result
    
    def simulate_with_selected_model(self, order_book, side, quantity_usd, 
                                    volatility, avg_daily_volume, 
                                    fee_percentage=0.001, 
                                    execution_timeframe=None,
                                    market_resilience=0.1):
        """
        Simulate market order using the automatically selected model.
        
        Args:
            order_book: Current order book state
            side: 'buy' or 'sell'
            quantity_usd: Order size in USD
            volatility: Market volatility
            avg_daily_volume: Average daily trading volume in USD
            fee_percentage: Trading fee percentage
            execution_timeframe: Optional timeframe for execution in hours
            market_resilience: Market resilience factor for Almgren-Chriss
            
        Returns:
            Dictionary with simulation results
        """
        # Select appropriate model
        selection = self.select_model(order_book, side, quantity_usd, 
                                     volatility, avg_daily_volume, execution_timeframe)
        model_type = selection["selected_model"]
        
        # Simulate using the selected model
        if model_type == "orderbook":
            # Use simple order book simulation
            result = simulate_market_order(order_book, side, quantity_usd)
            
        elif model_type == "ml_basic":
            # Use basic ML model
            result = simulate_market_order_with_ml(
                order_book, side, quantity_usd, volatility, fee_percentage
            )
            
        elif model_type == "ml_enhanced" and self.has_enhanced_models:
            # Use enhanced ML model
            result = simulate_market_order_with_enhanced_ml(
                order_book, side, quantity_usd, volatility, fee_percentage
            )
            
        elif model_type == "almgren_chriss" and self.has_almgren_chriss:
            import almgren_chriss as ac
            
            if execution_timeframe is not None and execution_timeframe > 0:
                # Use optimal execution strategy for large orders with timeframe
                result = ac.compare_execution_strategies(
                    order_book, side, quantity_usd, execution_timeframe,
                    volatility, market_resilience
                )
            else:
                # Use Almgren-Chriss for immediate execution
                result = ac.simulate_market_order_with_impact(
                    order_book, side, quantity_usd, volatility,
                    market_resilience, None, fee_percentage
                )
        else:
            # Fallback to order book simulation
            result = simulate_market_order(order_book, side, quantity_usd)
            selection["rationale"] += " (fallback due to model unavailability)"
        
        # Add selection details to result
        result["model_selection"] = selection
        
        return result

    def compare_all_models(self, order_book, side, quantity_usd, volatility, 
                         avg_daily_volume, fee_percentage=0.001,
                         execution_timeframe=None):
        """
        Run simulations using all available models for comparison.
        
        Args:
            order_book: Current order book state
            side: 'buy' or 'sell'
            quantity_usd: Order size in USD
            volatility: Market volatility
            avg_daily_volume: Average daily trading volume in USD
            fee_percentage: Trading fee percentage
            execution_timeframe: Optional timeframe for execution in hours
            
        Returns:
            Dictionary with results from all models
        """
        results = {
            "order_details": {
                "side": side,
                "quantity_usd": quantity_usd,
                "volatility": volatility,
                "relative_size": quantity_usd / avg_daily_volume if avg_daily_volume > 0 else None
            }
        }
        
        # 1. Order book simulation
        try:
            results["orderbook"] = simulate_market_order(order_book, side, quantity_usd)
        except Exception as e:
            results["orderbook"] = {"error": str(e)}
        
        # 2. Basic ML model
        try:
            results["ml_basic"] = simulate_market_order_with_ml(
                order_book, side, quantity_usd, volatility, fee_percentage
            )
        except Exception as e:
            results["ml_basic"] = {"error": str(e)}
        
        # 3. Enhanced ML model (if available)
        if self.has_enhanced_models:
            try:
                results["ml_enhanced"] = simulate_market_order_with_enhanced_ml(
                    order_book, side, quantity_usd, volatility, fee_percentage
                )
            except Exception as e:
                results["ml_enhanced"] = {"error": str(e)}
        
        # 4. Almgren-Chriss model (if available)
        if self.has_almgren_chriss:
            import almgren_chriss as ac
            try:
                if execution_timeframe is not None and execution_timeframe > 0:
                    results["almgren_chriss"] = ac.compare_execution_strategies(
                        order_book, side, quantity_usd, execution_timeframe,
                        volatility, 0.1  # default market resilience
                    )
                else:
                    results["almgren_chriss"] = ac.simulate_market_order_with_impact(
                        order_book, side, quantity_usd, volatility,
                        0.1, None, fee_percentage
                    )
            except Exception as e:
                results["almgren_chriss"] = {"error": str(e)}
        
        # Add model selection recommendation
        selection = self.select_model(
            order_book, side, quantity_usd, volatility, 
            avg_daily_volume, execution_timeframe
        )
        results["recommended_model"] = selection
        
        return results

# Example usage
if __name__ == "__main__":
    # Create a sample order book
    order_book = OrderBook("BTC-USDT")
    sample_data = {
        'bids': [['49900', '1.5'], ['49800', '2.0']],
        'asks': [['50000', '1.0'], ['50100', '3.0']]
    }
    order_book.update(sample_data)
    
    # Sample parameters
    side = "buy"
    quantity_usd = 100000  # $100k order
    volatility = 0.35  # 35% annualized
    avg_daily_volume = 5000000  # $5M daily volume
    
    # Create model selector
    model_selector = ModelSelectionStrategy()
    
    # Compare all models
    comparison = model_selector.compare_all_models(
        order_book, side, quantity_usd, volatility, avg_daily_volume,
        execution_timeframe=2  # 2-hour execution window
    )
    
    # Print results
    import json
    print(json.dumps(comparison.get("recommended_model"), indent=2)) 