"""
Regression models for slippage prediction.

This module implements various regression models to predict slippage in market orders
based on historical data and current market conditions.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
from orderbook import OrderBook


class SlippageDataCollector:
    """Collects and manages historical slippage data for model training."""
    
    def __init__(self, data_path: str = "slippage_data.csv"):
        self.data_path = data_path
        self.data = None
        self._load_data()
    
    def _load_data(self):
        if os.path.exists(self.data_path):
            try:
                self.data = pd.read_csv(self.data_path)
            except Exception as e:
                print(f"Error loading data: {e}")
                self.data = pd.DataFrame()
        else:
            self.data = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'side', 'quantity_usd', 'order_pct_of_depth',
                'spread_bps', 'volatility', 'depth_imbalance', 'slippage_bps'
            ])
    
    def add_execution_data(self, order_result, order_book, volatility):
        """Add execution result to the dataset."""
        if not order_result.get("success", False):
            return
        
        # Calculate market depth metrics
        depth_data = self._calculate_market_depth(order_book)
        
        # Calculate slippage in basis points
        slippage_bps = order_result.get("slippage_pct", 0) * 100
        
        # Calculate what percentage of market depth this order represented
        side = order_result.get("side", "").lower()
        quantity_usd = order_result.get("quantity_usd", 0)
        
        if side == 'buy':
            depth_usd = depth_data.get("ask_value_usd", 0)
        else:
            depth_usd = depth_data.get("bid_value_usd", 0)
        
        order_pct_of_depth = (quantity_usd / depth_usd * 100) if depth_usd > 0 else 0
        
        # Record the data
        new_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': order_book.symbol,
            'side': side,
            'quantity_usd': quantity_usd,
            'order_pct_of_depth': order_pct_of_depth,
            'spread_bps': depth_data.get("spread_bps", 0),
            'volatility': volatility,
            'depth_imbalance': depth_data.get("depth_imbalance", 0),
            'slippage_bps': slippage_bps
        }
        
        self.data = pd.concat([self.data, pd.DataFrame([new_data])], ignore_index=True)
        self._save_data()
    
    def _calculate_market_depth(self, order_book, depth_levels=5):
        """Calculate market depth metrics from the order book."""
        bids = order_book.get_bids(depth_levels)
        asks = order_book.get_asks(depth_levels)
        
        bid_volume = sum(qty for _, qty in bids)
        ask_volume = sum(qty for _, qty in asks)
        
        bid_value_usd = sum(price * qty for price, qty in bids)
        ask_value_usd = sum(price * qty for price, qty in asks)
        
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        if not best_bid or not best_ask:
            return {"bid_volume": 0, "ask_volume": 0, "bid_value_usd": 0,
                    "ask_value_usd": 0, "mid_price": 0, "spread": 0,
                    "spread_bps": 0, "depth_imbalance": 0}
        
        mid_price = (best_bid[0] + best_ask[0]) / 2
        spread = best_ask[0] - best_bid[0]
        spread_bps = (spread / mid_price) * 10000
        
        depth_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        return {
            "bid_volume": bid_volume, "ask_volume": ask_volume,
            "bid_value_usd": bid_value_usd, "ask_value_usd": ask_value_usd,
            "mid_price": mid_price, "spread": spread,
            "spread_bps": spread_bps, "depth_imbalance": depth_imbalance
        }
    
    def _save_data(self):
        """Save the current data to CSV."""
        try:
            self.data.to_csv(self.data_path, index=False)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def get_training_data(self):
        """Prepare the dataset for model training."""
        if self.data is None or len(self.data) < 10:
            raise ValueError("Not enough data for training (minimum 10 records needed)")
        
        features = ['quantity_usd', 'order_pct_of_depth', 'spread_bps', 
                    'volatility', 'depth_imbalance']
        
        X = pd.get_dummies(self.data[features + ['side']], drop_first=True)
        y = self.data['slippage_bps']
        
        return X, y


class SlippageModel:
    """Regression models for predicting slippage."""
    
    def __init__(self, model_path="slippage_model.joblib"):
        self.model_path = model_path
        self.linear_model = None
        self.quantile_models = {}
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load existing model if available."""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.linear_model = model_data.get('linear_model')
                self.quantile_models = model_data.get('quantile_models', {})
                self.feature_names = model_data.get('feature_names')
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def train(self, X, y, test_size=0.2):
        """Train regression models on historical data."""
        if len(X) < 10:
            raise ValueError("Not enough data for training (minimum 10 records needed)")
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.linear_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        self.linear_model.fit(X_train, y_train)
        linear_score = self.linear_model.score(X_test, y_test)
        
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            self.quantile_models[f"q{int(q*100)}"] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', QuantileRegressor(quantile=q, alpha=0.5, solver='highs'))
            ])
            self.quantile_models[f"q{int(q*100)}"].fit(X_train, y_train)
        
        self._save_model()
        
        return {
            "model_type": "Linear + Quantile Regression",
            "samples_count": len(X),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "linear_r2_score": linear_score,
            "quantiles_trained": quantiles
        }
    
    def _save_model(self):
        """Save the current model."""
        if self.linear_model is not None:
            model_data = {
                'linear_model': self.linear_model,
                'quantile_models': self.quantile_models,
                'feature_names': self.feature_names
            }
            try:
                joblib.dump(model_data, self.model_path)
            except Exception as e:
                print(f"Error saving model: {e}")
    
    def predict_slippage(self, order_book, side, quantity_usd, volatility, risk_level='mean'):
        """Predict slippage for a given order."""
        if self.linear_model is None:
            return {"error": "Model not trained. Train the model first."}
        
        features = self._calculate_prediction_features(order_book, side, quantity_usd, volatility)
        
        if self.feature_names:
            pred_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
            
            for feature, value in features.items():
                if feature in self.feature_names:
                    pred_df[feature] = value
                
            if 'side_sell' in self.feature_names:
                pred_df['side_sell'] = 1 if side.lower() == 'sell' else 0
        else:
            pred_df = pd.DataFrame([features])
        
        predictions = {}
        predictions['mean'] = float(self.linear_model.predict(pred_df)[0])
        
        for name, model in self.quantile_models.items():
            try:
                predictions[name] = float(model.predict(pred_df)[0])
            except Exception:
                predictions[name] = predictions.get('mean', 0)
        
        if risk_level in predictions:
            predicted_slippage_bps = predictions[risk_level]
        else:
            predicted_slippage_bps = predictions['mean']
        
        predicted_slippage_pct = predicted_slippage_bps / 100.0
        
        if side.lower() == 'buy':
            reference_price = order_book.get_best_ask()[0] if order_book.get_best_ask() else 0
        else:
            reference_price = order_book.get_best_bid()[0] if order_book.get_best_bid() else 0
        
        adjusted_price = reference_price * (1 + (predicted_slippage_pct / 100.0)) if reference_price > 0 else 0
        
        return {
            "predicted_slippage_bps": predicted_slippage_bps,
            "predicted_slippage_pct": predicted_slippage_pct,
            "reference_price": reference_price,
            "predicted_execution_price": adjusted_price,
            "prediction_models": list(predictions.keys()),
            "all_predictions": predictions,
            "risk_level": risk_level
        }
    
    def _calculate_prediction_features(self, order_book, side, quantity_usd, volatility):
        """Calculate features needed for slippage prediction."""
        depth_calculator = SlippageDataCollector()
        depth_data = depth_calculator._calculate_market_depth(order_book)
        
        if side.lower() == 'buy':
            depth_usd = depth_data.get("ask_value_usd", 0)
        else:
            depth_usd = depth_data.get("bid_value_usd", 0)
        
        order_pct_of_depth = (quantity_usd / depth_usd * 100) if depth_usd > 0 else 0
        
        return {
            'quantity_usd': quantity_usd,
            'order_pct_of_depth': order_pct_of_depth,
            'spread_bps': depth_data.get("spread_bps", 0),
            'volatility': volatility,
            'depth_imbalance': depth_data.get("depth_imbalance", 0)
        }


def simulate_market_order_with_prediction(
    order_book,
    side,
    quantity_usd,
    volatility,
    slippage_model=None,
    risk_level='q50',
    fee_percentage=0.001
):
    """Simulate a market order execution using regression model slippage prediction."""
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    
    if not best_bid or not best_ask:
        return {
            "success": False,
            "error": "Insufficient liquidity in order book"
        }
    
    mid_price = (best_bid[0] + best_ask[0]) / 2
    
    if side.lower() == 'buy':
        expected_price = best_ask[0]
    else:
        expected_price = best_bid[0]
    
    if slippage_model is not None:
        prediction = slippage_model.predict_slippage(
            order_book, side, quantity_usd, volatility, risk_level
        )
        
        adjusted_price = prediction.get("predicted_execution_price", expected_price)
        slippage_pct = prediction.get("predicted_slippage_pct", 0)
    else:
        adjusted_price = expected_price
        slippage_pct = 0
    
    executed_quantity = quantity_usd / adjusted_price if adjusted_price > 0 else 0
    total_cost = quantity_usd
    fee_usd = total_cost * fee_percentage
    slippage_usd = abs(adjusted_price - expected_price) * executed_quantity
    
    # Market orders are always 100% taker
    maker_proportion = 0.0
    taker_proportion = 1.0
    
    result = {
        "success": True,
        "side": side,
        "quantity_usd": quantity_usd,
        "filled_usd": total_cost,
        "unfilled_usd": 0,
        "executed_quantity": executed_quantity,
        "avg_execution_price": adjusted_price,
        "expected_price": expected_price,
        "mid_price": mid_price,
        "slippage_pct": slippage_pct,
        "slippage_usd": slippage_usd,
        "fee_percentage": fee_percentage,
        "fee_usd": fee_usd,
        "total_cost": total_cost + fee_usd if side.lower() == 'buy' else total_cost - fee_usd,
        "is_complete": True,
        "model_type": "regression",
        "risk_level": risk_level,
        "maker_proportion": maker_proportion,
        "taker_proportion": taker_proportion,
        "order_type": "market"
    }
    
    if slippage_model is not None:
        result["prediction"] = prediction
    
    return result


if __name__ == "__main__":
    from orderbook import OrderBook
    
    book = OrderBook("BTC-USDT")
    sample_data = {
        'bids': [['50000.0', '1.5'], ['49900.0', '2.3'], ['49800.0', '5.0']],
        'asks': [['50100.0', '1.0'], ['50200.0', '3.2'], ['50300.0', '2.5']]
    }
    book.update(sample_data)
    
    model = SlippageModel()
    
    X_synth = pd.DataFrame([
        {'quantity_usd': 10000, 'order_pct_of_depth': 20, 'spread_bps': 10, 
         'volatility': 0.3, 'depth_imbalance': 0.1, 'side_sell': 0}
    ] * 20)
    y_synth = X_synth['quantity_usd'] * 0.0001 + X_synth['spread_bps'] * 0.5
    
    model.train(X_synth, y_synth)
    prediction = model.predict_slippage(book, 'buy', 100000.0, 0.3)
    
    result = simulate_market_order_with_prediction(book, 'buy', 100000.0, 0.3, model)
    print("Slippage Prediction Model Successfully Implemented!")
