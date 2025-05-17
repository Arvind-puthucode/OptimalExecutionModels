"""
Enhanced regression models for slippage prediction.

This module extends the basic regression models with:
1. Feature importance visualization
2. Advanced regression techniques (Random Forest, Gradient Boosting)
3. Time-series forecasting capabilities
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO

# Import the original regression models
from regression_models import SlippageModel, SlippageDataCollector


class EnhancedSlippageModel(SlippageModel):
    """Enhanced regression models for predicting slippage with advanced features."""
    
    def __init__(self, model_path="enhanced_slippage_model.joblib"):
        super().__init__(model_path)
        self.models = {}
        self.feature_importances = {}
        self.model_performances = {}
        self.load_enhanced_model()
    
    def load_enhanced_model(self):
        """Load existing enhanced model if available."""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.models = model_data.get('models', {})
                self.feature_importances = model_data.get('feature_importances', {})
                self.model_performances = model_data.get('model_performances', {})
                self.feature_names = model_data.get('feature_names')
            except Exception as e:
                print(f"Error loading enhanced model: {e}")
    
    def train(self, X, y, test_size=0.2):
        """Train multiple regression models on historical data."""
        if len(X) < 10:
            raise ValueError("Not enough data for training (minimum 10 records needed)")
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train the base linear model
        base_result = super().train(X, y, test_size)
        self.models['linear'] = self.linear_model
        
        # Train Random Forest model
        self.models['random_forest'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        self.models['random_forest'].fit(X_train, y_train)
        rf_score = self.models['random_forest'].score(X_test, y_test)
        
        # Train Gradient Boosting model
        self.models['gradient_boosting'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        self.models['gradient_boosting'].fit(X_train, y_train)
        gb_score = self.models['gradient_boosting'].score(X_test, y_test)
        
        # Calculate and store model performances
        self.model_performances = {
            'linear': {
                'r2_score': base_result['linear_r2_score'],
                'rmse': np.sqrt(mean_squared_error(y_test, self.models['linear'].predict(X_test))),
                'mae': mean_absolute_error(y_test, self.models['linear'].predict(X_test))
            },
            'random_forest': {
                'r2_score': rf_score,
                'rmse': np.sqrt(mean_squared_error(y_test, self.models['random_forest'].predict(X_test))),
                'mae': mean_absolute_error(y_test, self.models['random_forest'].predict(X_test))
            },
            'gradient_boosting': {
                'r2_score': gb_score,
                'rmse': np.sqrt(mean_squared_error(y_test, self.models['gradient_boosting'].predict(X_test))),
                'mae': mean_absolute_error(y_test, self.models['gradient_boosting'].predict(X_test))
            }
        }
        
        # Extract feature importances
        self._extract_feature_importances(X)
        
        # Save the enhanced model
        self._save_enhanced_model()
        
        return {
            "models_trained": list(self.models.keys()),
            "samples_count": len(X),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "model_performances": self.model_performances
        }
    
    def _extract_feature_importances(self, X):
        """Extract feature importances from tree-based models."""
        for model_name, model in self.models.items():
            if model_name in ['random_forest', 'gradient_boosting']:
                # Get the regressor from the pipeline
                regressor = model.named_steps['regressor']
                # Get feature importances
                importances = regressor.feature_importances_
                # Store as a dictionary with feature names
                self.feature_importances[model_name] = dict(zip(X.columns, importances))
    
    def _save_enhanced_model(self):
        """Save the current enhanced model."""
        model_data = {
            'models': self.models,
            'feature_importances': self.feature_importances,
            'model_performances': self.model_performances,
            'feature_names': self.feature_names
        }
        try:
            joblib.dump(model_data, self.model_path)
        except Exception as e:
            print(f"Error saving enhanced model: {e}")
    
    def predict_slippage(self, order_book, side, quantity_usd, volatility, model_type='gradient_boosting', risk_level='mean'):
        """Predict slippage using the specified model type."""
        if not self.models or model_type not in self.models:
            # Fall back to the base model prediction
            return super().predict_slippage(order_book, side, quantity_usd, volatility, risk_level)
        
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
        
        selected_model = self.models[model_type]
        predicted_slippage_bps = float(selected_model.predict(pred_df)[0])
        predicted_slippage_pct = predicted_slippage_bps / 100.0
        
        if side.lower() == 'buy':
            reference_price = order_book.get_best_ask()[0] if order_book.get_best_ask() else 0
        else:
            reference_price = order_book.get_best_bid()[0] if order_book.get_best_bid() else 0
        
        adjusted_price = reference_price * (1 + (predicted_slippage_pct / 100.0)) if reference_price > 0 else 0
        
        # Include predictions from all models for comparison
        all_predictions = {}
        for name, model in self.models.items():
            all_predictions[name] = float(model.predict(pred_df)[0])
        
        return {
            "predicted_slippage_bps": predicted_slippage_bps,
            "predicted_slippage_pct": predicted_slippage_pct,
            "reference_price": reference_price,
            "predicted_execution_price": adjusted_price,
            "model_used": model_type,
            "all_model_predictions": all_predictions
        }
    
    def plot_feature_importance(self, model_type='gradient_boosting', plot_type='bar'):
        """Generate a feature importance visualization."""
        if model_type not in self.feature_importances:
            return None
        
        importances = self.feature_importances[model_type]
        
        # Convert to DataFrame for easier plotting
        imp_df = pd.DataFrame({
            'Feature': list(importances.keys()),
            'Importance': list(importances.values())
        }).sort_values('Importance', ascending=False)
        
        if plot_type == 'bar':
            # Create bar plot with Plotly
            fig = px.bar(
                imp_df, 
                x='Feature', 
                y='Importance',
                title=f'Feature Importance ({model_type})',
                color='Importance',
                labels={'Importance': 'Relative Importance'}
            )
        else:  # pie chart
            fig = px.pie(
                imp_df, 
                values='Importance', 
                names='Feature',
                title=f'Feature Importance ({model_type})',
                labels={'Importance': 'Relative Importance'}
            )
        
        return fig
    
    def compare_model_performance(self):
        """Generate a visualization comparing performance of different models."""
        if not self.model_performances:
            return None
        
        # Create DataFrame for visualization
        metrics = []
        for model_name, performance in self.model_performances.items():
            for metric_name, value in performance.items():
                metrics.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': value
                })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Create grouped bar chart with Plotly
        fig = px.bar(
            metrics_df,
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            labels={'Value': 'Score/Error Value'}
        )
        
        return fig


class TimeSeriesSlippageModel:
    """Time-series forecasting model for slippage prediction."""
    
    def __init__(self, data_path="slippage_timeseries.joblib"):
        self.data_path = data_path
        self.models = {}
        self.time_features = []
        self.load_model()
    
    def load_model(self):
        """Load existing time series model if available."""
        if os.path.exists(self.data_path):
            try:
                model_data = joblib.load(self.data_path)
                self.models = model_data.get('models', {})
                self.time_features = model_data.get('time_features', [])
            except Exception as e:
                print(f"Error loading time series model: {e}")
    
    def prepare_time_features(self, data):
        """Prepare time-based features from timestamp data."""
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in data.columns:
            if isinstance(data['timestamp'].iloc[0], str):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        else:
            # If no timestamp column, we can't create time features
            return data, []
        
        # Extract time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['day_of_month'] = data['timestamp'].dt.day
        data['week_of_year'] = data['timestamp'].dt.isocalendar().week
        data['month'] = data['timestamp'].dt.month
        
        # Calculate rolling statistics (requires sorted data)
        data = data.sort_values('timestamp')
        
        # Add lag features for the target variable
        for lag in [1, 3, 5, 10]:
            data[f'slippage_lag_{lag}'] = data['slippage_bps'].shift(lag)
        
        # Add rolling mean features
        for window in [3, 5, 10]:
            data[f'slippage_rolling_mean_{window}'] = data['slippage_bps'].rolling(window=window).mean()
        
        # Add rolling volatility
        for window in [5, 10]:
            data[f'slippage_rolling_std_{window}'] = data['slippage_bps'].rolling(window=window).std()
        
        # Drop rows with NaN values created by lag and rolling features
        data = data.dropna()
        
        # List of time-based feature columns
        time_features = [
            'hour', 'day_of_week', 'day_of_month', 'week_of_year', 'month'
        ] + [f'slippage_lag_{lag}' for lag in [1, 3, 5, 10]] + \
          [f'slippage_rolling_mean_{window}' for window in [3, 5, 10]] + \
          [f'slippage_rolling_std_{window}' for window in [5, 10]]
        
        return data, time_features
    
    def train(self, data, test_size=0.2):
        """Train time series models on historical data."""
        if len(data) < 30:
            raise ValueError("Not enough data for time series training (minimum 30 records needed)")
        
        # Prepare time features
        prepared_data, time_features = self.prepare_time_features(data)
        self.time_features = time_features
        
        if not time_features:
            raise ValueError("No time features could be created. Ensure data has a timestamp column.")
        
        # Combine regular features with time features
        regular_features = ['quantity_usd', 'order_pct_of_depth', 'spread_bps', 
                            'volatility', 'depth_imbalance']
        
        if 'side' in prepared_data.columns:
            X = pd.get_dummies(prepared_data[regular_features + time_features + ['side']], drop_first=True)
        else:
            X = prepared_data[regular_features + time_features]
        
        y = prepared_data['slippage_bps']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False  # No shuffle for time series
        )
        
        # Train Random Forest model
        self.models['ts_random_forest'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        self.models['ts_random_forest'].fit(X_train, y_train)
        rf_score = self.models['ts_random_forest'].score(X_test, y_test)
        
        # Train Gradient Boosting model
        self.models['ts_gradient_boosting'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        self.models['ts_gradient_boosting'].fit(X_train, y_train)
        gb_score = self.models['ts_gradient_boosting'].score(X_test, y_test)
        
        # Save the model
        self._save_model()
        
        return {
            "models_trained": list(self.models.keys()),
            "time_features_used": time_features,
            "samples_count": len(X),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "random_forest_r2": rf_score,
            "gradient_boosting_r2": gb_score
        }
    
    def _save_model(self):
        """Save the time series model."""
        model_data = {
            'models': self.models,
            'time_features': self.time_features
        }
        try:
            joblib.dump(model_data, self.data_path)
        except Exception as e:
            print(f"Error saving time series model: {e}")
    
    def forecast_slippage_trend(self, data, days_ahead=7, model_type='ts_gradient_boosting'):
        """Forecast slippage trend for the next N days."""
        if model_type not in self.models:
            return None
        
        # Prepare time features
        prepared_data, _ = self.prepare_time_features(data)
        
        if prepared_data.empty or len(prepared_data) < 10:
            return None
        
        # Get the last row for creating the forecast base
        last_row = prepared_data.iloc[-1].copy()
        
        # Create forecast data points
        forecast_rows = []
        
        for i in range(1, days_ahead + 1):
            forecast_date = last_row['timestamp'] + timedelta(days=i)
            new_row = last_row.copy()
            new_row['timestamp'] = forecast_date
            
            # Update time features for the new date
            new_row['hour'] = forecast_date.hour
            new_row['day_of_week'] = forecast_date.dayofweek
            new_row['day_of_month'] = forecast_date.day
            new_row['week_of_year'] = forecast_date.isocalendar().week
            new_row['month'] = forecast_date.month
            
            forecast_rows.append(new_row)
        
        # Create DataFrame from forecast rows
        forecast_df = pd.DataFrame(forecast_rows)
        
        # Get feature columns matching the trained model
        regular_features = ['quantity_usd', 'order_pct_of_depth', 'spread_bps', 
                           'volatility', 'depth_imbalance']
        
        if 'side_sell' in forecast_df.columns:
            X_forecast = forecast_df[regular_features + self.time_features + ['side_sell']]
        else:
            # Create dummy columns if needed
            forecast_df['side_sell'] = 0  # Default to "buy"
            X_forecast = forecast_df[regular_features + self.time_features + ['side_sell']]
        
        # Make predictions
        predictions = self.models[model_type].predict(X_forecast)
        
        # Create results
        results = pd.DataFrame({
            'date': [row['timestamp'] for row in forecast_rows],
            'slippage_bps': predictions
        })
        
        return results
    
    def plot_slippage_trend(self, data, forecast_data):
        """Plot historical and forecasted slippage trends."""
        if forecast_data is None or len(forecast_data) == 0:
            return None
        
        # Prepare historical data
        historical = data.copy()
        if 'timestamp' in historical.columns:
            if isinstance(historical['timestamp'].iloc[0], str):
                historical['timestamp'] = pd.to_datetime(historical['timestamp'])
            
            historical = historical.sort_values('timestamp')
            
            # Create a combined plot
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=historical['timestamp'],
                y=historical['slippage_bps'],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Add forecast data
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['slippage_bps'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Add a vertical line to separate historical and forecast data
            last_historical_date = historical['timestamp'].max()
            
            fig.add_shape(
                type="line",
                x0=last_historical_date,
                y0=min(historical['slippage_bps'].min(), forecast_data['slippage_bps'].min()),
                x1=last_historical_date,
                y1=max(historical['slippage_bps'].max(), forecast_data['slippage_bps'].max()),
                line=dict(color="black", width=2, dash="dash"),
            )
            
            # Update layout
            fig.update_layout(
                title='Slippage Trend Forecast',
                xaxis_title='Date',
                yaxis_title='Slippage (bps)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
        
        return None


def simulate_market_order_with_enhanced_ml(
    order_book,
    side,
    quantity_usd,
    volatility,
    fee_percentage=0.001,
    model_type='gradient_boosting',
    risk_level='mean'
):
    """Simulate a market order execution using enhanced regression models."""
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
    
    # Use enhanced model for prediction
    model = EnhancedSlippageModel()
    prediction = model.predict_slippage(
        order_book, side, quantity_usd, volatility, model_type, risk_level
    )
    
    adjusted_price = prediction.get("predicted_execution_price", expected_price)
    slippage_pct = prediction.get("predicted_slippage_pct", 0)
    
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
        "model_type": model_type,
        "risk_level": risk_level,
        "maker_proportion": maker_proportion,
        "taker_proportion": taker_proportion,
        "order_type": "market",
        "prediction": prediction
    }
    
    return result


if __name__ == "__main__":
    from orderbook import OrderBook
    
    # Create a sample order book
    book = OrderBook("BTC-USDT")
    sample_data = {
        'bids': [['50000.0', '1.5'], ['49900.0', '2.3'], ['49800.0', '5.0']],
        'asks': [['50100.0', '1.0'], ['50200.0', '3.2'], ['50300.0', '2.5']]
    }
    book.update(sample_data)
    
    # Test enhanced model
    enhanced_model = EnhancedSlippageModel()
    
    # Generate synthetic data for testing
    X_synth = pd.DataFrame([
        {'quantity_usd': 10000, 'order_pct_of_depth': 20, 'spread_bps': 10, 
         'volatility': 0.3, 'depth_imbalance': 0.1, 'side_sell': 0}
    ] * 20)
    y_synth = X_synth['quantity_usd'] * 0.0001 + X_synth['spread_bps'] * 0.5
    
    # Train model
    enhanced_model.train(X_synth, y_synth)
    
    # Test time series model
    # Create synthetic time series data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    ts_data = pd.DataFrame({
        'timestamp': dates,
        'quantity_usd': np.random.normal(10000, 1000, 50),
        'order_pct_of_depth': np.random.normal(20, 5, 50),
        'spread_bps': np.random.normal(10, 2, 50),
        'volatility': np.random.normal(0.3, 0.05, 50),
        'depth_imbalance': np.random.normal(0, 0.1, 50),
        'side': ['buy' if i % 2 == 0 else 'sell' for i in range(50)],
        'slippage_bps': np.random.normal(5, 1, 50) + np.arange(50) * 0.1  # Trend component
    })
    
    ts_model = TimeSeriesSlippageModel()
    try:
        ts_model.train(ts_data)
        print("Enhanced Regression Models Successfully Implemented!")
    except Exception as e:
        print(f"Time series model training failed: {e}") 