"""
Model Validation Metrics for Slippage Prediction

This module implements model validation metrics for the Almgren-Chriss model
and other slippage prediction models.

The validation metrics include:
1. Mean Absolute Percentage Error (MAPE)
2. Mean Absolute Error (MAE)
3. R-squared (R²)
4. Prediction vs. Actual visualizations
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict

# Import local modules
from orderbook import OrderBook
import almgren_chriss as ac
from almgren_chriss_calibration import AlmgrenChrissCalibrator


class ModelValidator:
    """
    Validates and tracks performance of slippage prediction models.
    """
    
    def __init__(self, data_path: str = "slippage_data.csv", 
                 stats_path: str = "model_validation_stats.json"):
        self.data_path = data_path
        self.stats_path = stats_path
        self.validation_stats = {}
        self._load_stats()
    
    def _load_stats(self):
        """Load validation statistics from file if available."""
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r') as f:
                    self.validation_stats = json.load(f)
            except Exception as e:
                print(f"Error loading validation stats: {e}")
                self.validation_stats = {}
    
    def _save_stats(self):
        """Save validation statistics to file."""
        try:
            with open(self.stats_path, 'w') as f:
                json.dump(self.validation_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving validation stats: {e}")
    
    def load_historical_data(self, days_back: int = None):
        """
        Load historical execution data for validation.
        
        Args:
            days_back: Optional number of days to look back
            
        Returns:
            DataFrame with historical data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Historical data file not found: {self.data_path}")
        
        try:
            data = pd.read_csv(self.data_path)
            
            # Filter to recent data if specified
            if days_back is not None:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                cutoff_date = datetime.now() - timedelta(days=days_back)
                data = data[data['timestamp'] >= cutoff_date]
            
            # Filter out rows with missing or invalid values
            data = data.dropna(subset=['slippage_bps', 'quantity_usd', 'spread_bps', 'volatility'])
            return data
        except Exception as e:
            raise ValueError(f"Error loading historical data: {e}")
    
    def calculate_mape(self, actual, predicted):
        """Calculate Mean Absolute Percentage Error."""
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def validate_almgren_chriss(self, symbol: str = None, days_back: int = 30):
        """
        Validate Almgren-Chriss model performance on historical data.
        
        Args:
            symbol: Market symbol to validate (e.g., 'BTC-USDT')
                   If None, validates for all markets with sufficient data
            days_back: Number of days to look back for validation data
            
        Returns:
            Dictionary with validation results
        """
        try:
            data = self.load_historical_data(days_back)
            
            # If symbol is provided, filter data for that market
            if symbol:
                market_data = data[data['symbol'] == symbol]
                if len(market_data) < 10:  # Need sufficient data for validation
                    raise ValueError(f"Insufficient data for {symbol}. Need at least 10 records.")
                markets = [symbol]
            else:
                # Group by symbol and filter out markets with insufficient data
                markets = [s for s, g in data.groupby('symbol') if len(g) >= 10]
            
            # Get calibrated parameters
            calibrator = AlmgrenChrissCalibrator()
            
            validation_results = {}
            
            for market in markets:
                market_data = data[data['symbol'] == market]
                
                # Ensure numeric columns
                for col in ['bid_price', 'ask_price', 'bid_volume', 'ask_volume', 'volatility', 'slippage_bps', 'quantity_usd']:
                    if col not in market_data.columns:
                        market_data[col] = 0.0  # Add default values if missing
                    else:
                        # Ensure values are numeric
                        market_data[col] = pd.to_numeric(market_data[col], errors='coerce')
                        # Fill any NaN with reasonable defaults
                        if col in ['bid_price', 'ask_price']:
                            market_data[col] = market_data[col].fillna(50000.0)
                        elif col in ['bid_volume', 'ask_volume']:
                            market_data[col] = market_data[col].fillna(1.0)
                        elif col == 'volatility':
                            market_data[col] = market_data[col].fillna(0.3)
                        elif col == 'slippage_bps':
                            market_data[col] = market_data[col].fillna(0.0)
                        elif col == 'quantity_usd':
                            market_data[col] = market_data[col].fillna(1000.0)
                
                # Create a sample order book for this market
                order_book = OrderBook(market)
                
                # Use average order book state from the data
                avg_bid_price = market_data['bid_price'].mean()
                avg_ask_price = market_data['ask_price'].mean()
                avg_bid_qty = market_data['bid_volume'].mean()
                avg_ask_qty = market_data['ask_volume'].mean()
                
                # Create a sample order book
                sample_data = {
                    'bids': [[str(avg_bid_price), str(avg_bid_qty)]],
                    'asks': [[str(avg_ask_price), str(avg_ask_qty)]]
                }
                order_book.update(sample_data)
                
                # Prepare data for validation
                quantities = market_data['quantity_usd'].values
                
                # Get most common side or default to "buy"
                sides = []
                if 'side' in market_data.columns:
                    sides = market_data['side'].values
                else:
                    sides = ['buy'] * len(quantities)
                    
                volatility_values = market_data['volatility'].values
                actual_slippage_bps = market_data['slippage_bps'].values
                
                # Get calibrated parameters for this market
                market_params = calibrator.get_market_parameters(market, volatility_values.mean())
                gamma_scale = market_params.get('gamma_scale', 1.0)
                eta_scale = market_params.get('eta_scale', 1.0)
                
                # Make predictions
                predicted_slippage_bps = []
                
                for i in range(len(quantities)):
                    quantity = quantities[i]
                    side = sides[i]
                    volatility = volatility_values[i]
                    
                    try:
                        # Calculate market depth
                        depth = ac.calculate_market_depth(order_book)
                        
                        # Calculate base parameters and apply calibration
                        base_gamma = ac.estimate_temporary_impact(depth, volatility)
                        base_eta = ac.estimate_permanent_impact(depth)
                        
                        gamma = base_gamma * gamma_scale
                        eta = base_eta * eta_scale
                        
                        # Calculate predicted impact
                        direction = 1 if str(side).lower() == 'buy' else -1
                        temporary_impact = direction * gamma * quantity
                        permanent_impact = direction * eta * quantity
                        total_impact = permanent_impact + 0.5 * temporary_impact
                        
                        # Convert to basis points
                        predicted_bps = total_impact * 10000
                        predicted_slippage_bps.append(predicted_bps)
                    except Exception as e:
                        # Use default value if calculation fails
                        predicted_slippage_bps.append(0.0)
                        print(f"Prediction error for {market} record {i}: {e}")
                
                # Calculate validation metrics
                predicted_slippage_bps = np.array(predicted_slippage_bps)
                try:
                    mape = self.calculate_mape(actual_slippage_bps, predicted_slippage_bps)
                except Exception:
                    mape = 0.0
                    
                mae = np.mean(np.abs(actual_slippage_bps - predicted_slippage_bps))
                
                # Calculate coefficient of determination (R²)
                y_mean = np.mean(actual_slippage_bps)
                ss_total = np.sum((actual_slippage_bps - y_mean) ** 2)
                ss_residual = np.sum((actual_slippage_bps - predicted_slippage_bps) ** 2)
                r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                
                # Store validation results
                validation_results[market] = {
                    'mape': mape,
                    'mae': mae,
                    'r2': r2,
                    'sample_count': len(market_data),
                    'time_period_days': days_back,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Update validation stats history
                if market not in self.validation_stats:
                    self.validation_stats[market] = []
                
                self.validation_stats[market].append(validation_results[market])
            
            # Save updated validation stats
            self._save_stats()
            
            return validation_results
        
        except Exception as e:
            print(f"Validation failed: {e}")
            return {}
    
    def plot_prediction_vs_actual(self, symbol: str, days_back: int = 30):
        """
        Generate a plot comparing predicted vs. actual slippage.
        
        Args:
            symbol: Market symbol to plot
            days_back: Number of days to look back for data
            
        Returns:
            Base64-encoded PNG image of the plot
        """
        data = self.load_historical_data(days_back)
        market_data = data[data['symbol'] == symbol]
        
        if len(market_data) < 10:
            raise ValueError(f"Insufficient data for {symbol}. Need at least 10 records.")
        
        # Create a sample order book for this market
        order_book = OrderBook(symbol)
        
        # Use average order book state from the data
        avg_bid_price = market_data.get('bid_price', 50000).mean()
        avg_ask_price = market_data.get('ask_price', 50100).mean()
        avg_bid_qty = market_data.get('bid_volume', 1.0).mean()
        avg_ask_qty = market_data.get('ask_volume', 1.0).mean()
        
        # Create a sample order book
        sample_data = {
            'bids': [[str(avg_bid_price), str(avg_bid_qty)]],
            'asks': [[str(avg_ask_price), str(avg_ask_qty)]]
        }
        order_book.update(sample_data)
        
        # Get calibrated parameters
        calibrator = AlmgrenChrissCalibrator()
        market_params = calibrator.get_market_parameters(symbol, market_data['volatility'].mean())
        gamma_scale = market_params.get('gamma_scale', 1.0)
        eta_scale = market_params.get('eta_scale', 1.0)
        
        # Prepare data
        quantities = market_data['quantity_usd'].values
        sides = market_data['side'].values
        volatility_values = market_data['volatility'].values
        actual_slippage_bps = market_data['slippage_bps'].values
        
        # Make predictions
        predicted_slippage_bps = []
        
        for i in range(len(quantities)):
            quantity = quantities[i]
            side = sides[i]
            volatility = volatility_values[i]
            
            # Calculate market depth
            depth = ac.calculate_market_depth(order_book)
            
            # Calculate base parameters and apply calibration
            base_gamma = ac.estimate_temporary_impact(depth, volatility)
            base_eta = ac.estimate_permanent_impact(depth)
            
            gamma = base_gamma * gamma_scale
            eta = base_eta * eta_scale
            
            # Calculate predicted impact
            direction = 1 if side.lower() == 'buy' else -1
            temporary_impact = direction * gamma * quantity
            permanent_impact = direction * eta * quantity
            total_impact = permanent_impact + 0.5 * temporary_impact
            
            # Convert to basis points
            predicted_bps = total_impact * 10000
            predicted_slippage_bps.append(predicted_bps)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Sort by order size for better visualization
        idx = np.argsort(quantities)
        sorted_quantities = quantities[idx]
        sorted_actual = actual_slippage_bps[idx]
        sorted_predicted = np.array(predicted_slippage_bps)[idx]
        
        # Plot
        plt.scatter(sorted_quantities, sorted_actual, alpha=0.7, label='Actual')
        plt.scatter(sorted_quantities, sorted_predicted, alpha=0.7, label='Predicted')
        
        # Add regression lines
        from scipy.stats import linregress
        
        # Actual regression line
        slope_actual, intercept_actual, _, _, _ = linregress(sorted_quantities, sorted_actual)
        plt.plot(sorted_quantities, intercept_actual + slope_actual * sorted_quantities, 
                 'r', label='Actual Trend')
        
        # Predicted regression line
        slope_pred, intercept_pred, _, _, _ = linregress(sorted_quantities, sorted_predicted)
        plt.plot(sorted_quantities, intercept_pred + slope_pred * sorted_quantities, 
                 'b', label='Predicted Trend')
        
        # Labels and title
        plt.xlabel('Order Size (USD)')
        plt.ylabel('Slippage (bps)')
        plt.title(f'Actual vs. Predicted Slippage for {symbol} (last {days_back} days)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add stats
        mape = self.calculate_mape(sorted_actual, sorted_predicted)
        mae = np.mean(np.abs(sorted_actual - sorted_predicted))
        y_mean = np.mean(sorted_actual)
        ss_total = np.sum((sorted_actual - y_mean) ** 2)
        ss_residual = np.sum((sorted_actual - sorted_predicted) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        plt.figtext(0.15, 0.85, f'MAPE: {mape:.2f}%\nMAE: {mae:.2f} bps\nR²: {r2:.4f}', 
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def get_market_validation_history(self, symbol: str):
        """
        Get validation history for a specific market.
        
        Args:
            symbol: Market symbol
            
        Returns:
            List of validation stats over time
        """
        if symbol in self.validation_stats:
            return self.validation_stats[symbol]
        else:
            return []
    
    def get_market_validation_summary(self):
        """
        Get a summary of validation metrics for all markets.
        
        Returns:
            DataFrame with validation summary
        """
        summary_data = []
        
        for market, history in self.validation_stats.items():
            if history:
                latest = history[-1]
                summary_data.append({
                    'market': market,
                    'mape': latest['mape'],
                    'mae': latest['mae'],
                    'r2': latest['r2'],
                    'sample_count': latest['sample_count'],
                    'last_validation': latest['timestamp']
                })
        
        if summary_data:
            return pd.DataFrame(summary_data)
        else:
            return pd.DataFrame(columns=['market', 'mape', 'mae', 'r2', 'sample_count', 'last_validation'])

    def plot_optimal_execution_comparison(self, symbol: str, quantity: float, time_horizon: float, 
                                         volatility: float = None, lambda_values: List[float] = None):
        """
        Plot optimal execution schedules for different risk aversion levels.
        
        Args:
            symbol: Market symbol to analyze
            quantity: Order quantity to execute
            time_horizon: Execution time horizon in hours
            volatility: Market volatility (if None, uses average from historical data)
            lambda_values: List of risk aversion parameters to compare
            
        Returns:
            Base64-encoded image string of the plot
        """
        import matplotlib.pyplot as plt
        import almgren_chriss as ac
        from almgren_chriss_calibration import AlmgrenChrissCalibrator
        
        # Get calibrated parameters
        calibrator = AlmgrenChrissCalibrator()
        
        # Create a sample order book for this market
        order_book = OrderBook(symbol)
        
        # Load some historical data to get order book state
        data = self.load_historical_data(days_back=30)
        market_data = data[data['symbol'] == symbol]
        
        if len(market_data) < 10:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Calculate order book state
        avg_bid_price = market_data.get('bid_price', 50000).mean()
        avg_ask_price = market_data.get('ask_price', 50100).mean()
        avg_bid_qty = market_data.get('bid_volume', 1.0).mean()
        avg_ask_qty = market_data.get('ask_volume', 1.0).mean()
        
        # Create order book
        sample_data = {
            'bids': [[str(avg_bid_price), str(avg_bid_qty)]],
            'asks': [[str(avg_ask_price), str(avg_ask_qty)]]
        }
        order_book.update(sample_data)
        
        # Use provided volatility or get from data
        if volatility is None:
            volatility = market_data['volatility'].mean()
        
        # Get depth metrics
        depth = ac.calculate_market_depth(order_book)
        
        # Get impact parameters
        gamma = ac.estimate_temporary_impact(depth, volatility)
        eta = ac.estimate_permanent_impact(depth)
        
        # Apply calibration if available
        market_params = calibrator.get_market_parameters(symbol, volatility)
        gamma_scale = market_params.get('gamma_scale', 1.0)
        eta_scale = market_params.get('eta_scale', 1.0)
        
        # Scale the parameters
        gamma *= gamma_scale
        eta *= eta_scale
        
        # Use default lambda values if none provided
        if lambda_values is None:
            lambda_values = [0.0, 0.5, 1.0, 2.0, 5.0]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot TWAP as reference
        time_points = np.linspace(0, time_horizon, 11)
        twap_inventory = quantity * (1 - time_points / time_horizon)
        plt.plot(time_points, twap_inventory, 'k--', label='TWAP (Linear)')
        
        # Colors for different strategies
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        
        # Plot optimal strategies for different risk aversion levels
        for i, lambda_risk in enumerate(lambda_values):
            # Generate optimal schedule
            schedule = ac.generate_optimal_execution_schedule(
                quantity, time_horizon, eta, gamma, volatility, lambda_risk
            )
            
            # Plot inventory curve
            color = colors[i % len(colors)]
            plt.plot(schedule['time_points'], schedule['inventory'], 
                     marker='o', color=color, 
                     label=f'λ={lambda_risk:.1f} (κ={schedule["kappa"]:.4f})')
        
        # Add labels and title
        plt.xlabel('Time (hours)')
        plt.ylabel('Remaining Quantity')
        plt.title(f'Optimal Execution Schedules for {symbol}\n'
                  f'Quantity: {quantity:.2f}, η={eta:.6f}, γ={gamma:.6f}, σ={volatility:.2f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str

    def compare_model_predictions(self, symbol: str = None, days_back: int = 30, min_order_size: float = 1000):
        """
        Compare the predictions from different models against actual slippage.
        
        Args:
            symbol: Market symbol to analyze (if None, analyzes first market with sufficient data)
            days_back: Number of days to look back for data
            min_order_size: Minimum order size to include in comparison
            
        Returns:
            Dictionary with comparison results and base64-encoded plot
        """
        # Import models
        import almgren_chriss as ac
        from market_order_simulator import simulate_market_order, simulate_market_order_with_ml
        try:
            from enhanced_regression_models import simulate_market_order_with_enhanced_ml
            has_enhanced_models = True
        except ImportError:
            has_enhanced_models = False
        
        # Load historical data
        data = self.load_historical_data(days_back)
        
        # Filter by symbol if provided
        if symbol:
            data = data[data['symbol'] == symbol]
        
        # Filter by minimum order size
        data = data[data['quantity_usd'] >= min_order_size]
        
        if len(data) < 10:
            raise ValueError(f"Insufficient data for comparison. Need at least 10 records.")
        
        # Get the first symbol if not specified
        if not symbol:
            symbol = data['symbol'].iloc[0]
            data = data[data['symbol'] == symbol]
        
        # Prepare results dictionary and lists for plotting
        actual_slippage = []
        ob_predictions = []
        ml_predictions = []
        ac_predictions = []
        enhanced_predictions = []
        quantities = []
        
        # Loop through each historical record
        for i, row in data.iterrows():
            # Extract data from the record
            side = row['side']
            quantity_usd = row['quantity_usd']
            volatility = row['volatility']
            actual_slip = row['slippage_bps']
            
            # Create sample order book from record
            order_book = OrderBook(symbol)
            bid_price = row.get('bid_price', 50000)
            ask_price = row.get('ask_price', 50100)
            bid_qty = row.get('bid_volume', 1.0)
            ask_qty = row.get('ask_volume', 1.0)
            
            sample_data = {
                'bids': [[str(bid_price), str(bid_qty)]],
                'asks': [[str(ask_price), str(ask_qty)]]
            }
            order_book.update(sample_data)
            
            # Make predictions with each model
            try:
                # Order book prediction
                ob_result = simulate_market_order(order_book, side, quantity_usd)
                ob_slip = ob_result.get('slippage_bps', 0)
                
                # ML prediction
                ml_result = simulate_market_order_with_ml(order_book, side, quantity_usd, volatility)
                ml_slip = ml_result.get('slippage_bps', 0)
                
                # Almgren-Chriss prediction
                ac_result = ac.simulate_market_order_with_impact(order_book, side, quantity_usd, volatility)
                ac_slip = ac_result.get('impact_bps', 0)
                
                # Enhanced ML prediction if available
                if has_enhanced_models:
                    enhanced_result = simulate_market_order_with_enhanced_ml(order_book, side, quantity_usd, volatility)
                    enhanced_slip = enhanced_result.get('slippage_bps', 0)
                else:
                    enhanced_slip = None
                
                # Append to lists for plotting
                actual_slippage.append(actual_slip)
                ob_predictions.append(ob_slip)
                ml_predictions.append(ml_slip)
                ac_predictions.append(ac_slip)
                if enhanced_slip is not None:
                    enhanced_predictions.append(enhanced_slip)
                quantities.append(quantity_usd)
                
            except Exception as e:
                print(f"Error processing record {i}: {e}")
        
        # Convert to numpy arrays
        actual_slippage = np.array(actual_slippage)
        ob_predictions = np.array(ob_predictions)
        ml_predictions = np.array(ml_predictions)
        ac_predictions = np.array(ac_predictions)
        enhanced_predictions = np.array(enhanced_predictions) if enhanced_predictions else None
        quantities = np.array(quantities)
        
        # Calculate metrics for each model
        from sklearn.metrics import mean_absolute_error, r2_score
        
        results = {
            "data_points": len(actual_slippage),
            "symbol": symbol,
            "time_period_days": days_back,
            "orderbook_model": {
                "mae": mean_absolute_error(actual_slippage, ob_predictions),
                "mape": self.calculate_mape(actual_slippage, ob_predictions),
                "r2": r2_score(actual_slippage, ob_predictions)
            },
            "ml_model": {
                "mae": mean_absolute_error(actual_slippage, ml_predictions),
                "mape": self.calculate_mape(actual_slippage, ml_predictions),
                "r2": r2_score(actual_slippage, ml_predictions)
            },
            "almgren_chriss_model": {
                "mae": mean_absolute_error(actual_slippage, ac_predictions),
                "mape": self.calculate_mape(actual_slippage, ac_predictions),
                "r2": r2_score(actual_slippage, ac_predictions)
            }
        }
        
        if enhanced_predictions is not None:
            results["enhanced_ml_model"] = {
                "mae": mean_absolute_error(actual_slippage, enhanced_predictions),
                "mape": self.calculate_mape(actual_slippage, enhanced_predictions),
                "r2": r2_score(actual_slippage, enhanced_predictions)
            }
        
        # Create comparison plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 10))
        
        # Create scatter plots
        plt.subplot(2, 2, 1)
        plt.scatter(quantities, actual_slippage, alpha=0.6, label='Actual', color='black')
        plt.scatter(quantities, ob_predictions, alpha=0.6, label='Order Book', color='blue')
        plt.xlabel('Order Size (USD)')
        plt.ylabel('Slippage (bps)')
        plt.title('Order Book Model vs Actual')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.scatter(quantities, actual_slippage, alpha=0.6, label='Actual', color='black')
        plt.scatter(quantities, ml_predictions, alpha=0.6, label='ML Model', color='green')
        plt.xlabel('Order Size (USD)')
        plt.ylabel('Slippage (bps)')
        plt.title('ML Model vs Actual')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.scatter(quantities, actual_slippage, alpha=0.6, label='Actual', color='black')
        plt.scatter(quantities, ac_predictions, alpha=0.6, label='Almgren-Chriss', color='red')
        plt.xlabel('Order Size (USD)')
        plt.ylabel('Slippage (bps)')
        plt.title('Almgren-Chriss Model vs Actual')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        if enhanced_predictions is not None:
            plt.scatter(quantities, actual_slippage, alpha=0.6, label='Actual', color='black')
            plt.scatter(quantities, enhanced_predictions, alpha=0.6, label='Enhanced ML', color='purple')
            plt.title('Enhanced ML Model vs Actual')
        else:
            # Model comparison scatter plot
            plt.scatter(ac_predictions, actual_slippage, alpha=0.6, label='Almgren-Chriss', color='red')
            plt.scatter(ml_predictions, actual_slippage, alpha=0.6, label='ML Model', color='green')
            plt.scatter(ob_predictions, actual_slippage, alpha=0.6, label='Order Book', color='blue')
            plt.plot([min(actual_slippage), max(actual_slippage)], 
                    [min(actual_slippage), max(actual_slippage)], 'k--', alpha=0.5)
            plt.xlabel('Predicted Slippage (bps)')
            plt.ylabel('Actual Slippage (bps)')
            plt.title('Model Predictions Comparison')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle(f'Slippage Model Comparison for {symbol} (Last {days_back} Days)', fontsize=14)
        plt.subplots_adjust(top=0.93)
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Add visualization to results
        results["visualization"] = img_str
        
        return results

    def analyze_parameter_stability(self, symbol: str, window_size: int = 7, overlap: int = 1):
        """
        Analyze the stability of Almgren-Chriss parameters over time.
        
        Args:
            symbol: Market symbol to analyze
            window_size: Size of the rolling window in days
            overlap: Number of days to overlap between windows
            
        Returns:
            Dictionary with parameter stability analysis and visualization
        """
        from almgren_chriss_calibration import AlmgrenChrissCalibrator
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        # Load historical data
        data = self.load_historical_data()
        market_data = data[data['symbol'] == symbol]
        
        if len(market_data) < 30:
            raise ValueError(f"Insufficient data for {symbol}. Need at least 30 records.")
        
        # Convert timestamp to datetime
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
        
        # Sort by timestamp
        market_data = market_data.sort_values('timestamp')
        
        # Get unique dates
        dates = market_data['timestamp'].dt.date.unique()
        
        # Prepare lists for tracking parameters
        timestamps = []
        gamma_scales = []
        eta_scales = []
        r2_values = []
        volatilities = []
        
        # Create calibrator
        calibrator = AlmgrenChrissCalibrator()
        
        # Analyze each time window
        for i in range(0, len(dates) - window_size + 1, overlap):
            window_dates = dates[i:i+window_size]
            start_date = pd.Timestamp(window_dates[0])
            end_date = pd.Timestamp(window_dates[-1])
            
            # Filter data for this window
            window_data = market_data[
                (market_data['timestamp'].dt.date >= window_dates[0]) &
                (market_data['timestamp'].dt.date <= window_dates[-1])
            ]
            
            if len(window_data) < 10:
                continue  # Skip windows with insufficient data
            
            try:
                # Create a temporary copy of the calibrator to avoid affecting the main one
                temp_calibrator = AlmgrenChrissCalibrator()
                
                # Override the data path to use our filtered data
                temp_calibrator.data_path = None
                
                # Create a custom load method for our filtered data
                def custom_load():
                    return window_data
                
                # Monkey patch the load method
                temp_calibrator.load_historical_data = custom_load
                
                # Calibrate parameters for this window
                results = temp_calibrator.calibrate_parameters_robust(symbol)
                
                if symbol in results:
                    result = results[symbol]
                    timestamps.append(end_date)
                    gamma_scales.append(result['gamma_scale'])
                    eta_scales.append(result['eta_scale'])
                    r2_values.append(result['validation_r2'])
                    volatilities.append(window_data['volatility'].mean())
            
            except Exception as e:
                print(f"Error calibrating window {start_date} to {end_date}: {e}")
        
        # Create the visualization
        plt.figure(figsize=(12, 10))
        
        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        gamma_scales = np.array(gamma_scales)
        eta_scales = np.array(eta_scales)
        r2_values = np.array(r2_values)
        volatilities = np.array(volatilities)
        
        # Plot gamma and eta scales
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, gamma_scales, 'b-', label='Gamma Scale (Temporary Impact)')
        plt.plot(timestamps, eta_scales, 'r-', label='Eta Scale (Permanent Impact)')
        plt.title(f'Almgren-Chriss Parameters Over Time for {symbol}')
        plt.ylabel('Parameter Scale')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot R² values
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, r2_values, 'g-', label='R² (Model Fit)')
        plt.ylabel('R² Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot volatility
        plt.subplot(3, 1, 3)
        plt.plot(timestamps, volatilities, 'k-', label='Market Volatility')
        plt.ylabel('Volatility')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format dates on x-axis
        date_format = DateFormatter('%Y-%m-%d')
        for ax in plt.gcf().axes:
            ax.xaxis.set_major_formatter(date_format)
        
        plt.tight_layout()
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Analyze parameter stability
        gamma_stability = {
            "mean": np.mean(gamma_scales),
            "std": np.std(gamma_scales),
            "min": np.min(gamma_scales),
            "max": np.max(gamma_scales),
            "coefficient_of_variation": np.std(gamma_scales) / np.mean(gamma_scales) if np.mean(gamma_scales) > 0 else 0
        }
        
        eta_stability = {
            "mean": np.mean(eta_scales),
            "std": np.std(eta_scales),
            "min": np.min(eta_scales),
            "max": np.max(eta_scales),
            "coefficient_of_variation": np.std(eta_scales) / np.mean(eta_scales) if np.mean(eta_scales) > 0 else 0
        }
        
        # Correlation analysis
        correlations = {
            "gamma_eta_correlation": np.corrcoef(gamma_scales, eta_scales)[0, 1],
            "gamma_volatility_correlation": np.corrcoef(gamma_scales, volatilities)[0, 1],
            "eta_volatility_correlation": np.corrcoef(eta_scales, volatilities)[0, 1],
            "r2_volatility_correlation": np.corrcoef(r2_values, volatilities)[0, 1]
        }
        
        # Return results
        return {
            "symbol": symbol,
            "window_size_days": window_size,
            "window_count": len(timestamps),
            "date_range": {
                "start": timestamps[0].strftime('%Y-%m-%d') if len(timestamps) > 0 else None,
                "end": timestamps[-1].strftime('%Y-%m-%d') if len(timestamps) > 0 else None
            },
            "gamma_scale_stability": gamma_stability,
            "eta_scale_stability": eta_stability,
            "correlations": correlations,
            "visualization": img_str
        }

# Example usage
if __name__ == "__main__":
    validator = ModelValidator()
    
    try:
        results = validator.validate_almgren_chriss()
        print("Validation results:")
        for market, result in results.items():
            print(f"\n{market}:")
            print(f"  MAPE: {result['mape']:.2f}%")
            print(f"  MAE: {result['mae']:.2f} bps")
            print(f"  R²: {result['r2']:.4f}")
            print(f"  Sample count: {result['sample_count']}")
    except Exception as e:
        print(f"Validation failed: {e}") 