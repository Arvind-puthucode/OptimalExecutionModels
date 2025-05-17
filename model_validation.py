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
            
            # Create a sample order book for this market
            order_book = OrderBook(market)
            
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
            
            # Prepare data for validation
            quantities = market_data['quantity_usd'].values
            sides = market_data['side'].values
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
            
            # Calculate validation metrics
            predicted_slippage_bps = np.array(predicted_slippage_bps)
            mape = self.calculate_mape(actual_slippage_bps, predicted_slippage_bps)
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
            
            # Add new validation record
            self.validation_stats[market].append({
                'mape': mape,
                'mae': mae,
                'r2': r2,
                'sample_count': len(market_data),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Only keep the last 30 validation records
            if len(self.validation_stats[market]) > 30:
                self.validation_stats[market] = self.validation_stats[market][-30:]
        
        # Save validation stats
        self._save_stats()
        
        return validation_results
    
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