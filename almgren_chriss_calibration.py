"""
Almgren-Chriss Model Parameter Calibration

This module implements automatic parameter calibration for the Almgren-Chriss
market impact model based on historical execution data.

The calibration process optimizes the gamma (temporary impact) and eta (permanent impact)
parameters for different markets to minimize prediction errors.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import json
import os
from datetime import datetime
import math
from scipy.optimize import minimize

# Import local modules
from orderbook import OrderBook
import almgren_chriss as ac


class AlmgrenChrissCalibrator:
    """Calibrates parameters for the Almgren-Chriss market impact model."""
    
    def __init__(self, data_path: str = "slippage_data.csv", 
                 params_path: str = "almgren_chriss_params.json"):
        self.data_path = data_path
        self.params_path = params_path
        self.market_params = {}
        self._load_params()
    
    def _load_params(self):
        """Load calibrated parameters from file if available."""
        if os.path.exists(self.params_path):
            try:
                with open(self.params_path, 'r') as f:
                    self.market_params = json.load(f)
            except Exception as e:
                print(f"Error loading parameters: {e}")
                self.market_params = {}
    
    def _save_params(self):
        """Save calibrated parameters to file."""
        try:
            with open(self.params_path, 'w') as f:
                json.dump(self.market_params, f, indent=2)
        except Exception as e:
            print(f"Error saving parameters: {e}")
    
    def load_historical_data(self):
        """Load historical execution data for calibration."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Historical data file not found: {self.data_path}")
        
        try:
            data = pd.read_csv(self.data_path)
            # Filter out rows with missing or invalid values
            data = data.dropna(subset=['slippage_bps', 'quantity_usd', 'spread_bps', 'volatility'])
            return data
        except Exception as e:
            raise ValueError(f"Error loading historical data: {e}")
    
    def _objective_function(self, params, data, side, order_book, quantities, volatility):
        """
        Objective function for optimization.
        
        Args:
            params: List of parameters [gamma_scale, eta_scale]
            data: Historical execution data
            side: Trade side (buy/sell)
            order_book: OrderBook state
            quantities: Order quantities
            volatility: Market volatility
            
        Returns:
            Mean absolute percentage error between predicted and actual slippage
        """
        gamma_scale, eta_scale = params
        
        # Ensure parameters are within reasonable bounds
        gamma_scale = max(0.01, min(10.0, gamma_scale))
        eta_scale = max(0.01, min(10.0, eta_scale))
        
        actual_slippage = data['slippage_bps'].values
        predicted_slippage = []
        
        # Calculate market depth once
        depth = ac.calculate_market_depth(order_book)
        
        for i, quantity in enumerate(quantities):
            # Calculate base gamma and eta
            base_gamma = ac.estimate_temporary_impact(depth, volatility)
            base_eta = ac.estimate_permanent_impact(depth)
            
            # Apply scaling factors
            gamma = base_gamma * gamma_scale
            eta = base_eta * eta_scale
            
            # Calculate predicted impact
            direction = 1 if side.lower() == 'buy' else -1
            temporary_impact = direction * gamma * quantity
            permanent_impact = direction * eta * quantity
            total_impact = permanent_impact + 0.5 * temporary_impact
            
            # Convert to basis points
            predicted_bps = total_impact * 10000
            predicted_slippage.append(predicted_bps)
        
        # Calculate error
        mape = mean_absolute_percentage_error(actual_slippage, predicted_slippage)
        return mape
    
    def calibrate_parameters(self, symbol: str = None):
        """
        Calibrate Almgren-Chriss parameters using historical data.
        
        Args:
            symbol: Market symbol to calibrate (e.g., 'BTC-USDT'). 
                   If None, calibrates for all markets with sufficient data.
                   
        Returns:
            Dictionary with calibration results
        """
        data = self.load_historical_data()
        
        # If symbol is provided, filter data for that market
        if symbol:
            market_data = data[data['symbol'] == symbol]
            if len(market_data) < 30:  # Need sufficient data for calibration
                raise ValueError(f"Insufficient data for {symbol}. Need at least 30 records.")
            markets = [symbol]
        else:
            # Group by symbol and filter out markets with insufficient data
            markets = [s for s, g in data.groupby('symbol') if len(g) >= 30]
        
        calibration_results = {}
        
        for market in markets:
            market_data = data[data['symbol'] == market]
            
            # Split data for training and validation
            train_data, val_data = train_test_split(market_data, test_size=0.3, random_state=42)
            
            # Create a sample order book for this market
            order_book = OrderBook(market)
            
            # Calculate average order book state from the data
            # In a real implementation, you'd use actual order book snapshots
            avg_bid_price = train_data.get('bid_price', 50000).mean()
            avg_ask_price = train_data.get('ask_price', 50100).mean()
            avg_bid_qty = train_data.get('bid_volume', 1.0).mean()
            avg_ask_qty = train_data.get('ask_volume', 1.0).mean()
            
            # Create a sample order book
            sample_data = {
                'bids': [[str(avg_bid_price), str(avg_bid_qty)]],
                'asks': [[str(avg_ask_price), str(avg_ask_qty)]]
            }
            order_book.update(sample_data)
            
            # Get quantities and side info from training data
            quantities = train_data['quantity_usd'].values
            side = train_data['side'].iloc[0]  # Use most common side
            volatility = train_data['volatility'].mean()
            
            # Initial parameter guess
            initial_params = [1.0, 1.0]  # [gamma_scale, eta_scale]
            
            # Optimize the parameters
            result = minimize(
                self._objective_function,
                initial_params,
                args=(train_data, side, order_book, quantities, volatility),
                method='Nelder-Mead',
                bounds=[(0.01, 10.0), (0.01, 10.0)]
            )
            
            # Extract optimized parameters
            gamma_scale, eta_scale = result.x
            
            # Calculate validation error
            val_quantities = val_data['quantity_usd'].values
            val_actual = val_data['slippage_bps'].values
            val_predicted = []
            
            # Calculate market depth for validation
            depth = ac.calculate_market_depth(order_book)
            
            for quantity in val_quantities:
                base_gamma = ac.estimate_temporary_impact(depth, volatility)
                base_eta = ac.estimate_permanent_impact(depth)
                
                gamma = base_gamma * gamma_scale
                eta = base_eta * eta_scale
                
                direction = 1 if side.lower() == 'buy' else -1
                temporary_impact = direction * gamma * quantity
                permanent_impact = direction * eta * quantity
                total_impact = permanent_impact + 0.5 * temporary_impact
                
                val_predicted.append(total_impact * 10000)  # Convert to basis points
            
            # Calculate validation metrics
            val_mape = mean_absolute_percentage_error(val_actual, val_predicted)
            val_mae = mean_absolute_error(val_actual, val_predicted)
            val_r2 = r2_score(val_actual, val_predicted)
            
            # Store calibrated parameters
            self.market_params[market] = {
                'gamma_scale': gamma_scale,
                'eta_scale': eta_scale,
                'volatility_base': volatility,
                'last_calibration': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sample_count': len(train_data),
                'validation_mape': val_mape,
                'validation_mae': val_mae,
                'validation_r2': val_r2
            }
            
            calibration_results[market] = {
                'gamma_scale': gamma_scale,
                'eta_scale': eta_scale,
                'training_records': len(train_data),
                'validation_records': len(val_data),
                'validation_mape': val_mape,
                'validation_mae': val_mae,
                'validation_r2': val_r2
            }
        
        # Save calibrated parameters
        self._save_params()
        
        return calibration_results
    
    def get_market_parameters(self, symbol: str, volatility: float) -> Dict[str, float]:
        """
        Get calibrated parameters for a specific market.
        
        Args:
            symbol: Market symbol (e.g., 'BTC-USDT')
            volatility: Current market volatility
            
        Returns:
            Dictionary with calibrated gamma and eta parameters
        """
        if symbol in self.market_params:
            params = self.market_params[symbol]
            gamma_scale = params.get('gamma_scale', 1.0)
            eta_scale = params.get('eta_scale', 1.0)
            volatility_base = params.get('volatility_base', volatility)
            
            # Adjust for current volatility relative to base volatility used in calibration
            volatility_adjustment = math.sqrt(volatility / volatility_base) if volatility_base > 0 else 1.0
            
            return {
                'gamma_scale': gamma_scale,
                'eta_scale': eta_scale,
                'volatility_adjustment': volatility_adjustment
            }
        else:
            # Return default values if no calibrated parameters exist
            return {
                'gamma_scale': 1.0,
                'eta_scale': 1.0,
                'volatility_adjustment': 1.0
            }


def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


# Example usage
if __name__ == "__main__":
    calibrator = AlmgrenChrissCalibrator()
    
    try:
        results = calibrator.calibrate_parameters()
        print("Calibration results:")
        for market, result in results.items():
            print(f"\n{market}:")
            print(f"  Gamma Scale: {result['gamma_scale']:.4f}")
            print(f"  Eta Scale: {result['eta_scale']:.4f}")
            print(f"  Validation MAPE: {result['validation_mape']:.2f}%")
            print(f"  Validation MAE: {result['validation_mae']:.2f} bps")
            print(f"  Validation R²: {result['validation_r2']:.4f}")
    except Exception as e:
        print(f"Calibration failed: {e}") 