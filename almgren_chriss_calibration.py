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
    
    def handle_outliers(self, data: pd.DataFrame, columns: List[str], method: str = 'iqr', k: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers in the calibration data.
        
        Args:
            data: DataFrame containing the calibration data
            columns: List of columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore', or 'percentile')
            k: Threshold factor for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        clean_data = data.copy()
        
        for col in columns:
            if col not in clean_data.columns:
                continue
            
            if method == 'iqr':
                # Interquartile range method
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - k * IQR
                upper_bound = Q3 + k * IQR
                
                # Identify outliers
                outliers = (clean_data[col] < lower_bound) | (clean_data[col] > upper_bound)
                print(f"Found {outliers.sum()} outliers in {col} using IQR method")
                
            elif method == 'zscore':
                # Z-score method
                mean = clean_data[col].mean()
                std = clean_data[col].std()
                z_scores = (clean_data[col] - mean) / std
                outliers = abs(z_scores) > k
                print(f"Found {outliers.sum()} outliers in {col} using Z-score method")
                
            elif method == 'percentile':
                # Percentile method
                lower_bound = clean_data[col].quantile(0.01)
                upper_bound = clean_data[col].quantile(0.99)
                outliers = (clean_data[col] < lower_bound) | (clean_data[col] > upper_bound)
                print(f"Found {outliers.sum()} outliers in {col} using percentile method")
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            # Handle outliers - can either remove or cap
            if outliers.sum() > 0:
                # Option 1: Cap outliers at the bounds
                clean_data.loc[clean_data[col] < lower_bound, col] = lower_bound
                clean_data.loc[clean_data[col] > upper_bound, col] = upper_bound
                
                # Option 2: Remove outliers (commented out)
                # clean_data = clean_data[~outliers]
        
        return clean_data

    def log_transform_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply log transformation to improve linearity for regression.
        
        Args:
            data: DataFrame containing the calibration data
            columns: List of columns to log transform
            
        Returns:
            DataFrame with log-transformed columns
        """
        transformed_data = data.copy()
        
        for col in columns:
            if col in transformed_data.columns:
                # Ensure values are positive for log transform
                min_val = transformed_data[col].min()
                if min_val <= 0:
                    # Add small offset to make all values positive
                    offset = abs(min_val) + 1e-6
                    transformed_data[col] = transformed_data[col] + offset
                
                # Apply log transform
                transformed_data[f'log_{col}'] = np.log(transformed_data[col])
        
        return transformed_data

    def calibrate_parameters_robust(self, symbol: str = None, outlier_method: str = 'iqr', 
                                   log_transform: bool = True, constraint_bounds: bool = True):
        """
        Robustly calibrate Almgren-Chriss parameters with outlier handling and constraints.
        
        Args:
            symbol: Market symbol to calibrate (e.g., 'BTC-USDT')
            outlier_method: Method for outlier detection ('iqr', 'zscore', or 'percentile')
            log_transform: Whether to apply log transformation for better linearity
            constraint_bounds: Whether to constrain parameters within typical bounds
            
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
            
            # Handle outliers in key calibration columns
            market_data = self.handle_outliers(
                market_data, 
                columns=['slippage_bps', 'quantity_usd', 'volatility'],
                method=outlier_method
            )
            
            # Apply log transformation if requested
            if log_transform:
                market_data = self.log_transform_data(
                    market_data, 
                    columns=['slippage_bps', 'quantity_usd']
                )
                # Use transformed columns for regression
                slippage_col = 'log_slippage_bps'
                quantity_col = 'log_quantity_usd'
            else:
                slippage_col = 'slippage_bps'
                quantity_col = 'quantity_usd'
            
            # Split data for training and validation
            train_data, val_data = train_test_split(market_data, test_size=0.3, random_state=42)
            
            # Create a sample order book for this market
            order_book = OrderBook(market)
            
            # Ensure we have numeric columns with proper values
            for df in [train_data, val_data]:
                for col in ['bid_price', 'ask_price', 'bid_volume', 'ask_volume', 'volatility']:
                    if col not in df.columns:
                        df[col] = 0.0  # Add default values if missing
                    else:
                        # Ensure values are numeric
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Fill any NaN with reasonable defaults
                        if col in ['bid_price', 'ask_price']:
                            df[col].fillna(50000.0, inplace=True)
                        elif col in ['bid_volume', 'ask_volume']:
                            df[col].fillna(1.0, inplace=True)
                        elif col == 'volatility':
                            df[col].fillna(0.3, inplace=True)
            
            # Calculate average order book state from the data
            avg_bid_price = train_data['bid_price'].mean()
            avg_ask_price = train_data['ask_price'].mean()
            avg_bid_qty = train_data['bid_volume'].mean()
            avg_ask_qty = train_data['ask_volume'].mean()
            
            # Create a sample order book
            sample_data = {
                'bids': [[str(avg_bid_price), str(avg_bid_qty)]],
                'asks': [[str(avg_ask_price), str(avg_ask_qty)]]
            }
            order_book.update(sample_data)
            
            # Get quantities and side info from training data
            quantities = train_data['quantity_usd'].values
            
            # Get the most common side or default to "buy"
            if 'side' in train_data.columns:
                # Use most common side
                sides = train_data['side'].value_counts()
                side = sides.index[0] if len(sides) > 0 else "buy"
            else:
                side = "buy"
            
            # Get volatility or use default
            if 'volatility' in train_data.columns:
                volatility = train_data['volatility'].mean()
            else:
                volatility = 0.3  # Default volatility
            
            # Initial parameter guess - more reasonable starting point
            initial_params = [0.8, 1.2]  # [gamma_scale, eta_scale]
            
            # Use robust optimization with constraints
            from scipy.optimize import minimize
            
            if constraint_bounds:
                # Add constraints to keep parameters in reasonable ranges
                bounds = [(0.1, 5.0), (0.1, 5.0)]  # gamma_scale, eta_scale
                options = {'maxiter': 100}
                method = 'L-BFGS-B'
            else:
                bounds = [(0.01, 10.0), (0.01, 10.0)]
                options = None
                method = 'Nelder-Mead'
            
            # Optimize the parameters with robust approach
            try:
                result = minimize(
                    self._objective_function,
                    initial_params,
                    args=(train_data, side, order_book, quantities, volatility),
                    method=method,
                    bounds=bounds,
                    options=options
                )
                
                # Extract optimized parameters
                gamma_scale, eta_scale = result.x
                
                # Apply typical bounds if requested
                if constraint_bounds:
                    gamma_scale = max(0.1, min(gamma_scale, 5.0))
                    eta_scale = max(0.1, min(eta_scale, 5.0))
                
                # Calculate validation error using original (untransformed) values
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
                    'validation_r2': val_r2,
                    'log_transform_used': log_transform,
                    'outlier_method': outlier_method
                }
                
                calibration_results[market] = {
                    'gamma_scale': gamma_scale,
                    'eta_scale': eta_scale,
                    'training_records': len(train_data),
                    'validation_records': len(val_data),
                    'validation_mape': val_mape,
                    'validation_mae': val_mae,
                    'validation_r2': val_r2,
                    'optimization_success': result.success,
                    'optimization_message': str(result.message) if hasattr(result, 'message') else None
                }
            except Exception as e:
                print(f"Optimization error for {market}: {e}")
                # Use default parameters
                calibration_results[market] = {
                    'gamma_scale': 1.0,
                    'eta_scale': 1.0,
                    'training_records': len(train_data),
                    'validation_records': len(val_data),
                    'validation_mape': 0.0,
                    'validation_mae': 0.0,
                    'validation_r2': 0.0,
                    'optimization_success': False,
                    'optimization_message': str(e)
                }
        
        # Save calibrated parameters
        self._save_params()
        
        return calibration_results

    def get_dynamic_parameters(self, symbol: str, volatility: float, 
                              time_of_day: float = None, recent_volume: float = None) -> Dict[str, float]:
        """
        Get dynamically adjusted parameters based on current market conditions.
        
        Args:
            symbol: Market symbol (e.g., 'BTC-USDT')
            volatility: Current market volatility
            time_of_day: Hour of day (0-23.99)
            recent_volume: Recent trading volume relative to average
            
        Returns:
            Dictionary with dynamically adjusted parameters
        """
        # Get base calibrated parameters
        base_params = self.get_market_parameters(symbol, volatility)
        gamma_scale = base_params.get('gamma_scale', 1.0)
        eta_scale = base_params.get('eta_scale', 1.0)
        volatility_adjustment = base_params.get('volatility_adjustment', 1.0)
        
        # Apply time-of-day adjustment if provided
        tod_adjustment = 1.0
        if time_of_day is not None:
            # Higher impact during market open/close or low liquidity periods
            # This is a simple model - can be refined with actual market data
            hour = time_of_day
            if 0 <= hour < 2 or 22 <= hour < 24:  # Late night/early morning
                tod_adjustment = 1.2  # 20% higher impact
            elif 8 <= hour < 10 or 14 <= hour < 16:  # Market open/close periods
                tod_adjustment = 1.1  # 10% higher impact
            elif 10 <= hour < 14:  # Mid-day
                tod_adjustment = 0.95  # 5% lower impact
        
        # Apply volume adjustment if provided
        volume_adjustment = 1.0
        if recent_volume is not None:
            # Lower impact when volume is high, higher when volume is low
            if recent_volume > 1.5:  # 50% above average volume
                volume_adjustment = 0.9  # 10% lower impact
            elif recent_volume < 0.7:  # 30% below average volume
                volume_adjustment = 1.15  # 15% higher impact
        
        # Combine all adjustments
        final_gamma_scale = gamma_scale * tod_adjustment * volume_adjustment
        final_eta_scale = eta_scale * tod_adjustment  # Permanent impact less affected by time
        
        return {
            'gamma_scale': final_gamma_scale,
            'eta_scale': final_eta_scale,
            'volatility_adjustment': volatility_adjustment,
            'tod_adjustment': tod_adjustment,
            'volume_adjustment': volume_adjustment
        }

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
        results = calibrator.calibrate_parameters_robust()
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