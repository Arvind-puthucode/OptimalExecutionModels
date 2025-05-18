"""
Almgren-Chriss Model Example

This script demonstrates how to:
1. Calibrate the Almgren-Chriss market impact model
2. Generate optimal execution schedules
3. Compare different market impact models
4. Visualize the results
"""

from typing import Dict, Any
import json
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

# Import local modules
from orderbook import OrderBook
import almgren_chriss as ac
from almgren_chriss_calibration import AlmgrenChrissCalibrator
from model_validation import ModelValidator
from market_order_simulator import ModelSelectionStrategy


def calibrate_model():
    """Calibrate the Almgren-Chriss model and show results."""
    print("\n1. CALIBRATING ALMGREN-CHRISS MODEL")
    print("==================================")
    
    calibrator = AlmgrenChrissCalibrator()
    
    try:
        # Use robust calibration with outlier handling and log transformation
        results = calibrator.calibrate_parameters_robust()
        
        print("Calibration results:")
        for market, result in results.items():
            print(f"\n{market}:")
            print(f"  Gamma Scale: {result['gamma_scale']:.4f}")
            print(f"  Eta Scale: {result['eta_scale']:.4f}")
            print(f"  Validation MAPE: {result['validation_mape']:.2f}%")
            print(f"  Validation MAE: {result['validation_mae']:.2f} bps")
            print(f"  Validation R²: {result['validation_r2']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Calibration failed: {e}")
        return {}


def generate_execution_schedule(symbol="BTC-USDT", quantity=5.0, time_horizon=4.0):
    """Generate optimal execution schedule for a large order."""
    print("\n2. GENERATING OPTIMAL EXECUTION SCHEDULE")
    print("======================================")
    
    # Create a sample order book
    order_book = OrderBook(symbol)
    sample_data = {
        'bids': [['49900', '1.5'], ['49800', '2.0']],
        'asks': [['50000', '1.0'], ['50100', '3.0']]
    }
    order_book.update(sample_data)
    
    # Sample parameters
    side = "buy"
    volatility = 0.35  # 35% annualized
    
    # Get calibrated parameters
    calibrator = AlmgrenChrissCalibrator()
    market_params = calibrator.get_market_parameters(symbol, volatility)
    
    # Calculate depth metrics
    depth = ac.calculate_market_depth(order_book)
    
    # Get impact parameters
    gamma = ac.estimate_temporary_impact(depth, volatility)
    eta = ac.estimate_permanent_impact(depth)
    
    # Apply calibration
    gamma_scale = market_params.get('gamma_scale', 1.0)
    eta_scale = market_params.get('eta_scale', 1.0)
    gamma *= gamma_scale
    eta *= eta_scale
    
    print(f"Market parameters for {symbol}:")
    print(f"  Temporary impact factor (gamma): {gamma:.8f}")
    print(f"  Permanent impact factor (eta): {gamma:.8f}")
    print(f"  Volatility: {volatility}")
    
    # Generate schedules with different risk aversion levels
    lambda_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    schedules = []
    
    print(f"\nOptimal execution schedules for {quantity} {symbol} over {time_horizon} hours:")
    print("-" * 70)
    print(f"{'Time (h)':10s} | {'Linear':10s} | {'λ=0.0':10s} | {'λ=1.0':10s} | {'λ=5.0':10s}")
    print("-" * 70)
    
    # Generate time points
    time_points = np.linspace(0, time_horizon, 11)
    
    # Calculate linear TWAP for reference
    twap_inventory = quantity * (1 - time_points / time_horizon)
    
    # Generate schedules
    for lambda_risk in lambda_values:
        schedule = ac.generate_optimal_execution_schedule(
            quantity, time_horizon, eta, gamma, volatility, lambda_risk
        )
        schedules.append(schedule)
    
    # Display schedules
    for i in range(len(time_points)):
        time = time_points[i]
        twap = twap_inventory[i]
        risk_0 = schedules[0]['inventory'][i]
        risk_1 = schedules[2]['inventory'][i]  # Lambda = 1.0
        risk_5 = schedules[4]['inventory'][i]  # Lambda = 5.0
        
        print(f"{time:10.2f} | {twap:10.4f} | {risk_0:10.4f} | {risk_1:10.4f} | {risk_5:10.4f}")
    
    print("\nExpected costs in basis points:")
    for i, lambda_risk in enumerate(lambda_values):
        print(f"  λ={lambda_risk:.1f}: {schedules[i]['expected_cost_bps']:.2f} bps")
    
    # Plot schedules
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, twap_inventory, 'k--', label='TWAP (Linear)')
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, lambda_risk in enumerate(lambda_values):
        schedule = schedules[i]
        plt.plot(schedule['time_points'], schedule['inventory'], 
                marker='o', color=colors[i], 
                label=f'λ={lambda_risk:.1f} (κ={schedule["kappa"]:.4f})')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Remaining Quantity')
    plt.title(f'Optimal Execution Schedules for {quantity} {symbol}\n'
              f'η={eta:.6f}, γ={gamma:.6f}, σ={volatility:.2f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot to file
    plt.savefig("optimal_execution_schedule.png")
    plt.close()
    
    print("\nPlot saved to 'optimal_execution_schedule.png'")
    
    return schedules


def compare_market_impact_models(symbol="BTC-USDT"):
    """Compare different market impact models."""
    print("\n3. COMPARING MARKET IMPACT MODELS")
    print("==============================")
    
    # Create sample order book
    order_book = OrderBook(symbol)
    sample_data = {
        'bids': [['49900', '1.5'], ['49800', '2.0']],
        'asks': [['50000', '1.0'], ['50100', '3.0']]
    }
    order_book.update(sample_data)
    
    # Sample parameters
    side = "buy"
    quantities = [1000, 10000, 50000, 100000, 500000]  # USD
    volatility = 0.35  # 35% annualized
    avg_daily_volume = 5000000  # $5M daily volume
    
    try:
        # Create model selector
        model_selector = ModelSelectionStrategy()
        
        # Compare models for different order sizes
        print(f"Model selection for different order sizes (Symbol: {symbol}):")
        print("-" * 90)
        print(f"{'Order Size ($)':15s} | {'Selected Model':20s} | {'Rationale':50s}")
        print("-" * 90)
        
        for quantity in quantities:
            selection = model_selector.select_model(
                order_book, side, quantity, volatility, avg_daily_volume
            )
            
            print(f"{quantity:15.0f} | {selection['selected_model']:20s} | {selection['rationale']}")
        
        # Run detailed comparison for medium-sized order
        medium_order = 50000  # $50k
        print(f"\nDetailed comparison for ${medium_order} order:")
        
        comparison = model_selector.compare_all_models(
            order_book, side, medium_order, volatility, avg_daily_volume
        )
        
        # Extract costs from each model
        models = []
        costs = []
        
        for model_name, result in comparison.items():
            if model_name in ["orderbook", "ml_basic", "ml_enhanced", "almgren_chriss"]:
                if isinstance(result, dict) and "error" not in result:
                    cost = None
                    if "slippage_bps" in result:
                        cost = result["slippage_bps"]
                    elif "expected_cost_bps" in result:
                        cost = result["expected_cost_bps"]
                    elif "impact_bps" in result:
                        cost = result["impact_bps"]
                    
                    if cost is not None:
                        models.append(model_name)
                        costs.append(cost)
        
        # Print costs
        if models:
            print("-" * 50)
            print(f"{'Model':20s} | {'Expected Cost (bps)':20s}")
            print("-" * 50)
            
            for i in range(len(models)):
                print(f"{models[i]:20s} | {costs[i]:20.2f}")
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(models, costs, color=['blue', 'green', 'purple', 'red'][:len(models)])
            plt.ylabel('Expected Cost (bps)')
            plt.title(f'Market Impact Model Comparison\n${medium_order} {side.upper()} Order in {symbol}')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Save plot to file
            plt.savefig("model_comparison.png")
            plt.close()
            
            print("\nComparison plot saved to 'model_comparison.png'")
        else:
            print("\nNo valid model comparisons available.")
    except Exception as e:
        print(f"\nError in model comparison: {e}")
    
    # Run comparison for large order with execution timeframe
    try:
        large_order = 500000  # $500k
        execution_time = 2.0  # 2 hours
        
        print(f"\nOptimal execution comparison for ${large_order} order over {execution_time} hours:")
        
        # Get execution strategies comparison
        ac_comparison = ac.compare_execution_strategies(
            order_book, side, large_order, execution_time, volatility
        )
        
        # Print execution cost comparison
        print("-" * 60)
        print(f"{'Execution Strategy':30s} | {'Expected Cost (bps)':20s}")
        print("-" * 60)
        
        immediate = ac_comparison["immediate_execution"]["expected_cost_bps"]
        twap = ac_comparison["twap_execution"]["expected_cost_bps"]
        optimal = ac_comparison["optimal_execution"]["expected_cost_bps"]
        
        print(f"{'Immediate Execution':30s} | {immediate:20.2f}")
        print(f"{'Linear TWAP':30s} | {twap:20.2f}")
        print(f"{'Optimal Almgren-Chriss':30s} | {optimal:20.2f}")
    except Exception as e:
        print(f"\nError in execution strategy comparison: {e}")
    
    return comparison


def validate_model_accuracy():
    """Validate model accuracy against historical data."""
    print("\n4. VALIDATING MODEL ACCURACY")
    print("==========================")
    
    validator = ModelValidator()
    
    try:
        # Validate Almgren-Chriss model
        ac_results = validator.validate_almgren_chriss(days_back=30)
        
        print("Almgren-Chriss model validation results:")
        for market, result in ac_results.items():
            print(f"\n{market}:")
            print(f"  MAPE: {result['mape']:.2f}%")
            print(f"  MAE: {result['mae']:.2f} bps")
            print(f"  R²: {result['r2']:.4f}")
            print(f"  Sample count: {result['sample_count']}")
        
        # Compare with other models
        model_comparison = validator.compare_model_predictions(days_back=30)
        
        print("\nModel comparison results:")
        print(f"  Symbol: {model_comparison['symbol']}")
        print(f"  Data points: {model_comparison['data_points']}")
        
        print("\nAccuracy metrics:")
        print("-" * 60)
        print(f"{'Model':20s} | {'MAE (bps)':12s} | {'MAPE (%)':12s} | {'R²':12s}")
        print("-" * 60)
        
        for model, metrics in model_comparison.items():
            if model in ["orderbook_model", "ml_model", "almgren_chriss_model", "enhanced_ml_model"]:
                print(f"{model:20s} | {metrics['mae']:12.2f} | {metrics['mape']:12.2f} | {metrics['r2']:12.4f}")
        
        print("\nModel comparison plot saved to 'model_validation.png'")
        
        # Save the visualization to file
        if "visualization" in model_comparison:
            # Extract base64 data
            img_data = model_comparison["visualization"].split(",")[1]
            with open("model_validation.png", "wb") as f:
                f.write(base64.b64decode(img_data))
        
        return model_comparison
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return {}


if __name__ == "__main__":
    print("ALMGREN-CHRISS MODEL DEMONSTRATION")
    print("=================================")
    
    # Run all demonstrations
    calibrate_model()
    generate_execution_schedule()
    compare_market_impact_models()
    
    try:
        validate_model_accuracy()
    except Exception as e:
        print(f"\nNote: Model validation requires historical data. Error: {e}")
    
    print("\nDemonstration complete!") 