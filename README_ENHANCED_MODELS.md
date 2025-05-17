# Enhanced Regression Models for Slippage Prediction

This documentation describes the enhanced regression models implemented for slippage prediction in the Market Order Simulator.

## Overview

The enhanced regression models extend the basic linear regression approach with:

1. **Feature importance visualization** - Helps users understand what drives execution costs
2. **Advanced regression techniques** - Includes Random Forest and Gradient Boosting models
3. **Time-series forecasting** - Analyzes temporal patterns in market behavior

## Features

### Multiple Regression Techniques

- **Linear Regression**: The baseline model, good for simple slippage prediction
- **Random Forest**: An ensemble learning method using multiple decision trees
- **Gradient Boosting**: An advanced technique that builds trees sequentially

### Feature Importance Analysis

The implementation includes visualizations that show which factors contribute most to slippage predictions:

- **Bar charts**: Shows relative importance of each feature
- **Pie charts**: Shows proportional contribution of features

Common important features for slippage prediction include:
- Order size relative to market depth
- Spread width in basis points
- Market volatility
- Depth imbalance between bids and asks

### Time-Series Analysis

The time-series forecasting capabilities include:

- **Historical pattern analysis**: Identifies trends and seasonality in slippage data
- **Future projection**: Predicts expected slippage trends over time
- **Temporal feature extraction**: Uses time-based features like hour of day, day of week, etc.

## Implementation Details

### Classes

1. **EnhancedSlippageModel**
   - Extends the base `SlippageModel`
   - Trains and manages multiple regression models
   - Provides feature importance visualization
   - Compares model performance

2. **TimeSeriesSlippageModel**
   - Specialized model for time-series forecasting
   - Creates temporal features from timestamp data
   - Generates slippage trend forecasts

### Primary Functions

- `simulate_market_order_with_enhanced_ml()`: Simulates market orders using advanced ML models
- `plot_feature_importance()`: Visualizes feature importance
- `compare_model_performance()`: Creates comparison charts of different models
- `forecast_slippage_trend()`: Predicts future slippage trends

## Usage

### Selecting a Model

In the Streamlit interface, you can select different regression model types:

1. **Standard Linear**: The original linear regression model
2. **Random Forest**: May provide better accuracy for complex relationships
3. **Gradient Boosting**: Often the best performer for slippage prediction

### Visualizing Feature Importance

The "Model Analysis" tab provides tools to:
- View feature importance for different model types
- Compare performance metrics between models
- Generate time-series forecasts for future slippage trends

### Interpreting Results

The model comparison metrics include:
- **R² Score**: Measures how well the model explains variance (higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

## Requirements

The enhanced models require the following Python packages:
- scikit-learn
- pandas
- numpy
- plotly
- matplotlib
- joblib

## Future Enhancements

Potential improvements to the slippage prediction models:
- Neural network implementation for more complex pattern recognition
- Hyperparameter optimization for model tuning
- Market-specific model training for different assets
- Cross-market slippage correlations 