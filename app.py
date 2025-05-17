import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import logging
from orderbook import OrderBook
from orderbook_client import OrderBookClient  # type: ignore
from market_order_simulator import simulate_market_order, calculate_order_metrics, simulate_market_order_with_ml
from regression_models import SlippageModel, SlippageDataCollector
# Import our enhanced regression models
from enhanced_regression_models import EnhancedSlippageModel, TimeSeriesSlippageModel, simulate_market_order_with_enhanced_ml
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import seaborn as sns

# Import our new modules if available
try:
    from almgren_chriss import simulate_market_order_with_impact
    from almgren_chriss_calibration import AlmgrenChrissCalibrator
    from model_validation import ModelValidator
    ALMGREN_CHRISS_AVAILABLE = True
except ImportError:
    ALMGREN_CHRISS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)

# Setup page config
st.set_page_config(
    page_title="Market Order Simulator",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Market Order Simulator")
st.markdown("Simulate market orders and analyze their execution based on real-time order book data.")


# Initialize the OrderBook global state
if 'order_book' not in st.session_state:
    st.session_state.order_book = OrderBook("BTC-USDT-SWAP")
    
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = 0
    
if 'connected' not in st.session_state:
    st.session_state.connected = False
    
if 'connection_attempts' not in st.session_state:
    st.session_state.connection_attempts = 0


def update_session_order_book():
    """Update the session state order book with data from the client"""
    # Get client instance
    client = OrderBookClient.get_instance()
    
    # Only update if there's data
    if client.has_data():
        # Get data from client
        bids, asks = client.get_bids_asks()
        
        # Convert to format expected by OrderBook.update()
        processed_data = {
            'bids': [[str(price), str(qty)] for price, qty in bids.items()],
            'asks': [[str(price), str(qty)] for price, qty in asks.items()]
        }
        
        # Update session state order book
        st.session_state.order_book.update(processed_data)
        st.session_state.last_update_time = client.last_update_time
        return True
    return False


def on_orderbook_update(data):
    """Callback function to handle order book updates - only logging, no session state access"""
    try:
        logging.info(f"Received order book update: {len(data.get('bids', []))} bids, {len(data.get('asks', []))} asks")
    except Exception as e:
        logging.error(f"Error in order book update callback: {e}")


def init_client():
    """Initialize order book client and start connection"""
    # Get client singleton
    client = OrderBookClient.get_instance()
    
    # Register callback for order book updates (just for logging)
    client.remove_callback(on_orderbook_update)  # Remove if already registered
    client.add_callback(on_orderbook_update)
    
    # Start connection if not already connected
    if not client.is_connected():
        success = client.connect()
        st.session_state.connected = success
        if success:
            # Increment connection attempts on success (to track reconnections)
            st.session_state.connection_attempts += 1
        return success
    else:
        st.session_state.connected = True
        return True


def plot_orderbook(order_book, depth=15, highlight_levels=None):
    """Create a Plotly visualization of the order book"""
    bids = order_book.get_bids(depth)
    asks = order_book.get_asks(depth)
    
    # Create bid and ask dataframes
    bid_prices = [bid[0] for bid in bids]
    bid_sizes = [bid[1] for bid in bids]
    bid_totals = np.cumsum(bid_sizes)
    
    ask_prices = [ask[0] for ask in asks]
    ask_sizes = [ask[1] for ask in asks]
    ask_totals = np.cumsum(ask_sizes)
    
    # Create the figure
    fig = go.Figure()
    
    # Add bid bars in green
    fig.add_trace(go.Bar(
        x=bid_prices,
        y=bid_sizes,
        name='Bids',
        marker_color='rgba(0, 128, 0, 0.7)'
    ))
    
    # Add ask bars in red
    fig.add_trace(go.Bar(
        x=ask_prices,
        y=ask_sizes,
        name='Asks',
        marker_color='rgba(255, 0, 0, 0.7)'
    ))
    
    # Highlight levels that were walked in the simulation
    if highlight_levels:
        if highlight_levels['side'].lower() == 'buy':
            highlight_prices = [level['price'] for level in highlight_levels['execution_detail']]
            highlight_sizes = [level['quantity'] for level in highlight_levels['execution_detail']]
            
            fig.add_trace(go.Bar(
                x=highlight_prices,
                y=highlight_sizes,
                name='Executed',
                marker_color='rgba(255, 165, 0, 0.9)'
            ))
        else:
            highlight_prices = [level['price'] for level in highlight_levels['execution_detail']]
            highlight_sizes = [level['quantity'] for level in highlight_levels['execution_detail']]
            
            fig.add_trace(go.Bar(
                x=highlight_prices,
                y=highlight_sizes,
                name='Executed',
                marker_color='rgba(0, 191, 255, 0.9)'
            ))
    
    # Add cumulative volume lines
    fig.add_trace(go.Scatter(
        x=bid_prices,
        y=bid_totals,
        name='Cumulative Bid Volume',
        line=dict(color='green', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=ask_prices,
        y=ask_totals,
        name='Cumulative Ask Volume',
        line=dict(color='red', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Order Book Depth',
        xaxis_title='Price',
        yaxis=dict(
            title='Size',
            side='left'
        ),
        yaxis2=dict(
            title='Cumulative Size',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(orientation='h', y=1.05),
        height=500
    )
    
    return fig


def get_sample_orderbook(symbol):
    """Create a sample order book for demonstration"""
    book = OrderBook(symbol)
    
    if symbol == "BTC-USDT-SWAP":
        mid_price = 50000
        # Generate some sample bids and asks
        bids = [[str(mid_price - i * 100), str(1 + i * 0.5)] for i in range(1, 20)]
        asks = [[str(mid_price + i * 100), str(1 + i * 0.4)] for i in range(1, 20)]
    elif symbol == "ETH-USDT-SWAP":
        mid_price = 2500
        bids = [[str(mid_price - i * 5), str(10 + i * 2)] for i in range(1, 20)]
        asks = [[str(mid_price + i * 5), str(10 + i * 1.5)] for i in range(1, 20)]
    else:  # SOL-USDT-SWAP
        mid_price = 100
        bids = [[str(mid_price - i * 0.5), str(100 + i * 20)] for i in range(1, 20)]
        asks = [[str(mid_price + i * 0.5), str(100 + i * 15)] for i in range(1, 20)]
    
    book.update({'bids': bids, 'asks': asks})
    return book


def format_currency(value):
    """Format a value as currency"""
    return f"${value:.2f}"


def format_percentage(value):
    """Format a value as percentage"""
    return f"{value:.4f}%"


def apply_volatility(simulation_result, volatility_factor):
    """Apply a volatility factor to the simulation results"""
    if not simulation_result["success"]:
        return simulation_result
    
    # Clone the result to avoid modifying the original
    result = dict(simulation_result)
    
    # Apply volatility to the execution price
    price_change_pct = (np.random.random() - 0.5) * volatility_factor * 2  # -volatility to +volatility
    
    # Adjust the execution price
    original_price = result["avg_execution_price"]
    result["avg_execution_price"] = original_price * (1 + price_change_pct / 100)
    
    # Recalculate slippage
    result["slippage_pct"] = ((result["avg_execution_price"] - result["expected_price"]) / result["expected_price"]) * 100
    if result["side"].lower() == 'sell':
        result["slippage_pct"] = -result["slippage_pct"]
    
    # Recalculate slippage in USD
    result["slippage_usd"] = abs(result["avg_execution_price"] - result["expected_price"]) * result["executed_quantity"]
    
    # Recalculate total cost
    result["filled_usd"] = result["avg_execution_price"] * result["executed_quantity"]
    
    return result


def calculate_almgren_chriss_impact(order_size_usd, market_depth, volatility, side="buy"):
    """
    Calculate price impact using Almgren-Chriss model
    
    Args:
        order_size_usd: Order size in USD
        market_depth: Depth of market (typically sum of quantities in order book)
        volatility: Market volatility
        side: Order side (buy/sell)
        
    Returns:
        Estimated price impact in percentage
    """
    # Constants for the model (should be calibrated for specific markets)
    sigma = volatility  # Market volatility
    gamma = 0.1  # Market impact coefficient (higher = more impact)
    
    # Calculate temporary impact
    if market_depth > 0:
        temporary_impact = gamma * sigma * np.sqrt(order_size_usd / market_depth)
    else:
        temporary_impact = 0.5  # Fallback for low liquidity
    
    # Adjust for side (sell orders have negative impact)
    if side.lower() == "sell":
        temporary_impact = -temporary_impact
        
    return temporary_impact * 100  # Convert to percentage


def plot_orderbook_heatmap(order_book, depth=20, normalize=True):
    """
    Create a heatmap visualization of the order book liquidity distribution.
    
    Args:
        order_book: OrderBook instance
        depth: Number of price levels to include
        normalize: Whether to normalize the quantities for better visualization
    
    Returns:
        Plotly figure
    """
    bids = order_book.get_bids(depth)
    asks = order_book.get_asks(depth)
    
    # Check if we have data to display
    if not bids and not asks:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No order book data available",
            showarrow=False,
            font=dict(size=15, color="red"),
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(
            title="Order Book Heatmap - No Data",
            height=250
        )
        return fig
    
    # Create dataframes for bids and asks
    bid_prices = [float(bid[0]) for bid in bids]
    bid_quantities = [float(bid[1]) for bid in bids]
    bid_depths = np.cumsum(bid_quantities) if bid_quantities else []
    
    ask_prices = [float(ask[0]) for ask in asks]
    ask_quantities = [float(ask[1]) for ask in asks]
    ask_depths = np.cumsum(ask_quantities) if ask_quantities else []
    
    # Create a single array for all price levels
    if bid_prices and ask_prices:
        all_prices = np.concatenate([np.array(bid_prices)[::-1], np.array(ask_prices)])
    elif bid_prices:
        all_prices = np.array(bid_prices)
    elif ask_prices:
        all_prices = np.array(ask_prices)
    else:
        all_prices = np.array([])
    
    # For normalized view
    if normalize:
        max_bid_qty = max(bid_quantities) if bid_quantities else 0
        max_ask_qty = max(ask_quantities) if ask_quantities else 0
        max_qty = max(max_bid_qty, max_ask_qty) if (max_bid_qty > 0 or max_ask_qty > 0) else 1
        
        bid_intensity = np.array(bid_quantities) / max_qty if bid_quantities else np.array([])
        ask_intensity = np.array(ask_quantities) / max_qty if ask_quantities else np.array([])
    else:
        bid_intensity = bid_quantities
        ask_intensity = ask_quantities
    
    # Create the heatmap figure
    fig = go.Figure()
    
    # Add bid heatmap (greens) if we have bids
    if bid_prices and bid_quantities:
        fig.add_trace(go.Heatmap(
            z=[bid_intensity],
            x=bid_prices,
            y=['Bids'],
            colorscale=[[0, 'rgba(0,50,0,0.2)'], [1, 'rgba(0,255,0,1)']],
            showscale=False,
            text=[[f"Price: {p}<br>Quantity: {q}<br>Depth: {d}" for p, q, d in zip(bid_prices, bid_quantities, bid_depths)]],
            hoverinfo='text'
        ))
    
    # Add ask heatmap (reds) if we have asks
    if ask_prices and ask_quantities:
        fig.add_trace(go.Heatmap(
            z=[ask_intensity],
            x=ask_prices,
            y=['Asks'],
            colorscale=[[0, 'rgba(50,0,0,0.2)'], [1, 'rgba(255,0,0,1)']],
            showscale=False,
            text=[[f"Price: {p}<br>Quantity: {q}<br>Depth: {d}" for p, q, d in zip(ask_prices, ask_quantities, ask_depths)]],
            hoverinfo='text'
        ))
    
    # Highlight the spread
    if bid_prices and ask_prices:
        best_bid = max(bid_prices)
        best_ask = min(ask_prices)
        spread = best_ask - best_bid
        spread_pct = spread / best_bid * 100
        
        # Add spread marker
        fig.add_shape(
            type="rect",
            x0=best_bid,
            x1=best_ask,
            y0=-0.5,
            y1=1.5,
            fillcolor="rgba(100,100,100,0.2)",
            line=dict(color="gray", width=1, dash="dot"),
            layer="below"
        )
        
        # Add spread annotation
        spread_text = f"Spread: {spread:.2f} ({spread_pct:.4f}%)"
    else:
        spread_text = "No spread available"

    # Update layout
    fig.update_layout(
        title=f"Order Book Heatmap - Liquidity Distribution",
        xaxis_title="Price",
        yaxis=dict(
            title="",
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Bids', 'Asks']
        ),
        height=250,
        margin=dict(l=50, r=50, t=80, b=50),
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                showarrow=False,
                text=spread_text,
                xref="paper",
                yref="paper",
                font=dict(size=12)
            )
        ]
    )
    
    return fig


def plot_order_walk_visualization(order_simulation, order_book, depth=20):
    """
    Visualize how a market order "walks the book" by highlighting the filled price levels.
    
    Args:
        order_simulation: Result from simulate_market_order function
        order_book: OrderBook instance
        depth: Number of price levels to show
    
    Returns:
        Plotly figure
    """
    if not order_simulation["success"]:
        fig = go.Figure()
        fig.add_annotation(
            text="Order simulation failed - insufficient liquidity",
            showarrow=False,
            font=dict(size=15, color="red"),
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(height=400)
        return fig
    
    side = order_simulation["side"]
    execution_details = order_simulation["execution_detail"]
    
    # Get data from order book
    bids = order_book.get_bids(depth)
    asks = order_book.get_asks(depth)
    
    # Create price level arrays
    bid_prices = [float(bid[0]) for bid in bids]
    bid_quantities = [float(bid[1]) for bid in bids]
    
    ask_prices = [float(ask[0]) for ask in asks]
    ask_quantities = [float(ask[1]) for ask in asks]
    
    # Create execution price and qty arrays
    exec_prices = [level["price"] for level in execution_details]
    exec_quantities = [level["quantity"] for level in execution_details]
    exec_values = [level["price"] * level["quantity"] for level in execution_details]
    
    # Determine cumulative volume and percentage of order filled at each price
    cumulative_qty = np.cumsum(exec_quantities)
    total_qty = sum(exec_quantities)
    pct_filled = (cumulative_qty / total_qty * 100) if total_qty > 0 else [0]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for the original order book
    if side.lower() == "buy":
        # For buys we filled ask side
        fig.add_trace(go.Bar(
            x=ask_prices,
            y=ask_quantities,
            name="Ask Liquidity",
            marker_color="rgba(255, 0, 0, 0.3)",
            hoverinfo="text",
            hovertext=[f"Price: {p}<br>Quantity: {q}" for p, q in zip(ask_prices, ask_quantities)]
        ))
        
        # Add execution bars
        fig.add_trace(go.Bar(
            x=exec_prices,
            y=exec_quantities,
            name="Executed",
            marker_color="rgba(255, 165, 0, 0.9)",
            hoverinfo="text",
            hovertext=[f"Price: {p}<br>Quantity: {q}<br>Value: ${v:.2f}<br>Filled: {pct:.1f}%" 
                     for p, q, v, pct in zip(exec_prices, exec_quantities, exec_values, pct_filled)]
        ))
    else:
        # For sells we filled bid side
        fig.add_trace(go.Bar(
            x=bid_prices,
            y=bid_quantities,
            name="Bid Liquidity",
            marker_color="rgba(0, 128, 0, 0.3)",
            hoverinfo="text",
            hovertext=[f"Price: {p}<br>Quantity: {q}" for p, q in zip(bid_prices, bid_quantities)]
        ))
        
        # Add execution bars
        fig.add_trace(go.Bar(
            x=exec_prices,
            y=exec_quantities,
            name="Executed",
            marker_color="rgba(0, 191, 255, 0.9)",
            hoverinfo="text",
            hovertext=[f"Price: {p}<br>Quantity: {q}<br>Value: ${v:.2f}<br>Filled: {pct:.1f}%" 
                     for p, q, v, pct in zip(exec_prices, exec_quantities, exec_values, pct_filled)]
        ))
    
    # Add price distribution line
    fig.add_trace(go.Scatter(
        x=exec_prices,
        y=pct_filled,
        mode="lines+markers",
        name="% Order Filled",
        line=dict(color="black", width=2),
        marker=dict(size=8, color="purple"),
        yaxis="y2"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{side.capitalize()} Order Execution - Walking the Book",
        xaxis_title="Price",
        yaxis=dict(
            title="Quantity",
            side="left"
        ),
        yaxis2=dict(
            title="% of Order Filled",
            side="right",
            overlaying="y",
            range=[0, 100],
            ticksuffix="%"
        ),
        legend=dict(x=0.01, y=0.99, orientation="h"),
        height=400
    )
    
    # Add a vertical line at the average execution price
    avg_price = order_simulation["avg_execution_price"]
    fig.add_shape(
        type="line",
        x0=avg_price,
        x1=avg_price,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add annotation for average price
    fig.add_annotation(
        x=avg_price,
        y=1,
        yref="paper",
        text=f"Avg Price: {avg_price:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    return fig


def create_execution_quality_dashboard(execution_history):
    """
    Create a comprehensive execution quality dashboard
    
    Args:
        execution_history: DataFrame with historical execution data
    
    Returns:
        List of plotly figures
    """
    if execution_history is None or execution_history.empty:
        return [go.Figure().update_layout(
            title="No execution history available",
            annotations=[dict(
                text="Execute some orders to see performance metrics",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )]
    
    figures = []
    
    # 1. Slippage over time
    slippage_fig = px.line(
        execution_history, 
        x='timestamp', 
        y='slippage_bps',
        color='side',
        labels={'slippage_bps': 'Slippage (bps)', 'timestamp': 'Time'},
        title='Slippage Over Time',
        hover_data=['quantity_usd', 'avg_execution_price']
    )
    
    # Add a horizontal line at y=0 for reference
    slippage_fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        xref="paper",
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Calculate and display average slippage
    avg_slippage = execution_history['slippage_bps'].mean()
    slippage_fig.add_annotation(
        x=0.98,
        y=0.03,
        xref="paper",
        yref="paper",
        text=f"Avg Slippage: {avg_slippage:.2f} bps",
        showarrow=False,
        font=dict(color="black", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # Update layout
    slippage_fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    figures.append(slippage_fig)
    
    # 2. Slippage distribution histogram
    hist_fig = px.histogram(
        execution_history,
        x='slippage_bps',
        color='side',
        marginal='violin',
        nbins=20,
        title='Slippage Distribution',
        labels={'slippage_bps': 'Slippage (bps)'}
    )
    
    # Add vertical line at mean
    hist_fig.add_vline(
        x=avg_slippage,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean: {avg_slippage:.2f}",
        annotation_position="top right"
    )
    
    # Add vertical line at 0
    hist_fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="gray"
    )
    
    # Update layout
    hist_fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    figures.append(hist_fig)
    
    # 3. Slippage vs Order Size scatter plot
    scatter_fig = px.scatter(
        execution_history,
        x='quantity_usd',
        y='slippage_bps',
        color='side',
        size='order_pct_of_depth',
        hover_data=['timestamp', 'avg_execution_price'],
        title='Slippage vs Order Size',
        labels={
            'quantity_usd': 'Order Size (USD)',
            'slippage_bps': 'Slippage (bps)',
            'order_pct_of_depth': '% of Book Depth'
        },
        trendline='ols'
    )
    
    # Add horizontal line at y=0
    scatter_fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        xref="paper",
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Update layout
    scatter_fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    figures.append(scatter_fig)
    
    # 4. Model accuracy metrics if predictions are available
    if 'predicted_slippage_bps' in execution_history.columns:
        # Create prediction error field
        execution_history['prediction_error'] = execution_history['slippage_bps'] - execution_history['predicted_slippage_bps']
        
        # RMSE and MAE
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(execution_history['slippage_bps'], execution_history['predicted_slippage_bps']))
        mae = mean_absolute_error(execution_history['slippage_bps'], execution_history['predicted_slippage_bps'])
        
        # Create scatter plot of actual vs predicted
        pred_fig = px.scatter(
            execution_history,
            x='slippage_bps',
            y='predicted_slippage_bps',
            color='side', 
            hover_data=['timestamp', 'quantity_usd'],
            title=f'Model Accuracy - Actual vs Predicted Slippage (RMSE: {rmse:.2f}, MAE: {mae:.2f})',
            labels={
                'slippage_bps': 'Actual Slippage (bps)',
                'predicted_slippage_bps': 'Predicted Slippage (bps)'
            }
        )
        
        # Add 45-degree line (perfect predictions)
        min_val = min(execution_history['slippage_bps'].min(), execution_history['predicted_slippage_bps'].min())
        max_val = max(execution_history['slippage_bps'].max(), execution_history['predicted_slippage_bps'].max())
        
        pred_fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Perfect Prediction'
        ))
        
        # Update layout
        pred_fig.update_layout(
            height=350,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        figures.append(pred_fig)
        
        # 5. Prediction error over time
        error_fig = px.line(
            execution_history.sort_values('timestamp'),
            x='timestamp',
            y='prediction_error',
            color='side',
            title='Prediction Error Over Time',
            labels={
                'prediction_error': 'Prediction Error (bps)',
                'timestamp': 'Time'
            }
        )
        
        # Add a horizontal line at y=0 (no error)
        error_fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=0,
            y1=0,
            xref="paper",
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Update layout
        error_fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        figures.append(error_fig)
    
    return figures


def create_scenario_analysis(order_book, scenarios):
    """
    Create a comparative visualization for different execution scenarios
    
    Args:
        order_book: Current OrderBook instance
        scenarios: List of scenario dictionaries with keys:
                  - name: Scenario name
                  - side: Buy or sell
                  - quantity_usd: Order size in USD
                  - simulation_result: Result from simulation function
                  
    Returns:
        List of plotly figures for comparing scenarios
    """
    if not scenarios or len(scenarios) == 0:
        return [go.Figure().update_layout(
            title="No scenarios to compare",
            height=300
        )]
    
    figures = []
    
    # Prepare data for the comparison
    scenario_names = [s['name'] for s in scenarios]
    avg_prices = [s['simulation_result'].get('avg_execution_price', 0) for s in scenarios]
    slippages = [s['simulation_result'].get('slippage_bps', 0) for s in scenarios]
    quantities = [s['quantity_usd'] for s in scenarios]
    fees = [s['simulation_result'].get('fee', 0) for s in scenarios]
    total_costs = [s['simulation_result'].get('total_cost', 0) for s in scenarios]
    sides = [s['side'] for s in scenarios]
    
    # 1. Basic comparison bar chart
    comparison_df = pd.DataFrame({
        'Scenario': scenario_names,
        'Avg Price': avg_prices,
        'Slippage (bps)': slippages,
        'Order Size': quantities,
        'Fee': fees,
        'Total Cost': total_costs,
        'Side': sides
    })
    
    # Price comparison
    price_fig = px.bar(
        comparison_df,
        x='Scenario',
        y='Avg Price',
        color='Side',
        title='Average Execution Price by Scenario',
        labels={'Avg Price': 'Average Price'},
        hover_data=['Order Size', 'Slippage (bps)']
    )
    price_fig.update_layout(height=300)
    figures.append(price_fig)
    
    # Slippage comparison
    slippage_fig = px.bar(
        comparison_df,
        x='Scenario',
        y='Slippage (bps)',
        color='Side',
        title='Slippage by Scenario',
        labels={'Slippage (bps)': 'Slippage (bps)'},
        hover_data=['Order Size', 'Avg Price']
    )
    
    # Add a horizontal line at y=0
    slippage_fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(scenario_names) - 0.5,
        y0=0,
        y1=0,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    slippage_fig.update_layout(height=300)
    figures.append(slippage_fig)
    
    # 2. Execution details comparison (if available)
    # Create a more detailed comparison of execution details across scenarios
    execution_details = []
    for scenario in scenarios:
        if 'execution_detail' in scenario['simulation_result']:
            for level in scenario['simulation_result']['execution_detail']:
                execution_details.append({
                    'Scenario': scenario['name'],
                    'Side': scenario['side'],
                    'Price': level['price'],
                    'Quantity': level['quantity'],
                    'Value': level['price'] * level['quantity'],
                    'Order Size': scenario['quantity_usd']
                })
    
    if execution_details:
        execution_df = pd.DataFrame(execution_details)
        
        # Price levels filled per scenario
        levels_fig = px.bar(
            execution_df,
            x='Price',
            y='Quantity',
            color='Scenario',
            facet_col='Side',
            title='Price Levels Filled by Scenario',
            hover_data=['Value'],
            barmode='group'
        )
        levels_fig.update_layout(height=400)
        figures.append(levels_fig)
        
        # Cumulative execution by price level
        cumulative_fig = px.bar(
            execution_df,
            x='Price',
            y='Quantity',
            color='Scenario',
            facet_col='Side',
            title='Cumulative Execution by Price Level',
            hover_data=['Value'],
            barmode='relative'
        )
        cumulative_fig.update_layout(height=400)
        figures.append(cumulative_fig)
    
    # 3. Total cost comparison
    cost_fig = px.bar(
        comparison_df,
        x='Scenario',
        y='Total Cost',
        color='Side',
        title='Total Execution Cost by Scenario',
        labels={'Total Cost': 'Total Cost (USD)'},
        hover_data=['Order Size', 'Fee', 'Slippage (bps)']
    )
    
    # Add cost breakdown as stacked bars if we have fee information
    if all(fee > 0 for fee in fees):
        # Create a stacked bar with market impact and fees
        market_impact = [total - fee for total, fee in zip(total_costs, fees)]
        
        cost_breakdown = pd.DataFrame({
            'Scenario': scenario_names + scenario_names,
            'Cost Component': ['Market Impact'] * len(scenario_names) + ['Fees'] * len(scenario_names),
            'Cost': market_impact + fees,
            'Side': sides + sides
        })
        
        cost_breakdown_fig = px.bar(
            cost_breakdown,
            x='Scenario',
            y='Cost',
            color='Cost Component',
            title='Cost Breakdown by Scenario',
            labels={'Cost': 'Cost (USD)'},
            barmode='stack',
            hover_data=['Side']
        )
        cost_breakdown_fig.update_layout(height=350)
        figures.append(cost_breakdown_fig)
    
    cost_fig.update_layout(height=300)
    figures.append(cost_fig)
    
    # 4. Percentage difference from best execution
    if len(scenarios) > 1:
        # Find the best (lowest) price for buys and the best (highest) price for sells
        buy_scenarios = [s for s in scenarios if s['side'].lower() == 'buy']
        sell_scenarios = [s for s in scenarios if s['side'].lower() == 'sell']
        
        comparison_df['Best Price Difference %'] = 0.0
        
        if buy_scenarios:
            best_buy_price = min([s['simulation_result'].get('avg_execution_price', float('inf')) 
                                for s in buy_scenarios if s['simulation_result'].get('success', False)])
            
            for i, scenario in enumerate(scenarios):
                if scenario['side'].lower() == 'buy' and scenario['simulation_result'].get('success', False):
                    price = scenario['simulation_result'].get('avg_execution_price', 0)
                    if best_buy_price > 0:
                        comparison_df.loc[i, 'Best Price Difference %'] = (price - best_buy_price) / best_buy_price * 100
        
        if sell_scenarios:
            best_sell_price = max([s['simulation_result'].get('avg_execution_price', 0) 
                                 for s in sell_scenarios if s['simulation_result'].get('success', False)])
            
            for i, scenario in enumerate(scenarios):
                if scenario['side'].lower() == 'sell' and scenario['simulation_result'].get('success', False):
                    price = scenario['simulation_result'].get('avg_execution_price', 0)
                    if best_sell_price > 0:
                        comparison_df.loc[i, 'Best Price Difference %'] = (best_sell_price - price) / best_sell_price * 100
        
        # Create the price difference chart
        diff_fig = px.bar(
            comparison_df,
            x='Scenario',
            y='Best Price Difference %',
            color='Side',
            title='Execution Price Difference from Best Scenario (%)',
            labels={'Best Price Difference %': 'Difference from Best Price (%)'},
            hover_data=['Avg Price', 'Order Size']
        )
        
        diff_fig.update_layout(height=300)
        figures.append(diff_fig)
    
    return figures

# Function to update simulation with scenario parameters
def run_scenario_simulation(order_book, side, quantity_usd, model_type, volatility=0.2, fee_percentage=0.001):
    """Run a market order simulation with the specified parameters"""
    if model_type == 'basic':
        # Basic simulation without ML prediction
        simulation_result = simulate_market_order(order_book, side, quantity_usd)
        
        # Calculate metrics
        if simulation_result["success"]:
            metrics = calculate_order_metrics(simulation_result, fee_percentage)
            simulation_result.update(metrics)
        
    elif model_type == 'ml':
        # ML-based simulation
        simulation_result = simulate_market_order_with_ml(
            order_book, side, quantity_usd, volatility, fee_percentage
        )
    
    elif model_type == 'enhanced_ml' and 'ENHANCED_MODELS_AVAILABLE' in globals() and ENHANCED_MODELS_AVAILABLE:
        # Enhanced ML simulation
        simulation_result = simulate_market_order_with_enhanced_ml(
            order_book, side, quantity_usd, volatility, fee_percentage
        )
    
    elif model_type == 'almgren_chriss' and ALMGREN_CHRISS_AVAILABLE:
        # Almgren-Chriss simulation
        simulation_result = simulate_market_order_with_impact(
            order_book, side, quantity_usd, volatility
        )
        
        # Calculate metrics
        if simulation_result["success"]:
            metrics = calculate_order_metrics(simulation_result, fee_percentage)
            simulation_result.update(metrics)
    
    else:
        # Fallback to basic if model not available
        simulation_result = simulate_market_order(order_book, side, quantity_usd)
        
        # Calculate metrics
        if simulation_result["success"]:
            metrics = calculate_order_metrics(simulation_result, fee_percentage)
            simulation_result.update(metrics)
    
    return simulation_result

def save_execution_to_history(simulation_result, side, quantity_usd, order_book):
    """Save execution results to the history dataframe for dashboard analysis"""
    if not simulation_result.get("success", False):
        return
    
    # Get current time
    timestamp = datetime.now()
    
    # Calculate order percentage of depth
    side_depth = sum(qty for _, qty in order_book.get_asks(100)) if side.lower() == "buy" else sum(qty for _, qty in order_book.get_bids(100))
    order_pct_of_depth = (quantity_usd / side_depth) * 100 if side_depth > 0 else 0
    
    # Calculate depth imbalance
    bid_depth = sum(qty for _, qty in order_book.get_bids(20))
    ask_depth = sum(qty for _, qty in order_book.get_asks(20))
    total_depth = bid_depth + ask_depth
    depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
    
    # Get spread in basis points
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    if best_bid and best_ask:
        mid_price = (best_bid[0] + best_ask[0]) / 2
        spread_bps = (best_ask[0] - best_bid[0]) / mid_price * 10000  # Convert to basis points
    else:
        spread_bps = 0
    
    # Create a new record
    new_record = {
        'timestamp': timestamp,
        'symbol': order_book.symbol,
        'side': side.lower(),
        'quantity_usd': quantity_usd,
        'order_pct_of_depth': order_pct_of_depth,
        'spread_bps': spread_bps,
        'volatility': simulation_result.get('volatility', 0.2),
        'depth_imbalance': depth_imbalance,
        'avg_execution_price': simulation_result.get('avg_execution_price', 0),
        'slippage_bps': simulation_result.get('slippage_bps', 0),
        'fee': simulation_result.get('fee', 0),
        'total_cost': simulation_result.get('total_cost', 0)
    }
    
    # Add predicted slippage if available
    if 'predicted_slippage' in simulation_result:
        new_record['predicted_slippage_bps'] = simulation_result['predicted_slippage']
    
    # Convert to DataFrame
    record_df = pd.DataFrame([new_record])
    
    # Append to existing history
    if 'execution_history' in st.session_state:
        st.session_state.execution_history = pd.concat([st.session_state.execution_history, record_df])
    else:
        st.session_state.execution_history = record_df
    
    # Log
    logging.info(f"Saved execution data for ML training, current records: {len(st.session_state.execution_history)}")

def main():
    # Initialize connection to order book data
    connection_status = init_client()
    
    # Update session state order book from client (live data)
    if connection_status:
        data_updated = update_session_order_book()
        if data_updated:
            logging.info("Updated session state order book from client")
    
    # Sidebar inputs
    st.sidebar.header("Order Parameters")
    
    # Connection status indicator
    connection_indicator = st.sidebar.empty()
    
    # Get client
    client = OrderBookClient.get_instance()
    
    # Check real connection status
    real_connection = client.is_connected() and client.has_data()
    
    if real_connection:
        connection_indicator.success("Connected to order book stream")
    else:
        connection_indicator.warning("Not connected to order book stream")
        if st.sidebar.button("Attempt Reconnection"):
            # Reset client and try to reconnect
            client.reset()
            st.session_state.connected = False
            reconnected = init_client()
            if reconnected:
                # Wait a moment for data to arrive
                time.sleep(1)
                data_updated = update_session_order_book()
                if data_updated:
                    st.sidebar.success("Successfully reconnected and received data")
                else:
                    st.sidebar.warning("Reconnected but no data received yet")
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to reconnect")
    
    # Add debug information in an expandable section
    with st.sidebar.expander("Debug Information", expanded=False):
        last_update = time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update_time)) if st.session_state.last_update_time > 0 else 'Never'
        client_update = time.strftime('%H:%M:%S', time.localtime(client.last_update_time)) if client.last_update_time > 0 else 'Never'
        
        st.markdown(f"Connected to API: {client.is_connected()}")
        st.markdown(f"Client has data: {client.has_data()}")
        st.markdown(f"Session state connected: {st.session_state.connected}")
        st.markdown(f"Last Update (Session): {last_update}")
        st.markdown(f"Last Update (Client): {client_update}")
        st.markdown(f"Connection Attempts: {st.session_state.connection_attempts}")
        st.markdown(f"Cached Order Book Size: {len(st.session_state.order_book.bids)} bids, {len(st.session_state.order_book.asks)} asks")
        st.markdown(f"Client Order Book Size: {len(client.bids)} bids, {len(client.asks)} asks")
        
        if st.button("Force Update from Client", key="force_update"):
            updated = update_session_order_book()
            if updated:
                st.success("Successfully updated order book from client")
                st.experimental_rerun()
            else:
                st.error("Failed to update - no data available from client")
        
        if st.button("Print Debug Info", key="debug_info"):
            logging.info(f"Client connected: {client.is_connected()}")
            logging.info(f"Client has data: {client.has_data()}")
            logging.info(f"Session state connected: {st.session_state.connected}")
            logging.info(f"Order book state (client): {client.get_order_book_snapshot()}")
    
    # Sidebar tabs
    sidebar_tab1, sidebar_tab2, sidebar_tab3 = st.sidebar.tabs(["Simulation", "Model Validation", "Model Analysis"])
    
    with sidebar_tab1:
        # Market selection
        market = st.selectbox(
            "Select Market",
            options=["BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "SOL-USDT"],
            index=0,
            help="Trading pair for simulation"
        )
        
        # Order configuration
        order_side = st.radio(
            "Order Side",
            options=["Buy", "Sell"],
            horizontal=True,
            help="Market buy or sell order"
        )
        
        quantity_usd = st.number_input(
            "Order Size (USD)",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
            help="Order quantity in USD"
        )
        
        # Volatility slider
        volatility = st.slider(
            "Market Volatility",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Annualized volatility (higher values = more slippage)"
        )
        
        # Fee tier selection
        fee_tier = st.selectbox(
            "Fee Tier",
            options=["VIP 0 (0.10%)", "VIP 1 (0.08%)", "VIP 2 (0.05%)", "VIP 3 (0.03%)", "VIP 4 (0.02%)"],
            index=0,
            help="Trading fee tier (affects total cost)"
        )
        
        # Extract fee percentage from the selection
        fee_percentage = float(fee_tier.split("(")[1].split("%")[0]) / 100
        
        # Advanced options
        with st.expander("Advanced Options"):
            execution_delay = st.slider(
                "Execution Delay (ms)",
                min_value=0,
                max_value=500,
                value=50,
                step=10,
                help="Simulated delay between order placement and execution"
            )
            
            price_impact_factor = st.slider(
                "Price Impact Factor",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Multiplier for the estimated price impact"
            )
            
            use_live_data = st.checkbox(
                "Use Live Data",
                value=True,
                help="Use real-time order book data. If disabled, will use simulated data."
            )
            
            use_ml_model = st.checkbox(
                "Use ML-based Slippage Prediction",
                value=False,
                help="Use machine learning to predict slippage instead of simulating market order execution"
            )
            
            # Add enhanced ML model selection
            if use_ml_model:
                regression_model_type = st.selectbox(
                    "Regression Model Type",
                    options=["Standard Linear", "Random Forest", "Gradient Boosting"],
                    index=2,
                    help="Select the regression model type to use for slippage prediction"
                )
                
                # Map UI selection to model type
                model_mapping = {
                    "Standard Linear": "linear",
                    "Random Forest": "random_forest",
                    "Gradient Boosting": "gradient_boosting"
                }
                
                model_type = model_mapping[regression_model_type]
                
                risk_level = st.select_slider(
                    "Risk Level",
                    options=["Conservative (q90)", "Moderate (q75)", "Balanced (q50)", "Aggressive (q25)", "Optimistic (q10)", "Mean"],
                    value="Balanced (q50)",
                    help="Risk level for slippage prediction. Higher percentiles give more conservative estimates."
                )
                # Extract the risk level code from the selection
                risk_level_code = risk_level.split("(")[1].split(")")[0] if "(" in risk_level else "mean"
            else:
                model_type = "gradient_boosting"  # Default
                risk_level_code = "q50"  # Default
            
            # Impact model selection 
            impact_model = st.selectbox(
                "Impact Model",
                options=["Standard", "Almgren-Chriss", "Square Root"],
                index=0,
                help="Select market impact model for price impact calculation"
            )
            
            # Add option to use calibrated parameters if Almgren-Chriss is selected
            if impact_model == "Almgren-Chriss" and ALMGREN_CHRISS_AVAILABLE:
                use_calibration = st.checkbox(
                    "Use Calibrated Parameters",
                    value=True,
                    help="Use automatically calibrated parameters for the Almgren-Chriss model"
                )
            else:
                use_calibration = True
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            simulate_button = st.button("Simulate", type="primary", use_container_width=True)
        with col2:
            reset_button = st.button("Reset", type="secondary", use_container_width=True)
        with col3:
            if st.button("Train Model", use_container_width=True):
                st.info("Training slippage prediction model...")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Initialize data collector
                collector = SlippageDataCollector()
                
                # Check if we have enough data
                try:
                    X, y = collector.get_training_data()
                    
                    # Update progress
                    progress_bar.progress(30)
                    
                    # Initialize and train enhanced model
                    try:
                        model = EnhancedSlippageModel()
                        training_result = model.train(X, y)
                        
                        # Update progress
                        progress_bar.progress(100)
                        
                        # Show training results
                        st.success(f"Enhanced model trained successfully with {training_result['samples_count']} samples")
                        st.info(f"Models trained: {', '.join(training_result['models_trained'])}")
                        
                        # Show model performance
                        if "model_performances" in training_result:
                            perf_metrics = pd.DataFrame()
                            for model_name, metrics in training_result["model_performances"].items():
                                model_metrics = pd.DataFrame({model_name: metrics}, index=metrics.keys())
                                perf_metrics = pd.concat([perf_metrics, model_metrics], axis=1)
                            
                            st.dataframe(perf_metrics)
                    except Exception as e:
                        # Fall back to standard model if enhanced fails
                        model = SlippageModel()
                        training_result = model.train(X, y)
                        
                        # Update progress
                        progress_bar.progress(100)
                        
                        # Show training results
                        st.warning(f"Enhanced model training failed, used standard model. Error: {str(e)}")
                        st.success(f"Model trained successfully with {training_result['samples_count']} samples")
                        st.info(f"R² Score: {training_result['linear_r2_score']:.4f}")
                except ValueError as e:
                    st.error(f"Error training model: {str(e)}")
                    st.info("Collecting more data from order executions...")
                finally:
                    # Remove progress bar
                    progress_bar.empty()
    
    # Model Validation Tab
    with sidebar_tab2:
        if ALMGREN_CHRISS_AVAILABLE:
            st.header("Almgren-Chriss Model Validation")
            
            # Market selection for validation
            val_market = st.selectbox(
                "Select Market",
                options=["BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "SOL-USDT", "All Markets"],
                index=0,
                key="val_market",
                help="Market to validate"
            )
            
            # Time period selection
            val_days_back = st.slider(
                "Days to Look Back",
                min_value=1,
                max_value=90,
                value=30,
                key="val_days_back",
                help="Number of days of historical data to use for validation"
            )
            
            # Action buttons for validation
            val_col1, val_col2 = st.columns(2)
            with val_col1:
                validate_button = st.button("Validate Model", type="primary", use_container_width=True)
            with val_col2:
                calibrate_button = st.button("Calibrate Parameters", type="secondary", use_container_width=True)
            
            # Display validation results
            if validate_button:
                st.info("Validating Almgren-Chriss model performance...")
                
                # Create a progress bar
                val_progress = st.progress(0)
                
                try:
                    # Initialize validator
                    validator = ModelValidator()
                    
                    # Update progress
                    val_progress.progress(30)
                    
                    # Validate model
                    if val_market == "All Markets":
                        results = validator.validate_almgren_chriss(days_back=val_days_back)
                    else:
                        results = validator.validate_almgren_chriss(symbol=val_market, days_back=val_days_back)
                    
                    # Update progress
                    val_progress.progress(100)
                    
                    # Show validation summary
                    st.success(f"Model validation completed. Results for {len(results)} markets.")
                    
                    # Create and display validation results dataframe
                    val_data = []
                    for market, metrics in results.items():
                        val_data.append({
                            'Market': market,
                            'MAPE (%)': metrics['mape'],
                            'MAE (bps)': metrics['mae'],
                            'R²': metrics['r2'],
                            'Samples': metrics['sample_count']
                        })
                    
                    val_df = pd.DataFrame(val_data)
                    st.dataframe(val_df, use_container_width=True)
                    
                    # If only one market was validated, show detailed plot
                    if val_market != "All Markets" and val_market in results:
                        st.subheader(f"Prediction Accuracy for {val_market}")
                        
                        # Generate and display validation plot
                        img_str = validator.plot_prediction_vs_actual(val_market, val_days_back)
                        st.image(f"data:image/png;base64,{img_str}", use_column_width=True)
                
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
                finally:
                    # Remove progress bar
                    val_progress.empty()
            
            # Handle calibration button
            if calibrate_button:
                st.info("Calibrating Almgren-Chriss model parameters...")
                
                # Create a progress bar
                cal_progress = st.progress(0)
                
                try:
                    # Initialize calibrator
                    calibrator = AlmgrenChrissCalibrator()
                    
                    # Update progress
                    cal_progress.progress(30)
                    
                    # Calibrate parameters
                    if val_market == "All Markets":
                        results = calibrator.calibrate_parameters()
                    else:
                        results = calibrator.calibrate_parameters(symbol=val_market)
                    
                    # Update progress
                    cal_progress.progress(100)
                    
                    # Show calibration results
                    st.success(f"Parameter calibration completed for {len(results)} markets.")
                    
                    # Create and display calibration results dataframe
                    cal_data = []
                    for market, params in results.items():
                        cal_data.append({
                            'Market': market,
                            'Gamma Scale': params['gamma_scale'],
                            'Eta Scale': params['eta_scale'],
                            'Validation MAPE (%)': params['validation_mape'],
                            'Validation R²': params['validation_r2'],
                            'Training Samples': params['training_records']
                        })
                    
                    cal_df = pd.DataFrame(cal_data)
                    st.dataframe(cal_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Calibration failed: {str(e)}")
                finally:
                    # Remove progress bar
                    cal_progress.empty()
        else:
            st.info("Almgren-Chriss model validation features are not available.")
            st.info("Please make sure the required modules are installed.")
    
    # Model Analysis tab
    with sidebar_tab3:
        st.header("Regression Model Analysis")
        
        # Feature importance visualization
        st.subheader("Feature Importance")
        
        viz_model_type = st.selectbox(
            "Model Type for Analysis",
            options=["Random Forest", "Gradient Boosting"],
            index=1,
            key="viz_model_type"
        )
        
        viz_plot_type = st.radio(
            "Plot Type",
            options=["Bar Chart", "Pie Chart"],
            horizontal=True,
            key="viz_plot_type"
        )
        
        # Map selections to model values
        viz_model_mapping = {
            "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting"
        }
        
        viz_plot_mapping = {
            "Bar Chart": "bar",
            "Pie Chart": "pie"
        }
        
        selected_model = viz_model_mapping[viz_model_type]
        selected_plot = viz_plot_mapping[viz_plot_type]
        
        # Show feature importance button
        if st.button("Show Feature Importance", key="show_importance"):
            # Initialize enhanced model
            model = EnhancedSlippageModel()
            
            # Generate plot
            fig = model.plot_feature_importance(selected_model, selected_plot)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.info("""
                Feature importance shows which factors contribute most to slippage prediction.
                Higher values indicate stronger influence on execution costs.
                """)
            else:
                st.warning(f"No feature importance data available for {viz_model_type}. Please train the model first.")
        
        # Model comparison
        st.subheader("Model Performance Comparison")
        
        if st.button("Compare Models", key="compare_models"):
            # Initialize enhanced model
            model = EnhancedSlippageModel()
            
            # Generate performance comparison plot
            fig = model.compare_model_performance()
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.info("""
                Model comparison shows how different regression techniques perform:
                - R² Score: Higher is better (closer to 1.0)
                - RMSE (Root Mean Squared Error): Lower is better
                - MAE (Mean Absolute Error): Lower is better
                """)
            else:
                st.warning("No model performance data available. Please train the models first.")
        
        # Time series analysis
        st.subheader("Time Series Analysis")
        
        forecast_days = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to forecast slippage trends"
        )
        
        if st.button("Generate Forecast", key="generate_forecast"):
            # Initialize time series model
            ts_model = TimeSeriesSlippageModel()
            
            # Get historical data
            collector = SlippageDataCollector()
            
            if collector.data is not None and len(collector.data) >= 30:
                try:
                    # Train the model if not already trained
                    training_result = ts_model.train(collector.data)
                    
                    # Generate forecast
                    forecast_data = ts_model.forecast_slippage_trend(
                        collector.data,
                        days_ahead=forecast_days
                    )
                    
                    # Generate plot
                    fig = ts_model.plot_slippage_trend(collector.data, forecast_data)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.info(f"Forecast generated for the next {forecast_days} days based on historical patterns.")
                    else:
                        st.warning("Could not generate forecast visualization.")
                except Exception as e:
                    st.error(f"Time series forecasting failed: {str(e)}")
            else:
                st.warning("Not enough historical data for time series forecasting. Need at least 30 records.")
                st.info(f"Current records: {0 if collector.data is None else len(collector.data)}")
    
    # Initialize session state for storing simulation results
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
    # Initialize session state variables for scenarios
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = []
        
    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = pd.DataFrame()
    
    # Main panel
    col1, col2 = st.columns(2)
    
    # Get the order book - live or sample
    has_valid_data = real_connection and len(st.session_state.order_book.bids) > 0 and len(st.session_state.order_book.asks) > 0
    
    # If we're connected but session state doesn't have data, try to update it
    if real_connection and not has_valid_data:
        has_valid_data = update_session_order_book()
    
    if use_live_data and has_valid_data:
        order_book = st.session_state.order_book
        # Check if the order book has been updated recently
        if time.time() - st.session_state.last_update_time > 30:
            # Show warning if the order book hasn't been updated recently
            st.warning("Order book data may be stale. Last update was over 30 seconds ago.")
    else:
        if use_live_data and not has_valid_data:
            st.warning("No live data available. Using simulated order book data.")
        order_book = get_sample_orderbook(market)
    
    # Order book visualization
    with col1:
        st.subheader("Order Book Depth Chart")
        
        # Show live data indicator
        if use_live_data and has_valid_data:
            st.success("Using live order book data")
        else:
            st.info("Using simulated order book data")
        
        # Placeholder for order book visualization
        orderbook_placeholder = st.empty()
        
        # Display current market metrics
        st.subheader("Market Metrics")
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        spread = order_book.get_spread()
        
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric(label="Best Bid", value=f"${best_bid[0]:.2f}" if best_bid else "$0.00")
        with metrics_cols[1]:
            st.metric(label="Best Ask", value=f"${best_ask[0]:.2f}" if best_ask else "$0.00")
        with metrics_cols[2]:
            st.metric(label="Spread", value=f"${spread:.2f}" if spread else "$0.00")
    
    # Simulation results
    with col2:
        st.subheader("Simulation Results")
        results_placeholder = st.empty()
    
    # Draw the initial order book visualization
    with orderbook_placeholder.container():
        fig = plot_orderbook(order_book)
        st.plotly_chart(fig, use_container_width=True)
    
    # Simulation logic
    if simulate_button:
        # Start measuring execution time
        start_time = time.time()
        
        # Calculate market depth for Almgren-Chriss model
        bids = order_book.get_bids(10)
        asks = order_book.get_asks(10)
        
        bid_depth = sum(qty for _, qty in bids)
        ask_depth = sum(qty for _, qty in asks)
        
        total_depth = bid_depth + ask_depth
        
        # Calculate Almgren-Chriss impact
        ac_impact = calculate_almgren_chriss_impact(
            quantity_usd, 
            total_depth,
            volatility,
            order_side.lower()
        )
        
        # Calculate Square Root impact (for comparison)
        sq_impact = np.sqrt(quantity_usd / (total_depth * (best_bid[0] if best_bid else 1))) * volatility * 100
        if order_side.lower() == "sell":
            sq_impact = -sq_impact
        
        # Run the simulation based on the selected method
        if use_ml_model:
            # Check if we should use enhanced ML models
            if regression_model_type != "Standard Linear":
                # Use enhanced ML-based prediction with advanced models
                actual_result = simulate_market_order_with_enhanced_ml(
                    order_book,
                    order_side.lower(),
                    quantity_usd,
                    volatility,
                    fee_percentage,
                    model_type,
                    risk_level_code
                )
            else:
                # Use standard ML-based prediction
                actual_result = simulate_market_order_with_ml(
                    order_book,
                    order_side.lower(),
                    quantity_usd,
                    volatility,
                    fee_percentage,
                    risk_level_code
                )
            
            # For ML model, expected and actual are the same since this is a prediction
            expected_metrics = actual_result
            actual_metrics = actual_result
        elif impact_model == "Almgren-Chriss" and ALMGREN_CHRISS_AVAILABLE:
            # Use the Almgren-Chriss model for simulation
            market_resilience = 0.3  # Could be made a UI parameter
            market_cap = None  # Could be made a UI parameter or fetched from an API
            
            actual_result = simulate_market_order_with_impact(
                order_book,
                order_side.lower(),
                quantity_usd,
                volatility,
                market_resilience,
                market_cap,
                fee_percentage,
                use_calibration
            )
            
            # For Almgren-Chriss, expected and actual are the same
            expected_metrics = actual_result
            actual_metrics = actual_result
            
            # Save the execution data for ML training
            try:
                data_collector = SlippageDataCollector()
                data_collector.add_execution_data(
                    order_result=actual_result,
                    order_book=order_book,
                    volatility=volatility
                )
                logging.info(f"Saved execution data for ML training, current records: {len(data_collector.data)}")
                
                # Also save to our execution history for dashboard
                save_execution_to_history(
                    actual_result,
                    order_side.lower(),
                    quantity_usd,
                    order_book
                )
            except Exception as e:
                logging.error(f"Error saving execution data: {e}")
        else:
            # Run the standard simulation
            expected_result = simulate_market_order(
                order_book,
                order_side.lower(),
                quantity_usd,
                price_impact_factor
            )
            
            # Calculate expected metrics
            expected_metrics = calculate_order_metrics(expected_result, fee_percentage)
            
            # Simulate execution delay
            if execution_delay > 0:
                time.sleep(execution_delay / 1000)  # Convert ms to seconds
            
            # Apply volatility to get the "actual" results
            actual_result = apply_volatility(expected_result, volatility)
            
            # Calculate actual metrics
            actual_metrics = calculate_order_metrics(actual_result, fee_percentage)
            
            # Save the execution data for ML training
            try:
                data_collector = SlippageDataCollector()
                data_collector.add_execution_data(
                    order_result=actual_metrics,
                    order_book=order_book,
                    volatility=volatility
                )
                logging.info(f"Saved execution data for ML training, current records: {len(data_collector.data)}")
                
                # Also save to our execution history for dashboard
                save_execution_to_history(
                    actual_metrics,
                    order_side.lower(),
                    quantity_usd,
                    order_book
                )
            except Exception as e:
                logging.error(f"Error saving execution data: {e}")
        
        # Store results in session state
        st.session_state.simulation_results = {
            "expected": expected_metrics,
            "actual": actual_metrics,
            "order_book": order_book,  # Store order book state for visualization
            "highlight_levels": actual_result if "execution_detail" in actual_result else None,
            "market": market,
            "model_type": "ml" if use_ml_model else "almgren_chriss" if impact_model == "Almgren-Chriss" else "simulation"
        }
        
        # Add simulation metadata if missing
        if use_ml_model and "execution_time_ms" not in actual_metrics:
            actual_metrics["execution_time_ms"] = 0
            actual_metrics["levels_walked"] = 0
            actual_metrics["execution_detail"] = []
        
        # Calculate internal latency
        end_time = time.time()
        internal_latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Add latency to the metrics
        actual_metrics["internal_latency_ms"] = internal_latency
        actual_metrics["external_latency_ms"] = execution_delay
        
        # Add impact model metrics to all simulation flows
        actual_metrics["ac_impact_pct"] = ac_impact
        actual_metrics["sq_impact_pct"] = sq_impact
        actual_metrics["selected_impact_model"] = impact_model
        
        # Update the order book visualization with executed levels
        with orderbook_placeholder.container():
            fig = plot_orderbook(order_book, highlight_levels=actual_metrics)
            st.plotly_chart(fig, use_container_width=True)
        
        # Update market metrics display
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        spread = order_book.get_spread()
        
        with metrics_cols[0]:
            st.metric(label="Best Bid", value=f"${best_bid[0]:.2f}" if best_bid else "$0.00")
        with metrics_cols[1]:
            st.metric(label="Best Ask", value=f"${best_ask[0]:.2f}" if best_ask else "$0.00")
        with metrics_cols[2]:
            st.metric(label="Spread", value=f"${spread:.2f}" if spread else "$0.00")
        
        # Display simulation results
        with results_placeholder.container():
            if actual_metrics["success"]:
                # Success message
                if use_ml_model:
                    st.success(f"ML-based slippage prediction completed successfully.")
                else:
                    st.success(f"Order simulation completed successfully.")
                
                # Results container
                results_container = st.container()
                
                # Results metrics
                st.markdown("### Order Execution Metrics")
                
                # Expected vs Actual metrics in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    if use_ml_model:
                        st.markdown("**ML Prediction Details**")
                        st.metric(label="Predicted Price", value=format_currency(actual_metrics["avg_execution_price"]))
                        st.metric(label="Predicted Slippage", value=format_percentage(actual_metrics["slippage_pct"]))
                        st.metric(label="Trading Fee", value=format_currency(actual_metrics["fee_usd"]))
                        st.metric(label="Total Cost", value=format_currency(actual_metrics["total_cost"]))
                        st.metric(label="Risk Level", value=actual_metrics.get("risk_level", "Default"))
                        
                        # Show model type and risk level
                        model_display_name = next((k for k, v in model_mapping.items() if v == model_type), model_type)
                        st.metric(label="Model Type", value=model_display_name)
                        
                        # Maker/Taker information
                        st.metric(label="Taker Proportion", value=f"{actual_metrics.get('taker_proportion', 1.0)*100:.0f}%")
                        
                        # If prediction contains quantile information, show it
                        if "prediction" in actual_metrics and "all_predictions" in actual_metrics["prediction"]:
                            st.markdown("**Prediction Ranges**")
                            predictions = actual_metrics["prediction"]["all_predictions"]
                            for q_name, q_value in predictions.items():
                                if q_name != risk_level_code:  # Skip the currently selected one
                                    st.metric(label=f"Slippage ({q_name})", value=format_percentage(q_value/100))
                    else:
                        st.markdown("**Expected Values**")
                        st.metric(label="Expected Price", value=format_currency(expected_metrics["avg_execution_price"]))
                        st.metric(label="Expected Slippage", value=format_percentage(expected_metrics["slippage_pct"]))
                        st.metric(label="Expected Fee", value=format_currency(expected_metrics["fee_usd"]))
                        st.metric(label="Expected Impact", value=format_currency(expected_metrics["slippage_usd"]))
                        st.metric(label="Expected Net Cost", value=format_currency(expected_metrics["total_cost"]))
                        
                        # Maker/Taker information
                        st.metric(label="Taker Proportion", value=f"{expected_metrics.get('taker_proportion', 1.0)*100:.0f}%")
                
                with col2:
                    if use_ml_model:
                        st.markdown("**Order Details**")
                        st.metric(label="Market", value=market)
                        st.metric(label="Side", value=order_side)
                        st.metric(label="Quantity USD", value=format_currency(quantity_usd))
                        st.metric(label="Executed Quantity", value=f"{actual_metrics['executed_quantity']:.6f}")
                        st.metric(label="Fee Percentage", value=f"{fee_percentage*100:.2f}%")
                        
                        # Display weighted fee if available
                        if "weighted_fee_pct" in actual_metrics:
                            weighted_fee = actual_metrics["weighted_fee_pct"] * 100
                            st.metric(label="Weighted Fee", value=f"{weighted_fee:.2f}%")
                    else:
                        st.markdown("**Actual Values**")
                        st.metric(
                            label="Actual Price",
                            value=format_currency(actual_metrics["avg_execution_price"]),
                            delta=format_currency(actual_metrics["avg_execution_price"] - expected_metrics["avg_execution_price"])
                        )
                        st.metric(
                            label="Actual Slippage",
                            value=format_percentage(actual_metrics["slippage_pct"]),
                            delta=format_percentage(actual_metrics["slippage_pct"] - expected_metrics["slippage_pct"])
                        )
                        st.metric(
                            label="Trading Fee",
                            value=format_currency(actual_metrics["fee_usd"]),
                            delta=format_currency(actual_metrics["fee_usd"] - expected_metrics["fee_usd"])
                        )
                        st.metric(
                            label="Market Impact",
                            value=format_currency(actual_metrics["slippage_usd"]),
                            delta=format_currency(actual_metrics["slippage_usd"] - expected_metrics["slippage_usd"])
                        )
                        st.metric(
                            label="Net Cost",
                            value=format_currency(actual_metrics["total_cost"]),
                            delta=format_currency(actual_metrics["total_cost"] - expected_metrics["total_cost"])
                        )
                        
                        # Maker/Taker information
                        st.metric(label="Taker Proportion", value=f"{actual_metrics.get('taker_proportion', 1.0)*100:.0f}%")
                        
                        # Display weighted fee if available
                        if "weighted_fee_pct" in actual_metrics:
                            weighted_fee = actual_metrics["weighted_fee_pct"] * 100
                            st.metric(label="Weighted Fee", value=f"{weighted_fee:.2f}%")
                
                # Add Impact Model Comparison
                st.markdown("### Impact Model Comparison")
                impact_cols = st.columns(3)
                with impact_cols[0]:
                    st.metric(
                        label="Standard Impact", 
                        value=format_percentage(actual_metrics["slippage_pct"]),
                        help="Impact calculated from actual price movement"
                    )
                with impact_cols[1]:
                    st.metric(
                        label="Almgren-Chriss Impact", 
                        value=format_percentage(actual_metrics["ac_impact_pct"]),
                        delta=format_percentage(actual_metrics["ac_impact_pct"] - actual_metrics["slippage_pct"]),
                        help="Impact estimated using Almgren-Chriss model"
                    )
                with impact_cols[2]:
                    st.metric(
                        label="Square Root Impact", 
                        value=format_percentage(actual_metrics["sq_impact_pct"]),
                        delta=format_percentage(actual_metrics["sq_impact_pct"] - actual_metrics["slippage_pct"]),
                        help="Impact estimated using Square Root model"
                    )
                
                # Add visual comparison of impact models
                impact_data = pd.DataFrame({
                    'Model': ['Actual', 'Almgren-Chriss', 'Square Root'],
                    'Impact (%)': [
                        actual_metrics["slippage_pct"],
                        actual_metrics["ac_impact_pct"],
                        actual_metrics["sq_impact_pct"]
                    ]
                })
                
                impact_fig = px.bar(
                    impact_data, 
                    x='Model', 
                    y='Impact (%)',
                    title='Market Impact Model Comparison',
                    color='Model',
                    height=300
                )
                st.plotly_chart(impact_fig, use_container_width=True)
                
                # Latency information
                st.markdown("### Performance Metrics")
                latency_cols = st.columns(2)
                with latency_cols[0]:
                    st.metric(label="Internal Latency", value=f"{actual_metrics['internal_latency_ms']:.2f} ms")
                with latency_cols[1]:
                    st.metric(label="External Latency", value=f"{actual_metrics['external_latency_ms']:.2f} ms")
                
                # Additional details expandable section
                with st.expander("Order Details"):
                    # Order parameters
                    st.markdown(f"**Order Parameters**")
                    st.markdown(f"- Symbol: {market}")
                    st.markdown(f"- Side: {order_side}")
                    st.markdown(f"- Quantity: ${quantity_usd:.2f}")
                    st.markdown(f"- Fee Tier: {fee_tier}")
                    st.markdown(f"- Selected Impact Model: {impact_model}")
                    
                    # Maker/Taker information
                    st.markdown(f"**Maker/Taker Classification**")
                    maker_proportion = actual_metrics.get("maker_proportion", 0.0)
                    taker_proportion = actual_metrics.get("taker_proportion", 1.0)
                    st.markdown(f"- Maker Proportion: {maker_proportion*100:.1f}%")
                    st.markdown(f"- Taker Proportion: {taker_proportion*100:.1f}%")
                    st.markdown(f"- Classification Method: {actual_metrics.get('prediction_method', 'rule-based')}")
                    
                    # Display maker/taker proportions as pie chart
                    maker_taker_data = pd.DataFrame({
                        'Type': ['Maker', 'Taker'],
                        'Proportion': [maker_proportion, taker_proportion]
                    })
                    
                    maker_taker_fig = px.pie(
                        maker_taker_data, 
                        values='Proportion', 
                        names='Type',
                        title='Maker/Taker Proportions',
                        color='Type',
                        color_discrete_map={'Maker': 'blue', 'Taker': 'red'}
                    )
                    st.plotly_chart(maker_taker_fig, use_container_width=True)
                    
                    if "weighted_fee_pct" in actual_metrics:
                        weighted_fee = actual_metrics["weighted_fee_pct"] * 100
                        st.markdown(f"- Weighted Fee Rate: {weighted_fee:.4f}%")
                        standard_fee = fee_percentage * 100
                        fee_savings = standard_fee - weighted_fee
                        if fee_savings > 0:
                            st.markdown(f"- Potential Fee Savings: {fee_savings:.4f}%")
                    
                    # Regression vs Actual Slippage Comparison
                    if use_ml_model:
                        st.markdown("**Regression Model Performance**")
                        
                        # Create a comparison table
                        if "prediction" in actual_metrics:
                            pred = actual_metrics["prediction"]
                            
                            # For enhanced models, we have predictions from multiple models
                            if "all_model_predictions" in pred:
                                predictions = pred["all_model_predictions"]
                                st.markdown(f"- Model Used: {model_display_name}")
                                
                                # Create dataframe for visualization
                                pred_data = []
                                for model_name, pred_value in predictions.items():
                                    # Map internal model names to display names
                                    display_name = next((k for k, v in model_mapping.items() if v == model_name), model_name)
                                    pred_data.append({
                                        'Model': display_name,
                                        'Predicted Slippage (bps)': pred_value
                                    })
                                
                                pred_df = pd.DataFrame(pred_data)
                                
                                # Sort by predicted slippage
                                pred_df = pred_df.sort_values('Predicted Slippage (bps)', ascending=False)
                                
                                # Display table
                                st.dataframe(pred_df, use_container_width=True)
                                
                                # Bar chart of predictions
                                pred_fig = px.bar(
                                    pred_df,
                                    x='Model',
                                    y='Predicted Slippage (bps)', 
                                    title='Slippage Predictions by Model',
                                    height=300
                                )
                                
                                # Add a line for the selected model
                                selected_value = predictions[model_type]
                                pred_fig.add_hline(
                                    y=selected_value, 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text=f"Selected ({model_display_name})"
                                )
                                
                                st.plotly_chart(pred_fig, use_container_width=True)
                                
                                # Feature importance if available
                                if regression_model_type != "Standard Linear":
                                    st.markdown("**Feature Importance**")
                                    st.markdown("The key factors influencing slippage in this prediction:")
                                    
                                    # Initialize model to get feature importance
                                    model = EnhancedSlippageModel()
                                    imp_fig = model.plot_feature_importance(model_type, "bar")
                                    
                                    if imp_fig:
                                        st.plotly_chart(imp_fig, use_container_width=True)
                                    else:
                                        st.info("Feature importance visualization not available. Train the model first.")
                            elif "all_predictions" in pred:
                                st.markdown("**Regression Model Performance**")
                                
                                # Create a comparison table
                                pred = pred["all_predictions"]
                                
                                st.markdown(f"- Predicted Slippage (Selected): {format_percentage(pred[risk_level_code])}")
                                
                                # Create dataframe for visualization
                                pred_data = []
                                for q_name, q_value in pred.items():
                                    pred_data.append({
                                        'Model': q_name,
                                        'Predicted Slippage (%)': q_value
                                    })
                                
                                pred_df = pd.DataFrame(pred_data)
                                
                                # Sort by predicted slippage
                                pred_df = pred_df.sort_values('Predicted Slippage (%)', ascending=False)
                                
                                # Display table
                                st.dataframe(pred_df, use_container_width=True)
                                
                                # Bar chart of predictions
                                pred_fig = px.bar(
                                    pred_df,
                                    x='Model',
                                    y='Predicted Slippage (%)', 
                                    title='Slippage Predictions by Model',
                                    height=300
                                )
                                
                                # Add a line for the selected model
                                selected_value = pred[risk_level_code]
                                pred_fig.add_hline(
                                    y=selected_value, 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text=f"Selected ({risk_level_code})"
                                )
                                
                                st.plotly_chart(pred_fig, use_container_width=True)
                    
                    # Execution details
                    st.markdown(f"**Execution Details**")
                    st.markdown(f"- Executed Quantity: {actual_metrics['executed_quantity']:.6f}")
                    st.markdown(f"- Price Levels Walked: {actual_metrics['levels_walked']}")
                    st.markdown(f"- Order Complete: {actual_metrics['is_complete']}")
                    
                    # Create a DataFrame with the execution details
                    if actual_metrics["execution_detail"]:
                        detail_df = pd.DataFrame(actual_metrics["execution_detail"])
                        st.dataframe(detail_df)
            else:
                # Error message
                st.error(f"Simulation failed: {actual_metrics['error']}")
    
    # Reset button logic
    if reset_button:
        # Reset the app state
        st.session_state.simulation_results = None
        st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.caption("Market Order Simulator | Data from OKX | Built with Streamlit")

    # Add the new visualization tabs
    with st.expander("Advanced Visualizations", expanded=True):
        viz_tabs = st.tabs([
            "Order Book Heatmap", 
            "Execution Analysis", 
            "Execution Quality Dashboard",
            "Scenario Analysis"
        ])
        
        # 1. Order Book Heatmap tab
        with viz_tabs[0]:
            st.subheader("Order Book Heatmap")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                heatmap_depth = st.slider("Depth to Display", 5, 50, 20, 
                                         help="Number of price levels to display in the heatmap")
            with col2:
                normalize = st.checkbox("Normalize Quantities", value=True, 
                                      help="Normalize quantities for better visualization")
            with col3:
                show_table = st.checkbox("Show Table View", value=False,
                                       help="Display order book data in tabular format")
            
            # Add visualization type options
            viz_options = st.columns([1, 1, 1])
            with viz_options[0]:
                viz_type = st.selectbox(
                    "Visualization Type",
                    ["Standard Heatmap", "Enhanced Depth View", "Step Chart"],
                    help="Select the visualization type best suited for your data"
                )
            with viz_options[1]:
                use_log_scale = st.checkbox("Use Log Scale", value=False,
                                         help="Apply logarithmic scaling to better visualize different quantity sizes")
            with viz_options[2]:
                highlight_spread = st.checkbox("Highlight Spread", value=True,
                                          help="Highlight the spread between bid and ask")
            
            # Display the heatmap or alternative visualization based on selection
            if viz_type == "Standard Heatmap":
                heatmap_fig = plot_orderbook_heatmap(st.session_state.order_book, heatmap_depth, normalize)
                st.plotly_chart(heatmap_fig, use_container_width=True)
            elif viz_type == "Enhanced Depth View":
                # Get order book data
                bids = st.session_state.order_book.get_bids(heatmap_depth)
                asks = st.session_state.order_book.get_asks(heatmap_depth)
                
                # Create enhanced depth view
                fig = go.Figure()
                
                # Process bid data
                if bids:
                    bid_prices = [float(bid[0]) for bid in bids]
                    bid_quantities = [float(bid[1]) for bid in bids]
                    
                    # Apply log scale if selected
                    if use_log_scale and any(q > 0 for q in bid_quantities):
                        bid_quantities = [np.log10(q) if q > 0 else 0 for q in bid_quantities]
                    
                    # Add bids as filled area
                    fig.add_trace(go.Scatter(
                        x=bid_prices,
                        y=bid_quantities,
                        fill='tozeroy',
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(0, 128, 0, 0.5)',
                        name='Bids',
                        hovertemplate='Price: $%{x:.2f}<br>Quantity: %{text}',
                        text=[f"{q:.6f}" for q in [float(bid[1]) for bid in bids]]
                    ))
                
                # Process ask data
                if asks:
                    ask_prices = [float(ask[0]) for ask in asks]
                    ask_quantities = [float(ask[1]) for ask in asks]
                    
                    # Apply log scale if selected
                    if use_log_scale and any(q > 0 for q in ask_quantities):
                        ask_quantities = [np.log10(q) if q > 0 else 0 for q in ask_quantities]
                    
                    # Add asks as filled area
                    fig.add_trace(go.Scatter(
                        x=ask_prices,
                        y=ask_quantities,
                        fill='tozeroy',
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(255, 0, 0, 0.5)',
                        name='Asks',
                        hovertemplate='Price: $%{x:.2f}<br>Quantity: %{text}',
                        text=[f"{q:.6f}" for q in [float(ask[1]) for ask in asks]]
                    ))
                
                # Highlight the spread if requested
                if highlight_spread and bids and asks:
                    best_bid = max(float(bid[0]) for bid in bids)
                    best_ask = min(float(ask[0]) for ask in asks)
                    spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / 2
                    
                    fig.add_shape(
                        type="rect",
                        x0=best_bid,
                        x1=best_ask,
                        y0=0,
                        y1=max(fig.data[0]['y']) if len(fig.data) > 0 and len(fig.data[0]['y']) > 0 else 1,
                        fillcolor="rgba(150, 150, 150, 0.2)",
                        line=dict(width=0),
                        layer="below"
                    )
                    
                    # Add spread annotation
                    fig.add_annotation(
                        x=mid_price,
                        y=0,
                        text=f"Spread: ${spread:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="#636363",
                        ax=0,
                        ay=-30,
                        bordercolor="#c7c7c7",
                        borderwidth=2,
                        borderpad=4,
                        bgcolor="#ff7f0e",
                        opacity=0.8
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"Enhanced Order Book Depth View{' (Log Scale)' if use_log_scale else ''}",
                    xaxis_title="Price",
                    yaxis_title=f"{'Log10(Quantity)' if use_log_scale else 'Quantity'}",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Step Chart":
                # Get order book data
                bids = st.session_state.order_book.get_bids(heatmap_depth)
                asks = st.session_state.order_book.get_asks(heatmap_depth)
                
                # Create step chart visualization
                fig = go.Figure()
                
                # Process bid data for cumulative representation
                if bids:
                    bid_prices = [float(bid[0]) for bid in bids]
                    bid_quantities = [float(bid[1]) for bid in bids]
                    bid_cumulative = np.cumsum(bid_quantities)
                    
                    # Apply log scale if selected
                    if use_log_scale and any(q > 0 for q in bid_cumulative):
                        bid_cumulative = [np.log10(q) if q > 0 else 0 for q in bid_cumulative]
                    
                    # Add bids as step chart - reverse the order for bid side
                    fig.add_trace(go.Scatter(
                        x=bid_prices[::-1],
                        y=bid_cumulative[::-1],
                        mode='lines',
                        line=dict(shape='hv', width=3, color='green'),
                        name='Cumulative Bids',
                        hovertemplate='Price: $%{x:.2f}<br>Cumulative Quantity: %{text}',
                        text=[f"{q:.6f}" for q in np.cumsum([float(bid[1]) for bid in bids])[::-1]]
                    ))
                
                # Process ask data for cumulative representation
                if asks:
                    ask_prices = [float(ask[0]) for ask in asks]
                    ask_quantities = [float(ask[1]) for ask in asks]
                    ask_cumulative = np.cumsum(ask_quantities)
                    
                    # Apply log scale if selected
                    if use_log_scale and any(q > 0 for q in ask_cumulative):
                        ask_cumulative = [np.log10(q) if q > 0 else 0 for q in ask_cumulative]
                    
                    # Add asks as step chart
                    fig.add_trace(go.Scatter(
                        x=ask_prices,
                        y=ask_cumulative,
                        mode='lines',
                        line=dict(shape='hv', width=3, color='red'),
                        name='Cumulative Asks',
                        hovertemplate='Price: $%{x:.2f}<br>Cumulative Quantity: %{text}',
                        text=[f"{q:.6f}" for q in np.cumsum([float(ask[1]) for ask in asks])]
                    ))
                
                # Highlight the spread if requested
                if highlight_spread and bids and asks:
                    best_bid = max(float(bid[0]) for bid in bids)
                    best_ask = min(float(ask[0]) for ask in asks)
                    spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Add vertical lines for best bid and ask
                    fig.add_shape(
                        type="line",
                        x0=best_bid, y0=0,
                        x1=best_bid, y1=max(fig.data[0]['y']) if len(fig.data) > 0 and len(fig.data[0]['y']) > 0 else 1,
                        line=dict(color="darkgreen", width=2, dash="dash")
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=best_ask, y0=0,
                        x1=best_ask, y1=max(fig.data[1]['y']) if len(fig.data) > 1 and len(fig.data[1]['y']) > 0 else 1,
                        line=dict(color="darkred", width=2, dash="dash")
                    )
                    
                    # Add spread annotation
                    fig.add_annotation(
                        x=mid_price,
                        y=0,
                        text=f"Spread: ${spread:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="#636363",
                        ax=0,
                        ay=-30,
                        bordercolor="#c7c7c7",
                        borderwidth=2,
                        borderpad=4,
                        bgcolor="#ff7f0e",
                        opacity=0.8
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"Order Book Depth - Step Chart{' (Log Scale)' if use_log_scale else ''}",
                    xaxis_title="Price",
                    yaxis_title=f"{'Log10(Cumulative Quantity)' if use_log_scale else 'Cumulative Quantity'}",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display tabular view if requested
            if show_table:
                st.subheader("Order Book Data (Tabular View)")
                
                # Get order book data
                bids = st.session_state.order_book.get_bids(heatmap_depth)
                asks = st.session_state.order_book.get_asks(heatmap_depth)
                
                # Create dataframes
                if bids:
                    bid_df = pd.DataFrame(bids, columns=["Price", "Quantity"])
                    bid_df["Cumulative"] = bid_df["Quantity"].cumsum()
                    bid_df["Side"] = "Bid"
                    bid_df["Value ($)"] = bid_df["Price"].astype(float) * bid_df["Quantity"].astype(float)
                    bid_df = bid_df[["Side", "Price", "Quantity", "Cumulative", "Value ($)"]]
                else:
                    bid_df = pd.DataFrame(columns=["Side", "Price", "Quantity", "Cumulative", "Value ($)"])
                
                if asks:
                    ask_df = pd.DataFrame(asks, columns=["Price", "Quantity"])
                    ask_df["Cumulative"] = ask_df["Quantity"].cumsum()
                    ask_df["Side"] = "Ask"
                    ask_df["Value ($)"] = ask_df["Price"].astype(float) * ask_df["Quantity"].astype(float)
                    ask_df = ask_df[["Side", "Price", "Quantity", "Cumulative", "Value ($)"]]
                else:
                    ask_df = pd.DataFrame(columns=["Side", "Price", "Quantity", "Cumulative", "Value ($)"])
                
                # Format the dataframes
                for df in [bid_df, ask_df]:
                    if not df.empty:
                        df["Price"] = df["Price"].astype(float).map("${:.2f}".format)
                        df["Quantity"] = df["Quantity"].astype(float).map("{:.6f}".format)
                        df["Cumulative"] = df["Cumulative"].astype(float).map("{:.6f}".format)
                        df["Value ($)"] = df["Value ($)"].map("${:.2f}".format)
                
                # Display two tables side-by-side
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Bids")
                    st.dataframe(bid_df, use_container_width=True)
                
                with col2:
                    st.subheader("Asks")
                    st.dataframe(ask_df, use_container_width=True)
                
                # Add summary metrics
                st.subheader("Order Book Summary")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                # Get bid/ask prices
                best_bid = st.session_state.order_book.get_best_bid()
                best_bid_price = best_bid[0] if best_bid else 0
                best_ask = st.session_state.order_book.get_best_ask()
                best_ask_price = best_ask[0] if best_ask else 0
                spread = best_ask_price - best_bid_price if best_bid and best_ask else 0
                
                # Calculate volumes for metrics
                total_bid_volume = sum(float(bid[1]) for bid in bids) if bids else 0
                total_ask_volume = sum(float(ask[1]) for ask in asks) if asks else 0
                total_volume = total_bid_volume + total_ask_volume
                
                with summary_col1:
                    # Calculate mid price
                    mid_price = (best_bid_price + best_ask_price) / 2 if best_bid and best_ask else 0
                    st.metric("Mid Price", f"${mid_price:.2f}")
                
                with summary_col2:
                    # Show spread
                    st.metric("Spread", f"${spread:.2f}")
                
                with summary_col3:
                    # Calculate spread in basis points
                    if best_bid_price > 0:
                        spread_bps = (spread / best_bid_price) * 10000
                        st.metric("Spread (bps)", f"{spread_bps:.2f}")
                    else:
                        st.metric("Spread (bps)", "N/A")
                
                with summary_col4:
                    # Calculate bid/ask ratio (imbalance)
                    if total_ask_volume > 0:
                        bid_ask_ratio = total_bid_volume / total_ask_volume
                        st.metric("Bid/Ask Ratio", f"{bid_ask_ratio:.2f}")
                    else:
                        st.metric("Bid/Ask Ratio", "N/A")
                
                # Add volume metrics in a second row
                volume_col1, volume_col2, volume_col3, volume_col4 = st.columns(4)
                
                with volume_col1:
                    st.metric("Bid Volume", f"{total_bid_volume:.6f}")
                
                with volume_col2:
                    st.metric("Ask Volume", f"{total_ask_volume:.6f}")
                
                with volume_col3:
                    st.metric("Total Volume", f"{total_volume:.6f}")
                
                with volume_col4:
                    # Calculate market pressure (positive = buying pressure, negative = selling pressure)
                    if total_volume > 0:
                        market_pressure = (total_bid_volume - total_ask_volume) / total_volume
                        pressure_text = f"{market_pressure:.2%}"
                        st.metric("Market Pressure", pressure_text, 
                                 delta=f"{'Buy' if market_pressure > 0 else 'Sell'} Pressure",
                                 delta_color="normal" if market_pressure > 0 else "inverse")
                    else:
                        st.metric("Market Pressure", "N/A")
        
        # 2. Execution Analysis tab - Walking the Book
        with viz_tabs[1]:
            st.subheader("Order Execution Analysis")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                exec_side = st.selectbox("Order Side", ["Buy", "Sell"], key="exec_side")
                
            with col2:
                exec_qty = st.number_input("Order Size (USD)", 
                                         min_value=100.0, max_value=500000.0, value=10000.0, step=1000.0,
                                         key="exec_qty")
            
            # Run the simulation when user clicks the button
            if st.button("Simulate Execution", key="exec_sim_btn"):
                with st.spinner("Simulating order execution..."):
                    # Run the simulation
                    exec_simulation = simulate_market_order(
                        st.session_state.order_book, exec_side.lower(), exec_qty
                    )
                    
                    # Store the simulation in session state
                    st.session_state.exec_simulation = exec_simulation
            
            # Show the execution visualization
            if 'exec_simulation' in st.session_state:
                if st.session_state.exec_simulation["success"]:
                    # Display the walking the book visualization
                    walk_fig = plot_order_walk_visualization(
                        st.session_state.exec_simulation, 
                        st.session_state.order_book
                    )
                    st.plotly_chart(walk_fig, use_container_width=True)
                    
                    # Display execution summary
                    metrics = calculate_order_metrics(st.session_state.exec_simulation, 0.001)
                    
                    st.subheader("Execution Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg. Execution Price", f"${metrics['avg_execution_price']:.2f}")
                    with col2:
                        st.metric("Slippage", f"{metrics['slippage_pct']:.2f}%")
                    with col3:
                        st.metric("Total Cost", f"${metrics['total_cost']:.2f}")
                else:
                    st.error("Simulation failed: Insufficient liquidity")
        
        # 3. Execution Quality Dashboard tab
        with viz_tabs[2]:
            st.subheader("Execution Quality Dashboard")
            
            if st.session_state.execution_history.empty:
                st.info("No execution history available. Execute some orders to see performance metrics.")
            else:
                # Create the dashboard
                dashboard_figs = create_execution_quality_dashboard(st.session_state.execution_history)
                
                # Display each figure in the dashboard
                for fig in dashboard_figs:
                    st.plotly_chart(fig, use_container_width=True)
        
        # 4. Scenario Analysis tab
        with viz_tabs[3]:
            st.subheader("Scenario Analysis")
            
            with st.form("scenario_form"):
                st.subheader("Add New Scenario")
                
                col1, col2 = st.columns(2)
                with col1:
                    scenario_name = st.text_input("Scenario Name", value=f"Scenario {len(st.session_state.scenarios) + 1}")
                    scenario_side = st.selectbox("Order Side", ["Buy", "Sell"], key="scenario_side")
                    scenario_qty = st.number_input(
                        "Order Size (USD)", 
                        min_value=100.0, 
                        max_value=1000000.0, 
                        value=10000.0,
                        step=1000.0,
                        key="scenario_qty"
                    )
                
                with col2:
                    scenario_model = st.selectbox(
                        "Execution Model", 
                        ["Basic", "ML-based", "Enhanced ML", "Almgren-Chriss"],
                        key="scenario_model",
                        format_func=lambda x: {
                            "Basic": "Basic (No ML)",
                            "ML-based": "ML Prediction",
                            "Enhanced ML": "Enhanced ML",
                            "Almgren-Chriss": "Almgren-Chriss Model"
                        }.get(x, x)
                    )
                    
                    scenario_volatility = st.slider(
                        "Volatility", 
                        min_value=0.0, 
                        max_value=1.0,
                        value=0.2,
                        step=0.05,
                        key="scenario_volatility"
                    )
                    
                    scenario_fee = st.slider(
                        "Fee Percentage",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.1,
                        step=0.01,
                        key="scenario_fee",
                        format="%.2f%%"
                    ) / 100  # Convert from percentage to decimal
                
                submit_scenario = st.form_submit_button("Add Scenario")
            
            if submit_scenario:
                with st.spinner("Running scenario simulation..."):
                    # Map the model selection to the API names
                    model_map = {
                        "Basic": "basic",
                        "ML-based": "ml",
                        "Enhanced ML": "enhanced_ml",
                        "Almgren-Chriss": "almgren_chriss"
                    }
                    model_type = model_map.get(scenario_model, "basic")
                    
                    # Run the simulation
                    simulation_result = run_scenario_simulation(
                        st.session_state.order_book,
                        scenario_side.lower(),
                        scenario_qty,
                        model_type,
                        scenario_volatility,
                        scenario_fee
                    )
                    
                    # Add to scenarios list
                    st.session_state.scenarios.append({
                        'name': scenario_name,
                        'side': scenario_side.lower(),
                        'quantity_usd': scenario_qty,
                        'model_type': model_type,
                        'volatility': scenario_volatility,
                        'fee_percentage': scenario_fee,
                        'simulation_result': simulation_result
                    })
                    
                    st.success(f"Added scenario: {scenario_name}")
            
            # Display existing scenarios
            if st.session_state.scenarios:
                st.subheader("Manage Scenarios")
                
                # Show a table of current scenarios
                scenario_df = pd.DataFrame([
                    {
                        'Name': s['name'],
                        'Side': s['side'].capitalize(),
                        'Order Size': f"${s['quantity_usd']:,.2f}",
                        'Model': s['model_type'].replace('_', ' ').title(),
                        'Avg. Price': f"${s['simulation_result'].get('avg_execution_price', 0):,.2f}",
                        'Slippage': f"{s['simulation_result'].get('slippage_bps', 0):.2f} bps"
                    } for s in st.session_state.scenarios
                ])
                
                st.dataframe(scenario_df, use_container_width=True)
                
                # Option to clear all scenarios
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("Clear All Scenarios"):
                        st.session_state.scenarios = []
                        st.experimental_rerun()
                
                # Display the scenario comparison visualizations
                if len(st.session_state.scenarios) > 0:
                    st.subheader("Scenario Comparison")
                    scenario_figs = create_scenario_analysis(
                        st.session_state.order_book,
                        st.session_state.scenarios
                    )
                    
                    # Display each comparison figure
                    for fig in scenario_figs:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add scenarios to compare different execution strategies.")


if __name__ == "__main__":
    main() 