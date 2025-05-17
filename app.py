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
import plotly.graph_objects as go
import plotly.express as px


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
    
    # Order parameters
    quantity_usd = st.sidebar.number_input(
        "Quantity (USD)",
        min_value=10.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0,
        help="Amount in USD to trade"
    )
    
    volatility = st.sidebar.slider(
        "Market Volatility",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Simulated market volatility factor"
    )
    
    fee_tier = st.sidebar.selectbox(
        "Fee Tier",
        options=["Tier 1 (0.10%)", "Tier 2 (0.08%)", "Tier 3 (0.06%)", "Tier 4 (0.04%)"],
        index=0,
        help="Trading fee tier"
    )
    
    # Extract fee percentage from the selected tier
    fee_percentage = float(fee_tier.split("(")[1].split("%")[0]) / 100
    
    # Order type selection
    order_side = st.sidebar.radio(
        "Order Side",
        options=["Buy", "Sell"],
        index=0
    )
    
    # Market selection
    market = st.sidebar.selectbox(
        "Market",
        options=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
        index=0,
        help="Select trading pair. Note: Currently, only BTC-USDT-SWAP is connected with real data."
    )
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
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
        
        # Impact model selection 
        impact_model = st.selectbox(
            "Impact Model",
            options=["Standard", "Almgren-Chriss", "Square Root"],
            index=0,
            help="Select market impact model for price impact calculation"
        )
        
        if use_ml_model:
            risk_level = st.select_slider(
                "Risk Level",
                options=["Conservative (q90)", "Moderate (q75)", "Balanced (q50)", "Aggressive (q25)", "Optimistic (q10)", "Mean"],
                value="Balanced (q50)",
                help="Risk level for slippage prediction. Higher percentiles give more conservative estimates."
            )
            # Extract the risk level code from the selection
            risk_level_code = risk_level.split("(")[1].split(")")[0] if "(" in risk_level else "mean"
        else:
            risk_level_code = "q50"  # Default
    
    # Action buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        simulate_button = st.button("Simulate", type="primary", use_container_width=True)
    with col2:
        reset_button = st.button("Reset", type="secondary", use_container_width=True)
    with col3:
        if st.button("Train Model", use_container_width=True):
            st.sidebar.info("Training slippage prediction model...")
            
            # Create a progress bar
            progress_bar = st.sidebar.progress(0)
            
            # Initialize data collector
            collector = SlippageDataCollector()
            
            # Check if we have enough data
            try:
                X, y = collector.get_training_data()
                
                # Update progress
                progress_bar.progress(30)
                
                # Initialize and train model
                model = SlippageModel()
                training_result = model.train(X, y)
                
                # Update progress
                progress_bar.progress(100)
                
                # Show training results
                st.sidebar.success(f"Model trained successfully with {training_result['samples_count']} samples")
                st.sidebar.info(f"R² Score: {training_result['linear_r2_score']:.4f}")
            except ValueError as e:
                st.sidebar.error(f"Error training model: {str(e)}")
                st.sidebar.info("Collecting more data from order executions...")
            finally:
                # Remove progress bar
                progress_bar.empty()
    
    # Initialize session state for storing simulation results
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
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
    
    # Simulate button logic
    if simulate_button:
        # Record the start time
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
            # Use ML-based prediction
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
            except Exception as e:
                logging.error(f"Error saving execution data: {e}")
        
        # If using ML model, set actual_metrics to be the same as the prediction
        if use_ml_model:
            actual_metrics = actual_result
            
            # Add simulation metadata
            actual_metrics["execution_time_ms"] = 0
            actual_metrics["levels_walked"] = 0
            actual_metrics["execution_detail"] = []
        
        # Calculate internal latency
        end_time = time.time()
        internal_latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Add latency to the metrics
        actual_metrics["internal_latency_ms"] = internal_latency
        actual_metrics["external_latency_ms"] = execution_delay
        
        # Add impact model metrics
        actual_metrics["ac_impact_pct"] = ac_impact
        actual_metrics["sq_impact_pct"] = sq_impact
        actual_metrics["selected_impact_model"] = impact_model
        
        # Store simulation results in session state
        st.session_state.simulation_results = {
            "expected": expected_metrics,
            "actual": actual_metrics,
            "order_book": order_book,
            "market": market,
            "model_type": "ml" if use_ml_model else "simulation"
        }
        
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
                    if use_ml_model and "prediction" in actual_metrics:
                        st.markdown("**Regression Model Performance**")
                        
                        # Create a comparison table
                        pred = actual_metrics["prediction"]
                        
                        if "all_predictions" in pred:
                            predictions = pred["all_predictions"]
                            st.markdown(f"- Predicted Slippage (Selected): {format_percentage(actual_metrics['slippage_pct'])}")
                            
                            # Create dataframe for visualization
                            pred_data = []
                            for q_name, q_value in predictions.items():
                                pred_data.append({
                                    'Model': f"{q_name}",
                                    'Predicted Slippage (%)': q_value/100
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
                            
                            pred_fig.add_hline(
                                y=actual_metrics['slippage_pct'], 
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


if __name__ == "__main__":
    main() 