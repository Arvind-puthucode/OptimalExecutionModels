import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import logging
from orderbook import OrderBook
from orderbook_client import OrderBookClient  # type: ignore
from market_order_simulator import simulate_market_order, calculate_order_metrics
import plotly.graph_objects as go


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


def on_orderbook_update(data):
    """Callback function to handle order book updates"""
    try:
        # Update the order book in session state
        st.session_state.order_book.update(data)
        st.session_state.last_update_time = time.time()
        logging.info(f"Updated order book: {len(data.get('bids', []))} bids, {len(data.get('asks', []))} asks")
    except Exception as e:
        logging.error(f"Error in order book update callback: {e}")


def init_client():
    """Initialize order book client and start connection"""
    # Get client singleton
    client = OrderBookClient.get_instance()
    
    # Register callback for order book updates
    client.remove_callback(on_orderbook_update)  # Remove if already registered
    client.add_callback(on_orderbook_update)
    
    # Start connection if not already connected
    if not client.is_connected():
        success = client.connect()
        st.session_state.connected = success
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


def main():
    # Initialize connection to order book data
    connection_status = init_client()
    
    # Sidebar inputs
    st.sidebar.header("Order Parameters")
    
    # Connection status indicator
    connection_indicator = st.sidebar.empty()
    if connection_status:
        connection_indicator.success("Connected to order book stream")
    else:
        connection_indicator.warning("Not connected to order book stream")
        if st.sidebar.button("Attempt Reconnection"):
            # Reset client and try to reconnect
            client = OrderBookClient.get_instance()
            client.reset()
            st.session_state.connected = False
            reconnected = init_client()
            if reconnected:
                st.sidebar.success("Successfully reconnected")
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to reconnect")
    
    # Add debug information in an expandable section
    with st.sidebar.expander("Debug Information", expanded=False):
        last_update = time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update_time)) if st.session_state.last_update_time > 0 else 'Never'
        
        st.markdown(f"Connection Status: {'Connected' if connection_status else 'Disconnected'}")
        st.markdown(f"Last Update Time: {last_update}")
        st.markdown(f"Connection Attempts: {st.session_state.connection_attempts}")
        st.markdown(f"Order Book Size: {len(st.session_state.order_book.bids)} bids, {len(st.session_state.order_book.asks)} asks")
        
        if st.button("Print Debug Info"):
            client = OrderBookClient.get_instance()
            logging.info(f"Client connected: {client.is_connected()}")
            logging.info(f"Session state connected: {st.session_state.connected}")
            logging.info(f"Order book state: {st.session_state.order_book.get_snapshot()}")
    
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
    
    # Action buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        simulate_button = st.button("Simulate", type="primary", use_container_width=True)
    with col2:
        reset_button = st.button("Reset", type="secondary", use_container_width=True)
    
    # Initialize session state for storing simulation results
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
    # Main panel
    col1, col2 = st.columns(2)
    
    # Get the order book - live or sample
    has_valid_data = connection_status and len(st.session_state.order_book.bids) > 0 and len(st.session_state.order_book.asks) > 0
    
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
        
        # Run the simulation
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
        
        # Calculate internal latency
        end_time = time.time()
        internal_latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Add latency to the metrics
        actual_metrics["internal_latency_ms"] = internal_latency
        actual_metrics["external_latency_ms"] = execution_delay
        
        # Store simulation results in session state
        st.session_state.simulation_results = {
            "expected": expected_metrics,
            "actual": actual_metrics,
            "order_book": order_book,
            "market": market
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
                st.success(f"Order simulation completed successfully.")
                
                # Results container
                results_container = st.container()
                
                # Results metrics
                st.markdown("### Order Execution Metrics")
                
                # Expected vs Actual metrics in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Expected Values**")
                    st.metric(label="Expected Price", value=format_currency(expected_metrics["avg_execution_price"]))
                    st.metric(label="Expected Slippage", value=format_percentage(expected_metrics["slippage_pct"]))
                    st.metric(label="Expected Fee", value=format_currency(expected_metrics["fee_usd"]))
                    st.metric(label="Expected Impact", value=format_currency(expected_metrics["slippage_usd"]))
                    st.metric(label="Expected Net Cost", value=format_currency(expected_metrics["total_cost"]))
                
                with col2:
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