# Market Order Simulator

A real-time market order simulator that visualizes order book data and simulates market order execution with slippage and fee calculation.

## Features

- Real-time L2 order book visualization
- Market order simulation with "walking the book" logic
- Slippage and trading fee calculation
- Market impact analysis
- Interactive Streamlit web interface

## Components

- `orderbook_client.py`: WebSocket client that connects to the order book data stream
- `orderbook.py`: Data structure to maintain the L2 order book state
- `market_order_simulator.py`: Logic to simulate market order execution
- `app.py`: Streamlit web interface

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

## Configuration

The simulation allows you to configure:

- Order quantity (in USD)
- Market volatility factor
- Fee tier
- Order side (buy/sell)
- Market (e.g., BTC-USDT-SWAP)
- Execution delay
- Price impact factor

## Data Source

The app connects to the following WebSocket endpoint to receive real-time order book data:
```
wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP
```

## License

MIT 