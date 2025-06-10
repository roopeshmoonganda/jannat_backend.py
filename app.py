import os
import json
import time
from datetime import datetime, timedelta
import requests
import math
import numpy as np
from collections import deque
import threading
import logging # Import logging module
import sys # Import sys module for stdout
import pytz # For timezone-aware market hours

# --- Fyers API V3 Imports ---
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

# --- Configuration ---
# Use environment variable for Flask backend URL, default for local testing
FLASK_BACKEND_URL = os.environ.get("FLASK_BACKEND_URL", "https://jannat-backend-py.onrender.com")

# This path must match the persistent disk mount path configured in Render
# Set this as an environment variable in Render for your service, e.g., /var/data
PERSISTENT_DISK_BASE_PATH = os.environ.get("PERSISTENT_DISK_PATH", "./jannat_data") # Default to local folder for testing

# Fyers API credentials (pulled from environment variables)
FYERS_APP_ID = os.environ.get("FYERS_APP_ID")
FYERS_SECRET_ID = os.environ.get("FYERS_SECRET_ID")
FYERS_REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI") # e.g., https://your-backend-service.onrender.com/fyers_auth_callback

# File paths for persistent storage on Render's disk or local directory
ACCESS_TOKEN_STORAGE_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_access_token.json")
CAPITAL_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_capital.json")
TRADE_LOG_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_trade_log.json")
ALGO_LOG_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_algo_engine.log")

# Define a separate directory for Fyers WebSocket logs
FYERS_WS_LOG_DIR = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_ws_logs")

# Trading Parameters (from your original snippet, maintain these)
BASE_CAPITAL = 100000.0
# Define the symbols your strategies will trade and monitor
# Ensure these are valid Fyers symbols (e.g., "NSE:RELIANCE-EQ", "NSE:NIFTY50-INDEX")
# This will be used for WebSocket subscription and strategy.
SYMBOLS_TO_MONITOR = ["NSE:NIFTY50-INDEX", "NSE:BANKNIFTY-INDEX", "NSE:RELIANCE-EQ"]
# You would define your actual trading symbols here, e.g.,
# TRADING_SYMBOLS = ["NSE:RELIANCE-EQ", "NSE:INFY-EQ"]

WINDOW_SIZE = 100 # For storing historical data if needed for indicators
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 26
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
STOP_LOSS_PERCENT = 0.005  # 0.5%
TARGET_PERCENT = 0.01      # 1%

# --- Configure Logging for Algo Engine ---
# This setup sends logs to both stdout (for Render's console) and a persistent file.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG to capture all messages

# Clear existing handlers to prevent duplicate output if this block runs multiple times (e.g., in development)
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close() # Important: close file handlers to release file locks

# Create a StreamHandler to output logs to stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s'))
logger.addHandler(stream_handler)

# Create a FileHandler to output logs to a persistent file
try:
    os.makedirs(os.path.dirname(ALGO_LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(ALGO_LOG_FILE)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Algo Engine logging to persistent file: {ALGO_LOG_FILE}")
except Exception as e:
    logger.error(f"Failed to set up file logger in jannat_algo_engine: {e}", exc_info=True)


# --- Global Variables for Algo State and Data ---
current_capital_data = {"current_balance": 0.0, "pnl_today": 0.0} # Managed by algo, exposed to frontend
active_trades = [] # List of dictionaries for currently active (open) trades
fyers = None # Fyers API client instance for REST calls (e.g., placing orders)
websocket_client = None # Fyers WebSocket client instance for real-time data
latest_prices = {} # Dictionary to store latest prices from WebSocket: {'symbol': {'ltp': price, 'timestamp': time_str}}

# Data structures for your strategy (from your snippet)
historical_data = {symbol: deque(maxlen=WINDOW_SIZE) for symbol in SYMBOLS_TO_MONITOR}
trade_signals_queue = deque(maxlen=100) # Assuming this is used for signals


# --- Helper Functions for Persistence ---

def load_capital_data():
    """Loads capital data from a JSON file, creating if not exists."""
    os.makedirs(os.path.dirname(CAPITAL_FILE), exist_ok=True) # Ensure directory exists
    if os.path.exists(CAPITAL_FILE):
        try:
            with open(CAPITAL_FILE, 'r') as f:
                content = f.read()
                if content:
                    return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Capital data file {CAPITAL_FILE} is corrupted. Initializing with BASE_CAPITAL.", exc_info=True)
        except Exception as e:
            logger.error(f"Error loading capital data: {e}", exc_info=True)
    logger.info("Capital data file not found or corrupted. Initializing with BASE_CAPITAL.")
    return {"current_balance": BASE_CAPITAL, "pnl_today": 0.0}

def save_capital_data(data):
    """Saves current capital data to a JSON file."""
    os.makedirs(os.path.dirname(CAPITAL_FILE), exist_ok=True) # Ensure directory exists
    try:
        with open(CAPITAL_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug("Capital data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving capital data: {e}", exc_info=True)

def log_trade(trade_data):
    """Appends a closed trade's data to a JSON log file for historical trades."""
    os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True) # Ensure directory exists
    
    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE, 'r') as f:
                content = f.read()
                if content:
                    trades = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Trade log file {TRADE_LOG_FILE} is corrupted. Starting new log.", exc_info=True)
            trades = []
        except Exception as e:
            logger.error(f"Error reading trade log file: {e}", exc_info=True)

    trades.append(trade_data)

    try:
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(trades, f, indent=4)
        logger.info(f"Trade logged: {trade_data.get('symbol')} {trade_data.get('signal')} @ {trade_data.get('entry_price')} PnL: {trade_data.get('pnl'):.2f}")
    except Exception as e:
        logger.error(f"Error writing trade log: {e}", exc_info=True)


# --- Fyers API & WebSocket Integration ---

def load_access_token():
    """Loads the Fyers access token from the persistent storage file."""
    if not os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
        logger.error(f"Access token file not found at {ACCESS_TOKEN_STORAGE_FILE}. "
                     "Please ensure Fyers authentication has completed successfully via the backend /login endpoint.")
        return None
    try:
        with open(ACCESS_TOKEN_STORAGE_FILE, 'r') as f:
            token_data = json.load(f)
            return token_data.get("access_token")
    except json.JSONDecodeError:
        logger.error(f"Error decoding access token JSON from {ACCESS_TOKEN_STORAGE_FILE}. File might be corrupted.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error loading access token: {e}", exc_info=True)
        return None

def init_fyers_client():
    """Initializes and returns the FyersModel client for REST API calls."""
    global fyers
    access_token = load_access_token()
    if not access_token:
        logger.error("Fyers access token not available. Cannot initialize REST client.")
        fyers = None
        return None
    
    if not all([FYERS_APP_ID, FYERS_SECRET_ID]): # Redirect URI not strictly needed for client init
        logger.error("Missing Fyers API credentials (APP_ID or SECRET_ID). Cannot initialize REST client.")
        fyers = None
        return None

    try:
        fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=FYERS_APP_ID)
        # You can add a small test call here to verify the token, e.g., get_profile()
        logger.info("Fyers REST client initialization attempt complete.")
        return fyers
    except Exception as e:
        logger.error(f"Error initializing Fyers REST client: {e}", exc_info=True)
        fyers = None
        return None

def connect_fyers_websocket():
    """Establishes and manages Fyers WebSocket connection for real-time data."""
    global websocket_client

    access_token = load_access_token()
    if not access_token:
        logger.error("No Fyers access token found for WebSocket connection. Skipping WS connection.")
        return False

    client_id_for_ws = FYERS_APP_ID
    if "-" in FYERS_APP_ID:
        client_id_for_ws = FYERS_APP_ID.split("-")[0] 

    if websocket_client and websocket_client.is_connected:
        logger.info("Fyers WebSocket already connected.")
        return True
    elif websocket_client: # Client object exists but not connected, attempt reconnect
        logger.info("Fyers WebSocket not connected, but client exists. Attempting reconnect.")
        try:
            websocket_client.connect()
            return True
        except Exception as e:
            logger.error(f"Error reconnecting Fyers WebSocket: {e}", exc_info=True)
            websocket_client = None # Reset client object if reconnect fails
    
    def on_open(ws_app):
        logger.info("Fyers WebSocket connection opened.")
        # Subscribe to all symbols defined in SYMBOLS_TO_MONITOR
        ws_app.subscribe(symbols=SYMBOLS_TO_MONITOR, data_type="symbolData")
        logger.info(f"Subscribed to symbols for real-time data: {SYMBOLS_TO_MONITOR}")

    def on_message(ws_app, message):
        global latest_prices, historical_data
        # Process the received message and update the global latest_prices dictionary
        if isinstance(message, dict) and "symbol" in message and "ltp" in message:
            symbol = message["symbol"]
            ltp = message["ltp"]
            
            latest_prices[symbol] = {
                "ltp": ltp,
                "timestamp": datetime.now().isoformat()
            }
            # Append to historical data deque for indicator calculations
            if symbol in historical_data:
                historical_data[symbol].append(ltp)
            
            # logger.debug(f"Updated LTP for {symbol}: {ltp}") # This can be very verbose

    def on_error(ws_app, error):
        logger.error(f"Fyers WebSocket error: {error}", exc_info=True)

    def on_close(ws_app):
        logger.info("Fyers WebSocket connection closed.")
        global websocket_client
        websocket_client = None

    try:
        os.makedirs(FYERS_WS_LOG_DIR, exist_ok=True) # Ensure WS log directory exists
        websocket_client = data_ws.FyersDataSocket(
            access_token=f"{client_id_for_ws}:{access_token}",
            log_path=FYERS_WS_LOG_DIR, # Save WebSocket library logs to persistent disk
            litemode=True, # Subscribe to basic LTP updates for symbols
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            # Removed 'on_open' as per the TypeError traceback
            enable_websocket_log=True
        )
        logger.info("Attempting to connect Fyers WebSocket...")
        websocket_client.connect()
        return True
    except Exception as e:
        logger.error(f"Failed to connect Fyers WebSocket: {e}", exc_info=True)
        websocket_client = None
        return False

def close_fyers_websocket():
    """Closes the Fyers WebSocket connection if open."""
    global websocket_client
    if websocket_client and websocket_client.is_connected:
        logger.info("Closing Fyers WebSocket connection.")
        websocket_client.close()
        websocket_client = None


# --- Market Hours Check ---
def is_market_open():
    """Checks if the Indian stock market (NSE/BSE) is currently open."""
    ist = pytz.timezone('Asia/Kolkata')
    now_utc = datetime.now(pytz.utc) # Get current UTC time, aware
    now_ist = datetime.now(ist) # Get current IST time, aware

    market_open_time = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    is_weekday = 0 <= now_ist.weekday() <= 4 # Monday (0) to Friday (4)

    is_open = is_weekday and (market_open_time <= now_ist <= market_close_time)
    
    # --- ENHANCED DEBUGGING LOGS ---
    logger.debug(f"DEBUG: Current UTC time (aware): {now_utc}")
    logger.debug(f"DEBUG: Current IST time (aware): {now_ist}")
    logger.debug(f"DEBUG: Market Open Time IST: {market_open_time}")
    logger.debug(f"DEBUG: Market Close Time IST: {market_close_time}")
    logger.debug(f"DEBUG: Is Weekday ({now_ist.strftime('%A')}): {is_weekday}")
    logger.debug(f"DEBUG: Is Market Open calculated: {is_open}")
    # --- END DEBUG LOGS ---

    if not is_open:
        if not is_weekday:
            logger.debug(f"Market is closed: Weekend ({now_ist.strftime('%A')}).")
        elif now_ist < market_open_time:
            logger.debug(f"Market is closed: Before opening hours. Current IST: {now_ist.strftime('%H:%M:%S')}")
        elif now_ist > market_close_time:
            logger.debug(f"Market is closed: After closing hours. Current IST: {now_ist.strftime('%H:%M:%S')}")
    else:
        logger.debug(f"Market is OPEN. Current IST: {now_ist.strftime('%H:%M:%S')}")
    
    return is_open


# --- Strategy Indicator Calculations (Placeholders for your existing logic) ---
# Assuming your original code had functions like these.
# You would implement the actual calculation logic here using 'historical_data' or other data sources.

def calculate_ema(prices, period):
    if len(prices) < period:
        return None
    ema = [0.0] * len(prices)
    smoothing_factor = 2 / (period + 1)
    
    # Simple moving average for the first EMA calculation
    ema[period - 1] = sum(prices[:period]) / period
    
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * smoothing_factor + ema[i-1]
    return ema[-1] # Return the latest EMA

def calculate_rsi(prices, period=RSI_PERIOD):
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0] # Losses are positive values

    avg_gain = np.mean(gains[:period]) if len(gains) > 0 else 0
    avg_loss = np.mean(losses[:period]) if len(losses) > 0 else 0

    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50 # Avoid division by zero
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast_period=MACD_FAST_PERIOD, slow_period=MACD_SLOW_PERIOD, signal_period=MACD_SIGNAL_PERIOD):
    if len(prices) < max(fast_period, slow_period, signal_period):
        return None, None, None

    # This is a simplified calculation for the latest MACD, typically requires more historical EMAs
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    if ema_fast is None or ema_slow is None:
        return None, None, None

    macd_line = ema_fast - ema_slow
    
    # To calculate the MACD signal line, you'd need a series of MACD line values
    # For now, we'll return None for signal and histogram unless a series is available.
    # In a full implementation, you'd store a history of macd_line values and then calculate EMA of *that* for signal line.
    
    return macd_line, None, None # macd_line, signal_line, histogram


# --- Core Algo Engine Functionality ---

def execute_strategy(algo_status_dict, app_logger_instance):
    """
    Main function to execute the trading strategy. Runs in a separate thread.
    algo_status_dict: A shared dictionary to control the algo's running state from the Flask app.
    app_logger_instance: The Flask app's logger instance (optional, for Flask-specific logging within algo).
    """
    global fyers, websocket_client, current_capital_data, active_trades, latest_prices, historical_data

    logger.info("Jannat Algo Engine thread started. Initializing...")
    
    current_capital_data = load_capital_data() # Load initial capital data

    # Initialize Fyers REST client (for placing orders, etc.)
    fyers = init_fyers_client()
    if not fyers:
        logger.error("Fyers REST client initialization failed. Algo will run in simulated trading mode only (no real orders).")
        # Algo continues to run, but only simulates trades and updates local state.

    # Connect WebSocket for real-time data streaming
    connect_fyers_websocket()

    logger.info(f"Starting with capital: {current_capital_data['current_balance']:.2f}")

    # Main loop for the algo engine
    while algo_status_dict.get("status") == "running":
        try:
            # --- Market Hours Check ---
            if not is_market_open():
                logger.info("Market is closed. Pausing algo activity and closing WebSocket.")
                close_fyers_websocket()
                time.sleep(60) # Wait for 1 minute before re-checking market status
                continue # Skip trading logic until market re-opens
            
            # --- WebSocket Connection Check (if market is open) ---
            if not websocket_client or not websocket_client.is_connected:
                logger.warning("Market is OPEN, but WebSocket is not connected. Attempting to reconnect.")
                connect_fyers_websocket() # Attempt to reconnect
                if not websocket_client or not websocket_client.is_connected:
                    logger.error("Failed to connect WebSocket. Real-time data will not be streamed for decisions.")
                    time.sleep(30) # Wait before next attempt to avoid hammering
                    continue

            logger.info("Algo engine running: Market is open and WebSocket is active.")

            # --- YOUR TRADING STRATEGY GOES HERE ---
            # This is where you will integrate your RSI, MACD, etc., logic.
            # You now have access to 'latest_prices' and 'historical_data' from WebSocket.

            for symbol in SYMBOLS_TO_MONITOR:
                current_ltp_info = latest_prices.get(symbol)
                
                if not current_ltp_info:
                    logger.debug(f"No real-time data for {symbol} yet. Skipping strategy for this symbol.")
                    continue

                current_ltp = current_ltp_info["ltp"]
                
                # Ensure enough historical data for calculations
                if len(historical_data[symbol]) < max(EMA_SLOW_PERIOD, RSI_PERIOD, MACD_SLOW_PERIOD) + 1:
                    logger.info(f"Collecting historical data for {symbol}. Current points: {len(historical_data[symbol])}/{WINDOW_SIZE}")
                    # Continue collecting data; strategy won't activate until enough data points.
                    continue
                
                # --- Calculate Indicators ---
                # Example: You would use historical_data[symbol] (which is a deque of prices)
                # to calculate your indicators.
                prices_list = list(historical_data[symbol]) # Convert deque to list for numpy/slicing
                
                # Replace with your actual indicator calculations
                current_rsi = calculate_rsi(prices_list)
                macd_line, macd_signal_line, macd_histogram = calculate_macd(prices_list)
                
                logger.debug(f"Symb
