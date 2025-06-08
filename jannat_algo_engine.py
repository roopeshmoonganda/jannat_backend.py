import os
import json
import time
from datetime import datetime, timedelta
import requests
import math
import numpy as np
from collections import deque # For storing recent ticks for candle reconstruction
import threading # For running the algo in a separate thread, if needed by Flask

# --- Fyers API V3 Imports ---
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws # Import for data WebSocket client

# --- Configuration ---
# Use environment variable for Flask backend URL, default to localhost for testing
FLASK_BACKEND_URL = os.environ.get("FLASK_BACKEND_URL", "https://jannat-backend-py.onrender.com") 

# File paths for persistent storage on Render's disk or local directory
# This path needs to be correctly set up on your Render service as a persistent disk mount point.
PERSISTENT_DISK_BASE_PATH = os.environ.get("PERSISTENT_DISK_PATH", ".") 
ACCESS_TOKEN_STORAGE_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_access_token.json")
CAPITAL_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_capital.json")
TRADE_LOG_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_trade_log.json")

# Define a separate directory for Fyers WebSocket logs
FYERS_WS_LOG_DIR = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_ws_logs")

# Trading Parameters
BASE_CAPITAL = 100000.0
# Changed SYMBOL_SPOT_BASE_NAME to just the instrument name, not with "NSE:"
# This allows dynamic generation of futures/options symbols easily
SYMBOL_SPOT_BASE_NAME = "BANKNIFTY" 
OPTION_EXPIRY_DAYS_AHEAD = 7 # Used in get_nearest_expiry_date (Placeholder, actual expiry logic needed)
BANKNIFTY_STRIKE_INTERVAL = 100
NIFTY_STRIKE_INTERVAL = 50 # Not used for BANKNIFTY, but good to have if you expand

# Strategy Parameters (Supertrend and MACD)
SUPER_TREND_PERIOD = 10
SUPER_TREND_MULTIPLIER = 3
FAST_MA_PERIOD = 12
SLOW_MA_PERIOD = 26
SIGNAL_PERIOD = 9

# Trade Management Parameters
TARGET_MULTIPLIER = 0.005 # 0.5% target
STOP_LOSS_MULTIPLIER = 0.002 # 0.2% stop loss
TRADE_QUANTITY_PER_LOT_BANKNIFTY = 15 # Example: BankNifty lot size
MAX_ACTIVE_TRADES = 1 # Max number of concurrent trades

# --- Global Variables for Algo Engine State ---
active_trades = {} # Stores details of current active trades (key: symbol, value: trade_object)
# Initialize with default structure, actual values loaded/reset later
jannat_capital = {"current_balance": BASE_CAPITAL, "pnl_today": 0.0, "last_day_reset": datetime.now().isoformat()}

# --- Global for Tick Data and Candle Reconstruction ---
# Dictionary to hold ticks for each symbol: {'NSE:BANKNIFTY24SEPFUT': deque([(timestamp, price), ...]), ...}
symbol_ticks = {}
# Dictionary to store last completed candles: {'NSE:BANKNIFTY24SEPFUT': [timestamp, open, high, low, close, volume], ...}
last_completed_candles = {}
CANDLE_RESOLUTION_SECONDS = 300 # 5 minutes for example (5 * 60)

# Global dictionary to store latest prices for subscribed symbols
latest_prices = {}
websocket_client = None # To store the Fyers WebSocket client instance

# Logger placeholder (will be replaced by Flask app.logger)
class SimpleLogger:
    def info(self, message):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - {message}")
    def warning(self, message):
        print(f"[WARNING] {datetime.now().strftime('%H:%M:%S')} - {message}")
    def error(self, message, exc_info=False): # Added exc_info for full tracebacks
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - {message}")
        if exc_info:
            import traceback
            traceback.print_exc()
    def debug(self, message): # Added for debug messages
        print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} - {message}")

logger = SimpleLogger() # Default logger, will be overridden by Flask's app.logger


# --- Fyers WebSocket Data Handlers ---
def on_message(message):
    """
    Callback function to process incoming messages from Fyers WebSocket.
    Processes both tick data and general messages.
    """
    global latest_prices, logger
    if 'symbol' in message and 'ltp' in message:
        latest_prices[message['symbol']] = message['ltp']
        logger.debug(f"Received tick for {message['symbol']}: LTP = {message['ltp']}")
        # This will also populate symbol_ticks for candle reconstruction
        on_ticks_callback(websocket_client, [message]) # Pass message as a list of one tick
    elif 's' in message and message['s'] == 'ok' and 'msg' in message:
        logger.info(f"WebSocket Message: {message['msg']}")
    else:
        logger.debug(f"Received non-tick WebSocket message: {message}")

def on_error(message):
    global logger
    logger.error(f"WebSocket Error: {message}")

def on_close():
    global logger
    logger.info("WebSocket connection closed.")

def on_open(): # Renamed from on_connect for clarity but used as on_connect in FyersDataSocket
    global logger
    logger.info("WebSocket connection opened.")

# --- Helper Functions ---

def get_fyers_access_token():
    """Reads the Fyers access token from a file."""
    if not os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
        logger.error(f"Access token file not found at {ACCESS_TOKEN_STORAGE_FILE}. Please authenticate via the Flask backend.")
        return None
    try:
        with open(ACCESS_TOKEN_STORAGE_FILE, 'r') as f:
            token_data = json.load(f)
            return token_data.get('access_token')
    except json.JSONDecodeError:
        logger.error("Error decoding access token JSON. File might be corrupted.")
        return None
    except Exception as e:
        logger.error(f"Error reading access token: {e}", exc_info=True)
        return None

def save_capital_data():
    """Saves the current capital data to a persistent file."""
    try:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(CAPITAL_FILE), exist_ok=True)
        with open(CAPITAL_FILE, 'w') as f:
            json.dump(jannat_capital, f, indent=4)
        logger.info("Capital data saved.")
    except Exception as e:
        logger.error(f"Error saving capital data: {e}", exc_info=True)

def load_capital_data():
    """Loads capital data from a persistent file, or initializes if not found."""
    global jannat_capital
    
    # Define a default structure for jannat_capital to ensure all keys are present
    default_capital_data = {
        "current_balance": BASE_CAPITAL,
        "pnl_today": 0.0,
        "last_day_reset": datetime.now().isoformat()
    }

    if os.path.exists(CAPITAL_FILE):
        try:
            with open(CAPITAL_FILE, 'r') as f:
                loaded_data = json.load(f)
            
            # Merge loaded data with defaults to ensure all keys are present
            jannat_capital = {**default_capital_data, **loaded_data}
            
            # Check for new day reset
            last_reset_date = datetime.fromisoformat(jannat_capital.get("last_day_reset", datetime.min.isoformat())).date()
            if last_reset_date < datetime.now().date():
                logger.info(f"New day detected. Resetting PnL from {jannat_capital.get('pnl_today', 0.0):.2f} to 0.")
                jannat_capital["pnl_today"] = 0.0
                jannat_capital["last_day_reset"] = datetime.now().isoformat()
                save_capital_data() # Save the reset PnL
            logger.info(f"Capital data loaded: {jannat_capital}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error loading capital data ({e}). Initializing with base capital.", exc_info=True)
            jannat_capital = default_capital_data # Use default data for initialization
            save_capital_data()
    else:
        logger.info("No existing capital data found. Initializing with base capital.")
        jannat_capital = default_capital_data # Use default data for initialization
        save_capital_data()

def log_trade(trade_details):
    """Logs trade details to a JSON file."""
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
    
    trade_log = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE, 'r') as f:
                content = f.read()
                if content: # Check if file is not empty
                    trade_log = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Trade log file {TRADE_LOG_FILE} is corrupted. Starting new log.")
            trade_log = [] # Reset if corrupted
        except Exception as e:
            logger.error(f"Error reading trade log: {e}", exc_info=True)
            trade_log = [] # Reset on other errors

    try:
        trade_log.append(trade_details)
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(trade_log, f, indent=4)
        logger.info(f"Trade logged: {trade_details}")
    except Exception as e:
        logger.error(f"Error writing trade log: {e}", exc_info=True)

def is_market_open():
    """Checks if the market is open for trading (e.g., NSE equity market hours)."""
    now = datetime.now()
    # IST Market Hours: 9:15 AM to 3:30 PM
    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    # Check if it's a weekday (Monday=0 to Friday=4)
    if now.weekday() < 5 and market_open_time <= now <= market_close_time:
        return True
    return False

def get_current_month_futures_symbol():
    """
    Generates the current month's BankNifty futures symbol in the format:
    NSE:BANKNIFTYYYMONFUT (e.g., NSE:BANKNIFTY25JUNFUT)
    """
    now = datetime.now()
    year_short = now.strftime('%y') # Last two digits of the year (e.g., 25)
    month_abbr = now.strftime('%b').upper() # Three-letter month abbreviation (e.g., JUN, JUL)
    return f"NSE:{SYMBOL_SPOT_BASE_NAME}{year_short}{month_abbr}FUT"

def get_current_option_symbol(base_symbol_name, expiry_date, is_call, strike_price):
    """Constructs the Fyers option symbol."""
    # Example: NSE:BANKNIFTY24SEP46000CE
    # Format: EXCHANGE:INSTRUMENT + YY + MON (3 chars) + Strike + PE/CE
    
    year_short = expiry_date.strftime('%y')
    month_short = expiry_date.strftime('%b').upper() # e.g., JAN, FEB
    option_type = "CE" if is_call else "PE"
    strike_str = str(int(strike_price))

    return f"NSE:{base_symbol_name}{year_short}{month_short}{strike_str}{option_type}"


def get_current_atm_strike(current_spot_price, strike_interval):
    """Calculates the At-The-Money (ATM) strike price."""
    return round(current_spot_price / strike_interval) * strike_interval

def get_nearest_expiry_date(option_expiry_days_ahead):
    """Finds the nearest weekly or monthly expiry date for options."""
    # NOTE: This is a simplified logic. For accurate expiry,
    # you'd ideally fetch a list of valid expiry dates from Fyers API or master data.
    today = datetime.now().date()
    
    # Try to find next Thursday (weekly expiry for BankNifty/Nifty)
    days_until_thursday = (3 - today.weekday() + 7) % 7 # 3 is Thursday (Mon=0 to Sun=6)
    next_thursday = today + timedelta(days=days_until_thursday)

    # If the calculated next_thursday is today and it's past market close,
    # or if today is Thursday and it's already a holiday/passed expiry,
    # then it should roll to the next Thursday.
    # This simplified logic might not cover all edge cases like holidays or monthly expiries.
    # For a robust system, you'd query Fyers for actual expiry dates.
    return next_thursday


# --- Indicator Calculations ---

def calculate_supertrend(candles, period, multiplier):
    """Calculates Supertrend indicator."""
    if len(candles) < period:
        return [], [] # Return empty lists if not enough data

    highs = np.array([c[2] for c in candles]) # [timestamp, open, high, low, close, volume]
    lows = np.array([c[3] for c in candles])
    closes = np.array([c[4] for c in candles])

    # Calculate Average True Range (ATR)
    tr = np.zeros(len(closes))
    if len(closes) > 0:
        tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    
    atr = np.array([np.mean(tr[max(0, i - period + 1):i+1]) for i in range(len(tr))])

    # Calculate Basic Upper Band and Lower Band
    basic_upper_band = (highs + lows) / 2 + multiplier * atr
    basic_lower_band = (highs + lows) / 2 - multiplier * atr

    # Calculate Final Upper Band and Lower Band
    final_upper_band = np.copy(basic_upper_band)
    final_lower_band = np.copy(basic_lower_band)

    for i in range(1, len(closes)):
        if closes[i-1] > final_upper_band[i-1]:
            final_upper_band[i] = max(basic_upper_band[i], final_upper_band[i-1])
        else:
            final_upper_band[i] = basic_upper_band[i]

        if closes[i-1] < final_lower_band[i-1]:
            final_lower_band[i] = min(basic_lower_band[i], final_lower_band[i-1])
        else:
            final_lower_band[i] = basic_lower_band[i]

    # Calculate Supertrend
    supertrend = np.zeros(len(closes))
    trend = np.zeros(len(closes)) # 1 for uptrend, -1 for downtrend

    for i in range(len(closes)):
        if i == 0:
            if closes[i] > ((highs[i] + lows[i]) / 2):
                trend[i] = 1
            else:
                trend[i] = -1
            supertrend[i] = basic_lower_band[i] if trend[i] == 1 else basic_upper_band[i]
        else:
            if closes[i] > supertrend[i-1] and trend[i-1] == 1:
                trend[i] = 1
                supertrend[i] = max(final_lower_band[i], supertrend[i-1])
            elif closes[i] < supertrend[i-1] and trend[i-1] == -1:
                trend[i] = -1
                supertrend[i] = min(final_upper_band[i], supertrend[i-1])
            elif closes[i] > supertrend[i-1] and trend[i-1] == -1:
                trend[i] = 1
                supertrend[i] = final_lower_band[i]
            elif closes[i] < supertrend[i-1] and trend[i-1] == 1:
                trend[i] = -1
                supertrend[i] = final_upper_band[i]
            else:
                trend[i] = trend[i-1]
                supertrend[i] = supertrend[i-1]


    return supertrend.tolist(), trend.tolist()

def calculate_macd(candles, fast_period, slow_period, signal_period):
    """Calculates MACD indicator."""
    min_required_candles = max(fast_period, slow_period) + signal_period - 1 
    if len(candles) < min_required_candles:
        return [], [], []

    closes = np.array([c[4] for c in candles])

    ema_fast = _calculate_ema(closes, fast_period)
    ema_slow = _calculate_ema(closes, slow_period)

    # MACD Line
    if len(ema_fast) > len(ema_slow):
        macd_line = ema_fast[-len(ema_slow):] - ema_slow
    elif len(ema_slow) > len(ema_fast):
        macd_line = ema_fast - ema_slow[-len(ema_fast):]
    else:
        macd_line = ema_fast - ema_slow

    if len(macd_line) < signal_period:
        return macd_line.tolist(), [], []

    signal_line = _calculate_ema(macd_line, signal_period)
    
    # Histogram
    histogram = macd_line[-len(signal_line):] - signal_line

    return macd_line.tolist(), signal_line.tolist(), histogram.tolist()

def _calculate_ema(data, period):
    """Helper to calculate Exponential Moving Average (EMA)."""
    if len(data) < period:
        return np.array([])
    
    ema_values = np.zeros_like(data, dtype=float)
    
    ema_values[period - 1] = np.mean(data[:period])
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(data)):
        ema_values[i] = ((data[i] - ema_values[i-1]) * multiplier) + ema_values[i-1]

    return ema_values[period-1:]


# --- Tick Data Processing and Candle Reconstruction ---

def initialize_tick_data_buffers(symbols):
    """Initializes deque for each symbol to store incoming ticks."""
    for symbol in symbols:
        symbol_ticks[symbol] = deque() # Stores (timestamp, price) tuples

def on_ticks_callback(ws_client, ticks):
    """
    Callback function to process incoming ticks from Fyers WebSocket.
    """
    global symbol_ticks, last_completed_candles, logger
    for tick in ticks:
        symbol = tick.get('symbol')
        timestamp_seconds = tick.get('timestamp') 
        ltp = tick.get('ltp')

        if symbol and timestamp_seconds and ltp is not None:
            tick_time = datetime.fromtimestamp(timestamp_seconds)
            
            if symbol not in symbol_ticks:
                symbol_ticks[symbol] = deque()
            
            symbol_ticks[symbol].append((tick_time, ltp))

            # Attempt to build a candle from the accumulated ticks
            build_candle_from_ticks(symbol)


def build_candle_from_ticks(symbol):
    """
    Reconstructs candles from accumulated ticks for a given symbol.
    """
    global symbol_ticks, last_completed_candles, logger, CANDLE_RESOLUTION_SECONDS

    if symbol not in symbol_ticks or not symbol_ticks[symbol]:
        return

    current_time = datetime.now()
    ticks_for_symbol = symbol_ticks[symbol]

    market_open_hour = 9
    market_open_minute = 15
    
    market_open_today = current_time.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
    
    if current_time < market_open_today:
        return # Market not yet open

    elapsed_seconds = (current_time - market_open_today).total_seconds()
    
    # Calculate the current candle's start time and its corresponding end time
    current_candle_start_seconds_offset = math.floor(elapsed_seconds / CANDLE_RESOLUTION_SECONDS) * CANDLE_RESOLUTION_SECONDS
    current_candle_start_time = market_open_today + timedelta(seconds=current_candle_start_seconds_offset)
    current_candle_end_time = current_candle_start_time + timedelta(seconds=CANDLE_RESOLUTION_SECONDS)

    # Process ticks that are *older* than or within the *just completed* candle interval.
    # Collect ticks that fall within the completed candle interval (i.e., time < current_candle_start_time)
    # This logic assumes ticks arrive roughly in order and attempts to finalize a candle
    # as soon as its interval ends.
    
    # This is a critical section for tick processing. A more robust solution for production
    # would involve a loop that identifies *all* completed candle intervals from the deque
    # and finalizes them sequentially before returning.
    
    # For now, let's simplify to process if the *first* tick in queue is past a candle end,
    # or if current time is past a candle end and we have some ticks.

    ticks_for_completed_candle = deque()
    # Move ticks that belong to a *completed* candle interval (i.e., their timestamp is less than
    # the start time of the *current* candle interval)
    while ticks_for_symbol and ticks_for_symbol[0][0] < current_candle_start_time:
        ticks_for_completed_candle.append(ticks_for_symbol.popleft())

    # If we have ticks for a completed candle and the current time is beyond its end
    if ticks_for_completed_candle and current_time >= current_candle_end_time:
        # The start time of this completed candle would be `current_candle_start_time - timedelta(seconds=CANDLE_RESOLUTION_SECONDS)`
        # because current_candle_start_time refers to the *current* interval's start.
        completed_candle_start_time = current_candle_start_time - timedelta(seconds=CANDLE_RESOLUTION_SECONDS)
        
        sorted_ticks = sorted(list(ticks_for_completed_candle), key=lambda x: x[0])

        if sorted_ticks:
            open_price = sorted_ticks[0][1]
            close_price = sorted_ticks[-1][1]
            high_price = max(t[1] for t in sorted_ticks)
            low_price = min(t[1] for t in sorted_ticks)
            volume = len(sorted_ticks) # Simple tick count as volume for now

            new_candle = [int(completed_candle_start_time.timestamp()), open_price, high_price, low_price, close_price, volume]
            
            # This is simplified: in reality, you'd append this to a list of historical candles
            # for the symbol, not just replace the last_completed_candles.
            last_completed_candles[symbol] = new_candle
            logger.info(f"New {CANDLE_RESOLUTION_SECONDS/60}-min candle for {symbol}: {new_candle}")
            # The strategy logic will pick this up in the main loop.


def start_fyers_websocket(access_token, app_logger_instance):
    """Initializes and starts the Fyers WebSocket connection for data."""
    global websocket_client, logger
    logger = app_logger_instance

    logger.info("Attempting to start Fyers WebSocket...")
    try:
        os.makedirs(FYERS_WS_LOG_DIR, exist_ok=True)
        logger.info(f"Fyers WebSocket log directory ensured: {FYERS_WS_LOG_DIR}")

        websocket_client = data_ws.FyersDataSocket(
            access_token=access_token,
            log_path=FYERS_WS_LOG_DIR, 
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_message=on_message,    # Corrected parameter name
            on_error=on_error,       # Corrected parameter name
            on_close=on_close,       # Corrected parameter name
            on_connect=on_open,      # Corrected parameter name (used on_open function)
            # Removed: run_background=True  <--- This line was removed based on previous error
        )

        logger.info("Connecting to Fyers WebSocket...")
        websocket_client.connect()
        logger.info("Fyers WebSocket connection initiated.")

        current_futures_symbol = get_current_month_futures_symbol()
        initial_symbols_to_watch = [current_futures_symbol]

        websocket_client.subscribe(symbols=initial_symbols_to_watch, data_type="symbolData") 
        logger.info(f"Subscribed to Fyers WebSocket for initial symbols: {initial_symbols_to_watch}")
        return websocket_client
    except Exception as e:
        logger.error(f"Failed to start Fyers WebSocket: {e}", exc_info=True)
        return None


# --- Trade Management Functions ---

def place_order_api(symbol, side, quantity, product_type, order_type, price=0, trade_mode='LIVE'):
    """Sends a trade request to the Flask backend's /trade/execute endpoint."""
    url = f"{FLASK_BACKEND_URL}/trade/execute"
    payload = {
        "symbol": symbol,
        "signal": "BUY" if side == 1 else "SELL", # Convert side (1/-1) to string (BUY/SELL)
        "quantity": quantity,
        "product_type": product_type,
        "order_type": order_type,
        "entryPrice": price, 
        "trade_mode": trade_mode
    }
    logger.info(f"Sending order placement request to backend for {symbol} ({'BUY' if side == 1 else 'SELL'})...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        if result.get("success"):
            logger.info(f"Order placed successfully for {symbol}. Order ID: {result.get('orderId')}")
            return True, result.get('orderId')
        else:
            logger.error(f"Failed to place order for {symbol}: {result.get('message')}")
            return False, result.get('message')
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error while placing order for {symbol}: {e}", exc_info=True)
        return False, str(e)


def monitor_and_manage_trade(trade_obj):
    """
    Monitors an active trade for stop-loss, target, or end-of-day exit.
    """
    global active_trades, jannat_capital, logger, latest_prices

    symbol = trade_obj['symbol']
    entry_price = trade_obj['entry_price']
    signal = trade_obj['signal']
    quantity = trade_obj['quantity']
    target_price = trade_obj['target_price']
    stop_loss_price = trade_obj['stop_loss_price']

    # Use the latest price from the global latest_prices dictionary
    current_price = latest_prices.get(symbol) # Get the real-time price of the traded option/future

    if current_price is None:
        logger.warning(f"Could not get current price for {symbol} from latest_prices for trade monitoring. Skipping this check.")
        return # Skip monitoring if no price available yet

    pnl = 0.0
    exit_reason = None
    exit_price = None

    if signal == 'BUY':
        if current_price >= target_price:
            pnl = (target_price - entry_price) * quantity
            exit_price = target_price
            exit_reason = "Target Hit"
        elif current_price <= stop_loss_price:
            pnl = (stop_loss_price - entry_price) * quantity
            exit_price = stop_loss_price
            exit_reason = "Stop Loss Hit"
    elif signal == 'SELL':
        if current_price <= target_price: # For SELL, target is lower than entry
            pnl = (entry_price - target_price) * quantity
            exit_price = target_price
            exit_reason = "Target Hit"
        elif current_price >= stop_loss_price: # For SELL, SL is higher than entry
            pnl = (entry_price - stop_loss_price) * quantity
            exit_price = stop_loss_price
            exit_reason = "Stop Loss Hit"

    # End of day square off (Example: square off at 3:25 PM IST)
    now = datetime.now()
    square_off_time = now.replace(hour=15, minute=25, second=0, microsecond=0)
    if now.weekday() < 5 and now >= square_off_time and exit_reason is None:
        pnl = (current_price - entry_price) * quantity if signal == 'BUY' else (entry_price - current_price) * quantity
        exit_price = current_price
        exit_reason = "End of Day Square Off"
        logger.info(f"Squaring off {symbol} trade due to end of day. Current PnL: {pnl:.2f}")


    if exit_reason:
        logger.info(f"Trade for {symbol} exited. Reason: {exit_reason}, PnL: {pnl:.2f}")
        jannat_capital['current_balance'] += pnl
        jannat_capital['pnl_today'] += pnl
        save_capital_data()

        trade_obj['exit_time'] = datetime.now().isoformat()
        trade_obj['exit_price'] = exit_price
        trade_obj['pnl'] = pnl
        trade_obj['status'] = "CLOSED"
        trade_obj['exit_reason'] = exit_reason
        log_trade(trade_obj) # Log the closed trade

        if symbol in active_trades:
            del active_trades[symbol] # Remove from active trades

        # Place exit order (if LIVE trade_mode)
        if trade_obj.get('trade_mode') == 'LIVE': # Only place live exit order if original trade was LIVE
            exit_side = -1 if signal == 'BUY' else 1 # Opposite side for exit (SELL if BUY, BUY if SELL)
            logger.info(f"Placing exit order for {symbol} at {exit_price}")
            success, message = place_order_api(symbol, exit_side, quantity, trade_obj['product_type'], 'MARKET', trade_mode='LIVE')
            if not success:
                logger.error(f"Failed to place exit order for {symbol}: {message}")


# --- Centralized Functions for Frontend Polling ---
def get_trade_details():
    """Provides current trade details for frontend."""
    return [trade for trade in active_trades.values()] # Return list of active trade objects

def get_capital_data():
    """Provides current capital data for frontend."""
    return jannat_capital


# --- Main Algo Engine Logic ---

def execute_strategy(algo_status_dict, app_logger_instance):
    """
    Main function to run the algorithmic trading strategy.
    This is designed to run in a separate thread.
    """
    global logger, websocket_client
    logger = app_logger_instance

    logger.info("Jannat Algo Engine started. Loading capital data...")
    load_capital_data()
    logger.info("Capital data loaded. Attempting Fyers API Setup for WebSocket...")

    access_token = get_fyers_access_token()
    if not access_token:
        logger.error("Cannot start algo: Fyers access token is missing or invalid. Stopping algo.")
        algo_status_dict["status"] = "stopped"
        return

    websocket_client = start_fyers_websocket(access_token, logger)
    if not websocket_client:
        logger.error("Fyers WebSocket initialization failed. Stopping algo.")
        algo_status_dict["status"] = "stopped"
        return
    
    logger.info("Fyers WebSocket initialized and connected. Entering main algo loop...")

    last_candle_processed_timestamp = {} 
    
    current_futures_symbol = get_current_month_futures_symbol()
    
    initialize_tick_data_buffers([current_futures_symbol])

    while algo_status_dict["status"] == "running":
        if not is_market_open():
            logger.info("Market is closed. Waiting for market open.")
            if websocket_client and websocket_client.is_connected:
                logger.info("Market closed, closing Fyers WebSocket connection.")
                # CORRECTED: Changed .disconnect() to .close_connection()
                websocket_client.close_connection() 
                websocket_client = None
            time.sleep(60) 
            continue
        
        if not websocket_client or not websocket_client.is_connected:
            logger.info("Market is open, reconnecting Fyers WebSocket.")
            websocket_client = start_fyers_websocket(access_token, logger)
            if not websocket_client:
                logger.error("Failed to reconnect WebSocket. Skipping this cycle.")
                time.sleep(10)
                continue

        processed_any_candle_in_this_iteration = False
        
        futures_candle = last_completed_candles.get(current_futures_symbol)

    # --- Main Algo Loop - Now driven by new candle availability ---
    last_candle_processed_timestamp = {} # Track the timestamp of the last candle we processed for each symbol
    
    # Get the current month's futures symbol
    current_futures_symbol = get_current_month_futures_symbol()
    
    # Initialize symbol_ticks for the futures symbol
    initialize_tick_data_buffers([current_futures_symbol])

    # NOTE ON INDICATOR HISTORY:
    # For accurate indicator calculations (like Supertrend, MACD), you NEED a history of `period` candles.
    # The `build_candle_from_ticks` function currently only provides the *latest* completed candle.
    # In a production system, you would maintain a `deque` or list of the last N completed candles
    # for each symbol (where N is greater than your largest indicator period).
    # This example proceeds with the latest candle, making indicators less reliable until a history is built.

    while algo_status_dict["status"] == "running":
        if not is_market_open():
            logger.info("Market is closed. Waiting for market open.")
            # Disconnect WebSocket if market is closed to save resources
            if websocket_client and websocket_client.is_connected:
                logger.info("Market closed, disconnecting Fyers WebSocket.")
                websocket_client.disconnect() # Correct method to close the connection
                websocket_client = None # Clear client
            time.sleep(60) # Check every minute if market is closed
            continue
        
        # If market just opened, reconnect WebSocket
        if not websocket_client or not websocket_client.is_connected:
            logger.info("Market is open, reconnecting Fyers WebSocket.")
            websocket_client = start_fyers_websocket(access_token, logger)
            if not websocket_client:
                logger.error("Failed to reconnect WebSocket. Skipping this cycle.")
                time.sleep(10) # Wait a bit before retrying
                continue

        processed_any_candle_in_this_iteration = False
        
        # Process futures candle for signal generation
        futures_candle = last_completed_candles.get(current_futures_symbol)
        
        if futures_candle:
            candle_timestamp = futures_candle[0] # Get timestamp of the completed candle

            if current_futures_symbol in last_candle_processed_timestamp and candle_timestamp == last_candle_processed_timestamp[current_futures_symbol]:
                pass # Already processed this candle, wait for a new one
            else:
                logger.info(f"Processing new candle for {current_futures_symbol}: {futures_candle}")
                last_candle_processed_timestamp[current_futures_symbol] = candle_timestamp
                processed_any_candle_in_this_iteration = True

                # IMPORTANT: Placeholder for actual historical data.
                # In a real system, you'd use a `deque` of the last X candles.
                num_required_candles = max(SUPER_TREND_PERIOD, SLOW_MA_PERIOD + SIGNAL_PERIOD)
                # This will make indicators very volatile for a new connection/symbol
                past_candles_for_indicators = [futures_candle] * num_required_candles 
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # CRITICAL AREA FOR IMPROVEMENT: Replace this with actual historical candle accumulation.


                # --- Strategy Logic ---
                supertrend_values, supertrend_trend = calculate_supertrend(
                    past_candles_for_indicators, SUPER_TREND_PERIOD, SUPER_TREND_MULTIPLIER
                )
                macd_line, signal_line, macd_histogram = calculate_macd(
                    past_candles_for_indicators, FAST_MA_PERIOD, SLOW_MA_PERIOD, SIGNAL_PERIOD
                )

                if not supertrend_values or not macd_line or not supertrend_trend or not signal_line or not macd_histogram:
                    logger.warning(f"Indicators not calculated for {current_futures_symbol}. Skipping trade signal generation due to insufficient data or calculation error.")
                    processed_any_candle_in_this_iteration = False 
                    continue

                latest_supertrend = supertrend_values[-1]
                latest_supertrend_trend = supertrend_trend[-1] # 1 for up, -1 for down
                latest_macd_line = macd_line[-1]
                latest_signal_line = signal_line[-1]
                latest_macd_histogram = macd_histogram[-1]

                current_futures_price_for_signal = futures_candle[4] # Use candle close as current price for this check

                logger.info(f"Strategy check for {current_futures_symbol}: ST: {latest_supertrend:.2f}, Trend: {latest_supertrend_trend}, MACD: {latest_macd_line:.2f}, Signal: {latest_signal_line:.2f}, Hist: {latest_macd_histogram:.2f}")

                trade_signal = None
                if latest_supertrend_trend == 1 and latest_macd_line > latest_signal_line and latest_macd_histogram > 0:
                    trade_signal = 'BUY'
                elif latest_supertrend_trend == -1 and latest_macd_line < latest_signal_line and latest_macd_histogram < 0:
                    trade_signal = 'SELL'
                
                # VIX filter (placeholder)
                # if not filter_high_vix(current_vix_value): 
                #     logger.info("High VIX detected. Not placing trades.")
                #     trade_signal = None


                if trade_signal and len(active_trades) < MAX_ACTIVE_TRADES:
                    logger.info(f"Strong trade signal detected for {current_futures_symbol}: {trade_signal}")

                    # Determine ATM strike and option symbol
                    atm_strike = get_current_atm_strike(current_futures_price_for_signal, BANKNIFTY_STRIKE_INTERVAL)
                    expiry_date = get_nearest_expiry_date(OPTION_EXPIRY_DAYS_AHEAD)
                    is_call = True if trade_signal == 'BUY' else False # Buy Call for BUY, Buy Put for SELL
                    option_symbol = get_current_option_symbol(SYMBOL_SPOT_BASE_NAME, expiry_date, is_call, atm_strike)
                    
                    target_symbol_to_trade = option_symbol # This is the option symbol to trade

                    # Subscribe to the specific option symbol if not already subscribed
                    if target_symbol_to_trade not in latest_prices and websocket_client:
                        try:
                            logger.info(f"Dynamically subscribing to option symbol for trade: {target_symbol_to_trade}")
                            websocket_client.subscribe(symbols=[target_symbol_to_trade], data_type="symbolData")
                            time.sleep(1.5) # Give a moment for initial tick to arrive
                        except Exception as e:
                            logger.error(f"Failed to subscribe to option symbol {target_symbol_to_trade} dynamically: {e}", exc_info=True)
                            continue # Skip this trade if subscription fails

                    # Fetch option_current_price from latest_prices for entry
                    option_current_price = latest_prices.get(target_symbol_to_trade)
                    if option_current_price is None:
                        logger.warning(f"No real-time price available for {target_symbol_to_trade} after subscription. Skipping order placement for this signal.")
                        continue # Skip this trade if no real-time price after attempting subscription
                    
                    logger.info(f"Real-time option price for {target_symbol_to_trade}: {option_current_price:.2f}")

                    product_type = "MIS" # Margin Intraday Square off
                    order_type = "MARKET" # Or "LIMIT" if you want to specify entry_price
                    quantity = TRADE_QUANTITY_PER_LOT_BANKNIFTY 

                    entry_price = option_current_price # Assume market order, so entry is current price

                    calculated_target = entry_price * (1 + TARGET_MULTIPLIER) if trade_signal == 'BUY' else entry_price * (1 - TARGET_MULTIPLIER)
                    calculated_stop_loss = entry_price * (1 - STOP_LOSS_MULTIPLIER) if trade_signal == 'BUY' else entry_price * (1 + STOP_LOSS_MULTIPLIER)

                    logger.info(f"Attempting to place {trade_signal} trade for {target_symbol_to_trade} @ {entry_price:.2f} (Target: {calculated_target:.2f}, SL: {calculated_stop_loss:.2f})")

                    # Place order
                    success, order_id = place_order_api(
                        target_symbol_to_trade,
                        1 if trade_signal == 'BUY' else -1, # side: 1 for BUY, -1 for SELL
                        quantity,
                        product_type,
                        order_type,
                        price=entry_price,
                        trade_mode='PAPER' # Change to 'LIVE' for live trading
                    )

                    if success:
                        trade_details_for_monitoring = {
                            "order_id": order_id,
                            "symbol": target_symbol_to_trade,
                            "signal": trade_signal,
                            "entry_time": datetime.now().isoformat(),
                            "entry_price": entry_price,
                            "target_price": calculated_target,
                            "stop_loss_price": calculated_stop_loss,
                            "quantity": quantity,
                            "product_type": product_type,
                            "order_type": order_type,
                            "trade_mode": 'PAPER', 
                            "status": "ACTIVE"
                        }
                        active_trades[target_symbol_to_trade] = trade_details_for_monitoring
                        log_trade(trade_details_for_monitoring)
                        logger.info(f"Trade placed successfully for {target_symbol_to_trade}. Now monitoring...")

                    else:
                        logger.error("Failed to place trade.")
                else:
                    if trade_signal:
                        logger.info(f"Trade signal for {current_futures_symbol} but maximum active trades ({MAX_ACTIVE_TRADES}) reached.")
                    else:
                        logger.debug("No strong trade signal or filter condition not met for current candle.")
        else:
            logger.debug(f"No new completed futures candle for {current_futures_symbol} available yet. Waiting for ticks...")
        
        # After processing signals, check/monitor active trades
        for symbol, trade_obj in list(active_trades.items()): 
            monitor_and_manage_trade(trade_obj)
        
        # Small sleep if no new candle was processed to prevent busy-waiting
        if not processed_any_candle_in_this_iteration:
            time.sleep(1) 

# --- Bonus Intelligence Functions (Placeholders) ---
def filter_high_vix(vix_value):
    """
    Placeholder for VIX filter.
    Requires fetching VIX data from an external source and defining a threshold.
    e.g., if vix_value > 20: return False (do not trade in high volatility)
    """
    return True # Always allow for now. Implement real VIX check later.

def get_current_spot_price(symbol):
    """
    Returns the latest known price for a symbol from the real-time tick data.
    """
    global latest_prices
    return latest_prices.get(symbol) # Safely get price, returns None if not found


# --- Initial Setup for direct run (for testing purposes) ---
if __name__ == "__main__":
    # When running directly, use SimpleLogger and simulate Flask app context
    print("Running Jannat Algo Engine directly for testing.")
    
    # Simulate an app_logger for direct execution
    class MockAppLogger:
        def info(self, message):
            print(f"[INFO][AppLogger] {datetime.now().strftime('%H:%M:%S')} - {message}")
        def warning(self, message):
            print(f"[WARNING][AppLogger] {datetime.now().strftime('%H:%M:%S')} - {message}")
        def error(self, message, exc_info=False):
            print(f"[ERROR][AppLogger] {datetime.now().strftime('%H:%M:%S')} - {message}")
            if exc_info:
                import traceback
                traceback.print_exc()
        def debug(self, message):
            print(f"[DEBUG][AppLogger] {datetime.now().strftime('%H:%M:%S')} - {message}")
    
    mock_app_logger = MockAppLogger()

    # Simulate algo_status_dict for direct execution
    mock_algo_status_dict = {"status": "running", "last_update": datetime.now().isoformat()}

    # Set up dummy environment variables for local testing without actual Fyers credentials
    os.environ["FYERS_APP_ID"] = "dummy_app_id"
    os.environ["FYERS_SECRET_ID"] = "dummy_secret"
    os.environ["FYERS_REDIRECT_URI"] = "http://localhost:5000/fyers_auth_callback"
    os.environ["FLASK_BACKEND_URL"] = "http://localhost:5000"
    os.environ["PERSISTENT_DISK_PATH"] = "./jannat_data" # Create a local folder for persistent data

    # Create dummy directories if they don't exist
    os.makedirs(os.path.dirname(ACCESS_TOKEN_STORAGE_FILE), exist_ok=True)
    os.makedirs(FYERS_WS_LOG_DIR, exist_ok=True)

    # Create a dummy access token file for testing purposes
    if not os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
        with open(ACCESS_TOKEN_STORAGE_FILE, 'w') as f:
            json.dump({"access_token": "YOUR_DUMMY_FYERS_ACCESS_TOKEN_FOR_TESTING_ONLY"}, f)
        print(f"Created dummy access token file at {ACCESS_TOKEN_STORAGE_FILE}. Replace with real token for live trading.")

    execute_strategy(mock_algo_status_dict, mock_app_logger)
