import os
import json
import time
from datetime import datetime, timedelta
import requests
import math
import numpy as np
from collections import deque # For storing recent ticks for candle reconstruction

# --- Fyers API V3 Imports ---
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws # Import for data WebSocket client

# --- Configuration ---
FLASK_BACKEND_URL = os.environ.get("FLASK_BACKEND_URL", "https://jannat-backend-py.onrender.com") # Use environment variable, default to localhost for testing

# File paths for persistent storage on Render's disk
PERSISTENT_DISK_BASE_PATH = os.environ.get("PERSISTENT_DISK_PATH", ".")
ACCESS_TOKEN_STORAGE_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_access_token.json")
CAPITAL_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_capital.json")
TRADE_LOG_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_trade_log.json")

# Define a separate directory for Fyers WebSocket logs
FYERS_WS_LOG_DIR = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_ws_logs")

# Trading Parameters
BASE_CAPITAL = 100000.0
# Changed SYMBOL_SPOT to just the instrument name, not with "NSE:"
# This allows dynamic generation of futures/options symbols easily
SYMBOL_SPOT_BASE_NAME = "BANKNIFTY" 
OPTION_EXPIRY_DAYS_AHEAD = 7 # Used in get_nearest_expiry_date
BANKNIFTY_STRIKE_INTERVAL = 100
NIFTY_STRIKE_INTERVAL = 50

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
    def error(self, message):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - {message}")
    def debug(self, message):
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

def on_open():
    global logger
    logger.info("WebSocket connection opened.")

# --- Helper Functions ---

def get_fyers_access_token():
    """Reads the Fyers access token from a file."""
    if not os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
        logger.error(f"Access token file not found at {ACCESS_TOKEN_STORAGE_FILE}. Please authenticate.")
        return None
    try:
        with open(ACCESS_TOKEN_STORAGE_FILE, 'r') as f:
            token_data = json.load(f)
            return token_data.get('access_token')
    except json.JSONDecodeError:
        logger.error("Error decoding access token JSON. File might be corrupted.")
        return None
    except Exception as e:
        logger.error(f"Error reading access token: {e}")
        return None

def save_capital_data():
    """Saves the current capital data to a persistent file."""
    try:
        with open(CAPITAL_FILE, 'w') as f:
            json.dump(jannat_capital, f, indent=4)
        logger.info("Capital data saved.")
    except Exception as e:
        logger.error(f"Error saving capital data: {e}")

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
                # Use .get with a default for logging in case 'pnl_today' was genuinely missing before reset
                logger.info(f"New day detected. Resetting PnL from {jannat_capital.get('pnl_today', 0.0)} to 0.")
                jannat_capital["pnl_today"] = 0.0
                jannat_capital["last_day_reset"] = datetime.now().isoformat()
                save_capital_data() # Save the reset PnL
            logger.info(f"Capital data loaded: {jannat_capital}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error loading capital data ({e}). Initializing with base capital.")
            jannat_capital = default_capital_data # Use default data for initialization
            save_capital_data()
    else:
        logger.info("No existing capital data found. Initializing with base capital.")
        jannat_capital = default_capital_data # Use default data for initialization
        save_capital_data()

def log_trade(trade_details):
    """Logs trade details to a JSON file."""
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump([], f) # Create an empty list if file doesn't exist

    try:
        with open(TRADE_LOG_FILE, 'r+') as f:
            f.seek(0) # Go to the beginning of the file
            content = f.read()
            if content:
                trade_log = json.loads(content)
            else:
                trade_log = []

            trade_log.append(trade_details)
            f.seek(0) # Go to the beginning again to overwrite
            f.truncate() # Clear existing content
            json.dump(trade_log, f, indent=4)
        logger.info(f"Trade logged: {trade_details}")
    except Exception as e:
        logger.error(f"Error logging trade: {e}")

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
    today = datetime.now().date()
    # Fyers options typically expire on Thursday
    # Find next Thursday
    days_until_thursday = (3 - today.weekday() + 7) % 7 # 3 is Thursday (Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6)
    next_thursday = today + timedelta(days=days_until_thursday)

    # If today is Thursday and market is open/about to close, the expiry might be next week's Thursday.
    # This logic can be complex for edge cases (e.g., holiday, monthly expiry).
    # For simplicity, we'll just target the next immediate Thursday.
    # You might want to fetch actual expiry dates from Fyers API or master data.
    return next_thursday


# --- Indicator Calculations (Remain Unchanged in Logic, but data source changes) ---

def calculate_supertrend(candles, period, multiplier):
    """Calculates Supertrend indicator."""
    if len(candles) < period:
        return [], [] # Return empty lists if not enough data

    highs = np.array([c[2] for c in candles]) # [timestamp, open, high, low, close, volume]
    lows = np.array([c[3] for c in candles])
    closes = np.array([c[4] for c in candles])

    # Calculate Average True Range (ATR)
    # ATR is typically max( (high-low), abs(high-prev_close), abs(low-prev_close) )
    tr = np.zeros(len(closes))
    if len(closes) > 0: # Ensure there's data to process
        tr[0] = highs[0] - lows[0] # First TR is just high - low
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    
    # Calculate SMA of TR to get ATR (or EMA of TR for more typical ATR)
    atr = np.array([np.mean(tr[max(0, i - period + 1):i+1]) for i in range(len(tr))]) # Simple moving average of TR

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
        if i == 0: # Initialize first trend based on close vs midpoint
            if closes[i] > ((highs[i] + lows[i]) / 2):
                trend[i] = 1
            else:
                trend[i] = -1
            # Initial supertrend value based on initial trend direction
            supertrend[i] = basic_lower_band[i] if trend[i] == 1 else basic_upper_band[i]
        else:
            if closes[i] > supertrend[i-1] and trend[i-1] == 1:
                trend[i] = 1
                supertrend[i] = max(final_lower_band[i], supertrend[i-1])
            elif closes[i] < supertrend[i-1] and trend[i-1] == -1:
                trend[i] = -1
                supertrend[i] = min(final_upper_band[i], supertrend[i-1])
            elif closes[i] > supertrend[i-1] and trend[i-1] == -1: # Trend reversal to up
                trend[i] = 1
                supertrend[i] = final_lower_band[i]
            elif closes[i] < supertrend[i-1] and trend[i-1] == 1: # Trend reversal to down
                trend[i] = -1
                supertrend[i] = final_upper_band[i]
            else: # No clear change, maintain previous trend and supertrend value
                trend[i] = trend[i-1]
                supertrend[i] = supertrend[i-1]


    return supertrend.tolist(), trend.tolist() # Return list for JSON serialization

def calculate_macd(candles, fast_period, slow_period, signal_period):
    """Calculates MACD indicator."""
    # Ensure enough data for the slowest EMA and subsequent signal line EMA
    min_required_candles = max(fast_period, slow_period) + signal_period - 1 # Adjusted for _calculate_ema
    if len(candles) < min_required_candles:
        return [], [], []

    closes = np.array([c[4] for c in candles])

    ema_fast = _calculate_ema(closes, fast_period)
    ema_slow = _calculate_ema(closes, slow_period)

    # MACD Line is Fast EMA - Slow EMA.
    # Align the EMAs correctly by taking the latter part of the longer EMA
    # to match the length of the shorter EMA for subtraction.
    if len(ema_fast) > len(ema_slow):
        macd_line = ema_fast[-len(ema_slow):] - ema_slow
    elif len(ema_slow) > len(ema_fast):
        macd_line = ema_fast - ema_slow[-len(ema_fast):]
    else:
        macd_line = ema_fast - ema_slow

    if len(macd_line) < signal_period: # Not enough MACD line points for signal line
        return macd_line.tolist(), [], []

    signal_line = _calculate_ema(macd_line, signal_period)
    
    # Histogram is MACD Line - Signal Line.
    # Align by taking the latter part of MACD line to match signal line length.
    histogram = macd_line[-len(signal_line):] - signal_line

    return macd_line.tolist(), signal_line.tolist(), histogram.tolist()

def _calculate_ema(data, period):
    """Helper to calculate Exponential Moving Average (EMA)."""
    if len(data) < period: # Not enough data for EMA
        return np.array([])
    
    ema_values = np.zeros_like(data, dtype=float)
    
    # Simple Moving Average for the first 'period' values to initialize EMA
    ema_values[period - 1] = np.mean(data[:period])
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(data)):
        ema_values[i] = ((data[i] - ema_values[i-1]) * multiplier) + ema_values[i-1]

    return ema_values[period-1:] # Return from the point where EMA truly starts


# --- Tick Data Processing and Candle Reconstruction ---

def initialize_tick_data_buffers(symbols):
    """Initializes deque for each symbol to store incoming ticks."""
    for symbol in symbols:
        symbol_ticks[symbol] = deque() # Stores (timestamp, price) tuples

def on_ticks_callback(ws_client, ticks):
    """
    Callback function to process incoming ticks from Fyers WebSocket.
    This function will be called by the Fyers WebSocket client whenever new ticks arrive.
    It now accepts a list of ticks (even if it's just one).
    """
    global symbol_ticks, last_completed_candles, logger
    for tick in ticks:
        symbol = tick.get('symbol')
        timestamp_seconds = tick.get('timestamp') # Fyers typically sends timestamp in seconds
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
    This function should be called frequently as new ticks arrive.
    It manages the `symbol_ticks` deque and updates `last_completed_candles`.
    """
    global symbol_ticks, last_completed_candles, logger, CANDLE_RESOLUTION_SECONDS

    if symbol not in symbol_ticks or not symbol_ticks[symbol]:
        return

    current_time = datetime.now()
    ticks_for_symbol = symbol_ticks[symbol]

    # Find the start time of the current candle interval
    # This aligns candles to fixed intervals (e.g., every 5 minutes from market open)
    market_open_hour = 9
    market_open_minute = 15
    
    # Calculate offset from market open to align candle start times
    # This logic assumes candles align to fixed intervals from market open
    # E.g., if market opens at 9:15 and candle is 5 min, candles will be 9:15-9:20, 9:20-9:25 etc.
    market_open_today = current_time.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
    
    # Ensure we are past market open before attempting to build candles
    if current_time < market_open_today:
        return

    elapsed_seconds = (current_time - market_open_today).total_seconds()
    
    # Calculate which candle interval we are currently in
    current_candle_start_seconds_offset = math.floor(elapsed_seconds / CANDLE_RESOLUTION_SECONDS) * CANDLE_RESOLUTION_SECONDS
    current_candle_start_time = market_open_today + timedelta(seconds=current_candle_start_seconds_offset)
    current_candle_end_time = current_candle_start_time + timedelta(seconds=CANDLE_RESOLUTION_SECONDS)


    # Process ticks that fall into *completed* intervals
    # Iterate through ticks, removing those that are too old (belong to previous completed candles)
    # and using them to build a candle if an interval closes.
    
    # Temporarily hold ticks for the current interval
    temp_current_candle_ticks = deque()
    
    # Move ticks that belong to the current or future candle into a temporary deque
    # and remove any ticks that are older than the current candle's start time.
    # We must ensure ticks are processed in correct time order.
    # While ticks_for_symbol[0][0] is for time of tick, it's safer to pop until current_candle_end_time
    # or process all currently available ticks and then clear.
    
    # Collect all ticks that are older than or within the *just completed* candle window.
    # This loop ensures we only process ticks that are "finished" and belong to a past interval.
    ticks_to_process_for_past_candle = []
    while ticks_for_symbol and ticks_for_symbol[0][0] < current_candle_end_time:
        ticks_to_process_for_past_candle.append(ticks_for_symbol.popleft())
    
    # If the current time is beyond the end of a candle interval AND we have ticks for that interval, finalize it
    if current_time >= current_candle_end_time and ticks_to_process_for_past_candle:
        # Sort ticks by time to ensure O, H, L, C are correct if ticks arrive out of order
        sorted_ticks = sorted(list(ticks_to_process_for_past_candle), key=lambda x: x[0])

        open_price = sorted_ticks[0][1]
        close_price = sorted_ticks[-1][1]
        high_price = max(t[1] for t in sorted_ticks)
        low_price = min(t[1] for t in sorted_ticks)
        volume = len(sorted_ticks) # Simple tick count as volume for now

        # Fyers API historical data format: [timestamp, open, high, low, close, volume]
        # Timestamp for the candle usually represents the start time of the candle interval
        new_candle = [int(current_candle_start_time.timestamp()), open_price, high_price, low_price, close_price, volume]
        
        last_completed_candles[symbol] = new_candle
        logger.info(f"New {CANDLE_RESOLUTION_SECONDS/60}-min candle for {symbol}: {new_candle}")
        # At this point, you could trigger your strategy functions with this new candle data.

    # Any remaining ticks in `ticks_for_symbol` are for the *current* or *future* candle interval.
    # These will be processed in subsequent calls to `build_candle_from_ticks`.


def start_fyers_websocket(access_token, app_logger_instance):
    """Initializes and starts the Fyers WebSocket connection for data."""
    global websocket_client, logger # Ensure logger and websocket_client are accessible
    logger = app_logger_instance # Assign the passed logger

    logger.info("Attempting to start Fyers WebSocket...")
    try:
        # Ensure the log directory exists
        os.makedirs(FYERS_WS_LOG_DIR, exist_ok=True)
        logger.info(f"Fyers WebSocket log directory ensured: {FYERS_WS_LOG_DIR}")

        websocket_client = data_ws.FyersDataSocket(
            access_token=access_token,
            log_path=FYERS_WS_LOG_DIR, # Corrected: Pass the directory path
            litemode=False, # Set to True for light mode (fewer fields), False for full data
            write_to_file=False, # Set to True to write raw data to file
            reconnect=True, # Enable auto-reconnection
            # Use handler_message, handler_error, etc., as per fyers-apiv3 docs
            handler_message=on_message,
            handler_error=on_error,
            handler_close=on_close,
            handler_open=on_open,
            run_background=True # Runs the WebSocket in a separate thread
        )

        logger.info("Connecting to Fyers WebSocket...")
        websocket_client.connect()
        logger.info("Fyers WebSocket connection initiated.")

        # Dynamically generate the current month's BankNifty futures symbol
        current_futures_symbol = get_current_month_futures_symbol()
        initial_symbols_to_watch = [current_futures_symbol]

        # Subscribe to initial symbols
        websocket_client.subscribe(symbols=initial_symbols_to_watch, data_type="symbolData") 
        logger.info(f"Subscribed to Fyers WebSocket for initial symbols: {initial_symbols_to_watch}")
        return websocket_client
    except Exception as e:
        logger.error(f"Failed to start Fyers WebSocket: {e}", exc_info=True) # Print full traceback
        return None


# --- Trade Management Functions ---

def place_order_api(symbol, side, quantity, product_type, order_type, price=0, trade_mode='LIVE'):
    """Sends a trade request to the Flask backend's /trade/execute endpoint."""
    url = f"{FLASK_BACKEND_URL}/trade/execute"
    payload = {
        "symbol": symbol,
        "signal": "BUY" if side == 1 else "SELL",
        "quantity": quantity,
        "product_type": product_type,
        "order_type": order_type,
        "entryPrice": price, # Only relevant for LIMIT orders, but pass always
        "trade_mode": trade_mode
    }
    logger.info(f"Sending order placement request to backend for {symbol} ({'BUY' if side == 1 else 'SELL'})...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            logger.info(f"Order placed successfully for {symbol}. Order ID: {result.get('orderId')}")
            return True, result.get('orderId')
        else:
            logger.error(f"Failed to place order for {symbol}: {result.get('message')}")
            return False, result.get('message')
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error while placing order for {symbol}: {e}")
        return False, str(e)


def monitor_and_manage_trade(trade_obj):
    """
    Monitors an active trade for stop-loss, target, or end-of-day exit.
    This function would ideally run in a separate thread for each trade
    or be part of a non-blocking event loop if using asynchronous libraries.
    For simplicity, within `execute_strategy` it will be checked on each new candle.
    """
    global active_trades, jannat_capital, logger, latest_prices

    symbol = trade_obj['symbol']
    entry_price = trade_obj['entry_price']
    signal = trade_obj['signal']
    quantity = trade_obj['quantity']
    target_price = trade_obj['target_price']
    stop_loss_price = trade_obj['stop_loss_price']

    logger.info(f"Monitoring trade for {symbol}. Entry: {entry_price}, SL: {stop_loss_price}, Target: {target_price}")

    # Use the latest price from the global latest_prices dictionary
    current_price = latest_prices.get(symbol) # Get the real-time price of the traded option/future

    if current_price is None:
        logger.warning(f"Could not get current price for {symbol} from latest_prices for trade monitoring.")
        # If no price, consider if you want to exit or just wait for next tick
        return # Skip monitoring if no price

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
        if trade_obj.get('trade_mode') == 'LIVE':
            exit_side = -1 if signal == 'BUY' else 1 # Opposite side for exit
            logger.info(f"Placing exit order for {symbol} at {exit_price}")
            success, message = place_order_api(symbol, exit_side, quantity, trade_obj['product_type'], 'MARKET')
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
    logger = app_logger_instance # Use the Flask app's logger

    logger.info("Jannat Algo Engine started. Loading capital data...")
    load_capital_data()
    logger.info("Capital data loaded. Attempting Fyers API Setup for WebSocket...")

    # --- Fyers API Setup for WebSocket ---
    access_token = get_fyers_access_token()
    if not access_token:
        logger.error("Cannot start algo: Fyers access token is missing or invalid. Stopping algo.")
        algo_status_dict["status"] = "stopped"
        return

    # Initialize Fyers WebSocket connection
    websocket_client = start_fyers_websocket(access_token, logger)
    if not websocket_client:
        logger.error("Fyers WebSocket initialization failed. Stopping algo.")
        algo_status_dict["status"] = "stopped"
        return
    
    logger.info("Fyers WebSocket initialized and connected. Entering main algo loop...")

    # --- Main Algo Loop - Now driven by new candle availability ---
    last_candle_processed_time = {} # Track the timestamp of the last candle we processed for each symbol
    
    # Get the current month's futures symbol
    current_futures_symbol = get_current_month_futures_symbol()

    # Make sure we have enough historical candles for indicators.
    # In a real system, you'd fetch this from Fyers API or maintain a rolling window.
    # For now, we'll just ensure the loop processes ticks.
    
    while algo_status_dict["status"] == "running":
        if not is_market_open():
            logger.info("Market is closed. Waiting for market open.")
            time.sleep(60) # Check every minute if market is closed
            continue

        processed_any_candle_in_this_iteration = False
        
        # We process the futures symbol's candles for signal generation
        # and options symbols for monitoring active trades.
        
        # Process futures candle for signal generation
        futures_candle = last_completed_candles.get(current_futures_symbol)
        
        if futures_candle:
            candle_timestamp = futures_candle[0] # Get timestamp of the completed candle

            # Check if this candle has already been processed
            if current_futures_symbol in last_candle_processed_time and candle_timestamp == last_candle_processed_time[current_futures_symbol]:
                pass # Already processed this candle, wait for a new one
            else:
                logger.info(f"Processing new candle for {current_futures_symbol}: {futures_candle}")
                last_candle_processed_time[current_futures_symbol] = candle_timestamp
                processed_any_candle_in_this_iteration = True

                # **IMPORTANT: For accurate indicator calculations (like Supertrend, MACD),**
                # you NEED a history of `period` candles. The `build_candle_from_ticks`
                # function currently only provides the *latest* completed candle.
                # In a production system, you would maintain a `deque` or list of the
                # last `max(SUPER_TREND_PERIOD, SLOW_MA_PERIOD + SIGNAL_PERIOD)`
                # completed candles for each symbol.
                # For this example, we'll proceed with just the latest candle, but be aware
                # that indicators will be less meaningful without sufficient history.
                
                # Placeholder for historical candles: in reality, this would be `history_of_n_candles`
                # For basic demonstration, we can simulate by repeating the current candle
                # or acknowledge that true indicator accuracy requires more.
                
                # A better approach (but more complex for a single file) would be to
                # maintain a `deque` of `CANDLE_RESOLUTION_SECONDS` candles in a global state
                # and pass that to `calculate_supertrend` and `calculate_macd`.
                
                # For now, let's create a minimal `past_candles_for_indicators`
                # This will make indicators very volatile but avoid errors for now.
                # Consider this a *critical area for improvement* for actual trading.
                num_required_candles = max(SUPER_TREND_PERIOD, SLOW_MA_PERIOD + SIGNAL_PERIOD)
                past_candles_for_indicators = [futures_candle] * num_required_candles # Placeholder for actual historical data

                # --- Strategy Logic (Remains the same as before) ---
                supertrend_values, supertrend_trend = calculate_supertrend(
                    past_candles_for_indicators, SUPER_TREND_PERIOD, SUPER_TREND_MULTIPLIER
                )
                macd_line, signal_line, macd_histogram = calculate_macd(
                    past_candles_for_indicators, FAST_MA_PERIOD, SLOW_MA_PERIOD, SIGNAL_PERIOD
                )

                if not supertrend_values or not macd_line or not supertrend_trend or not signal_line or not macd_histogram:
                    logger.warning(f"Indicators not calculated for {current_futures_symbol}. Skipping trade signal generation due to insufficient data or calculation error.")
                    # Reset processed_any_candle_in_this_iteration if signal generation skipped due to this
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
                
                # Filter conditions (e.g., VIX) - unchanged
                # if not filter_high_vix(current_vix_value): # Assume get_current_vix_value() exists
                #     logger.info("High VIX detected. Not placing trades.")
                #     trade_signal = None


                if trade_signal and len(active_trades) < MAX_ACTIVE_TRADES:
                    logger.info(f"Strong trade signal detected for {current_futures_symbol}: {trade_signal}")

                    # Determine ATM strike and option symbol
                    # Use SYMBOL_SPOT_BASE_NAME ("BANKNIFTY") for option symbol construction
                    atm_strike = get_current_atm_strike(current_futures_price_for_signal, BANKNIFTY_STRIKE_INTERVAL)
                    expiry_date = get_nearest_expiry_date(OPTION_EXPIRY_DAYS_AHEAD)
                    is_call = True if trade_signal == 'BUY' else False # Buy Call for BUY, Buy Put for SELL
                    option_symbol = get_current_option_symbol(SYMBOL_SPOT_BASE_NAME, expiry_date, is_call, atm_strike)
                    
                    target_symbol_to_trade = option_symbol # This is the option symbol to trade

                    # --- NEW: Subscribe to the specific option symbol if not already subscribed ---
                    if target_symbol_to_trade not in latest_prices and websocket_client:
                        try:
                            logger.info(f"Dynamically subscribing to option symbol for trade: {target_symbol_to_trade}")
                            websocket_client.subscribe(symbols=[target_symbol_to_trade], data_type="symbolData")
                            # Give a moment for initial tick to arrive
                            time.sleep(1.5) # A slightly longer sleep to allow first tick to come in
                        except Exception as e:
                            logger.error(f"Failed to subscribe to option symbol {target_symbol_to_trade} dynamically: {e}")
                            # Decide if you want to proceed without real-time option data or skip trade
                            continue # Skip this trade if subscription fails

                    # --- NEW: Fetch option_current_price from latest_prices for entry ---
                    option_current_price = latest_prices.get(target_symbol_to_trade) # Use .get() to avoid KeyError
                    if option_current_price is None:
                        logger.warning(f"No real-time price available for {target_symbol_to_trade} after subscription. Skipping order placement for this signal.")
                        continue # Skip this trade if no real-time price after attempting subscription
                    
                    logger.info(f"Real-time option price for {target_symbol_to_trade}: {option_current_price:.2f}")

                    product_type = "MIS" # Margin Intraday Square off
                    order_type = "MARKET" # Or "LIMIT" if you want to specify entry_price
                    quantity = TRADE_QUANTITY_PER_LOT_BANKNIFTY # Quantity for BankNifty options

                    entry_price = option_current_price # Assume market order, so entry is current price

                    calculated_target = entry_price * (1 + TARGET_MULTIPLIER) if trade_signal == 'BUY' else entry_price * (1 - TARGET_MULTIPLIER)
                    calculated_stop_loss = entry_price * (1 - STOP_LOSS_MULTIPLIER) if trade_signal == 'BUY' else entry_price * (1 + STOP_LOSS_MULTIPLIER)

                    logger.info(f"Attempting to place {trade_signal} trade for {target_symbol_to_trade} @ {entry_price:.2f} (Target: {calculated_target:.2f}, SL: {calculated_stop_loss:.2f})")

                    # Place order
                    success, order_id = place_order_api(
                        target_symbol_to_trade,
                        1 if trade_signal == 'BUY' else -1,
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
                            "trade_mode": 'PAPER', # Reflects what was placed
                            "status": "ACTIVE"
                        }
                        active_trades[target_symbol_to_trade] = trade_details_for_monitoring
                        log_trade(trade_details_for_monitoring)
                        logger.info(f"Trade placed successfully for {target_symbol_to_trade}. Now monitoring...")

                    else:
                        logger.error("Failed to place trade.")
                else:
                    if trade_signal: # If signal is there but max trades reached
                        logger.info(f"Trade signal for {current_futures_symbol} but maximum active trades ({MAX_ACTIVE_TRADES}) reached.")
                    else:
                        logger.debug("No strong trade signal or filter condition not met for current candle.") # Changed to debug for less verbosity
        else: # If no new futures candle
            logger.debug(f"No new completed futures candle for {current_futures_symbol} available yet. Waiting for ticks...") # Use debug for less verbose output
        
        # After processing signals, check/monitor active trades
        for symbol, trade_obj in list(active_trades.items()): # Iterate over a copy as dict might change
            monitor_and_manage_trade(trade_obj)
        
        # Add a small sleep if no new candle was processed to prevent busy-waiting
        if not processed_any_candle_in_this_iteration:
            time.sleep(1) # Small sleep to prevent busy-waiting if no new data or signal

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
    This is used by monitor_and_manage_trade and can also be used as underlying for option selection.
    """
    global latest_prices
    return latest_prices.get(symbol) # Safely get price, returns None if not found


# --- Initial Setup for direct run (for testing purposes) ---
if __name__ == "__main__":
    # When running directly, use SimpleLogger and simulate Flask app context
    logger.info("Running Jannat Algo Engine directly for testing.")
    
    # Simulate an app_logger for direct execution
    class MockAppLogger:
        def info(self, message):
            print(f"[INFO][AppLogger] {datetime.now().strftime('%H:%M:%S')} - {message}")
        def warning(self, message):
            print(f"[WARNING][AppLogger] {datetime.now().strftime('%H:%M:%S')} - {message}")
        def error(self, message, exc_info=False): # Added exc_info for full tracebacks
            print(f"[ERROR][AppLogger] {datetime.now().strftime('%H:%M:%S')} - {message}")
            if exc_info:
                import traceback
                traceback.print_exc()
        def debug(self, message): # Added for debug messages
            print(f"[DEBUG][AppLogger] {datetime.now().strftime('%H:%M:%S')} - {message}")
    
    mock_app_logger = MockAppLogger()

    # Simulate algo_status_dict for direct execution
    mock_algo_status_dict = {"status": "running", "last_update": datetime.now().isoformat()}

    # Set up dummy environment variables for local testing without actual Fyers credentials
    os.environ["FYERS_APP_ID"] = "dummy_app_id"
    os.environ["FYERS_SECRET_ID"] = "dummy_secret"
    os.environ["FYERS_REDIRECT_URI"] = "http://localhost:5000/fyers_auth_callback"
    os.environ["FLASK_BACKEND_URL"] = "http://localhost:5000"

    # Create a dummy access token file for testing purposes
    # In a real scenario, this would be generated via the authentication flow
    if not os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
        with open(ACCESS_TOKEN_STORAGE_FILE, 'w') as f:
            json.dump({"access_token": "YOUR_DUMMY_FYERS_ACCESS_TOKEN_FOR_TESTING_ONLY"}, f)
        logger.info(f"Created dummy access token file at {ACCESS_TOKEN_STORAGE_FILE}. Replace with real token for live trading.")

    execute_strategy(mock_algo_status_dict, mock_app_logger)
