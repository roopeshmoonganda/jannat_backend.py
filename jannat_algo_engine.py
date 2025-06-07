import os
import json
import time
from datetime import datetime, timedelta
import requests
import math
from flask import Flask, jsonify, request # flask is imported for logger, not for routes here
import numpy as np # For numerical operations, especially for indicators

# --- Configuration ---
# URL of your deployed Flask backend
# IMPORTANT: Change this to your actual deployed Flask backend URL
FLASK_BACKEND_URL = "https://jannat-backend-py.onrender.com/" # <--- **UPDATE THIS URL**

# File paths for persistent storage on Render's disk
# The "PERSISTENT_DISK_PATH" environment variable will be set by Render.
# If running locally, it defaults to the current directory.\
PERSISTENT_DISK_BASE_PATH = os.environ.get("PERSISTENT_DISK_PATH", ".")
ACCESS_TOKEN_STORAGE_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_access_token.json")
CAPITAL_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_capital.json")
TRADE_LOG_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_trade_log.json")


# Trading Parameters
BASE_CAPITAL = 100000.0 # Initial capital for paper trading
SYMBOL_SPOT = "NSE:BANKNIFTY" # Base symbol for ATM option selection
OPTION_EXPIRY_DAYS_AHEAD = 7 # Number of days to look ahead for option expiry (adjust as needed for weekly/monthly)
BANKNIFTY_STRIKE_INTERVAL = 100 # BankNifty strikes are typically 100 apart
NIFTY_STRIKE_INTERVAL = 50 # Nifty strikes are typically 50 apart
TARGET_PERCENT = 0.015 # 1.5% target
STOP_LOSS_PERCENT = 0.01 # 1.0% stop loss
QUANTITY_PER_TRADE = 15 # Example: 1 lot of BankNifty (15 shares)
PRODUCT_TYPE = "MIS" # MIS, CNC, NRML
ORDER_TYPE = "MARKET" # MARKET or LIMIT
TRADE_MODE = "PAPER" # "PAPER" for simulated trades, "LIVE" for real trades
TRADE_INTERVAL_SECONDS = 60 * 5 # Check and trade every 5 minutes

# Global variables for trade management and PnL tracking
current_day = datetime.now().date() # Initial global declaration
daily_pnl = 0.0
total_trades = 0
capital_data = {} # Stores current capital, daily PnL, etc.
trade_log = [] # Stores details of executed trades

# --- Helper Functions for Persistence ---

def load_capital_data():
    global capital_data, daily_pnl, total_trades, current_day # Make current_day global here too
    try:
        if os.path.exists(CAPITAL_FILE):
            with open(CAPITAL_FILE, "r") as f:
                loaded_data = json.load(f)
                capital_data = loaded_data
                last_recorded_date = datetime.fromisoformat(loaded_data.get('last_recorded_date', datetime.min.isoformat())).date()

                if last_recorded_date != datetime.now().date():
                    # It's a new day, reset daily PnL and trade count
                    app.logger.info(f"New day detected. Resetting daily PnL and trade count for {datetime.now().date()}")
                    capital_data['daily_pnl'] = 0.0
                    capital_data['total_trades_today'] = 0
                    capital_data['last_recorded_date'] = datetime.now().isoformat()
                    daily_pnl = 0.0
                    total_trades = 0
                else:
                    daily_pnl = capital_data.get('daily_pnl', 0.0)
                    total_trades = capital_data.get('total_trades_today', 0)
                app.logger.info(f"Capital data loaded: {capital_data}")
        else:
            capital_data = {
                "current_capital": BASE_CAPITAL,
                "daily_pnl": 0.0,
                "total_trades_today": 0,
                "last_recorded_date": datetime.now().isoformat()
            }
            app.logger.info("No capital data found. Initializing with base capital.")
        current_day = datetime.now().date() # Ensure current_day is always aligned
    except (IOError, json.JSONDecodeError) as e:
        app.logger.error(f"Error loading capital data: {e}")
        capital_data = { # Re-initialize if error occurs
            "current_capital": BASE_CAPITAL,
            "daily_pnl": 0.0,
            "total_trades_today": 0,
            "last_recorded_date": datetime.now().isoformat()
        }
        daily_pnl = 0.0
        total_trades = 0
    save_capital_data() # Save immediately after loading/initializing to ensure 'last_recorded_date' is current


def save_capital_data():
    try:
        capital_data['daily_pnl'] = daily_pnl
        capital_data['total_trades_today'] = total_trades
        capital_data['last_recorded_date'] = datetime.now().isoformat() # Update timestamp on each save
        with open(CAPITAL_FILE, "w") as f:
            json.dump(capital_data, f, indent=4)
        app.logger.info("Capital data saved.")
    except IOError as e:
        app.logger.error(f"Error saving capital data: {e}")

def load_trade_log():
    global trade_log
    try:
        if os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, "r") as f:
                trade_log = json.load(f)
            app.logger.info("Trade log loaded.")
    except (IOError, json.JSONDecodeError) as e:
        app.logger.error(f"Error loading trade log: {e}")
        trade_log = [] # Reset if error
    save_trade_log() # Save immediately after loading/initializing

def save_trade_log():
    try:
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(trade_log, f, indent=4)
        app.logger.info("Trade log saved.")
    except IOError as e:
        app.logger.error(f"Error saving trade log: {e}")

# --- Market Status & Time Functions ---

def is_market_open():
    """Checks if the market is currently open based on IST."""
    current_time_ist = datetime.now().time() # Assuming the server is in IST or handles timezone
    return MARKET_OPEN_TIME <= current_time_ist <= MARKET_CLOSE_TIME

def is_within_trading_window():
    """Checks if within typical trading hours, including a buffer before close."""
    current_time_ist = datetime.now().time()
    return MARKET_OPEN_TIME <= current_time_ist <= MARKET_CUTOFF_TIME

# --- Backend Communication Functions ---

def fetch_ohlcv_from_backend(symbol, interval, days):
    """Fetches OHLCV data from the Flask backend."""
    # Backend expects 'resolution', 'range_from', 'range_to'
    payload = {
        "symbol": symbol,
        "resolution": interval, # Corrected key name from 'interval' to 'resolution' for backend
        "date_format": "1", # Fyers expects this
        "range_from": (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        "range_to": datetime.now().strftime('%Y-%m-%d'),
        "cont_flag": "1" # For continuous historical data
    }
    try:
        # --- CRITICAL FIX: Changed to requests.post and passing json=payload ---
        response = requests.post(f"{FLASK_BACKEND_URL}data/ohlcv", json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        if data.get('success'):
            return data['data'] # <--- Backend returns candles under 'data' key
        else:
            app.logger.error(f"Failed to fetch OHLCV: {data.get('message', 'Unknown error from backend')}")
            return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Flask backend for OHLCV: {e}")
        return None

def fetch_quote_from_backend(symbol):
    """Fetches real-time quote data from the Flask backend."""
    payload = {"symbols": [symbol]}
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}data/quote", json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            return data['data']
        else:
            app.logger.error(f"Failed to fetch quote: {data.get('message', 'Unknown error from backend')}")
            return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Flask backend for quote: {e}")
        return None

def execute_trade_on_backend(symbol, signal, entry_price, target, stop_loss, atm_strike, quantity, product_type, order_type, trade_mode):
    """Executes a trade via the Flask backend."""
    payload = {
        "symbol": symbol,
        "signal": signal,
        "entryPrice": entry_price,
        "target": target,
        "stopLoss": stop_loss,
        "atmStrike": atm_strike, # May not be directly used by Fyers API, but for your internal logic
        "quantity": quantity,
        "product_type": product_type,
        "order_type": order_type,
        "trade_mode": trade_mode
    }
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}trade/execute", json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            app.logger.info(f"Trade executed on backend: {data.get('message')}")
            return data['orderId']
        else:
            app.logger.error(f"Failed to execute trade on backend: {data.get('message', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Flask backend for trade execution: {e}")
        return None

# --- Option Chain & Strike Calculation ---

def get_closest_strike(spot_price, strike_interval):
    """Calculates the ATM strike price."""
    return round(spot_price / strike_interval) * strike_interval

def get_option_symbols(spot_price, expiry_days_ahead):
    """
    Generates relevant option symbols for BANKNIFTY/NIFTY.
    This is a simplified example; a real implementation would fetch expiry dates from Fyers API.
    For now, it assumes weekly expiry (Thursday) for NIFTY/BANKNIFTY.
    """
    current_date = datetime.now()
    # Find next Thursday for weekly expiry
    days_until_thursday = (3 - current_date.weekday() + 7) % 7 # 3 is Thursday (Mon=0)
    next_thursday = current_date + timedelta(days=days_until_thursday)
    # If today is Thursday and market is open past expiry time, consider next Thursday
    if current_date.weekday() == 3 and current_date.time() > MARKET_CLOSE_TIME:
        next_thursday += timedelta(days=7) # Move to next week's Thursday

    expiry_date_str = next_thursday.strftime('%y%b').upper() # YYMON (e.g., 25JUN)
    # Fyers format for weekly options sometimes includes day: e.g., BANKNIFTY25606
    # For weekly options, it's typically YYMDD (25606 for June 6, 2025)
    # This requires a more robust expiry date fetching or generation based on Fyers conventions.
    # For simplicity, we'll try to generate a common weekly format.
    # Fyers symbol format: IndexYYMthDaySTRIKECE/PE (e.g., BANKNIFTY2460623000CE)
    # Let's use a simpler placeholder and rely on backend for full symbol generation if possible,
    # or improve this for actual Fyers weekly symbols.
    
    # Example for BANKNIFTY weekly symbol format (e.g., BANKNIFTY25607 for Jun 7, 2025)
    expiry_suffix = next_thursday.strftime('%y%m%d')

    atm_strike = get_closest_strike(spot_price, BANKNIFTY_STRIKE_INTERVAL) # Assuming BANKNIFTY for now
    ce_symbol = f"NSE:BANKNIFTY{expiry_suffix}{atm_strike}CE"
    pe_symbol = f"NSE:BANKNIFTY{expiry_suffix}{atm_strike}PE"

    app.logger.info(f"Generated ATM CE: {ce_symbol}, PE: {pe_symbol}")
    return ce_symbol, pe_symbol, atm_strike

# --- Technical Analysis (Simplified) ---

def calculate_sma(candles, period):
    """Calculates Simple Moving Average."""
    if len(candles) < period:
        return None
    # Assuming candles is a list of dicts with 'close' key
    close_prices = [c['close'] for c in candles]
    return np.mean(close_prices[-period:])

def calculate_rsi(candles, period=14):
    """Calculates Relative Strength Index."""
    if len(candles) < period + 1: # Need at least period + 1 data points
        return None

    close_prices = np.array([c['close'] for c in candles])
    deltas = np.diff(close_prices)
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]

    avg_gain = np.mean(gains[:period]) if len(gains[:period]) > 0 else 0
    avg_loss = np.mean(losses[:period]) if len(losses[:period]) > 0 else 0

    if avg_loss == 0: # Avoid division by zero
        return 100 if avg_gain > 0 else 50 # If no losses, RSI is 100 (if gains) or 50 (if no gains/losses)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Core Algo Engine Logic ---

def execute_strategy():
    """Main function to run the trading strategy."""
    global daily_pnl, total_trades, capital_data, trade_log, current_day # Added current_day to global

    load_capital_data() # Load capital data at start
    load_trade_log() # Load trade log at start

    app.logger.info("Jannat Algo Trading Engine Started.")

    while True:
        current_time = datetime.now()

        # Update current_day at the start of each loop iteration to handle day changes
        if current_time.date() != current_day:
            app.logger.info(f"New day detected: {current_time.date()}. Resetting daily PnL and trade count.")
            daily_pnl = 0.0
            total_trades = 0
            current_day = current_time.date() # This assignment now refers to the global variable
            # Ensure capital data is saved with new date and reset PnL for the new day
            capital_data['daily_pnl'] = daily_pnl
            capital_data['total_trades_today'] = total_trades
            capital_data['last_recorded_date'] = current_day.isoformat()
            save_capital_data()
            # Optionally clear trade log for the new day or archive it
            # trade_log = []
            # save_trade_log()


        if is_market_open() and is_within_trading_window():
            app.logger.info(f"Market is open and within trading window. Current Capital: {capital_data['current_capital']:.2f}, Daily PnL: {daily_pnl:.2f}, Trades Today: {total_trades}")

            # 1. Fetch live spot price for BANKNIFTY
            spot_quote_data = fetch_quote_from_backend(SYMBOL_SPOT)
            if not spot_quote_data or not spot_quote_data[0].get('v'):
                app.logger.error("Failed to fetch live spot price for BANKNIFTY. Skipping trade cycle.")
                time.sleep(TRADE_INTERVAL_SECONDS)
                continue

            current_spot_price = spot_quote_data[0]['v']['lp'] # Last Traded Price
            app.logger.info(f"Current {SYMBOL_SPOT} Spot Price: {current_spot_price:.2f}")

            # 2. Fetch OHLCV data for strategy calculation
            ohlcv_data = fetch_ohlcv_from_backend(SYMBOL_SPOT, "5", 7) # 5-min candles for 7 days
            if not ohlcv_data:
                app.logger.error("Failed to fetch OHLCV data. Skipping trade cycle.")
                time.sleep(TRADE_INTERVAL_SECONDS)
                continue

            # 3. Determine trade signal
            signal, target_option_type = determine_signal(ohlcv_data, current_spot_price)

            if signal and target_option_type and filter_high_vix(20): # Assuming VIX check is always true for now
                app.logger.info(f"Signal: {signal} {target_option_type}")

                # 4. Get relevant option strike and symbol
                option_ce_symbol, option_pe_symbol, atm_strike = get_option_symbols(current_spot_price, OPTION_EXPIRY_DAYS_AHEAD)

                target_option_symbol = option_ce_symbol if target_option_type == "CE" else option_pe_symbol
                app.logger.info(f"Target Option Symbol: {target_option_symbol}")

                # 5. Fetch quote for the target option to get its current price
                option_quote_data = fetch_quote_from_backend(target_option_symbol)
                if not option_quote_data or not option_quote_data[0].get('v'):
                    app.logger.error(f"Failed to fetch live quote for {target_option_symbol}. Skipping trade cycle.")
                    time.sleep(TRADE_INTERVAL_SECONDS)
                    continue

                option_entry_price = option_quote_data[0]['v']['lp']
                app.logger.info(f"Option Entry Price ({target_option_symbol}): {option_entry_price:.2f}")

                # Calculate Target and Stop Loss for the option
                option_target_price = option_entry_price * (1 + TARGET_PERCENT)
                option_stop_loss_price = option_entry_price * (1 - STOP_LOSS_PERCENT)

                app.logger.info(f"Calculated Option Target: {option_target_price:.2f}, Stop Loss: {option_stop_loss_price:.2f}")

                # 6. Execute trade via backend
                order_id = execute_trade_on_backend(
                    symbol=target_option_symbol,
                    signal=signal, # "BUY"
                    entry_price=option_entry_price,
                    target=option_target_price,
                    stop_loss=option_stop_loss_price,
                    atm_strike=atm_strike, # Passed for logging/info, not directly used by Fyers
                    quantity=QUANTITY_PER_TRADE,
                    product_type=PRODUCT_TYPE,
                    order_type=ORDER_TYPE,
                    trade_mode=TRADE_MODE
                )

                if order_id:
                    app.logger.info(f"Trade successfully placed with Order ID: {order_id}")
                    total_trades += 1
                    # Simulate PnL for paper trading (this is very basic, a real system needs to track positions)
                    simulated_pnl = (option_target_price - option_entry_price) * QUANTITY_PER_TRADE # Simplified
                    daily_pnl += simulated_pnl
                    capital_data['current_capital'] += simulated_pnl
                    capital_data['daily_pnl'] = daily_pnl
                    capital_data['total_trades_today'] = total_trades

                    # Log the trade
                    trade_details = {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": target_option_symbol,
                        "signal": signal,
                        "entry_price": option_entry_price,
                        "target_price": option_target_price,
                        "stop_loss_price": option_stop_loss_price,
                        "quantity": QUANTITY_PER_TRADE,
                        "order_id": order_id,
                        "trade_mode": TRADE_MODE,
                        "simulated_pnl": simulated_pnl if TRADE_MODE == "PAPER" else "N/A"
                    }
                    trade_log.append(trade_details)
                    save_trade_log()
                    save_capital_data()

                    app.logger.info(f"Current Capital: {capital_data['current_capital']:.2f}, Daily PnL: {daily_pnl:.2f}, Trades Today: {total_trades}")
                else:
                    app.logger.error("Failed to place trade.")
            else:
                app.logger.info("No trade signal or filter condition not met.")
        else:
            app.logger.info("Market is closed or not within trading window.")
            # Reset daily counters at start of new day if not already
            if not is_market_open() and (current_time.time() > MARKET_CLOSE_TIME or current_time.time() < MARKET_OPEN_TIME):
                # Ensure daily PnL and trade counts reset only once per new day.
                # The `load_capital_data` function now handles this date-based check.
                pass

        time.sleep(TRADE_INTERVAL_SECONDS) # Wait before next check

# --- Bonus Intelligence Functions (Placeholders) ---
def filter_high_vix(vix_value):
    """
    Placeholder for VIX filter.
    Requires fetching VIX data from an external source.
    e.g., if vix_value > 20: return False
    """
    return True # Always allow for now

# --- Initial Setup ---
# A very basic logger setup for demonstration.
# In a real app, use Flask's app.logger or a dedicated logging module.
class SimpleLogger:
    def info(self, message):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - {message}")
    def warning(self, message):
        print(f"[WARNING] {datetime.now().strftime('%H:%M:%S')} - {message}")
    def error(self, message):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - {message}")

app = Flask(__name__) # Use Flask's logger for consistency, even if not a web route
app.logger = SimpleLogger() # Override with simple logger for console output if...

if __name__ == "__main__":
    # If running this file directly for testing, you might want a placeholder
    # For actual algo, it's called by app.py via threading.
    app.logger.info("Running Jannat Algo Engine directly for testing.")
    execute_strategy()
