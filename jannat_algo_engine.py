import os
import json
import time
from datetime import datetime, timedelta
import requests
import math
from flask import Flask, jsonify, request
import numpy as np # For numerical operations, especially for indicators

# --- Configuration ---
# URL of your deployed Flask backend
# IMPORTANT: Change this to your actual deployed Flask backend URL
FLASK_BACKEND_URL = "https://jannat-backend-py.onrender.com" # <--- **UPDATE THIS URL**

# File paths for persistent storage on Render's disk
# The "PERSISTENT_DISK_PATH" environment variable will be set by Render.
# If running locally, it defaults to the current directory.
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

# Strategy Flags & Controls
AUTO_MODE_ON = True # Set to False to temporarily stop the algo
LAST_TRADE_DIRECTION = None # To prevent repeat trades in same direction (BUY/SELL)
RSI_RESET_THRESHOLD = {"BUY": 50, "SELL": 50} # RSI must cross this to reset for a new trade in the same direction
LAST_TRADE_TIMESTAMP = None # To prevent rapid consecutive trades
TRADE_INTERVAL_SECONDS = 30 # How often to check for new candles/signals (e.g., 30-60 seconds)
# Market Hours (IST - adjust if your server is in a different timezone)
MARKET_OPEN_TIME = datetime.strptime("09:16", "%H:%M").time()
MARKET_CLOSE_TIME = datetime.strptime("15:25", "%H:%M").time()
INITIAL_BUFFER_MINUTES = 10 # Don't trade in the first X minutes of market open

# Risk Management
DAILY_TARGET_PERCENT = 1.3 # 1.3% of total capital per day
TRAILING_SL_PERCENT = 0.7 # 0.7% trailing SL per trade
MAX_DAILY_LOSS_COUNT = 2 # Cut-off if > 2 losses in a day

# Global state for the algo
current_capital = BASE_CAPITAL
daily_pnl = 0.0
trades_today = []
daily_loss_count = 0
candles_1min = [] # Stores recent 1-min candles (e.g., last 200)
candles_5min = [] # Stores recent 5-min candles (e.g., last 200)

# --- Helper Functions for File Operations ---
def load_capital_data():
    """Loads capital and PnL data from JSON file."""
    global current_capital, daily_pnl
    if os.path.exists(CAPITAL_FILE):
        try:
            with open(CAPITAL_FILE, 'r') as f:
                data = json.load(f)
                # Check for new day and reset daily PnL and loss count
                if data.get('last_run_date') == datetime.now().strftime('%Y-%m-%d'):
                    current_capital = data.get('current_capital', BASE_CAPITAL)
                    daily_pnl = data.get('daily_pnl', 0.0)
                    # daily_loss_count is not explicitly loaded here, assumes fresh start daily or managed by trade logic
                    app.logger.info(f"Loaded capital: {current_capital}, Daily PnL: {daily_pnl} for today.")
                else:
                    # New day, reset daily PnL and loss count but carry over capital
                    current_capital = data.get('current_capital', BASE_CAPITAL)
                    daily_pnl = 0.0
                    global daily_loss_count
                    daily_loss_count = 0 # Reset for new day
                    app.logger.info("New day. Resetting daily PnL and loss count.")
        except Exception as e:
            app.logger.error(f"Error loading capital data: {e}. Starting with base capital.")
            current_capital = BASE_CAPITAL
            daily_pnl = 0.0
            daily_loss_count = 0
    else:
        app.logger.info("Capital file not found. Starting with base capital.")
        current_capital = BASE_CAPITAL
        daily_pnl = 0.0
        daily_loss_count = 0
    save_capital_data() # Save to create file if not exists or update date

def save_capital_data():
    """Saves current capital and daily PnL to JSON file."""
    try:
        os.makedirs(os.path.dirname(CAPITAL_FILE), exist_ok=True)
        with open(CAPITAL_FILE, 'w') as f:
            json.dump({
                'current_capital': current_capital,
                'daily_pnl': daily_pnl,
                'last_run_date': datetime.now().strftime('%Y-%m-%d')
            }, f, indent=4)
        app.logger.info(f"Capital saved: {current_capital}, Daily PnL: {daily_pnl}")
    except IOError as e:
        app.logger.error(f"Could not write capital data to disk: {e}")

def load_trade_log():
    """Loads trade log from JSON file."""
    global trades_today
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE, 'r') as f:
                data = json.load(f)
                # Only load today's trades
                today_date = datetime.now().strftime('%Y-%m-%d')
                trades_today = [trade for trade in data.get('trades', []) if trade.get('entry_time', '').startswith(today_date)]
                app.logger.info(f"Loaded {len(trades_today)} trades for today.")
        except Exception as e:
            app.logger.error(f"Error loading trade log: {e}. Starting with empty log.")
            trades_today = []
    else:
        app.logger.info("Trade log file not found. Starting with empty log.")
        trades_today = []

def save_trade_log():
    """Saves all trades to JSON file."""
    # Read existing trades, merge with today's, then save
    all_trades = []
    try:
        if os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, 'r') as f:
                existing_data = json.load(f)
                # Filter out old trades that are not from today, then add today's trades
                all_trades = [trade for trade in existing_data.get('trades', []) if not trade.get('entry_time', '').startswith(datetime.now().strftime('%Y-%m-%d'))]
        
        all_trades.extend(trades_today)
        os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump({'trades': all_trades}, f, indent=4)
        app.logger.info(f"Trade log saved with {len(all_trades)} entries.")
    except IOError as e:
        app.logger.error(f"Could not write trade log to disk: {e}")
    except Exception as e:
        app.logger.error(f"Error saving trade log: {e}")

# --- Communication with Flask Backend ---
def fetch_ohlcv_from_backend(symbol, interval, days):
    """Fetches OHLCV data from the Flask backend."""
    try:
        response = requests.get(f"{FLASK_BACKEND_URL}/data/ohlcv", params={
            "symbol": symbol,
            "interval": interval,
            "days": days
        })
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            return data['candles']
        else:
            app.logger.error(f"Failed to fetch OHLCV: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Flask backend for OHLCV: {e}")
        return None

def fetch_quote_from_backend(symbol):
    """Fetches live quote data from the Flask backend."""
    try:
        response = requests.get(f"{FLASK_BACKEND_URL}/data/quote", params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            # The 'v' field contains quote details, including 'lp' (last price)
            quote_data = data['quote']
            return quote_data.get('lp') # Last Traded Price
        else:
            app.logger.error(f"Failed to fetch quote: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Flask backend for quote: {e}")
        return None

def place_trade_via_backend(symbol, signal, entry_price, target, stop_loss, quantity, product_type="MIS", order_type="MARKET", trade_mode="PAPER"):
    """Places a trade via the Flask backend."""
    payload = {
        "symbol": symbol,
        "signal": signal,
        "entryPrice": entry_price,
        "target": target,
        "stopLoss": stop_loss,
        "quantity": quantity,
        "productType": product_type,
        "orderType": order_type,
        "tradeMode": trade_mode,
        "atmStrike": 0 # This field is passed but not used by backend in generic order_data
    }
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}/trade/execute", json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            app.logger.info(f"Trade placed successfully: {data.get('message')}. Order ID: {data.get('orderId')}")
            return data.get('orderId')
        else:
            app.logger.error(f"Failed to place trade: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Flask backend for trade execution: {e}")
        return None

# --- Data Management ---
def update_local_candles(symbol, interval_str, history_days, target_list):
    """
    Fetches candles from backend and updates the local list.
    Ensures local list contains only candles from the last `history_days`.
    """
    new_candles = fetch_ohlcv_from_backend(symbol, interval_str, history_days)
    if new_candles:
        # Fyers candles are [timestamp, open, high, low, close, volume]
        # Convert timestamp to datetime objects for easier comparison
        formatted_candles = []
        for c in new_candles:
            try:
                # Fyers API timestamps are usually Unix timestamps (in seconds).
                # The provided code snippet suggests it might be in milliseconds, check actual API response.
                # Assuming seconds here for now, adjust if needed.
                dt_object = datetime.fromtimestamp(c[0])
                formatted_candles.append({
                    "time": dt_object,
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "volume": c[5]
                })
            except (TypeError, ValueError) as e:
                app.logger.error(f"Error processing candle timestamp {c[0]}: {e}")
                continue # Skip malformed candle

        # Keep only unique candles (important for real-time updates)
        # Use a set of (time, close) to identify unique candles
        existing_candle_keys = {(c["time"], c["close"]) for c in target_list}
        unique_new_candles = [c for c in formatted_candles if (c["time"], c["close"]) not in existing_candle_keys]

        # Append new unique candles and sort by time
        target_list.extend(unique_new_candles)
        target_list.sort(key=lambda x: x["time"])

        # Remove candles older than the specified history_days
        earliest_time = datetime.now() - timedelta(days=history_days)
        target_list[:] = [c for c in target_list if c["time"] >= earliest_time]

        app.logger.info(f"Updated {interval_str} candles. Total: {len(target_list)}")
        return True
    return False

def get_latest_candle(interval_str):
    """Returns the latest closed candle for a given interval."""
    target_list = candles_1min if interval_str == "1min" else candles_5min
    if target_list and len(target_list) >= 2: # Need at least 2 for a 'closed' candle
        # The latest candle in the sorted list might still be forming.
        # We want the *last fully closed* candle.
        # This implies checking the second-to-last candle.
        # For simplicity, if fetching historical, assume last one is closed.
        # If live, you'd need to confirm candle closure via timestamp alignment.
        # For now, let's just take the last candle in the list.
        return target_list[-1]
    return None

# --- Indicator Calculations ---
def calculate_rsi(closes, period=14):
    """Calculates Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return None

    deltas = np.diff(closes)
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate initial RS
    if avg_loss == 0:
        rs = 100 if avg_gain > 0 else 0 # Avoid division by zero
    else:
        rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Calculate subsequent RS and RSI using smoothing
    for i in range(period, len(closes) -1):
        gain = deltas[i] if deltas[i] > 0 else 0
        loss = -deltas[i] if deltas[i] < 0 else 0

        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

        if avg_loss == 0:
            rs = 100 if avg_gain > 0 else 0
        else:
            rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_supertrend(highs, lows, closes, period=10, multiplier=3):
    """Calculates Supertrend indicator."""
    if len(closes) < period:
        return None

    # Calculate Average True Range (ATR)
    atr_values = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr_values.append(tr)
    
    if len(atr_values) < period:
        return None
    
    atr = np.mean(atr_values[-period:]) # Simple Moving Average of True Range

    # Calculate Basic Upper Band and Basic Lower Band
    basic_upper_band = (highs + lows) / 2 + multiplier * atr
    basic_lower_band = (highs + lows) / 2 - multiplier * atr

    final_upper_band = [0.0] * len(closes)
    final_lower_band = [0.0] * len(closes)
    supertrend = [0.0] * len(closes)
    
    for i in range(period, len(closes)):
        if i == period:
            final_upper_band[i] = basic_upper_band[i]
            final_lower_band[i] = basic_lower_band[i]
        else:
            final_upper_band[i] = basic_upper_band[i] if basic_upper_band[i] < final_upper_band[i-1] or closes[i-1] > final_upper_band[i-1] else final_upper_band[i-1]
            final_lower_band[i] = basic_lower_band[i] if basic_lower_band[i] > final_lower_band[i-1] or closes[i-1] < final_lower_band[i-1] else final_lower_band[i-1]
        
        if supertrend[i-1] == final_upper_band[i-1] and closes[i] <= final_upper_band[i]:
            supertrend[i] = final_upper_band[i]
        elif supertrend[i-1] == final_upper_band[i-1] and closes[i] > final_upper_band[i]:
            supertrend[i] = final_lower_band[i]
        elif supertrend[i-1] == final_lower_band[i-1] and closes[i] >= final_lower_band[i]:
            supertrend[i] = final_lower_band[i]
        elif supertrend[i-1] == final_lower_band[i-1] and closes[i] < final_lower_band[i]:
            supertrend[i] = final_upper_band[i]
        else: # Initial case
             supertrend[i] = final_lower_band[i] if closes[i] > final_lower_band[i] else final_upper_band[i]

    # Determine trend based on latest candle and supertrend
    if closes[-1] > supertrend[-1]:
        return "BUY" # Uptrend
    elif closes[-1] < supertrend[-1]:
        return "SELL" # Downtrend
    return "NEUTRAL" # Sideways or unconfirmed


def check_rsi_filter(rsi_value, min_rsi=40, max_rsi=60):
    """Checks if 1-min RSI is within acceptable range for a trade."""
    if rsi_value is None:
        return False
    return rsi_value < min_rsi or rsi_value > max_rsi

def check_supertrend_confirmation(supertrend_direction, trade_direction):
    """Checks if 5-min Supertrend matches entry direction."""
    return supertrend_direction == trade_direction

def check_volume_spike(volumes, multiplier=1.5, lookback_periods=20):
    """Checks if current volume is significantly higher than average."""
    if len(volumes) < lookback_periods:
        return False
    
    recent_volumes = np.array(volumes[-lookback_periods-1:-1]) # Exclude current candle
    average_volume = np.mean(recent_volumes)
    
    current_volume = volumes[-1]
    return current_volume > (average_volume * multiplier)

def check_breakout_candle(latest_candle_1min, prev_candle_1min, trade_direction):
    """Checks for a breakout candle (higher high for CE, lower low for PE)."""
    if not latest_candle_1min or not prev_candle_1min:
        return False
    
    if trade_direction == "BUY": # Looking for CE (Call Option) entry
        # Current candle's close is higher than its open AND
        # Current candle's high is higher than previous candle's high
        return latest_candle_1min["close"] > latest_candle_1min["open"] and \
               latest_candle_1min["high"] > prev_candle_1min["high"]
    elif trade_direction == "SELL": # Looking for PE (Put Option) entry
        # Current candle's close is lower than its open AND
        # Current candle's low is lower than previous candle's low
        return latest_candle_1min["close"] < latest_candle_1min["open"] and \
               latest_candle_1min["low"] < prev_candle_1min["low"]
    return False

def check_support_resistance(latest_candle_1min, closes, highs, lows, trade_direction, lookback=50):
    """
    Simplified S/R check: Checks if current price is breaking through a recent
    high/low, rather than bouncing off it. Avoids trading near obvious S/R.
    This is a very basic placeholder and can be greatly expanded.
    """
    if len(closes) < lookback:
        return True # Not enough data to check S/R, proceed with caution

    recent_high = np.max(highs[-lookback:])
    recent_low = np.min(lows[-lookback:])
    
    # If buying, we want to be breaking above recent resistance, not stuck below it.
    # If selling, we want to be breaking below recent support, not stuck above it.
    
    # Simple check: If current close is very close to a recent high (for buy) or low (for sell)
    # and not clearly breaking out, it might be a false signal.
    # For now, let's just check if price is not excessively far from recent high/low
    # in a way that suggests a whipsaw.
    
    # A more robust S/R check would involve identifying explicit levels.
    # For this implementation, we'll make a very simple check:
    # If we are looking to BUY, we want to be *above* a recent significant low.
    # If we are looking to SELL, we want to be *below* a recent significant high.
    
    # This is a very rough interpretation. A better approach would be to detect if a breakout
    # is occurring *through* a predefined or dynamically identified S/R level.
    # Given the complexity, this will be a simple "allow" for now, indicating it's not blocking.
    # Proper S/R detection and breakout confirmation is a complex topic beyond this scope.
    return True # Placeholder for now, assume no strong S/R barrier blocking trade.


# --- ATM Option Selection ---
def get_option_expiry():
    """Calculates the next upcoming option expiry date (weekly/monthly)."""
    today = datetime.now()
    # Find next Friday (for weekly options)
    days_until_friday = (4 - today.weekday() + 7) % 7
    if days_until_friday == 0: # If today is Friday, check if it's past market close
        if today.time() > MARKET_CLOSE_TIME:
            days_until_friday = 7 # Go to next Friday
    next_friday = today + timedelta(days=days_until_friday)

    # For monthly expiry (last Thursday of the month) - more complex logic
    # For simplicity, let's just target the next Friday for now.
    # You'll need to adapt this for monthly if you trade them.

    # Fyers format for expiry: YYMMMFF (e.g., 24JUN for June 2024, 246 for June 2024)
    # For weekly, it's typically YYMDD (e.g., 24606 for June 6th 2024 if it's an expiry date)
    # Or YYMMDD for NIFTY/BANKNIFTY weekly options (e.g., 24606 for June 6, 2024)
    # Fyers symbol format often uses YYMNDD (Year, Month Letter for Jan-Sep/O-N-D, Day)
    # e.g., for June 7, 2025: BANKNIFTY25607 (25 for year, 6 for month, 07 for day)
    # This is highly dependent on Fyers' exact symbol convention for weekly options.
    # Let's assume YYMDD format for now, which is common for short-term options.
    
    # Fyers API V3 symbol conventions for weekly options usually look like:
    # NSE:BANKNIFTY24JUN134800CE (24 for year, JUN for month, 13 for day) or
    # NSE:BANKNIFTY24613 (24 for year, 6 for June, 13 for day)
    # The common format for Bank Nifty weekly options is like 'BANKNIFTY24606' (YYMMDD)
    # For example, June 6, 2024, if it's an expiry date.
    
    expiry_date_str = next_friday.strftime("%y%m%d") # Format as YYMMDD
    return expiry_date_str

def get_atm_strike(spot_price, strike_interval):
    """Calculates the nearest ATM strike price."""
    return round(spot_price / strike_interval) * strike_interval

def get_option_symbol(base_symbol, expiry_date_str, strike_price, option_type):
    """Constructs the Fyers option symbol (e.g., NSE:BANKNIFTY2460745000CE)."""
    # Fyers option symbol format: NSE:<BASESYMBOL><YYMMDD><STRIKE><OPTIONTYPE>
    # Example: NSE:BANKNIFTY2460745000CE (for June 7th 2024, 45000 CE)
    # Make sure expiry_date_str is in YYMMDD format.
    return f"NSE:{base_symbol}{expiry_date_str}{int(strike_price)}{option_type}"


def select_atm_option(base_symbol, trade_direction, spot_price):
    """
    Selects the ATM option symbol based on spot price and trade direction.
    Simplification: Directly picks ATM. Does not pick based on max volume/OI
    across multiple strikes as that requires a full options chain API or pre-fetched data.
    """
    if "BANKNIFTY" in base_symbol:
        strike_interval = BANKNIFTY_STRIKE_INTERVAL
    elif "NIFTY" in base_symbol:
        strike_interval = NIFTY_STRIKE_INTERVAL
    else:
        app.logger.error(f"Unsupported base symbol: {base_symbol}")
        return None

    atm_strike = get_atm_strike(spot_price, strike_interval)
    expiry_date_str = get_option_expiry() # Get next weekly expiry (YYMMDD format)

    option_type = "CE" if trade_direction == "BUY" else "PE"

    option_symbol = get_option_symbol(base_symbol.split(':')[-1], expiry_date_str, atm_strike, option_type)
    app.logger.info(f"Selected ATM option symbol: {option_symbol}")
    return option_symbol


# --- Risk Management ---
def check_daily_target_achieved():
    """Checks if daily profit target is achieved."""
    return daily_pnl >= (BASE_CAPITAL * DAILY_TARGET_PERCENT / 100)

def reset_rsi_for_new_trade(current_rsi, last_trade_direction):
    """Checks if RSI has reset to allow a new trade in the same direction."""
    if last_trade_direction == "BUY" and current_rsi <= RSI_RESET_THRESHOLD["BUY"]:
        return True
    if last_trade_direction == "SELL" and current_rsi >= RSI_RESET_THRESHOLD["SELL"]:
        return True
    return False

def check_max_daily_losses():
    """Checks if maximum daily losses have been exceeded."""
    return daily_loss_count >= MAX_DAILY_LOSS_COUNT

# --- Trade Execution (Paper Mode Simulation) ---
class PaperTrade:
    """Represents a simulated paper trade."""
    def __init__(self, symbol, signal, entry_price, quantity, target, stop_loss):
        self.symbol = symbol
        self.signal = signal
        self.entry_time = datetime.now()
        self.entry_price = entry_price
        self.quantity = quantity
        self.target = target
        self.initial_stop_loss = stop_loss
        self.trailing_sl = stop_loss # Current trailing stop loss
        self.is_open = True
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.reason = "Open"
        
        # Calculate initial profit/loss points for easy tracking
        self.profit_point = self.entry_price + (self.entry_price * (target - self.entry_price) / self.entry_price) if signal == "BUY" else \
                            self.entry_price - (self.entry_price * (self.entry_price - target) / self.entry_price)
        self.loss_point = self.entry_price - (self.entry_price * (self.entry_price - stop_loss) / self.entry_price) if signal == "BUY" else \
                          self.entry_price + (self.entry_price * (stop_loss - self.entry_price) / self.entry_price)

    def update_trailing_sl(self, current_price):
        """Updates trailing stop loss based on price movement."""
        if not self.is_open:
            return

        trailing_amount = self.entry_price * TRAILING_SL_PERCENT / 100

        if self.signal == "BUY":
            # Trailing SL moves up as price moves up
            new_sl = current_price - trailing_amount
            self.trailing_sl = max(self.trailing_sl, new_sl)
        elif self.signal == "SELL":
            # Trailing SL moves down as price moves down
            new_sl = current_price + trailing_amount
            self.trailing_sl = min(self.trailing_sl, new_sl)
        
        # Ensure trailing SL is not worse than initial stop loss
        if self.signal == "BUY":
            self.trailing_sl = max(self.trailing_sl, self.initial_stop_loss)
        elif self.signal == "SELL":
            self.trailing_sl = min(self.trailing_sl, self.initial_stop_loss)

    def check_exit(self, current_price):
        """Checks if target or stop loss is hit."""
        global daily_pnl, current_capital, daily_loss_count, LAST_TRADE_DIRECTION

        if not self.is_open:
            return

        if self.signal == "BUY":
            if current_price >= self.target:
                self.close_trade(current_price, "Target Hit")
            elif current_price <= self.trailing_sl:
                self.close_trade(current_price, "SL Hit")
        elif self.signal == "SELL":
            if current_price <= self.target: # For SELL, target is lower price
                self.close_trade(current_price, "Target Hit")
            elif current_price >= self.trailing_sl: # For SELL, SL is higher price
                self.close_trade(current_price, "SL Hit")

    def close_trade(self, exit_price, reason):
        """Closes the simulated trade and updates PnL."""
        global daily_pnl, current_capital, daily_loss_count, LAST_TRADE_DIRECTION
        
        self.exit_time = datetime.now()
        self.exit_price = exit_price
        self.is_open = False
        self.reason = reason

        if self.signal == "BUY":
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        elif self.signal == "SELL":
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
        
        daily_pnl += self.pnl
        current_capital += self.pnl # Compounding capital
        
        if self.pnl < 0:
            daily_loss_count += 1
            app.logger.warning(f"Trade Loss: {self.pnl:.2f}. Daily losses: {daily_loss_count}")
        else:
            daily_loss_count = 0 # Reset consecutive losses on a profitable trade

        LAST_TRADE_DIRECTION = None # Reset last trade direction after trade closes

        trade_info = {
            "symbol": self.symbol,
            "signal": self.signal,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "reason": self.reason
        }
        trades_today.append(trade_info)
        save_trade_log()
        save_capital_data()

        app.logger.info(f"Trade Closed: Symbol: {self.symbol}, PnL: {self.pnl:.2f}, Reason: {reason}")
        app.logger.info(f"Current Capital: {current_capital:.2f}, Daily PnL: {daily_pnl:.2f}")

current_open_trade = None # Only one trade open at a time for simplicity

def execute_strategy():
    """
    Main strategy execution logic.
    This function will be called in a loop.
    """
    global current_open_trade, LAST_TRADE_DIRECTION, daily_loss_count, current_capital, daily_pnl, LAST_TRADE_TIMESTAMP
    global AUTO_MODE_ON

    if not AUTO_MODE_ON:
        app.logger.info("Auto mode is OFF. Skipping strategy execution.")
        return

    if check_max_daily_losses():
        app.logger.warning(f"Max daily losses ({MAX_DAILY_LOSS_COUNT}) exceeded. Stopping algo for the day.")
        AUTO_MODE_ON = False # Disable auto mode for the rest of the day
        return

    if check_daily_target_achieved():
        app.logger.info(f"Daily profit target ({DAILY_TARGET_PERCENT}%) achieved. Stopping algo for the day.")
        AUTO_MODE_ON = False # Disable auto mode for the rest of the day
        return

    # Fetch latest data
    success_1min = update_local_candles(SYMBOL_SPOT, "1min", 3, candles_1min)
    success_5min = update_local_candles(SYMBOL_SPOT, "5min", 7, candles_5min)

    if not success_1min or not success_5min:
        app.logger.error("Failed to update candles. Skipping strategy check.")
        return

    latest_spot_price = fetch_quote_from_backend(SYMBOL_SPOT)
    if latest_spot_price is None:
        app.logger.error("Failed to fetch live spot price. Skipping strategy check.")
        return

    # Get closed candles for indicator calculations
    # Ensure we get candles that are fully formed at the current time
    # This logic assumes the backend provides completed candles.
    # For robust real-time, you'd process tick data and form candles yourself.
    
    # We need enough data points for RSI and Supertrend calculation.
    # RSI typically needs at least 15-20 periods. Supertrend needs at least `period`.
    
    if len(candles_1min) < 20 or len(candles_5min) < 20: # Arbitrary minimums for indicator calculation
        app.logger.info("Not enough historical data for indicators. Waiting for more candles.")
        return

    # Extract closes, highs, lows for numpy arrays
    closes_1min = np.array([c["close"] for c in candles_1min])
    highs_1min = np.array([c["high"] for c in candles_1min])
    lows_1min = np.array([c["low"] for c in candles_1min])
    volumes_1min = np.array([c["volume"] for c in candles_1min])
    
    closes_5min = np.array([c["close"] for c in candles_5min])
    highs_5min = np.array([c["high"] for c in candles_5min])
    lows_5min = np.array([c["low"] for c in candles_5min])

    # Get the last two 1-min candles for breakout check
    latest_1min_candle = candles_1min[-1]
    prev_1min_candle = candles_1min[-2] if len(candles_1min) >= 2 else None

    # Check for a currently open trade
    if current_open_trade and current_open_trade.is_open:
        current_open_trade.update_trailing_sl(latest_spot_price)
        current_open_trade.check_exit(latest_spot_price)
        app.logger.info(f"Open trade: {current_open_trade.symbol}, Entry: {current_open_trade.entry_price}, Current Price: {latest_spot_price}, Trailing SL: {current_open_trade.trailing_sl:.2f}, Target: {current_open_trade.target:.2f}")
        return # Don't look for new trades if one is already open

    # --- Strategy Logic ---
    signal = None # "BUY" for CE, "SELL" for PE
    
    # Calculate indicators
    rsi_1min = calculate_rsi(closes_1min)
    supertrend_5min_direction = calculate_supertrend(highs_5min, lows_5min, closes_5min)
    volume_spike = check_volume_spike(volumes_1min)

    app.logger.info(f"Indicators: RSI (1min): {rsi_1min:.2f}, Supertrend (5min): {supertrend_5min_direction}, Volume Spike: {volume_spike}")

    # Determine potential trade direction based on RSI
    if rsi_1min is not None:
        if rsi_1min > 60: # Overbought, potential SELL (PE) signal
            signal = "SELL"
        elif rsi_1min < 40: # Oversold, potential BUY (CE) signal
            signal = "BUY"
    
    if signal:
        # Check if previous trade was in same direction and RSI hasn't reset
        if LAST_TRADE_DIRECTION == signal and not reset_rsi_for_new_trade(rsi_1min, signal):
            app.logger.info(f"RSI not reset for {signal} direction. Skipping trade.")
            signal = None # Invalidate signal
        
        # Filter 2: Supertrend Confirmation
        if signal and not check_supertrend_confirmation(supertrend_5min_direction, signal):
            app.logger.info(f"Supertrend ({supertrend_5min_direction}) does not confirm {signal} signal. Skipping.")
            signal = None

        # Filter 3: Volume Spike
        if signal and not volume_spike:
            app.logger.info(f"No volume spike for {signal} signal. Skipping.")
            signal = None

        # Filter 4: Breakout Candle
        if signal and (prev_1min_candle is None or not check_breakout_candle(latest_1min_candle, prev_1min_candle, signal)):
            app.logger.info(f"No breakout candle for {signal} signal. Skipping.")
            signal = None

        # Filter 5: Support/Resistance Check (simplified placeholder)
        if signal and not check_support_resistance(latest_1min_candle, closes_1min, highs_1min, lows_1min, signal):
            app.logger.info(f"S/R check failed for {signal} signal. Skipping.")
            signal = None

    # If all filters pass, prepare and execute trade
    if signal:
        app.logger.info(f"All filters passed! Generating {signal} trade signal.")

        # ATM Option Selection
        option_symbol = select_atm_option(SYMBOL_SPOT, signal, latest_spot_price)
        if not option_symbol:
            app.logger.error("Failed to select ATM option symbol. Cannot place trade.")
            return
        
        # For paper trading, quantity is based on capital
        # Calculate quantity such that total trade value is a certain percentage of capital or a fixed amount
        # For simplicity, let's target a fixed number of lots or a small portion of capital
        # Assuming 1 lot is roughly 15-25 shares for BankNifty options
        # Let's assume a rough option premium is around 100-300 for ATM.
        # Max 1 lot for now, or calculate based on 5% of capital (adjust as needed)
        
        # Fetching quantity
        # Assuming a target trade value of 5% of current_capital
        target_trade_value = current_capital * 0.05 # 5% of current capital
        
        # Attempt to get a rough price for the option to calculate quantity
        # This is important as option premiums vary widely
        option_ltp = fetch_quote_from_backend(option_symbol)
        if not option_ltp or option_ltp <= 0:
            app.logger.error(f"Could not get LTP for {option_symbol}. Cannot calculate quantity.")
            return

        # BankNifty lot size is 15, Nifty lot size is 50.
        lot_size = 15 if "BANKNIFTY" in SYMBOL_SPOT else 50
        
        # Calculate maximum possible quantity in lots
        # Example: if capital is 100k, and target trade value 5k, and option price is 200,
        # then max shares = 5000 / 200 = 25 shares.
        # Quantity must be a multiple of lot_size.
        max_shares = math.floor(target_trade_value / option_ltp)
        quantity = (max_shares // lot_size) * lot_size
        
        if quantity == 0:
            app.logger.warning(f"Calculated quantity is 0 for {option_symbol}. Not placing trade.")
            return

        # Determine entry price, target, stop loss for the option
        # For simplicity, let's use the current LTP of the option for entry and calculate SL/Target from there.
        # This is crucial: the entry_price, target, stop_loss are for the OPTION, not the spot.
        option_entry_price = option_ltp
        
        # For paper trading, we estimate target/stop loss based on the option's value
        # Adjust these percentages as needed based on your option trading strategy
        option_target_percentage = 0.05 # 5% profit on option price
        option_sl_percentage = 0.02 # 2% loss on option price

        if signal == "BUY":
            option_target = option_entry_price * (1 + option_target_percentage)
            option_stop_loss = option_entry_price * (1 - option_sl_percentage)
        else: # SELL
            option_target = option_entry_price * (1 - option_target_percentage)
            option_stop_loss = option_entry_price * (1 + option_sl_percentage)
        
        # Create a paper trade object
        current_open_trade = PaperTrade(
            symbol=option_symbol,
            signal=signal,
            entry_price=option_entry_price,
            quantity=quantity,
            target=option_target,
            stop_loss=option_stop_loss
        )
        app.logger.info(f"Simulated {signal} trade initiated for {option_symbol} at {option_entry_price} (Qty: {quantity})")
        app.logger.info(f"Target: {option_target:.2f}, Initial SL: {option_stop_loss:.2f}")

        # Record the last trade timestamp to prevent rapid re-entries
        LAST_TRADE_TIMESTAMP = datetime.now()
        LAST_TRADE_DIRECTION = signal # Store the direction of this trade

        # This is where you would call place_trade_via_backend for LIVE trading
        # order_id = place_trade_via_backend(option_symbol, signal, option_entry_price, option_target, option_stop_loss, quantity, trade_mode="LIVE")
        # if order_id:
        #     app.logger.info(f"LIVE Trade placed with ID: {order_id}")
        # else:
        #     app.logger.error("Failed to place LIVE trade.")
        #     # Handle live trade failure (e.g., revert current_open_trade state)

    else:
        app.logger.info("No trade signal generated after filters.")


# --- Main Loop and Scheduling ---
def is_market_open():
    """Checks if the current time is within market hours (IST)."""
    now = datetime.now().time()
    return MARKET_OPEN_TIME <= now <= MARKET_CLOSE_TIME

def is_within_trading_window():
    """Checks if trading is allowed based on initial buffer."""
    now = datetime.now()
    open_time = datetime.combine(now.date(), MARKET_OPEN_TIME)
    trading_start_time = open_time + timedelta(minutes=INITIAL_BUFFER_MINUTES)
    return now >= trading_start_time

def main_loop():
    """Main loop for the trading engine."""
    global AUTO_MODE_ON, daily_pnl, current_capital, daily_loss_count

    app.logger.info("Jannat Algo Trading Engine Started.")
    load_capital_data() # Load capital at start
    load_trade_log() # Load previous trades for the day

    while True:
        current_time = datetime.now()
        app.logger.info(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        app.logger.info(f"Auto Mode: {'ON' if AUTO_MODE_ON else 'OFF'}")
        app.logger.info(f"Daily PnL: {daily_pnl:.2f}, Current Capital: {current_capital:.2f}")
        app.logger.info(f"Trades Today: {len(trades_today)}, Daily Losses: {daily_loss_count}")


        if is_market_open() and is_within_trading_window():
            if AUTO_MODE_ON:
                execute_strategy()
            else:
                app.logger.info("Auto mode is OFF, but market is open. Waiting for it to be turned ON.")
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
app.logger = SimpleLogger() # Override with simple logger for console output if not running flask app itself


if __name__ == "__main__":
    main_loop()
