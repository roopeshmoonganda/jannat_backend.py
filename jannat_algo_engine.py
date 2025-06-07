import os
import json
import time
from datetime import datetime, timedelta
import requests
import math
from flask import Flask
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

# --- UPDATED PROFIT/LOSS TARGETS ---
TARGET_PERCENT = 0.02 # 2.0% target
STOP_LOSS_PERCENT = 0.01 # 1.0% stop loss

# --- TRAILING STOP LOSS PARAMETERS ---
# Activate trailing SL when running profit reaches this percentage
TRAILING_SL_PROFIT_ACTIVATION_PERCENT = 0.01 # 1% profit
# Lock in this percentage of profit when trailing SL activates/moves
TRAILING_SL_LOCK_PERCENT = 0.008 # 0.8% profit locked

QUANTITY_PER_TRADE = 15 # Example: 1 lot of BankNifty (15 shares)
PRODUCT_TYPE = "MIS" # MIS, CNC, NRML
ORDER_TYPE = "MARKET" # MARKET or LIMIT
TRADE_MODE = "PAPER" # "PAPER" for simulated trades, "LIVE" for real trades
TRADE_INTERVAL_SECONDS = 60 * 1 # Check and trade every 1 minute for faster scalping checks


# Global variables for trade management and PnL tracking
current_day = datetime.now().date() # Initial global declaration
daily_pnl = 0.0
total_trades = 0
capital_data = {} # Stores current capital, daily PnL, etc.
trade_log = [] # Stores details of executed trades

# Market Hours (IST)
MARKET_OPEN_TIME = datetime.strptime("09:15", "%H:%M").time()
MARKET_CLOSE_TIME = datetime.strptime("15:30", "%H:%M").time()
MARKET_CUTOFF_TIME = datetime.strptime("15:20", "%H:%M").time() # Stop trading before market close


# --- Helper Functions for Persistence ---

def load_capital_data():
    global capital_data, daily_pnl, total_trades, current_day
    try:
        if os.path.exists(CAPITAL_FILE):
            with open(CAPITAL_FILE, "r") as f:
                loaded_data = json.load(f)
                capital_data = loaded_data
                last_recorded_date = datetime.fromisoformat(loaded_data.get('last_recorded_date', datetime.min.isoformat())).date()

                if last_recorded_date != datetime.now().date():
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
        current_day = datetime.now().date()
    except (IOError, json.JSONDecodeError) as e:
        app.logger.error(f"Error loading capital data: {e}")
        capital_data = {
            "current_capital": BASE_CAPITAL,
            "daily_pnl": 0.0,
            "total_trades_today": 0,
            "last_recorded_date": datetime.now().isoformat()
        }
        daily_pnl = 0.0
        total_trades = 0
    save_capital_data()


def save_capital_data():
    try:
        capital_data['daily_pnl'] = daily_pnl
        capital_data['total_trades_today'] = total_trades
        capital_data['last_recorded_date'] = datetime.now().isoformat()
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
        trade_log = []
    save_trade_log()

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
    current_time_ist = datetime.now().time()
    return MARKET_OPEN_TIME <= current_time_ist <= MARKET_CLOSE_TIME

def is_within_trading_window():
    """Checks if within typical trading hours, including a buffer before close."""
    current_time_ist = datetime.now().time()
    return MARKET_OPEN_TIME <= current_time_ist <= MARKET_CUTOFF_TIME

# --- Backend Communication Functions ---

def fetch_ohlcv_from_backend(symbol, resolution, days):
    """Fetches OHLCV data from the Flask backend for a given resolution."""
    payload = {
        "symbol": symbol,
        "resolution": resolution, # Use resolution parameter
        "date_format": "1",
        "range_from": (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        "range_to": datetime.now().strftime('%Y-%m-%d'),
        "cont_flag": "1"
    }
    try:
        response = requests.post(f"{FLASK_BACKEND_URL}data/ohlcv", json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            return data['data']
        else:
            app.logger.error(f"Failed to fetch OHLCV for {symbol} ({resolution}): {data.get('message', 'Unknown error from backend')}")
            return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Flask backend for OHLCV ({resolution}): {e}")
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
            app.logger.error(f"Failed to fetch quote for {symbol}: {data.get('message', 'Unknown error from backend')}")
            return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to Flask backend for quote {symbol}: {e}")
        return None

def execute_trade_on_backend(symbol, signal, entry_price, target, stop_loss, atm_strike, quantity, product_type, order_type, trade_mode):
    """Executes a trade via the Flask backend."""
    payload = {
        "symbol": symbol,
        "signal": signal,
        "entryPrice": entry_price,
        "target": target,
        "stopLoss": stop_loss,
        "atmStrike": atm_strike,
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
    """
    current_date = datetime.now()
    days_until_thursday = (3 - current_date.weekday() + 7) % 7
    next_thursday = current_date + timedelta(days=days_until_thursday)
    if current_date.weekday() == 3 and current_date.time() > MARKET_CLOSE_TIME:
        next_thursday += timedelta(days=7)

    expiry_suffix = next_thursday.strftime('%y%m%d')

    atm_strike = get_closest_strike(spot_price, BANKNIFTY_STRIKE_INTERVAL)
    ce_symbol = f"NSE:BANKNIFTY{expiry_suffix}{atm_strike}CE"
    pe_symbol = f"NSE:BANKNIFTY{expiry_suffix}{atm_strike}PE"

    app.logger.info(f"Generated ATM CE: {ce_symbol}, PE: {pe_symbol}")
    return ce_symbol, pe_symbol, atm_strike

# --- Technical Analysis (Enhanced) ---

def calculate_sma(candles, period):
    """Calculates Simple Moving Average."""
    if len(candles) < period:
        return None
    close_prices = [c['close'] for c in candles]
    return np.mean(close_prices[-period:])

def calculate_rsi(candles, period=14):
    """Calculates Relative Strength Index."""
    if len(candles) < period + 1:
        return None

    close_prices = np.array([c['close'] for c in candles])
    deltas = np.diff(close_prices)
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]

    avg_gain = np.mean(gains[:period]) if len(gains[:period]) > 0 else 0
    avg_loss = np.mean(losses[:period]) if len(losses[:period]) > 0 else 0

    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_supertrend(candles, atr_period=10, multiplier=3.0):
    """
    Calculates the Supertrend indicator.
    Candles should have 'high', 'low', 'close' keys.
    Returns a tuple: (supertrend_value, direction)
    Direction: 1 for uptrend (green), -1 for downtrend (red), 0 for flat/no clear trend yet.
    """
    if len(candles) < atr_period:
        return None, 0

    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])
    closes = np.array([c['close'] for c in candles])

    tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1)))
    tr[0] = (highs[0] - lows[0]) # Handle first value where np.roll looks at invalid index

    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = ((atr[i-1] * (atr_period - 1)) + tr[i]) / atr_period

    basic_upper_band = ((highs + lows) / 2) + (multiplier * atr)
    basic_lower_band = ((highs + lows) / 2) - (multiplier * atr)

    final_upper_band = np.copy(basic_upper_band)
    final_lower_band = np.copy(basic_lower_band)

    supertrend = np.zeros_like(closes)
    direction = np.zeros_like(closes, dtype=int) # 1 for up, -1 for down

    # Initialize first direction based on close vs. bands or assume flat
    if closes[atr_period-1] > basic_upper_band[atr_period-1]:
        direction[atr_period-1] = 1
    elif closes[atr_period-1] < basic_lower_band[atr_period-1]:
        direction[atr_period-1] = -1
    else:
        direction[atr_period-1] = 0 # No clear trend initially

    for i in range(atr_period, len(candles)):
        # Calculate current bands
        current_basic_upper = basic_upper_band[i]
        current_basic_lower = basic_lower_band[i]
        
        # Adjust final bands based on previous direction
        if direction[i-1] == 1: # If previous was uptrend
            final_lower_band[i] = max(current_basic_lower, final_lower_band[i-1])
            final_upper_band[i] = current_basic_upper # Upper band can move freely
        else: # If previous was downtrend or flat
            final_upper_band[i] = min(current_basic_upper, final_upper_band[i-1])
            final_lower_band[i] = current_basic_lower # Lower band can move freely

        # Determine current direction
        if closes[i] > final_upper_band[i-1]:
            direction[i] = 1 # Price moved above upper band, new uptrend
        elif closes[i] < final_lower_band[i-1]:
            direction[i] = -1 # Price moved below lower band, new downtrend
        else:
            direction[i] = direction[i-1] # Retain previous direction if no cross

        # Set Supertrend value based on direction
        if direction[i] == 1:
            supertrend[i] = final_lower_band[i]
        else:
            supertrend[i] = final_upper_band[i]

    return supertrend[-1], direction[-1] # Return last value and its direction

def calculate_macd(candles, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates MACD, Signal Line, and MACD Histogram.
    Candles should have 'close' keys.
    Returns (macd_value, signal_line_value, histogram_value) for the latest candle.
    """
    if len(candles) < slow_period + signal_period: # Needs enough data for all EMAs
        return None, None, None

    close_prices = np.array([c['close'] for c in candles])

    # Function to calculate EMA
    def ema(prices, period):
        ema_values = np.zeros_like(prices)
        if len(prices) < period: return ema_values
        
        smoothing_factor = 2 / (period + 1)
        
        # Calculate initial EMA (simple average of first 'period' prices)
        ema_values[period-1] = np.mean(prices[:period]) 

        for i in range(period, len(prices)):
            ema_values[i] = (prices[i] * smoothing_factor) + (ema_values[i-1] * (1 - smoothing_factor))
        return ema_values

    ema_fast = ema(close_prices, fast_period)
    ema_slow = ema(close_prices, slow_period)

    # MACD Line (ensure calculations start after enough data points)
    macd_line = ema_fast[slow_period-1:] - ema_slow[slow_period-1:] # Align lengths

    # Signal Line (EMA of MACD Line)
    signal_line = ema(macd_line, signal_period) 

    # MACD Histogram
    histogram = macd_line[-len(signal_line):] - signal_line # Align lengths for histogram

    if len(macd_line) > 0 and len(signal_line) > 0 and len(histogram) > 0:
        return macd_line[-1], signal_line[-1], histogram[-1]
    return None, None, None


def determine_signal(ohlcv_5min, current_spot_price):
    """
    Determines trade signal (BUY/SELL) based on enhanced indicators.
    Also returns the target option type (CE/PE).
    """
    # For scalping, consider getting 1-min data here as well for confirmations
    # ohlcv_1min = fetch_ohlcv_from_backend(SYMBOL_SPOT, "1", 2) # Example: 1-min for 2 days

    if not ohlcv_5min or len(ohlcv_5min) < 100: # Need sufficient candles for indicators
        app.logger.warning("Insufficient 5-min OHLCV data for signal determination.")
        return None, None

    # Ensure candles are sorted oldest to newest for indicator calculation
    # Fyers API usually returns oldest first, but confirm or sort if necessary
    # ohlcv_5min.sort(key=lambda x: x['timestamp'])

    # Get the latest candle's close price and volume for checks
    latest_close = ohlcv_5min[-1]['close']
    latest_volume = ohlcv_5min[-1]['volume']

    # --- Indicator Calculations (5-min timeframe) ---
    sma_50 = calculate_sma(ohlcv_5min, 50)
    sma_20 = calculate_sma(ohlcv_5min, 20) # Faster SMA
    rsi = calculate_rsi(ohlcv_5min, 14)
    supertrend_val, supertrend_dir = calculate_supertrend(ohlcv_5min, atr_period=10, multiplier=3.0)
    macd_line, signal_line, macd_hist = calculate_macd(ohlcv_5min)

    # --- Volume Check ---
    # Consider last N candles average for volume
    avg_volume = np.mean([c['volume'] for c in ohlcv_5min[-30:]]) if len(ohlcv_5min) >= 30 else 0
    volume_confirm = (latest_volume > (avg_volume * 1.5)) and (avg_volume > 0) # Volume spike condition, avoid div by zero

    app.logger.info(f"Indicators (5-min): SMA20={sma_20:.2f}, SMA50={sma_50:.2f}, RSI={rsi:.2f}, Supertrend={supertrend_val:.2f} ({'Up' if supertrend_dir == 1 else 'Down' if supertrend_dir == -1 else 'Flat'}), MACD={macd_line:.2f}, Signal={signal_line:.2f}, Hist={macd_hist:.2f}, Volume_Confirm={volume_confirm}")

    # --- Signal Logic (Enhanced with Price Action elements, Volume, Supertrend, MACD) ---
    # This is a robust combination, but still simplified price action.

    # Buy Signal (Call Option - CE)
    # Conditions: Price above faster SMA, faster SMA above slower SMA (uptrend), RSI confirms momentum,
    # Supertrend confirms uptrend, MACD shows bullish momentum, and volume confirms.
    if (latest_close > sma_20 and # Price above short-term trend
        sma_20 > sma_50 and # Short-term trend above long-term trend
        rsi is not None and rsi > 55 and # Stronger RSI for confirmation
        supertrend_dir == 1 and # Supertrend is green (uptrend)
        macd_line is not None and signal_line is not None and macd_line > signal_line and macd_hist > 0 and # MACD Buy Crossover and positive histogram
        volume_confirm): # Volume confirms the move
        app.logger.info("Strong BUY (CE) signal detected!")
        return "BUY", "CE"

    # Sell Signal (Put Option - PE)
    # Conditions: Price below faster SMA, faster SMA below slower SMA (downtrend), RSI confirms momentum,
    # Supertrend confirms downtrend, MACD shows bearish momentum, and volume confirms.
    elif (latest_close < sma_20 and # Price below short-term trend
          sma_20 < sma_50 and # Short-term trend below long-term trend
          rsi is not None and rsi < 45 and # Weaker RSI for confirmation
          supertrend_dir == -1 and # Supertrend is red (downtrend)
          macd_line is not None and signal_line is not None and macd_line < signal_line and macd_hist < 0 and # MACD Sell Crossover and negative histogram
          volume_confirm): # Volume confirms the move
        app.logger.info("Strong SELL (PE) signal detected!")
        return "BUY", "PE" # We buy PE for a 'sell' market view (shorting the index)

    return None, None # No strong signal

def monitor_and_manage_trade(trade_details):
    """
    Monitors a simulated open trade for target, fixed SL, or trailing SL.
    This function simulates continuous monitoring and exit for PAPER_TRADE mode.
    For LIVE trading, this would involve Fyers API order modification and position monitoring.
    """
    global daily_pnl, total_trades, capital_data, trade_log

    symbol = trade_details['symbol']
    entry_price = trade_details['entry_price']
    initial_target_price = trade_details['target_price']
    initial_stop_loss_price = trade_details['stop_loss_price']
    signal = trade_details['signal'] # "BUY" for CE or PE
    quantity = trade_details['quantity']

    current_stop_loss = initial_stop_loss_price
    highest_price_achieved = entry_price # Track highest price for trailing SL (initially entry price)
    trailing_sl_active = False

    app.logger.info(f"Monitoring simulated trade for {symbol} (Entry: {entry_price:.2f}, Target: {initial_target_price:.2f}, SL: {initial_stop_loss_price:.2f})")

    monitoring_start_time = datetime.now()
    MAX_MONITORING_DURATION_SECONDS = 60 * 30 # Monitor for max 30 minutes per trade, adjust as needed
    MONITOR_INTERVAL_SECONDS = 5 # Check every 5 seconds for faster response to price action

    while (datetime.now() - monitoring_start_time).total_seconds() < MAX_MONITORING_DURATION_SECONDS:
        current_quote_data = fetch_quote_from_backend(symbol)
        if not current_quote_data or not current_quote_data[0].get('v'):
            app.logger.error(f"Failed to fetch live quote for {symbol} during monitoring. Continuing...")
            time.sleep(MONITOR_INTERVAL_SECONDS)
            continue

        current_price = current_quote_data[0]['v']['lp']
        current_pnl_absolute = (current_price - entry_price) * quantity
        current_profit_percent = (current_price - entry_price) / entry_price
        
        app.logger.info(f"Monitoring {symbol}: Current Price: {current_price:.2f}, Current PnL: {current_pnl_absolute:.2f} ({current_profit_percent:.2%}), Trailing SL: {'Active' if trailing_sl_active else 'Inactive'}, Current SL Level: {current_stop_loss:.2f}, Target: {initial_target_price:.2f}")

        # --- Check for Exit Conditions ---
        simulated_pnl_on_exit = 0.0

        # Check for Target Hit
        if current_price >= initial_target_price:
            app.logger.info(f"Target hit for {symbol}! Exiting trade.")
            simulated_pnl_on_exit = (initial_target_price - entry_price) * quantity
            exit_type = "Target Hit"
            break # Exit monitoring loop

        # Check for Fixed or Trailing Stop Loss Hit
        if current_price <= current_stop_loss:
            app.logger.info(f"Stop Loss hit for {symbol}! Exiting trade. Exit price: {current_price:.2f}")
            simulated_pnl_on_exit = (current_price - entry_price) * quantity # Actual PnL at SL hit price
            exit_type = "Stop Loss Hit"
            break # Exit monitoring loop

        # --- Trailing Stop Loss Logic ---
        # Only for long positions (BUY signal for CE or PE - assuming we always 'buy' an option)
        # Update highest price achieved
        highest_price_achieved = max(highest_price_achieved, current_price)

        # Calculate running profit percentage from entry using the highest_price_achieved
        running_profit_percent = (highest_price_achieved - entry_price) / entry_price

        if running_profit_percent >= TRAILING_SL_PROFIT_ACTIVATION_PERCENT:
            if not trailing_sl_active:
                app.logger.info(f"Trailing SL activated for {symbol}! Profit reached {running_profit_percent:.2%}.")
                trailing_sl_active = True

            # Calculate new trailing stop loss price
            # SL moves up to lock in TRAILING_SL_LOCK_PERCENT of profit from highest achieved price
            new_trailing_sl = highest_price_achieved * (1 - TRAILING_SL_LOCK_PERCENT)
            
            # Ensure trailing SL never moves down from its last position and is always above initial fixed SL
            # The current_stop_loss should only increase if new_trailing_sl is higher
            if new_trailing_sl > current_stop_loss:
                current_stop_loss = new_trailing_sl
                app.logger.info(f"Trailing SL moved to {current_stop_loss:.2f}")
            
        time.sleep(MONITOR_INTERVAL_SECONDS)
    else: # If loop completes without hitting target/SL
        app.logger.info(f"Trade for {symbol} timed out after {MAX_MONITORING_DURATION_SECONDS/60:.0f} minutes without hitting target or SL. Exiting at current price: {current_price:.2f}")
        simulated_pnl_on_exit = (current_price - entry_price) * quantity
        exit_type = "Timed Out"

    # --- Record Simulated Exit ---
    daily_pnl += simulated_pnl_on_exit
    capital_data['current_capital'] += simulated_pnl_on_exit
    capital_data['daily_pnl'] = daily_pnl
    
    trade_details.update({
        "exit_price": current_price if exit_type == "Timed Out" else (initial_target_price if exit_type == "Target Hit" else current_price), # Capture exact exit price
        "exit_time": datetime.now().isoformat(),
        "simulated_pnl": simulated_pnl_on_exit,
        "exit_type": exit_type
    })
    trade_log.append(trade_details)
    save_trade_log()
    save_capital_data()

    app.logger.info(f"Trade completed for {symbol}. PnL: {simulated_pnl_on_exit:.2f}. New Capital: {capital_data['current_capital']:.2f}, Daily PnL: {daily_pnl:.2f}")
    
    return True # Trade was managed to completion


# --- Core Algo Engine Logic ---

def execute_strategy():
    """Main function to run the trading strategy."""
    global daily_pnl, total_trades, capital_data, trade_log, current_day

    load_capital_data()
    load_trade_log()

    app.logger.info("Jannat Algo Trading Engine Started.")

    while True:
        current_time = datetime.now()

        if current_time.date() != current_day:
            app.logger.info(f"New day detected: {current_time.date()}. Resetting daily PnL and trade count.")
            daily_pnl = 0.0
            total_trades = 0
            current_day = current_time.date()
            capital_data['daily_pnl'] = daily_pnl
            capital_data['total_trades_today'] = total_trades
            capital_data['last_recorded_date'] = current_day.isoformat()
            save_capital_data()


        if is_market_open() and is_within_trading_window():
            app.logger.info(f"Market is open and within trading window. Current Capital: {capital_data['current_capital']:.2f}, Daily PnL: {daily_pnl:.2f}, Trades Today: {total_trades}")

            # 1. Fetch live spot price for BANKNIFTY
            spot_quote_data = fetch_quote_from_backend(SYMBOL_SPOT)
            if not spot_quote_data or not spot_quote_data[0].get('v'):
                app.logger.error("Failed to fetch live spot price for BANKNIFTY. Skipping trade cycle.")
                time.sleep(TRADE_INTERVAL_SECONDS)
                continue

            current_spot_price = spot_quote_data[0]['v']['lp']
            app.logger.info(f"Current {SYMBOL_SPOT} Spot Price: {current_spot_price:.2f}")

            # 2. Fetch OHLCV data for strategy calculation (using 5-min for indicators)
            # To add 1-min analysis:
            # ohlcv_1min_data = fetch_ohlcv_from_backend(SYMBOL_SPOT, "1", 2) # Fetch 1-min candles for 2 days
            # Then pass both to determine_signal: determine_signal(ohlcv_5min_data, ohlcv_1min_data, current_spot_price)
            ohlcv_5min_data = fetch_ohlcv_from_backend(SYMBOL_SPOT, "5", 7) # 5-min candles for 7 days

            if not ohlcv_5min_data: # or not ohlcv_1min_data: if using 1-min
                app.logger.error("Failed to fetch OHLCV data. Skipping trade cycle.")
                time.sleep(TRADE_INTERVAL_SECONDS)
                continue

            # 3. Determine trade signal
            signal, target_option_type = determine_signal(ohlcv_5min_data, current_spot_price)

            # Placeholder for VIX filter - ensure it's implemented for real use
            if signal and target_option_type and filter_high_vix(20):
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

                app.logger.info(f"Calculated Option Target: {option_target_price:.2f}, Initial Stop Loss: {option_stop_loss_price:.2f}")

                # 6. Execute trade via backend (This is simulated placement in paper mode)
                order_id = execute_trade_on_backend(
                    symbol=target_option_symbol,
                    signal=signal,
                    entry_price=option_entry_price,
                    target=option_target_price,
                    stop_loss=option_stop_loss_price,
                    atm_strike=atm_strike,
                    quantity=QUANTITY_PER_TRADE,
                    product_type=PRODUCT_TYPE,
                    order_type=ORDER_TYPE,
                    trade_mode=TRADE_MODE
                )

                if order_id:
                    app.logger.info(f"Trade successfully placed with Order ID: {order_id}")
                    total_trades += 1 # Increment trade count upon placement

                    trade_details_for_monitoring = {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": target_option_symbol,
                        "signal": signal,
                        "entry_price": option_entry_price,
                        "target_price": option_target_price,
                        "stop_loss_price": option_stop_loss_price,
                        "quantity": QUANTITY_PER_TRADE,
                        "order_id": order_id,
                        "trade_mode": TRADE_MODE
                    }
                    
                    # --- Monitor and Manage the Trade until exit (Target/SL/Trailing SL hit) ---
                    # This function will run in its own internal loop until the trade is exited.
                    monitor_and_manage_trade(trade_details_for_monitoring)
                    
                else:
                    app.logger.error("Failed to place trade.")
            else:
                app.logger.info("No strong trade signal or filter condition not met.")
        else:
            app.logger.info("Market is closed or not within trading window.")
            pass # Load capital data handles new day reset

        time.sleep(TRADE_INTERVAL_SECONDS) # Wait before next check/trade attempt

# --- Bonus Intelligence Functions (Placeholders) ---
def filter_high_vix(vix_value):
    """
    Placeholder for VIX filter.
    Requires fetching VIX data from an external source and defining a threshold.
    e.g., if vix_value > 20: return False (do not trade in high volatility)
    """
    return True # Always allow for now. Implement real VIX check later.

# --- Initial Setup ---
class SimpleLogger:
    def info(self, message):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - {message}")
    def warning(self, message):
        print(f"[WARNING] {datetime.now().strftime('%H:%M:%S')} - {message}")
    def error(self, message):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - {message}")

app = Flask(__name__)
app.logger = SimpleLogger()

if __name__ == "__main__":
    app.logger.info("Running Jannat Algo Engine directly for testing.")
    execute_strategy()
