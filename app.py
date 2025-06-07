import os
import json
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from flask_cors import CORS
import threading
from jannat_algo_engine import execute_strategy # Corrected: 'execute_strategy' imported

# --- Fyers API V3 Imports ---
# Make sure you have the latest fyers-apiv3 installed: pip install fyers-apiv3
from fyers_apiv3 import fyersModel # Correct import for the main Fyers client model (includes SessionModel)
from fyers_apiv3.FyersWebsocket import data_ws # Correct import for the data WebSocket client


app = Flask(__name__)
CORS(app) # Enable CORS for deployment compatibility


# --- Configuration (Load from Environment Variables for Production) ---
# For development, you can hardcode, but for deployment, USE ENVIRONMENT VARIABLES
# Example:
# export FYERS_APP_ID="YOUR_APP_ID"
# export FYERS_SECRET_ID="YOUR_SECRET_ID"
# export FYERS_REDIRECT_URI="YOUR_REDIRECT_URI_FOR_AUTH_FLOW"
# export FYERS_CLIENT_ID="YOUR_APP_ID-100" # Typically App ID + "-100"


FYERS_APP_ID = os.environ.get("FYERS_APP_ID", "YOUR_APP_ID_FROM_FYERS")
FYERS_SECRET_ID = os.environ.get("FYERS_SECRET_ID", "YOUR_SECRET_ID_FROM_FYERS")
FYERS_REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI", "https://jannat-backend-py.onrender.com/fyers_auth_callback") # Ensure this matches your Render URL
FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID", FYERS_APP_ID + "-100") # Default to App ID + "-100"

# Global variable to store the Fyers API client instance
fyers_api_client = None
# Global variable to store the access token (for simplicity; better to use a database in production)
ACCESS_TOKEN = None

# --- File paths for persistent storage on Render's disk ---
# The "PERSISTENT_DISK_PATH" environment variable will be set by Render.
# If running locally, it defaults to the current directory.
PERSISTENT_DISK_BASE_PATH = os.environ.get("PERSISTENT_DISK_PATH", ".")
ACCESS_TOKEN_STORAGE_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_access_token.json")

# --- Helper Functions ---
def save_access_token(token_data):
    try:
        with open(ACCESS_TOKEN_STORAGE_FILE, "w") as f:
            json.dump(token_data, f)
        app.logger.info(f"Access token saved to {ACCESS_TOKEN_STORAGE_FILE}")
    except IOError as e:
        app.logger.error(f"Error saving access token: {e}")

def load_access_token():
    global ACCESS_TOKEN
    try:
        if os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
            with open(ACCESS_TOKEN_STORAGE_FILE, "r") as f:
                token_data = json.load(f)
                ACCESS_TOKEN = token_data.get('access_token')
                app.logger.info(f"Access token loaded from {ACCESS_TOKEN_STORAGE_FILE}")
                return token_data
        app.logger.warning("No access token file found.")
        return None
    except json.JSONDecodeError as e:
        app.logger.error(f"Error decoding access token JSON: {e}")
        return None
    except IOError as e:
        app.logger.error(f"Error loading access token: {e}")
        return None

def initialize_fyers_client(access_token):
    global fyers_api_client
    if access_token:
        try:
            fyers_api_client = fyersModel.FyersModel(
                client_id=FYERS_CLIENT_ID,
                is_async=False, # Set to True for async operations
                token=access_token,
                log_path=os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_logs") # Path for Fyers API logs
            )
            app.logger.info("Fyers API client initialized with access token.")
            return True
        except Exception as e:
            app.logger.error(f"Error initializing Fyers API client: {e}")
            return False
    return False

# --- Flask Routes ---

@app.route("/")
def home():
    return "Jannat Algo Backend is running!"

@app.route("/generate_auth_url", methods=["GET"])
def generate_auth_url():
    session = fyersModel.SessionModel(
        client_id=FYERS_CLIENT_ID,
        secret_key=FYERS_SECRET_ID,
        redirect_uri=FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )
    auth_url = session.generate_authcode()
    app.logger.info(f"Generated Fyers Auth URL: {auth_url}")
    return jsonify({"auth_url": auth_url})

@app.route("/fyers_auth_callback", methods=["GET"])
def fyers_auth_callback():
    global ACCESS_TOKEN
    auth_code = request.args.get("auth_code")
    if not auth_code:
        app.logger.error("Authorization code not found in callback.")
        return jsonify({"success": False, "message": "Authorization code not found."}), 400

    session = fyersModel.SessionModel(
        client_id=FYERS_CLIENT_ID,
        secret_key=FYERS_SECRET_ID,
        redirect_uri=FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )

    try:
        session.set_token(auth_code)
        response = session.generate_token()

        if response and response.get('s') == 'ok':
            ACCESS_TOKEN = response.get("access_token")
            if ACCESS_TOKEN:
                save_access_token({"access_token": ACCESS_TOKEN, "timestamp": datetime.now().isoformat()})
                initialize_fyers_client(ACCESS_TOKEN)
                app.logger.info("Fyers access token obtained and saved successfully.")
                return jsonify({"success": True, "message": "Fyers authentication successful!", "access_token_present": True}), 200
            else:
                app.logger.error("Access token not found in Fyers response.")
                return jsonify({"success": False, "message": "Access token not found in Fyers response."}), 500
        else:
            error_message = response.get('message', 'Unknown error during token generation.')
            app.logger.error(f"Fyers token generation failed: {error_message}")
            return jsonify({"success": False, "message": f"Fyers token generation failed: {error_message}"}), 500
    except Exception as e:
        app.logger.error(f"Error during Fyers authentication callback: {e}")
        return jsonify({"success": False, "message": f"Authentication failed: {e}"}), 500

@app.route("/validate_credentials", methods=["GET"])
def validate_credentials():
    # Attempt to load token and initialize client if not already
    global fyers_api_client, ACCESS_TOKEN
    if not fyers_api_client and ACCESS_TOKEN:
        initialize_fyers_client(ACCESS_TOKEN)
    elif not fyers_api_client:
        token_data = load_access_token()
        if token_data:
            initialize_fyers_client(token_data.get('access_token'))

    if fyers_api_client:
        try:
            # Make a simple API call to validate credentials, e.g., get user profile
            profile_info = fyers_api_client.get_profile()
            if profile_info and profile_info.get('s') == 'ok':
                user_name = profile_info.get('data', {}).get('name', 'Unknown User')
                app.logger.info(f"Fyers credentials valid for: {user_name}")
                return jsonify({"status": "success", "message": f"Credentials valid for {user_name}"}), 200
            else:
                error_message = profile_info.get('message', 'Failed to get profile information.')
                app.logger.error(f"Fyers credentials invalid: {error_message}")
                return jsonify({"status": "error", "message": f"Invalid Fyers credentials: {error_message}"}), 401
        except Exception as e:
            app.logger.error(f"Error validating Fyers credentials: {e}")
            return jsonify({"status": "error", "message": f"Error validating credentials: {e}"}), 500
    else:
        app.logger.warning("Fyers client not initialized. No access token found or initialization failed.")
        return jsonify({"status": "error", "message": "Fyers client not initialized. Please authenticate."}), 401


@app.route("/save_and_validate_credentials", methods=["POST"])
def save_and_validate_credentials_route():
    data = request.json
    client_id = data.get("client_id")
    secret_key = data.get("secret_key")
    redirect_uri = data.get("redirect_uri") # This should be the one from Fyers App settings

    if not all([client_id, secret_key, redirect_uri]):
        return jsonify({"success": False, "message": "Missing Fyers credentials."}), 400

    # In a real application, you would save these securely (e.g., to a database)
    # For now, we'll just attempt to generate the auth URL to validate.
    # Note: For Render, environment variables are the primary way to persist these.
    # This route is more for local testing or initial setup.

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )

    try:
        auth_url = session.generate_authcode()
        app.logger.info(f"Successfully generated auth URL for provided credentials.")
        return jsonify({
            "success": True,
            "message": "Credentials seem valid, proceed to Fyers authentication.",
            "auth_url": auth_url
        }), 200
    except Exception as e:
        app.logger.error(f"Failed to generate auth URL with provided credentials: {e}")
        return jsonify({"success": False, "message": f"Invalid Fyers credentials provided: {e}"}), 401


@app.route("/data/ohlcv", methods=["POST"])
def get_ohlcv_data():
    global fyers_api_client
    if not fyers_api_client:
        return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401

    data = request.json
    symbol = data.get('symbol')
    resolution = data.get('resolution', '15') # e.g., "1", "5", "15", "60", "240", "D"
    range_from = data.get('range_from') # YYYY-MM-DD
    range_to = data.get('range_to')     # YYYY-MM-DD

    if not all([symbol, resolution, range_from, range_to]):
        return jsonify({"success": False, "message": "Missing data parameters."}), 400

    try:
        history_data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",  # 1 for YYYY-MM-DD
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1" # 1 for historical data
        }
        response = fyers_api_client.history(data=history_data)

        if response.get('s') == 'ok':
            # Format data if needed, Fyers returns [timestamp, open, high, low, close, volume]
            candles = response.get('candles', [])
            # Convert timestamp to human-readable format if desired
            formatted_candles = [
                {
                    "time": datetime.fromtimestamp(c[0]).strftime('%Y-%m-%d %H:%M:%S'),
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "volume": c[5]
                } for c in candles
            ]
            return jsonify({"success": True, "data": formatted_candles}), 200
        else:
            return jsonify({"success": False, "message": response.get('message', 'Failed to fetch OHLCV data.')}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"Error fetching OHLCV data: {e}"}), 500

@app.route("/data/quote", methods=["POST"])
def get_quote_data():
    global fyers_api_client
    if not fyers_api_client:
        return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401

    data = request.json
    symbols = data.get('symbols') # List of symbols, e.g., ["NSE:BANKNIFTY", "NSE:NIFTY"]

    if not symbols or not isinstance(symbols, list):
        return jsonify({"success": False, "message": "Missing or invalid symbols parameter (must be a list).", "symbols": symbols}), 400

    try:
        # Fyers API expects symbols as a comma-separated string
        symbols_str = ','.join(symbols)
        response = fyers_api_client.quotes(data={"symbols": symbols_str})

        if response.get('s') == 'ok':
            return jsonify({"success": True, "data": response.get('d', [])}), 200
        else:
            return jsonify({"success": False, "message": response.get('message', 'Failed to fetch quote data.')}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"Error fetching quote data: {e}"}), 500


@app.route("/trade/execute", methods=["POST"])
def execute_trade():
    global fyers_api_client
    if not fyers_api_client:
        return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401

    data = request.json
    symbol = data.get('symbol')
    signal = data.get('signal')
    entry_price = data.get('entryPrice')
    target = data.get('target')
    stop_loss = data.get('stopLoss')
    atm_strike = data.get('atmStrike')
    quantity = data.get('quantity')
    product_type = data.get('productType') # e.g., "MIS", "CNC", "NRML"
    order_type = data.get('orderType') # "LIMIT", "MARKET"
    trade_mode = data.get('tradeMode') # "PAPER" or "LIVE"

    if not all([symbol, signal, entry_price, target, stop_loss, quantity, product_type, order_type, trade_mode]):
        return jsonify({"success": False, "message": "Missing trade parameters."}), 400

    app.logger.info(f"Received trade request: {data}")

    # --- Paper Trading Logic ---
    if trade_mode == "PAPER":
        simulated_order_id = f"SIMULATED_ORDER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        app.logger.info(f"Simulating paper trade for {symbol} ({signal}) - Order ID: {simulated_order_id}")
        return jsonify({"success": True, "message": "Paper trade simulated successfully.", "orderId": simulated_order_id}), 200

    # --- Live Trading Logic ---
    elif trade_mode == "LIVE":
        try:
            # Determine side: 1 for BUY, -1 for SELL
            side = 1 if signal == "BUY" else -1

            # V3 Order Data Structure - check Fyers API V3 documentation for exact keys and values
            # This is a generic structure; for options, you'll need the correct instrument symbol.
            order_data = {
                "symbol": symbol,
                "qty": int(quantity),
                "type": 2,  # 1 for LIMIT, 2 for MARKET, 3 for SL-M, 4 for SL-L
                "side": side,
                "productType": product_type,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False,
                "stopLoss": float(stop_loss) if stop_loss else 0, # SL for SL/SL-M/SL-L orders
                "takeProfit": float(target) if target else 0 # Target for take profit (if supported as part of single order)
            }


            if order_type == "LIMIT":
                order_data["type"] = 1
                order_data["limitPrice"] = float(entry_price)
            # Add conditions for SL-M, SL-L if your application requires them


            response = fyers_api_client.place_order(data=order_data)


            if response.get('s') == 'ok':
                order_id = response['id']
                app.logger.info(f"Live trade placed for {symbol} ({signal}) - Fyers Order ID: {order_id}")
                return jsonify({"success": True, "message": "Live trade placed successfully.", "orderId": order_id}), 200
            else:
                app.logger.error(f"Fyers order placement error: {response.get('message', 'Unknown error')}")
                return jsonify({"success": False, "message": response.get('message', 'Failed to place live order.')}), 500
        except Exception as e:
            app.logger.error(f"Error placing live trade: {e}")
            return jsonify({"success": False, "message": f"Internal server error placing live trade: {e}"}), 500
    else:
        return jsonify({"success": False, "message": "Invalid trade mode specified."}), 400

# CORRECTED INDENTATION: This route must be at the same level as other @app.route decorators
@app.route('/start_algo', methods=['GET', 'POST'])
def start_algo():
    try:
        # CORRECTED: Call execute_strategy from jannat_algo_engine
        threading.Thread(target=execute_strategy).start()
        return jsonify({"status": "Algo Started"}), 200
    except Exception as e:
        return jsonify({"status": "Failed to start algo", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
