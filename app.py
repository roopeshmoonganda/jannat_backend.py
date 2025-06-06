import os
import json
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from flask_cors import CORS

# Install the Fyers API Python SDK: pip install fyers-api==1.0.9
from fyers_api import accessToken, fyersModel


app = Flask(__name__)
CORS(app)


# --- Configuration (Load from Environment Variables for Production) ---
# For development, you can hardcode, but for deployment, USE ENVIRONMENT VARIABLES
# Example:
# export FYERS_APP_ID="YOUR_APP_ID"
# export FYERS_SECRET_ID="YOUR_SECRET_ID"
# export FYERS_REDIRECT_URI="YOUR_REDIRECT_URI_FOR_AUTH_FLOW"
# export FYERS_CLIENT_ID="YOUR_APP_ID-100" # Typically App ID + "-100"
# export ACCESS_TOKEN_FILE="fyers_access_token.json" # File to store dynamically generated access token


FYERS_APP_ID = os.environ.get("FYERS_APP_ID", "YOUR_APP_ID_FROM_FYERS")
FYERS_SECRET_ID = os.environ.get("FYERS_SECRET_ID", "YOUR_SECRET_ID_FROM_FYERS")
FYERS_REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI", "https://google.com")
FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID", f"{FYERS_APP_ID}-100" if FYERS_APP_ID != "YOUR_APP_ID_FROM_FYERS" else "")


# This file will store the access token generated after authentication
ACCESS_TOKEN_STORAGE_FILE = "fyers_access_token.json"


# Global Fyers client instance
fyers = None


def load_access_token():
    """Loads the access token from a file."""
    if os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
        with open(ACCESS_TOKEN_STORAGE_FILE, 'r') as f:
            data = json.load(f)
            # Check for expiry if Fyers provides it
            # For simplicity, we assume the token is valid if present.
            # In production, you'd handle token refresh or re-authentication.
            return data.get('access_token')
    return None


def save_access_token(token):
    """Saves the access token to a file."""
    with open(ACCESS_TOKEN_STORAGE_FILE, 'w') as f:
        json.dump({'access_token': token, 'timestamp': datetime.now().isoformat()}, f)


@app.before_request
def check_fyers_client_initialized():
    """Ensures Fyers client is initialized before processing requests."""
    global fyers
    if fyers is None:
        access_token = load_access_token()
        if access_token:
            try:
                # Use fyersModel for 1.0.9 compatibility
                fyers = fyersModel.FyersModel(client_id=FYERS_CLIENT_ID, token=access_token, log_path=os.getcwd())
                app.logger.info("Fyers client initialized from stored token.")
            except Exception as e:
                app.logger.error(f"Error initializing Fyers client with stored token: {e}")
                fyers = None # Reset if initialization fails
        else:
            app.logger.warning("Fyers access token not found or expired. Please authenticate via /generate_auth_url.")


# --- Fyers Authentication Flow ---
@app.route("/generate_auth_url")
def generate_auth_url():
    """Generates the Fyers authentication URL for manual login."""
    try:
        # Use accessToken.SessionModel for 1.0.9 compatibility
        session = accessToken.SessionModel()
        session.set_param({
            "app_id": FYERS_CLIENT_ID,
            "secret_key": FYERS_SECRET_ID,
            "redirect_uri": FYERS_REDIRECT_URI,
            "response_type": "code",
            "grant_type": "authorization_code"})
        
        response = session.generate_authcode()
        return jsonify({"success": True, "auth_url": response}), 200
    except Exception as e:
        app.logger.error(f"Error generating Fyers auth URL: {e}")
        return jsonify({"success": False, "message": f"Failed to generate auth URL: {e}"}), 500


@app.route("/fyers_auth_callback")
def fyers_auth_callback():
    """Callback endpoint for Fyers OAuth flow."""
    auth_code = request.args.get('auth_code')
    if not auth_code:
        error = request.args.get('error')
        return jsonify({"success": False, "message": f"Fyers authentication failed: {error}"}), 400

    try:
        # Use accessToken.SessionModel for 1.0.9 compatibility
        session = accessToken.SessionModel()
        session.set_param({
            "app_id": FYERS_CLIENT_ID,
            "secret_key": FYERS_SECRET_ID,
            "redirect_uri": FYERS_REDIRECT_URI,
            "response_type": "code",
            "grant_type": "authorization_code"
        })
        session.set_token(auth_code)
        response = session.generate_token()
        access_token = response["access_token"]
        save_access_token(access_token)

        global fyers
        # Use fyersModel for 1.0.9 compatibility
        fyers = fyersModel.FyersModel(client_id=FYERS_CLIENT_ID, token=access_token, log_path=os.getcwd())
        app.logger.info("Fyers client initialized successfully with new token.")

        return jsonify({"success": True, "message": "Fyers token generated and saved successfully!", "access_token": access_token}), 200
    except Exception as e:
        app.logger.error(f"Error processing Fyers auth callback: {e}")
        return jsonify({"success": False, "message": f"Failed to generate token: {e}"}), 500


# Endpoint for the React Native app to validate credentials (App ID, Secret ID, Client ID)
# and ensure backend has a valid access token.
@app.route("/validate_credentials", methods=["POST"])
def validate_credentials():
    data = request.json
    app_id = data.get('app_id')
    secret_id = data.get('secret_id')
    client_id = data.get('client_id')
    access_key = data.get('access_key')

    if not all([app_id, secret_id, client_id, access_key]):
        return jsonify({"success": False, "message": "Missing App ID, Secret ID, Client ID, or Access Key"}), 400

    # Basic validation: Check if provided IDs match backend config (environment variables)
    if app_id != FYERS_APP_ID or secret_id != FYERS_SECRET_ID or client_id != FYERS_CLIENT_ID:
        return jsonify({"success": False, "message": "App ID, Secret ID, or Client ID mismatch with backend configuration."}), 401

    # Attempt to initialize Fyers client with the provided access key for a quick check
    # In a real scenario, you'd want to verify token validity with Fyers if possible.
    try:
        # Use fyersModel for 1.0.9 compatibility
        test_fyers = fyersModel.FyersModel(client_id=client_id, token=access_key, log_path=os.getcwd())
        # Try a simple API call to verify token (e.g., get profile)
        profile_data = test_fyers.get_profile()
        if profile_data and profile_data.get('s') == 'ok':
            # If valid, save this access token for future use by the backend
            save_access_token(access_key)
            global fyers
            fyers = test_fyers
            return jsonify({"success": True, "message": "Credentials validated and Fyers client initialized."}), 200
        else:
            app.logger.error(f"Access Key validation failed (Fyers response: {profile_data})")
            return jsonify({"success": False, "message": "Access Key invalid or expired."}), 401
    except Exception as e:
        app.logger.error(f"Error validating access key: {e}")
        return jsonify({"success": False, "message": f"Error validating credentials: {e}"}), 500


@app.route("/save_and_validate_credentials", methods=["POST"])
def save_and_validate_credentials():
    # This endpoint is primarily for when the user manually enters App ID/Secret ID.
    # It doesn't handle access token generation itself; it just validates against backend's config.
    # For a full flow, this would trigger an authentication URL generation for the user to open.
    data = request.json
    app_id = data.get('app_id')
    secret_id = data.get('secret_id')

    if not all([app_id, secret_id]):
        return jsonify({"success": False, "message": "Missing App ID or Secret ID"}), 400

    # For now, just compare with backend's configured FYERS_APP_ID and FYERS_SECRET_ID
    if app_id == FYERS_APP_ID and secret_id == FYERS_SECRET_ID:
        return jsonify({"success": True, "message": "App ID and Secret ID matched backend configuration. Now generate/load access token."}), 200
    else:
        return jsonify({"success": False, "message": "App ID or Secret ID mismatch with backend configuration. Please check your credentials or backend setup."}), 401


# --- Data Endpoints ---
@app.route("/data/ohlcv")
def get_ohlcv():
    global fyers
    if not fyers:
        return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401

    symbol = request.args.get('symbol', "NSE:BANKNIFTY")
    interval = request.args.get('interval', "1")
    days = request.args.get('days', "3")

    try:
        data = {
            "symbol": symbol,
            "resolution": interval,
            "date_format": "1",
            "range_from": (datetime.now() - timedelta(days=int(days))).strftime('%Y-%m-%d'),
            "range_to": datetime.now().strftime('%Y-%m-%d'),
            "cont_flag": "1"
        }
        response = fyers.history(data=data)
        if response.get('s') == 'ok' and response.get('candles'):
            # Fyers API returns [timestamp, open, high, low, close, volume]
            return jsonify({"success": True, "candles": response['candles']}), 200
        else:
            app.logger.error(f"Fyers OHLCV API error: {response.get('message', 'Unknown error')}")
            return jsonify({"success": False, "message": response.get('message', 'Failed to fetch OHLCV data from Fyers.')}), 500
    except Exception as e:
        app.logger.error(f"Error fetching OHLCV data: {e}")
        return jsonify({"success": False, "message": f"Internal server error fetching OHLCV: {e}"}), 500


@app.route("/data/quote")
def get_quote():
    global fyers
    if not fyers:
        return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401

    symbol = request.args.get('symbol', "NSE:BANKNIFTY")

    try:
        data = {"symbols": symbol}
        response = fyers.quotes(data=data)
        if response.get('s') == 'ok' and response.get('d') and response['d'][0].get('v'):
            return jsonify({"success": True, "quote": response['d'][0]['v']}), 200
        else:
            app.logger.error(f"Fyers Quote API error: {response.get('message', 'Unknown error')}")
            return jsonify({"success": False, "message": response.get('message', 'Failed to fetch quote from Fyers.')}), 500
    except Exception as e:
        app.logger.error(f"Error fetching quote: {e}")
        return jsonify({"success": False, "message": f"Internal server error fetching quote: {e}"}), 500


# --- Trade Execution Endpoint ---
@app.route("/trade/execute", methods=["POST"])
def execute_trade():
    global fyers
    if not fyers:
        return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401

    data = request.json
    symbol = data.get('symbol')
    signal = data.get('signal')
    entry_price = data.get('entryPrice')
    target = data.get('target')
    stop_loss = data.get('stopLoss')
    atm_strike = data.get('atmStrike')
    quantity = data.get('quantity')
    product_type = data.get('productType')
    order_type = data.get('orderType')
    trade_mode = data.get('tradeMode')


    if not all([symbol, signal, entry_price, target, stop_loss, quantity, product_type, order_type, trade_mode]):
        return jsonify({"success": False, "message": "Missing trade parameters."}), 400

    app.logger.info(f"Received trade request: {data}")

    # --- Paper Trading Logic ---
    if trade_mode == "PAPER":
        # Simulate a trade, no actual Fyers API call
        simulated_order_id = f"SIMULATED_ORDER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        app.logger.info(f"Simulating paper trade for {symbol} ({signal}) - Order ID: {simulated_order_id}")
        return jsonify({"success": True, "message": "Paper trade simulated successfully.", "orderId": simulated_order_id}), 200

    # --- Live Trading Logic ---
    elif trade_mode == "LIVE":
        try:
            # Determine side: 1 for BUY, -1 for SELL
            side = 1 if signal == "BUY" else -1

            # Fyers order data structure (adjust as per Fyers API documentation for the instrument)
            # This is a generic structure; for options, you'll need the correct instrument symbol.
            order_data = {
                "symbol": symbol, # e.g., "NSE:BANKNIFTY24JUN45000CE"
                "qty": int(quantity),
                "type": 2, # Assuming 2 for market order, adjust for LIMIT etc.
                "side": side,
                "productType": product_type,
                "limitPrice": 0, # Only for LIMIT orders
                "stopPrice": 0, # Only for SL orders
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": "False",
                "Hedge": {},
                "reduceOnly": "False",
            }

            if order_type == "LIMIT":
                order_data["type"] = 1 # Update type to LIMIT
                order_data["limitPrice"] = float(entry_price)
                order_data["stopPrice"] = 0 # Ensure stopPrice is 0 for LIMIT orders

            # Execute order via Fyers API
            response = fyers.place_order(data=order_data)

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


if __name__ == "__main__":
    # In production, use a WSGI server like Gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
