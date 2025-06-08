import os
import json
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from flask_cors import CORS
import threading
from jannat_algo_engine import execute_strategy, get_trade_details, get_capital_data # Corrected: 'execute_strategy', 'get_trade_details', 'get_capital_data' imported

# --- Fyers API V3 Imports ---
# Make sure you have the latest fyers-apiv3 installed: pip install fyers-apiv3
from fyers_apiv3 import fyersModel # Correct import for the main Fyers client model (includes SessionModel)
from fyers_apiv3.FyersWebsocket import data_ws # Correct import for the data WebSocket client


app = Flask(__name__)
CORS(app) # Enable CORS for deployment compatibility


# --- Configuration (Load from Environment Variables for Production) ---
FYERS_APP_ID = os.environ.get("FYERS_APP_ID", "YOUR_APP_ID_FROM_FYERS")
FYERS_SECRET_ID = os.environ.get("FYERS_SECRET_ID", "YOUR_SECRET_ID_FROM_FYERS")
FYERS_REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI", "https://jannat-backend-py.onrender.com/fyers_auth_callback") # Default to Render URL if not set
FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID", f"{FYERS_APP_ID}-100") # Typically App ID + "-100"

# Global variable to track if algo is running (for frontend polling)
algo_running_status = {"status": "stopped", "last_update": None}
algo_thread = None # To hold the reference to the algo thread


@app.route('/')
def home():
    return "Jannat Algo Backend is running!"

@app.route('/generate_auth_url', methods=['GET'])
def generate_auth_url():
    try:
        session = fyersModel.SessionModel(
            client_id=FYERS_CLIENT_ID,
            secret_key=FYERS_SECRET_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type='code',
            state='sample_state'
        )
        auth_url = session.generate_authcode()
        return jsonify({"success": True, "auth_url": auth_url}), 200
    except Exception as e:
        app.logger.error(f"Error generating auth URL: {e}")
        return jsonify({"success": False, "message": f"Error generating auth URL: {e}"}), 500

@app.route('/fyers_auth_callback', methods=['GET'])
def fyers_auth_callback():
    auth_code = request.args.get('auth_code')
    if not auth_code:
        return jsonify({"success": False, "message": "Auth code not found in callback."}), 400

    try:
        session = fyersModel.SessionModel(
            client_id=FYERS_CLIENT_ID,
            secret_key=FYERS_SECRET_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type='code',
            state='sample_state'
        )
        session.set_token(auth_code)
        response = session.generate_access_token()

        if response.get('s') == 'ok':
            access_token = response['access_token']
            # Save access token to a file (or database)
            with open('fyers_access_token.json', 'w') as f:
                json.dump({"access_token": access_token}, f)
            return jsonify({"success": True, "message": "Access token retrieved and saved."}), 200
        else:
            app.logger.error(f"Failed to generate access token: {response}")
            return jsonify({"success": False, "message": f"Failed to generate access token: {response.get('message', 'Unknown error')}"}), 500
    except Exception as e:
        app.logger.error(f"Error in Fyers auth callback: {e}")
        return jsonify({"success": False, "message": f"Internal server error in auth callback: {e}"}), 500

@app.route('/validate_credentials', methods=['GET'])
def validate_credentials():
    try:
        # Check if access token exists and is valid
        if not os.path.exists('fyers_access_token.json'):
            return jsonify({"status": "error", "message": "Access token not found. Please authenticate."}), 401

        with open('fyers_access_token.json', 'r') as f:
            token_data = json.load(f)
            access_token = token_data.get('access_token')

        if not access_token:
            return jsonify({"status": "error", "message": "Access token is empty. Please re-authenticate."}), 401

        fyers_api_client = fyersModel.FyersModel(
            client_id=FYERS_CLIENT_ID,
            is_token=True,
            token=access_token,
            log_path=os.getcwd()
        )
        profile_info = fyers_api_client.get_profile() # Test API call to validate token

        if profile_info and profile_info.get('s') == 'ok':
            return jsonify({"status": "success", "message": "Fyers credentials are valid."}), 200
        else:
            return jsonify({"status": "error", "message": f"Invalid Fyers credentials or token expired: {profile_info.get('message', 'Unknown error')}"}), 401
    except Exception as e:
        app.logger.error(f"Error validating credentials: {e}")
        return jsonify({"status": "error", "message": f"Internal server error validating credentials: {e}"}), 500

@app.route('/save_and_validate_credentials', methods=['POST'])
def save_and_validate_credentials():
    data = request.get_json()
    client_id = data.get('client_id')
    secret_key = data.get('secret_key')
    redirect_uri = data.get('redirect_uri')

    if not all([client_id, secret_key, redirect_uri]):
        return jsonify({"success": False, "message": "Missing client_id, secret_key, or redirect_uri"}), 400

    # For simplicity, we just return an auth URL here.
    # In a real scenario, you might save these temporarily or use them to generate the URL.
    # For this backend, we are already expecting them from env variables.
    # This route primarily serves to return the auth_url for the frontend to open.
    try:
        # Use provided client_id and secret_key for session model to generate auth URL
        session = fyersModel.SessionModel(
            client_id=client_id, # Use provided client_id
            secret_key=secret_key, # Use provided secret_key
            redirect_uri=redirect_uri, # Use provided redirect_uri
            response_type='code',
            state='sample_state'
        )
        auth_url = session.generate_authcode()
        return jsonify({"success": True, "auth_url": auth_url, "message": "Auth URL generated. Please complete Fyers authentication."}), 200
    except Exception as e:
        app.logger.error(f"Error in /save_and_validate_credentials: {e}")
        return jsonify({"success": False, "message": f"Error generating auth URL with provided credentials: {e}"}), 500


@app.route('/data/ohlcv', methods=['POST'])
def get_ohlcv_data():
    data = request.get_json()
    symbol = data.get('symbol')
    resolution = data.get('resolution')
    range_from = data.get('range_from')
    range_to = data.get('range_to')

    if not all([symbol, resolution, range_from, range_to]):
        return jsonify({"success": False, "message": "Missing symbol, resolution, range_from, or range_to"}), 400

    try:
        with open('fyers_access_token.json', 'r') as f:
            token_data = json.load(f)
            access_token = token_data.get('access_token')

        if not access_token:
            return jsonify({"success": False, "message": "Access token not found. Cannot fetch data."}), 401

        fyers_api_client = fyersModel.FyersModel(
            client_id=FYERS_CLIENT_ID,
            is_token=True,
            token=access_token,
            log_path=os.getcwd()
        )

        # Convert date strings to datetime objects for Fyers API
        from_date_obj = datetime.strptime(range_from, '%Y-%m-%d')
        to_date_obj = datetime.strptime(range_to, '%Y-%m-%d')

        history_data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1", # 1 for historical data
            "range_from": from_date_obj.strftime('%Y-%m-%d'),
            "range_to": to_date_obj.strftime('%Y-%m-%d'),
            "cont_flag": "1" # Continue flag for fetching more data if available
        }
        response = fyers_api_client.history(data=history_data)

        if response.get('s') == 'ok':
            return jsonify({"success": True, "data": response.get('candles', [])}), 200
        else:
            return jsonify({"success": False, "message": response.get('message', 'Failed to fetch OHLCV data')}), 500
    except Exception as e:
        app.logger.error(f"Error fetching OHLCV data: {e}")
        return jsonify({"success": False, "message": f"Internal server error fetching OHLCV data: {e}"}), 500


@app.route('/data/quote', methods=['POST'])
def get_quote_data():
    data = request.get_json()
    symbols = data.get('symbols') # Expects a list of symbols

    if not symbols:
        return jsonify({"success": False, "message": "Missing symbols"}), 400

    try:
        with open('fyers_access_token.json', 'r') as f:
            token_data = json.load(f)
            access_token = token_data.get('access_token')

        if not access_token:
            return jsonify({"success": False, "message": "Access token not found. Cannot fetch data."}), 401

        fyers_api_client = fyersModel.FyersModel(
            client_id=FYERS_CLIENT_ID,
            is_token=True,
            token=access_token,
            log_path=os.getcwd()
        )

        quote_data = {"symbols": ",".join(symbols)} # Fyers API expects comma-separated string
        response = fyers_api_client.quotes(data=quote_data)

        if response.get('s') == 'ok':
            return jsonify({"success": True, "data": response.get('d', [])}), 200
        else:
            return jsonify({"success": False, "message": response.get('message', 'Failed to fetch quote data')}), 500
    except Exception as e:
        app.logger.error(f"Error fetching quote data: {e}")
        return jsonify({"success": False, "message": f"Internal server error fetching quote data: {e}"}), 500


@app.route('/trade/execute', methods=['POST'])
def execute_trade():
    trade_details = request.get_json()
    symbol = trade_details.get('symbol')
    signal = trade_details.get('signal') # BUY or SELL
    entry_price = trade_details.get('entryPrice')
    target = trade_details.get('target')
    stop_loss = trade_details.get('stopLoss')
    quantity = trade_details.get('quantity')
    product_type = trade_details.get('product_type') # e.g., "MIS", "CNC", "INTRADAY"
    order_type = trade_details.get('order_type') # e.g., "LIMIT", "MARKET"
    trade_mode = trade_details.get('trade_mode', 'PAPER') # PAPER or LIVE

    if not all([symbol, signal, quantity, product_type, order_type]):
        return jsonify({"success": False, "message": "Missing required trade parameters."}), 400

    if trade_mode == 'PAPER':
        try:
            # Simulate paper trade success
            # Here you would typically log the paper trade in your trade log file
            # For simplicity, returning a simulated success
            trade_id = f"PAPER_TRADE_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            app.logger.info(f"Paper trade simulated for {symbol} ({signal}) - Trade ID: {trade_id}")
            return jsonify({"success": True, "message": "Paper trade simulated successfully.", "orderId": trade_id}), 200
        except Exception as e:
            app.logger.error(f"Error simulating paper trade: {e}")
            return jsonify({"success": False, "message": f"Internal server error simulating paper trade: {e}"}), 500
    elif trade_mode == 'LIVE':
        try:
            with open('fyers_access_token.json', 'r') as f:
                token_data = json.load(f)
                access_token = token_data.get('access_token')

            if not access_token:
                return jsonify({"success": False, "message": "Access token not found. Cannot place live trade."}), 401

            fyers_api_client = fyersModel.FyersModel(
                client_id=FYERS_CLIENT_ID,
                is_token=True,
                token=access_token,
                log_path=os.getcwd()
            )

            side = 1 if signal.upper() == 'BUY' else -1 # 1 for BUY, -1 for SELL
            limit_price = float(entry_price) if order_type.upper() == 'LIMIT' else 0 # Set limit price if order type is LIMIT

            order_data = {
                "symbol": symbol,
                "qty": int(quantity),
                "type": 2 if order_type.upper() == 'MARKET' else 1, # 1 for LIMIT, 2 for MARKET
                "side": side,
                "productType": product_type.upper(), # e.g., "MIS", "CNC"
                "limitPrice": limit_price,
                "stopPrice": 0, # Not using stopPrice directly here, SL will be managed by algo
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": "False"
            }

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

@app.route('/start_algo', methods=['GET', 'POST'])
def start_algo():
    global algo_running_status, algo_thread
    try:
        if algo_thread and algo_thread.is_alive():
            return jsonify({"status": "Algo Already Running", "message": "The algorithmic trading engine is already active."}), 200

        # Pass the global status dict and app logger to the algo engine
        threading.Thread(target=execute_strategy, args=(algo_running_status, app.logger,)).start()
        algo_running_status["status"] = "running"
        algo_running_status["last_update"] = datetime.now().isoformat()
        app.logger.info("Algorithmic trading engine started via API.")
        return jsonify({"status": "Algo Started", "message": "Algorithmic trading engine initiated successfully."}), 200
    except Exception as e:
        app.logger.error(f"Error starting algo: {e}")
        return jsonify({"status": "error", "message": f"Failed to start algo: {e}"}), 500

# --- NEW ENDPOINTS FOR FRONTEND ---
@app.route('/current_status', methods=['GET'])
def get_current_status():
    """
    Returns the current status of the algo and any active trade details.
    """
    global algo_running_status
    try:
        current_trade_details = get_trade_details() # This will be from jannat_algo_engine
        current_capital_data = get_capital_data() # This will be from jannat_algo_engine

        response_data = {
            "algo_status": algo_running_status["status"],
            "algo_last_update": algo_running_status["last_update"],
            "current_trade": current_trade_details,
            "capital_data": current_capital_data
        }
        return jsonify(response_data), 200
    except Exception as e:
        app.logger.error(f"Error fetching current status: {e}")
        return jsonify({"error": "Failed to retrieve current status", "message": str(e)}), 500

@app.route('/trade_history', methods=['GET'])
def get_trade_history():
    """
    Returns the full trade log from jannat_trade_log.json.
    """
    TRADE_LOG_FILE = os.path.join(os.environ.get("PERSISTENT_DISK_PATH", "."), "jannat_trade_log.json")
    if not os.path.exists(TRADE_LOG_FILE):
        return jsonify({"history": [], "message": "Trade log file not found."}), 200
    try:
        with open(TRADE_LOG_FILE, 'r') as f:
            trade_history = json.load(f)
        return jsonify({"history": trade_history}), 200
    except json.JSONDecodeError:
        app.logger.error("Error decoding jannat_trade_log.json. File might be empty or corrupted.")
        return jsonify({"history": [], "message": "Error reading trade log. File might be corrupted."}), 200
    except Exception as e:
        app.logger.error(f"Error fetching trade history: {e}")
        return jsonify({"error": "Failed to retrieve trade history", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
