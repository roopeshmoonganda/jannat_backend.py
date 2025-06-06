import os
import json
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from flask_cors import CORS # Added for deployment
# Install the Fyers API Python SDK: pip install fyers-api==1.0.9
[span_1](start_span)from fyers_api import accessToken, fyersModel # Changed import[span_1](end_span)


app = Flask(__name__)
[span_2](start_span)CORS(app) # Enable CORS for deployment compatibility[span_2](end_span)


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
[span_3](start_span)FYERS_REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI", "https://google.com") # Must match your Fyers app settings[span_3](end_span)
[span_4](start_span)FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID", f"{FYERS_APP_ID}-100" if FYERS_APP_ID != "YOUR_APP_ID_FROM_FYERS" else "") # Example: "YOUR_APP_ID-100"[span_4](end_span)




# [span_5](start_span)This file will store the access token generated after authentication[span_5](end_span)
[span_6](start_span)ACCESS_TOKEN_STORAGE_FILE = "fyers_access_token.json"[span_6](end_span)




# Global Fyers client instance
fyers = None




def load_access_token():
    """Loads the access token from a file."""
    if os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
        with open(ACCESS_TOKEN_STORAGE_FILE, 'r') as f:
            data = json.load(f)
            # Check for expiry if Fyers provides it
            # [span_7](start_span)For simplicity, we assume the token is valid if present.[span_7](end_span)
            # [span_8](start_span)In production, you'd handle token refresh or re-authentication.[span_8](end_span)
            [span_9](start_span)return data.get('access_token')[span_9](end_span)
    return None




def save_access_token(token):
    """Saves the access token to a file."""
    with open(ACCESS_TOKEN_STORAGE_FILE, 'w') as f:
        [span_10](start_span)[span_11](start_span)json.dump({'access_token': token, 'timestamp': datetime.now().isoformat()}, f)[span_10](end_span)[span_11](end_span)




@app.before_request
def check_fyers_client_initialized():
    """Ensures Fyers client is initialized before processing requests."""
    global fyers
    if fyers is None:
        access_token = load_access_token()
        if access_token:
            try:
                # [span_12](start_span)Use fyersModel for 1.0.9 compatibility[span_12](end_span)
                [span_13](start_span)fyers = fyersModel.FyersModel(client_id=FYERS_CLIENT_ID, token=access_token, log_path=os.getcwd())[span_13](end_span)
                app.logger.info("Fyers client initialized from stored token.")
            except Exception as e:
                app.logger.error(f"Error initializing Fyers client with stored token: {e}")
                fyers = None # Reset if initialization fails
        else:
            [span_14](start_span)app.logger.warning("Fyers access token not found or expired. Please authenticate via /generate_auth_url.")[span_14](end_span)




# --- Fyers Authentication Flow ---
@app.route("/generate_auth_url")
def generate_auth_url():
    """Generates the Fyers authentication URL for manual login."""
    try:
        # [span_15](start_span)Use accessToken.SessionModel for 1.0.9 compatibility[span_15](end_span)
        session = accessToken.SessionModel(
            client_id=FYERS_CLIENT_ID,
            secret_key=FYERS_SECRET_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            [span_16](start_span)response_type='code', # 'code' for authorization code flow[span_16](end_span)
            [span_17](start_span)grant_type='authorization_code'[span_17](end_span)
        )
        [span_18](start_span)response = session.generate_authcode()[span_18](end_span)
        return jsonify({"success": True, "auth_url": response}), 200
    except Exception as e:
        app.logger.error(f"Error generating Fyers auth URL: {e}")
        return jsonify({"success": False, "message": f"Failed to generate auth URL: {e}"}), 500




@app.route("/fyers_auth_callback")
def fyers_auth_callback():
    """Callback endpoint for Fyers OAuth flow."""
    [span_19](start_span)[span_20](start_span)auth_code = request.args.get('auth_code')[span_19](end_span)[span_20](end_span)
    if not auth_code:
        [span_21](start_span)error = request.args.get('error')[span_21](end_span)
        return jsonify({"success": False, "message": f"Fyers authentication failed: {error}"}), 400


    try:
        # [span_22](start_span)Use accessToken.SessionModel for 1.0.9 compatibility[span_22](end_span)
        session = accessToken.SessionModel(
            client_id=FYERS_CLIENT_ID,
            secret_key=FYERS_SECRET_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type='code',
            grant_type='authorization_code'
        )
        [span_23](start_span)[span_24](start_span)session.set_token(auth_code)[span_23](end_span)[span_24](end_span)
        [span_25](start_span)[span_26](start_span)response = session.generate_token()[span_25](end_span)[span_26](end_span)
        [span_27](start_span)[span_28](start_span)access_token = response["access_token"][span_27](end_span)[span_28](end_span)
        [span_29](start_span)[span_30](start_span)save_access_token(access_token) # Store the access token securely[span_29](end_span)[span_30](end_span)
        
        global fyers
        # [span_31](start_span)Use fyersModel for 1.0.9 compatibility[span_31](end_span)
        [span_32](start_span)fyers = fyersModel.FyersModel(client_id=FYERS_CLIENT_ID, token=access_token, log_path=os.getcwd())[span_32](end_span)
        app.logger.info("Fyers client initialized successfully with new token.")


        [span_33](start_span)return jsonify({"success": True, "message": "Fyers token generated and saved successfully!", "access_token": access_token}), 200[span_33](end_span)
    except Exception as e:
        app.logger.error(f"Error processing Fyers auth callback: {e}")
        [span_34](start_span)[span_35](start_span)return jsonify({"success": False, "message": f"Failed to generate token: {e}"}), 500[span_34](end_span)[span_35](end_span)




# Endpoint for the React Native app to validate credentials (App ID, Secret ID, Client ID)
# and ensure backend has a valid access token.
@app.route("/validate_credentials", methods=["POST"])
def validate_credentials():
    data = request.json
    app_id = data.get('app_id')
    secret_id = data.get('secret_id')
    client_id = data.get('client_id')
    [span_36](start_span)access_key = data.get('access_key') # Access key sent from fyers_token.txt[span_36](end_span)


    if not all([app_id, secret_id, client_id, access_key]):
        [span_37](start_span)return jsonify({"success": False, "message": "Missing App ID, Secret ID, Client ID, or Access Key"}), 400[span_37](end_span)


    # [span_38](start_span)Basic validation: Check if provided IDs match backend config (environment variables)[span_38](end_span)
    if app_id != FYERS_APP_ID or secret_id != FYERS_SECRET_ID or client_id != FYERS_CLIENT_ID:
        [span_39](start_span)return jsonify({"success": False, "message": "App ID, Secret ID, or Client ID mismatch with backend configuration."}), 401[span_39](end_span)


    # [span_40](start_span)Attempt to initialize Fyers client with the provided access key for a quick check[span_40](end_span)
    # [span_41](start_span)In a real scenario, you'd want to verify token validity with Fyers if possible.[span_41](end_span)
    try:
        # [span_42](start_span)Use fyersModel for 1.0.9 compatibility[span_42](end_span)
        [span_43](start_span)test_fyers = fyersModel.FyersModel(client_id=client_id, token=access_key, log_path=os.getcwd())[span_43](end_span)
        # [span_44](start_span)Try a simple API call to verify token (e.g., get profile)[span_44](end_span)
        [span_45](start_span)profile_data = test_fyers.get_profile()[span_45](end_span)
        if profile_data and profile_data.get('s') == 'ok':
            # [span_46](start_span)If valid, save this access token for future use by the backend[span_46](end_span)
            [span_47](start_span)save_access_token(access_key)[span_47](end_span)
            global fyers
            [span_48](start_span)fyers = test_fyers # Set the global Fyers client[span_48](end_span)
            [span_49](start_span)return jsonify({"success": True, "message": "Credentials validated and Fyers client initialized."}), 200[span_49](end_span)
        else:
            app.logger.error(f"Access Key validation failed (Fyers response: {profile_data})")
            [span_50](start_span)return jsonify({"success": False, "message": "Access Key invalid or expired."}), 401[span_50](end_span)
    except Exception as e:
        [span_51](start_span)app.logger.error(f"Error validating access key: {e}")[span_51](end_span)
        [span_52](start_span)return jsonify({"success": False, "message": f"Error validating credentials: {e}"}), 500[span_52](end_span)




@app.route("/save_and_validate_credentials", methods=["POST"])
def save_and_validate_credentials():
    # [span_53](start_span)This endpoint is primarily for when the user manually enters App ID/Secret ID.[span_53](end_span)
    # [span_54](start_span)It doesn't handle access token generation itself; it just validates against backend's config.[span_54](end_span)
    # [span_55](start_span)For a full flow, this would trigger an authentication URL generation for the user to open.[span_55](end_span)
    data = request.json
    app_id = data.get('app_id')
    secret_id = data.get('secret_id')


    if not all([app_id, secret_id]):
        [span_56](start_span)return jsonify({"success": False, "message": "Missing App ID or Secret ID"}), 400[span_56](end_span)


    # [span_57](start_span)For now, just compare with backend's configured FYERS_APP_ID and FYERS_SECRET_ID[span_57](end_span)
    if app_id == FYERS_APP_ID and secret_id == FYERS_SECRET_ID:
        return jsonify({"success": True, "message": "App ID and Secret ID matched backend configuration. Now generate/load access token."}), 200
    else:
        [span_58](start_span)return jsonify({"success": False, "message": "App ID or Secret ID mismatch with backend configuration. Please check your credentials or backend setup."}), 401[span_58](end_span)




# --- Data Endpoints ---
@app.route("/data/ohlcv")
def get_ohlcv():
    global fyers
    if not fyers:
        [span_59](start_span)return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401[span_59](end_span)


    symbol = request.args.get('symbol', "NSE:BANKNIFTY")
    interval = request.args.get('interval', "1")
    days = request.args.get('days', "3")


    try:
        data = {
            [span_60](start_span)"symbol": symbol, #[span_60](end_span)
            [span_61](start_span)"resolution": interval, #[span_61](end_span)
            [span_62](start_span)"date_format": "1", #[span_62](end_span)
            [span_63](start_span)"range_from": (datetime.now() - timedelta(days=int(days))).strftime('%Y-%m-%d'), #[span_63](end_span)
            [span_64](start_span)"range_to": datetime.now().strftime('%Y-%m-%d'), #[span_64](end_span)
            [span_65](start_span)"cont_flag": "1" #[span_65](end_span)
        }
        [span_66](start_span)response = fyers.history(data=data)[span_66](end_span)
        [span_67](start_span)if response.get('s') == 'ok' and response.get('candles'): #[span_67](end_span)
            # [span_68](start_span)Fyers API returns [timestamp, open, high, low, close, volume][span_68](end_span)
            [span_69](start_span)return jsonify({"success": True, "candles": response['candles']}), 200[span_69](end_span)
        else:
            app.logger.error(f"Fyers OHLCV API error: {response.get('message', 'Unknown error')}")
            [span_70](start_span)return jsonify({"success": False, "message": response.get('message', 'Failed to fetch OHLCV data from Fyers.')}), 500[span_70](end_span)
    except Exception as e:
        [span_71](start_span)app.logger.error(f"Error fetching OHLCV data: {e}")[span_71](end_span)
        [span_72](start_span)return jsonify({"success": False, "message": f"Internal server error fetching OHLCV: {e}"}), 500[span_72](end_span)




@app.route("/data/quote")
def get_quote():
    global fyers
    if not fyers:
        [span_73](start_span)return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401[span_73](end_span)


    symbol = request.args.get('symbol', "NSE:BANKNIFTY")


    try:
        [span_74](start_span)data = {"symbols": symbol}[span_74](end_span)
        [span_75](start_span)response = fyers.quotes(data=data)[span_75](end_span)
        [span_76](start_span)if response.get('s') == 'ok' and response.get('d') and response['d'][0].get('v'):[span_76](end_span)
            [span_77](start_span)return jsonify({"success": True, "quote": response['d'][0]['v']}), 200[span_77](end_span)
        else:
            app.logger.error(f"Fyers Quote API error: {response.get('message', 'Unknown error')}")
            [span_78](start_span)return jsonify({"success": False, "message": response.get('message', 'Failed to fetch quote from Fyers.')}), 500[span_78](end_span)
    except Exception as e:
        [span_79](start_span)app.logger.error(f"Error fetching quote: {e}")[span_79](end_span)
        [span_80](start_span)return jsonify({"success": False, "message": f"Internal server error fetching quote: {e}"}), 500[span_80](end_span)




# --- Trade Execution Endpoint ---
@app.route("/trade/execute", methods=["POST"])
def execute_trade():
    global fyers
    if not fyers:
        [span_81](start_span)return jsonify({"success": False, "message": "Fyers client not initialized. Authenticate first."}), 401[span_81](end_span)


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
    [span_82](start_span)trade_mode = data.get('tradeMode') # "PAPER" or "LIVE"[span_82](end_span)


    if not all([symbol, signal, entry_price, target, stop_loss, quantity, product_type, order_type, trade_mode]):
        [span_83](start_span)return jsonify({"success": False, "message": "Missing trade parameters."}), 400[span_83](end_span)


    [span_84](start_span)app.logger.info(f"Received trade request: {data}")[span_84](end_span)


    # --- Paper Trading Logic ---
    if trade_mode == "PAPER":
        # Simulate a trade, no actual Fyers API call
        [span_85](start_span)simulated_order_id = f"SIMULATED_ORDER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"[span_85](end_span)
        [span_86](start_span)app.logger.info(f"Simulating paper trade for {symbol} ({signal}) - Order ID: {simulated_order_id}")[span_86](end_span)
        [span_87](start_span)return jsonify({"success": True, "message": "Paper trade simulated successfully.", "orderId": simulated_order_id}), 200[span_87](end_span)


    # --- Live Trading Logic ---
    [span_88](start_span)elif trade_mode == "LIVE": #[span_88](end_span)
        try:
            # [span_89](start_span)Determine side: 1 for BUY, -1 for SELL[span_89](end_span)
            [span_90](start_span)side = 1 if signal == "BUY" else -1[span_90](end_span)
            
            # [span_91](start_span)Fyers order data structure (adjust as per Fyers API documentation for the instrument)[span_91](end_span)
            # [span_92](start_span)This is a generic structure; for options, you'll need the correct instrument symbol.[span_92](end_span)
            order_data = {
                [span_93](start_span)"symbol": symbol, # e.g., "NSE:BANKNIFTY24JUN45000CE"[span_93](end_span)
                [span_94](start_span)"qty": int(quantity), #[span_94](end_span)
                [span_95](start_span)"type": 2,  # 2 for MARKET order, 1 for LIMIT order[span_95](end_span)
                [span_96](start_span)"side": side, #[span_96](end_span)
                [span_97](start_span)"productType": product_type, # e.g., "MIS"[span_97](end_span)
                [span_98](start_span)"limitPrice": 0, # For MARKET order[span_98](end_span)
                [span_99](start_span)"stopPrice": 0, # For MARKET order[span_99](end_span)
                [span_100](start_span)"validity": "DAY", #[span_100](end_span)
                [span_101](start_span)"disclosedQty": 0, #[span_101](end_span)
                [span_102](start_span)"offlineOrder": "False", #[span_102](end_span)
                [span_103](start_span)"Hedge": {}, #[span_103](end_span)
                [span_104](start_span)"reduceOnly": "False", #[span_104](end_span)
            }


            if order_type == "LIMIT":
                [span_105](start_span)order_data["type"] = 1[span_105](end_span)
                [span_106](start_span)order_data["limitPrice"] = float(entry_price) # Use entryPrice as limit price[span_106](end_span)
                [span_107](start_span)order_data["stopPrice"] = 0 # No stop price for simple limit order[span_107](end_span)


            # [span_108](start_span)Execute order via Fyers API[span_108](end_span)
            [span_109](start_span)response = fyers.place_order(data=order_data)[span_109](end_span)


            [span_110](start_span)if response.get('s') == 'ok':[span_110](end_span)
                [span_111](start_span)order_id = response['id'][span_111](end_span)
                [span_112](start_span)app.logger.info(f"Live trade placed for {symbol} ({signal}) - Fyers Order ID: {order_id}")[span_112](end_span)
                [span_113](start_span)return jsonify({"success": True, "message": "Live trade placed successfully.", "orderId": order_id}), 200[span_113](end_span)
            else:
                app.logger.error(f"Fyers order placement error: {response.get('message', 'Unknown error')}")
                [span_114](start_span)return jsonify({"success": False, "message": response.get('message', 'Failed to place live order.')}), 500[span_114](end_span)
        except Exception as e:
            [span_115](start_span)app.logger.error(f"Error placing live trade: {e}")[span_115](end_span)
            [span_116](start_span)return jsonify({"success": False, "message": f"Internal server error placing live trade: {e}"}), 500[span_116](end_span)
    else:
        [span_117](start_span)return jsonify({"success": False, "message": "Invalid trade mode specified."}), 400[span_117](end_span)




if __name__ == "__main__":
    # In production, use a WSGI server like Gunicorn
    [span_118](start_span)app.run(host="0.0.0.0", port=5000, debug=True) # debug=True for development, set to False in production[span_118](end_span)
