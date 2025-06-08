from flask import Flask, jsonify, request, redirect, url_for
from flask_cors import CORS
import threading
import json
import os
from datetime import datetime, timedelta
import logging # Import logging module for Flask app
import sys # Import sys module for stdout

# Import your algo engine functions and its logger
# Note: The algo_engine_logger is imported but the algo engine
# is now configured to log directly to sys.stdout and a file.
# The app.logger will still be used for Flask-specific logs.
from jannat_algo_engine import execute_strategy, get_trade_details, get_capital_data

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- Flask App Logging Configuration ---
# Get the logger for the Flask app
app.logger.setLevel(logging.INFO) # Set default log level for Flask app
# Remove default handler if it exists to avoid duplicate logs in some environments
if app.logger.handlers:
    for handler in app.logger.handlers:
        app.logger.removeHandler(handler)
        handler.close() # Important: close file handlers to release locks

# Add a StreamHandler to output Flask logs to stdout
flask_stream_handler = logging.StreamHandler(sys.stdout)
flask_stream_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s'))
app.logger.addHandler(flask_stream_handler)


# --- Configuration (ensure this matches jannat_algo_engine.py) ---
# This path must match the persistent disk mount path configured in Render
PERSISTENT_DISK_BASE_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/var/data")
ACCESS_TOKEN_STORAGE_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "fyers_access_token.json")
TRADE_LOG_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_trade_log.json") # Assuming this is where past trades are saved
ALGO_ENGINE_LOG_FILE = os.path.join(PERSISTENT_DISK_BASE_PATH, "jannat_algo_engine.log")

# --- Fyers API Credentials (from environment variables) ---
FYERS_APP_ID = os.environ.get("FYERS_APP_ID")
FYERS_SECRET_ID = os.environ.get("FYERS_SECRET_ID")
FYERS_REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI") # This should point to your Render backend URL, e.g., https://your-service.onrender.com/fyers_auth_callback

# Global dictionary to control algo thread status
algo_status_dict = {"status": "stopped", "last_update": datetime.now().isoformat()}
algo_thread = None # Global variable to hold the algo thread instance

# --- Fyers Authentication Endpoints ---
@app.route('/login')
def login():
    """Initiates the Fyers authentication process."""
    if not FYERS_APP_ID or not FYERS_REDIRECT_URI:
        app.logger.error("Fyers APP_ID or REDIRECT_URI not set in environment variables.")
        return jsonify({"error": "Fyers API credentials not configured"}), 500

    # Assuming fyersModel is imported and available, it's used for generating login URL
    # This requires an instance of FyersModel for session handling, which is usually created during token generation
    # For login URL, we need to create a temporary session model
    try:
        from fyers_apiv3 import fyersModel # Import here to avoid circular dependency if algo_engine imports app.py parts
        session = fyersModel.SessionModel(
            client_id=FYERS_APP_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type='code',
            state='sample_state', # A state parameter to prevent CSRF, can be dynamic
            secret_key=FYERS_SECRET_ID, # Secret key needed for token generation, but often for authcode only client_id/redirect_uri are exposed
            grant_type='authorization_code'
        )
        generate_authcode_url = session.generate_authcode()
        app.logger.info(f"Redirecting for Fyers authentication: {generate_authcode_url}")
        return redirect(generate_authcode_url)
    except Exception as e:
        app.logger.error(f"Error generating Fyers authcode URL: {e}")
        return jsonify({"error": f"Failed to generate Fyers login URL: {e}"}), 500


@app.route('/fyers_auth_callback')
def fyers_auth_callback():
    """Handles the callback from Fyers after successful authentication."""
    auth_code = request.args.get('auth_code')
    if not auth_code:
        error = request.args.get('error')
        app.logger.error(f"Fyers authentication failed or no auth_code received. Error: {error}")
        return jsonify({"status": "Failed", "message": f"Authentication failed: {error}"}), 400

    app.logger.info(f"Received auth_code: {auth_code}")

    try:
        from fyers_apiv3 import fyersModel
        session = fyersModel.SessionModel(
            client_id=FYERS_APP_ID,
            secret_key=FYERS_SECRET_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type='code',
            grant_type='authorization_code'
        )
        session.set_token(auth_code)
        response = session.generate_token()

        if response.get("code") == 200 and response.get("access_token"):
            access_token = response["access_token"]
            app.logger.info("Access token generated successfully.")

            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(ACCESS_TOKEN_STORAGE_FILE), exist_ok=True)
            with open(ACCESS_TOKEN_STORAGE_FILE, 'w') as f:
                json.dump({"access_token": access_token}, f)
            app.logger.info(f"Access token saved to {ACCESS_TOKEN_STORAGE_FILE}")
            return jsonify({"status": "Success", "message": "Fyers authenticated and token saved."}), 200
        else:
            app.logger.error(f"Failed to generate access token: {response}")
            return jsonify({"status": "Failed", "message": f"Failed to generate access token: {response.get('message', 'Unknown error')}"}), 500
    except Exception as e:
        app.logger.error(f"Error during Fyers token generation: {e}")
        return jsonify({"status": "Error", "message": f"Token generation error: {e}"}), 500

# --- Algo Engine Control Endpoints ---
@app.route('/start_algo', methods=['GET', 'POST'])
def start_algo():
    global algo_thread, algo_status_dict
    # Check if thread is already running or if status is 'running' but thread is dead
    if algo_status_dict["status"] == "running" and algo_thread and algo_thread.is_alive():
        app.logger.info("Attempted to start algo, but it's already running.")
        return jsonify({"status": "Algo already running"}), 200

    try:
        # The algo_status_dict allows the algo engine to check its own status and stop
        # app.logger is passed for compatibility, but the algo engine now uses its own configured logger
        algo_thread = threading.Thread(target=execute_strategy, args=(algo_status_dict, app.logger,))
        algo_thread.daemon = True # Allows Flask app to exit gracefully even if algo thread is running
        algo_thread.start()
        algo_status_dict["status"] = "running"
        algo_status_dict["last_update"] = datetime.now().isoformat()
        app.logger.info("Algo engine started in background thread.")
        return jsonify({"status": "Algo Started"}), 200
    except Exception as e:
        app.logger.error(f"Error starting algo: {e}", exc_info=True) # Log traceback
        # Reset status if starting failed
        algo_status_dict["status"] = "stopped"
        return jsonify({"status": "Error", "message": f"Failed to start algo: {e}"}), 500

@app.route('/stop_algo', methods=['GET', 'POST'])
def stop_algo():
    global algo_status_dict, algo_thread
    if algo_status_dict["status"] == "stopped":
        app.logger.info("Attempted to stop algo, but it's already stopped.")
        return jsonify({"status": "Algo already stopped"}), 200

    app.logger.info("Stopping algo engine initiated.")
    algo_status_dict["status"] = "stopped" # This signal will stop the loop in execute_strategy
    algo_status_dict["last_update"] = datetime.now().isoformat()

    # Optional: Wait for the thread to actually finish for graceful shutdown.
    # Be careful with `join()` in a web server context as it can block the request.
    # if algo_thread and algo_thread.is_alive():
    #     app.logger.info("Waiting for algo thread to terminate...")
    #     algo_thread.join(timeout=5) # Wait up to 5 seconds for thread to finish
    #     if algo_thread.is_alive():
    #         app.logger.warning("Algo thread did not terminate gracefully within timeout.")
    #     else:
    #         app.logger.info("Algo thread terminated gracefully.")

    return jsonify({"status": "Algo Stopped"}), 200


# --- Frontend Data Endpoints ---

@app.route('/api/status', methods=['GET'])
def get_algo_status():
    """Returns the current status of the algo engine."""
    global algo_status_dict
    return jsonify(algo_status_dict), 200

@app.route('/api/capital', methods=['GET'])
def get_current_capital():
    """Returns current capital data (balance, pnl_today, etc.)."""
    capital_data = get_capital_data() # Assuming this function exists in jannat_algo_engine
    return jsonify(capital_data), 200

@app.route('/api/active_trades', methods=['GET'])
def get_current_active_trades():
    """Returns a list of currently active trades."""
    active_trades = get_trade_details() # Assuming this function exists in jannat_algo_engine
    return jsonify(active_trades), 200

@app.route('/api/past_trades', methods=['GET'])
def get_historical_trades():
    """Returns a list of historical (closed) trades from the JSON log file."""
    if not os.path.exists(TRADE_LOG_FILE):
        return jsonify([]), 200 # Return empty list if file doesn't exist

    try:
        with open(TRADE_LOG_FILE, 'r') as f:
            content = f.read()
            if not content: # Handle empty file
                return jsonify([]), 200
            trade_log = json.loads(content)
            # Ensure proper date/time format for frontend if needed, e.g., convert to ISO format
            # For simplicity, returning as is, assuming frontend handles date parsing
            return jsonify(trade_log), 200
    except json.JSONDecodeError:
        app.logger.error(f"Error decoding trade log JSON at {TRADE_LOG_FILE}. File might be corrupted.", exc_info=True)
        return jsonify([]), 500
    except Exception as e:
        app.logger.error(f"Error reading trade log: {e}", exc_info=True)
        return jsonify([]), 500

@app.route('/api/logs', methods=['GET'])
def get_production_logs():
    """Reads and returns the last N lines of the algo engine's log file."""
    num_lines = request.args.get('lines', default=500, type=int) # Default to 500 lines for logs
    
    logs = []
    if os.path.exists(ALGO_ENGINE_LOG_FILE):
        try:
            with open(ALGO_ENGINE_LOG_FILE, 'r') as f:
                all_lines = f.readlines()
                logs = [line.strip() for line in all_lines[-num_lines:]]
        except Exception as e:
            app.logger.error(f"Error reading algo engine log file: {e}", exc_info=True)
            logs = [f"Error reading logs: {e}"] # Return error message within logs
    else:
        logs = [f"Algo engine log file not found at {ALGO_ENGINE_LOG_FILE}. Ensure it's configured to write to persistent disk."]

    return jsonify({"logs": logs}), 200 # Return logs in a dictionary under 'logs' key


# --- Flask Entry Point ---
if __name__ == '__main__':
    # Ensure directories for persistent data are created on startup
    app.logger.info(f"Ensuring persistent disk path exists: {PERSISTENT_DISK_BASE_PATH}")
    os.makedirs(PERSISTENT_DISK_BASE_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(ACCESS_TOKEN_STORAGE_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(ALGO_ENGINE_LOG_FILE), exist_ok=True)

    # For local testing, ensure a dummy token exists if you're not going through auth every time
    # In a production environment, this file should be created via the /login flow
    if not os.path.exists(ACCESS_TOKEN_STORAGE_FILE):
        app.logger.warning(f"Access token file not found at {ACCESS_TOKEN_STORAGE_FILE}. "
                           "Algo will not run without a valid Fyers access token. "
                           "Please authenticate via /login endpoint.")
        # You might create a dummy file for local development if you manually set the token for debugging
        # with open(ACCESS_TOKEN_STORAGE_FILE, 'w') as f:
        #     json.dump({"access_token": "YOUR_DUMMY_FYERS_ACCESS_TOKEN_FOR_DEV"}, f)

    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
