from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os
import subprocess
import threading


app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

class OracleServer:
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.models = {}  # LSTM models
        self.rf_models = {}  # Random Forest models
        self.norm_params = {}
        self.history_data = {}
        self.symbols = ['AAPL', 'MSFT']
        self.features = ['Close', 'High', 'Low', 'Open', 'Volume']
        self.output_folder = "project_files"
        self.updating = False
        self.update_status = {"step": "", "message": "", "progress": 0}
        self.load_resources()
        
    def load_resources(self):
        """Load all necessary resources for each symbol and feature"""
        for symbol in self.symbols:
            try:
                symbol_dir = os.path.join(self.output_folder, symbol)
                if not os.path.exists(symbol_dir):
                    os.makedirs(symbol_dir, exist_ok=True)
                    print(f"Created directory for {symbol}")
                    continue
                
                # Initialize data structures for this symbol
                self.models[symbol] = {}
                self.rf_models[symbol] = {}
                self.history_data[symbol] = {}
                
                # Load norm parameters
                norm_path = os.path.join(symbol_dir, f'norm_params_{symbol}.npy')
                if os.path.exists(norm_path):
                    self.norm_params[symbol] = np.load(norm_path, allow_pickle=True).item()
                    print(f"Loaded normalization parameters for {symbol}")
                
                # Load history data for each feature
                for feature in self.features:
                    history_path = os.path.join(symbol_dir, f'full_history_{symbol}_{feature}.npy')
                    if os.path.exists(history_path):
                        self.history_data[symbol][feature] = np.load(history_path)
                        print(f"Loaded {feature} history for {symbol}")
                    
                    # Load LSTM model for this feature
                    model_path = os.path.join(symbol_dir, f'model_{symbol}_{feature}.keras')
                    if os.path.exists(model_path):
                        self.models[symbol][feature] = load_model(model_path)
                        print(f"Loaded LSTM model for {symbol} - {feature}")
                    
                    # Load RF model for this feature
                    rf_model_path = os.path.join(symbol_dir, f'model_rf_{symbol}_{feature}.joblib')
                    if os.path.exists(rf_model_path):
                        self.rf_models[symbol][feature] = joblib.load(rf_model_path)
                        print(f"Loaded RF model for {symbol} - {feature}")
                
                print(f"Successfully loaded resources for {symbol}")
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
    
    def download_latest_data(self, symbol=None):
        """Download latest stock data for one or all symbols"""
        self.update_status = {"step": "download", "message": "Downloading latest stock data...", "progress": 20}
        
        # Ensure output directory exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        if symbol is not None:
            symbols_to_process = [symbol]
        else:
            symbols_to_process = self.symbols
        
        success = True
        for sym in symbols_to_process:
            try:
                symbol_dir = os.path.join(self.output_folder, sym)
                os.makedirs(symbol_dir, exist_ok=True)
                
                # Download latest data using yfinance
                print(f"Downloading latest data for {sym}...")
                stock = yf.Ticker(sym)
                df = stock.history(period='3y')  # Download 3 years of data
                
                if df.empty:
                    print(f"No data found for {sym}")
                    continue
                
                # Save raw data for each feature
                for feature in self.features:
                    if feature in df.columns:
                        raw_path = os.path.join(symbol_dir, f'{sym}_latest_{feature}.npy')
                        np.save(raw_path, df[feature].values)
                        
                        # Also save full history
                        hist_path = os.path.join(symbol_dir, f'full_history_{sym}_{feature}.npy')
                        np.save(hist_path, df[feature].values)
                        
                        print(f"Saved {feature} data for {sym}")
                
                # If it's a new symbol, add to the list
                if sym not in self.symbols:
                    self.symbols.append(sym)
                    print(f"Added new symbol: {sym}")
                
                print(f"Successfully downloaded data for {sym}")
            except Exception as e:
                print(f"Error downloading {sym} data: {str(e)}")
                success = False
        
        return success
    
    def update_data(self, specific_symbol=None):
        """Update data and models for one or all symbols"""
        if self.updating:
            return False, "Data update already in progress, please try again later"
        
        self.updating = True
        self.update_status = {"step": "start", "message": "Starting data update...", "progress": 0}
        
        try:
            # Step 1: Download latest data
            if not self.download_latest_data(specific_symbol):
                self.update_status = {"step": "error", "message": "Failed to download data", "progress": 0}
                self.updating = False
                return False, "Failed to download latest data"
            
            # Step 2: Data preprocessing
            self.update_status = {"step": "prepare", "message": "Preprocessing data...", "progress": 40}
            print("Starting data preprocessing...")
            
            # If updating a specific symbol, pass it as an argument
            if specific_symbol:
                result1 = subprocess.run(['python', 'data_preparation.py', specific_symbol], 
                                        capture_output=True, text=True, check=False)
            else:
                result1 = subprocess.run(['python', 'data_preparation.py'], 
                                        capture_output=True, text=True, check=False)
            
            if result1.returncode != 0:
                error_msg = f"Data preprocessing failed: {result1.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "Data preprocessing failed", "progress": 40}
                self.updating = False
                return False, error_msg
            
            # Step 3: Train LSTM models
            self.update_status = {"step": "lstm", "message": "Training LSTM models...", "progress": 60}
            print("Starting LSTM model training...")
            
            if specific_symbol:
                result2 = subprocess.run(['python', 'train_model.py', specific_symbol], 
                                        capture_output=True, text=True, check=False)
            else:
                result2 = subprocess.run(['python', 'train_model.py'], 
                                        capture_output=True, text=True, check=False)
            
            if result2.returncode != 0:
                error_msg = f"LSTM model training failed: {result2.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "LSTM model training failed", "progress": 60}
                self.updating = False
                return False, error_msg
            
            # Step 4: Train Random Forest models
            self.update_status = {"step": "rf", "message": "Training Random Forest models...", "progress": 80}
            print("Starting Random Forest model training...")
            
            if specific_symbol:
                result3 = subprocess.run(['python', 'train_model_rf.py', specific_symbol], 
                                        capture_output=True, text=True, check=False)
            else:
                result3 = subprocess.run(['python', 'train_model_rf.py'], 
                                        capture_output=True, text=True, check=False)
            
            if result3.returncode != 0:
                error_msg = f"Random Forest model training failed: {result3.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "Random Forest model training failed", "progress": 80}
                self.updating = False
                return False, error_msg
            
            # Step 5: Reload resources
            self.update_status = {"step": "reload", "message": "Reloading models and data...", "progress": 95}
            print("Reloading models and data...")
            self.load_resources()
            
            # Update complete
            self.update_status = {"step": "complete", "message": "Data and models successfully updated", "progress": 100}
            self.updating = False
            return True, "Data and models successfully updated"
        
        except Exception as e:
            self.updating = False
            error_msg = f"Error during update process: {str(e)}"
            self.update_status = {"step": "error", "message": error_msg, "progress": 0}
            print(error_msg)
            return False, error_msg
    
    def async_update_data(self, symbol=None):
        """Asynchronously execute data update"""
        threading.Thread(target=lambda: self._async_update_worker(symbol)).start()
        return True, "Data update started, please wait..."
    
    def _async_update_worker(self, symbol=None):
        """Asynchronous update worker thread"""
        self.update_data(symbol)
    
    def get_update_status(self):
        """Get current update status"""
        return self.update_status
    
    def get_available_symbols(self):
        """Get list of available symbols"""
        return self.symbols
    
    def add_new_symbol(self, symbol):
        """Add and process a new symbol"""
        try:
            # Check if symbol is valid
            stock = yf.Ticker(symbol)
            info = stock.info
            if 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
                return False, f"Invalid symbol: {symbol}"
            
            # Add symbol to list if not already present
            if symbol not in self.symbols:
                # Start the update process for this symbol
                success, message = self.async_update_data(symbol)
                if success:
                    return True, f"Adding {symbol}. Data download and model training started."
                else:
                    return False, message
            else:
                return True, f"{symbol} is already available."
                
        except Exception as e:
            return False, f"Error adding symbol {symbol}: {str(e)}"
    
    def predict_multiple_days(self, symbol, feature, days=3, model_type='lstm'):
        """Predict multiple days for a specific feature"""
        try:
            if symbol not in self.symbols:
                return None
                
            if model_type == 'lstm':
                if symbol not in self.models or feature not in self.models[symbol]:
                    return None
                
                # Get the latest data
                if symbol not in self.norm_params or feature not in self.history_data[symbol]:
                    return None
                    
                # Get normalization parameters and history data
                norm_params = self.norm_params[symbol]
                last_sequence = self.history_data[symbol][feature][-60:]
                
                # Create input sequence with all features
                input_sequence = []
                for feat in self.features:
                    if feat in self.history_data[symbol]:
                        feat_data = self.history_data[symbol][feat][-60:]
                        # Normalize
                        if feat in norm_params:
                            mean, std = norm_params[feat]
                            feat_data = (feat_data - mean) / std
                        input_sequence.append(feat_data)
                
                # Stack features to create multi-feature input
                input_array = np.column_stack(input_sequence)
                input_array = input_array.reshape(1, 60, len(input_sequence))
                
                # Predict
                predictions = self.models[symbol][feature].predict(input_array, verbose=0)[0]
                
                # Denormalize
                mean, std = norm_params[feature]
                denorm_predictions = predictions * std + mean
                
                # Error correction - compare with last known price
                last_known_price = self.history_data[symbol][feature][-1]
                if abs(denorm_predictions[0] - last_known_price) / last_known_price > 0.05:
                    correction = last_known_price / denorm_predictions[0]
                    denorm_predictions = denorm_predictions * correction
                
                return denorm_predictions
                
            elif model_type == 'rf':
                if symbol not in self.rf_models or feature not in self.rf_models[symbol]:
                    return None
                
                # Get normalization parameters and history data
                if symbol not in self.norm_params or feature not in self.history_data[symbol]:
                    return None
                
                norm_params = self.norm_params[symbol]
                
                # Prepare input for RF model
                input_sequence = []
                for feat in self.features:
                    if feat in self.history_data[symbol]:
                        feat_data = self.history_data[symbol][feat][-60:]
                        # Normalize
                        if feat in norm_params:
                            mean, std = norm_params[feat]
                            feat_data = (feat_data - mean) / std
                        input_sequence.append(feat_data)
                
                # Stack features to create multi-feature input
                input_array = np.column_stack(input_sequence)
                input_array = input_array.reshape(1, -1)
                
                # Predict with RF model
                predictions = self.rf_models[symbol][feature].predict(input_array)[0]
                
                # Denormalize
                mean, std = norm_params[feature]
                denorm_predictions = predictions * std + mean
                
                # Error correction
                last_known_price = self.history_data[symbol][feature][-1]
                if abs(denorm_predictions[0] - last_known_price) / last_known_price > 0.05:
                    correction = last_known_price / denorm_predictions[0]
                    denorm_predictions = denorm_predictions * correction
                
                return denorm_predictions
                
            return None
            
        except Exception as e:
            print(f"Prediction error ({model_type}): {str(e)}")
            return None
    
    def get_plot_data(self, symbol, target_feature=None, target_date=None):
        """Generate plot data including predictions for all features"""
        if symbol not in self.symbols:
            return None
            
        features_to_plot = [target_feature] if target_feature else self.features
        result = {'features': {}}
        
        # Get dates for x-axis
        current_date = datetime.now()
        dates = [
            (current_date - timedelta(days=30-i)).strftime('%Y-%m-%d')
            for i in range(30)
        ]
        
        # Add future dates
        future_dates = [
            (current_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            for i in range(3)
        ]
        
        result['dates'] = dates + future_dates
        
        # Add history and predictions for each requested feature
        for feature in features_to_plot:
            if feature not in self.history_data[symbol]:
                continue
                
            # Get history data (last 30 days)
            history = self.history_data[symbol][feature][-30:].tolist()
            
            # Get predictions for both models
            lstm_predictions = self.predict_multiple_days(symbol, feature, days=3, model_type='lstm')
            rf_predictions = self.predict_multiple_days(symbol, feature, days=3, model_type='rf')
            
            # Determine which day to highlight (if any)
            highlight_index = None
            if target_date:
                try:
                    target_date_obj = self._parse_date(target_date)
                    for i, date_str in enumerate(future_dates):
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        if date_obj.date() == target_date_obj.date():
                            highlight_index = 30 + i  # 30 days of history + index in future
                            break
                except:
                    pass
            
            result['features'][feature] = {
                'history': history,
                'lstm_predictions': lstm_predictions.tolist() if lstm_predictions is not None else None,
                'rf_predictions': rf_predictions.tolist() if rf_predictions is not None else None,
                'highlight_index': highlight_index
            }
        
        return result
    
    def _parse_date(self, date_string):
        """Parse date string into datetime object"""
        try:
            # Try various formats
            today = datetime.now()
            
            # Handle relative dates
            if date_string.lower() == 'tomorrow':
                return today + timedelta(days=1)
            elif date_string.lower() == 'day after tomorrow':
                return today + timedelta(days=2)
            elif date_string.lower() == 'next day':
                return today + timedelta(days=1)
            elif date_string.lower() in ['in 2 days', 'in two days']:
                return today + timedelta(days=2)
            elif date_string.lower() in ['in 3 days', 'in three days']:
                return today + timedelta(days=3)
                
            # Try to parse as explicit date
            for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d-%m-%Y', '%d/%m/%Y'):
                try:
                    return datetime.strptime(date_string, fmt)
                except:
                    pass
                    
            # If all parsing attempts fail
            return None
        except:
            return None
    
    def process_request(self, request):
        """Process natural language request and extract symbol, feature, and date"""
        try:
            req = request.lower().strip()
            
            # Extract symbol
            symbol = None
            for sym in self.symbols:
                if sym.lower() in req:
                    symbol = sym
                    break
                    
            if not symbol:
                # Try to extract company names
                company_mapping = {
                    'apple': 'AAPL',
                    'microsoft': 'MSFT'
                }
                
                for company, sym in company_mapping.items():
                    if company in req:
                        symbol = sym
                        break
            
            if not symbol:
                return json.dumps({'error': 'Please specify a valid company/symbol'})
            
            # Extract feature
            feature = None
            feature_keywords = {
                'close': 'Close',
                'closing': 'Close',
                'high': 'High',
                'highest': 'High',
                'low': 'Low',
                'lowest': 'Low',
                'open': 'Open',
                'opening': 'Open',
                'volume': 'Volume'
            }
            
            for keyword, feat in feature_keywords.items():
                if keyword in req:
                    feature = feat
                    break
            
            # Extract date
            date_keywords = [
                'tomorrow', 'day after tomorrow', 'next day',
                'in 2 days', 'in two days', 'in 3 days', 'in three days'
            ]
            
            target_date = None
            for date_kw in date_keywords:
                if date_kw in req:
                    target_date = date_kw
                    break
            
            # Get predictions and plot data
            features_to_predict = [feature] if feature else self.features
            predictions = {}
            
            for feat in features_to_predict:
                lstm_pred = self.predict_multiple_days(symbol, feat)
                rf_pred = self.predict_multiple_days(symbol, feat, model_type='rf')
                
                if lstm_pred is not None or rf_pred is not None:
                    predictions[feat] = {
                        'lstm': lstm_pred.tolist() if lstm_pred is not None else None,
                        'rf': rf_pred.tolist() if rf_pred is not None else None
                    }
            
            if not predictions:
                return json.dumps({'error': 'Prediction failed'})
                
            plot_data = self.get_plot_data(symbol, feature, target_date)
            
            # Determine which day to report in message
            day_idx = 0
            if target_date:
                date_obj = self._parse_date(target_date)
                today = datetime.now()
                days_diff = (date_obj.date() - today.date()).days
                if 0 < days_diff <= 3:
                    day_idx = days_diff - 1
            
            # Build prediction message
            message = f"{symbol} prediction results:\n"
            
            for feat, pred in predictions.items():
                lstm_val = pred['lstm'][day_idx] if pred['lstm'] else None
                rf_val = pred['rf'][day_idx] if pred['rf'] else None
                
                if lstm_val is not None or rf_val is not None:
                    message += f"\n{feat}:\n"
                    if lstm_val is not None:
                        message += f"LSTM model: ${lstm_val:.2f}\n"
                    if rf_val is not None:
                        message += f"Random Forest model: ${rf_val:.2f}\n"
                
            return json.dumps({
                'symbol': symbol,
                'feature': feature,
                'target_date': target_date,
                'predictions': predictions,
                'message': message,
                'plot_data': plot_data
            })
            
        except Exception as e:
            return json.dumps({'error': str(e)})
    

oracle = OracleServer()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        query = data.get('query', '').lower().strip()
        
        # Process the natural language query
        result_json = oracle.process_request(query)
        result = json.loads(result_json)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/update_data', methods=['POST'])
def update_data():
    """API endpoint to update data and models"""
    try:
        # Get specific symbol if provided
        data = request.json
        symbol = data.get('symbol') if data else None
        
        # Asynchronously execute update
        success, message = oracle.async_update_data(symbol)
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/update_status', methods=['GET'])
def update_status():
    """API endpoint to get update status"""
    status = oracle.get_update_status()
    return jsonify(status)

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """API endpoint to get available symbols"""
    symbols = oracle.get_available_symbols()
    return jsonify({
        'symbols': symbols
    })

@app.route('/add_symbol', methods=['POST'])
def add_symbol():
    """API endpoint to add a new symbol"""
    try:
        data = request.json
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({
                'success': False,
                'message': 'No symbol provided'
            })
        
        success, message = oracle.add_new_symbol(symbol)
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    oracle.load_resources()
    app.run(host='localhost', port=5001, debug=True)