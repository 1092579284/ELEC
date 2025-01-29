import socket
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta

class OracleServer:
    def __init__(self, host='localhost', port=5001):  # Changed port to 5001
        self.host = host
        self.port = port
        self.models = {}
        self.norm_params = {}
        self.symbols = ['AAPL', 'MSFT']
        self.load_models()
        
    def load_models(self):
        """Load trained models and normalization parameters"""
        for symbol in self.symbols:
            try:
                self.models[symbol] = load_model(f'model_{symbol}.keras')  # Changed to .keras extension
                self.norm_params[symbol] = np.load(f'norm_params_{symbol}.npy')
                print(f"Loaded model for {symbol}")
            except Exception as e:
                print(f"Error loading model for {symbol}: {e}")
    
    def predict_next_day(self, symbol):
        """Make prediction for next day's stock price"""
        try:
            if symbol not in self.models:
                return f"Model for {symbol} is not loaded."
                
            # Get recent data
            stock = yf.Ticker(symbol)
            df = stock.history(period='70d')  # Get enough data for sequence
            close_prices = df['Close'].values
            
            # Normalize data
            mean, std = self.norm_params[symbol]
            normalized_data = (close_prices - mean) / std
            
            # Prepare sequence for prediction
            sequence = normalized_data[-60:].reshape(1, 60, 1)
            
            # Make prediction
            pred_normalized = self.models[symbol].predict(sequence, verbose=0)[0][0]
            
            # Denormalize prediction
            prediction = (pred_normalized * std) + mean
            
            return prediction
            
        except Exception as e:
            return f"Error making prediction: {e}"
    
    def process_request(self, request):
        """Process client request and return response"""
        try:
            request = request.lower()
            
            # Parse stock symbol from request
            if 'apple' in request or 'aapl' in request:
                symbol = 'AAPL'
            elif 'microsoft' in request or 'msft' in request:
                symbol = 'MSFT'
            else:
                return "I can only predict Apple (AAPL) or Microsoft (MSFT) stock prices."
            
            prediction = self.predict_next_day(symbol)
            
            if isinstance(prediction, float):
                response = f"Based on my analysis, {symbol}'s stock price tomorrow will be approximately ${prediction:.2f}"
            else:
                response = str(prediction)
                
            return response
            
        except Exception as e:
            return f"Error processing request: {e}"
    
    def start(self):
        """Start the server"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow socket reuse
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"Oracle server listening on {self.host}:{self.port}")
            
            while True:
                client_socket, address = server_socket.accept()
                print(f"Connection from {address}")
                
                try:
                    while True:  # Keep connection alive for multiple requests
                        request = client_socket.recv(1024).decode('utf-8')
                        if not request or request.lower() in ['exit', 'quit']:
                            print(f"Connection from {address} closed.")
                            break
                        
                        # Process request and send response
                        response = self.process_request(request)
                        client_socket.send(response.encode('utf-8'))
                        
                except Exception as e:
                    print(f"Error handling client: {e}")
                finally:
                    client_socket.close()
                    
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            server_socket.close()


if __name__ == "__main__":
    server = OracleServer()
    server.start()
