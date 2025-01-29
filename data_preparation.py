import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_stock_data(symbol, period='2y'):
    """
    Download stock data using yfinance
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

def prepare_data(df, sequence_length=60):
    """
    Prepare data for neural network training
    """
    data = df['Close'].values
    mean = np.mean(data)
    std = np.std(data)
    data_normalized = (data - mean) / std
    
    X, y = [], []
    for i in range(len(data_normalized) - sequence_length):
        X.append(data_normalized[i:(i + sequence_length)])
        y.append(data_normalized[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, mean, std

def main():
    # Define folder for storing files
    output_folder = "project_files"
    os.makedirs(output_folder, exist_ok=True)
    
    symbols = ['AAPL', 'MSFT']
    for symbol in symbols:
        print(f"Downloading data for {symbol}...")
        df = download_stock_data(symbol)
        
        # Save raw data
        raw_data_path = os.path.join(output_folder, f'data_{symbol}.csv')
        df.to_csv(raw_data_path)
        print(f"Raw data saved to: {os.path.abspath(raw_data_path)}")
        
        # Prepare data
        X, y, mean, std = prepare_data(df)
        
        # Save processed data
        X_path = os.path.join(output_folder, f'X_{symbol}.npy')
        y_path = os.path.join(output_folder, f'y_{symbol}.npy')
        norm_params_path = os.path.join(output_folder, f'norm_params_{symbol}.npy')
        
        np.save(X_path, X)
        print(f"Processed feature data saved to: {os.path.abspath(X_path)}")
        np.save(y_path, y)
        print(f"Processed target data saved to: {os.path.abspath(y_path)}")
        np.save(norm_params_path, np.array([mean, std]))
        print(f"Normalization parameters saved to: {os.path.abspath(norm_params_path)}")

if __name__ == "__main__":
    main()
