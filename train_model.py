import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

def create_model(sequence_length):
    """Create an LSTM model for time series forecasting."""
    model = Sequential([
        LSTM(units=50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(units=50, activation='relu'),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(symbol, output_folder, epochs=50, batch_size=32):
    """Train model using prepared data."""
    X_path = os.path.join(output_folder, f'X_{symbol}.npy')
    y_path = os.path.join(output_folder, f'y_{symbol}.npy')
    model_path = os.path.join(output_folder, f'model_{symbol}.keras')

    X = np.load(X_path)
    y = np.load(y_path)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = create_model(X.shape[1])
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    model.save(model_path)
    print(f"Model saved to: {os.path.abspath(model_path)}")
    return history

def main():
    output_folder = "project_files"
    os.makedirs(output_folder, exist_ok=True)

    symbols = ['AAPL', 'MSFT']
    for symbol in symbols:
        print(f"\nTraining model for {symbol}...")
        X_path = os.path.join(output_folder, f'X_{symbol}.npy')
        if not os.path.exists(X_path):
            print(f"Prepared data for {symbol} not found in {os.path.abspath(X_path)}. Run data preparation script first.")
            continue

        train_model(symbol, output_folder)

if __name__ == "__main__":
    main()
