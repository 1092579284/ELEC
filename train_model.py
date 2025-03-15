import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def create_model(input_shape):
    """创建单步预测LSTM模型"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)  # 输出1天预测
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss=tf.keras.losses.Huber(),
                 metrics=['mae', 'mse'])
    
    return model


def train_model(symbol, output_folder, epochs=100, batch_size=64):
    """训练模型"""
    X_path = os.path.join(output_folder, f'X_{symbol}.npy')
    y_path = os.path.join(output_folder, f'y_{symbol}.npy')
    model_path = os.path.join(output_folder, f'model_{symbol}.keras')

    # 加载数据
    X = np.load(X_path)
    y = np.load(y_path)
    
    # 数据分割
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 创建模型
    model = create_model((X.shape[1], 1))
    
    # 回调函数
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # 训练
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 保存最终模型
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return history

def main():
    output_folder = "project_files"
    symbols = ['AAPL', 'MSFT']
    
    for symbol in symbols:
        print(f"\nTraining {symbol}...")
        if not os.path.exists(os.path.join(output_folder, f'X_{symbol}.npy')):
            print(f"Data not found for {symbol}")
            continue
        train_model(symbol, output_folder)

if __name__ == "__main__":
    main()