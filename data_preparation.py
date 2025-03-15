import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def download_stock_data(symbol, period='3y'):
    """下载股票数据"""
    # 检查是否已有下载好的NPY文件
    npy_path = os.path.join("project_files", f'{symbol}_latest.npy')
    if os.path.exists(npy_path):
        print(f"使用已下载的数据: {npy_path}")
        # 加载NPY数据
        close_values = np.load(npy_path)
        
        # 删除临时NPY文件
        os.remove(npy_path)
        
        # 创建一个简单的DataFrame，只包含Close列
        dates = [datetime.now() - timedelta(days=len(close_values)-i) for i in range(len(close_values))]
        df = pd.DataFrame({
            'Close': close_values
        }, index=dates)
        
        return df
    else:
        # 如果没有找到NPY文件，则直接下载
        print(f"直接下载 {symbol} 数据...")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df

def prepare_data(df, sequence_length=60, forecast_days=1):
    """生成多目标序列数据"""
    data = df['Close'].values
    mean = np.mean(data)
    std = np.std(data)
    data_normalized = (data - mean) / std
    
    X, y = [], []
    for i in range(len(data_normalized) - sequence_length - forecast_days + 1):
        X.append(data_normalized[i:(i + sequence_length)])
        y.append(data_normalized[i + sequence_length : i + sequence_length + forecast_days])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, mean, std

def main():
    output_folder = "project_files"
    os.makedirs(output_folder, exist_ok=True)
    
    symbols = ['AAPL', 'MSFT']
    for symbol in symbols:
        print(f"处理 {symbol} 数据...")
        try:
            # 下载或使用已下载的数据
            df = download_stock_data(symbol)
            
            # 保存原始数据
            raw_path = os.path.join(output_folder, f'full_history_{symbol}.npy')
            np.save(raw_path, df['Close'].values)
            print(f"保存完整历史数据到 {raw_path}")
            
            # 准备训练数据
            X, y, mean, std = prepare_data(df)
            
            # 保存处理后的数据
            np.save(os.path.join(output_folder, f'X_{symbol}.npy'), X)
            np.save(os.path.join(output_folder, f'y_{symbol}.npy'), y)
            np.save(os.path.join(output_folder, f'norm_params_{symbol}.npy'), 
                   np.array([mean, std]))
            
            print(f"成功处理 {symbol} 数据")
            
        except Exception as e:
            print(f"处理 {symbol} 时出错: {str(e)}")

if __name__ == "__main__":
    main()