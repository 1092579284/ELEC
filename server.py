from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import os


app = Flask(__name__)
CORS(app)  # 允许跨域请求

class OracleServer:
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.models = {}
        self.norm_params = {}
        self.history_data = {}
        self.symbols = ['AAPL', 'MSFT']
        self.output_folder = "project_files"
        self.load_resources()
        
    def load_resources(self):
        """加载所有必需资源"""
        for symbol in self.symbols:
            try:
                # 加载模型
                model_path = os.path.join(self.output_folder, f'model_{symbol}.keras')
                self.models[symbol] = load_model(model_path)
                
                # 加载归一化参数
                norm_path = os.path.join(self.output_folder, f'norm_params_{symbol}.npy')
                self.norm_params[symbol] = np.load(norm_path)
                
                # 加载历史数据
                history_path = os.path.join(self.output_folder, f'full_history_{symbol}.npy')
                self.history_data[symbol] = np.load(history_path)
                
                print(f"Loaded resources for {symbol}")
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
    
    def recursive_predict(self, symbol, days=1):
        """预测单日价格"""
        try:
            # 获取最新数据
            mean, std = self.norm_params[symbol]
            last_60_days = self.history_data[symbol][-60:]
            normalized_seq = (last_60_days - mean) / std
            
            # 准备输入序列
            current_seq = normalized_seq.reshape(1, 60, 1)
            
            # 预测
            pred = self.models[symbol].predict(current_seq, verbose=0)[0][0]
            
            # 反归一化
            prediction = pred * std + mean
            
            # 误差修正
            last_known_price = self.history_data[symbol][-1]
            if abs(prediction - last_known_price) / last_known_price > 0.05:
                correction = last_known_price / prediction
                prediction = prediction * correction
                
            return prediction
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

    def apply_error_correction(self, predictions, last_price):
        """误差修正策略"""
        if abs(predictions[0] - last_price) / last_price > 0.05:
            correction = last_price / predictions[0]
            return predictions * correction
        return predictions
    
    def get_plot_data(self, symbol):
        """生成图表数据"""
        history = self.history_data[symbol][-30:].tolist()
        prediction = self.recursive_predict(symbol)
        
        dates = [
            (datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d')
            for i in range(30)
        ]
        
        pred_date = [(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')]
        
        return {
            'history': history,
            'predictions': [prediction],
            'dates': dates + pred_date
        }

    
    def process_request(self, request):
        """处理所有请求为1日预测"""
        try:
            req = request.lower().strip()
            symbol = None
            
            # 简化符号识别逻辑
            if any(kw in req for kw in ['aapl', 'apple']):
                symbol = 'AAPL'
            elif any(kw in req for kw in ['msft', 'microsoft']):
                symbol = 'MSFT'
            
            if not symbol:
                return json.dumps({'error': 'Unsupported symbol'})
            
            # 返回1日预测
            pred = self.recursive_predict(symbol)
            if pred is None:
                return json.dumps({'error': 'Prediction failed'})
                
            plot_data = self.get_plot_data(symbol)
            return json.dumps({
                'symbol': symbol,
                'prediction': pred.tolist(),
                'message': f"1-Day forecast for {symbol}: {pred[-1]:.2f} (Final day)",
                'plot_data': plot_data
            })
            
        except Exception as e:
            return json.dumps({'error': str(e)})
    
    def start(self):
        """启动服务器"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"Server listening on {self.host}:{self.port}")
            
            while True:
                conn, addr = s.accept()
                print(f"Connected by {addr}")
                try:
                    while True:
                        data = conn.recv(1024).decode('utf-8')
                        if not data or data.lower() in ['exit', 'quit']:
                            break
                            
                        response = self.process_request(data)
                        conn.sendall(response.encode('utf-8'))
                finally:
                    conn.close()
    


oracle = OracleServer()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        query = data.get('query', '').lower().strip()
        
        # 简化符号识别逻辑
        symbol = None
        if any(kw in query for kw in ['aapl', 'apple']):
            symbol = 'AAPL'
        elif any(kw in query for kw in ['msft', 'microsoft']):
            symbol = 'MSFT'
        
        if not symbol:
            return jsonify({'error': '不支持的股票代码'})
        
        # 获取预测
        prediction = oracle.recursive_predict(symbol)
        if prediction is None:
            return jsonify({'error': '预测失败'})
        
        plot_data = oracle.get_plot_data(symbol)
        
        return jsonify({
            'symbol': symbol,
            'prediction': float(prediction),
            'message': f"{symbol} 明日预测价格: ${prediction:.2f}",
            'plot_data': plot_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    oracle.load_resources()
    app.run(host='localhost', port=5001, debug=True)
