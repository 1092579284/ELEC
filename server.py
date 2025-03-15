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
CORS(app)  # 允许跨域请求

class OracleServer:
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.models = {}
        self.rf_models = {}  # 随机森林模型
        self.norm_params = {}
        self.history_data = {}
        self.symbols = ['AAPL', 'MSFT']
        self.output_folder = "project_files"
        self.updating = False
        self.update_status = {"step": "", "message": "", "progress": 0}
        self.load_resources()
        
    def load_resources(self):
        """加载所有必需资源"""
        for symbol in self.symbols:
            try:
                # 加载LSTM模型
                model_path = os.path.join(self.output_folder, f'model_{symbol}.keras')
                self.models[symbol] = load_model(model_path)
                
                # 加载随机森林模型
                rf_model_path = os.path.join(self.output_folder, f'model_rf_{symbol}.joblib')
                if os.path.exists(rf_model_path):
                    self.rf_models[symbol] = joblib.load(rf_model_path)
                    print(f"Loaded RF model for {symbol}")
                else:
                    print(f"RF model for {symbol} not found")
                
                # 加载归一化参数
                norm_path = os.path.join(self.output_folder, f'norm_params_{symbol}.npy')
                self.norm_params[symbol] = np.load(norm_path)
                
                # 加载历史数据
                history_path = os.path.join(self.output_folder, f'full_history_{symbol}.npy')
                self.history_data[symbol] = np.load(history_path)
                
                print(f"Loaded resources for {symbol}")
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
    
    def download_latest_data(self):
        """下载最新的股票数据"""
        self.update_status = {"step": "download", "message": "正在下载最新股票数据...", "progress": 20}
        
        # 确保输出目录存在
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        success = True
        for symbol in self.symbols:
            try:
                # 直接使用yfinance下载最新数据
                print(f"下载 {symbol} 最新数据...")
                stock = yf.Ticker(symbol)
                df = stock.history(period='3y')  # 下载3年数据
                
                # 直接保存为NPY格式
                data_path = os.path.join(self.output_folder, f'{symbol}_latest.npy')
                np.save(data_path, df['Close'].values)
                
                # 同时保存原始数据供后续处理
                raw_path = os.path.join(self.output_folder, f'full_history_{symbol}.npy')
                np.save(raw_path, df['Close'].values)
                
                print(f"已保存 {symbol} 数据到 {data_path}")
            except Exception as e:
                print(f"下载 {symbol} 数据出错: {str(e)}")
                success = False
        
        return success
    
    def update_data(self):
        """更新数据和模型的过程"""
        if self.updating:
            return False, "数据更新已在进行中，请稍后再试"
        
        self.updating = True
        self.update_status = {"step": "start", "message": "开始更新数据...", "progress": 0}
        
        try:
            # 第1步：下载最新数据
            if not self.download_latest_data():
                self.update_status = {"step": "error", "message": "下载数据失败", "progress": 0}
                self.updating = False
                return False, "下载最新数据失败"
            
            # 第2步：数据预处理
            self.update_status = {"step": "prepare", "message": "正在进行数据预处理...", "progress": 40}
            print("开始数据预处理...")
            result1 = subprocess.run(['python', 'data_preparation.py'], 
                                    capture_output=True, text=True, check=False)
            
            if result1.returncode != 0:
                error_msg = f"数据预处理失败: {result1.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "数据预处理失败", "progress": 40}
                self.updating = False
                return False, error_msg
            
            # 第3步：训练LSTM模型
            self.update_status = {"step": "lstm", "message": "正在训练LSTM模型...", "progress": 60}
            print("开始训练LSTM模型...")
            result2 = subprocess.run(['python', 'train_model.py'], 
                                    capture_output=True, text=True, check=False)
            
            if result2.returncode != 0:
                error_msg = f"LSTM模型训练失败: {result2.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "LSTM模型训练失败", "progress": 60}
                self.updating = False
                return False, error_msg
            
            # 第4步：训练随机森林模型
            self.update_status = {"step": "rf", "message": "正在训练随机森林模型...", "progress": 80}
            print("开始训练随机森林模型...")
            result3 = subprocess.run(['python', 'train_model_rf.py'], 
                                    capture_output=True, text=True, check=False)
            
            if result3.returncode != 0:
                error_msg = f"随机森林模型训练失败: {result3.stderr}"
                print(error_msg)
                self.update_status = {"step": "error", "message": "随机森林模型训练失败", "progress": 80}
                self.updating = False
                return False, error_msg
            
            # 第5步：重新加载资源
            self.update_status = {"step": "reload", "message": "正在重新加载模型和数据...", "progress": 95}
            print("重新加载模型和数据...")
            self.load_resources()
            
            # 完成更新
            self.update_status = {"step": "complete", "message": "数据和模型已成功更新", "progress": 100}
            self.updating = False
            return True, "数据和模型已成功更新"
        
        except Exception as e:
            self.updating = False
            error_msg = f"更新过程中发生错误: {str(e)}"
            self.update_status = {"step": "error", "message": error_msg, "progress": 0}
            print(error_msg)
            return False, error_msg
    
    def async_update_data(self):
        """异步执行数据更新"""
        threading.Thread(target=self._async_update_worker).start()
        return True, "数据更新已开始，请稍后..."
    
    def _async_update_worker(self):
        """异步更新工作线程"""
        self.update_data()
    
    def get_update_status(self):
        """获取当前更新状态"""
        return self.update_status
    
    def recursive_predict(self, symbol, days=1):
        """使用LSTM模型预测单日价格"""
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
            print(f"LSTM Prediction error: {str(e)}")
            return None
    
    def rf_predict(self, symbol):
        """使用随机森林模型预测单日价格"""
        try:
            if symbol not in self.rf_models:
                print(f"No RF model for {symbol}")
                return None
                
            # 获取最新数据
            mean, std = self.norm_params[symbol]
            last_60_days = self.history_data[symbol][-60:]
            
            # 将3D数据重塑为2D以适应RF模型
            normalized_seq = (last_60_days - mean) / std
            flattened_seq = normalized_seq.reshape(1, -1)
            
            # 预测
            pred = self.rf_models[symbol].predict(flattened_seq)[0]
            
            # 反归一化
            prediction = pred * std + mean
            
            # 误差修正
            last_known_price = self.history_data[symbol][-1]
            if abs(prediction - last_known_price) / last_known_price > 0.05:
                correction = last_known_price / prediction
                prediction = prediction * correction
                
            return prediction
            
        except Exception as e:
            print(f"RF Prediction error: {str(e)}")
            return None

    def apply_error_correction(self, predictions, last_price):
        """误差修正策略"""
        if abs(predictions[0] - last_price) / last_price > 0.05:
            correction = last_price / predictions[0]
            return predictions * correction
        return predictions
    
    def get_plot_data(self, symbol):
        """生成图表数据，包含两种算法的预测结果"""
        history = self.history_data[symbol][-30:].tolist()
        lstm_prediction = self.recursive_predict(symbol)
        rf_prediction = self.rf_predict(symbol)
        
        dates = [
            (datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d')
            for i in range(30)
        ]
        
        pred_date = [(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')]
        
        predictions = {
            'lstm': lstm_prediction if lstm_prediction is not None else None,
            'rf': rf_prediction if rf_prediction is not None else None
        }
        
        return {
            'history': history,
            'predictions': predictions,
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
            
            # 获取两种算法的预测结果
            lstm_pred = self.recursive_predict(symbol)
            rf_pred = self.rf_predict(symbol)
            
            if lstm_pred is None and rf_pred is None:
                return json.dumps({'error': 'Prediction failed'})
                
            plot_data = self.get_plot_data(symbol)
            
            # 构建预测消息
            message = f"{symbol} 预测结果:\n"
            if lstm_pred is not None:
                message += f"LSTM模型: ${lstm_pred:.2f}\n"
            if rf_pred is not None:
                message += f"随机森林模型: ${rf_pred:.2f}"
                
            return json.dumps({
                'symbol': symbol,
                'lstm_prediction': lstm_pred.tolist() if lstm_pred is not None else None,
                'rf_prediction': rf_pred.tolist() if rf_pred is not None else None,
                'message': message,
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
        
        # 获取两种算法的预测
        lstm_prediction = oracle.recursive_predict(symbol)
        rf_prediction = oracle.rf_predict(symbol)
        
        if lstm_prediction is None and rf_prediction is None:
            return jsonify({'error': '预测失败'})
        
        plot_data = oracle.get_plot_data(symbol)
        
        # 构建预测消息
        message = f"{symbol} 预测结果:\n"
        if lstm_prediction is not None:
            message += f"LSTM模型: ${lstm_prediction:.2f}\n"
        if rf_prediction is not None:
            message += f"随机森林模型: ${rf_prediction:.2f}"
        
        return jsonify({
            'symbol': symbol,
            'lstm_prediction': float(lstm_prediction) if lstm_prediction is not None else None,
            'rf_prediction': float(rf_prediction) if rf_prediction is not None else None,
            'message': message,
            'plot_data': plot_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/update_data', methods=['POST'])
def update_data():
    """更新数据和模型的API端点"""
    try:
        # 异步执行更新，不阻塞响应
        success, message = oracle.async_update_data()
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
    """获取更新状态的API端点"""
    status = oracle.get_update_status()
    return jsonify(status)

if __name__ == "__main__":
    oracle.load_resources()
    app.run(host='localhost', port=5001, debug=True)