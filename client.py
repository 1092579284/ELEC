import socket
import json
import matplotlib.pyplot as plt
from datetime import datetime
import time

def plot_stock_data(plot_data):
    """自动绘制股票走势图"""
    plt.figure(figsize=(12, 6))
    
    # 合并历史与预测数据
    full_data = plot_data['history'] + plot_data['predictions']
    
    # 主趋势线
    plt.plot(
        plot_data['dates'],
        full_data,
        color='#2c7fb8',
        linewidth=2,
        zorder=1
    )
    
    # 预测部分高亮
    plt.plot(
        plot_data['dates'][-1:],  # 修改为只显示1天预测
        plot_data['predictions'],
        color='#ff7f0e',
        linestyle='--',
        marker='o',
        markersize=8,
        linewidth=2,
        zorder=2,
        label='Predicted'
    )
    
    # 格式设置
    plt.title(f"Stock Price Trend - {plot_data['symbol']}", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 标注预测开始点
    plt.axvline(
        x=plot_data['dates'][29],
        color='red',
        linestyle=':',
        linewidth=1,
        label='Prediction Start'
    )
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    host = 'localhost'
    port = 5001
    
    print("Stock Prediction Client (Enter 'exit' to quit)")
    print("Supported symbols: AAPL, MSFT")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # 创建新的socket连接
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # 设置超时
                s.settimeout(10)
                
                # 尝试连接，最多重试3次
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        s.connect((host, port))
                        break
                    except socket.error as e:
                        if attempt < max_retries - 1:
                            print(f"连接失败，正在重试 ({attempt+1}/{max_retries})...")
                            time.sleep(2)  # 等待2秒后重试
                        else:
                            raise e
                
                # 发送请求
                s.sendall(user_input.encode('utf-8'))
                
                # 接收响应
                response_data = b""
                while True:
                    try:
                        chunk = s.recv(4096)
                        if not chunk:
                            break
                        response_data += chunk
                        # 尝试解析JSON，如果成功则退出循环
                        try:
                            json.loads(response_data.decode('utf-8'))
                            break
                        except json.JSONDecodeError:
                            # 数据不完整，继续接收
                            continue
                    except socket.timeout:
                        print("接收数据超时")
                        break
                
                # 处理响应
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    
                    if 'error' in response:
                        print(f"Error: {response['error']}")
                        continue
                    
                    # 显示文本预测
                    print(f"\nOracle: {response['message']}")
                    
                    # 自动显示图表
                    if 'plot_data' in response:
                        plot_data = response['plot_data']
                        plot_data['symbol'] = response['symbol']
                        plot_stock_data(plot_data)
                    else:
                        print("No chart data available")
                        
                except json.JSONDecodeError:
                    print("Invalid server response")
                    print(f"Raw response: {response_data.decode('utf-8', errors='replace')}")
                
        except socket.error as e:
            print(f"Connection error: {str(e)}")
            print("请确保服务器已启动并正在运行")
            time.sleep(2)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
