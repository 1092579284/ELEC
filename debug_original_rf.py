import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import traceback
import sys

def train_rf_model_debug(symbol, output_folder, n_estimators=100, max_depth=10):
    """训练随机森林模型，并详细记录调试信息"""
    try:
        print(f"开始加载数据 {symbol}...")
        X_path = os.path.join(output_folder, f'X_{symbol}.npy')
        y_path = os.path.join(output_folder, f'y_{symbol}.npy')
        model_path = os.path.join(output_folder, f'model_rf_{symbol}.joblib')

        # 加载数据
        X = np.load(X_path)
        print(f"加载 X 成功，形状: {X.shape}")
        y = np.load(y_path)
        print(f"加载 y 成功，形状: {y.shape}")
        
        # 将3D数据重塑为2D (samples, features)
        print("重塑 X 数据...")
        X_reshaped = X.reshape(X.shape[0], -1)
        print(f"重塑后 X 形状: {X_reshaped.shape}")
        
        # 数据分割
        print("分割训练集和测试集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y, test_size=0.2, random_state=42
        )
        print(f"训练集 X 形状: {X_train.shape}")
        print(f"测试集 X 形状: {X_test.shape}")
        print(f"训练集 y 形状: {y_train.shape}")
        print(f"测试集 y 形状: {y_test.shape}")

        # 创建模型
        print("创建随机森林模型...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # 训练模型
        print(f"开始训练随机森林模型 {symbol}...")
        # 查看训练数据的详细信息
        print(f"训练集 X 样例: \n{X_train[:2]}")
        print(f"训练集 y 样例: \n{y_train[:2]}")
        
        # 检查y的形状，如果需要压平
        if len(y_train.shape) > 1 and y_train.shape[1] == 1:
            print("压平 y_train...")
            y_train = y_train.ravel()
            y_test = y_test.ravel()
        
        model.fit(X_train, y_train)
        
        # 评估模型
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"训练集 R² 得分: {train_score:.4f}")
        print(f"测试集 R² 得分: {test_score:.4f}")
        
        # 保存模型
        print(f"保存模型到 {model_path}...")
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
        
        return model, train_score, test_score
    
    except Exception as e:
        print(f"训练 {symbol} 随机森林模型时出错: {str(e)}")
        traceback.print_exc()
        return None, 0, 0

def main():
    output_folder = "project_files"
    symbols = ['AAPL', 'MSFT']
    
    for symbol in symbols:
        print(f"\n=== 调试训练 {symbol} 随机森林模型 ===")
        if not os.path.exists(os.path.join(output_folder, f'X_{symbol}.npy')):
            print(f"未找到 {symbol} 的数据")
            continue
        train_rf_model_debug(symbol, output_folder)

if __name__ == "__main__":
    print("系统信息:")
    print(f"Python 版本: {sys.version}")
    try:
        import sklearn
        print(f"scikit-learn 版本: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn 未安装")
    
    main() 