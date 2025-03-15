import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

def create_rf_model():
    """创建随机森林回归模型"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    return model

def train_rf_model(symbol, output_folder, n_estimators=100, max_depth=10):
    """训练随机森林模型"""
    X_path = os.path.join(output_folder, f'X_{symbol}.npy')
    y_path = os.path.join(output_folder, f'y_{symbol}.npy')
    model_path = os.path.join(output_folder, f'model_rf_{symbol}.joblib')

    # 加载数据
    X = np.load(X_path)
    y = np.load(y_path)
    
    # 将3D数据重塑为2D (samples, features)
    X_reshaped = X.reshape(X.shape[0], -1)
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42
    )
    
    # 检查y的形状，如果需要压平
    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    # 创建模型
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # 训练模型
    print(f"开始训练随机森林模型 {symbol}...")
    model.fit(X_train, y_train)
    
    # 评估模型
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"训练集 R2 得分: {train_score:.4f}")
    print(f"测试集 R2 得分: {test_score:.4f}")
    
    # 保存模型
    joblib.dump(model, model_path)
    print(f"模型已保存到 {model_path}")
    
    return model, train_score, test_score

def predict_rf(model, X_input, norm_params):
    """使用随机森林模型进行预测"""
    mean, std = norm_params
    
    # 归一化输入
    X_norm = (X_input - mean) / std
    X_reshaped = X_norm.reshape(1, -1)
    
    # 预测
    prediction = model.predict(X_reshaped)[0]
    
    return prediction

def main():
    output_folder = "project_files"
    symbols = ['AAPL', 'MSFT']
    
    for symbol in symbols:
        print(f"\n训练 {symbol} 随机森林模型...")
        if not os.path.exists(os.path.join(output_folder, f'X_{symbol}.npy')):
            print(f"未找到 {symbol} 的数据")
            continue
        train_rf_model(symbol, output_folder)

if __name__ == "__main__":
    main() 