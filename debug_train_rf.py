import os
import numpy as np
import traceback
import sys

def check_files_exist():
    """检查必要的数据文件是否存在"""
    output_folder = "project_files"
    symbols = ['AAPL', 'MSFT']
    all_exist = True
    
    print("检查文件存在性：")
    for symbol in symbols:
        X_path = os.path.join(output_folder, f'X_{symbol}.npy')
        y_path = os.path.join(output_folder, f'y_{symbol}.npy')
        
        X_exists = os.path.exists(X_path)
        y_exists = os.path.exists(y_path)
        
        print(f"{symbol} 数据文件:")
        print(f"  - X_{symbol}.npy: {'存在' if X_exists else '不存在'}")
        print(f"  - y_{symbol}.npy: {'存在' if y_exists else '不存在'}")
        
        if not X_exists or not y_exists:
            all_exist = False
    
    return all_exist

def check_data_shapes():
    """检查数据的形状是否正确"""
    output_folder = "project_files"
    symbols = ['AAPL', 'MSFT']
    
    print("\n检查数据形状：")
    for symbol in symbols:
        try:
            X_path = os.path.join(output_folder, f'X_{symbol}.npy')
            y_path = os.path.join(output_folder, f'y_{symbol}.npy')
            
            if os.path.exists(X_path) and os.path.exists(y_path):
                X = np.load(X_path)
                y = np.load(y_path)
                
                print(f"{symbol} 数据形状:")
                print(f"  - X 形状: {X.shape}")
                print(f"  - y 形状: {y.shape}")
                
                # 确保X可以被重塑为2D
                try:
                    X_reshaped = X.reshape(X.shape[0], -1)
                    print(f"  - X 重塑后形状: {X_reshaped.shape}")
                except Exception as e:
                    print(f"  - X 重塑错误: {str(e)}")
        except Exception as e:
            print(f"{symbol} 数据加载错误: {str(e)}")

def check_directories():
    """检查必要的目录是否存在并可写入"""
    output_folder = "project_files"
    
    print("\n检查目录权限：")
    if not os.path.exists(output_folder):
        print(f"目录 {output_folder} 不存在")
        try:
            os.makedirs(output_folder)
            print(f"已创建目录 {output_folder}")
        except Exception as e:
            print(f"创建目录失败: {str(e)}")
    else:
        print(f"目录 {output_folder} 存在")
        
    # 检查写入权限
    try:
        test_file = os.path.join(output_folder, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("测试写入权限")
        os.remove(test_file)
        print(f"目录 {output_folder} 可写入")
    except Exception as e:
        print(f"目录写入测试失败: {str(e)}")

def try_training():
    """尝试训练一个小型随机森林模型"""
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    
    print("\n尝试训练小型随机森林模型：")
    
    try:
        # 创建一些简单的测试数据
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        # 创建一个小型的随机森林模型
        model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        
        # 训练模型
        model.fit(X, y)
        
        # 保存模型
        test_model_path = "project_files/test_rf_model.joblib"
        joblib.dump(model, test_model_path)
        
        # 加载模型
        loaded_model = joblib.load(test_model_path)
        
        # 预测
        pred = loaded_model.predict(X[:1])
        
        print("训练、保存、加载和预测成功")
        
        # 清理
        os.remove(test_model_path)
    except Exception as e:
        print(f"随机森林测试模型训练失败: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== 随机森林模型训练调试 ===")
    files_exist = check_files_exist()
    check_data_shapes()
    check_directories()
    try_training()
    
    print("\n调试信息:")
    print(f"Python 版本: {sys.version}")
    print(f"numpy 版本: {np.__version__}")
    
    try:
        import sklearn
        print(f"scikit-learn 版本: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn 未安装")
    
    try:
        import joblib
        print(f"joblib 版本: {joblib.__version__}")
    except ImportError:
        print("joblib 未安装")
        
    print("\n如果上述测试都通过，请检查train_model_rf.py的具体错误") 