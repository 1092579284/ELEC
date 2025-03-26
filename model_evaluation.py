import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from tabulate import tabulate

def load_data(symbol, output_folder, feature):
    """Load data for a specific stock and feature"""
    symbol_dir = os.path.join(output_folder, symbol)
    X_path = os.path.join(symbol_dir, f'X_{symbol}.npy')
    y_path = os.path.join(symbol_dir, f'y_{symbol}_{feature}.npy')
    norm_params_path = os.path.join(symbol_dir, f'norm_params_{symbol}.npy')
    
    # Load data
    X = np.load(X_path)
    y = np.load(y_path)
    norm_params = np.load(norm_params_path, allow_pickle=True).item()
    
    # Data split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Reshape test data for Random Forest
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    return X_test, y_test, X_test_reshaped, norm_params

def load_models(symbol, output_folder, feature):
    """Load LSTM and Random Forest models"""
    symbol_dir = os.path.join(output_folder, symbol)
    lstm_model_path = os.path.join(symbol_dir, f'model_{symbol}_{feature}.keras')
    rf_model_path = os.path.join(symbol_dir, f'model_rf_{symbol}_{feature}.joblib')
    
    lstm_model = None
    rf_model = None
    
    if os.path.exists(lstm_model_path):
        try:
            lstm_model = load_model(lstm_model_path)
            print(f"LSTM model loaded: {lstm_model_path}")
        except Exception as e:
            print(f"Failed to load LSTM model: {str(e)}")
    
    if os.path.exists(rf_model_path):
        try:
            rf_model = joblib.load(rf_model_path)
            print(f"Random Forest model loaded: {rf_model_path}")
        except Exception as e:
            print(f"Failed to load Random Forest model: {str(e)}")
    
    return lstm_model, rf_model

def evaluate_models(X_test, y_test, X_test_reshaped, lstm_model, rf_model, norm_params, feature):
    """Evaluate model performance and return metrics"""
    results = {}
    mean, std = norm_params[feature]
    
    # If LSTM model exists
    if lstm_model is not None:
        # Predict
        lstm_preds = lstm_model.predict(X_test)
        
        # Denormalize predictions and ground truth
        lstm_preds_denorm = lstm_preds * std + mean
        y_test_denorm = y_test * std + mean
        
        # Calculate metrics (for each forecast day)
        lstm_metrics = {
            'day': [],
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        for day in range(y_test.shape[1]):
            mse = mean_squared_error(y_test_denorm[:, day], lstm_preds_denorm[:, day])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_denorm[:, day], lstm_preds_denorm[:, day])
            r2 = r2_score(y_test_denorm[:, day], lstm_preds_denorm[:, day])
            
            lstm_metrics['day'].append(day + 1)
            lstm_metrics['mse'].append(mse)
            lstm_metrics['rmse'].append(rmse)
            lstm_metrics['mae'].append(mae)
            lstm_metrics['r2'].append(r2)
        
        results['lstm'] = {
            'predictions': lstm_preds_denorm,
            'metrics': lstm_metrics
        }
    
    # If Random Forest model exists
    if rf_model is not None:
        # Predict
        rf_preds = rf_model.predict(X_test_reshaped)
        
        # Denormalize predictions and ground truth
        rf_preds_denorm = rf_preds * std + mean
        y_test_denorm = y_test * std + mean
        
        # Calculate metrics (for each forecast day)
        rf_metrics = {
            'day': [],
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        for day in range(y_test.shape[1]):
            mse = mean_squared_error(y_test_denorm[:, day], rf_preds_denorm[:, day])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_denorm[:, day], rf_preds_denorm[:, day])
            r2 = r2_score(y_test_denorm[:, day], rf_preds_denorm[:, day])
            
            rf_metrics['day'].append(day + 1)
            rf_metrics['mse'].append(mse)
            rf_metrics['rmse'].append(rmse)
            rf_metrics['mae'].append(mae)
            rf_metrics['r2'].append(r2)
        
        results['rf'] = {
            'predictions': rf_preds_denorm,
            'metrics': rf_metrics
        }
    
    return results, y_test_denorm

def display_metrics_table(results, symbol, feature):
    """Display model evaluation metrics in a table format"""
    print(f"\n{symbol} - {feature} Feature Prediction Results Comparison")
    
    if 'lstm' in results and 'rf' in results:
        # Create comparison table
        table_data = []
        headers = ["Forecast Day", "LSTM-MSE", "RF-MSE", "LSTM-RMSE", "RF-RMSE", "LSTM-MAE", "RF-MAE", "LSTM-R²", "RF-R²"]
        
        lstm_metrics = results['lstm']['metrics']
        rf_metrics = results['rf']['metrics']
        
        for i in range(len(lstm_metrics['day'])):
            row = [
                f"Day {lstm_metrics['day'][i]}",
                f"{lstm_metrics['mse'][i]:.4f}",
                f"{rf_metrics['mse'][i]:.4f}",
                f"{lstm_metrics['rmse'][i]:.4f}",
                f"{rf_metrics['rmse'][i]:.4f}",
                f"{lstm_metrics['mae'][i]:.4f}",
                f"{rf_metrics['mae'][i]:.4f}",
                f"{lstm_metrics['r2'][i]:.4f}",
                f"{rf_metrics['r2'][i]:.4f}"
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
        
        # Calculate average metrics
        lstm_avg_mse = np.mean(lstm_metrics['mse'])
        rf_avg_mse = np.mean(rf_metrics['mse'])
        lstm_avg_rmse = np.mean(lstm_metrics['rmse'])
        rf_avg_rmse = np.mean(rf_metrics['rmse'])
        lstm_avg_mae = np.mean(lstm_metrics['mae'])
        rf_avg_mae = np.mean(rf_metrics['mae'])
        lstm_avg_r2 = np.mean(lstm_metrics['r2'])
        rf_avg_r2 = np.mean(rf_metrics['r2'])
        
        print("\nAverage Metrics:")
        avg_data = [
            ["Average", f"{lstm_avg_mse:.4f}", f"{rf_avg_mse:.4f}", 
             f"{lstm_avg_rmse:.4f}", f"{rf_avg_rmse:.4f}", 
             f"{lstm_avg_mae:.4f}", f"{rf_avg_mae:.4f}", 
             f"{lstm_avg_r2:.4f}", f"{rf_avg_r2:.4f}"]
        ]
        print(tabulate(avg_data, headers=headers, tablefmt="pretty"))
        
        # Output which model performs better overall
        if lstm_avg_mse < rf_avg_mse:
            print(f"\nBased on MSE, LSTM model performs better overall for {feature} prediction")
        else:
            print(f"\nBased on MSE, Random Forest model performs better overall for {feature} prediction")
    
    elif 'lstm' in results:
        # Only LSTM model
        table_data = []
        headers = ["Forecast Day", "MSE", "RMSE", "MAE", "R²"]
        
        lstm_metrics = results['lstm']['metrics']
        
        for i in range(len(lstm_metrics['day'])):
            row = [
                f"Day {lstm_metrics['day'][i]}",
                f"{lstm_metrics['mse'][i]:.4f}",
                f"{lstm_metrics['rmse'][i]:.4f}",
                f"{lstm_metrics['mae'][i]:.4f}",
                f"{lstm_metrics['r2'][i]:.4f}"
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
        
    elif 'rf' in results:
        # Only Random Forest model
        table_data = []
        headers = ["Forecast Day", "MSE", "RMSE", "MAE", "R²"]
        
        rf_metrics = results['rf']['metrics']
        
        for i in range(len(rf_metrics['day'])):
            row = [
                f"Day {rf_metrics['day'][i]}",
                f"{rf_metrics['mse'][i]:.4f}",
                f"{rf_metrics['rmse'][i]:.4f}",
                f"{rf_metrics['mae'][i]:.4f}",
                f"{rf_metrics['r2'][i]:.4f}"
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

def export_prediction_data_to_csv(results, y_test_denorm, symbol, feature):
    """Export prediction data to CSV for further analysis"""
    output_dir = os.path.join("project_files", symbol, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for actual values
    samples = len(y_test_denorm)
    data = {'Sample_Index': range(samples)}
    
    # Add actual values for each forecast day
    for day in range(y_test_denorm.shape[1]):
        data[f'Actual_Day{day+1}'] = y_test_denorm[:, day]
    
    # Add model predictions
    if 'lstm' in results:
        lstm_preds = results['lstm']['predictions']
        for day in range(lstm_preds.shape[1]):
            data[f'LSTM_Day{day+1}'] = lstm_preds[:, day]
    
    if 'rf' in results:
        rf_preds = results['rf']['predictions']
        for day in range(rf_preds.shape[1]):
            data[f'RF_Day{day+1}'] = rf_preds[:, day]
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, f"{symbol}_{feature}_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Prediction data exported to CSV: {csv_path}")

def plot_key_metrics(results, y_test_denorm, symbol, feature):
    """Plot key metrics comparison between models for the research paper"""
    if not results or len(results) < 1:
        return
        
    output_dir = os.path.join("project_files", symbol, "evaluation", "report_figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. RMSE by Forecast Day - For "Model Performance Over Time" section
    if 'lstm' in results and 'rf' in results:
        lstm_metrics = results['lstm']['metrics']
        rf_metrics = results['rf']['metrics']
        days = lstm_metrics['day']
        
        plt.figure(figsize=(10, 6))
        plt.plot(days, lstm_metrics['rmse'], 'o-', label='LSTM', color='blue', linewidth=2)
        plt.plot(days, rf_metrics['rmse'], 's-', label='Random Forest', color='red', linewidth=2)
        plt.title(f"{symbol} - {feature} - RMSE by Forecast Day", fontsize=14)
        plt.xlabel("Forecast Day", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.xticks(days, fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_rmse_by_day.png"), dpi=300)
        plt.close()
        print(f"Figure 1: RMSE by Forecast Day - For 'Model Performance Over Time' section")
        
        # 2. Day 1 Metrics Comparison - For "Prediction Accuracy Comparison" section
        metrics = ['MSE', 'RMSE', 'MAE']
        lstm_values = [lstm_metrics['mse'][0], lstm_metrics['rmse'][0], lstm_metrics['mae'][0]]
        rf_values = [rf_metrics['mse'][0], rf_metrics['rmse'][0], rf_metrics['mae'][0]]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, lstm_values, width, label='LSTM', color='blue', alpha=0.8)
        plt.bar(x + width/2, rf_values, width, label='Random Forest', color='red', alpha=0.8)
        
        # Add values on top of bars
        for i, v in enumerate(lstm_values):
            plt.text(i - width/2, v + max(lstm_values + rf_values) * 0.02, f"{v:.2f}", 
                     ha='center', va='bottom', fontsize=10)
        for i, v in enumerate(rf_values):
            plt.text(i + width/2, v + max(lstm_values + rf_values) * 0.02, f"{v:.2f}", 
                     ha='center', va='bottom', fontsize=10)
        
        plt.title(f"{symbol} - {feature} - Day 1 Prediction Error Metrics", fontsize=14)
        plt.xlabel("Metrics", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.xticks(x, metrics, fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_day1_metrics.png"), dpi=300)
        plt.close()
        print(f"Figure 2: Day 1 Metrics Comparison - For 'Prediction Accuracy Comparison' section")
        
        # 3. R² Comparison - For "Model Performance Over Time" section
        plt.figure(figsize=(10, 6))
        plt.plot(days, lstm_metrics['r2'], 'o-', label='LSTM', color='blue', linewidth=2)
        plt.plot(days, rf_metrics['r2'], 's-', label='Random Forest', color='red', linewidth=2)
        plt.title(f"{symbol} - {feature} - R² by Forecast Day", fontsize=14)
        plt.xlabel("Forecast Day", fontsize=12)
        plt.ylabel("R²", fontsize=12)
        plt.xticks(days, fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_r2_by_day.png"), dpi=300)
        plt.close()
        print(f"Figure 3: R² by Forecast Day - For 'Model Performance Over Time' section")
    
    # 4. Day 1 Prediction Comparison - For "Prediction Accuracy Comparison" section
    if ('lstm' in results or 'rf' in results) and y_test_denorm is not None:
        plt.figure(figsize=(12, 6))
        # Show only first 20 samples for clarity
        samples = min(20, len(y_test_denorm))
        
        # Plot actual values
        plt.plot(range(samples), y_test_denorm[:samples, 0], 'o-', label="Actual", color='black', linewidth=2)
        
        # Plot predictions
        if 'lstm' in results:
            lstm_preds = results['lstm']['predictions']
            plt.plot(range(samples), lstm_preds[:samples, 0], 's-', label="LSTM", color='blue', linewidth=2)
        
        if 'rf' in results:
            rf_preds = results['rf']['predictions']
            plt.plot(range(samples), rf_preds[:samples, 0], '^-', label="Random Forest", color='red', linewidth=2)
        
        plt.title(f"{symbol} - {feature} - Day 1 Prediction Comparison", fontsize=14)
        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel(f"{feature} Value", fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_day1_predictions.png"), dpi=300)
        plt.close()
        print(f"Figure 4: Day 1 Prediction Comparison - For 'Prediction Accuracy Comparison' section")
    
    print(f"\nKey figures for the research paper saved to {output_dir}")

def plot_error_distribution(results, y_test_denorm, symbol, feature):
    """Plot error distribution for error analysis section"""
    if not results or len(results) < 1 or y_test_denorm is None:
        return
        
    output_dir = os.path.join("project_files", symbol, "evaluation", "report_figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Error Distribution - For "Error Distribution Analysis" section
    plt.figure(figsize=(12, 6))
    
    if 'lstm' in results:
        lstm_preds = results['lstm']['predictions']
        lstm_errors = lstm_preds[:, 0] - y_test_denorm[:, 0]
        sns.kdeplot(lstm_errors, label='LSTM', color='blue', linewidth=2)
    
    if 'rf' in results:
        rf_preds = results['rf']['predictions']
        rf_errors = rf_preds[:, 0] - y_test_denorm[:, 0]
        sns.kdeplot(rf_errors, label='Random Forest', color='red', linewidth=2)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title(f"{symbol} - {feature} - Day 1 Error Distribution", fontsize=14)
    plt.xlabel("Prediction Error", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_error_distribution.png"), dpi=300)
    plt.close()
    print(f"Figure 5: Error Distribution - For 'Error Distribution Analysis' section")
    
    # 2. Error vs Actual Value - For "Error Distribution Analysis" section
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    if 'lstm' in results:
        ax1.scatter(y_test_denorm[:, 0], lstm_errors, alpha=0.5, label='LSTM Error', color='blue')
        ax1.axhline(y=0, color='red', linestyle='-', linewidth=1)
        ax1.set_title(f"{symbol} - {feature} - LSTM Error vs Actual Value", fontsize=14)
        ax1.set_xlabel("Actual Value", fontsize=12)
        ax1.set_ylabel("Prediction Error", fontsize=12)
        ax1.tick_params(labelsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
    
    if 'rf' in results:
        ax2.scatter(y_test_denorm[:, 0], rf_errors, alpha=0.5, label='RF Error', color='red')
        ax2.axhline(y=0, color='red', linestyle='-', linewidth=1)
        ax2.set_title(f"{symbol} - {feature} - RF Error vs Actual Value", fontsize=14)
        ax2.set_xlabel("Actual Value", fontsize=12)
        ax2.set_ylabel("Prediction Error", fontsize=12)
        ax2.tick_params(labelsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_error_vs_actual.png"), dpi=300)
    plt.close()
    print(f"Figure 6: Error vs Actual Value - For 'Error Distribution Analysis' section")

def export_results_to_csv(results, symbol, feature):
    """Export evaluation results to CSV file"""
    output_dir = os.path.join("project_files", symbol, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame and save
    if 'lstm' in results and 'rf' in results:
        lstm_metrics = results['lstm']['metrics']
        rf_metrics = results['rf']['metrics']
        
        data = {
            'Forecast_Day': [f"Day_{day}" for day in lstm_metrics['day']],
            'LSTM_MSE': lstm_metrics['mse'],
            'RF_MSE': rf_metrics['mse'],
            'LSTM_RMSE': lstm_metrics['rmse'],
            'RF_RMSE': rf_metrics['rmse'],
            'LSTM_MAE': lstm_metrics['mae'],
            'RF_MAE': rf_metrics['mae'],
            'LSTM_R2': lstm_metrics['r2'],
            'RF_R2': rf_metrics['r2']
        }
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f"{symbol}_{feature}_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results exported to CSV: {csv_path}")

def create_performance_heatmap(symbols, features, output_folder):
    """Create heatmap comparing model performance across different features"""
    combined_data = []
    
    for symbol in symbols:
        for feature in features:
            eval_file = os.path.join(output_folder, symbol, "evaluation", f"{symbol}_{feature}_comparison.csv")
            
            if os.path.exists(eval_file):
                df = pd.read_csv(eval_file)
                
                # Get Day 1 metrics
                if len(df) > 0:
                    day1_data = df.iloc[0]
                    
                    if 'LSTM_RMSE' in day1_data and 'RF_RMSE' in day1_data:
                        combined_data.append({
                            'Symbol': symbol,
                            'Feature': feature,
                            'LSTM_RMSE': day1_data['LSTM_RMSE'],
                            'RF_RMSE': day1_data['RF_RMSE'],
                            'LSTM_R2': day1_data['LSTM_R2'],
                            'RF_R2': day1_data['RF_R2'],
                            'RMSE_Diff': day1_data['RF_RMSE'] - day1_data['LSTM_RMSE'],
                            'R2_Diff': day1_data['LSTM_R2'] - day1_data['RF_R2']
                        })
    
    if not combined_data:
        print("No data available for heatmap")
        return
        
    # Create DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Create output directory
    output_dir = os.path.join(output_folder, "report_figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. RMSE Heatmap - For "Different Features Prediction Difficulty" section
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for RMSE
    rmse_pivot = combined_df.pivot(index="Feature", columns="Symbol", values="RMSE_Diff")
    
    # Create heatmap
    sns.heatmap(rmse_pivot, annot=True, cmap='RdBu_r', center=0, fmt=".2f", 
                annot_kws={"size": 12}, cbar_kws={"label": "RMSE Difference (RF - LSTM)"})
    
    plt.title("RMSE Difference Across Features (RF - LSTM)", fontsize=14)
    plt.xlabel("Stock Symbol", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_rmse_heatmap.png"), dpi=300)
    plt.close()
    print(f"Figure 7: RMSE Heatmap - For 'Different Features Prediction Difficulty' section")
    
    # 2. Create radar chart - For "Radar Chart Analysis" section
    # Calculate average metrics across symbols for each feature
    avg_metrics = combined_df.groupby('Feature')[['LSTM_RMSE', 'RF_RMSE', 'LSTM_R2', 'RF_R2']].mean().reset_index()
    
    for feature in avg_metrics['Feature']:
        feature_data = avg_metrics[avg_metrics['Feature'] == feature]
        
        # Get metrics
        metrics = ['RMSE', 'R²']
        lstm_values = [
            feature_data['LSTM_RMSE'].values[0],
            feature_data['LSTM_R2'].values[0]
        ]
        
        rf_values = [
            feature_data['RF_RMSE'].values[0],
            feature_data['RF_R2'].values[0]
        ]
        
        # Normalize values for radar chart
        # For RMSE (lower is better), invert the values and normalize
        max_rmse = max(feature_data['LSTM_RMSE'].values[0], feature_data['RF_RMSE'].values[0])
        norm_lstm_rmse = 1 - (feature_data['LSTM_RMSE'].values[0] / max_rmse)
        norm_rf_rmse = 1 - (feature_data['RF_RMSE'].values[0] / max_rmse)
        
        # For R2 (higher is better), just normalize to 0-1
        norm_lstm_r2 = feature_data['LSTM_R2'].values[0]
        norm_rf_r2 = feature_data['RF_R2'].values[0]
        
        # Final normalized values
        norm_lstm = [norm_lstm_rmse, norm_lstm_r2]
        norm_rf = [norm_rf_rmse, norm_rf_r2]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        norm_lstm += norm_lstm[:1]  # Close the loop
        norm_rf += norm_rf[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, norm_lstm, 'b-', linewidth=2, label='LSTM')
        ax.fill(angles, norm_lstm, 'b', alpha=0.1)
        
        ax.plot(angles, norm_rf, 'r-', linewidth=2, label='Random Forest')
        ax.fill(angles, norm_rf, 'r', alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        
        # Add legend and title
        ax.legend(loc='upper right', fontsize=12)
        plt.title(f"{feature} - Model Performance Comparison", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{feature}_radar_chart.png"), dpi=300)
        plt.close()
        print(f"Figure 8: Radar Chart for {feature} - For 'Radar Chart Analysis' section")
    
    # 3. Create multi-day RMSE comparison for "Model Performance Over Time" section
    for symbol in symbols:
        multi_day_data = []
        
        for feature in features:
            eval_file = os.path.join(output_folder, symbol, "evaluation", f"{symbol}_{feature}_comparison.csv")
            
            if os.path.exists(eval_file):
                df = pd.read_csv(eval_file)
                
                if len(df) > 1:  # Ensure we have multiple days
                    feature_data = {
                        'Feature': feature,
                        'Days': list(range(1, len(df) + 1)),
                        'LSTM_RMSE': df['LSTM_RMSE'].values,
                        'RF_RMSE': df['RF_RMSE'].values
                    }
                    multi_day_data.append(feature_data)
        
        if multi_day_data:
            plt.figure(figsize=(15, 10))
            
            for i, data in enumerate(multi_day_data):
                plt.subplot(len(multi_day_data), 1, i + 1)
                
                plt.plot(data['Days'], data['LSTM_RMSE'], 'o-', label='LSTM', color='blue', linewidth=2)
                plt.plot(data['Days'], data['RF_RMSE'], 's-', label='Random Forest', color='red', linewidth=2)
                
                plt.title(f"{symbol} - {data['Feature']} - RMSE by Forecast Day", fontsize=12)
                plt.xlabel("Forecast Day" if i == len(multi_day_data) - 1 else "", fontsize=10)
                plt.ylabel("RMSE", fontsize=10)
                plt.xticks(data['Days'])
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{symbol}_multi_feature_rmse.png"), dpi=300)
            plt.close()
            print(f"Figure 9: Multi-feature RMSE Comparison for {symbol} - For 'Model Performance Over Time' section")

def main():
    output_folder = "project_files"
    symbols = ['AAPL', 'MSFT']
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    # Create output directory for report figures
    report_dir = os.path.join(output_folder, "report_figures")
    os.makedirs(report_dir, exist_ok=True)
    
    for symbol in symbols:
        print(f"\nGenerating report figures for {symbol}...")
        
        for feature in features:
            print(f"\nFeature: {feature}")
            
            try:
                # Load data
                X_test, y_test, X_test_reshaped, norm_params = load_data(symbol, output_folder, feature)
                
                # Load models
                lstm_model, rf_model = load_models(symbol, output_folder, feature)
                
                if lstm_model is None and rf_model is None:
                    print(f"No models found for {symbol}'s {feature} feature, skipping evaluation")
                    continue
                
                # Evaluate models
                results, y_test_denorm = evaluate_models(X_test, y_test, X_test_reshaped, lstm_model, rf_model, 
                                                      norm_params, feature)
                
                # Display metrics table
                display_metrics_table(results, symbol, feature)
                
                # Export prediction data to CSV for analysis
                export_prediction_data_to_csv(results, y_test_denorm, symbol, feature)
                
                # Generate key figures for the research paper
                plot_key_metrics(results, y_test_denorm, symbol, feature)
                
                # Generate error distribution figures
                plot_error_distribution(results, y_test_denorm, symbol, feature)
                
                # Export results to CSV
                export_results_to_csv(results, symbol, feature)
                
            except Exception as e:
                print(f"Error generating figures for {symbol}'s {feature} feature: {str(e)}")
    
    # Create comparative analysis figures across features
    try:
        create_performance_heatmap(symbols, features, output_folder)
    except Exception as e:
        print(f"Error creating performance heatmap: {str(e)}")
    
    print("\nAll figures for the research paper have been generated")
    print(f"Report figures are saved in the '{report_dir}' and symbol-specific 'report_figures' directories")
    
    # Print figure placement guide
    print("\n--- Figure Placement Guide for Research Paper ---")
    print("1. RMSE by Forecast Day Charts: 'Model Performance Over Time' section")
    print("2. Day 1 Metrics Comparison: 'Prediction Accuracy Comparison' section")
    print("3. R² by Forecast Day Charts: 'Model Performance Over Time' section")
    print("4. Day 1 Prediction Comparison: 'Prediction Accuracy Comparison' section")
    print("5. Error Distribution: 'Error Distribution Analysis' section")
    print("6. Error vs Actual Value: 'Error Distribution Analysis' section")
    print("7. RMSE Heatmap: 'Different Features Prediction Difficulty' section")
    print("8. Radar Charts: 'Radar Chart Analysis' section")
    print("9. Multi-feature RMSE Comparison: 'Model Performance Over Time' section")

if __name__ == "__main__":
    main() 