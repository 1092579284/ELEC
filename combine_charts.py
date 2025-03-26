import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

def create_combined_rmse_charts(stocks, features):
    """Combine RMSE by Forecast Day charts for all features into one chart per stock"""
    for stock in stocks:
        # Create a figure with subplots for each feature
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            
            # Skip if we've plotted all features
            if i >= len(features):
                break
                
            # Path to CSV file with metrics
            metrics_file = os.path.join("project_files", stock, "evaluation", f"{stock}_{feature}_comparison.csv")
            
            if not os.path.exists(metrics_file):
                continue
                
            # Load metrics data
            df = pd.read_csv(metrics_file)
            
            # Extract forecast days and RMSE values
            days = [int(day.split('_')[1]) for day in df['Forecast_Day']]
            lstm_rmse = df['LSTM_RMSE'].values
            rf_rmse = df['RF_RMSE'].values
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Plot RMSE by day
            ax.plot(days, lstm_rmse, 'o-', label='LSTM', color='blue', linewidth=2)
            ax.plot(days, rf_rmse, 's-', label='Random Forest', color='red', linewidth=2)
            
            ax.set_title(f"{feature} - RMSE by Forecast Day", fontsize=14)
            ax.set_xlabel("Forecast Day", fontsize=12)
            ax.set_ylabel("RMSE", fontsize=12)
            ax.set_xticks(days)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"{stock} - RMSE by Forecast Day Across Features", fontsize=16, y=1.02)
        
        # Create output directory
        output_dir = os.path.join("project_files", stock, "evaluation", "combined_figures")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{stock}_combined_rmse_chart.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created combined RMSE chart for {stock}")

def create_combined_r2_charts(stocks, features):
    """Combine R² by Forecast Day charts for all features into one chart per stock"""
    for stock in stocks:
        # Create a figure with subplots for each feature
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            
            # Skip if we've plotted all features
            if i >= len(features):
                break
                
            # Path to CSV file with metrics
            metrics_file = os.path.join("project_files", stock, "evaluation", f"{stock}_{feature}_comparison.csv")
            
            if not os.path.exists(metrics_file):
                continue
                
            # Load metrics data
            df = pd.read_csv(metrics_file)
            
            # Extract forecast days and R² values
            days = [int(day.split('_')[1]) for day in df['Forecast_Day']]
            lstm_r2 = df['LSTM_R2'].values
            rf_r2 = df['RF_R2'].values
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Plot R² by day
            ax.plot(days, lstm_r2, 'o-', label='LSTM', color='blue', linewidth=2)
            ax.plot(days, rf_r2, 's-', label='Random Forest', color='red', linewidth=2)
            
            ax.set_title(f"{feature} - R² by Forecast Day", fontsize=14)
            ax.set_xlabel("Forecast Day", fontsize=12)
            ax.set_ylabel("R²", fontsize=12)
            ax.set_xticks(days)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"{stock} - R² by Forecast Day Across Features", fontsize=16, y=1.02)
        
        # Create output directory
        output_dir = os.path.join("project_files", stock, "evaluation", "combined_figures")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{stock}_combined_r2_chart.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created combined R² chart for {stock}")

def create_combined_day1_metrics(stocks, features):
    """Combine Day 1 Metrics Comparison charts for all features into one chart per stock"""
    for stock in stocks:
        # Create a figure with subplots for each feature
        fig = plt.figure(figsize=(18, 15))
        gs = GridSpec(3, 2, figure=fig)
        
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            
            # Skip if we've plotted all features
            if i >= len(features):
                break
                
            # Path to CSV file with metrics
            metrics_file = os.path.join("project_files", stock, "evaluation", f"{stock}_{feature}_comparison.csv")
            
            if not os.path.exists(metrics_file):
                continue
                
            # Load metrics data
            df = pd.read_csv(metrics_file)
            
            # Get Day 1 metrics
            if len(df) > 0:
                day1_data = df.iloc[0]
                
                # Create subplot
                ax = fig.add_subplot(gs[row, col])
                
                # Extract metrics for Day 1
                metrics = ['MSE', 'RMSE', 'MAE']
                lstm_values = [day1_data['LSTM_MSE'], day1_data['LSTM_RMSE'], day1_data['LSTM_MAE']]
                rf_values = [day1_data['RF_MSE'], day1_data['RF_RMSE'], day1_data['RF_MAE']]
                
                # Create bar chart
                x = np.arange(len(metrics))
                width = 0.35
                
                ax.bar(x - width/2, lstm_values, width, label='LSTM', color='blue', alpha=0.8)
                ax.bar(x + width/2, rf_values, width, label='Random Forest', color='red', alpha=0.8)
                
                # Add values on top of bars
                for j, v in enumerate(lstm_values):
                    ax.text(j - width/2, v + max(lstm_values + rf_values) * 0.02, f"{v:.2f}", 
                           ha='center', va='bottom', fontsize=10)
                for j, v in enumerate(rf_values):
                    ax.text(j + width/2, v + max(lstm_values + rf_values) * 0.02, f"{v:.2f}", 
                           ha='center', va='bottom', fontsize=10)
                
                ax.set_title(f"{feature} - Day 1 Prediction Error Metrics", fontsize=14)
                ax.set_xlabel("Metrics", fontsize=12)
                ax.set_ylabel("Value", fontsize=12)
                ax.set_xticks(x, metrics, fontsize=11)
                ax.legend(fontsize=11)
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"{stock} - Day 1 Metrics Comparison Across Features", fontsize=16, y=1.02)
        
        # Create output directory
        output_dir = os.path.join("project_files", stock, "evaluation", "combined_figures")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{stock}_combined_day1_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created combined Day 1 Metrics chart for {stock}")

def create_combined_error_distribution(stocks, features):
    """Combine Error Distribution charts for all features into one chart per stock"""
    for stock in stocks:
        # Create a figure with subplots for each feature
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            
            # Skip if we've plotted all features
            if i >= len(features):
                break
                
            # Path to CSV file with predictions
            predictions_file = os.path.join("project_files", stock, "evaluation", f"{stock}_{feature}_predictions.csv")
            
            if not os.path.exists(predictions_file):
                continue
                
            # Load predictions data
            df = pd.read_csv(predictions_file)
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Calculate errors for day 1
            if 'Actual_Day1' in df.columns:
                actual_col = 'Actual_Day1'
                
                if 'LSTM_Day1' in df.columns:
                    lstm_errors = df['LSTM_Day1'] - df[actual_col]
                    sns.kdeplot(lstm_errors, ax=ax, label='LSTM', color='blue', linewidth=2)
                
                if 'RF_Day1' in df.columns:
                    rf_errors = df['RF_Day1'] - df[actual_col]
                    sns.kdeplot(rf_errors, ax=ax, label='Random Forest', color='red', linewidth=2)
                
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax.set_title(f"{feature} - Day 1 Error Distribution", fontsize=14)
                ax.set_xlabel("Prediction Error", fontsize=12)
                ax.set_ylabel("Density", fontsize=12)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"{stock} - Error Distribution Across Features", fontsize=16, y=1.02)
        
        # Create output directory
        output_dir = os.path.join("project_files", stock, "evaluation", "combined_figures")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{stock}_combined_error_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created combined Error Distribution chart for {stock}")

def create_combined_error_vs_actual(stocks, features):
    """Combine Error vs Actual Value charts for all features into one chart per stock"""
    for stock in stocks:
        plt.figure(figsize=(20, 15))
        
        for i, feature in enumerate(features):
            # Skip if we've plotted all features
            if i >= len(features):
                break
                
            # Path to CSV file with predictions
            predictions_file = os.path.join("project_files", stock, "evaluation", f"{stock}_{feature}_predictions.csv")
            
            if not os.path.exists(predictions_file):
                continue
                
            # Load predictions data
            df = pd.read_csv(predictions_file)
            
            # Calculate errors for day 1
            if 'Actual_Day1' in df.columns:
                actual_col = 'Actual_Day1'
                
                # LSTM plot
                if 'LSTM_Day1' in df.columns:
                    plt.subplot(len(features), 2, 2*i+1)
                    lstm_errors = df['LSTM_Day1'] - df[actual_col]
                    plt.scatter(df[actual_col], lstm_errors, alpha=0.5, color='blue')
                    plt.axhline(y=0, color='red', linestyle='-', linewidth=1)
                    plt.title(f"{feature} - LSTM Error vs Actual Value", fontsize=14)
                    plt.xlabel("Actual Value", fontsize=12)
                    plt.ylabel("Prediction Error", fontsize=12)
                    plt.grid(True, alpha=0.3)
                
                # Random Forest plot
                if 'RF_Day1' in df.columns:
                    plt.subplot(len(features), 2, 2*i+2)
                    rf_errors = df['RF_Day1'] - df[actual_col]
                    plt.scatter(df[actual_col], rf_errors, alpha=0.5, color='red')
                    plt.axhline(y=0, color='red', linestyle='-', linewidth=1)
                    plt.title(f"{feature} - RF Error vs Actual Value", fontsize=14)
                    plt.xlabel("Actual Value", fontsize=12)
                    plt.ylabel("Prediction Error", fontsize=12) 
                    plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"{stock} - Error vs Actual Value Across Features", fontsize=16, y=1.0)
        
        # Create output directory
        output_dir = os.path.join("project_files", stock, "evaluation", "combined_figures")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{stock}_combined_error_vs_actual.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created combined Error vs Actual Value chart for {stock}")

def main():
    stocks = ['AAPL', 'MSFT']
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    print("Generating combined charts...")
    
    # Create combined RMSE charts
    create_combined_rmse_charts(stocks, features)
    
    # Create combined R² charts
    create_combined_r2_charts(stocks, features)
    
    # Create combined Day 1 Metrics charts
    create_combined_day1_metrics(stocks, features)
    
    # Create combined Error Distribution charts
    create_combined_error_distribution(stocks, features)
    
    # Create combined Error vs Actual Value charts
    create_combined_error_vs_actual(stocks, features)
    
    print("\nAll combined charts have been generated")
    print("Charts are saved in the 'project_files/{stock}/evaluation/combined_figures' directories")
    
    # Print figure placement guide
    print("\n--- Updated Figure Placement Guide for Research Paper ---")
    print("1. Combined RMSE Charts: 'Model Performance Over Time' section")
    print("2. Combined Day 1 Metrics: 'Prediction Accuracy Comparison' section")
    print("3. Combined R² Charts: 'Model Performance Over Time' section")
    print("4. Combined Error Distribution: 'Error Distribution Analysis' section")
    print("5. Combined Error vs Actual Value: 'Error Distribution Analysis' section")

if __name__ == "__main__":
    main() 