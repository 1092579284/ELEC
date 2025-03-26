import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate

def load_csv_files(symbol, feature):
    """Load prediction and metrics CSV files for a specific symbol and feature"""
    base_dir = os.path.join("project_files", symbol, "evaluation")
    prediction_file = os.path.join(base_dir, f"{symbol}_{feature}_predictions.csv")
    metrics_file = os.path.join(base_dir, f"{symbol}_{feature}_comparison.csv")
    
    prediction_df = None
    metrics_df = None
    
    if os.path.exists(prediction_file):
        prediction_df = pd.read_csv(prediction_file)
        print(f"Loaded prediction data for {symbol} - {feature}")
    
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        print(f"Loaded metrics data for {symbol} - {feature}")
    
    return prediction_df, metrics_df

def visualize_prediction_trends(prediction_df, symbol, feature, output_dir):
    """Visualize the prediction trends over time"""
    if prediction_df is None:
        print(f"No prediction data available for {symbol} - {feature}")
        return
    
    # Create a figure with multiple subplots for each prediction day
    days = [col for col in prediction_df.columns if col.startswith('Actual_Day')]
    n_days = len(days)
    
    # Use a subset of data for visualization (first 100 samples or all if less)
    sample_size = min(100, len(prediction_df))
    sample_df = prediction_df.iloc[:sample_size].copy()
    
    # Add a date-like index for better visualization
    sample_df['Date'] = pd.date_range(start='2023-01-01', periods=len(sample_df))
    
    plt.figure(figsize=(15, 10))
    for i, day in enumerate(range(1, n_days + 1)):
        plt.subplot(n_days, 1, i + 1)
        
        actual_col = f'Actual_Day{day}'
        lstm_col = f'LSTM_Day{day}'
        rf_col = f'RF_Day{day}'
        
        plt.plot(sample_df['Date'], sample_df[actual_col], 'k-', label='Actual')
        
        if lstm_col in sample_df.columns:
            plt.plot(sample_df['Date'], sample_df[lstm_col], 'b-', label='LSTM')
        
        if rf_col in sample_df.columns:
            plt.plot(sample_df['Date'], sample_df[rf_col], 'r-', label='RF')
        
        plt.title(f"{symbol} - {feature} - Day {day} Prediction Trend")
        plt.ylabel(f"{feature} Value")
        plt.legend()
        plt.grid(True)
        
        # Only show x-label for the bottom plot
        if i == n_days - 1:
            plt.xlabel("Date")
        
        # Format x-axis to show dates better
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_prediction_trends.png"))
    plt.close()
    print(f"Saved prediction trends visualization for {symbol} - {feature}")

def visualize_error_distribution(prediction_df, symbol, feature, output_dir):
    """Visualize error distribution for both models"""
    if prediction_df is None:
        print(f"No prediction data available for {symbol} - {feature}")
        return
        
    # Calculate errors for day 1 predictions (most important)
    day = 1
    actual_col = f'Actual_Day{day}'
    data = []
    
    if f'LSTM_Day{day}' in prediction_df.columns:
        lstm_errors = prediction_df[f'LSTM_Day{day}'] - prediction_df[actual_col]
        data.append(("LSTM", lstm_errors))
    
    if f'RF_Day{day}' in prediction_df.columns:
        rf_errors = prediction_df[f'RF_Day{day}'] - prediction_df[actual_col]
        data.append(("Random Forest", rf_errors))
    
    if not data:
        return
        
    # Create error distribution plot
    plt.figure(figsize=(12, 6))
    for name, errors in data:
        sns.kdeplot(errors, label=name)
    
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(f"{symbol} - {feature} - Day 1 Prediction Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_error_distribution.png"))
    plt.close()
    print(f"Saved error distribution visualization for {symbol} - {feature}")

def visualize_error_vs_actual(prediction_df, symbol, feature, output_dir):
    """Visualize prediction error vs actual value to see if errors are correlated with value magnitude"""
    if prediction_df is None:
        print(f"No prediction data available for {symbol} - {feature}")
        return
    
    # Focus on day 1 predictions
    day = 1
    actual_col = f'Actual_Day{day}'
    
    plt.figure(figsize=(15, 7))
    
    # Plot for LSTM
    if f'LSTM_Day{day}' in prediction_df.columns:
        plt.subplot(1, 2, 1)
        lstm_errors = prediction_df[f'LSTM_Day{day}'] - prediction_df[actual_col]
        plt.scatter(prediction_df[actual_col], lstm_errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f"{symbol} - {feature} - LSTM Error vs Actual Value")
        plt.xlabel("Actual Value")
        plt.ylabel("Prediction Error")
        plt.grid(True)
    
    # Plot for Random Forest
    if f'RF_Day{day}' in prediction_df.columns:
        plt.subplot(1, 2, 2)
        rf_errors = prediction_df[f'RF_Day{day}'] - prediction_df[actual_col]
        plt.scatter(prediction_df[actual_col], rf_errors, alpha=0.5, color='green')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f"{symbol} - {feature} - RF Error vs Actual Value")
        plt.xlabel("Actual Value")
        plt.ylabel("Prediction Error")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_error_vs_actual.png"))
    plt.close()
    print(f"Saved error vs actual visualization for {symbol} - {feature}")

def visualize_metrics_comparison(metrics_df, symbol, feature, output_dir):
    """Visualize metrics comparison across forecast days"""
    if metrics_df is None:
        print(f"No metrics data available for {symbol} - {feature}")
        return
    
    # Create radar chart for metrics comparison
    if 'LSTM_MSE' in metrics_df.columns and 'RF_MSE' in metrics_df.columns:
        # Create radar chart for Day 1 metrics
        day1_metrics = metrics_df.iloc[0]
        
        # Metrics to compare (excluding forecast day column)
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        
        # Prepare data
        lstm_values = [
            day1_metrics['LSTM_MSE'] / max(day1_metrics['LSTM_MSE'], day1_metrics['RF_MSE']),
            day1_metrics['LSTM_RMSE'] / max(day1_metrics['LSTM_RMSE'], day1_metrics['RF_RMSE']),
            day1_metrics['LSTM_MAE'] / max(day1_metrics['LSTM_MAE'], day1_metrics['RF_MAE']),
            day1_metrics['LSTM_R2']  # Already normalized between 0-1
        ]
        
        rf_values = [
            day1_metrics['RF_MSE'] / max(day1_metrics['LSTM_MSE'], day1_metrics['RF_MSE']),
            day1_metrics['RF_RMSE'] / max(day1_metrics['LSTM_RMSE'], day1_metrics['RF_RMSE']),
            day1_metrics['RF_MAE'] / max(day1_metrics['LSTM_MAE'], day1_metrics['RF_MAE']),
            day1_metrics['RF_R2']  # Already normalized between 0-1
        ]
        
        # For R2, higher is better, so invert normalization
        lstm_values[3] = lstm_values[3]
        rf_values[3] = rf_values[3]
        
        # For error metrics (MSE, RMSE, MAE), lower is better, so invert
        for i in range(3):
            lstm_values[i] = 1 - lstm_values[i]
            rf_values[i] = 1 - rf_values[i]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        lstm_values += lstm_values[:1]  # Close the loop
        rf_values += rf_values[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, lstm_values, 'b-', linewidth=2, label='LSTM')
        ax.fill(angles, lstm_values, 'b', alpha=0.1)
        
        ax.plot(angles, rf_values, 'r-', linewidth=2, label='Random Forest')
        ax.fill(angles, rf_values, 'r', alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add legend and title
        ax.legend(loc='upper right')
        plt.title(f"{symbol} - {feature} - Day 1 Model Performance Comparison")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_radar_chart.png"))
        plt.close()
        print(f"Saved radar chart visualization for {symbol} - {feature}")

    # Create a heatmap for performance difference across days
    if len(metrics_df) > 1:  # Only if we have multiple forecast days
        # Calculate performance difference (LSTM - RF)
        # Negative values mean RF is better, positive values mean LSTM is better
        diff_data = {
            'Day': metrics_df['Forecast_Day'],
            'MSE_Diff': metrics_df['RF_MSE'] - metrics_df['LSTM_MSE'],
            'RMSE_Diff': metrics_df['RF_RMSE'] - metrics_df['LSTM_RMSE'],
            'MAE_Diff': metrics_df['RF_MAE'] - metrics_df['LSTM_MAE'],
            'R2_Diff': metrics_df['LSTM_R2'] - metrics_df['RF_R2']  # Note: for R2, higher is better
        }
        
        diff_df = pd.DataFrame(diff_data)
        
        # Format for heatmap
        heatmap_data = diff_df.set_index('Day')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True, fmt=".4f")
        plt.title(f"{symbol} - {feature} - Performance Difference (LSTM - RF)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{symbol}_{feature}_performance_heatmap.png"))
        plt.close()
        print(f"Saved performance heatmap visualization for {symbol} - {feature}")

def main():
    symbols = ['AAPL', 'MSFT']
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    for symbol in symbols:
        print(f"\nVisualizing results for {symbol}...")
        
        for feature in features:
            print(f"\nFeature: {feature}")
            
            try:
                # Create output directory
                output_dir = os.path.join("project_files", symbol, "evaluation", "visualizations")
                os.makedirs(output_dir, exist_ok=True)
                
                # Load CSV files
                prediction_df, metrics_df = load_csv_files(symbol, feature)
                
                if prediction_df is None and metrics_df is None:
                    print(f"No data found for {symbol} - {feature}, skipping visualization")
                    continue
                
                # Generate visualizations
                if prediction_df is not None:
                    visualize_prediction_trends(prediction_df, symbol, feature, output_dir)
                    visualize_error_distribution(prediction_df, symbol, feature, output_dir)
                    visualize_error_vs_actual(prediction_df, symbol, feature, output_dir)
                
                if metrics_df is not None:
                    visualize_metrics_comparison(metrics_df, symbol, feature, output_dir)
                
                print(f"All visualizations for {symbol} - {feature} saved to {output_dir}")
                
            except Exception as e:
                print(f"Error visualizing {symbol}'s {feature} data: {str(e)}")

if __name__ == "__main__":
    main() 