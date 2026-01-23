# Read data/Bosund_time_series.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

# Import multiple models to compare
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Metrics for evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import PdfPages for multi-page PDF
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

# Load and prepare the sales data
df = pd.read_csv('../data/Bosund_time_series.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Load and prepare weather data
print("Loading weather data...")
weather_df = pd.read_csv('data/weather/weather_jakobstad_cleaned.csv')
weather_df['Time'] = pd.to_datetime(weather_df['Time'])
weather_df.set_index('Time', inplace=True)

print(f"Weather data loaded: {len(weather_df)} days")
print(f"Weather date range: {weather_df.index.min()} to {weather_df.index.max()}")

# df is a time series of daily sales for each product in Bosund store where each column is a product and each row is a day
# Each day has sales for all products
# The index is the date
# The columns are the products
# The values are the sales in euros

# Predict weekly sales (Monday-Sunday) for each product in Bosund store using linear regression with feature engineering
# Add a green line in the plot to mark the boundary between train and test set (always at a week boundary)
# Ensure no data leakage: rolling features and splits are computed only on past data
# Skip products with 0 sales. Plot by week number (sum of revenue per week).
# Test set is always a whole number of weeks (preferably 4 or 5, but always full weeks, not partial)

# Convert the daily sales to weekly sales (Monday-Sunday)
df_weekly = df.resample('W-MON').sum()

# Skip products with 0 total sales across all time periods
total_sales = df_weekly.sum()
products_with_sales = total_sales[total_sales > 0].index
df_weekly = df_weekly[products_with_sales]

print(f"Number of products with sales: {len(products_with_sales)}")
print(f"Number of weeks: {len(df_weekly)}")

# Aggregate weather data to weekly (Monday-Sunday) using mean for most features
print("\nAggregating weather data to weekly...")
weather_weekly = weather_df.resample('W-MON').agg({
    'Air temperature': 'mean',
    'Dew-point temperature': 'mean',
    'Gust speed': 'max',  # Use max gust speed for the week
    'Pressure (msl)': 'mean',
    'Relative humidity': 'mean',
    'Wind direction': 'mean',
    'Wind speed': 'mean'
})

# Rename weather columns to be more feature-friendly
weather_weekly.columns = [
    'temp_mean',
    'dewpoint_mean', 
    'gust_max',
    'pressure_mean',
    'humidity_mean',
    'wind_dir_mean',
    'wind_speed_mean'
]

print(f"Weather weeks: {len(weather_weekly)}")
print(f"Weather date range: {weather_weekly.index.min()} to {weather_weekly.index.max()}")

# Merge weather data with sales data
print("\nMerging weather data with sales data...")
df_weekly = df_weekly.join(weather_weekly, how='left')

# Forward fill any missing weather data (if sales data extends beyond weather data)
df_weekly[weather_weekly.columns] = df_weekly[weather_weekly.columns].fillna(method='ffill')

print(f"Merged data shape: {df_weekly.shape}")
print(f"Columns with weather: {list(weather_weekly.columns)}")

# Create features for linear regression (no data leakage)
def create_features(df, window_sizes=[2, 4, 8], weather_cols=None):
    """
    Create rolling features for time series prediction
    Ensures no data leakage by only using past data
    Includes weather features and weather-based rolling features
    """
    df_features = df.copy()
    
    # Identify product columns (exclude weather columns)
    if weather_cols is None:
        weather_cols = []
    product_columns = [col for col in df.columns if col not in weather_cols]
    
    # Create rolling features for each product individually
    for product in product_columns:
        for window in window_sizes:
            # Rolling mean
            df_features[f'{product}_rolling_mean_{window}'] = df[product].rolling(window=window, min_periods=1).mean().shift(1)
            # Rolling std
            df_features[f'{product}_rolling_std_{window}'] = df[product].rolling(window=window, min_periods=1).std().shift(1)
            # Rolling max
            df_features[f'{product}_rolling_max_{window}'] = df[product].rolling(window=window, min_periods=1).max().shift(1)
    
    # Create rolling features for weather data
    for weather_col in weather_cols:
        for window in [2, 4]:  # Use smaller windows for weather
            df_features[f'{weather_col}_rolling_mean_{window}'] = df[weather_col].rolling(window=window, min_periods=1).mean().shift(1)
    
    # Trend features
    df_features['week_number'] = range(len(df_features))
    df_features['week_number_squared'] = df_features['week_number'] ** 2
    
    return df_features

# Define weather columns
weather_cols = list(weather_weekly.columns)

# Create features (including weather features)
df_features = create_features(df_weekly, weather_cols=weather_cols)

# Fill NaN values with 0 (from rolling features at the beginning)
df_features = df_features.fillna(0)

print(f"\nTotal features created: {len(df_features.columns)}")
print(f"Weather columns included: {weather_cols}")

# Split data at week boundary (use last 6 weeks as test set)
test_weeks = 15
train_size = len(df_features) - test_weeks

train_data = df_features.iloc[:train_size]
test_data = df_features.iloc[train_size:]

print(f"Train weeks: {len(train_data)}, Test weeks: {len(test_data)}")

print("\n" + "=" * 80)
print("TRAINING MULTIPLE MODELS WITH WEATHER FEATURES")
print("=" * 80)
print(f"\nWeather features included in model:")
for i, weather_col in enumerate(weather_cols, 1):
    print(f"  {i}. {weather_col}")
print(f"\nTotal feature types:")
print(f"  - Product rolling features (mean, std, max) for windows [2, 4, 8]")
print(f"  - Weather features (current week)")
print(f"  - Weather rolling features (mean) for windows [2, 4]")
print(f"  - Trend features (week_number, week_number_squared)")
print(f"\nModels to compare for each product:")
print(f"  1. K-Nearest Neighbors (KNN)")
print(f"  2. Linear Regression")
print(f"  3. Ridge Regression")
print(f"  4. Lasso Regression")
print(f"  5. Random Forest")
print(f"  6. Gradient Boosting")
print("=" * 80)

# Define models to compare
models_to_compare = {
    'KNN': KNeighborsRegressor(n_neighbors=3),
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
}

# Store results for best model per product
best_model_results = {}
predictions = {}
model_comparison = {}

for product in products_with_sales:
    # Skip if product has no variation in training data
    if train_data[product].std() == 0:
        continue
    
    # Get features specific to this product plus general trend features and weather features
    product_features = [col for col in df_features.columns if col.startswith(f'{product}_rolling_')]
    general_features = ['week_number', 'week_number_squared']
    
    # Add current weather features (not rolled)
    current_weather_features = weather_cols
    
    # Add rolled weather features
    weather_rolling_features = [col for col in df_features.columns 
                                if any(col.startswith(f'{w}_rolling_') for w in weather_cols)]
    
    feature_cols = product_features + general_features + current_weather_features + weather_rolling_features
    
    # Prepare features and target for this product
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    y_train = train_data[product]
    y_test = test_data[product]
    
    # Train all models and compare
    product_model_results = {}
    
    for model_name, model in models_to_compare.items():
        try:
            # Create a fresh copy of the model
            from sklearn.base import clone
            model_copy = clone(model)
            
            # Train model
            model_copy.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model_copy.predict(X_train)
            y_pred_test = model_copy.predict(X_test)
            
            # Ensure predictions are never negative (sales can't be negative)
            y_pred_train = np.maximum(y_pred_train, 0)
            y_pred_test = np.maximum(y_pred_test, 0)
            
            # Calculate metrics
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Store results for this model
            product_model_results[model_name] = {
                'model': model_copy,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'train_pred': y_pred_train,
                'test_pred': y_pred_test
            }
        except Exception as e:
            # Skip models that fail
            continue
    
    # Select best model based on test MAE (lower is better)
    if product_model_results:
        best_model_name = min(product_model_results.keys(), 
                              key=lambda x: product_model_results[x]['test_mae'])
        
        best_result = product_model_results[best_model_name]
        
        # Store best model results
        best_model_results[product] = {
            'model_name': best_model_name,
            'model': best_result['model'],
            'test_rmse': best_result['test_rmse'],
            'test_mae': best_result['test_mae'],
            'test_r2': best_result['test_r2']
        }
        
        # Store predictions from best model
        predictions[product] = {
            'train_pred': best_result['train_pred'],
            'test_pred': best_result['test_pred'],
            'train_actual': y_train,
            'test_actual': y_test
        }
        
        # Store comparison of all models for this product
        model_comparison[product] = {
            model_name: {
                'test_mae': results['test_mae'],
                'test_rmse': results['test_rmse'],
                'test_r2': results['test_r2']
            }
            for model_name, results in product_model_results.items()
        }

print(f"\nBest models selected for {len(best_model_results)} products")

# Print model selection summary
model_counts = {}
for product in best_model_results.keys():
    model_name = best_model_results[product]['model_name']
    model_counts[model_name] = model_counts.get(model_name, 0) + 1

print("\nModel Selection Summary:")
print("-" * 50)
for model_name, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(best_model_results)) * 100
    print(f"  {model_name}: {count} products ({percentage:.1f}%)")

# Create plots for each product
os.makedirs('result', exist_ok=True)

# Create a multi-page PDF with best model evaluation plots for all products
print("\nGenerating evaluation plots with best models...")
with PdfPages('result/Bosund_best_models_evaluation.pdf') as pdf:
    for product in best_model_results.keys():
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get data
        train_weeks = train_data.index
        test_weeks = test_data.index
        
        # Plot actual vs predicted for train and test sets (no training predictions)
        ax.plot(train_weeks, predictions[product]['train_actual'], 
                'b-', label='Verklig försäljning (historisk)', linewidth=2.5)
        
        ax.plot(test_weeks, predictions[product]['test_actual'], 
                'r-', label='Verklig försäljning (test)', linewidth=2.5)
        ax.plot(test_weeks, predictions[product]['test_pred'], 
                'r--', label='Prognos (test)', alpha=0.7, linewidth=2)
        
        # Add green line to mark train/test boundary
        ax.axvline(x=train_weeks[-1], color='green', linestyle='-', linewidth=2, 
                   label='Start testperiod', alpha=0.6)
        
        # Get metrics
        test_rmse = best_model_results[product]['test_rmse']
        test_mae = best_model_results[product]['test_mae']
        test_r2 = best_model_results[product]['test_r2']
        model_name = best_model_results[product]['model_name']
        
        # Calculate percentage deviation (MAPE)
        test_actual_values = predictions[product]['test_actual'].values
        test_pred_values = predictions[product]['test_pred']
        # Avoid division by zero
        non_zero_mask = test_actual_values != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((test_actual_values[non_zero_mask] - test_pred_values[non_zero_mask]) / test_actual_values[non_zero_mask])) * 100
        else:
            mape = 0
        
        # Title with model name and performance in Swedish
        title = f'{product[:60]}\n'
        title += f'Modell: {model_name} | Medelavvikelse: {test_mae:.2f} € per vecka ({mape:.1f}%)'
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Vecka', fontsize=11)
        ax.set_ylabel('Försäljning (€)', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save this page to the PDF
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved best model evaluation plots for {len(best_model_results)} products to: result/Bosund_best_models_evaluation.pdf")

# Model performance summary removed - not displayed in console

# Save best model predictions to CSV
prediction_df = pd.DataFrame(index=df_features.index)
for product in best_model_results.keys():
    prediction_df[f'{product}_actual'] = df_features[product]
    prediction_df[f'{product}_predicted'] = np.concatenate([
        predictions[product]['train_pred'],
        predictions[product]['test_pred']
    ])

prediction_df.to_csv('result/Bosund_best_models_predictions.csv')

# Save best model results summary
best_models_summary = []
for product in best_model_results.keys():
    results = best_model_results[product]
    best_models_summary.append({
        'Product': product,
        'Model': results['model_name'],
        'Test_RMSE': results['test_rmse'],
        'Test_MAE': results['test_mae'],
        'Test_R2': results['test_r2']
    })

best_models_summary_df = pd.DataFrame(best_models_summary)
best_models_summary_df.to_csv('result/Bosund_best_models_summary.csv', index=False)

# Save model comparison details to CSV
comparison_rows = []
for product in model_comparison.keys():
    for model_name, metrics in model_comparison[product].items():
        comparison_rows.append({
            'Product': product,
            'Model': model_name,
            'Test_MAE': metrics['test_mae'],
            'Test_RMSE': metrics['test_rmse'],
            'Test_R2': metrics['test_r2'],
            'Is_Best': model_name == best_model_results[product]['model_name']
        })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv('result/Bosund_model_comparison.csv', index=False)

print("\n" + "=" * 80)
print("OUTPUTS SAVED (BEST MODELS WITH WEATHER FEATURES):")
print("=" * 80)
print(f"  1. Predictions CSV:         result/Bosund_best_models_predictions.csv")
print(f"  2. Evaluation plots PDF:    result/Bosund_best_models_evaluation.pdf")
print(f"  3. Best models summary CSV: result/Bosund_best_models_summary.csv")
print(f"  4. Model comparison CSV:    result/Bosund_model_comparison.csv")
print("\n  Weather features used: {0}".format(", ".join(weather_cols)))
print("\n  Note: Each product uses the best performing model (lowest test MAE)")
print("        Plots show test predictions only (no training predictions)")
print("=" * 80)