# Read data/Bosund_time_series.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
warnings.filterwarnings('ignore')

# Load and prepare the data
df = pd.read_csv('../data/Bosund_time_series.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

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

# Create features for linear regression (no data leakage)
def create_features(df, window_sizes=[2, 4, 8]):
    """
    Create rolling features for time series prediction
    Ensures no data leakage by only using past data
    """
    df_features = df.copy()
    
    # Create rolling features for each product individually
    for product in df.columns:
        for window in window_sizes:
            # Rolling mean
            df_features[f'{product}_rolling_mean_{window}'] = df[product].rolling(window=window, min_periods=1).mean().shift(1)
            # Rolling std
            df_features[f'{product}_rolling_std_{window}'] = df[product].rolling(window=window, min_periods=1).std().shift(1)
            # Rolling max
            df_features[f'{product}_rolling_max_{window}'] = df[product].rolling(window=window, min_periods=1).max().shift(1)
    
    # Trend features
    df_features['week_number'] = range(len(df_features))
    df_features['week_number_squared'] = df_features['week_number'] ** 2
    
    return df_features

# Create features
df_features = create_features(df_weekly)

# Fill NaN values with 0 (from rolling features at the beginning)
df_features = df_features.fillna(0)

# Split data at week boundary (use last 4 weeks as test set)
test_weeks = 4
train_size = len(df_features) - test_weeks

train_data = df_features.iloc[:train_size]
test_data = df_features.iloc[train_size:]

print(f"Train weeks: {len(train_data)}, Test weeks: {len(test_data)}")

# Train models and make predictions for each product
predictions = {}
models = {}

for product in products_with_sales:
    # Skip if product has no variation in training data
    if train_data[product].std() == 0:
        continue
    
    # Get features specific to this product plus general trend features
    product_features = [col for col in df_features.columns if col.startswith(f'{product}_rolling_')]
    general_features = ['week_number', 'week_number_squared']
    feature_cols = product_features + general_features
    
    # Prepare features and target for this product
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    y_train = train_data[product]
    y_test = test_data[product]
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Ensure predictions are never negative (sales can't be negative)
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)
    
    # Store results
    models[product] = model
    predictions[product] = {
        'train_pred': y_pred_train,
        'test_pred': y_pred_test,
        'train_actual': y_train,
        'test_actual': y_test
    }

print(f"Models trained for {len(models)} products")

# Create plots for each product
os.makedirs('result', exist_ok=True)

# Import PdfPages for multi-page PDF
from matplotlib.backends.backend_pdf import PdfPages

# Create a multi-page PDF with all products (no training predictions, but keep test predictions)
print("\nGenerating PDF with test set predictions...")
with PdfPages('result/Bosund_all_products_predictions.pdf') as pdf:
    for product in models.keys():
        fig = plt.figure(figsize=(12, 6))
        
        # Plot training data
        train_weeks = train_data.index
        test_weeks = test_data.index
        
        # Only plot actual values for training (no predicted dashed lines for train)
        plt.plot(train_weeks, predictions[product]['train_actual'], 
                 'b-', label='Verklig försäljning (historisk)', linewidth=2)
        
        # Plot test data actual values AND predictions
        plt.plot(test_weeks, predictions[product]['test_actual'], 
                 'r-', label='Verklig försäljning (test)', linewidth=2)
        plt.plot(test_weeks, predictions[product]['test_pred'], 
                 'r--', label='Prognos (test)', alpha=0.7, linewidth=2)
        
        # Add green line to mark train/test boundary
        plt.axvline(x=train_weeks[-1], color='green', linestyle='-', linewidth=2, 
                    label='Gräns träning/test', alpha=0.6)
        
        plt.title(f'{product} - Veckoförsäljning', fontsize=10)
        plt.xlabel('Vecka')
        plt.ylabel('Försäljning (€)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save this page to the PDF
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved all {len(models)} products to: result/Bosund_all_products_predictions.pdf")

# Save predictions to CSV (but don't print performance comparison)
prediction_df = pd.DataFrame(index=df_features.index)

for product in models.keys():
    prediction_df[f'{product}_actual'] = df_features[product]
    prediction_df[f'{product}_predicted'] = np.concatenate([
        predictions[product]['train_pred'],
        predictions[product]['test_pred']
    ])

prediction_df.to_csv('result/Bosund_weekly_predictions.csv')
print(f"\nPredictions saved to: result/Bosund_weekly_predictions.csv")
print(f"All product plots saved to: result/Bosund_all_products_predictions.pdf")
print("\nNote: PDF shows actual sales data with test set predictions (no training predictions, no performance comparison).")

