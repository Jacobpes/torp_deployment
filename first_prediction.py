# Exploratory Data Analysis - Order Export Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Load the data
df = pd.read_csv('data/order_export_20250930T20240405.csv')

# Drop all rows later than 28.9.2025 to make it last evenly with last day of week 39
df = df[df['updated'] <= '2025-09-29']

# Data Cleaning and Preparation
# Convert updated column to datetime
df['updated'] = pd.to_datetime(df['updated'])

# Drop broken products: SIA Glass KNÄCK 200ml/115g, Häggbloms POTATIS 5 kg, Torp JORDGUBBSLEMONAD 500ml, SataMaito MJÖLK Fettfri 1L
df = df[~df['name'].isin(['SIA Glass KNÄCK 200ml/115g', 'Häggbloms POTATIS 5 kg', 'Torp JORDGUBBSLEMONAD 500ml', 'SataMaito MJÖLK Fettfri 1L'])]

# Extract temporal features
df['date'] = df['updated'].dt.date
df['hour'] = df['updated'].dt.hour
df['day_of_week'] = df['updated'].dt.day_name()
df['month'] = df['updated'].dt.month
df['year'] = df['updated'].dt.year

# Calculate total revenue per order line
df['total_revenue'] = df['quantity'] * df['line_price']

# Predict weekly sales (Monday-Sunday) for each product in Bosund store using linear regression with feature engineering
# Add a green line in the plot to mark the boundary between train and test set (always at a week boundary)
# Ensure no data leakage: rolling features and splits are computed only on past data
# Skip products with 0 sales. Plot by week number (sum of revenue per week).
# Test set is always a whole number of weeks (preferably 4 or 5, but always full weeks, not partial)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def create_features(df):
    """Create features for training data only"""
    df = df.copy()
    df['prev_day_sales'] = df['line_price'].shift(1)
    df['rolling_3d'] = df['line_price'].shift(1).rolling(3).mean()
    df['rolling_7d'] = df['line_price'].shift(1).rolling(7).mean()
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    return df

def create_test_features(train_df, test_df):
    """Create features for test data using only training data to prevent leakage"""
    test_features = test_df.copy()
    
    # Use the last values from training data for rolling features
    test_features['prev_day_sales'] = train_df['line_price'].iloc[-1]  # Last training day
    test_features['rolling_3d'] = train_df['line_price'].tail(3).mean()  # Last 3 days from training
    test_features['rolling_7d'] = train_df['line_price'].tail(7).mean()  # Last 7 days from training
    
    # These don't need special handling
    test_features['day_of_week'] = pd.to_datetime(test_features['date']).dt.dayofweek
    test_features['month'] = pd.to_datetime(test_features['date']).dt.month
    
    return test_features

# Define all stores to process
stores = ['Bosund kiosk', 'Kronoby kiosk', 'Sisbacka kiosk', 'V-lift kiosk', 
          'Sunds kiosk', 'Torp kiosk', 'Granholmen kiosk', 'Holm kiosk', 'Lager kållby']

# Process each store
for store_name in stores:
    print(f"\nProcessing store: {store_name}")
    
    # Filter store data and ensure date is datetime
    store_df = df[df['store'] == store_name].copy()
    store_df['date'] = pd.to_datetime(store_df['updated']).dt.date

    # Find the global last date and align the test set to full weeks (Monday-Sunday)
    global_last_date = store_df['date'].max()
    
    # Check if the week containing the last date is complete
    # Find the Sunday of the week that contains the last date
    last_date_dt = pd.to_datetime(global_last_date)
    last_date_weekday = last_date_dt.weekday()  # Monday=0, Sunday=6
    last_date_sunday = last_date_dt + pd.Timedelta(days=6 - last_date_weekday)
    
    # Check if this week is complete (has data for all 7 days)
    week_start = last_date_sunday - pd.Timedelta(days=6)  # Monday of the week
    week_dates = pd.date_range(week_start, last_date_sunday, freq='D')
    week_data_count = store_df[store_df['date'].isin(week_dates.date)].groupby('date').size().count()
    
    # If the week containing the last date is incomplete, exclude it from test set
    if week_data_count < 7:
        print(f"Week {last_date_sunday.isocalendar().week} (containing last date) is incomplete ({week_data_count}/7 days), excluding from test set")
        test_end_date = last_date_sunday - pd.Timedelta(days=7)  # End at previous Sunday
    else:
        test_end_date = last_date_sunday
    
    # Set test set to be the last N full weeks (e.g. 4 or 5 weeks, but always full weeks)
    n_test_weeks = 4
    test_start_date = test_end_date - pd.Timedelta(days=7 * n_test_weeks - 1)
    test_start_date = test_start_date.date()
    test_end_date = test_end_date.date()

    print(f"Global last date: {global_last_date}")
    print(f"Test set: {n_test_weeks} full weeks, from {test_start_date} to {test_end_date}")

    products = store_df['name'].unique()

    results = []
    plot_products = []
    plot_data = []
    split_lines = []  # To store the week where the train/test split occurs for each product

    for product in products:
        product_df = store_df[store_df['name'] == product].copy()
        # Aggregate daily sales for this product
        daily_product_sales = product_df.groupby('date')['line_price'].sum().reset_index()
        daily_product_sales = daily_product_sales.sort_values('date').reset_index(drop=True)
        
        # Skip if all sales are zero
        if (daily_product_sales['line_price'] == 0).all():
            continue

        # If not enough data, skip
        if len(daily_product_sales) < 10:
            continue

        # Split into train/test FIRST, before any feature engineering
        daily_product_sales['date_dt'] = pd.to_datetime(daily_product_sales['date'])
        train = daily_product_sales[daily_product_sales['date_dt'] < pd.to_datetime(test_start_date)].copy()
        test = daily_product_sales[(daily_product_sales['date_dt'] >= pd.to_datetime(test_start_date)) & (daily_product_sales['date_dt'] <= pd.to_datetime(test_end_date))].copy()
        
        # Skip if not enough data in train or test set
        if len(train) < 10 or len(test) < 7:  # Need at least 10 training days and 1 week of test data
            continue

        # Feature engineering: create features ONLY on training data
        train_features = create_features(train)
        
        # Handle NaN values in training features
        feature_cols = ['prev_day_sales', 'rolling_3d', 'rolling_7d', 'day_of_week', 'month']
        for col in feature_cols:
            if train_features[col].isna().any():
                mean_val = train_features[col].mean()
                train_features[col] = train_features[col].fillna(mean_val)
        
        # Create test features using only training data (no leakage)
        test_features = create_test_features(train, test)
        
        # Prepare features and target
        X_train = train_features[feature_cols].values
        y_train = train_features['line_price'].values
        X_test = test_features[feature_cols].values
        y_test = test_features['line_price'].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({'product': product, 'rmse': rmse})

        # For plotting: group actual and predicted by week number (sum revenue per week)
        # Prepare a DataFrame for all data (train + test) with predictions for test
        all_plot_df = daily_product_sales[['date_dt', 'line_price']].copy()
        all_plot_df['actual'] = all_plot_df['line_price']
        all_plot_df['pred'] = np.nan
        # Fill predictions only for test set dates
        test_date_mask = all_plot_df['date_dt'].isin(test['date_dt'])
        all_plot_df.loc[test_date_mask, 'pred'] = y_pred

        # Add year and week columns
        all_plot_df['year'] = all_plot_df['date_dt'].dt.isocalendar().year
        all_plot_df['week'] = all_plot_df['date_dt'].dt.isocalendar().week

        # Group by year and week for actual and predicted
        weekly_actual = all_plot_df.groupby(['year', 'week'])['actual'].sum().reset_index()
        weekly_pred = all_plot_df.groupby(['year', 'week'])['pred'].sum().reset_index()

        # Merge for plotting
        weekly_plot = pd.merge(weekly_actual, weekly_pred, on=['year', 'week'], how='outer').sort_values(['year', 'week'])
        # For x-axis, use a tuple (year, week) to avoid ambiguity across years
        weekly_plot['year_week'] = weekly_plot.apply(lambda row: f"{int(row['year'])}-W{int(row['week']):02d}", axis=1)

        # Find the week where the test set starts (for green line)
        test_start_year = pd.to_datetime(test_start_date).isocalendar().year
        test_start_week = pd.to_datetime(test_start_date).isocalendar().week
        split_week_label = f"{int(test_start_year)}-W{int(test_start_week):02d}"
        split_lines.append(split_week_label)

        plot_products.append(product)
        plot_data.append((weekly_plot['year_week'], weekly_plot['actual'], weekly_plot['pred'], rmse))

    # Prepare subplots: one for each product that passed the filter
    n_products = len(plot_products)
    if n_products == 0:
        print(f"No products with sufficient data found for {store_name}")
        continue
        
    n_cols = 3
    n_rows = int(np.ceil(n_products / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), sharex=False)
    axes = axes.flatten()

    for i, (product, (year_weeks, actual, pred, rmse)) in enumerate(zip(plot_products, plot_data)):
        ax = axes[i]
        # Plot actuals as a solid blue line
        ax.plot(year_weeks, actual, label='Actual', marker='o', color='tab:blue')
        # Plot predictions as a solid orange line, but only for test weeks, and connect to last train week
        # Find the split index
        split_week = split_lines[i]
        if split_week in list(year_weeks):
            split_idx = list(year_weeks).index(split_week)
            # To connect the last train week to the first test week, build a combined prediction curve
            # The prediction curve is: NaN for train weeks, then last train actual, then test predictions
            pred_curve = [np.nan] * split_idx
            # Connect with the last actual value from train
            if split_idx > 0:
                pred_curve.append(actual.iloc[split_idx-1])
            else:
                pred_curve.append(np.nan)
            # Then the test predictions
            pred_curve += list(pred.iloc[split_idx:])
            # The x-axis for the prediction curve
            pred_x = list(year_weeks[:split_idx+1]) + list(year_weeks[split_idx:])
            ax.plot(pred_x, pred_curve, label='Predicted', marker='o', linestyle='--', color='tab:orange')
            # Add green vertical line at the train/test split
            ax.axvline(x=year_weeks.iloc[split_idx], color='green', linestyle='-', linewidth=2, label='Train/Test Split')
        else:
            # If the split week is not in the x-axis (e.g. no sales that week), just plot as usual
            ax.plot(year_weeks, pred, label='Predicted', marker='o', linestyle='--', color='tab:orange')
        ax.set_title(f"{product}\nRMSE: {rmse:.2f}")
        ax.set_xlabel('Year-Week')
        ax.set_ylabel('Weekly Sales (€)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for j in range(n_products, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'{store_name}: Actual vs Predicted (Linear Regression, NO DATA LEAKAGE) for All Products (nonzero only, weekly sum, full weeks only)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Create safe filename from store name
    safe_store_name = store_name.replace(' ', '_').replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
    plt.savefig(f'plots/{safe_store_name}_actual_vs_predicted_linear_regression_no_data_leakage.pdf')
    plt.close()

    # Show RMSE for each product
    results_df = pd.DataFrame(results).sort_values('rmse')
    print(f"Linear Regression RMSE per product for {store_name} (sorted):")
    print(results_df)

    # For each product in store, fit a linear regression model on all available history, then predict sales for the next 4 full weeks (Monday to Sunday).
    # Plot the full history and the next month's predictions for each product, aggregated by week.
    # Sales can never be negative.

    future_weeks = 4
    products = store_df['name'].unique()
    
    if len(products) == 0:
        print(f"No products found for {store_name}")
        continue
        
    fig, axes = plt.subplots(len(products), 1, figsize=(18, 6 * len(products)), sharex=False)

    if len(products) == 1:
        axes = [axes]

    for idx, product in enumerate(products):
        prod_df = store_df[store_df['name'] == product].copy()
        daily = prod_df.groupby('date')['line_price'].sum().reset_index()
        daily = daily.sort_values('date')
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.set_index('date').asfreq('D', fill_value=0).reset_index()

        # Feature engineering
        daily['dayofweek'] = daily['date'].dt.dayofweek
        daily['month'] = daily['date'].dt.month
        daily['day'] = daily['date'].dt.day
        daily['is_weekend'] = daily['dayofweek'].isin([5, 6]).astype(int)

        # Prepare X, y
        X = daily[['dayofweek', 'month', 'day', 'is_weekend']]
        y = daily['line_price']

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Prepare future dates: always full weeks (Monday to Sunday), for the next 4 weeks
        last_date = daily['date'].max()
        # Find the next Monday after last_date (if last_date is Monday, start from next week)
        days_ahead = (7 - last_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        first_future_monday = last_date + pd.Timedelta(days=days_ahead)
        total_future_days = future_weeks * 7
        future_dates = pd.date_range(first_future_monday, periods=total_future_days, freq='D')
        future_df = pd.DataFrame({'date': future_dates})
        future_df['dayofweek'] = future_df['date'].dt.dayofweek
        future_df['month'] = future_df['date'].dt.month
        future_df['day'] = future_df['date'].dt.day
        future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
        X_future = future_df[['dayofweek', 'month', 'day', 'is_weekend']]

        # Predict future, ensure no negative sales
        y_future = model.predict(X_future)
        y_future = np.maximum(y_future, 0)

        # Combine history and future for weekly aggregation
        history_df = daily[['date', 'line_price']].copy()
        history_df['type'] = 'History'
        history_df = history_df.rename(columns={'line_price': 'sales'})

        future_df['sales'] = y_future
        future_df['type'] = 'Forecast'
        combined_df = pd.concat([history_df, future_df[['date', 'sales', 'type']]], ignore_index=True)

        # Add year and week columns for grouping
        combined_df['year'] = combined_df['date'].dt.isocalendar().year
        combined_df['week'] = combined_df['date'].dt.isocalendar().week

        # Aggregate by week and type
        weekly = combined_df.groupby(['year', 'week', 'type'])['sales'].sum().reset_index()
        # For x-axis, use a tuple (year, week) to avoid ambiguity across years
        weekly['year_week'] = weekly.apply(lambda row: f"{int(row['year'])}-W{int(row['week']):02d}", axis=1)

        # Split history and forecast for plotting
        weekly_history = weekly[weekly['type'] == 'History']
        weekly_forecast = weekly[weekly['type'] == 'Forecast']

        # Find the first forecast week that is not already in history
        history_weeks = set(weekly_history['year_week'])
        forecast_weeks = list(weekly_forecast['year_week'])
        forecast_sales = list(weekly_forecast['sales'])

        ax = axes[idx]
        # Plot history
        ax.plot(weekly_history['year_week'], weekly_history['sales'], label='History (weekly sum)', marker='o', color='tab:blue')

        # Connect the last history week to the first forecast week with a "constant" line
        if forecast_weeks:
            last_hist_week = weekly_history['year_week'].iloc[-1]
            last_hist_sales = weekly_history['sales'].iloc[-1]
            first_forecast_week = forecast_weeks[0]
            first_forecast_sales = forecast_sales[0]

            # If the first forecast week is the same as the last history week, just plot the forecast as usual
            if first_forecast_week == last_hist_week:
                # Plot a horizontal line (constant) from last history week to first forecast week (which are the same)
                # Plot forecast points, skipping the first (overlap) week for orange dots, but keep the constant line
                plot_forecast_weeks = forecast_weeks[1:]
                plot_forecast_sales = forecast_sales[1:]
                # Draw a constant line from last history week to first forecast week (which are the same)
                ax.plot([last_hist_week, first_forecast_week], [last_hist_sales, first_forecast_sales],
                        color='tab:orange', linestyle='--', marker='o', label='Forecast (weekly sum)')
                # If more forecast weeks, draw a constant line from first forecast week to the next, and so on
                prev_week = first_forecast_week
                prev_sales = first_forecast_sales
                for w, s in zip(plot_forecast_weeks, plot_forecast_sales):
                    ax.plot([prev_week, w], [prev_sales, s], color='tab:orange', linestyle='--', marker='o')
                    prev_week = w
                    prev_sales = s
            else:
                # Draw a constant line from last history week to first forecast week
                ax.plot([last_hist_week, first_forecast_week], [last_hist_sales, first_forecast_sales],
                        color='tab:orange', linestyle='--', marker='o', label='Forecast (weekly sum)')
                # Then for the rest of the forecast weeks, draw constant lines between each week
                prev_week = first_forecast_week
                prev_sales = first_forecast_sales
                for w, s in zip(forecast_weeks[1:], forecast_sales[1:]):
                    ax.plot([prev_week, w], [prev_sales, s], color='tab:orange', linestyle='--', marker='o')
                    prev_week = w
                    prev_sales = s

        ax.set_title(f"{product} - {store_name}: Weekly History and Next 4 Weeks Forecast")
        ax.set_xlabel('Year-Week')
        ax.set_ylabel('Weekly Sales (€)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    # Create safe filename from store name
    safe_store_name = store_name.replace(' ', '_').replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
    plt.savefig(f'plots/{safe_store_name}_future_4_weeks_forecast.pdf')
    plt.close()