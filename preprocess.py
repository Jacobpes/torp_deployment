import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

success = True

try:
    # Read the dataframe
    df = pd.read_csv('data/order_export_20250930T20240405.csv')

    # For each store, create a time series DataFrame where each column is a product (by name)
    # and each cell is the revenue for that product on that day. Fill missing values with 0.

    # Convert 'updated' to datetime and extract date only
    df['date'] = pd.to_datetime(df['updated']).dt.date

    # Get the full date range in the data
    all_dates = pd.date_range(df['date'].min(), df['date'].max())

    # Get all unique stores
    stores = df['store'].unique()

    for store in stores:
        # Filter data for the current store
        store_df = df[df['store'] == store]
        # Pivot: index=date, columns=product name, values=sum of line_price
        pivot = store_df.pivot_table(
            index='date',
            columns='name',
            values='line_price',
            aggfunc='sum',
            fill_value=0
        )
        # Reindex to include all dates, fill missing with 0
        pivot = pivot.reindex(all_dates.date, fill_value=0)
        pivot.index.name = 'date'
        # Save to CSV
        filename = f"data/{store.split()[0]}_time_series.csv"
        pivot.reset_index().to_csv(filename, index=False, sep=';')
except Exception as e:
    print(f"An error occurred: {e}")
    success = False

if success:
    print("Process completed successfully.")
else:
    print("Process failed.")
