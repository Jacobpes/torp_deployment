"""
Grafisk plocklista per shop - en produkt per sida med försäljningshistorik

Detta script:
1. Läser plocklistan från picking_list_results.csv
2. Hämtar historisk försäljningsdata för varje produkt
3. Skapar en grafisk PDF per butik med en produkt per sida
4. Varje sida visar:
   - Produktinformation
   - Ett års försäljningshistorik (veckovis)
- Prognostiserad försäljning för nästa vecka
   - Nuvarande lager och behov av påfyllning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsRegressor
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    _ADVANCED_ML = True
except ImportError:
    _ADVANCED_ML = False
import os
import sys
import glob
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Hitta projektets rot-katalog (en nivå upp från scripts/ om scriptet är i scripts/)
# When running as executable, use sys.executable to find where .exe is located
if getattr(sys, 'frozen', False):
    # Use sys.executable to find where .exe file is located (same as main.py)
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
else:
    SCRIPT_DIR = Path(__file__).parent.resolve()
    if SCRIPT_DIR.name == 'scripts':
        PROJECT_ROOT = SCRIPT_DIR.parent
    else:
        PROJECT_ROOT = SCRIPT_DIR

# Konfiguration - använd absoluta sökvägar
# When running as executable, use organized folder structure
if getattr(sys, 'frozen', False):
    PICKING_LIST_PATH = PROJECT_ROOT / 'output' / 'plocklistor' / 'picking_list_results.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'output' / 'plocklistor'  # Save in output/plocklistor subdirectory
    DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'  # Downloads in data/nedladdningar
    LEVERANSFREKVENS_PATH = PROJECT_ROOT / 'data' / 'parametrar' / 'Leveransfrekvens.csv'
else:
    PICKING_LIST_PATH = PROJECT_ROOT / 'picking_list_results.csv'
    OUTPUT_DIR = PROJECT_ROOT / 'picking_list_graphical'
    DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'
    LEVERANSFREKVENS_PATH = PROJECT_ROOT / 'data' / 'parametrar' / 'Leveransfrekvens.csv'
FORECAST_WEEKS = 4  # Antal veckor att visa prognos för i grafen (för visning)

def find_latest_file(pattern, directory):
    """
    Hittar den senaste filen baserat på datum i filnamnet.
    Pattern ska vara t.ex. 'product_sales_*.csv' eller 'stock_report_*.csv'
    """
    directory = Path(directory)
    files = glob.glob(str(directory / pattern))
    if not files:
        raise FileNotFoundError(f"Inga filer matchar mönstret {pattern} i {directory}")
    
    # Extrahera datum från filnamn och sortera
    file_dates = []
    for file in files:
        # Försök hitta datum i formatet YYYY-MM-DD eller YYYY-MM-DD_to_YYYY-MM-DD
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(file))
        if date_match:
            # Om det finns två datum (t.ex. 2024-02-12_to_2025-12-09), ta det sista
            all_dates = re.findall(r'(\d{4}-\d{2}-\d{2})', os.path.basename(file))
            if all_dates:
                latest_date_in_filename = max(all_dates)
                file_dates.append((latest_date_in_filename, file))
    
    if not file_dates:
        raise ValueError(f"Kunde inte hitta datum i filnamn för {pattern}")
    
    # Sortera efter datum (senaste först)
    file_dates.sort(key=lambda x: x[0], reverse=True)
    latest_file = file_dates[0][1]
    print(f"  Hittade senaste fil: {os.path.basename(latest_file)} (datum: {file_dates[0][0]})")
    return latest_file

def load_leveransfrekvens(file_path):
    """Läser Leveransfrekvens.csv och returnerar en dictionary med leveransfrekvens per butik"""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Varning: Leveransfrekvens-fil saknas: {file_path}")
        print("  Använder standardvärde: 7 dagar för alla butiker")
        return {}  # Returnera tom dict, scriptet använder default 7 dagar
    
    try:
        # Läs CSV med utf-8-sig för att hantera BOM om det finns
        df = pd.read_csv(str(file_path), sep=';', encoding='utf-8-sig')
        
        print(f"  Hittade kolumner: {list(df.columns)}")
        
        # Hitta rätt kolumnnamn (kan vara 'store' eller 'Store' etc.)
        store_col = None
        freq_col = None
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'store' in col_lower or 'butik' in col_lower:
                store_col = col
            if 'leveransfrekvens' in col_lower or 'frekvens' in col_lower:
                freq_col = col
        
        if store_col is None or freq_col is None:
            # Fallback: använd första och andra kolumnen
            if len(df.columns) >= 2:
                store_col = df.columns[0]
                freq_col = df.columns[1]
                print(f"  Varning: Kunde inte hitta kolumnnamn, använder: '{store_col}' och '{freq_col}'")
            else:
                print(f"  Fel: CSV-filen har för få kolumner. Förväntar minst 2 kolumner.")
                return {}
        
        # Skapa dictionary: butik -> leveransfrekvens_dagar
        parametrar = {}
        for _, row in df.iterrows():
            butik_raw = row[store_col]
            if pd.notna(butik_raw):
                butik = str(butik_raw).strip()
                if butik:  # Hoppa över tomma rader
                    try:
                        freq_raw = row[freq_col]
                        if pd.notna(freq_raw):
                            # Ta bort eventuella icke-numeriska tecken
                            freq_str = str(freq_raw).strip()
                            freq_clean = ''.join(c for c in freq_str if c.isdigit() or c == '.')
                            if freq_clean:
                                leveransfrekvens = int(float(freq_clean))
                                parametrar[butik] = leveransfrekvens
                    except (ValueError, TypeError) as e:
                        print(f"  Varning: Kunde inte tolka frekvens för '{butik}': {freq_raw}")
        
        print(f"Laddade leveransfrekvenser för {len(parametrar)} butiker")
        if len(parametrar) > 0:
            print(f"  Exempel: {list(parametrar.items())[:3]}")
        return parametrar
    except Exception as e:
        print(f"Varning: Kunde inte ladda leveransfrekvenser: {e}")
        print("  Använder standardvärde: 7 dagar för alla butiker")
        import traceback
        traceback.print_exc()
        return {}

def load_and_prepare_sales_data(file_path):
    """Läser och förbereder försäljningsdata"""
    file_path = Path(file_path)
    print(f"Läser försäljningsdata från {file_path}...")
    df = pd.read_csv(str(file_path))
    
    # Kontrollera vilken struktur filen har och mappa kolumner därefter
    if 'period' in df.columns:
        # Ny struktur med period-baserad data
        df = df.rename(columns={
            'period': 'date',
            'store_name': 'store',
            'product_name': 'name',
            'total_quantity_sold': 'quantity',
            'total_sales': 'line_price'
        })
        # Konvertera datum från period-kolumnen
        df['date'] = pd.to_datetime(df['date']).dt.date
    elif 'created_at' in df.columns:
        # Gammal struktur med individuella orderrader
        df = df.rename(columns={
            'created_at': 'updated',
            'store_name': 'store',
            'product_name': 'name',
            'line_total': 'line_price'
        })
        # Konvertera datum
        df['updated'] = pd.to_datetime(df['updated'])
        df['date'] = df['updated'].dt.date
    else:
        # Försök hitta datumkolumn
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'created' in col.lower() or 'period' in col.lower()]
        if date_cols:
            df['date'] = pd.to_datetime(df[date_cols[0]]).dt.date
        else:
            raise ValueError(f"Kunde inte hitta datumkolumn i filen. Tillgängliga kolumner: {list(df.columns)}")
        
        # Mappa övriga kolumner om de finns
        if 'store_name' in df.columns:
            df['store'] = df['store_name']
        if 'product_name' in df.columns:
            df['name'] = df['product_name']
        if 'quantity' not in df.columns:
            # Försök hitta quantity-kolumn
            qty_cols = [col for col in df.columns if 'quantity' in col.lower()]
            if qty_cols:
                df['quantity'] = df[qty_cols[0]]
            else:
                raise ValueError(f"Kunde inte hitta quantity-kolumn i filen. Tillgängliga kolumner: {list(df.columns)}")
    
    # Behåll unit-kolumnen om den finns, annars sätt standard
    if 'unit' not in df.columns:
        df['unit'] = 'st'  # Standardenhet om kolumnen saknas
    
    # Filtrera bort order med status "error" om order_status kolumnen finns
    if 'order_status' in df.columns:
        initial_count = len(df)
        df = df[df['order_status'] == 'complete'].copy()
        filtered_count = len(df)
        if initial_count != filtered_count:
            print(f"  Filtrerade bort {initial_count - filtered_count} rader med order_status != 'complete'")
    
    # Säkerställ att nödvändiga kolumner finns
    required_cols = ['date', 'store', 'name', 'quantity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Saknade kolumner: {missing_cols}. Tillgängliga kolumner: {list(df.columns)}")
    
    print(f"  Laddat {len(df)} rader")
    print(f"  Butiker: {df['store'].nunique()}")
    print(f"  Produkter: {df['name'].nunique()}")
    
    return df

def _empty_forecast_df(first_future_monday, future_weeks):
    """Returns a forecast DataFrame filled with zeros."""
    rows = []
    for i in range(future_weeks):
        ws = first_future_monday + pd.Timedelta(weeks=i)
        iso = ws.isocalendar()
        rows.append({
            'week_start': ws,
            'quantity': 0.0,
            'week_number': iso.week,
            'year': iso.year,
            'week_year': f"{iso.year}-W{int(iso.week):02d}"
        })
    return pd.DataFrame(rows)


def _forecast_weeks_short_history(daily_sales, first_future_monday, future_weeks, ref_date):
    """
    For products with < 5 months history.
    Uses KNN (k=min(4, n_complete_weeks), weights='distance') on weekly
    aggregated data with features (week_of_year, month) so each future week
    gets the weighted average of the 4 closest historical weeks -- giving
    a slightly different prediction per week.
    """
    ds = daily_sales.copy()
    ds['week'] = ds['date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = ds.groupby('week')['quantity'].sum().reset_index()
    weekly = weekly.sort_values('week')

    current_week_start = ref_date - pd.Timedelta(days=ref_date.dayofweek)
    complete = weekly[weekly['week'] < current_week_start].copy()

    if len(complete) == 0:
        complete = weekly.copy()
    if len(complete) == 0:
        return _empty_forecast_df(first_future_monday, future_weeks)

    complete['week_of_year'] = complete['week'].dt.isocalendar().week.astype(int)
    complete['month'] = complete['week'].dt.month

    k = min(4, len(complete))
    model = KNeighborsRegressor(n_neighbors=k, weights='distance')
    model.fit(complete[['week_of_year', 'month']], complete['quantity'])

    rows = []
    for i in range(future_weeks):
        ws = first_future_monday + pd.Timedelta(weeks=i)
        iso = ws.isocalendar()
        pred = max(0.0, float(model.predict([[int(iso.week), ws.month]])[0]))
        rows.append({
            'week_start': ws,
            'quantity': pred,
            'week_number': iso.week,
            'year': iso.year,
            'week_year': f"{iso.year}-W{int(iso.week):02d}"
        })
    return pd.DataFrame(rows)


def _compute_mae(y_true, y_pred):
    """Mean absolute error that works even without sklearn.metrics."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def _forecast_weeks_best_model(daily_sales, first_future_monday, future_weeks, today):
    """
    For products with >= 5 months history.
    Tests KNN, Linear, Ridge, RandomForest, GradientBoosting against a
    4-week-average baseline on a 28-day validation window, picks the winner,
    then predicts day-by-day for each future week and aggregates to weekly
    totals -- naturally producing different values per week.
    Falls back to KNN-only when advanced sklearn modules are unavailable.
    """
    ds = daily_sales.copy()

    ds['dayofweek'] = ds['date'].dt.dayofweek
    ds['month'] = ds['date'].dt.month
    ds['day'] = ds['date'].dt.day
    ds['weekofyear'] = ds['date'].dt.isocalendar().week.astype(int)
    ds['is_weekend'] = ds['dayofweek'].isin([5, 6]).astype(int)

    feature_cols = ['dayofweek', 'month', 'day', 'weekofyear', 'is_weekend']
    X = ds[feature_cols]
    y = ds['quantity']

    val_days = min(28, len(ds) // 4)
    if val_days < 7:
        return _forecast_weeks_short_history(daily_sales, first_future_monday, future_weeks, today)

    X_train, X_val = X.iloc[:-val_days], X.iloc[-val_days:]
    y_train, y_val = y.iloc[:-val_days], y.iloc[-val_days:]

    # Baseline: flat daily rate from the 4 latest complete weeks in training data
    train_ds = ds.iloc[:-val_days].copy()
    train_ds['week'] = train_ds['date'].dt.to_period('W').apply(lambda r: r.start_time)
    train_weekly = train_ds.groupby('week')['quantity'].sum()
    train_end = train_ds['date'].iloc[-1]
    cws = train_end - pd.Timedelta(days=train_end.dayofweek)
    complete_train_weeks = train_weekly[train_weekly.index < cws]
    if len(complete_train_weeks) > 0:
        baseline_daily = complete_train_weeks.tail(min(4, len(complete_train_weeks))).mean() / 7.0
    elif len(train_weekly) > 0:
        baseline_daily = train_weekly.mean() / 7.0
    else:
        baseline_daily = 0.0
    baseline_mae = _compute_mae(y_val, [baseline_daily] * len(y_val))

    candidates = {
        'knn_3': KNeighborsRegressor(n_neighbors=3),
        'knn_5': KNeighborsRegressor(n_neighbors=5),
        'knn_7': KNeighborsRegressor(n_neighbors=7),
    }
    if _ADVANCED_ML:
        candidates['linear'] = LinearRegression()
        candidates['ridge'] = Ridge(alpha=1.0)
        candidates['rf'] = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        candidates['gb'] = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)

    best_mae = baseline_mae
    best_model = None

    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            preds = np.maximum(model.predict(X_val), 0)
            mae = _compute_mae(y_val, preds)
            if mae < best_mae:
                best_mae = mae
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return _forecast_weeks_short_history(daily_sales, first_future_monday, future_weeks, today)

    best_model.fit(X, y)

    total_days = future_weeks * 7
    future_dates = pd.date_range(first_future_monday, periods=total_days, freq='D')
    fdf = pd.DataFrame({'date': future_dates})
    fdf['dayofweek'] = fdf['date'].dt.dayofweek
    fdf['month'] = fdf['date'].dt.month
    fdf['day'] = fdf['date'].dt.day
    fdf['weekofyear'] = fdf['date'].dt.isocalendar().week.astype(int)
    fdf['is_weekend'] = fdf['dayofweek'].isin([5, 6]).astype(int)

    fdf['quantity'] = np.maximum(best_model.predict(fdf[feature_cols]), 0)

    fdf['week_start'] = fdf['date'] - pd.to_timedelta(fdf['date'].dt.dayofweek, unit='D')
    weekly = fdf.groupby('week_start')['quantity'].sum().reset_index()
    weekly = weekly.sort_values('week_start')
    weekly['week_number'] = weekly['week_start'].dt.isocalendar().week
    weekly['year'] = weekly['week_start'].dt.isocalendar().year
    weekly['week_year'] = weekly.apply(
        lambda r: f"{int(r['year'])}-W{int(r['week_number']):02d}", axis=1
    )
    return weekly


def predict_weekly_sales(product_df, future_weeks=4):
    """
    Prognostiserar veckovis försäljning per future week.

    - < 5 months history  → KNN (k=4) on weekly data: each future week
      is the distance-weighted average of the 4 closest historical weeks
    - >= 5 months history → best daily model from comparison, day-by-day
      predictions aggregated to weekly totals (naturally different per week)

    Returns DataFrame: week_start, quantity, week_number, year, week_year
    """
    daily_sales = product_df.groupby('date')['quantity'].sum().reset_index()
    daily_sales = daily_sales.sort_values('date')
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)

    today = pd.Timestamp.now().normalize()
    first_date = daily_sales['date'].min()

    date_range = pd.date_range(start=first_date, end=today, freq='D')
    daily_sales = daily_sales.set_index('date').reindex(date_range, fill_value=0).reset_index()
    daily_sales = daily_sales.rename(columns={'index': 'date'})
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)

    days_with_data = len(daily_sales)
    months_with_data = days_with_data / 30.0

    tomorrow = today + pd.Timedelta(days=1)
    days_until_monday = (7 - tomorrow.weekday()) % 7
    first_future_monday = tomorrow if days_until_monday == 0 else tomorrow + pd.Timedelta(days=days_until_monday)

    if months_with_data < 5:
        return _forecast_weeks_short_history(daily_sales, first_future_monday, future_weeks, today)
    else:
        return _forecast_weeks_best_model(daily_sales, first_future_monday, future_weeks, today)

def build_weekly_forecast_from_predicted_sales(predicted_sales_units, delivery_frequency_days, future_weeks=4):
    """
    Skapar en veckovis prognos baserat på plocklistans prognosvärde.
    predicted_sales_units är total prognos för delivery_frequency_days dagar.
    """
    try:
        if pd.isna(predicted_sales_units):
            return pd.DataFrame(columns=['week_start', 'quantity', 'week_number', 'year', 'week_year'])
        predicted_sales_units = float(predicted_sales_units)
    except (TypeError, ValueError):
        return pd.DataFrame(columns=['week_start', 'quantity', 'week_number', 'year', 'week_year'])
    
    try:
        delivery_frequency_days = float(delivery_frequency_days)
    except (TypeError, ValueError):
        delivery_frequency_days = 7.0
    
    if delivery_frequency_days <= 0:
        delivery_frequency_days = 7.0
    
    # Konvertera prognos för leveransperioden till veckovärde
    weekly_value = predicted_sales_units * (7.0 / delivery_frequency_days)
    
    forecast_df = pd.DataFrame(columns=['week_start', 'quantity', 'week_number', 'year', 'week_year'])
    today = pd.Timestamp.now().normalize()
    tomorrow = today + pd.Timedelta(days=1)
    days_until_monday = (7 - tomorrow.weekday()) % 7
    if days_until_monday == 0:
        first_future_monday = tomorrow
    else:
        first_future_monday = tomorrow + pd.Timedelta(days=days_until_monday)
    
    for week_num in range(future_weeks):
        week_start = first_future_monday + pd.Timedelta(weeks=week_num)
        iso_week = week_start.isocalendar().week
        iso_year = week_start.isocalendar().year
        forecast_df = pd.concat([forecast_df, pd.DataFrame({
            'week_start': [week_start],
            'quantity': [weekly_value],
            'week_number': [iso_week],
            'year': [iso_year],
            'week_year': [f"{iso_year}-W{int(iso_week):02d}"]
        })], ignore_index=True)
    
    return forecast_df

def get_weekly_sales_history(product_df, store_name, product_name, weeks_back=52):
    """
    Hämtar veckovis försäljningshistorik för en produkt.
    Returnerar DataFrame med veckonummer och försäljning.
    Säkerställer att alla veckor är representerade (sätter till 0 om saknas).
    """
    # Filtrera på produkt och butik
    product_sales = product_df[
        (product_df['store'] == store_name) & 
        (product_df['name'] == product_name)
    ].copy()
    
    if len(product_sales) == 0:
        return pd.DataFrame(columns=['week_start', 'quantity', 'week_number', 'year', 'week_year'])
    
    # Konvertera datum
    product_sales['date'] = pd.to_datetime(product_sales['date'])
    
    # Beräkna veckostart (måndag)
    product_sales['week_start'] = product_sales['date'] - pd.to_timedelta(
        product_sales['date'].dt.dayofweek, unit='D'
    )
    
    # Beräkna veckonummer (ISO veckonummer)
    product_sales['week_number'] = product_sales['week_start'].dt.isocalendar().week
    product_sales['year'] = product_sales['week_start'].dt.isocalendar().year
    
    # Skapa kombinerad vecka-år kolumn för att hantera veckor över årsskiften
    product_sales['week_year'] = product_sales['year'].astype(str) + '-W' + product_sales['week_number'].astype(str).str.zfill(2)
    
    # Aggregera per vecka
    weekly_sales = product_sales.groupby(['week_start', 'week_number', 'year', 'week_year'])['quantity'].sum().reset_index()
    weekly_sales = weekly_sales.sort_values('week_start')
    
    # Begränsa till senaste X veckor
    if len(weekly_sales) > 0:
        latest_week = weekly_sales['week_start'].max()
        cutoff_date = latest_week - pd.Timedelta(weeks=weeks_back)
        weekly_sales = weekly_sales[weekly_sales['week_start'] >= cutoff_date]
        
        # Skapa komplett serie av veckor från cutoff till latest
        all_weeks = pd.date_range(start=cutoff_date, end=latest_week, freq='W-MON')
        # Konvertera till Series för att använda isocalendar()
        all_weeks_series = pd.Series(all_weeks)
        week_numbers = all_weeks_series.dt.isocalendar().week
        years = all_weeks_series.dt.isocalendar().year
        # Skapa week_year-strängar
        week_year_str = [f"{y}-W{w:02d}" for y, w in zip(years.values, week_numbers.values)]
        all_weeks_df = pd.DataFrame({
            'week_start': all_weeks,
            'week_number': week_numbers.values,
            'year': years.values,
            'week_year': week_year_str
        })
        
        # Merge med faktisk försäljning, fyll saknade veckor med 0
        weekly_sales = all_weeks_df.merge(
            weekly_sales[['week_start', 'quantity']], 
            on='week_start', 
            how='left'
        )
        weekly_sales['quantity'] = weekly_sales['quantity'].fillna(0)
        
        # Sortera efter veckostart
        weekly_sales = weekly_sales.sort_values('week_start')
    
    return weekly_sales

def create_product_page(fig, sales_history, product_info, predicted_sales, weekly_forecast, unit='st'):
    """
    Skapar en sida för en produkt med graf och information.
    weekly_forecast: DataFrame med veckovis prognos för 4 veckor framåt
    """
    fig.clear()
    
    # Skapa layout med subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 2.5, 1], hspace=0.35, wspace=0.3)
    
    # Titelområde
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # Huvudgraf
    ax_main = fig.add_subplot(gs[1, :])
    
    # Info-boxar
    ax_info1 = fig.add_subplot(gs[2, 0])
    ax_info1.axis('off')
    
    ax_info2 = fig.add_subplot(gs[2, 1])
    ax_info2.axis('off')
    
    # Produktinformation
    product_name = product_info['product_name']
    product_code = product_info.get('product_code', 'N/A')
    if pd.isna(product_code) or product_code == '':
        product_code = 'N/A'
    store_name = product_info['store_name']
    
    # Beräkna nästa vecka (måndag) för plocklistan
    # Använd första veckan från weekly_forecast om tillgänglig, annars beräkna från sales_history
    next_week_label = "N/A"
    if len(weekly_forecast) > 0:
        first_forecast_week = weekly_forecast['week_start'].min()
        next_week_number = first_forecast_week.isocalendar().week
        next_week_year = first_forecast_week.isocalendar().year
        next_week_label = f"{next_week_year}-W{next_week_number:02d}"
    elif len(sales_history) > 0:
        last_date = sales_history['week_start'].max()
        days_ahead = (7 - last_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_week_monday = last_date + pd.Timedelta(days=days_ahead)
        next_week_number = next_week_monday.isocalendar().week
        next_week_year = next_week_monday.isocalendar().year
        next_week_label = f"{next_week_year}-W{next_week_number:02d}"
    
    # Titel med veckoinformation för plocklistan
    title_text = f"{product_name}\n{store_name}\nPlocklista för vecka: {next_week_label}"
    ax_title.text(0.5, 0.5, title_text, 
                  ha='center', va='center', 
                  fontsize=14, fontweight='bold',
                  wrap=True)
    
    # Rita försäljningshistorik
    if len(sales_history) > 0:
        # Sortera efter veckostart
        sales_history = sales_history.sort_values('week_start')
        
        # Använd veckonummer för x-axel (skapa en index för positionering)
        x_positions = range(len(sales_history))
        x_labels = sales_history['week_year'].values
        
        # Rita historisk försäljning med fylld yta under
        ax_main.fill_between(x_positions, 0, sales_history['quantity'], 
                            alpha=0.3, color='blue', label='Historisk försäljning')
        ax_main.plot(x_positions, sales_history['quantity'], 
                    'b-', linewidth=2.5, marker='o', markersize=5, zorder=3)
        
        # Rita prognostiserad försäljning för 4 veckor framåt
        if len(weekly_forecast) > 0:
            # Sortera prognos efter veckostart
            weekly_forecast = weekly_forecast.sort_values('week_start')
            
            # Beräkna x-positioner för prognoserna (efter historiken)
            forecast_x_start = len(x_positions)
            forecast_x_positions = range(forecast_x_start, forecast_x_start + len(weekly_forecast))
            forecast_x_labels = weekly_forecast['week_year'].values
            
            # Rita prognoslinje med markerade punkter
            ax_main.plot(forecast_x_positions, weekly_forecast['quantity'], 
                        'r-', linewidth=2.5, marker='o', markersize=8, 
                        label=f'Prognos ({FORECAST_WEEKS} veckor)', 
                        zorder=5, markeredgecolor='darkred', markeredgewidth=2)
            
            # Lägg till text vid varje prognospunkt
            for i, (x_pos, row) in enumerate(zip(forecast_x_positions, weekly_forecast.itertuples())):
                ax_main.annotate(f'{row.quantity:.1f}', 
                               xy=(x_pos, row.quantity),
                               xytext=(0, 15), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='red',
                               ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Vertikal linje för att markera början av framtiden
            ax_main.axvline(x=forecast_x_start - 0.5, color='r', linestyle='--', 
                          alpha=0.5, linewidth=2, zorder=1)
            
            # Kombinera alla x-labels (historia + prognos)
            all_x_positions = list(x_positions) + list(forecast_x_positions)
            all_x_labels = list(x_labels) + list(forecast_x_labels)
        else:
            all_x_positions = x_positions
            all_x_labels = x_labels
        
        # Formatera x-axel med veckonummer
        # Visa var 4:e vecka för att undvika överfyllning
        step = max(1, len(all_x_positions) // 20)  # Max 20 etiketter
        visible_ticks = all_x_positions[::step]
        visible_labels = all_x_labels[::step]
        ax_main.set_xticks(visible_ticks)
        ax_main.set_xticklabels(visible_labels, rotation=45, ha='right')
        
        ax_main.set_xlabel('Vecka', fontsize=12, fontweight='bold')
        ax_main.set_ylabel(f'Försäljning ({unit})', fontsize=12, fontweight='bold')
        ax_main.set_title(f'Försäljningshistorik (senaste 52 veckor) och prognos ({FORECAST_WEEKS} veckor framåt)', 
                         fontsize=13, fontweight='bold', pad=15)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Sätt y-axel minimum till 0
        y_max_history = sales_history['quantity'].max() if len(sales_history) > 0 else 0
        y_max_forecast = weekly_forecast['quantity'].max() if len(weekly_forecast) > 0 else 0
        y_max = max(y_max_history, y_max_forecast, predicted_sales if predicted_sales > 0 else 0)
        ax_main.set_ylim(bottom=0, top=max(y_max * 1.1, 1))
    else:
        ax_main.text(0.5, 0.5, 'Ingen försäljningshistorik tillgänglig', 
                    ha='center', va='center', fontsize=14, style='italic')
        ax_main.set_title('Försäljningshistorik', fontsize=13, fontweight='bold')
        ax_main.axis('off')
    
    # Info-box 1: Lagerstatus
    current_stock = product_info['current_stock']
    stock_warning_limit = product_info['stock_warning_limit']
    expected_stock_after = product_info['expected_stock_after_sales']
    fill_up = product_info['fill_up_quantity']
    
    info_text1 = f"""LAGERSTATUS

Nuvarande lager: {current_stock:.1f} {unit}
Varningsgräns: {stock_warning_limit:.1f} {unit}
Förväntat lager efter försäljning: {expected_stock_after:.1f} {unit}
Prognostiserad försäljning (vecka {next_week_label}): {predicted_sales:.1f} {unit}"""
    
    ax_info1.text(0.05, 0.95, info_text1, 
                  ha='left', va='top', 
                  fontsize=10, family='monospace',
                  transform=ax_info1.transAxes,
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=10))
    
    # Info-box 2: Påfyllningsbehov
    needs_refill = product_info['needs_refill']
    status_color = 'lightcoral' if needs_refill else 'lightgreen'
    status_text = 'BEHÖVER PÅFYLLNING' if needs_refill else 'LAGER OK'
    
    info_text2 = f"""PÅFYLLNING

Behöver fylla på: {fill_up:.1f} {unit}
Status: {status_text}
Produktkod: {product_code}"""
    
    ax_info2.text(0.05, 0.95, info_text2, 
                  ha='left', va='top', 
                  fontsize=10, family='monospace',
                  transform=ax_info2.transAxes,
                  bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.7, pad=10))

def generate_graphical_picking_list(picking_list_df, sales_df):
    """
    Genererar grafiska PDF:er för varje butik.
    """
    import os
    
    # Skapa output-katalog om den inte finns
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Gruppera efter butik
    stores = picking_list_df['store_name'].unique()
    
    for store_name in stores:
        print(f"\nGenererar grafisk plocklista för: {store_name}")
        
        # Filtrera produkter för denna butik
        store_products = picking_list_df[picking_list_df['store_name'] == store_name].copy()
        
        # Beräkna needs_refill baserat på Påfyllningsbehov > 0
        if 'Påfyllningsbehov' in store_products.columns:
            store_products['needs_refill'] = store_products['Påfyllningsbehov'] > 0
        else:
            store_products['needs_refill'] = False
        
        # Sortera efter behov av påfyllning (först de som behöver mest)
        # Använd Påfyllningsbehov istället för fill_up_quantity
        sort_cols = ['needs_refill']
        if 'Påfyllningsbehov' in store_products.columns:
            sort_cols.append('Påfyllningsbehov')
        store_products = store_products.sort_values(sort_cols, ascending=[False, False])
        
        # Skapa PDF-filnamn (ersätt specialtecken)
        safe_store_name = store_name.replace('/', '_').replace('\\', '_')
        pdf_filename = OUTPUT_DIR / f'Plocklista_{safe_store_name}.pdf'
        
        print(f"  Skapar PDF: {pdf_filename}")
        print(f"  Antal produkter: {len(store_products)}")
        
        # Skapa PDF med matplotlib
        with PdfPages(str(pdf_filename)) as pdf:
            for idx, (_, product_row) in enumerate(store_products.iterrows(), 1):
                # Använd rätt kolumnnamn för produktnamn
                product_name_col = 'Produktnamn' if 'Produktnamn' in product_row.index else 'product_name'
                product_name_display = str(product_row[product_name_col])[:50] if product_name_col in product_row.index else 'Okänt produkt'
                print(f"    Sidan {idx}/{len(store_products)}: {product_name_display}...")
                
                # Hämta försäljningshistorik för denna produkt
                product_name_in_picking = product_row[product_name_col] if product_name_col in product_row.index else product_row.get('product_name', '')
                
                # Försök hitta matchande produktnamn i sales data
                store_sales = sales_df[sales_df['store'] == store_name].copy()
                product_names_in_sales = store_sales['name'].unique()
                
                # Normalisera produktnamn för matchning
                product_name_normalized = product_name_in_picking.strip().lower()
                
                matched_product_name = None
                
                # Försök exakt match först
                for sales_name in product_names_in_sales:
                    if product_name_normalized == sales_name.strip().lower():
                        matched_product_name = sales_name
                        break
                
                # Om ingen exakt match, försök partiell match
                if matched_product_name is None:
                    for sales_name in product_names_in_sales:
                        sales_normalized = sales_name.strip().lower()
                        if (product_name_normalized in sales_normalized or 
                            sales_normalized in product_name_normalized):
                            matched_product_name = sales_name
                            break
                
                # Hämta enhet och historik om matchning hittades
                unit = 'st'  # Standardenhet
                weekly_forecast = pd.DataFrame(columns=['week_start', 'quantity', 'week_number', 'year', 'week_year'])
                
                if matched_product_name:
                    # Hämta enhet från första matchningen
                    matched_sales = store_sales[store_sales['name'] == matched_product_name]
                    if len(matched_sales) > 0 and 'unit' in matched_sales.columns:
                        unit_value = matched_sales['unit'].iloc[0]
                        if pd.notna(unit_value) and unit_value != '':
                            unit = str(unit_value).strip()
                    
                    sales_history = get_weekly_sales_history(
                        sales_df, store_name, matched_product_name, weeks_back=52
                    )
                    
                    # Skapa produkt-specifik DataFrame för prognos
                    product_sales_df = matched_sales.copy()
                    # Konvertera date till datetime (hantera både date-objekt och strings)
                    if len(product_sales_df) > 0:
                        if 'date' in product_sales_df.columns:
                            # Om date redan är datetime, behåll det, annars konvertera
                            if not pd.api.types.is_datetime64_any_dtype(product_sales_df['date']):
                                product_sales_df['date'] = pd.to_datetime(product_sales_df['date'])
                        elif 'updated' in product_sales_df.columns:
                            # Fallback till updated om date saknas
                            product_sales_df['date'] = pd.to_datetime(product_sales_df['updated'])
                    
                    # Prognostisera 4 veckor framåt
                    try:
                        weekly_forecast = predict_weekly_sales(product_sales_df, future_weeks=FORECAST_WEEKS)
                    except Exception as e:
                        print(f"      Varning: Kunde inte skapa prognos: {e}")
                        weekly_forecast = pd.DataFrame(columns=['week_start', 'quantity', 'week_number', 'year', 'week_year'])
                else:
                    # Ingen matchning, skapa tom historik
                    sales_history = pd.DataFrame(columns=['week_start', 'quantity', 'week_number', 'year', 'week_year'])
                
                # Skapa figur för denna produkt
                fig = plt.figure(figsize=(11.69, 8.27))  # A4-landskap
                
                # Konvertera product_row till dict och mappa kolumnnamn
                product_info = product_row.to_dict()
                
                # Mappa kolumnnamn från CSV till förväntade namn i create_product_page
                # CSV har: Produktnamn, Produktkod, saldo_denna_butik, Varningsgräns, Prognosticerad_försäljning, Påfyllningsbehov
                # create_product_page förväntar: product_name, product_code, current_stock, stock_warning_limit, predicted_sales_units, fill_up_quantity, needs_refill, expected_stock_after_sales
                
                # Mappa kolumner
                if 'Produktnamn' in product_info:
                    product_info['product_name'] = product_info['Produktnamn']
                if 'Produktkod' in product_info:
                    product_info['product_code'] = product_info['Produktkod']
                if 'saldo_denna_butik' in product_info:
                    product_info['current_stock'] = product_info['saldo_denna_butik']
                if 'Varningsgräns' in product_info:
                    product_info['stock_warning_limit'] = product_info['Varningsgräns']
                if 'Prognosticerad_försäljning' in product_info:
                    product_info['predicted_sales_units'] = product_info['Prognosticerad_försäljning']
                if 'Leveransfrekvens_dagar' in product_info:
                    product_info['delivery_frequency_days'] = product_info['Leveransfrekvens_dagar']
                if 'Påfyllningsbehov' in product_info:
                    product_info['fill_up_quantity'] = product_info['Påfyllningsbehov']
                    product_info['needs_refill'] = product_info['Påfyllningsbehov'] > 0
                else:
                    product_info['fill_up_quantity'] = 0
                    product_info['needs_refill'] = False
                
                # Beräkna expected_stock_after_sales om det saknas
                if 'expected_stock_after_sales' not in product_info:
                    current_stock = product_info.get('current_stock', 0)
                    predicted_sales = product_info.get('predicted_sales_units', 0)
                    product_info['expected_stock_after_sales'] = max(0, current_stock - predicted_sales)
                
                # Skapa sidan
                predicted_sales_value = product_info.get('predicted_sales_units', 0)
                delivery_frequency_days = product_info.get('delivery_frequency_days', 7)
                
                # Only fall back to flat forecast when ML forecast is empty
                if len(weekly_forecast) == 0:
                    weekly_forecast = build_weekly_forecast_from_predicted_sales(
                        predicted_sales_value, delivery_frequency_days, future_weeks=FORECAST_WEEKS
                    )
                
                create_product_page(fig, sales_history, product_info, 
                                  predicted_sales_value, weekly_forecast, unit=unit)
                
                # Lägg till sidan i PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        print(f"  ✓ Klar! PDF sparad: {pdf_filename}")

def main():
    """Huvudfunktion"""
    print("="*80)
    print("GRAFISK PLOCKLISTA GENERATOR")
    print("="*80)
    print(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LEVERANSFREKVENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Hitta senaste försäljningsfil automatiskt
    print("\nHittar senaste försäljningsfil...")
    latest_sales_file = find_latest_file('product_sales_*.csv', DATA_DOWNLOADS_DIR)
    
    # Ladda leveransfrekvenser (används för framtida utökningar)
    print(f"\nLaddar leveransfrekvenser från: {LEVERANSFREKVENS_PATH}")
    parametrar = load_leveransfrekvens(LEVERANSFREKVENS_PATH)
    
    # Ladda plocklista
    print(f"\nLäser plocklista från {PICKING_LIST_PATH}...")
    picking_list_df = pd.read_csv(str(PICKING_LIST_PATH), sep=';')
    print(f"  Laddat {len(picking_list_df)} produkter")
    print(f"  Butiker: {picking_list_df['store_name'].nunique()}")
    
    # Ladda försäljningsdata
    sales_df = load_and_prepare_sales_data(latest_sales_file)
    
    # Generera grafiska PDF:er
    generate_graphical_picking_list(picking_list_df, sales_df)
    
    print(f"\n{'='*80}")
    print("KLAR!")
    print(f"{'='*80}")
    print(f"Grafiska plocklistor sparade i mappen: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
