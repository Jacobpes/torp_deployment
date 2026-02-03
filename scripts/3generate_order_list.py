"""
Orderlista baserad på beställningsfrekvens per leverantör

Detta script:
1. Läser beställningsfrekvens från parametrar/Beställningsfrekvens.csv
2. Hämtar leverantörsinformation från product_sales_items
3. För varje leverantör, prognostiserar försäljning för beställningsfrekvens-perioden
4. Skapar en CSV-fil per leverantör i orderlistor/ med:
   - Alla produkter från leverantören
   - Prognosticerad försäljning per butik för beställningsfrekvens-perioden
   - Saldo (lagersaldo)
   - stock_warning_limit (multiplicerat med antal butiker som säljer produkten)
   - beställningsbehov (beräknat så att saldo inte går under stock_warning_limit)
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
import os
import sys
import glob
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
try:
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("Warning: openpyxl not available. Excel export will be skipped.")

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
    OUTPUT_DIR = PROJECT_ROOT / 'output' / 'orderlistor'  # Save in output/orderlistor subdirectory
    DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'  # Downloads in data/nedladdningar
    BESTALLNINGSFREKVENS_PATH = PROJECT_ROOT / 'data' / 'parametrar' / 'Beställningsfrekvens.csv'
else:
    OUTPUT_DIR = PROJECT_ROOT / 'orderlistor'
    DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'
    BESTALLNINGSFREKVENS_PATH = PROJECT_ROOT / 'data' / 'parametrar' / 'Beställningsfrekvens.csv'

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

def load_bestallningsfrekvens(file_path):
    """Läser parametrar/Beställningsfrekvens.csv och returnerar dictionary med frekvens per leverantör"""
    file_path = Path(file_path)
    print(f"\nLäser beställningsfrekvens från {file_path}...")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Beställningsfrekvens-fil hittades inte: {file_path}")
    
    # CSV-struktur (efter fix):
    # Kolumn 1: "Leverantör" - leverantörsnamn (t.ex. "Snellman", "Arla")
    # Kolumn 2: "e-post" - email-adress
    # Kolumn 3: "Beställnings frekvens/antal dagar" - frekvens i dagar (t.ex. "7", "21")
    
    # Läs CSV manuellt för att undvika problem med BOM och pandas tolkningsfel
    import csv
    rows = []
    with open(str(file_path), 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            rows.append(row)
    
    if len(rows) < 2:
        print(f"  Fel: CSV-filen har för få rader ({len(rows)}). Förväntar minst header + 1 datarad.")
        return {}
    
    # Första raden är header, resten är data
    header = rows[0]
    data_rows = rows[1:]
    
    print(f"  Hittade {len(header)} kolumner: {header[:4]}...")
    print(f"  Hittade {len(data_rows)} datarader")
    
    # Skapa dictionary: leverantör -> beställningsfrekvens (dagar)
    frekvenser = {}
    
    # Kolumnindex: kolumn 0 = Leverantör, kolumn 2 = Beställnings frekvens/antal dagar
    leverantor_col_idx = 0
    frekvens_col_idx = 2
    
    if len(header) <= frekvens_col_idx:
        print(f"  Fel: CSV-filen har för få kolumner ({len(header)}). Förväntar minst {frekvens_col_idx + 1} kolumner.")
        return frekvenser
    
    print(f"  Använder kolumn {leverantor_col_idx} för Leverantör: '{header[leverantor_col_idx]}'")
    print(f"  Använder kolumn {frekvens_col_idx} för Frekvens: '{header[frekvens_col_idx]}'")
    
    for row_idx, row in enumerate(data_rows):
        if len(row) <= max(leverantor_col_idx, frekvens_col_idx):
            continue
            
        leverantor_raw = row[leverantor_col_idx].strip() if leverantor_col_idx < len(row) else ''
        frekvens_raw = row[frekvens_col_idx].strip() if frekvens_col_idx < len(row) else ''
        
        leverantor = leverantor_raw if leverantor_raw else None
        
        # Validera leverantörsnamn - hoppa över om det är tomt
        if not leverantor or leverantor.lower() in ['nan', 'none', '']:
            continue
            
        # Validera att frekvens_raw inte är tom
        if not frekvens_raw or frekvens_raw.strip() == '':
            continue
            
        try:
            # Konvertera frekvens till heltal
            freq_str = str(frekvens_raw).strip()
            
            # Ta bort alla icke-numeriska tecken förutom decimaltecken
            freq_clean = ''.join(c for c in freq_str if c.isdigit() or c == '.')
            
            if freq_clean:
                frekvens_dagar = int(float(freq_clean))
                # Validera att frekvensen är rimlig (mellan 1 och 365 dagar)
                if 1 <= frekvens_dagar <= 365:
                    frekvenser[leverantor] = frekvens_dagar
                else:
                    print(f"  Varning: Ovanlig frekvens för '{leverantor}': {frekvens_dagar} dagar (hoppar över)")
            else:
                print(f"  Varning: Kunde inte extrahera nummer från frekvens för '{leverantor}': '{frekvens_raw}'")
        except (ValueError, TypeError) as e:
            # Hoppa över rader där frekvensen inte kan tolkas
            print(f"  Varning: Kunde inte tolka frekvens för '{leverantor}': '{frekvens_raw}' (fel: {e})")
    
    print(f"  Laddade beställningsfrekvenser för {len(frekvenser)} leverantörer")
    if len(frekvenser) > 0:
        print(f"  Exempel leverantörer: {list(frekvenser.keys())[:5]}")
        # Visa några exempel med frekvenser
        for lev, freq in list(frekvenser.items())[:5]:
            print(f"    {lev}: {freq} dagar")
    else:
        print(f"  VARNING: Inga leverantörer laddades! Kontrollera CSV-filen.")
        print(f"  Förväntad struktur: Kolumn 0 = Leverantör, Kolumn 2 = Beställnings frekvens/antal dagar")
        print(f"  Kontrollera att frekvenskolumnen innehåller numeriska värden (1-365)")
    
    return frekvenser

def load_supplier_mapping(file_path):
    """
    Läser product_sales_items och skapar mapping: produkt -> leverantör
    Returnerar tuple: (supplier_mapping, unit_mapping)
    supplier_mapping: (product_name, store_name) -> supplier_name
    unit_mapping: (product_name, store_name) -> unit
    """
    file_path = Path(file_path)
    print(f"\nLäser leverantörsinformation från {file_path}...")
    df = pd.read_csv(str(file_path))
    
    # Filtrera bort order med status "error"
    if 'order_status' in df.columns:
        df = df[df['order_status'] == 'complete'].copy()
    
    # Skapa mapping: produktnamn -> leverantör
    # Använd både produktnamn och butik för att hantera samma produkt från olika leverantörer
    supplier_mapping = {}
    unit_mapping = {}
    
    for _, row in df.iterrows():
        if pd.notna(row.get('supplier_name')) and pd.notna(row.get('product_name')):
            product_name = str(row['product_name']).strip()
            supplier_name = str(row['supplier_name']).strip()
            store_name = str(row.get('store_name', '')).strip()
            
            if product_name and supplier_name:
                # Skapa nyckel baserat på produktnamn och butik
                key = (product_name.lower(), store_name)
                if key not in supplier_mapping:
                    supplier_mapping[key] = supplier_name
                    # Spara också enhet om den finns
                    if 'unit' in row and pd.notna(row['unit']):
                        unit_mapping[key] = str(row['unit']).strip()
                    else:
                        unit_mapping[key] = 'st'
                # Om samma produkt har olika leverantörer i samma butik, ta den första
    
    print(f"  Laddade leverantörsmappning för {len(supplier_mapping)} produkt-butik-kombinationer")
    return supplier_mapping, unit_mapping

def get_product_unit(product_name, store_name, unit_mapping):
    """
    Hämtar enhet för en produkt från unit_mapping.
    Returnerar enhet eller 'st' som standard.
    """
    key = (product_name.lower(), store_name)
    if key in unit_mapping:
        return unit_mapping[key]
    return 'st'

def load_and_prepare_sales_data(file_path):
    """Läser och förbereder försäljningsdata"""
    file_path = Path(file_path)
    print(f"\nLäser försäljningsdata från {file_path}...")
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
        # Behåll unit-kolumnen om den finns (den ska redan finnas efter rename)
        if 'unit' not in df.columns:
            # Om unit saknas, sätt standardvärde
            df['unit'] = 'st'
    elif 'created_at' in df.columns:
        # Gammal struktur med individuella orderrader (product_sales_items)
        df = df.rename(columns={
            'created_at': 'updated',
            'store_name': 'store',
            'product_name': 'name',
            'line_total': 'line_price'
        })
        # Konvertera datum
        df['updated'] = pd.to_datetime(df['updated'])
        df['date'] = df['updated'].dt.date
        # Behåll unit-kolumnen om den finns (product_sales_items har 'unit' kolumn)
        if 'unit' not in df.columns:
            df['unit'] = 'st'
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
        df['unit'] = 'st'
    
    # Filtrera bort order med status "error" om order_status kolumnen finns
    if 'order_status' in df.columns:
        initial_count = len(df)
        df = df[df['order_status'] == 'complete'].copy()
        filtered_count = len(df)
        if initial_count != filtered_count:
            print(f"  Filtrerade bort {initial_count - filtered_count} rader med order_status != 'complete'")
    
    # Fyll NaN-värden i quantity med 0
    if 'quantity' in df.columns:
        nan_count = df['quantity'].isna().sum()
        if nan_count > 0:
            print(f"  Fyller {nan_count} NaN-värden i quantity med 0")
            df['quantity'] = df['quantity'].fillna(0)
    
    # Fyll NaN-värden i unit med 'st'
    if 'unit' in df.columns:
        nan_count = df['unit'].isna().sum()
        if nan_count > 0:
            df['unit'] = df['unit'].fillna('st')
    
    print(f"  Laddat {len(df)} rader")
    print(f"  Butiker: {df['store'].nunique()}")
    print(f"  Produkter: {df['name'].nunique()}")
    print(f"  Datumintervall: {df['date'].min()} till {df['date'].max()}")
    
    return df

def load_stock_data(file_path):
    """Läser lagerdata"""
    file_path = Path(file_path)
    print(f"\nLäser lagerdata från {file_path}...")
    stock_df = pd.read_csv(str(file_path))
    
    # Normalisera produktnamn för matchning
    stock_df['product_name_normalized'] = stock_df['product_name'].str.strip().str.lower()
    
    # Hantera negativa stock-värden (sätt till 0)
    stock_df['stock'] = stock_df['stock'].clip(lower=0)
    
    print(f"  Laddat {len(stock_df)} rader")
    print(f"  Butiker: {stock_df['store_name'].nunique()}")
    print(f"  Produkter: {stock_df['product_name'].nunique()}")
    
    return stock_df


def build_stock_index(stock_df):
    """Bygger uppslag per butik för produkter i stock_report."""
    stock_by_store = {}
    if stock_df is None or len(stock_df) == 0:
        return stock_by_store
    for store_name, group in stock_df.groupby('store_name'):
        stock_by_store[store_name] = (
            group['product_name_normalized']
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
    return stock_by_store


def product_exists_in_stock(product_name, store_name, stock_by_store):
    """Returnerar True om produkten finns i stock_report för given butik."""
    if store_name not in stock_by_store:
        return False
    product_norm = str(product_name).strip().lower()
    stock_names = stock_by_store[store_name]
    if product_norm in stock_names:
        return True
    # Partiell matchning för att hantera små namnvariationer
    for stock_name in stock_names:
        if product_norm in stock_name or stock_name in product_norm:
            return True
    return False


def filter_sales_to_stock(sales_df, stock_df):
    """
    Filtrerar bort produkter som inte finns i stock_report.
    Produkter som saknas i stock_report betraktas som utgångna.
    """
    stock_by_store = build_stock_index(stock_df)
    if not stock_by_store:
        return sales_df
    allowed_pairs = set()
    unique_pairs = sales_df[['store', 'name']].drop_duplicates()
    for store_name, product_name in unique_pairs.itertuples(index=False):
        if product_exists_in_stock(product_name, store_name, stock_by_store):
            allowed_pairs.add((store_name, product_name))
    if not allowed_pairs:
        return sales_df.iloc[0:0].copy()
    mask = [(s, n) in allowed_pairs for s, n in zip(sales_df['store'], sales_df['name'])]
    filtered_df = sales_df[mask].copy()
    removed = len(sales_df) - len(filtered_df)
    if removed > 0:
        print(f"  Filtrerade bort {removed} rader (produkter ej i stock_report)")
    return filtered_df


def filter_mapping_to_stock(mapping, stock_df):
    """Filtrerar mapping till produkter som finns i stock_report."""
    stock_by_store = build_stock_index(stock_df)
    if not stock_by_store:
        return mapping
    filtered = {}
    for (product_name, store_name), value in mapping.items():
        if product_exists_in_stock(product_name, store_name, stock_by_store):
            filtered[(product_name, store_name)] = value
    return filtered

def predict_product_sales(product_df, forecast_days):
    """
    Prognostiserar försäljning för en produkt baserat på historisk data.
    Returnerar totalt antal enheter som förväntas säljas under forecast_days dagar.
    """
    # Aggregera daglig försäljning
    daily_sales = product_df.groupby('date')['quantity'].sum().reset_index()
    daily_sales = daily_sales.sort_values('date')
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    
    # Fyll NaN-värden med 0
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)
    
    # Hämta dagens datum
    today = pd.Timestamp.now().normalize()
    first_date = daily_sales['date'].min()
    last_date_in_data = daily_sales['date'].max()
    
    # Säkerställ att data går till dagens datum
    if last_date_in_data < today:
        date_range = pd.date_range(start=first_date, end=today, freq='D')
        daily_sales = daily_sales.set_index('date').reindex(date_range, fill_value=0).reset_index()
        daily_sales = daily_sales.rename(columns={'index': 'date'})
        last_date = today
    else:
        daily_sales = daily_sales[daily_sales['date'] <= today].copy()
        date_range = pd.date_range(start=first_date, end=today, freq='D')
        daily_sales = daily_sales.set_index('date').reindex(date_range, fill_value=0).reset_index()
        daily_sales = daily_sales.rename(columns={'index': 'date'})
        last_date = today
    
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)
    
    # Räkna antal dagar med data
    days_with_data = len(daily_sales)
    # Räkna antal månader (ungefär 30 dagar per månad)
    months_with_data = days_with_data / 30.0
    
    # Fall 1: Mindre än 4 veckors data (< 28 dagar)
    # Använd veckovis genomsnitt av all data som är > 0
    if days_with_data < 28:
        # Aggregera till veckovis data för att få korrekt veckovis genomsnitt
        daily_sales['week'] = daily_sales['date'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly_sales = daily_sales.groupby('week')['quantity'].sum().reset_index()
        
        # Filtrera bort veckor med 0 försäljning för att få genomsnitt av faktisk försäljning
        non_zero_weeks = weekly_sales[weekly_sales['quantity'] > 0]['quantity']
        if len(non_zero_weeks) > 0:
            # Beräkna genomsnittlig veckovis försäljning
            avg_weekly_sales = non_zero_weeks.mean()
            # Konvertera till dagligt genomsnitt för att prognostisera för forecast_days
            avg_daily_sales = avg_weekly_sales / 7.0
            # Prognostisera för forecast_days dagar framåt
            predicted_total = avg_daily_sales * forecast_days
            return max(0.0, float(predicted_total))
        else:
            # Om all försäljning är 0, returnera 0
            return 0.0
    
    # Fall 2: Mindre än 10 månaders data (< 300 dagar)
    # Använd 4 veckors genomsnitt
    elif months_with_data < 10:
        # Aggregera till veckovis data för de senaste 4 veckorna
        daily_sales['week'] = daily_sales['date'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly_sales = daily_sales.groupby('week')['quantity'].sum().reset_index()
        
        # Ta de senaste 4 veckorna (eller så många som finns)
        weeks_to_look_back = min(4, len(weekly_sales))
        if weeks_to_look_back > 0:
            recent_weekly_sales = weekly_sales.tail(weeks_to_look_back)['quantity']
            # Om det finns försäljning i de senaste veckorna, använd genomsnitt
            if recent_weekly_sales.sum() > 0:
                avg_weekly_sales = recent_weekly_sales.mean()
            else:
                # Om ingen försäljning i senaste veckorna, använd genomsnitt av alla veckor med försäljning
                non_zero_weeks = weekly_sales[weekly_sales['quantity'] > 0]['quantity']
                if len(non_zero_weeks) > 0:
                    avg_weekly_sales = non_zero_weeks.mean()
                else:
                    avg_weekly_sales = 0.0
            
            # Konvertera till dagligt genomsnitt för att prognostisera för forecast_days
            avg_daily_sales = avg_weekly_sales / 7.0
            # Prognostisera för forecast_days dagar framåt
            predicted_total = avg_daily_sales * forecast_days
            return max(0.0, float(predicted_total))
        else:
            return 0.0
    
    # Feature engineering
    daily_sales['dayofweek'] = daily_sales['date'].dt.dayofweek
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['day'] = daily_sales['date'].dt.day
    daily_sales['is_weekend'] = daily_sales['dayofweek'].isin([5, 6]).astype(int)
    
    # Träna modell
    X = daily_sales[['dayofweek', 'month', 'day', 'is_weekend']]
    y = daily_sales['quantity']
    
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)
    
    # Prognostisera framtida försäljning
    tomorrow = last_date + pd.Timedelta(days=1)
    future_dates = pd.date_range(tomorrow, periods=forecast_days, freq='D')
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df['dayofweek'] = future_df['date'].dt.dayofweek
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
    X_future = future_df[['dayofweek', 'month', 'day', 'is_weekend']]
    
    y_future = model.predict(X_future)
    y_future = np.maximum(y_future, 0)
    
    return float(y_future.sum())

def get_product_stock_info(stock_df, product_name, store_name):
    """
    Hämtar lagerinformation för en produkt i en specifik butik.
    Returnerar (stock, stock_warning_limit) eller (0, 0) om inte hittad.
    """
    product_normalized = product_name.strip().lower()
    store_normalized = store_name.strip()
    
    # Försök hitta match
    matched = stock_df[
        (stock_df['product_name_normalized'] == product_normalized) &
        (stock_df['store_name'] == store_normalized)
    ]
    
    if len(matched) > 0:
        stock = float(matched.iloc[0]['stock'])
        warning_limit = float(matched.iloc[0]['stock_warning_limit'])
        return stock, warning_limit
    
    # Försök partiell match
    matched = stock_df[
        (stock_df['product_name_normalized'].str.contains(product_normalized, na=False, regex=False)) &
        (stock_df['store_name'] == store_normalized)
    ]
    
    if len(matched) > 0:
        stock = float(matched.iloc[0]['stock'])
        warning_limit = float(matched.iloc[0]['stock_warning_limit'])
        return stock, warning_limit
    
    return 0.0, 0.0

def count_stores_selling_product(stock_df, product_name):
    """
    Räknar antal butiker som säljer en produkt (baserat på stock_report).
    """
    product_normalized = product_name.strip().lower()
    
    # Försök exakt match
    matched = stock_df[stock_df['product_name_normalized'] == product_normalized]
    
    if len(matched) == 0:
        # Försök partiell match
        matched = stock_df[
            stock_df['product_name_normalized'].str.contains(product_normalized, na=False, regex=False)
        ]
    
    if len(matched) > 0:
        return matched['store_name'].nunique()
    
    return 0

def calculate_bestallningsbehov(current_stock_total, predicted_sales_total, stock_warning_limit_total, order_frequency_days):
    """
    Beräknar beställningsbehov så att saldo inte går under stock_warning_limit
    under beställningsfrekvens-perioden.
    
    Logik:
    - Efter order_frequency_days: current_stock_total + beställningsbehov - predicted_sales_total >= stock_warning_limit_total
    - beställningsbehov >= stock_warning_limit_total + predicted_sales_total - current_stock_total
    """
    bestallningsbehov = stock_warning_limit_total + predicted_sales_total - current_stock_total
    return max(0.0, bestallningsbehov)

def match_supplier_name(supplier_from_mapping, supplier_from_frekvens):
    """
    Försöker matcha leverantörsnamn mellan olika källor.
    Returnerar matchat namn eller None.
    """
    if not supplier_from_mapping or not supplier_from_frekvens:
        return None
    
    mapping_normalized = str(supplier_from_mapping).strip().lower()
    frekvens_normalized = str(supplier_from_frekvens).strip().lower()
    
    # Ta bort vanliga suffix och prefix för bättre matchning
    def normalize_name(name):
        # Ta bort "Ab", "Oy", "Ab Oy" etc.
        name = re.sub(r'\b(ab|oy|ab oy|aboy)\b', '', name)
        # Ta bort extra mellanslag
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    mapping_clean = normalize_name(mapping_normalized)
    frekvens_clean = normalize_name(frekvens_normalized)
    
    # Exakt match
    if mapping_clean == frekvens_clean or mapping_normalized == frekvens_normalized:
        return supplier_from_frekvens
    
    # Partiell match (en innehåller den andra)
    if mapping_clean in frekvens_clean or frekvens_clean in mapping_clean:
        return supplier_from_frekvens
    
    # Matcha på första ordet (t.ex. "Snellman" matchar "Robin Snellman")
    mapping_first_word = mapping_clean.split()[0] if mapping_clean.split() else ''
    frekvens_first_word = frekvens_clean.split()[0] if frekvens_clean.split() else ''
    
    if mapping_first_word and frekvens_first_word and mapping_first_word == frekvens_first_word:
        return supplier_from_frekvens
    
    # Matcha på sista ordet
    mapping_last_word = mapping_clean.split()[-1] if mapping_clean.split() else ''
    frekvens_last_word = frekvens_clean.split()[-1] if frekvens_clean.split() else ''
    
    if mapping_last_word and frekvens_last_word and mapping_last_word == frekvens_last_word:
        return supplier_from_frekvens
    
    return None

def process_suppliers(sales_df, stock_df, supplier_mapping, unit_mapping, bestallningsfrekvenser):
    """
    Processar alla leverantörer och skapar orderlistor.
    """
    print("\n" + "="*80)
    print("GENERERAR ORDERLISTOR PER LEVERANTÖR")
    print("="*80)
    
    # Skapa output-katalog
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Gruppera produkter per leverantör
    supplier_products = {}  # supplier -> list of (product_name, store_name)
    
    # Gå igenom alla produkter i sales data och matcha med leverantörer
    for product_name in sales_df['name'].unique():
        for store_name in sales_df[sales_df['name'] == product_name]['store'].unique():
            key = (product_name.lower(), store_name)
            if key in supplier_mapping:
                supplier = supplier_mapping[key]
                if supplier not in supplier_products:
                    supplier_products[supplier] = []
                if (product_name, store_name) not in supplier_products[supplier]:
                    supplier_products[supplier].append((product_name, store_name))
    
    print(f"Hittade produkter för {len(supplier_products)} leverantörer")
    
    # Processa varje leverantör
    for supplier_name, products in supplier_products.items():
        # Hitta matchande leverantör i beställningsfrekvens
        matched_supplier = None
        order_frequency = None
        
        # Försök hitta exakt match först
        if supplier_name in bestallningsfrekvenser:
            matched_supplier = supplier_name
            order_frequency = bestallningsfrekvenser[supplier_name]
        else:
            # Försök hitta partiell match (både vägar)
            for frekvens_supplier, freq in bestallningsfrekvenser.items():
                if match_supplier_name(supplier_name, frekvens_supplier):
                    matched_supplier = frekvens_supplier
                    order_frequency = freq
                    break
                # Försök även omvänt
                if match_supplier_name(frekvens_supplier, supplier_name):
                    matched_supplier = frekvens_supplier
                    order_frequency = freq
                    break
        
        if order_frequency is None:
            # Använd standardvärde 7 dagar om ingen match hittas
            order_frequency = 7
            print(f"\nVarning: Hittade ingen beställningsfrekvens för leverantör '{supplier_name}'.")
            print(f"  Använder standardvärde: {order_frequency} dagar")
        
        print(f"\nProcessar leverantör: {supplier_name} (beställningsfrekvens: {order_frequency} dagar)")
        
        # Skapa orderlista för denna leverantör
        order_list = []
        
        for product_name, store_name in products:
            # Hämta försäljningsdata för denna produkt i denna butik
            product_sales = sales_df[
                (sales_df['name'] == product_name) &
                (sales_df['store'] == store_name)
            ].copy()
            
            if len(product_sales) == 0:
                continue
            
            # Prognostisera försäljning för beställningsfrekvens-perioden
            try:
                predicted_sales = predict_product_sales(product_sales, order_frequency)
            except Exception as e:
                print(f"  Varning: Kunde inte prognostisera för {product_name} i {store_name}: {e}")
                predicted_sales = 0.0
            
            # Hämta lagerinformation
            stock, warning_limit = get_product_stock_info(stock_df, product_name, store_name)
            
            # Hämta enhet för produkten från unit_mapping
            unit = get_product_unit(product_name, store_name, unit_mapping)
            
            # Lägg till i orderlista
            order_list.append({
                'Produktnamn': product_name,
                'Butik': store_name,
                'Enhet': unit,
                'Prognosticerad_försäljning': predicted_sales,
                'Saldo': stock,
                'stock_warning_limit': warning_limit
            })
        
        if len(order_list) == 0:
            print(f"  Inga produkter att beställa för {supplier_name}")
            continue
        
        # Konvertera till DataFrame och aggregera per produkt (summera över butiker)
        order_df = pd.DataFrame(order_list)
        
        # Gruppera per produkt och summera
        # Hämta enhet från första förekomsten (samma produkt bör ha samma enhet)
        unit_per_product = order_df.groupby('Produktnamn')['Enhet'].first().reset_index()
        unit_per_product.columns = ['Produktnamn', 'Enhet']
        
        aggregated = order_df.groupby('Produktnamn').agg({
            'Prognosticerad_försäljning': 'sum',  # Total försäljning över alla butiker
            'Saldo': 'sum',  # Total saldo över alla butiker
            'stock_warning_limit': 'sum',  # Total warning limit över alla butiker
            'Butik': lambda x: ', '.join(x.unique())  # Lista butiker
        }).reset_index()
        
        # Lägg till enhet
        aggregated = aggregated.merge(unit_per_product, on='Produktnamn')
        
        # Räkna antal butiker per produkt
        stores_per_product = order_df.groupby('Produktnamn')['Butik'].nunique().reset_index()
        stores_per_product.columns = ['Produktnamn', 'Antal_butiker']
        aggregated = aggregated.merge(stores_per_product, on='Produktnamn')
        
        # stock_warning_limit är redan summerad över alla butiker (rad 661), ingen ytterligare multiplikation behövs
        
        # Beräkna beställningsbehov
        aggregated['beställningsbehov'] = aggregated.apply(
            lambda row: calculate_bestallningsbehov(
                row['Saldo'],
                row['Prognosticerad_försäljning'],
                row['stock_warning_limit'],
                order_frequency
            ),
            axis=1
        )
        
        # Lägg till orderfrekvens som kolumn
        aggregated['Orderfrekvens_dagar'] = order_frequency
        
        # Sortera efter beställningsbehov (högst först)
        aggregated = aggregated.sort_values('beställningsbehov', ascending=False)
        
        # Ordna kolumner i logisk ordning
        column_order = [
            'Produktnamn',
            'Enhet',
            'Orderfrekvens_dagar',
            'Prognosticerad_försäljning',
            'Saldo',
            'stock_warning_limit',
            'Antal_butiker',
            'beställningsbehov',
            'Butik'
        ]
        # Lägg bara till kolumner som faktiskt finns
        available_columns = [col for col in column_order if col in aggregated.columns]
        aggregated = aggregated[available_columns]
        
        # Formatera data
        # Prognosticerad_försäljning och beställningsbehov ska ha 1 decimal
        if 'Prognosticerad_försäljning' in aggregated.columns:
            aggregated['Prognosticerad_försäljning'] = aggregated['Prognosticerad_försäljning'].round(1)
        if 'beställningsbehov' in aggregated.columns:
            aggregated['beställningsbehov'] = aggregated['beställningsbehov'].round(1)
        # Saldo och stock_warning_limit kan också ha 1 decimal för konsistens
        if 'Saldo' in aggregated.columns:
            aggregated['Saldo'] = aggregated['Saldo'].round(1)
        if 'stock_warning_limit' in aggregated.columns:
            aggregated['stock_warning_limit'] = aggregated['stock_warning_limit'].round(1)
        
        # Spara till CSV
        safe_supplier_name = supplier_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_file_csv = OUTPUT_DIR / f'Orderlista_{safe_supplier_name}.csv'
        aggregated.to_csv(str(output_file_csv), index=False, sep=';', encoding='utf-8-sig')
        
        # Spara till Excel med auto-anpassade kolumnbredder
        if OPENPYXL_AVAILABLE:
            output_file_xlsx = OUTPUT_DIR / f'Orderlista_{safe_supplier_name}.xlsx'
            with pd.ExcelWriter(str(output_file_xlsx), engine='openpyxl') as writer:
                aggregated.to_excel(writer, sheet_name='Orderlista', index=False)
                worksheet = writer.sheets['Orderlista']
                
                # Auto-anpassa kolumnbredder
                for idx, col in enumerate(aggregated.columns, 1):
                    max_length = max(
                        aggregated[col].astype(str).map(len).max(),
                        len(str(col))
                    )
                    # Sätt en max-bredd för att undvika för breda kolumner
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[get_column_letter(idx)].width = adjusted_width
        
        print(f"  [OK] Sparade orderlista med {len(aggregated)} produkter till {output_file_csv}")
        if OPENPYXL_AVAILABLE:
            print(f"  [OK] Sparade Excel-fil: {output_file_xlsx}")
        print(f"    Totalt beställningsbehov: {aggregated['beställningsbehov'].sum():.1f} enheter")

def main():
    """Huvudfunktion"""
    print("="*80)
    print("ORDERLISTA GENERATOR - PER LEVERANTÖR MED BESTÄLLNINGSFREKVENS")
    print("="*80)
    print(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    BESTALLNINGSFREKVENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Hitta senaste filer automatiskt
    print("\nHittar senaste datafiler...")
    latest_sales_file = find_latest_file('product_sales_*.csv', DATA_DOWNLOADS_DIR)
    latest_stock_file = find_latest_file('stock_report_*.csv', DATA_DOWNLOADS_DIR)
    latest_items_file = find_latest_file('product_sales_items_*.csv', DATA_DOWNLOADS_DIR)
    
    # Ladda beställningsfrekvenser
    bestallningsfrekvenser = load_bestallningsfrekvens(BESTALLNINGSFREKVENS_PATH)
    
    # Ladda leverantörsmappning och enhetsmappning
    supplier_mapping, unit_mapping = load_supplier_mapping(latest_items_file)
    
    # Ladda försäljnings- och lagerdata
    sales_df = load_and_prepare_sales_data(latest_sales_file)
    stock_df = load_stock_data(latest_stock_file)

    # Filtrera bort produkter som inte finns i stock_report
    sales_df = filter_sales_to_stock(sales_df, stock_df)
    supplier_mapping = filter_mapping_to_stock(supplier_mapping, stock_df)
    unit_mapping = filter_mapping_to_stock(unit_mapping, stock_df)
    
    # Processa leverantörer och skapa orderlistor
    process_suppliers(sales_df, stock_df, supplier_mapping, unit_mapping, bestallningsfrekvenser)
    
    print(f"\n{'='*80}")
    print("KLAR!")
    print(f"{'='*80}")
    print(f"Orderlistor sparade i mappen: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
