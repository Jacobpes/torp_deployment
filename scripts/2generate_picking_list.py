"""
Produktvis prognos och plocklista per shop

Detta script:
1. Gör produktvis prognos för varje shop baserat på historisk försäljning
2. Beräknar plocklista: hur mycket som behöver fyllas på i lager
   så att lagret hamnar precis på stock_warning_limit efter prognosticerad försäljning
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
    OUTPUT_DIR = PROJECT_ROOT / 'output' / 'plocklistor'  # Save in output/plocklistor subdirectory
    DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'  # Downloads in data/nedladdningar
    LEVERANSFREKVENS_PATH = PROJECT_ROOT / 'data' / 'parametrar' / 'Leveransfrekvens.csv'
    PICKING_LIST_RESULTS_PATH = PROJECT_ROOT / 'output' / 'plocklistor' / 'picking_list_results.csv'
else:
    OUTPUT_DIR = PROJECT_ROOT / 'plocklistor'
    DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'
    LEVERANSFREKVENS_PATH = PROJECT_ROOT / 'data' / 'parametrar' / 'Leveransfrekvens.csv'
    PICKING_LIST_RESULTS_PATH = PROJECT_ROOT / 'picking_list_results.csv'

def find_latest_file(pattern, directory):
    """
    Hittar den senaste filen baserat på datum i filnamnet.
    Pattern ska vara t.ex. 'product_sales_*.csv' eller 'stock_report_*.csv'
    """
    directory = Path(directory)
    
    # Kontrollera att katalogen finns
    if not directory.exists():
        raise FileNotFoundError(f"Katalogen finns inte: {directory}")
    
    files = glob.glob(str(directory / pattern))
    if not files:
        # Visa vad som faktiskt finns i katalogen för debugging
        existing_files = list(directory.glob('*.csv'))
        if existing_files:
            print(f"  Hittade {len(existing_files)} CSV-filer i {directory}, men inga matchar mönstret {pattern}")
            print(f"  Exempel filer: {[f.name for f in existing_files[:5]]}")
        else:
            print(f"  Katalogen {directory} är tom eller innehåller inga CSV-filer")
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
    
    # Säkerställ att nödvändiga kolumner finns
    required_cols = ['date', 'store', 'name', 'quantity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Saknade kolumner: {missing_cols}. Tillgängliga kolumner: {list(df.columns)}")
    
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
    
    # Normalisera produktnamn och butiksnamn för matchning
    stock_df['product_name_normalized'] = stock_df['product_name'].str.strip().str.lower()
    stock_df['store_name_normalized'] = stock_df['store_name'].str.strip()
    
    # Hantera negativa stock-värden (sätt till 0)
    stock_df['stock'] = stock_df['stock'].clip(lower=0)
    
    print(f"  Laddat {len(stock_df)} rader")
    print(f"  Butiker: {stock_df['store_name'].nunique()}")
    print(f"  Produkter: {stock_df['product_name'].nunique()}")
    
    return stock_df


def predict_product_sales(product_df, forecast_days):
    """
    Prognostiserar försäljning för en produkt baserat på historisk data.
    Returnerar totalt antal enheter som förväntas säljas under framtida perioden.
    Prognostiserar för exakt forecast_days dagar från morgondagen.
    """
    # Aggregera daglig försäljning (i antal enheter, inte värde)
    daily_sales = product_df.groupby('date')['quantity'].sum().reset_index()
    daily_sales = daily_sales.sort_values('date')
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    
    # Fyll NaN-värden med 0
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)
    
    # Hämta dagens datum och första datumet i data
    today = pd.Timestamp.now().normalize()
    first_date = daily_sales['date'].min()
    last_date_in_data = daily_sales['date'].max()
    
    # Säkerställ att data går till dagens datum
    # Om sista datumet i data är tidigare än idag, fyll i saknade dagar till idag
    if last_date_in_data < today:
        # Skapa fullständig datumserie från första datumet till dagens datum
        date_range = pd.date_range(start=first_date, end=today, freq='D')
        daily_sales = daily_sales.set_index('date').reindex(date_range, fill_value=0).reset_index()
        daily_sales = daily_sales.rename(columns={'index': 'date'})
        last_date = today
    else:
        # Om data går längre än idag, använd bara data till idag
        daily_sales = daily_sales[daily_sales['date'] <= today].copy()
        # Fyll i saknade dagar från första datumet till idag
        date_range = pd.date_range(start=first_date, end=today, freq='D')
        daily_sales = daily_sales.set_index('date').reindex(date_range, fill_value=0).reset_index()
        daily_sales = daily_sales.rename(columns={'index': 'date'})
        last_date = today
    
    # Säkerställ att quantity är 0 för alla saknade värden
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
    
    # Förbered features och target
    X = daily_sales[['dayofweek', 'month', 'day', 'is_weekend']]
    y = daily_sales['quantity']
    
    # Träna modell
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)
    
    # Förbered framtida datum: prognostisera för forecast_days dagar från morgondagen
    tomorrow = last_date + pd.Timedelta(days=1)
    # Prognostisera för exakt forecast_days dagar framåt
    future_dates = pd.date_range(tomorrow, periods=forecast_days, freq='D')
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df['dayofweek'] = future_df['date'].dt.dayofweek
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
    X_future = future_df[['dayofweek', 'month', 'day', 'is_weekend']]
    
    # Prognostisera framtida försäljning (säkra att inga negativa värden)
    y_future = model.predict(X_future)
    y_future = np.maximum(y_future, 0)
    
    # Returnera totalt antal enheter som förväntas säljas under forecast_days dagar
    return float(y_future.sum())

def calculate_picking_quantity(current_stock, predicted_sales_total, stock_warning_limit, delivery_frequency_days):
    """
    Beräknar hur mycket som behöver fyllas på i lager.
    
    Logik:
    - Om försäljningsprognosen förverkligas, ska lagret stanna på varningsgränsen
    - Formel: saldo_denna_butik + Påfyllningsbehov - Prognosticerad_försäljning = Varningsgräns
    - Påfyllningsbehov = Varningsgräns - saldo_denna_butik + Prognosticerad_försäljning
    - Om resultatet är negativt (för mycket lager), returnera 0
    """
    # Beräkna påfyllningsbehov så att efter försäljningsprognosen hamnar lagret på varningsgränsen
    fill_up = stock_warning_limit - current_stock + predicted_sales_total
    return max(0.0, fill_up)

def match_product_name(sales_name, stock_names):
    """
    Försöker matcha produktnamn mellan försäljningsdata och lagerdata.
    Returnerar matchat namn eller None.
    """
    sales_normalized = sales_name.strip().lower()
    
    # Exakt match
    for stock_name in stock_names:
        if sales_normalized == stock_name.strip().lower():
            return stock_name
    
    # Partiell match (innehåller)
    for stock_name in stock_names:
        if sales_normalized in stock_name.strip().lower() or stock_name.strip().lower() in sales_normalized:
            return stock_name
    
    return None

def load_unit_mapping(file_path):
    """
    Läser product_sales_items och skapar mapping: (product_name, store_name) -> unit
    Returnerar unit_mapping dictionary.
    """
    file_path = Path(file_path)
    print(f"\nLäser enhetsinformation från {file_path}...")
    df = pd.read_csv(str(file_path))
    
    # Filtrera bort order med status "error"
    if 'order_status' in df.columns:
        df = df[df['order_status'] == 'complete'].copy()
    
    # Skapa mapping: (product_name, store_name) -> unit
    unit_mapping = {}
    
    for _, row in df.iterrows():
        if pd.notna(row.get('product_name')):
            product_name = str(row['product_name']).strip()
            store_name = str(row.get('store_name', '')).strip()
            
            if product_name:
                # Skapa nyckel baserat på produktnamn och butik
                key = (product_name.lower(), store_name)
                if key not in unit_mapping:
                    # Spara enhet om den finns
                    if 'unit' in row and pd.notna(row['unit']):
                        unit_mapping[key] = str(row['unit']).strip()
                    else:
                        unit_mapping[key] = 'st'
    
    print(f"  Laddade enhetsmappning för {len(unit_mapping)} produkt-butik-kombinationer")
    return unit_mapping

def get_product_unit(product_name, store_name, unit_mapping):
    """
    Hämtar enhet för en produkt från unit_mapping.
    Returnerar enhet eller 'st' som standard.
    """
    key = (product_name.lower(), store_name)
    if key in unit_mapping:
        return unit_mapping[key]
    return 'st'

def process_all_stores(sales_df, stock_df, parametrar, unit_mapping):
    """
    Processar alla butiker och produkter för att generera plocklista.
    Sparar en CSV-fil per butik i plocklistor/ mappen.
    parametrar: dictionary med leveransfrekvens per butik
    """
    print("\n" + "="*80)
    print("GENERERAR PROGNOSER OCH PLOCKLISTA")
    print("="*80)
    
    # Skapa output-katalog
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stores = sales_df['store'].unique()
    all_results = []
    
    for store_name in stores:
        # Hämta leveransfrekvens för denna butik (standard: 7 dagar om inte angiven)
        delivery_frequency = parametrar.get(store_name, 7)
        print(f"\nProcessar butik: {store_name} (leveransfrekvens: {delivery_frequency} dagar)")
        
        store_sales = sales_df[sales_df['store'] == store_name].copy()
        store_stock = stock_df[stock_df['store_name'] == store_name].copy()
        
        # Skapa lookup-dictionary för lagerdata
        stock_lookup = {}
        for _, row in store_stock.iterrows():
            product_name = row['product_name']
            product_name_normalized = product_name.lower().strip()
            stock_lookup[product_name_normalized] = {
                'product_name': product_name,
                'stock': max(0, row['stock']),  # Säkerställ att stock >= 0
                'stock_warning_limit': row['stock_warning_limit'],
                'product_id': row['product_id'],
                'product_code': row['product_code']
            }
        
        # Hämta alla produkter som sålts i denna butik
        products = store_sales['name'].unique()
        print(f"  Antal produkter i försäljningsdata: {len(products)}")
        
        store_results = []
        for product_name in products:
            product_sales = store_sales[store_sales['name'] == product_name].copy()
            
            # Hitta matchande produkt i lagerdata
            matched_stock = None
            product_name_normalized = product_name.strip().lower()
            
            # Försök hitta exakt match först
            if product_name_normalized in stock_lookup:
                matched_stock = stock_lookup[product_name_normalized]
            else:
                # Försök hitta partiell match
                for key, value in stock_lookup.items():
                    if product_name_normalized in key or key in product_name_normalized:
                        matched_stock = value
                        break
            
            # Om ingen match hittades, hoppa över produkten
            if matched_stock is None:
                continue
            
            # Prognostisera försäljning för leveransfrekvens-perioden
            try:
                predicted_sales = predict_product_sales(product_sales, delivery_frequency)
            except Exception as e:
                print(f"    Varning: Kunde inte prognostisera för {product_name}: {e}")
                predicted_sales = 0.0
            
            # Hämta nuvarande lagerstatus
            current_stock = matched_stock['stock']
            stock_warning_limit = matched_stock['stock_warning_limit']
            
            # Beräkna hur mycket som behöver fyllas på
            # (så att efter 3 dagar hamnar vi på stock_warning_limit)
            fill_up_quantity = calculate_picking_quantity(
                current_stock, 
                predicted_sales, 
                stock_warning_limit,
                delivery_frequency
            )
            
            # Hämta enhet för produkten
            unit = get_product_unit(product_name, store_name, unit_mapping)
            
            # Lägg till resultat
            # Formatera Produktkod som sträng (inte float)
            product_code = matched_stock['product_code']
            if pd.notna(product_code):
                # Konvertera till int om det är ett nummer, annars behåll som sträng
                try:
                    product_code = str(int(float(product_code)))
                except (ValueError, TypeError):
                    product_code = str(product_code)
            else:
                product_code = ''
            
            result = {
                'store_name': store_name,  # Lägg till butiksnamn för script 4
                'Produktnamn': matched_stock['product_name'],
                'Produktkod': product_code,
                'Produkt_ID': int(matched_stock['product_id']) if pd.notna(matched_stock['product_id']) else 0,
                'Leveransfrekvens_dagar': delivery_frequency,
                'saldo_denna_butik': current_stock,
                'Varningsgräns': stock_warning_limit,
                'Prognosticerad_försäljning': predicted_sales,
                'Påfyllningsbehov': fill_up_quantity,
                'Enhet': unit,
            }
            
            store_results.append(result)
        
        print(f"  Processade {len(store_results)} produkter med matchande lagerdata")
        
        if len(store_results) > 0:
            # Skapa DataFrame för denna butik
            store_df = pd.DataFrame(store_results)
            
            # Sortera efter behov av påfyllning (högst först)
            store_df = store_df.sort_values(['Påfyllningsbehov'], 
                                          ascending=[False])
            
            # Ordna kolumner i logisk ordning
            column_order = [
                'Produktnamn',
                'Produktkod',
                'Produkt_ID',
                'Leveransfrekvens_dagar',
                'saldo_denna_butik',
                'Varningsgräns',
                'Prognosticerad_försäljning',
                'Påfyllningsbehov',
                'Enhet',
            ]
            available_columns = [col for col in column_order if col in store_df.columns]
            store_df = store_df[available_columns]
            
            # Formatera data
            # Produkt_ID ska vara heltal (inte float)
            if 'Produkt_ID' in store_df.columns:
                store_df['Produkt_ID'] = store_df['Produkt_ID'].fillna(0).astype(int)
            
            # Produktkod ska vara sträng (inte float)
            if 'Produktkod' in store_df.columns:
                def format_product_code(val):
                    if pd.isna(val) or val == '':
                        return ''
                    try:
                        # Konvertera till int om möjligt, annars behåll som sträng
                        return str(int(float(val)))
                    except (ValueError, TypeError):
                        return str(val)
                store_df['Produktkod'] = store_df['Produktkod'].apply(format_product_code)
            
            # Prognosticerad_försäljning och Påfyllningsbehov ska ha 1 decimal
            if 'Prognosticerad_försäljning' in store_df.columns:
                store_df['Prognosticerad_försäljning'] = store_df['Prognosticerad_försäljning'].round(1)
            if 'Påfyllningsbehov' in store_df.columns:
                store_df['Påfyllningsbehov'] = store_df['Påfyllningsbehov'].round(1)
            # Saldo och Varningsgräns kan också ha 1 decimal för konsistens
            if 'saldo_denna_butik' in store_df.columns:
                store_df['saldo_denna_butik'] = store_df['saldo_denna_butik'].round(1)
            if 'Varningsgräns' in store_df.columns:
                store_df['Varningsgräns'] = store_df['Varningsgräns'].round(1)
            
            # Spara till CSV
            safe_store_name = store_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_file_csv = OUTPUT_DIR / f'{safe_store_name}.csv'
            store_df.to_csv(str(output_file_csv), index=False, sep=';', encoding='utf-8-sig')
            
            # Spara till Excel med auto-anpassade kolumnbredder
            if OPENPYXL_AVAILABLE:
                output_file_xlsx = OUTPUT_DIR / f'{safe_store_name}.xlsx'
                with pd.ExcelWriter(str(output_file_xlsx), engine='openpyxl') as writer:
                    store_df.to_excel(writer, sheet_name='Plocklista', index=False)
                    worksheet = writer.sheets['Plocklista']
                    
                    # Auto-anpassa kolumnbredder
                    for idx, col in enumerate(store_df.columns, 1):
                        max_length = max(
                            store_df[col].astype(str).map(len).max(),
                            len(str(col))
                        )
                        # Sätt en max-bredd för att undvika för breda kolumner
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[get_column_letter(idx)].width = adjusted_width
            
            print(f"  [OK] Sparade plocklista med {len(store_df)} produkter till {output_file_csv}")
            if OPENPYXL_AVAILABLE:
                print(f"  [OK] Sparade Excel-fil: {output_file_xlsx}")
            print(f"    Totalt påfyllningsbehov: {store_df['Påfyllningsbehov'].sum():.1f} enheter")
        
        all_results.extend(store_results)
    
    return all_results

def main():
    """Huvudfunktion"""
    print("="*80)
    print("PRODUKTVIS PROGNOS OCH PLOCKLISTA GENERATOR")
    print("="*80)
    print(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LEVERANSFREKVENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Hitta senaste filer automatiskt
    print("\nHittar senaste datafiler...")
    print(f"  Söker i: {DATA_DOWNLOADS_DIR}")
    print(f"  Katalog finns: {DATA_DOWNLOADS_DIR.exists()}")
    
    if not DATA_DOWNLOADS_DIR.exists():
        if getattr(sys, 'frozen', False):
            raise FileNotFoundError(
                f"Data-katalogen finns inte: {DATA_DOWNLOADS_DIR}\n"
                f"Kontrollera att nedladdade CSV-filer finns i samma mapp som executable-filen."
            )
        else:
            raise FileNotFoundError(
                f"Data-katalogen finns inte: {DATA_DOWNLOADS_DIR}\n"
                f"Kontrollera att 'data/downloads' katalogen finns i projektets rot."
            )
    
    latest_sales_file = find_latest_file('product_sales_*.csv', DATA_DOWNLOADS_DIR)
    latest_stock_file = find_latest_file('stock_report_*.csv', DATA_DOWNLOADS_DIR)
    latest_items_file = find_latest_file('product_sales_items_*.csv', DATA_DOWNLOADS_DIR)
    
    # Ladda leveransfrekvenser
    print(f"\nLaddar leveransfrekvenser från: {LEVERANSFREKVENS_PATH}")
    parametrar = load_leveransfrekvens(LEVERANSFREKVENS_PATH)
    
    # Ladda enhetsmappning från product_sales_items
    unit_mapping = load_unit_mapping(latest_items_file)
    
    # Ladda data
    sales_df = load_and_prepare_sales_data(latest_sales_file)
    stock_df = load_stock_data(latest_stock_file)
    
    # Processa alla butiker (sparar automatiskt CSV-filer per butik)
    results = process_all_stores(sales_df, stock_df, parametrar, unit_mapping)
    
    # Visa sammanfattning
    print(f"\n{'='*80}")
    print("SAMMANFATTNING")
    print(f"{'='*80}")
    
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        print(f"Totalt antal produkter i alla plocklistor: {len(results_df)}")
        print(f"Totalt antal enheter att fylla på: {results_df['Påfyllningsbehov'].sum():.1f}")
        
        # Formatera data
        # Produkt_ID ska vara heltal (inte float)
        if 'Produkt_ID' in results_df.columns:
            results_df['Produkt_ID'] = results_df['Produkt_ID'].fillna(0).astype(int)
        
        # Produktkod ska vara sträng (inte float)
        if 'Produktkod' in results_df.columns:
            def format_product_code(val):
                if pd.isna(val) or val == '':
                    return ''
                try:
                    return str(int(float(val)))
                except (ValueError, TypeError):
                    return str(val)
            results_df['Produktkod'] = results_df['Produktkod'].apply(format_product_code)
        
        # Prognosticerad_försäljning och Påfyllningsbehov ska ha 1 decimal
        if 'Prognosticerad_försäljning' in results_df.columns:
            results_df['Prognosticerad_försäljning'] = results_df['Prognosticerad_försäljning'].round(1)
        if 'Påfyllningsbehov' in results_df.columns:
            results_df['Påfyllningsbehov'] = results_df['Påfyllningsbehov'].round(1)
        # Saldo och Varningsgräns kan också ha 1 decimal för konsistens
        if 'saldo_denna_butik' in results_df.columns:
            results_df['saldo_denna_butik'] = results_df['saldo_denna_butik'].round(1)
        if 'Varningsgräns' in results_df.columns:
            results_df['Varningsgräns'] = results_df['Varningsgräns'].round(1)
        
        # Spara sammanfattande CSV-fil för script 4
        results_df.to_csv(str(PICKING_LIST_RESULTS_PATH), index=False, sep=';', encoding='utf-8-sig')
        print(f"\n[OK] Sparade sammanfattande plocklista till: {PICKING_LIST_RESULTS_PATH}")
        
        # Spara även som Excel
        if OPENPYXL_AVAILABLE:
            excel_path = PICKING_LIST_RESULTS_PATH.with_suffix('.xlsx')
            with pd.ExcelWriter(str(excel_path), engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Plocklistor', index=False)
                worksheet = writer.sheets['Plocklistor']
                
                # Auto-anpassa kolumnbredder
                for idx, col in enumerate(results_df.columns, 1):
                    max_length = max(
                        results_df[col].astype(str).map(len).max(),
                        len(str(col))
                    )
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[get_column_letter(idx)].width = adjusted_width
            print(f"[OK] Sparade Excel-fil: {excel_path}")
        
        # Räkna antal butiker
        stores_processed = sales_df['store'].nunique()
        print(f"\nAntal butiker processade: {stores_processed}")
        print(f"Plocklistor sparade i mappen: {OUTPUT_DIR}/")
    else:
        print("Inga produkter att processa.")
    
    print(f"\n{'='*80}")
    print("KLAR!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

