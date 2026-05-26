"""
Produktvis prognos och plocklista per shop

Detta script:
1. Gör produktvis prognos för varje shop baserat på historisk försäljning
2. Beräknar plocklista: hur mycket som behöver fyllas på i lager
   så att lagret hamnar precis på stock_warning_limit efter prognosticerad försäljning
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("Warning: openpyxl not available. Excel export will be skipped.")

# Ensure the script directory is on sys.path so we can import _common
# whether we are run as a script or via importlib (main.py / PyInstaller).
_script_dir = Path(__file__).parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from _common import (
    get_project_root,
    get_param_file_path,
    safe_filename,
    clear_readonly,
    preflight_writable,
    find_latest_file,
    load_leveransfrekvens,
    load_and_prepare_sales_data,
    predict_product_sales,
    normalize_name,
    normalize_name_series,
    count_normalised_changes,
)

PROJECT_ROOT = get_project_root()

# OUTPUT_DIR contains end-user files (PDF, XLSX). SYSTEM_DATA_DIR contains
# the working CSV files that the next pipeline step reads. Splitting these
# keeps the user's "output" folder clean - they only see the finished
# reports there, not the intermediate machine-readable data.
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'plocklistor'
SYSTEM_DATA_DIR = PROJECT_ROOT / 'system_data' / 'plocklistor'
DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'

PICKING_LIST_RESULTS_PATH = SYSTEM_DATA_DIR / 'picking_list_results.csv'
PICKING_LIST_SUMMARY_XLSX = OUTPUT_DIR / 'Plocklistor_sammanställning.xlsx'

# Exakta kolumner och ordning för plocklistor enligt
# data/template/picking_list_results.xlsx. Alla output-filer (per-butik
# och sammanställning) ska följa samma struktur så användaren kan lita
# på att layouten är identisk överallt.
PICKING_LIST_TEMPLATE_COLUMNS = [
    'store_name',
    'Produktnamn',
    'Påfyllningsbehov',
    'saldo_denna_butik',
    'Varningsgräns',
    'Enhet',
    'Leveransfrekvens_dagar',
    'Produktkod',
]


def _apply_picking_list_template(df):
    """
    Reducerar och omordnar df till de kolumner template-filen definierar.
    Saknade kolumner läggs till som tomma så ordningen alltid blir konsekvent;
    extra kolumner droppas så användaren slipper interna fält som
    Produkt_ID/Prognosticerad_försäljning i sin Excel.
    """
    for col in PICKING_LIST_TEMPLATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[PICKING_LIST_TEMPLATE_COLUMNS].copy()

LEVERANSFREKVENS_PATH = get_param_file_path('Leveransfrekvens.csv')


def load_stock_data(file_path):
    """Läser lagerdata"""
    file_path = Path(file_path)
    print(f"\nLäser lagerdata från {file_path}...")
    stock_df = pd.read_csv(str(file_path))

    before = len(stock_df)
    stock_df = stock_df.dropna(subset=['product_name', 'store_name'])
    dropped = before - len(stock_df)
    if dropped > 0:
        print(f"  Removed {dropped} rows with missing product_name or store_name")

    # Normalise whitespace so stock_report and sales data agree on store
    # identity even if one source has a stray NBSP, double space or BOM.
    raw_store = stock_df['store_name'].astype('string')
    raw_product = stock_df['product_name'].astype('string')
    stock_df['store_name'] = normalize_name_series(stock_df['store_name'])
    stock_df['product_name'] = normalize_name_series(stock_df['product_name'])
    changed_store = count_normalised_changes(raw_store, stock_df['store_name'])
    changed_product = count_normalised_changes(raw_product, stock_df['product_name'])
    if changed_store or changed_product:
        print(
            f"  Normaliserade whitespace i lagerdata: "
            f"{changed_store} butiksnamn och {changed_product} produktnamn justerade"
        )

    stock_df['stock'] = pd.to_numeric(stock_df['stock'], errors='coerce').fillna(0)
    stock_df['stock_warning_limit'] = pd.to_numeric(
        stock_df['stock_warning_limit'], errors='coerce').fillna(0)

    stock_df['product_name_normalized'] = stock_df['product_name'].str.lower()
    stock_df['store_name_normalized'] = stock_df['store_name']

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
    product_norm = normalize_name(product_name).lower()
    stock_names = stock_by_store[store_name]
    if product_norm in stock_names:
        return True
    for stock_name in stock_names:
        if product_norm in stock_name or stock_name in product_norm:
            return True
    return False


def filter_sales_to_stock(sales_df, stock_df):
    """Filtrerar bort produkter som inte finns i stock_report."""
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
    mask = [(s, n) in allowed_pairs for s, n in zip(
        sales_df['store'], sales_df['name'])]
    filtered_df = sales_df[mask].copy()
    removed = len(sales_df) - len(filtered_df)
    if removed > 0:
        print(f"  Filtrerade bort {removed} rader (produkter ej i stock_report)")
    return filtered_df


def filter_unit_mapping_to_stock(unit_mapping, stock_df):
    """Filtrerar unit_mapping till produkter som finns i stock_report."""
    stock_by_store = build_stock_index(stock_df)
    if not stock_by_store:
        return unit_mapping
    filtered = {}
    for (product_name, store_name), unit in unit_mapping.items():
        if product_exists_in_stock(product_name, store_name, stock_by_store):
            filtered[(product_name, store_name)] = unit
    return filtered


def calculate_picking_quantity(current_stock, predicted_sales_total, stock_warning_limit, delivery_frequency_days):
    """
    Beräknar hur mycket som behöver fyllas på i lager.

    Logik:
    - Om försäljningsprognosen förverkligas, ska lagret stanna på varningsgränsen
    - Formel: saldo_denna_butik + Påfyllningsbehov - Prognosticerad_försäljning = Varningsgräns
    - Påfyllningsbehov = Varningsgräns - saldo_denna_butik + Prognosticerad_försäljning

    Returnerar ett tecken-bärande tal:
    - Positivt: så många enheter behöver fyllas på.
    - 0: lagret landar exakt på varningsgränsen efter förväntad försäljning.
    - Negativt: överskott - så många enheter finns det "tillgodo" jämfört
      med varningsgränsen efter förväntad försäljning. Synligt i per-butik-
      filerna så användaren kan se var det finns extra lager.
    """
    return stock_warning_limit - current_stock + predicted_sales_total


def load_unit_mapping(file_path):
    """
    Läser product_sales_items och skapar mapping: (product_name, store_name) -> unit
    """
    file_path = Path(file_path)
    print(f"\nLäser enhetsinformation från {file_path}...")
    df = pd.read_csv(str(file_path))

    if 'order_status' in df.columns:
        df = df[df['order_status'] == 'complete'].copy()

    unit_mapping = {}
    for _, row in df.iterrows():
        if not pd.notna(row.get('product_name')):
            continue
        # normalize_name handles NBSP, ZWSP, double spaces - not just .strip().
        product_name = normalize_name(row['product_name'])
        store_name = normalize_name(row.get('store_name', ''))
        if not product_name:
            continue
        key = (product_name.lower(), store_name)
        if key in unit_mapping:
            continue
        if 'unit' in row and pd.notna(row['unit']):
            unit_mapping[key] = normalize_name(row['unit']) or 'st'
        else:
            unit_mapping[key] = 'st'

    print(f"  Laddade enhetsmappning för {len(unit_mapping)} produkt-butik-kombinationer")
    return unit_mapping


def get_product_unit(product_name, store_name, unit_mapping):
    """Hämtar enhet för en produkt från unit_mapping. Standard: 'st'."""
    key = (normalize_name(product_name).lower(), normalize_name(store_name))
    return unit_mapping.get(key, 'st')


def _write_dataframe(df, csv_path, excel_path=None, sheet_name='Plocklista'):
    """
    Skriver DataFrame till CSV och (om openpyxl är tillgängligt) Excel,
    rensar read-only-attributet på filerna så att alla användare kan
    redigera dem.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        clear_readonly(csv_path)
    df.to_csv(str(csv_path), index=False, sep=';', encoding='utf-8-sig')
    clear_readonly(csv_path)

    if excel_path is not None and OPENPYXL_AVAILABLE:
        excel_path = Path(excel_path)
        if excel_path.exists():
            clear_readonly(excel_path)
        with pd.ExcelWriter(str(excel_path), engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns, 1):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col)),
                )
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[get_column_letter(idx)].width = adjusted_width
        clear_readonly(excel_path)


def process_all_stores(sales_df, stock_df, parametrar, unit_mapping):
    """
    Processar alla butiker och produkter för att generera plocklista.
    Sparar en CSV-fil per butik i plocklistor/ mappen.
    parametrar: dictionary med leveransfrekvens per butik
    """
    print("\n" + "=" * 80)
    print("GENERERAR PROGNOSER OCH PLOCKLISTA")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stores = sales_df['store'].unique()
    all_results = []

    for store_name in stores:
        delivery_frequency = parametrar.get(store_name, 7)
        print(
            f"\nProcessar butik: {store_name} "
            f"(leveransfrekvens: {delivery_frequency} dagar)"
        )

        store_sales = sales_df[sales_df['store'] == store_name].copy()
        store_stock = stock_df[stock_df['store_name'] == store_name].copy()

        stock_lookup = {}
        for _, row in store_stock.iterrows():
            product_name = row['product_name']
            if not isinstance(product_name, str):
                continue
            product_name_normalized = normalize_name(product_name).lower()
            stock_lookup[product_name_normalized] = {
                'product_name': product_name,
                'stock': max(0, row['stock']),
                'stock_warning_limit': row['stock_warning_limit'],
                'product_id': row['product_id'],
                'product_code': row['product_code'],
            }

        products = store_sales['name'].unique()
        print(f"  Antal produkter i försäljningsdata: {len(products)}")

        store_results = []
        skipped_without_stock = 0
        for product_name in products:
            if pd.isna(product_name):
                continue
            product_name = str(product_name)
            product_sales = store_sales[store_sales['name'] == product_name].copy()

            matched_stock = None
            product_name_normalized = normalize_name(product_name).lower()
            if product_name_normalized in stock_lookup:
                matched_stock = stock_lookup[product_name_normalized]
            else:
                for key, value in stock_lookup.items():
                    if product_name_normalized in key or key in product_name_normalized:
                        matched_stock = value
                        break

            if matched_stock is None:
                skipped_without_stock += 1
                continue

            try:
                predicted_sales = predict_product_sales(product_sales, delivery_frequency)
            except Exception as e:
                print(f"    Varning: Kunde inte prognostisera för {product_name}: {e}")
                predicted_sales = 0.0

            current_stock = matched_stock['stock']
            stock_warning_limit = matched_stock['stock_warning_limit']

            fill_up_quantity = calculate_picking_quantity(
                current_stock,
                predicted_sales,
                stock_warning_limit,
                delivery_frequency,
            )

            unit = get_product_unit(product_name, store_name, unit_mapping)

            product_code = matched_stock['product_code']
            if pd.notna(product_code):
                try:
                    product_code = str(int(float(product_code)))
                except (ValueError, TypeError):
                    product_code = str(product_code)
            else:
                product_code = ''

            store_results.append({
                'store_name': store_name,
                'Produktnamn': matched_stock['product_name'],
                'Produktkod': product_code,
                'Produkt_ID': int(float(matched_stock['product_id']))
                if pd.notna(matched_stock['product_id']) else 0,
                'Leveransfrekvens_dagar': delivery_frequency,
                'saldo_denna_butik': current_stock,
                'Varningsgräns': stock_warning_limit,
                'Prognosticerad_försäljning': predicted_sales,
                'Påfyllningsbehov': fill_up_quantity,
                'Enhet': unit,
            })

        print(f"  Processade {len(store_results)} produkter med matchande lagerdata")
        if skipped_without_stock > 0:
            print(
                f"  Hoppade över {skipped_without_stock} produkter "
                f"utan stock_report-match (betraktade som utgångna)"
            )

        if len(store_results) > 0:
            store_df = pd.DataFrame(store_results)
            store_df = store_df.sort_values(['Påfyllningsbehov'], ascending=[False])

            if 'Produktkod' in store_df.columns:
                def format_product_code(val):
                    if pd.isna(val) or val == '':
                        return ''
                    try:
                        return str(int(float(val)))
                    except (ValueError, TypeError):
                        return str(val)
                store_df['Produktkod'] = store_df['Produktkod'].apply(format_product_code)

            for col in ('Påfyllningsbehov', 'saldo_denna_butik', 'Varningsgräns'):
                if col in store_df.columns:
                    store_df[col] = store_df[col].round(1)

            # Reducera till template-kolumnerna i template-ordning. Interna
            # fält (Produkt_ID, Prognosticerad_försäljning) hamnar inte i
            # användarens Excel.
            store_df = _apply_picking_list_template(store_df)

            safe_store_name = safe_filename(store_name)
            # CSV is internal machine data → system_data/. XLSX is what the
            # end user opens in Excel → output/.
            output_file_csv = SYSTEM_DATA_DIR / f'{safe_store_name}.csv'
            output_file_xlsx = OUTPUT_DIR / f'{safe_store_name}.xlsx' if OPENPYXL_AVAILABLE else None

            try:
                _write_dataframe(store_df, output_file_csv, output_file_xlsx, sheet_name='Plocklista')
            except PermissionError as e:
                print(
                    f"  [WARNING] Kunde inte skriva plocklista för {store_name}: {e}\n"
                    f"  Hoppar över denna butik men fortsätter med resten."
                )
                all_results.extend(store_results)
                continue

            print(f"  [OK] Sparade plocklista med {len(store_df)} produkter till {output_file_csv}")
            if OPENPYXL_AVAILABLE and output_file_xlsx is not None:
                print(f"  [OK] Sparade Excel-fil: {output_file_xlsx}")
            print(
                f"    Totalt påfyllningsbehov: "
                f"{store_df['Påfyllningsbehov'].sum():.1f} enheter"
            )

        all_results.extend(store_results)

    return all_results


def main():
    """Huvudfunktion"""
    print("=" * 80)
    print("PRODUKTVIS PROGNOS OCH PLOCKLISTA GENERATOR")
    print("=" * 80)
    print(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LEVERANSFREKVENS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Fail fast if either folder isn't writable, rather than blowing up
    # mid-loop after we've already spent time on forecasting.
    preflight_writable(OUTPUT_DIR)
    preflight_writable(SYSTEM_DATA_DIR)

    print("\nHittar senaste datafiler...")
    print(f"  Söker i: {DATA_DOWNLOADS_DIR}")
    print(f"  Katalog finns: {DATA_DOWNLOADS_DIR.exists()}")

    if not DATA_DOWNLOADS_DIR.exists():
        if getattr(sys, 'frozen', False):
            raise FileNotFoundError(
                f"Data-katalogen finns inte: {DATA_DOWNLOADS_DIR}\n"
                f"Kontrollera att nedladdade CSV-filer finns i samma mapp som executable-filen."
            )
        raise FileNotFoundError(
            f"Data-katalogen finns inte: {DATA_DOWNLOADS_DIR}\n"
            f"Kontrollera att 'data/downloads' katalogen finns i projektets rot."
        )

    latest_sales_file = find_latest_file('product_sales_*.csv', DATA_DOWNLOADS_DIR)
    latest_stock_file = find_latest_file('stock_report_*.csv', DATA_DOWNLOADS_DIR)
    latest_items_file = find_latest_file('product_sales_items_*.csv', DATA_DOWNLOADS_DIR)

    print(f"\nLaddar leveransfrekvenser från: {LEVERANSFREKVENS_PATH}")
    parametrar = load_leveransfrekvens(LEVERANSFREKVENS_PATH)

    unit_mapping = load_unit_mapping(latest_items_file)

    sales_df = load_and_prepare_sales_data(latest_sales_file)
    stock_df = load_stock_data(latest_stock_file)

    sales_df = filter_sales_to_stock(sales_df, stock_df)
    unit_mapping = filter_unit_mapping_to_stock(unit_mapping, stock_df)

    # Surface stores that have sales data but no entry in Leveransfrekvens.csv.
    # If the name only differs by whitespace this is where a typo becomes
    # visible - both names appear in the warning and the user can fix the CSV.
    sales_stores = set(sales_df['store'].unique())
    config_stores = set(parametrar.keys())
    unknown_stores = sorted(sales_stores - config_stores)
    if unknown_stores:
        print(
            "\n[INFO] Följande butiker saknas i Leveransfrekvens.csv "
            "(default 7 dagar används):"
        )
        for s in unknown_stores:
            # repr() makes any hidden whitespace visible in the log.
            print(f"  - {s!r}")
        configured_without_sales = sorted(config_stores - sales_stores)
        if configured_without_sales:
            print(
                "  (Tips: dessa butiker finns i Leveransfrekvens.csv men "
                "INTE i försäljningsdata - jämför med listan ovan om "
                "namnen råkar skilja sig åt i whitespace:)"
            )
            for s in configured_without_sales:
                print(f"  - {s!r}")

    results = process_all_stores(sales_df, stock_df, parametrar, unit_mapping)

    print(f"\n{'=' * 80}")
    print("SAMMANFATTNING")
    print(f"{'=' * 80}")

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        total_rows = len(results_df)

        if 'Produktkod' in results_df.columns:
            def format_product_code(val):
                if pd.isna(val) or val == '':
                    return ''
                try:
                    return str(int(float(val)))
                except (ValueError, TypeError):
                    return str(val)
            results_df['Produktkod'] = results_df['Produktkod'].apply(format_product_code)

        for col in ('Påfyllningsbehov', 'saldo_denna_butik', 'Varningsgräns'):
            if col in results_df.columns:
                results_df[col] = results_df[col].round(1)

        # Filtrera sammanställningen till enbart rader med faktiskt
        # påfyllningsbehov (>0). Per-butik-filerna behåller all data
        # (inklusive negativa värden som visar överskott), men den
        # här sammanfattningen är en åtgärdslista - bara det som
        # faktiskt behöver fyllas på.
        results_df = results_df[results_df['Påfyllningsbehov'] > 0].copy()
        results_df = results_df.sort_values(
            ['store_name', 'Påfyllningsbehov'], ascending=[True, False]
        )

        print(
            f"Totalt antal rader (alla butiker, även överskott): {total_rows}"
        )
        print(
            f"Rader med påfyllningsbehov (>0): {len(results_df)} "
            f"(övriga filtreras bort från picking_list_results.csv)"
        )
        if len(results_df) > 0:
            print(
                f"Totalt antal enheter att fylla på: "
                f"{results_df['Påfyllningsbehov'].sum():.1f}"
            )

        # Samma kolumnstruktur här som i per-butik-filerna, enligt template.
        results_df = _apply_picking_list_template(results_df)

        excel_summary = PICKING_LIST_SUMMARY_XLSX if OPENPYXL_AVAILABLE else None
        _write_dataframe(
            results_df, PICKING_LIST_RESULTS_PATH, excel_summary, sheet_name='Plocklistor'
        )
        print(f"\n[OK] Sparade sammanfattande plocklista (CSV) till: {PICKING_LIST_RESULTS_PATH}")
        if excel_summary is not None:
            print(f"[OK] Sparade Excel-fil: {excel_summary}")

        stores_processed = sales_df['store'].nunique()
        print(f"\nAntal butiker processade: {stores_processed}")
        print(f"Excel-plocklistor (för användare) sparade i: {OUTPUT_DIR}/")
        print(f"CSV-plocklistor (systemdata) sparade i:     {SYSTEM_DATA_DIR}/")
    else:
        print("Inga produkter att processa.")

    print(f"\n{'=' * 80}")
    print("KLAR!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
