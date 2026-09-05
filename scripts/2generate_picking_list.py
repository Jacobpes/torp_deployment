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
    import openpyxl  # noqa: F401
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
    find_item_metadata_file,
    load_leveransfrekvens,
    load_sales_history,
    predict_product_sales,
    normalize_name,
    normalize_name_series,
    count_normalised_changes,
    filter_active_products,
    filter_garbage_product_names,
    deduplicate_stock,
    assert_not_parameter_path,
    reraise_if_write_locked,
    load_and_sync_product_format,
    sort_dataframe_by_product_format,
    sort_dataframe_by_store_and_format,
    write_formatted_excel,
    preflight_torp_excel_files,
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

# Avancerad plocklista: INKLUDERAR rader med Påfyllningsbehov <= 0 så
# användaren kan se vilka produkter som har överskott (negativa värden) i
# olika butiker - användbart för att bestämma omfördelningar mellan butiker.
ADVANCED_PICKING_LIST_XLSX = OUTPUT_DIR / 'avancerad_plocklista.xlsx'
ADVANCED_PICKING_LIST_CSV = SYSTEM_DATA_DIR / 'avancerad_plocklista.csv'

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
    """Läser lagerdata och filtrerar bort inaktiva produkter."""
    file_path = Path(file_path)
    print(f"\nLäser lagerdata från {file_path}...")
    stock_df = pd.read_csv(str(file_path))

    # Inaktiva produkter (product_status != 1) ska ignoreras genom hela
    # pipelinen. Görs här innan något annat filter byggs så att alla
    # nedströms steg (filter_sales_to_stock, build_stock_index, ...)
    # automatiskt arbetar mot endast aktiva produkter.
    stock_df = filter_active_products(stock_df)

    # Skräpnamn (t.ex. en produkt som heter bara "R") ska aldrig nå
    # plocklistan. Filtreras tidigt så hela pipelinen slipper se dem.
    stock_df = filter_garbage_product_names(stock_df)

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

    # Slå ihop duplicerade (butik, produktnamn)-rader så plocklistan inte
    # visar samma produkt flera gånger. Görs efter numeric coerce så att
    # SUM-aggregeringen får riktiga floats.
    stock_df = deduplicate_stock(stock_df)

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


def _write_dataframe(df, csv_path, excel_path=None, sheet_name='Plocklista',
                     product_format=None):
    """
    Skriver DataFrame till CSV och (om openpyxl är tillgängligt) Excel,
    rensar read-only-attributet på filerna så att alla användare kan
    redigera dem. Excel-filer får produktnamn-färger enligt format.xlsx.
    """
    csv_path = Path(csv_path)
    assert_not_parameter_path(csv_path)
    if excel_path is not None:
        assert_not_parameter_path(excel_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        clear_readonly(csv_path)
    df.to_csv(str(csv_path), index=False, sep=';', encoding='utf-8-sig')
    clear_readonly(csv_path)

    if excel_path is not None and OPENPYXL_AVAILABLE:
        excel_path = Path(excel_path)
        if excel_path.exists():
            clear_readonly(excel_path)
        try:
            write_formatted_excel(
                df, excel_path, sheet_name, product_format, product_col='Produktnamn',
            )
            clear_readonly(excel_path)
        except OSError as e:
            reraise_if_write_locked(excel_path, e)


def _normalise_product_code(product_code):
    """Format a stock_report product_code as a clean integer string when possible."""
    if not pd.notna(product_code) or product_code == '':
        return ''
    try:
        return str(int(float(product_code)))
    except (ValueError, TypeError):
        return str(product_code)


def _normalise_product_id(product_id):
    """Convert stock_report product_id to int; 0 if missing/garbage."""
    if not pd.notna(product_id):
        return 0
    try:
        return int(float(product_id))
    except (ValueError, TypeError):
        return 0


def _match_sales_for_product(product_name_normalized, sales_by_product):
    """
    Look up the sales rows for a given normalised product name in this store.

    Two-pass match (mirrors the old sales-first iteration but flipped):
      1. Exact normalised-lowercase name hit.
      2. Substring fallback in either direction (handles trailing notes
         like "Mjölkdryck 3 % Laktosfri 1 L" vs "Mjölkdryck 3% Laktosfri 1L").

    Returns the sales DataFrame slice or None when no sales row exists at
    all for this product+store pair (treated as predicted_sales=0).
    """
    if not sales_by_product:
        return None
    hit = sales_by_product.get(product_name_normalized)
    if hit is not None:
        return hit
    for sales_name_norm, group in sales_by_product.items():
        if (product_name_normalized in sales_name_norm
                or sales_name_norm in product_name_normalized):
            return group
    return None


def process_all_stores(sales_df, stock_df, parametrar, unit_mapping,
                       product_format=None):
    """
    Generate the picking list for every (store, product) pair that exists in
    stock_report. Earlier versions iterated sales-active products only,
    which silently dropped any product that hadn't sold recently (or any
    store that hadn't sold at all on the day of the daily snapshot). The
    user-facing advanced picking list ("avancerad plocklista") explicitly
    needs ALL products listed so they can be redistributed between stores
    even when they aren't selling, so this loop now uses stock_report as
    the canonical universe and looks up sales as an optional input to
    the forecast.

    For products with no historical sales, predicted_sales = 0 and
    Påfyllningsbehov = Varningsgräns − saldo_denna_butik (i.e. just
    "bring stock up to the warning limit").
    """
    print("\n" + "=" * 80)
    print("GENERERAR PROGNOSER OCH PLOCKLISTA")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Use stock_report as the source of truth for stores AND products.
    # Sorting gives deterministic per-run output ordering and a stable diff
    # when comparing two days' picking lists side-by-side.
    stores = sorted(
        s for s in stock_df['store_name'].unique() if isinstance(s, str)
    )
    all_results = []

    for store_name in stores:
        delivery_frequency = parametrar.get(store_name, 7)
        print(
            f"\nProcessar butik: {store_name} "
            f"(leveransfrekvens: {delivery_frequency} dagar)"
        )

        store_stock = stock_df[stock_df['store_name'] == store_name].copy()
        store_sales = sales_df[sales_df['store'] == store_name].copy()

        # Pre-index sales by normalised product name so the per-product
        # lookup inside the hot loop is O(1) instead of re-filtering
        # the full sales_df ~500 times per store.
        sales_by_product = {}
        for sales_name, group in store_sales.groupby('name', sort=False):
            if not isinstance(sales_name, str) or not sales_name:
                continue
            key = normalize_name(sales_name).lower()
            if key:
                sales_by_product[key] = group

        print(
            f"  Antal produkter i stock_report: {len(store_stock)} "
            f"(varav {len(sales_by_product)} med försäljningshistorik)"
        )

        store_results = []
        no_sales_count = 0
        for _, stock_row in store_stock.iterrows():
            product_name = stock_row.get('product_name')
            if not isinstance(product_name, str) or not product_name.strip():
                continue
            product_name_normalized = normalize_name(product_name).lower()

            product_sales = _match_sales_for_product(
                product_name_normalized, sales_by_product
            )

            if product_sales is None or len(product_sales) == 0:
                predicted_sales = 0.0
                no_sales_count += 1
            else:
                try:
                    predicted_sales = predict_product_sales(
                        product_sales, delivery_frequency
                    )
                except Exception as e:
                    print(
                        f"    Varning: Kunde inte prognostisera för "
                        f"{product_name}: {e}"
                    )
                    predicted_sales = 0.0

            current_stock = max(0.0, float(stock_row.get('stock', 0) or 0))
            stock_warning_limit = float(stock_row.get('stock_warning_limit', 0) or 0)

            fill_up_quantity = calculate_picking_quantity(
                current_stock,
                predicted_sales,
                stock_warning_limit,
                delivery_frequency,
            )

            unit = get_product_unit(product_name, store_name, unit_mapping)
            product_code = _normalise_product_code(stock_row.get('product_code'))
            product_id = _normalise_product_id(stock_row.get('product_id'))

            store_results.append({
                'store_name': store_name,
                'Produktnamn': product_name,
                'Produktkod': product_code,
                'Produkt_ID': product_id,
                'Leveransfrekvens_dagar': delivery_frequency,
                'saldo_denna_butik': current_stock,
                'Varningsgräns': stock_warning_limit,
                'Prognosticerad_försäljning': predicted_sales,
                'Påfyllningsbehov': fill_up_quantity,
                'Enhet': unit,
            })

        if no_sales_count > 0:
            print(
                f"  {no_sales_count} produkter saknar försäljningshistorik "
                f"(predicted_sales=0; Påfyllningsbehov = Varningsgräns - saldo)"
            )
        print(f"  Processade {len(store_results)} produkter totalt")

        if len(store_results) > 0:
            store_df = pd.DataFrame(store_results)
            store_df = sort_dataframe_by_product_format(
                store_df, product_format, 'Produktnamn',
            )

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
                _write_dataframe(
                    store_df, output_file_csv, output_file_xlsx,
                    sheet_name='Plocklista', product_format=product_format,
                )
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
    preflight_torp_excel_files(PROJECT_ROOT)

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

    latest_stock_file = find_latest_file('stock_report_*.csv', DATA_DOWNLOADS_DIR)
    # Items-filen från SFTP är ibland tom (bara header, ~200 byte) eller
    # saknas helt. find_item_metadata_file faller då tillbaka på senaste
    # dagliga product_sales_*.csv (samma item-schema) istället för att
    # avbryta hela körningen - enhetsmappningen behöver bara unit-kolumnen.
    latest_items_file = find_item_metadata_file(
        DATA_DOWNLOADS_DIR, min_size_bytes=1024
    )

    print(f"\nLaddar leveransfrekvenser från: {LEVERANSFREKVENS_PATH}")
    parametrar = load_leveransfrekvens(LEVERANSFREKVENS_PATH)

    unit_mapping = load_unit_mapping(latest_items_file)

    # Forecasting needs the full historical sales window, not just the
    # latest one-day product_sales_*.csv snapshot. With one day's data,
    # predict_product_sales degenerates to 0 for almost every product
    # (the 4-week-average baseline has no complete weeks to average), so
    # we use load_sales_history which stitches the cumulative items file
    # together with newer daily snapshots.
    print("\nLäser försäljningshistorik (items-fil + senaste dagliga snapshots)...")
    sales_df = load_sales_history(DATA_DOWNLOADS_DIR)
    stock_df = load_stock_data(latest_stock_file)

    sales_df = filter_sales_to_stock(sales_df, stock_df)
    unit_mapping = filter_unit_mapping_to_stock(unit_mapping, stock_df)

    # Surface stores that we'll process (i.e. stores with stock data) but
    # which lack an entry in Leveransfrekvens.csv. If the name only differs
    # by whitespace this is where a typo becomes visible - both the stock
    # name and the config name appear in the warning and the user can fix
    # the CSV by hand.
    stock_stores = set(stock_df['store_name'].unique())
    config_stores = set(parametrar.keys())
    unknown_stores = sorted(stock_stores - config_stores)
    if unknown_stores:
        print(
            "\n[INFO] Följande butiker (från stock_report) saknas i "
            "Leveransfrekvens.csv (default 7 dagar används):"
        )
        for s in unknown_stores:
            # repr() makes any hidden whitespace visible in the log.
            print(f"  - {s!r}")
        configured_without_stock = sorted(config_stores - stock_stores)
        if configured_without_stock:
            print(
                "  (Tips: dessa butiker finns i Leveransfrekvens.csv men "
                "INTE i stock_report - jämför med listan ovan om "
                "namnen råkar skilja sig åt i whitespace:)"
            )
            for s in configured_without_stock:
                print(f"  - {s!r}")

    product_format = load_and_sync_product_format(stock_df)

    results = process_all_stores(
        sales_df, stock_df, parametrar, unit_mapping, product_format=product_format,
    )

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

        # Avancerad plocklista: ALLA rader inkl. överskott (negativa
        # Påfyllningsbehov). Sorterad så att största behoven är överst
        # och största överskotten längst ner per butik - underlättar
        # beslut om omfördelning mellan butiker.
        advanced_df = results_df.copy()
        advanced_df = sort_dataframe_by_store_and_format(
            advanced_df, product_format, 'store_name', 'Produktnamn',
        )
        advanced_df = _apply_picking_list_template(advanced_df)
        advanced_xlsx = ADVANCED_PICKING_LIST_XLSX if OPENPYXL_AVAILABLE else None
        _write_dataframe(
            advanced_df, ADVANCED_PICKING_LIST_CSV, advanced_xlsx,
            sheet_name='Avancerad plocklista', product_format=product_format,
        )
        print(
            f"\n[OK] Sparade avancerad plocklista (alla rader inkl. överskott) "
            f"med {len(advanced_df)} rader"
        )
        if advanced_xlsx is not None:
            print(f"     Excel: {advanced_xlsx}")
        print(f"     CSV:   {ADVANCED_PICKING_LIST_CSV}")

        # Filtrera sammanställningen till enbart rader med faktiskt
        # påfyllningsbehov (>0). Per-butik-filerna behåller all data
        # (inklusive negativa värden som visar överskott), men den
        # här sammanfattningen är en åtgärdslista - bara det som
        # faktiskt behöver fyllas på.
        results_df = results_df[results_df['Påfyllningsbehov'] > 0].copy()
        results_df = sort_dataframe_by_store_and_format(
            results_df, product_format, 'store_name', 'Produktnamn',
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
            results_df, PICKING_LIST_RESULTS_PATH, excel_summary,
            sheet_name='Plocklistor', product_format=product_format,
        )
        print(f"\n[OK] Sparade sammanfattande plocklista (CSV) till: {PICKING_LIST_RESULTS_PATH}")
        if excel_summary is not None:
            print(f"[OK] Sparade Excel-fil: {excel_summary}")

        stores_processed = stock_df['store_name'].nunique()
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
