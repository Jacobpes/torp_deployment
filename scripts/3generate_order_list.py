"""
Orderlista baserad på beställningsfrekvens per leverantör

Detta script:
1. Läser beställningsfrekvens från parametrar/Beställningsfrekvens.csv
2. Hämtar leverantörsinformation från product_sales_items
3. För varje leverantör, prognostiserar försäljning för beställningsfrekvens-perioden
4. Skapar detaljerad orderlista per leverantör i output/orderlistor/
5. Skapar ren leverantörs-Excel (Kupa kod, produkt, enhet, behov) i
   output/orderlistor_leverantor/ för verifiering och vidarebefordran
"""

import csv
import re
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
    load_sales_history,
    predict_product_sales,
    normalize_name,
    normalize_name_series,
    count_normalised_changes,
    filter_active_products,
    filter_garbage_product_names,
    deduplicate_stock,
    assert_not_parameter_path,
    load_price_log,
    save_price_log,
    track_purchase_price,
    extract_latest_prices_from_items,
    read_text_with_encoding_fallback,
    reraise_if_write_locked,
    load_product_format,
    sort_dataframe_by_product_format,
    write_formatted_excel,
    preflight_torp_excel_files,
)

PROJECT_ROOT = get_project_root()

# OUTPUT_DIR contains the detailed XLSX files the user opens for planning.
# SUPPLIER_OUTPUT_DIR holds a clean one-file-per-supplier Excel ready to
# verify and forward without structural edits. SYSTEM_DATA_DIR holds the
# raw CSV copies for downstream tooling / debugging.
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'orderlistor'
SUPPLIER_OUTPUT_DIR = PROJECT_ROOT / 'output' / 'orderlistor_leverantor'
SYSTEM_DATA_DIR = PROJECT_ROOT / 'system_data' / 'orderlistor'
DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'

# Inköpspris-loggen ligger på en stabil plats utanför per-leverantörs-
# katalogerna så den följer med över alla körningar. Loggas historik per
# product_code så vi kan upptäcka prisförändringar mellan körningar.
PRICE_LOG_PATH = PROJECT_ROOT / 'system_data' / 'inkopspris_log.json'

BESTALLNINGSFREKVENS_PATH = get_param_file_path('Beställningsfrekvens.csv')


def load_bestallningsfrekvens(file_path):
    """
    Läser parametrar/Beställningsfrekvens.csv → {leverantör: frekvens_dagar}.

    Faller tillbaka till tomt dict (→ default 7 dagar per leverantör) om filen
    saknas eller är trasig, istället för att krascha hela steg 3. Det matchar
    beteendet hos load_leveransfrekvens och gör att .exe:n förblir körbar
    även om någon råkar radera parameterfilen från sharen.
    """
    file_path = Path(file_path)
    print(f"\nLäser beställningsfrekvens från {file_path}...")

    if not file_path.exists():
        print(
            f"  Varning: Beställningsfrekvens-fil saknas: {file_path}\n"
            f"  Använder standardvärde: 7 dagar för alla leverantörer."
        )
        return {}

    try:
        text = read_text_with_encoding_fallback(file_path)
        rows = []
        reader = csv.reader(text.splitlines(), delimiter=';')
        for row in reader:
            rows.append(row)
    except (OSError, UnicodeDecodeError) as e:
        print(
            f"  Varning: Kunde inte läsa {file_path}: {e}\n"
            f"  Använder standardvärde: 7 dagar för alla leverantörer."
        )
        return {}

    if len(rows) < 2:
        print(
            f"  Fel: CSV-filen har för få rader ({len(rows)}). "
            f"Förväntar minst header + 1 datarad."
        )
        return {}

    header = rows[0]
    data_rows = rows[1:]

    print(f"  Hittade {len(header)} kolumner: {header[:4]}...")
    print(f"  Hittade {len(data_rows)} datarader")

    frekvenser = {}

    leverantor_col_idx = 0
    frekvens_col_idx = 2

    if len(header) <= frekvens_col_idx:
        print(
            f"  Fel: CSV-filen har för få kolumner ({len(header)}). "
            f"Förväntar minst {frekvens_col_idx + 1} kolumner."
        )
        return frekvenser

    print(
        f"  Använder kolumn {leverantor_col_idx} för Leverantör: "
        f"'{header[leverantor_col_idx]}'"
    )
    print(
        f"  Använder kolumn {frekvens_col_idx} för Frekvens: "
        f"'{header[frekvens_col_idx]}'"
    )

    for row in data_rows:
        if len(row) <= max(leverantor_col_idx, frekvens_col_idx):
            continue
        # normalize_name (not just .strip()) so suppliers with NBSP, ZWSP
        # or stray double spaces in the CSV match the form coming from
        # product_sales_items via load_supplier_mapping.
        leverantor_raw = normalize_name(
            row[leverantor_col_idx] if leverantor_col_idx < len(row) else ''
        )
        frekvens_raw = (
            row[frekvens_col_idx].strip() if frekvens_col_idx < len(row) else ''
        )

        leverantor = leverantor_raw if leverantor_raw else None
        if not leverantor or leverantor.lower() in ['nan', 'none', '']:
            continue
        if not frekvens_raw or frekvens_raw.strip() == '':
            continue

        try:
            freq_str = str(frekvens_raw).strip()
            freq_clean = ''.join(c for c in freq_str if c.isdigit() or c == '.')
            if freq_clean:
                frekvens_dagar = int(float(freq_clean))
                if 1 <= frekvens_dagar <= 365:
                    frekvenser[leverantor] = frekvens_dagar
                else:
                    print(
                        f"  Varning: Ovanlig frekvens för '{leverantor}': "
                        f"{frekvens_dagar} dagar (hoppar över)"
                    )
            else:
                print(
                    f"  Varning: Kunde inte extrahera nummer från frekvens för "
                    f"'{leverantor}': '{frekvens_raw}'"
                )
        except (ValueError, TypeError) as e:
            print(
                f"  Varning: Kunde inte tolka frekvens för '{leverantor}': "
                f"'{frekvens_raw}' (fel: {e})"
            )

    print(f"  Laddade beställningsfrekvenser för {len(frekvenser)} leverantörer")
    if len(frekvenser) > 0:
        print(f"  Exempel leverantörer: {list(frekvenser.keys())[:5]}")
        for lev, freq in list(frekvenser.items())[:5]:
            print(f"    {lev}: {freq} dagar")
    else:
        print(
            "  VARNING: Inga leverantörer laddades! "
            "Kontrollera CSV-filen."
        )

    return frekvenser


def load_supplier_mapping(file_path):
    """
    Läser product_sales_items och returnerar
    (supplier_mapping, unit_mapping, kupa_kod_mapping).

    - supplier_mapping / unit_mapping: nyckel (product_name_lower, store_name)
    - kupa_kod_mapping: nyckel product_name_lower → supplier_item_code
      (leverantörens artikelkod / "kupa kod")
    """
    file_path = Path(file_path)
    print(f"\nLäser leverantörsinformation från {file_path}...")
    df = pd.read_csv(str(file_path))

    if 'order_status' in df.columns:
        df = df[df['order_status'] == 'complete'].copy()

    supplier_mapping = {}
    unit_mapping = {}
    kupa_kod_mapping = {}

    has_kupa = 'supplier_item_code' in df.columns

    for _, row in df.iterrows():
        if not (pd.notna(row.get('supplier_name')) and pd.notna(row.get('product_name'))):
            continue
        product_name = normalize_name(row['product_name'])
        supplier_name = normalize_name(row['supplier_name'])
        store_name = normalize_name(row.get('store_name', ''))
        if not (product_name and supplier_name):
            continue
        key = (product_name.lower(), store_name)
        if key not in supplier_mapping:
            supplier_mapping[key] = supplier_name
            if 'unit' in row and pd.notna(row['unit']):
                unit_mapping[key] = normalize_name(row['unit']) or 'st'
            else:
                unit_mapping[key] = 'st'
        if has_kupa and product_name.lower() not in kupa_kod_mapping:
            raw_kupa = row.get('supplier_item_code')
            if pd.notna(raw_kupa) and str(raw_kupa).strip() not in ('', 'nan'):
                kupa_val = str(raw_kupa).strip()
                # Drop trailing .0 from numeric Excel/CSV floats.
                if kupa_val.endswith('.0'):
                    try:
                        kupa_val = str(int(float(kupa_val)))
                    except (ValueError, TypeError):
                        pass
                kupa_kod_mapping[product_name.lower()] = kupa_val

    print(
        f"  Laddade leverantörsmappning för {len(supplier_mapping)} "
        f"produkt-butik-kombinationer"
    )
    print(f"  Laddade kupa kod för {len(kupa_kod_mapping)} produkter")
    return supplier_mapping, unit_mapping, kupa_kod_mapping


def get_product_unit(product_name, store_name, unit_mapping):
    """Hämtar enhet för en produkt från unit_mapping. Standard: 'st'."""
    key = (normalize_name(product_name).lower(), normalize_name(store_name))
    return unit_mapping.get(key, 'st')


def load_stock_data(file_path):
    """Läser lagerdata och filtrerar bort inaktiva produkter."""
    file_path = Path(file_path)
    print(f"\nLäser lagerdata från {file_path}...")
    stock_df = pd.read_csv(str(file_path))

    # Inaktiva produkter (product_status != 1) ska ignoreras genom hela
    # pipelinen. Filtrera tidigt så build_stock_index och filter_sales_to_stock
    # automatiskt utesluter dem från orderlistorna.
    stock_df = filter_active_products(stock_df)

    # Skräpnamn ('R' etc.) ska aldrig nå orderlistan.
    stock_df = filter_garbage_product_names(stock_df)

    before = len(stock_df)
    stock_df = stock_df.dropna(subset=['product_name', 'store_name'])
    if before != len(stock_df):
        print(f"  Removed {before - len(stock_df)} rows with missing product_name or store_name")

    # Normalise whitespace so stock_report and sales data align on store/product
    # identity even with stray NBSP, ZWSP, double spaces or BOM.
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

    # Slå ihop duplicerade (butik, produktnamn)-rader så orderlistan inte
    # visar samma produkt flera gånger.
    stock_df = deduplicate_stock(stock_df)

    stock_df['product_name_normalized'] = stock_df['product_name'].str.lower()
    stock_df['stock'] = stock_df['stock'].clip(lower=0)

    print(f"  Laddat {len(stock_df)} rader")
    print(f"  Butiker: {stock_df['store_name'].nunique()}")
    print(f"  Produkter: {stock_df['product_name'].nunique()}")

    return stock_df


def build_stock_index(stock_df):
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


def filter_mapping_to_stock(mapping, stock_df):
    stock_by_store = build_stock_index(stock_df)
    if not stock_by_store:
        return mapping
    filtered = {}
    for (product_name, store_name), value in mapping.items():
        if product_exists_in_stock(product_name, store_name, stock_by_store):
            filtered[(product_name, store_name)] = value
    return filtered


def get_product_stock_info(stock_df, product_name, store_name):
    """Returns (stock, stock_warning_limit) or (0, 0) if not found."""
    product_normalized = normalize_name(product_name).lower()
    store_normalized = normalize_name(store_name)

    matched = stock_df[
        (stock_df['product_name_normalized'] == product_normalized)
        & (stock_df['store_name'] == store_normalized)
    ]
    if len(matched) > 0:
        return float(matched.iloc[0]['stock']), float(matched.iloc[0]['stock_warning_limit'])

    matched = stock_df[
        (stock_df['product_name_normalized'].str.contains(
            product_normalized, na=False, regex=False))
        & (stock_df['store_name'] == store_normalized)
    ]
    if len(matched) > 0:
        return float(matched.iloc[0]['stock']), float(matched.iloc[0]['stock_warning_limit'])

    return 0.0, 0.0


def calculate_bestallningsbehov(current_stock_total, predicted_sales_total,
                                stock_warning_limit_total, order_frequency_days):
    """
    Efter order_frequency_days ska:
      current_stock_total + beställningsbehov - predicted_sales_total >= stock_warning_limit_total
    """
    bestallningsbehov = stock_warning_limit_total + predicted_sales_total - current_stock_total
    return max(0.0, bestallningsbehov)


def match_supplier_name(supplier_from_mapping, supplier_from_frekvens):
    """Försöker matcha leverantörsnamn mellan olika källor."""
    if not supplier_from_mapping or not supplier_from_frekvens:
        return None

    mapping_normalized = normalize_name(supplier_from_mapping).lower()
    frekvens_normalized = normalize_name(supplier_from_frekvens).lower()

    def strip_company_suffixes(name):
        # Strip common Finnish/Swedish corporate suffixes so e.g. "Snellman"
        # matches "Snellman Oy". Whitespace already normalised by caller, but
        # collapse again after suffix removal in case "Foo Ab" -> "Foo  ".
        name = re.sub(r'\b(ab|oy|ab oy|aboy)\b', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    mapping_clean = strip_company_suffixes(mapping_normalized)
    frekvens_clean = strip_company_suffixes(frekvens_normalized)

    if mapping_clean == frekvens_clean or mapping_normalized == frekvens_normalized:
        return supplier_from_frekvens
    if mapping_clean in frekvens_clean or frekvens_clean in mapping_clean:
        return supplier_from_frekvens

    mapping_first = mapping_clean.split()[0] if mapping_clean.split() else ''
    frekvens_first = frekvens_clean.split()[0] if frekvens_clean.split() else ''
    if mapping_first and frekvens_first and mapping_first == frekvens_first:
        return supplier_from_frekvens

    mapping_last = mapping_clean.split()[-1] if mapping_clean.split() else ''
    frekvens_last = frekvens_clean.split()[-1] if frekvens_clean.split() else ''
    if mapping_last and frekvens_last and mapping_last == frekvens_last:
        return supplier_from_frekvens

    return None


def _write_orderlista(df, csv_path, excel_path, product_format=None):
    """Skriv orderlista till CSV + Excel utan read-only-attribut."""
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
                df, excel_path, 'Orderlista', product_format, product_col='Produktnamn',
            )
            clear_readonly(excel_path)
        except OSError as e:
            reraise_if_write_locked(excel_path, e)


def _build_name_to_code_map(sales_df):
    """{produktnamn → product_code} från sales_df, för uppslag i pris-loggen.

    Om samma produktnamn har flera koder vinner den nyaste sales-raden
    (sort_values descending på timestamp innan drop_duplicates).

    Note: load_and_prepare_sales_data renames the items file's
    ``created_at`` to ``updated``, so we look for either name to stay
    compatible with raw items dumps and the standardised pipeline data.
    Previously we only checked for ``created_at`` and silently fell
    through to "no time sort" - which meant the dedup kept an arbitrary
    code instead of the most recent one when a product had been
    re-coded over time.
    """
    if 'product_code' not in sales_df.columns or 'name' not in sales_df.columns:
        return {}
    df = sales_df[['name', 'product_code']].copy()
    timestamp_col = next(
        (c for c in ('updated', 'created_at', 'date') if c in sales_df.columns),
        None,
    )
    if timestamp_col is not None:
        df['_dt'] = pd.to_datetime(sales_df[timestamp_col], errors='coerce')
        df = df.sort_values('_dt', ascending=False)
    df = df.dropna(subset=['name', 'product_code'])
    df = df.drop_duplicates(subset=['name'], keep='first')
    return dict(zip(df['name'], df['product_code']))


def process_suppliers(sales_df, stock_df, supplier_mapping, unit_mapping,
                     bestallningsfrekvenser, latest_prices=None,
                     price_log=None, product_format=None,
                     kupa_kod_mapping=None):
    print("\n" + "=" * 80)
    print("GENERERAR ORDERLISTOR PER LEVERANTÖR")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    preflight_writable(SUPPLIER_OUTPUT_DIR)

    latest_prices = latest_prices or {}
    price_log = price_log if price_log is not None else {'products': {}}
    kupa_kod_mapping = kupa_kod_mapping or {}
    name_to_code = _build_name_to_code_map(sales_df)
    price_changes = []

    supplier_products = {}
    for product_name in sales_df['name'].unique():
        if pd.isna(product_name):
            continue
        product_name = str(product_name)
        for store_name in sales_df[sales_df['name'] == product_name]['store'].unique():
            if pd.isna(store_name):
                continue
            store_name = str(store_name)
            key = (product_name.lower(), store_name)
            if key in supplier_mapping:
                supplier = supplier_mapping[key]
                supplier_products.setdefault(supplier, [])
                if (product_name, store_name) not in supplier_products[supplier]:
                    supplier_products[supplier].append((product_name, store_name))

    print(f"Hittade produkter för {len(supplier_products)} leverantörer")

    successful = []
    failed = []

    for supplier_name, products in supplier_products.items():
        try:
            order_frequency = None
            if supplier_name in bestallningsfrekvenser:
                order_frequency = bestallningsfrekvenser[supplier_name]
            else:
                for frekvens_supplier, freq in bestallningsfrekvenser.items():
                    if match_supplier_name(supplier_name, frekvens_supplier) \
                            or match_supplier_name(frekvens_supplier, supplier_name):
                        order_frequency = freq
                        break

            if order_frequency is None:
                order_frequency = 7
                print(
                    f"\nVarning: Hittade ingen beställningsfrekvens för leverantör "
                    f"'{supplier_name}'. Använder standardvärde: {order_frequency} dagar"
                )

            print(
                f"\nProcessar leverantör: {supplier_name} "
                f"(beställningsfrekvens: {order_frequency} dagar)"
            )

            order_list = []
            for product_name, store_name in products:
                product_sales = sales_df[
                    (sales_df['name'] == product_name)
                    & (sales_df['store'] == store_name)
                ].copy()
                if len(product_sales) == 0:
                    continue

                try:
                    predicted_sales = predict_product_sales(product_sales, order_frequency)
                except Exception as e:
                    print(
                        f"  Varning: Kunde inte prognostisera för "
                        f"{product_name} i {store_name}: {e}"
                    )
                    predicted_sales = 0.0

                stock, warning_limit = get_product_stock_info(stock_df, product_name, store_name)
                unit = get_product_unit(product_name, store_name, unit_mapping)

                order_list.append({
                    'Produktnamn': product_name,
                    'Butik': store_name,
                    'Enhet': unit,
                    'Prognosticerad_försäljning': predicted_sales,
                    'Saldo': stock,
                    'stock_warning_limit': warning_limit,
                })

            if len(order_list) == 0:
                print(f"  Inga produkter att beställa för {supplier_name}")
                continue

            order_df = pd.DataFrame(order_list)

            unit_per_product = order_df.groupby('Produktnamn')['Enhet'].first().reset_index()
            unit_per_product.columns = ['Produktnamn', 'Enhet']

            aggregated = order_df.groupby('Produktnamn').agg({
                'Prognosticerad_försäljning': 'sum',
                'Saldo': 'sum',
                'stock_warning_limit': 'sum',
                'Butik': lambda x: ', '.join(x.unique()),
            }).reset_index()

            aggregated = aggregated.merge(unit_per_product, on='Produktnamn')

            stores_per_product = order_df.groupby('Produktnamn')['Butik'].nunique().reset_index()
            stores_per_product.columns = ['Produktnamn', 'Antal_butiker']
            aggregated = aggregated.merge(stores_per_product, on='Produktnamn')

            aggregated['beställningsbehov'] = aggregated.apply(
                lambda row: calculate_bestallningsbehov(
                    row['Saldo'],
                    row['Prognosticerad_försäljning'],
                    row['stock_warning_limit'],
                    order_frequency,
                ),
                axis=1,
            )

            aggregated['Orderfrekvens_dagar'] = order_frequency

            # Lägg till kolumnen senast_kända_inköpspris per produkt.
            # Logiken: läs aktuellt pris från items om finns, jämför mot
            # vad loggen säger sedan tidigare, uppdatera loggen om priset
            # ändrats. Fallback-ordning vid saknat aktuellt pris: senaste
            # värdet i loggen, annars tomt.
            from _common import _normalise_product_code as _norm_code
            prices_for_orderlista = []
            for _, row in aggregated.iterrows():
                prod_name = row['Produktnamn']
                code = name_to_code.get(prod_name)
                code_key = _norm_code(code) if code is not None else None
                current_price = latest_prices.get(code_key) if code_key else None
                price, changed = track_purchase_price(
                    price_log, code, prod_name, current_price
                )
                prices_for_orderlista.append(price)
                if changed and code_key:
                    price_changes.append({
                        'product_code': code_key,
                        'product_name': prod_name,
                        'supplier': supplier_name,
                        'new_price': price,
                    })
            aggregated['senast_kända_inköpspris'] = prices_for_orderlista

            def _produktkod(name):
                code = _norm_code(name_to_code.get(name))
                if code:
                    return code
                if product_format is not None:
                    for entry in product_format.entries:
                        if entry['norm'] == normalize_name(name).lower() and entry.get('code'):
                            return entry['code']
                return ''

            aggregated['Produktkod'] = aggregated['Produktnamn'].map(_produktkod)
            aggregated['Kupa kod'] = aggregated['Produktnamn'].map(
                lambda n: kupa_kod_mapping.get(normalize_name(n).lower(), '')
            )

            def _qty_per_box(name):
                if product_format is None:
                    return ''
                return product_format.qty_per_box_by_norm.get(
                    normalize_name(name).lower(), ''
                )

            aggregated['Mängd per låda'] = aggregated['Produktnamn'].map(_qty_per_box)

            aggregated = sort_dataframe_by_product_format(
                aggregated, product_format, 'Produktnamn',
            )

            column_order = [
                'Kupa kod',
                'Produktkod',
                'Produktnamn',
                'Enhet',
                'Mängd per låda',
                'Orderfrekvens_dagar',
                'Prognosticerad_försäljning',
                'Saldo',
                'stock_warning_limit',
                'Antal_butiker',
                'beställningsbehov',
                'senast_kända_inköpspris',
                'Butik',
            ]
            available_columns = [c for c in column_order if c in aggregated.columns]
            aggregated = aggregated[available_columns]

            for col in ('Prognosticerad_försäljning', 'beställningsbehov',
                        'Saldo', 'stock_warning_limit'):
                if col in aggregated.columns:
                    aggregated[col] = aggregated[col].round(1)
            if 'senast_kända_inköpspris' in aggregated.columns:
                aggregated['senast_kända_inköpspris'] = pd.to_numeric(
                    aggregated['senast_kända_inköpspris'], errors='coerce'
                ).round(2)

            safe_supplier_name = safe_filename(supplier_name)
            # CSV → system_data (intern systemdata). XLSX → output (för användaren).
            output_file_csv = SYSTEM_DATA_DIR / f'Orderlista_{safe_supplier_name}.csv'
            output_file_xlsx = (
                OUTPUT_DIR / f'Orderlista_{safe_supplier_name}.xlsx'
                if OPENPYXL_AVAILABLE else None
            )

            _write_orderlista(aggregated, output_file_csv, output_file_xlsx,
                              product_format=product_format)

            # Leverantörsfil: ren Excel utan interna planeringskolumner,
            # avsedd att verifieras och vidarebefordras som den är.
            if OPENPYXL_AVAILABLE:
                supplier_cols = [
                    c for c in (
                        'Kupa kod',
                        'Produktkod',
                        'Produktnamn',
                        'Enhet',
                        'Mängd per låda',
                        'beställningsbehov',
                    )
                    if c in aggregated.columns
                ]
                supplier_df = aggregated[supplier_cols].copy()
                # Visa bara rader som faktiskt ska beställas.
                if 'beställningsbehov' in supplier_df.columns:
                    supplier_df = supplier_df[
                        pd.to_numeric(
                            supplier_df['beställningsbehov'], errors='coerce'
                        ).fillna(0) > 0
                    ].copy()
                supplier_xlsx = (
                    SUPPLIER_OUTPUT_DIR / f'Orderlista_{safe_supplier_name}.xlsx'
                )
                _write_orderlista(
                    supplier_df,
                    SYSTEM_DATA_DIR / f'Orderlista_{safe_supplier_name}_leverantor.csv',
                    supplier_xlsx,
                    product_format=product_format,
                )
                print(f"  [OK] Sparade leverantörs-Excel: {supplier_xlsx}")

            print(
                f"  [OK] Sparade orderlista med {len(aggregated)} produkter "
                f"till {output_file_csv}"
            )
            if OPENPYXL_AVAILABLE and output_file_xlsx is not None:
                print(f"  [OK] Sparade Excel-fil: {output_file_xlsx}")
            print(
                f"    Totalt beställningsbehov: "
                f"{aggregated['beställningsbehov'].sum():.1f} enheter"
            )
            successful.append(supplier_name)

        except Exception as e:
            # En enskild leverantörs fel ska inte stoppa resten av rapporten.
            import traceback
            print(f"  [WARNING] Kunde inte generera orderlista för {supplier_name}: {e}")
            if not isinstance(e, PermissionError):
                traceback.print_exc()
            failed.append((supplier_name, str(e)))

    print(f"\n{'=' * 80}")
    print(f"Klara orderlistor: {len(successful)} av {len(supplier_products)}")
    if failed:
        print(f"Misslyckade leverantörer ({len(failed)}):")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print(f"{'=' * 80}")
    return price_changes


def main():
    print("=" * 80)
    print("ORDERLISTA GENERATOR - PER LEVERANTÖR MED BESTÄLLNINGSFREKVENS")
    print("=" * 80)
    print(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    BESTALLNINGSFREKVENS_PATH.parent.mkdir(parents=True, exist_ok=True)

    preflight_writable(OUTPUT_DIR)
    preflight_writable(SUPPLIER_OUTPUT_DIR)
    preflight_writable(SYSTEM_DATA_DIR)
    preflight_torp_excel_files(PROJECT_ROOT)

    print("\nHittar senaste datafiler...")
    latest_stock_file = find_latest_file('stock_report_*.csv', DATA_DOWNLOADS_DIR)
    # Items-filen från SFTP är ibland tom (bara header, ~200 byte) eller
    # saknas helt. find_item_metadata_file faller då tillbaka på senaste
    # dagliga product_sales_*.csv (samma item-schema) istället för att
    # avbryta hela körningen - leverantörsmappning/priser finns även där.
    latest_items_file = find_item_metadata_file(
        DATA_DOWNLOADS_DIR, min_size_bytes=1024
    )

    bestallningsfrekvenser = load_bestallningsfrekvens(BESTALLNINGSFREKVENS_PATH)
    supplier_mapping, unit_mapping, kupa_kod_mapping = load_supplier_mapping(
        latest_items_file
    )

    # Use full historical sales for forecasting. The earlier one-day
    # snapshot (product_sales_<today>.csv) made predict_product_sales
    # collapse to ~0 for almost every product, which in turn made
    # beställningsbehov disappear from the order lists.
    print("\nLäser försäljningshistorik (items-fil + senaste dagliga snapshots)...")
    sales_df = load_sales_history(DATA_DOWNLOADS_DIR, keep_unit=True)
    stock_df = load_stock_data(latest_stock_file)

    sales_df = filter_sales_to_stock(sales_df, stock_df)
    supplier_mapping = filter_mapping_to_stock(supplier_mapping, stock_df)
    unit_mapping = filter_mapping_to_stock(unit_mapping, stock_df)

    # Inköpspris-tracking: läs senaste priser från items, jämför mot
    # tidigare körningars logg, uppdatera och spara. Crash-safe - om
    # filen saknas eller är trasig fortsätter pipelinen med tomma värden.
    print(f"\nLäser inköpsprislogg från: {PRICE_LOG_PATH}")
    price_log = load_price_log(PRICE_LOG_PATH)
    try:
        items_df_for_prices = pd.read_csv(str(latest_items_file))
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(
            f"  [VARNING] Kunde inte läsa priser från {latest_items_file}: {e}\n"
            f"           Använder fallback från loggen om finns."
        )
        items_df_for_prices = pd.DataFrame()
    latest_prices = extract_latest_prices_from_items(items_df_for_prices)
    print(f"  Aktuella priser från items: {len(latest_prices)} produkter")
    print(f"  Tidigare loggade produkter: {len(price_log.get('products', {}))}")

    product_format = load_product_format()
    if product_format is None:
        print("  [INFO] Ingen format.xlsx hittades – orderlistor sorteras utan produktformat")

    price_changes = process_suppliers(
        sales_df, stock_df, supplier_mapping, unit_mapping,
        bestallningsfrekvenser,
        latest_prices=latest_prices, price_log=price_log,
        product_format=product_format,
        kupa_kod_mapping=kupa_kod_mapping,
    ) or []

    save_price_log(price_log, PRICE_LOG_PATH)
    if price_changes:
        print(f"\n[INFO] {len(price_changes)} prisförändring(ar) loggade denna körning:")
        for ch in price_changes[:10]:
            print(
                f"  - {ch['product_name']!r} ({ch['supplier']!r}): "
                f"nytt pris {ch['new_price']}"
            )
        if len(price_changes) > 10:
            print(f"  ... och {len(price_changes) - 10} till (se {PRICE_LOG_PATH})")

    print(f"\n{'=' * 80}")
    print("KLAR!")
    print(f"{'=' * 80}")
    print(f"Excel-orderlistor (för användare) sparade i: {OUTPUT_DIR}/")
    print(f"Leverantörs-Excel (att vidarebefordra) sparade i: {SUPPLIER_OUTPUT_DIR}/")
    print(f"CSV-orderlistor (systemdata) sparade i:     {SYSTEM_DATA_DIR}/")


if __name__ == "__main__":
    main()
