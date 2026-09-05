"""
Grafisk plocklista per shop - en produkt per sida med försäljningshistorik

Detta script:
1. Läser plocklistan från picking_list_results.csv
2. Hämtar historisk försäljningsdata för varje produkt
3. Skapar en grafisk PDF per butik med en produkt per sida

Robusthet:
- Varje butiks PDF byggs först i en lokal temp-katalog och flyttas sedan
  atomiskt till output-katalogen med retry. På så sätt påverkas inte
  pågående arbete av tillfälliga SMB- eller AV-låsningar på nätverkssharen.
- En enskild butiks fel stoppar inte resten - alla butiker som kan
  genereras genereras, och en summering visar vilka som lyckades/misslyckades.
"""

import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
# Tvinga headless Agg-backend INNAN pyplot importeras. Annars försöker
# matplotlib hitta ett GUI-backend (TkAgg/Qt5Agg) som inte är bundlat
# i .exe:n och scriptet kraschar med ImportError.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_script_dir = Path(__file__).parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from _common import (
    get_project_root,
    get_param_file_path,
    safe_filename,
    clear_readonly,
    preflight_writable,
    atomic_copy_with_retry,
    load_leveransfrekvens,
    load_sales_history,
    predict_weekly_sales,
    normalize_name,
    normalize_name_series,
    count_normalised_changes,
    safe_unlink,
    load_product_format,
    sort_dataframe_by_product_format,
)


_REQUIRED_PICKING_LIST_COLUMNS = ('store_name', 'Produktnamn')


def _load_picking_list_safely(path):
    """
    Load picking_list_results.csv defensively. Returns an empty DataFrame
    (with the columns we expect) when the file is missing, empty, malformed
    or unreadable. This avoids a hard crash of step 4 just because the
    upstream step produced no output or the file got truncated on a flaky
    network share.

    The caller decides what to do with an empty result; typically it should
    log "no picking list to process" and exit step 4 cleanly so the rest of
    the pipeline (and a future re-run) continues to work.
    """
    empty = pd.DataFrame(columns=list(_REQUIRED_PICKING_LIST_COLUMNS))
    if not path.exists():
        print(
            f"  [INFO] Plocklist-filen finns inte: {path}\n"
            f"         Steg 4 hoppas över (steg 2 producerade ingen plocklista, "
            f"eller filen har inte synkroniserats hit än)."
        )
        return empty
    try:
        df = pd.read_csv(str(path), sep=';')
    except pd.errors.EmptyDataError:
        print(f"  [INFO] Plocklist-filen är tom: {path}. Steg 4 hoppas över.")
        return empty
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
        print(
            f"  [WARNING] Kunde inte läsa plocklista från {path}: {e}\n"
            f"            Steg 4 hoppas över."
        )
        return empty

    missing = [c for c in _REQUIRED_PICKING_LIST_COLUMNS if c not in df.columns]
    if missing:
        print(
            f"  [WARNING] Plocklist-filen saknar kolumner {missing}: {path}\n"
            f"            Förmodligen ett gammalt eller manuellt redigerat format. "
            f"Steg 4 hoppas över."
        )
        return empty
    return df

PROJECT_ROOT = get_project_root()

# OUTPUT_DIR contains the PDF reports the user opens. PICKING_LIST_PATH
# points to the internal CSV that step 2 wrote to system_data/.
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'plocklistor'
SYSTEM_DATA_DIR = PROJECT_ROOT / 'system_data' / 'plocklistor'
PICKING_LIST_PATH = SYSTEM_DATA_DIR / 'picking_list_results.csv'
DATA_DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'nedladdningar'

LEVERANSFREKVENS_PATH = get_param_file_path('Leveransfrekvens.csv')
FORECAST_WEEKS = 4  # Antal veckor att visa prognos för i grafen


def _safe_float(val, default=0.0):
    """Safely convert a value to float, returning default for NaN/None/non-numeric."""
    try:
        result = float(val)
        return result if not np.isnan(result) else default
    except (TypeError, ValueError):
        return default


def build_weekly_forecast_from_predicted_sales(predicted_sales_units, delivery_frequency_days, future_weeks=4):
    """
    Skapar en veckovis prognos baserat på plocklistans prognosvärde.
    Används som fallback när den ML-baserade prognosen är tom.
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

    weekly_value = predicted_sales_units * (7.0 / delivery_frequency_days)

    today = pd.Timestamp.now().normalize()
    tomorrow = today + pd.Timedelta(days=1)
    days_until_monday = (7 - tomorrow.weekday()) % 7
    first_future_monday = tomorrow if days_until_monday == 0 else tomorrow + \
        pd.Timedelta(days=days_until_monday)

    rows = []
    for week_num in range(future_weeks):
        week_start = first_future_monday + pd.Timedelta(weeks=week_num)
        iso_week = week_start.isocalendar().week
        iso_year = week_start.isocalendar().year
        rows.append({
            'week_start': week_start,
            'quantity': weekly_value,
            'week_number': iso_week,
            'year': iso_year,
            'week_year': f"{iso_year}-W{int(iso_week):02d}",
        })
    return pd.DataFrame(rows)


def get_weekly_sales_history(product_df, store_name, product_name, weeks_back=52):
    """
    Hämtar veckovis försäljningshistorik för en produkt.
    Säkerställer att alla veckor är representerade (sätter till 0 om saknas).
    """
    product_sales = product_df[
        (product_df['store'] == store_name)
        & (product_df['name'] == product_name)
    ].copy()

    if len(product_sales) == 0:
        return pd.DataFrame(columns=['week_start', 'quantity', 'week_number', 'year', 'week_year'])

    product_sales['date'] = pd.to_datetime(product_sales['date'])
    product_sales['week_start'] = product_sales['date'] - pd.to_timedelta(
        product_sales['date'].dt.dayofweek, unit='D'
    )
    product_sales['week_number'] = product_sales['week_start'].dt.isocalendar().week
    product_sales['year'] = product_sales['week_start'].dt.isocalendar().year
    product_sales['week_year'] = (
        product_sales['year'].astype(str) + '-W'
        + product_sales['week_number'].astype(str).str.zfill(2)
    )

    weekly_sales = product_sales.groupby(
        ['week_start', 'week_number', 'year', 'week_year']
    )['quantity'].sum().reset_index()
    weekly_sales = weekly_sales.sort_values('week_start')

    if len(weekly_sales) == 0:
        return weekly_sales

    today = pd.Timestamp.now().normalize()
    current_week_monday = today - pd.Timedelta(days=today.dayofweek)
    latest_week = max(weekly_sales['week_start'].max(), current_week_monday)
    cutoff_date = latest_week - pd.Timedelta(weeks=weeks_back)
    weekly_sales = weekly_sales[weekly_sales['week_start'] >= cutoff_date]

    all_weeks = pd.date_range(start=cutoff_date, end=latest_week, freq='W-MON')
    all_weeks_series = pd.Series(all_weeks)
    week_numbers = all_weeks_series.dt.isocalendar().week
    years = all_weeks_series.dt.isocalendar().year
    week_year_str = [f"{y}-W{w:02d}" for y, w in zip(years.values, week_numbers.values)]
    all_weeks_df = pd.DataFrame({
        'week_start': all_weeks,
        'week_number': week_numbers.values,
        'year': years.values,
        'week_year': week_year_str,
    })

    weekly_sales = all_weeks_df.merge(
        weekly_sales[['week_start', 'quantity']],
        on='week_start',
        how='left',
    )
    weekly_sales['quantity'] = weekly_sales['quantity'].fillna(0)
    weekly_sales = weekly_sales.sort_values('week_start')
    return weekly_sales


def create_product_page(fig, sales_history, product_info, predicted_sales, weekly_forecast, unit='st'):
    """
    Skapar en sida för en produkt med graf och information.
    weekly_forecast: DataFrame med veckovis prognos för 4 veckor framåt
    """
    fig.clear()

    try:
        predicted_sales = float(predicted_sales) if pd.notna(predicted_sales) else 0.0
    except (TypeError, ValueError):
        predicted_sales = 0.0

    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 2.5, 1], hspace=0.35, wspace=0.3)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_main = fig.add_subplot(gs[1, :])
    ax_info1 = fig.add_subplot(gs[2, 0])
    ax_info1.axis('off')
    ax_info2 = fig.add_subplot(gs[2, 1])
    ax_info2.axis('off')

    product_name = product_info['product_name']
    product_code = product_info.get('product_code', 'N/A')
    if pd.isna(product_code) or product_code == '':
        product_code = 'N/A'
    store_name = product_info['store_name']

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
        next_week_label = f"{next_week_monday.isocalendar().year}-W{next_week_monday.isocalendar().week:02d}"

    title_text = f"{product_name}\n{store_name}\nPlocklista för vecka: {next_week_label}"
    ax_title.text(0.5, 0.5, title_text,
                  ha='center', va='center',
                  fontsize=14, fontweight='bold', wrap=True)

    if len(sales_history) > 0:
        sales_history = sales_history.sort_values('week_start')
        x_positions = range(len(sales_history))
        x_labels = sales_history['week_year'].values

        ax_main.fill_between(x_positions, 0, sales_history['quantity'],
                             alpha=0.3, color='blue', label='Historisk försäljning')
        ax_main.plot(x_positions, sales_history['quantity'],
                     'b-', linewidth=2.5, marker='o', markersize=5, zorder=3)

        if len(weekly_forecast) > 0:
            weekly_forecast = weekly_forecast.sort_values('week_start')
            forecast_x_start = len(x_positions)
            forecast_x_positions = range(
                forecast_x_start, forecast_x_start + len(weekly_forecast))
            forecast_x_labels = weekly_forecast['week_year'].values

            ax_main.plot(forecast_x_positions, weekly_forecast['quantity'],
                         'r-', linewidth=2.5, marker='o', markersize=8,
                         label=f'Prognos ({FORECAST_WEEKS} veckor)',
                         zorder=5, markeredgecolor='darkred', markeredgewidth=2)

            for x_pos, row in zip(forecast_x_positions, weekly_forecast.itertuples()):
                ax_main.annotate(
                    f'{row.quantity:.1f}',
                    xy=(x_pos, row.quantity),
                    xytext=(0, 15), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='red',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                )

            ax_main.axvline(x=forecast_x_start - 0.5, color='r', linestyle='--',
                            alpha=0.5, linewidth=2, zorder=1)

            all_x_positions = list(x_positions) + list(forecast_x_positions)
            all_x_labels = list(x_labels) + list(forecast_x_labels)
        else:
            all_x_positions = list(x_positions)
            all_x_labels = list(x_labels)

        step = max(1, len(all_x_positions) // 20)
        visible_ticks = all_x_positions[::step]
        visible_labels = all_x_labels[::step]
        ax_main.set_xticks(visible_ticks)
        ax_main.set_xticklabels(visible_labels, rotation=45, ha='right')

        ax_main.set_xlabel('Vecka', fontsize=12, fontweight='bold')
        ax_main.set_ylabel(f'Försäljning ({unit})', fontsize=12, fontweight='bold')
        ax_main.set_title(
            f'Försäljningshistorik (senaste 52 veckor) och prognos '
            f'({FORECAST_WEEKS} veckor framåt)',
            fontsize=13, fontweight='bold', pad=15,
        )
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.legend(loc='best', fontsize=10, framealpha=0.9)

        y_max_history = sales_history['quantity'].max() if len(sales_history) > 0 else 0
        y_max_forecast = weekly_forecast['quantity'].max() if len(weekly_forecast) > 0 else 0
        y_max = max(y_max_history, y_max_forecast,
                    predicted_sales if predicted_sales > 0 else 0)
        ax_main.set_ylim(bottom=0, top=max(y_max * 1.1, 1))
    else:
        ax_main.text(0.5, 0.5, 'Ingen försäljningshistorik tillgänglig',
                     ha='center', va='center', fontsize=14, style='italic')
        ax_main.set_title('Försäljningshistorik', fontsize=13, fontweight='bold')
        ax_main.axis('off')

    current_stock = _safe_float(product_info.get('current_stock', 0))
    stock_warning_limit = _safe_float(product_info.get('stock_warning_limit', 0))
    expected_stock_after = _safe_float(product_info.get('expected_stock_after_sales', 0))
    fill_up = _safe_float(product_info.get('fill_up_quantity', 0))

    info_text1 = (
        f"LAGERSTATUS\n\n"
        f"Nuvarande lager: {current_stock:.1f} {unit}\n"
        f"Varningsgräns: {stock_warning_limit:.1f} {unit}\n"
        f"Förväntat lager efter försäljning: {expected_stock_after:.1f} {unit}\n"
        f"Prognostiserad försäljning (vecka {next_week_label}): {predicted_sales:.1f} {unit}"
    )
    ax_info1.text(0.05, 0.95, info_text1,
                  ha='left', va='top',
                  fontsize=10, family='monospace',
                  transform=ax_info1.transAxes,
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=10))

    needs_refill = product_info['needs_refill']
    status_color = 'lightcoral' if needs_refill else 'lightgreen'
    status_text = 'BEHÖVER PÅFYLLNING' if needs_refill else 'LAGER OK'

    info_text2 = (
        f"PÅFYLLNING\n\n"
        f"Behöver fylla på: {fill_up:.1f} {unit}\n"
        f"Status: {status_text}\n"
        f"Produktkod: {product_code}"
    )
    ax_info2.text(0.05, 0.95, info_text2,
                  ha='left', va='top',
                  fontsize=10, family='monospace',
                  transform=ax_info2.transAxes,
                  bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.7, pad=10))


def _build_pdf_for_store(store_name, store_products, sales_df, tmp_pdf_path):
    """
    Bygger PDF:en för en butik i en lokal temp-fil. Returnerar antal produkter
    som lyckades renderas. Anropas innanför per-butik try/except.
    """
    pages_written = 0
    with PdfPages(str(tmp_pdf_path)) as pdf:
        for idx, (_, product_row) in enumerate(store_products.iterrows(), 1):
            product_name_col = 'Produktnamn' if 'Produktnamn' in product_row.index else 'product_name'
            product_name_display = (
                str(product_row[product_name_col])[:50]
                if product_name_col in product_row.index else 'Okänt produkt'
            )
            print(
                f"    Sidan {idx}/{len(store_products)}: {product_name_display}..."
            )

            product_name_in_picking = (
                product_row[product_name_col]
                if product_name_col in product_row.index
                else product_row.get('product_name', '')
            )

            store_sales = sales_df[sales_df['store'] == store_name].copy()
            product_names_in_sales = store_sales['name'].unique()

            product_name_normalized = normalize_name(product_name_in_picking).lower()
            matched_product_name = None

            for sales_name in product_names_in_sales:
                if product_name_normalized == normalize_name(sales_name).lower():
                    matched_product_name = sales_name
                    break

            if matched_product_name is None:
                for sales_name in product_names_in_sales:
                    sales_normalized = normalize_name(sales_name).lower()
                    if (product_name_normalized in sales_normalized
                            or sales_normalized in product_name_normalized):
                        matched_product_name = sales_name
                        break

            unit = 'st'
            weekly_forecast = pd.DataFrame(
                columns=['week_start', 'quantity', 'week_number', 'year', 'week_year']
            )

            if matched_product_name:
                matched_sales = store_sales[store_sales['name'] == matched_product_name]
                if len(matched_sales) > 0 and 'unit' in matched_sales.columns:
                    unit_value = matched_sales['unit'].iloc[0]
                    if pd.notna(unit_value) and unit_value != '':
                        unit = normalize_name(unit_value) or 'st'

                sales_history = get_weekly_sales_history(
                    sales_df, store_name, matched_product_name, weeks_back=52
                )

                product_sales_df = matched_sales.copy()
                if len(product_sales_df) > 0:
                    if 'date' in product_sales_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(product_sales_df['date']):
                            product_sales_df['date'] = pd.to_datetime(product_sales_df['date'])
                    elif 'updated' in product_sales_df.columns:
                        product_sales_df['date'] = pd.to_datetime(product_sales_df['updated'])

                try:
                    weekly_forecast = predict_weekly_sales(
                        product_sales_df, future_weeks=FORECAST_WEEKS
                    )
                except Exception as e:
                    print(f"      Varning: Kunde inte skapa prognos: {e}")
                    weekly_forecast = pd.DataFrame(
                        columns=['week_start', 'quantity', 'week_number', 'year', 'week_year']
                    )
            else:
                sales_history = pd.DataFrame(
                    columns=['week_start', 'quantity', 'week_number', 'year', 'week_year']
                )

            fig = plt.figure(figsize=(11.69, 8.27))  # A4-landskap

            product_info = product_row.to_dict()
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

            if 'expected_stock_after_sales' not in product_info:
                current_stock = product_info.get('current_stock', 0)
                predicted_sales = product_info.get('predicted_sales_units', 0)
                product_info['expected_stock_after_sales'] = max(
                    0, current_stock - predicted_sales)

            predicted_sales_value = product_info.get('predicted_sales_units', 0)
            delivery_frequency_days = product_info.get('delivery_frequency_days', 7)

            if len(weekly_forecast) == 0:
                weekly_forecast = build_weekly_forecast_from_predicted_sales(
                    predicted_sales_value, delivery_frequency_days, future_weeks=FORECAST_WEEKS
                )

            create_product_page(
                fig, sales_history, product_info,
                predicted_sales_value, weekly_forecast, unit=unit,
            )

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            pages_written += 1
    return pages_written


def generate_graphical_picking_list(picking_list_df, sales_df, product_format=None):
    """
    Genererar grafiska PDF:er för varje butik.

    Varje butik renderas oberoende av de andra: om en butik misslyckas
    (t.ex. permission denied på utgångsfilen) loggas felet och loopen
    fortsätter med nästa butik. En summering visas i slutet.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preflight_writable(OUTPUT_DIR)

    stores = list(picking_list_df['store_name'].unique())
    print(f"\nButiker att processa ({len(stores)}): {stores}")

    successful = []
    failed = []

    for store_name in stores:
        try:
            print(f"\nGenererar grafisk plocklista för: {store_name}")

            store_products = picking_list_df[
                picking_list_df['store_name'] == store_name
            ].copy()

            store_products = sort_dataframe_by_product_format(
                store_products, product_format, 'Produktnamn',
            )

            safe_store_name = safe_filename(store_name)
            final_pdf = OUTPUT_DIR / f'Plocklista_{safe_store_name}.pdf'

            print(f"  Skapar PDF: {final_pdf}")
            print(f"  Antal produkter: {len(store_products)}")

            # Bygg först i en lokal temp-katalog så vi inte slåss med
            # nätverkssharens AV/SMB-låsningar under själva renderingen.
            with tempfile.TemporaryDirectory(prefix='plocklista_') as tmpdir:
                tmp_pdf = Path(tmpdir) / f'Plocklista_{safe_store_name}.pdf'
                pages = _build_pdf_for_store(store_name, store_products, sales_df, tmp_pdf)

                # Flytta till slutdestinationen med retry vid transienta
                # PermissionError, och rensa ev. read-only-attribut så
                # vem som helst kan redigera/radera filen senare.
                atomic_copy_with_retry(tmp_pdf, final_pdf)

            print(f"  [OK] Klar! PDF sparad: {final_pdf} ({pages} sidor)")
            successful.append(store_name)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"  [WARNING] Kunde inte generera PDF för {store_name}: {err_msg}")
            if not isinstance(e, PermissionError):
                traceback.print_exc()
            failed.append((store_name, err_msg))
            continue

    print(f"\n{'=' * 80}")
    print("GRAFISKA PLOCKLISTOR - SAMMANFATTNING")
    print(f"{'=' * 80}")
    print(f"Lyckades: {len(successful)} av {len(stores)}")
    if successful:
        for name in successful:
            print(f"  [OK]      {name}")
    if failed:
        print(f"\nMisslyckade ({len(failed)}):")
        for name, err in failed:
            print(f"  [FEL]     {name}: {err}")
    print(f"{'=' * 80}")


def main():
    """Huvudfunktion"""
    print("=" * 80)
    print("GRAFISK PLOCKLISTA GENERATOR")
    print("=" * 80)
    print(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LEVERANSFREKVENS_PATH.parent.mkdir(parents=True, exist_ok=True)

    preflight_writable(OUTPUT_DIR)

    # Leveransfrekvenser läses för loggning / framtida bruk; själva
    # plocklistan har redan Leveransfrekvens_dagar per rad från skript 2.
    print(f"\nLaddar leveransfrekvenser från: {LEVERANSFREKVENS_PATH}")
    load_leveransfrekvens(LEVERANSFREKVENS_PATH)

    print(f"\nLäser plocklista från {PICKING_LIST_PATH}...")
    picking_list_df = _load_picking_list_safely(PICKING_LIST_PATH)

    if len(picking_list_df) == 0:
        print(
            "\n[INFO] Ingen plocklista att rita upp - hoppar över steg 4 "
            "utan fel."
        )
        print("\n" + "=" * 80)
        print("KLAR (inget att göra)!")
        print("=" * 80)
        return

    before = len(picking_list_df)
    picking_list_df = picking_list_df.dropna(subset=['store_name', 'Produktnamn'])
    dropped = before - len(picking_list_df)
    if dropped > 0:
        print(f"  Removed {dropped} rows with missing store_name or Produktnamn")

    # Defence in depth: even though script 2 writes normalised names, normalise
    # again on read so a manually edited picking_list_results.csv (with stray
    # whitespace from Excel) doesn't silently fragment a store into duplicates.
    raw_store = picking_list_df['store_name'].astype('string')
    raw_product = picking_list_df['Produktnamn'].astype('string')
    picking_list_df['store_name'] = normalize_name_series(picking_list_df['store_name'])
    picking_list_df['Produktnamn'] = normalize_name_series(picking_list_df['Produktnamn'])
    changed_store = count_normalised_changes(raw_store, picking_list_df['store_name'])
    changed_product = count_normalised_changes(raw_product, picking_list_df['Produktnamn'])
    if changed_store or changed_product:
        print(
            f"  Normaliserade whitespace i plocklistan: "
            f"{changed_store} butiksnamn och {changed_product} produktnamn justerade"
        )

    for col in ('saldo_denna_butik', 'Varningsgräns',
                'Prognosticerad_försäljning', 'Påfyllningsbehov', 'Leveransfrekvens_dagar'):
        if col in picking_list_df.columns:
            picking_list_df[col] = pd.to_numeric(picking_list_df[col], errors='coerce').fillna(0)

    print(f"  Laddat {len(picking_list_df)} produkter")
    print(f"  Butiker: {picking_list_df['store_name'].nunique()}")

    # Historiken som ritas i PDF:erna kräver hela tidsserien, inte bara
    # senaste dagens snapshot. Tidigare användes product_sales_*.csv,
    # vilket bara innehåller en dags transaktioner (~500 rader) och då
    # blev grafens "senaste 52 veckor" i praktiken bara den innevarande
    # veckan med nollor överallt annars. Vi tar nu hela cumulative
    # items-filen som primär källa och kompletterar med eventuella
    # dagliga snapshots för datum efter items-filens slutdatum så att
    # även de senaste dagarna kommer med (items-exporten på SFTP:n
    # ligger ofta 1-3 dygn efter dagliga snapshots).
    print("\nLäser försäljningshistorik (items-fil + senaste dagliga snapshots)...")
    sales_df = load_sales_history(DATA_DOWNLOADS_DIR, keep_unit=True)

    product_format = load_product_format()
    generate_graphical_picking_list(picking_list_df, sales_df, product_format=product_format)

    print(f"\n{'=' * 80}")
    print("KLAR!")
    print(f"{'=' * 80}")
    print(f"Grafiska plocklistor sparade i mappen: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
