"""
Shared utilities for the Torp report generation pipeline.

This module centralises code that was previously duplicated across
2generate_picking_list.py, 3generate_order_list.py and
4generate_graphical_picking_list.py:

  - Path resolution (PROJECT_ROOT, parameter file lookup with _MEIPASS fallback)
  - Filename sanitisation (Windows-safe, handles all forbidden characters)
  - Write-permission helpers (clear read-only attribute, preflight write check,
    atomic temp+move with retry for fragile network shares)
  - Data loading helpers (find_latest_file, load_leveransfrekvens,
    load_and_prepare_sales_data)
  - Forecasting (predict_product_sales for total-period forecasts, plus the
    weekly-forecast helpers used by the graphical picking list)
"""

import os
import re
import sys
import stat
import glob
import time
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    _ADVANCED_ML = True
except ImportError:
    _ADVANCED_ML = False

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_project_root():
    """
    Returns the project root directory.

    - When frozen (PyInstaller .exe): the directory that contains the .exe.
      This is the user-facing folder where data/, output/ and parameter files
      live next to the executable.
    - When running as a regular Python script: the parent of scripts/ (or
      the script's own directory if it isn't inside scripts/).
    """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent.resolve()
    here = Path(__file__).parent.resolve()
    if here.name == 'scripts':
        return here.parent
    return here


def get_param_file_path(filename):
    """
    Locate a parameter file (e.g. Leveransfrekvens.csv).

    Search order:
      1. <project_root>/data/parametrar/<filename>  (user-editable, deployable)
      2. <_MEIPASS>/data/parametrar/<filename>     (bundled inside .exe, fallback)

    Returns the first existing path. If neither exists, returns option 1 so
    the caller can report a clear "file missing" error.
    """
    rel = Path('data') / 'parametrar' / filename
    primary = get_project_root() / rel
    if primary.exists():
        return primary
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            bundled = Path(meipass) / rel
            if bundled.exists():
                return bundled
    return primary


# ---------------------------------------------------------------------------
# Filename safety
# ---------------------------------------------------------------------------

# Windows forbids these characters in filenames, plus control characters.
_FORBIDDEN_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name):
    """
    Sanitise a string so it can be used as a Windows filename component.
    Replaces forbidden characters with underscore, normalises whitespace
    (NBSP, tabs, zero-width chars → regular space, collapsed) and strips
    trailing whitespace/dots, which Windows silently truncates.
    """
    if name is None:
        return '_'
    cleaned = _FORBIDDEN_FILENAME_CHARS.sub('_', str(name))
    cleaned = normalize_name(cleaned)
    cleaned = cleaned.rstrip('.')
    return cleaned or '_'


# ---------------------------------------------------------------------------
# Whitespace normalisation for names (store names, product names, etc.)
# ---------------------------------------------------------------------------
#
# A surprising number of bugs come from invisible whitespace differences in
# names that arrive via CSV exports from Excel, Word documents or copy-paste
# from emails. Examples we have to handle:
#
#   - Trailing/leading ASCII space:  "Sandsund kiosk " vs "Sandsund kiosk"
#   - Non-breaking space (U+00A0):   "Sandsund\u00a0kiosk" - looks identical
#     but != "Sandsund kiosk". Python's .strip() handles outer NBSP but
#     internal NBSP between words must be replaced.
#   - Multiple internal spaces:      "Sandsund  kiosk" vs "Sandsund kiosk"
#   - Tab characters:                "Sandsund\tkiosk"
#   - Zero-width characters:         U+200B (ZWSP), U+200C, U+200D, U+FEFF (BOM).
#     These are not in str.isspace() so .strip() does NOT remove them. They
#     appear invisibly when text is pasted from web pages or emails.
#
# normalize_name does all of this in one pass: remove zero-width characters,
# replace every Unicode whitespace run with a single ASCII space, and strip.

# Zero-width characters that survive .strip() and break exact matching.
_INVISIBLE_CHARS = re.compile(r'[\u200b-\u200d\ufeff]')
# Any run of whitespace (ASCII, NBSP, ideographic, etc.) collapses to one space.
_WHITESPACE_RUN = re.compile(r'\s+')


def normalize_name(value):
    """
    Normalise a single name/text value for reliable string matching.

    Returns the input unchanged for NaN/None. For other values: removes
    invisible characters, collapses any whitespace run to a single ASCII
    space, and strips outer whitespace.
    """
    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    s = str(value)
    s = _INVISIBLE_CHARS.sub('', s)
    s = _WHITESPACE_RUN.sub(' ', s).strip()
    return s


def normalize_name_series(series):
    """
    Vectorised normalize_name for a pandas Series. NaN is preserved.

    Uses regex .str operations which keep NaN automatically, much faster
    than .apply(normalize_name) on large frames.
    """
    s = series.astype('string')
    s = s.str.replace(_INVISIBLE_CHARS, '', regex=True)
    s = s.str.replace(_WHITESPACE_RUN, ' ', regex=True).str.strip()
    return s.astype(object)


def count_normalised_changes(original_series, normalised_series):
    """
    Count how many cells differ between the original and the normalised
    series. Used for diagnostic logging at data-ingestion points.
    """
    a = original_series.astype('string').fillna('')
    b = normalised_series.astype('string').fillna('')
    return int((a != b).sum())


# ---------------------------------------------------------------------------
# Write-permission helpers
# ---------------------------------------------------------------------------

def clear_readonly(path):
    """
    Ensure the file at `path` is writable by removing the read-only attribute.
    Silent no-op if the file does not exist.

    On Windows the FAT-style read-only attribute is the common reason a
    "Permission denied" error shows up when re-writing existing output files.
    NTFS ACLs are inherited from the parent directory and are not changed here.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        os.chmod(str(p), stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass
    if os.name == 'nt':
        try:
            import subprocess
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            subprocess.run(
                ['attrib', '-R', str(p)],
                check=False,
                capture_output=True,
                creationflags=flags,
            )
        except Exception:
            pass


def safe_unlink(path):
    """
    Delete a file if it exists. Returns True if a file was actually removed,
    False if the path didn't exist or wasn't a regular file.

    Never raises FileNotFoundError - "already gone" is considered success
    (that's the desired end state anyway). Other OSErrors (e.g. permission
    denied because someone has the file open) are caught and logged as a
    warning, returning False, so the calling pipeline keeps running.
    """
    p = Path(path)
    if not p.exists():
        # Already gone - that's success in the "be content" sense.
        return False
    try:
        # missing_ok=True so a race condition between the exists() check
        # above and the unlink() call below still doesn't crash.
        p.unlink(missing_ok=True)
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        # File is open in another process, ACL denies delete, etc.
        # Don't crash - the caller almost always just wants the file gone
        # and can move on if we can't manage it.
        print(f"  [WARNING] Kunde inte radera {p}: {e}")
        return False


def preflight_writable(directory):
    """
    Verify that `directory` exists and we can write into it. Raises
    PermissionError with a clear message if not. Used to fail fast at the
    start of a step instead of blowing up after minutes of work.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    probe = d / f'.write_test_{os.getpid()}_{int(time.time() * 1000)}'
    try:
        probe.write_text('ok', encoding='utf-8')
    except Exception as e:
        raise PermissionError(
            f"Output-katalogen är inte skrivbar: {d}\n"
            f"Ursprungligt fel: {e}\n"
            f"Kontrollera NTFS-rättigheter och att ingen annan process håller filer öppna."
        ) from e
    safe_unlink(probe)


def atomic_copy_with_retry(src, dst, attempts=3, delay=2.0):
    r"""
    Move `src` to `dst`, retrying on transient PermissionError.

    Intended for moving a file produced locally to a network share
    (\\server\share\...). SMB shares, antivirus scanning and PDF readers
    sometimes hold a brief lock on the destination; retrying smooths over
    those transients without losing the produced file.

    Side effects:
      - Creates dst.parent if it doesn't exist.
      - Clears the read-only attribute on dst (before and after copy).
      - Removes src on success.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    last_err = None
    for i in range(attempts):
        try:
            if dst.exists():
                clear_readonly(dst)
            shutil.copy2(str(src), str(dst))
            safe_unlink(src)
            clear_readonly(dst)
            return
        except OSError as e:
            # PermissionError ärver från OSError - täcker både låsta filer
            # och transienta SMB/NFS-fel.
            last_err = e
            if i < attempts - 1:
                time.sleep(delay)

    raise last_err


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_latest_file(pattern, directory):
    """
    Find the most recent file in `directory` matching `pattern` (e.g.
    'product_sales_*.csv'), using the latest YYYY-MM-DD found in the
    filename as the sort key. Raises FileNotFoundError if nothing matches.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Katalogen finns inte: {directory}")

    files = glob.glob(str(directory / pattern))
    if not files:
        existing = list(directory.glob('*.csv'))
        if existing:
            print(
                f"  Hittade {len(existing)} CSV-filer i {directory}, "
                f"men inga matchar mönstret {pattern}"
            )
            print(f"  Exempel filer: {[f.name for f in existing[:5]]}")
        else:
            print(f"  Katalogen {directory} är tom eller innehåller inga CSV-filer")
        raise FileNotFoundError(f"Inga filer matchar mönstret {pattern} i {directory}")

    file_dates = []
    for file in files:
        all_dates = re.findall(r'(\d{4}-\d{2}-\d{2})', os.path.basename(file))
        if all_dates:
            file_dates.append((max(all_dates), file))

    if not file_dates:
        raise ValueError(f"Kunde inte hitta datum i filnamn för {pattern}")

    file_dates.sort(key=lambda x: x[0], reverse=True)
    latest_file = file_dates[0][1]
    print(
        f"  Hittade senaste fil: {os.path.basename(latest_file)} "
        f"(datum: {file_dates[0][0]})"
    )
    return latest_file


# ---------------------------------------------------------------------------
# Leveransfrekvens (delivery frequency per store)
# ---------------------------------------------------------------------------

def load_leveransfrekvens(file_path):
    """
    Read Leveransfrekvens.csv → dict {store_name: delivery_frequency_days}.
    Returns an empty dict and logs a warning if the file is missing or
    malformed; callers fall back to a default frequency in that case.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Varning: Leveransfrekvens-fil saknas: {file_path}")
        print("  Använder standardvärde: 7 dagar för alla butiker")
        return {}

    try:
        df = pd.read_csv(str(file_path), sep=';', encoding='utf-8-sig')
        print(f"  Hittade kolumner: {list(df.columns)}")

        store_col = None
        freq_col = None
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'store' in col_lower or 'butik' in col_lower:
                store_col = col
            if 'leveransfrekvens' in col_lower or 'frekvens' in col_lower:
                freq_col = col

        if store_col is None or freq_col is None:
            if len(df.columns) >= 2:
                store_col, freq_col = df.columns[0], df.columns[1]
                print(
                    f"  Varning: Kunde inte hitta kolumnnamn, "
                    f"använder: '{store_col}' och '{freq_col}'"
                )
            else:
                print("  Fel: CSV-filen har för få kolumner. Förväntar minst 2 kolumner.")
                return {}

        parametrar = {}
        for _, row in df.iterrows():
            butik_raw = row[store_col]
            if not pd.notna(butik_raw):
                continue
            # normalize_name (not just .strip()) so a stray NBSP or double
            # space in the CSV resolves to the same key as the sales data.
            butik = normalize_name(butik_raw)
            if not butik:
                continue
            try:
                freq_raw = row[freq_col]
                if pd.notna(freq_raw):
                    freq_str = str(freq_raw).strip()
                    freq_clean = ''.join(c for c in freq_str if c.isdigit() or c == '.')
                    if freq_clean:
                        parametrar[butik] = int(float(freq_clean))
            except (ValueError, TypeError):
                print(f"  Varning: Kunde inte tolka frekvens för '{butik}': {freq_raw}")

        print(f"Laddade leveransfrekvenser för {len(parametrar)} butiker")
        if parametrar:
            print(f"  Exempel: {list(parametrar.items())[:3]}")
        return parametrar
    except Exception as e:
        print(f"Varning: Kunde inte ladda leveransfrekvenser: {e}")
        print("  Använder standardvärde: 7 dagar för alla butiker")
        import traceback
        traceback.print_exc()
        return {}


# ---------------------------------------------------------------------------
# Sales data loading
# ---------------------------------------------------------------------------

def load_and_prepare_sales_data(file_path, keep_unit=False):
    """
    Read and normalise product sales data from product_sales_*.csv or the
    transactional product_sales_items_*.csv layout. Returns a DataFrame with
    columns: date, store, name, quantity (+ unit if keep_unit=True).
    """
    file_path = Path(file_path)
    print(f"Läser försäljningsdata från {file_path}...")
    df = pd.read_csv(str(file_path))

    if 'period' in df.columns:
        df = df.rename(columns={
            'period': 'date',
            'store_name': 'store',
            'product_name': 'name',
            'total_quantity_sold': 'quantity',
            'total_sales': 'line_price',
        })
        df['date'] = pd.to_datetime(df['date']).dt.date
        if keep_unit and 'unit' not in df.columns:
            df['unit'] = 'st'
    elif 'created_at' in df.columns:
        df = df.rename(columns={
            'created_at': 'updated',
            'store_name': 'store',
            'product_name': 'name',
            'line_total': 'line_price',
        })
        df['updated'] = pd.to_datetime(df['updated'])
        df['date'] = df['updated'].dt.date
        if keep_unit and 'unit' not in df.columns:
            df['unit'] = 'st'
    else:
        date_cols = [c for c in df.columns
                     if 'date' in c.lower() or 'created' in c.lower() or 'period' in c.lower()]
        if date_cols:
            df['date'] = pd.to_datetime(df[date_cols[0]]).dt.date
        else:
            raise ValueError(
                f"Kunde inte hitta datumkolumn i filen. "
                f"Tillgängliga kolumner: {list(df.columns)}"
            )
        if 'store_name' in df.columns:
            df['store'] = df['store_name']
        if 'product_name' in df.columns:
            df['name'] = df['product_name']
        if 'quantity' not in df.columns:
            qty_cols = [c for c in df.columns if 'quantity' in c.lower()]
            if qty_cols:
                df['quantity'] = df[qty_cols[0]]
            else:
                raise ValueError(
                    f"Kunde inte hitta quantity-kolumn i filen. "
                    f"Tillgängliga kolumner: {list(df.columns)}"
                )

    if keep_unit and 'unit' not in df.columns:
        df['unit'] = 'st'

    required = ['date', 'store', 'name', 'quantity']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Saknade kolumner: {missing}. Tillgängliga kolumner: {list(df.columns)}"
        )

    if 'order_status' in df.columns:
        before = len(df)
        df = df[df['order_status'] == 'complete'].copy()
        if before != len(df):
            print(
                f"  Filtrerade bort {before - len(df)} rader med "
                f"order_status != 'complete'"
            )

    if 'quantity' in df.columns:
        nan_count = df['quantity'].isna().sum()
        if nan_count:
            print(f"  Fyller {nan_count} NaN-värden i quantity med 0")
            df['quantity'] = df['quantity'].fillna(0)

    if keep_unit and 'unit' in df.columns:
        nan_count = df['unit'].isna().sum()
        if nan_count:
            df['unit'] = df['unit'].fillna('st')

    before = len(df)
    df = df.dropna(subset=['name', 'store'])
    if before != len(df):
        print(f"  Removed {before - len(df)} rows with missing product name or store")

    # Normalise whitespace in name/store columns so that small differences
    # in CSV exports (trailing space, NBSP, double space, BOM) don't cause
    # "Sandsund kiosk" and "Sandsund kiosk " to be treated as separate stores.
    raw_store = df['store'].astype('string')
    raw_name = df['name'].astype('string')
    df['store'] = normalize_name_series(df['store'])
    df['name'] = normalize_name_series(df['name'])

    store_changed = count_normalised_changes(raw_store, df['store'])
    name_changed = count_normalised_changes(raw_name, df['name'])
    if store_changed or name_changed:
        print(
            f"  Normaliserade whitespace i butiks-/produktnamn: "
            f"{store_changed} butiksnamn och {name_changed} produktnamn justerade"
        )

    if keep_unit and 'unit' in df.columns:
        df['unit'] = normalize_name_series(df['unit'])

    print(f"  Laddat {len(df)} rader")
    print(f"  Butiker: {df['store'].nunique()}")
    print(f"  Produkter: {df['name'].nunique()}")
    print(f"  Datumintervall: {df['date'].min()} till {df['date'].max()}")

    return df


# ---------------------------------------------------------------------------
# Forecasting — total over a forecast window (used by scripts 2 and 3)
# ---------------------------------------------------------------------------

def _compute_mae(y_true, y_pred):
    """Mean absolute error that works even without sklearn.metrics."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def _predict_weekly_average(daily_sales, forecast_days, ref_date):
    """
    Forecast using the mean of the 4 latest *complete* calendar weeks.
    The current (possibly incomplete) week is excluded so the average is
    not pulled down by a partial week.
    """
    ds = daily_sales.copy()
    ds['week'] = ds['date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sales = ds.groupby('week')['quantity'].sum().reset_index()
    weekly_sales = weekly_sales.sort_values('week')

    current_week_start = ref_date - pd.Timedelta(days=ref_date.dayofweek)
    complete_weeks = weekly_sales[weekly_sales['week'] < current_week_start]

    if len(complete_weeks) == 0:
        complete_weeks = weekly_sales

    weeks_to_use = min(4, len(complete_weeks))
    if weeks_to_use > 0:
        avg_weekly = complete_weeks.tail(weeks_to_use)['quantity'].mean()
        return max(0.0, float(avg_weekly / 7.0 * forecast_days))

    return 0.0


def _predict_best_model(daily_sales, forecast_days, today):
    """
    Train several candidate models, evaluate each on a 28-day validation
    window, and use the winner to produce the final forecast. Falls back to
    a 4-week-average baseline if no ML model beats it.
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
        return _predict_weekly_average(daily_sales, forecast_days, today)

    X_train, X_val = X.iloc[:-val_days], X.iloc[-val_days:]
    y_train, y_val = y.iloc[:-val_days], y.iloc[-val_days:]

    train_end = ds['date'].iloc[-val_days - 1]
    baseline_total = _predict_weekly_average(
        ds[['date', 'quantity']].iloc[:-val_days].copy(), val_days, train_end
    )
    baseline_daily = baseline_total / val_days if val_days > 0 else 0.0
    baseline_mae = _compute_mae(y_val, [baseline_daily] * len(y_val))

    candidates = {
        'knn_3': KNeighborsRegressor(n_neighbors=3),
        'knn_5': KNeighborsRegressor(n_neighbors=5),
        'knn_7': KNeighborsRegressor(n_neighbors=7),
    }
    if _ADVANCED_ML:
        candidates['linear'] = LinearRegression()
        candidates['ridge'] = Ridge(alpha=1.0)
        candidates['rf'] = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42)
        candidates['gb'] = GradientBoostingRegressor(
            n_estimators=50, max_depth=5, random_state=42)

    best_mae = baseline_mae
    best_model = None
    for _, model in candidates.items():
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
        return _predict_weekly_average(daily_sales, forecast_days, today)

    best_model.fit(X, y)

    tomorrow = today + pd.Timedelta(days=1)
    future_dates = pd.date_range(tomorrow, periods=forecast_days, freq='D')
    fdf = pd.DataFrame({'date': future_dates})
    fdf['dayofweek'] = fdf['date'].dt.dayofweek
    fdf['month'] = fdf['date'].dt.month
    fdf['day'] = fdf['date'].dt.day
    fdf['weekofyear'] = fdf['date'].dt.isocalendar().week.astype(int)
    fdf['is_weekend'] = fdf['dayofweek'].isin([5, 6]).astype(int)

    y_future = np.maximum(best_model.predict(fdf[feature_cols]), 0)
    return float(y_future.sum())


def predict_product_sales(product_df, forecast_days):
    """
    Total expected sales for `forecast_days` days, based on historical data.

      - < 5 months of history → average of the 4 latest complete weeks
      - >= 5 months of history → best of KNN/Linear/Ridge/RF/GB, otherwise
        the 4-week-average baseline
    """
    daily_sales = product_df.groupby('date')['quantity'].sum().reset_index()
    daily_sales = daily_sales.sort_values('date')
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)

    today = pd.Timestamp.now().normalize()
    first_date = daily_sales['date'].min()

    date_range = pd.date_range(start=first_date, end=today, freq='D')
    daily_sales = daily_sales.set_index('date').reindex(
        date_range, fill_value=0).reset_index()
    daily_sales = daily_sales.rename(columns={'index': 'date'})
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)

    months_with_data = len(daily_sales) / 30.0
    if months_with_data < 5:
        return _predict_weekly_average(daily_sales, forecast_days, today)
    return _predict_best_model(daily_sales, forecast_days, today)


# ---------------------------------------------------------------------------
# Forecasting — week-by-week (used by script 4 for the graphical picking list)
# ---------------------------------------------------------------------------

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
            'week_year': f"{iso.year}-W{int(iso.week):02d}",
        })
    return pd.DataFrame(rows)


def _forecast_weeks_short_history(daily_sales, first_future_monday, future_weeks, ref_date):
    """For < 5 months history: KNN on weekly-aggregated data."""
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
            'week_year': f"{iso.year}-W{int(iso.week):02d}",
        })
    return pd.DataFrame(rows)


def _forecast_weeks_best_model(daily_sales, first_future_monday, future_weeks, today):
    """For >= 5 months history: best of KNN/Linear/Ridge/RF/GB, day-by-day, aggregated to weekly."""
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

    train_ds = ds.iloc[:-val_days].copy()
    train_ds['week'] = train_ds['date'].dt.to_period('W').apply(lambda r: r.start_time)
    train_weekly = train_ds.groupby('week')['quantity'].sum()
    train_end = train_ds['date'].iloc[-1]
    cws = train_end - pd.Timedelta(days=train_end.dayofweek)
    complete_train_weeks = train_weekly[train_weekly.index < cws]
    if len(complete_train_weeks) > 0:
        baseline_daily = complete_train_weeks.tail(
            min(4, len(complete_train_weeks))).mean() / 7.0
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
        candidates['rf'] = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42)
        candidates['gb'] = GradientBoostingRegressor(
            n_estimators=50, max_depth=5, random_state=42)

    best_mae = baseline_mae
    best_model = None
    for _, model in candidates.items():
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
    future_dates = pd.date_range(
        first_future_monday, periods=total_days, freq='D')
    fdf = pd.DataFrame({'date': future_dates})
    fdf['dayofweek'] = fdf['date'].dt.dayofweek
    fdf['month'] = fdf['date'].dt.month
    fdf['day'] = fdf['date'].dt.day
    fdf['weekofyear'] = fdf['date'].dt.isocalendar().week.astype(int)
    fdf['is_weekend'] = fdf['dayofweek'].isin([5, 6]).astype(int)
    fdf['quantity'] = np.maximum(best_model.predict(fdf[feature_cols]), 0)

    fdf['week_start'] = fdf['date'] - pd.to_timedelta(
        fdf['date'].dt.dayofweek, unit='D')
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
    Week-by-week sales forecast for the next `future_weeks` weeks.
    Returns DataFrame with: week_start, quantity, week_number, year, week_year.
    """
    daily_sales = product_df.groupby('date')['quantity'].sum().reset_index()
    daily_sales = daily_sales.sort_values('date')
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)

    today = pd.Timestamp.now().normalize()
    first_date = daily_sales['date'].min()

    date_range = pd.date_range(start=first_date, end=today, freq='D')
    daily_sales = daily_sales.set_index('date').reindex(
        date_range, fill_value=0).reset_index()
    daily_sales = daily_sales.rename(columns={'index': 'date'})
    daily_sales['quantity'] = daily_sales['quantity'].fillna(0)

    months_with_data = len(daily_sales) / 30.0

    tomorrow = today + pd.Timedelta(days=1)
    days_until_monday = (7 - tomorrow.weekday()) % 7
    first_future_monday = tomorrow if days_until_monday == 0 else tomorrow + \
        pd.Timedelta(days=days_until_monday)

    if months_with_data < 5:
        return _forecast_weeks_short_history(daily_sales, first_future_monday, future_weeks, today)
    return _forecast_weeks_best_model(daily_sales, first_future_monday, future_weeks, today)
