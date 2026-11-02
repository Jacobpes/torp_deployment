"""Chart series for in-app picking graphs (history + forecast)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pandas as pd

from api.config import DOWNLOADS_DIR, PROJECT_ROOT, SYSTEM_PLOCK_DIR
from api.files import store_allowed
from api.models import User

_SCRIPTS = PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_cache_lock = threading.Lock()
_sales_cache: dict = {"mtime": None, "df": None, "loaded_at": 0.0}


def _normalize(name: str) -> str:
    try:
        from _common import normalize_name

        return normalize_name(str(name))
    except Exception:
        return " ".join(str(name).split())


def _downloads_mtime() -> float:
    if not DOWNLOADS_DIR.exists():
        return 0.0
    latest = 0.0
    for pattern in ("product_sales_items_*.csv", "product_sales_*.csv"):
        for path in DOWNLOADS_DIR.glob(pattern):
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                continue
    return latest


def get_sales_df() -> pd.DataFrame:
    """Cached sales history (reload when download files change)."""
    mtime = _downloads_mtime()
    with _cache_lock:
        if (
            _sales_cache["df"] is not None
            and _sales_cache["mtime"] == mtime
            and (time.time() - float(_sales_cache["loaded_at"])) < 3600
        ):
            return _sales_cache["df"]

    from _common import load_sales_history

    df = load_sales_history(DOWNLOADS_DIR, keep_unit=True)
    with _cache_lock:
        _sales_cache["df"] = df
        _sales_cache["mtime"] = mtime
        _sales_cache["loaded_at"] = time.time()
    return df


def list_chart_stores(user: User) -> list[str]:
    if not SYSTEM_PLOCK_DIR.exists():
        return []
    stores: list[str] = []
    skip = {"picking_list_results.csv", "avancerad_plocklista.csv"}
    for path in sorted(SYSTEM_PLOCK_DIR.glob("*.csv")):
        if path.name in skip:
            continue
        store = path.stem
        if store_allowed(user, store):
            stores.append(store)
    return stores


def _picking_csv_for_store(store: str) -> Path | None:
    path = SYSTEM_PLOCK_DIR / f"{store}.csv"
    if path.is_file():
        return path
    # Fallback: case-insensitive match
    if not SYSTEM_PLOCK_DIR.exists():
        return None
    for candidate in SYSTEM_PLOCK_DIR.glob("*.csv"):
        if candidate.stem.casefold() == store.casefold():
            return candidate
    return None


def list_products_for_store(store: str) -> list[dict]:
    path = _picking_csv_for_store(store)
    if path is None:
        return []
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # Encoding fallback
    if "Produktnamn" not in df.columns:
        df = pd.read_csv(path, sep=";")

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if "p" in key and "fyll" in key:
            rename[col] = "fill_up"
        elif key in ("saldo_denna_butik", "saldo"):
            rename[col] = "stock"
        elif "varnings" in key:
            rename[col] = "warning"
        elif key in ("produktnamn", "product_name"):
            rename[col] = "product"
        elif key in ("enhet", "unit"):
            rename[col] = "unit"
        elif key in ("produktkod", "product_code"):
            rename[col] = "code"
        elif "leverans" in key:
            rename[col] = "delivery_days"
    df = df.rename(columns=rename)
    if "product" not in df.columns:
        return []

    rows: list[dict] = []
    for _, row in df.iterrows():
        name = row.get("product")
        if name is None or (isinstance(name, float) and pd.isna(name)):
            continue
        name = str(name).strip()
        if not name:
            continue

        def _num(val, default=0.0):
            try:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return default
                return float(val)
            except (TypeError, ValueError):
                return default

        rows.append(
            {
                "product": name,
                "unit": _normalize(str(row.get("unit") or "st")) or "st",
                "code": str(row.get("code") or "") if pd.notna(row.get("code")) else "",
                "stock": _num(row.get("stock")),
                "warning": _num(row.get("warning")),
                "fill_up": _num(row.get("fill_up")),
                "delivery_days": int(_num(row.get("delivery_days"), 7)),
            }
        )

    rows.sort(key=lambda r: (-r["fill_up"], r["product"].casefold()))
    return rows


def get_weekly_sales_history(product_df, store_name, product_name, weeks_back=52):
    """Weekly sales with zero-filled gaps (same logic as PDF generator)."""
    product_sales = product_df[
        (product_df["store"] == store_name) & (product_df["name"] == product_name)
    ].copy()
    empty = pd.DataFrame(
        columns=["week_start", "quantity", "week_number", "year", "week_year"]
    )
    if len(product_sales) == 0:
        return empty

    product_sales["date"] = pd.to_datetime(product_sales["date"])
    product_sales["week_start"] = product_sales["date"] - pd.to_timedelta(
        product_sales["date"].dt.dayofweek, unit="D"
    )
    product_sales["week_number"] = product_sales["week_start"].dt.isocalendar().week
    product_sales["year"] = product_sales["week_start"].dt.isocalendar().year
    product_sales["week_year"] = (
        product_sales["year"].astype(str)
        + "-W"
        + product_sales["week_number"].astype(str).str.zfill(2)
    )

    weekly = (
        product_sales.groupby(["week_start", "week_number", "year", "week_year"])[
            "quantity"
        ]
        .sum()
        .reset_index()
        .sort_values("week_start")
    )
    if len(weekly) == 0:
        return weekly

    today = pd.Timestamp.now().normalize()
    current_week_monday = today - pd.Timedelta(days=today.dayofweek)
    latest_week = max(weekly["week_start"].max(), current_week_monday)
    cutoff = latest_week - pd.Timedelta(weeks=weeks_back)
    weekly = weekly[weekly["week_start"] >= cutoff]

    all_weeks = pd.date_range(start=cutoff, end=latest_week, freq="W-MON")
    series = pd.Series(all_weeks)
    week_numbers = series.dt.isocalendar().week
    years = series.dt.isocalendar().year
    all_df = pd.DataFrame(
        {
            "week_start": all_weeks,
            "week_number": week_numbers.values,
            "year": years.values,
            "week_year": [f"{y}-W{int(w):02d}" for y, w in zip(years, week_numbers)],
        }
    )
    weekly = all_df.merge(weekly[["week_start", "quantity"]], on="week_start", how="left")
    weekly["quantity"] = weekly["quantity"].fillna(0)
    return weekly.sort_values("week_start")


def _match_sales_product(store_sales: pd.DataFrame, product_name: str) -> str | None:
    try:
        from _common import names_equivalent
    except Exception:
        names_equivalent = None  # type: ignore[assignment]

    target = _normalize(product_name).lower()
    names = store_sales["name"].dropna().unique()
    for sales_name in names:
        if target == _normalize(sales_name).lower():
            return str(sales_name)
    if names_equivalent is None:
        return None
    for sales_name in names:
        if names_equivalent(product_name, sales_name):
            return str(sales_name)
    return None


def _series_to_points(df: pd.DataFrame) -> list[dict]:
    points = []
    if df is None or len(df) == 0:
        return points
    for _, row in df.iterrows():
        week_start = row["week_start"]
        if hasattr(week_start, "date"):
            week_start = week_start.date().isoformat()
        else:
            week_start = str(week_start)[:10]
        points.append(
            {
                "week_start": week_start,
                "week_year": str(row.get("week_year", "")),
                "quantity": round(float(row["quantity"]), 2),
            }
        )
    return points


def build_chart(store: str, product: str) -> dict:
    products = list_products_for_store(store)
    meta = next((p for p in products if p["product"] == product), None)
    if meta is None:
        # Allow exact product even if not in picking list (admin exploration)
        meta = {
            "product": product,
            "unit": "st",
            "code": "",
            "stock": 0.0,
            "warning": 0.0,
            "fill_up": 0.0,
            "delivery_days": 7,
        }

    sales_df = get_sales_df()
    store_sales = sales_df[sales_df["store"] == store] if len(sales_df) else sales_df
    # Also try normalized store match
    if len(store_sales) == 0 and len(sales_df):
        target = _normalize(store).lower()
        mask = sales_df["store"].map(lambda s: _normalize(s).lower() == target)
        store_sales = sales_df[mask]
        if len(store_sales):
            store = str(store_sales["store"].iloc[0])

    matched = _match_sales_product(store_sales, product) if len(store_sales) else None
    history = pd.DataFrame(
        columns=["week_start", "quantity", "week_number", "year", "week_year"]
    )
    forecast = history.copy()
    unit = meta["unit"]

    if matched:
        history = get_weekly_sales_history(sales_df, store, matched, weeks_back=52)
        matched_sales = store_sales[store_sales["name"] == matched].copy()
        if len(matched_sales) and "unit" in matched_sales.columns:
            u = matched_sales["unit"].iloc[0]
            if pd.notna(u) and str(u).strip():
                unit = _normalize(str(u)) or unit

        product_sales = matched_sales.copy()
        if "date" in product_sales.columns:
            product_sales["date"] = pd.to_datetime(product_sales["date"], errors="coerce")
        try:
            from _common import predict_weekly_sales

            forecast = predict_weekly_sales(product_sales, future_weeks=4)
        except Exception:
            forecast = pd.DataFrame(
                columns=["week_start", "quantity", "week_number", "year", "week_year"]
            )

    return {
        "store": store,
        "product": product,
        "matched_sales_name": matched,
        "unit": unit,
        "history": _series_to_points(history),
        "forecast": _series_to_points(forecast),
        "meta": {
            "stock": meta["stock"],
            "warning": meta["warning"],
            "fill_up": meta["fill_up"],
            "delivery_days": meta["delivery_days"],
            "code": meta["code"],
        },
    }
