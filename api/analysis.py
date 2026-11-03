"""Sales contribution (försäljningsbidrag) for the analysis tab.

Customer price on each completed order line is ``line_total``
(quantity × unit_price). Purchase cost is estimated from a 25% VAT and a
25% margin on the price excluding VAT::

    exkl. moms = line_total / 1.25
    inköp = exkl. moms × 0.75

Contribution is sales excluding VAT minus that purchase cost. Because the
margin is the same for every product, margin percent is the same too.
Euro contribution still ranks products by how much they add, and the
share of total contribution ranks them as a percent of the selection.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from api.config import DOWNLOADS_DIR, PROJECT_ROOT
from api.files import allowed_stores_for_user
from api.models import User

_SCRIPTS = PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import (  # noqa: E402
    canonical_store_name,
    canonical_store_series,
    normalize_name_series,
    stores_equivalent,
)

# 25% VAT, then a 25% margin on the price excluding VAT.
# cost = ex_vat / MARKUP_FACTOR, and 1 - 1/MARKUP_FACTOR = 0.25.
VAT_FACTOR = 1.25
MARKUP_FACTOR = 1 / 0.75
TOP_PRODUCTS = 12

_cache_lock = threading.Lock()
_facts_cache: dict = {"mtime": None, "df": None, "loaded_at": 0.0}


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


def _parse_date(value: str | None, field: str) -> date | None:
    if value is None or not str(value).strip():
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field} måste vara YYYY-MM-DD") from exc


def _money(revenue: float, quantity: float, lines: int, products: int) -> dict:
    ex_vat = revenue / VAT_FACTOR
    cost = ex_vat / MARKUP_FACTOR
    contribution = ex_vat - cost
    margin = (contribution / ex_vat * 100.0) if ex_vat else 0.0
    avg_price = (revenue / quantity) if quantity else 0.0
    per_unit = (contribution / quantity) if quantity else 0.0
    return {
        "revenue_gross": round(revenue, 2),
        "revenue_ex_vat": round(ex_vat, 2),
        "purchase_cost": round(cost, 2),
        "contribution": round(contribution, 2),
        "margin_pct": round(margin, 1),
        "quantity": round(quantity, 2),
        "lines": int(lines),
        "products": int(products),
        "avg_price": round(avg_price, 2),
        "contribution_per_unit": round(per_unit, 2),
    }


def _totals(frame: pd.DataFrame) -> dict:
    if frame is None or len(frame) == 0:
        return _money(0.0, 0.0, 0, 0)
    return _money(
        float(frame["line_total"].sum()),
        float(frame["quantity"].sum()),
        int(len(frame)),
        int(frame["name"].nunique()),
    )


def _load_raw_frames() -> pd.DataFrame:
    """Latest cumulative items file, plus daily snapshots dated after it."""
    import re

    if not DOWNLOADS_DIR.exists():
        raise FileNotFoundError(f"Saknar katalogen {DOWNLOADS_DIR}")

    usecols = [
        "order_item_id",
        "created_at",
        "store_name",
        "order_status",
        "product_name",
        "quantity",
        "unit_price",
        "line_total",
    ]

    def _read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, usecols=lambda c: c in usecols)

    def _end_date(path: Path) -> str | None:
        found = re.findall(r"(\d{4}-\d{2}-\d{2})", path.name)
        return max(found) if found else None

    items = [
        path
        for path in sorted(DOWNLOADS_DIR.glob("product_sales_items_*.csv"))
        if path.stat().st_size >= 1024
    ]
    daily = [
        path
        for path in sorted(DOWNLOADS_DIR.glob("product_sales_*.csv"))
        if not path.name.startswith("product_sales_items_") and path.stat().st_size >= 64
    ]
    if not items and not daily:
        raise FileNotFoundError("Hittade ingen försäljningsfil")

    frames: list[pd.DataFrame] = []
    items_end = None
    if items:
        latest_items = items[-1]
        items_end = _end_date(latest_items)
        frames.append(_read(latest_items))
    else:
        frames.extend(_read(path) for path in daily)
        daily = []

    for path in daily:
        daily_end = _end_date(path)
        if items_end and daily_end and daily_end > items_end:
            frames.append(_read(path))

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "order_item_id" in df.columns:
        df = df.drop_duplicates(subset=["order_item_id"], keep="last")
    return df


def get_sales_facts() -> pd.DataFrame:
    """Cached completed order lines: date, store, name, quantity, line_total."""
    mtime = _downloads_mtime()
    with _cache_lock:
        cached = _facts_cache["df"]
        if (
            cached is not None
            and _facts_cache["mtime"] == mtime
            and (time.time() - float(_facts_cache["loaded_at"])) < 3600
        ):
            return cached

    raw = _load_raw_frames()
    if "order_status" in raw.columns:
        status = raw["order_status"].astype("string").str.strip().str.lower()
        raw = raw[status == "complete"].copy()

    raw["date"] = pd.to_datetime(raw["created_at"], errors="coerce").dt.date
    raw["store"] = canonical_store_series(raw["store_name"])
    raw["name"] = normalize_name_series(raw["product_name"])
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce").fillna(0.0)
    raw["line_total"] = pd.to_numeric(raw["line_total"], errors="coerce").fillna(0.0)
    raw = raw.dropna(subset=["date", "store", "name"])
    raw = raw[(raw["store"].astype(str).str.len() > 0) & (raw["name"].astype(str).str.len() > 0)]
    facts = raw[["date", "store", "name", "quantity", "line_total"]].copy()

    with _cache_lock:
        _facts_cache["df"] = facts
        _facts_cache["mtime"] = mtime
        _facts_cache["loaded_at"] = time.time()
    return facts


def _visible_stores(user: User, stores: list[str]) -> list[str]:
    allowed = allowed_stores_for_user(user)
    if allowed is None:
        return stores
    return [store for store in stores if any(stores_equivalent(store, name) for name in allowed)]


def _bucket_for(start: date, end: date) -> str:
    days = (end - start).days
    if days <= 62:
        return "day"
    if days <= 400:
        return "week"
    return "month"


def _add_period(frame: pd.DataFrame, bucket: str) -> pd.DataFrame:
    ts = pd.to_datetime(frame["date"])
    out = frame.copy()
    if bucket == "month":
        out["period"] = ts.dt.strftime("%Y-%m")
    elif bucket == "week":
        monday = ts - pd.to_timedelta(ts.dt.dayofweek, unit="D")
        out["period"] = monday.dt.strftime("%Y-%m-%d")
    else:
        out["period"] = ts.dt.strftime("%Y-%m-%d")
    return out


def _series(frame: pd.DataFrame, bucket: str) -> list[dict]:
    if len(frame) == 0:
        return []
    keyed = _add_period(frame, bucket)
    grouped = (
        keyed.groupby("period", as_index=False)
        .agg(line_total=("line_total", "sum"), quantity=("quantity", "sum"), lines=("line_total", "size"))
        .sort_values("period")
    )
    points = []
    for row in grouped.itertuples(index=False):
        totals = _money(float(row.line_total), float(row.quantity), int(row.lines), 0)
        points.append(
            {
                "period": str(row.period),
                "contribution": totals["contribution"],
                "revenue_gross": totals["revenue_gross"],
                "quantity": totals["quantity"],
            }
        )
    return points


def _by_store(frame: pd.DataFrame, total_contribution: float) -> list[dict]:
    if len(frame) == 0:
        return []
    grouped = frame.groupby("store", as_index=False).agg(
        line_total=("line_total", "sum"),
        quantity=("quantity", "sum"),
        lines=("line_total", "size"),
        products=("name", "nunique"),
    )
    rows = []
    for row in grouped.itertuples(index=False):
        totals = _money(float(row.line_total), float(row.quantity), int(row.lines), int(row.products))
        share = (totals["contribution"] / total_contribution * 100.0) if total_contribution else 0.0
        rows.append(
            {
                "store": str(row.store),
                "contribution": totals["contribution"],
                "revenue_gross": totals["revenue_gross"],
                "quantity": totals["quantity"],
                "margin_pct": totals["margin_pct"],
                "share_pct": round(share, 1),
            }
        )
    rows.sort(key=lambda item: (-item["contribution"], item["store"].casefold()))
    return rows


def _products(frame: pd.DataFrame, total_contribution: float) -> list[dict]:
    if len(frame) == 0:
        return []
    grouped = frame.groupby("name", as_index=False).agg(
        line_total=("line_total", "sum"),
        quantity=("quantity", "sum"),
        lines=("line_total", "size"),
    )
    rows = []
    for row in grouped.itertuples(index=False):
        totals = _money(float(row.line_total), float(row.quantity), int(row.lines), 1)
        share = (totals["contribution"] / total_contribution * 100.0) if total_contribution else 0.0
        rows.append(
            {
                "product": str(row.name),
                "contribution": totals["contribution"],
                "revenue_gross": totals["revenue_gross"],
                "quantity": totals["quantity"],
                "margin_pct": totals["margin_pct"],
                "share_pct": round(share, 1),
                "avg_price": totals["avg_price"],
                "contribution_per_unit": totals["contribution_per_unit"],
            }
        )
    rows.sort(key=lambda item: (-item["contribution"], item["product"].casefold()))
    return rows


def _stacked(frame: pd.DataFrame, label_col: str, limit: int | None = None) -> list[dict]:
    """Segments of contribution per store, one row per label (product or period)."""
    if len(frame) == 0:
        return []
    work = frame
    if limit is not None and label_col == "name":
        ranking = (
            work.groupby("name")["line_total"].sum().sort_values(ascending=False).head(limit).index
        )
        work = work[work["name"].isin(ranking)]
    grouped = work.groupby([label_col, "store"], as_index=False)["line_total"].sum()
    order = (
        work.groupby(label_col)["line_total"].sum().sort_values(ascending=False).index.tolist()
    )
    if label_col == "period":
        order = sorted(order)
    buckets: dict[str, list[dict]] = {str(label): [] for label in order}
    for row in grouped.itertuples(index=False):
        label = str(getattr(row, label_col))
        revenue = float(row.line_total)
        contribution = _money(revenue, 0.0, 0, 0)["contribution"]
        if contribution == 0:
            continue
        buckets.setdefault(label, []).append({"store": str(row.store), "contribution": contribution})
    return [{"label": label, "segments": buckets[label]} for label in buckets]


def build_analysis(
    user: User,
    date_from: str | None = None,
    date_to: str | None = None,
    store: str | None = None,
    product: str | None = None,
) -> dict:
    start_arg = _parse_date(date_from, "date_from")
    end_arg = _parse_date(date_to, "date_to")

    facts = get_sales_facts()
    all_stores = sorted(facts["store"].dropna().unique().tolist(), key=str.casefold) if len(facts) else []
    visible = _visible_stores(user, all_stores)
    scoped = facts[facts["store"].isin(visible)] if len(facts) else facts

    if len(scoped) == 0:
        data_min = data_max = date.today()
    else:
        data_min = min(scoped["date"])
        data_max = max(scoped["date"])

    end = end_arg or data_max
    start = start_arg or (end - timedelta(days=29))
    if start > end:
        raise ValueError("date_from får inte ligga efter date_to")

    chosen_store = store.strip() if store and store.strip() else None
    if chosen_store:
        chosen_store = canonical_store_name(chosen_store)
        visible_by_key = {name.casefold(): name for name in visible}
        matched = visible_by_key.get(
            chosen_store.casefold() if isinstance(chosen_store, str) else ""
        )
        if matched is None:
            raise PermissionError("Butiken ingår inte i din behörighet")
        chosen_store = matched

    window = scoped[(scoped["date"] >= start) & (scoped["date"] <= end)]
    if chosen_store:
        window = window[window["store"] == chosen_store]

    totals = _totals(window)
    bucket = _bucket_for(start, end)
    products = _products(window, totals["contribution"])

    chosen_product = product.strip() if product and product.strip() else None
    focus = None
    if chosen_product:
        focused = window[window["name"] == chosen_product]
        focus_totals = _totals(focused)
        focus = {
            "product": chosen_product,
            "totals": focus_totals,
            "by_store": _by_store(focused, focus_totals["contribution"]),
            "series": _series(focused, bucket),
            "stacked_series": _stacked(_add_period(focused, bucket), "period") if len(focused) else [],
            "stacked_bar": _stacked(focused, "name"),
        }

    return {
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "bounds": {"min_date": data_min.isoformat(), "max_date": data_max.isoformat()},
        "bucket": bucket,
        "store": chosen_store,
        "stores": visible,
        "assumptions": {
            "vat_factor": VAT_FACTOR,
            "markup_factor": MARKUP_FACTOR,
            "margin_pct": round((1 - 1 / MARKUP_FACTOR) * 100, 1),
        },
        "totals": totals,
        "series": _series(window, bucket),
        "by_store": _by_store(window, totals["contribution"]),
        "products": products,
        "stacked_products": _stacked(window, "name", TOP_PRODUCTS),
        "stacked_series": _stacked(_add_period(window, bucket), "period") if len(window) else [],
        "focus": focus,
    }
