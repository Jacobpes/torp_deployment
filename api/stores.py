"""Discover store names from pipeline data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from api.config import DOWNLOADS_DIR, PARAM_DIR, PROJECT_ROOT

_SCRIPTS = PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _normalize(name: str) -> str:
    try:
        from _common import canonical_store_name

        value = canonical_store_name(str(name))
        return value if isinstance(value, str) else " ".join(str(name).split())
    except Exception:
        return " ".join(str(name).split())


def canonical_store_name(name: str) -> str:
    """Current store name. 'Bosund kiosk' and 'Bosund butik' are the same."""
    return _normalize(name)


def discover_stores() -> list[str]:
    """Union of stores from latest stock_report and Leveransfrekvens.csv."""
    names: set[str] = set()

    stock = _latest_stock_report()
    if stock is not None:
        try:
            df = pd.read_csv(stock, usecols=lambda c: c in ("store_name",))
            if "store_name" in df.columns:
                for raw in df["store_name"].dropna().unique():
                    n = _normalize(raw)
                    if n:
                        names.add(n)
        except Exception:
            pass

    lev = PARAM_DIR / "Leveransfrekvens.csv"
    if lev.exists():
        try:
            df = pd.read_csv(lev, sep=";", encoding="utf-8-sig")
            col = next(
                (c for c in df.columns if str(c).strip().lower() in ("store", "butik", "store_name")),
                df.columns[0],
            )
            for raw in df[col].dropna().unique():
                n = _normalize(raw)
                if n and n.lower() not in ("store", "butik", "store_name"):
                    names.add(n)
        except Exception:
            pass

    return sorted(names, key=lambda s: s.casefold())


def _latest_stock_report() -> Path | None:
    if not DOWNLOADS_DIR.exists():
        return None
    files = sorted(DOWNLOADS_DIR.glob("stock_report_*.csv"))
    return files[-1] if files else None
