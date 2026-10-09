"""List and authorize downloadable report files."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from api.config import ORDER_DIR, ORDER_SUPPLIER_DIR, PLOCK_DIR, PROJECT_ROOT
from api.models import User
from api.stores import discover_stores

_SCRIPTS = PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _safe_filename(name: str) -> str:
    try:
        from _common import safe_filename

        return safe_filename(name)
    except Exception:
        return "".join(c if c not in '<>:"/\\|?*' else "_" for c in str(name)).strip() or "_"


def allowed_stores_for_user(user: User) -> set[str] | None:
    """None means all stores (admin)."""
    if user.role == "admin":
        return None
    return {s.store_name for s in user.stores}


def store_allowed(user: User, store_name: str) -> bool:
    allowed = allowed_stores_for_user(user)
    if allowed is None:
        return True
    return store_name in allowed


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _match_store_from_plock_name(filename: str, known_stores: list[str]) -> tuple[str | None, str]:
    """Return (store_name, kind) for a plocklistor filename."""
    # Prefer longest store name match so "Lager källby" beats shorter prefixes.
    stores_sorted = sorted(known_stores, key=len, reverse=True)
    if filename.startswith("Plocklista_") and filename.lower().endswith(".pdf"):
        rest = filename[len("Plocklista_") : -4]
        for store in stores_sorted:
            if _safe_filename(store) == rest or store == rest:
                return store, "pdf"
        return rest or None, "pdf"
    if filename.lower().endswith(".xlsx"):
        stem = filename[:-5]
        if stem in (
            "avancerad_plocklista",
            "Plocklistor_sammanställning",
            "Plocklistor_sammanstallning",
        ):
            return None, "other"
        for store in stores_sorted:
            if _safe_filename(store) == stem or store == stem:
                return store, "xlsx"
        # Fallback: treat stem as store name if it looks like a kiosk/lager file
        if "kiosk" in stem.casefold() or stem.casefold().startswith("lager"):
            return stem, "xlsx"
    return None, "other"


def list_files_for_user(user: User) -> list[dict]:
    known = discover_stores()
    # Also include stores from ACL and from existing files
    allowed = allowed_stores_for_user(user)
    if allowed is not None:
        known = sorted(set(known) | allowed, key=lambda s: s.casefold())

    results: list[dict] = []

    if PLOCK_DIR.exists():
        for path in sorted(PLOCK_DIR.iterdir()):
            if not path.is_file():
                continue
            store, kind = _match_store_from_plock_name(path.name, known)
            if kind == "other":
                if user.role != "admin":
                    continue
                results.append(
                    {
                        "store": "",
                        "kind": "other",
                        "name": path.name,
                        "size": path.stat().st_size,
                        "modified_at": _mtime(path),
                        "relative": f"plocklistor/{path.name}",
                    }
                )
                continue
            if store is None:
                continue
            if not store_allowed(user, store):
                continue
            results.append(
                {
                    "store": store,
                    "kind": kind,
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified_at": _mtime(path),
                    "relative": f"plocklistor/{path.name}",
                }
            )

    if user.role == "admin":
        for folder, kind, prefix in (
            (ORDER_DIR, "order_xlsx", "orderlistor"),
            (ORDER_SUPPLIER_DIR, "order_supplier_xlsx", "orderlistor_leverantor"),
        ):
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue
                if not path.name.lower().endswith((".xlsx", ".csv")):
                    continue
                results.append(
                    {
                        "store": "",
                        "kind": kind,
                        "name": path.name,
                        "size": path.stat().st_size,
                        "modified_at": _mtime(path),
                        "relative": f"{prefix}/{path.name}",
                    }
                )

    return results


def resolve_download(user: User, relative: str) -> Path:
    """Resolve a relative output path and enforce ACL."""
    rel = relative.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise PermissionError("Invalid path")

    mapping = {
        "plocklistor": PLOCK_DIR,
        "orderlistor": ORDER_DIR,
        "orderlistor_leverantor": ORDER_SUPPLIER_DIR,
    }
    parts = rel.split("/", 1)
    if len(parts) != 2 or parts[0] not in mapping:
        raise FileNotFoundError("Unknown file location")
    folder_key, name = parts
    if "/" in name or "\\" in name:
        raise PermissionError("Invalid filename")

    if folder_key != "plocklistor" and user.role != "admin":
        raise PermissionError("Admin only")

    base = mapping[folder_key]
    path = (base / name).resolve()
    if not str(path).startswith(str(base.resolve())):
        raise PermissionError("Path escape blocked")
    if not path.is_file():
        raise FileNotFoundError(name)

    if folder_key == "plocklistor":
        known = discover_stores()
        allowed = allowed_stores_for_user(user)
        if allowed is not None:
            known = sorted(set(known) | allowed, key=lambda s: s.casefold())
        store, kind = _match_store_from_plock_name(name, known)
        if kind == "other":
            if user.role != "admin":
                raise PermissionError("Admin only")
        elif store is None or not store_allowed(user, store):
            raise PermissionError("Store not allowed")

    return path
