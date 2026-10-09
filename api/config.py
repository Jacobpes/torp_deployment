"""Runtime configuration for the Torp web API."""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "nedladdningar"
PARAM_DIR = DATA_DIR / "parametrar"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOCK_DIR = OUTPUT_DIR / "plocklistor"
ORDER_DIR = OUTPUT_DIR / "orderlistor"
ORDER_SUPPLIER_DIR = OUTPUT_DIR / "orderlistor_leverantor"
SYSTEM_DATA_DIR = PROJECT_ROOT / "system_data"

# Persist auth DB next to pipeline data
DB_PATH = SYSTEM_DATA_DIR / "web_auth.db"
DATABASE_URL = os.environ.get("TORP_DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")

SESSION_SECRET = os.environ.get("TORP_SESSION_SECRET") or secrets.token_hex(32)
SESSION_COOKIE = "torp_session"
SESSION_MAX_AGE = int(os.environ.get("TORP_SESSION_MAX_AGE", str(60 * 60 * 24 * 7)))

ADMIN_USER = os.environ.get("TORP_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("TORP_ADMIN_PASSWORD", "admin")

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "TORP_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

PIPELINE_LOG = PROJECT_ROOT / "torp_report_generator.log"
JOB_STATE_PATH = SYSTEM_DATA_DIR / "web_job_state.json"
SYSTEM_PLOCK_DIR = SYSTEM_DATA_DIR / "plocklistor"
