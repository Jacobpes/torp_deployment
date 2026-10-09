"""Background runner for main.py pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.config import JOB_STATE_PATH, PIPELINE_LOG, PROJECT_ROOT, SYSTEM_DATA_DIR

_process: subprocess.Popen | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_state() -> dict[str, Any]:
    if not JOB_STATE_PATH.exists():
        return {
            "running": False,
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "last_error": None,
        }
    try:
        return json.loads(JOB_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "running": False,
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "last_error": None,
        }


def _save_state(state: dict[str, Any]) -> None:
    SYSTEM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOB_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _refresh_running() -> dict[str, Any]:
    global _process
    state = _load_state()

    if _process is not None:
        code = _process.poll()
        if code is None:
            state["running"] = True
            state["pid"] = _process.pid
            _save_state(state)
            return state
        state["running"] = False
        state["exit_code"] = code
        state["finished_at"] = _iso(_utcnow())
        if code != 0:
            state["last_error"] = f"Pipeline exited with code {code}"
        else:
            state["last_error"] = None
        _process = None
        _save_state(state)
        return state

    if state.get("running") and not _pid_alive(state.get("pid")):
        state["running"] = False
        if state.get("finished_at") is None:
            state["finished_at"] = _iso(_utcnow())
        if state.get("exit_code") is None:
            state["exit_code"] = -1
            state["last_error"] = state.get("last_error") or "Pipeline process ended unexpectedly"
        _save_state(state)
    return state


def start_pipeline() -> dict[str, Any]:
    global _process
    state = _refresh_running()
    if state.get("running"):
        raise RuntimeError("A report job is already running")

    main_py = PROJECT_ROOT / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"main.py not found at {main_py}")

    log_path = PROJECT_ROOT / "web_pipeline.log"
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    _process = subprocess.Popen(
        [sys.executable, str(main_py)],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    state = {
        "running": True,
        "pid": _process.pid,
        "started_at": _iso(_utcnow()),
        "finished_at": None,
        "exit_code": None,
        "last_error": None,
        "log_path": str(log_path),
    }
    _save_state(state)
    return state


def read_log_tail(max_chars: int = 8000) -> str:
    state = _load_state()
    candidates = [
        Path(state["log_path"]) if state.get("log_path") else None,
        PROJECT_ROOT / "web_pipeline.log",
        PIPELINE_LOG,
    ]
    for path in candidates:
        if path is None or not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:]
        except OSError:
            continue
    return ""


def get_status() -> dict[str, Any]:
    state = _refresh_running()
    return {
        "running": bool(state.get("running")),
        "pid": state.get("pid"),
        "started_at": _parse_iso(state.get("started_at")),
        "finished_at": _parse_iso(state.get("finished_at")),
        "exit_code": state.get("exit_code"),
        "last_error": state.get("last_error"),
        "log_tail": read_log_tail(),
    }
