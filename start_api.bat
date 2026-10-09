@echo off
REM Start Torp web API (FastAPI) on http://127.0.0.1:8000
cd /d "%~dp0"
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
