# Torp web app

Run the existing Python report pipeline behind a small web UI: admins manage users and store access; store users download their picking lists.

## Stack

- **API:** FastAPI (`api/`) — auth, ACL, job runner for `main.py`, file downloads
- **UI:** Next.js + TypeScript (`web/`) — login, admin, downloads
- **DB:** SQLite at `system_data/web_auth.db`

## Setup

```bash
# API deps (from repo root)
pip install -r requirements-web.txt
# also needs existing pipeline deps: pandas, openpyxl, scikit-learn, paramiko, ...

# Frontend
cd web
npm install
```

Copy `.env.web.example` values into your environment (or set them in the shell). Change `TORP_ADMIN_PASSWORD` and `TORP_SESSION_SECRET` before any real deployment.

Default first-run admin (only if the DB is empty):

- user: `admin`
- password: `admin`

## Run (development)

Terminal 1 — API:

```bash
# from repo root
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
cd web
npm run dev
```

Open http://localhost:3000 — Next.js proxies `/api/*` to the FastAPI server.

## Roles

| Role | Capabilities |
|------|----------------|
| `admin` | Create users, assign stores, start report job, download all plock/order files |
| `user` | Download Excel/PDF picking lists only for assigned stores; view charts for those stores |

Store names must match pipeline names (e.g. `Bosund kiosk`, `Lager källby`).

## Charts

Open **Grafer** in the UI (`/charts`). The API serves weekly history + 4-week forecast from the same sales data and forecast helpers as the PDF generator (`GET /api/charts/{store}/series?product=...`), ACL-filtered by store.
## Notes

- Starting a job runs `python main.py` in the project root (download + picking + order + PDFs). Only one job at a time.
- Job log tail is shown in Admin; full log also lands in `web_pipeline.log`.
- The Windows `.exe` / `Prognosautomation` flow is unchanged and can run in parallel on another machine.
