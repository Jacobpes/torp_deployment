AI Prediction tool for Torp shop

## Web app

FastAPI + Next.js UI for running reports on a server, managing users, and store-scoped downloads.

See [web/README.md](web/README.md).

Quick start:

```bash
pip install -r requirements-web.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

cd web
npm install
npm run dev
```

Open http://localhost:3000 (default admin `admin` / `admin` on first empty DB — change via `TORP_ADMIN_*` env vars; see `.env.web.example`).
