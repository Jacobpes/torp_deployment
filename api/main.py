"""Bootstrap admin user and FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from api.auth import hash_password
from api.config import ADMIN_PASSWORD, ADMIN_USER, CORS_ORIGINS
from api.database import SessionLocal, init_db
from api.models import User
from api.routers import admin_router, analysis_router, auth_router, charts_router, files_router


def bootstrap_admin() -> None:
    db = SessionLocal()
    try:
        count = db.scalar(select(func.count()).select_from(User)) or 0
        if count > 0:
            return
        admin = User(
            username=ADMIN_USER.strip() or "admin",
            password_hash=hash_password(ADMIN_PASSWORD or "admin"),
            role="admin",
            is_active=True,
        )
        db.add(admin)
        db.commit()
        print(f"[torp-api] Seeded admin user '{admin.username}'")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    bootstrap_admin()
    yield


app = FastAPI(title="Torp Report API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(files_router)
app.include_router(charts_router)
app.include_router(analysis_router)


@app.get("/api/health")
def health():
    return {"ok": True}
