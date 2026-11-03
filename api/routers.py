"""API routers."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import jobs
from api.auth import (
    clear_session_cookie,
    get_current_user,
    hash_password,
    require_admin,
    set_session_cookie,
    user_to_out,
    verify_password,
)
from api.analysis import build_analysis
from api.charts import build_chart, list_chart_stores, list_products_for_store
from api.database import get_db
from api.files import list_files_for_user, resolve_download, store_allowed
from api.models import User, UserStore
from api.schemas import (
    AnalysisOut,
    ChartProductOut,
    ChartSeriesOut,
    CreateUserRequest,
    FileOut,
    JobStatusOut,
    LoginRequest,
    PatchUserRequest,
    SetStoresRequest,
    StoreOut,
    UserOut,
)
from api.stores import discover_stores

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])
files_router = APIRouter(prefix="/api/files", tags=["files"])
charts_router = APIRouter(prefix="/api/charts", tags=["charts"])
analysis_router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@auth_router.post("/login", response_model=UserOut)
def login(
    body: LoginRequest,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    user = db.scalar(select(User).where(User.username == body.username.strip()))
    if user is None or not user.is_active or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    set_session_cookie(response, user.id)
    return user_to_out(user)


@auth_router.post("/logout")
def logout(response: Response):
    clear_session_cookie(response)
    return {"ok": True}


@auth_router.get("/me", response_model=UserOut)
def me(user: Annotated[User, Depends(get_current_user)]):
    return user_to_out(user)


@admin_router.get("/users", response_model=list[UserOut])
def list_users(
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    users = db.scalars(select(User).order_by(User.username)).all()
    return [user_to_out(u) for u in users]


@admin_router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    body: CreateUserRequest,
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username required")
    existing = db.scalar(select(User).where(User.username == username))
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")
    user = User(
        username=username,
        password_hash=hash_password(body.password),
        role=body.role,
        is_active=True,
    )
    db.add(user)
    db.flush()
    for store in sorted({s.strip() for s in body.stores if s and s.strip()}):
        db.add(UserStore(user_id=user.id, store_name=store))
    db.commit()
    db.refresh(user)
    return user_to_out(user)


@admin_router.patch("/users/{user_id}", response_model=UserOut)
def patch_user(
    user_id: int,
    body: PatchUserRequest,
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if body.password is not None:
        user.password_hash = hash_password(body.password)
    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active
    db.commit()
    db.refresh(user)
    return user_to_out(user)


@admin_router.put("/users/{user_id}/stores", response_model=UserOut)
def set_user_stores(
    user_id: int,
    body: SetStoresRequest,
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.stores.clear()
    db.flush()
    for store in sorted({s.strip() for s in body.stores if s and s.strip()}):
        db.add(UserStore(user_id=user.id, store_name=store))
    db.commit()
    db.refresh(user)
    return user_to_out(user)


@admin_router.get("/stores", response_model=list[StoreOut])
def list_stores(_: Annotated[User, Depends(require_admin)]):
    return [{"name": n} for n in discover_stores()]


@admin_router.post("/jobs/run", response_model=JobStatusOut)
def run_job(_: Annotated[User, Depends(require_admin)]):
    try:
        jobs.start_pipeline()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return jobs.get_status()


@admin_router.get("/jobs/status", response_model=JobStatusOut)
def job_status(_: Annotated[User, Depends(require_admin)]):
    return jobs.get_status()


@files_router.get("", response_model=list[FileOut])
def list_files(user: Annotated[User, Depends(get_current_user)]):
    rows = list_files_for_user(user)
    return [
        {
            "store": r["store"],
            "kind": r["kind"],
            "name": r["name"],
            "path": r["relative"],
            "size": r["size"],
            "modified_at": r["modified_at"],
        }
        for r in rows
    ]


@files_router.get("/download")
def download_file(
    path: str,
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        file_path = resolve_download(user, path)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@charts_router.get("/stores", response_model=list[StoreOut])
def chart_stores(user: Annotated[User, Depends(get_current_user)]):
    return [{"name": s} for s in list_chart_stores(user)]


@charts_router.get("/{store}/products", response_model=list[ChartProductOut])
def chart_products(store: str, user: Annotated[User, Depends(get_current_user)]):
    if not store_allowed(user, store):
        raise HTTPException(status_code=403, detail="Store not allowed")
    products = list_products_for_store(store)
    if not products:
        raise HTTPException(status_code=404, detail="No picking list for store")
    return products


@charts_router.get("/{store}/series", response_model=ChartSeriesOut)
def chart_series(
    store: str,
    product: str,
    user: Annotated[User, Depends(get_current_user)],
):
    if not store_allowed(user, store):
        raise HTTPException(status_code=403, detail="Store not allowed")
    if not product.strip():
        raise HTTPException(status_code=400, detail="product required")
    try:
        return build_chart(store, product.strip())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart failed: {e}") from e


@analysis_router.get("", response_model=AnalysisOut)
def analysis(
    user: Annotated[User, Depends(get_current_user)],
    date_from: str | None = None,
    date_to: str | None = None,
    store: str | None = None,
    product: str | None = None,
):
    try:
        return build_analysis(
            user,
            date_from=date_from,
            date_to=date_to,
            store=store,
            product=product,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysen misslyckades: {e}") from e
