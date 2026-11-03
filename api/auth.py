"""Password hashing and session helpers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from api.config import SESSION_COOKIE, SESSION_MAX_AGE, SESSION_SECRET
from api.database import get_db
from api.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="torp-session")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_session_token(user_id: int) -> str:
    return _serializer.dumps({"uid": user_id})


def read_session_token(token: str) -> int | None:
    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired, TypeError, ValueError):
        return None
    uid = data.get("uid")
    return int(uid) if uid is not None else None


def set_session_cookie(response: Response, user_id: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=create_session_token(user_id),
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


def get_current_user(
    db: Annotated[Session, Depends(get_db)],
    torp_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> User:
    if not torp_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user_id = read_session_token(torp_session)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive")
    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user


def _canonical_user_stores(user: User) -> list[str]:
    from api.stores import canonical_store_name

    seen: dict[str, str] = {}
    for link in user.stores:
        name = canonical_store_name(link.store_name)
        if name:
            seen.setdefault(name.casefold(), name)
    return sorted(seen.values(), key=str.casefold)


def user_to_out(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "is_active": user.is_active,
        "stores": _canonical_user_stores(user),
        "created_at": user.created_at,
    }
