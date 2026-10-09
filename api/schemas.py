"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["admin", "user"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserOut(BaseModel):
    id: int
    username: str
    role: Role
    is_active: bool
    stores: list[str]
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=4, max_length=256)
    role: Role = "user"
    stores: list[str] = Field(default_factory=list)


class PatchUserRequest(BaseModel):
    password: str | None = Field(default=None, min_length=4, max_length=256)
    role: Role | None = None
    is_active: bool | None = None


class SetStoresRequest(BaseModel):
    stores: list[str]


class StoreOut(BaseModel):
    name: str


class FileOut(BaseModel):
    store: str
    kind: Literal["xlsx", "pdf", "order_xlsx", "order_supplier_xlsx", "other"]
    name: str
    path: str
    size: int
    modified_at: datetime | None = None


class JobStatusOut(BaseModel):
    running: bool
    pid: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    last_error: str | None = None
    log_tail: str = ""


class ChartPoint(BaseModel):
    week_start: str
    week_year: str
    quantity: float


class ChartProductOut(BaseModel):
    product: str
    unit: str
    code: str = ""
    stock: float = 0
    warning: float = 0
    fill_up: float = 0
    delivery_days: int = 7


class ChartMetaOut(BaseModel):
    stock: float
    warning: float
    fill_up: float
    delivery_days: int
    code: str = ""


class ChartSeriesOut(BaseModel):
    store: str
    product: str
    matched_sales_name: str | None = None
    unit: str
    history: list[ChartPoint]
    forecast: list[ChartPoint]
    meta: ChartMetaOut
