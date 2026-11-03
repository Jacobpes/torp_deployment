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


class AnalysisTotals(BaseModel):
    revenue_gross: float
    revenue_ex_vat: float
    purchase_cost: float
    contribution: float
    margin_pct: float
    quantity: float
    lines: int
    products: int
    avg_price: float
    contribution_per_unit: float


class AnalysisPoint(BaseModel):
    period: str
    contribution: float
    revenue_gross: float
    quantity: float


class AnalysisStoreRow(BaseModel):
    store: str
    contribution: float
    revenue_gross: float
    quantity: float
    margin_pct: float
    share_pct: float


class AnalysisProductRow(BaseModel):
    product: str
    contribution: float
    revenue_gross: float
    quantity: float
    margin_pct: float
    share_pct: float
    avg_price: float
    contribution_per_unit: float


class AnalysisSegment(BaseModel):
    store: str
    contribution: float


class AnalysisStackedRow(BaseModel):
    label: str
    segments: list[AnalysisSegment]


class AnalysisFocus(BaseModel):
    product: str
    totals: AnalysisTotals
    by_store: list[AnalysisStoreRow]
    series: list[AnalysisPoint]
    stacked_series: list[AnalysisStackedRow]
    stacked_bar: list[AnalysisStackedRow]


class AnalysisOut(BaseModel):
    date_from: str
    date_to: str
    bounds: dict[str, str]
    bucket: Literal["day", "week", "month"]
    store: str | None = None
    stores: list[str]
    assumptions: dict[str, float]
    totals: AnalysisTotals
    series: list[AnalysisPoint]
    by_store: list[AnalysisStoreRow]
    products: list[AnalysisProductRow]
    stacked_products: list[AnalysisStackedRow]
    stacked_series: list[AnalysisStackedRow]
    focus: AnalysisFocus | None = None
