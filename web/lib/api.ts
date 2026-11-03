export type Role = "admin" | "user";

export type User = {
  id: number;
  username: string;
  role: Role;
  is_active: boolean;
  stores: string[];
  created_at?: string | null;
};

export type FileItem = {
  store: string;
  kind: "xlsx" | "pdf" | "order_xlsx" | "order_supplier_xlsx" | "other";
  name: string;
  path: string;
  size: number;
  modified_at?: string | null;
};

export type ChartPoint = {
  week_start: string;
  week_year: string;
  quantity: number;
};

export type ChartProduct = {
  product: string;
  unit: string;
  code: string;
  stock: number;
  warning: number;
  fill_up: number;
  delivery_days: number;
};

export type ChartSeries = {
  store: string;
  product: string;
  matched_sales_name?: string | null;
  unit: string;
  history: ChartPoint[];
  forecast: ChartPoint[];
  meta: {
    stock: number;
    warning: number;
    fill_up: number;
    delivery_days: number;
    code: string;
  };
};

export type AnalysisTotals = {
  revenue_gross: number;
  revenue_ex_vat: number;
  purchase_cost: number;
  contribution: number;
  margin_pct: number;
  quantity: number;
  lines: number;
  products: number;
  avg_price: number;
  contribution_per_unit: number;
};

export type AnalysisPoint = {
  period: string;
  contribution: number;
  revenue_gross: number;
  quantity: number;
};

export type AnalysisStoreRow = {
  store: string;
  contribution: number;
  revenue_gross: number;
  quantity: number;
  margin_pct: number;
  share_pct: number;
};

export type AnalysisProductRow = {
  product: string;
  contribution: number;
  revenue_gross: number;
  quantity: number;
  margin_pct: number;
  share_pct: number;
  avg_price: number;
  contribution_per_unit: number;
};

export type AnalysisStackedRow = {
  label: string;
  segments: { store: string; contribution: number }[];
};

export type AnalysisFocus = {
  product: string;
  totals: AnalysisTotals;
  by_store: AnalysisStoreRow[];
  series: AnalysisPoint[];
  stacked_series: AnalysisStackedRow[];
  stacked_bar: AnalysisStackedRow[];
};

export type AnalysisReport = {
  date_from: string;
  date_to: string;
  bounds: { min_date: string; max_date: string };
  bucket: "day" | "week" | "month";
  store: string | null;
  stores: string[];
  assumptions: { vat_factor: number; markup_factor: number; margin_pct: number };
  totals: AnalysisTotals;
  series: AnalysisPoint[];
  by_store: AnalysisStoreRow[];
  products: AnalysisProductRow[];
  stacked_products: AnalysisStackedRow[];
  stacked_series: AnalysisStackedRow[];
  focus: AnalysisFocus | null;
};

export type JobStatus = {
  running: boolean;
  pid?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  last_error?: string | null;
  log_tail: string;
};

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail =
      typeof data === "object" && data && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : res.statusText;
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return data as T;
}

export async function api<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(path, {
    ...init,
    headers,
    credentials: "include",
  });
  return parseJson<T>(res);
}

export function downloadUrl(path: string): string {
  return `/api/files/download?path=${encodeURIComponent(path)}`;
}

const euroFormat = new Intl.NumberFormat("sv-SE", {
  style: "currency",
  currency: "EUR",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const numberFormat = new Intl.NumberFormat("sv-SE", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 2,
});

export function formatEuro(value: number): string {
  return euroFormat.format(value);
}

export function formatNumber(value: number): string {
  return numberFormat.format(value);
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
