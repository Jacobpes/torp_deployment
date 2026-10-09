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

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
