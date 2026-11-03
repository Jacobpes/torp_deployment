"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnalysisStackedRow, AnalysisStoreRow } from "@/lib/api";
import { formatEuro, formatNumber } from "@/lib/api";

const COLORS = [
  "#1f4638",
  "#c45c26",
  "#1f4f9c",
  "#2f6b4f",
  "#8a5a12",
  "#6b3a6a",
  "#3d7a8c",
  "#a33b2c",
  "#5c6b2f",
  "#8c6a3d",
];

const tooltipStyle = {
  borderRadius: 10,
  border: "1px solid rgba(18,32,24,0.12)",
  background: "#f4f7f1",
};

function colorFor(store: string, stores: string[]) {
  const index = Math.max(0, stores.indexOf(store));
  return COLORS[index % COLORS.length];
}

function shorten(label: string, max = 36) {
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

export function pivotStacked(rows: AnalysisStackedRow[], stores: string[]) {
  return rows.map((row) => {
    const point: Record<string, string | number> = { label: row.label };
    for (const store of stores) point[store] = 0;
    for (const segment of row.segments) point[segment.store] = segment.contribution;
    return point;
  });
}

export function storesInRows(rows: AnalysisStackedRow[], preferred: string[]) {
  const seen = new Set<string>();
  for (const row of rows) {
    for (const segment of row.segments) seen.add(segment.store);
  }
  const ordered = preferred.filter((store) => seen.has(store));
  for (const store of seen) {
    if (!ordered.includes(store)) ordered.push(store);
  }
  return ordered;
}

function EuroTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number; color?: string }[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const rows = payload.filter((item) => Number(item.value) !== 0);
  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      {(rows.length ? rows : payload).map((item) => (
        <div key={String(item.name)} style={{ color: item.color }}>
          {item.name}: {formatEuro(Number(item.value) || 0)}
        </div>
      ))}
    </div>
  );
}

export function ContributionTrend({
  points,
}: {
  points: { period: string; contribution: number; revenue_gross: number; quantity: number }[];
}) {
  if (!points.length) {
    return (
      <div className="chart-empty">
        <p className="muted">Ingen försäljning i vald period.</p>
      </div>
    );
  }
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={points} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
          <CartesianGrid stroke="rgba(18,32,24,0.08)" vertical={false} />
          <XAxis
            dataKey="period"
            tick={{ fill: "#4a5c52", fontSize: 11 }}
            minTickGap={28}
          />
          <YAxis
            tick={{ fill: "#4a5c52", fontSize: 11 }}
            width={64}
            tickFormatter={(value: number) => formatNumber(value)}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value: number, name: string) => {
              if (name === "quantity") return [formatNumber(value), "Antal"];
              if (name === "revenue_gross") return [formatEuro(value), "Omsättning"];
              return [formatEuro(value), "Försäljningsbidrag"];
            }}
          />
          <Legend
            formatter={(value) =>
              value === "contribution"
                ? "Försäljningsbidrag"
                : value === "revenue_gross"
                  ? "Omsättning"
                  : "Antal"
            }
          />
          <Line
            type="monotone"
            dataKey="contribution"
            stroke="#c45c26"
            strokeWidth={2.5}
            dot={false}
            name="contribution"
          />
          <Line
            type="monotone"
            dataKey="revenue_gross"
            stroke="#1f4638"
            strokeWidth={1.5}
            dot={false}
            name="revenue_gross"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function StackedContribution({
  rows,
  stores,
  layout = "horizontal",
}: {
  rows: AnalysisStackedRow[];
  stores: string[];
  layout?: "horizontal" | "vertical";
}) {
  const keys = storesInRows(rows, stores);
  const data = pivotStacked(rows, keys);
  if (!data.length || !keys.length) {
    return (
      <div className="chart-empty">
        <p className="muted">Inget att visa för urvalet.</p>
      </div>
    );
  }
  const vertical = layout === "vertical";
  const height = vertical ? Math.max(280, data.length * 38 + 48) : 340;
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout={vertical ? "vertical" : "horizontal"}
          margin={{ top: 12, right: 16, left: 8, bottom: 8 }}
        >
          <CartesianGrid stroke="rgba(18,32,24,0.08)" vertical={false} />
          {vertical ? (
            <XAxis
              type="number"
              tick={{ fill: "#4a5c52", fontSize: 11 }}
              tickFormatter={(value: number) => formatNumber(value)}
            />
          ) : (
            <XAxis
              dataKey="label"
              tick={{ fill: "#4a5c52", fontSize: 11 }}
              minTickGap={24}
            />
          )}
          {vertical ? (
            <YAxis
              type="category"
              dataKey="label"
              width={210}
              tick={{ fill: "#4a5c52", fontSize: 11 }}
              tickFormatter={(value: string) => shorten(value)}
            />
          ) : (
            <YAxis
              tick={{ fill: "#4a5c52", fontSize: 11 }}
              width={64}
              tickFormatter={(value: number) => formatNumber(value)}
            />
          )}
          <Tooltip content={<EuroTooltip />} />
          <Legend />
          {keys.map((store) => (
            <Bar
              key={store}
              dataKey={store}
              stackId="contribution"
              fill={colorFor(store, stores)}
              name={store}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function StoreBars({ rows }: { rows: AnalysisStoreRow[] }) {
  if (!rows.length) {
    return (
      <div className="chart-empty">
        <p className="muted">Ingen butik i urvalet.</p>
      </div>
    );
  }
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={Math.max(260, rows.length * 36 + 40)}>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 12, right: 16, left: 8, bottom: 8 }}
        >
          <CartesianGrid stroke="rgba(18,32,24,0.08)" vertical={false} />
          <XAxis
            type="number"
            tick={{ fill: "#4a5c52", fontSize: 11 }}
            tickFormatter={(value: number) => formatNumber(value)}
          />
          <YAxis
            type="category"
            dataKey="store"
            width={160}
            tick={{ fill: "#4a5c52", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value: number) => [formatEuro(value), "Försäljningsbidrag"]}
          />
          <Bar dataKey="contribution" fill="#1f4638" radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
