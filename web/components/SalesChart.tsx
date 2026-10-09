"use client";

import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";
import type { ChartSeries } from "@/lib/api";

type Row = {
  label: string;
  history?: number | null;
  forecast?: number | null;
};

export function SalesChart({ series }: { series: ChartSeries }) {
  const data = useMemo(() => {
    const rows: Row[] = [];
    for (const p of series.history) {
      rows.push({
        label: p.week_year,
        history: p.quantity,
        forecast: null,
      });
    }
    for (const p of series.forecast) {
      rows.push({
        label: p.week_year,
        history: null,
        forecast: p.quantity,
      });
    }
    // Connect history to forecast visually: set last history point's forecast
    // equal to last history qty so the red line starts from the boundary.
    if (rows.length && series.forecast.length) {
      const lastHistIdx = series.history.length - 1;
      if (lastHistIdx >= 0) {
        rows[lastHistIdx] = {
          ...rows[lastHistIdx],
          forecast: rows[lastHistIdx].history ?? null,
        };
      }
    }
    return rows;
  }, [series]);

  const splitLabel =
    series.history.length > 0
      ? series.history[series.history.length - 1]?.week_year
      : undefined;

  if (data.length === 0) {
    return (
      <div className="chart-empty">
        <p className="muted">Ingen försäljningshistorik för denna produkt.</p>
      </div>
    );
  }

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={data} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="rgba(18,32,24,0.08)" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: "#4a5c52", fontSize: 11 }}
            interval="preserveStartEnd"
            minTickGap={28}
          />
          <YAxis
            tick={{ fill: "#4a5c52", fontSize: 11 }}
            width={42}
            label={{
              value: series.unit,
              angle: -90,
              position: "insideLeft",
              fill: "#4a5c52",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={{
              borderRadius: 10,
              border: "1px solid rgba(18,32,24,0.12)",
              background: "#f4f7f1",
            }}
            formatter={(value: number, name: string) => [
              `${value} ${series.unit}`,
              name === "history" ? "Historik" : "Prognos",
            ]}
          />
          <Legend
            formatter={(value) =>
              value === "history" ? "Historik (52 v)" : "Prognos (4 v)"
            }
          />
          {splitLabel ? (
            <ReferenceLine
              x={splitLabel}
              stroke="#c45c26"
              strokeDasharray="4 4"
              strokeOpacity={0.55}
            />
          ) : null}
          <Area
            type="monotone"
            dataKey="history"
            stroke="#1f4f9c"
            fill="#1f4f9c"
            fillOpacity={0.22}
            strokeWidth={2}
            connectNulls={false}
            dot={false}
            activeDot={{ r: 4 }}
            name="history"
          />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#c45c26"
            strokeWidth={2.5}
            dot={{ r: 4, fill: "#c45c26" }}
            connectNulls
            name="forecast"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
