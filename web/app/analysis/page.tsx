"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AuthGate } from "@/components/AuthGate";
import {
  StackedContribution,
  StoreBars,
  ContributionTrend,
} from "@/components/AnalysisCharts";
import {
  api,
  formatEuro,
  formatNumber,
  type AnalysisProductRow,
  type AnalysisReport,
  type AnalysisTotals,
} from "@/lib/api";

type SortKey = "contribution" | "margin_pct" | "share_pct" | "contribution_per_unit";

const SORTS: { id: SortKey; label: string }[] = [
  { id: "contribution", label: "Euro och cent" },
  { id: "margin_pct", label: "Marginal %" },
  { id: "share_pct", label: "Andel %" },
  { id: "contribution_per_unit", label: "Per enhet" },
];

function isoDate(value: Date): string {
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, "0");
  const day = String(value.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function addDays(iso: string, days: number): string {
  const date = new Date(`${iso}T00:00:00`);
  date.setDate(date.getDate() + days);
  return isoDate(date);
}

function Kpis({ totals, caption }: { totals: AnalysisTotals; caption: string }) {
  const cards = [
    ["Försäljningsbidrag", formatEuro(totals.contribution), true],
    ["Omsättning", formatEuro(totals.revenue_gross), false],
    ["Omsättning exkl. moms", formatEuro(totals.revenue_ex_vat), false],
    ["Inköp", formatEuro(totals.purchase_cost), false],
    ["Marginal", `${formatNumber(totals.margin_pct)} %`, false],
    ["Antal sålda", formatNumber(totals.quantity), false],
    ["Snittpris", formatEuro(totals.avg_price), false],
    ["Bidrag / enhet", formatEuro(totals.contribution_per_unit), false],
  ] as const;
  return (
    <div className="stack">
      <p className="muted">{caption}</p>
      <div className="kpi-grid">
        {cards.map(([label, value, hero]) => (
          <div key={label} className={hero ? "kpi hero" : "kpi"}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function AnalysisPanel() {
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [store, setStore] = useState("");
  const [product, setProduct] = useState("");
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("contribution");
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams();
    if (ready && dateFrom) params.set("date_from", dateFrom);
    if (ready && dateTo) params.set("date_to", dateTo);
    if (store) params.set("store", store);
    if (product) params.set("product", product);
    const queryString = params.toString();
    setLoading(true);
    setError(null);
    api<AnalysisReport>(`/api/analysis${queryString ? `?${queryString}` : ""}`)
      .then((data) => {
        setReport(data);
        if (!ready) {
          setDateFrom(data.date_from);
          setDateTo(data.date_to);
          setReady(true);
        }
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Kunde inte hämta analysen"),
      )
      .finally(() => setLoading(false));
  }, [ready, dateFrom, dateTo, store, product]);

  const totals = report?.focus?.totals ?? report?.totals;
  const rows = useMemo(() => {
    const source = report?.products ?? [];
    const q = query.trim().toLowerCase();
    const filtered = q
      ? source.filter((row) => row.product.toLowerCase().includes(q))
      : source;
    return [...filtered].sort((a, b) => {
      const delta = b[sortKey] - a[sortKey];
      if (delta !== 0) return delta;
      return a.product.localeCompare(b.product, "sv");
    });
  }, [report, query, sortKey]);

  function applyPreset(kind: "7" | "30" | "90" | "year" | "all") {
    if (!report) return;
    const end = report.bounds.max_date;
    if (kind === "all") {
      setDateFrom(report.bounds.min_date);
      setDateTo(end);
      return;
    }
    if (kind === "year") {
      setDateFrom(`${end.slice(0, 4)}-01-01`);
      setDateTo(end);
      return;
    }
    const days = Number(kind);
    setDateFrom(addDays(end, -(days - 1)));
    setDateTo(end);
  }

  const bucketLabel =
    report?.bucket === "day" ? "per dag" : report?.bucket === "week" ? "per vecka" : "per månad";
  const scopeLabel = report?.focus
    ? report.focus.product
    : store
      ? store
      : "Alla butiker";

  return (
    <div className="stack">
      <section className="panel stack">
        <div>
          <h1>Analysverktyg</h1>
          <p className="muted">
            Försäljningsbidrag, omsättning och antal för en produkt, en butik
            eller hela kedjan. Begränsa perioden och klicka en produkt för att
            se hur bidraget fördelas mellan butikerna.
          </p>
        </div>
        {error ? <p className="error">{error}</p> : null}
        <div className="seg">
          <button type="button" onClick={() => applyPreset("7")}>
            7 dagar
          </button>
          <button type="button" onClick={() => applyPreset("30")}>
            30 dagar
          </button>
          <button type="button" onClick={() => applyPreset("90")}>
            90 dagar
          </button>
          <button type="button" onClick={() => applyPreset("year")}>
            I år
          </button>
          <button type="button" onClick={() => applyPreset("all")}>
            Allt
          </button>
        </div>
        <div className="row" style={{ alignItems: "stretch" }}>
          <label className="field">
            Från
            <input
              type="date"
              value={dateFrom}
              min={report?.bounds.min_date}
              max={dateTo || report?.bounds.max_date}
              onChange={(e) => setDateFrom(e.target.value)}
            />
          </label>
          <label className="field">
            Till
            <input
              type="date"
              value={dateTo}
              min={dateFrom || report?.bounds.min_date}
              max={report?.bounds.max_date}
              onChange={(e) => setDateTo(e.target.value)}
            />
          </label>
          <label className="field" style={{ minWidth: 220, flex: 1 }}>
            Butik
            <select value={store} onChange={(e) => setStore(e.target.value)}>
              <option value="">Alla butiker</option>
              {(report?.stores ?? []).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ minWidth: 280, flex: 1 }}>
            Sök produkt
            <ProductCombobox
              products={(report?.products ?? []).map((row) => row.product)}
              query={query}
              selected={product}
              onQuery={setQuery}
              onSelect={setProduct}
            />
          </label>
        </div>
        {report ? (
          <p className="note">
            Försäljningsbidrag = omsättning exkl. {Math.round((report.assumptions.vat_factor - 1) * 100)} %
            moms − inköp. Marginalen är {formatNumber(report.assumptions.margin_pct)} % av
            omsättningen exkl. moms för varje produkt.
            Euro och andel visar vilka produkter och butiker som bidrar mest.
            Bara avslutade köp räknas.
          </p>
        ) : null}
        {loading ? <p className="muted">Hämtar siffror…</p> : null}
      </section>

      {report && totals ? (
        <>
          <section className="panel stack">
            <Kpis
              totals={totals}
              caption={`${scopeLabel} · ${report.date_from} – ${report.date_to}`}
            />
          </section>

          <section className="panel stack">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>Utveckling {bucketLabel}</h2>
              {product ? (
                <button type="button" className="btn ghost" onClick={() => setProduct("")}>
                  Visa alla produkter
                </button>
              ) : null}
            </div>
            {report.focus ? (
              <StackedContribution
                rows={report.focus.stacked_series}
                stores={report.stores}
              />
            ) : (
              <ContributionTrend points={report.series} />
            )}
          </section>

          <section className="panel stack">
            <h2>
              {report.focus
                ? "Butikernas andel av produkten"
                : "Produkter per butik"}
            </h2>
            <p className="muted">
              {report.focus
                ? "En stapel, uppdelad i färg per butik."
                : "De 12 produkter med högst försäljningsbidrag. Varje färg är en butik."}
            </p>
            <StackedContribution
              rows={report.focus ? report.focus.stacked_bar : report.stacked_products}
              stores={report.stores}
              layout="vertical"
            />
          </section>

          <section className="panel stack">
            <h2>
              {report.focus
                ? "Butiker för den valda produkten"
                : store
                  ? `Bidrag i ${store}`
                  : "Butikernas försäljningsbidrag"}
            </h2>
            <StoreBars rows={report.focus ? report.focus.by_store : report.by_store} />
          </section>

          <section className="panel stack">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2>Produkter</h2>
              <div className="seg" role="group" aria-label="Sortering">
                {SORTS.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={sortKey === item.id ? "active" : undefined}
                    onClick={() => setSortKey(item.id)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="table-scroll">
              <table className="table">
                <thead>
                  <tr>
                    <th>Produkt</th>
                    <th className="num">Bidrag</th>
                    <th className="num">Andel</th>
                    <th className="num">Marginal</th>
                    <th className="num">Omsättning</th>
                    <th className="num">Antal</th>
                    <th className="num">Snittpris</th>
                    <th className="num">Per enhet</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <ProductRow
                      key={row.product}
                      row={row}
                      active={row.product === product}
                      onSelect={() =>
                        setProduct((current) => (current === row.product ? "" : row.product))
                      }
                    />
                  ))}
                  {!rows.length ? (
                    <tr>
                      <td colSpan={8}>
                        <p className="muted">Inga produkter i urvalet.</p>
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}

function ProductCombobox({
  products,
  query,
  selected,
  onQuery,
  onSelect,
}: {
  products: string[];
  query: string;
  selected: string;
  onQuery: (value: string) => void;
  onSelect: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const options = useMemo(() => {
    const names = [...products].sort((a, b) => a.localeCompare(b, "sv"));
    const needle = query.trim().toLocaleLowerCase("sv");
    if (!needle || needle === selected.toLocaleLowerCase("sv")) return names;
    return names.filter((name) => name.toLocaleLowerCase("sv").includes(needle));
  }, [products, query, selected]);

  useEffect(() => {
    setActive(0);
  }, [query]);

  useEffect(() => {
    const node = listRef.current?.children[active] as HTMLElement | undefined;
    node?.scrollIntoView({ block: "nearest" });
  }, [active, open]);

  useEffect(() => {
    function onPointerDown(event: MouseEvent) {
      const root = rootRef.current;
      if (!root) return;
      const target = event.target as Node;
      if (root.contains(target)) return;
      if (root.closest("label")?.contains(target)) return;
      setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, []);

  function choose(name: string) {
    onQuery(name);
    onSelect(name);
    setOpen(false);
  }

  function clear() {
    onQuery("");
    onSelect("");
    setOpen(false);
  }

  return (
    <div className="combo" ref={rootRef}>
      <input
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
        aria-controls="product-combobox-list"
        placeholder="Skriv för att filtrera…"
        value={query}
        onFocus={() => setOpen(true)}
        onMouseDown={() => setOpen(true)}
        onChange={(event) => {
          const value = event.target.value;
          onQuery(value);
          setOpen(true);
          if (selected && value !== selected) onSelect("");
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setOpen(true);
            setActive((index) => Math.min(index + 1, Math.max(options.length - 1, 0)));
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setOpen(true);
            setActive((index) => Math.max(index - 1, 0));
          } else if (event.key === "Enter" && open && options[active]) {
            event.preventDefault();
            choose(options[active]);
          } else if (event.key === "Escape") {
            setOpen(false);
          }
        }}
      />
      {query || selected ? (
        <button type="button" className="combo-clear" onClick={clear}>
          Rensa
        </button>
      ) : null}
      {open ? (
        <ul className="combo-list" id="product-combobox-list" role="listbox" ref={listRef}>
          {options.map((name, index) => (
            <li key={name}>
              <button
                type="button"
                role="option"
                aria-selected={name === selected}
                className={index === active ? "active" : undefined}
                onMouseEnter={() => setActive(index)}
                onMouseDown={(event) => {
                  event.preventDefault();
                  choose(name);
                }}
              >
                {name}
              </button>
            </li>
          ))}
          {!options.length ? <li className="combo-empty">Ingen produkt matchar.</li> : null}
        </ul>
      ) : null}
    </div>
  );
}

function ProductRow({
  row,
  active,
  onSelect,
}: {
  row: AnalysisProductRow;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <tr className={active ? "row-active" : undefined}>
      <td>
        <button type="button" className="row-btn" onClick={onSelect}>
          {row.product}
        </button>
      </td>
      <td className="num">{formatEuro(row.contribution)}</td>
      <td className="num">{formatNumber(row.share_pct)} %</td>
      <td className="num">{formatNumber(row.margin_pct)} %</td>
      <td className="num">{formatEuro(row.revenue_gross)}</td>
      <td className="num">{formatNumber(row.quantity)}</td>
      <td className="num">{formatEuro(row.avg_price)}</td>
      <td className="num">{formatEuro(row.contribution_per_unit)}</td>
    </tr>
  );
}

export default function AnalysisPage() {
  return (
    <AuthGate>
      {() => <AnalysisPanel />}
    </AuthGate>
  );
}
