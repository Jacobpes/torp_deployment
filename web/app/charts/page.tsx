"use client";

import { useEffect, useMemo, useState } from "react";
import { AuthGate } from "@/components/AuthGate";
import { SalesChart } from "@/components/SalesChart";
import {
  api,
  type ChartProduct,
  type ChartSeries,
} from "@/lib/api";

function ChartsPanel() {
  const [stores, setStores] = useState<string[]>([]);
  const [store, setStore] = useState("");
  const [products, setProducts] = useState<ChartProduct[]>([]);
  const [product, setProduct] = useState("");
  const [series, setSeries] = useState<ChartSeries | null>(null);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loadingProducts, setLoadingProducts] = useState(false);
  const [loadingSeries, setLoadingSeries] = useState(false);

  useEffect(() => {
    api<{ name: string }[]>("/api/charts/stores")
      .then((rows) => {
        const names = rows.map((r) => r.name);
        setStores(names);
        if (names.length) setStore(names[0]);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Kunde inte hämta butiker"),
      );
  }, []);

  useEffect(() => {
    if (!store) return;
    setLoadingProducts(true);
    setError(null);
    setSeries(null);
    setProduct("");
    api<ChartProduct[]>(`/api/charts/${encodeURIComponent(store)}/products`)
      .then((rows) => {
        setProducts(rows);
        if (rows.length) setProduct(rows[0].product);
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Kunde inte hämta produkter"),
      )
      .finally(() => setLoadingProducts(false));
  }, [store]);

  useEffect(() => {
    if (!store || !product) return;
    setLoadingSeries(true);
    setError(null);
    api<ChartSeries>(
      `/api/charts/${encodeURIComponent(store)}/series?product=${encodeURIComponent(product)}`,
    )
      .then(setSeries)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Kunde inte hämta graf"),
      )
      .finally(() => setLoadingSeries(false));
  }, [store, product]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return products;
    return products.filter(
      (p) =>
        p.product.toLowerCase().includes(q) ||
        (p.code || "").toLowerCase().includes(q),
    );
  }, [products, filter]);

  const selectedMeta = products.find((p) => p.product === product);

  return (
    <div className="stack">
      <section className="panel stack">
        <div>
          <h1>Prognoser</h1>
          <p className="muted">
            Försäljningshistorik och 4-veckors prognos – samma underlag som i
            plocklista-PDF:erna.
          </p>
        </div>
        {error ? <p className="error">{error}</p> : null}
        <div className="row" style={{ alignItems: "stretch" }}>
          <label className="field" style={{ minWidth: 220, flex: 1 }}>
            Butik
            <select
              value={store}
              onChange={(e) => setStore(e.target.value)}
              disabled={!stores.length}
            >
              {stores.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ minWidth: 220, flex: 2 }}>
            Sök produkt
            <input
              type="text"
              placeholder="Namn eller kod…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </label>
        </div>
      </section>

      <div className="charts-layout">
        <aside className="panel product-list">
          <h2>Produkter</h2>
          {loadingProducts ? (
            <p className="muted">Laddar…</p>
          ) : (
            <ul>
              {filtered.map((p) => (
                <li key={p.product}>
                  <button
                    type="button"
                    className={
                      p.product === product ? "product-btn active" : "product-btn"
                    }
                    onClick={() => setProduct(p.product)}
                  >
                    <span className="product-name">{p.product}</span>
                    <span className="product-meta">
                      Påfyllning {p.fill_up} · saldo {p.stock}
                    </span>
                  </button>
                </li>
              ))}
              {!filtered.length ? (
                <li>
                  <p className="muted">Inga produkter matchar.</p>
                </li>
              ) : null}
            </ul>
          )}
        </aside>

        <section className="panel stack chart-panel">
          {selectedMeta ? (
            <div>
              <h2>{selectedMeta.product}</h2>
              <p className="muted">
                {store}
                {selectedMeta.code ? ` · kod ${selectedMeta.code}` : ""} ·{" "}
                {selectedMeta.unit}
              </p>
              <div className="row" style={{ marginTop: "0.75rem" }}>
                <span className="pill">Saldo {selectedMeta.stock}</span>
                <span className="pill">Varning {selectedMeta.warning}</span>
                <span className="pill ok">Påfyllning {selectedMeta.fill_up}</span>
              </div>
            </div>
          ) : (
            <h2>Välj en produkt</h2>
          )}

          {loadingSeries ? (
            <p className="muted">Beräknar graf…</p>
          ) : series ? (
            <SalesChart series={series} />
          ) : (
            <p className="muted">Ingen graf att visa.</p>
          )}
        </section>
      </div>
    </div>
  );
}

export default function ChartsPage() {
  return (
    <AuthGate>
      {() => <ChartsPanel />}
    </AuthGate>
  );
}
