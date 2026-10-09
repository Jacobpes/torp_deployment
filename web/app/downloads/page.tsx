"use client";

import { useEffect, useMemo, useState } from "react";
import { AuthGate } from "@/components/AuthGate";
import {
  api,
  downloadUrl,
  formatBytes,
  type FileItem,
  type User,
} from "@/lib/api";

function DownloadsPanel({ user }: { user: User }) {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<FileItem[]>("/api/files")
      .then(setFiles)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Kunde inte hämta filer"),
      );
  }, []);

  const byStore = useMemo(() => {
    const map = new Map<string, FileItem[]>();
    for (const f of files) {
      if (!f.store) continue;
      const list = map.get(f.store) || [];
      list.push(f);
      map.set(f.store, list);
    }
    return Array.from(map.entries()).sort((a, b) =>
      a[0].localeCompare(b[0], "sv"),
    );
  }, [files]);

  const adminExtra = useMemo(
    () => files.filter((f) => !f.store || f.kind.startsWith("order") || f.kind === "other"),
    [files],
  );

  return (
    <div className="stack">
      <section className="panel stack">
        <div>
          <h1>Filer</h1>
          <p className="muted">
            {user.role === "admin"
              ? "Alla plocklistor och orderlistor."
              : "Plocklistor för dina tilldelade butiker."}
          </p>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {byStore.length === 0 && !error ? (
        <section className="panel">
          <p className="muted">Inga filer tillgängliga ännu.</p>
        </section>
      ) : null}

      <div className="store-grid">
        {byStore.map(([store, items], idx) => (
          <article
            key={store}
            className="store-card"
            style={{ animationDelay: `${idx * 40}ms` }}
          >
            <h3>{store}</h3>
            <div className="stack">
              {items.map((file) => (
                <div key={file.path} className="row">
                  <a
                    className="btn ghost"
                    href={downloadUrl(file.path)}
                    style={{ borderRadius: 10 }}
                  >
                    {file.kind === "pdf" ? "PDF" : "Excel"} · {formatBytes(file.size)}
                  </a>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>

      {user.role === "admin" && adminExtra.length > 0 ? (
        <section className="panel stack">
          <h2>Övrigt (admin)</h2>
          <table className="table">
            <thead>
              <tr>
                <th>Fil</th>
                <th>Typ</th>
                <th>Storlek</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {adminExtra.map((file) => (
                <tr key={file.path}>
                  <td>{file.name}</td>
                  <td>
                    <span className="pill">{file.kind}</span>
                  </td>
                  <td>{formatBytes(file.size)}</td>
                  <td>
                    <a className="btn ghost" href={downloadUrl(file.path)}>
                      Ladda ner
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
    </div>
  );
}

export default function DownloadsPage() {
  return (
    <AuthGate>
      {(user) => <DownloadsPanel user={user} />}
    </AuthGate>
  );
}
