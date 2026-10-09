"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { AuthGate } from "@/components/AuthGate";
import { api, type JobStatus, type User } from "@/lib/api";

function AdminPanel() {
  const [users, setUsers] = useState<User[]>([]);
  const [stores, setStores] = useState<string[]>([]);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const [newUser, setNewUser] = useState({
    username: "",
    password: "",
    role: "user" as "admin" | "user",
  });
  const [selectedStores, setSelectedStores] = useState<Record<number, string[]>>(
    {},
  );
  const [passwordEdits, setPasswordEdits] = useState<Record<number, string>>({});

  const load = useCallback(async () => {
    const [u, s, j] = await Promise.all([
      api<User[]>("/api/admin/users"),
      api<{ name: string }[]>("/api/admin/stores"),
      api<JobStatus>("/api/admin/jobs/status"),
    ]);
    setUsers(u);
    setStores(s.map((x) => x.name));
    setJob(j);
    const map: Record<number, string[]> = {};
    for (const user of u) map[user.id] = [...user.stores];
    setSelectedStores(map);
  }, []);

  useEffect(() => {
    load().catch((err) =>
      setError(err instanceof Error ? err.message : "Kunde inte ladda admin"),
    );
  }, [load]);

  useEffect(() => {
    if (!job?.running) return;
    const id = window.setInterval(() => {
      api<JobStatus>("/api/admin/jobs/status")
        .then(setJob)
        .catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(id);
  }, [job?.running]);

  const storeOptions = useMemo(() => stores, [stores]);

  async function createUser(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setMessage(null);
    try {
      await api<User>("/api/admin/users", {
        method: "POST",
        body: JSON.stringify({
          ...newUser,
          stores: [],
        }),
      });
      setNewUser({ username: "", password: "", role: "user" });
      setMessage("Användare skapad");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kunde inte skapa användare");
    }
  }

  async function saveStores(userId: number) {
    setError(null);
    setMessage(null);
    try {
      await api(`/api/admin/users/${userId}/stores`, {
        method: "PUT",
        body: JSON.stringify({ stores: selectedStores[userId] || [] }),
      });
      setMessage("Butiksåtkomst sparad");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kunde inte spara butiker");
    }
  }

  async function patchUser(
    userId: number,
    body: { password?: string; is_active?: boolean; role?: "admin" | "user" },
  ) {
    setError(null);
    setMessage(null);
    try {
      await api(`/api/admin/users/${userId}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      });
      if (body.password) {
        setPasswordEdits((prev) => ({ ...prev, [userId]: "" }));
      }
      setMessage("Användare uppdaterad");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kunde inte uppdatera");
    }
  }

  async function runJob() {
    setError(null);
    setMessage(null);
    try {
      const status = await api<JobStatus>("/api/admin/jobs/run", {
        method: "POST",
      });
      setJob(status);
      setMessage("Rapportkörning startad");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kunde inte starta jobb");
    }
  }

  function toggleStore(userId: number, store: string) {
    setSelectedStores((prev) => {
      const current = new Set(prev[userId] || []);
      if (current.has(store)) current.delete(store);
      else current.add(store);
      return { ...prev, [userId]: Array.from(current).sort() };
    });
  }

  return (
    <div className="stack">
      <section className="panel stack">
        <div>
          <h1>Admin</h1>
          <p className="muted">
            Skapa användare, tilldela butiker och kör prognospipelinen.
          </p>
        </div>
        {error ? <p className="error">{error}</p> : null}
        {message ? <p className="muted">{message}</p> : null}
      </section>

      <section className="panel stack">
        <h2>Kör rapporter</h2>
        <div className="row">
          <button
            type="button"
            className="btn accent"
            onClick={runJob}
            disabled={!!job?.running}
          >
            {job?.running ? "Kör…" : "Starta nedladdning + rapporter"}
          </button>
          <span className={`pill ${job?.running ? "warn" : "ok"}`}>
            {job?.running ? "Pågår" : "Redo"}
          </span>
          {job?.exit_code != null && !job.running ? (
            <span className="muted">Senaste exit: {job.exit_code}</span>
          ) : null}
        </div>
        {job?.last_error ? <p className="error">{job.last_error}</p> : null}
        {job?.log_tail ? <pre className="log-box">{job.log_tail}</pre> : null}
      </section>

      <section className="panel stack">
        <h2>Ny användare</h2>
        <form className="stack" onSubmit={createUser}>
          <div className="row" style={{ alignItems: "stretch" }}>
            <label className="field" style={{ flex: 1, minWidth: 160 }}>
              Användarnamn
              <input
                value={newUser.username}
                onChange={(e) =>
                  setNewUser((p) => ({ ...p, username: e.target.value }))
                }
                required
              />
            </label>
            <label className="field" style={{ flex: 1, minWidth: 160 }}>
              Lösenord
              <input
                type="password"
                value={newUser.password}
                onChange={(e) =>
                  setNewUser((p) => ({ ...p, password: e.target.value }))
                }
                required
                minLength={4}
              />
            </label>
            <label className="field" style={{ minWidth: 140 }}>
              Roll
              <select
                value={newUser.role}
                onChange={(e) =>
                  setNewUser((p) => ({
                    ...p,
                    role: e.target.value as "admin" | "user",
                  }))
                }
              >
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select>
            </label>
          </div>
          <button className="btn" type="submit">
            Skapa
          </button>
        </form>
      </section>

      <section className="panel stack">
        <h2>Användare & butiker</h2>
        <div className="stack">
          {users.map((user) => (
            <article key={user.id} className="store-card stack">
              <div className="row">
                <h3 style={{ margin: 0 }}>{user.username}</h3>
                <span className="pill">{user.role}</span>
                <span className={`pill ${user.is_active ? "ok" : "warn"}`}>
                  {user.is_active ? "aktiv" : "inaktiv"}
                </span>
              </div>

              {user.role === "user" ? (
                <>
                  <div className="checks">
                    {storeOptions.map((store) => (
                      <label key={store}>
                        <input
                          type="checkbox"
                          checked={(selectedStores[user.id] || []).includes(
                            store,
                          )}
                          onChange={() => toggleStore(user.id, store)}
                        />
                        {store}
                      </label>
                    ))}
                  </div>
                  <button
                    type="button"
                    className="btn ghost"
                    onClick={() => saveStores(user.id)}
                  >
                    Spara butiker
                  </button>
                </>
              ) : (
                <p className="muted">Admin har åtkomst till alla butiker.</p>
              )}

              <div className="row">
                <input
                  type="password"
                  placeholder="Nytt lösenord"
                  value={passwordEdits[user.id] || ""}
                  onChange={(e) =>
                    setPasswordEdits((p) => ({
                      ...p,
                      [user.id]: e.target.value,
                    }))
                  }
                  style={{ maxWidth: 220 }}
                />
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => {
                    const pw = passwordEdits[user.id];
                    if (!pw || pw.length < 4) {
                      setError("Lösenord måste vara minst 4 tecken");
                      return;
                    }
                    void patchUser(user.id, { password: pw });
                  }}
                >
                  Byt lösenord
                </button>
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() =>
                    void patchUser(user.id, { is_active: !user.is_active })
                  }
                >
                  {user.is_active ? "Inaktivera" : "Aktivera"}
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

export default function AdminPage() {
  return (
    <AuthGate adminOnly>
      {() => <AdminPanel />}
    </AuthGate>
  );
}
