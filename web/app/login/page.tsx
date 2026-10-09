"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type User } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const user = await api<User>("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      router.replace(user.role === "admin" ? "/admin" : "/downloads");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inloggning misslyckades");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-stage">
      <form className="login-card stack" onSubmit={onSubmit}>
        <div>
          <h1 className="brand-hero">Torp</h1>
          <p className="lede">
            Logga in för plocklistor och prognosrapporter per butik.
          </p>
        </div>
        <label className="field">
          Användarnamn
          <input
            type="text"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </label>
        <label className="field">
          Lösenord
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        {error ? <p className="error">{error}</p> : null}
        <button className="btn accent" type="submit" disabled={busy}>
          {busy ? "Loggar in…" : "Logga in"}
        </button>
      </form>
    </div>
  );
}
