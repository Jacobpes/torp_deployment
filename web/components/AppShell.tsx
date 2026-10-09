"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import type { User } from "@/lib/api";
import { api } from "@/lib/api";

export function AppShell({
  user,
  children,
}: {
  user: User;
  children: React.ReactNode;
}) {
  const router = useRouter();

  async function logout() {
    await api("/api/auth/logout", { method: "POST" });
    router.replace("/login");
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          Torp
          <span>{user.role === "admin" ? "admin" : "plocklistor"}</span>
        </div>
        <nav className="nav">
          <Link href="/">Hem</Link>
          {user.role === "admin" ? <Link href="/admin">Admin</Link> : null}
          <Link href="/charts">Grafer</Link>
          <Link href="/downloads">Filer</Link>
          <span className="pill">{user.username}</span>
          <button type="button" className="linkish" onClick={logout}>
            Logga ut
          </button>
        </nav>
      </header>
      <main className="main">{children}</main>
    </div>
  );
}
