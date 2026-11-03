"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
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
  const path = usePathname();

  function itemClass(href: string) {
    const active = href === "/" ? path === "/" : path.startsWith(href);
    return active ? "nav-active" : undefined;
  }

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
          <Link href="/" className={itemClass("/")}>
            Hem
          </Link>
          {user.role === "admin" ? (
            <Link href="/admin" className={itemClass("/admin")}>
              Admin
            </Link>
          ) : null}
          <Link href="/charts" className={itemClass("/charts")}>
            Prognoser
          </Link>
          <Link href="/analysis" className={itemClass("/analysis")}>
            Analysverktyg
          </Link>
          <Link href="/downloads" className={itemClass("/downloads")}>
            Filer
          </Link>
          <span className="pill">{user.username}</span>
          <button type="button" className="linkish" onClick={logout}>
            Logga ut
          </button>
        </nav>
      </header>
      <main className={path.startsWith("/analysis") ? "main main-wide" : "main"}>
        {children}
      </main>
    </div>
  );
}
