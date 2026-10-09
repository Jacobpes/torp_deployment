"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { api, type User } from "@/lib/api";

export function AuthGate({
  children,
  adminOnly = false,
}: {
  children: (user: User) => React.ReactNode;
  adminOnly?: boolean;
}) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    let cancelled = false;
    api<User>("/api/auth/me")
      .then((u) => {
        if (cancelled) return;
        if (adminOnly && u.role !== "admin") {
          router.replace("/");
          return;
        }
        setUser(u);
      })
      .catch(() => {
        if (!cancelled) router.replace("/login");
      });
    return () => {
      cancelled = true;
    };
  }, [adminOnly, router]);

  if (!user) {
    return (
      <div className="login-stage">
        <p className="muted">Laddar…</p>
      </div>
    );
  }

  return <AppShell user={user}>{children(user)}</AppShell>;
}
