"use client";

import Link from "next/link";
import { AuthGate } from "@/components/AuthGate";

export default function HomePage() {
  return (
    <AuthGate>
      {(user) => (
        <section className="panel stack">
          <div>
            <h1>Hej {user.username}</h1>
            <p className="muted">
              {user.role === "admin"
                ? "Administrera användare, butiksåtkomst och kör rapporter."
                : "Ladda ner plocklistor för dina butiker."}
            </p>
          </div>
          <div className="row">
            <Link className="btn accent" href="/analysis">
              Analysverktyg
            </Link>
            <Link className="btn" href="/charts">
              Prognoser
            </Link>
            <Link className="btn" href="/downloads">
              Öppna filer
            </Link>
            {user.role === "admin" ? (
              <Link className="btn ghost" href="/admin">
                Admin
              </Link>
            ) : null}
          </div>
          {user.role !== "admin" && user.stores.length > 0 ? (
            <p className="muted">
              Dina butiker: {user.stores.join(", ")}
            </p>
          ) : null}
        </section>
      )}
    </AuthGate>
  );
}
