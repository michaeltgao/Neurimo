import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { createVisit, listVisits, type Visit } from "../api/visits";
import { getChild, type Child } from "../api/children";
import { useAuth } from "../context/AuthContext";

export default function ChildDetailPage() {
  const { childId } = useParams();
  const childIdNum = Number(childId);
  const nav = useNavigate();
  const { logout } = useAuth();

  function onSignOut() {
    logout();
    nav("/signin");
  }

  const [child, setChild] = useState<Child | null>(null);
  const [visits, setVisits] = useState<Visit[]>([]);
  const [err, setErr] = useState<string | null>(null);

  // create visit form
  const [visitDate, setVisitDate] = useState("");
  const [ageMonths, setAgeMonths] = useState<number>(12);

  const validChildId = useMemo(() => Number.isFinite(childIdNum) && childIdNum > 0, [childIdNum]);

  useEffect(() => {
    if (!validChildId) return;

    let cancelled = false;

    Promise.all([getChild(childIdNum), listVisits(childIdNum)])
      .then(([found, v]) => {
        if (cancelled) return;
        setErr(null);
        setChild(found);
        setVisits(v);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const err = e as { response?: { data?: { detail?: string } }; message?: string };
        setErr(err?.response?.data?.detail ?? err?.message ?? "Failed to load child");
      });

    return () => {
      cancelled = true;
    };
  }, [childIdNum, validChildId]);

  async function onCreateVisit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);

    if (!visitDate) {
      setErr("visit_date is required");
      return;
    }

    try {
      const created = await createVisit(childIdNum, { visit_date: visitDate, age_months: ageMonths });
      setVisitDate("");
      setAgeMonths(12);
      // go to visit page
      nav(`/visits/${created.child_id}-${created.visit_number}`);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Create visit failed");
    }
  }

  if (!validChildId) {
    return (
      <div style={{ padding: 24, color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>
        Invalid patient ID
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#fff" }}>
      {/* Header */}
      <header
        style={{
          borderBottom: "1px solid var(--color-border)",
          padding: "12px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button onClick={() => nav("/children")} style={{ fontSize: "0.8125rem" }}>
            Back
          </button>
          <h1 style={{ fontSize: "1rem", fontWeight: 600 }}>Neurimo</h1>
        </div>
        <button onClick={onSignOut} style={{ fontSize: "0.8125rem" }}>
          Sign Out
        </button>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 960, margin: "0 auto", padding: "32px 24px" }}>
        {/* Patient info */}
        {child ? (
          <div style={{ marginBottom: 32 }}>
            <h1>{child.pseudo_id}</h1>
            <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", marginTop: 4 }}>
              Born {child.birthdate} · {child.sex === "M" ? "Male" : child.sex === "F" ? "Female" : child.sex}
            </p>
          </div>
        ) : (
          <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>Loading...</p>
        )}

        {err && (
          <div
            style={{
              padding: "12px 16px",
              marginBottom: 24,
              backgroundColor: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: 8,
              color: "var(--color-error)",
              fontSize: "0.875rem",
            }}
          >
            {err}
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
          {/* Create visit */}
          <div>
            <h2 style={{ marginBottom: 16 }}>New visit</h2>
            <form onSubmit={onCreateVisit} style={{ display: "grid", gap: 16 }}>
              <div style={{ display: "grid", gap: 6 }}>
                <label htmlFor="visitDate">Visit date</label>
                <input
                  id="visitDate"
                  type="date"
                  value={visitDate}
                  onChange={(e) => setVisitDate(e.target.value)}
                />
              </div>

              <div style={{ display: "grid", gap: 6 }}>
                <label htmlFor="ageMonths">Age (months)</label>
                <input
                  id="ageMonths"
                  type="number"
                  value={ageMonths}
                  min={6}
                  max={60}
                  onChange={(e) => setAgeMonths(Number(e.target.value))}
                />
              </div>

              <button type="submit" className="primary" style={{ marginTop: 8 }}>
                Create visit
              </button>
            </form>
          </div>

          {/* Visits list */}
          <div>
            <h2 style={{ marginBottom: 16 }}>Visit history</h2>
            {visits.length === 0 ? (
              <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>No visits yet.</p>
            ) : (
              <div style={{ display: "grid", gap: 8 }}>
                {visits.map((v) => (
                  <div
                    key={v.id}
                    onClick={() => nav(`/visits/${v.child_id}-${v.visit_number}`)}
                    style={{
                      padding: "12px 16px",
                      border: "1px solid var(--color-border)",
                      borderRadius: 8,
                      cursor: "pointer",
                      transition: "background-color 0.15s ease",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--color-bg-secondary)")}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                  >
                    <div style={{ fontWeight: 500, fontSize: "0.875rem" }}>
                      Visit {v.visit_number}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)", marginTop: 2 }}>
                      {v.visit_date} · {v.age_months} months
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
