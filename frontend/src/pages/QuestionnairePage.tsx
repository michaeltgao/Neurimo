import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getVisit, type Visit } from "../api/visits";
import QuestionnaireForm from "../components/QuestionnaireForm";
import { submitQuestionnaire, getQuestionnaire, type Questionnaire } from "../api/reports";
import { useAuth } from "../context/AuthContext";

export default function QuestionnairePage() {
  const { visitId } = useParams();
  const nav = useNavigate();
  const { logout } = useAuth();

  function onSignOut() {
    logout();
    nav("/signin");
  }

  // Validate format: should be "childId-visitNumber" (e.g., "22-1")
  const isValid = useMemo(() => visitId ? /^\d+-\d+$/.test(visitId) : false, [visitId]);

  const [visit, setVisit] = useState<Visit | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [initialQuestionnaire, setInitialQuestionnaire] = useState<Questionnaire | undefined>(undefined);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isValid) return;

    let cancelled = false;
    setLoading(true);

    getVisit(visitId!)
      .then((v) => {
        if (cancelled) return;
        setVisit(v);
        setErr(null);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const err = e as { response?: { data?: { detail?: string } }; message?: string };
        setErr(err?.response?.data?.detail ?? err?.message ?? "Failed to load visit");
      });

    // Load previously saved questionnaire
    getQuestionnaire(visitId!)
      .then((q) => {
        if (cancelled) return;
        setInitialQuestionnaire(q);
        setSaved(true); // Already saved previously
      })
      .catch(() => {
        // Ignore errors - just means no questionnaire saved yet
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [visitId, isValid]);

  if (!isValid) {
    return (
      <div style={{ padding: 24, color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>
        Invalid visit ID
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
          <button onClick={() => nav(-1)} style={{ fontSize: "0.8125rem" }}>
            Back
          </button>
          <h1 style={{ fontSize: "1rem", fontWeight: 600 }}>Neurimo</h1>
        </div>
        <button onClick={onSignOut} style={{ fontSize: "0.8125rem" }}>
          Sign Out
        </button>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 640, margin: "0 auto", padding: "32px 24px" }}>
        {/* Visit info */}
        <div style={{ marginBottom: 32 }}>
          <h1>{visit ? "Clinical Questionnaire" : "Loading..."}</h1>
          {visit && (
            <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", marginTop: 4 }}>
              Visit {visitId?.split("-")[1]} · {visit.visit_date} · {visit.age_months} months old
            </p>
          )}
        </div>

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

        <div style={{ marginBottom: 24 }}>
          {loading ? (
            <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>Loading...</p>
          ) : (
            <QuestionnaireForm
              key={initialQuestionnaire ? "loaded" : "new"}
              initial={initialQuestionnaire}
              onSubmit={async (q) => {
                await submitQuestionnaire(visitId!, q);
                setSaved(true);
              }}
            />
          )}
        </div>

        {/* Next step */}
        <div
          style={{
            padding: 16,
            backgroundColor: "var(--color-bg-secondary)",
            borderRadius: 8,
            border: "1px solid var(--color-border)",
          }}
        >
          <div style={{ fontSize: "0.875rem", fontWeight: 500, marginBottom: 4 }}>Next step</div>
          <div style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)", marginBottom: 12 }}>
            {saved ? "Questionnaire saved. View the assessment report." : "Save the questionnaire to continue."}
          </div>
          <button
            onClick={() => nav(`/visits/${visitId}/report`)}
            disabled={!saved}
            className={saved ? "primary" : ""}
          >
            View Report
          </button>
        </div>
      </main>
    </div>
  );
}
