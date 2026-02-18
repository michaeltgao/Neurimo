import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getVisit, type Visit } from "../api/visits";
import QuestionnaireForm from "../components/QuestionnaireForm";
import FamilyHistoryForm, { type FamilyHistoryData } from "../components/FamilyHistoryForm";
import { submitQuestionnaire, getQuestionnaire, type Questionnaire } from "../api/reports";
import { useAuth } from "../context/AuthContext";
import logo from "../assets/logo.png";

// Icons
const ArrowLeftIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

const ClipboardIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
    <rect x="8" y="2" width="8" height="4" rx="1" ry="1" />
  </svg>
);

const CalendarIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
    <line x1="16" y1="2" x2="16" y2="6" />
    <line x1="8" y1="2" x2="8" y2="6" />
    <line x1="3" y1="10" x2="21" y2="10" />
  </svg>
);

const CheckCircleIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>
);

const FileTextIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </svg>
);

const ArrowRightIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12" />
    <polyline points="12 5 19 12 12 19" />
  </svg>
);

// Card wrapper style
const cardStyle: React.CSSProperties = {
  backgroundColor: "#fff",
  borderRadius: 12,
  border: "1px solid var(--color-border)",
  boxShadow: "0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06)",
  padding: 24,
};

// Progress step component
const ProgressStep = ({
  number,
  title,
  completed,
  active
}: {
  number: number;
  title: string;
  completed: boolean;
  active: boolean;
}) => (
  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
    <div style={{
      width: 32,
      height: 32,
      borderRadius: "50%",
      backgroundColor: completed ? "#dcfce7" : active ? "var(--color-text)" : "var(--color-bg-tertiary)",
      color: completed ? "#166534" : active ? "#fff" : "var(--color-text-secondary)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontWeight: 600,
      fontSize: "0.875rem",
    }}>
      {completed ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : number}
    </div>
    <span style={{
      fontSize: "0.875rem",
      fontWeight: 500,
      color: completed ? "#166534" : active ? "var(--color-text)" : "var(--color-text-secondary)",
    }}>
      {title}
    </span>
  </div>
);

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
  const [questionnaireSaved, setQuestionnaireSaved] = useState(false);
  const [familyHistorySaved, setFamilyHistorySaved] = useState(false);
  const [initialQuestionnaire, setInitialQuestionnaire] = useState<Questionnaire | undefined>(undefined);
  const [loading, setLoading] = useState(true);

  const allSaved = questionnaireSaved && familyHistorySaved;

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
        setQuestionnaireSaved(true);
        if (q.family_history && Object.keys(q.family_history).length > 0) {
          setFamilyHistorySaved(true);
        }
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
    <div style={{ minHeight: "100vh", backgroundColor: "var(--color-bg-secondary)" }}>
      {/* Header */}
      <header
        style={{
          borderBottom: "1px solid var(--color-border)",
          padding: "12px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          backgroundColor: "#fff",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button
            onClick={() => nav(-1)}
            style={{
              fontSize: "0.8125rem",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <ArrowLeftIcon />
            Back
          </button>
          <img src={logo} alt="Neurimo" style={{ height: 40 }} />
        </div>
        <button onClick={onSignOut} style={{ fontSize: "0.8125rem" }}>
          Sign Out
        </button>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 800, margin: "0 auto", padding: "32px 24px" }}>
        {/* Visit info card */}
        {visit ? (
          <div style={{
            ...cardStyle,
            marginBottom: 24,
            display: "flex",
            alignItems: "center",
            gap: 16,
          }}>
            <div style={{
              padding: 12,
              backgroundColor: "var(--color-bg-tertiary)",
              borderRadius: 12,
              color: "var(--color-text-secondary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}>
              <ClipboardIcon />
            </div>
            <div style={{ flex: 1 }}>
              <h1 style={{ fontSize: "1.5rem", marginBottom: 4 }}>Developmental Questionnaire</h1>
              <div style={{
                color: "var(--color-text-secondary)",
                fontSize: "0.875rem",
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <CalendarIcon />
                  Visit {visitId?.split("-")[1]} · {visit.visit_date}
                </span>
                <span style={{
                  padding: "2px 10px",
                  backgroundColor: "var(--color-bg-tertiary)",
                  borderRadius: 4,
                  fontSize: "0.75rem",
                  fontWeight: 500,
                }}>
                  {visit.age_months} months old
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div style={{
            ...cardStyle,
            marginBottom: 24,
            animation: "pulse 1.5s ease-in-out infinite",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{ width: 44, height: 44, borderRadius: 12, backgroundColor: "var(--color-bg-tertiary)" }} />
              <div style={{ flex: 1 }}>
                <div style={{ height: 24, width: "50%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4, marginBottom: 8 }} />
                <div style={{ height: 14, width: "30%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4 }} />
              </div>
            </div>
          </div>
        )}

        {/* Progress steps */}
        <div style={{
          ...cardStyle,
          marginBottom: 24,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 32,
        }}>
          <ProgressStep number={1} title="Questionnaire" completed={questionnaireSaved} active={!questionnaireSaved} />
          <div style={{ width: 40, height: 2, backgroundColor: questionnaireSaved ? "#bbf7d0" : "var(--color-bg-tertiary)" }} />
          <ProgressStep number={2} title="Family History" completed={familyHistorySaved} active={questionnaireSaved && !familyHistorySaved} />
          <div style={{ width: 40, height: 2, backgroundColor: allSaved ? "#bbf7d0" : "var(--color-bg-tertiary)" }} />
          <ProgressStep number={3} title="View Report" completed={false} active={allSaved} />
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
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {err}
          </div>
        )}

        {/* Developmental Questionnaire */}
        <div style={{ ...cardStyle, marginBottom: 24 }}>
          {loading ? (
            <div style={{ padding: 20, textAlign: "center" }}>
              <div style={{
                width: 32,
                height: 32,
                border: "3px solid var(--color-bg-tertiary)",
                borderTopColor: "var(--color-accent)",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
                margin: "0 auto 12px",
              }} />
              <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", margin: 0 }}>
                Loading questionnaire...
              </p>
            </div>
          ) : (
            <QuestionnaireForm
              key={`questionnaire-${visitId}`}
              initial={initialQuestionnaire}
              onSubmit={async (q) => {
                const saved = await submitQuestionnaire(visitId!, q);
                setInitialQuestionnaire(saved);
                setQuestionnaireSaved(true);
              }}
              onUnsave={() => setQuestionnaireSaved(false)}
            />
          )}
        </div>

        {/* Family History */}
        {!loading && (
          <div style={{ ...cardStyle, marginBottom: 24 }}>
            <FamilyHistoryForm
              key={`family-history-${visitId}`}
              initial={initialQuestionnaire?.family_history ?? undefined}
              onSubmit={async (familyHistory: FamilyHistoryData) => {
                const saved = await submitQuestionnaire(visitId!, {
                  ...initialQuestionnaire,
                  regression: initialQuestionnaire?.regression ?? false,
                  seizures: initialQuestionnaire?.seizures ?? false,
                  motor_delay: initialQuestionnaire?.motor_delay ?? false,
                  global_delay: initialQuestionnaire?.global_delay ?? false,
                  family_history_asd_ndd: initialQuestionnaire?.family_history_asd_ndd ?? false,
                  dysmorphic_features: initialQuestionnaire?.dysmorphic_features ?? false,
                  macrocephaly: initialQuestionnaire?.macrocephaly ?? false,
                  microcephaly: initialQuestionnaire?.microcephaly ?? false,
                  family_history: familyHistory,
                });
                setInitialQuestionnaire(saved);
                setFamilyHistorySaved(true);
              }}
              onUnsave={() => setFamilyHistorySaved(false)}
            />
          </div>
        )}

        {/* Next step card */}
        <div
          style={{
            ...cardStyle,
            backgroundColor: allSaved ? "#f0fdf4" : "#fff",
            borderColor: allSaved ? "#bbf7d0" : "var(--color-border)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{
              padding: 10,
              backgroundColor: allSaved ? "#dcfce7" : "var(--color-bg-tertiary)",
              borderRadius: 10,
              color: allSaved ? "#166534" : "var(--color-text-secondary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}>
              {allSaved ? <CheckCircleIcon /> : <FileTextIcon />}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 2 }}>
                {allSaved ? "Ready to View Report" : "Complete Both Forms"}
              </div>
              <div style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
                {allSaved
                  ? "All forms completed. View the developmental assessment report."
                  : `Complete ${!questionnaireSaved && !familyHistorySaved ? "both forms" : !questionnaireSaved ? "the questionnaire" : "family history"} to continue.`}
              </div>
            </div>
            <button
              onClick={() => nav(`/visits/${visitId}/report`)}
              disabled={!allSaved}
              className={allSaved ? "primary" : ""}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "10px 16px",
              }}
            >
              View Report
              <ArrowRightIcon />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
