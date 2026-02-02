import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getReport, type Report } from "../api/reports";
import { getVisitVideos, type Video } from "../api/videos";
import { useAuth } from "../context/AuthContext";

export default function ReportPage() {
  const { visitId } = useParams();
  const nav = useNavigate();
  const { logout } = useAuth();

  function onSignOut() {
    logout();
    nav("/signin");
  }

  // Validate format: should be "childId-visitNumber" (e.g., "22-1")
  const isValid = useMemo(() => visitId ? /^\d+-\d+$/.test(visitId) : false, [visitId]);

  const [report, setReport] = useState<Report | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    if (!isValid) return;
    let cancelled = false;

    getReport(visitId!)
      .then((r) => {
        if (cancelled) return;
        setReport(r);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const err = e as { response?: { data?: { detail?: string } }; message?: string };
        setErr(err?.response?.data?.detail ?? err?.message ?? "Failed to load report");
      });

    getVisitVideos(visitId!)
      .then((v) => {
        if (cancelled) return;
        setVideos(v);
      })
      .catch(() => {
        // Ignore - videos section just won't show
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

  // Determine risk level styling
  const getRiskStyle = (bucket: string) => {
    const lower = bucket.toLowerCase();
    if (lower.includes("low")) {
      return { bg: "#dcfce7", color: "#166534" };
    } else if (lower.includes("medium") || lower.includes("moderate")) {
      return { bg: "#fef3c7", color: "#92400e" };
    } else if (lower.includes("high")) {
      return { bg: "#fef2f2", color: "#991b1b" };
    }
    return { bg: "var(--color-bg-tertiary)", color: "var(--color-text)" };
  };

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
        <div style={{ marginBottom: 32 }}>
          <h1>Assessment Report</h1>
          {report && (
            <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", marginTop: 4 }}>
              Visit {visitId?.split("-")[1]} · {report.visit.visit_date} · {report.visit.age_months} months old
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

        {!report ? (
          <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>Loading...</p>
        ) : (
          <div style={{ display: "grid", gap: 16 }}>
            {/* Risk assessment */}
            <div
              style={{
                padding: 20,
                borderRadius: 8,
                backgroundColor: getRiskStyle(report.asd_risk_bucket).bg,
                border: "1px solid transparent",
              }}
            >
              <div
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 500,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: 8,
                  color: getRiskStyle(report.asd_risk_bucket).color,
                  opacity: 0.8,
                }}
              >
                ASD Risk Level
              </div>
              <div
                style={{
                  fontSize: "1.5rem",
                  fontWeight: 600,
                  color: getRiskStyle(report.asd_risk_bucket).color,
                }}
              >
                {report.asd_risk_bucket}
              </div>
            </div>

            {/* Explanations */}
            <div
              style={{
                padding: 16,
                borderRadius: 8,
                border: "1px solid var(--color-border)",
              }}
            >
              <h2 style={{ marginBottom: 12 }}>Key observations</h2>
              {report.explanations.length === 0 ? (
                <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", margin: 0 }}>
                  No additional observations.
                </p>
              ) : (
                <ul
                  style={{
                    margin: 0,
                    paddingLeft: 20,
                    display: "grid",
                    gap: 8,
                  }}
                >
                  {report.explanations.map((x, i) => (
                    <li key={i} style={{ fontSize: "0.875rem", color: "var(--color-text-secondary)" }}>
                      {x}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* Prior visits */}
            <div
              style={{
                padding: 16,
                borderRadius: 8,
                border: "1px solid var(--color-border)",
              }}
            >
              <h2 style={{ marginBottom: 12 }}>Visit history</h2>
              {report.prior_visits.length === 0 ? (
                <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", margin: 0 }}>
                  No prior visits.
                </p>
              ) : (
                <div style={{ display: "grid", gap: 8 }}>
                  {report.prior_visits.map((v) => (
                    <div
                      key={v.id}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "8px 12px",
                        backgroundColor: "var(--color-bg-secondary)",
                        borderRadius: 6,
                        fontSize: "0.875rem",
                      }}
                    >
                      <span style={{ color: "var(--color-text-secondary)" }}>
                        {v.visit_date} · {v.age_months} months
                      </span>
                      <span style={{ fontWeight: 500 }}>{v.asd_risk_bucket}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Assisted reviews */}
            {videos.length > 0 && (
              <div
                style={{
                  padding: 16,
                  borderRadius: 8,
                  border: "1px solid var(--color-border)",
                }}
              >
                <h2 style={{ marginBottom: 12 }}>Video recordings</h2>
                <div style={{ display: "grid", gap: 8 }}>
                  {videos.map((v) => (
                    <Link
                      key={v.id}
                      to={`/videos/${v.id}/assisted-review`}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "8px 12px",
                        backgroundColor: "var(--color-bg-secondary)",
                        borderRadius: 6,
                        fontSize: "0.875rem",
                        textDecoration: "none",
                        color: "inherit",
                      }}
                    >
                      <span style={{ fontWeight: 500 }}>
                        {v.task_type.replace("_", " ")}
                      </span>
                      <span style={{ color: "var(--color-primary)", fontSize: "0.8125rem" }}>
                        View assisted review →
                      </span>
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Disclaimer */}
            <p
              style={{
                fontSize: "0.75rem",
                color: "var(--color-text-tertiary)",
                margin: 0,
                paddingTop: 8,
              }}
            >
              This tool provides decision support only and does not constitute a standalone diagnosis.
              Clinical judgment should always be applied.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
