import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getVisit, type Visit } from "../api/visits";
import VisitTaskCard from "../components/VisitTaskCard";
import { uploadVisitVideo, getVisitVideos, type TaskType } from "../api/videos";
import { useAuth } from "../context/AuthContext";
import logo from "../assets/logo.png";

// Icons
const ArrowLeftIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

const VideoIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="23 7 16 12 23 17 23 7" />
    <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
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

const ClipboardIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
    <rect x="8" y="2" width="8" height="4" rx="1" ry="1" />
  </svg>
);

const CheckCircleIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
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

// Section header with icon
const SectionHeader = ({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle?: string }) => (
  <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 20 }}>
    <div style={{
      padding: 10,
      backgroundColor: "var(--color-bg-tertiary)",
      borderRadius: 10,
      color: "var(--color-text-secondary)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    }}>
      {icon}
    </div>
    <div>
      <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 2 }}>{title}</h2>
      {subtitle && (
        <p style={{ color: "var(--color-text-secondary)", fontSize: "0.8125rem", margin: 0 }}>
          {subtitle}
        </p>
      )}
    </div>
  </div>
);

// Progress indicator
const ProgressBar = ({ completed, total }: { completed: number; total: number }) => {
  const percentage = (completed / total) * 100;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <div style={{
        flex: 1,
        height: 8,
        backgroundColor: "var(--color-bg-tertiary)",
        borderRadius: 4,
        overflow: "hidden",
      }}>
        <div style={{
          width: `${percentage}%`,
          height: "100%",
          backgroundColor: completed === total ? "#16a34a" : "#16a34a",
          borderRadius: 4,
          transition: "width 0.3s ease",
        }} />
      </div>
    </div>
  );
};

export default function VisitPage() {
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

  const [uploaded, setUploaded] = useState<Record<TaskType, boolean>>({
    imitation: false,
    joint_attention: false,
    free_play: false,
  });
  const [fileNames, setFileNames] = useState<Record<TaskType, string>>({
    imitation: "",
    joint_attention: "",
    free_play: "",
  });

  useEffect(() => {
    if (!isValid) return;

    let cancelled = false;

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

    // Load previously uploaded videos
    getVisitVideos(visitId!)
      .then((videos) => {
        if (cancelled) return;
        const uploadedState: Record<TaskType, boolean> = {
          imitation: false,
          joint_attention: false,
          free_play: false,
        };
        const fileNameState: Record<TaskType, string> = {
          imitation: "",
          joint_attention: "",
          free_play: "",
        };
        for (const video of videos) {
          uploadedState[video.task_type] = true;
          // Extract filename from storage_path
          const pathParts = video.storage_path.split("/");
          fileNameState[video.task_type] = pathParts[pathParts.length - 1];
        }
        setUploaded(uploadedState);
        setFileNames(fileNameState);
      })
      .catch(() => {
        // Ignore errors - just means no videos uploaded yet
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

  async function handleUpload(task: TaskType, file: File) {
    try {
      await uploadVisitVideo(visitId!, task, file);
      setUploaded((prev) => ({ ...prev, [task]: true }));
      setFileNames((prev) => ({ ...prev, [task]: file.name }));
      setErr(null);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Upload failed");
    }
  }

  const allUploaded = uploaded.imitation && uploaded.joint_attention && uploaded.free_play;
  const uploadedCount = [uploaded.imitation, uploaded.joint_attention, uploaded.free_play].filter(Boolean).length;

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
            onClick={() => nav(`/children/${visitId?.split("-")[0]}`)}
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
              <h1 style={{ fontSize: "1.5rem", marginBottom: 4 }}>Visit {visitId?.split("-")[1]}</h1>
              <div style={{
                color: "var(--color-text-secondary)",
                fontSize: "0.875rem",
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <CalendarIcon />
                  {visit.visit_date}
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
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              color: allUploaded ? "var(--color-success)" : "var(--color-text-secondary)",
            }}>
              {allUploaded && <CheckCircleIcon />}
              <span style={{ fontSize: "0.8125rem", fontWeight: 500 }}>
                {uploadedCount}/3 uploaded
              </span>
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
                <div style={{ height: 24, width: "30%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4, marginBottom: 8 }} />
                <div style={{ height: 14, width: "20%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4 }} />
              </div>
            </div>
          </div>
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

        {/* Video tasks card */}
        <div style={{ ...cardStyle, marginBottom: 24 }}>
          <SectionHeader
            icon={<VideoIcon />}
            title="Video Tasks"
            subtitle="Upload behavioral assessment videos for analysis"
          />

          {/* Progress bar */}
          <div style={{ marginBottom: 20 }}>
            <ProgressBar completed={uploadedCount} total={3} />
          </div>

          <div style={{ display: "grid", gap: 16 }}>
            <VisitTaskCard
              title="Task A — Imitation"
              instructions="Upload a 30-60s clip of parents performing gestures and encouraging child to imitate."
              status={uploaded.imitation ? "uploaded" : "missing"}
              fileName={fileNames.imitation}
              onUpload={(file) => handleUpload("imitation", file)}
            />

            <VisitTaskCard
              title="Task B — Joint Attention"
              instructions="Upload a 30-60s clip where the parent points to an object and directs attention."
              status={uploaded.joint_attention ? "uploaded" : "missing"}
              fileName={fileNames.joint_attention}
              onUpload={(file) => handleUpload("joint_attention", file)}
            />

            <VisitTaskCard
              title="Task C — Free Social Play"
              instructions="Upload a 30-60s clip of natural play."
              status={uploaded.free_play ? "uploaded" : "missing"}
              fileName={fileNames.free_play}
              onUpload={(file) => handleUpload("free_play", file)}
            />
          </div>
        </div>

        {/* Next step card */}
        <div
          style={{
            ...cardStyle,
            backgroundColor: allUploaded ? "#f0fdf4" : "#fff",
            borderColor: allUploaded ? "#bbf7d0" : "var(--color-border)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{
              padding: 10,
              backgroundColor: allUploaded ? "#dcfce7" : "var(--color-bg-tertiary)",
              borderRadius: 10,
              color: allUploaded ? "#166534" : "var(--color-text-secondary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}>
              {allUploaded ? <CheckCircleIcon /> : <ClipboardIcon />}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 2 }}>
                {allUploaded ? "Ready for Questionnaire" : "Complete Video Uploads"}
              </div>
              <div style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
                {allUploaded
                  ? "All videos uploaded successfully. Proceed to the developmental questionnaire."
                  : `Upload ${3 - uploadedCount} more video${3 - uploadedCount !== 1 ? 's' : ''} to continue.`}
              </div>
            </div>
            <button
              onClick={() => nav(`/visits/${visitId}/questionnaire`)}
              disabled={!allUploaded}
              className={allUploaded ? "primary" : ""}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "10px 16px",
              }}
            >
              Continue
              <ArrowRightIcon />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
