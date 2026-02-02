import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getVisit, type Visit } from "../api/visits";
import VisitTaskCard from "../components/VisitTaskCard";
import { uploadVisitVideo, getVisitVideos, type TaskType } from "../api/videos";
import { useAuth } from "../context/AuthContext";


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
      <main style={{ maxWidth: 720, margin: "0 auto", padding: "32px 24px" }}>
        {/* Visit info */}
        <div style={{ marginBottom: 32 }}>
          <h1>{visit ? `Visit ${visitId?.split("-")[1]}` : "Loading..."}</h1>
          {visit && (
            <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", marginTop: 4 }}>
              {visit.visit_date} · {visit.age_months} months old
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
          <h2 style={{ marginBottom: 16 }}>Video tasks</h2>
          <div style={{ display: "grid", gap: 12 }}>
            <VisitTaskCard
              title="Task A — Imitation"
              instructions="Upload a 30-60s clip of parents performing gestures, such as clapping and arm-raising, and encouraging child to imitate."
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
              instructions="Upload a 60-90s clip of natural play (peekaboo / talking / interaction)."
              status={uploaded.free_play ? "uploaded" : "missing"}
              fileName={fileNames.free_play}
              onUpload={(file) => handleUpload("free_play", file)}
            />
          </div>
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
            {allUploaded ? "All videos uploaded. Continue to questionnaire." : "Upload all three videos to continue."}
          </div>
          <button
            onClick={() => nav(`/visits/${visitId}/questionnaire`)}
            disabled={!allUploaded}
            className={allUploaded ? "primary" : ""}
          >
            Continue to Questionnaire
          </button>
        </div>
      </main>
    </div>
  );
}
