import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getVideo, type Video } from "../api/videoMeta";
import { getVideoStaticUrl, getGuidedReviewData } from "../api/videos";
import { GuidedVideoPlayer } from "../components/guided-review";
import type { GuidedReviewData } from "../types/guidedReview";

export default function AssistedReviewPage() {
  const { videoId } = useParams();
  const navigate = useNavigate();
  const idNum = useMemo(() => Number(videoId), [videoId]);

  const [video, setVideo] = useState<Video | null>(null);
  const [guidedData, setGuidedData] = useState<GuidedReviewData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(idNum) || idNum <= 0) return;

    (async () => {
      try {
        setErr(null);
        const v = await getVideo(idNum);
        setVideo(v);
      } catch (e: unknown) {
        const err = e as { response?: { data?: { detail?: string } }; message?: string };
        setErr(err?.response?.data?.detail ?? err?.message ?? "Failed to load assisted review");
      }
    })();
  }, [idNum]);

  // Once we have video metadata, probe for actual duration and fetch real guided review data
  useEffect(() => {
    if (!video) return;

    const videoUrl = getVideoStaticUrl(video.storage_path);

    // Create a hidden video element to probe duration
    const probe = document.createElement("video");
    probe.src = videoUrl;
    probe.preload = "metadata";

    const fetchGuidedData = async (durationMs: number) => {
      try {
        const apiData = await getGuidedReviewData(video.id, durationMs);
        setGuidedData(apiData as unknown as GuidedReviewData);
      } catch (apiErr) {
        const error = apiErr as { response?: { data?: { detail?: string } }; message?: string };
        setErr(error?.response?.data?.detail ?? error?.message ?? "Failed to load guided review data");
      }
    };

    probe.onloadedmetadata = () => {
      const durationMs = Math.round(probe.duration * 1000);
      fetchGuidedData(durationMs);
    };

    probe.onerror = () => {
      // Fallback with default duration
      fetchGuidedData(10000);
    };

    return () => {
      probe.src = "";
    };
  }, [video]);

  if (!Number.isFinite(idNum) || idNum <= 0) {
    return (
      <div style={{ padding: 24, fontFamily: "system-ui" }}>
        Invalid video id
      </div>
    );
  }

  if (err) {
    return (
      <div style={{ padding: 24, color: "crimson", fontFamily: "system-ui" }}>
        {err}
      </div>
    );
  }

  if (!video || !guidedData) {
    return (
      <div
        style={{
          minHeight: "100vh",
          backgroundColor: "#f9fafb",
          fontFamily: "system-ui, -apple-system, sans-serif",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Header skeleton */}
        <header
          style={{
            padding: "12px 24px",
            backgroundColor: "#fff",
            borderBottom: "1px solid #e5e7eb",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div
              style={{
                width: 70,
                height: 32,
                backgroundColor: "#e5e7eb",
                borderRadius: 6,
              }}
            />
            <h1
              style={{
                margin: 0,
                fontSize: "1.125rem",
                fontWeight: 600,
                color: "#111827",
              }}
            >
              Assisted Review
            </h1>
          </div>
        </header>

        {/* Loading content */}
        <main
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 24,
          }}
        >
          <div
            style={{
              textAlign: "center",
              maxWidth: 400,
            }}
          >
            {/* Animated spinner */}
            <div
              style={{
                width: 48,
                height: 48,
                margin: "0 auto 20px",
                border: "3px solid #e5e7eb",
                borderTopColor: "#6366f1",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
              }}
            />
            <h2
              style={{
                fontSize: "1.125rem",
                fontWeight: 600,
                color: "#111827",
                marginBottom: 8,
              }}
            >
              Loading Analysis
            </h2>
            <p
              style={{
                fontSize: "0.875rem",
                color: "#6b7280",
                margin: 0,
                lineHeight: 1.5,
              }}
            >
              Preparing video and AI analysis data...
            </p>

            {/* Progress indicators */}
            <div
              style={{
                marginTop: 24,
                display: "flex",
                flexDirection: "column",
                gap: 12,
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 16px",
                  backgroundColor: "#fff",
                  borderRadius: 8,
                  border: "1px solid #e5e7eb",
                }}
              >
                <div
                  style={{
                    width: 20,
                    height: 20,
                    borderRadius: "50%",
                    backgroundColor: video ? "#dcfce7" : "#fef3c7",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {video ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#166534" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    <div
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        backgroundColor: "#f59e0b",
                      }}
                    />
                  )}
                </div>
                <span style={{ fontSize: "0.8125rem", color: "#374151" }}>
                  {video ? "Video loaded" : "Loading video..."}
                </span>
              </div>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 16px",
                  backgroundColor: "#fff",
                  borderRadius: 8,
                  border: "1px solid #e5e7eb",
                }}
              >
                <div
                  style={{
                    width: 20,
                    height: 20,
                    borderRadius: "50%",
                    backgroundColor: guidedData ? "#dcfce7" : video ? "#fef3c7" : "#f3f4f6",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {guidedData ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#166534" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : video ? (
                    <div
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        backgroundColor: "#f59e0b",
                      }}
                    />
                  ) : (
                    <div
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        backgroundColor: "#d1d5db",
                      }}
                    />
                  )}
                </div>
                <span style={{ fontSize: "0.8125rem", color: "#374151" }}>
                  {guidedData ? "Analysis ready" : video ? "Loading analysis..." : "Waiting..."}
                </span>
              </div>
            </div>
          </div>
        </main>

        <style>{`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#f9fafb",
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* Header */}
      <header
        style={{
          padding: "12px 24px",
          backgroundColor: "#fff",
          borderBottom: "1px solid #e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button
            onClick={() => navigate(`/visits/${video.child_id}-${video.visit_number}/report?section=assisted-review`)}
            style={{
              padding: "6px 12px",
              backgroundColor: "transparent",
              border: "1px solid #e5e7eb",
              borderRadius: 6,
              fontSize: "0.8125rem",
              color: "#374151",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            <span>←</span> Back
          </button>
          <h1
            style={{
              margin: 0,
              fontSize: "1.125rem",
              fontWeight: 600,
              color: "#111827",
            }}
          >
            Assisted Review
          </h1>
        </div>

        <div
          style={{
            fontSize: "0.8125rem",
            color: "#6b7280",
          }}
        >
          Video ID: {video.id}
        </div>
      </header>

      {/* Main Content */}
      <main style={{ padding: "24px" }}>
        <GuidedVideoPlayer
          data={guidedData}
          hasNextVideo={false}
        />
      </main>
    </div>
  );
}
