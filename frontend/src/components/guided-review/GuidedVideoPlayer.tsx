import { useRef, useState, useCallback, useEffect } from "react";
import type {
  GuidedReviewData,
  PlaybackMode,
  PauseCard as PauseCardType,
} from "../../types/guidedReview";
import PauseCard from "./PauseCard";
import SummaryCard from "./SummaryCard";
import ReviewTimeline from "./ReviewTimeline";
import VideoOverlayLayer from "./VideoOverlayLayer";

type Props = {
  data: GuidedReviewData;
  onNextVideo?: () => void;
  hasNextVideo?: boolean;
};

export default function GuidedVideoPlayer({
  data,
  onNextVideo,
  hasNextVideo,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const lastCheckedTimeRef = useRef<number>(0);

  // Playback state
  const [mode, setMode] = useState<PlaybackMode>("idle");
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [currentFlagIndex, setCurrentFlagIndex] = useState<number | null>(null);
  const [flagsVisited, setFlagsVisited] = useState<Set<string>>(new Set());

  // UI state
  const [activePauseCard, setActivePauseCard] = useState<PauseCardType | null>(null);

  // Overlay toggles
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showHeadGaze, setShowHeadGaze] = useState(true);
  const [showParent, setShowParent] = useState(true);

  // Fullscreen state
  const [isFullscreen, setIsFullscreen] = useState(false);

  const { flaggedMoments, durationMs, videoUrl } = data;

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  // Toggle fullscreen
  const handleToggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => {
        console.error("Failed to enter fullscreen:", err);
      });
    } else {
      document.exitFullscreen();
    }
  }, []);

  // Handle time updates
  const handleTimeUpdate = useCallback(() => {
    if (!videoRef.current || mode !== "playing") return;

    const currentMs = videoRef.current.currentTime * 1000;
    setCurrentTimeMs(currentMs);

    // Check if we've hit a flag
    const hitFlag = flaggedMoments.find(
      (f) =>
        !flagsVisited.has(f.id) &&
        f.pauseAtMs <= currentMs &&
        f.pauseAtMs > lastCheckedTimeRef.current
    );

    if (hitFlag) {
      // Pause at this flag
      videoRef.current.pause();
      setMode("paused_at_flag");
      setCurrentFlagIndex(flaggedMoments.indexOf(hitFlag));
      setActivePauseCard(hitFlag.pauseCard);
      setFlagsVisited((prev) => new Set([...prev, hitFlag.id]));
    }

    lastCheckedTimeRef.current = currentMs;
  }, [mode, flaggedMoments, flagsVisited]);

  // Handle video ended
  const handleEnded = useCallback(() => {
    setMode("complete");
    setActivePauseCard(null);
  }, []);

  // Play/Pause controls
  const handlePlay = useCallback(() => {
    if (!videoRef.current) return;
    videoRef.current.play();
    setMode("playing");
  }, []);

  const handleContinue = useCallback(() => {
    if (!videoRef.current) return;
    setActivePauseCard(null);
    videoRef.current.play();
    setMode("playing");
  }, []);

  const handleReplay = useCallback(() => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = 0;
    lastCheckedTimeRef.current = 0;
    setCurrentTimeMs(0);
    setFlagsVisited(new Set());
    setCurrentFlagIndex(null);
    setActivePauseCard(null);
    setMode("idle");
  }, []);

  const handleSkipToEnd = useCallback(() => {
    setMode("complete");
    setActivePauseCard(null);
    if (videoRef.current) {
      videoRef.current.pause();
    }
  }, []);

  const handleSeek = useCallback((timeMs: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = timeMs / 1000;
    setCurrentTimeMs(timeMs);
    lastCheckedTimeRef.current = timeMs;
  }, []);

  // Get current flag description for bottom bar
  const currentFlagDescription = currentFlagIndex !== null
    ? `${flaggedMoments[currentFlagIndex].expected.type} → ${flaggedMoments[currentFlagIndex].observed.status.replace("_", " ")}`
    : null;

  return (
    <div
      ref={containerRef}
      style={{
        fontFamily: "system-ui, -apple-system, sans-serif",
        maxWidth: isFullscreen ? "100%" : 900,
        margin: "0 auto",
        backgroundColor: isFullscreen ? "#000" : "transparent",
        height: isFullscreen ? "100vh" : "auto",
        display: isFullscreen ? "flex" : "block",
        flexDirection: "column",
      }}
    >
      {/* Top Bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 16px",
          backgroundColor: "#f9fafb",
          borderRadius: "12px 12px 0 0",
          borderBottom: "1px solid #e5e7eb",
        }}
      >
        {/* Task + Age */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span
            style={{
              padding: "4px 12px",
              backgroundColor: "#eff6ff",
              color: "#1d4ed8",
              borderRadius: 6,
              fontSize: "0.8125rem",
              fontWeight: 500,
            }}
          >
            {data.taskTypeDisplay}
          </span>
          <span style={{ fontSize: "0.8125rem", color: "#6b7280" }}>
            {data.ageBucket}
          </span>
        </div>

        {/* Quality Badge */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              padding: "4px 10px",
              backgroundColor:
                data.quality.overallQuality === "good" ? "#dcfce7" :
                data.quality.overallQuality === "medium" ? "#fef3c7" :
                "#fee2e2",
              color:
                data.quality.overallQuality === "good" ? "#166534" :
                data.quality.overallQuality === "medium" ? "#92400e" :
                "#991b1b",
              borderRadius: 6,
              fontSize: "0.75rem",
              fontWeight: 500,
              textTransform: "capitalize",
            }}
          >
            Tracking: {data.quality.overallQuality}
          </span>
          <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>
            Face: {data.quality.faceVisibilityPct}%
          </span>
        </div>
      </div>

      {/* Key Drivers */}
      {data.keyDrivers.length > 0 && (
        <div
          style={{
            padding: "8px 16px",
            backgroundColor: "#fff",
            borderBottom: "1px solid #e5e7eb",
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexWrap: "wrap",
          }}
        >
          <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>
            Key drivers:
          </span>
          {data.keyDrivers.map((driver) => {
            // Use color from backend (scales with ML risk level) or fallback to severity-based
            const textColor = driver.color ||
              (driver.severity === "high" ? "#991b1b" :
               driver.severity === "medium" ? "#92400e" :
               "#6b7280");
            // Derive background from text color (lighter version)
            const bgColor =
              textColor === "#6b7280" ? "#f3f4f6" :     // gray -> light gray bg
              textColor.includes("dc2626") || textColor.includes("991b1b") ? "#fee2e2" :  // red -> light red bg
              textColor.includes("d97706") || textColor.includes("92400e") ? "#fef3c7" :  // orange -> light orange bg
              textColor.includes("2563eb") || textColor.includes("1d4ed8") ? "#dbeafe" :  // blue -> light blue bg
              "#f3f4f6";  // default light gray
            return (
              <span
                key={driver.id}
                style={{
                  padding: "2px 8px",
                  backgroundColor: bgColor,
                  color: textColor,
                  borderRadius: 4,
                  fontSize: "0.75rem",
                  fontWeight: 500,
                }}
              >
                {driver.label}
                {driver.count && driver.count > 1 && ` ×${driver.count}`}
              </span>
            );
          })}
        </div>
      )}

      {/* Video Container */}
      <div
        style={{
          position: "relative",
          backgroundColor: "#0a0a0a",
          aspectRatio: isFullscreen ? undefined : "16/9",
          flex: isFullscreen ? 1 : undefined,
          minHeight: isFullscreen ? 0 : undefined,
        }}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          muted={data.taskType !== "joint_attention"}
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleEnded}
          onPlay={() => setMode("playing")}
          onPause={() => {
            if (mode === "playing") {
              // Only set to idle if not paused at flag
              // (user manually paused)
            }
          }}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            display: "block",
          }}
        />

        {/* Video Annotation Overlay Layer */}
        <VideoOverlayLayer
          videoRef={videoRef}
          videoId={data.videoId}
          durationMs={durationMs}
          currentTimeMs={currentTimeMs}
          showSkeleton={showSkeleton}
          showHeadGaze={showHeadGaze}
          showParent={showParent}
        />

        {/* Overlay: Idle state - show play button */}
        {mode === "idle" && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "rgba(0,0,0,0.3)",
            }}
          >
            <button
              onClick={handlePlay}
              style={{
                width: 80,
                height: 80,
                borderRadius: "50%",
                backgroundColor: "#fff",
                border: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
                transition: "transform 0.15s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "scale(1.05)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "scale(1)";
              }}
            >
              <svg width="32" height="32" viewBox="0 0 24 24" fill="#0a0a0a">
                <polygon points="6,4 20,12 6,20" />
              </svg>
            </button>
          </div>
        )}

        {/* Pause Card Overlay */}
        {mode === "paused_at_flag" && activePauseCard && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              backgroundColor: "rgba(0,0,0,0.5)",
            }}
          >
            <PauseCard card={activePauseCard} riskBucket={data.riskBucket} onContinue={handleContinue} />
          </div>
        )}

        {/* Summary Card Overlay */}
        {mode === "complete" && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              backgroundColor: "rgba(0,0,0,0.5)",
            }}
          >
            <SummaryCard
              summary={data.summary}
              quality={data.quality}
              riskBucket={data.riskBucket}
              onReplay={handleReplay}
              onClose={() => setMode("idle")}
              onNext={onNextVideo}
              hasNextVideo={hasNextVideo}
            />
          </div>
        )}
      </div>

      {/* Timeline */}
      <div
        style={{
          padding: "8px 16px",
          backgroundColor: "#fff",
          borderTop: "1px solid #e5e7eb",
        }}
      >
        <ReviewTimeline
          durationMs={durationMs}
          currentTimeMs={currentTimeMs}
          flaggedMoments={flaggedMoments}
          currentFlagIndex={currentFlagIndex}
          onSeek={handleSeek}
        />
      </div>

      {/* Bottom Controls */}
      <div
        style={{
          padding: "12px 16px",
          backgroundColor: "#f9fafb",
          borderRadius: "0 0 12px 12px",
          borderTop: "1px solid #e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        {/* Left: Flag info */}
        <div style={{ fontSize: "0.8125rem", color: "#6b7280" }}>
          {currentFlagIndex !== null ? (
            <>
              <strong style={{ color: "#374151" }}>
                Flag {currentFlagIndex + 1} of {flaggedMoments.length}:
              </strong>{" "}
              {currentFlagDescription}
            </>
          ) : flaggedMoments.length > 0 ? (
            <span>{flaggedMoments.length} flagged moments</span>
          ) : (
            <span style={{ color: "#22c55e" }}>No flagged moments</span>
          )}
        </div>

        {/* Right: Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {/* Overlay toggles */}
          <div
            style={{
              display: "flex",
              gap: 4,
              marginRight: 8,
              paddingRight: 8,
              borderRight: "1px solid #e5e7eb",
            }}
          >
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 2,
                fontSize: "0.75rem",
                color: "#6b7280",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={showSkeleton}
                onChange={(e) => setShowSkeleton(e.target.checked)}
                style={{ width: 14, height: 14 }}
              />
              Skeleton
            </label>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 2,
                fontSize: "0.75rem",
                color: "#6b7280",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={showHeadGaze}
                onChange={(e) => setShowHeadGaze(e.target.checked)}
                style={{ width: 14, height: 14 }}
              />
              Head/Gaze
            </label>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 2,
                fontSize: "0.75rem",
                color: "#6b7280",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={showParent}
                onChange={(e) => setShowParent(e.target.checked)}
                style={{ width: 14, height: 14 }}
              />
              Parent
            </label>
          </div>

          {/* Restart button */}
          <button
            onClick={handleReplay}
            style={{
              padding: "6px 12px",
              backgroundColor: "#fff",
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
            <span>◀</span> Restart
          </button>

          {/* Skip to end button */}
          {mode !== "complete" && (
            <button
              onClick={handleSkipToEnd}
              style={{
                padding: "6px 12px",
                backgroundColor: "#fff",
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
              Skip to End <span>→</span>
            </button>
          )}

          {/* Fullscreen button */}
          <button
            onClick={handleToggleFullscreen}
            style={{
              padding: "6px 12px",
              backgroundColor: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 6,
              fontSize: "0.8125rem",
              color: "#374151",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
            title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isFullscreen ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
