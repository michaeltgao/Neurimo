import type { FlaggedMoment } from "../../types/guidedReview";
import { formatTimestamp } from "../../types/guidedReview";

type Props = {
  durationMs: number;
  currentTimeMs: number;
  flaggedMoments: FlaggedMoment[];
  currentFlagIndex: number | null;
  onSeek?: (timeMs: number) => void;
};

export default function ReviewTimeline({
  durationMs,
  currentTimeMs,
  flaggedMoments,
  currentFlagIndex,
  onSeek,
}: Props) {
  const progress = durationMs > 0 ? (currentTimeMs / durationMs) * 100 : 0;

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onSeek) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const timeMs = percentage * durationMs;
    onSeek(Math.max(0, Math.min(durationMs, timeMs)));
  };

  return (
    <div style={{ padding: "12px 0" }}>
      {/* Timeline Bar */}
      <div
        onClick={handleClick}
        style={{
          position: "relative",
          height: 8,
          backgroundColor: "#e5e7eb",
          borderRadius: 4,
          cursor: onSeek ? "pointer" : "default",
          overflow: "visible",
        }}
      >
        {/* Expected Windows (behind progress) */}
        {flaggedMoments.map((flag) => {
          const startPct = (flag.windowStartMs / durationMs) * 100;
          const widthPct = ((flag.windowEndMs - flag.windowStartMs) / durationMs) * 100;

          return (
            <div
              key={`window-${flag.id}`}
              style={{
                position: "absolute",
                left: `${startPct}%`,
                width: `${widthPct}%`,
                top: 0,
                bottom: 0,
                backgroundColor: "#dbeafe",
                opacity: 0.6,
                borderRadius: 2,
              }}
            />
          );
        })}

        {/* Progress */}
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: `${progress}%`,
            backgroundColor: "#0a0a0a",
            borderRadius: 4,
            transition: "width 0.1s linear",
          }}
        />

        {/* Flag Markers */}
        {flaggedMoments.map((flag, index) => {
          const positionPct = (flag.pauseAtMs / durationMs) * 100;
          const isActive = currentFlagIndex === index;
          const isVisited = currentTimeMs >= flag.pauseAtMs;

          return (
            <div
              key={flag.id}
              style={{
                position: "absolute",
                left: `${positionPct}%`,
                top: "50%",
                transform: "translate(-50%, -50%)",
                width: isActive ? 16 : 12,
                height: isActive ? 16 : 12,
                borderRadius: "50%",
                backgroundColor: flag.markerColor,
                border: isActive ? "3px solid #fff" : "2px solid #fff",
                boxShadow: isActive
                  ? "0 0 0 2px #0a0a0a, 0 2px 8px rgba(0,0,0,0.3)"
                  : "0 1px 3px rgba(0,0,0,0.2)",
                cursor: "pointer",
                transition: "all 0.15s ease",
                zIndex: isActive ? 10 : 5,
                opacity: isVisited && !isActive ? 0.6 : 1,
              }}
              onClick={(e) => {
                e.stopPropagation();
                if (onSeek) {
                  // Seek to just before the pause point
                  onSeek(Math.max(0, flag.pauseAtMs - 100));
                }
              }}
              title={`Flag ${index + 1}: ${flag.expected.description}`}
            />
          );
        })}

        {/* Playhead */}
        <div
          style={{
            position: "absolute",
            left: `${progress}%`,
            top: "50%",
            transform: "translate(-50%, -50%)",
            width: 14,
            height: 14,
            borderRadius: "50%",
            backgroundColor: "#0a0a0a",
            border: "2px solid #fff",
            boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
            zIndex: 20,
          }}
        />
      </div>

      {/* Time Display */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 8,
          fontSize: "0.75rem",
          color: "#6b7280",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <span>{formatTimestamp(currentTimeMs)}</span>
        <span>{formatTimestamp(durationMs)}</span>
      </div>
    </div>
  );
}
