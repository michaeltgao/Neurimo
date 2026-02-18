import type { PauseCard as PauseCardType } from "../../types/guidedReview";

type RiskBucket = "low" | "low-moderate" | "moderate" | "moderate-high" | "elevated" | "high";

type Props = {
  card: PauseCardType;
  riskBucket?: RiskBucket;
  onContinue: () => void;
};

// Get background and border colors that scale with ML risk level
function getStatusColors(status: string, riskBucket: RiskBucket) {
  const isLowRisk = riskBucket === "low" || riskBucket === "low-moderate";

  if (status === "observed") {
    return { bg: "#f0fdfa", border: "#99f6e4" }; // Always green for success
  }

  if (isLowRisk) {
    // Low risk: neutral gray for all issue types
    return { bg: "#f9fafb", border: "#e5e7eb" };
  } else if (riskBucket === "moderate") {
    // Moderate: soft blue
    return { bg: "#eff6ff", border: "#bfdbfe" };
  } else if (riskBucket === "moderate-high" || riskBucket === "elevated") {
    // Moderate-high: amber/orange
    return { bg: "#fffbeb", border: "#fde68a" };
  } else {
    // High: red (original behavior)
    if (status === "delayed") return { bg: "#fffbeb", border: "#fde68a" };
    if (status === "partial") return { bg: "#eff6ff", border: "#bfdbfe" };
    if (status === "not_observed") return { bg: "#fef2f2", border: "#fecaca" };
    if (status === "flagged") return { bg: "#fffbeb", border: "#fde68a" }; // Amber for flagged behaviors
    return { bg: "#f9fafb", border: "#e5e7eb" };
  }
}

export default function PauseCard({ card, riskBucket = "moderate", onContinue }: Props) {
  const {
    status,
    statusIcon,
    statusLabel,
    statusColor,
    prompt,
    expectation,
    observation,
    tracking,
    flagIndex,
    flagTotal,
  } = card;

  // Get colors based on risk bucket (scales with ML risk level)
  const { bg: bgColor, border: borderColor } = getStatusColors(status, riskBucket);

  return (
    <div
      style={{
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        backgroundColor: bgColor,
        border: `2px solid ${borderColor}`,
        borderRadius: 12,
        padding: "20px 24px",
        minWidth: 320,
        maxWidth: 400,
        boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
        zIndex: 100,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* Status Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 16,
        }}
      >
        <span
          style={{
            fontSize: "1.5rem",
            lineHeight: 1,
          }}
        >
          {statusIcon}
        </span>
        <span
          style={{
            fontSize: "1.125rem",
            fontWeight: 600,
            color: statusColor,
            textTransform: "uppercase",
            letterSpacing: "0.025em",
          }}
        >
          {statusLabel}
        </span>
      </div>

      {/* Prompt Info */}
      <div style={{ marginBottom: 12 }}>
        <div
          style={{
            fontSize: "0.875rem",
            color: "#374151",
            marginBottom: 4,
          }}
        >
          <strong>{prompt.type}</strong> at {prompt.timestamp}
        </div>
        {prompt.confidence < 0.8 && (
          <div
            style={{
              fontSize: "0.75rem",
              color: "#6b7280",
            }}
          >
            Detection confidence: {Math.round(prompt.confidence * 100)}%
          </div>
        )}
      </div>

      {/* Expectation */}
      <div
        style={{
          fontSize: "0.8125rem",
          color: "#6b7280",
          marginBottom: 8,
        }}
      >
        Expected: {expectation.description} within {expectation.windowDuration}
      </div>

      {/* Observation */}
      <div
        style={{
          fontSize: "0.875rem",
          fontWeight: 500,
          color: statusColor,
          marginBottom: 16,
          padding: "8px 12px",
          backgroundColor: "rgba(255,255,255,0.6)",
          borderRadius: 6,
        }}
      >
        {status === "observed" && observation.latencyDisplay && (
          <>Observed at {observation.latencyDisplay}</>
        )}
        {status === "delayed" && observation.latencyDisplay && (
          <>Response at {observation.latencyDisplay} (delayed)</>
        )}
        {status === "partial" && (
          <>Partial response: {observation.description}</>
        )}
        {status === "not_observed" && (
          <>No response detected in window</>
        )}
        {status === "uncertain" && (
          <>Unable to determine (low tracking quality)</>
        )}
        {status === "flagged" && (
          <>{observation.description}</>
        )}
      </div>

      {/* Tracking Quality */}
      <div
        style={{
          fontSize: "0.75rem",
          color: "#9ca3af",
          marginBottom: 20,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span>Tracking:</span>
        <span
          style={{
            padding: "2px 8px",
            borderRadius: 4,
            backgroundColor: tracking.quality === "good" ? "#dcfce7" :
                            tracking.quality === "medium" ? "#fef3c7" :
                            "#fee2e2",
            color: tracking.quality === "good" ? "#166534" :
                   tracking.quality === "medium" ? "#92400e" :
                   "#991b1b",
            fontWeight: 500,
            textTransform: "capitalize",
          }}
        >
          {tracking.quality} ({tracking.qualityPct}%)
        </span>
        {!tracking.faceVisible && (
          <span style={{ color: "#dc2626" }}>Face not visible</span>
        )}
      </div>

      {/* Continue Button */}
      <button
        onClick={onContinue}
        style={{
          width: "100%",
          padding: "12px 20px",
          backgroundColor: "#0a0a0a",
          color: "#fff",
          border: "none",
          borderRadius: 8,
          fontSize: "0.9375rem",
          fontWeight: 500,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
          transition: "background-color 0.15s",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "#262626";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "#0a0a0a";
        }}
      >
        <span style={{ fontSize: "1rem" }}>▶</span>
        Continue
      </button>

      {/* Flag Counter */}
      <div
        style={{
          marginTop: 12,
          fontSize: "0.75rem",
          color: "#9ca3af",
          textAlign: "center",
        }}
      >
        Flag {flagIndex + 1} of {flagTotal}
      </div>
    </div>
  );
}
