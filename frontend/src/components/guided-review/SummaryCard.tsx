import type {
  AnnotationSummary,
  VideoQualityMetrics,
} from "../../types/guidedReview";

type TrialOutcome = {
  status: "observed" | "delayed" | "partial" | "not_observed" | "uncertain";
};

type RiskBucket = "low" | "low-moderate" | "moderate" | "moderate-high" | "elevated" | "high";

type Props = {
  summary: AnnotationSummary;
  quality: VideoQualityMetrics;
  riskBucket?: RiskBucket;
  onReplay: () => void;
  onClose: () => void;
  onNext?: () => void;
  hasNextVideo?: boolean;
};

// Get outcome box colors that scale with ML risk level
function getOutcomeConfig(status: TrialOutcome["status"], riskBucket: RiskBucket) {
  const isLowRisk = riskBucket === "low" || riskBucket === "low-moderate";

  // Success always green
  if (status === "observed") {
    return { icon: "✓", bg: "#dcfce7", color: "#166534" };
  }

  // Uncertain always gray
  if (status === "uncertain") {
    return { icon: "?", bg: "#f3f4f6", color: "#6b7280" };
  }

  // Scale colors based on risk level
  if (isLowRisk) {
    // Low risk: neutral gray for all issues (informational)
    return {
      delayed: { icon: "○", bg: "#f3f4f6", color: "#6b7280" },
      partial: { icon: "○", bg: "#f3f4f6", color: "#6b7280" },
      not_observed: { icon: "○", bg: "#f3f4f6", color: "#6b7280" },
    }[status] || { icon: "?", bg: "#f3f4f6", color: "#6b7280" };
  } else if (riskBucket === "moderate") {
    // Moderate: soft blue
    return {
      delayed: { icon: "◐", bg: "#dbeafe", color: "#1d4ed8" },
      partial: { icon: "◐", bg: "#dbeafe", color: "#1d4ed8" },
      not_observed: { icon: "○", bg: "#dbeafe", color: "#1d4ed8" },
    }[status] || { icon: "?", bg: "#f3f4f6", color: "#6b7280" };
  } else if (riskBucket === "moderate-high" || riskBucket === "elevated") {
    // Moderate-high: amber/orange
    return {
      delayed: { icon: "⚠", bg: "#fef3c7", color: "#92400e" },
      partial: { icon: "◐", bg: "#fef3c7", color: "#92400e" },
      not_observed: { icon: "⚠", bg: "#fef3c7", color: "#92400e" },
    }[status] || { icon: "?", bg: "#f3f4f6", color: "#6b7280" };
  } else {
    // High risk: red (original alarming colors)
    return {
      delayed: { icon: "⚠", bg: "#fef3c7", color: "#92400e" },
      partial: { icon: "◐", bg: "#dbeafe", color: "#1d4ed8" },
      not_observed: { icon: "✗", bg: "#fee2e2", color: "#991b1b" },
    }[status] || { icon: "?", bg: "#f3f4f6", color: "#6b7280" };
  }
}

function OutcomeBox({ status, riskBucket = "moderate" }: { status: TrialOutcome["status"]; riskBucket?: RiskBucket }) {
  const config = getOutcomeConfig(status, riskBucket);

  return (
    <div
      style={{
        width: 32,
        height: 32,
        borderRadius: 6,
        backgroundColor: config.bg,
        color: config.color,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "0.875rem",
        fontWeight: 600,
      }}
    >
      {config.icon}
    </div>
  );
}

function SummarySection({
  title,
  trials,
  statLine,
  clinicalNote,
  riskBucket = "moderate",
}: {
  title: string;
  trials: TrialOutcome[];
  statLine: string;
  clinicalNote: string;
  riskBucket?: RiskBucket;
}) {
  return (
    <div
      style={{
        padding: 16,
        backgroundColor: "#f9fafb",
        borderRadius: 8,
        border: "1px solid #e5e7eb",
      }}
    >
      <div
        style={{
          fontSize: "0.8125rem",
          fontWeight: 600,
          color: "#374151",
          marginBottom: 12,
          textTransform: "uppercase",
          letterSpacing: "0.025em",
        }}
      >
        {title}
      </div>

      {/* Trial outcome boxes */}
      <div
        style={{
          display: "flex",
          gap: 8,
          marginBottom: 12,
        }}
      >
        {trials.map((trial, i) => (
          <OutcomeBox key={i} status={trial.status} riskBucket={riskBucket} />
        ))}
      </div>

      {/* Stat line */}
      <div
        style={{
          fontSize: "0.875rem",
          fontWeight: 500,
          color: "#111827",
          marginBottom: 4,
        }}
      >
        {statLine}
      </div>

      {/* Clinical note */}
      <div
        style={{
          fontSize: "0.8125rem",
          color: "#6b7280",
          fontStyle: "italic",
        }}
      >
        "{clinicalNote}"
      </div>
    </div>
  );
}

export default function SummaryCard({
  summary,
  quality,
  riskBucket = "moderate",
  onReplay,
  onClose,
  onNext,
  hasNextVideo,
}: Props) {
  const isLowRisk = riskBucket === "low" || riskBucket === "low-moderate";

  // Build trial outcomes from summary data
  const buildTrials = (): { title: string; trials: TrialOutcome[]; statLine: string; clinicalNote: string }[] => {
    const sections: { title: string; trials: TrialOutcome[]; statLine: string; clinicalNote: string }[] = [];

    if (summary.nameResponse) {
      const nr = summary.nameResponse;
      const trials: TrialOutcome[] = [];

      for (let i = 0; i < nr.observed; i++) trials.push({ status: "observed" });
      for (let i = 0; i < nr.delayed; i++) trials.push({ status: "delayed" });
      for (let i = 0; i < nr.notObserved; i++) trials.push({ status: "not_observed" });
      for (let i = 0; i < nr.uncertain; i++) trials.push({ status: "uncertain" });

      sections.push({
        title: "Name Response",
        trials,
        statLine: `${nr.observed} of ${nr.total} observed`,
        clinicalNote: nr.clinicalNote,
      });
    }

    if (summary.jointAttention) {
      const ja = summary.jointAttention;
      const trials: TrialOutcome[] = [];

      for (let i = 0; i < ja.observed; i++) trials.push({ status: "observed" });
      for (let i = 0; i < ja.delayed; i++) trials.push({ status: "delayed" });
      for (let i = 0; i < ja.notObserved; i++) trials.push({ status: "not_observed" });
      for (let i = 0; i < ja.uncertain; i++) trials.push({ status: "uncertain" });

      // Softer language for low risk
      let statLine: string;
      if (isLowRisk) {
        // Use neutral language when overall risk is low
        const responded = ja.observed + ja.delayed;
        if (responded > 0) {
          statLine = `${responded} of ${ja.total} responses detected`;
        } else {
          statLine = `${ja.total} trials reviewed`;
        }
      } else {
        statLine = `${ja.observed} of ${ja.total} shared attention${ja.delayed > 0 ? ` (${ja.delayed} delayed)` : ""}`;
      }

      sections.push({
        title: "Joint Attention",
        trials,
        statLine,
        clinicalNote: ja.clinicalNote,
      });
    }

    if (summary.imitation) {
      const im = summary.imitation;
      const trials: TrialOutcome[] = [];

      for (let i = 0; i < im.full; i++) trials.push({ status: "observed" });
      for (let i = 0; i < im.partial; i++) trials.push({ status: "partial" });
      for (let i = 0; i < im.none; i++) trials.push({ status: "not_observed" });

      sections.push({
        title: "Imitation",
        trials,
        statLine: `${im.full} full, ${im.partial} partial, ${im.none} none (${Math.round(im.rateInclusive * 100)}%)`,
        clinicalNote: im.clinicalNote,
      });
    }

    if (summary.freePlay) {
      const fp = summary.freePlay;
      sections.push({
        title: "Free Play Observations",
        trials: [], // No discrete trials for free play
        statLine: `Social looks: ${fp.socialLooks} | Points: ${fp.spontaneousPoints} | Transitions: ${fp.toyTransitions}`,
        clinicalNote: fp.clinicalNote,
      });
    }

    return sections;
  };

  const sections = buildTrials();

  const qualityColor = quality.overallQuality === "good" ? "#166534" :
                       quality.overallQuality === "medium" ? "#92400e" :
                       "#991b1b";
  const qualityBg = quality.overallQuality === "good" ? "#dcfce7" :
                    quality.overallQuality === "medium" ? "#fef3c7" :
                    "#fee2e2";

  return (
    <div
      style={{
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        backgroundColor: "#fff",
        borderRadius: 16,
        padding: 24,
        minWidth: 400,
        maxWidth: 500,
        maxHeight: "80vh",
        overflowY: "auto",
        boxShadow: "0 12px 48px rgba(0,0,0,0.2)",
        zIndex: 100,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* Exit Button */}
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          width: 32,
          height: 32,
          borderRadius: "50%",
          backgroundColor: "#f3f4f6",
          border: "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "1rem",
          color: "#6b7280",
          transition: "background-color 0.15s",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "#e5e7eb";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "#f3f4f6";
        }}
        title="Close"
      >
        ✕
      </button>

      {/* Header */}
      <div
        style={{
          textAlign: "center",
          marginBottom: 24,
        }}
      >
        <div
          style={{
            width: 48,
            height: 48,
            borderRadius: "50%",
            backgroundColor: "#dcfce7",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 12px",
          }}
        >
          <span style={{ fontSize: "1.5rem" }}>✓</span>
        </div>
        <h2
          style={{
            margin: 0,
            fontSize: "1.25rem",
            fontWeight: 600,
            color: "#111827",
          }}
        >
          Review Complete
        </h2>
      </div>

      {/* Sections */}
      <div style={{ display: "grid", gap: 12, marginBottom: 20 }}>
        {sections.map((section, i) => (
          <SummarySection key={i} {...section} riskBucket={riskBucket} />
        ))}

        {/* Data Quality Section */}
        <div
          style={{
            padding: 16,
            backgroundColor: "#f9fafb",
            borderRadius: 8,
            border: "1px solid #e5e7eb",
          }}
        >
          <div
            style={{
              fontSize: "0.8125rem",
              fontWeight: 600,
              color: "#374151",
              marginBottom: 8,
              textTransform: "uppercase",
              letterSpacing: "0.025em",
            }}
          >
            Data Quality
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              fontSize: "0.875rem",
              color: "#6b7280",
            }}
          >
            <span
              style={{
                padding: "2px 10px",
                borderRadius: 4,
                backgroundColor: qualityBg,
                color: qualityColor,
                fontWeight: 500,
                textTransform: "capitalize",
              }}
            >
              {quality.overallQuality}
            </span>
            <span>Face visible: {quality.faceVisibilityPct}%</span>
            <span>•</span>
            <span>
              {quality.overallQuality === "good" ? "High reliability" :
               quality.overallQuality === "medium" ? "Moderate reliability" :
               "Low reliability"}
            </span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div
        style={{
          display: "flex",
          gap: 12,
        }}
      >
        <button
          onClick={onReplay}
          style={{
            flex: 1,
            padding: "12px 20px",
            backgroundColor: "#f3f4f6",
            color: "#374151",
            border: "1px solid #e5e7eb",
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
            e.currentTarget.style.backgroundColor = "#e5e7eb";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "#f3f4f6";
          }}
        >
          <span>◀</span>
          Replay Video
        </button>

        {hasNextVideo && onNext && (
          <button
            onClick={onNext}
            style={{
              flex: 1,
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
            Next Video
            <span>→</span>
          </button>
        )}
      </div>
    </div>
  );
}
