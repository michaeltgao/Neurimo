import { useMemo } from "react";

type TimelineVisit = {
  id: number;
  age_months: number;
  visit_date: string;
  asd_risk_bucket: string;
  visit_number: number;
  is_current: boolean;
};

type Props = {
  visits: TimelineVisit[];
  currentVisitId: number;
};

const TICK_MONTHS = [12, 15, 18, 24];

function bucketColor(bucket: string): { dot: string; label: string } {
  const b = bucket.toLowerCase();
  if (b === "low")
    return { dot: "#22c55e", label: "#166534" };
  if (b === "moderate")
    return { dot: "#f59e0b", label: "#92400e" };
  if (b === "moderate-high")
    return { dot: "#f97316", label: "#9a3412" };
  if (b === "high")
    return { dot: "#ef4444", label: "#991b1b" };
  return { dot: "#a3a3a3", label: "#525252" };
}

export default function Timeline({ visits, currentVisitId }: Props) {
  const { axisMin, axisMax } = useMemo(() => {
    if (visits.length === 0) return { axisMin: 12, axisMax: 24 };
    const ages = visits.map((v) => v.age_months);
    return {
      axisMin: Math.min(12, ...ages) - 1,
      axisMax: Math.max(24, ...ages) + 1,
    };
  }, [visits]);

  const range = axisMax - axisMin;

  function pct(months: number): number {
    return ((months - axisMin) / range) * 100;
  }

  if (visits.length === 0) {
    return (
      <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", margin: 0 }}>
        No visits with predictions yet.
      </p>
    );
  }

  return (
    <div style={{ padding: "12px 0 0" }}>
      {/* Axis area */}
      <div style={{ position: "relative", height: 80, marginBottom: 4 }}>
        {/* Horizontal line */}
        <div
          style={{
            position: "absolute",
            top: 36,
            left: 0,
            right: 0,
            height: 2,
            backgroundColor: "var(--color-border)",
            borderRadius: 1,
          }}
        />

        {/* Tick marks */}
        {TICK_MONTHS.filter((m) => m >= axisMin && m <= axisMax).map((m) => (
          <div
            key={m}
            style={{
              position: "absolute",
              left: `${pct(m)}%`,
              transform: "translateX(-50%)",
              textAlign: "center",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 30,
                left: "50%",
                transform: "translateX(-50%)",
                width: 1,
                height: 14,
                backgroundColor: "var(--color-text-tertiary)",
              }}
            />
            <div
              style={{
                position: "absolute",
                top: 50,
                left: "50%",
                transform: "translateX(-50%)",
                fontSize: "0.6875rem",
                color: "var(--color-text-tertiary)",
                whiteSpace: "nowrap",
              }}
            >
              {m}mo
            </div>
          </div>
        ))}

        {/* Visit dots */}
        {visits.map((v) => {
          const isCurrent = v.id === currentVisitId;
          const colors = bucketColor(v.asd_risk_bucket);
          const size = isCurrent ? 14 : 10;

          return (
            <div
              key={v.id}
              style={{
                position: "absolute",
                left: `${pct(v.age_months)}%`,
                top: 36 - size / 2,
                transform: "translateX(-50%)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              {/* Label above dot */}
              <div
                style={{
                  position: "absolute",
                  bottom: size + 4,
                  whiteSpace: "nowrap",
                  fontSize: "0.6875rem",
                  fontWeight: isCurrent ? 600 : 400,
                  color: colors.label,
                }}
              >
                {v.asd_risk_bucket}
              </div>

              {/* Dot */}
              <div
                style={{
                  width: size,
                  height: size,
                  borderRadius: "50%",
                  backgroundColor: colors.dot,
                  border: isCurrent ? "2px solid var(--color-text)" : "2px solid #fff",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
                  cursor: "default",
                }}
                title={`Visit ${v.visit_number} · ${v.visit_date} · ${v.age_months}mo · ${v.asd_risk_bucket}`}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
