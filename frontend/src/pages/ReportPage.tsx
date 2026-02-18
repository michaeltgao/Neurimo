import React, { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { getReport, getQuestionnaire, type Report, type Questionnaire } from "../api/reports";
import { getVisitVideos, type Video } from "../api/videos";
import { getChild, type Child } from "../api/children";
import { useAuth } from "../context/AuthContext";

/* ─── Helpers ─── */

/**
 * Convert probability (0.0-1.0) to display score (0-100).
 * Falls back to bucket midpoint if no score available.
 *
 * Risk buckets (4 levels):
 *   low:      0-25  (prob <= 0.25)
 *   moderate:      26-50 (prob 0.26-0.50)
 *   moderate-high: 51-75 (prob 0.51-0.75)
 *   high:     76-100 (prob > 0.75)
 */
function toDisplayScore(riskScore: number | null | undefined, bucket: string): number {
  if (riskScore != null) return Math.round(riskScore * 100);
  // Fallback to bucket midpoint when no probability available
  const b = bucket.toLowerCase();
  if (b === "low") return 13;        // midpoint of 0-25
  if (b === "moderate") return 38;   // midpoint of 26-50
  if (b === "moderate-high") return 63;   // midpoint of 51-75
  if (b.includes("high")) return 88; // midpoint of 76-100
  return 50;
}

function getRiskStyle(bucket: string) {
  const b = bucket.toLowerCase();
  if (b === "low") return { bg: "#dcfce7", color: "#166534", border: "#bbf7d0", label: "Low" };
  if (b === "moderate") return { bg: "#fef3c7", color: "#92400e", border: "#fde68a", label: "Moderate" };
  if (b === "moderate-high") return { bg: "#fef2f2", color: "#b45309", border: "#fed7aa", label: "Moderate-High" };
  if (b.includes("high")) return { bg: "#fef2f2", color: "#991b1b", border: "#fecaca", label: "High" };
  return { bg: "#f5f5f5", color: "#525252", border: "#e5e5e5", label: bucket };
}

function dotColor(bucket: string): string {
  const b = bucket.toLowerCase();
  if (b === "low") return "#22c55e";
  if (b === "moderate") return "#f59e0b";
  if (b === "moderate-high") return "#f97316";
  return "#ef4444";
}

/* ─── Task Observation Categorization ─── */

type TaskType = "imitation" | "joint_attention" | "free_play";

const TASK_KEYWORDS: Record<TaskType, string[]> = {
  imitation: ["imitat", "copy", "mirror", "repeat", "mimic", "gesture"],
  joint_attention: ["attention", "point", "gaze", "look", "eye contact", "shared", "follow", "engage"],
  free_play: ["play", "explor", "repetitive", "motor", "movement", "hand", "toy", "object", "spin", "line up"],
};

const DEFAULT_OBSERVATIONS: Record<TaskType, string[]> = {
  imitation: [
    "Copies simple gestures when demonstrated",
    "Attempts to repeat sounds and vocalizations",
    "Shows interest in mimicking facial expressions",
  ],
  joint_attention: [
    "Follows pointing gestures appropriately",
    "Engages in reciprocal interaction with caregiver",
    "Shows interest in shared activities",
  ],
  free_play: [
    "Engages with toys in age-appropriate manner",
    "Shows varied and flexible play patterns",
    "Demonstrates typical exploratory behavior",
  ],
};

function categorizeExplanations(explanations: string[]): Record<TaskType, { flagged: string[]; hasFlags: boolean }> {
  const result: Record<TaskType, { flagged: string[]; hasFlags: boolean }> = {
    imitation: { flagged: [], hasFlags: false },
    joint_attention: { flagged: [], hasFlags: false },
    free_play: { flagged: [], hasFlags: false },
  };

  for (const explanation of explanations) {
    const lower = explanation.toLowerCase();
    let matched = false;

    for (const [task, keywords] of Object.entries(TASK_KEYWORDS) as [TaskType, string[]][]) {
      if (keywords.some((kw) => lower.includes(kw))) {
        result[task].flagged.push(explanation);
        result[task].hasFlags = true;
        matched = true;
        break;
      }
    }

    // If no specific match, add to free_play as general behavioral observation
    if (!matched && explanation.length > 0) {
      result.free_play.flagged.push(explanation);
      result.free_play.hasFlags = true;
    }
  }

  return result;
}

const QUESTIONNAIRE_SECTIONS = [
  {
    title: "Social Interaction",
    keys: [
      { key: "social_responds_to_name", label: "Responds when name is called" },
      { key: "social_eye_contact", label: "Makes eye contact during interactions" },
      { key: "social_interest_in_children", label: "Shows interest in other children" },
      { key: "social_smile_response", label: "Smiles in response to your smile" },
      { key: "social_share_enjoyment", label: "Shares enjoyment or interests with you" },
    ],
  },
  {
    title: "Communication",
    keys: [
      { key: "comm_gestures", label: "Uses gestures (pointing, waving) to communicate" },
      { key: "comm_verbal_requests", label: "Responds to simple verbal requests" },
      { key: "comm_show_things", label: "Tries to show you things they find interesting" },
      { key: "comm_babble_words", label: "Babbles or attempts to use words" },
      { key: "comm_imitate", label: "Imitates sounds or actions" },
    ],
  },
  {
    title: "Repetitive Behaviors",
    keys: [
      { key: "rep_repetitive_movements", label: "Engages in repetitive movements (hand flapping, rocking)" },
      { key: "rep_insist_sameness", label: "Insists on sameness or becomes upset by changes" },
      { key: "rep_intense_interests", label: "Has unusually intense interests in specific objects" },
      { key: "rep_line_up_objects", label: "Lines up toys or objects in specific patterns" },
      { key: "rep_difficulty_transitions", label: "Has difficulty with transitions between activities" },
    ],
  },
  {
    title: "Sensory Responses",
    keys: [
      { key: "sens_unusual_reactions", label: "Has unusual reactions to sounds, textures, or lights" },
      { key: "sens_seek_sensory", label: "Seeks out certain sensory experiences" },
      { key: "sens_distressed_noises", label: "Becomes distressed by everyday noises" },
      { key: "sens_food_texture", label: "Has strong food or texture preferences/aversions" },
    ],
  },
];

function responseBadgeStyle(value: string): { bg: string; color: string } {
  switch (value) {
    case "always":
    case "often":
      return { bg: "#dcfce7", color: "#166534" };
    case "sometimes":
      return { bg: "#fef3c7", color: "#92400e" };
    case "rarely":
      return { bg: "#fed7aa", color: "#9a3412" };
    case "never":
      return { bg: "#fecaca", color: "#991b1b" };
    default:
      return { bg: "#f5f5f5", color: "#525252" };
  }
}

const FAMILY_HISTORY_CONDITIONS = [
  { key: "anxiety", label: "Anxiety" },
  { key: "adhd", label: "ADHD/ADD" },
  { key: "asd", label: "Autism Spectrum Disorder" },
  { key: "bipolar", label: "Bipolar Disorder" },
  { key: "depression", label: "Depression" },
  { key: "epilepsy", label: "Epilepsy/Seizure Disorder" },
  { key: "genetic", label: "Genetic Condition" },
  { key: "intellectual_disability", label: "Intellectual Disability" },
  { key: "language_disorder", label: "Language Disorder" },
  { key: "learning_disability", label: "Learning Disability" },
  { key: "tics", label: "Motor or Vocal Tics" },
  { key: "psychosis", label: "Psychosis or Schizophrenia" },
];

const FAMILY_MEMBERS = [
  { key: "mother", label: "Mother" },
  { key: "father", label: "Father" },
  { key: "brother", label: "Brother" },
  { key: "sister", label: "Sister" },
  { key: "grandparent", label: "Grandparent" },
  { key: "aunt_uncle", label: "Aunt/Uncle" },
  { key: "other", label: "Other" },
];

/* ─── Risk Chart (SVG) ─── */

/**
 * Generate a smooth cubic bezier path through points using Catmull-Rom spline
 * tension: 0 = straight lines, 1 = very curved (default 0.5 for natural curves)
 */
function smoothPath(
  points: { x: number; y: number }[],
  tension: number = 0.5
): string {
  if (points.length < 2) return "";
  if (points.length === 2) {
    return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`;
  }

  const path: string[] = [`M ${points[0].x} ${points[0].y}`];

  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];

    // Calculate control points using Catmull-Rom to Cubic Bezier conversion
    const cp1x = p1.x + ((p2.x - p0.x) * tension) / 6;
    const cp1y = p1.y + ((p2.y - p0.y) * tension) / 6;
    const cp2x = p2.x - ((p3.x - p1.x) * tension) / 6;
    const cp2y = p2.y - ((p3.y - p1.y) * tension) / 6;

    path.push(`C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`);
  }

  return path.join(" ");
}

function RiskChart({ visits }: { visits: Report["prior_visits"] }) {
  const [hoveredPoint, setHoveredPoint] = useState<{
    age: number;
    score: number;
    bucket: string;
    x: number;
    y: number;
    visitNumber: number;
  } | null>(null);

  if (visits.length === 0) return null;

  const sorted = [...visits].sort((a, b) => a.age_months - b.age_months);
  const points = sorted.map((v) => ({
    age: v.age_months,
    score: toDisplayScore(v.risk_score, v.asd_risk_bucket),
    bucket: v.asd_risk_bucket,
    isCurrent: v.is_current,
    visitNumber: v.visit_number,
  }));

  const W = 700;
  const H = 300;
  const PAD = { top: 20, right: 30, bottom: 50, left: 50 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const ages = points.map((p) => p.age);
  const minAge = Math.min(12, ...ages);
  const maxAge = Math.max(24, ...ages);

  const xScale = (age: number) => {
    const range = maxAge - minAge;
    if (range === 0) return PAD.left + plotW / 2;
    return PAD.left + ((age - minAge) / range) * plotW;
  };
  const yScale = (score: number) => PAD.top + plotH - (score / 100) * plotH;

  // Convert points to x,y coordinates and generate smooth curve
  const xyPoints = points.map((p) => ({ x: xScale(p.age), y: yScale(p.score) }));
  const linePath = smoothPath(xyPoints, 0.5);

  const gridValues = [0, 25, 50, 75, 100];
  const xTicks = [12, 15, 18, 24].filter((t) => t >= minAge && t <= maxAge);

  return (
    <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
        {/* Grid lines */}
        {gridValues.map((v) => (
          <g key={v}>
            <line
              x1={PAD.left}
              y1={yScale(v)}
              x2={W - PAD.right}
              y2={yScale(v)}
              stroke="#e5e5e5"
              strokeDasharray={v === 0 ? "none" : "4,4"}
            />
            <text x={PAD.left - 8} y={yScale(v) + 4} textAnchor="end" fontSize="11" fill="#a3a3a3">
              {v}
            </text>
          </g>
        ))}

        {/* X-axis ticks */}
        {xTicks.map((t) => (
          <text key={t} x={xScale(t)} y={H - PAD.bottom + 20} textAnchor="middle" fontSize="11" fill="#a3a3a3">
            {t}
          </text>
        ))}

        {/* Axis labels */}
        <text x={W / 2} y={H - 5} textAnchor="middle" fontSize="12" fill="#a3a3a3">
          Age (months)
        </text>
        <text
          x={12}
          y={H / 2}
          textAnchor="middle"
          fontSize="12"
          fill="#a3a3a3"
          transform={`rotate(-90, 12, ${H / 2})`}
        >
          Risk Score
        </text>

        {/* Line */}
        <path d={linePath} fill="none" stroke="#0a0a0a" strokeWidth="2" />

        {/* Interactive Points */}
        {points.map((p, i) => {
          const cx = xScale(p.age);
          const cy = yScale(p.score);
          const isHovered = hoveredPoint?.age === p.age;
          return (
            <g key={i}>
              {/* Larger invisible hit area for easier hovering */}
              <circle
                cx={cx}
                cy={cy}
                r={20}
                fill="transparent"
                style={{ cursor: "pointer" }}
                onMouseEnter={() =>
                  setHoveredPoint({
                    age: p.age,
                    score: p.score,
                    bucket: p.bucket,
                    x: (cx / W) * 100,
                    y: (cy / H) * 100,
                    visitNumber: p.visitNumber,
                  })
                }
                onMouseLeave={() => setHoveredPoint(null)}
              />
              {/* Visible point */}
              <circle
                cx={cx}
                cy={cy}
                r={isHovered ? 9 : p.isCurrent ? 7 : 5}
                fill={dotColor(p.bucket)}
                stroke="#fff"
                strokeWidth="2"
                style={{
                  transition: "r 0.15s ease",
                  pointerEvents: "none",
                }}
              />
            </g>
          );
        })}
      </svg>

      {/* Hover Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: "absolute",
            left: `${hoveredPoint.x}%`,
            top: `${hoveredPoint.y}%`,
            transform: "translate(-50%, -120%)",
            backgroundColor: "#fff",
            border: "1px solid #e5e5e5",
            borderRadius: 8,
            padding: "8px 12px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            pointerEvents: "none",
            zIndex: 10,
            whiteSpace: "nowrap",
          }}
        >
          <div style={{ fontSize: "0.75rem", fontWeight: 600, marginBottom: 4 }}>
            Visit {hoveredPoint.visitNumber}
          </div>
          <div style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
            Age: <strong style={{ color: "#0a0a0a" }}>{hoveredPoint.age} months</strong>
          </div>
          <div style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
            Score: <strong style={{ color: dotColor(hoveredPoint.bucket) }}>{hoveredPoint.score}</strong>
            <span
              style={{
                marginLeft: 6,
                fontSize: "0.6875rem",
                padding: "1px 6px",
                borderRadius: 4,
                backgroundColor: getRiskStyle(hoveredPoint.bucket).bg,
                color: getRiskStyle(hoveredPoint.bucket).color,
              }}
            >
              {getRiskStyle(hoveredPoint.bucket).label}
            </span>
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 12, flexWrap: "wrap" }}>
        {[
          { color: "#22c55e", label: "Low (0-25)" },
          { color: "#f59e0b", label: "Moderate (26-50)" },
          { color: "#f97316", label: "Moderate-High (51-75)" },
          { color: "#ef4444", label: "High (76-100)" },
        ].map((item) => (
          <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                backgroundColor: item.color,
              }}
            />
            <span style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Sidebar Navigation Types ─── */
type SidebarSection = "dashboard" | "assessments" | "questionnaire" | "assisted-review" | "care-plan";

const SIDEBAR_ITEMS: { key: SidebarSection; label: string; icon: React.ReactNode }[] = [
  {
    key: "dashboard",
    label: "Dashboard",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
      </svg>
    ),
  },
  {
    key: "assessments",
    label: "Assessments",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
        <polyline points="14,2 14,8 20,8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10,9 9,9 8,9" />
      </svg>
    ),
  },
  {
    key: "questionnaire",
    label: "Questionnaire & History",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 11l3 3L22 4" />
        <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
      </svg>
    ),
  },
  {
    key: "assisted-review",
    label: "Assisted Review",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="23 7 16 12 23 17 23 7" />
        <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
      </svg>
    ),
  },
  {
    key: "care-plan",
    label: "Care Plan",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2" />
        <rect x="8" y="2" width="8" height="4" rx="1" ry="1" />
        <path d="M9 14l2 2 4-4" />
      </svg>
    ),
  },
];

/* ─── Main Component ─── */

export default function ReportPage() {
  const { visitId } = useParams();
  const [searchParams] = useSearchParams();
  const nav = useNavigate();
  const { logout } = useAuth();

  function onSignOut() {
    logout();
    nav("/signin");
  }

  const isValid = useMemo(() => (visitId ? /^\d+-\d+$/.test(visitId) : false), [visitId]);
  const childIdNum = useMemo(() => {
    if (!visitId) return 0;
    return Number(visitId.split("-")[0]);
  }, [visitId]);

  // Get initial section from URL query param
  const initialSection = useMemo(() => {
    const section = searchParams.get("section");
    if (section && ["dashboard", "assessments", "questionnaire", "assisted-review", "care-plan"].includes(section)) {
      return section as SidebarSection;
    }
    return "dashboard";
  }, [searchParams]);

  const [report, setReport] = useState<Report | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [child, setChild] = useState<Child | null>(null);
  const [questionnaire, setQuestionnaire] = useState<Questionnaire | null>(null);
  const [err, setErr] = useState("");
  const [activeSection, setActiveSection] = useState<SidebarSection>(initialSection);
  const [questionnaireTab, setQuestionnaireTab] = useState<"responses" | "family">("responses");

  useEffect(() => {
    if (!isValid) return;
    let cancelled = false;

    getReport(visitId!)
      .then((r) => {
        if (!cancelled) setReport(r);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const err = e as { response?: { data?: { detail?: string } }; message?: string };
        setErr(err?.response?.data?.detail ?? err?.message ?? "Failed to load report");
      });

    getVisitVideos(visitId!)
      .then((v) => {
        if (!cancelled) setVideos(v);
      })
      .catch(() => {});

    if (childIdNum > 0) {
      getChild(childIdNum)
        .then((c) => {
          if (!cancelled) setChild(c);
        })
        .catch(() => {});
    }

    getQuestionnaire(visitId!)
      .then((q) => {
        if (!cancelled) setQuestionnaire(q);
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [visitId, isValid, childIdNum]);

  /* Computed KPI values */
  const kpiData = useMemo(() => {
    if (!report) return null;

    const score = toDisplayScore(report.risk_score, report.asd_risk_bucket);
    const visits = report.prior_visits;
    const sorted = [...visits].sort((a, b) => a.age_months - b.age_months);
    const current = sorted.find((v) => v.is_current);
    const previous = current
      ? sorted.filter((v) => !v.is_current && v.age_months < current.age_months).at(-1)
      : undefined;

    let scoreChange = 0;
    let changePercent = 0;
    let previousAge = 0;

    if (current && previous) {
      const prevScore = toDisplayScore(previous.risk_score, previous.asd_risk_bucket);
      scoreChange = score - prevScore;
      changePercent = prevScore > 0 ? Math.round((scoreChange / prevScore) * 100) : 0;
      previousAge = previous.age_months;
    }

    return {
      score,
      scoreChange,
      changePercent,
      previousAge,
      visitCount: visits.length,
      concernCount: report.explanations.length,
    };
  }, [report]);

  const riskStyle = report ? getRiskStyle(report.asd_risk_bucket) : null;

  if (!isValid) {
    return (
      <div style={{ padding: 24, color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>
        Invalid visit ID
      </div>
    );
  }

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
          <button onClick={() => nav(`/visits/${visitId}`)} style={{ fontSize: "0.8125rem" }}>
            Back
          </button>
          <h1 style={{ fontSize: "1.125rem", fontWeight: 600, letterSpacing: "-0.025em" }}>
            Patient Dashboard
          </h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {report && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: "0.75rem",
                color: "var(--color-text-secondary)",
              }}
            >
              <span>
                Patient:{" "}
                <strong style={{ color: "var(--color-text)" }}>
                  {child?.pseudo_id ?? `#${report.visit.child_id}`}
                </strong>
              </span>
              <span style={{ color: "var(--color-border)" }}>|</span>
              <span>
                Age:{" "}
                <strong style={{ color: "var(--color-text)" }}>
                  {report.visit.age_months} months
                </strong>
              </span>
              <span style={{ color: "var(--color-border)" }}>|</span>
              <span>
                Last Visit:{" "}
                <strong style={{ color: "var(--color-text)" }}>{report.visit.visit_date}</strong>
              </span>
            </div>
          )}
          <button
            onClick={() => nav(`/visits/${visitId}`)}
            style={{ fontSize: "0.8125rem", display: "flex", alignItems: "center" }}
            title="Edit Videos & Questionnaire"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" />
              <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" />
            </svg>
          </button>
          <button
            onClick={() => nav("/children")}
            style={{ fontSize: "0.8125rem", display: "flex", alignItems: "center" }}
            title="Home"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
              <polyline points="9 22 9 12 15 12 15 22" />
            </svg>
          </button>
          <button onClick={onSignOut} style={{ fontSize: "0.8125rem" }}>
            Sign Out
          </button>
        </div>
      </header>

      {/* Main content with sidebar */}
      <div style={{ display: "flex", minHeight: "calc(100vh - 57px)" }}>
        {/* Sidebar */}
        {report && report.asd_risk_bucket !== "pending" && report.asd_risk_bucket !== "insufficient_data" && (
          <aside
            style={{
              width: 220,
              borderRight: "1px solid var(--color-border)",
              backgroundColor: "#fafafa",
              padding: "24px 0",
              flexShrink: 0,
            }}
          >
            <nav style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {SIDEBAR_ITEMS.map((item) => {
                const isActive = activeSection === item.key;
                return (
                  <button
                    key={item.key}
                    onClick={() => setActiveSection(item.key)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      padding: "10px 20px",
                      border: "none",
                      background: isActive ? "#fff" : "transparent",
                      borderLeft: isActive ? "3px solid var(--color-accent)" : "3px solid transparent",
                      color: isActive ? "var(--color-text)" : "var(--color-text-secondary)",
                      fontSize: "0.875rem",
                      fontWeight: isActive ? 500 : 400,
                      cursor: "pointer",
                      textAlign: "left",
                      transition: "all 0.15s ease",
                      width: "100%",
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = "#f0f0f0";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = "transparent";
                      }
                    }}
                  >
                    <span style={{ opacity: isActive ? 1 : 0.6 }}>{item.icon}</span>
                    {item.label}
                  </button>
                );
              })}
            </nav>

            {/* Patient info in sidebar */}
            <div
              style={{
                margin: "24px 16px 0",
                padding: "12px",
                backgroundColor: "#fff",
                borderRadius: 8,
                border: "1px solid var(--color-border)",
              }}
            >
              <div style={{ fontSize: "0.6875rem", color: "var(--color-text-tertiary)", marginBottom: 4 }}>
                PATIENT
              </div>
              <div style={{ fontSize: "0.8125rem", fontWeight: 500 }}>
                {child?.pseudo_id ?? `#${report.visit.child_id}`}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)", marginTop: 4 }}>
                {report.visit.age_months} months
              </div>
            </div>
          </aside>
        )}

        {/* Main content area */}
        <main style={{ flex: 1, padding: "32px 24px", maxWidth: report && report.asd_risk_bucket !== "pending" && report.asd_risk_bucket !== "insufficient_data" ? 900 : 1080, margin: "0 auto" }}>
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
          ) : report.asd_risk_bucket === "pending" || report.asd_risk_bucket === "insufficient_data" ? (
            <div
              style={{
                padding: 32,
                borderRadius: 8,
                border: "1px solid var(--color-border)",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: 8 }}>
                {report.asd_risk_bucket === "pending"
                  ? "Analysis in Progress"
                  : "Waiting for Videos"}
              </div>
              <p
                style={{
                  fontSize: "0.875rem",
                  color: "var(--color-text-secondary)",
                  margin: "0 0 16px",
                  maxWidth: 480,
                  marginLeft: "auto",
                  marginRight: "auto",
                }}
              >
                {report.asd_risk_bucket === "pending"
                  ? "Videos are being processed. This typically completes within a minute. Please refresh to check for results."
                  : "Upload all 3 behavioral videos (joint attention, imitation, free play) and submit the questionnaire to generate a report."}
              </p>
              <button onClick={() => window.location.reload()} className="primary" style={{ fontSize: "0.8125rem" }}>
                Refresh
              </button>
            </div>
          ) : (
            <div style={{ display: "grid", gap: 32 }}>
              {/* ══════════════════════════════════════════════════════════════════
                  DASHBOARD SECTION
                  ══════════════════════════════════════════════════════════════════ */}
              {activeSection === "dashboard" && (
                <>
                  {/* ── Assessment Overview ── */}
            <section>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
                <h1 style={{ fontSize: "1.25rem" }}>Assessment Overview</h1>
                <span
                  style={{
                    fontSize: "0.6875rem",
                    fontWeight: 500,
                    padding: "2px 10px",
                    borderRadius: 12,
                    backgroundColor: "#eff6ff",
                    color: "#2563eb",
                  }}
                >
                  {report.visit.age_months} Month Visit
                </span>
              </div>
              <p
                style={{
                  fontSize: "0.8125rem",
                  color: "var(--color-text-secondary)",
                  margin: "0 0 16px",
                }}
              >
                Current risk level and key performance indicators
              </p>

              {kpiData && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(4, 1fr)",
                    gap: 12,
                  }}
                >
                  {/* Risk Score */}
                  <div
                    style={{
                      padding: 16,
                      borderRadius: 8,
                      border: "1px solid var(--color-border)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        marginBottom: 8,
                      }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a3a3a3" strokeWidth="2">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                      </svg>
                      {kpiData.scoreChange !== 0 && (
                        <span
                          style={{
                            fontSize: "0.6875rem",
                            fontWeight: 500,
                            color: kpiData.scoreChange > 0 ? "#991b1b" : "#166534",
                          }}
                        >
                          {kpiData.scoreChange > 0 ? "+" : ""}
                          {kpiData.scoreChange} pts
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>
                      Overall Risk Score
                    </div>
                    <div style={{ fontSize: "1.5rem", fontWeight: 600, marginTop: 2 }}>
                      {kpiData.score}
                      <span
                        style={{
                          fontSize: "1rem",
                          fontWeight: 400,
                          color: "var(--color-text-secondary)",
                        }}
                      >
                        /100
                      </span>
                    </div>
                  </div>

                  {/* Change from Last Visit */}
                  <div
                    style={{
                      padding: 16,
                      borderRadius: 8,
                      border: "1px solid var(--color-border)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        marginBottom: 8,
                      }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a3a3a3" strokeWidth="2">
                        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
                        <polyline points="17 6 23 6 23 12" />
                      </svg>
                      {kpiData.previousAge > 0 && (
                        <span
                          style={{
                            fontSize: "0.6875rem",
                            fontWeight: 500,
                            color: "var(--color-text-secondary)",
                          }}
                        >
                          vs. {kpiData.previousAge}mo
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>
                      Change from Last Visit
                    </div>
                    <div style={{ fontSize: "1.5rem", fontWeight: 600, marginTop: 2 }}>
                      {kpiData.previousAge > 0 ? (
                        <>
                          {kpiData.changePercent > 0 ? "+" : ""}
                          {kpiData.changePercent}%
                        </>
                      ) : (
                        <span style={{ fontSize: "1rem", color: "var(--color-text-tertiary)" }}>
                          N/A
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Visits Completed */}
                  <div
                    style={{
                      padding: 16,
                      borderRadius: 8,
                      border: "1px solid var(--color-border)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        marginBottom: 8,
                      }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a3a3a3" strokeWidth="2">
                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                        <line x1="16" y1="2" x2="16" y2="6" />
                        <line x1="8" y1="2" x2="8" y2="6" />
                        <line x1="3" y1="10" x2="21" y2="10" />
                      </svg>
                      <span
                        style={{
                          fontSize: "0.6875rem",
                          fontWeight: 500,
                          color: "#166534",
                        }}
                      >
                        Complete
                      </span>
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>
                      Visits Completed
                    </div>
                    <div style={{ fontSize: "1.5rem", fontWeight: 600, marginTop: 2 }}>
                      {kpiData.visitCount}
                      <span
                        style={{
                          fontSize: "1rem",
                          fontWeight: 400,
                          color: "var(--color-text-secondary)",
                        }}
                      >
                        {" "}
                        total
                      </span>
                    </div>
                  </div>

                  {/* Observations Flagged */}
                  <div
                    style={{
                      padding: 16,
                      borderRadius: 8,
                      border: "1px solid var(--color-border)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        marginBottom: 8,
                      }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a3a3a3" strokeWidth="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                        <line x1="12" y1="9" x2="12" y2="13" />
                        <line x1="12" y1="17" x2="12.01" y2="17" />
                      </svg>
                      {kpiData.concernCount > 0 && (
                        <span
                          style={{
                            fontSize: "0.6875rem",
                            fontWeight: 500,
                            color: kpiData.concernCount > 2 ? "#991b1b" : "#92400e",
                          }}
                        >
                          {kpiData.concernCount > 2 ? "Review" : "Monitor"}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>
                      Observations Flagged
                    </div>
                    <div style={{ fontSize: "1.5rem", fontWeight: 600, marginTop: 2 }}>
                      {kpiData.concernCount}
                    </div>
                  </div>
                </div>
              )}
            </section>

            {/* ── Risk Assessment Summary ── */}
            <section>
              <div
                style={{
                  padding: 16,
                  borderRadius: 8,
                  border: "1px solid var(--color-border)",
                }}
              >
                <h2 style={{ marginBottom: 4 }}>Risk Assessment Summary</h2>
                <p
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--color-text-secondary)",
                    margin: "0 0 12px",
                  }}
                >
                  Patient: {child?.pseudo_id ?? `#${report.visit.child_id}`} | Assessed:{" "}
                  {report.visit.visit_date}
                </p>

                <div
                  style={{
                    padding: "16px 20px",
                    borderRadius: 8,
                    backgroundColor: riskStyle!.bg,
                    border: `1px solid ${riskStyle!.border}`,
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                  }}
                >
                  {/* Risk icon: checkmark for low, info for moderate, exclamation for moderate-high/high */}
                  {report.asd_risk_bucket === "low" ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={riskStyle!.color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M8 12l3 3 5-6" />
                    </svg>
                  ) : report.asd_risk_bucket === "moderate" ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={riskStyle!.color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 16v-4" />
                      <circle cx="12" cy="8" r="0.5" fill={riskStyle!.color} />
                    </svg>
                  ) : (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={riskStyle!.color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M12 8v4" />
                      <circle cx="12" cy="16" r="0.5" fill={riskStyle!.color} />
                    </svg>
                  )}
                  <div>
                    <div
                      style={{
                        fontSize: "0.75rem",
                        fontWeight: 500,
                        color: riskStyle!.color,
                        marginBottom: 2,
                      }}
                    >
                      Current Risk Level
                    </div>
                    <div
                      style={{
                        fontSize: "1.5rem",
                        fontWeight: 600,
                        color: riskStyle!.color,
                      }}
                    >
                      {riskStyle!.label}
                    </div>
                  </div>
                </div>

                <p
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--color-text-tertiary)",
                    margin: "12px 0 0",
                  }}
                >
                  This assessment is based on standardized behavioral observations and developmental
                  milestones. Results should be reviewed in context of clinical evaluation and family
                  history.
                </p>
              </div>
            </section>

            {/* ── Developmental Trends ── */}
            <section>
              <h1 style={{ fontSize: "1.125rem", marginBottom: 4 }}>Developmental Trends</h1>
              <p
                style={{
                  fontSize: "0.8125rem",
                  color: "var(--color-text-secondary)",
                  margin: "0 0 16px",
                }}
              >
                Longitudinal risk progression across assessment periods
              </p>

              <div
                style={{
                  padding: 16,
                  borderRadius: 8,
                  border: "1px solid var(--color-border)",
                }}
              >
                <h2 style={{ marginBottom: 4 }}>Developmental Timeline</h2>
                <p
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--color-text-secondary)",
                    margin: "0 0 12px",
                  }}
                >
                  Risk trend across multiple assessment visits
                </p>
                <RiskChart visits={report.prior_visits} />
              </div>
            </section>

                  {/* ── Visit History (Dashboard) ── */}
                  {report.prior_visits.length > 0 && (
                    <section>
                      <div
                        style={{
                          padding: 16,
                          borderRadius: 8,
                          border: "1px solid var(--color-border)",
                        }}
                      >
                        <h2 style={{ marginBottom: 12 }}>Visit History</h2>
                        <div style={{ display: "grid", gap: 8 }}>
                          {[...report.prior_visits]
                            .sort((a, b) => b.age_months - a.age_months)
                            .map((v) => {
                              const vStyle = getRiskStyle(v.asd_risk_bucket);
                              const clickable = !v.is_current;
                              return (
                                <div
                                  key={v.id}
                                  onClick={
                                    clickable
                                      ? () =>
                                          nav(
                                            `/visits/${report.visit.child_id}-${v.visit_number}/report`
                                          )
                                      : undefined
                                  }
                                  style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    padding: "10px 12px",
                                    backgroundColor: v.is_current
                                      ? "var(--color-bg-tertiary)"
                                      : "var(--color-bg-secondary)",
                                    borderRadius: 6,
                                    fontSize: "0.875rem",
                                    cursor: clickable ? "pointer" : "default",
                                    transition: "background-color 0.15s",
                                  }}
                                  onMouseEnter={(e) => {
                                    if (clickable)
                                      (e.currentTarget as HTMLDivElement).style.backgroundColor =
                                        "var(--color-bg-tertiary)";
                                  }}
                                  onMouseLeave={(e) => {
                                    if (clickable)
                                      (e.currentTarget as HTMLDivElement).style.backgroundColor =
                                        "var(--color-bg-secondary)";
                                  }}
                                >
                                  <div>
                                    <span style={{ fontWeight: 500 }}>
                                      Visit {v.visit_number}
                                    </span>
                                    <span
                                      style={{
                                        color: "var(--color-text-secondary)",
                                        marginLeft: 8,
                                      }}
                                    >
                                      {v.visit_date} &middot; {v.age_months}mo
                                    </span>
                                    {v.is_current && (
                                      <span
                                        style={{
                                          marginLeft: 8,
                                          fontSize: "0.6875rem",
                                          padding: "1px 6px",
                                          borderRadius: 4,
                                          backgroundColor: "#eff6ff",
                                          color: "#2563eb",
                                        }}
                                      >
                                        Current
                                      </span>
                                    )}
                                  </div>
                                  <span
                                    style={{
                                      fontSize: "0.75rem",
                                      fontWeight: 500,
                                      padding: "2px 8px",
                                      borderRadius: 4,
                                      backgroundColor: vStyle.bg,
                                      color: vStyle.color,
                                    }}
                                  >
                                    {vStyle.label}
                                  </span>
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    </section>
                  )}
                </>
              )}

              {/* ══════════════════════════════════════════════════════════════════
                  ASSESSMENTS SECTION - Behavioral Domain Analysis
                  ══════════════════════════════════════════════════════════════════ */}
              {activeSection === "assessments" && (
                <>
                  {/* ── Visit Header ── */}
                  <section>
                    <div style={{ marginBottom: 24 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
                        <h1 style={{ fontSize: "1.25rem", fontWeight: 600 }}>
                          Visit {report.prior_visits.find(v => v.is_current)?.visit_number ?? 1}: {report.visit.age_months} Month Assessment
                        </h1>
                        {riskStyle && (
                          <span
                            style={{
                              fontSize: "0.6875rem",
                              fontWeight: 500,
                              padding: "2px 10px",
                              borderRadius: 12,
                              backgroundColor: riskStyle.bg,
                              color: riskStyle.color,
                            }}
                          >
                            {riskStyle.label} Risk
                          </span>
                        )}
                      </div>
                      <p style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)", margin: 0 }}>
                        Assessed on {report.visit.visit_date}
                      </p>
                    </div>

                    <div
                      style={{
                        padding: 20,
                        borderRadius: 8,
                        border: "1px solid var(--color-border)",
                      }}
                    >
                      <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 4 }}>Behavioral Domain Analysis</h2>
                      <p
                        style={{
                          fontSize: "0.8125rem",
                          color: "var(--color-text-secondary)",
                          margin: "0 0 20px",
                        }}
                      >
                        Current assessment period findings
                      </p>

                      {(() => {
                        // Use backend-categorized explanations if available, fallback to client-side categorization
                        const categorized = report.explanations_by_task
                          ? {
                              imitation: { flagged: report.explanations_by_task.imitation, hasFlags: report.explanations_by_task.imitation.length > 0 },
                              joint_attention: { flagged: report.explanations_by_task.joint_attention, hasFlags: report.explanations_by_task.joint_attention.length > 0 },
                              free_play: { flagged: report.explanations_by_task.free_play, hasFlags: report.explanations_by_task.free_play.length > 0 },
                            }
                          : categorizeExplanations(report.explanations);
                        const tasks: { key: TaskType; title: string; description: string }[] = [
                          { key: "imitation", title: "Imitation", description: "Ability to copy actions, sounds, and expressions" },
                          { key: "joint_attention", title: "Joint Attention", description: "Ability to share attention with others regarding objects or events" },
                          { key: "free_play", title: "Free Play", description: "Spontaneous play behaviors and exploratory patterns" },
                        ];

                        // Determine flag styling intensity based on overall risk level
                        const riskLevel = report.asd_risk_bucket.toLowerCase();
                        const isHighRisk = riskLevel.includes("high");
                        const flagStyle = isHighRisk
                          ? { bg: "#fef2f2", border: "#fecaca", iconColor: "#ef4444", textColor: "#991b1b", obsBg: "#fff", obsBorder: "1px solid #fecaca" }
                          : { bg: "#fefce8", border: "#fef08a", iconColor: "#f59e0b", textColor: "#92400e", obsBg: "#fffbeb", obsBorder: "1px solid #fef08a" };

                        return (
                          <div style={{ display: "grid", gap: 16 }}>
                            {tasks.map((task) => {
                              const taskData = categorized[task.key];
                              const observations = taskData.hasFlags ? taskData.flagged : DEFAULT_OBSERVATIONS[task.key];
                              const isFlagged = taskData.hasFlags;

                              return (
                                <div
                                  key={task.key}
                                  style={{
                                    padding: 16,
                                    borderRadius: 8,
                                    border: isFlagged ? `1px solid ${flagStyle.border}` : "1px solid var(--color-border)",
                                    backgroundColor: isFlagged ? flagStyle.bg : "transparent",
                                  }}
                                >
                                  <div style={{ marginBottom: 8 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                                      <h3 style={{ fontSize: "0.9375rem", fontWeight: 600, margin: 0 }}>{task.title}</h3>
                                      {isFlagged ? (
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={flagStyle.iconColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                          <circle cx="12" cy="12" r="10" />
                                          <line x1="12" y1="8" x2="12" y2="12" />
                                          <line x1="12" y1="16" x2="12.01" y2="16" />
                                        </svg>
                                      ) : (
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                          <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
                                          <polyline points="22 4 12 14.01 9 11.01" />
                                        </svg>
                                      )}
                                    </div>
                                    <p style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)", margin: 0 }}>
                                      {task.description}
                                    </p>
                                  </div>

                                  {/* Clinical Observations */}
                                  <div
                                    style={{
                                      backgroundColor: isFlagged ? flagStyle.obsBg : "var(--color-bg-secondary)",
                                      borderRadius: 6,
                                      padding: 12,
                                      marginTop: 12,
                                      border: isFlagged ? flagStyle.obsBorder : "none",
                                    }}
                                  >
                                    <div style={{ fontSize: "0.6875rem", fontWeight: 600, color: isFlagged ? flagStyle.textColor : "var(--color-text-tertiary)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
                                      {isFlagged ? "Flagged Observations:" : "Clinical Observations:"}
                                    </div>
                                    <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "grid", gap: 6 }}>
                                      {observations.map((obs, idx) => (
                                        <li key={idx} style={{ display: "flex", alignItems: "flex-start", gap: 8, fontSize: "0.8125rem", color: isFlagged ? flagStyle.textColor : "var(--color-text-secondary)" }}>
                                          <span style={{ color: isFlagged ? flagStyle.iconColor : "var(--color-text-tertiary)", marginTop: 2 }}>•</span>
                                          {obs}
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        );
                      })()}
                    </div>
                  </section>
                </>
              )}

              {/* ══════════════════════════════════════════════════════════════════
                  QUESTIONNAIRE & FAMILY HISTORY SECTION
                  ══════════════════════════════════════════════════════════════════ */}
              {activeSection === "questionnaire" && (
                <>
                  {/* ── Section Header ── */}
                  <div style={{ marginBottom: 24 }}>
                    <h1 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: 4 }}>Questionnaire & Family History</h1>
                    <p style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)", margin: 0 }}>
                      Caregiver-reported observations and family medical background
                    </p>
                  </div>

                  {/* ── Tabs ── */}
                  <div
                    style={{
                      backgroundColor: "#fff",
                      borderRadius: 12,
                      border: "1px solid var(--color-border)",
                      overflow: "hidden",
                    }}
                  >
                    {/* Tab headers */}
                    <div style={{ display: "flex", borderBottom: "1px solid var(--color-border)" }}>
                      <button
                        onClick={() => setQuestionnaireTab("responses")}
                        style={{
                          flex: 1,
                          padding: "14px 20px",
                          fontSize: "0.875rem",
                          fontWeight: 500,
                          border: "none",
                          background: questionnaireTab === "responses" ? "#fff" : "var(--color-bg-secondary)",
                          color: questionnaireTab === "responses" ? "var(--color-text)" : "var(--color-text-secondary)",
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          gap: 8,
                          borderBottom: questionnaireTab === "responses" ? "2px solid var(--color-text)" : "2px solid transparent",
                          marginBottom: -1,
                          transition: "all 0.15s ease",
                        }}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M9 11l3 3L22 4" />
                          <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
                        </svg>
                        Questionnaire
                      </button>
                      <button
                        onClick={() => setQuestionnaireTab("family")}
                        style={{
                          flex: 1,
                          padding: "14px 20px",
                          fontSize: "0.875rem",
                          fontWeight: 500,
                          border: "none",
                          background: questionnaireTab === "family" ? "#fff" : "var(--color-bg-secondary)",
                          color: questionnaireTab === "family" ? "var(--color-text)" : "var(--color-text-secondary)",
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          gap: 8,
                          borderBottom: questionnaireTab === "family" ? "2px solid var(--color-text)" : "2px solid transparent",
                          marginBottom: -1,
                          transition: "all 0.15s ease",
                        }}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                          <circle cx="12" cy="7" r="4" />
                        </svg>
                        Family History
                      </button>
                    </div>

                    {/* Tab content */}
                    <div style={{ padding: 24 }}>
                      {/* Questionnaire Responses Tab */}
                      {questionnaireTab === "responses" && (
                        <>
                          {questionnaire?.responses && Object.keys(questionnaire.responses).length > 0 ? (
                            <div style={{ display: "grid", gap: 24 }}>
                              {QUESTIONNAIRE_SECTIONS.map((section) => {
                                const answered = section.keys.filter(
                                  (k) => questionnaire.responses?.[k.key]
                                );
                                if (answered.length === 0) return null;
                                return (
                                  <div key={section.title}>
                                    <div
                                      style={{
                                        fontSize: "0.8125rem",
                                        fontWeight: 600,
                                        color: "var(--color-text)",
                                        marginBottom: 12,
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 8,
                                      }}
                                    >
                                      <div
                                        style={{
                                          width: 4,
                                          height: 16,
                                          backgroundColor: "var(--color-text)",
                                          borderRadius: 2,
                                        }}
                                      />
                                      {section.title}
                                    </div>
                                    <div style={{ display: "grid", gap: 8 }}>
                                      {section.keys.map((item) => {
                                        const val = questionnaire.responses?.[item.key];
                                        if (!val) return null;
                                        const badge = responseBadgeStyle(val);
                                        return (
                                          <div
                                            key={item.key}
                                            style={{
                                              display: "flex",
                                              alignItems: "center",
                                              justifyContent: "space-between",
                                              padding: "10px 14px",
                                              borderRadius: 8,
                                              backgroundColor: "var(--color-bg-secondary)",
                                              border: "1px solid var(--color-border)",
                                            }}
                                          >
                                            <span style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
                                              {item.label}
                                            </span>
                                            <span
                                              style={{
                                                fontSize: "0.75rem",
                                                fontWeight: 500,
                                                padding: "4px 10px",
                                                borderRadius: 6,
                                                backgroundColor: badge.bg,
                                                color: badge.color,
                                                textTransform: "capitalize" as const,
                                              }}
                                            >
                                              {val}
                                            </span>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <div style={{ textAlign: "center", padding: "40px 20px" }}>
                              <div style={{ color: "var(--color-text-tertiary)", marginBottom: 8 }}>
                                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M9 11l3 3L22 4" />
                                  <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
                                </svg>
                              </div>
                              <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", margin: 0 }}>
                                No questionnaire responses recorded
                              </p>
                            </div>
                          )}
                        </>
                      )}

                      {/* Family History Tab */}
                      {questionnaireTab === "family" && (
                        <>
                          {questionnaire?.family_history && Object.keys(questionnaire.family_history).length > 0 ? (
                            <div style={{ overflowX: "auto" }}>
                              <table
                                style={{
                                  width: "100%",
                                  borderCollapse: "collapse",
                                  fontSize: "0.8125rem",
                                  minWidth: 600,
                                }}
                              >
                                <thead>
                                  <tr>
                                    <th
                                      style={{
                                        textAlign: "left",
                                        padding: "12px 14px",
                                        backgroundColor: "var(--color-bg-secondary)",
                                        fontWeight: 600,
                                        borderBottom: "1px solid var(--color-border)",
                                        borderTopLeftRadius: 8,
                                      }}
                                    >
                                      Condition
                                    </th>
                                    {FAMILY_MEMBERS.map((member, idx) => (
                                      <th
                                        key={member.key}
                                        style={{
                                          textAlign: "center",
                                          padding: "12px 8px",
                                          backgroundColor: "var(--color-bg-secondary)",
                                          fontWeight: 600,
                                          borderBottom: "1px solid var(--color-border)",
                                          whiteSpace: "nowrap",
                                          borderTopRightRadius: idx === FAMILY_MEMBERS.length - 1 ? 8 : 0,
                                        }}
                                      >
                                        {member.label}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {FAMILY_HISTORY_CONDITIONS.map((condition, idx) => {
                                    const conditionData = questionnaire.family_history?.[condition.key];
                                    const hasAny = conditionData && Object.values(conditionData).some(Boolean);
                                    if (!hasAny) return null;
                                    return (
                                      <tr key={condition.key}>
                                        <td
                                          style={{
                                            padding: "12px 14px",
                                            borderBottom: idx === FAMILY_HISTORY_CONDITIONS.length - 1 ? "none" : "1px solid var(--color-border)",
                                            fontWeight: 500,
                                          }}
                                        >
                                          {condition.label}
                                        </td>
                                        {FAMILY_MEMBERS.map((member) => (
                                          <td
                                            key={member.key}
                                            style={{
                                              textAlign: "center",
                                              padding: "12px 8px",
                                              borderBottom: idx === FAMILY_HISTORY_CONDITIONS.length - 1 ? "none" : "1px solid var(--color-border)",
                                            }}
                                          >
                                            {conditionData?.[member.key] ? (
                                              <span
                                                style={{
                                                  display: "inline-flex",
                                                  alignItems: "center",
                                                  justifyContent: "center",
                                                  width: 22,
                                                  height: 22,
                                                  borderRadius: 6,
                                                  backgroundColor: "#f0fdf4",
                                                  color: "#16a34a",
                                                }}
                                              >
                                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                                  <polyline points="20 6 9 17 4 12" />
                                                </svg>
                                              </span>
                                            ) : (
                                              <span style={{ color: "var(--color-text-tertiary)" }}>—</span>
                                            )}
                                          </td>
                                        ))}
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          ) : (
                            <div style={{ textAlign: "center", padding: "40px 20px" }}>
                              <div style={{ color: "var(--color-text-tertiary)", marginBottom: 8 }}>
                                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                                  <circle cx="12" cy="7" r="4" />
                                </svg>
                              </div>
                              <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", margin: 0 }}>
                                No family history recorded
                              </p>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </>
              )}

              {/* ══════════════════════════════════════════════════════════════════
                  ASSISTED REVIEW SECTION - Video Recordings
                  ══════════════════════════════════════════════════════════════════ */}
              {activeSection === "assisted-review" && (
                <>
                  {/* ── Section Header ── */}
                  <div style={{ marginBottom: 24 }}>
                    <h1 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: 4 }}>Assisted Review</h1>
                    <p style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)", margin: 0 }}>
                      Review behavioral assessment recordings with assisted analysis
                    </p>
                  </div>

                  {/* ── Video Cards Grid ── */}
                  {videos.length > 0 ? (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
                      {(() => {
                        const taskConfig: Record<string, { title: string; description: string; icon: React.ReactNode; color: string; bgColor: string }> = {
                          imitation: {
                            title: "Imitation",
                            description: "Gesture copying and mimicking behaviors",
                            color: "#7c3aed",
                            bgColor: "#f5f3ff",
                            icon: (
                              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                                <circle cx="9" cy="7" r="4" />
                                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                              </svg>
                            ),
                          },
                          joint_attention: {
                            title: "Joint Attention",
                            description: "Shared focus and gaze following",
                            color: "#0891b2",
                            bgColor: "#ecfeff",
                            icon: (
                              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0891b2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                                <circle cx="12" cy="12" r="3" />
                              </svg>
                            ),
                          },
                          free_play: {
                            title: "Free Play",
                            description: "Spontaneous play and exploration",
                            color: "#059669",
                            bgColor: "#ecfdf5",
                            icon: (
                              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="10" />
                                <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                                <line x1="9" y1="9" x2="9.01" y2="9" />
                                <line x1="15" y1="9" x2="15.01" y2="9" />
                              </svg>
                            ),
                          },
                        };

                        return videos.map((v) => {
                          const config = taskConfig[v.task_type] || {
                            title: v.task_type.replace("_", " "),
                            description: "Behavioral assessment",
                            color: "#6b7280",
                            bgColor: "#f9fafb",
                            icon: (
                              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2">
                                <polygon points="23 7 16 12 23 17 23 7" />
                                <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                              </svg>
                            ),
                          };

                          return (
                            <Link
                              key={v.id}
                              to={`/videos/${v.id}/assisted-review`}
                              style={{
                                display: "block",
                                textDecoration: "none",
                                color: "inherit",
                                borderRadius: 12,
                                border: "1px solid var(--color-border)",
                                overflow: "hidden",
                                transition: "all 0.2s ease",
                                backgroundColor: "#fff",
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.borderColor = config.color;
                                e.currentTarget.style.boxShadow = `0 4px 12px ${config.color}20`;
                                e.currentTarget.style.transform = "translateY(-2px)";
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = "var(--color-border)";
                                e.currentTarget.style.boxShadow = "none";
                                e.currentTarget.style.transform = "translateY(0)";
                              }}
                            >
                              {/* Thumbnail placeholder */}
                              <div
                                style={{
                                  height: 120,
                                  backgroundColor: config.bgColor,
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                  position: "relative",
                                }}
                              >
                                <div
                                  style={{
                                    width: 56,
                                    height: 56,
                                    borderRadius: "50%",
                                    backgroundColor: "#fff",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                                  }}
                                >
                                  {config.icon}
                                </div>
                                {/* Play button overlay */}
                                <div
                                  style={{
                                    position: "absolute",
                                    bottom: 8,
                                    right: 8,
                                    width: 32,
                                    height: 32,
                                    borderRadius: "50%",
                                    backgroundColor: "rgba(0,0,0,0.7)",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                  }}
                                >
                                  <svg width="14" height="14" viewBox="0 0 24 24" fill="#fff">
                                    <polygon points="5 3 19 12 5 21 5 3" />
                                  </svg>
                                </div>
                              </div>

                              {/* Card content */}
                              <div style={{ padding: 16 }}>
                                <h3 style={{ fontSize: "0.9375rem", fontWeight: 600, margin: "0 0 6px" }}>
                                  {config.title}
                                </h3>
                                <p style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)", margin: "0 0 12px", lineHeight: 1.4 }}>
                                  {config.description}
                                </p>
                                <div
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 4,
                                    fontSize: "0.8125rem",
                                    fontWeight: 500,
                                    color: config.color,
                                  }}
                                >
                                  <span>View Analysis</span>
                                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <line x1="5" y1="12" x2="19" y2="12" />
                                    <polyline points="12 5 19 12 12 19" />
                                  </svg>
                                </div>
                              </div>
                            </Link>
                          );
                        });
                      })()}
                    </div>
                  ) : (
                    <div
                      style={{
                        padding: 48,
                        borderRadius: 12,
                        border: "2px dashed var(--color-border)",
                        textAlign: "center",
                        backgroundColor: "var(--color-bg-secondary)",
                      }}
                    >
                      <div style={{ marginBottom: 16 }}>
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-tertiary)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                          <polygon points="23 7 16 12 23 17 23 7" />
                          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                        </svg>
                      </div>
                      <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 4 }}>No Videos Uploaded</h3>
                      <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", margin: 0 }}>
                        Upload assessment videos to enable AI-assisted behavioral analysis.
                      </p>
                    </div>
                  )}

                  {/* ── Info card ── */}
                  {videos.length > 0 && (
                    <div
                      style={{
                        marginTop: 24,
                        padding: 16,
                        borderRadius: 8,
                        backgroundColor: "#eff6ff",
                        border: "1px solid #bfdbfe",
                        display: "flex",
                        alignItems: "flex-start",
                        gap: 12,
                      }}
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 2 }}>
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 16v-4" />
                        <path d="M12 8h.01" />
                      </svg>
                      <div>
                        <div style={{ fontWeight: 500, fontSize: "0.875rem", color: "#1e40af", marginBottom: 4 }}>
                          Assisted Review Mode
                        </div>
                        <p style={{ fontSize: "0.8125rem", color: "#1e40af", margin: 0, lineHeight: 1.5 }}>
                          Each video includes behavioral markers and timestamps highlighting key observations.
                          Use these insights to guide your clinical assessment and documentation.
                        </p>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* ══════════════════════════════════════════════════════════════════
                  CARE PLAN SECTION
                  ══════════════════════════════════════════════════════════════════ */}
              {activeSection === "care-plan" && (
                <>
                  {/* ── Section Header ── */}
                  <div style={{ marginBottom: 24 }}>
                    <h1 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: 4 }}>Care Plan</h1>
                    <p style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)", margin: 0 }}>
                      Recommended interventions and follow-up timeline
                    </p>
                  </div>

                  {/* ── Timeline ── */}
                  <div
                    style={{
                      backgroundColor: "#fff",
                      borderRadius: 12,
                      border: "1px solid var(--color-border)",
                      padding: 24,
                      marginBottom: 24,
                    }}
                  >
                    <div style={{ position: "relative" }}>
                      {/* Timeline items */}
                      {[
                        {
                          title: "Early Intervention Referral",
                          description: "Immediate referral to state early intervention program for eligibility determination",
                          priority: "completed",
                          icon: (
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                              <polyline points="22 4 12 14.01 9 11.01" />
                            </svg>
                          ),
                        },
                        {
                          title: "Developmental Evaluation",
                          description: "Full multidisciplinary assessment including psychology, OT, and speech-language evaluation",
                          priority: "in_progress",
                          icon: (
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                              <polyline points="14 2 14 8 20 8" />
                              <line x1="16" y1="13" x2="8" y2="13" />
                              <line x1="16" y1="17" x2="8" y2="17" />
                            </svg>
                          ),
                        },
                        {
                          title: "Family Support Session",
                          description: "Connect with parent support group and schedule initial family counseling",
                          priority: "in_progress",
                          icon: (
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                              <circle cx="9" cy="7" r="4" />
                              <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                            </svg>
                          ),
                        },
                        {
                          title: "Follow-up Screening",
                          description: "27-month developmental screening to monitor progression and intervention effectiveness",
                          priority: "in_progress",
                          icon: (
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                              <line x1="16" y1="2" x2="16" y2="6" />
                              <line x1="8" y1="2" x2="8" y2="6" />
                              <line x1="3" y1="10" x2="21" y2="10" />
                            </svg>
                          ),
                        },
                      ].map((item, index, arr) => {
                        const priorityStyles: Record<string, { dot: string; bg: string; text: string; label: string }> = {
                          completed: { dot: "#16a34a", bg: "#f0fdf4", text: "#166534", label: "Completed" },
                          in_progress: { dot: "#ea580c", bg: "#fff7ed", text: "#9a3412", label: "In Progress" },
                        };
                        const style = priorityStyles[item.priority];
                        const isLast = index === arr.length - 1;

                        return (
                          <div
                            key={index}
                            style={{
                              display: "flex",
                              gap: 16,
                              paddingBottom: isLast ? 0 : 24,
                              position: "relative",
                            }}
                          >
                            {/* Timeline line */}
                            {!isLast && (
                              <div
                                style={{
                                  position: "absolute",
                                  left: 15,
                                  top: 32,
                                  bottom: 0,
                                  width: 2,
                                  backgroundColor: "#e5e7eb",
                                }}
                              />
                            )}

                            {/* Timeline dot */}
                            <div
                              style={{
                                width: 32,
                                height: 32,
                                borderRadius: "50%",
                                backgroundColor: style.dot,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                flexShrink: 0,
                                color: "#fff",
                                zIndex: 1,
                              }}
                            >
                              {item.icon}
                            </div>

                            {/* Content card */}
                            <div
                              style={{
                                flex: 1,
                                backgroundColor: style.bg,
                                borderRadius: 10,
                                padding: 16,
                                border: `1px solid ${style.dot}20`,
                              }}
                            >
                              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 8 }}>
                                <h3 style={{ fontSize: "0.9375rem", fontWeight: 600, margin: 0, color: style.text }}>
                                  {item.title}
                                </h3>
                                <span
                                  style={{
                                    fontSize: "0.6875rem",
                                    fontWeight: 500,
                                    padding: "3px 8px",
                                    borderRadius: 4,
                                    backgroundColor: style.dot,
                                    color: "#fff",
                                    flexShrink: 0,
                                  }}
                                >
                                  {style.label}
                                </span>
                              </div>
                              <p style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)", margin: 0, lineHeight: 1.5 }}>
                                {item.description}
                              </p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}

              {/* ── Footer (dashboard only) ── */}
              {activeSection === "dashboard" && (
                <footer
                  style={{
                    paddingTop: 16,
                    borderTop: "1px solid var(--color-border)",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <span
                    style={{ fontSize: "0.75rem", color: "var(--color-text-tertiary)" }}
                  >
                    &copy; {new Date().getFullYear()} Neurimo
                  </span>
                  <span
                    style={{ fontSize: "0.75rem", color: "var(--color-text-tertiary)" }}
                  >
                    For clinical decision support only &middot; Not a standalone diagnosis
                  </span>
                </footer>
              )}
          </div>
        )}
      </main>
      </div>
    </div>
  );
}
