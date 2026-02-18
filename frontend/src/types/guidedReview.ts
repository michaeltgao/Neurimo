/**
 * Guided Review Types
 *
 * Data structures for the video annotation overlay system
 * with guided pause-at-flags playback.
 */

export type TaskType = "name_response" | "joint_attention" | "imitation" | "free_play";

// ═══════════════════════════════════════════════════════════════
// STATUS TYPES
// ═══════════════════════════════════════════════════════════════

export type ObservationStatus = "observed" | "delayed" | "partial" | "not_observed" | "uncertain" | "flagged";

export type QualityLevel = "good" | "medium" | "low";

// ═══════════════════════════════════════════════════════════════
// PAUSE CARD
// ═══════════════════════════════════════════════════════════════

export type PauseCard = {
  // Status
  status: ObservationStatus;
  statusIcon: "✓" | "⚠" | "◐" | "✗" | "?" | "⚑";
  statusLabel: string;
  statusColor: string;

  // What triggered this
  prompt: {
    type: string;                    // "Name call"
    timestamp: string;               // "0:02.1"
    confidence: number;              // 0.86
  };

  // What was expected
  expectation: {
    description: string;             // "Head turn toward speaker"
    windowDuration: string;          // "1.5s"
  };

  // What was observed
  observation: {
    description: string;             // "None" | "Head turn at 0:02.8"
    latencyMs?: number;              // 700
    latencyDisplay?: string;         // "0.7s"
  };

  // Data quality
  tracking: {
    quality: QualityLevel;
    qualityPct: number;              // 89
    faceVisible: boolean;
  };

  // Position in sequence
  flagIndex: number;                 // 0
  flagTotal: number;                 // 3
};

// ═══════════════════════════════════════════════════════════════
// FLAGGED MOMENT
// ═══════════════════════════════════════════════════════════════

export type FlaggedMoment = {
  id: string;
  promptId: string;

  // Timing (in milliseconds)
  promptAtMs: number;
  windowStartMs: number;
  windowEndMs: number;
  pauseAtMs: number;                 // = windowEndMs

  // Expected
  expected: {
    type: string;                    // "orient_to_speaker"
    description: string;             // "Head turn toward speaker"
    windowDurationMs: number;
  };

  // Observed
  observed: {
    status: ObservationStatus;
    description: string;             // "None" | "Head turn at 0:02.8"
    tObservedMs?: number;
    latencyMs?: number;
  };

  // Quality during this window
  trackingQuality: number;           // 0-1
  trackingLabel: QualityLevel;
  faceVisibleDuringWindow: boolean;

  // For timeline marker
  markerColor: string;

  // Pre-computed pause card
  pauseCard: PauseCard;
};

// ═══════════════════════════════════════════════════════════════
// KEY DRIVER
// ═══════════════════════════════════════════════════════════════

export type KeyDriver = {
  id: string;
  label: string;                     // "No orienting to name"
  count?: number;                    // 2
  severity: "high" | "medium" | "low";
  color?: string;                    // Color that scales with ML risk level (e.g., "#6b7280")
  linkedFlagIds: string[];
};

// ═══════════════════════════════════════════════════════════════
// QUALITY METRICS
// ═══════════════════════════════════════════════════════════════

export type VideoQualityMetrics = {
  overallQuality: QualityLevel;
  trackingConfidenceAvg: number;     // 0-1
  faceVisibilityPct: number;         // 0-100
  outOfViewPct: number;              // 0-100
};

// ═══════════════════════════════════════════════════════════════
// SUMMARY (for top bar + end card)
// ═══════════════════════════════════════════════════════════════

export type TaskSummary = {
  total: number;
  observed: number;
  delayed: number;
  notObserved: number;
  uncertain: number;
  clinicalNote: string;
};

export type ImitationSummary = {
  total: number;
  full: number;
  partial: number;
  none: number;
  rate: number;
  rateInclusive: number;
  clinicalNote: string;
};

export type FreePlaySummary = {
  socialLooks: number;
  spontaneousPoints: number;
  toyTransitions: number;
  repetitiveEpisodes: number;
  clinicalNote: string;
};

export type AnnotationSummary = {
  // Risk
  riskScore: number;
  riskConfidenceBand: number;
  riskBucket: "low" | "moderate" | "moderate-high" | "high";

  // Key drivers (clickable tags)
  keyDrivers: KeyDriver[];

  // Counts
  totalPrompts: number;
  totalFlags: number;

  // Per-category (for summary card)
  nameResponse?: TaskSummary;
  jointAttention?: TaskSummary;
  imitation?: ImitationSummary;
  freePlay?: FreePlaySummary;
};

// ═══════════════════════════════════════════════════════════════
// GUIDED REVIEW DATA (API response shape)
// ═══════════════════════════════════════════════════════════════

export type GuidedReviewData = {
  videoId: number;
  videoUrl: string;
  durationMs: number;
  fps: number;

  // For top bar
  taskType: TaskType;
  taskTypeDisplay: string;           // "Joint Attention"
  ageBucket: string;                 // "12-15 mo"
  riskBucket: "low" | "low-moderate" | "moderate" | "moderate-high" | "elevated" | "high";
  quality: VideoQualityMetrics;
  keyDrivers: KeyDriver[];

  // For guided playback
  flaggedMoments: FlaggedMoment[];

  // For summary card
  summary: AnnotationSummary;
};

// ═══════════════════════════════════════════════════════════════
// PLAYBACK STATE
// ═══════════════════════════════════════════════════════════════

export type PlaybackMode = "idle" | "playing" | "paused_at_flag" | "complete";

export type GuidedReviewState = {
  mode: PlaybackMode;
  currentTimeMs: number;

  // Flags
  currentFlagIndex: number | null;
  flagsVisited: Set<string>;

  // UI state
  showPauseCard: boolean;
  activePauseCard: PauseCard | null;
  showSummaryCard: boolean;

  // Overlay toggles
  showSkeleton: boolean;
  showHeadGaze: boolean;
  showDetailed: boolean;
};

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

export function getStatusIcon(status: ObservationStatus): "✓" | "⚠" | "◐" | "✗" | "?" | "⚑" {
  switch (status) {
    case "observed": return "✓";
    case "delayed": return "⚠";
    case "partial": return "◐";
    case "not_observed": return "✗";
    case "uncertain": return "?";
    case "flagged": return "⚑";
  }
}

export function getStatusColor(status: ObservationStatus): string {
  switch (status) {
    case "observed": return "#0D9488";    // Teal
    case "delayed": return "#D97706";     // Amber
    case "partial": return "#3B82F6";     // Blue
    case "not_observed": return "#DC2626"; // Coral/Red
    case "uncertain": return "#6B7280";   // Gray
    case "flagged": return "#D97706";     // Amber (attention-worthy)
  }
}

export function getStatusLabel(status: ObservationStatus): string {
  switch (status) {
    case "observed": return "Observed";
    case "delayed": return "Delayed";
    case "partial": return "Partial";
    case "not_observed": return "Not Observed";
    case "uncertain": return "Uncertain";
    case "flagged": return "Flagged";
  }
}

export function formatTimestamp(ms: number): string {
  const totalSeconds = ms / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

export function formatDuration(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}
