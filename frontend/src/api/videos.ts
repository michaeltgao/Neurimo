import { api } from "./client";

export type TaskType = "imitation" | "joint_attention" | "free_play";

export type Video = {
  id: number;
  visit_id: number;
  task_type: TaskType;
  storage_path: string;
  status: string;
  created_at: string;
};

export async function uploadVisitVideo(visitId: string, taskType: TaskType, file: File): Promise<Video> {
  const form = new FormData();
  form.append("file", file);

  const res = await api.post<Video>(`/visits/${visitId}/videos`, form, {
    params: { task_type: taskType },
    headers: { "Content-Type": "multipart/form-data" },
  });

  return res.data;
}

export async function getVisitVideos(visitId: string): Promise<Video[]> {
  const res = await api.get<Video[]>(`/visits/${visitId}/videos`);
  return res.data;
}

export type AnnotationEvent = {
  type: string;
  start_frame: number;
  end_frame: number;
  label?: string;
};

export type VideoAnnotations = {
  version: string;
  video_id: number;
  task_type: TaskType;
  fps: number | null;
  events: AnnotationEvent[];
  signals: Record<string, number[]>;
  landmarks: Record<string, unknown>[];
  notes?: string;
};

export async function getVideoAnnotations(videoId: number): Promise<VideoAnnotations> {
  const res = await api.get<VideoAnnotations>(`/videos/${videoId}/annotations`);
  return res.data;
}

// If backend serves /static, build a direct URL for the MP4
export function getVideoStaticUrl(storagePath: string): string {
  const base = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

  if (storagePath.startsWith("/static/")) return `${base}${storagePath}`;

  const idx = storagePath.indexOf("data/");
  const rel = idx >= 0 ? storagePath.slice(idx + "data/".length) : storagePath;

  return `${base}/static/${rel}`;
}

// Guided Review Data types (matching backend output)
export type GuidedReviewData = {
  videoId: number;
  videoUrl: string;
  durationMs: number;
  taskTypeDisplay: string;
  ageBucket: string;
  riskBucket: "low" | "low-moderate" | "moderate" | "moderate-high" | "elevated" | "high";
  flaggedMoments: FlaggedMoment[];
  quality: VideoQualityMetrics;
  summary: AnnotationSummary;
  keyDrivers: KeyDriver[];
};

export type FlaggedMoment = {
  id: string;
  pauseAtMs: number;
  windowStartMs: number;
  windowEndMs: number;
  markerColor: string;
  expected: {
    type: string;
    description: string;
  };
  observed: {
    status: string;
    latencyMs: number | null;
    description: string;
  };
  pauseCard: PauseCard;
};

export type PauseCard = {
  status: string;
  statusIcon: string;
  statusLabel: string;
  statusColor: string;
  prompt: {
    type: string;
    timestamp: string;
    confidence: number;
  };
  expectation: {
    description: string;
    windowDuration: string;
  };
  observation: {
    latencyDisplay: string | null;
    description: string;
  };
  tracking: {
    quality: string;
    qualityPct: number;
    faceVisible: boolean;
  };
  flagIndex: number;
  flagTotal: number;
};

export type VideoQualityMetrics = {
  overallQuality: "good" | "medium" | "poor";
  faceVisibilityPct: number;
};

export type AnnotationSummary = {
  nameResponse?: {
    observed: number;
    delayed: number;
    notObserved: number;
    uncertain: number;
    total: number;
    clinicalNote: string;
  };
  jointAttention?: {
    observed: number;
    delayed: number;
    notObserved: number;
    uncertain: number;
    total: number;
    clinicalNote: string;
  };
  imitation?: {
    full: number;
    partial: number;
    none: number;
    rateInclusive: number;
    clinicalNote: string;
  };
  freePlay?: {
    socialLooks: number;
    spontaneousPoints: number;
    toyTransitions: number;
    clinicalNote: string;
  };
};

export type KeyDriver = {
  id: string;
  label: string;
  severity: "high" | "medium" | "low";
  color?: string;  // Color that scales with ML risk level
  count?: number;
};

export async function getGuidedReviewData(videoId: number, durationMs: number): Promise<GuidedReviewData> {
  const res = await api.get<GuidedReviewData>(`/videos/${videoId}/guided-review`, {
    params: { duration_ms: durationMs },
  });
  return res.data;
}

// Overlay Data types (for real-time video annotation overlays)
import type { OverlayData } from "../types/overlayTypes";

export async function getOverlayData(videoId: number, durationMs: number): Promise<OverlayData> {
  const res = await api.get<OverlayData>(`/videos/${videoId}/overlay-data`, {
    params: { duration_ms: durationMs },
  });
  return res.data;
}
