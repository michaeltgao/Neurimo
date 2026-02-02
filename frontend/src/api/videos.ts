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
