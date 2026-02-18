import { api } from "./client";

export type Video = {
  id: number;
  task_type: string;
  storage_path: string;
  child_id: number;
  visit_number: number;
};

export async function getVideo(videoId: number): Promise<Video> {
  const res = await api.get(`/videos/${videoId}`);
  return res.data;
}