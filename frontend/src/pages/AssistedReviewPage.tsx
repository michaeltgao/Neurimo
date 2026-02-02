import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { getVideoAnnotations, type VideoAnnotations } from "../api/videos";

// You need a way to get the video URL.
// Easiest: pass storage_path in the route state OR fetch video record.
// For now: assume you can fetch video metadata (recommended).
import { getVideo, type Video } from "../api/videoMeta"; // create this if you don't have it
import { getVideoStaticUrl } from "../api/videos";

export default function AssistedReviewPage() {
  const { videoId } = useParams();
  const idNum = useMemo(() => Number(videoId), [videoId]);

  const [video, setVideo] = useState<Video | null>(null);
  const [ann, setAnn] = useState<VideoAnnotations | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(idNum) || idNum <= 0) return;

    (async () => {
      try {
        setErr(null);
        const v = await getVideo(idNum);
        setVideo(v);
        const a = await getVideoAnnotations(idNum);
        setAnn(a);
      } catch (e: unknown) {
        const err = e as { response?: { data?: { detail?: string } }; message?: string };
        setErr(err?.response?.data?.detail ?? err?.message ?? "Failed to load assisted review");
      }
    })();
  }, [idNum]);

  if (!Number.isFinite(idNum) || idNum <= 0) return <div style={{ padding: 24 }}>Invalid video id</div>;
  if (err) return <div style={{ padding: 24, color: "crimson" }}>{err}</div>;
  if (!video) return <div style={{ padding: 24 }}>Loading video...</div>;

  const videoUrl = getVideoStaticUrl(video.storage_path);

  return (
    <div style={{ padding: 24, maxWidth: 1000, margin: "0 auto", fontFamily: "system-ui" }}>
      <h1 style={{ marginTop: 0 }}>Assisted Review</h1>
      <div style={{ opacity: 0.8, marginBottom: 12 }}>
        Task: <b>{video.task_type}</b> • Video ID: {video.id}
      </div>

      <div style={{ position: "relative", width: "100%", borderRadius: 12, overflow: "hidden", border: "1px solid #ddd" }}>
        <video src={videoUrl} controls style={{ width: "100%", display: "block" }} />
        {/* v1.5: add <canvas> overlay here */}
      </div>

      <div style={{ marginTop: 16, padding: 12, border: "1px solid #eee", borderRadius: 12 }}>
        <div style={{ fontWeight: 700 }}>Annotations</div>
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
          {ann ? JSON.stringify(ann, null, 2) : "Loading annotations..."}
        </pre>
      </div>
    </div>
  );
}
