from __future__ import annotations

import argparse
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore


# Model URLs for MediaPipe Tasks
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
POSE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# Cache directory for models
MODEL_CACHE_DIR = Path.home() / ".cache" / "mediapipe_models"


def _download_model(url: str, filename: str) -> Path:
    """Download model file if not cached, return path."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_CACHE_DIR / filename
    if not model_path.exists():
        print(f"Downloading {filename}...")
        # Use unverified SSL context to handle macOS certificate issues
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ssl_context) as response:
            model_path.write_bytes(response.read())
        print(f"Downloaded to {model_path}")
    return model_path


def get_model_paths() -> Tuple[Path, Path]:
    """Get paths to face and pose landmarker models, downloading if needed."""
    face_model = _download_model(FACE_LANDMARKER_URL, "face_landmarker.task")
    pose_model = _download_model(POSE_LANDMARKER_URL, "pose_landmarker_lite.task")
    return face_model, pose_model


# ---- CONFIG: adjust to match your manifest columns ----
TASKS = {
    "joint_attention": "joint_attention_path",
    "imitation": "imitation_path",
    "free_play": "free_play_path",
}

# ---- QC thresholds tuned for 8–10 second clips ----
MIN_DURATION_SEC = 7.0
MAX_DURATION_SEC = 12.0

# We sample every N frames for speed. With 30fps and stride=15:
# 10s video -> 300 frames -> 20 sampled frames
SAMPLE_EVERY_N = 15

# Max frames to process per video (for speed)
MAX_FRAMES_PER_VIDEO = 200

# If fps is lower (e.g., 15fps), sampled frames ~75.
MIN_FRAMES_USED = 30

# Tracking ratios: with short clips, don't set too strict initially
MIN_POSE_RATIO = 0.20
MIN_FACE_RATIO = 0.02


# If face+pose missing on most frames, it's unusable
MAX_OUT_OF_VIEW_RATIO = 0.80


@dataclass
class QCMetrics:
    duration_sec: float
    fps_est: float
    frames_used: int
    face_detected_ratio: float
    pose_detected_ratio: float
    out_of_view_ratio: float
    passed: bool
    reason: str


def safe_float(x, default: float = 0.0) -> float:
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return default
        return xf
    except Exception:
        return default


def qc_single_video(
    video_path: str,
    sample_every_n: int = SAMPLE_EVERY_N,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
    face_landmarker: Optional[Any] = None,
    pose_landmarker: Optional[Any] = None,
) -> QCMetrics:
    p = Path(video_path)
    if not p.exists():
        return QCMetrics(
            duration_sec=0.0,
            fps_est=0.0,
            frames_used=0,
            face_detected_ratio=0.0,
            pose_detected_ratio=0.0,
            out_of_view_ratio=1.0,
            passed=False,
            reason="file_missing",
        )

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return QCMetrics(
            duration_sec=0.0,
            fps_est=0.0,
            frames_used=0,
            face_detected_ratio=0.0,
            pose_detected_ratio=0.0,
            out_of_view_ratio=1.0,
            passed=False,
            reason="cannot_open_video",
        )

    fps_est = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0)

    face_present = 0
    pose_present = 0
    both_missing = 0
    frames_used = 0

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # sampling
        if sample_every_n > 1 and (frame_idx % sample_every_n != 0):
            frame_idx += 1
            continue

        frame_idx += 1
        frames_used += 1

        # Stop if we've processed enough frames
        if frames_used >= max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_ok = False
        pose_ok = False

        if face_landmarker is not None:
            face_result = face_landmarker.detect(mp_image)
            face_ok = len(face_result.face_landmarks) > 0

        if pose_landmarker is not None:
            pose_result = pose_landmarker.detect(mp_image)
            pose_ok = len(pose_result.pose_landmarks) > 0

        if face_ok:
            face_present += 1
        if pose_ok:
            pose_present += 1
        if (not face_ok) and (not pose_ok):
            both_missing += 1

    cap.release()

    if frames_used <= 0:
        return QCMetrics(
            duration_sec=0.0,
            fps_est=fps_est,
            frames_used=0,
            face_detected_ratio=0.0,
            pose_detected_ratio=0.0,
            out_of_view_ratio=1.0,
            passed=False,
            reason="no_frames_read",
        )

    # duration estimate based on sampled frames
    duration_sec = safe_float((frames_used * sample_every_n) / max(fps_est, 1e-6), 0.0)

    face_ratio = face_present / frames_used
    pose_ratio = pose_present / frames_used
    out_of_view_ratio = both_missing / frames_used

    passed = True
    reasons = []

    if duration_sec < MIN_DURATION_SEC:
        passed = False
        reasons.append("too_short")
    if duration_sec > MAX_DURATION_SEC:
        passed = False
        reasons.append("too_long")
    if frames_used < MIN_FRAMES_USED:
        passed = False
        reasons.append("too_few_frames")
    if pose_ratio < MIN_POSE_RATIO:
        passed = False
        reasons.append("low_pose_tracking")
    if face_ratio < MIN_FACE_RATIO:
        passed = False
        reasons.append("low_face_tracking")
    if out_of_view_ratio > MAX_OUT_OF_VIEW_RATIO:
        passed = False
        reasons.append("mostly_out_of_view")

    reason = "pass" if passed else "|".join(reasons)

    return QCMetrics(
        duration_sec=duration_sec,
        fps_est=fps_est,
        frames_used=frames_used,
        face_detected_ratio=face_ratio,
        pose_detected_ratio=pose_ratio,
        out_of_view_ratio=out_of_view_ratio,
        passed=passed,
        reason=reason,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to data/manifest.csv")
    parser.add_argument("--out", default="data/derived/qc.csv", help="Output CSV path")
    parser.add_argument("--min_tasks_pass", type=int, default=2, help="Require >= this many tasks passing per child")
    parser.add_argument("--sample_every_n", type=int, default=SAMPLE_EVERY_N, help="Use every Nth frame (default: 15)")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES_PER_VIDEO, help="Max frames to process per video (default: 200)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    missing_cols = [col for col in TASKS.values() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {missing_cols}. Present: {list(df.columns)}")

    # Download models and create landmarkers
    face_model_path, pose_model_path = get_model_paths()

    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    rows: List[Dict] = []
    total_children = len(df)
    total_videos = total_children * len(TASKS)
    video_count = 0

    # Setup output file for incremental writes
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = False

    print(f"Processing {total_children} children, {total_videos} videos total...")

    for _, r in df.iterrows():
        child_id = str(r["child_id"])

        for task_name, col in TASKS.items():
            video_count += 1
            video_path = str(r[col])
            print(f"[{video_count}/{total_videos}] child={child_id} task={task_name}", flush=True)

            m = qc_single_video(
                video_path,
                sample_every_n=args.sample_every_n,
                max_frames=args.max_frames,
                face_landmarker=face_landmarker,
                pose_landmarker=pose_landmarker,
            )

            row = {
                "child_id": child_id,
                "task_type": task_name,
                "video_path": video_path,
                "duration_sec": round(m.duration_sec, 3),
                "fps_est": round(m.fps_est, 3),
                "frames_used": int(m.frames_used),
                "face_detected_ratio": round(m.face_detected_ratio, 4),
                "pose_detected_ratio": round(m.pose_detected_ratio, 4),
                "out_of_view_ratio": round(m.out_of_view_ratio, 4),
                "qc_pass": bool(m.passed),
                "qc_reason": m.reason,
            }
            rows.append(row)

            # Write incrementally to CSV
            row_df = pd.DataFrame([row])
            row_df.to_csv(out_path, mode="a", header=not header_written, index=False)
            header_written = True

    # Build summary from collected rows (CSV already written incrementally)
    qc_df = pd.DataFrame(rows)
    pass_counts = qc_df.groupby("child_id")["qc_pass"].sum().reset_index(name="tasks_passed")
    usable = pass_counts[pass_counts["tasks_passed"] >= args.min_tasks_pass].copy()
    unusable = pass_counts[pass_counts["tasks_passed"] < args.min_tasks_pass].copy()

    usable_path = out_path.parent / "usable_children.csv"
    unusable_path = out_path.parent / "unusable_children.csv"
    usable.to_csv(usable_path, index=False)
    unusable.to_csv(unusable_path, index=False)

    # Cleanup landmarkers
    face_landmarker.close()
    pose_landmarker.close()

    print(f"Wrote QC results: {out_path}")
    print(f"Usable children (>= {args.min_tasks_pass}/3 passing): {len(usable)}")
    print(f"Unusable children: {len(unusable)}")
    print(f"Wrote: {usable_path}")
    print(f"Wrote: {unusable_path}")


if __name__ == "__main__":
    main()
