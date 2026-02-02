from __future__ import annotations

import argparse
import csv
import math
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore


# ---------------------------
# Model URLs + caching
# ---------------------------
HAND_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_CACHE_DIR = Path.home() / ".cache" / "mediapipe_models"


def _download_model(url: str, filename: str) -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_CACHE_DIR / filename
    if not model_path.exists():
        print(f"Downloading {filename}...")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ssl_context) as resp:
            model_path.write_bytes(resp.read())
        print(f"Downloaded to {model_path}")
    return model_path


def get_model_paths() -> Tuple[Path, Path]:
    hand_model = _download_model(HAND_LANDMARKER_URL, "hand_landmarker.task")
    pose_model = _download_model(POSE_LANDMARKER_URL, "pose_landmarker_lite.task")
    return hand_model, pose_model


# ---------------------------
# Geometry helpers
# ---------------------------
def _angle_deg(dx: float, dy: float) -> float:
    # angle in image coords, 0 = right, 90 = down
    return float((math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0)


def _circular_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _in_adult_region(x: float, y: float, mode: str) -> bool:
    """
    Adult hands tend to enter from top or sides.
    Normalized coords in [0,1].
    mode:
      - "top_or_side": y <= y_max OR x <= side_x OR x >= 1-side_x
      - "top_only": y <= y_max
      - "none": always True
    """
    if mode == "none":
        return True

    y_max = 0.62
    side_x = 0.18

    if mode == "top_only":
        return y <= y_max

    return (y <= y_max) or (x <= side_x) or (x >= 1.0 - side_x)


@dataclass
class PointSample:
    t_sec: float
    angle_deg: float
    conf: float  # proxy confidence


@dataclass
class PointSegment:
    t_start: float
    t_end: float
    angle_deg: float
    stability: float
    n_samples: int


# ---------------------------
# Hand vector extraction (STRICT)
# ---------------------------
def _hand_point_vector(
    hand_result: Any,
    adult_region: str,
    min_vec_len: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Returns best hand pointing vector (dx, dy, conf, wrist_x, wrist_y) or None.

    Strong gating to avoid child hand false positives:
      - wrist must be in adult region (top or side)
      - index_tip - wrist vector must be long enough (pointiness proxy)
    """
    if not hand_result or not getattr(hand_result, "hand_landmarks", None):
        return None

    hands = hand_result.hand_landmarks
    handedness = getattr(hand_result, "handedness", None)

    best = None
    best_score = -1.0

    for i, lms in enumerate(hands):
        wrist = lms[0]
        tip = lms[8]  # index_tip

        wx, wy = float(wrist.x), float(wrist.y)
        if not _in_adult_region(wx, wy, adult_region):
            continue

        dx = float(tip.x - wrist.x)
        dy = float(tip.y - wrist.y)
        mag = math.hypot(dx, dy)

        # Pointiness gate
        if mag < min_vec_len:
            continue

        conf = 0.5
        if handedness and i < len(handedness) and handedness[i]:
            top = handedness[i][0]
            conf = float(getattr(top, "score", 0.0) or 0.0)

        score = conf * mag
        if score > best_score:
            best_score = score
            best = (dx, dy, conf, wx, wy)

    return best


# ---------------------------
# Segment building (stable angles)
# ---------------------------
def _finalize_segment(seg: List[PointSample], max_angle_jitter_deg: float) -> PointSegment:
    angles = [s.angle_deg for s in seg]
    angle = float(np.median(angles))
    diffs = [_circular_diff_deg(a, angle) for a in angles]
    mad = float(np.median(diffs))
    stability = float(max(0.0, 1.0 - (mad / max_angle_jitter_deg)))
    return PointSegment(
        t_start=float(seg[0].t_sec),
        t_end=float(seg[-1].t_sec),
        angle_deg=angle,
        stability=stability,
        n_samples=len(seg),
    )


def detect_point_segments(
    samples: List[PointSample],
    min_frames: int,
    max_angle_jitter_deg: float,
    max_gap_sec: float,
    merge_gap_sec: float,
) -> List[PointSegment]:
    if not samples:
        return []

    samples = sorted(samples, key=lambda s: s.t_sec)

    segments: List[PointSegment] = []
    current: List[PointSample] = [samples[0]]

    for s in samples[1:]:
        prev = current[-1]
        if (s.t_sec - prev.t_sec) > max_gap_sec:
            if len(current) >= min_frames:
                segments.append(_finalize_segment(current, max_angle_jitter_deg))
            current = [s]
            continue

        base_angle = float(np.median([x.angle_deg for x in current]))
        if _circular_diff_deg(s.angle_deg, base_angle) <= max_angle_jitter_deg:
            current.append(s)
        else:
            if len(current) >= min_frames:
                segments.append(_finalize_segment(current, max_angle_jitter_deg))
            current = [s]

    if len(current) >= min_frames:
        segments.append(_finalize_segment(current, max_angle_jitter_deg))

    # Merge adjacent segments if close
    if not segments:
        return []

    merged: List[PointSegment] = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if (seg.t_start - last.t_end) <= merge_gap_sec and _circular_diff_deg(seg.angle_deg, last.angle_deg) <= max_angle_jitter_deg:
            total_n = last.n_samples + seg.n_samples
            new_angle = float(np.median([last.angle_deg, seg.angle_deg]))
            new_stability = float((last.stability * last.n_samples + seg.stability * seg.n_samples) / max(1, total_n))
            merged[-1] = PointSegment(
                t_start=last.t_start,
                t_end=seg.t_end,
                angle_deg=new_angle,
                stability=new_stability,
                n_samples=total_n,
            )
        else:
            merged.append(seg)

    return merged


# ---------------------------
# Video processing (HAND-ONLY by default)
# ---------------------------
def process_video_hand_only(
    video_path: Path,
    sample_every_n: int,
    max_frames: int,
    hand_landmarker: Any,
    adult_region: str,
    min_vec_len: float,
) -> List[PointSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0

    out: List[PointSample] = []
    frames_used = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if sample_every_n > 1 and (frame_idx % sample_every_n != 0):
            frame_idx += 1
            continue

        t_sec = frame_idx / fps
        frame_idx += 1
        frames_used += 1
        if frames_used > max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            hand_res = hand_landmarker.detect(mp_image)
        except Exception:
            continue

        hv = _hand_point_vector(hand_res, adult_region=adult_region, min_vec_len=min_vec_len)
        if hv is None:
            continue

        dx, dy, conf, _wx, _wy = hv
        out.append(PointSample(float(t_sec), _angle_deg(dx, dy), float(conf)))

    cap.release()
    return out


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Detect adult pointing segments (HAND-ONLY, strict adult region gating).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", default="joint_attention", choices=["joint_attention", "imitation", "free_play"])
    ap.add_argument("--out", default="data/derived/point_events.csv")

    ap.add_argument("--sample_every_n", type=int, default=2)
    ap.add_argument("--max_frames", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--adult_region", default="top_or_side", choices=["top_or_side", "top_only", "none"])
    ap.add_argument("--min_vec_len", type=float, default=0.10, help="Index_tip-wrist min length (normalized). Increase to reduce false positives.")

    ap.add_argument("--min_frames", type=int, default=8)
    ap.add_argument("--max_angle_jitter_deg", type=float, default=10.0)
    ap.add_argument("--max_gap_sec", type=float, default=0.30)
    ap.add_argument("--merge_gap_sec", type=float, default=0.50)

    # If you later truly have adult pose in frame, you can enable pose fallback.
    ap.add_argument("--enable_pose_fallback", action="store_true", help="(Not recommended) Enable pose fallback.")

    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    col = {
        "joint_attention": "joint_attention_path",
        "imitation": "imitation_path",
        "free_play": "free_play_path",
    }[args.task]
    if col not in df.columns:
        raise ValueError(f"Manifest missing column '{col}'. Present: {list(df.columns)}")

    hand_model, pose_model = get_model_paths()

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    # Pose fallback is intentionally not used by default.
    pose_landmarker = None
    if args.enable_pose_fallback:
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["child_id", "task_type", "t_start", "t_end", "angle_deg", "method", "stability", "n_samples"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    for i in range(n):
        child_id = str(df.iloc[i]["child_id"])
        video_path = Path(str(df.iloc[i][col]))
        if not video_path.exists():
            continue

        print(f"[{i+1}/{n}] child={child_id} task={args.task} video={video_path.name}", flush=True)

        # HAND-ONLY detection
        samples = process_video_hand_only(
            video_path=video_path,
            sample_every_n=args.sample_every_n,
            max_frames=args.max_frames,
            hand_landmarker=hand_landmarker,
            adult_region=args.adult_region,
            min_vec_len=args.min_vec_len,
        )

        segs = detect_point_segments(
            samples=samples,
            min_frames=args.min_frames,
            max_angle_jitter_deg=args.max_angle_jitter_deg,
            max_gap_sec=args.max_gap_sec,
            merge_gap_sec=args.merge_gap_sec,
        )

        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            for s in segs:
                w.writerow(
                    {
                        "child_id": child_id,
                        "task_type": args.task,
                        "t_start": round(s.t_start, 3),
                        "t_end": round(s.t_end, 3),
                        "angle_deg": round(s.angle_deg, 2),
                        "method": "hand",
                        "stability": round(s.stability, 4),
                        "n_samples": int(s.n_samples),
                    }
                )

    hand_landmarker.close()
    if pose_landmarker is not None:
        pose_landmarker.close()

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

