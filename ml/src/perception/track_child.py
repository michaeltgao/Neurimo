from __future__ import annotations

import argparse
import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore


# ---------------------------
# Configuration dataclasses
# ---------------------------
@dataclass
class SmoothingConfig:
    """Configuration for bbox smoothing and gap handling."""
    enabled: bool = True
    method: str = "ema"  # "ema" or "kalman"
    ema_alpha: float = 0.3  # smoothing factor (0=no smoothing, 1=no memory)
    short_gap_threshold_sec: float = 0.5  # gaps <= this are interpolated


@dataclass
class VideoQualityReport:
    """Per-video quality metrics for downstream filtering/analysis."""
    detection_rate: float  # % frames with child detected
    longest_gap_sec: float  # longest detection gap in seconds
    longest_gap_frames: int  # longest gap in frame count
    avg_bbox_area: float  # mean bbox area (normalized 0-1, proxy for distance)
    bbox_area_std: float  # std of bbox area (stability proxy)
    fps: float
    duration_sec: float
    n_frames_processed: int
    width: int
    height: int
    aspect_ratio: float  # width/height
    is_vertical_video: bool  # height > width
    motion_blur_proxy: Optional[float] = None  # laplacian variance if computed
    parent_detection_rate: float = 0.0  # % frames with parent detected


# ---------------------------
# Model URLs + caching
# ---------------------------
POSE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

MODEL_CACHE_DIR = Path.home() / ".cache" / "mediapipe_models"


def _download_model(url: str, filename: str) -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_CACHE_DIR / filename
    if not model_path.exists():
        print(f"Downloading {filename}...")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ssl_context) as response:
            model_path.write_bytes(response.read())
        print(f"Downloaded to {model_path}")
    return model_path


def get_model_paths() -> Tuple[Path, Path]:
    pose_model = _download_model(POSE_LANDMARKER_URL, "pose_landmarker_lite.task")
    hand_model = _download_model(HAND_LANDMARKER_URL, "hand_landmarker.task")
    return pose_model, hand_model


# ---------------------------
# Helpers
# ---------------------------
def _lm_list_to_xyzw(lms: Any, expected_n: int) -> np.ndarray:
    """
    Convert MediaPipe landmark list into (expected_n, 4) float32: [x,y,z,vis].
    If visibility missing, defaults to 1.0.
    """
    out = np.full((expected_n, 4), np.nan, dtype=np.float32)
    if lms is None:
        return out

    n = min(len(lms), expected_n)
    for i in range(n):
        lm = lms[i]
        vis = getattr(lm, "visibility", None)
        vis = vis if vis is not None else 1.0
        out[i, 0] = float(lm.x)
        out[i, 1] = float(lm.y)
        out[i, 2] = float(lm.z)
        out[i, 3] = float(vis)
    return out


def _choose_best_pose(pose_result: Any) -> Tuple[Optional[Any], float]:
    """
    PoseLandmarker may return multiple poses. We pick the best by mean visibility.
    Returns raw landmark list + score.
    """
    if not pose_result or not getattr(pose_result, "pose_landmarks", None):
        return None, 0.0

    poses = pose_result.pose_landmarks  # list of landmark lists
    best = None
    best_score = -1.0
    for lms in poses:
        vis = [(v if (v := getattr(lm, "visibility", None)) is not None else 1.0) for lm in lms]
        score = float(np.mean(vis)) if vis else 0.0
        if score > best_score:
            best_score = score
            best = lms

    return best, float(best_score if best is not None else 0.0)


def _split_hands(hand_result: Any) -> Tuple[Optional[Any], Optional[Any], float]:
    """
    HandLandmarker returns up to N hands. If handedness exists, map to left/right.
    Returns raw landmark lists for left/right + mean handedness score.
    """
    if not hand_result or not getattr(hand_result, "hand_landmarks", None):
        return None, None, 0.0

    hands = hand_result.hand_landmarks
    handedness = getattr(hand_result, "handedness", None)

    left = None
    right = None
    scores = []

    for i, lms in enumerate(hands):
        label = None
        score_i = 0.5
        if handedness and i < len(handedness) and handedness[i]:
            top = handedness[i][0]
            label = getattr(top, "category_name", None) or getattr(top, "label", None)
            score_i = float(getattr(top, "score", 0.0) or 0.0)
        scores.append(score_i)

        if (label or "").lower() == "left":
            left = lms
        elif (label or "").lower() == "right":
            right = lms
        else:
            if left is None:
                left = lms
            elif right is None:
                right = lms

    return left, right, float(np.mean(scores) if scores else 0.0)


# ---------------------------
# Bbox utilities
# ---------------------------
def bbox_from_pose_landmarks(
    lms: Any,
    min_vis: float = 0.35,
    padding: Tuple[float, float] = (0.06, 0.08),
) -> Tuple[Optional[np.ndarray], float]:
    """
    Derive bounding box from pose landmarks.

    Args:
        lms: MediaPipe pose landmarks list
        min_vis: minimum visibility threshold
        padding: (pad_x, pad_y) to add around detected region

    Returns:
        (bbox_array, confidence) where bbox_array is [x0, y0, x1, y1] or None
        confidence is mean visibility of used landmarks
    """
    if lms is None:
        return None, 0.0

    xs, ys, vis_scores = [], [], []
    for lm in lms:
        v = float(getattr(lm, "visibility", 0.0) or 0.0)
        if v >= min_vis:
            xs.append(float(lm.x))
            ys.append(float(lm.y))
            vis_scores.append(v)

    if len(xs) < 8:
        return None, 0.0

    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    pad_x, pad_y = padding

    x0 = max(0.0, x0 - pad_x)
    x1 = min(1.0, x1 + pad_x)
    y0 = max(0.0, y0 - pad_y)
    y1 = min(1.0, y1 + pad_y)

    confidence = float(np.mean(vis_scores))
    return np.array([x0, y0, x1, y1], dtype=np.float32), confidence


def classify_pose_as_child_or_adult(lms: Any, min_vis: float = 0.30) -> Optional[str]:
    """
    Classify a detected pose as 'child' or 'adult' based on position.
    Child is generally centered; adult comes from sides/top/behind.

    Returns: "child", "adult", or None if insufficient landmarks
    """
    if lms is None:
        return None

    # Get bbox for position analysis
    bbox, conf = bbox_from_pose_landmarks(lms, min_vis=min_vis)
    if bbox is None:
        return None

    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    # Adult indicators: pose from edges or top
    # Left or right edges (adult reaching from side)
    if cx <= 0.20 or cx >= 0.80:
        return "adult"
    # Top region (adult reaching from above/behind)
    if cy <= 0.25:
        return "adult"
    # Upper corners (diagonal reach from behind)
    if cy <= 0.35 and (cx <= 0.25 or cx >= 0.75):
        return "adult"

    # Child is centered
    return "child"


def choose_child_and_parent_poses(
    poses: List[List[Any]],
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """
    From multiple detected poses, identify child and parent poses.

    Returns: (child_pose_landmarks, parent_pose_landmarks)
    """
    if not poses:
        return None, None

    if len(poses) == 1:
        # Single pose - classify based on position
        classification = classify_pose_as_child_or_adult(poses[0], min_vis=0.25)
        if classification == "adult":
            # If only pose is classified as adult, still treat as child for tracking
            return poses[0], None
        return poses[0], None

    # Multiple poses - classify each
    child_pose = None
    parent_pose = None

    for lms in poses:
        classification = classify_pose_as_child_or_adult(lms, min_vis=0.25)
        if classification == "child" and child_pose is None:
            child_pose = lms
        elif classification == "adult" and parent_pose is None:
            parent_pose = lms

    # Fallback: if no child found, use most centered pose
    if child_pose is None:
        scored: List[Tuple[float, int]] = []
        for i, lms in enumerate(poses):
            bbox, _ = bbox_from_pose_landmarks(lms, min_vis=0.25)
            if bbox is None:
                centered = 0.0
            else:
                cx = (bbox[0] + bbox[2]) / 2.0
                centered = 1.0 - abs(cx - 0.5)
            scored.append((centered, i))
        scored.sort(reverse=True)
        child_pose = poses[scored[0][1]]
        if len(scored) > 1 and parent_pose is None:
            parent_pose = poses[scored[1][1]]

    return child_pose, parent_pose


# ---------------------------
# Smoothing utilities
# ---------------------------
def smooth_bbox_ema(bbox_sequence: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply Exponential Moving Average smoothing to bbox sequence.
    Handles NaN values by not updating EMA state during gaps.

    Args:
        bbox_sequence: (N, 5) array of [x0, y0, x1, y1, conf]
        alpha: smoothing factor (0.3 = moderate smoothing)

    Returns:
        (N, 5) smoothed bbox sequence
    """
    if bbox_sequence.size == 0:
        return bbox_sequence

    N = bbox_sequence.shape[0]
    smoothed = np.full_like(bbox_sequence, np.nan, dtype=np.float32)

    # Convert to centroid/size representation
    raw_cx = (bbox_sequence[:, 0] + bbox_sequence[:, 2]) / 2
    raw_cy = (bbox_sequence[:, 1] + bbox_sequence[:, 3]) / 2
    raw_w = bbox_sequence[:, 2] - bbox_sequence[:, 0]
    raw_h = bbox_sequence[:, 3] - bbox_sequence[:, 1]
    raw_conf = bbox_sequence[:, 4]

    # Find first valid detection
    valid_mask = np.isfinite(raw_cx)
    first_valid_indices = np.where(valid_mask)[0]
    if len(first_valid_indices) == 0:
        return smoothed

    i0 = first_valid_indices[0]
    sm_cx, sm_cy = raw_cx[i0], raw_cy[i0]
    sm_w, sm_h = raw_w[i0], raw_h[i0]

    for i in range(i0, N):
        if np.isfinite(raw_cx[i]):
            # Update EMA
            sm_cx = alpha * raw_cx[i] + (1 - alpha) * sm_cx
            sm_cy = alpha * raw_cy[i] + (1 - alpha) * sm_cy
            sm_w = alpha * raw_w[i] + (1 - alpha) * sm_w
            sm_h = alpha * raw_h[i] + (1 - alpha) * sm_h

            # Reconstruct bbox
            smoothed[i, 0] = sm_cx - sm_w / 2
            smoothed[i, 1] = sm_cy - sm_h / 2
            smoothed[i, 2] = sm_cx + sm_w / 2
            smoothed[i, 3] = sm_cy + sm_h / 2
            smoothed[i, 4] = raw_conf[i]
        # else: keep NaN (gap)

    return smoothed


# ---------------------------
# Gap handling utilities
# ---------------------------
def detect_gaps(
    detected_flags: np.ndarray,
    t_sec: np.ndarray,
    short_gap_threshold_sec: float = 0.5,
) -> List[Tuple[int, int, float, str]]:
    """
    Find detection gaps in the sequence.

    Returns:
        List of (start_idx, end_idx, duration_sec, gap_type)
        where gap_type is "short" or "long"
    """
    if len(detected_flags) == 0:
        return []

    gaps = []
    in_gap = False
    gap_start = 0

    for i, detected in enumerate(detected_flags):
        if not detected and not in_gap:
            in_gap = True
            gap_start = i
        elif detected and in_gap:
            gap_end = i
            if gap_start > 0 and gap_end < len(t_sec):
                duration = float(t_sec[gap_end] - t_sec[gap_start])
            else:
                duration = float((gap_end - gap_start) / 30.0)  # fallback estimate
            gap_type = "short" if duration <= short_gap_threshold_sec else "long"
            gaps.append((gap_start, gap_end, duration, gap_type))
            in_gap = False

    # Handle gap at end
    if in_gap:
        gap_end = len(detected_flags)
        if gap_start > 0:
            duration = float(t_sec[-1] - t_sec[gap_start])
        else:
            duration = float((gap_end - gap_start) / 30.0)
        gap_type = "short" if duration <= short_gap_threshold_sec else "long"
        gaps.append((gap_start, gap_end, duration, gap_type))

    return gaps


def interpolate_short_gaps(
    bbox_sequence: np.ndarray,
    gaps: List[Tuple[int, int, float, str]],
    t_sec: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation for short gaps only. Long gaps remain NaN.

    Args:
        bbox_sequence: (N, 5) array
        gaps: list from detect_gaps
        t_sec: timestamps

    Returns:
        (N, 5) with short gaps interpolated
    """
    if bbox_sequence.size == 0:
        return bbox_sequence

    result = bbox_sequence.copy()

    for gap_start, gap_end, duration, gap_type in gaps:
        if gap_type != "short":
            continue

        # Need valid points before and after gap
        if gap_start == 0 or gap_end >= len(bbox_sequence):
            continue

        before_idx = gap_start - 1
        after_idx = gap_end

        if not np.isfinite(result[before_idx, 0]) or not np.isfinite(result[after_idx, 0]):
            continue

        # Linear interpolation for each coordinate
        for col in range(5):
            v0 = result[before_idx, col]
            v1 = result[after_idx, col]
            t0 = t_sec[before_idx]
            t1 = t_sec[after_idx]

            for i in range(gap_start, gap_end):
                if t1 - t0 > 1e-6:
                    alpha = (t_sec[i] - t0) / (t1 - t0)
                else:
                    alpha = 0.5
                result[i, col] = v0 + alpha * (v1 - v0)

    return result


# ---------------------------
# Quality report computation
# ---------------------------
def compute_quality_report(
    t_sec: np.ndarray,
    child_bbox: np.ndarray,
    parent_bbox: np.ndarray,
    video_info: Dict[str, Any],
) -> VideoQualityReport:
    """
    Compute per-video quality metrics.

    Args:
        t_sec: timestamps array
        child_bbox: (N, 5) child bbox array
        parent_bbox: (N, 5) parent bbox array
        video_info: dict with fps, width, height, duration_sec

    Returns:
        VideoQualityReport dataclass
    """
    N = len(t_sec) if len(t_sec) > 0 else 0
    fps = video_info.get("fps", 30.0)
    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    duration_sec = video_info.get("duration_sec", t_sec[-1] - t_sec[0] if N > 1 else 0.0)

    # Detection rate
    child_detected = np.isfinite(child_bbox[:, 0]) if child_bbox.size > 0 else np.array([])
    detection_rate = float(child_detected.sum()) / N if N > 0 else 0.0

    # Gap analysis
    gaps = detect_gaps(child_detected, t_sec, short_gap_threshold_sec=0.5)
    longest_gap_sec = max((g[2] for g in gaps), default=0.0)
    longest_gap_frames = max((g[1] - g[0] for g in gaps), default=0)

    # Bbox area stats
    valid_mask = child_detected
    if valid_mask.any():
        areas = (child_bbox[valid_mask, 2] - child_bbox[valid_mask, 0]) * \
                (child_bbox[valid_mask, 3] - child_bbox[valid_mask, 1])
        avg_bbox_area = float(np.mean(areas))
        bbox_area_std = float(np.std(areas))
    else:
        avg_bbox_area = 0.0
        bbox_area_std = 0.0

    # Aspect ratio
    aspect_ratio = width / height if height > 0 else 1.0
    is_vertical_video = height > width

    # Parent detection rate
    parent_detected = np.isfinite(parent_bbox[:, 0]) if parent_bbox.size > 0 else np.array([])
    parent_detection_rate = float(parent_detected.sum()) / N if N > 0 else 0.0

    return VideoQualityReport(
        detection_rate=detection_rate,
        longest_gap_sec=longest_gap_sec,
        longest_gap_frames=longest_gap_frames,
        avg_bbox_area=avg_bbox_area,
        bbox_area_std=bbox_area_std,
        fps=fps,
        duration_sec=duration_sec,
        n_frames_processed=N,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        is_vertical_video=is_vertical_video,
        motion_blur_proxy=None,  # optional, skip by default
        parent_detection_rate=parent_detection_rate,
    )


@dataclass
class TracksArrays:
    """Extended track arrays with bbox tracking and quality metrics."""
    # Existing fields (backward compatible)
    t_sec: np.ndarray  # (N,)
    pose: np.ndarray  # (N,33,4) child pose landmarks [x,y,z,vis]
    lh: np.ndarray  # (N,21,4) OR empty (0,21,4)
    rh: np.ndarray  # (N,21,4) OR empty (0,21,4)

    # New fields for bbox tracking
    child_bbox: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), dtype=np.float32))  # (N,5) [x0,y0,x1,y1,conf]
    parent_bbox: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), dtype=np.float32))  # (N,5) [x0,y0,x1,y1,conf]
    is_smoothed: bool = False


def extract_tracks_arrays_for_video(
    video_path: str,
    sample_every_n: int,
    max_frames: int,
    pose_landmarker: Any,
    hand_landmarker: Optional[Any],
    smoothing_config: Optional[SmoothingConfig] = None,
    compute_quality: bool = True,
) -> Tuple[TracksArrays, Optional[VideoQualityReport]]:
    """
    Enhanced extraction with bbox tracking, smoothing, and quality metrics.

    Args:
        video_path: path to video file
        sample_every_n: frame sampling rate
        max_frames: maximum frames to process
        pose_landmarker: MediaPipe pose landmarker (should be configured for num_poses=2)
        hand_landmarker: optional hand landmarker
        smoothing_config: smoothing parameters (None = no smoothing)
        compute_quality: whether to generate quality report

    Returns:
        (TracksArrays, VideoQualityReport) or (TracksArrays, None)
    """
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing video: {p}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {p}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t_list: List[float] = []
    pose_list: List[np.ndarray] = []
    lh_list: List[np.ndarray] = []
    rh_list: List[np.ndarray] = []
    child_bbox_list: List[np.ndarray] = []
    parent_bbox_list: List[np.ndarray] = []

    any_lh = False
    any_rh = False

    frames_used = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # sample
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

        # Pose detection (may return multiple poses)
        pose_res = pose_landmarker.detect(mp_image)
        poses = pose_res.pose_landmarks if pose_res and pose_res.pose_landmarks else []

        # Choose child and parent poses
        child_pose_raw, parent_pose_raw = choose_child_and_parent_poses(poses)

        # Convert to arrays
        pose_arr = _lm_list_to_xyzw(child_pose_raw, expected_n=33)

        # Extract bboxes
        child_bbox_arr = np.full((5,), np.nan, dtype=np.float32)
        parent_bbox_arr = np.full((5,), np.nan, dtype=np.float32)

        if child_pose_raw is not None:
            bbox, conf = bbox_from_pose_landmarks(child_pose_raw, min_vis=0.35)
            if bbox is not None:
                child_bbox_arr[:4] = bbox
                child_bbox_arr[4] = conf

        if parent_pose_raw is not None:
            bbox, conf = bbox_from_pose_landmarks(parent_pose_raw, min_vis=0.35)
            if bbox is not None:
                parent_bbox_arr[:4] = bbox
                parent_bbox_arr[4] = conf

        # Hands
        lh_arr = np.full((21, 4), np.nan, dtype=np.float32)
        rh_arr = np.full((21, 4), np.nan, dtype=np.float32)

        if hand_landmarker is not None:
            hand_res = hand_landmarker.detect(mp_image)
            lh_raw, rh_raw, _hs = _split_hands(hand_res)
            if lh_raw is not None:
                lh_arr = _lm_list_to_xyzw(lh_raw, expected_n=21)
                any_lh = True
            if rh_raw is not None:
                rh_arr = _lm_list_to_xyzw(rh_raw, expected_n=21)
                any_rh = True

        t_list.append(float(t_sec))
        pose_list.append(pose_arr)
        lh_list.append(lh_arr)
        rh_list.append(rh_arr)
        child_bbox_list.append(child_bbox_arr)
        parent_bbox_list.append(parent_bbox_arr)

    cap.release()

    # Stack arrays
    t = np.asarray(t_list, dtype=np.float32)
    pose = np.stack(pose_list, axis=0).astype(np.float32) if pose_list else np.zeros((0, 33, 4), dtype=np.float32)
    lh_full = np.stack(lh_list, axis=0).astype(np.float32) if lh_list else np.zeros((0, 21, 4), dtype=np.float32)
    rh_full = np.stack(rh_list, axis=0).astype(np.float32) if rh_list else np.zeros((0, 21, 4), dtype=np.float32)
    child_bbox = np.stack(child_bbox_list, axis=0).astype(np.float32) if child_bbox_list else np.zeros((0, 5), dtype=np.float32)
    parent_bbox = np.stack(parent_bbox_list, axis=0).astype(np.float32) if parent_bbox_list else np.zeros((0, 5), dtype=np.float32)

    # If no hands EVER detected, store empty arrays as requested
    lh = lh_full if any_lh else np.zeros((0, 21, 4), dtype=np.float32)
    rh = rh_full if any_rh else np.zeros((0, 21, 4), dtype=np.float32)

    # Apply smoothing and gap handling
    is_smoothed = False
    if smoothing_config is not None and smoothing_config.enabled and child_bbox.size > 0:
        # Detect gaps for interpolation
        child_detected = np.isfinite(child_bbox[:, 0])
        gaps = detect_gaps(child_detected, t, smoothing_config.short_gap_threshold_sec)

        # Interpolate short gaps first
        child_bbox = interpolate_short_gaps(child_bbox, gaps, t)

        # Apply smoothing
        if smoothing_config.method == "ema":
            child_bbox = smooth_bbox_ema(child_bbox, alpha=smoothing_config.ema_alpha)
        # kalman can be added here later

        # Same for parent if detected
        if parent_bbox.size > 0 and np.isfinite(parent_bbox[:, 0]).any():
            parent_detected = np.isfinite(parent_bbox[:, 0])
            parent_gaps = detect_gaps(parent_detected, t, smoothing_config.short_gap_threshold_sec)
            parent_bbox = interpolate_short_gaps(parent_bbox, parent_gaps, t)
            if smoothing_config.method == "ema":
                parent_bbox = smooth_bbox_ema(parent_bbox, alpha=smoothing_config.ema_alpha)

        is_smoothed = True

    tracks = TracksArrays(
        t_sec=t,
        pose=pose,
        lh=lh,
        rh=rh,
        child_bbox=child_bbox,
        parent_bbox=parent_bbox,
        is_smoothed=is_smoothed,
    )

    # Compute quality report
    quality_report = None
    if compute_quality:
        video_info = {
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        }
        quality_report = compute_quality_report(t, child_bbox, parent_bbox, video_info)

    return tracks, quality_report


def save_tracks_npz(out_path: Path, arr: TracksArrays) -> None:
    """Save TracksArrays to compressed NPZ file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        # Existing fields (backward compatible)
        t_sec=arr.t_sec,
        pose=arr.pose,
        lh=arr.lh,
        rh=arr.rh,
        # New fields
        child_bbox=arr.child_bbox,
        parent_bbox=arr.parent_bbox,
        is_smoothed=np.array([arr.is_smoothed]),  # store as array for NPZ compatibility
    )


# ---------------------------
# CLI
# ---------------------------
TASK_TO_COL = {
    "joint_attention": "joint_attention_path",
    "imitation": "imitation_path",
    "free_play": "free_play_path",
}


def main():
    ap = argparse.ArgumentParser(
        description="Export per-video landmarks + bboxes to NPZ for feature extraction."
    )
    # Existing args
    ap.add_argument("--manifest", required=True, help="CSV with child_id + *_path columns.")
    ap.add_argument("--task", required=True, choices=list(TASK_TO_COL.keys()))
    ap.add_argument("--out_dir", default="data/derived/tracks", help="Output directory for .npz files.")
    ap.add_argument("--sample_every_n", type=int, default=2)
    ap.add_argument("--max_frames", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N rows.")
    ap.add_argument("--overwrite", action="store_true")

    # New args for bbox tracking and quality
    ap.add_argument("--num_poses", type=int, default=2,
                    help="Number of poses to detect (2=child+parent)")
    ap.add_argument("--no_smoothing", action="store_true",
                    help="Disable bbox smoothing")
    ap.add_argument("--smoothing_method", default="ema", choices=["ema", "kalman"],
                    help="Smoothing method for bbox tracks")
    ap.add_argument("--ema_alpha", type=float, default=0.3,
                    help="EMA smoothing factor (0=no smoothing, 1=no memory)")
    ap.add_argument("--short_gap_threshold", type=float, default=0.5,
                    help="Gaps <= this (sec) are interpolated")
    ap.add_argument("--quality_report_out", default=None,
                    help="Output CSV path for per-video quality reports")
    ap.add_argument("--no_quality", action="store_true",
                    help="Skip quality report computation")

    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    col = TASK_TO_COL[args.task]

    if col not in df.columns:
        raise ValueError(f"Manifest missing column '{col}'. Present: {list(df.columns)}")

    # Build smoothing config
    smoothing_config = None
    if not args.no_smoothing:
        smoothing_config = SmoothingConfig(
            enabled=True,
            method=args.smoothing_method,
            ema_alpha=args.ema_alpha,
            short_gap_threshold_sec=args.short_gap_threshold,
        )

    pose_model, hand_model = get_model_paths()

    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=args.num_poses,  # detect multiple poses for child+parent
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    print(f"Exporting tracks: task={args.task} videos={n} out_dir={out_dir}")
    if smoothing_config:
        print(f"  Smoothing: {smoothing_config.method} (alpha={smoothing_config.ema_alpha})")
    else:
        print("  Smoothing: disabled")

    # Collect quality reports
    quality_reports: List[Dict[str, Any]] = []

    for i in range(n):
        child_id = str(df.iloc[i]["child_id"])
        video_path = str(df.iloc[i][col])

        out_path = out_dir / f"{child_id}_{args.task}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[{i+1}/{n}] skip existing: {out_path.name}", flush=True)
            continue

        print(f"[{i+1}/{n}] child={child_id} task={args.task}", flush=True)

        try:
            arr, quality_report = extract_tracks_arrays_for_video(
                video_path=video_path,
                sample_every_n=args.sample_every_n,
                max_frames=args.max_frames,
                pose_landmarker=pose_landmarker,
                hand_landmarker=hand_landmarker,
                smoothing_config=smoothing_config,
                compute_quality=not args.no_quality,
            )
            save_tracks_npz(out_path, arr)

            # Collect quality report
            if quality_report is not None:
                report_dict = {
                    "child_id": child_id,
                    "task": args.task,
                    "detection_rate": quality_report.detection_rate,
                    "longest_gap_sec": quality_report.longest_gap_sec,
                    "longest_gap_frames": quality_report.longest_gap_frames,
                    "avg_bbox_area": quality_report.avg_bbox_area,
                    "bbox_area_std": quality_report.bbox_area_std,
                    "fps": quality_report.fps,
                    "duration_sec": quality_report.duration_sec,
                    "n_frames_processed": quality_report.n_frames_processed,
                    "width": quality_report.width,
                    "height": quality_report.height,
                    "aspect_ratio": quality_report.aspect_ratio,
                    "is_vertical_video": quality_report.is_vertical_video,
                    "parent_detection_rate": quality_report.parent_detection_rate,
                }
                quality_reports.append(report_dict)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue

    pose_landmarker.close()
    hand_landmarker.close()

    # Save quality reports CSV
    if args.quality_report_out and quality_reports:
        qr_path = Path(args.quality_report_out)
        qr_path.parent.mkdir(parents=True, exist_ok=True)
        qr_df = pd.DataFrame(quality_reports)
        qr_df.to_csv(qr_path, index=False)
        print(f"Quality report saved to: {qr_path}")

    print("Done.")


if __name__ == "__main__":
    main()
