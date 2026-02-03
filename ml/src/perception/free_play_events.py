"""
Free-play behavioral event detection from pre-computed tracks.

Loads tracks from trackchild.py npz files and produces:
- free_play_events.csv: time-stamped behavioral events with confidence
- free_play_summary.csv: per-video feature aggregates

Event types:
A) Engagement & regulation (child-centered):
   - OFF_SCREEN: child bbox missing for extended periods
   - BODY_ACTIVITY_LEVEL: low/med/high based on pose velocity
   - PERIODIC_MOTION: periodic oscillations in hands/torso (conservative)

B) Social interaction:
   - PARENT_PRESENT: parent bbox exists
   - CLOSE_PROXIMITY: child-parent bbox overlap/distance

C) Manual interaction:
   - HAND_ACTIVE: hands present + high velocity in workspace
   - HAND_TO_FACE: wrist close to face landmarks
   - HANDS_NEAR: left/right wrists close together with motion
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
@dataclass
class EventConfig:
    """Configuration for event detection thresholds."""
    # Off-screen detection
    off_screen_min_dur: float = 0.5  # minimum gap duration to create OFF_SCREEN event

    # Activity level thresholds (normalized coords/sec)
    activity_low_threshold: float = 0.02
    activity_high_threshold: float = 0.05
    activity_min_dur: float = 0.5  # minimum duration for activity burst events
    activity_ema_window: int = 3  # EMA smoothing window

    # Repetitive motion detection
    rep_freq_min: float = 1.5  # Hz
    rep_freq_max: float = 5.5  # Hz
    rep_normalized_threshold: float = 0.15  # normalized periodicity score (scale-invariant)
    rep_min_amplitude: float = 0.006
    rep_min_dur: float = 1.0  # conservative: require sustained repetition
    rep_window_sec: float = 2.0  # window for local periodicity detection

    # Parent presence
    parent_min_dur: float = 0.5

    # Close proximity
    proximity_iou_threshold: float = 0.05  # IoU threshold for "close"
    proximity_center_dist_threshold: float = 0.25  # normalized distance threshold
    proximity_min_dur: float = 0.5

    # Hand active detection
    hand_speed_threshold: float = 0.04  # normalized coords/sec
    hand_workspace_y_min: float = 0.4  # hands must be in lower portion (workspace)
    hand_active_min_dur: float = 0.3

    # Hand to face
    hand_face_distance_threshold: float = 0.12  # normalized distance
    hand_face_min_dur: float = 0.2

    # Hands together (clap)
    hands_together_threshold: float = 0.10  # wrist-to-wrist distance
    hands_together_min_dur: float = 0.15


# ============================================================
# Utility functions
# ============================================================
def nanmed_smooth(x: np.ndarray, win: int = 2) -> np.ndarray:
    """Apply median smoothing, handling NaNs."""
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=float)
    n = x.size
    for i in range(n):
        j0 = max(0, i - win)
        j1 = min(n, i + win + 1)
        out[i] = float(np.nanmedian(x[j0:j1]))
    return out


def ema_smooth(x: np.ndarray, alpha: float = 0.3, max_hold_frames: int = 2) -> np.ndarray:
    """
    Exponential moving average smoothing, handling NaNs.

    Args:
        x: Input signal
        alpha: Smoothing factor (0=no smoothing, 1=no memory)
        max_hold_frames: Maximum consecutive NaN frames to hold value (default=2).
                         Beyond this, output NaN to avoid fake activity during long gaps.
    """
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return np.full_like(x, np.nan)

    # Find first valid
    first_valid = np.where(valid)[0][0]
    state = x[first_valid]
    gap_count = 0

    for i in range(x.size):
        if np.isfinite(x[i]):
            state = alpha * x[i] + (1 - alpha) * state
            out[i] = state
            gap_count = 0
        else:
            gap_count += 1
            if gap_count <= max_hold_frames:
                out[i] = state  # hold for very short gaps only
            else:
                out[i] = np.nan  # output NaN for longer gaps
    return out


def interp_nans(x: np.ndarray, min_valid_frac: float = 0.10) -> np.ndarray:
    """Interpolate NaN values, returning all NaN if insufficient valid points."""
    if x.size == 0:
        return x
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if good.sum() < max(8, int(min_valid_frac * x.size)):
        return np.full_like(x, np.nan, dtype=float)
    out = x.astype(float).copy()
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def compute_bbox_visibility(
    bbox: np.ndarray, min_conf: float = 0.2, min_area: float = 1e-4
) -> np.ndarray:
    """
    Compute per-frame visibility from bbox array.

    Uses confidence AND area to determine visibility, not just isfinite.
    This handles cases where bbox defaults to zeros for missing data.

    Args:
        bbox: (N, 5) array [x0, y0, x1, y1, conf]
        min_conf: Minimum confidence threshold
        min_area: Minimum bbox area (normalized, e.g., 1e-4)

    Returns:
        (N,) boolean visibility array
    """
    if bbox.size == 0:
        return np.zeros(0, dtype=bool)

    # Check for valid coordinates (not NaN)
    coords_valid = np.isfinite(bbox[:, 0]) & np.isfinite(bbox[:, 2])

    # Check confidence
    conf = bbox[:, 4] if bbox.shape[1] > 4 else np.ones(len(bbox))
    conf_ok = conf > min_conf

    # Check area (handles zeros-for-missing case)
    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    area_ok = area > min_area

    return coords_valid & conf_ok & area_ok


def get_missing_mask(original: np.ndarray) -> np.ndarray:
    """
    Get mask of frames where original data was missing (NaN).
    Used to invalidate computed values during tracking gaps.
    """
    if original.ndim == 1:
        return ~np.isfinite(original)
    elif original.ndim == 2:
        return ~np.isfinite(original).any(axis=1)
    else:
        # For higher dims, check first element
        return ~np.isfinite(original.reshape(original.shape[0], -1)).any(axis=1)


def spectral_peak(
    signal: np.ndarray, fps: float, fmin: float, fmax: float, min_amplitude: float = 0.005
) -> Tuple[float, float, float]:
    """
    Find dominant frequency in signal within [fmin, fmax] Hz band.

    Returns (frequency, raw_power, normalized_score).
    normalized_score = peak_power / (variance + eps), which is scale-invariant.
    Returns (0, 0, 0) if insufficient signal.
    """
    if signal.size < 30 or not np.isfinite(signal).all():
        return 0.0, 0.0, 0.0

    variance = float(np.var(signal))
    amplitude = float(np.std(signal))
    if amplitude < min_amplitude:
        return 0.0, 0.0, 0.0

    s = signal - float(np.mean(signal))
    window = np.hanning(s.size)
    y = np.fft.rfft(s * window)
    freqs = np.fft.rfftfreq(s.size, d=1.0 / fps)
    power = np.abs(y) ** 2
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    k = int(np.argmax(power[mask]))
    peak_power = float(power[mask][k])
    peak_freq = float(freqs[mask][k])

    # Normalized score: power relative to signal variance (scale-invariant)
    eps = 1e-8
    normalized_score = peak_power / (variance * signal.size + eps)

    return peak_freq, peak_power, normalized_score


def windowed_periodicity(
    signal: np.ndarray,
    fps: float,
    window_sec: float,
    freq_min: float,
    freq_max: float,
    normalized_threshold: float = 0.15,
) -> np.ndarray:
    """
    Detect periodicity in sliding windows using normalized score.

    Args:
        signal: Input signal
        fps: Frames per second
        window_sec: Window size in seconds
        freq_min, freq_max: Frequency band to search
        normalized_threshold: Threshold for normalized periodicity score (scale-invariant)

    Returns:
        Per-frame boolean flags indicating periodic segments.
    """
    n = signal.size
    win_frames = max(int(window_sec * fps), 30)
    flags = np.zeros(n, dtype=bool)

    if n < win_frames:
        return flags

    for i in range(n - win_frames + 1):
        chunk = signal[i : i + win_frames]
        if not np.isfinite(chunk).all():
            continue
        freq, _power, norm_score = spectral_peak(chunk, fps, freq_min, freq_max)
        if norm_score > normalized_threshold and freq > 0:
            center = i + win_frames // 2
            radius = win_frames // 4
            flags[max(0, center - radius) : min(n, center + radius)] = True

    return flags


def segments_from_bool(
    t: np.ndarray, flags: np.ndarray, min_dur: float
) -> List[Tuple[float, float]]:
    """Convert boolean flag array to list of (start, end) time segments."""
    if t.size == 0:
        return []
    segs: List[Tuple[float, float]] = []
    in_seg = False
    s0 = 0.0
    for ti, fi in zip(t, flags):
        if fi and not in_seg:
            in_seg = True
            s0 = float(ti)
        if (not fi) and in_seg:
            e0 = float(ti)
            if e0 - s0 >= min_dur:
                segs.append((s0, e0))
            in_seg = False
    if in_seg:
        e0 = float(t[-1])
        if e0 - s0 >= min_dur:
            segs.append((s0, e0))
    return segs


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bboxes [x0, y0, x1, y1, ...]."""
    if not np.isfinite(bbox1[0]) or not np.isfinite(bbox2[0]):
        return 0.0

    x0_1, y0_1, x1_1, y1_1 = bbox1[:4]
    x0_2, y0_2, x1_2, y1_2 = bbox2[:4]

    # Intersection
    xi0 = max(x0_1, x0_2)
    yi0 = max(y0_1, y0_2)
    xi1 = min(x1_1, x1_2)
    yi1 = min(y1_1, y1_2)

    inter_w = max(0.0, xi1 - xi0)
    inter_h = max(0.0, yi1 - yi0)
    inter_area = inter_w * inter_h

    # Union
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - inter_area

    if union_area < 1e-8:
        return 0.0
    return float(inter_area / union_area)


def bbox_center_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute center-to-center distance between bboxes."""
    if not np.isfinite(bbox1[0]) or not np.isfinite(bbox2[0]):
        return float("inf")
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def compute_speed(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute per-frame speed from (N, 2) xy coordinates.
    Returns (N,) array with speed[0] = speed[1].
    """
    if xy.shape[0] < 2:
        return np.zeros(max(1, xy.shape[0]), dtype=float)

    dt = np.diff(t)
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.sqrt(dx * dx + dy * dy) / np.maximum(dt, 1e-6)
    # Pad to match length
    speed = np.concatenate([[speed[0] if speed.size else 0.0], speed])
    return speed.astype(float)


# ============================================================
# Event dataclass
# ============================================================
@dataclass
class EventSeg:
    """Single behavioral event segment."""
    child_id: str
    task_type: str
    event_type: str
    t_start: float
    t_end: float
    confidence: float
    meta: str


# ============================================================
# Track loading
# ============================================================
@dataclass
class TracksData:
    """Loaded track data from npz file."""
    t_sec: np.ndarray  # (N,)
    pose: np.ndarray  # (N, 33, 4) [x, y, z, vis]
    lh: np.ndarray  # (N, 21, 4) or (0, 21, 4)
    rh: np.ndarray  # (N, 21, 4) or (0, 21, 4)
    child_bbox: np.ndarray  # (N, 5) [x0, y0, x1, y1, conf]
    parent_bbox: np.ndarray  # (N, 5) [x0, y0, x1, y1, conf]
    fps: float
    sample_every_n: int


def load_tracks(npz_path: Path) -> Optional[TracksData]:
    """Load track data from npz file."""
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path)
        t_sec = data["t_sec"]
        pose = data["pose"]
        lh = data.get("lh", np.zeros((0, 21, 4), dtype=np.float32))
        rh = data.get("rh", np.zeros((0, 21, 4), dtype=np.float32))
        child_bbox = data.get("child_bbox", np.zeros((len(t_sec), 5), dtype=np.float32))
        parent_bbox = data.get("parent_bbox", np.zeros((len(t_sec), 5), dtype=np.float32))

        # Handle metadata
        fps_arr = data.get("fps", np.array([30.0]))
        fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0
        sample_arr = data.get("sample_every_n", np.array([1]))
        sample_every_n = int(sample_arr[0]) if sample_arr.size > 0 else 1

        return TracksData(
            t_sec=t_sec.astype(float),
            pose=pose.astype(float),
            lh=lh.astype(float),
            rh=rh.astype(float),
            child_bbox=child_bbox.astype(float),
            parent_bbox=parent_bbox.astype(float),
            fps=fps,
            sample_every_n=sample_every_n,
        )
    except Exception as e:
        print(f"  Error loading {npz_path}: {e}")
        return None


# ============================================================
# Signal computation
# ============================================================
def compute_frame_signals(tracks: TracksData) -> Dict[str, np.ndarray]:
    """
    Compute frame-level signals from track data.

    Returns dict with:
    - child_visible: (N,) bool
    - parent_visible: (N,) bool
    - pose_speed: (N,) float - median speed of stable landmarks (NaN during gaps)
    - lh_speed, rh_speed: (N,) float - wrist speeds (NaN during gaps)
    - lh_visible, rh_visible: (N,) bool
    - lh_wrist_xy, rh_wrist_xy: (N, 2) float
    - nose_xy: (N, 2) float
    - shoulder_mid_xy: (N, 2) float
    - hip_mid_xy: (N, 2) float
    """
    n = len(tracks.t_sec)
    t = tracks.t_sec
    pose = tracks.pose  # (N, 33, 4)

    # Child/parent visibility using confidence + area (handles zeros-for-missing case)
    child_visible = compute_bbox_visibility(tracks.child_bbox, min_conf=0.2, min_area=1e-4)
    parent_visible = compute_bbox_visibility(tracks.parent_bbox, min_conf=0.2, min_area=1e-4)

    # Stable pose landmarks for activity: shoulders (11, 12), hips (23, 24), nose (0)
    # These are less noisy than wrists for overall body activity
    stable_idx = [0, 11, 12, 23, 24]  # nose, shoulders, hips
    stable_xy = pose[:, stable_idx, :2]  # (N, 5, 2)

    # Track which frames had valid original pose data (before interpolation)
    pose_was_valid = np.isfinite(stable_xy[:, 0, 0])  # use nose as proxy

    # Compute per-landmark speed and take median
    landmark_speed_list: List[np.ndarray] = []
    for k in range(len(stable_idx)):
        xy_k = stable_xy[:, k, :]  # (N, 2)
        xy_k = np.where(np.isfinite(xy_k), xy_k, np.nan)
        # Interpolate for speed calculation
        xy_interp = np.column_stack([interp_nans(xy_k[:, 0]), interp_nans(xy_k[:, 1])])
        speed_k = compute_speed(xy_interp, t)
        landmark_speed_list.append(speed_k)
    landmark_speeds = np.stack(landmark_speed_list, axis=1)  # (N, 5)
    pose_speed = np.nanmedian(landmark_speeds, axis=1)

    # Invalidate speed during tracking gaps (where original data was missing)
    # This prevents fake activity from interpolation through long gaps
    pose_speed[~pose_was_valid] = np.nan

    pose_speed = ema_smooth(pose_speed, alpha=0.4, max_hold_frames=2)

    # Hand visibility and positions
    has_lh = tracks.lh.shape[0] > 0
    has_rh = tracks.rh.shape[0] > 0

    lh_visible = np.zeros(n, dtype=bool)
    rh_visible = np.zeros(n, dtype=bool)
    lh_wrist_xy = np.full((n, 2), np.nan, dtype=float)
    rh_wrist_xy = np.full((n, 2), np.nan, dtype=float)
    lh_speed = np.full(n, np.nan, dtype=float)  # NaN by default, not zeros
    rh_speed = np.full(n, np.nan, dtype=float)

    if has_lh and tracks.lh.shape[0] == n:
        lh_wrist_xy = tracks.lh[:, 0, :2]  # wrist is landmark 0
        lh_visible = np.isfinite(lh_wrist_xy[:, 0])
        lh_interp = np.column_stack([interp_nans(lh_wrist_xy[:, 0]), interp_nans(lh_wrist_xy[:, 1])])
        lh_speed = compute_speed(lh_interp, t)
        # Invalidate speed where original data was missing
        lh_speed[~lh_visible] = np.nan
        lh_speed = ema_smooth(lh_speed, alpha=0.4, max_hold_frames=2)

    if has_rh and tracks.rh.shape[0] == n:
        rh_wrist_xy = tracks.rh[:, 0, :2]
        rh_visible = np.isfinite(rh_wrist_xy[:, 0])
        rh_interp = np.column_stack([interp_nans(rh_wrist_xy[:, 0]), interp_nans(rh_wrist_xy[:, 1])])
        rh_speed = compute_speed(rh_interp, t)
        # Invalidate speed where original data was missing
        rh_speed[~rh_visible] = np.nan
        rh_speed = ema_smooth(rh_speed, alpha=0.4, max_hold_frames=2)

    # Key pose landmarks for other detections
    nose_xy = pose[:, 0, :2]
    lsho_xy = pose[:, 11, :2]
    rsho_xy = pose[:, 12, :2]
    lhip_xy = pose[:, 23, :2]
    rhip_xy = pose[:, 24, :2]

    shoulder_mid_xy = (lsho_xy + rsho_xy) / 2
    hip_mid_xy = (lhip_xy + rhip_xy) / 2

    # Pose wrists (alternative to hand landmarks)
    pose_lwrist_xy = pose[:, 15, :2]
    pose_rwrist_xy = pose[:, 16, :2]

    return {
        "child_visible": child_visible,
        "parent_visible": parent_visible,
        "pose_speed": pose_speed,
        "lh_speed": lh_speed,
        "rh_speed": rh_speed,
        "lh_visible": lh_visible,
        "rh_visible": rh_visible,
        "lh_wrist_xy": lh_wrist_xy,
        "rh_wrist_xy": rh_wrist_xy,
        "pose_lwrist_xy": pose_lwrist_xy,
        "pose_rwrist_xy": pose_rwrist_xy,
        "nose_xy": nose_xy,
        "shoulder_mid_xy": shoulder_mid_xy,
        "hip_mid_xy": hip_mid_xy,
    }


# ============================================================
# Event detectors
# ============================================================
def detect_off_screen_events(
    tracks: TracksData, signals: Dict[str, np.ndarray], cfg: EventConfig
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Detect OFF_SCREEN events when child bbox is missing for extended periods.
    Returns (segments, stats).
    """
    t = tracks.t_sec
    visible = signals["child_visible"]

    # Find gaps (inverse of visible)
    off_screen_flags = ~visible
    segs = segments_from_bool(t, off_screen_flags, cfg.off_screen_min_dur)

    total_off = sum(e - s for s, e in segs)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    stats = {
        "off_screen_time_sec": round(total_off, 3),
        "off_screen_time_frac": round(total_off / max(duration, 1e-6), 4),
        "off_screen_bout_count": len(segs),
        "longest_off_screen_sec": round(max((e - s for s, e in segs), default=0.0), 3),
    }
    return segs, stats


def detect_activity_level_events(
    tracks: TracksData, signals: Dict[str, np.ndarray], cfg: EventConfig
) -> Tuple[List[Tuple[float, float, str]], Dict[str, Any]]:
    """
    Detect BODY_ACTIVITY_LEVEL events (HIGH bursts).
    Returns (segments with level label, stats).
    """
    t = tracks.t_sec
    speed = signals["pose_speed"]
    visible = signals["child_visible"]

    # Classify each frame
    activity_level = np.zeros(len(t), dtype=int)  # 0=low, 1=med, 2=high
    activity_level[speed >= cfg.activity_low_threshold] = 1
    activity_level[speed >= cfg.activity_high_threshold] = 2

    # Only count when visible
    activity_level[~visible] = 0

    # Detect high activity bursts
    high_flags = (activity_level == 2) & visible
    high_segs = segments_from_bool(t, high_flags, cfg.activity_min_dur)

    # Stats
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    valid_frames = visible.sum()
    if valid_frames > 0:
        frac_low = float((activity_level == 0)[visible].sum()) / valid_frames
        frac_med = float((activity_level == 1)[visible].sum()) / valid_frames
        frac_high = float((activity_level == 2)[visible].sum()) / valid_frames
    else:
        frac_low = frac_med = frac_high = 0.0

    high_total = sum(e - s for s, e in high_segs)

    stats = {
        "activity_frac_low": round(frac_low, 4),
        "activity_frac_med": round(frac_med, 4),
        "activity_frac_high": round(frac_high, 4),
        "activity_burst_count": len(high_segs),
        "activity_burst_total_sec": round(high_total, 3),
        "activity_bursts_per_min": round(len(high_segs) / max(duration / 60, 1e-6), 3),
        "mean_activity_burst_dur": round(high_total / max(len(high_segs), 1), 3),
    }

    # Return with level label for event CSV
    labeled_segs = [(s, e, "HIGH") for s, e in high_segs]
    return labeled_segs, stats


def detect_periodic_motion_events(
    tracks: TracksData, signals: Dict[str, np.ndarray], cfg: EventConfig
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Detect PERIODIC_MOTION via periodic oscillations in body parts.

    Uses normalized periodicity score (power/variance) which is scale-invariant
    and more robust across different camera distances and frame rates.

    Conservative detection to minimize false positives.
    """
    t = tracks.t_sec
    n = len(t)
    fps = tracks.fps / max(tracks.sample_every_n, 1)

    # Multiple signal candidates for repetitive motion
    pose = tracks.pose

    # 1. Vertical wrist oscillation (arm flapping)
    lw_y = pose[:, 15, 1]  # left wrist y
    rw_y = pose[:, 16, 1]  # right wrist y
    wrist_y = np.nanmean(np.stack([lw_y, rw_y], axis=1), axis=1)
    wrist_y = interp_nans(wrist_y)
    wrist_y = nanmed_smooth(wrist_y, win=2)

    # 2. Horizontal hip oscillation (rocking)
    hip_x = signals["hip_mid_xy"][:, 0]
    hip_x = interp_nans(hip_x)
    hip_x = nanmed_smooth(hip_x, win=2)

    # 3. Shoulder horizontal oscillation
    sho_x = signals["shoulder_mid_xy"][:, 0]
    sho_x = interp_nans(sho_x)
    sho_x = nanmed_smooth(sho_x, win=2)

    # Analyze each signal using normalized score (scale-invariant)
    best_freq = 0.0
    best_norm_score = 0.0
    best_type = "none"
    rep_flags = np.zeros(n, dtype=bool)

    for signal, name in [(wrist_y, "arm_flap"), (hip_x, "rocking_hip"), (sho_x, "rocking_shoulder")]:
        if not np.isfinite(signal).any():
            continue
        amplitude = float(np.nanstd(signal))
        if amplitude < cfg.rep_min_amplitude:
            continue

        # Global periodicity check (using normalized score)
        f, _p, norm_score = spectral_peak(signal, fps, cfg.rep_freq_min, cfg.rep_freq_max, cfg.rep_min_amplitude)
        if norm_score > best_norm_score:
            best_norm_score = norm_score
            best_freq = f
            best_type = name

        # Windowed periodicity for segment detection (using normalized threshold)
        if amplitude >= cfg.rep_min_amplitude:
            w_flags = windowed_periodicity(
                signal, fps, cfg.rep_window_sec, cfg.rep_freq_min, cfg.rep_freq_max,
                normalized_threshold=cfg.rep_normalized_threshold
            )
            rep_flags = rep_flags | w_flags

    # Only keep when child visible
    rep_flags = rep_flags & signals["child_visible"]

    segs = segments_from_bool(t, rep_flags, cfg.rep_min_dur)

    total_rep = sum(e - s for s, e in segs)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    stats = {
        "periodic_motion_time_sec": round(total_rep, 3),
        "periodic_motion_time_frac": round(total_rep / max(duration, 1e-6), 4),
        "periodic_motion_bout_count": len(segs),
        "periodic_motion_freq_hz": round(best_freq, 3) if total_rep > 0 else 0.0,
        "periodic_motion_type": best_type if total_rep > 0 else "none",
        "periodic_motion_norm_score": round(best_norm_score, 4) if total_rep > 0 else 0.0,
    }
    return segs, stats


def detect_parent_present_events(
    tracks: TracksData, signals: Dict[str, np.ndarray], cfg: EventConfig
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """Detect PARENT_PRESENT events when parent bbox is visible."""
    t = tracks.t_sec
    visible = signals["parent_visible"]

    segs = segments_from_bool(t, visible, cfg.parent_min_dur)

    total = sum(e - s for s, e in segs)

    stats = {
        "parent_present_time_sec": round(total, 3),
        "parent_present_ratio": round(float(visible.mean()), 4),
    }
    return segs, stats


def detect_close_proximity_events(
    tracks: TracksData, signals: Dict[str, np.ndarray], cfg: EventConfig
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """Detect CLOSE_PROXIMITY events when child and parent bboxes are close."""
    t = tracks.t_sec
    n = len(t)
    child_bbox = tracks.child_bbox
    parent_bbox = tracks.parent_bbox

    close_flags = np.zeros(n, dtype=bool)
    for i in range(n):
        if not signals["child_visible"][i] or not signals["parent_visible"][i]:
            continue

        iou = bbox_iou(child_bbox[i], parent_bbox[i])
        dist = bbox_center_distance(child_bbox[i], parent_bbox[i])

        if iou > cfg.proximity_iou_threshold or dist < cfg.proximity_center_dist_threshold:
            close_flags[i] = True

    segs = segments_from_bool(t, close_flags, cfg.proximity_min_dur)

    total = sum(e - s for s, e in segs)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    stats = {
        "close_proximity_time_sec": round(total, 3),
        "close_proximity_time_frac": round(total / max(duration, 1e-6), 4),
        "close_proximity_bout_count": len(segs),
    }
    return segs, stats


def detect_hand_active_events(
    tracks: TracksData, signals: Dict[str, np.ndarray], cfg: EventConfig
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Detect HAND_ACTIVE events: hands present + high velocity + in workspace region.

    Workspace is defined as the lower portion of the child's body (hands near lap/front).
    """
    t = tracks.t_sec
    n = len(t)
    child_bbox = tracks.child_bbox

    lh_active = np.zeros(n, dtype=bool)
    rh_active = np.zeros(n, dtype=bool)

    for i in range(n):
        if not signals["child_visible"][i]:
            continue

        # Get workspace y threshold
        # When bbox is valid: use relative position within bbox
        # When bbox is missing: use absolute frame y (cfg value is treated as frame-relative)
        bbox_valid = np.isfinite(child_bbox[i, 0]) and (child_bbox[i, 3] - child_bbox[i, 1]) > 0.01
        if bbox_valid:
            # Workspace is lower portion of child bbox (ratio applied to bbox height)
            workspace_y_min = child_bbox[i, 1] + (child_bbox[i, 3] - child_bbox[i, 1]) * cfg.hand_workspace_y_min
        else:
            # Fallback: use absolute frame y threshold (lower half of frame)
            workspace_y_min = 0.4  # absolute frame y, not the config ratio

        # Left hand
        if signals["lh_visible"][i]:
            wy = signals["lh_wrist_xy"][i, 1]
            speed = signals["lh_speed"][i]
            if np.isfinite(wy) and np.isfinite(speed) and wy >= workspace_y_min and speed > cfg.hand_speed_threshold:
                lh_active[i] = True

        # Right hand
        if signals["rh_visible"][i]:
            wy = signals["rh_wrist_xy"][i, 1]
            speed = signals["rh_speed"][i]
            if np.isfinite(wy) and np.isfinite(speed) and wy >= workspace_y_min and speed > cfg.hand_speed_threshold:
                rh_active[i] = True

    # Combine: either hand active
    hand_active = lh_active | rh_active
    segs = segments_from_bool(t, hand_active, cfg.hand_active_min_dur)

    total = sum(e - s for s, e in segs)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    stats = {
        "hand_active_time_sec": round(total, 3),
        "hand_active_ratio": round(total / max(duration, 1e-6), 4),
        "hand_active_bouts_per_min": round(len(segs) / max(duration / 60, 1e-6), 3),
        "left_hand_active_frac": round(float(lh_active.mean()), 4),
        "right_hand_active_frac": round(float(rh_active.mean()), 4),
    }
    return segs, stats


def detect_hand_to_face_events(
    tracks: TracksData, signals: Dict[str, np.ndarray], cfg: EventConfig
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """Detect HAND_TO_FACE events: wrist close to nose/face landmarks."""
    t = tracks.t_sec
    n = len(t)
    pose = tracks.pose

    # Face landmarks from pose: 0=nose, 1/2=eyes, 7/8=ears
    face_idx = [0, 1, 2, 7, 8]

    hand_face_flags = np.zeros(n, dtype=bool)

    for i in range(n):
        if not signals["child_visible"][i]:
            continue

        # Get face points
        face_pts = pose[i, face_idx, :2]
        if not np.isfinite(face_pts).any():
            continue

        best_dist = float("inf")

        # Check left hand/wrist
        for wrist_xy in [signals["lh_wrist_xy"][i], signals["pose_lwrist_xy"][i]]:
            if np.isfinite(wrist_xy).all():
                dists = np.sqrt(((wrist_xy - face_pts) ** 2).sum(axis=1))
                best_dist = min(best_dist, float(np.nanmin(dists)))

        # Check right hand/wrist
        for wrist_xy in [signals["rh_wrist_xy"][i], signals["pose_rwrist_xy"][i]]:
            if np.isfinite(wrist_xy).all():
                dists = np.sqrt(((wrist_xy - face_pts) ** 2).sum(axis=1))
                best_dist = min(best_dist, float(np.nanmin(dists)))

        if best_dist < cfg.hand_face_distance_threshold:
            hand_face_flags[i] = True

    segs = segments_from_bool(t, hand_face_flags, cfg.hand_face_min_dur)

    total = sum(e - s for s, e in segs)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    stats = {
        "hand_to_face_time_sec": round(total, 3),
        "hand_to_face_time_frac": round(total / max(duration, 1e-6), 4),
        "hand_to_face_count": len(segs),
    }
    return segs, stats


def detect_hands_near_events(
    tracks: TracksData, signals: Dict[str, np.ndarray], cfg: EventConfig
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Detect HANDS_NEAR events: both wrists close together.

    To distinguish clap-like motion from static hands-clasped/resting, we require
    either motion (combined wrist speed) OR open→close transitions nearby.

    Returns (segments, stats) where stats includes both raw "hands_near" and
    motion-filtered "hands_near_with_motion" counts.
    """
    t = tracks.t_sec
    n = len(t)
    fps = tracks.fps / max(tracks.sample_every_n, 1)

    # Compute wrist distance time series
    wrist_dist = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if not signals["child_visible"][i]:
            continue
        lw = signals["lh_wrist_xy"][i] if signals["lh_visible"][i] else signals["pose_lwrist_xy"][i]
        rw = signals["rh_wrist_xy"][i] if signals["rh_visible"][i] else signals["pose_rwrist_xy"][i]
        if np.isfinite(lw).all() and np.isfinite(rw).all():
            wrist_dist[i] = float(np.sqrt(((lw - rw) ** 2).sum()))

    # Basic "near" flags (just distance threshold)
    near_flags = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isfinite(wrist_dist[i]) and wrist_dist[i] < cfg.hands_together_threshold:
            near_flags[i] = True

    # Enhanced: require motion context for "clap-like" behavior
    # Either: (1) combined hand speed is high, OR (2) open→close transition nearby
    near_with_motion_flags = np.zeros(n, dtype=bool)
    open_threshold = cfg.hands_together_threshold * 2.0  # "open" = 2x the close threshold
    context_frames = max(int(0.4 * fps), 3)  # look 0.4s around for open frames

    for i in range(n):
        if not near_flags[i]:
            continue

        # Check 1: Is there combined hand motion?
        lh_spd = signals["lh_speed"][i] if np.isfinite(signals["lh_speed"][i]) else 0.0
        rh_spd = signals["rh_speed"][i] if np.isfinite(signals["rh_speed"][i]) else 0.0
        combined_speed = lh_spd + rh_spd
        has_motion = combined_speed > cfg.hand_speed_threshold * 1.5

        # Check 2: Is there an "open" frame nearby? (open→close transition)
        has_open_nearby = False
        for j in range(max(0, i - context_frames), min(n, i + context_frames + 1)):
            if j != i and np.isfinite(wrist_dist[j]) and wrist_dist[j] > open_threshold:
                has_open_nearby = True
                break

        if has_motion or has_open_nearby:
            near_with_motion_flags[i] = True

    # Segment the motion-filtered flags
    segs = segments_from_bool(t, near_with_motion_flags, cfg.hands_together_min_dur)

    # Also compute raw near segments for comparison
    raw_segs = segments_from_bool(t, near_flags, cfg.hands_together_min_dur)

    total = sum(e - s for s, e in segs)
    raw_total = sum(e - s for s, e in raw_segs)

    stats = {
        "hands_near_count": len(raw_segs),  # all instances of hands close
        "hands_near_total_sec": round(raw_total, 3),
        "hands_near_with_motion_count": len(segs),  # motion-filtered (more clap-like)
        "hands_near_with_motion_sec": round(total, 3),
    }
    return segs, stats


# ============================================================
# Main processing
# ============================================================
def process_free_play_tracks(
    child_id: str,
    tracks: TracksData,
    cfg: EventConfig,
) -> Tuple[List[EventSeg], Dict[str, Any]]:
    """
    Process tracks data to extract events and summary features.

    Returns: (events_list, summary_dict)
    """
    if tracks.t_sec.size < 15:
        return [], {"child_id": child_id, "task_type": "free_play", "error": "insufficient_frames"}

    # Compute frame-level signals
    signals = compute_frame_signals(tracks)

    # Check minimum tracking quality
    if signals["child_visible"].mean() < 0.3:
        return [], {"child_id": child_id, "task_type": "free_play", "error": "poor_tracking"}

    # Detect all event types
    events: List[EventSeg] = []

    def add_events(event_type: str, segs: List[Tuple[float, float]], conf: float, meta: str = "") -> None:
        for s, e in segs:
            events.append(EventSeg(child_id, "free_play", event_type, float(s), float(e), float(conf), meta))

    # A) Engagement & regulation
    off_segs, off_stats = detect_off_screen_events(tracks, signals, cfg)
    add_events("OFF_SCREEN", off_segs, 0.90, "child_bbox_missing")

    activity_segs, activity_stats = detect_activity_level_events(tracks, signals, cfg)
    for s, e, level in activity_segs:
        events.append(EventSeg(child_id, "free_play", f"ACTIVITY_{level}", s, e, 0.75, f"level={level}"))

    rep_segs, rep_stats = detect_periodic_motion_events(tracks, signals, cfg)
    rep_conf = 0.60 if rep_stats["periodic_motion_bout_count"] > 0 else 0.50
    add_events("PERIODIC_MOTION", rep_segs, rep_conf, f"type={rep_stats.get('periodic_motion_type', 'unknown')}")

    # B) Social interaction
    parent_segs, parent_stats = detect_parent_present_events(tracks, signals, cfg)
    add_events("PARENT_PRESENT", parent_segs, 0.85, "parent_bbox_detected")

    prox_segs, prox_stats = detect_close_proximity_events(tracks, signals, cfg)
    add_events("CLOSE_PROXIMITY", prox_segs, 0.75, "bbox_overlap_or_near")

    # C) Manual interaction
    hand_segs, hand_stats = detect_hand_active_events(tracks, signals, cfg)
    add_events("HAND_ACTIVE", hand_segs, 0.70, "workspace_motion")

    face_segs, face_stats = detect_hand_to_face_events(tracks, signals, cfg)
    add_events("HAND_TO_FACE", face_segs, 0.80, f"dist<{cfg.hand_face_distance_threshold}")

    near_segs, near_stats = detect_hands_near_events(tracks, signals, cfg)
    add_events("HANDS_NEAR", near_segs, 0.75, f"wrist_dist<{cfg.hands_together_threshold},with_motion")

    # Build summary
    duration = float(tracks.t_sec[-1] - tracks.t_sec[0]) if tracks.t_sec.size > 1 else 0.0
    summary = {
        "child_id": child_id,
        "task_type": "free_play",
        "duration_sec": round(duration, 3),
        # Detection coverage
        "child_visible_ratio": round(float(signals["child_visible"].mean()), 4),
        "parent_present_ratio": round(parent_stats["parent_present_ratio"], 4),
        "hand_detection_ratio": round(float((signals["lh_visible"] | signals["rh_visible"]).mean()), 4),
        # Off-screen
        **{k: v for k, v in off_stats.items()},
        # Activity
        **{k: v for k, v in activity_stats.items()},
        # Repetitive motion
        **{k: v for k, v in rep_stats.items()},
        # Proximity
        **{k: v for k, v in prox_stats.items()},
        # Hand events
        **{k: v for k, v in hand_stats.items()},
        **{k: v for k, v in face_stats.items()},
        **{k: v for k, v in near_stats.items()},
    }

    return events, summary


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Detect free-play behavioral events from pre-computed tracks."
    )
    ap.add_argument("--manifest", required=True, help="CSV with child_id column")
    ap.add_argument("--tracks_dir", default="data/derived/tracks", help="Directory containing *_free_play.npz files")
    ap.add_argument("--out_events", default="data/derived/free_play_events.csv")
    ap.add_argument("--out_summary", default="data/derived/free_play_summary.csv")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N videos (0=all)")

    # Event detection thresholds
    ap.add_argument("--off_screen_min_dur", type=float, default=0.5)
    ap.add_argument("--activity_low_thr", type=float, default=0.02)
    ap.add_argument("--activity_high_thr", type=float, default=0.05)
    ap.add_argument("--activity_min_dur", type=float, default=0.5)
    ap.add_argument("--rep_freq_min", type=float, default=1.5)
    ap.add_argument("--rep_freq_max", type=float, default=5.5)
    ap.add_argument("--rep_normalized_thr", type=float, default=0.15, help="Normalized periodicity score threshold (scale-invariant)")
    ap.add_argument("--rep_min_dur", type=float, default=1.0)
    ap.add_argument("--proximity_iou_thr", type=float, default=0.05)
    ap.add_argument("--proximity_dist_thr", type=float, default=0.25)
    ap.add_argument("--hand_speed_thr", type=float, default=0.04)
    ap.add_argument("--hand_face_dist_thr", type=float, default=0.12)
    ap.add_argument("--hands_together_thr", type=float, default=0.10)

    args = ap.parse_args()

    # Build config
    cfg = EventConfig(
        off_screen_min_dur=args.off_screen_min_dur,
        activity_low_threshold=args.activity_low_thr,
        activity_high_threshold=args.activity_high_thr,
        activity_min_dur=args.activity_min_dur,
        rep_freq_min=args.rep_freq_min,
        rep_freq_max=args.rep_freq_max,
        rep_normalized_threshold=args.rep_normalized_thr,
        rep_min_dur=args.rep_min_dur,
        proximity_iou_threshold=args.proximity_iou_thr,
        proximity_center_dist_threshold=args.proximity_dist_thr,
        hand_speed_threshold=args.hand_speed_thr,
        hand_face_distance_threshold=args.hand_face_dist_thr,
        hands_together_threshold=args.hands_together_thr,
    )

    # Load manifest
    df = pd.read_csv(args.manifest)
    if "child_id" not in df.columns:
        raise ValueError(f"Manifest missing 'child_id'. Columns: {list(df.columns)}")

    tracks_dir = Path(args.tracks_dir)
    out_events = Path(args.out_events)
    out_summary = Path(args.out_summary)

    # Create output directories
    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    # Define output fields
    event_fields = ["child_id", "task_type", "event_type", "t_start", "t_end", "confidence", "meta"]

    summary_fields = [
        "child_id", "task_type", "duration_sec",
        "child_visible_ratio", "parent_present_ratio", "hand_detection_ratio",
        # Off-screen
        "off_screen_time_sec", "off_screen_time_frac", "off_screen_bout_count", "longest_off_screen_sec",
        # Activity
        "activity_frac_low", "activity_frac_med", "activity_frac_high",
        "activity_burst_count", "activity_burst_total_sec", "activity_bursts_per_min", "mean_activity_burst_dur",
        # Periodic motion
        "periodic_motion_time_sec", "periodic_motion_time_frac", "periodic_motion_bout_count",
        "periodic_motion_freq_hz", "periodic_motion_type", "periodic_motion_norm_score",
        # Proximity
        "close_proximity_time_sec", "close_proximity_time_frac", "close_proximity_bout_count",
        # Hand events
        "hand_active_time_sec", "hand_active_ratio", "hand_active_bouts_per_min",
        "left_hand_active_frac", "right_hand_active_frac",
        "hand_to_face_time_sec", "hand_to_face_time_frac", "hand_to_face_count",
        "hands_near_count", "hands_near_total_sec",
        "hands_near_with_motion_count", "hands_near_with_motion_sec",
    ]

    # Initialize output files
    with open(out_events, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=event_fields).writeheader()
    with open(out_summary, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    # Process each child
    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    processed = 0
    errors = 0

    for i in range(n):
        child_id = str(df.iloc[i]["child_id"])
        npz_path = tracks_dir / f"{child_id}_free_play.npz"

        if not npz_path.exists():
            print(f"[{i+1}/{n}] child={child_id} - tracks not found, skipping")
            continue

        print(f"[{i+1}/{n}] child={child_id}", flush=True)

        tracks = load_tracks(npz_path)
        if tracks is None:
            print(f"  Error loading tracks")
            errors += 1
            continue

        events, summary = process_free_play_tracks(child_id, tracks, cfg)

        if "error" in summary:
            print(f"  Skipped: {summary.get('error')}")
            errors += 1
            continue

        # Write events
        with open(out_events, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=event_fields)
            for e in events:
                w.writerow({
                    "child_id": e.child_id,
                    "task_type": e.task_type,
                    "event_type": e.event_type,
                    "t_start": round(e.t_start, 3),
                    "t_end": round(e.t_end, 3),
                    "confidence": round(e.confidence, 4),
                    "meta": e.meta,
                })

        # Write summary
        with open(out_summary, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            # Fill missing fields with 0 or empty
            row = {k: summary.get(k, 0 if k != "repetitive_motion_type" else "none") for k in summary_fields}
            w.writerow(row)

        processed += 1

    print(f"\nDone. Processed: {processed}, Errors: {errors}")
    print(f"Wrote: {out_events}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
