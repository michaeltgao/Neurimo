from __future__ import annotations

import argparse
import csv
import math
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


# ============================================================
# Models
# ============================================================
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
CACHE_DIR = Path.home() / ".cache" / "mediapipe_models"


def _download(url: str, filename: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / filename
    if not p.exists():
        print(f"Downloading {filename}...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ctx) as r:
            p.write_bytes(r.read())
        print(f"Downloaded to {p}")
    return p


def get_models() -> Tuple[Path, Path]:
    return _download(HAND_MODEL_URL, "hand_landmarker.task"), _download(POSE_MODEL_URL, "pose_landmarker_lite.task")


# ============================================================
# Helpers
# ============================================================
def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def safe_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and fps > 0 else 30.0


def smooth_med(x: np.ndarray, win: int = 2) -> np.ndarray:
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=float)
    n = x.size
    for i in range(n):
        j0 = max(0, i - win)
        j1 = min(n, i + win + 1)
        out[i] = float(np.nanmedian(x[j0:j1]))
    return out


def interp_nans(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if good.sum() < max(8, int(0.10 * x.size)):
        return np.full_like(x, np.nan, dtype=float)
    out = x.astype(float).copy()
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def apply_jump_limiter(xy: np.ndarray, max_jump: float = 0.18) -> np.ndarray:
    """
    Apply jump limiter to xy position sequence to prevent teleportation artifacts.

    When a position jumps more than max_jump from the previous valid position,
    it's treated as invalid (set to NaN) for speed calculations.

    Args:
        xy: (N, 2) array of (x, y) positions
        max_jump: maximum allowed jump in normalized coordinates (default 0.18)

    Returns:
        (N, 2) array with large jumps replaced by NaN
    """
    if xy.size == 0 or len(xy.shape) != 2 or xy.shape[0] < 2:
        return xy

    result = xy.astype(float).copy()
    prev_valid_x = np.nan
    prev_valid_y = np.nan

    for i in range(len(result)):
        x, y = result[i, 0], result[i, 1]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        if np.isfinite(prev_valid_x) and np.isfinite(prev_valid_y):
            jump = math.hypot(x - prev_valid_x, y - prev_valid_y)
            if jump > max_jump:
                # Large jump - treat as invalid for speed purposes
                result[i, 0] = np.nan
                result[i, 1] = np.nan
                continue

        # Valid position - update tracker
        prev_valid_x, prev_valid_y = x, y

    return result


def spectral_peak(signal: np.ndarray, fps: float, fmin: float, fmax: float) -> Tuple[float, float]:
    """Return (peak_hz, peak_power) in [fmin,fmax]."""
    if signal.size < 20 or not np.isfinite(signal).all():
        return 0.0, 0.0
    s = signal - float(np.mean(signal))
    window = np.hanning(s.size)
    y = np.fft.rfft(s * window)
    freqs = np.fft.rfftfreq(s.size, d=1.0 / fps)
    power = (np.abs(y) ** 2)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0, 0.0
    k = int(np.argmax(power[mask]))
    return float(freqs[mask][k]), float(power[mask][k])


def segments_from_bool(times: np.ndarray, flags: np.ndarray, min_dur: float) -> List[Tuple[float, float]]:
    if times.size == 0:
        return []
    segs: List[Tuple[float, float]] = []
    in_seg = False
    s0 = 0.0
    for t, f in zip(times, flags):
        if f and not in_seg:
            in_seg = True
            s0 = float(t)
        if (not f) and in_seg:
            e0 = float(t)
            if e0 - s0 >= min_dur:
                segs.append((s0, e0))
            in_seg = False
    if in_seg:
        e0 = float(times[-1])
        if e0 - s0 >= min_dur:
            segs.append((s0, e0))
    return segs


def first_time_in_window(after_t: float, times: List[float], window_sec: float) -> Optional[float]:
    lo = after_t
    hi = after_t + window_sec
    for t in sorted(times):
        if t >= lo and t <= hi:
            return t
    return None


# ============================================================
# Regions fallback if pose bbox missing
# Child is generally centered, adult comes from sides/top/behind
# ============================================================
def in_child_region_fallback(x: float, y: float) -> bool:
    """Child is in center region of frame, typically lower half."""
    return (0.25 <= x <= 0.75) and (y >= 0.35)


def in_adult_region_fallback(x: float, y: float) -> bool:
    """Adult comes from sides, top, or behind - peripheral regions only."""
    # Sides (far left or far right)
    if x <= 0.20 or x >= 0.80:
        return True
    # Top region (above child's typical area)
    if y <= 0.30:
        return True
    # Upper corners (diagonal from behind)
    if y <= 0.45 and (x <= 0.30 or x >= 0.70):
        return True
    return False


# ============================================================
# Pose helpers (child bbox + child arms-up)
# ============================================================
def pose_bbox(lms: List[Any], min_vis: float = 0.35) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for lm in lms:
        v = float(getattr(lm, "visibility", 0.0))
        if v >= min_vis:
            xs.append(float(lm.x))
            ys.append(float(lm.y))
    if len(xs) < 8:
        return None
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    pad_x, pad_y = 0.05, 0.07
    x0 = max(0.0, x0 - pad_x)
    x1 = min(1.0, x1 + pad_x)
    y0 = max(0.0, y0 - pad_y)
    y1 = min(1.0, y1 + pad_y)
    return (x0, y0, x1, y1)


def point_in_bbox(x: float, y: float, bb: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bb
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def bbox_iou(bb1: Optional[Tuple[float, float, float, float]], bb2: Optional[Tuple[float, float, float, float]]) -> float:
    """Compute IoU between two bboxes (x0, y0, x1, y1). Returns 0.0 if either is None."""
    if bb1 is None or bb2 is None:
        return 0.0
    x0_1, y0_1, x1_1, y1_1 = bb1
    x0_2, y0_2, x1_2, y1_2 = bb2

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


def choose_child_and_adult_poses_temporal(
    poses: List[Any],
    prev_child_bbox: Optional[Tuple[float, float, float, float]],
    iou_weight: float = 0.6,
    min_vis: float = 0.30,
) -> Tuple[Optional[Any], Optional[Any], Optional[Tuple[float, float, float, float]]]:
    """
    Choose child and adult poses using temporal continuity (IoU with previous child bbox).

    Args:
        poses: List of pose landmark lists from MediaPipe
        prev_child_bbox: Previous frame's child bbox for temporal continuity
        iou_weight: Weight for IoU score vs position score (0-1)
        min_vis: Minimum visibility for classification

    Returns:
        (child_pose, adult_pose, child_bbox) - any can be None
    """
    if not poses:
        return None, None, None

    if len(poses) == 1:
        # Single pose - classify based on position but treat as child for tracking continuity
        lms = poses[0]
        classification = classify_pose_as_child_or_adult(lms, min_vis=min_vis)
        bbox = pose_bbox(lms, min_vis=0.35)
        # Even if classified as adult, if only one pose, treat as child for tracking
        return lms, None, bbox

    # Score each pose with combined position + temporal continuity
    pose_scores: List[Tuple[float, int, str, Optional[Tuple[float, float, float, float]]]] = []

    for i, lms in enumerate(poses):
        bbox = pose_bbox(lms, min_vis=0.35)
        classification = classify_pose_as_child_or_adult(lms, min_vis=min_vis)

        # Position score: centered = more likely child
        if bbox is None:
            position_score = 0.0
        else:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            # Higher score for centered positions
            position_score = (1.0 - abs(cx - 0.5)) * (1.0 - abs(cy - 0.5) * 0.5)
            # Penalty for edge positions (adult-like)
            if cx <= 0.20 or cx >= 0.80 or cy <= 0.25:
                position_score *= 0.3

        # Temporal continuity score (IoU with previous child bbox)
        iou = bbox_iou(bbox, prev_child_bbox) if prev_child_bbox is not None else 0.0

        # Combined score
        if prev_child_bbox is not None:
            combined_score = iou_weight * iou + (1 - iou_weight) * position_score
        else:
            combined_score = position_score

        pose_scores.append((combined_score, i, classification or "unknown", bbox))

    # Sort by combined score (descending)
    pose_scores.sort(reverse=True, key=lambda x: x[0])

    # Select child as highest scoring pose
    child_idx = pose_scores[0][1]
    child_pose = poses[child_idx]
    child_bbox = pose_scores[0][3]
    adult_pose = None

    # Select adult from remaining poses (prefer adult-classified)
    for _score, idx, classification, _bbox in pose_scores[1:]:
        if classification == "adult":
            adult_pose = poses[idx]
            break

    # If no adult found, use second-best pose as adult
    if adult_pose is None and len(pose_scores) > 1:
        adult_pose = poses[pose_scores[1][1]]

    return child_pose, adult_pose, child_bbox


def child_arms_up(lms: List[Any]) -> bool:
    """Check if child's wrists are above their shoulders (arm raising).

    More lenient detection - only needs one arm to be raised with sufficient visibility.
    """
    def vis(i: int) -> float:
        return float(getattr(lms[i], "visibility", 0.0))

    margin = 0.02

    # Check left arm: left wrist (15) above left shoulder (11)
    l_ok = False
    if vis(15) >= 0.20 and vis(11) >= 0.20:
        l_ok = float(lms[15].y) < float(lms[11].y) - margin

    # Check right arm: right wrist (16) above right shoulder (12)
    r_ok = False
    if vis(16) >= 0.20 and vis(12) >= 0.20:
        r_ok = float(lms[16].y) < float(lms[12].y) - margin

    # Also check if wrists are high on screen (above 0.35) even with lower visibility
    # This catches cases where child raises arms but landmarks are partially occluded
    l_high = vis(15) >= 0.15 and float(lms[15].y) < 0.35
    r_high = vis(16) >= 0.15 and float(lms[16].y) < 0.35

    return bool(l_ok or r_ok or l_high or r_high)


def adult_arms_up(lms: List[Any]) -> bool:
    """Check if adult's arms are raised (wrists above shoulders).

    More tolerant of partial visibility since adult may be partially off-screen.
    """
    def vis(i: int) -> float:
        return float(getattr(lms[i], "visibility", 0.0))

    # Need at least one shoulder visible (adult may be partially off-screen)
    if max(vis(11), vis(12)) < 0.15:
        return False
    # Need at least one wrist visible
    if max(vis(15), vis(16)) < 0.15:
        return False

    margin = 0.03
    l_ok = vis(15) >= 0.15 and vis(11) >= 0.15 and float(lms[15].y) < float(lms[11].y) - margin
    r_ok = vis(16) >= 0.15 and vis(12) >= 0.15 and float(lms[16].y) < float(lms[12].y) - margin

    # Also check if wrist is near top of frame (going off-screen while raised)
    l_near_top = vis(15) >= 0.10 and float(lms[15].y) < 0.15
    r_near_top = vis(16) >= 0.10 and float(lms[16].y) < 0.15

    return bool(l_ok or r_ok or l_near_top or r_near_top)


def classify_pose_as_child_or_adult(lms: List[Any], min_vis: float = 0.30) -> Optional[str]:
    """
    Classify a detected pose as 'child' or 'adult' based on position.
    Child is generally centered; adult comes from sides/top/behind.
    Returns None if insufficient landmarks visible.

    Adults may be partially off-screen, so we use lower visibility thresholds
    and check for landmarks near frame edges.
    """
    # Get visible landmarks to compute center
    xs, ys = [], []
    edge_count = 0  # Count landmarks near frame edges (suggests adult partially off-screen)

    for lm in lms:
        v = float(getattr(lm, "visibility", 0.0))
        if v >= min_vis:
            x, y = float(lm.x), float(lm.y)
            xs.append(x)
            ys.append(y)
            # Check if near edge of frame
            if x <= 0.05 or x >= 0.95 or y <= 0.05:
                edge_count += 1

    # For adults partially off-screen, accept fewer visible landmarks
    if len(xs) < 3:
        return None
    if len(xs) < 5 and edge_count == 0:
        return None

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    # Child is centered (middle of frame, lower portion)
    is_centered_x = 0.25 <= cx <= 0.75
    is_lower_half = cy >= 0.35

    # Adult comes from periphery (sides, top, behind)
    is_peripheral_x = cx <= 0.25 or cx >= 0.75
    is_upper = cy <= 0.35
    is_upper_corner = cy <= 0.45 and (cx <= 0.30 or cx >= 0.70)

    # If many landmarks are near edges, likely an adult partially off-screen
    is_partial_offscreen = edge_count >= 2

    if is_centered_x and is_lower_half and not is_partial_offscreen:
        return "child"
    elif is_peripheral_x or is_upper or is_upper_corner or is_partial_offscreen:
        return "adult"
    else:
        # Default: if centered, likely child
        return "child" if is_centered_x else "adult"


def get_pose_wrist_positions(lms: List[Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Get left and right wrist positions from pose landmarks if visible."""
    def vis(i: int) -> float:
        return float(getattr(lms[i], "visibility", 0.0))

    left_wrist = None
    right_wrist = None

    if vis(15) >= 0.25:
        left_wrist = (float(lms[15].x), float(lms[15].y))
    if vis(16) >= 0.25:
        right_wrist = (float(lms[16].x), float(lms[16].y))

    return left_wrist, right_wrist


# ============================================================
# Data
# ============================================================
@dataclass
class Event:
    child_id: str
    task_type: str
    subject: str
    primitive: str
    kind: str
    t_sec: float
    confidence: float
    meta: str


@dataclass
class HandDet:
    t: float
    wx: float
    wy: float
    conf: float


# ============================================================
# New dataclasses for two-channel detection
# ============================================================
@dataclass
class ParentAction:
    """Parent's demonstration of an action."""
    action_type: str           # "CLAP", "ARMS_UP"
    t_start: float             # when action begins
    t_end: float               # when action ends
    confidence: float          # detection confidence (0-1)
    n_repetitions: int         # e.g., number of claps in sequence
    detection_method: str      # "distance", "speed", "pose"


@dataclass
class ChildAttempt:
    """Child's attempt to imitate an action."""
    attempt_id: int            # sequential attempt number
    action_type: str           # "CLAP", "ARMS_UP"
    t_start: float             # when attempt begins
    t_end: float               # when attempt ends
    confidence: float          # detection confidence (0-1)
    n_repetitions: int         # e.g., number of claps
    detection_method: str      # "distance", "speed", "pose"
    is_response: bool = False  # True if within response window of parent action
    latency_from_demo_sec: Optional[float] = None  # time since parent action end


@dataclass
class ImitationTrial:
    """A single parent demo → child response trial."""
    trial_id: int
    action_type: str                             # "CLAP", "ARMS_UP"
    parent_action: ParentAction                  # the demonstration
    child_attempts: List[ChildAttempt] = field(default_factory=list)
    first_success_attempt_id: Optional[int] = None  # which attempt succeeded
    attempts_before_success: int = 0             # 0 if first attempt succeeded
    imitation_success: bool = False              # did child successfully imitate?
    imitation_latency_sec: Optional[float] = None  # time to first successful response


@dataclass
class ImitationEvent:
    """Event for CSV output."""
    child_id: str
    task_type: str
    event_type: str           # "PARENT_ACTION_START", "PARENT_ACTION_END", etc.
    action_type: str          # "CLAP", "ARMS_UP"
    subject: str              # "parent", "child"
    t_sec: float
    confidence: float
    attempt_id: Optional[int] = None         # for child attempts
    trial_id: Optional[int] = None           # links to parent action
    latency_sec: Optional[float] = None      # for success events
    attempts_before_success: Optional[int] = None
    meta: str = ""                           # additional info


# ============================================================
# CLAP: distance-based + speed fallback (with jitter guards)
# ============================================================
def pick_two_hands(hands: List[HandDet], max_wrist_dist: float) -> Optional[Tuple[HandDet, HandDet]]:
    if len(hands) < 2:
        return None
    best = None
    best_score = -1.0
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            h1, h2 = hands[i], hands[j]
            d = euclid((h1.wx, h1.wy), (h2.wx, h2.wy))
            if d > max_wrist_dist:
                continue
            score = (h1.conf + h2.conf) / (1.0 + d)
            if score > best_score:
                best_score = score
                best = (h1, h2)
    return best


def clap_thresholds(dist: np.ndarray) -> Tuple[float, float]:
    q15 = float(np.nanquantile(dist, 0.15))
    q65 = float(np.nanquantile(dist, 0.65))
    close_th = float(np.clip(q15 * 1.30, 0.04, 0.25))
    open_th = float(np.clip(q65 * 0.95, close_th + 0.03, 0.60))
    return close_th, open_th


def detect_claps_from_distance(
    times: np.ndarray,
    dist: np.ndarray,
    conf: np.ndarray,
    fps_eff: float,
    min_claps: int,
    periodic_min: float,
) -> List[Tuple[float, float]]:
    if times.size < 12:
        return []
    d = smooth_med(dist, win=2)
    close_th, open_th = clap_thresholds(d)

    claps: List[Tuple[float, float]] = []
    state = "open"
    last = -1e9
    min_sep = 0.18

    for t, dv, cv in zip(times, d, conf):
        if not np.isfinite(dv):
            continue
        if state == "open":
            if dv <= close_th and (float(t) - last) >= min_sep:
                claps.append((float(t), float(cv)))
                last = float(t)
                state = "closed"
        else:
            if dv >= open_th:
                state = "open"

    if len(claps) == 0:
        return []

    dd = interp_nans(d.copy())
    if not np.isfinite(dd).all():
        return claps if len(claps) >= min_claps else []

    _, pwr = spectral_peak(dd, fps_eff, 1.0, 3.2)
    var = float(np.var(dd))
    per_score = float(pwr / (var + 1e-6))

    if len(claps) >= min_claps:
        return claps
    if per_score >= periodic_min:
        return claps
    return []


def group_demo(event_times: List[float], max_gap: float, min_events: int) -> Optional[Tuple[float, float]]:
    if len(event_times) < min_events:
        return None
    ts = sorted(event_times)
    groups: List[List[float]] = []
    cur = [ts[0]]
    for t in ts[1:]:
        if t - cur[-1] <= max_gap:
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)
    groups.sort(key=lambda g: (len(g), g[-1] - g[0]), reverse=True)
    best = groups[0]
    if len(best) < min_events:
        return None
    return (best[0], best[-1])


def detect_speed_peaks(times: np.ndarray, speed: np.ndarray, peak_thr_std: float, min_sep_sec: float = 0.18) -> List[float]:
    if times.size < 12:
        return []
    s = smooth_med(speed, win=2)
    m = float(np.nanmean(s))
    sd = float(np.nanstd(s) + 1e-6)
    thr = m + peak_thr_std * sd

    peaks: List[float] = []
    last = -1e9
    for i in range(1, len(s) - 1):
        if not np.isfinite(s[i]):
            continue
        if s[i] > thr and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            t = float(times[i])
            if t - last >= min_sep_sec:
                peaks.append(t)
                last = t
    return peaks


def clap_demo_from_speed(
    t: np.ndarray,
    speed: np.ndarray,
    fps_eff: float,
    peak_thr_std: float,
    min_peaks: int,
    periodic_min: float,
    min_speed_abs: float,
    min_speed_var: float,
) -> Optional[Tuple[float, float, float, int]]:
    if t.size < 16:
        return None
    s = interp_nans(speed)
    if not np.isfinite(s).all():
        return None
    s = smooth_med(s, win=2)

    # jitter guard: if there's not enough motion energy, don't treat as demo
    if float(np.nanmax(s)) < min_speed_abs:
        return None
    if float(np.nanvar(s)) < min_speed_var:
        return None

    peaks = detect_speed_peaks(t, s, peak_thr_std=peak_thr_std, min_sep_sec=0.18)
    if len(peaks) == 0:
        return None

    _, pwr = spectral_peak(s, fps_eff, 1.0, 3.2)
    var = float(np.var(s))
    per_score = float(pwr / (var + 1e-6))

    if len(peaks) >= min_peaks:
        demo = group_demo(peaks, max_gap=1.4, min_events=min_peaks)
        if demo is None:
            demo = (peaks[0], peaks[-1])
        if per_score < periodic_min:
            return None
        return (float(demo[0]), float(demo[1]), float(per_score), int(len(peaks)))

    # If only 1 peak: require strong periodicity + strong speed burst
    if per_score >= (periodic_min * 1.25) and float(np.nanmax(s)) >= (min_speed_abs * 1.4):
        return (float(peaks[0]), float(peaks[-1]), float(per_score), int(len(peaks)))

    return None


# ============================================================
# Adult ARMS_UP demo: "raise then disappear" detector
# ============================================================
def arms_up_raise_then_disappear(
    t: np.ndarray,
    adult_best_xy: List[Tuple[float, float]],
    demo_max_rise_sec: float,
    y_high: float,
    y_low: float,
    vanish_min_sec: float,
) -> Optional[Tuple[float, float]]:
    """
    If adult wrist y goes from low-on-screen (large y) -> high (small y) quickly,
    and then wrist becomes missing/outside (NaN) soon after, count as ARMS_UP demo.

    Also detects arms going off the sides of the frame.
    """
    xy = np.array(adult_best_xy, dtype=float)
    x = xy[:, 0].astype(float)
    y = xy[:, 1].astype(float)
    is_ok = np.isfinite(y) & np.isfinite(x)

    # nothing to do if almost no adult wrist data
    if int(is_ok.sum()) < max(6, int(0.08 * len(y))):
        return None

    # scan for a "raise" transition (upward motion)
    for i in range(len(y)):
        if not is_ok[i]:
            continue
        if y[i] < y_high:  # already high
            continue

        # look forward up to demo_max_rise_sec
        t0 = float(t[i])
        jmax = int(np.searchsorted(t, t0 + demo_max_rise_sec, side="right"))
        jmax = min(jmax, len(y))

        # find first frame where wrist is high OR near edge of frame
        j_hi = None
        for j in range(i + 1, jmax):
            if not is_ok[j]:
                continue
            # Wrist moved high (arm raised)
            if y[j] <= y_low:
                j_hi = j
                break
            # Wrist moved to edge (arm going off-screen to side)
            if x[j] <= 0.08 or x[j] >= 0.92:
                j_hi = j
                break

        if j_hi is None:
            continue

        # after reaching high/edge, see if it "vanishes" for vanish_min_sec
        t_hi = float(t[j_hi])
        kmax = int(np.searchsorted(t, t_hi + vanish_min_sec, side="right"))
        kmax = min(kmax, len(y))
        if kmax - j_hi < 3:
            continue

        vanish_ratio = float((~is_ok[j_hi:kmax]).sum()) / float(max(1, (kmax - j_hi)))
        if vanish_ratio >= 0.50:  # Lowered threshold for partial off-screen
            return (t0, t_hi)

    # Also scan for movement toward edges (side motion for arm raising from side)
    for i in range(len(x)):
        if not is_ok[i]:
            continue
        # Start from center-ish position
        if not (0.25 <= x[i] <= 0.75):
            continue

        t0 = float(t[i])
        jmax = int(np.searchsorted(t, t0 + demo_max_rise_sec, side="right"))
        jmax = min(jmax, len(x))

        # find frame where wrist moved to edge
        j_edge = None
        for j in range(i + 1, jmax):
            if not is_ok[j]:
                continue
            if x[j] <= 0.10 or x[j] >= 0.90 or y[j] <= 0.15:
                j_edge = j
                break

        if j_edge is None:
            continue

        # Check if it vanishes after reaching edge
        t_edge = float(t[j_edge])
        kmax = int(np.searchsorted(t, t_edge + vanish_min_sec, side="right"))
        kmax = min(kmax, len(x))
        if kmax - j_edge < 3:
            continue

        vanish_ratio = float((~is_ok[j_edge:kmax]).sum()) / float(max(1, (kmax - j_edge)))
        if vanish_ratio >= 0.50:
            return (t0, t_edge)

    return None


# ============================================================
# Confidence scoring
# ============================================================
def compute_action_confidence(
    n_repetitions: int,
    duration_sec: float,
    stability: float,        # 0-1, higher = more stable
    detector_conf: float,    # raw MediaPipe/detector confidence
    periodicity_score: float = 0.0,  # for rhythmic actions like claps
    base_confidence: float = 0.7,
) -> float:
    """
    Compute multi-factor confidence for an action detection.

    Weights:
    - 30% repetition count (more = better, up to 5)
    - 25% duration (longer = better, up to 2s)
    - 20% stability (lower variance = better)
    - 15% detector confidence
    - 10% periodicity (for rhythmic actions)
    """
    # Repetition factor: 1 rep = 0.3, 5+ reps = 1.0
    rep_factor = min(1.0, (n_repetitions - 1) / 4.0) if n_repetitions > 0 else 0.0
    rep_factor = max(0.3, rep_factor) if n_repetitions >= 1 else 0.0

    # Duration factor: 0.2s = 0.1, 2s+ = 1.0
    dur_factor = min(1.0, duration_sec / 2.0) if duration_sec > 0 else 0.0

    # Stability factor: already 0-1
    stab_factor = float(np.clip(stability, 0.0, 1.0))

    # Detector confidence factor: already 0-1
    det_factor = float(np.clip(detector_conf, 0.0, 1.0))

    # Periodicity factor: normalize (typical values 5-20 for good claps)
    per_factor = min(1.0, periodicity_score / 15.0) if periodicity_score > 0 else 0.0

    # Weighted combination
    confidence = (
        base_confidence * rep_factor * 0.30 +
        base_confidence * dur_factor * 0.25 +
        base_confidence * stab_factor * 0.20 +
        det_factor * 0.15 +
        base_confidence * per_factor * 0.10
    )

    return float(np.clip(confidence, 0.0, 1.0))


# ============================================================
# Parent action detection
# ============================================================
def detect_parent_clap_actions(
    t_arr: np.ndarray,
    adult_frame_hands: List[List[HandDet]],
    adult_best_xy: List[Tuple[float, float]],
    fps_eff: float,
    min_claps_dist: int = 2,
    periodic_min_dist: float = 8.0,
    speed_peak_thr_std: float = 2.1,
    min_peaks: int = 2,
    periodic_min_speed: float = 6.0,
    min_speed_abs: float = 0.06,
    min_speed_var: float = 0.0006,
) -> List[ParentAction]:
    """
    Detect parent CLAP demonstrations.
    Returns list of ParentAction with timing and confidence.
    """
    actions: List[ParentAction] = []

    # Build distance time series for two-hand claps (full-length with NaNs for missing)
    def build_pair_ts_full(frame_hands: List[List[HandDet]], max_dist: float, min_conf: float):
        """Build full-length distance array aligned to t_arr. Missing frames get NaN."""
        n = len(t_arr)
        ds = np.full(n, np.nan, dtype=float)
        cs = np.full(n, np.nan, dtype=float)
        for i, hs in enumerate(frame_hands):
            hs2 = [h for h in hs if h.conf >= min_conf]
            pair = pick_two_hands(hs2, max_wrist_dist=max_dist)
            if pair is None:
                continue
            h1, h2 = pair
            ds[i] = euclid((h1.wx, h1.wy), (h2.wx, h2.wy))
            cs[i] = float(min(h1.conf, h2.conf))
        return t_arr.copy(), ds, cs

    a_t, a_d, a_c = build_pair_ts_full(adult_frame_hands, max_dist=0.98, min_conf=0.14)

    # Distance-based clap detection
    adult_claps_dist = detect_claps_from_distance(
        a_t, a_d, a_c, fps_eff=fps_eff,
        min_claps=min_claps_dist,
        periodic_min=periodic_min_dist,
    ) if a_t.size > 0 else []

    # Try to group into a demo
    demo_dist = group_demo([t for t, _ in adult_claps_dist], max_gap=1.4, min_events=max(2, min_claps_dist))

    if demo_dist is not None:
        # Distance-based demo found
        demo_start, demo_end = float(demo_dist[0]), float(demo_dist[1])
        n_claps = len(adult_claps_dist)

        # Compute periodicity for confidence
        if a_d.size > 0:
            dd = interp_nans(a_d.copy())
            if np.isfinite(dd).all():
                _, pwr = spectral_peak(dd, fps_eff, 1.0, 3.2)
                var = float(np.var(dd))
                per_score = float(pwr / (var + 1e-6))
            else:
                per_score = 0.0
        else:
            per_score = 0.0

        conf = compute_action_confidence(
            n_repetitions=n_claps,
            duration_sec=demo_end - demo_start,
            stability=0.7,  # distance-based is relatively stable
            detector_conf=float(np.mean([c for _, c in adult_claps_dist])) if adult_claps_dist else 0.5,
            periodicity_score=per_score,
        )

        actions.append(ParentAction(
            action_type="CLAP",
            t_start=demo_start,
            t_end=demo_end,
            confidence=conf,
            n_repetitions=n_claps,
            detection_method="distance",
        ))
    else:
        # Speed fallback
        xy = np.array(adult_best_xy, dtype=float)
        if xy.size > 0 and len(xy.shape) == 2 and xy.shape[0] > 1:
            # Apply jump limiter to prevent teleportation artifacts from hand swaps
            xy = apply_jump_limiter(xy, max_jump=0.18)

            dx = np.diff(xy[:, 0])
            dy = np.diff(xy[:, 1])
            dt = np.diff(t_arr)
            with np.errstate(invalid="ignore", divide="ignore"):
                speed = np.sqrt(dx * dx + dy * dy) / np.maximum(dt, 1e-6)
            t_speed = t_arr[1:]

            # Guard: require adult hand detections
            adult_hand_frames = sum(1 for hs in adult_frame_hands if len(hs) > 0)
            adult_hand_ratio = adult_hand_frames / max(1, len(adult_frame_hands))

            if adult_hand_ratio >= 0.06:
                demo_speed = clap_demo_from_speed(
                    t_speed, speed, fps_eff=fps_eff,
                    peak_thr_std=speed_peak_thr_std,
                    min_peaks=min_peaks,
                    periodic_min=periodic_min_speed,
                    min_speed_abs=min_speed_abs,
                    min_speed_var=min_speed_var,
                )

                if demo_speed is not None:
                    demo_start, demo_end, per_score, n_peaks = demo_speed
                    conf = compute_action_confidence(
                        n_repetitions=n_peaks,
                        duration_sec=demo_end - demo_start,
                        stability=0.5,  # speed-based is less stable
                        detector_conf=0.5,
                        periodicity_score=per_score,
                    )
                    actions.append(ParentAction(
                        action_type="CLAP",
                        t_start=float(demo_start),
                        t_end=float(demo_end),
                        confidence=conf,
                        n_repetitions=n_peaks,
                        detection_method=f"speed;per={per_score:.1f}",
                    ))

    return actions


def detect_parent_arms_actions(
    t_arr: np.ndarray,
    adult_frame_hands: List[List[HandDet]],
    adult_arms_flags: List[bool],
    adult_best_xy: List[Tuple[float, float]],
    arms_min_seg_dur: float = 0.18,
    use_raise_then_disappear: bool = True,
) -> List[ParentAction]:
    """
    Detect parent ARMS_UP demonstrations.
    Returns list of ParentAction with timing and confidence.
    """
    actions: List[ParentAction] = []

    # High position flags (from pose or hand position)
    adult_high_flags: List[bool] = []
    for i, hs in enumerate(adult_frame_hands):
        pose_arms_up = adult_arms_flags[i] if i < len(adult_arms_flags) else False
        hand_high = False
        hand_near_edge = False

        if len(hs) >= 2:
            ys = sorted([h.wy for h in hs])[:2]
            hand_high = bool(ys[0] < 0.40 and ys[1] < 0.40)
            for h in hs:
                if h.wy < 0.12 or h.wx < 0.08 or h.wx > 0.92:
                    hand_near_edge = True
                    break
        elif len(hs) == 1:
            hand_high = bool(hs[0].wy < 0.35)
            if hs[0].wy < 0.12 or hs[0].wx < 0.08 or hs[0].wx > 0.92:
                hand_near_edge = True

        adult_high_flags.append(bool(pose_arms_up or hand_high or hand_near_edge))

    adult_high = np.array(adult_high_flags, dtype=bool)
    adult_high_segs = segments_from_bool(t_arr, adult_high, min_dur=arms_min_seg_dur)

    if len(adult_high_segs) > 0:
        seg_start, seg_end = adult_high_segs[0]
        conf = compute_action_confidence(
            n_repetitions=1,
            duration_sec=seg_end - seg_start,
            stability=0.8,
            detector_conf=0.7,
            periodicity_score=0.0,
        )
        actions.append(ParentAction(
            action_type="ARMS_UP",
            t_start=float(seg_start),
            t_end=float(seg_end),
            confidence=conf,
            n_repetitions=1,
            detection_method="hands_high",
        ))
    elif use_raise_then_disappear:
        # Fallback: raise-then-disappear
        rt = arms_up_raise_then_disappear(
            t_arr,
            adult_best_xy=adult_best_xy,
            demo_max_rise_sec=0.65,
            y_high=0.60,
            y_low=0.38,
            vanish_min_sec=0.35,
        )
        if rt is not None:
            demo_start, demo_end = float(rt[0]), float(rt[1])
            conf = compute_action_confidence(
                n_repetitions=1,
                duration_sec=demo_end - demo_start,
                stability=0.6,
                detector_conf=0.5,
                periodicity_score=0.0,
            )
            actions.append(ParentAction(
                action_type="ARMS_UP",
                t_start=demo_start,
                t_end=demo_end,
                confidence=conf,
                n_repetitions=1,
                detection_method="raise_then_disappear",
            ))

    return actions


# ============================================================
# Child attempt detection
# ============================================================
def detect_child_clap_attempts(
    t_arr: np.ndarray,
    child_frame_hands: List[List[HandDet]],
    child_best_xy: List[Tuple[float, float]],
    parent_clap_actions: List[ParentAction],
    fps_eff: float,
    response_window_sec: float,
    min_claps_dist: int = 1,
    periodic_min_dist: float = 8.0,
    speed_peak_thr_std: float = 2.1,
    min_peaks: int = 1,
    min_speed_abs: float = 0.055,
    allow_single_burst: bool = True,
) -> List[ChildAttempt]:
    """
    Detect child CLAP attempts and tag as responses to parent actions.
    """
    attempts: List[ChildAttempt] = []
    attempt_id = 0

    # Build distance time series (full-length with NaNs for missing)
    def build_pair_ts_full(frame_hands: List[List[HandDet]], max_dist: float, min_conf: float):
        """Build full-length distance array aligned to t_arr. Missing frames get NaN."""
        n = len(t_arr)
        ds = np.full(n, np.nan, dtype=float)
        cs = np.full(n, np.nan, dtype=float)
        for i, hs in enumerate(frame_hands):
            hs2 = [h for h in hs if h.conf >= min_conf]
            pair = pick_two_hands(hs2, max_wrist_dist=max_dist)
            if pair is None:
                continue
            h1, h2 = pair
            ds[i] = euclid((h1.wx, h1.wy), (h2.wx, h2.wy))
            cs[i] = float(min(h1.conf, h2.conf))
        return t_arr.copy(), ds, cs

    c_t, c_d, c_c = build_pair_ts_full(child_frame_hands, max_dist=0.72, min_conf=0.16)

    # Distance-based clap detection
    child_claps_dist = detect_claps_from_distance(
        c_t, c_d, c_c, fps_eff=fps_eff,
        min_claps=min_claps_dist,
        periodic_min=periodic_min_dist,
    ) if c_t.size > 0 else []

    # Filter early false positives
    MIN_VALID_CLAP_TIME = 0.5
    child_claps_dist = [(t, c) for t, c in child_claps_dist if t >= MIN_VALID_CLAP_TIME]

    # Group claps into attempts (claps within 1.4s of each other)
    if child_claps_dist:
        clap_groups: List[List[Tuple[float, float]]] = []
        current_group: List[Tuple[float, float]] = [child_claps_dist[0]]

        for t, c in child_claps_dist[1:]:
            if t - current_group[-1][0] <= 1.4:
                current_group.append((t, c))
            else:
                clap_groups.append(current_group)
                current_group = [(t, c)]
        clap_groups.append(current_group)

        for group in clap_groups:
            t_start = group[0][0]
            t_end = group[-1][0]
            n_claps = len(group)
            mean_conf = float(np.mean([c for _, c in group]))

            # Check if this is a response to a parent action
            is_response = False
            latency: Optional[float] = None
            for pa in parent_clap_actions:
                if pa.t_end <= t_start <= pa.t_end + response_window_sec:
                    is_response = True
                    latency = t_start - pa.t_end
                    break

            conf = compute_action_confidence(
                n_repetitions=n_claps,
                duration_sec=t_end - t_start + 0.1,  # add small duration for single claps
                stability=0.7,
                detector_conf=mean_conf,
                periodicity_score=0.0,
            )

            attempts.append(ChildAttempt(
                attempt_id=attempt_id,
                action_type="CLAP",
                t_start=t_start,
                t_end=t_end,
                confidence=conf,
                n_repetitions=n_claps,
                detection_method="distance",
                is_response=is_response,
                latency_from_demo_sec=latency,
            ))
            attempt_id += 1

    # Speed-based detection for response window (if no distance-based found)
    if parent_clap_actions and not any(a.is_response for a in attempts):
        xy = np.array(child_best_xy, dtype=float)
        if xy.size > 0 and len(xy.shape) == 2 and xy.shape[0] > 1:
            # Apply jump limiter to prevent teleportation artifacts from hand swaps
            xy = apply_jump_limiter(xy, max_jump=0.18)

            dx = np.diff(xy[:, 0])
            dy = np.diff(xy[:, 1])
            dt = np.diff(t_arr)
            with np.errstate(invalid="ignore", divide="ignore"):
                speed = np.sqrt(dx * dx + dy * dy) / np.maximum(dt, 1e-6)
            t_speed = t_arr[1:]

            for pa in parent_clap_actions:
                mask = (t_speed >= pa.t_end) & (t_speed <= pa.t_end + response_window_sec)
                if not np.any(mask):
                    continue

                child_speed_window = interp_nans(speed)[mask]
                t_window = t_speed[mask]
                child_speed_smooth = smooth_med(child_speed_window, win=2)

                if float(np.nanmax(child_speed_smooth)) >= min_speed_abs:
                    peaks = detect_speed_peaks(t_window, child_speed_smooth, peak_thr_std=speed_peak_thr_std, min_sep_sec=0.18)

                    if len(peaks) >= min_peaks:
                        t_start = float(peaks[0])
                        t_end = float(peaks[-1]) if len(peaks) > 1 else t_start
                        latency = t_start - pa.t_end

                        conf = compute_action_confidence(
                            n_repetitions=len(peaks),
                            duration_sec=t_end - t_start + 0.1,
                            stability=0.5,
                            detector_conf=0.5,
                            periodicity_score=0.0,
                        )

                        attempts.append(ChildAttempt(
                            attempt_id=attempt_id,
                            action_type="CLAP",
                            t_start=t_start,
                            t_end=t_end,
                            confidence=conf,
                            n_repetitions=len(peaks),
                            detection_method="speed",
                            is_response=True,
                            latency_from_demo_sec=latency,
                        ))
                        attempt_id += 1

                    elif allow_single_burst and len(peaks) == 1 and float(np.nanmax(child_speed_smooth)) >= min_speed_abs * 1.35:
                        t_start = float(peaks[0])
                        latency = t_start - pa.t_end

                        conf = compute_action_confidence(
                            n_repetitions=1,
                            duration_sec=0.2,
                            stability=0.4,
                            detector_conf=0.5,
                            periodicity_score=0.0,
                        )

                        attempts.append(ChildAttempt(
                            attempt_id=attempt_id,
                            action_type="CLAP",
                            t_start=t_start,
                            t_end=t_start,
                            confidence=conf,
                            n_repetitions=1,
                            detection_method="speed_burst",
                            is_response=True,
                            latency_from_demo_sec=latency,
                        ))
                        attempt_id += 1

    return attempts


def detect_child_arms_attempts(
    t_arr: np.ndarray,
    child_arms_flags: List[bool],
    parent_arms_actions: List[ParentAction],
    response_window_sec: float,
    min_seg_dur: float = 0.12,
) -> List[ChildAttempt]:
    """
    Detect child ARMS_UP attempts and tag as responses to parent actions.
    """
    attempts: List[ChildAttempt] = []
    attempt_id = 0

    child_high = np.array(child_arms_flags, dtype=bool)
    child_high_segs = segments_from_bool(t_arr, child_high, min_dur=min_seg_dur)

    for seg_start, seg_end in child_high_segs:
        # Check if this is a response to a parent action
        is_response = False
        latency: Optional[float] = None
        for pa in parent_arms_actions:
            if pa.t_end <= seg_start <= pa.t_end + response_window_sec:
                is_response = True
                latency = seg_start - pa.t_end
                break

        conf = compute_action_confidence(
            n_repetitions=1,
            duration_sec=seg_end - seg_start,
            stability=0.8,
            detector_conf=0.7,
            periodicity_score=0.0,
        )

        attempts.append(ChildAttempt(
            attempt_id=attempt_id,
            action_type="ARMS_UP",
            t_start=seg_start,
            t_end=seg_end,
            confidence=conf,
            n_repetitions=1,
            detection_method="pose",
            is_response=is_response,
            latency_from_demo_sec=latency,
        ))
        attempt_id += 1

    # Also check individual frames for brief arm raises not forming segments
    if parent_arms_actions:
        for pa in parent_arms_actions:
            # Check if we already have a response for this action
            has_response = any(
                a.is_response and a.latency_from_demo_sec is not None
                for a in attempts
            )
            if has_response:
                continue

            # Check individual frames in response window
            for i, (t_frame, arm_up) in enumerate(zip(t_arr, child_arms_flags)):
                if arm_up and pa.t_end <= t_frame <= pa.t_end + response_window_sec:
                    # Require at least 2 consecutive frames
                    if i + 1 < len(child_arms_flags) and child_arms_flags[i + 1]:
                        latency = t_frame - pa.t_end
                        conf = compute_action_confidence(
                            n_repetitions=1,
                            duration_sec=0.1,
                            stability=0.6,
                            detector_conf=0.5,
                            periodicity_score=0.0,
                        )

                        attempts.append(ChildAttempt(
                            attempt_id=attempt_id,
                            action_type="ARMS_UP",
                            t_start=t_frame,
                            t_end=t_frame,
                            confidence=conf,
                            n_repetitions=1,
                            detection_method="pose_brief",
                            is_response=True,
                            latency_from_demo_sec=latency,
                        ))
                        attempt_id += 1
                        break

    return attempts


# ============================================================
# Trial matching
# ============================================================
def match_attempts_to_trials(
    parent_actions: List[ParentAction],
    child_attempts: List[ChildAttempt],
    response_window_sec: float,
) -> List[ImitationTrial]:
    """
    Match child attempts to parent demonstrations to form trials.

    For each parent action:
    1. Find all child attempts within response_window_sec after action end
    2. Mark first successful attempt (if any)
    3. Count attempts_before_success
    4. Compute imitation_latency
    """
    trials: List[ImitationTrial] = []

    for trial_id, pa in enumerate(parent_actions):
        # Find matching child attempts
        matching_attempts = [
            ca for ca in child_attempts
            if ca.action_type == pa.action_type and ca.is_response
            and ca.latency_from_demo_sec is not None
            and 0 <= ca.latency_from_demo_sec <= response_window_sec
        ]

        # Sort by time
        matching_attempts.sort(key=lambda a: a.t_start)

        # Determine success
        imitation_success = len(matching_attempts) > 0
        first_success_id = matching_attempts[0].attempt_id if imitation_success else None
        attempts_before = 0  # first attempt is success if any
        imitation_latency = matching_attempts[0].latency_from_demo_sec if imitation_success else None

        trials.append(ImitationTrial(
            trial_id=trial_id,
            action_type=pa.action_type,
            parent_action=pa,
            child_attempts=matching_attempts,
            first_success_attempt_id=first_success_id,
            attempts_before_success=attempts_before,
            imitation_success=imitation_success,
            imitation_latency_sec=imitation_latency,
        ))

    return trials


# ============================================================
# Main per-video
# ============================================================
def process_video(
    child_id: str,
    video_path: Path,
    hand_landmarker: Any,
    pose_landmarker: Any,
    sample_every_n: int,
    max_frames: int,
    response_window_sec: float,
    # parameters
    adult_speed_peak_thr_std: float,
    child_speed_peak_thr_std: float,
    adult_min_peaks: int,
    child_min_peaks: int,
    clap_min_claps_adult_dist: int,
    clap_min_claps_child_dist: int,
    clap_periodic_min_dist: float,
    clap_periodic_min_speed: float,
    # new guards / generosity
    adult_demo_min_speed_abs: float,
    adult_demo_min_speed_var: float,
    child_resp_min_speed_abs: float,
    child_resp_allow_single_burst: bool,
    arms_min_seg_dur: float,
    arms_raise_then_disappear: bool,
) -> Tuple[List[Event], Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {}

    fps = safe_fps(cap)
    fps_eff = fps / max(1, sample_every_n)

    times: List[float] = []
    adult_frame_hands: List[List[HandDet]] = []
    child_frame_hands: List[List[HandDet]] = []

    adult_best_xy: List[Tuple[float, float]] = []
    child_best_xy: List[Tuple[float, float]] = []

    child_arms_flags: List[bool] = []
    adult_arms_flags: List[bool] = []  # Track adult arm raising from pose
    events: List[Event] = []

    frame_idx = 0
    used = 0

    # Track previous child bbox for temporal continuity (prevents random flipping)
    prev_child_bbox: Optional[Tuple[float, float, float, float]] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if sample_every_n > 1 and (frame_idx % sample_every_n != 0):
            frame_idx += 1
            continue

        t = frame_idx / fps
        frame_idx += 1
        used += 1
        if used > max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Pose detection - detect multiple poses and classify as adult/child
        child_bb = None
        adult_bb = None
        child_arms = False
        adult_arms = False
        try:
            pres = pose_landmarker.detect(mp_image)
        except Exception:
            pres = None

        if pres is not None and getattr(pres, "pose_landmarks", None) and len(pres.pose_landmarks) > 0:
            # Use temporal tracking to prevent random child/adult flipping between frames
            child_pose, adult_pose, new_child_bb = choose_child_and_adult_poses_temporal(
                poses=list(pres.pose_landmarks),
                prev_child_bbox=prev_child_bbox,
                iou_weight=0.6,
                min_vis=0.30,
            )

            # Update temporal tracker
            if new_child_bb is not None:
                prev_child_bbox = new_child_bb
                child_bb = new_child_bb

            # Extract child info
            if child_pose is not None:
                if child_bb is None:
                    child_bb = pose_bbox(child_pose, min_vis=0.35)
                child_arms = child_arms_up(child_pose)

            # Extract adult info
            if adult_pose is not None:
                adult_bb = pose_bbox(adult_pose, min_vis=0.35)
                adult_arms = adult_arms_up(adult_pose)

        child_arms_flags.append(bool(child_arms))
        adult_arms_flags.append(bool(adult_arms))

        # Hands classify
        frame_adult: List[HandDet] = []
        frame_child: List[HandDet] = []

        try:
            hres = hand_landmarker.detect(mp_image)
        except Exception:
            hres = None

        if hres is not None and getattr(hres, "hand_landmarks", None):
            hands = hres.hand_landmarks
            handed = getattr(hres, "handedness", None)

            for i, lms in enumerate(hands):
                wrist = lms[0]
                wx, wy = float(wrist.x), float(wrist.y)

                conf = 0.5
                if handed and i < len(handed) and handed[i]:
                    top = handed[i][0]
                    conf = float(getattr(top, "score", 0.0) or 0.0)

                det = HandDet(t=float(t), wx=wx, wy=wy, conf=conf)

                # Classify hand as child or adult based on pose bboxes or fallback regions
                if child_bb is not None or adult_bb is not None:
                    in_child = child_bb is not None and point_in_bbox(wx, wy, child_bb)
                    in_adult = adult_bb is not None and point_in_bbox(wx, wy, adult_bb)

                    if in_child and not in_adult:
                        frame_child.append(det)
                    elif in_adult and not in_child:
                        frame_adult.append(det)
                    elif in_child and in_adult:
                        # Both boxes overlap - use fallback logic
                        if in_child_region_fallback(wx, wy):
                            frame_child.append(det)
                        else:
                            frame_adult.append(det)
                    else:
                        # Not in either bbox - use fallback
                        if in_child_region_fallback(wx, wy):
                            frame_child.append(det)
                        elif in_adult_region_fallback(wx, wy):
                            frame_adult.append(det)
                else:
                    # No pose bboxes - use fallback regions
                    if in_child_region_fallback(wx, wy):
                        frame_child.append(det)
                    elif in_adult_region_fallback(wx, wy):
                        frame_adult.append(det)

        adult_frame_hands.append(frame_adult)
        child_frame_hands.append(frame_child)

        if frame_adult:
            best = max(frame_adult, key=lambda d: d.conf)
            adult_best_xy.append((best.wx, best.wy))
        else:
            adult_best_xy.append((math.nan, math.nan))

        if frame_child:
            best = max(frame_child, key=lambda d: d.conf)
            child_best_xy.append((best.wx, best.wy))
        else:
            child_best_xy.append((math.nan, math.nan))

        times.append(float(t))

    cap.release()

    if len(times) == 0:
        return [], {}

    t_arr = np.array(times, dtype=float)

    # =========================
    # Use new two-channel detection
    # =========================

    # Detect parent actions
    parent_clap_actions = detect_parent_clap_actions(
        t_arr=t_arr,
        adult_frame_hands=adult_frame_hands,
        adult_best_xy=adult_best_xy,
        fps_eff=fps_eff,
        min_claps_dist=clap_min_claps_adult_dist,
        periodic_min_dist=clap_periodic_min_dist,
        speed_peak_thr_std=adult_speed_peak_thr_std,
        min_peaks=adult_min_peaks,
        periodic_min_speed=clap_periodic_min_speed,
        min_speed_abs=adult_demo_min_speed_abs,
        min_speed_var=adult_demo_min_speed_var,
    )

    parent_arms_actions = detect_parent_arms_actions(
        t_arr=t_arr,
        adult_frame_hands=adult_frame_hands,
        adult_arms_flags=adult_arms_flags,
        adult_best_xy=adult_best_xy,
        arms_min_seg_dur=arms_min_seg_dur,
        use_raise_then_disappear=arms_raise_then_disappear,
    )

    # Detect child attempts
    child_clap_attempts = detect_child_clap_attempts(
        t_arr=t_arr,
        child_frame_hands=child_frame_hands,
        child_best_xy=child_best_xy,
        parent_clap_actions=parent_clap_actions,
        fps_eff=fps_eff,
        response_window_sec=response_window_sec,
        min_claps_dist=clap_min_claps_child_dist,
        periodic_min_dist=clap_periodic_min_dist,
        speed_peak_thr_std=child_speed_peak_thr_std,
        min_peaks=child_min_peaks,
        min_speed_abs=child_resp_min_speed_abs,
        allow_single_burst=child_resp_allow_single_burst,
    )

    child_arms_attempts = detect_child_arms_attempts(
        t_arr=t_arr,
        child_arms_flags=child_arms_flags,
        parent_arms_actions=parent_arms_actions,
        response_window_sec=response_window_sec,
        min_seg_dur=0.12,
    )

    # Match attempts to trials
    clap_trials = match_attempts_to_trials(parent_clap_actions, child_clap_attempts, response_window_sec)
    arms_trials = match_attempts_to_trials(parent_arms_actions, child_arms_attempts, response_window_sec)

    # =========================
    # Generate events (backward compatible format)
    # =========================

    # Parent clap events
    for pa in parent_clap_actions:
        events.append(Event(child_id, "imitation", "adult", "CLAP", "PARENT_ACTION_START", pa.t_start, pa.confidence, pa.detection_method))
        events.append(Event(child_id, "imitation", "adult", "CLAP", "PARENT_ACTION_END", pa.t_end, pa.confidence, pa.detection_method))
        # Also emit old-style events for backward compat
        events.append(Event(child_id, "imitation", "adult", "CLAP", "DEMO_START", pa.t_start, pa.confidence, pa.detection_method))
        events.append(Event(child_id, "imitation", "adult", "CLAP", "DEMO_END", pa.t_end, pa.confidence, pa.detection_method))

    # Parent arms events
    for pa in parent_arms_actions:
        events.append(Event(child_id, "imitation", "adult", "ARMS_UP", "PARENT_ACTION_START", pa.t_start, pa.confidence, pa.detection_method))
        events.append(Event(child_id, "imitation", "adult", "ARMS_UP", "PARENT_ACTION_END", pa.t_end, pa.confidence, pa.detection_method))
        # Also emit old-style events for backward compat
        events.append(Event(child_id, "imitation", "adult", "ARMS_UP", "DEMO_START", pa.t_start, pa.confidence, pa.detection_method))
        events.append(Event(child_id, "imitation", "adult", "ARMS_UP", "DEMO_END", pa.t_end, pa.confidence, pa.detection_method))

    # Child clap attempts
    for ca in child_clap_attempts:
        events.append(Event(child_id, "imitation", "child", "CLAP", "CHILD_ATTEMPT_START", ca.t_start, ca.confidence, ca.detection_method))
        events.append(Event(child_id, "imitation", "child", "CLAP", "CHILD_ATTEMPT_END", ca.t_end, ca.confidence, ca.detection_method))
        if ca.is_response:
            events.append(Event(child_id, "imitation", "child", "CLAP", "RESPONSE", ca.t_start, ca.confidence, "after_demo"))

    # Child arms attempts
    for ca in child_arms_attempts:
        events.append(Event(child_id, "imitation", "child", "ARMS_UP", "CHILD_ATTEMPT_START", ca.t_start, ca.confidence, ca.detection_method))
        events.append(Event(child_id, "imitation", "child", "ARMS_UP", "CHILD_ATTEMPT_END", ca.t_end, ca.confidence, ca.detection_method))
        if ca.is_response:
            events.append(Event(child_id, "imitation", "child", "ARMS_UP", "RESPONSE", ca.t_start, ca.confidence, "after_demo"))

    # Trial success/failure events
    for trial in clap_trials:
        if trial.imitation_success:
            latency_str = f"latency={trial.imitation_latency_sec:.3f}" if trial.imitation_latency_sec else ""
            events.append(Event(child_id, "imitation", "child", "CLAP", "IMITATION_SUCCESS",
                                trial.child_attempts[0].t_start if trial.child_attempts else trial.parent_action.t_end,
                                trial.child_attempts[0].confidence if trial.child_attempts else 0.5, latency_str))
        else:
            events.append(Event(child_id, "imitation", "child", "CLAP", "IMITATION_FAILURE",
                                trial.parent_action.t_end, 0.0, "no_response"))

    for trial in arms_trials:
        if trial.imitation_success:
            latency_str = f"latency={trial.imitation_latency_sec:.3f}" if trial.imitation_latency_sec else ""
            events.append(Event(child_id, "imitation", "child", "ARMS_UP", "IMITATION_SUCCESS",
                                trial.child_attempts[0].t_start if trial.child_attempts else trial.parent_action.t_end,
                                trial.child_attempts[0].confidence if trial.child_attempts else 0.5, latency_str))
        else:
            events.append(Event(child_id, "imitation", "child", "ARMS_UP", "IMITATION_FAILURE",
                                trial.parent_action.t_end, 0.0, "no_response"))

    # =========================
    # Compute summary metrics (backward compatible + new)
    # =========================

    # Clap metrics
    clap_demo_present = 1 if parent_clap_actions else 0
    clap_demo_method = parent_clap_actions[0].detection_method if parent_clap_actions else ""
    adult_clap_count = sum(pa.n_repetitions for pa in parent_clap_actions)
    child_clap_count = len(child_clap_attempts)
    clap_response_present = 1 if any(t.imitation_success for t in clap_trials) else 0
    _clap_lat = clap_trials[0].imitation_latency_sec if clap_trials and clap_trials[0].imitation_success else None
    clap_latency = _clap_lat if _clap_lat is not None else math.nan
    clap_attempts_before_success = clap_trials[0].attempts_before_success if clap_trials and clap_trials[0].imitation_success else 0
    clap_total_child_attempts = len(child_clap_attempts)
    clap_mean_attempt_conf = float(np.mean([ca.confidence for ca in child_clap_attempts])) if child_clap_attempts else 0.0

    # Arms metrics
    arms_demo_present = 1 if parent_arms_actions else 0
    arms_response_present = 1 if any(t.imitation_success for t in arms_trials) else 0
    _arms_lat = arms_trials[0].imitation_latency_sec if arms_trials and arms_trials[0].imitation_success else None
    arms_latency = _arms_lat if _arms_lat is not None else math.nan
    child_arms_up_total = len(child_arms_attempts)
    arms_attempts_before_success = arms_trials[0].attempts_before_success if arms_trials and arms_trials[0].imitation_success else 0
    arms_total_child_attempts = len(child_arms_attempts)
    arms_mean_attempt_conf = float(np.mean([ca.confidence for ca in child_arms_attempts])) if child_arms_attempts else 0.0

    # Overall metrics
    demo_prims = clap_demo_present + arms_demo_present
    resp_prims = clap_response_present + arms_response_present
    imitation_score = float(resp_prims / demo_prims) if demo_prims > 0 else 0.0

    total_attempts_before_success = clap_attempts_before_success + arms_attempts_before_success

    # Mean latency across successful imitations
    latencies = []
    if clap_trials and clap_trials[0].imitation_success and clap_trials[0].imitation_latency_sec is not None:
        latencies.append(clap_trials[0].imitation_latency_sec)
    if arms_trials and arms_trials[0].imitation_success and arms_trials[0].imitation_latency_sec is not None:
        latencies.append(arms_trials[0].imitation_latency_sec)
    mean_imitation_latency = float(np.mean(latencies)) if latencies else math.nan

    summary: Dict[str, Any] = {
        "child_id": child_id,
        "task_type": "imitation",

        # Backward compatible fields
        "clap_demo_present": int(clap_demo_present),
        "clap_demo_method": clap_demo_method,
        "adult_clap_count": int(adult_clap_count),
        "child_clap_count": int(child_clap_count),
        "clap_response_present": int(clap_response_present),
        "clap_latency_sec": "" if not np.isfinite(clap_latency) else round(float(clap_latency), 3),

        "arms_demo_present": int(arms_demo_present),
        "arms_response_present": int(arms_response_present),
        "arms_latency_sec": "" if not np.isfinite(arms_latency) else round(float(arms_latency), 3),
        "child_arms_up_total": int(child_arms_up_total),

        "demo_primitives": int(demo_prims),
        "responded_primitives": int(resp_prims),
        "imitation_score": round(float(imitation_score), 4),

        # NEW fields for attempt tracking
        "clap_attempts_before_success": int(clap_attempts_before_success),
        "clap_total_child_attempts": int(clap_total_child_attempts),
        "clap_mean_attempt_confidence": round(float(clap_mean_attempt_conf), 4),

        "arms_attempts_before_success": int(arms_attempts_before_success),
        "arms_total_child_attempts": int(arms_total_child_attempts),
        "arms_mean_attempt_confidence": round(float(arms_mean_attempt_conf), 4),

        "total_attempts_before_success": int(total_attempts_before_success),
        "mean_imitation_latency_sec": "" if not np.isfinite(mean_imitation_latency) else round(float(mean_imitation_latency), 3),
    }

    return events, summary


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Imitation (CLAP + ARMS_UP only) — imitate AFTER, generous time, fixed edge cases.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_events", default="data/derived/imit_events.csv")
    ap.add_argument("--out_summary", default="data/derived/imit_summary.csv")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--sample_every_n", type=int, default=2)
    ap.add_argument("--max_frames", type=int, default=900)

    # generous after-demo response window
    ap.add_argument("--response_window_sec", type=float, default=7.5)

    # adult demo: speed fallback knobs + jitter guards
    ap.add_argument("--adult_speed_peak_thr_std", type=float, default=2.1)
    ap.add_argument("--adult_min_peaks", type=int, default=2)
    ap.add_argument("--clap_periodic_min_speed", type=float, default=6.0)
    ap.add_argument("--adult_demo_min_speed_abs", type=float, default=0.06)   # guards false demos (child 1)
    ap.add_argument("--adult_demo_min_speed_var", type=float, default=0.0006) # guards false demos

    # distance-based clap knobs
    ap.add_argument("--clap_min_claps_adult_dist", type=int, default=2)
    ap.add_argument("--clap_min_claps_child_dist", type=int, default=1)
    ap.add_argument("--clap_periodic_min_dist", type=float, default=8.0)

    # child response speed fallback (more generous for child 15)
    ap.add_argument("--child_speed_peak_thr_std", type=float, default=2.1)
    ap.add_argument("--child_min_peaks", type=int, default=1)
    ap.add_argument("--child_resp_min_speed_abs", type=float, default=0.055)
    ap.add_argument("--child_resp_allow_single_burst", action="store_true")

    # arms up
    ap.add_argument("--arms_min_seg_dur", type=float, default=0.18)
    ap.add_argument("--arms_raise_then_disappear", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if "imitation_path" not in df.columns:
        raise ValueError(f"Manifest missing 'imitation_path'. Columns: {list(df.columns)}")

    hand_model, pose_model = get_models()

    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=4,  # Reduced from 10 to limit low-quality detections
        )
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=3,  # Detect multiple poses to distinguish adult vs child
        )
    )

    out_e = Path(args.out_events)
    out_s = Path(args.out_summary)
    out_e.parent.mkdir(parents=True, exist_ok=True)
    out_s.parent.mkdir(parents=True, exist_ok=True)

    with open(out_e, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["child_id", "task_type", "subject", "primitive", "kind", "t_sec", "confidence", "meta"],
        )
        w.writeheader()

    summary_fields = [
        "child_id", "task_type",
        # Backward compatible fields
        "clap_demo_present", "clap_demo_method", "adult_clap_count", "child_clap_count",
        "clap_response_present", "clap_latency_sec",
        "arms_demo_present", "arms_response_present", "arms_latency_sec", "child_arms_up_total",
        "demo_primitives", "responded_primitives", "imitation_score",
        # NEW fields for attempt tracking
        "clap_attempts_before_success", "clap_total_child_attempts", "clap_mean_attempt_confidence",
        "arms_attempts_before_success", "arms_total_child_attempts", "arms_mean_attempt_confidence",
        "total_attempts_before_success", "mean_imitation_latency_sec",
    ]
    with open(out_s, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    for i in range(n):
        child_id = str(df.iloc[i]["child_id"])
        vp = Path(str(df.iloc[i]["imitation_path"]))
        if not vp.exists():
            continue

        print(f"[{i+1}/{n}] child={child_id} imitation video={vp.name}", flush=True)

        events, summary = process_video(
            child_id=child_id,
            video_path=vp,
            hand_landmarker=hand_landmarker,
            pose_landmarker=pose_landmarker,
            sample_every_n=args.sample_every_n,
            max_frames=args.max_frames,
            response_window_sec=args.response_window_sec,

            adult_speed_peak_thr_std=args.adult_speed_peak_thr_std,
            child_speed_peak_thr_std=args.child_speed_peak_thr_std,
            adult_min_peaks=args.adult_min_peaks,
            child_min_peaks=args.child_min_peaks,
            clap_min_claps_adult_dist=args.clap_min_claps_adult_dist,
            clap_min_claps_child_dist=args.clap_min_claps_child_dist,
            clap_periodic_min_dist=args.clap_periodic_min_dist,
            clap_periodic_min_speed=args.clap_periodic_min_speed,

            adult_demo_min_speed_abs=args.adult_demo_min_speed_abs,
            adult_demo_min_speed_var=args.adult_demo_min_speed_var,
            child_resp_min_speed_abs=args.child_resp_min_speed_abs,
            child_resp_allow_single_burst=bool(args.child_resp_allow_single_burst),
            arms_min_seg_dur=args.arms_min_seg_dur,
            arms_raise_then_disappear=bool(args.arms_raise_then_disappear),
        )

        with open(out_e, "a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["child_id", "task_type", "subject", "primitive", "kind", "t_sec", "confidence", "meta"],
            )
            for e in events:
                w.writerow(
                    {
                        "child_id": e.child_id,
                        "task_type": e.task_type,
                        "subject": e.subject,
                        "primitive": e.primitive,
                        "kind": e.kind,
                        "t_sec": round(float(e.t_sec), 3),
                        "confidence": round(float(e.confidence), 4),
                        "meta": e.meta,
                    }
                )

        with open(out_s, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            row = {k: "" for k in summary_fields}
            row.update(summary)
            w.writerow(row)

    hand_landmarker.close()
    pose_landmarker.close()
    print(f"Wrote: {out_e}")
    print(f"Wrote: {out_s}")


if __name__ == "__main__":
    main()


