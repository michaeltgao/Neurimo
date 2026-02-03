from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import numpy as np


# Pose indices (MediaPipe pose landmarker)
POSE_NOSE = 0
POSE_L_EYE_INNER = 1
POSE_R_EYE_INNER = 4
POSE_L_EAR = 7
POSE_R_EAR = 8
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
POSE_L_WRIST = 15
POSE_R_WRIST = 16

# Small epsilon for division safety
EPS = 1e-8


@dataclass
class CommonConfig:
    # Visibility thresholds (normalized 0..1)
    pose_vis_thr: float = 0.35

    # Fraction of landmarks that must be visible for pose to be "present"
    pose_landmark_frac_thr: float = 0.35

    # Stillness threshold in normalized units/sec for wrist speed
    still_speed_thr: float = 0.020

    # Autocorr settings
    autocorr_min_lag_sec: float = 0.20
    autocorr_max_lag_sec: float = 2.00

    # --- Paper-aligned feature thresholds ---

    # Gaze proxy: ear visibility asymmetry threshold
    # If |left_ear_vis - right_ear_vis| > threshold, head is turned away
    ear_asymmetry_thr: float = 0.25

    # Gaze proxy: nose x deviation from shoulder midpoint
    # If nose deviates > threshold from midpoint, likely looking away
    nose_deviation_thr: float = 0.08

    # Gaze proxy: nose-to-parent center threshold (normalized frame width)
    # Only triggers if parent bbox valid AND deviation exceeds this
    nose_to_parent_thr: float = 0.15

    # Non-engaged movement: centroid speed threshold (normalized units/sec)
    # Speed above this = restless/roaming behavior
    roaming_speed_thr: float = 0.03

    # Non-engaged movement: jitter threshold (std of velocity)
    jitter_thr: float = 0.015

    # Non-engaged movement: max gap (seconds) to interpolate across
    # Larger gaps are treated as discontinuities (no speed computed)
    max_interp_gap_sec: float = 0.5

    # Proximity detection: bbox IoU threshold
    proximity_iou_thr: float = 0.02

    # Proximity detection: center distance threshold (normalized units)
    # Closer than this = close proximity
    proximity_dist_thr: float = 0.20

    # Minimum confidence for bbox to be considered valid
    bbox_conf_thr: float = 0.3


def _interp_1d(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs. If too many NaNs, returns all-NaN."""
    if x.size == 0:
        return x
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if good.sum() < max(8, int(0.10 * x.size)):
        return np.full_like(x, np.nan, dtype=float)
    out = x.astype(float).copy()
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def _safe_dt(t: np.ndarray) -> np.ndarray:
    if t.size < 2:
        return np.array([], dtype=float)
    dt = np.diff(t.astype(float))
    dt[dt <= 1e-6] = np.nan
    return dt


def _speed_from_xy(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """
    xy: (N,2) normalized coordinates
    returns speed per step: (N-1,)
    """
    if t.size < 2 or xy.shape[0] < 2:
        return np.array([], dtype=float)
    dt = _safe_dt(t)
    dx = np.diff(xy[:, 0].astype(float))
    dy = np.diff(xy[:, 1].astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        sp = np.sqrt(dx * dx + dy * dy) / dt
    return sp


def _autocorr_peak(x: np.ndarray, fps_eff: float, min_lag_sec: float, max_lag_sec: float) -> float:
    """
    Returns the maximum autocorrelation value in [min_lag_sec, max_lag_sec].
    x should be 1D finite vector.
    """
    if x.size < 20 or not np.isfinite(x).all():
        return 0.0
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0

    # convert lag bounds to samples
    min_lag = int(max(1, round(min_lag_sec * fps_eff)))
    max_lag = int(max(min_lag + 1, round(max_lag_sec * fps_eff)))
    max_lag = min(max_lag, x.size - 2)
    if max_lag <= min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        v = float(np.dot(x[:-lag], x[lag:]) / denom)
        if v > best:
            best = v
    return float(best)


# -----------------------------------------------------------------------------
# Paper-aligned feature helpers
# -----------------------------------------------------------------------------


def _bbox_centroid(bbox: np.ndarray) -> np.ndarray:
    """
    Extract centroid (cx, cy) from bbox array.
    bbox: (N, 5) with [x0, y0, x1, y1, conf] or (N, 4) with [x0, y0, x1, y1]
    Returns: (N, 2) array of centroids
    """
    cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
    cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
    return np.column_stack([cx, cy])


def _bbox_valid(bbox: Optional[np.ndarray], conf_thr: float = 0.3) -> np.ndarray:
    """
    Returns boolean mask for frames with valid bbox.
    Valid = coordinates finite AND confidence >= threshold (if conf column exists)
    """
    if bbox is None or bbox.shape[0] == 0:
        return np.array([], dtype=bool)

    # Check coordinates are finite
    coords_valid = np.isfinite(bbox[:, :4]).all(axis=1)

    # Check confidence if present
    if bbox.shape[1] >= 5:
        conf_valid = bbox[:, 4] >= conf_thr
        return coords_valid & conf_valid
    return coords_valid


def _bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute IoU between two bboxes [x0, y0, x1, y1, ...].
    Returns 0.0 if either bbox is invalid.
    """
    if not np.isfinite(bbox1[:4]).all() or not np.isfinite(bbox2[:4]).all():
        return 0.0

    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    inter = (x1 - x0) * (y1 - y0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def _bbox_center_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Euclidean distance between bbox centers.
    Returns NaN if either bbox is invalid.
    """
    if not np.isfinite(bbox1[:4]).all() or not np.isfinite(bbox2[:4]).all():
        return float("nan")

    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0

    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def _point_in_bbox(x: float, y: float, bbox: np.ndarray) -> bool:
    """Check if point (x, y) is inside bbox [x0, y0, x1, y1, ...]."""
    if not np.isfinite([x, y]).all() or not np.isfinite(bbox[:4]).all():
        return False
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _compute_gaze_away_mask(
    pose: np.ndarray,
    parent_bbox: Optional[np.ndarray],
    cfg: CommonConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame mask for "not looking at parent" using gaze proxy.

    Gaze proxy uses head orientation estimated from:
    1. Ear visibility asymmetry (if one ear much more visible, head is turned)
    2. Nose x-position relative to shoulder midpoint

    Optional (only if parent bbox valid):
    3. Nose x-position relative to parent bbox center

    Returns:
        (gaze_away_mask, gaze_valid_mask)
        - gaze_away_mask: (N,) bool, True when child is NOT looking at parent
        - gaze_valid_mask: (N,) bool, True when gaze can be estimated
    """
    N = pose.shape[0]
    gaze_away = np.zeros(N, dtype=bool)
    gaze_valid = np.zeros(N, dtype=bool)

    # Extract keypoints and visibility (vectorized)
    nose_x = pose[:, POSE_NOSE, 0]
    nose_vis = pose[:, POSE_NOSE, 3]
    l_ear_vis = pose[:, POSE_L_EAR, 3]
    r_ear_vis = pose[:, POSE_R_EAR, 3]
    l_shoulder_x = pose[:, POSE_L_SHOULDER, 0]
    r_shoulder_x = pose[:, POSE_R_SHOULDER, 0]
    l_shoulder_vis = pose[:, POSE_L_SHOULDER, 3]
    r_shoulder_vis = pose[:, POSE_R_SHOULDER, 3]

    # Precompute parent bbox validity once (not per-frame)
    if parent_bbox is not None and parent_bbox.shape[0] == N:
        parent_ok = _bbox_valid(parent_bbox, cfg.bbox_conf_thr)
        parent_cx = (parent_bbox[:, 0] + parent_bbox[:, 2]) / 2.0
    else:
        parent_ok = np.zeros(N, dtype=bool)
        parent_cx = np.full(N, np.nan)

    # Vectorized validity checks
    nose_ok = nose_vis >= cfg.pose_vis_thr
    has_l_ear = l_ear_vis >= cfg.pose_vis_thr
    has_r_ear = r_ear_vis >= cfg.pose_vis_thr
    has_any_ear = has_l_ear | has_r_ear

    # Gaze valid = nose visible AND at least one ear visible
    gaze_valid = nose_ok & has_any_ear

    # Method 1: Ear asymmetry (vectorized)
    ear_asymmetry = np.abs(l_ear_vis - r_ear_vis)
    away_ear = ear_asymmetry > cfg.ear_asymmetry_thr

    # Method 2: Nose deviation from shoulder midpoint (vectorized)
    has_shoulders = (l_shoulder_vis >= cfg.pose_vis_thr) & (r_shoulder_vis >= cfg.pose_vis_thr)
    shoulder_mid = (l_shoulder_x + r_shoulder_x) / 2.0
    nose_dev = np.abs(nose_x - shoulder_mid)
    away_shoulders = has_shoulders & np.isfinite(nose_dev) & (nose_dev > cfg.nose_deviation_thr)

    # Method 3: Nose to parent (only when parent_ok, uses configurable threshold)
    nose_to_parent_dist = np.abs(nose_x - parent_cx)
    away_parent = parent_ok & np.isfinite(nose_to_parent_dist) & (nose_to_parent_dist > cfg.nose_to_parent_thr)

    # Combine: looking away if ANY method triggers (while gaze is valid)
    gaze_away = gaze_valid & (away_ear | away_shoulders | away_parent)

    return gaze_away, gaze_valid


def _compute_non_engaged_movement_mask(
    t_sec: np.ndarray,
    child_bbox: Optional[np.ndarray],
    pose: np.ndarray,
    cfg: CommonConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-interval mask for "non-engaged movement" (restlessness/roaming).

    Uses:
    1. Child bbox centroid velocity (high = roaming)
    2. Centroid jitter (high variability = restlessness)

    Gap handling: Intervals that cross large time gaps (> max_interp_gap_sec) are
    excluded to avoid interpolation artifacts creating artificial speed spikes.

    Returns:
        (non_engaged_mask, movement_valid_mask)
        - non_engaged_mask: (N-1,) bool for intervals, True = restless/roaming
        - movement_valid_mask: (N-1,) bool for intervals, True = movement measurable
    """
    N = len(t_sec)
    if N < 2:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    # Interval-based: we have N-1 intervals between N frames
    n_intervals = N - 1
    non_engaged = np.zeros(n_intervals, dtype=bool)
    interval_valid = np.zeros(n_intervals, dtype=bool)

    # Compute dt for each interval
    dt = np.diff(t_sec.astype(float))

    # Detect gaps: intervals where dt > max_interp_gap_sec are invalid
    gap_mask = dt > cfg.max_interp_gap_sec

    # Try bbox centroid first, fall back to pose nose
    raw_ok: np.ndarray
    if child_bbox is not None and child_bbox.shape[0] == N:
        bbox_ok = _bbox_valid(child_bbox, cfg.bbox_conf_thr)
        if bbox_ok.sum() > 10:
            centroid = _bbox_centroid(child_bbox)
            cx_raw = np.where(bbox_ok, centroid[:, 0], np.nan)
            cy_raw = np.where(bbox_ok, centroid[:, 1], np.nan)
            raw_ok = bbox_ok
        else:
            cx_raw = cy_raw = None
            raw_ok = np.zeros(N, dtype=bool)
    else:
        cx_raw = cy_raw = None
        raw_ok = np.zeros(N, dtype=bool)

    # Fallback to pose nose position
    if cx_raw is None or raw_ok.sum() < 10:
        nose_xy = pose[:, POSE_NOSE, :2].astype(float)
        nose_vis = pose[:, POSE_NOSE, 3]
        raw_ok = nose_vis >= cfg.pose_vis_thr
        if raw_ok.sum() > 10:
            cx_raw = np.where(raw_ok, nose_xy[:, 0], np.nan)
            cy_raw = np.where(raw_ok, nose_xy[:, 1], np.nan)
        else:
            return non_engaged, interval_valid

    if cx_raw is None:
        return non_engaged, interval_valid

    # Interval validity: both endpoints must be valid AND no gap
    interval_valid = raw_ok[:-1] & raw_ok[1:] & ~gap_mask

    # Compute speed only on valid intervals (no interpolation needed)
    speed = np.full(n_intervals, np.nan)
    dx = np.diff(cx_raw)
    dy = np.diff(cy_raw)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.sqrt(dx * dx + dy * dy) / dt
    speed[~interval_valid] = np.nan

    # Compute rolling jitter using only valid speeds
    # Use a window of ~0.5 sec at estimated fps
    dt_med = float(np.nanmedian(dt)) if np.isfinite(dt).any() else 1.0 / 30.0
    window = max(3, min(15, int(0.5 / dt_med)))

    jitter = np.zeros(n_intervals)
    for i in range(n_intervals):
        if not interval_valid[i]:
            continue
        start = max(0, i - window // 2)
        end = min(n_intervals, i + window // 2 + 1)
        win_speed = speed[start:end]
        win_valid = interval_valid[start:end]
        win_speed = win_speed[win_valid & np.isfinite(win_speed)]
        if win_speed.size >= 3:
            jitter[i] = float(np.std(win_speed))

    # Mark intervals as non-engaged if high speed or high jitter
    for i in range(n_intervals):
        if not interval_valid[i]:
            continue
        sp = speed[i] if np.isfinite(speed[i]) else 0.0
        jit = jitter[i]

        if sp > cfg.roaming_speed_thr or jit > cfg.jitter_thr:
            non_engaged[i] = True

    return non_engaged, interval_valid


def _compute_proximity_mask(
    child_bbox: Optional[np.ndarray],
    parent_bbox: Optional[np.ndarray],
    parent_pose: Optional[np.ndarray],
    cfg: CommonConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame mask for parent-child proximity.

    NOTE: This is a proximity proxy, not true physical contact detection.
    True contact detection would require parent wrist keypoints, which are
    not currently stored in tracks. When parent_pose is available, we use
    wrist-in-bbox as a stronger signal; otherwise we fall back to bbox
    overlap/distance which is a weaker proximity indicator.

    Methods (in order of preference):
    1. Parent wrist keypoints inside child bbox (if parent_pose available)
    2. Bbox IoU >= threshold (overlap proxy)
    3. Bbox center distance <= threshold (closeness proxy)

    Returns:
        (proximity_mask, proximity_valid_mask)
        - proximity_mask: (N,) bool, True when close proximity detected
        - proximity_valid_mask: (N,) bool, True when proximity can be measured
    """
    if child_bbox is None or child_bbox.shape[0] == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    N = child_bbox.shape[0]
    proximity = np.zeros(N, dtype=bool)
    proximity_valid = np.zeros(N, dtype=bool)

    child_ok = _bbox_valid(child_bbox, cfg.bbox_conf_thr)

    # Method 1: Parent wrist keypoints inside child bbox (preferred, rare)
    if parent_pose is not None and parent_pose.shape[0] == N:
        parent_lwrist = parent_pose[:, POSE_L_WRIST, :2]
        parent_rwrist = parent_pose[:, POSE_R_WRIST, :2]
        parent_lwrist_vis = parent_pose[:, POSE_L_WRIST, 3]
        parent_rwrist_vis = parent_pose[:, POSE_R_WRIST, 3]

        for i in range(N):
            if not child_ok[i]:
                continue

            l_visible = parent_lwrist_vis[i] >= cfg.pose_vis_thr
            r_visible = parent_rwrist_vis[i] >= cfg.pose_vis_thr

            if l_visible or r_visible:
                proximity_valid[i] = True

            l_contact = l_visible and _point_in_bbox(
                parent_lwrist[i, 0], parent_lwrist[i, 1], child_bbox[i]
            )
            r_contact = r_visible and _point_in_bbox(
                parent_rwrist[i, 0], parent_rwrist[i, 1], child_bbox[i]
            )

            if l_contact or r_contact:
                proximity[i] = True

        if proximity_valid.any():
            return proximity, proximity_valid

    # Method 2: Fallback to bbox proximity (weaker signal)
    if parent_bbox is None or parent_bbox.shape[0] != N:
        return proximity, proximity_valid

    parent_ok = _bbox_valid(parent_bbox, cfg.bbox_conf_thr)

    # Vectorized IoU and distance computation
    for i in range(N):
        if not (child_ok[i] and parent_ok[i]):
            continue

        proximity_valid[i] = True

        iou = _bbox_iou(child_bbox[i], parent_bbox[i])
        if iou >= cfg.proximity_iou_thr:
            proximity[i] = True
            continue

        dist = _bbox_center_distance(child_bbox[i], parent_bbox[i])
        if np.isfinite(dist) and dist <= cfg.proximity_dist_thr:
            proximity[i] = True

    return proximity, proximity_valid


def load_tracks_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (t_sec, pose, lh, rh)
    pose: (N,33,4)
    lh/rh: (N,21,4) OR empty (0,21,4)

    Note: For new format with bbox fields, use load_tracks_npz_extended().
    """
    data = np.load(npz_path, allow_pickle=False)
    t = data["t_sec"].astype(float)
    pose = data["pose"].astype(float)
    lh = data["lh"].astype(float)
    rh = data["rh"].astype(float)
    return t, pose, lh, rh


def load_tracks_npz_extended(
    npz_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], bool, float, int]:
    """
    Load tracks with backward compatibility for old/new NPZ formats.

    Returns:
        (t_sec, pose, lh, rh, child_bbox, parent_bbox, is_smoothed, fps, sample_every_n)

        - t_sec: (N,) timestamps
        - pose: (N,33,4) child pose landmarks
        - lh: (N,21,4) left hand OR empty
        - rh: (N,21,4) right hand OR empty
        - child_bbox: (N,5) [x0,y0,x1,y1,conf] or None if old format
        - parent_bbox: (N,5) [x0,y0,x1,y1,conf] or None if old format
        - is_smoothed: bool, False if old format or not smoothed
        - fps: float, 0.0 if old format
        - sample_every_n: int, 1 if old format
    """
    data = np.load(npz_path, allow_pickle=False)

    # Required fields (always present)
    t = data["t_sec"].astype(float)
    pose = data["pose"].astype(float)
    lh = data["lh"].astype(float)
    rh = data["rh"].astype(float)

    # New fields with defaults for old format
    child_bbox = None
    parent_bbox = None
    is_smoothed = False
    fps = 0.0
    sample_every_n = 1

    if "child_bbox" in data:
        child_bbox = data["child_bbox"].astype(float)
    if "parent_bbox" in data:
        parent_bbox = data["parent_bbox"].astype(float)
    if "is_smoothed" in data:
        is_smoothed = bool(data["is_smoothed"][0])
    if "fps" in data:
        fps = float(data["fps"][0])
    if "sample_every_n" in data:
        sample_every_n = int(data["sample_every_n"][0])

    return t, pose, lh, rh, child_bbox, parent_bbox, is_smoothed, fps, sample_every_n


def compute_common_features_from_tracks(
    t_sec: np.ndarray,
    pose: np.ndarray,
    lh: np.ndarray,
    rh: np.ndarray,
    cfg: Optional[CommonConfig] = None,
    face_present_ratio: Optional[float] = None,
    child_bbox: Optional[np.ndarray] = None,
    parent_bbox: Optional[np.ndarray] = None,
    parent_pose: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes common/global features from tracks arrays.

    Args:
        t_sec: (N,) timestamps
        pose: (N, 33, 4) child pose landmarks
        lh: (N, 21, 4) left hand landmarks
        rh: (N, 21, 4) right hand landmarks
        cfg: CommonConfig with thresholds
        face_present_ratio: From QC data, or None
        child_bbox: (N, 5) child bounding box [x0,y0,x1,y1,conf], optional
        parent_bbox: (N, 5) parent bounding box, optional
        parent_pose: (N, 33, 4) parent pose landmarks, optional (for contact detection)

    Returns:
        Dict with all common features including paper-aligned features:
        - lack_of_eye_contact_duration(_adj)
        - non_engaged_movement_duration(_adj)
        - physical_contact_duration(_adj)
    """
    cfg = cfg or CommonConfig()

    N = int(t_sec.shape[0])
    duration_sec = float(t_sec[-1] - t_sec[0]) if N >= 2 else 0.0

    out: Dict[str, float] = {
        "n_frames": float(N),
        "duration_sec": duration_sec,
        "face_present_ratio": float(face_present_ratio) if face_present_ratio is not None else float("nan"),
    }

    # Paper-aligned feature defaults
    paper_defaults = {
        "lack_of_eye_contact_duration": 0.0,
        "lack_of_eye_contact_duration_adj": float("nan"),
        "non_engaged_movement_duration": 0.0,
        "non_engaged_movement_duration_adj": float("nan"),
        "physical_contact_duration": 0.0,
        "physical_contact_duration_adj": float("nan"),
    }

    if N < 8 or pose.shape[0] != N:
        # too short / malformed
        out.update(
            {
                "pose_present_ratio": 0.0,
                "out_of_view_ratio": 1.0,
                "head_turn_rate": 0.0,
                "motion_energy_hands": 0.0,
                "repetitive_motion_score": 0.0,
                "stillness_ratio": 1.0,
                "hands_present_ratio": 0.0,
            }
        )
        out.update(paper_defaults)
        return out

    # pose present per frame: require some landmarks with vis >= threshold
    vis = pose[:, :, 3]
    pose_present = np.nanmean((vis >= cfg.pose_vis_thr).astype(float), axis=1)  # (N,)
    pose_ok = pose_present > cfg.pose_landmark_frac_thr
    pose_present_ratio = float(np.mean(pose_ok))

    out["pose_present_ratio"] = pose_present_ratio
    out["out_of_view_ratio"] = float(1.0 - pose_present_ratio)

    # hands present ratio (child-only hands already)
    def _hand_present(h: np.ndarray) -> np.ndarray:
        if h.shape[0] != N:  # empty stored (0,21,4)
            return np.zeros((N,), dtype=bool)
        # consider present if wrist x,y finite
        wrist_xy = h[:, 0, 0:2]
        return np.isfinite(wrist_xy).all(axis=1)

    lh_ok = _hand_present(lh)
    rh_ok = _hand_present(rh)
    hands_ok = lh_ok | rh_ok
    out["hands_present_ratio"] = float(np.mean(hands_ok))

    # head turn rate: abs derivative of nose x over time
    nose_x = pose[:, POSE_NOSE, 0].astype(float)
    nose_x = _interp_1d(nose_x)
    if not np.isfinite(nose_x).all() or N < 3:
        out["head_turn_rate"] = 0.0
    else:
        dt = _safe_dt(t_sec)
        dx = np.diff(nose_x)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.abs(dx) / dt
        rate = rate[np.isfinite(rate)]
        out["head_turn_rate"] = float(np.mean(rate)) if rate.size else 0.0

    # wrist speed (motion energy): mean speed of L/R wrists when pose visible
    lw = pose[:, POSE_L_WRIST, 0:2].astype(float)
    rw = pose[:, POSE_R_WRIST, 0:2].astype(float)

    # interpolate coordinates
    lwx = _interp_1d(lw[:, 0]); lwy = _interp_1d(lw[:, 1])
    rwx = _interp_1d(rw[:, 0]); rwy = _interp_1d(rw[:, 1])

    lw_xy = np.stack([lwx, lwy], axis=1)
    rw_xy = np.stack([rwx, rwy], axis=1)

    lw_speed = _speed_from_xy(t_sec, lw_xy)
    rw_speed = _speed_from_xy(t_sec, rw_xy)

    # combine speeds robustly
    if lw_speed.size and rw_speed.size:
        ws = np.nanmean(np.stack([lw_speed, rw_speed], axis=0), axis=0)
    elif lw_speed.size:
        ws = lw_speed
    elif rw_speed.size:
        ws = rw_speed
    else:
        ws = np.array([], dtype=float)

    ws = ws[np.isfinite(ws)]
    out["motion_energy_hands"] = float(np.mean(ws)) if ws.size else 0.0

    # stillness ratio: fraction of steps with speed below threshold (only when pose ok)
    if ws.size:
        # map pose_ok (N) to steps (N-1)
        pose_ok_steps = pose_ok[1:] if pose_ok.shape[0] > 1 else np.zeros((0,), dtype=bool)
        ws_full = _speed_from_xy(t_sec, np.nanmean(np.stack([lw_xy, rw_xy], axis=0), axis=0))
        if ws_full.size:
            mask = pose_ok_steps & np.isfinite(ws_full)
            if mask.sum() > 0:
                out["stillness_ratio"] = float(np.mean(ws_full[mask] < cfg.still_speed_thr))
            else:
                out["stillness_ratio"] = 1.0
        else:
            out["stillness_ratio"] = 1.0
    else:
        out["stillness_ratio"] = 1.0

    # repetitive motion score: autocorr peak of wrist speed over a lag range
    # estimate effective fps from timestamps
    if N >= 3:
        dt_med = float(np.nanmedian(np.diff(t_sec))) if np.isfinite(np.diff(t_sec)).any() else 1.0 / 15.0
        fps_eff = 1.0 / max(dt_med, 1e-6)
    else:
        fps_eff = 15.0

    # use ws_full to preserve uniform length
    ws_full = _speed_from_xy(t_sec, np.nanmean(np.stack([lw_xy, rw_xy], axis=0), axis=0))
    ws_full = ws_full.astype(float)
    ws_full = _interp_1d(ws_full) if np.isfinite(ws_full).sum() > 0 else np.full_like(ws_full, np.nan)

    if ws_full.size and np.isfinite(ws_full).all():
        out["repetitive_motion_score"] = _autocorr_peak(
            ws_full,
            fps_eff=fps_eff,
            min_lag_sec=cfg.autocorr_min_lag_sec,
            max_lag_sec=cfg.autocorr_max_lag_sec,
        )
    else:
        out["repetitive_motion_score"] = 0.0

    # -------------------------------------------------------------------------
    # Paper-aligned features (autism study)
    # -------------------------------------------------------------------------

    # Interval-based duration: dt[i] = time for interval i→i+1
    # Masks on frames use dt[i] for "frame i contributes dt[i] seconds"
    # Masks on intervals directly use dt[interval_idx]
    dt = np.diff(t_sec.astype(float)) if N >= 2 else np.array([])

    # 1. Lack of eye contact duration (gaze proxy via head orientation)
    # gaze masks are per-frame (N,), we sum dt for frames where gaze_away[:-1] is True
    gaze_away_mask, gaze_valid_mask = _compute_gaze_away_mask(pose, parent_bbox, cfg)

    if dt.size > 0 and gaze_away_mask.size == N:
        # Use frame mask on intervals: interval i is "away" if frame i is away
        lack_of_eye_contact_sec = float(np.sum(dt[gaze_away_mask[:-1]]))
        valid_gaze_duration = float(np.sum(dt[gaze_valid_mask[:-1]]))
    else:
        lack_of_eye_contact_sec = 0.0
        valid_gaze_duration = 0.0

    out["lack_of_eye_contact_duration"] = lack_of_eye_contact_sec

    if valid_gaze_duration > EPS:
        out["lack_of_eye_contact_duration_adj"] = lack_of_eye_contact_sec / valid_gaze_duration
    else:
        out["lack_of_eye_contact_duration_adj"] = float("nan")

    # 2. Non-engaged movement duration (restlessness/roaming via centroid velocity)
    # Returns interval masks (N-1,) directly
    non_engaged_mask, movement_valid_mask = _compute_non_engaged_movement_mask(
        t_sec, child_bbox, pose, cfg
    )

    if dt.size > 0 and non_engaged_mask.size == dt.size:
        non_engaged_sec = float(np.sum(dt[non_engaged_mask]))
        valid_movement_duration = float(np.sum(dt[movement_valid_mask]))
    else:
        non_engaged_sec = 0.0
        valid_movement_duration = 0.0

    out["non_engaged_movement_duration"] = non_engaged_sec

    if valid_movement_duration > EPS:
        out["non_engaged_movement_duration_adj"] = non_engaged_sec / valid_movement_duration
    else:
        out["non_engaged_movement_duration_adj"] = float("nan")

    # 3. Proximity duration (bbox overlap/closeness as contact proxy)
    # NOTE: This is proximity, not true physical contact. See _compute_proximity_mask docstring.
    proximity_mask, proximity_valid_mask = _compute_proximity_mask(
        child_bbox, parent_bbox, parent_pose, cfg
    )

    if dt.size > 0 and proximity_mask.size == N:
        proximity_sec = float(np.sum(dt[proximity_mask[:-1]]))
        valid_proximity_duration = float(np.sum(dt[proximity_valid_mask[:-1]]))
    else:
        proximity_sec = 0.0
        valid_proximity_duration = 0.0

    # Keep feature names as "physical_contact" for backward compatibility,
    # but understand this is actually proximity-based
    out["physical_contact_duration"] = proximity_sec

    if valid_proximity_duration > EPS:
        out["physical_contact_duration_adj"] = proximity_sec / valid_proximity_duration
    else:
        out["physical_contact_duration_adj"] = float("nan")

    return out


# Task prefix mapping
TASK_PREFIX_MAP = {
    "joint_attention": "ja",
    "imitation": "imit",
    "free_play": "fp",
}


def _empty_common_features() -> Dict[str, float]:
    """Returns a dict of common features with NaN values."""
    return {
        "n_frames": float("nan"),
        "duration_sec": float("nan"),
        "face_present_ratio": float("nan"),
        "pose_present_ratio": float("nan"),
        "out_of_view_ratio": float("nan"),
        "head_turn_rate": float("nan"),
        "motion_energy_hands": float("nan"),
        # Paper-aligned features
        "lack_of_eye_contact_duration": float("nan"),
        "lack_of_eye_contact_duration_adj": float("nan"),
        "non_engaged_movement_duration": float("nan"),
        "non_engaged_movement_duration_adj": float("nan"),
        "physical_contact_duration": float("nan"),
        "physical_contact_duration_adj": float("nan"),
        "repetitive_motion_score": float("nan"),
        "stillness_ratio": float("nan"),
        "hands_present_ratio": float("nan"),
    }


def extract_common_features_for_child(
    child_id: str,
    tracks_dir: Path,
    qc_df: pd.DataFrame,
    task_types: Optional[List[str]] = None,
    cfg: Optional[CommonConfig] = None,
) -> Dict[str, float]:
    """
    Extract common features for a child across all task types.

    For each task:
    1. Load tracks from NPZ
    2. Get face_detected_ratio from qc_df
    3. Call compute_common_features_from_tracks(face_present_ratio=face_ratio)
    4. Namespace results with task prefix (ja_, imit_, fp_)

    Args:
        child_id: Child identifier (will be converted to string for matching)
        tracks_dir: Directory containing {child_id}_{task_type}.npz files
        qc_df: DataFrame with columns: child_id, task_type, face_detected_ratio
        task_types: List of task types to process (default: all 3)
        cfg: Optional CommonConfig for feature extraction

    Returns:
        Dict with namespaced features, e.g.:
        {"ja_n_frames": ..., "ja_duration_sec": ..., "imit_n_frames": ..., ...}
    """
    if task_types is None:
        task_types = ["joint_attention", "imitation", "free_play"]

    features: Dict[str, float] = {}
    child_id_str = str(child_id)

    for task in task_types:
        prefix = TASK_PREFIX_MAP.get(task, task[:2])
        npz_path = tracks_dir / f"{child_id_str}_{task}.npz"

        # Get face_detected_ratio from QC for this (child_id, task_type)
        if len(qc_df) > 0 and "child_id" in qc_df.columns:
            qc_row = qc_df[
                (qc_df["child_id"].astype(str) == child_id_str) & (qc_df["task_type"] == task)
            ]
        else:
            qc_row = pd.DataFrame()

        if len(qc_row) > 0:
            face_ratio_val = qc_row.iloc[0].get("face_detected_ratio", np.nan)
            face_ratio = float(face_ratio_val) if pd.notna(face_ratio_val) else None
        else:
            face_ratio = None

        if npz_path.exists():
            # Use extended loader to get bbox data for paper-aligned features
            (
                t_sec,
                pose,
                lh,
                rh,
                child_bbox,
                parent_bbox,
                _is_smoothed,
                _fps,
                _sample_every_n,
            ) = load_tracks_npz_extended(npz_path)

            task_features = compute_common_features_from_tracks(
                t_sec=t_sec,
                pose=pose,
                lh=lh,
                rh=rh,
                cfg=cfg,
                face_present_ratio=face_ratio,
                child_bbox=child_bbox,
                parent_bbox=parent_bbox,
                parent_pose=None,  # Not stored in current format
            )
        else:
            task_features = _empty_common_features()

        # Namespace features by task prefix
        for key, value in task_features.items():
            features[f"{prefix}_{key}"] = value

    return features
