from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import numpy as np


# Pose indices (MediaPipe pose landmarker)
POSE_NOSE = 0
POSE_L_WRIST = 15
POSE_R_WRIST = 16


@dataclass
class CommonConfig:
    # Visibility thresholds (normalized 0..1)
    pose_vis_thr: float = 0.35

    # Stillness threshold in normalized units/sec for wrist speed
    still_speed_thr: float = 0.020

    # Autocorr settings
    autocorr_min_lag_sec: float = 0.20
    autocorr_max_lag_sec: float = 2.00


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    Load tracks with backward compatibility for old/new NPZ formats.

    Returns:
        (t_sec, pose, lh, rh, child_bbox, parent_bbox, is_smoothed)

        - t_sec: (N,) timestamps
        - pose: (N,33,4) child pose landmarks
        - lh: (N,21,4) left hand OR empty
        - rh: (N,21,4) right hand OR empty
        - child_bbox: (N,5) [x0,y0,x1,y1,conf] or None if old format
        - parent_bbox: (N,5) [x0,y0,x1,y1,conf] or None if old format
        - is_smoothed: bool, False if old format or not smoothed
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

    if "child_bbox" in data:
        child_bbox = data["child_bbox"].astype(float)
    if "parent_bbox" in data:
        parent_bbox = data["parent_bbox"].astype(float)
    if "is_smoothed" in data:
        is_smoothed = bool(data["is_smoothed"][0])

    return t, pose, lh, rh, child_bbox, parent_bbox, is_smoothed


def compute_common_features_from_tracks(
    t_sec: np.ndarray,
    pose: np.ndarray,
    lh: np.ndarray,
    rh: np.ndarray,
    cfg: Optional[CommonConfig] = None,
    face_present_ratio: Optional[float] = None,
) -> Dict[str, float]:
    """
    Computes common/global features from tracks arrays.
    If face_present_ratio is None, will emit NaN for face_present_ratio.
    """
    cfg = cfg or CommonConfig()

    N = int(t_sec.shape[0])
    out: Dict[str, float] = {
        "n_frames": float(N),
        "duration_sec": float(t_sec[-1] - t_sec[0]) if N >= 2 else 0.0,
        "face_present_ratio": float(face_present_ratio) if face_present_ratio is not None else float("nan"),
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
        return out

    # pose present per frame: require some landmarks with vis >= threshold
    vis = pose[:, :, 3]
    pose_present = np.nanmean((vis >= cfg.pose_vis_thr).astype(float), axis=1)  # (N,)
    pose_ok = pose_present > 0.35  # at least 35% of landmarks visible
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
        qc_row = qc_df[
            (qc_df["child_id"].astype(str) == child_id_str) & (qc_df["task_type"] == task)
        ]

        if len(qc_row) > 0:
            face_ratio_val = qc_row.iloc[0].get("face_detected_ratio", np.nan)
            face_ratio = float(face_ratio_val) if pd.notna(face_ratio_val) else None
        else:
            face_ratio = None

        if npz_path.exists():
            t_sec, pose, lh, rh = load_tracks_npz(npz_path)
            task_features = compute_common_features_from_tracks(
                t_sec=t_sec,
                pose=pose,
                lh=lh,
                rh=rh,
                cfg=cfg,
                face_present_ratio=face_ratio,
            )
        else:
            task_features = _empty_common_features()

        # Namespace features by task prefix
        for key, value in task_features.items():
            features[f"{prefix}_{key}"] = value

    return features
