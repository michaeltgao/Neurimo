from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.src.perception.mediapipe_runner import MediaPipeRunner


# ============================================================
# Utils
# ============================================================
def nanmed_smooth(x: np.ndarray, win: int = 2) -> np.ndarray:
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


def spectral_peak(
    signal: np.ndarray, fps: float, fmin: float, fmax: float, min_amplitude: float = 0.005
) -> Tuple[float, float]:
    """
    Find dominant frequency in signal within [fmin, fmax] Hz band.
    Returns (frequency, power). Returns (0, 0) if signal is too short,
    has insufficient amplitude, or no valid peak found.
    """
    if signal.size < 30 or not np.isfinite(signal).all():
        return 0.0, 0.0

    # Check signal has meaningful amplitude (avoid detecting noise)
    amplitude = float(np.std(signal))
    if amplitude < min_amplitude:
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


def segments_from_bool(t: np.ndarray, flags: np.ndarray, min_dur: float) -> List[Tuple[float, float]]:
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


def bbox_from_pose(lms: List[Any], min_vis: float = 0.35) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for lm in lms:
        v = float(getattr(lm, "visibility", 0.0) or 0.0)
        if v >= min_vis:
            xs.append(float(lm.x))
            ys.append(float(lm.y))
    if len(xs) < 8:
        return None
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    pad_x, pad_y = 0.06, 0.08
    x0 = max(0.0, x0 - pad_x)
    x1 = min(1.0, x1 + pad_x)
    y0 = max(0.0, y0 - pad_y)
    y1 = min(1.0, y1 + pad_y)
    return (x0, y0, x1, y1)


def in_bbox(x: float, y: float, bb: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bb
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def fallback_child_region(x: float, y: float) -> bool:
    # tighter child center region to avoid adult edge/top hands
    return (0.25 <= x <= 0.75) and (y >= 0.38)


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# ============================================================
# Event row
# ============================================================
@dataclass
class EventSeg:
    child_id: str
    task_type: str
    event_type: str
    t_start: float
    t_end: float
    confidence: float
    meta: str


# ============================================================
# Core per-video processing
# ============================================================
def choose_child_pose(poses: List[List[Any]]) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """
    If multiple poses detected, use position-based classification to identify child vs adult.
    Child is generally centered; adult comes from sides/top/behind.
    Return (child_pose, adult_pose)
    """
    if not poses:
        return None, None
    if len(poses) == 1:
        # Single pose - classify based on position
        classification = classify_pose_as_child_or_adult(poses[0], min_vis=0.25)
        if classification == "child":
            return poses[0], None
        else:
            # If only pose is classified as adult, still treat as child for tracking
            return poses[0], None

    # Multiple poses - classify each
    child_pose = None
    adult_pose = None

    for lms in poses:
        classification = classify_pose_as_child_or_adult(lms, min_vis=0.25)
        if classification == "child" and child_pose is None:
            child_pose = lms
        elif classification == "adult" and adult_pose is None:
            adult_pose = lms

    # Fallback: if no child found, use most centered pose
    if child_pose is None:
        scored: List[Tuple[float, int]] = []
        for i, lms in enumerate(poses):
            bb = bbox_from_pose(lms, min_vis=0.25)
            if bb is None:
                centered = 0.0
            else:
                cx = (bb[0] + bb[2]) / 2.0
                centered = 1.0 - abs(cx - 0.5)
            scored.append((centered, i))
        scored.sort(reverse=True)
        child_pose = poses[scored[0][1]]
        if len(scored) > 1:
            adult_pose = poses[scored[1][1]]

    return child_pose, adult_pose


def wrist_from_hand(hand_lms: List[Any]) -> Tuple[float, float]:
    w = hand_lms[0]
    return float(w.x), float(w.y)


def hand_label_and_score(handedness: Any, i: int) -> Tuple[str, float]:
    if not handedness or i >= len(handedness) or not handedness[i]:
        return "", 0.0
    top = handedness[i][0]
    label = getattr(top, "category_name", None) or getattr(top, "label", None) or ""
    score = float(getattr(top, "score", 0.0) or 0.0)
    return str(label), score


def is_adult_hand_region(wx: float, wy: float) -> bool:
    """
    Check if hand is in a region where adult hands typically appear.
    Adult hands come from sides (left/right edges) or top/above.
    Adult face is never visible - only arms/hands.
    """
    # Left or right edges (adult reaching from side)
    if wx <= 0.20 or wx >= 0.80:
        return True
    # Top region (adult reaching from above/behind)
    if wy <= 0.28:
        return True
    # Upper quadrants (diagonal reach from behind)
    if wy <= 0.40 and (wx <= 0.25 or wx >= 0.75):
        return True
    return False


def per_frame_adult_presence(
    hands_res: Any,
    child_bb: Optional[Tuple[float, float, float, float]],
    other_pose_present: bool,
    adult_hand_min_conf: float = 0.12,
) -> Tuple[bool, str, List[Tuple[float, float]]]:
    """
    Detect adult presence via hands. Strategies:
    1. If > 2 hands detected with good confidence, extras must be adult (child has 2 hands)
    2. Hands in peripheral regions (sides/top) are likely adult
    3. Hands clearly outside child bbox are likely adult

    Returns: (is_present, reason, list of adult hand positions)
    """
    adult_hand_positions: List[Tuple[float, float]] = []

    if hands_res is None or not getattr(hands_res, "hand_landmarks", None):
        return other_pose_present, "pose2" if other_pose_present else "", adult_hand_positions

    hands = hands_res.hand_landmarks
    handedness = getattr(hands_res, "handedness", None)

    # Gather all valid hands with positions
    valid_hands: List[Tuple[float, float, float]] = []  # (wx, wy, score)
    for i, lms in enumerate(hands):
        _, s = hand_label_and_score(handedness, i)
        if s > 0 and s < adult_hand_min_conf:
            continue
        wx, wy = wrist_from_hand(lms)
        valid_hands.append((wx, wy, s if s > 0 else 0.5))

    # Strategy 1: If more than 2 hands, extras are adult hands
    # Sort by how "child-like" the position is (centered + lower = more child-like)
    if len(valid_hands) > 2:
        def child_score(h: Tuple[float, float, float]) -> float:
            wx, wy = h[0], h[1]
            # Higher score = more likely child (centered x, lower y)
            x_center = 1.0 - abs(wx - 0.5) * 2  # 1.0 at center, 0.0 at edges
            y_lower = wy  # Higher y = lower in frame = more likely child
            return x_center + y_lower

        sorted_hands = sorted(valid_hands, key=child_score, reverse=True)
        # First 2 are child, rest are adult
        for wx, wy, _ in sorted_hands[2:]:
            adult_hand_positions.append((wx, wy))

    # Strategy 2: Check remaining hands for peripheral positions
    for wx, wy, _ in valid_hands:
        if (wx, wy) in adult_hand_positions:
            continue

        # Flag as adult if in peripheral region
        if is_adult_hand_region(wx, wy):
            adult_hand_positions.append((wx, wy))
        # Also flag if outside child bbox AND in upper portion of frame
        elif child_bb is not None:
            if not in_bbox(wx, wy, child_bb) and wy < 0.45:
                adult_hand_positions.append((wx, wy))

    if adult_hand_positions:
        reason = "extra_hands" if len(valid_hands) > 2 else "adult_hand_peripheral"
        return True, reason, adult_hand_positions

    if other_pose_present:
        return True, "pose2", adult_hand_positions

    return False, "", adult_hand_positions


def detect_adult_hand_motion(
    adult_hand_xy: List[Tuple[float, float]],
    t: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute adult hand motion/activity level.
    Returns (speed_array, mean_activity).
    """
    n = len(adult_hand_xy)
    if n < 2:
        return np.zeros(max(1, n), dtype=float), 0.0

    xy = np.array(adult_hand_xy, dtype=float)
    # Handle NaN entries
    valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])

    if valid.sum() < 2:
        return np.zeros(n, dtype=float), 0.0

    # Interpolate gaps
    for col in range(2):
        if (~valid).any() and valid.any():
            xy[~valid, col] = np.interp(
                np.where(~valid)[0],
                np.where(valid)[0],
                xy[valid, col]
            )

    # Compute speed
    dt = np.diff(t[:n])
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.sqrt(dx * dx + dy * dy) / np.maximum(dt, 1e-6)

    # Pad to match length
    speed = np.concatenate([[speed[0] if speed.size > 0 else 0.0], speed])
    mean_activity = float(np.nanmean(speed)) if np.isfinite(speed).any() else 0.0

    return speed.astype(float), mean_activity


def child_hand_to_face_flag(pose_xy: np.ndarray, lh_xy: Optional[np.ndarray], rh_xy: Optional[np.ndarray], thr: float) -> bool:
    """
    pose_xy: (33,2) child pose
    hand: (21,2) or None
    We detect if wrist (or index tip) is close to face landmarks.
    """
    if pose_xy.shape != (33, 2):
        return False

    # face-ish pose indices (pose model):
    # 0: nose, 1/2: eyes, 9/10: mouth corners (approx), 7/8: ears (varies)
    face_idx = [0, 1, 2, 7, 8, 9, 10]
    face_pts = pose_xy[face_idx, :]
    if not np.isfinite(face_pts).any():
        return False

    def min_dist(hand_xy: np.ndarray) -> float:
        # use wrist (0) + index tip (8) if present
        pts_list = []
        if hand_xy.shape[0] >= 9:
            pts_list.append(hand_xy[0])
            pts_list.append(hand_xy[8])
        else:
            pts_list.append(hand_xy[0])
        pts = np.array(pts_list, dtype=float)
        # distance to any face point
        d = np.sqrt(((pts[:, None, :] - face_pts[None, :, :]) ** 2).sum(axis=2))
        return float(np.nanmin(d))

    best = math.inf
    if lh_xy is not None and np.isfinite(lh_xy).any():
        best = min(best, min_dist(lh_xy))
    if rh_xy is not None and np.isfinite(rh_xy).any():
        best = min(best, min_dist(rh_xy))

    return bool(best < thr)


def compute_motion_energy(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """
    xy: (N,2) normalized coordinates for a joint. Returns per-step speed.
    """
    if t.size < 2:
        return np.zeros((0,), dtype=float)
    dt = np.diff(t)
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.sqrt(dx * dx + dy * dy) / np.maximum(dt, 1e-6)
    return speed.astype(float)


def classify_pose_as_child_or_adult(lms: List[Any], min_vis: float = 0.30) -> Optional[str]:
    """
    Classify a detected pose as 'child' or 'adult' based on position.
    Child is generally centered; adult comes from sides/top/behind.
    """
    xs, ys = [], []
    edge_count = 0

    for lm in lms:
        v = float(getattr(lm, "visibility", 0.0) or 0.0)
        if v >= min_vis:
            x, y = float(lm.x), float(lm.y)
            xs.append(x)
            ys.append(y)
            if x <= 0.05 or x >= 0.95 or y <= 0.05:
                edge_count += 1

    if len(xs) < 3:
        return None
    if len(xs) < 5 and edge_count == 0:
        return None

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    is_centered_x = 0.25 <= cx <= 0.75
    is_lower_half = cy >= 0.35
    is_peripheral_x = cx <= 0.25 or cx >= 0.75
    is_upper = cy <= 0.35
    is_partial_offscreen = edge_count >= 2

    if is_centered_x and is_lower_half and not is_partial_offscreen:
        return "child"
    elif is_peripheral_x or is_upper or is_partial_offscreen:
        return "adult"
    else:
        return "child" if is_centered_x else "adult"


def detect_object_focus(
    pose_xy: np.ndarray,
    pose_ok: np.ndarray,
    lh: np.ndarray,
    rh: np.ndarray,
    gaze_down_thr: float = 0.08,
) -> np.ndarray:
    """
    Detect when child is focused on an object (looking down, hands in lap/front area).
    Returns per-frame boolean flags.

    Heuristics:
    - Nose is below or near shoulder level (looking down)
    - Hands are in front/lap area (low y, centered x)
    - Low head movement (sustained attention)
    """
    n = pose_xy.shape[0]
    flags = np.zeros(n, dtype=bool)

    for i in range(n):
        if not pose_ok[i]:
            continue

        nose = pose_xy[i, 0, 0:2]
        lsho = pose_xy[i, 11, 0:2]
        rsho = pose_xy[i, 12, 0:2]

        if not (np.isfinite(nose).all() and np.isfinite(lsho).all() and np.isfinite(rsho).all()):
            continue

        # Shoulder midpoint
        sho_mid_y = (lsho[1] + rsho[1]) / 2.0

        # Looking down: nose y is close to or below shoulder y
        # (In normalized coords, larger y = lower on screen)
        looking_down = nose[1] >= sho_mid_y - gaze_down_thr

        # Check if hands are in object manipulation area (front/lap)
        hands_in_lap = False
        for h in [lh[i], rh[i]]:
            if np.isfinite(h).any():
                hx, hy = h[0, 0], h[0, 1]  # wrist
                # Lap/front area: centered x, lower y (hands below shoulders)
                if 0.25 <= hx <= 0.75 and hy >= sho_mid_y - 0.05:
                    hands_in_lap = True
                    break

        flags[i] = looking_down and hands_in_lap

    return flags


def detect_windowed_periodicity(
    signal: np.ndarray,
    fps: float,
    window_sec: float,
    freq_min: float,
    freq_max: float,
    power_thr: float,
) -> np.ndarray:
    """
    Detect periodicity in a sliding window. Returns per-frame flags.
    More sensitive than global periodicity for detecting bouts.
    """
    n = signal.size
    win_frames = max(int(window_sec * fps), 30)
    flags = np.zeros(n, dtype=bool)

    if n < win_frames:
        return flags

    for i in range(n - win_frames + 1):
        chunk = signal[i:i + win_frames]
        if not np.isfinite(chunk).all():
            continue

        freq, power = spectral_peak(chunk, fps, freq_min, freq_max)
        if power > power_thr and freq > 0:
            # Mark the window center
            center = i + win_frames // 2
            flags[max(0, center - win_frames // 4):min(n, center + win_frames // 4)] = True

    return flags


def process_free_play_video(
    child_id: str,
    video_path: Path,
    runner: MediaPipeRunner,
    sample_every_n: int,
    max_frames: int,
    # thresholds
    min_pose_vis: float,
    min_seg_dur: float,
    freeze_speed_thr: float,
    freeze_min_dur: float,
    hand_face_thr: float,
    flap_freq_min: float,
    flap_freq_max: float,
    flap_power_thr: float,
    rocking_freq_min: float,
    rocking_freq_max: float,
    engaged_head_turn_thr: float,
    engaged_min_dur: float,
) -> Tuple[List[EventSeg], Dict[str, Any]]:
    # Get video info for fps calculation
    try:
        video_info = runner.get_video_info(str(video_path))
        fps = video_info["fps"]
    except RuntimeError:
        return [], {}

    fps_eff = fps / max(1, sample_every_n)

    t_list: List[float] = []
    pose_list: List[np.ndarray] = []   # (33,4) x,y,z,vis
    lh_list: List[np.ndarray] = []     # (21,2)
    rh_list: List[np.ndarray] = []     # (21,2)

    pose_present: List[bool] = []
    adult_present: List[bool] = []
    adult_side: List[int] = []  # -1=left, +1=right, 0=unknown
    adult_meta: List[str] = []
    adult_hand_xy_list: List[Tuple[float, float]] = []  # Track adult hand positions

    child_bb_list: List[Optional[Tuple[float, float, float, float]]] = []

    # Iterate frames using the runner
    for frame_result in runner.iter_video(str(video_path), sample_every_n=sample_every_n, max_frames=max_frames):
        frame_time = frame_result.t_sec

        # Pose
        pres = frame_result.pose
        poses = pres.pose_landmarks if (pres and getattr(pres, "pose_landmarks", None)) else []
        child_pose_raw, other_pose_raw = choose_child_pose(poses)

        if child_pose_raw is None:
            pose_arr = np.full((33, 4), np.nan, dtype=np.float32)
            bb = None
            frame_pose_ok = False
        else:
            pose_arr = np.full((33, 4), np.nan, dtype=np.float32)
            for i in range(min(33, len(child_pose_raw))):
                lm = child_pose_raw[i]
                pose_arr[i, 0] = float(lm.x)
                pose_arr[i, 1] = float(lm.y)
                pose_arr[i, 2] = float(lm.z)
                pose_arr[i, 3] = float(getattr(lm, "visibility", 0.0) or 0.0)
            bb = bbox_from_pose(child_pose_raw, min_vis=min_pose_vis)
            frame_pose_ok = True

        other_pose_present = other_pose_raw is not None

        # Hands
        hres = frame_result.hands

        # adult presence (focus on hands from periphery - adult face never visible)
        ad_ok, ad_reason, ad_hands = per_frame_adult_presence(hres, bb, other_pose_present)
        adult_present.append(bool(ad_ok))
        adult_meta.append(ad_reason)

        # Track best adult hand position (use first detected, or NaN if none)
        if ad_hands:
            adult_hand_xy_list.append(ad_hands[0])  # Use first adult hand
        else:
            adult_hand_xy_list.append((math.nan, math.nan))

        # estimate adult side (left/right) from adult hand position
        side = 0
        if ad_ok and ad_hands:
            # Use the first detected adult hand position
            best_wx = ad_hands[0][0]
            if np.isfinite(best_wx):
                side = -1 if best_wx < 0.5 else +1
        adult_side.append(int(side))

        # child-only hands for hand-to-face / flap
        frame_lh_xy = np.full((21, 2), np.nan, dtype=np.float32)
        frame_rh_xy = np.full((21, 2), np.nan, dtype=np.float32)

        if hres is not None and getattr(hres, "hand_landmarks", None):
            hands = hres.hand_landmarks
            handedness = getattr(hres, "handedness", None)

            # gather hands that are child (wrist in bbox or in fallback)
            child_candidates: List[Tuple[str, float, Any]] = []
            for i, lms in enumerate(hands):
                label, score = hand_label_and_score(handedness, i)
                wrist_x, wrist_y = wrist_from_hand(lms)
                is_child = (in_bbox(wrist_x, wrist_y, bb) if bb is not None else fallback_child_region(wrist_x, wrist_y))
                if is_child:
                    rank = score if score > 0 else 0.2
                    child_candidates.append((label, rank, lms))

            # choose top 2
            child_candidates.sort(key=lambda x: x[1], reverse=True)
            chosen = child_candidates[:2]

            for label, _rank, lms in chosen:
                lab = (label or "").lower()
                arr = np.full((21, 2), np.nan, dtype=np.float32)
                for k in range(min(21, len(lms))):
                    arr[k, 0] = float(lms[k].x)
                    arr[k, 1] = float(lms[k].y)
                if "left" in lab and np.isnan(frame_lh_xy[0, 0]):
                    frame_lh_xy = arr
                elif "right" in lab and np.isnan(frame_rh_xy[0, 0]):
                    frame_rh_xy = arr
                else:
                    # slot into empty
                    if np.isnan(frame_lh_xy[0, 0]):
                        frame_lh_xy = arr
                    elif np.isnan(frame_rh_xy[0, 0]):
                        frame_rh_xy = arr

        t_list.append(float(frame_time))
        pose_list.append(pose_arr)
        lh_list.append(frame_lh_xy)
        rh_list.append(frame_rh_xy)
        pose_present.append(bool(frame_pose_ok))
        child_bb_list.append(bb)

    # Need minimum frames for reliable analysis
    if len(t_list) < 15:
        return [], {"child_id": child_id, "task_type": "free_play", "error": "insufficient_frames"}

    # Need minimum pose tracking coverage
    pose_coverage = sum(pose_present) / len(pose_present) if pose_present else 0.0
    if pose_coverage < 0.3:
        return [], {"child_id": child_id, "task_type": "free_play", "error": "poor_pose_tracking"}

    t = np.array(t_list, dtype=float)
    pose = np.stack(pose_list, axis=0).astype(float)          # (N,33,4)
    lh = np.stack(lh_list, axis=0).astype(float)              # (N,21,2)
    rh = np.stack(rh_list, axis=0).astype(float)              # (N,21,2)

    pose_ok = np.array(pose_present, dtype=bool)
    adult_ok_raw = np.array(adult_present, dtype=bool)
    adult_side_arr = np.array(adult_side, dtype=int)

    # Temporal smoothing for adult detection - fill small gaps (flickering)
    # If adult detected in neighboring frames, likely present in between
    adult_ok = adult_ok_raw.copy()
    for i in range(1, len(adult_ok) - 1):
        if not adult_ok[i] and adult_ok_raw[i - 1] and adult_ok_raw[i + 1]:
            adult_ok[i] = True
    # Also extend by 1 frame on each side of detected segments (handles edge flicker)
    adult_ok_extended = adult_ok.copy()
    for i in range(1, len(adult_ok)):
        if adult_ok[i] and not adult_ok[i - 1]:
            adult_ok_extended[i - 1] = True
    for i in range(len(adult_ok) - 1):
        if adult_ok[i] and not adult_ok[i + 1]:
            adult_ok_extended[i + 1] = True
    adult_ok = adult_ok_extended

    # -------------------------
    # Base signals
    # -------------------------
    # wrists (child): pose indices 15/16
    lw = pose[:, 15, 0:2]
    rw = pose[:, 16, 0:2]
    nose = pose[:, 0, 0:2]
    lhip = pose[:, 23, 0:2]
    rhip = pose[:, 24, 0:2]
    lsho = pose[:, 11, 0:2]
    rsho = pose[:, 12, 0:2]

    # compute speeds (use mean of wrists when available)
    wrist_xy = np.nanmean(np.stack([lw, rw], axis=1), axis=1)  # (N,2)
    wrist_xy = interp_nans(wrist_xy[:, 0])[:, None].repeat(2, axis=1) if False else wrist_xy  # no-op placeholder

    # speed from each wrist then combine
    lw_speed = compute_motion_energy(t, interp_nans(lw[:, 0])[:, None].repeat(2, axis=1) if False else lw)
    rw_speed = compute_motion_energy(t, interp_nans(rw[:, 0])[:, None].repeat(2, axis=1) if False else rw)
    # If joints are NaN often, speeds contain NaNs; handle:
    lw_speed = interp_nans(lw_speed) if np.isfinite(lw_speed).sum() > 0 else lw_speed
    rw_speed = interp_nans(rw_speed) if np.isfinite(rw_speed).sum() > 0 else rw_speed
    wrist_speed = np.nanmean(np.stack([lw_speed, rw_speed], axis=0), axis=0)  # (N-1,)

    wrist_speed_s = nanmed_smooth(wrist_speed, win=2)

    # -------------------------
    # FREEZE (disengagement proxy)
    # -------------------------
    freeze_flags = np.zeros((t.size,), dtype=bool)
    # map speed (N-1) to frames (N) by padding
    sp = np.concatenate([[wrist_speed_s[0] if wrist_speed_s.size else 0.0], wrist_speed_s])
    freeze_flags = pose_ok & np.isfinite(sp) & (sp < freeze_speed_thr)
    freeze_segs = segments_from_bool(t, freeze_flags, min_dur=freeze_min_dur)

    # -------------------------
    # HAND_TO_FACE
    # -------------------------
    hand_face_flags = np.zeros((t.size,), dtype=bool)
    for i in range(t.size):
        if not pose_ok[i]:
            continue
        pose_xy = pose[i, :, 0:2]
        lh_xy = lh[i] if np.isfinite(lh[i]).any() else None
        rh_xy = rh[i] if np.isfinite(rh[i]).any() else None
        hand_face_flags[i] = child_hand_to_face_flag(pose_xy, lh_xy, rh_xy, thr=hand_face_thr)
    hand_face_segs = segments_from_bool(t, hand_face_flags, min_dur=max(0.20, min_seg_dur))

    # -------------------------
    # REPETITIVE_MOTION / STEREOTYPY (comprehensive detection)
    # Includes: arm flapping, tapping, side-to-side movements, etc.
    # Looks for periodicity in multiple signals and takes the strongest
    # -------------------------
    min_rep_amplitude = 0.006  # ~0.6% of frame dimension (lowered for sensitivity)

    # Signal 1: Vertical wrist position (arm flapping - up/down)
    wy = np.nanmean(np.stack([lw[:, 1], rw[:, 1]], axis=1), axis=1)
    wy = interp_nans(wy)
    wy = nanmed_smooth(wy, win=2)
    wy_amplitude = float(np.std(wy)) if np.isfinite(wy).any() else 0.0
    f_wy, p_wy = spectral_peak(wy, fps_eff, flap_freq_min, flap_freq_max, min_amplitude=0.003)

    # Signal 2: Horizontal wrist position (side-to-side movements)
    wx = np.nanmean(np.stack([lw[:, 0], rw[:, 0]], axis=1), axis=1)
    wx = interp_nans(wx)
    wx = nanmed_smooth(wx, win=2)
    wx_amplitude = float(np.std(wx)) if np.isfinite(wx).any() else 0.0
    f_wx, p_wx = spectral_peak(wx, fps_eff, flap_freq_min, flap_freq_max, min_amplitude=0.003)

    # Signal 3: Wrist speed (tapping, drumming, general repetitive motion)
    # Pad speed to match frame count
    sp_padded = np.concatenate([[wrist_speed_s[0] if wrist_speed_s.size else 0.0], wrist_speed_s])
    sp_smooth = nanmed_smooth(sp_padded, win=2)
    sp_amplitude = float(np.std(sp_smooth)) if np.isfinite(sp_smooth).any() else 0.0
    f_sp, p_sp = spectral_peak(sp_smooth, fps_eff, flap_freq_min, flap_freq_max, min_amplitude=0.002)

    # Choose best signal (highest power with sufficient amplitude)
    candidates = []
    if wy_amplitude > min_rep_amplitude * 0.5:
        candidates.append(("vertical", f_wy, p_wy, wy_amplitude, wy))
    if wx_amplitude > min_rep_amplitude * 0.5:
        candidates.append(("horizontal", f_wx, p_wx, wx_amplitude, wx))
    if sp_amplitude > min_rep_amplitude * 0.3:  # Speed has different scale
        candidates.append(("speed", f_sp, p_sp, sp_amplitude, sp_smooth))

    # Select best candidate by power
    if candidates:
        best = max(candidates, key=lambda x: x[2])
        rep_type, f_flap, p_flap, best_amplitude, best_signal = best
    else:
        rep_type, f_flap, p_flap, best_amplitude, best_signal = "none", 0.0, 0.0, 0.0, wy

    # For backward compatibility, keep wy_amplitude for windowed detection
    wy_amplitude = max(wy_amplitude, wx_amplitude, sp_amplitude * 0.5)

    # NOTE: flap_segs creation moved below after all candidates (rocking, tapping) are gathered

    # -------------------------
    # ROCKING detection (now part of REPETITIVE_MOTION)
    # Torso lateral oscillation - add to repetitive motion candidates
    # -------------------------
    # use midpoint of hips x
    hipx = np.nanmean(np.stack([lhip[:, 0], rhip[:, 0]], axis=1), axis=1)
    hipx = interp_nans(hipx)
    hipx = nanmed_smooth(hipx, win=2)

    # Check for sufficient lateral movement amplitude
    hipx_amplitude = float(np.std(hipx)) if np.isfinite(hipx).any() else 0.0
    min_rock_amplitude = 0.012  # ~1.2% of frame width

    f_rock, p_rock = spectral_peak(hipx, fps_eff, rocking_freq_min, rocking_freq_max)

    # Also check shoulder midpoint for upper body rocking
    shox = np.nanmean(np.stack([lsho[:, 0], rsho[:, 0]], axis=1), axis=1)
    shox = interp_nans(shox)
    shox = nanmed_smooth(shox, win=2)
    shox_amplitude = float(np.std(shox)) if np.isfinite(shox).any() else 0.0

    f_rock_sho, p_rock_sho = spectral_peak(shox, fps_eff, rocking_freq_min, rocking_freq_max)

    # Add rocking to repetitive motion candidates
    if hipx_amplitude > min_rock_amplitude:
        candidates.append(("rocking_hip", f_rock, p_rock, hipx_amplitude, hipx))
    if shox_amplitude > min_rock_amplitude:
        candidates.append(("rocking_shoulder", f_rock_sho, p_rock_sho, shox_amplitude, shox))

    # -------------------------
    # TAPPING detection (fingertip movements) - add to REPETITIVE_MOTION
    # Analyze individual finger movements from hand landmarks
    # -------------------------
    # Fingertip indices in hand landmark model: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    fingertip_indices = [4, 8, 12, 16, 20]

    # Compute fingertip vertical position relative to wrist for each hand
    for hand_arr, hand_name in [(lh, "left"), (rh, "right")]:
        # Check if hand is tracked well enough
        valid_frames = int(np.isfinite(hand_arr[:, 0, 0]).sum())
        if valid_frames < 15:
            continue

        # Get wrist y position for this hand
        hand_wrist_y: np.ndarray = hand_arr[:, 0, 1]  # Landmark 0 is wrist

        # Compute mean fingertip y relative to wrist (detects tapping motion)
        fingertip_y: np.ndarray = np.zeros(hand_arr.shape[0], dtype=float)
        for fi in fingertip_indices:
            fingertip_y = fingertip_y + hand_arr[:, fi, 1]
        fingertip_y = fingertip_y / len(fingertip_indices)

        # Relative movement (fingertips to wrist)
        finger_rel_y = fingertip_y - hand_wrist_y
        finger_rel_y = interp_nans(finger_rel_y)
        finger_rel_y = nanmed_smooth(finger_rel_y, win=2)

        finger_amplitude = float(np.std(finger_rel_y)) if np.isfinite(finger_rel_y).any() else 0.0

        # Tapping has smaller movements - use lower threshold
        min_tap_amplitude = 0.003  # ~0.3% of frame dimension
        if finger_amplitude > min_tap_amplitude:
            # Pass lower min_amplitude to spectral_peak for tapping
            f_tap, p_tap = spectral_peak(finger_rel_y, fps_eff, flap_freq_min, flap_freq_max * 1.5, min_amplitude=0.002)
            if f_tap > 0:
                candidates.append((f"tapping_{hand_name}", f_tap, p_tap, finger_amplitude, finger_rel_y))

        # Also check raw fingertip speed (rapid movements)
        finger_speed = compute_motion_energy(t, np.stack([fingertip_y, np.zeros_like(fingertip_y)], axis=1))
        if finger_speed.size > 0:
            finger_speed = interp_nans(finger_speed)
            finger_speed_padded = np.concatenate([[finger_speed[0]], finger_speed])
            finger_speed_padded = nanmed_smooth(finger_speed_padded, win=2)
            speed_amplitude = float(np.std(finger_speed_padded)) if np.isfinite(finger_speed_padded).any() else 0.0
            if speed_amplitude > min_tap_amplitude * 0.5:
                f_tap_sp, p_tap_sp = spectral_peak(finger_speed_padded, fps_eff, flap_freq_min, flap_freq_max * 1.5, min_amplitude=0.001)
                if f_tap_sp > 0:
                    candidates.append((f"tap_speed_{hand_name}", f_tap_sp, p_tap_sp, speed_amplitude, finger_speed_padded))

    # Re-select best candidate after adding rocking and tapping
    if candidates:
        best = max(candidates, key=lambda x: x[2])
        rep_type, f_flap, p_flap, best_amplitude, best_signal = best
    else:
        rep_type, f_flap, p_flap, best_amplitude, best_signal = "none", 0.0, 0.0, 0.0, wy

    # Update flap_global based on new best candidate (lowered thresholds for sensitivity)
    # For tapping, periodicity may be weaker but still present
    flap_global = (p_flap > flap_power_thr * 0.3) and (f_flap > 0) and (best_amplitude > min_rep_amplitude * 0.2)

    # Create repetitive motion segments based on all candidates
    flap_flags = pose_ok & flap_global
    flap_segs = segments_from_bool(t, flap_flags, min_dur=max(0.30, min_seg_dur))

    # -------------------------
    # ADULT_PRESENT segments
    # -------------------------
    adult_segs = segments_from_bool(t, adult_ok, min_dur=max(0.20, min_seg_dur))

    # -------------------------
    # ADULT_HAND_ACTIVITY (track adult hand motion when present)
    # Adult hands come from sides/top - track their movement
    # -------------------------
    adult_hand_speed, adult_hand_mean_activity = detect_adult_hand_motion(
        adult_hand_xy_list, t
    )
    # Flag frames where adult hand is actively moving (engaging with child)
    adult_active_thr = 0.03  # movement threshold
    adult_hand_active = adult_ok & (adult_hand_speed > adult_active_thr)
    adult_active_segs = segments_from_bool(t, adult_hand_active, min_dur=max(0.20, min_seg_dur))

    # -------------------------
    # ENGAGED_WITH_ADULT (heuristic)
    # adult present + child orients toward adult side + child active (not freeze)
    # orientation proxy: nose x relative to shoulders midpoint
    # if adult on left, nose shifts left (smaller x), vice versa.
    # -------------------------
    shx = np.nanmean(np.stack([lsho[:, 0], rsho[:, 0]], axis=1), axis=1)
    shx = interp_nans(shx)
    nx = interp_nans(nose[:, 0])
    head_dx = nx - shx  # >0 means face slightly to right (very rough)

    # adult side: -1 left, +1 right
    # engaged if sign matches and magnitude above threshold
    # Also consider if adult hand is actively moving (engaging with child)
    engaged_flags = np.zeros((t.size,), dtype=bool)
    for i in range(t.size):
        if not pose_ok[i]:
            continue
        if not adult_ok[i]:
            continue
        if freeze_flags[i]:
            continue

        # Check if child is oriented toward adult
        oriented_to_adult = False
        if adult_side_arr[i] != 0:
            want = float(adult_side_arr[i]) * float(head_dx[i])
            oriented_to_adult = want > engaged_head_turn_thr

        # Check if adult hand is actively moving (engaging)
        adult_actively_engaging = adult_hand_active[i] if i < len(adult_hand_active) else False

        # Engaged if child oriented toward active adult, OR adult is very active
        engaged_flags[i] = bool(oriented_to_adult or (adult_actively_engaging and not freeze_flags[i]))

    engaged_segs = segments_from_bool(t, engaged_flags, min_dur=engaged_min_dur)

    # -------------------------
    # DISENGAGED_WITH_ADULT (adult present but child not oriented toward them)
    # -------------------------
    disengaged_flags = np.zeros((t.size,), dtype=bool)
    for i in range(t.size):
        if not pose_ok[i]:
            continue
        if not adult_ok[i]:
            continue
        # Adult present but child not engaged (not looking toward adult, or frozen)
        if not engaged_flags[i]:
            disengaged_flags[i] = True
    disengaged_segs = segments_from_bool(t, disengaged_flags, min_dur=engaged_min_dur)

    # -------------------------
    # OBJECT_FOCUS (repetitive attention on object - looking down, hands in lap)
    # -------------------------
    object_focus_flags = detect_object_focus(
        pose_xy=pose[:, :, 0:2],
        pose_ok=pose_ok,
        lh=lh,
        rh=rh,
        gaze_down_thr=0.08,
    )
    # Object focus is more meaningful when sustained and not just a brief glance
    object_focus_segs = segments_from_bool(t, object_focus_flags, min_dur=max(0.5, min_seg_dur))

    # -------------------------
    # Windowed periodicity detection for short bouts
    # Use the best signal from candidates for windowed detection
    # -------------------------
    if best_amplitude > min_rep_amplitude * 0.2:
        flap_windowed_flags = detect_windowed_periodicity(
            signal=best_signal,
            fps=fps_eff,
            window_sec=1.5,
            freq_min=flap_freq_min,
            freq_max=flap_freq_max * 1.5,  # Extended range for tapping
            power_thr=flap_power_thr * 0.25,  # Lower for windowed detection
        )
        flap_combined_flags = pose_ok & (flap_flags | flap_windowed_flags)
        flap_combined_segs = segments_from_bool(t, flap_combined_flags, min_dur=max(0.25, min_seg_dur))
        if len(flap_combined_segs) >= len(flap_segs):
            flap_segs = flap_combined_segs
            # Update rep_type if we found more segments via windowed
            if not flap_global and len(flap_combined_segs) > 0:
                flap_global = True  # For confidence calculation

    # -------------------------
    # Build event list + summary
    # -------------------------
    events: List[EventSeg] = []

    def add_segs(event_type: str, segs: List[Tuple[float, float]], conf: float, meta: str) -> None:
        for s, e in segs:
            events.append(EventSeg(child_id, "free_play", event_type, float(s), float(e), float(conf), meta))

    add_segs("FREEZE", freeze_segs, conf=0.75, meta=f"thr_speed<{freeze_speed_thr}")
    add_segs("HAND_TO_FACE", hand_face_segs, conf=0.80, meta=f"thr<{hand_face_thr}")
    # Repetitive motion confidence based on periodicity strength (includes flapping, tapping, rocking, etc.)
    rep_conf = min(0.85, 0.55 + 0.3 * min(1.0, p_flap / (flap_power_thr * 2))) if flap_global else 0.50
    add_segs("REPETITIVE_MOTION", flap_segs, conf=rep_conf, meta=f"type={rep_type},f={f_flap:.2f},p={p_flap:.1f}")
    add_segs("ADULT_PRESENT", adult_segs, conf=0.85, meta="hand_from_peripheral")
    add_segs("ADULT_HAND_ACTIVE", adult_active_segs, conf=0.70, meta=f"motion>{adult_active_thr}")
    add_segs("ENGAGED_WITH_ADULT", engaged_segs, conf=0.60, meta=f"head_thr>{engaged_head_turn_thr}")
    add_segs("DISENGAGED_WITH_ADULT", disengaged_segs, conf=0.60, meta="adult_present_not_engaged")
    add_segs("OBJECT_FOCUS", object_focus_segs, conf=0.65, meta="looking_down_hands_in_lap")

    dur = float(t[-1] - t[0]) if t.size > 1 else 0.0
    dur = max(dur, 1e-6)

    def seg_total(segs: List[Tuple[float, float]]) -> float:
        return float(sum(max(0.0, e - s) for s, e in segs))

    # Compute time fractions first so we can gate frequency features
    flap_time_frac = seg_total(flap_segs) / dur

    # Gate frequency features: only meaningful when corresponding time fraction > 0
    # Otherwise these are just noise from spectral analysis
    gated_flap_freq = f_flap if flap_time_frac > 0 else 0.0

    summary = {
        "child_id": child_id,
        "task_type": "free_play",
        "duration_sec": round(dur, 3),
        "pose_present_ratio": round(float(pose_ok.mean()), 4),
        "adult_present_ratio": round(float(adult_ok.mean()), 4),
        "adult_hand_active_time_frac": round(seg_total(adult_active_segs) / dur, 4),
        "adult_hand_mean_activity": round(float(adult_hand_mean_activity), 4),
        "freeze_time_frac": round(seg_total(freeze_segs) / dur, 4),
        "hand_to_face_time_frac": round(seg_total(hand_face_segs) / dur, 4),
        # Repetitive motion: includes flapping, tapping, rocking, side-to-side, etc.
        "repetitive_motion_time_frac": round(flap_time_frac, 4),
        "engaged_with_adult_time_frac": round(seg_total(engaged_segs) / dur, 4),
        "disengaged_with_adult_time_frac": round(seg_total(disengaged_segs) / dur, 4),
        # NOTE: This is a proxy (looking down + hands in lap), not true object detection
        "hands_near_torso_time_frac": round(seg_total(object_focus_segs) / dur, 4),
        "repetitive_motion_freq_hz": round(float(gated_flap_freq), 3),
    }

    return events, summary


# ============================================================
# CLI
# ============================================================
TASK_TO_COL = {"free_play": "free_play_path"}


def main():
    ap = argparse.ArgumentParser(description="Detect discrete free-play events (repetitive motion/stereotypy, hand-to-face, freeze, adult presence/engagement).")
    ap.add_argument("--manifest", required=True, help="CSV with child_id and free_play_path")
    ap.add_argument("--out_events", default="data/derived/free_play_events.csv")
    ap.add_argument("--out_summary", default="data/derived/free_play_summary.csv")
    ap.add_argument("--sample_every_n", type=int, default=2)
    ap.add_argument("--max_frames", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)

    # thresholds
    ap.add_argument("--min_pose_vis", type=float, default=0.35)
    ap.add_argument("--min_seg_dur", type=float, default=0.25)

    ap.add_argument("--freeze_speed_thr", type=float, default=0.020, help="Lower => stricter freeze. Normalized units/sec.")
    ap.add_argument("--freeze_min_dur", type=float, default=0.60)

    ap.add_argument("--hand_face_thr", type=float, default=0.085, help="Normalized distance threshold for wrist-to-face.")

    ap.add_argument("--flap_freq_min", type=float, default=1.5)
    ap.add_argument("--flap_freq_max", type=float, default=5.5)
    ap.add_argument("--flap_power_thr", type=float, default=12.0)

    ap.add_argument("--rocking_freq_min", type=float, default=0.4)
    ap.add_argument("--rocking_freq_max", type=float, default=1.8)

    ap.add_argument("--engaged_head_turn_thr", type=float, default=0.010)
    ap.add_argument("--engaged_min_dur", type=float, default=0.35)

    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if "free_play_path" not in df.columns:
        raise ValueError(f"Manifest missing 'free_play_path'. Columns: {list(df.columns)}")

    # Create runner with pose and hands enabled
    runner = MediaPipeRunner(
        enable_pose=True,
        enable_hands=True,
        enable_face=False,
        num_poses=2,   # allow adult pose too
        num_hands=8,   # allow multiple (adult+child); we'll filter
    )

    out_e = Path(args.out_events)
    out_s = Path(args.out_summary)
    out_e.parent.mkdir(parents=True, exist_ok=True)
    out_s.parent.mkdir(parents=True, exist_ok=True)

    events_fields = ["child_id", "task_type", "event_type", "t_start", "t_end", "confidence", "meta"]
    summary_fields = [
        "child_id", "task_type", "duration_sec",
        "pose_present_ratio", "adult_present_ratio",
        "adult_hand_active_time_frac", "adult_hand_mean_activity",
        "freeze_time_frac", "hand_to_face_time_frac", "repetitive_motion_time_frac",
        "engaged_with_adult_time_frac", "disengaged_with_adult_time_frac", "hands_near_torso_time_frac",
        "repetitive_motion_freq_hz",
    ]

    with open(out_e, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=events_fields).writeheader()
    with open(out_s, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    for i in range(n):
        child_id = str(df.iloc[i]["child_id"])
        vp = Path(str(df.iloc[i]["free_play_path"]))
        if not vp.exists():
            continue

        print(f"[{i+1}/{n}] child={child_id} free_play={vp.name}", flush=True)

        events, summary = process_free_play_video(
            child_id=child_id,
            video_path=vp,
            runner=runner,
            sample_every_n=args.sample_every_n,
            max_frames=args.max_frames,
            min_pose_vis=args.min_pose_vis,
            min_seg_dur=args.min_seg_dur,
            freeze_speed_thr=args.freeze_speed_thr,
            freeze_min_dur=args.freeze_min_dur,
            hand_face_thr=args.hand_face_thr,
            flap_freq_min=args.flap_freq_min,
            flap_freq_max=args.flap_freq_max,
            flap_power_thr=args.flap_power_thr,
            rocking_freq_min=args.rocking_freq_min,
            rocking_freq_max=args.rocking_freq_max,
            engaged_head_turn_thr=args.engaged_head_turn_thr,
            engaged_min_dur=args.engaged_min_dur,
        )

        with open(out_e, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=events_fields)
            for e in events:
                w.writerow(
                    {
                        "child_id": e.child_id,
                        "task_type": e.task_type,
                        "event_type": e.event_type,
                        "t_start": round(float(e.t_start), 3),
                        "t_end": round(float(e.t_end), 3),
                        "confidence": round(float(e.confidence), 4),
                        "meta": e.meta,
                    }
                )

        with open(out_s, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writerow(summary)

    runner.close()
    print(f"Wrote: {out_e}")
    print(f"Wrote: {out_s}")


if __name__ == "__main__":
    main()
