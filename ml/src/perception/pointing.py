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


def _in_adult_region(
    x: float,
    y: float,
    mode: str,
    y_max: float = 0.62,
    side_x: float = 0.18,
) -> bool:
    """
    Adult hands tend to enter from top or sides.
    Normalized coords in [0,1].

    Args:
        x, y: normalized position
        mode: "top_or_side", "top_only", or "none"
        y_max: threshold for "top" region (hands above this y are adult-like)
        side_x: threshold for side regions (hands within this x from edges are adult-like)

    Returns:
        True if position is in adult region
    """
    if mode == "none":
        return True

    if mode == "top_only":
        return y <= y_max

    return (y <= y_max) or (x <= side_x) or (x >= 1.0 - side_x)


def _angle_in_range(angle: float, range_tuple: Tuple[float, float]) -> bool:
    """Check if angle falls within range, handling wraparound."""
    low, high = range_tuple
    if low <= high:
        return low <= angle <= high
    else:
        # Wraps around 0/360
        return angle >= low or angle <= high


# ---------------------------
# Data classes (defined early for type hints)
# ---------------------------
@dataclass
class PointSample:
    t_sec: float
    angle_deg: float
    conf: float  # handedness/proxy confidence


@dataclass
class PointSegment:
    t_start: float
    t_end: float
    angle_deg: float
    stability: float
    n_samples: int
    mean_conf: float = 0.5  # average sample confidence


@dataclass
class OrientEvent:
    """Child orienting/turning toward or away from parent location."""
    event_type: str  # "ORIENT_TO_PARENT", "ORIENT_AWAY"
    t_start: float
    t_end: float
    head_angle_start_deg: float  # head direction at start
    head_angle_end_deg: float  # head direction at end
    angle_change_deg: float  # signed turn amount
    confidence: float  # based on landmark visibility
    duration_sec: float


@dataclass
class PointingEvent:
    """Parent pointing gesture event."""
    event_type: str  # "PARENT_POINT"
    t_start: float
    t_end: float
    point_angle_deg: float  # direction of pointing
    stability: float  # angle stability over duration
    confidence: float  # combined confidence score
    method: str  # "hand" or "pose"
    n_samples: int


@dataclass
class PointingBout:
    """Grouped pointing events forming a single "attempt"."""
    bout_id: int
    t_start: float  # start of first event in bout
    t_end: float  # end of last event in bout
    n_events: int  # number of pointing events in this bout
    total_duration_sec: float  # sum of event durations (not bout span)
    mean_confidence: float
    mean_angle_deg: float  # average pointing direction
    events: List[PointingEvent] = field(default_factory=list)


@dataclass
class HeadSample:
    """Per-frame head orientation sample."""
    t_sec: float
    head_angle_deg: float  # yaw angle: 0=right, 90=down, 180=left, 270=up
    confidence: float  # mean visibility of head landmarks


@dataclass
class OrientingSummary:
    """Per-video summary of orienting behavior."""
    child_id: str
    task_type: str
    n_orient_to_parent: int
    n_orient_away: int
    total_orient_duration_sec: float
    mean_orient_confidence: float
    # Raw pointing events
    n_parent_points: int
    total_point_duration_sec: float
    mean_point_confidence: float
    # Bout-level metrics (grouped pointing attempts)
    n_pointing_bouts: int = 0
    mean_bout_duration_sec: float = 0.0
    time_to_first_point_sec: Optional[float] = None  # None if no pointing


# Parent location angle ranges
# Vector from ear midpoint to nose: 0=right, 90=down, 180=left, 270=up
# When child faces camera, nose is below ear midpoint -> ~90 degrees
PARENT_ANGLE_RANGES = {
    "behind_camera": (60.0, 120.0),  # facing camera = facing parent (~90 deg)
    "left_side": (150.0, 210.0),     # child looking left (~180 deg)
    "right_side": (330.0, 30.0),     # child looking right (~0 deg, wraps around)
}


# ---------------------------
# Child/adult pose selection
# ---------------------------
def select_child_pose(
    poses: List[Any],
    min_visibility: float = 0.3,
) -> Optional[Any]:
    """
    Select the most likely child pose from multiple detected poses.

    Uses position-based heuristic: child is typically centered,
    adult hands/body enter from edges or top.

    Args:
        poses: list of pose landmarks from MediaPipe
        min_visibility: minimum visibility for nose landmark

    Returns:
        Most likely child pose landmarks, or None if none found
    """
    if not poses:
        return None

    if len(poses) == 1:
        return poses[0]

    # Score each pose by how "child-like" the position is
    best_pose = None
    best_score = -1.0

    for lms in poses:
        # Use nose position as proxy for pose center
        nose = lms[0] if len(lms) > 0 else None
        if nose is None:
            continue

        nose_vis = float(getattr(nose, "visibility", 0.0) or 0.0)
        if nose_vis < min_visibility:
            continue

        nx, ny = float(nose.x), float(nose.y)

        # Position score: centered = more likely child
        # Score is highest at (0.5, 0.5) and decreases toward edges
        center_score = (1.0 - abs(nx - 0.5) * 2.0) * (1.0 - abs(ny - 0.5) * 1.5)
        center_score = max(0.0, center_score)

        # Penalty for edge positions (adult-like)
        if nx <= 0.20 or nx >= 0.80:
            center_score *= 0.2  # Strong penalty for side edges
        if ny <= 0.25:
            center_score *= 0.3  # Strong penalty for top region

        if center_score > best_score:
            best_score = center_score
            best_pose = lms

    return best_pose if best_pose is not None else poses[0]


# ---------------------------
# Head orientation from pose
# ---------------------------
# MediaPipe pose landmark indices for head
NOSE_IDX = 0
LEFT_EYE_INNER_IDX = 1
LEFT_EYE_IDX = 2
LEFT_EYE_OUTER_IDX = 3
RIGHT_EYE_INNER_IDX = 4
RIGHT_EYE_IDX = 5
RIGHT_EYE_OUTER_IDX = 6
LEFT_EAR_IDX = 7
RIGHT_EAR_IDX = 8


def compute_head_orientation(
    pose_landmarks: Any,
    min_visibility: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """
    Compute head yaw orientation from pose landmarks.

    Uses the vector from midpoint of ears to nose as a proxy for
    head facing direction (gaze vector approximation).

    Args:
        pose_landmarks: MediaPipe pose landmarks
        min_visibility: minimum visibility threshold for landmarks

    Returns:
        (angle_deg, confidence) or None if insufficient landmarks
        angle_deg: 0=right, 90=down, 180=left, 270=up (image coords)
    """
    if pose_landmarks is None:
        return None

    lms = pose_landmarks

    # Get key landmarks
    nose = lms[NOSE_IDX] if len(lms) > NOSE_IDX else None
    left_ear = lms[LEFT_EAR_IDX] if len(lms) > LEFT_EAR_IDX else None
    right_ear = lms[RIGHT_EAR_IDX] if len(lms) > RIGHT_EAR_IDX else None

    if nose is None:
        return None

    # Check visibility
    nose_vis = float(getattr(nose, "visibility", 0.0) or 0.0)
    if nose_vis < min_visibility:
        return None

    # Get ear positions (use eyes as fallback)
    # Use Tuple[float, float] | None to help type checker understand x and y are set together
    left_pos: Optional[Tuple[float, float]] = None
    right_pos: Optional[Tuple[float, float]] = None
    left_vis = 0.0
    right_vis = 0.0

    if left_ear is not None:
        left_vis = float(getattr(left_ear, "visibility", 0.0) or 0.0)
        if left_vis >= min_visibility:
            left_pos = (float(left_ear.x), float(left_ear.y))

    if right_ear is not None:
        right_vis = float(getattr(right_ear, "visibility", 0.0) or 0.0)
        if right_vis >= min_visibility:
            right_pos = (float(right_ear.x), float(right_ear.y))

    # Fallback to eyes if ears not visible
    if left_pos is None and len(lms) > LEFT_EYE_IDX:
        left_eye = lms[LEFT_EYE_IDX]
        left_vis = float(getattr(left_eye, "visibility", 0.0) or 0.0)
        if left_vis >= min_visibility:
            left_pos = (float(left_eye.x), float(left_eye.y))

    if right_pos is None and len(lms) > RIGHT_EYE_IDX:
        right_eye = lms[RIGHT_EYE_IDX]
        right_vis = float(getattr(right_eye, "visibility", 0.0) or 0.0)
        if right_vis >= min_visibility:
            right_pos = (float(right_eye.x), float(right_eye.y))

    # Need at least one side reference
    if left_pos is None and right_pos is None:
        return None

    # Compute ear/eye midpoint
    if left_pos is not None and right_pos is not None:
        mid_x = (left_pos[0] + right_pos[0]) / 2.0
        mid_y = (left_pos[1] + right_pos[1]) / 2.0
        confidence = (nose_vis + left_vis + right_vis) / 3.0
    elif left_pos is not None:
        mid_x, mid_y = left_pos[0], left_pos[1]
        confidence = (nose_vis + left_vis) / 2.0
    else:
        # right_pos must be not None here due to the check above
        assert right_pos is not None  # Help type checker
        mid_x, mid_y = right_pos[0], right_pos[1]
        confidence = (nose_vis + right_vis) / 2.0

    # Vector from midpoint to nose = facing direction
    nose_x, nose_y = float(nose.x), float(nose.y)
    dx = nose_x - mid_x
    dy = nose_y - mid_y

    # Convert to angle (0=right, 90=down in image coords)
    angle_deg = _angle_deg(dx, dy)

    return (angle_deg, confidence)


# ---------------------------
# Orient event detection
# ---------------------------
def detect_orient_events(
    head_samples: List[HeadSample],
    parent_angle_range: Tuple[float, float],
    min_turn_deg: float = 25.0,
    min_duration_sec: float = 0.2,
    max_gap_sec: float = 0.3,
    smoothing_window: int = 3,
    stable_threshold_deg: float = 5.0,
    stable_time_sec: float = 0.15,
) -> List[OrientEvent]:
    """
    Detect child orienting toward or away from parent.

    Looks for significant head turns that end up facing toward
    (or away from) the parent's expected location.

    Args:
        head_samples: list of HeadSample sorted by time
        parent_angle_range: (low_deg, high_deg) where parent is located
        min_turn_deg: minimum angle change to count as a turn
        min_duration_sec: minimum event duration
        max_gap_sec: max gap in samples before splitting events
        smoothing_window: window size for angle smoothing
        stable_threshold_deg: angle change below this is considered stable
        stable_time_sec: time to remain stable before ending turn

    Returns:
        List of OrientEvent
    """
    if len(head_samples) < 3:
        return []

    samples = sorted(head_samples, key=lambda s: s.t_sec)

    # Smooth angles to reduce noise
    angles = np.array([s.head_angle_deg for s in samples])
    if smoothing_window > 1 and len(angles) >= smoothing_window:
        # Circular smoothing (handle wraparound)
        cos_vals = np.cos(np.radians(angles))
        sin_vals = np.sin(np.radians(angles))
        kernel = np.ones(smoothing_window) / smoothing_window
        cos_smooth = np.convolve(cos_vals, kernel, mode="same")
        sin_smooth = np.convolve(sin_vals, kernel, mode="same")
        angles = np.degrees(np.arctan2(sin_smooth, cos_smooth)) % 360.0

    events: List[OrientEvent] = []

    # Detect significant turns
    i = 0
    while i < len(samples) - 1:
        # Look for start of a turn
        start_angle = angles[i]
        start_time = samples[i].t_sec

        # Scan forward to find end of turn
        j = i + 1
        max_change = 0.0
        max_j = i
        stable_since_j = None  # index where we started stabilizing

        while j < len(samples):
            # Check for gap
            if samples[j].t_sec - samples[j - 1].t_sec > max_gap_sec:
                break

            curr_angle = angles[j]
            change = _circular_diff_deg(curr_angle, start_angle)

            if change > max_change + stable_threshold_deg:
                # Still turning - update max and reset stable tracker
                max_change = change
                max_j = j
                stable_since_j = None
            else:
                # Angle not increasing much - might be stabilizing
                if stable_since_j is None:
                    stable_since_j = j

                # Check if we've been stable long enough after a significant turn
                if max_change >= min_turn_deg and stable_since_j is not None:
                    stable_duration = samples[j].t_sec - samples[stable_since_j].t_sec
                    if stable_duration >= stable_time_sec:
                        break

            j += 1

        # Check if we found a significant turn
        if max_change >= min_turn_deg:
            end_idx = max_j
            end_time = samples[end_idx].t_sec
            end_angle = angles[end_idx]
            duration = end_time - start_time

            if duration >= min_duration_sec:
                # Determine if turn was toward or away from parent
                end_facing_parent = _angle_in_range(end_angle, parent_angle_range)
                start_facing_parent = _angle_in_range(start_angle, parent_angle_range)

                if end_facing_parent and not start_facing_parent:
                    event_type = "ORIENT_TO_PARENT"
                elif not end_facing_parent and start_facing_parent:
                    event_type = "ORIENT_AWAY"
                else:
                    # Turn within same region, skip
                    i = max_j + 1
                    continue

                # Signed angle change
                signed_change = _signed_angle_diff(end_angle, start_angle)

                # Average confidence
                confs = [samples[k].confidence for k in range(i, end_idx + 1)]
                avg_conf = float(np.mean(confs)) if confs else 0.5

                events.append(OrientEvent(
                    event_type=event_type,
                    t_start=start_time,
                    t_end=end_time,
                    head_angle_start_deg=start_angle,
                    head_angle_end_deg=end_angle,
                    angle_change_deg=signed_change,
                    confidence=avg_conf,
                    duration_sec=duration,
                ))

            i = max_j + 1
        else:
            i += 1

    return events


def _signed_angle_diff(to_angle: float, from_angle: float) -> float:
    """Compute signed angle difference (positive = clockwise in image coords)."""
    diff = to_angle - from_angle
    while diff > 180.0:
        diff -= 360.0
    while diff < -180.0:
        diff += 360.0
    return diff


# ---------------------------
# Pointing event conversion
# ---------------------------
def segment_to_pointing_event(
    seg: PointSegment,
    method: str = "hand",
    base_confidence: float = 0.7,
) -> PointingEvent:
    """
    Convert a PointSegment to a PointingEvent with confidence scoring.

    Confidence is based on:
    - Stability of the pointing angle (40%)
    - Duration (20%)
    - Sample density (20%)
    - Detector/handedness confidence (20%)
    """
    # Duration factor: longer is better, up to 2 seconds
    duration = seg.t_end - seg.t_start
    duration_factor = min(1.0, duration / 2.0)

    # Sample density factor
    if duration > 0:
        samples_per_sec = seg.n_samples / duration
        density_factor = min(1.0, samples_per_sec / 15.0)  # 15 fps = full score
    else:
        density_factor = 0.5

    # Detector confidence factor (handedness score from MediaPipe)
    detector_conf = seg.mean_conf

    # Combined confidence with weighted factors
    confidence = (
        base_confidence * seg.stability * 0.40 +
        base_confidence * duration_factor * 0.20 +
        base_confidence * density_factor * 0.20 +
        detector_conf * 0.20
    )

    return PointingEvent(
        event_type="PARENT_POINT",
        t_start=seg.t_start,
        t_end=seg.t_end,
        point_angle_deg=seg.angle_deg,
        stability=seg.stability,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method=method,
        n_samples=seg.n_samples,
    )


# ---------------------------
# Summary creation
# ---------------------------
def merge_pointing_events(
    events: List[PointingEvent],
    merge_gap_sec: float = 1.0,
) -> List[PointingBout]:
    """
    Group pointing events into "bouts" (attempts).

    Events within merge_gap_sec of each other are grouped into a single bout.
    This helps distinguish repeated brief points from sustained pointing.

    Args:
        events: list of PointingEvent sorted by t_start
        merge_gap_sec: max gap between events to group into same bout

    Returns:
        List of PointingBout
    """
    if not events:
        return []

    # Ensure sorted by start time
    events = sorted(events, key=lambda e: e.t_start)

    bouts: List[PointingBout] = []
    current_group: List[PointingEvent] = [events[0]]

    for e in events[1:]:
        last_end = current_group[-1].t_end
        if e.t_start - last_end <= merge_gap_sec:
            current_group.append(e)
        else:
            # Finalize current bout
            bouts.append(_create_bout(current_group, len(bouts)))
            current_group = [e]

    # Finalize last bout
    if current_group:
        bouts.append(_create_bout(current_group, len(bouts)))

    return bouts


def _create_bout(events: List[PointingEvent], bout_id: int) -> PointingBout:
    """Create a PointingBout from a group of events."""
    durations = [e.t_end - e.t_start for e in events]
    confs = [e.confidence for e in events]
    angles = [e.point_angle_deg for e in events]

    return PointingBout(
        bout_id=bout_id,
        t_start=events[0].t_start,
        t_end=events[-1].t_end,
        n_events=len(events),
        total_duration_sec=sum(durations),
        mean_confidence=float(np.mean(confs)),
        mean_angle_deg=float(np.mean(angles)),
        events=events,
    )


def create_orienting_summary(
    child_id: str,
    task_type: str,
    orient_events: List[OrientEvent],
    pointing_events: List[PointingEvent],
    pointing_bouts: Optional[List[PointingBout]] = None,
    video_start_time: float = 0.0,
) -> OrientingSummary:
    """
    Create per-video summary of orienting and pointing events.

    Args:
        child_id: child identifier
        task_type: task type (joint_attention, etc.)
        orient_events: list of orient events
        pointing_events: list of pointing events
        pointing_bouts: optional pre-computed bouts (computed if None)
        video_start_time: video start time for time_to_first_point calculation
    """
    n_orient_to = sum(1 for e in orient_events if e.event_type == "ORIENT_TO_PARENT")
    n_orient_away = sum(1 for e in orient_events if e.event_type == "ORIENT_AWAY")
    orient_durations = [e.duration_sec for e in orient_events]
    orient_confs = [e.confidence for e in orient_events]

    point_durations = [e.t_end - e.t_start for e in pointing_events]
    point_confs = [e.confidence for e in pointing_events]

    # Compute bout-level metrics
    if pointing_bouts is None and pointing_events:
        pointing_bouts = merge_pointing_events(pointing_events, merge_gap_sec=1.0)

    n_bouts = len(pointing_bouts) if pointing_bouts else 0
    bout_durations = [b.total_duration_sec for b in pointing_bouts] if pointing_bouts else []
    mean_bout_dur = float(np.mean(bout_durations)) if bout_durations else 0.0

    # Time to first point (from video start)
    time_to_first: Optional[float] = None
    if pointing_events:
        first_point_time = min(e.t_start for e in pointing_events)
        time_to_first = first_point_time - video_start_time

    return OrientingSummary(
        child_id=child_id,
        task_type=task_type,
        n_orient_to_parent=n_orient_to,
        n_orient_away=n_orient_away,
        total_orient_duration_sec=sum(orient_durations),
        mean_orient_confidence=float(np.mean(orient_confs)) if orient_confs else 0.0,
        n_parent_points=len(pointing_events),
        total_point_duration_sec=sum(point_durations),
        mean_point_confidence=float(np.mean(point_confs)) if point_confs else 0.0,
        n_pointing_bouts=n_bouts,
        mean_bout_duration_sec=mean_bout_dur,
        time_to_first_point_sec=time_to_first,
    )




# ---------------------------
# Hand vector extraction (STRICT)
# ---------------------------
# MediaPipe hand landmark indices
WRIST_IDX = 0
INDEX_TIP_IDX = 8
MIDDLE_MCP_IDX = 9  # Middle finger metacarpophalangeal joint


def _hand_point_vector(
    hand_result: Any,
    adult_region: str,
    min_vec_len: float,
    adult_y_max: float = 0.72,
    adult_side_x: float = 0.18,
    adaptive_threshold: bool = True,
    min_pointiness_ratio: float = 1.8,
    min_hand_conf: float = 0.55,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Returns best hand pointing vector (dx, dy, conf, wrist_x, wrist_y) or None.

    Strong gating to avoid child hand false positives:
      - wrist must be in adult region (top or side)
      - index_tip - wrist vector must be long enough (pointiness proxy)
      - handedness confidence must be above min_hand_conf

    Args:
        hand_result: MediaPipe hand detection result
        adult_region: "top_or_side", "top_only", or "none"
        min_vec_len: minimum pointing vector length (normalized, fallback threshold)
        adult_y_max: y threshold for adult region (normalized, lower = higher in frame)
        adult_side_x: x threshold for side regions (normalized)
        adaptive_threshold: if True, scale min_vec_len by hand size
        min_pointiness_ratio: when adaptive, require pointing vector >= ratio * hand_size
        min_hand_conf: minimum handedness confidence to accept hand detection
    """
    if not hand_result or not getattr(hand_result, "hand_landmarks", None):
        return None

    hands = hand_result.hand_landmarks
    handedness = getattr(hand_result, "handedness", None)

    best = None
    best_score = -1.0

    for i, lms in enumerate(hands):
        wrist = lms[WRIST_IDX]
        tip = lms[INDEX_TIP_IDX]
        middle_mcp = lms[MIDDLE_MCP_IDX]

        wx, wy = float(wrist.x), float(wrist.y)
        if not _in_adult_region(wx, wy, adult_region, y_max=adult_y_max, side_x=adult_side_x):
            continue

        dx = float(tip.x - wrist.x)
        dy = float(tip.y - wrist.y)
        mag = math.hypot(dx, dy)

        # Compute adaptive threshold based on hand size
        # Hand size proxy: wrist to middle finger MCP distance
        hand_size = math.hypot(
            float(middle_mcp.x - wrist.x),
            float(middle_mcp.y - wrist.y),
        )

        if adaptive_threshold and hand_size > 0.01:
            # Adaptive: pointing vector should be >= ratio * hand_size
            effective_min_len = min_pointiness_ratio * hand_size
            # Enforce full minimum floor (not half) to avoid false positives
            effective_min_len = max(effective_min_len, min_vec_len)
        else:
            # Fallback to fixed threshold
            effective_min_len = min_vec_len

        # Pointiness gate
        if mag < effective_min_len:
            continue

        # Get handedness confidence
        conf = 0.5
        if handedness and i < len(handedness) and handedness[i]:
            top = handedness[i][0]
            conf = float(getattr(top, "score", 0.0) or 0.0)

        # Gate on handedness confidence to reject low-quality detections
        if conf < min_hand_conf:
            continue

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
    # Compute mean handedness/detector confidence from samples
    mean_conf = float(np.mean([s.conf for s in seg])) if seg else 0.5
    return PointSegment(
        t_start=float(seg[0].t_sec),
        t_end=float(seg[-1].t_sec),
        angle_deg=angle,
        stability=stability,
        n_samples=len(seg),
        mean_conf=mean_conf,
    )


def detect_point_segments(
    samples: List[PointSample],
    min_frames: int,
    max_angle_jitter_deg: float,
    max_gap_sec: float,
    merge_gap_sec: float,
    min_duration_sec: float = 0.2,
    min_stability: float = 0.45,
    min_mean_conf: float = 0.5,
) -> List[PointSegment]:
    """
    Detect stable pointing segments from point samples.

    Args:
        samples: list of PointSample
        min_frames: minimum number of samples in segment
        max_angle_jitter_deg: max angle deviation within segment
        max_gap_sec: max time gap before splitting segment
        merge_gap_sec: max gap for merging adjacent segments
        min_duration_sec: minimum segment duration to accept
        min_stability: minimum stability score to accept (0-1)
        min_mean_conf: minimum mean handedness confidence to accept

    Returns:
        List of PointSegment that pass all filters
    """
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
            new_mean_conf = float((last.mean_conf * last.n_samples + seg.mean_conf * seg.n_samples) / max(1, total_n))
            merged[-1] = PointSegment(
                t_start=last.t_start,
                t_end=seg.t_end,
                angle_deg=new_angle,
                stability=new_stability,
                n_samples=total_n,
                mean_conf=new_mean_conf,
            )
        else:
            merged.append(seg)

    # Apply segment-level quality filters
    filtered: List[PointSegment] = []
    for seg in merged:
        duration = seg.t_end - seg.t_start
        if duration < min_duration_sec:
            continue
        if seg.stability < min_stability:
            continue
        if seg.mean_conf < min_mean_conf:
            continue
        filtered.append(seg)

    return filtered


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
    adult_y_max: float = 0.72,
    adult_side_x: float = 0.18,
    adaptive_threshold: bool = True,
    min_pointiness_ratio: float = 1.8,
    min_hand_conf: float = 0.55,
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

        hv = _hand_point_vector(
            hand_res,
            adult_region=adult_region,
            min_vec_len=min_vec_len,
            adult_y_max=adult_y_max,
            adult_side_x=adult_side_x,
            adaptive_threshold=adaptive_threshold,
            min_pointiness_ratio=min_pointiness_ratio,
            min_hand_conf=min_hand_conf,
        )
        if hv is None:
            continue

        dx, dy, conf, _wx, _wy = hv
        out.append(PointSample(float(t_sec), _angle_deg(dx, dy), float(conf)))

    cap.release()
    return out


# ---------------------------
# Enhanced video processing (with head orientation)
# ---------------------------
def process_video_with_orient(
    video_path: Path,
    sample_every_n: int,
    max_frames: int,
    pose_landmarker: Any,
    hand_landmarker: Optional[Any],
    adult_region: str,
    min_vec_len: float,
    head_min_vis: float = 0.3,
    adult_y_max: float = 0.72,
    adult_side_x: float = 0.18,
    adaptive_threshold: bool = True,
    min_pointiness_ratio: float = 1.8,
    min_hand_conf: float = 0.55,
) -> Tuple[List[HeadSample], List[PointSample]]:
    """
    Process video extracting both head orientation and pointing samples.

    Args:
        video_path: path to video file
        sample_every_n: frame sampling rate
        max_frames: max frames to process
        pose_landmarker: MediaPipe pose landmarker
        hand_landmarker: optional hand landmarker for pointing
        adult_region: region filter for adult hands
        min_vec_len: minimum pointing vector length
        head_min_vis: minimum visibility for head landmarks
        adult_y_max: y threshold for adult region (normalized)
        adult_side_x: x threshold for side adult regions (normalized)

    Returns:
        (head_samples, point_samples) tuple
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0

    head_samples: List[HeadSample] = []
    point_samples: List[PointSample] = []
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

        # Pose detection for head orientation
        try:
            pose_res = pose_landmarker.detect(mp_image)
            if pose_res and pose_res.pose_landmarks:
                # Select child pose using position-based heuristic
                child_pose = select_child_pose(pose_res.pose_landmarks, min_visibility=head_min_vis)
                head_result = compute_head_orientation(child_pose, min_visibility=head_min_vis)
                if head_result is not None:
                    angle, conf = head_result
                    head_samples.append(HeadSample(
                        t_sec=float(t_sec),
                        head_angle_deg=float(angle),
                        confidence=float(conf),
                    ))
        except Exception:
            pass

        # Hand detection for pointing (if hand_landmarker provided)
        if hand_landmarker is not None:
            try:
                hand_res = hand_landmarker.detect(mp_image)
                hv = _hand_point_vector(
                    hand_res,
                    adult_region=adult_region,
                    min_vec_len=min_vec_len,
                    adult_y_max=adult_y_max,
                    adult_side_x=adult_side_x,
                    adaptive_threshold=adaptive_threshold,
                    min_pointiness_ratio=min_pointiness_ratio,
                    min_hand_conf=min_hand_conf,
                )
                if hv is not None:
                    dx, dy, conf, _wx, _wy = hv
                    point_samples.append(PointSample(float(t_sec), _angle_deg(dx, dy), float(conf)))
            except Exception:
                pass

    cap.release()
    return head_samples, point_samples


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Detect orient-to-parent and pointing events for joint attention analysis."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", default="joint_attention", choices=["joint_attention", "imitation", "free_play"])
    ap.add_argument("--out", default="data/derived/orient_point_events.csv")

    ap.add_argument("--sample_every_n", type=int, default=2)
    ap.add_argument("--max_frames", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)

    # Pointing detection args
    ap.add_argument("--adult_region", default="top_or_side", choices=["top_or_side", "top_only", "none"])
    ap.add_argument("--min_vec_len", type=float, default=0.10,
                    help="Index_tip-wrist min length (normalized). Fallback when adaptive disabled.")
    ap.add_argument("--no_adaptive_vec_len", action="store_true",
                    help="Disable adaptive min_vec_len scaling based on hand size")
    ap.add_argument("--min_pointiness_ratio", type=float, default=1.8,
                    help="When adaptive, require pointing vector >= ratio * hand_size")
    ap.add_argument("--min_hand_conf", type=float, default=0.55,
                    help="Minimum handedness confidence to accept hand detection")
    ap.add_argument("--min_frames", type=int, default=6)
    ap.add_argument("--max_angle_jitter_deg", type=float, default=10.0)
    ap.add_argument("--max_gap_sec", type=float, default=0.25)
    ap.add_argument("--merge_gap_sec", type=float, default=0.40)
    ap.add_argument("--min_segment_duration", type=float, default=0.2,
                    help="Minimum segment duration (sec) to accept")
    ap.add_argument("--min_segment_stability", type=float, default=0.45,
                    help="Minimum segment stability (0-1) to accept")
    ap.add_argument("--min_segment_conf", type=float, default=0.5,
                    help="Minimum mean handedness confidence for segment")
    ap.add_argument("--adult_y_max", type=float, default=0.72,
                    help="Y threshold for adult region (normalized). Hands above this are adult-like.")
    ap.add_argument("--adult_side_x", type=float, default=0.18,
                    help="X threshold for side regions (normalized). Hands within this from edges are adult-like.")

    # Orient detection args
    ap.add_argument("--enable_orient", action="store_true",
                    help="Enable child orient-to-parent detection (requires pose)")
    ap.add_argument("--parent_location", default="behind_camera",
                    choices=["behind_camera", "left_side", "right_side"],
                    help="Expected parent location for orient detection")
    ap.add_argument("--min_turn_deg", type=float, default=25.0,
                    help="Minimum head turn angle to count as orient event")
    ap.add_argument("--min_orient_duration", type=float, default=0.2,
                    help="Minimum duration for orient event (sec)")
    ap.add_argument("--head_min_vis", type=float, default=0.3,
                    help="Minimum visibility for head landmarks")

    # Output args
    ap.add_argument("--summary_out", default=None,
                    help="Output CSV for per-video summary")
    ap.add_argument("--orient_out", default=None,
                    help="Separate output CSV for orient events only")
    ap.add_argument("--pointing_merge_gap", type=float, default=1.0,
                    help="Gap (sec) for merging pointing events into bouts. 0=disabled.")

    # Legacy compatibility
    ap.add_argument("--enable_pose_fallback", action="store_true",
                    help="(Deprecated) Use --enable_orient instead")

    args = ap.parse_args()

    # Handle legacy arg
    if args.enable_pose_fallback:
        args.enable_orient = True

    df = pd.read_csv(args.manifest)
    col = {
        "joint_attention": "joint_attention_path",
        "imitation": "imitation_path",
        "free_play": "free_play_path",
    }[args.task]
    if col not in df.columns:
        raise ValueError(f"Manifest missing column '{col}'. Present: {list(df.columns)}")

    hand_model, pose_model = get_model_paths()

    # Hand landmarker for pointing detection
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    # Pose landmarker for orient detection
    pose_landmarker = None
    if args.enable_orient:
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=2,  # Detect child + parent, use heuristic to select child
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Main events CSV (combined pointing + orient events)
    fieldnames = [
        "child_id", "task_type", "event_type", "t_start", "t_end",
        "angle_deg", "confidence", "method", "stability", "n_samples",
        "angle_change_deg", "duration_sec",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    # Optional orient-only CSV
    orient_path = Path(args.orient_out) if args.orient_out else None
    if orient_path:
        orient_path.parent.mkdir(parents=True, exist_ok=True)
        orient_fieldnames = [
            "child_id", "task_type", "event_type", "t_start", "t_end",
            "head_angle_start_deg", "head_angle_end_deg", "angle_change_deg",
            "confidence", "duration_sec",
        ]
        with open(orient_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=orient_fieldnames)
            w.writeheader()

    # Collect summaries
    summaries: List[OrientingSummary] = []

    # Get parent angle range
    parent_angle_range = PARENT_ANGLE_RANGES.get(args.parent_location, (160.0, 200.0))

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    print(f"Processing {n} videos for task={args.task}")
    if args.enable_orient:
        print(f"  Orient detection: enabled (parent={args.parent_location})")
    else:
        print("  Orient detection: disabled (use --enable_orient to enable)")

    for i in range(n):
        child_id = str(df.iloc[i]["child_id"])
        video_path = Path(str(df.iloc[i][col]))
        if not video_path.exists():
            print(f"[{i+1}/{n}] child={child_id} - video not found, skipping")
            continue

        print(f"[{i+1}/{n}] child={child_id} task={args.task}", flush=True)

        orient_events: List[OrientEvent] = []
        pointing_events: List[PointingEvent] = []

        if args.enable_orient and pose_landmarker is not None:
            # Use enhanced processing with both head orientation and pointing
            head_samples, point_samples = process_video_with_orient(
                video_path=video_path,
                sample_every_n=args.sample_every_n,
                max_frames=args.max_frames,
                pose_landmarker=pose_landmarker,
                hand_landmarker=hand_landmarker,
                adult_region=args.adult_region,
                min_vec_len=args.min_vec_len,
                head_min_vis=args.head_min_vis,
                adult_y_max=args.adult_y_max,
                adult_side_x=args.adult_side_x,
                adaptive_threshold=not args.no_adaptive_vec_len,
                min_pointiness_ratio=args.min_pointiness_ratio,
                min_hand_conf=args.min_hand_conf,
            )

            # Detect orient events
            orient_events = detect_orient_events(
                head_samples=head_samples,
                parent_angle_range=parent_angle_range,
                min_turn_deg=args.min_turn_deg,
                min_duration_sec=args.min_orient_duration,
                max_gap_sec=args.max_gap_sec,
            )

            # Detect pointing segments and convert to events
            point_segments = detect_point_segments(
                samples=point_samples,
                min_frames=args.min_frames,
                max_angle_jitter_deg=args.max_angle_jitter_deg,
                max_gap_sec=args.max_gap_sec,
                merge_gap_sec=args.merge_gap_sec,
                min_duration_sec=args.min_segment_duration,
                min_stability=args.min_segment_stability,
                min_mean_conf=args.min_segment_conf,
            )
            pointing_events = [segment_to_pointing_event(seg, method="hand") for seg in point_segments]

        else:
            # Hand-only detection (legacy behavior)
            point_samples = process_video_hand_only(
                video_path=video_path,
                sample_every_n=args.sample_every_n,
                max_frames=args.max_frames,
                hand_landmarker=hand_landmarker,
                adult_region=args.adult_region,
                min_vec_len=args.min_vec_len,
                adult_y_max=args.adult_y_max,
                adult_side_x=args.adult_side_x,
                adaptive_threshold=not args.no_adaptive_vec_len,
                min_pointiness_ratio=args.min_pointiness_ratio,
                min_hand_conf=args.min_hand_conf,
            )

            point_segments = detect_point_segments(
                samples=point_samples,
                min_frames=args.min_frames,
                max_angle_jitter_deg=args.max_angle_jitter_deg,
                max_gap_sec=args.max_gap_sec,
                merge_gap_sec=args.merge_gap_sec,
                min_duration_sec=args.min_segment_duration,
                min_stability=args.min_segment_stability,
                min_mean_conf=args.min_segment_conf,
            )
            pointing_events = [segment_to_pointing_event(seg, method="hand") for seg in point_segments]

        # Write events to main CSV
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)

            # Write orient events
            for e in orient_events:
                w.writerow({
                    "child_id": child_id,
                    "task_type": args.task,
                    "event_type": e.event_type,
                    "t_start": round(e.t_start, 3),
                    "t_end": round(e.t_end, 3),
                    "angle_deg": round(e.head_angle_end_deg, 2),
                    "confidence": round(e.confidence, 4),
                    "method": "head_pose",
                    "stability": "",
                    "n_samples": "",
                    "angle_change_deg": round(e.angle_change_deg, 2),
                    "duration_sec": round(e.duration_sec, 3),
                })

            # Write pointing events
            for e in pointing_events:
                w.writerow({
                    "child_id": child_id,
                    "task_type": args.task,
                    "event_type": e.event_type,
                    "t_start": round(e.t_start, 3),
                    "t_end": round(e.t_end, 3),
                    "angle_deg": round(e.point_angle_deg, 2),
                    "confidence": round(e.confidence, 4),
                    "method": e.method,
                    "stability": round(e.stability, 4),
                    "n_samples": e.n_samples,
                    "angle_change_deg": "",
                    "duration_sec": round(e.t_end - e.t_start, 3),
                })

        # Write orient events to separate file if requested
        if orient_path and orient_events:
            with open(orient_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=orient_fieldnames)
                for e in orient_events:
                    w.writerow({
                        "child_id": child_id,
                        "task_type": args.task,
                        "event_type": e.event_type,
                        "t_start": round(e.t_start, 3),
                        "t_end": round(e.t_end, 3),
                        "head_angle_start_deg": round(e.head_angle_start_deg, 2),
                        "head_angle_end_deg": round(e.head_angle_end_deg, 2),
                        "angle_change_deg": round(e.angle_change_deg, 2),
                        "confidence": round(e.confidence, 4),
                        "duration_sec": round(e.duration_sec, 3),
                    })

        # Compute pointing bouts if merge_gap > 0
        pointing_bouts = None
        if args.pointing_merge_gap > 0 and pointing_events:
            pointing_bouts = merge_pointing_events(pointing_events, merge_gap_sec=args.pointing_merge_gap)

        # Create summary
        summary = create_orienting_summary(
            child_id, args.task, orient_events, pointing_events,
            pointing_bouts=pointing_bouts,
            video_start_time=0.0,  # video typically starts at t=0
        )
        summaries.append(summary)

    hand_landmarker.close()
    if pose_landmarker is not None:
        pose_landmarker.close()

    print(f"Wrote events: {out_path}")
    if orient_path:
        print(f"Wrote orient events: {orient_path}")

    # Write summary CSV
    if args.summary_out and summaries:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_fieldnames = [
            "child_id", "task_type", "n_orient_to_parent", "n_orient_away",
            "total_orient_duration_sec", "mean_orient_confidence",
            "n_parent_points", "total_point_duration_sec", "mean_point_confidence",
            "n_pointing_bouts", "mean_bout_duration_sec", "time_to_first_point_sec",
        ]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fieldnames)
            w.writeheader()
            for s in summaries:
                w.writerow({
                    "child_id": s.child_id,
                    "task_type": s.task_type,
                    "n_orient_to_parent": s.n_orient_to_parent,
                    "n_orient_away": s.n_orient_away,
                    "total_orient_duration_sec": round(s.total_orient_duration_sec, 3),
                    "mean_orient_confidence": round(s.mean_orient_confidence, 4),
                    "n_parent_points": s.n_parent_points,
                    "total_point_duration_sec": round(s.total_point_duration_sec, 3),
                    "mean_point_confidence": round(s.mean_point_confidence, 4),
                    "n_pointing_bouts": s.n_pointing_bouts,
                    "mean_bout_duration_sec": round(s.mean_bout_duration_sec, 3),
                    "time_to_first_point_sec": round(s.time_to_first_point_sec, 3) if s.time_to_first_point_sec is not None else "",
                })
        print(f"Wrote summary: {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()

