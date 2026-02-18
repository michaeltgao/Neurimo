from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Pose landmark indices (MediaPipe)
POSE_NOSE = 0
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12

# Response detection config
RESPONSE_WINDOW_SEC = 3.0  # How long after cue to look for response
HEAD_TURN_THRESHOLD = 0.04  # Minimum head_signal change to count as response (normalized units)
MIN_VISIBILITY = 0.35  # Minimum visibility for landmarks to be considered valid
MIN_VALID_FRACTION = 0.3  # Minimum fraction of valid samples in response window


@dataclass
class TracksData:
    """Container for loaded track data with visibility info."""
    t_sec: np.ndarray  # (N,) timestamps
    head_signal: np.ndarray  # (N,) normalized head orientation (nose_x - shoulder_midpoint_x)
    valid_mask: np.ndarray  # (N,) bool mask where landmarks are visible


def _load_tracks(tracks_dir: Path, child_id: str, task: str) -> Optional[TracksData]:
    """
    Load tracks and compute normalized head orientation signal.

    Returns TracksData with:
        - head_signal: nose_x - shoulder_midpoint_x (reduces whole-body translation confounds)
        - valid_mask: frames where nose and both shoulders have sufficient visibility
    """
    npz_path = tracks_dir / f"{child_id}_{task}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=False)
    t_sec = data["t_sec"].astype(float)
    pose = data["pose"].astype(float)
    if pose.shape[0] == 0:
        return None

    # Extract landmarks and visibility
    nose_x = pose[:, POSE_NOSE, 0]
    nose_vis = pose[:, POSE_NOSE, 3]
    l_shoulder_x = pose[:, POSE_L_SHOULDER, 0]
    l_shoulder_vis = pose[:, POSE_L_SHOULDER, 3]
    r_shoulder_x = pose[:, POSE_R_SHOULDER, 0]
    r_shoulder_vis = pose[:, POSE_R_SHOULDER, 3]

    # Compute normalized head signal: nose relative to shoulder midpoint
    # This removes whole-body translation confounds (leaning, etc.)
    shoulder_mid_x = 0.5 * (l_shoulder_x + r_shoulder_x)
    head_signal = nose_x - shoulder_mid_x

    # Valid mask: require nose and both shoulders visible
    valid_mask = (
        (nose_vis >= MIN_VISIBILITY) &
        (l_shoulder_vis >= MIN_VISIBILITY) &
        (r_shoulder_vis >= MIN_VISIBILITY) &
        np.isfinite(head_signal)
    )

    return TracksData(t_sec=t_sec, head_signal=head_signal, valid_mask=valid_mask)


def _detect_head_turn_after_cue(
    tracks: TracksData,
    cue_time: float,
    window_sec: float = RESPONSE_WINDOW_SEC,
    threshold: float = HEAD_TURN_THRESHOLD,
    min_valid_frac: float = MIN_VALID_FRACTION,
) -> Tuple[bool, float, float]:
    """
    Detect if child turned head after a cue using normalized head signal.

    Uses head_signal (nose_x - shoulder_midpoint_x) to reduce whole-body
    translation confounds. Requires sufficient visible frames in window.

    Returns:
        (responded, latency_sec, signed_delta)
        - latency is NaN if no response
        - signed_delta indicates direction: positive = head moved right, negative = left
    """
    t_sec = tracks.t_sec
    head_signal = tracks.head_signal
    valid_mask = tracks.valid_mask

    # Find samples in response window
    window_mask = (t_sec >= cue_time) & (t_sec <= cue_time + window_sec)
    if window_mask.sum() < 2:
        return False, float("nan"), float("nan")

    window_t = t_sec[window_mask]
    window_signal = head_signal[window_mask]
    window_valid = valid_mask[window_mask]

    # Visibility gating: require minimum fraction of valid samples
    valid_frac = window_valid.sum() / len(window_valid)
    if valid_frac < min_valid_frac:
        return False, float("nan"), float("nan")

    # Only use valid samples
    if window_valid.sum() < 2:
        return False, float("nan"), float("nan")

    valid_t = window_t[window_valid]
    valid_signal = window_signal[window_valid]

    # Baseline from first valid sample after cue
    baseline = valid_signal[0]

    # Compute signed deltas from baseline
    deltas = valid_signal - baseline

    # Find first time absolute delta exceeds threshold
    abs_deltas = np.abs(deltas)
    response_idx = np.where(abs_deltas >= threshold)[0]
    if len(response_idx) == 0:
        return False, float("nan"), float("nan")

    first_response = response_idx[0]
    latency = max(0.0, valid_t[first_response] - cue_time)
    signed_delta = float(deltas[first_response])

    return True, float(latency), signed_delta


def _deduplicate_cues(times: np.ndarray, min_gap_sec: float = 1.0) -> np.ndarray:
    """Remove cues that are too close together, keeping the first of each cluster."""
    if len(times) == 0:
        return times
    times = np.sort(times)
    keep = [times[0]]
    for t in times[1:]:
        if t - keep[-1] >= min_gap_sec:
            keep.append(t)
    return np.array(keep)


def _compute_post_cue_head_change(
    tracks: TracksData,
    cue_time: float,
    window_sec: float = RESPONSE_WINDOW_SEC,
) -> Tuple[float, float]:
    """
    Compute magnitude and direction of head turn after cue.

    Returns:
        (max_abs_delta, signed_delta_at_max)
        - max_abs_delta: maximum absolute change from baseline
        - signed_delta_at_max: signed value at max change (positive = right, negative = left)
    """
    t_sec = tracks.t_sec
    head_signal = tracks.head_signal
    valid_mask = tracks.valid_mask

    window_mask = (t_sec >= cue_time) & (t_sec <= cue_time + window_sec)
    if window_mask.sum() < 2:
        return float("nan"), float("nan")

    window_signal = head_signal[window_mask]
    window_valid = valid_mask[window_mask]

    if window_valid.sum() < 2:
        return float("nan"), float("nan")

    valid_signal = window_signal[window_valid]
    baseline = valid_signal[0]
    deltas = valid_signal - baseline

    max_idx = np.argmax(np.abs(deltas))
    max_abs_delta = float(np.abs(deltas[max_idx]))
    signed_delta = float(deltas[max_idx])

    return max_abs_delta, signed_delta


def extract_joint_attention_features(
    child_id: str,
    point_events_df: pd.DataFrame,
    audio_events_df: Optional[pd.DataFrame] = None,
    tracks_dir: Optional[Path] = None,
    video_duration_sec: Optional[float] = None,
) -> Dict[str, float]:
    """
    Extract joint attention features from pointing, audio events, and child tracks.

    Args:
        child_id: Child identifier
        point_events_df: DataFrame from pointing.py output with columns:
            child_id, task_type, t_start, t_end, angle_deg, method, stability, n_samples
        audio_events_df: DataFrame from audio_events.py output with columns:
            child_id, task_type, event_type, t_start, t_end, confidence, matched_phrase
            Event types: CALL_ATTENTION, LOOK
        tracks_dir: Directory containing {child_id}_joint_attention.npz
        video_duration_sec: Optional video duration for computing coverage ratio.

    Returns:
        Dict with features:
            # Pointing features (adult-side)
            ja_point_segment_count: Number of pointing segments detected
            ja_total_point_duration_sec: Sum of all segment durations
            ja_mean_point_stability: Mean stability across segments (0-1)
            ja_point_coverage_ratio: point_duration / video_duration

            # Audio features (adult-side)
            ja_attention_call_count: Number of CALL_ATTENTION events
            ja_look_phrase_count: Number of LOOK events
            ja_audio_cue_count: Total audio attention cues
            ja_audio_cue_mean_confidence: Mean confidence of audio detections

            # Response features (child behavior)
            ja_attention_response_rate: Fraction of all audio cues child responded to
            ja_attention_response_latency_mean: Mean latency of attention responses (sec)
            ja_name_response_rate: Fraction of CALL_ATTENTION cues child responded to
            ja_name_response_latency_mean: Mean latency of name responses (sec)
            ja_follow_point_rate: Fraction of point segments child responded to (any movement)
            ja_follow_point_correct_dir_rate: Fraction of responses in correct direction
            ja_follow_point_latency_mean: Mean latency of point following (sec)
            ja_post_point_head_change_mean: Mean head turn magnitude after points

            # Paper-aligned features (autism identification study)
            ja_response_latency_s: First response latency to name call (sec)
            ja_parent_attempts: Total parent calling attempts (CALL_ATTENTION + LOOK)
            ja_orient_success: Binary (1.0 if child oriented at least once, 0.0 otherwise)
            ja_orient_latency_s: Latency to first successful orientation (sec, relative to cue)
            ja_first_orient_time_s: Absolute timestamp of first orientation (sec from video start)
    """
    child_id_str = str(child_id)
    features: Dict[str, float] = {}

    # Load child tracks if available
    tracks_data = None
    if tracks_dir is not None:
        tracks_data = _load_tracks(tracks_dir, child_id_str, "joint_attention")

    # === Pointing features (adult-side) ===
    if len(point_events_df) > 0 and "child_id" in point_events_df.columns:
        point_child_df = point_events_df[
            (point_events_df["child_id"].astype(str) == child_id_str)
            & (point_events_df["task_type"] == "joint_attention")
        ]
    else:
        point_child_df = pd.DataFrame()

    if len(point_child_df) == 0:
        features.update({
            "ja_point_segment_count": 0.0,
            "ja_total_point_duration_sec": 0.0,
            "ja_mean_point_stability": float("nan"),
            "ja_point_coverage_ratio": float("nan"),
        })
    else:
        segment_count = float(len(point_child_df))
        durations = point_child_df["t_end"] - point_child_df["t_start"]
        total_duration = float(durations.sum())
        mean_stability = float(point_child_df["stability"].mean())

        if video_duration_sec is not None and video_duration_sec > 0:
            coverage_ratio = total_duration / video_duration_sec
        else:
            coverage_ratio = float("nan")

        features.update({
            "ja_point_segment_count": segment_count,
            "ja_total_point_duration_sec": total_duration,
            "ja_mean_point_stability": mean_stability,
            "ja_point_coverage_ratio": coverage_ratio,
        })

    # === Audio features (adult-side) ===
    audio_child_df = None
    if audio_events_df is not None and len(audio_events_df) > 0 and "child_id" in audio_events_df.columns:
        audio_child_df = audio_events_df[
            (audio_events_df["child_id"].astype(str) == child_id_str)
            & (audio_events_df["task_type"] == "joint_attention")
        ]

    if audio_child_df is None or len(audio_child_df) == 0:
        features.update({
            "ja_attention_call_count": 0.0,
            "ja_look_phrase_count": 0.0,
            "ja_audio_cue_count": 0.0,
            "ja_audio_cue_mean_confidence": float("nan"),
            # Paper-aligned: parent attempts
            "ja_parent_attempts": 0.0,
        })
        audio_child_df = pd.DataFrame(columns=["event_type", "t_start"])  # Empty with correct columns for response analysis
    else:
        call_attention = audio_child_df[audio_child_df["event_type"] == "CALL_ATTENTION"]
        look_phrase = audio_child_df[audio_child_df["event_type"] == "LOOK"]

        attention_call_count = float(len(call_attention))
        look_phrase_count = float(len(look_phrase))
        audio_cue_count = attention_call_count + look_phrase_count
        mean_confidence = float(audio_child_df["confidence"].mean())

        features.update({
            "ja_attention_call_count": attention_call_count,
            "ja_look_phrase_count": look_phrase_count,
            "ja_audio_cue_count": audio_cue_count,
            "ja_audio_cue_mean_confidence": mean_confidence,
            # Paper-aligned: parent attempts (same as audio_cue_count)
            "ja_parent_attempts": audio_cue_count,
        })

    # === Response features (child behavior) ===
    # These require tracks data
    if tracks_data is None:
        features.update({
            "ja_attention_response_rate": float("nan"),
            "ja_attention_response_latency_mean": float("nan"),
            "ja_name_response_rate": float("nan"),
            "ja_name_response_latency_mean": float("nan"),
            "ja_follow_point_rate": float("nan"),
            "ja_follow_point_correct_dir_rate": float("nan"),
            "ja_follow_point_latency_mean": float("nan"),
            "ja_post_point_head_change_mean": float("nan"),
            # Paper-aligned features
            "ja_response_latency_s": float("nan"),
            "ja_orient_success": float("nan"),
            "ja_orient_latency_s": float("nan"),
            "ja_first_orient_time_s": float("nan"),
        })
    else:
        # Attention response analysis (all audio cues: CALL_ATTENTION + LOOK)
        all_cue_times = _deduplicate_cues(audio_child_df["t_start"].to_numpy(dtype=float))

        attention_responses = []
        attention_latencies = []
        first_orient_latency = float("nan")  # Latency relative to first successful cue
        first_orient_time = float("nan")  # Absolute timestamp of first orient
        for cue_time in all_cue_times:
            responded, latency, _ = _detect_head_turn_after_cue(tracks_data, cue_time)
            attention_responses.append(responded)
            if responded:
                attention_latencies.append(latency)
                # Capture first successful orient (paper-aligned)
                if np.isnan(first_orient_latency):
                    first_orient_latency = latency
                    first_orient_time = cue_time + latency

        if len(attention_responses) > 0:
            attention_response_rate = float(np.mean(attention_responses))
        else:
            attention_response_rate = float("nan")

        if len(attention_latencies) > 0:
            attention_response_latency_mean = float(np.mean(attention_latencies))
        else:
            attention_response_latency_mean = float("nan")

        # Paper-aligned: binary orient success (1.0 if any response, 0.0 otherwise)
        orient_success = 1.0 if len(attention_latencies) > 0 else 0.0

        # Name response analysis (CALL_ATTENTION only, with larger dedup gap)
        call_attention_df = audio_child_df[audio_child_df["event_type"] == "CALL_ATTENTION"]
        name_cue_times = _deduplicate_cues(
            call_attention_df["t_start"].to_numpy(dtype=float),
            min_gap_sec=2.0  # Larger gap for name calls (repeated calls are common)
        )

        name_responses = []
        name_latencies = []
        first_name_response_latency = float("nan")  # Paper-aligned: first response
        for cue_time in name_cue_times:
            responded, latency, _ = _detect_head_turn_after_cue(tracks_data, cue_time)
            name_responses.append(responded)
            if responded:
                name_latencies.append(latency)
                # Capture first successful response latency (paper-aligned)
                if np.isnan(first_name_response_latency):
                    first_name_response_latency = latency

        if len(name_responses) > 0:
            name_response_rate = float(np.mean(name_responses))
        else:
            name_response_rate = float("nan")

        if len(name_latencies) > 0:
            name_response_latency_mean = float(np.mean(name_latencies))
        else:
            name_response_latency_mean = float("nan")

        # Point following analysis with directional matching
        # Use angle_deg to check if child turned in the correct direction
        point_responses = []
        point_correct_direction = []
        point_latencies = []
        point_head_changes = []

        for _, row in point_child_df.iterrows():
            point_time = float(row["t_start"])
            point_angle = float(row.get("angle_deg", 90.0))  # Default to center if missing

            responded, latency, signed_delta = _detect_head_turn_after_cue(tracks_data, point_time)
            point_responses.append(responded)

            if responded:
                point_latencies.append(latency)

                # Directional matching: check if head moved toward point direction
                # Point angle convention: 0=right, 180=left (in image coords)
                # Head signal: positive = head moved right, negative = left
                # If point is to right (angle < 90), expect positive signed_delta
                # If point is to left (angle > 90), expect negative signed_delta
                expected_direction = 1.0 if point_angle < 90 else -1.0
                actual_direction = 1.0 if signed_delta > 0 else -1.0
                correct_dir = (expected_direction == actual_direction)
                point_correct_direction.append(correct_dir)

            head_change_abs, _ = _compute_post_cue_head_change(tracks_data, point_time)
            if np.isfinite(head_change_abs):
                point_head_changes.append(head_change_abs)

        if len(point_responses) > 0:
            follow_point_rate = float(np.mean(point_responses))
        else:
            follow_point_rate = float("nan")

        if len(point_correct_direction) > 0:
            follow_point_correct_dir_rate = float(np.mean(point_correct_direction))
        else:
            follow_point_correct_dir_rate = float("nan")

        if len(point_latencies) > 0:
            follow_point_latency_mean = float(np.mean(point_latencies))
        else:
            follow_point_latency_mean = float("nan")

        if len(point_head_changes) > 0:
            post_point_head_change_mean = float(np.mean(point_head_changes))
        else:
            post_point_head_change_mean = float("nan")

        features.update({
            "ja_attention_response_rate": attention_response_rate,
            "ja_attention_response_latency_mean": attention_response_latency_mean,
            "ja_name_response_rate": name_response_rate,
            "ja_name_response_latency_mean": name_response_latency_mean,
            "ja_follow_point_rate": follow_point_rate,
            "ja_follow_point_correct_dir_rate": follow_point_correct_dir_rate,
            "ja_follow_point_latency_mean": follow_point_latency_mean,
            "ja_post_point_head_change_mean": post_point_head_change_mean,
            # Paper-aligned features (autism identification study)
            "ja_response_latency_s": first_name_response_latency,
            "ja_orient_success": orient_success,
            "ja_orient_latency_s": first_orient_latency,
            "ja_first_orient_time_s": first_orient_time,
        })

    return features


def extract_response_outcomes(
    child_id: str,
    point_events_df: pd.DataFrame,
    audio_events_df: Optional[pd.DataFrame] = None,
    tracks_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Extract per-event response outcomes for guided review.

    This function uses the SAME detection logic as extract_joint_attention_features
    to ensure consistency between ML features and guided review display.

    Args:
        child_id: Child identifier
        point_events_df: DataFrame from pointing.py output
        audio_events_df: DataFrame from audio_events.py output
        tracks_dir: Directory containing {child_id}_joint_attention.npz

    Returns:
        List of dicts, each containing:
            - cue_type: "audio" or "point"
            - event_type: "CALL_ATTENTION", "LOOK", or "POINT"
            - t_start_sec: Cue start time
            - t_end_sec: Cue end time (for points)
            - matched_phrase: For audio events
            - point_angle_deg: For pointing events
            - responded: bool
            - latency_ms: Response latency (None if no response)
            - status: "observed", "delayed", "not_observed", or "uncertain"
    """
    child_id_str = str(child_id)
    outcomes: List[Dict[str, Any]] = []

    # Load child tracks if available
    tracks_data = None
    if tracks_dir is not None:
        tracks_data = _load_tracks(tracks_dir, child_id_str, "joint_attention")

    if tracks_data is None:
        # No tracking data - mark all as uncertain
        if audio_events_df is not None and len(audio_events_df) > 0:
            audio_child_df = audio_events_df[
                (audio_events_df["child_id"].astype(str) == child_id_str)
                & (audio_events_df["task_type"] == "joint_attention")
            ]
            for _, row in audio_child_df.iterrows():
                outcomes.append({
                    "cue_type": "audio",
                    "event_type": row["event_type"],
                    "t_start_sec": float(row["t_start"]),
                    "t_end_sec": float(row["t_end"]),
                    "matched_phrase": row.get("matched_phrase"),
                    "point_angle_deg": None,
                    "responded": False,
                    "latency_ms": None,
                    "status": "uncertain",
                })

        if len(point_events_df) > 0 and "child_id" in point_events_df.columns:
            point_child_df = point_events_df[
                (point_events_df["child_id"].astype(str) == child_id_str)
                & (point_events_df["task_type"] == "joint_attention")
            ]
            for _, row in point_child_df.iterrows():
                outcomes.append({
                    "cue_type": "point",
                    "event_type": "POINT",
                    "t_start_sec": float(row["t_start"]),
                    "t_end_sec": float(row["t_end"]),
                    "matched_phrase": None,
                    "point_angle_deg": float(row.get("point_angle_deg", row.get("angle_deg", 90.0))),
                    "responded": False,
                    "latency_ms": None,
                    "status": "uncertain",
                })

        return outcomes

    # Delayed threshold (same as used in guided review)
    DELAYED_THRESHOLD_MS = 1500.0

    # Process audio events
    if audio_events_df is not None and len(audio_events_df) > 0 and "child_id" in audio_events_df.columns:
        audio_child_df = audio_events_df[
            (audio_events_df["child_id"].astype(str) == child_id_str)
            & (audio_events_df["task_type"] == "joint_attention")
        ]

        for _, row in audio_child_df.iterrows():
            cue_time = float(row["t_start"])
            responded, latency, _ = _detect_head_turn_after_cue(tracks_data, cue_time)

            latency_ms = latency * 1000 if responded and not np.isnan(latency) else None

            if not responded:
                status = "not_observed"
            elif latency_ms is not None and latency_ms > DELAYED_THRESHOLD_MS:
                status = "delayed"
            else:
                status = "observed"

            outcomes.append({
                "cue_type": "audio",
                "event_type": row["event_type"],
                "t_start_sec": cue_time,
                "t_end_sec": float(row["t_end"]),
                "matched_phrase": row.get("matched_phrase"),
                "point_angle_deg": None,
                "responded": responded,
                "latency_ms": latency_ms,
                "status": status,
            })

    # Process pointing events
    if len(point_events_df) > 0 and "child_id" in point_events_df.columns:
        point_child_df = point_events_df[
            (point_events_df["child_id"].astype(str) == child_id_str)
            & (point_events_df["task_type"] == "joint_attention")
        ]

        for _, row in point_child_df.iterrows():
            point_time = float(row["t_start"])
            point_angle = float(row.get("point_angle_deg", row.get("angle_deg", 90.0)))

            responded, latency, _ = _detect_head_turn_after_cue(tracks_data, point_time)

            latency_ms = latency * 1000 if responded and not np.isnan(latency) else None

            if not responded:
                status = "not_observed"
            elif latency_ms is not None and latency_ms > DELAYED_THRESHOLD_MS:
                status = "delayed"
            else:
                status = "observed"

            outcomes.append({
                "cue_type": "point",
                "event_type": "POINT",
                "t_start_sec": point_time,
                "t_end_sec": float(row["t_end"]),
                "matched_phrase": None,
                "point_angle_deg": point_angle,
                "responded": responded,
                "latency_ms": latency_ms,
                "status": status,
            })

    # Sort by time
    outcomes.sort(key=lambda x: x["t_start_sec"])

    return outcomes
