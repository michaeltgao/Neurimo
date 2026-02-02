from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Pose landmark indices
POSE_NOSE = 0

# Response detection config
RESPONSE_WINDOW_SEC = 3.0  # How long after cue to look for response
HEAD_TURN_THRESHOLD = 0.03  # Minimum nose_x change to count as response (normalized units)


def _load_tracks(tracks_dir: Path, child_id: str, task: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load tracks and return (t_sec, nose_x) or None if missing."""
    npz_path = tracks_dir / f"{child_id}_{task}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=False)
    t_sec = data["t_sec"].astype(float)
    pose = data["pose"].astype(float)
    if pose.shape[0] == 0:
        return None
    nose_x = pose[:, POSE_NOSE, 0]
    return t_sec, nose_x


def _detect_head_turn_after_cue(
    t_sec: np.ndarray,
    nose_x: np.ndarray,
    cue_time: float,
    window_sec: float = RESPONSE_WINDOW_SEC,
    threshold: float = HEAD_TURN_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Detect if child turned head after a cue.

    Returns:
        (responded, latency_sec) - latency is NaN if no response
    """
    # Find samples in response window
    mask = (t_sec >= cue_time) & (t_sec <= cue_time + window_sec)
    if mask.sum() < 2:
        return False, float("nan")

    window_t = t_sec[mask]
    window_nose = nose_x[mask]

    # Check for valid data
    if not np.isfinite(window_nose).any():
        return False, float("nan")

    # Interpolate NaNs
    valid = np.isfinite(window_nose)
    if valid.sum() < 2:
        return False, float("nan")

    window_nose_interp = np.interp(
        window_t, window_t[valid], window_nose[valid]
    )

    # Compute cumulative head movement from start
    baseline = window_nose_interp[0]
    deltas = np.abs(window_nose_interp - baseline)

    # Find first time delta exceeds threshold
    response_idx = np.where(deltas >= threshold)[0]
    if len(response_idx) == 0:
        return False, float("nan")

    first_response = response_idx[0]
    latency = max(0.0, window_t[first_response] - cue_time)
    return True, float(latency)


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
    t_sec: np.ndarray,
    nose_x: np.ndarray,
    cue_time: float,
    window_sec: float = RESPONSE_WINDOW_SEC,
) -> float:
    """Compute magnitude of head turn after cue (max absolute delta from baseline)."""
    mask = (t_sec >= cue_time) & (t_sec <= cue_time + window_sec)
    if mask.sum() < 2:
        return float("nan")

    window_nose = nose_x[mask]
    valid = np.isfinite(window_nose)
    if valid.sum() < 2:
        return float("nan")

    window_nose_clean = window_nose[valid]
    baseline = window_nose_clean[0]
    max_delta = float(np.max(np.abs(window_nose_clean - baseline)))
    return max_delta


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
            ja_follow_point_rate: Fraction of point segments child followed
            ja_follow_point_latency_mean: Mean latency of point following (sec)
            ja_post_point_head_change_mean: Mean head turn magnitude after points
    """
    child_id_str = str(child_id)
    features: Dict[str, float] = {}

    # Load child tracks if available
    tracks_data = None
    if tracks_dir is not None:
        tracks_data = _load_tracks(tracks_dir, child_id_str, "joint_attention")

    # === Pointing features (adult-side) ===
    point_child_df = point_events_df[
        (point_events_df["child_id"].astype(str) == child_id_str)
        & (point_events_df["task_type"] == "joint_attention")
    ]

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
    if audio_events_df is not None and len(audio_events_df) > 0:
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
            "ja_follow_point_latency_mean": float("nan"),
            "ja_post_point_head_change_mean": float("nan"),
        })
    else:
        t_sec, nose_x = tracks_data

        # Attention response analysis (all audio cues: CALL_ATTENTION + LOOK)
        all_cue_times = _deduplicate_cues(audio_child_df["t_start"].to_numpy(dtype=float))

        attention_responses = []
        attention_latencies = []
        for cue_time in all_cue_times:
            responded, latency = _detect_head_turn_after_cue(t_sec, nose_x, cue_time)
            attention_responses.append(responded)
            if responded:
                attention_latencies.append(latency)

        if len(attention_responses) > 0:
            attention_response_rate = float(np.mean(attention_responses))
        else:
            attention_response_rate = float("nan")

        if len(attention_latencies) > 0:
            attention_response_latency_mean = float(np.mean(attention_latencies))
        else:
            attention_response_latency_mean = float("nan")

        # Name response analysis (CALL_ATTENTION only)
        call_attention_df = audio_child_df[audio_child_df["event_type"] == "CALL_ATTENTION"]
        name_cue_times = _deduplicate_cues(call_attention_df["t_start"].to_numpy(dtype=float))

        name_responses = []
        name_latencies = []
        for cue_time in name_cue_times:
            responded, latency = _detect_head_turn_after_cue(t_sec, nose_x, cue_time)
            name_responses.append(responded)
            if responded:
                name_latencies.append(latency)

        if len(name_responses) > 0:
            name_response_rate = float(np.mean(name_responses))
        else:
            name_response_rate = float("nan")

        if len(name_latencies) > 0:
            name_response_latency_mean = float(np.mean(name_latencies))
        else:
            name_response_latency_mean = float("nan")

        # Point following analysis (use smaller gap since pointing segments are discrete events)
        point_cue_times = _deduplicate_cues(point_child_df["t_start"].to_numpy(dtype=float), min_gap_sec=0.5)

        point_responses = []
        point_latencies = []
        point_head_changes = []
        for point_time in point_cue_times:
            responded, latency = _detect_head_turn_after_cue(t_sec, nose_x, point_time)
            point_responses.append(responded)
            if responded:
                point_latencies.append(latency)

            head_change = _compute_post_cue_head_change(t_sec, nose_x, point_time)
            if np.isfinite(head_change):
                point_head_changes.append(head_change)

        if len(point_responses) > 0:
            follow_point_rate = float(np.mean(point_responses))
        else:
            follow_point_rate = float("nan")

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
            "ja_follow_point_latency_mean": follow_point_latency_mean,
            "ja_post_point_head_change_mean": post_point_head_change_mean,
        })

    return features
