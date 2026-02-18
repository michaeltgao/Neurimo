"""
Free play feature extraction for Neurimo ML pipeline.

Extracts behavioral features from free play video analysis:
- Original 10 high-level ratio features (backward compatible)
- 22 new temporal/behavioral metrics for enhanced analysis

Features are prefixed with 'fp_' for free play task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_float(val, default: float = float("nan")) -> float:
    """Convert value to float, handling empty strings and None."""
    if pd.isna(val) or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _clamp01(x: float) -> float:
    """Clamp value to [0, 1] range, preserving NaN."""
    if pd.isna(x):
        return x
    return max(0.0, min(1.0, x))


def _extract_temporal_features_from_events(
    child_id: str,
    fp_events_df: pd.DataFrame,
    duration_sec: float,
) -> Dict[str, float]:
    """
    Extract temporal features from event-level data.

    New features based on event timing and patterns:
    - Activity state transitions
    - Periodic motion bout patterns
    - Proximity dynamics
    - Hand activity patterns
    - Engagement transitions

    Returns:
        Dict with temporal features. Rules for NaN vs 0:
        - NaN: Feature could not be computed (missing data/columns, invalid duration)
        - 0: Feature was computed but no events of that type were detected
    """
    child_id_str = str(child_id)

    # Default NaN values for all temporal features
    temporal_defaults = {
        "fp_activity_state_transition_rate": float("nan"),
        "fp_mean_activity_bout_duration": float("nan"),
        "fp_activity_bout_variability": float("nan"),
        "fp_periodic_first_onset_sec": float("nan"),
        "fp_periodic_total_bout_count": float("nan"),
        "fp_periodic_mean_bout_duration": float("nan"),
        "fp_periodic_inter_bout_interval_mean": float("nan"),
        "fp_hand_active_burst_rate": float("nan"),
        "fp_hand_to_face_count": float("nan"),
        "fp_hands_near_rhythmic_count": float("nan"),
        "fp_engagement_transition_rate": float("nan"),
        "fp_event_confidence_mean": float("nan"),
    }

    # Validate required columns exist
    required_cols = {"child_id", "task_type", "event_type", "t_start", "t_end"}
    if not required_cols.issubset(fp_events_df.columns):
        return temporal_defaults

    # Validate duration is finite and positive
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        return temporal_defaults

    child_events = fp_events_df[
        (fp_events_df["child_id"].astype(str) == child_id_str)
        & (fp_events_df["task_type"] == "free_play")
    ]

    if len(child_events) == 0:
        return temporal_defaults

    features: Dict[str, float] = {}

    # Safely convert event_type to string for filtering
    event_type_str = child_events["event_type"].astype(str)

    # Activity state transitions (count HIGH activity events)
    activity_events = child_events[event_type_str.str.startswith("ACTIVITY_")]
    if len(activity_events) > 0:
        # Transition rate = number of activity bursts per minute
        n_bursts = len(activity_events)
        features["fp_activity_state_transition_rate"] = n_bursts / (duration_sec / 60.0)

        # Mean bout duration
        durations = np.asarray((activity_events["t_end"] - activity_events["t_start"]).values)
        if len(durations) > 0:
            features["fp_mean_activity_bout_duration"] = float(np.mean(durations))
            if len(durations) > 1:
                mean_dur = float(np.mean(durations))
                if mean_dur > 0:
                    features["fp_activity_bout_variability"] = float(np.std(durations) / mean_dur)
    else:
        features["fp_activity_state_transition_rate"] = 0.0

    # Periodic motion temporal features
    periodic_events = child_events[event_type_str == "PERIODIC_MOTION"]
    if len(periodic_events) > 0:
        sorted_periodic = periodic_events.sort_values("t_start")
        features["fp_periodic_first_onset_sec"] = float(sorted_periodic.iloc[0]["t_start"])
        features["fp_periodic_total_bout_count"] = float(len(periodic_events))

        durations = np.asarray((sorted_periodic["t_end"] - sorted_periodic["t_start"]).values)
        features["fp_periodic_mean_bout_duration"] = float(np.mean(durations))

        # Inter-bout intervals
        if len(sorted_periodic) > 1:
            starts = np.asarray(sorted_periodic["t_start"].values)
            ends = np.asarray(sorted_periodic["t_end"].values[:-1])
            intervals = starts[1:] - ends
            if len(intervals) > 0:
                features["fp_periodic_inter_bout_interval_mean"] = float(np.mean(intervals))
    else:
        features["fp_periodic_total_bout_count"] = 0.0

    # Hand activity patterns
    hand_active_events = child_events[event_type_str == "HAND_ACTIVE"]
    if len(hand_active_events) > 0:
        features["fp_hand_active_burst_rate"] = len(hand_active_events) / (duration_sec / 60.0)
    else:
        features["fp_hand_active_burst_rate"] = 0.0

    # Hand to face count
    face_events = child_events[event_type_str == "HAND_TO_FACE"]
    features["fp_hand_to_face_count"] = float(len(face_events))

    # Hands near (rhythmic) count
    near_events = child_events[event_type_str == "HANDS_NEAR"]
    features["fp_hands_near_rhythmic_count"] = float(len(near_events))

    # Engagement transitions (proximity + parent present changes)
    proximity_events = child_events[event_type_str == "CLOSE_PROXIMITY"]
    parent_events = child_events[event_type_str == "PARENT_PRESENT"]
    engagement_count = len(proximity_events) + len(parent_events)
    if engagement_count > 0:
        features["fp_engagement_transition_rate"] = engagement_count / (duration_sec / 60.0)
    else:
        features["fp_engagement_transition_rate"] = 0.0

    # Mean confidence across all events
    if len(child_events) > 0 and "confidence" in child_events.columns:
        features["fp_event_confidence_mean"] = float(child_events["confidence"].mean())

    # Fill remaining defaults
    for key, default in temporal_defaults.items():
        if key not in features:
            features[key] = default

    return features


def _extract_proximity_dynamics_features(
    child_id: str,
    fp_events_df: pd.DataFrame,
    duration_sec: float,
) -> Dict[str, float]:
    """
    Extract proximity dynamics features from events.

    Measures approach/gap patterns and sustained proximity.

    Returns:
        Dict with proximity features. Rules for NaN vs 0:
        - NaN: Feature could not be computed (missing data/columns, invalid duration)
        - 0: Feature was computed but no proximity events were detected
    """
    child_id_str = str(child_id)

    defaults = {
        "fp_proximity_approach_count": float("nan"),
        "fp_proximity_gap_count": float("nan"),
        "fp_proximity_maintain_duration": float("nan"),
        "fp_proximity_first_approach_latency": float("nan"),
    }

    if fp_events_df is None or len(fp_events_df) == 0:
        return defaults

    # Validate required columns exist
    required_cols = {"child_id", "task_type", "event_type", "t_start", "t_end"}
    if not required_cols.issubset(fp_events_df.columns):
        return defaults

    # Validate duration is finite and positive
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        return defaults

    child_events = fp_events_df[
        (fp_events_df["child_id"].astype(str) == child_id_str)
        & (fp_events_df["task_type"] == "free_play")
    ]

    if len(child_events) == 0:
        return defaults

    # Safely convert event_type to string for filtering
    event_type_str = child_events["event_type"].astype(str)

    # Proximity events
    proximity_events = child_events[event_type_str == "CLOSE_PROXIMITY"]

    if len(proximity_events) > 0:
        sorted_prox = proximity_events.sort_values("t_start")

        # First approach latency
        defaults["fp_proximity_first_approach_latency"] = float(sorted_prox.iloc[0]["t_start"])

        # Total maintained proximity duration
        t_end = np.asarray(sorted_prox["t_end"].values)
        t_start = np.asarray(sorted_prox["t_start"].values)
        total_prox = (t_end - t_start).sum()
        defaults["fp_proximity_maintain_duration"] = float(total_prox)

        # Approach count = number of proximity segments
        defaults["fp_proximity_approach_count"] = float(len(proximity_events))

        # Gap count: number of significant breaks (>1 sec) between proximity segments
        if len(sorted_prox) > 1:
            ends = t_end[:-1]
            starts = t_start[1:]
            gaps = starts - ends
            defaults["fp_proximity_gap_count"] = float(np.sum(gaps > 1.0))
        else:
            defaults["fp_proximity_gap_count"] = 0.0
    else:
        defaults["fp_proximity_approach_count"] = 0.0
        defaults["fp_proximity_gap_count"] = 0.0
        defaults["fp_proximity_maintain_duration"] = 0.0

    return defaults


def _extract_engagement_features(
    child_id: str,
    fp_summary_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Extract engagement state features from summary data.
    """
    child_id_str = str(child_id)

    defaults = {
        "fp_engaged_time_frac": float("nan"),
        "fp_disengaged_time_frac": float("nan"),
    }

    # Get from summary
    if len(fp_summary_df) > 0 and "child_id" in fp_summary_df.columns:
        child_df = fp_summary_df[
            (fp_summary_df["child_id"].astype(str) == child_id_str)
            & (fp_summary_df["task_type"] == "free_play")
        ]
    else:
        child_df = pd.DataFrame()

    if len(child_df) == 0:
        return defaults

    row = child_df.iloc[0]

    # Engaged = close proximity or actively with adult
    engaged = _safe_float(row.get("engaged_with_adult_time_frac"))
    close_prox = _safe_float(row.get("close_proximity_time_frac", 0))

    if not np.isnan(engaged):
        defaults["fp_engaged_time_frac"] = _clamp01(engaged)
    elif not np.isnan(close_prox):
        defaults["fp_engaged_time_frac"] = _clamp01(close_prox)

    # Disengaged = not with adult + freeze time
    disengaged = _safe_float(row.get("disengaged_with_adult_time_frac"))
    freeze = _safe_float(row.get("freeze_time_frac", 0))

    if not np.isnan(disengaged):
        defaults["fp_disengaged_time_frac"] = _clamp01(disengaged)
    elif not np.isnan(freeze):
        defaults["fp_disengaged_time_frac"] = _clamp01(freeze)

    return defaults


def _extract_hand_coordination_features(
    child_id: str,
    fp_summary_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Extract hand coordination features from summary.
    """
    child_id_str = str(child_id)

    defaults = {
        "fp_hand_bilateral_ratio": float("nan"),
    }

    if len(fp_summary_df) > 0 and "child_id" in fp_summary_df.columns:
        child_df = fp_summary_df[
            (fp_summary_df["child_id"].astype(str) == child_id_str)
            & (fp_summary_df["task_type"] == "free_play")
        ]
    else:
        child_df = pd.DataFrame()

    if len(child_df) == 0:
        return defaults

    row = child_df.iloc[0]

    # Bilateral ratio from left/right active fractions
    left_frac = _safe_float(row.get("left_hand_active_frac", 0))
    right_frac = _safe_float(row.get("right_hand_active_frac", 0))

    if not np.isnan(left_frac) and not np.isnan(right_frac):
        total = left_frac + right_frac
        if total > 0:
            # Bilateral = how balanced are the hands (1.0 = perfectly balanced)
            min_frac = min(left_frac, right_frac)
            max_frac = max(left_frac, right_frac)
            if max_frac > 0:
                defaults["fp_hand_bilateral_ratio"] = min_frac / max_frac
            else:
                defaults["fp_hand_bilateral_ratio"] = 0.0

    return defaults


def _extract_social_response_features(
    child_id: str,
    fp_events_df: Optional[pd.DataFrame],
) -> Dict[str, float]:
    """
    Extract social response features.

    Measures response rates and latencies to social bids.

    Returns:
        Dict with social response features. Rules for NaN vs 0:
        - NaN: Feature could not be computed (missing data/columns)
        - 0: Feature was computed but no responses detected
    """
    defaults = {
        "fp_social_bid_response_rate": float("nan"),
        "fp_social_bid_response_latency_mean": float("nan"),
    }

    if fp_events_df is None or len(fp_events_df) == 0:
        return defaults

    # Validate required columns exist
    required_cols = {"child_id", "task_type", "event_type", "t_start", "t_end"}
    if not required_cols.issubset(fp_events_df.columns):
        return defaults

    child_id_str = str(child_id)
    child_events = fp_events_df[
        (fp_events_df["child_id"].astype(str) == child_id_str)
        & (fp_events_df["task_type"] == "free_play")
    ]

    if len(child_events) == 0:
        return defaults

    # Safely convert event_type to string for filtering
    event_type_str = child_events["event_type"].astype(str)

    # Parent present events as potential social bids
    parent_events = child_events[event_type_str == "PARENT_PRESENT"]

    # Child responses: close proximity or hand activity
    response_events = child_events[
        event_type_str.isin(["CLOSE_PROXIMITY", "HAND_ACTIVE"])
    ]

    if len(parent_events) > 0 and len(response_events) > 0:
        # Sort response events by time to ensure proper ordering
        response_events = response_events.sort_values("t_start")

        parent_starts = parent_events["t_start"].values
        parent_ends = parent_events["t_end"].values

        response_count = 0
        latencies = []
        used_response_indices: set[int] = set()

        for ps, pe in zip(parent_starts, parent_ends):
            # Find responses within 3 seconds of parent segment start
            window_end = min(ps + 3.0, pe)
            responses_in_window = response_events[
                (response_events["t_start"] >= ps)
                & (response_events["t_start"] <= window_end)
                & (~response_events.index.isin(used_response_indices))
            ]
            if len(responses_in_window) > 0:
                response_count += 1
                # Use min() to get the earliest response time (already sorted, but explicit)
                first_response_time = float(responses_in_window["t_start"].min())
                latencies.append(first_response_time - ps)
                # Mark this response as used to prevent double-counting
                first_response_idx = responses_in_window["t_start"].idxmin()
                used_response_indices.add(first_response_idx)

        if len(parent_events) > 0:
            defaults["fp_social_bid_response_rate"] = response_count / len(parent_events)

        if len(latencies) > 0:
            defaults["fp_social_bid_response_latency_mean"] = float(np.mean(latencies))

    return defaults


def _get_new_feature_defaults() -> Dict[str, float]:
    """Get default NaN values for all new features."""
    return {
        # Temporal features (12)
        "fp_activity_state_transition_rate": float("nan"),
        "fp_mean_activity_bout_duration": float("nan"),
        "fp_activity_bout_variability": float("nan"),
        "fp_periodic_first_onset_sec": float("nan"),
        "fp_periodic_total_bout_count": float("nan"),
        "fp_periodic_mean_bout_duration": float("nan"),
        "fp_periodic_inter_bout_interval_mean": float("nan"),
        "fp_hand_active_burst_rate": float("nan"),
        "fp_hand_to_face_count": float("nan"),
        "fp_hands_near_rhythmic_count": float("nan"),
        "fp_engagement_transition_rate": float("nan"),
        "fp_event_confidence_mean": float("nan"),
        # Proximity features (4)
        "fp_proximity_approach_count": float("nan"),
        "fp_proximity_gap_count": float("nan"),
        "fp_proximity_maintain_duration": float("nan"),
        "fp_proximity_first_approach_latency": float("nan"),
        # Engagement features (2)
        "fp_engaged_time_frac": float("nan"),
        "fp_disengaged_time_frac": float("nan"),
        # Coordination features (1)
        "fp_hand_bilateral_ratio": float("nan"),
        # Social response features (2)
        "fp_social_bid_response_rate": float("nan"),
        "fp_social_bid_response_latency_mean": float("nan"),
    }


def extract_free_play_features(
    child_id: str,
    fp_summary_df: pd.DataFrame,
    fp_events_df: Optional[pd.DataFrame] = None,
    tracks_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Extract comprehensive free play features.

    Args:
        child_id: Child identifier
        fp_summary_df: DataFrame from free_play_events.py output (free_play_summary.csv)
        fp_events_df: Optional DataFrame from free_play_events.csv for temporal features
        tracks_dir: Optional path to tracks directory for additional analysis

    Returns:
        Dict with features (all prefixed with fp_):

        ORIGINAL FEATURES (10, backward compatible):
            fp_adult_present_ratio: Ratio of frames with adult present
            fp_adult_hand_active_time_frac: Time fraction with adult hand active
            fp_adult_hand_mean_activity: Mean adult hand activity level
            fp_freeze_time_frac: Time fraction child is frozen (disengaged)
            fp_hand_to_face_time_frac: Time fraction hand is near face
            fp_repetitive_motion_time_frac: Time fraction with repetitive motion
            fp_engaged_with_adult_time_frac: Time fraction engaged with adult
            fp_disengaged_with_adult_time_frac: Time fraction disengaged from adult
            fp_hands_near_torso_time_frac: Time fraction with hands near torso
            fp_repetitive_motion_freq_hz: Dominant frequency of repetitive motion

        NEW TEMPORAL FEATURES (12):
            fp_activity_state_transition_rate: Activity bursts per minute
            fp_mean_activity_bout_duration: Mean duration of activity bouts
            fp_activity_bout_variability: CV of bout durations
            fp_periodic_first_onset_sec: Time to first periodic motion
            fp_periodic_total_bout_count: Number of periodic motion bouts
            fp_periodic_mean_bout_duration: Mean duration of periodic bouts
            fp_periodic_inter_bout_interval_mean: Mean time between periodic bouts
            fp_hand_active_burst_rate: Hand activity bursts per minute
            fp_hand_to_face_count: Number of hand-to-face events
            fp_hands_near_rhythmic_count: Number of rhythmic hand movements
            fp_engagement_transition_rate: Engagement transitions per minute
            fp_event_confidence_mean: Mean confidence of detected events

        NEW PROXIMITY FEATURES (4):
            fp_proximity_approach_count: Number of approach events
            fp_proximity_gap_count: Number of gaps (>1s) between proximity segments
            fp_proximity_maintain_duration: Total time in proximity
            fp_proximity_first_approach_latency: Time to first approach

        NEW ENGAGEMENT FEATURES (2):
            fp_engaged_time_frac: Combined engaged time fraction
            fp_disengaged_time_frac: Combined disengaged time fraction

        NEW COORDINATION FEATURES (1):
            fp_hand_bilateral_ratio: Balance of left/right hand activity

        NEW SOCIAL RESPONSE FEATURES (2):
            fp_social_bid_response_rate: Response rate to parent bids
            fp_social_bid_response_latency_mean: Mean response latency
    """
    child_id_str = str(child_id)

    # Default features for missing data
    default_features = {
        # Original 10 features
        "fp_adult_present_ratio": float("nan"),
        "fp_adult_hand_active_time_frac": float("nan"),
        "fp_adult_hand_mean_activity": float("nan"),
        "fp_freeze_time_frac": float("nan"),
        "fp_hand_to_face_time_frac": float("nan"),
        "fp_repetitive_motion_time_frac": float("nan"),
        "fp_engaged_with_adult_time_frac": float("nan"),
        "fp_disengaged_with_adult_time_frac": float("nan"),
        "fp_hands_near_torso_time_frac": float("nan"),
        "fp_repetitive_motion_freq_hz": float("nan"),
    }

    # Filter to this child + task_type (safety belt for duplicates/multiple visits)
    if len(fp_summary_df) > 0 and "child_id" in fp_summary_df.columns:
        child_df = fp_summary_df[
            (fp_summary_df["child_id"].astype(str) == child_id_str)
            & (fp_summary_df["task_type"] == "free_play")
        ]
    else:
        child_df = pd.DataFrame()

    if len(child_df) == 0:
        # Add new feature defaults
        new_feature_defaults = _get_new_feature_defaults()
        default_features.update(new_feature_defaults)
        return default_features

    # Pick best row if duplicates exist (highest pose_present_ratio = best tracking)
    child_df = child_df.copy()
    if "pose_present_ratio" in child_df.columns:
        child_df["_sort_key"] = child_df["pose_present_ratio"].fillna(-1)
        row = child_df.sort_values("_sort_key", ascending=False).iloc[0]
    else:
        row = child_df.iloc[0]

    # Extract original 10 features
    features = {
        "fp_adult_present_ratio": _clamp01(_safe_float(row.get("adult_present_ratio"))),
        "fp_adult_hand_active_time_frac": _clamp01(_safe_float(row.get("adult_hand_active_time_frac"))),
        "fp_adult_hand_mean_activity": _safe_float(row.get("adult_hand_mean_activity")),
        "fp_freeze_time_frac": _clamp01(_safe_float(row.get("freeze_time_frac"))),
        "fp_hand_to_face_time_frac": _clamp01(_safe_float(row.get("hand_to_face_time_frac"))),
        "fp_repetitive_motion_time_frac": _clamp01(_safe_float(row.get("repetitive_motion_time_frac"))),
        "fp_engaged_with_adult_time_frac": _clamp01(_safe_float(row.get("engaged_with_adult_time_frac"))),
        "fp_disengaged_with_adult_time_frac": _clamp01(_safe_float(row.get("disengaged_with_adult_time_frac"))),
        "fp_hands_near_torso_time_frac": _clamp01(_safe_float(row.get("hands_near_torso_time_frac"))),
        "fp_repetitive_motion_freq_hz": _safe_float(row.get("repetitive_motion_freq_hz")),
    }

    # Get duration for new features
    duration_sec = _safe_float(row.get("duration_sec", 0))

    # Extract new temporal features if events data available
    if fp_events_df is not None and len(fp_events_df) > 0:
        temporal_features = _extract_temporal_features_from_events(
            child_id, fp_events_df, duration_sec
        )
        features.update(temporal_features)

        proximity_features = _extract_proximity_dynamics_features(
            child_id, fp_events_df, duration_sec
        )
        features.update(proximity_features)

        social_features = _extract_social_response_features(
            child_id, fp_events_df
        )
        features.update(social_features)
    else:
        # Add defaults for temporal features
        new_defaults = _get_new_feature_defaults()
        for key in new_defaults:
            if key not in features:
                features[key] = new_defaults[key]

    # Extract engagement features from summary
    engagement_features = _extract_engagement_features(
        child_id, fp_summary_df
    )
    features.update(engagement_features)

    # Extract hand coordination features
    coord_features = _extract_hand_coordination_features(child_id, fp_summary_df)
    features.update(coord_features)

    return features


# Thresholds for flagging behaviors (consistent with clinical assessment)
REPETITIVE_MOTION_THRESHOLD = 0.15  # Flag if >15% of time in repetitive motion
HAND_TO_FACE_THRESHOLD = 0.10       # Flag if >10% of time hand-to-face


def extract_free_play_outcomes(
    child_id: str,
    fp_summary_df: pd.DataFrame,
    fp_events_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    Extract per-event outcomes for guided review consistency.

    Uses the SAME threshold logic as feature extraction to determine which
    behavioral events should be flagged, ensuring consistency between
    dashboard explanations and video replay.

    Args:
        child_id: Child identifier
        fp_summary_df: DataFrame from free_play_events.py output
        fp_events_df: Optional events DataFrame for individual events

    Returns:
        List of dicts with outcome info for each flagged event:
            - task_type: "free_play"
            - event_type: "PERIODIC_MOTION", "HAND_TO_FACE", etc.
            - t_start_sec: Event start time
            - t_end_sec: Event end time
            - flagged: Whether this event is part of a concerning pattern
            - status: "flagged" if concerning, "normal" otherwise
    """
    child_id_str = str(child_id)
    outcomes: List[Dict[str, Any]] = []

    # Determine which behavior types are flagged based on aggregate metrics
    flagged_types: set[str] = set()

    # Get summary for this child
    if len(fp_summary_df) > 0 and "child_id" in fp_summary_df.columns:
        child_df = fp_summary_df[
            (fp_summary_df["child_id"].astype(str) == child_id_str)
            & (fp_summary_df["task_type"] == "free_play")
        ]
    else:
        child_df = pd.DataFrame()

    if len(child_df) > 0:
        row = child_df.iloc[0]

        # Check repetitive motion threshold
        rep_motion_frac = _safe_float(row.get("repetitive_motion_time_frac", 0), default=0)
        if rep_motion_frac > REPETITIVE_MOTION_THRESHOLD:
            flagged_types.add("PERIODIC_MOTION")

        # Check hand-to-face threshold
        htf_frac = _safe_float(row.get("hand_to_face_time_frac", 0), default=0)
        if htf_frac > HAND_TO_FACE_THRESHOLD:
            flagged_types.add("HAND_TO_FACE")

    # If no events data or no flagged types, return empty
    if fp_events_df is None or len(fp_events_df) == 0 or not flagged_types:
        return outcomes

    # Filter events to this child
    child_events = fp_events_df[
        (fp_events_df["child_id"].astype(str) == child_id_str)
        & (fp_events_df["task_type"] == "free_play")
    ]

    if len(child_events) == 0:
        return outcomes

    # Create outcomes for each event of flagged types
    event_type_col = child_events["event_type"].astype(str)

    for _, event in child_events.iterrows():
        event_type = str(event.get("event_type", "")).upper()

        if event_type in flagged_types:
            outcomes.append({
                "task_type": "free_play",
                "event_type": event_type,
                "t_start_sec": float(event.get("t_start", 0)),
                "t_end_sec": float(event.get("t_end", 0)),
                "flagged": True,
                "status": "flagged",
            })

    return outcomes
