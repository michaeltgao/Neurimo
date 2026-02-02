from __future__ import annotations

from typing import Dict

import pandas as pd


def extract_free_play_features(
    child_id: str,
    fp_summary_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Extract free play features from the free play summary CSV.

    Args:
        child_id: Child identifier
        fp_summary_df: DataFrame from free_play_events.py output (free_play_summary.csv)
            with columns: child_id, task_type, duration_sec, pose_present_ratio,
            adult_present_ratio, adult_hand_active_time_frac, adult_hand_mean_activity,
            freeze_time_frac, hand_to_face_time_frac, repetitive_motion_time_frac,
            engaged_with_adult_time_frac, disengaged_with_adult_time_frac,
            hands_near_torso_time_frac, repetitive_motion_freq_hz

    Returns:
        Dict with features (all prefixed with fp_):
            # Note: fp_duration_sec and fp_pose_present_ratio are provided by common.py
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
    """
    child_id_str = str(child_id)

    # Filter to this child + task_type (safety belt for duplicates/multiple visits)
    child_df = fp_summary_df[
        (fp_summary_df["child_id"].astype(str) == child_id_str)
        & (fp_summary_df["task_type"] == "free_play")
    ]

    if len(child_df) == 0:
        # Note: fp_duration_sec and fp_pose_present_ratio are provided by common.py
        return {
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

    # Pick best row if duplicates exist (highest pose_present_ratio = best tracking)
    child_df = child_df.copy()
    child_df["_sort_key"] = child_df["pose_present_ratio"].fillna(-1)
    row = child_df.sort_values("_sort_key", ascending=False).iloc[0]

    def safe_float(val, default=float("nan")) -> float:
        """Convert value to float, handling empty strings and None."""
        if pd.isna(val) or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def clamp01(x: float) -> float:
        """Clamp value to [0, 1] range, preserving NaN."""
        if pd.isna(x):
            return x
        return max(0.0, min(1.0, x))

    # Note: fp_duration_sec and fp_pose_present_ratio are provided by common.py
    return {
        "fp_adult_present_ratio": clamp01(safe_float(row.get("adult_present_ratio"))),
        "fp_adult_hand_active_time_frac": clamp01(safe_float(row.get("adult_hand_active_time_frac"))),
        "fp_adult_hand_mean_activity": safe_float(row.get("adult_hand_mean_activity")),  # not a ratio
        "fp_freeze_time_frac": clamp01(safe_float(row.get("freeze_time_frac"))),
        "fp_hand_to_face_time_frac": clamp01(safe_float(row.get("hand_to_face_time_frac"))),
        "fp_repetitive_motion_time_frac": clamp01(safe_float(row.get("repetitive_motion_time_frac"))),
        "fp_engaged_with_adult_time_frac": clamp01(safe_float(row.get("engaged_with_adult_time_frac"))),
        "fp_disengaged_with_adult_time_frac": clamp01(safe_float(row.get("disengaged_with_adult_time_frac"))),
        "fp_hands_near_torso_time_frac": clamp01(safe_float(row.get("hands_near_torso_time_frac"))),
        "fp_repetitive_motion_freq_hz": safe_float(row.get("repetitive_motion_freq_hz")),  # Hz, not a ratio
    }
