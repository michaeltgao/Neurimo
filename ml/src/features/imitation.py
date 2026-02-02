from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def extract_imitation_features(
    child_id: str,
    imit_summary_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Extract imitation features from the imitation summary CSV.

    Args:
        child_id: Child identifier
        imit_summary_df: DataFrame from imitation.py output (imit_summary.csv) with columns:
            child_id, task_type, clap_demo_present, clap_demo_method, adult_clap_count,
            child_clap_count, clap_response_present, clap_latency_sec, arms_demo_present,
            arms_response_present, arms_latency_sec, child_arms_up_total,
            demo_primitives, responded_primitives, imitation_score

    Returns:
        Dict with features (all prefixed with imit_):
            imit_score: Overall imitation score (0-1)
            imit_clap_demo_present: Whether clap demo was detected (0/1)
            imit_clap_response_present: Whether child responded to clap (0/1)
            imit_clap_latency_sec: Latency of clap response (NaN if no response)
            imit_adult_clap_count: Number of adult claps detected
            imit_child_clap_count: Number of child claps detected
            imit_arms_demo_present: Whether arms-up demo was detected (0/1)
            imit_arms_response_present: Whether child responded to arms-up (0/1)
            imit_arms_latency_sec: Latency of arms response (NaN if no response)
            imit_child_arms_up_total: Total child arms-up events
            imit_demo_primitives: Number of demo primitives shown
            imit_responded_primitives: Number of primitives child responded to
    """
    child_id_str = str(child_id)

    # Filter to this child + task_type (safety belt for duplicates/multiple visits)
    child_df = imit_summary_df[
        (imit_summary_df["child_id"].astype(str) == child_id_str)
        & (imit_summary_df["task_type"] == "imitation")
    ]

    if len(child_df) == 0:
        return {
            "imit_score": float("nan"),
            "imit_clap_demo_present": float("nan"),
            "imit_clap_response_present": float("nan"),
            "imit_clap_latency_sec": float("nan"),
            "imit_adult_clap_count": float("nan"),
            "imit_child_clap_count": float("nan"),
            "imit_arms_demo_present": float("nan"),
            "imit_arms_response_present": float("nan"),
            "imit_arms_latency_sec": float("nan"),
            "imit_child_arms_up_total": float("nan"),
            "imit_demo_primitives": float("nan"),
            "imit_responded_primitives": float("nan"),
        }

    row = child_df.iloc[0]

    def safe_float(val, default=float("nan")) -> float:
        """Convert value to float, handling empty strings and None."""
        if pd.isna(val) or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def safe_bool_to_float(val) -> float:
        """Convert boolean-like value to 0.0/1.0, NaN for missing/unknown."""
        if pd.isna(val) or val == "":
            return float("nan")
        if isinstance(val, (bool, np.bool_)):
            return 1.0 if val else 0.0
        if isinstance(val, (int, float, np.integer, np.floating)):
            return 1.0 if val else 0.0
        if isinstance(val, str):
            lower = val.lower()
            if lower in ("true", "1", "yes"):
                return 1.0
            if lower in ("false", "0", "no"):
                return 0.0
            # Unknown strings (nan, none, missing, etc.) → NaN
            return float("nan")
        return float("nan")

    return {
        "imit_score": safe_float(row.get("imitation_score")),
        "imit_clap_demo_present": safe_bool_to_float(row.get("clap_demo_present")),
        "imit_clap_response_present": safe_bool_to_float(row.get("clap_response_present")),
        "imit_clap_latency_sec": safe_float(row.get("clap_latency_sec")),
        "imit_adult_clap_count": safe_float(row.get("adult_clap_count"), default=0.0),
        "imit_child_clap_count": safe_float(row.get("child_clap_count"), default=0.0),
        "imit_arms_demo_present": safe_bool_to_float(row.get("arms_demo_present")),
        "imit_arms_response_present": safe_bool_to_float(row.get("arms_response_present")),
        "imit_arms_latency_sec": safe_float(row.get("arms_latency_sec")),
        "imit_child_arms_up_total": safe_float(row.get("child_arms_up_total"), default=0.0),
        "imit_demo_primitives": safe_float(row.get("demo_primitives"), default=0.0),
        "imit_responded_primitives": safe_float(row.get("responded_primitives"), default=0.0),
    }
