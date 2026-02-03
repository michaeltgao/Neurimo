from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _is_present(val) -> bool:
    """Check if a value is present (not NaN/None/empty string)."""
    if pd.isna(val):
        return False
    if isinstance(val, str) and val.strip() == "":
        return False
    return True


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
            demo_primitives, responded_primitives, imitation_score,
            # New attempt tracking fields:
            clap_attempts_before_success, clap_total_child_attempts,
            arms_attempts_before_success, arms_total_child_attempts,
            total_attempts_before_success, mean_imitation_latency_sec

    Returns:
        Dict with features (all prefixed with imit_):

            # Core imitation metrics
            imit_score: Overall imitation score (0-1)
            imit_demo_primitives: Number of demo primitives shown
            imit_responded_primitives: Number of primitives child responded to

            # Clap-specific
            imit_clap_demo_present: Whether clap demo was detected (0/1)
            imit_clap_response_present: Whether child responded to clap (0/1)
            imit_clap_latency_sec: Latency of clap response (NaN if no response)
            imit_adult_clap_count: Number of adult claps detected
            imit_child_clap_count: Number of child claps detected
            imit_clap_attempts_before_success: Clap attempts before first success

            # Arms-up specific
            imit_arms_demo_present: Whether arms-up demo was detected (0/1)
            imit_arms_response_present: Whether child responded to arms-up (0/1)
            imit_arms_latency_sec: Latency of arms response (NaN if no response)
            imit_child_arms_up_total: Total child arms-up events
            imit_arms_attempts_before_success: Arms attempts before first success

            # Paper-aligned features (Mirror paper)
            imit_attempts_before_success: Total attempts before first successful imitation
                - 0 if child succeeded on first attempt
                - Higher values indicate more trials needed
                - NaN if no demo or attempt data missing
            imit_response_latency_sec: Mean response latency across successful imitations (sec)
                - Time from end of parent demo to start of child response
                - NaN if no successful imitations
    """
    child_id_str = str(child_id)

    # Filter to this child + task_type
    if len(imit_summary_df) > 0 and "child_id" in imit_summary_df.columns:
        child_df = imit_summary_df[
            (imit_summary_df["child_id"].astype(str) == child_id_str)
            & (imit_summary_df["task_type"] == "imitation")
        ]
    else:
        child_df = pd.DataFrame()

    # Define all output features with NaN defaults
    nan_features = {
        # Core metrics
        "imit_score": float("nan"),
        "imit_demo_primitives": float("nan"),
        "imit_responded_primitives": float("nan"),

        # Clap-specific
        "imit_clap_demo_present": float("nan"),
        "imit_clap_response_present": float("nan"),
        "imit_clap_latency_sec": float("nan"),
        "imit_adult_clap_count": float("nan"),
        "imit_child_clap_count": float("nan"),
        "imit_clap_attempts_before_success": float("nan"),

        # Arms-up specific
        "imit_arms_demo_present": float("nan"),
        "imit_arms_response_present": float("nan"),
        "imit_arms_latency_sec": float("nan"),
        "imit_child_arms_up_total": float("nan"),
        "imit_arms_attempts_before_success": float("nan"),

        # Paper-aligned features
        "imit_attempts_before_success": float("nan"),
        "imit_response_latency_sec": float("nan"),
    }

    if len(child_df) == 0:
        return nan_features

    # FIX #1: Handle duplicate rows deterministically
    # Sort by best performance metrics, take the best row
    if len(child_df) > 1:
        # Create sortable columns (handle missing values)
        sort_df = child_df.copy()
        sort_df["_sort_score"] = pd.to_numeric(
            sort_df["imitation_score"] if "imitation_score" in sort_df.columns else pd.Series(0, index=sort_df.index),
            errors="coerce"
        ).fillna(-1)
        sort_df["_sort_resp"] = pd.to_numeric(
            sort_df["responded_primitives"] if "responded_primitives" in sort_df.columns else pd.Series(0, index=sort_df.index),
            errors="coerce"
        ).fillna(-1)
        sort_df = sort_df.sort_values(
            by=["_sort_score", "_sort_resp"],
            ascending=[False, False],
        )
        child_df = sort_df.drop(columns=["_sort_score", "_sort_resp"])

    row = child_df.iloc[0]

    def safe_float(val, default: float = float("nan")) -> float:
        """Convert value to float, handling empty strings and None."""
        if pd.isna(val):
            return default
        if isinstance(val, str):
            val = val.strip()
            if val == "":
                return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def safe_bool_to_float(val) -> float:
        """Convert boolean-like value to 0.0/1.0, NaN for missing/unknown."""
        if pd.isna(val):
            return float("nan")
        if isinstance(val, (bool, np.bool_)):
            return 1.0 if val else 0.0
        if isinstance(val, (int, float, np.integer, np.floating)):
            return 1.0 if val else 0.0
        if isinstance(val, str):
            # FIX #2: Strip whitespace before checking
            lower = val.strip().lower()
            if lower == "":
                return float("nan")
            if lower in ("true", "1", "yes"):
                return 1.0
            if lower in ("false", "0", "no"):
                return 0.0
            # Unknown strings (nan, none, missing, etc.) → NaN
            return float("nan")
        return float("nan")

    def safe_float_or_none(val) -> Optional[float]:
        """Convert value to float, returning None if missing."""
        if pd.isna(val):
            return None
        if isinstance(val, str):
            val = val.strip()
            if val == "":
                return None
        try:
            result = float(val)
            return result if np.isfinite(result) else None
        except (ValueError, TypeError):
            return None

    # Extract core metrics
    features: Dict[str, float] = {
        "imit_score": safe_float(row.get("imitation_score")),
        "imit_demo_primitives": safe_float(row.get("demo_primitives"), default=0.0),
        "imit_responded_primitives": safe_float(row.get("responded_primitives"), default=0.0),
    }

    # Extract clap-specific features
    clap_response_present = safe_bool_to_float(row.get("clap_response_present"))
    features.update({
        "imit_clap_demo_present": safe_bool_to_float(row.get("clap_demo_present")),
        "imit_clap_response_present": clap_response_present,
        "imit_clap_latency_sec": safe_float(row.get("clap_latency_sec")),
        "imit_adult_clap_count": safe_float(row.get("adult_clap_count"), default=0.0),
        "imit_child_clap_count": safe_float(row.get("child_clap_count"), default=0.0),
        "imit_clap_attempts_before_success": safe_float(row.get("clap_attempts_before_success")),
    })

    # Extract arms-up specific features
    arms_response_present = safe_bool_to_float(row.get("arms_response_present"))
    features.update({
        "imit_arms_demo_present": safe_bool_to_float(row.get("arms_demo_present")),
        "imit_arms_response_present": arms_response_present,
        "imit_arms_latency_sec": safe_float(row.get("arms_latency_sec")),
        "imit_child_arms_up_total": safe_float(row.get("child_arms_up_total"), default=0.0),
        "imit_arms_attempts_before_success": safe_float(row.get("arms_attempts_before_success")),
    })

    # =========================================================================
    # Paper-aligned features (Mirror paper for autism identification)
    # =========================================================================

    # imit_attempts_before_success: Total attempts before first successful imitation
    # - Available directly from summary if using updated perception module
    # - Fallback: compute from clap + arms attempts
    total_attempts_raw = row.get("total_attempts_before_success")
    if _is_present(total_attempts_raw):
        attempts_before_success = safe_float(total_attempts_raw)
    else:
        # FIX #3: Fallback only if we actually have attempt data
        clap_attempts_raw = row.get("clap_attempts_before_success")
        arms_attempts_raw = row.get("arms_attempts_before_success")
        demo_prims = safe_float(row.get("demo_primitives"), default=0.0)

        have_clap_data = _is_present(clap_attempts_raw)
        have_arms_data = _is_present(arms_attempts_raw)

        if demo_prims > 0 and (have_clap_data or have_arms_data):
            # Sum available attempt counts (treat missing as 0 only if other is present)
            clap_attempts = safe_float(clap_attempts_raw, default=0.0) if have_clap_data else 0.0
            arms_attempts = safe_float(arms_attempts_raw, default=0.0) if have_arms_data else 0.0
            attempts_before_success = clap_attempts + arms_attempts
        else:
            # No demo or no attempt data → unknown
            attempts_before_success = float("nan")

    features["imit_attempts_before_success"] = attempts_before_success

    # imit_response_latency_sec: Mean response latency across successful imitations
    # FIX #4: Only include latency if response_present is confirmed true
    # FIX #5: Consistent naming (_sec not _s)
    mean_latency_raw = row.get("mean_imitation_latency_sec")
    if _is_present(mean_latency_raw):
        response_latency = safe_float(mean_latency_raw)
    else:
        # Fallback: compute mean from individual latencies, but only if response occurred
        latencies = []

        clap_lat = safe_float_or_none(row.get("clap_latency_sec"))
        if clap_lat is not None and clap_response_present == 1.0:
            latencies.append(clap_lat)

        arms_lat = safe_float_or_none(row.get("arms_latency_sec"))
        if arms_lat is not None and arms_response_present == 1.0:
            latencies.append(arms_lat)

        if latencies:
            response_latency = float(np.mean(latencies))
        else:
            response_latency = float("nan")

    features["imit_response_latency_sec"] = response_latency

    return features
