from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# QC columns to bring into merged features (per task)
QC_METRIC_COLS = [
    "face_detected_ratio",
    "pose_detected_ratio",
    "out_of_view_ratio",
    "frames_used",
    "duration_sec",
    "fps_est",
]


def _select_best_qc_row(task_qc: pd.DataFrame) -> Optional[pd.Series]:
    """
    Select the single best QC row for a task.

    Tie-breaker priority:
    1. Highest pose_detected_ratio (best tracking quality)
    2. Highest frames_used (most data)

    Args:
        task_qc: DataFrame of QC rows for a single (child, task) pair

    Returns:
        Best row as Series, or None if no rows
    """
    if len(task_qc) == 0:
        return None

    if len(task_qc) == 1:
        return task_qc.iloc[0]

    # Sort by pose_detected_ratio desc, then frames_used desc
    sort_cols = []
    if "pose_detected_ratio" in task_qc.columns:
        sort_cols.append("pose_detected_ratio")
    if "frames_used" in task_qc.columns:
        sort_cols.append("frames_used")

    if sort_cols:
        sorted_qc = task_qc.sort_values(sort_cols, ascending=False)
        return sorted_qc.iloc[0]

    # Fallback: just take first row
    return task_qc.iloc[0]


def _compute_task_quality(
    row: Dict[str, Any],
    task_prefix: str,
    qc_prefix: str,
) -> float:
    """
    Compute composite quality score for a task (0-1, higher = better).

    Uses QC metrics as primary source, falls back to feature-derived ratios.

    Quality formula:
        base = 0.50 * pose_ratio + 0.30 * face_ratio + 0.20 * (1 - out_of_view_ratio)
        if qc_pass == 0: base *= 0.5

    Args:
        row: Feature row dictionary
        task_prefix: Feature prefix (ja, imit, fp)
        qc_prefix: QC column prefix (qc_ja, qc_imit, qc_fp)

    Returns:
        Quality score between 0 and 1, or NaN if insufficient data
    """
    # Try QC metrics first (canonical source)
    pose_ratio = row.get(f"{qc_prefix}_pose_detected_ratio")
    face_ratio = row.get(f"{qc_prefix}_face_detected_ratio")
    out_of_view = row.get(f"{qc_prefix}_out_of_view_ratio")

    # Fall back to feature-derived ratios if QC metrics missing
    if pose_ratio is None or not np.isfinite(pose_ratio):
        pose_ratio = row.get(f"{task_prefix}_pose_present_ratio")
    if face_ratio is None or not np.isfinite(face_ratio):
        face_ratio = row.get(f"{task_prefix}_face_present_ratio")
    if out_of_view is None or not np.isfinite(out_of_view):
        # Derive from pose_present_ratio if available
        pose_present = row.get(f"{task_prefix}_pose_present_ratio")
        if pose_present is not None and np.isfinite(pose_present):
            out_of_view = 1.0 - pose_present

    # Compute weighted quality from available indicators
    indicators: List[Tuple[Optional[float], float]] = [
        (pose_ratio, 0.50),
        (face_ratio, 0.30),
    ]

    # out_of_view is inverted (lower is better)
    if out_of_view is not None and np.isfinite(out_of_view):
        indicators.append((1.0 - out_of_view, 0.20))

    valid_pairs = []
    for val, weight in indicators:
        if val is not None and np.isfinite(val):
            val = max(0.0, min(1.0, val))
            valid_pairs.append((val, weight))

    if not valid_pairs:
        return float("nan")

    total_weight = sum(w for _, w in valid_pairs)
    weighted_sum = sum(v * w for v, w in valid_pairs)
    base_quality = weighted_sum / total_weight

    # Apply QC penalty: if QC failed, reduce quality by 50%
    qc_pass = row.get(f"{qc_prefix}_pass")
    if qc_pass is not None and np.isfinite(qc_pass) and qc_pass < 0.5:
        base_quality *= 0.5

    return base_quality


def _safe_update(row: Dict[str, Any], features: Dict[str, Any], source: str) -> None:
    """Update row with features, raising on key collision."""
    normalized = {str(k): v for k, v in features.items()}
    for key in normalized:
        if key in row and key != "child_id":
            raise ValueError(f"Feature key collision: '{key}' from {source} already exists in row")
    row.update(normalized)


def merge_all_features(
    child_ids: List[str],
    common_features: Dict[str, Dict[str, float]],
    ja_features: Dict[str, Dict[str, float]],
    imit_features: Dict[str, Dict[str, float]],
    fp_features: Dict[str, Dict[str, float]],
    qc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all feature dictionaries into a single DataFrame with one row per child.

    Args:
        child_ids: List of child identifiers to include
        common_features: Dict mapping child_id -> common features dict
        ja_features: Dict mapping child_id -> joint attention features dict
        imit_features: Dict mapping child_id -> imitation features dict
        fp_features: Dict mapping child_id -> free play features dict
        qc_df: DataFrame with QC results (columns: child_id, task_type, qc_pass, ...)

    Returns:
        DataFrame with columns:
            child_id,
            # Common features per task (ja_*, imit_*, fp_*)
            # Task-specific features
            # QC metrics per task: qc_ja_*, qc_imit_*, qc_fp_*
            # QC summary: qc_tasks_passed
            # Quality scores: qc_ja_quality, qc_imit_quality, qc_fp_quality, qc_overall_quality
    """
    rows = []

    # Task mapping: task_type in QC -> (prefix for QC cols, prefix for features)
    task_config = {
        "joint_attention": ("qc_ja", "ja"),
        "imitation": ("qc_imit", "imit"),
        "free_play": ("qc_fp", "fp"),
    }

    for child_id in child_ids:
        child_id_str = str(child_id)
        row: Dict[str, Any] = {"child_id": child_id_str}

        # Add common features
        if child_id_str in common_features:
            _safe_update(row, common_features[child_id_str], "common_features")

        # Add joint attention features
        if child_id_str in ja_features:
            _safe_update(row, ja_features[child_id_str], "ja_features")

        # Add imitation features
        if child_id_str in imit_features:
            _safe_update(row, imit_features[child_id_str], "imit_features")

        # Add free play features
        if child_id_str in fp_features:
            _safe_update(row, fp_features[child_id_str], "fp_features")

        # Process QC data per task
        if len(qc_df) > 0 and "child_id" in qc_df.columns:
            child_qc = qc_df[qc_df["child_id"].astype(str) == child_id_str]
        else:
            child_qc = pd.DataFrame()
        tasks_passed = 0

        for task_type, (qc_prefix, feat_prefix) in task_config.items():
            task_qc = child_qc[child_qc["task_type"] == task_type]
            best_row = _select_best_qc_row(task_qc)

            if best_row is not None:
                # Add QC pass/fail
                qc_pass_val = best_row.get("qc_pass")
                passed = bool(qc_pass_val) if pd.notna(qc_pass_val) else False
                row[f"{qc_prefix}_pass"] = 1.0 if passed else 0.0
                if passed:
                    tasks_passed += 1

                # Add QC reason if available
                qc_reason = best_row.get("qc_reason")
                if pd.notna(qc_reason):
                    row[f"{qc_prefix}_reason"] = str(qc_reason)

                # Add QC metric columns
                for col in QC_METRIC_COLS:
                    if col in best_row.index:
                        val = best_row[col]
                        row[f"{qc_prefix}_{col}"] = float(val) if pd.notna(val) else float("nan")
            else:
                # No QC data for this task
                row[f"{qc_prefix}_pass"] = float("nan")
                for col in QC_METRIC_COLS:
                    row[f"{qc_prefix}_{col}"] = float("nan")

        row["qc_tasks_passed"] = float(tasks_passed)

        # Compute per-task quality scores (uses QC metrics primarily, falls back to features)
        row["qc_ja_quality"] = _compute_task_quality(row, "ja", "qc_ja")
        row["qc_imit_quality"] = _compute_task_quality(row, "imit", "qc_imit")
        row["qc_fp_quality"] = _compute_task_quality(row, "fp", "qc_fp")

        # Overall quality = mean of valid task qualities
        task_qualities = [row["qc_ja_quality"], row["qc_imit_quality"], row["qc_fp_quality"]]
        valid_qualities = [q for q in task_qualities if np.isfinite(q)]
        row["qc_overall_quality"] = float(np.mean(valid_qualities)) if valid_qualities else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def save_features(df: pd.DataFrame, out_path: Path, format: str = "csv") -> None:
    """
    Save feature DataFrame to file.

    Args:
        df: Feature DataFrame to save
        out_path: Output path
        format: Output format ('csv' or 'parquet')
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
