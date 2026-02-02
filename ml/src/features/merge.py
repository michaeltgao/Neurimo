from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _safe_update(row: Dict[str, Any], features: Dict[str, Any], source: str) -> None:
    """Update row with features, raising on key collision."""
    # Normalize keys to strings
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
        qc_df: DataFrame with QC results (columns: child_id, task_type, qc_pass)

    Returns:
        DataFrame with columns:
            child_id,
            # Common features per task (ja_*, imit_*, fp_*)
            # Task-specific features
            # QC flags: qc_ja_pass, qc_imit_pass, qc_fp_pass, qc_tasks_passed
    """
    rows = []

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

        # Add QC pass/fail flags
        child_qc = qc_df[qc_df["child_id"].astype(str) == child_id_str]

        task_map = {
            "joint_attention": "qc_ja_pass",
            "imitation": "qc_imit_pass",
            "free_play": "qc_fp_pass",
        }

        tasks_passed = 0
        for task_type, col_name in task_map.items():
            task_qc = child_qc[child_qc["task_type"] == task_type]
            if len(task_qc) > 0:
                # Use .any() to handle duplicates - pass if ANY row passes
                passed = bool(task_qc["qc_pass"].astype(bool).any())
                row[col_name] = 1.0 if passed else 0.0
                if passed:
                    tasks_passed += 1
            else:
                row[col_name] = float("nan")

        row["qc_tasks_passed"] = float(tasks_passed)

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
