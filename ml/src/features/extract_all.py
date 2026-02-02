"""
Feature extraction pipeline entry point.

Extracts all features (common + task-specific) and merges them into a single
output file ready for training.

Usage:
    python -m ml.src.features.extract_all \
        --manifest data/manifest.csv \
        --qc data/derived/qc.csv \
        --tracks_dir data/derived/tracks \
        --imit_summary data/derived/imit_summary.csv \
        --fp_summary data/derived/free_play_summary.csv \
        --point_events data/derived/point_events.csv \
        --audio_events data/derived/audio_events.csv \
        --out data/derived/features_merged.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .common import extract_common_features_for_child
from .free_play import extract_free_play_features
from .imitation import extract_imitation_features
from .joint_attention import extract_joint_attention_features
from .merge import merge_all_features, save_features


def _read_csv_or_empty(path: Optional[str]) -> pd.DataFrame:
    """Read CSV if path exists, otherwise return empty DataFrame."""
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {p} not found, using empty DataFrame")
        return pd.DataFrame()
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract all features and merge into a single output file."
    )
    ap.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest CSV with child_id column",
    )
    ap.add_argument(
        "--qc",
        required=True,
        help="Path to QC results CSV (from qc_video.py)",
    )
    ap.add_argument(
        "--tracks_dir",
        required=True,
        help="Directory containing {child_id}_{task_type}.npz files",
    )
    ap.add_argument(
        "--imit_summary",
        default=None,
        help="Path to imitation summary CSV (from imitation.py)",
    )
    ap.add_argument(
        "--fp_summary",
        default=None,
        help="Path to free play summary CSV (from free_play_events.py)",
    )
    ap.add_argument(
        "--point_events",
        default=None,
        help="Path to pointing events CSV (from pointing.py)",
    )
    ap.add_argument(
        "--audio_events",
        default=None,
        help="Path to audio events CSV (from audio_events.py)",
    )
    ap.add_argument(
        "--out",
        default="data/derived/features_merged.csv",
        help="Output path for merged features",
    )
    ap.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)",
    )
    ap.add_argument(
        "--min_tasks_pass",
        type=int,
        default=0,
        help="Only include children who passed at least N tasks (0 = all children)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N children (0 = no limit, useful for testing)",
    )
    ap.add_argument(
        "--require_tracks",
        action="store_true",
        help="Only include children who have at least one track file",
    )

    args = ap.parse_args()

    # Load all input data
    print("Loading input data...")
    manifest_df = pd.read_csv(args.manifest)
    qc_df = pd.read_csv(args.qc)
    imit_summary_df = _read_csv_or_empty(args.imit_summary)
    fp_summary_df = _read_csv_or_empty(args.fp_summary)
    point_events_df = _read_csv_or_empty(args.point_events)
    audio_events_df = _read_csv_or_empty(args.audio_events)

    tracks_dir = Path(args.tracks_dir)

    if "child_id" not in manifest_df.columns:
        raise ValueError(f"Manifest missing 'child_id' column. Columns: {list(manifest_df.columns)}")

    child_ids = [str(x) for x in manifest_df["child_id"].unique()]

    # Filter to usable children if requested
    if args.min_tasks_pass > 0:
        pass_counts = qc_df.groupby(qc_df["child_id"].astype(str))["qc_pass"].sum()
        usable_ids = set(pass_counts[pass_counts >= args.min_tasks_pass].index)
        before = len(child_ids)
        child_ids = [c for c in child_ids if c in usable_ids]
        print(f"  Filtered to {len(child_ids)}/{before} children with >= {args.min_tasks_pass} tasks passed")

    # Filter to children with track files if requested
    if args.require_tracks:
        def _has_any_track(child_id: str) -> bool:
            for task in ["joint_attention", "imitation", "free_play"]:
                if (tracks_dir / f"{child_id}_{task}.npz").exists():
                    return True
            return False

        before = len(child_ids)
        child_ids = [c for c in child_ids if _has_any_track(c)]
        print(f"  Filtered to {len(child_ids)}/{before} children with track files")

    # Apply limit for testing
    if args.limit > 0:
        child_ids = child_ids[: args.limit]
        print(f"  Limited to first {len(child_ids)} children")

    n_children = len(child_ids)
    print(f"Extracting features for {n_children} children...")

    # Extract features for each child
    common_features = {}
    ja_features = {}
    imit_features = {}
    fp_features = {}

    for i, child_id in enumerate(child_ids):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{n_children}] child_id={child_id}")

        # Common features (from tracks + QC face_detected_ratio)
        common_features[child_id] = extract_common_features_for_child(
            child_id=child_id,
            tracks_dir=tracks_dir,
            qc_df=qc_df,
        )

        # Get video duration from common features for JA coverage ratio
        ja_duration_raw = common_features[child_id].get("ja_duration_sec")
        ja_duration = None if (ja_duration_raw is None or pd.isna(ja_duration_raw)) else float(ja_duration_raw)

        # Joint attention features (from pointing + audio events + tracks for response detection)
        ja_features[child_id] = extract_joint_attention_features(
            child_id=child_id,
            point_events_df=point_events_df if len(point_events_df) > 0 else pd.DataFrame(),
            audio_events_df=audio_events_df if len(audio_events_df) > 0 else None,
            tracks_dir=tracks_dir,
            video_duration_sec=ja_duration,
        )

        # Imitation features (from imit_summary)
        imit_features[child_id] = extract_imitation_features(
            child_id=child_id,
            imit_summary_df=imit_summary_df,
        )

        # Free play features (from fp_summary)
        fp_features[child_id] = extract_free_play_features(
            child_id=child_id,
            fp_summary_df=fp_summary_df,
        )

    # Merge all features
    print("Merging features...")
    merged_df = merge_all_features(
        child_ids=child_ids,
        common_features=common_features,
        ja_features=ja_features,
        imit_features=imit_features,
        fp_features=fp_features,
        qc_df=qc_df,
    )

    # Save output
    out_path = Path(args.out)
    save_features(merged_df, out_path, format=args.format)

    print(f"Saved {len(merged_df)} rows x {len(merged_df.columns)} columns to {out_path}")

    # Print feature summary
    print("\nFeature columns:")
    for col in merged_df.columns:
        non_nan = merged_df[col].notna().sum()
        print(f"  {col}: {non_nan}/{len(merged_df)} non-null")


if __name__ == "__main__":
    main()
