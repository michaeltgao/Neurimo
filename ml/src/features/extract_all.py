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
        --fp_events data/derived/free_play_events.csv \
        --point_events data/derived/point_events.csv \
        --audio_events data/derived/audio_events.csv \
        --out data/derived/features_merged.csv \
        --workers 4  # Enable parallel processing
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .common import extract_common_features_for_child
from .free_play import extract_free_play_features
from .imitation import extract_imitation_features
from .joint_attention import extract_joint_attention_features
from .merge import merge_all_features, save_features

# Optional tqdm support for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def _read_csv_or_empty(path: Optional[str]) -> pd.DataFrame:
    """Read CSV if path exists, otherwise return empty DataFrame."""
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {p} not found, using empty DataFrame")
        return pd.DataFrame()
    return pd.read_csv(p)


def _prefilter_dataframes(
    child_ids: List[str],
    qc_df: pd.DataFrame,
    imit_summary_df: pd.DataFrame,
    fp_summary_df: pd.DataFrame,
    fp_events_df: pd.DataFrame,
    point_events_df: pd.DataFrame,
    audio_events_df: pd.DataFrame,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Pre-filter all DataFrames by child_id to avoid repeated filtering in the loop.

    Returns a dict mapping child_id -> {df_name: filtered_df}.
    This reduces O(N*M) filtering to O(N+M) where N=children, M=rows.
    """
    child_set = set(child_ids)

    # Convert child_id columns to string once for efficient filtering
    def _index_by_child(df: pd.DataFrame, child_col: str = "child_id") -> Dict[str, pd.DataFrame]:
        if df is None or len(df) == 0 or child_col not in df.columns:
            return {}
        df = df.copy()
        df["_child_str"] = df[child_col].astype(str)
        return {str(cid): group.drop(columns=["_child_str"]) for cid, group in df.groupby("_child_str") if cid in child_set}

    # Pre-index all DataFrames
    qc_indexed = _index_by_child(qc_df)
    imit_indexed = _index_by_child(imit_summary_df)
    fp_summary_indexed = _index_by_child(fp_summary_df)
    fp_events_indexed = _index_by_child(fp_events_df)
    point_indexed = _index_by_child(point_events_df)
    audio_indexed = _index_by_child(audio_events_df)

    # Build per-child lookup
    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    for cid in child_ids:
        result[cid] = {
            "qc": qc_indexed.get(cid, pd.DataFrame()),
            "imit_summary": imit_indexed.get(cid, pd.DataFrame()),
            "fp_summary": fp_summary_indexed.get(cid, pd.DataFrame()),
            "fp_events": fp_events_indexed.get(cid, pd.DataFrame()),
            "point_events": point_indexed.get(cid, pd.DataFrame()),
            "audio_events": audio_indexed.get(cid, pd.DataFrame()),
        }
    return result


def _filter_df_for_child(df: pd.DataFrame, child_id: str) -> pd.DataFrame:
    """Filter DataFrame to rows for a specific child_id."""
    if df is None or len(df) == 0 or "child_id" not in df.columns:
        return pd.DataFrame()
    return df[df["child_id"].astype(str) == child_id]


def _extract_features_for_child_prefiltered(
    child_id: str,
    tracks_dir: Path,
    prefiltered: Dict[str, pd.DataFrame],
) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Extract all features for a single child using pre-filtered DataFrames.
    Use this for sequential execution to avoid repeated filtering.
    """
    return _extract_child_features_impl(
        child_id=child_id,
        tracks_dir=tracks_dir,
        qc_df=prefiltered["qc"],
        point_df=prefiltered["point_events"],
        audio_df=prefiltered["audio_events"],
        imit_summary_df=prefiltered["imit_summary"],
        fp_summary_df=prefiltered["fp_summary"],
        fp_events_df=prefiltered["fp_events"],
    )


def _extract_features_for_child_full(
    child_id: str,
    tracks_dir: Path,
    qc_df: pd.DataFrame,
    point_events_df: pd.DataFrame,
    audio_events_df: pd.DataFrame,
    imit_summary_df: pd.DataFrame,
    fp_summary_df: pd.DataFrame,
    fp_events_df: pd.DataFrame,
) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Extract all features for a single child, filtering full DataFrames.
    Use this for parallel execution - filtering happens in worker process,
    avoiding pickle overhead of pre-filtered DataFrames.
    """
    return _extract_child_features_impl(
        child_id=child_id,
        tracks_dir=tracks_dir,
        qc_df=_filter_df_for_child(qc_df, child_id),
        point_df=_filter_df_for_child(point_events_df, child_id),
        audio_df=_filter_df_for_child(audio_events_df, child_id),
        imit_summary_df=_filter_df_for_child(imit_summary_df, child_id),
        fp_summary_df=_filter_df_for_child(fp_summary_df, child_id),
        fp_events_df=_filter_df_for_child(fp_events_df, child_id),
    )


def _extract_child_features_impl(
    child_id: str,
    tracks_dir: Path,
    qc_df: pd.DataFrame,
    point_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    imit_summary_df: pd.DataFrame,
    fp_summary_df: pd.DataFrame,
    fp_events_df: pd.DataFrame,
) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Core implementation for extracting all features for a single child.

    Args:
        child_id: Child identifier
        tracks_dir: Path to tracks directory
        qc_df: QC data for this child only
        point_df: Point events for this child only
        audio_df: Audio events for this child only
        imit_summary_df: Imitation summary for this child only
        fp_summary_df: Free play summary for this child only
        fp_events_df: Free play events for this child only

    Returns:
        Tuple of (child_id, common_features, ja_features, imit_features, fp_features)
    """
    # Common features (from tracks + QC face_detected_ratio)
    common_feats = extract_common_features_for_child(
        child_id=child_id,
        tracks_dir=tracks_dir,
        qc_df=qc_df,
    )

    # Get video duration from common features for JA coverage ratio
    ja_duration_raw = common_feats.get("ja_duration_sec")
    ja_duration = None if (ja_duration_raw is None or pd.isna(ja_duration_raw)) else float(ja_duration_raw)

    # Joint attention features
    ja_feats = extract_joint_attention_features(
        child_id=child_id,
        point_events_df=point_df if len(point_df) > 0 else pd.DataFrame(),
        audio_events_df=audio_df if len(audio_df) > 0 else None,
        tracks_dir=tracks_dir,
        video_duration_sec=ja_duration,
    )

    # Imitation features
    imit_feats = extract_imitation_features(
        child_id=child_id,
        imit_summary_df=imit_summary_df,
    )

    # Free play features
    fp_feats = extract_free_play_features(
        child_id=child_id,
        fp_summary_df=fp_summary_df,
        fp_events_df=fp_events_df if len(fp_events_df) > 0 else None,
    )

    return child_id, common_feats, ja_feats, imit_feats, fp_feats


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
        "--fp_events",
        default=None,
        help="Path to free play events CSV (from free_play_events.py) for temporal features",
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
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential). "
             "Set to 0 for auto (uses CPU count - 1).",
    )
    ap.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar even if tqdm is available",
    )

    args = ap.parse_args()

    # Load all input data
    print("Loading input data...")
    manifest_df = pd.read_csv(args.manifest)
    qc_df = pd.read_csv(args.qc)
    imit_summary_df = _read_csv_or_empty(args.imit_summary)
    fp_summary_df = _read_csv_or_empty(args.fp_summary)
    fp_events_df = _read_csv_or_empty(args.fp_events)
    point_events_df = _read_csv_or_empty(args.point_events)
    audio_events_df = _read_csv_or_empty(args.audio_events)

    tracks_dir = Path(args.tracks_dir)

    if "child_id" not in manifest_df.columns:
        raise ValueError(f"Manifest missing 'child_id' column. Columns: {list(manifest_df.columns)}")

    child_ids = [str(x) for x in manifest_df["child_id"].unique()]

    # Filter to usable children if requested
    if args.min_tasks_pass > 0:
        # Count distinct tasks passed per child (handles duplicates and non-binary qc_pass)
        tmp = qc_df.copy()
        tmp["_child_str"] = tmp["child_id"].astype(str)
        tmp["_passed"] = tmp["qc_pass"].astype(float) >= 0.5
        # Get max pass status per (child, task) to handle duplicates
        passed_by_task = tmp.groupby(["_child_str", "task_type"])["_passed"].max()
        # Count tasks passed per child
        pass_counts = passed_by_task.groupby("_child_str").sum()
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

    # Determine worker count
    n_workers = args.workers
    if n_workers == 0:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    n_workers = min(n_workers, n_children)  # Don't use more workers than children

    # Pre-filter DataFrames by child_id (O(N+M) instead of O(N*M))
    print("Pre-filtering DataFrames by child_id...")
    prefiltered = _prefilter_dataframes(
        child_ids=child_ids,
        qc_df=qc_df,
        imit_summary_df=imit_summary_df,
        fp_summary_df=fp_summary_df,
        fp_events_df=fp_events_df,
        point_events_df=point_events_df,
        audio_events_df=audio_events_df,
    )

    # Setup progress tracking
    use_progress = TQDM_AVAILABLE and not args.no_progress
    parallel_mode = n_workers > 1

    if parallel_mode:
        print(f"Extracting features for {n_children} children using {n_workers} workers...")
    else:
        print(f"Extracting features for {n_children} children...")

    # Extract features for each child
    common_features: Dict[str, Dict[str, float]] = {}
    ja_features: Dict[str, Dict[str, float]] = {}
    imit_features: Dict[str, Dict[str, float]] = {}
    fp_features: Dict[str, Dict[str, float]] = {}

    failed_children: List[Tuple[str, str]] = []  # (child_id, error_msg)

    if parallel_mode:
        # Parallel execution using ProcessPoolExecutor
        # Pass full DataFrames - filtering happens in worker process to avoid pickle overhead
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks with full DataFrames (not prefiltered)
            futures = {
                executor.submit(
                    _extract_features_for_child_full,
                    child_id,
                    tracks_dir,
                    qc_df,
                    point_events_df,
                    audio_events_df,
                    imit_summary_df,
                    fp_summary_df,
                    fp_events_df,
                ): child_id
                for child_id in child_ids
            }

            # Collect results with progress tracking
            if use_progress:
                completed_iter = tqdm(
                    as_completed(futures),
                    total=n_children,
                    desc="Extracting",
                    unit="child",
                )
            else:
                completed_iter = as_completed(futures)  # type: ignore[assignment]

            completed_count = 0
            for future in completed_iter:
                cid = futures[future]
                try:
                    child_id, common, ja, imit, fp = future.result()
                    common_features[child_id] = common
                    ja_features[child_id] = ja
                    imit_features[child_id] = imit
                    fp_features[child_id] = fp
                except Exception as e:
                    failed_children.append((cid, str(e)))

                completed_count += 1
                if not use_progress and completed_count % 10 == 0:
                    print(f"  [{completed_count}/{n_children}] completed")
    else:
        # Sequential execution (workers=1) - use prefiltered data for efficiency
        if use_progress:
            child_iter = tqdm(child_ids, desc="Extracting", unit="child")
        else:
            child_iter = child_ids  # type: ignore[assignment]

        for i, child_id in enumerate(child_iter):
            if not use_progress and ((i + 1) % 10 == 0 or i == 0):
                print(f"  [{i + 1}/{n_children}] child_id={child_id}")

            try:
                child_id, common, ja, imit, fp = _extract_features_for_child_prefiltered(
                    child_id=child_id,
                    tracks_dir=tracks_dir,
                    prefiltered=prefiltered[child_id],
                )
                common_features[child_id] = common
                ja_features[child_id] = ja
                imit_features[child_id] = imit
                fp_features[child_id] = fp
            except Exception as e:
                failed_children.append((child_id, str(e)))

    # Report failures
    if failed_children:
        print(f"\nWARNING: {len(failed_children)} children failed:")
        for cid, err in failed_children[:5]:
            print(f"  {cid}: {err}")
        if len(failed_children) > 5:
            print(f"  ... and {len(failed_children) - 5} more")

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
