"""
Train stacking meta-model using out-of-fold (OOF) predictions from per-task models.

Base inputs per child (default):
  - P_joint_logreg
  - P_imit_logreg
  - P_free_xgb

Optionally also:
  - P_joint_xgb

Assumptions:
- You already ran F1 and saved fold models to data/derived/models:
    joint_logreg_fold{0..k-1}.joblib
    imit_logreg_fold{0..k-1}.joblib
    free_xgb_fold{0..k-1}.joblib
- You have a single "wide" dataset CSV containing the feature columns for each task,
  plus an id and label column.
- You have splits.json from split.py with k-fold configuration.

This script:
1) Loads the same CV folds from splits.json (used in F1 training).
2) Loads each fold model and generates OOF proba predictions.
3) Fits a LogisticRegression meta-model on OOF predictions.
4) Evaluates meta-model via CV on OOF features (with nested threshold selection).
5) Saves meta-model + artifacts to data/derived/models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib  # type: ignore[import-untyped]

from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
)

# -----------------------------
# Config defaults (edit if needed)
# -----------------------------
DEFAULT_MODELS_DIR = Path("data/derived/models")
DEFAULT_OUT_DIR = Path("data/derived/models")

# Heuristic feature selection by prefix (fallback if you don't have explicit lists)
DEFAULT_TASK_PREFIXES = {
    "joint": ["ja_"],     # joint attention features
    "imit": ["imit_"],    # imitation features
    "free": ["fp_"],      # free play features
}

# -----------------------------
# Helpers
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, required=True,
                   help="Wide CSV with features + label. Example: data/derived/features.csv")
    p.add_argument("--splits", type=str, required=True,
                   help="Path to splits.json from split.py (must be kfold mode)")
    p.add_argument("--id_col", type=str, default="child_id")
    p.add_argument("--label_col", type=str, required=True,
                   help="Binary label column name (0/1). Example: y")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for meta-model (folds come from splits.json)")

    p.add_argument("--models_dir", type=str, default=str(DEFAULT_MODELS_DIR))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))

    # Base model specs: task -> model_name used in filenames
    p.add_argument("--joint_base", type=str, default="logreg",
                   help="Filename stem for joint model (e.g. logreg or xgb)")
    p.add_argument("--imit_base", type=str, default="logreg",
                   help="Filename stem for imit model (e.g. logreg or xgb)")
    p.add_argument("--free_base", type=str, default="xgb",
                   help="Filename stem for free model (e.g. logreg or xgb)")

    p.add_argument("--include_joint_xgb", action="store_true",
                   help="If set, also include P_joint_xgb as an extra meta feature (requires joint_xgb_fold*.joblib)")

    # Optional explicit feature lists (if you want deterministic selection)
    p.add_argument("--feature_cols_json", type=str, default="",
                   help="Optional JSON mapping task->list_of_feature_cols. Overrides prefix heuristic.")

    # Meta-model regularization
    p.add_argument("--meta_C", type=float, default=1.0,
                   help="Single C value (ignored if --meta_C_grid is set)")
    p.add_argument("--meta_class_weight", type=str, default="balanced",
                   help="Use 'balanced' or 'none' (ignored if --meta_class_weight_grid is set)")

    # Grid search options
    p.add_argument("--meta_C_grid", type=str, default="",
                   help="Comma-separated C values to search, e.g. '0.01,0.1,1,10'")
    p.add_argument("--meta_class_weight_grid", type=str, default="",
                   help="Comma-separated class_weight options, e.g. 'balanced,none'")
    p.add_argument("--select_metric", type=str, default="auprc",
                   choices=["auroc", "auprc", "f1", "balanced_acc"],
                   help="Metric to optimize during grid search (default: auprc)")

    # Task exclusion (e.g., drop weak imitation)
    p.add_argument("--exclude_tasks", type=str, default="",
                   help="Comma-separated tasks to exclude, e.g. 'imit'")

    # Confidence features
    p.add_argument("--add_confidence", action="store_true",
                   help="Add confidence features |p - 0.5| for each base model")

    # Scaling
    p.add_argument("--scale_meta_features", action="store_true",
                   help="Standardize meta-features before fitting (recommended if using confidence features)")

    # QC-aware training options
    p.add_argument("--quality_col", type=str, default=None,
                   help="Column name for sample weights (e.g., qc_overall_quality). "
                        "If provided, samples are weighted by this column during training.")
    p.add_argument("--quality_floor", type=float, default=0.3,
                   help="Minimum sample weight to prevent zero-weighting low-quality samples (default: 0.3)")

    return p.parse_args()


def load_splits(splits_path: Path) -> List[Dict]:
    """
    Load k-fold splits from JSON (generated by split.py).
    Returns list of {"fold": int, "train": [child_ids], "val": [child_ids]}.
    """
    with open(splits_path, "r") as f:
        data = json.load(f)

    if "folds" not in data or not isinstance(data["folds"], list):
        raise ValueError(
            f"splits.json must be in kfold mode (have 'folds' key). "
            f"Found keys: {list(data.keys())}"
        )

    folds = []
    for fold_data in data["folds"]:
        folds.append({
            "fold": int(fold_data["fold"]),
            "train": [str(x) for x in fold_data["train"]],
            "val": [str(x) for x in fold_data["val"]],
        })

    if len(folds) == 0:
        raise ValueError("splits.json contains no folds")

    return folds


def get_indices_for_ids(
    child_ids_all: List[str], target_ids: List[str]
) -> Tuple[List[int], List[str]]:
    """
    Get indices of child_ids_all that are in target_ids.

    Returns:
        indices: List of matched indices
        unmatched: List of target_ids not found in child_ids_all
    """
    target_set = set(target_ids)
    all_set = set(child_ids_all)
    indices = [i for i, cid in enumerate(child_ids_all) if cid in target_set]
    unmatched = [tid for tid in target_ids if tid not in all_set]
    return indices, unmatched


def load_feature_cols_mapping(path: str) -> Optional[Dict[str, List[str]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--feature_cols_json not found: {p}")
    with p.open("r") as f:
        mapping = json.load(f)
    # Expect keys: joint, imit, free
    for k in ("joint", "imit", "free"):
        if k not in mapping:
            raise ValueError(f"feature_cols_json missing key '{k}'")
    return mapping


def load_feature_cols_from_f1_metrics(
    models_dir: Path, task: str, model_name: str
) -> Optional[List[str]]:
    """
    Load feature_cols from F1 metrics JSON to ensure exact feature alignment.
    Returns None if metrics file doesn't exist or doesn't have feature_cols.
    """
    metrics_path = models_dir / f"{task}_{model_name}_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r") as f:
            metrics = json.load(f)
        feature_cols = metrics.get("feature_cols")
        if feature_cols and isinstance(feature_cols, list):
            return feature_cols
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def select_task_columns(
    df: pd.DataFrame,
    task: str,
    models_dir: Path,
    model_name: str,
    explicit: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Select feature columns for a task. Priority:
    1. Explicit mapping from --feature_cols_json
    2. feature_cols from F1 metrics JSON (ensures exact alignment with trained models)
    3. Fallback to prefix heuristic (with warning)
    """
    # Priority 1: Explicit mapping
    if explicit is not None:
        cols = explicit[task]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for task={task}: {missing[:10]}{'...' if len(missing)>10 else ''}")
        return cols

    # Priority 2: Load from F1 metrics JSON (recommended)
    f1_cols = load_feature_cols_from_f1_metrics(models_dir, task, model_name)
    if f1_cols is not None:
        missing = [c for c in f1_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature mismatch for task={task}: {len(missing)} columns from F1 metrics "
                f"not found in data. Examples: {missing[:5]}{'...' if len(missing)>5 else ''}"
            )
        return f1_cols

    # Priority 3: Fallback to prefix heuristic (with warning)
    prefixes = DEFAULT_TASK_PREFIXES.get(task)
    if not prefixes:
        raise ValueError(f"No prefixes configured for task={task}")

    cols = []
    for c in df.columns:
        if any(c.startswith(pref) for pref in prefixes):
            cols.append(c)

    if not cols:
        raise ValueError(
            f"No feature columns found for task={task} using prefixes={prefixes}. "
            f"Provide --feature_cols_json to specify exact columns."
        )

    print(f"  WARNING: Using prefix heuristic for {task} features ({len(cols)} cols). "
          f"Consider running F1 with metrics JSON for exact alignment.")
    return cols


def load_fold_model(models_dir: Path, task: str, model_name: str, fold: int):
    # Expected naming from F1:
    # data/derived/models/{task}_{model_name}_fold{fold}.joblib
    p = models_dir / f"{task}_{model_name}_fold{fold}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Missing fold model: {p}")
    return joblib.load(p)


def predict_proba_1(model, X: np.ndarray) -> np.ndarray:
    """
    Returns P(y=1). Works for sklearn + xgboost sklearn wrapper.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    # fallback: some models might have decision_function
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        # sigmoid
        return 1.0 / (1.0 + np.exp(-z))
    raise TypeError("Model has neither predict_proba nor decision_function")


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_hat = (p >= threshold).astype(int)
    out = {}
    # Protect against degenerate folds (all one class)
    try:
        out["auroc"] = float(roc_auc_score(y_true, p))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, p))
    except Exception:
        out["auprc"] = float("nan")
    out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))
    out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_hat))
    # Brier score: measures calibration (lower is better, 0 = perfect)
    try:
        out["brier"] = float(brier_score_loss(y_true, p))
    except Exception:
        out["brier"] = float("nan")
    return out


def mean_std(metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics], dtype=float)
        out[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return out


def find_optimal_threshold(y_true: np.ndarray, p: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """
    Find optimal threshold on OOF predictions.
    Returns (best_threshold, best_score).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        y_hat = (p >= t).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, y_hat, zero_division=0))
        elif metric == "balanced_acc":
            score = float(balanced_accuracy_score(y_true, y_hat))
        else:
            # For auroc/auprc, threshold doesn't matter (ranking metric)
            # So just use F1 for threshold selection
            score = float(f1_score(y_true, y_hat, zero_division=0))

        if score > best_score:
            best_score = score
            best_thresh = t

    return float(best_thresh), float(best_score)


def run_cv_with_config(
    Z: np.ndarray,
    y: np.ndarray,
    fold_indices: List[Tuple[List[int], List[int]]],
    C: float,
    class_weight: Optional[str],
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    scale_features: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]], List[float]]:
    """
    Run CV with specific config, return OOF predictions, mean/std metrics, and per-fold thresholds.

    Args:
        Z: Meta-feature matrix (OOF predictions from F1 models)
        y: Labels
        fold_indices: List of (train_idx, val_idx) tuples
        C: Regularization strength
        class_weight: Class weight strategy ("balanced" or None)
        seed: Random seed
        sample_weight: Optional per-sample weights for training
        scale_features: Whether to standardize meta-features
        verbose: Print per-fold metrics

    Returns:
        oof_preds: OOF predictions
        summary: Mean/std of metrics across folds
        fold_thresholds: Per-fold optimal thresholds (nested selection)
    """
    n = len(y)
    oof_preds = np.full(n, np.nan)
    fold_metrics = []
    fold_thresholds = []

    for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
        # Optional scaling (fit on train, transform both)
        if scale_features:
            scaler = StandardScaler()
            Z_train = scaler.fit_transform(Z[tr_idx])
            Z_val = scaler.transform(Z[va_idx])
        else:
            Z_train = Z[tr_idx]
            Z_val = Z[va_idx]

        model = LogisticRegression(
            C=C,
            solver="liblinear",
            class_weight=class_weight,
            max_iter=2000,
            random_state=seed,
        )
        if sample_weight is not None:
            sw_train = sample_weight[tr_idx]
            model.fit(Z_train, y[tr_idx], sample_weight=sw_train)
        else:
            model.fit(Z_train, y[tr_idx])

        p = model.predict_proba(Z_val)[:, 1]
        oof_preds[va_idx] = p

        # Nested threshold selection: find optimal threshold on TRAINING fold only
        p_train = model.predict_proba(Z_train)[:, 1]
        fold_thresh, _ = find_optimal_threshold(y[tr_idx], p_train, metric="f1")
        fold_thresholds.append(fold_thresh)

        # Compute metrics using the nested threshold
        metrics = compute_metrics(y[va_idx], p, threshold=fold_thresh)
        fold_metrics.append(metrics)

        if verbose:
            print(f"    Fold {fold_i}: auroc={metrics['auroc']:.3f}, f1={metrics['f1']:.3f}, "
                  f"brier={metrics['brier']:.3f}, thresh={fold_thresh:.3f}")

    summary = mean_std(fold_metrics)
    return oof_preds, summary, fold_thresholds


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    data_csv = Path(args.data_csv)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)

    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not found in data")
    if df[args.id_col].astype(str).duplicated().any():
        raise ValueError(f"Duplicate ids found in {args.id_col}; stacking expects one row per child.")
    if args.label_col not in df.columns:
        raise ValueError(f"label_col '{args.label_col}' not found in data")

    # Ensure label is 0/1 ints
    y = df[args.label_col].astype(int).to_numpy()
    # Convert to string for consistent matching with splits.json and output
    child_ids_all = df[args.id_col].astype(str).tolist()

    # Compute sample weights if quality column specified
    sample_weight: Optional[np.ndarray] = None
    if args.quality_col:
        if args.quality_col not in df.columns:
            raise ValueError(
                f"Quality column '{args.quality_col}' not found. "
                f"Available columns: {[c for c in df.columns if c.startswith('qc_')]}"
            )
        # Fill NaN with 0.5 (neutral), clip to [floor, 1.0]
        sample_weight = (
            df[args.quality_col]
            .fillna(0.5)
            .clip(args.quality_floor, 1.0)
            .to_numpy()
        )
        print(f"Using sample weights from '{args.quality_col}' (floor={args.quality_floor})")
        print(f"  Weight stats: min={sample_weight.min():.3f}, max={sample_weight.max():.3f}, "
              f"mean={sample_weight.mean():.3f}")

    # Load folds from splits.json (must match F1 training)
    splits_path = Path(args.splits)
    folds = load_splits(splits_path)
    n_folds = len(folds)

    feature_map = load_feature_cols_mapping(args.feature_cols_json)

    # Select per-task feature columns (prefer F1 metrics JSON for exact alignment)
    print("\nLoading feature columns...")
    joint_cols = select_task_columns(df, "joint", models_dir, args.joint_base, feature_map)
    print(f"  joint: {len(joint_cols)} features")
    imit_cols = select_task_columns(df, "imit", models_dir, args.imit_base, feature_map)
    print(f"  imit:  {len(imit_cols)} features")
    free_cols = select_task_columns(df, "free", models_dir, args.free_base, feature_map)
    print(f"  free:  {len(free_cols)} features")

    X_joint = df[joint_cols].to_numpy()
    X_imit  = df[imit_cols].to_numpy()
    X_free  = df[free_cols].to_numpy()

    # Prepare OOF containers
    p_joint = np.full(shape=(len(df),), fill_value=np.nan, dtype=float)
    p_imit  = np.full(shape=(len(df),), fill_value=np.nan, dtype=float)
    p_free  = np.full(shape=(len(df),), fill_value=np.nan, dtype=float)

    p_joint_xgb = None
    if args.include_joint_xgb:
        p_joint_xgb = np.full(shape=(len(df),), fill_value=np.nan, dtype=float)

    # Store fold indices for later use (meta-model CV)
    fold_indices: List[Tuple[List[int], List[int]]] = []

    print(f"\nStacking F2: generating OOF predictions with {n_folds}-fold CV (from splits.json)...")
    total_unmatched_train = 0
    total_unmatched_val = 0

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        tr_idx, unmatched_train = get_indices_for_ids(child_ids_all, fold_data["train"])
        va_idx, unmatched_val = get_indices_for_ids(child_ids_all, fold_data["val"])
        fold_indices.append((tr_idx, va_idx))

        total_unmatched_train += len(unmatched_train)
        total_unmatched_val += len(unmatched_val)

        if len(va_idx) == 0:
            raise ValueError(
                f"Fold {fold_idx} has no validation samples after ID matching. "
                f"Unmatched val IDs: {unmatched_val[:10]}{'...' if len(unmatched_val)>10 else ''}"
            )
        if len(tr_idx) == 0:
            raise ValueError(
                f"Fold {fold_idx} has no training samples after ID matching. "
                f"Unmatched train IDs: {unmatched_train[:10]}{'...' if len(unmatched_train)>10 else ''}"
            )

        # Load fold models trained in F1
        m_joint = load_fold_model(models_dir, "joint", args.joint_base, fold_idx)
        m_imit  = load_fold_model(models_dir, "imit",  args.imit_base,  fold_idx)
        m_free  = load_fold_model(models_dir, "free",  args.free_base,  fold_idx)

        p_joint[va_idx] = predict_proba_1(m_joint, X_joint[va_idx])
        p_imit[va_idx]  = predict_proba_1(m_imit,  X_imit[va_idx])
        p_free[va_idx]  = predict_proba_1(m_free,  X_free[va_idx])

        if args.include_joint_xgb:
            m_joint_xgb = load_fold_model(models_dir, "joint", "xgb", fold_idx)
            p_joint_xgb[va_idx] = predict_proba_1(m_joint_xgb, X_joint[va_idx])

        # quick fold metrics (base preds)
        mj = compute_metrics(y[va_idx], p_joint[va_idx])
        mi = compute_metrics(y[va_idx], p_imit[va_idx])
        mf = compute_metrics(y[va_idx], p_free[va_idx])
        print(
            f"  Fold {fold_idx}: "
            f"joint(auroc={mj['auroc']:.3f}, f1={mj['f1']:.3f}) | "
            f"imit(auroc={mi['auroc']:.3f}, f1={mi['f1']:.3f}) | "
            f"free(auroc={mf['auroc']:.3f}, f1={mf['f1']:.3f})"
        )

    # Report any ID matching issues
    if total_unmatched_train > 0 or total_unmatched_val > 0:
        print(f"  WARNING: {total_unmatched_train} train IDs and {total_unmatched_val} val IDs "
              f"from splits.json not found in data CSV")

    # Sanity: no NaNs left (with detailed diagnostics)
    nan_issues = []
    if np.isnan(p_joint).any():
        nan_count = np.isnan(p_joint).sum()
        nan_issues.append(f"p_joint has {nan_count} NaNs")
    if np.isnan(p_imit).any():
        nan_count = np.isnan(p_imit).sum()
        nan_issues.append(f"p_imit has {nan_count} NaNs")
    if np.isnan(p_free).any():
        nan_count = np.isnan(p_free).sum()
        nan_issues.append(f"p_free has {nan_count} NaNs")

    if nan_issues:
        # Find which samples have NaNs
        nan_mask = np.isnan(p_joint) | np.isnan(p_imit) | np.isnan(p_free)
        nan_ids = [child_ids_all[i] for i in np.where(nan_mask)[0][:10]]
        raise RuntimeError(
            f"OOF predictions contain NaNs: {', '.join(nan_issues)}. "
            f"This usually means some samples weren't in any validation fold. "
            f"Sample IDs with NaNs: {nan_ids}{'...' if nan_mask.sum()>10 else ''}"
        )

    # Parse excluded tasks
    exclude_set = set(t.strip() for t in args.exclude_tasks.split(",") if t.strip())

    # Build meta-feature matrix (respecting exclusions)
    meta_cols: List[str] = []
    meta_arrays: List[np.ndarray] = []

    if "joint" not in exclude_set:
        meta_cols.append(f"P_joint_{args.joint_base}")
        meta_arrays.append(p_joint)
        if args.add_confidence:
            meta_cols.append("conf_joint")
            meta_arrays.append(np.abs(p_joint - 0.5))

    if "imit" not in exclude_set:
        meta_cols.append(f"P_imit_{args.imit_base}")
        meta_arrays.append(p_imit)
        if args.add_confidence:
            meta_cols.append("conf_imit")
            meta_arrays.append(np.abs(p_imit - 0.5))

    if "free" not in exclude_set:
        meta_cols.append(f"P_free_{args.free_base}")
        meta_arrays.append(p_free)
        if args.add_confidence:
            meta_cols.append("conf_free")
            meta_arrays.append(np.abs(p_free - 0.5))

    if args.include_joint_xgb and "joint" not in exclude_set:
        if np.isnan(p_joint_xgb).any():
            raise RuntimeError("P_joint_xgb contains NaNs; check joint_xgb models exist and folds match.")
        meta_cols.append("P_joint_xgb")
        meta_arrays.append(p_joint_xgb)
        if args.add_confidence:
            meta_cols.append("conf_joint_xgb")
            meta_arrays.append(np.abs(p_joint_xgb - 0.5))

    if len(meta_arrays) == 0:
        raise ValueError("All tasks excluded! Nothing to stack.")

    Z = np.column_stack(meta_arrays)
    print(f"\nMeta-features ({len(meta_cols)}): {meta_cols}")

    # Build grid for C and class_weight
    if args.meta_C_grid:
        C_values = [float(x.strip()) for x in args.meta_C_grid.split(",")]
    else:
        C_values = [args.meta_C]

    if args.meta_class_weight_grid:
        cw_values = [x.strip() for x in args.meta_class_weight_grid.split(",")]
    else:
        cw_values = [args.meta_class_weight]

    # Grid search
    best_score = -1.0
    best_config: Dict = {}
    best_oof_preds: Optional[np.ndarray] = None
    best_summary: Dict = {}
    best_fold_thresholds: List[float] = []

    print(f"\nGrid search: {len(C_values)} C values x {len(cw_values)} class_weight options")
    print(f"Optimizing for: {args.select_metric}")
    if args.scale_meta_features:
        print("Meta-feature scaling: ENABLED")
    print()

    for C in C_values:
        for cw in cw_values:
            cw_parsed = None if cw == "none" else cw
            oof_preds, summary, fold_thresholds = run_cv_with_config(
                Z, y, fold_indices, C, cw_parsed, args.seed, sample_weight,
                scale_features=args.scale_meta_features,
                verbose=False,  # Set True for debugging
            )

            score = summary[args.select_metric][0]  # mean
            print(f"  C={C:<6} cw={cw:<10} => {args.select_metric}={score:.3f} (±{summary[args.select_metric][1]:.3f})")

            if score > best_score:
                best_score = score
                best_config = {"C": C, "class_weight": cw}
                best_oof_preds = oof_preds
                best_summary = summary
                best_fold_thresholds = fold_thresholds

    print(f"\nBest config: C={best_config['C']}, class_weight={best_config['class_weight']}")
    print(f"Best {args.select_metric}: {best_score:.3f}")

    # Use average of nested per-fold thresholds (unbiased)
    nested_threshold = float(np.mean(best_fold_thresholds))
    nested_threshold_std = float(np.std(best_fold_thresholds))
    print(f"\nNested threshold (avg of per-fold): {nested_threshold:.3f} ± {nested_threshold_std:.3f}")
    print(f"  Per-fold thresholds: {[f'{t:.3f}' for t in best_fold_thresholds]}")

    # Final model with best config
    best_cw = None if best_config["class_weight"] == "none" else best_config["class_weight"]
    meta = LogisticRegression(
        C=best_config["C"],
        solver="liblinear",
        class_weight=best_cw,
        max_iter=2000,
        random_state=args.seed,
    )

    print("\nBest config CV metrics (mean ± std):")
    print(f"  AUROC:  {best_summary['auroc'][0]:.3f} ± {best_summary['auroc'][1]:.3f}")
    print(f"  AUPRC:  {best_summary['auprc'][0]:.3f} ± {best_summary['auprc'][1]:.3f}")
    print(f"  F1:     {best_summary['f1'][0]:.3f} ± {best_summary['f1'][1]:.3f}")
    print(f"  BalAcc: {best_summary['balanced_acc'][0]:.3f} ± {best_summary['balanced_acc'][1]:.3f}")
    print(f"  Brier:  {best_summary['brier'][0]:.3f} ± {best_summary['brier'][1]:.3f} (lower is better)")

    # Fit final meta-model on full OOF Z (no leakage because Z is OOF)
    # Apply scaling if enabled (same as during CV)
    final_scaler: Optional[StandardScaler] = None
    Z_final = Z
    if args.scale_meta_features:
        final_scaler = StandardScaler()
        Z_final = final_scaler.fit_transform(Z)

    if sample_weight is not None:
        meta.fit(Z_final, y, sample_weight=sample_weight)
    else:
        meta.fit(Z_final, y)

    # Save artifacts
    out_model = out_dir / "stack_meta_logreg.joblib"
    joblib.dump(meta, out_model)

    out_scaler = None
    if final_scaler is not None:
        out_scaler = out_dir / "stack_meta_scaler.joblib"
        joblib.dump(final_scaler, out_scaler)

    # Use string IDs consistently for output
    oof_df = pd.DataFrame({
        args.id_col: child_ids_all,  # Use string IDs consistently
        args.label_col: y,
        "P_meta": best_oof_preds,
        **{c: Z[:, i] for i, c in enumerate(meta_cols)},
    })
    out_oof = out_dir / "stack_oof_predictions.csv"
    oof_df.to_csv(out_oof, index=False)

    weights = {
        "intercept": float(meta.intercept_[0]),
        "coef": {meta_cols[i]: float(meta.coef_[0, i]) for i in range(len(meta_cols))},
        "meta_cols": meta_cols,
        "meta_C": best_config["C"],
        "meta_class_weight": best_config["class_weight"],
        "scale_meta_features": args.scale_meta_features,
        "threshold_nested_mean": nested_threshold,
        "threshold_nested_std": nested_threshold_std,
        "threshold_per_fold": best_fold_thresholds,
        "n_folds": n_folds,
        "splits_path": str(args.splits),
        "seed": args.seed,
        "excluded_tasks": list(exclude_set) if exclude_set else [],
        "add_confidence": args.add_confidence,
        "select_metric": args.select_metric,
        "sample_weight_col": args.quality_col,
        "sample_weight_floor": args.quality_floor if args.quality_col else None,
        "metrics_cv_mean_std": {k: {"mean": v[0], "std": v[1]} for k, v in best_summary.items()},
        "base_model_feature_cols": {
            "joint": joint_cols,
            "imit": imit_cols,
            "free": free_cols,
        },
    }
    out_json = out_dir / "stack_meta_logreg_info.json"
    with out_json.open("w") as f:
        json.dump(weights, f, indent=2)

    print("\nSaved:")
    print(f"  Meta model: {out_model}")
    if out_scaler is not None:
        print(f"  Scaler:     {out_scaler}")
    print(f"  OOF preds:  {out_oof}")
    print(f"  Info JSON:  {out_json}")
    print("\nMeta coefficients (higher = more weight):")
    for k, v in weights["coef"].items():
        print(f"  {k}: {v:+.4f}")
    print(f"\nRecommended threshold: {nested_threshold:.3f} (nested CV, unbiased)")


if __name__ == "__main__":
    main()
