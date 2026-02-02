from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import (  # type: ignore
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
)


@dataclass
class EvalMetrics:
    n: int
    pos_rate: float
    auroc: Optional[float]
    auprc: Optional[float]
    accuracy: float
    balanced_accuracy: float
    f1: float


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    # AUROC undefined if only one class present in y_true
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> EvalMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    return EvalMetrics(
        n=int(len(y_true)),
        pos_rate=float(np.mean(y_true)) if len(y_true) else 0.0,
        auroc=_safe_auroc(y_true, y_prob),
        auprc=_safe_auprc(y_true, y_prob),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def aggregate_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and std of metrics across folds."""
    keys = ["auroc", "auprc", "accuracy", "balanced_accuracy", "f1"]
    agg: Dict[str, Any] = {"n_folds": len(fold_metrics)}

    for key in keys:
        values = [fm[key] for fm in fold_metrics if fm.get(key) is not None]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None

    # Also aggregate n and pos_rate
    agg["n_total"] = sum(fm["n"] for fm in fold_metrics)
    agg["pos_rate_mean"] = float(np.mean([fm["pos_rate"] for fm in fold_metrics]))

    return agg


def load_splits(splits_path: Path) -> Dict[str, Any]:
    """
    Load splits from JSON. Supports both holdout and k-fold formats from split.py.

    Returns:
        For holdout: {"mode": "holdout", "train": [...], "val": [...], "test": [...]}
        For kfold: {"mode": "kfold", "folds": [{"fold": 0, "train": [...], "val": [...]}, ...]}
    """
    with open(splits_path, "r") as f:
        data = json.load(f)

    # Detect k-fold mode
    if "folds" in data and isinstance(data["folds"], list):
        folds = []
        for fold_data in data["folds"]:
            folds.append({
                "fold": int(fold_data["fold"]),
                "train": [str(x) for x in fold_data["train"]],
                "val": [str(x) for x in fold_data["val"]],
            })
        return {"mode": "kfold", "folds": folds}

    # Handle holdout mode (nested under "split" key)
    if "split" in data and isinstance(data["split"], dict):
        splits = data["split"]
    else:
        splits = data

    # expected keys: train/val/test each list of child_ids (as str or int)
    for k in ("train", "val", "test"):
        if k not in splits:
            raise ValueError(f"splits.json missing key '{k}'. Keys: {list(splits.keys())}")

    return {
        "mode": "holdout",
        "train": [str(x) for x in splits["train"]],
        "val": [str(x) for x in splits["val"]],
        "test": [str(x) for x in splits["test"]],
    }


def load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    if "child_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Labels CSV must have columns child_id,label. Got: {list(df.columns)}")
    df = df[["child_id", "label"]].copy()
    df["child_id"] = df["child_id"].astype(str)
    # ensure 0/1 ints
    df["label"] = df["label"].astype(int)
    return df


def select_task_features(features_df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    task: joint | imit | free
    """
    prefix_map = {
        "joint": "ja_",
        "imit": "imit_",
        "free": "fp_",
    }
    if task not in prefix_map:
        raise ValueError(f"Unknown task={task}. Choose from {list(prefix_map.keys())}")

    prefix = prefix_map[task]
    # keep only prefixed feature columns; always keep child_id for merges
    cols = [c for c in features_df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix '{prefix}'. Available columns: {len(features_df.columns)}")

    # (Optional) drop obvious non-feature columns if any sneak in
    # In your merged file, child_id is separate, so we're good.

    X = features_df[cols].copy()
    return X, cols


def build_model(model_type: str, random_state: int) -> Pipeline:
    if model_type == "logreg":
        # Strong baseline: imputing + scaling + L2 logistic
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                )),
            ]
        )

    if model_type == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore[import-not-found]
        except Exception as e:
            raise ImportError(
                "XGBoost not available. Install with `pip install xgboost` (or add to your env)."
            ) from e

        # Good default for small/medium tabular dataset; tune later
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", xgb),
            ]
        )

    raise ValueError("model_type must be one of: logreg, xgb")


def _get_indices_for_ids(child_ids_all: List[str], target_ids: List[str]) -> List[int]:
    """Get indices of child_ids_all that are in target_ids."""
    target_set = set(target_ids)
    return [i for i, cid in enumerate(child_ids_all) if cid in target_set]


def _predict_prob(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Get predicted probabilities for class 1."""
    return model.predict_proba(X)[:, 1]


def _train_holdout(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    child_ids_all: List[str],
    splits: Dict[str, Any],
    args: argparse.Namespace,
    feature_cols: List[str],
    out_dir: Path,
) -> None:
    """Train and evaluate using holdout split."""
    train_idx = _get_indices_for_ids(child_ids_all, splits["train"])
    val_idx = _get_indices_for_ids(child_ids_all, splits["val"])
    test_idx = _get_indices_for_ids(child_ids_all, splits["test"])

    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(idxs) == 0:
            raise ValueError(
                f"Split '{name}' has 0 labeled children after filtering. "
                f"Fix splits.json or labels/child_id formatting."
            )

    X_train = X_all.iloc[train_idx].to_numpy()
    y_train = y_all[train_idx]
    X_val = X_all.iloc[val_idx].to_numpy()
    y_val = y_all[val_idx]
    X_test = X_all.iloc[test_idx].to_numpy()
    y_test = y_all[test_idx]

    model = build_model(args.model, random_state=args.random_state)
    model.fit(X_train, y_train)

    p_train = _predict_prob(model, X_train)
    p_val = _predict_prob(model, X_val)
    p_test = _predict_prob(model, X_test)

    metrics = {
        "task": args.task,
        "model": args.model,
        "mode": "holdout",
        "n_features": int(len(feature_cols)),
        "features_prefix": {"joint": "ja_", "imit": "imit_", "free": "fp_"}[args.task],
        "threshold": float(args.threshold),
        "splits": {
            "train": asdict(evaluate_binary(y_train, p_train, threshold=args.threshold)),
            "val": asdict(evaluate_binary(y_val, p_val, threshold=args.threshold)),
            "test": asdict(evaluate_binary(y_test, p_test, threshold=args.threshold)),
        },
        "feature_cols": feature_cols,
    }

    tag = f"{args.task}_{args.model}"
    model_path = out_dir / f"{tag}.joblib"
    metrics_path = out_dir / f"{tag}_metrics.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nVAL metrics:")
    print(json.dumps(metrics["splits"]["val"], indent=2))


def _train_kfold(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    child_ids_all: List[str],
    splits: Dict[str, Any],
    args: argparse.Namespace,
    feature_cols: List[str],
    out_dir: Path,
) -> None:
    """Train and evaluate using k-fold cross-validation."""
    folds = splits["folds"]
    n_folds = len(folds)
    tag = f"{args.task}_{args.model}"

    fold_val_metrics: List[Dict[str, Any]] = []
    fold_train_metrics: List[Dict[str, Any]] = []

    print(f"Running {n_folds}-fold cross-validation...")

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx = _get_indices_for_ids(child_ids_all, fold_data["train"])
        val_idx = _get_indices_for_ids(child_ids_all, fold_data["val"])

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {fold_idx} has empty train or val split after filtering.")

        X_train = X_all.iloc[train_idx].to_numpy()
        y_train = y_all[train_idx]
        X_val = X_all.iloc[val_idx].to_numpy()
        y_val = y_all[val_idx]

        model = build_model(args.model, random_state=args.random_state + fold_idx)
        model.fit(X_train, y_train)

        p_train = _predict_prob(model, X_train)
        p_val = _predict_prob(model, X_val)

        train_metrics = asdict(evaluate_binary(y_train, p_train, threshold=args.threshold))
        val_metrics = asdict(evaluate_binary(y_val, p_val, threshold=args.threshold))

        fold_train_metrics.append(train_metrics)
        fold_val_metrics.append(val_metrics)

        # Save per-fold model
        fold_model_path = out_dir / f"{tag}_fold{fold_idx}.joblib"
        joblib.dump(model, fold_model_path)

        auroc_str = f"{val_metrics['auroc']:.3f}" if val_metrics['auroc'] is not None else "N/A"
        print(f"  Fold {fold_idx}: val_auroc={auroc_str}, val_f1={val_metrics['f1']:.3f}")

    # Aggregate metrics
    agg_val = aggregate_fold_metrics(fold_val_metrics)
    agg_train = aggregate_fold_metrics(fold_train_metrics)

    metrics = {
        "task": args.task,
        "model": args.model,
        "mode": "kfold",
        "n_folds": n_folds,
        "n_features": int(len(feature_cols)),
        "features_prefix": {"joint": "ja_", "imit": "imit_", "free": "fp_"}[args.task],
        "threshold": float(args.threshold),
        "cv_summary": {
            "train": agg_train,
            "val": agg_val,
        },
        "per_fold": {
            "train": fold_train_metrics,
            "val": fold_val_metrics,
        },
        "feature_cols": feature_cols,
    }

    metrics_path = out_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {n_folds} fold models: {out_dir}/{tag}_fold*.joblib")
    print(f"Saved metrics: {metrics_path}")
    print("\nCV VAL metrics (mean ± std):")
    print(f"  AUROC: {agg_val['auroc_mean']:.3f} ± {agg_val['auroc_std']:.3f}" if agg_val['auroc_mean'] else "  AUROC: N/A")
    print(f"  AUPRC: {agg_val['auprc_mean']:.3f} ± {agg_val['auprc_std']:.3f}" if agg_val['auprc_mean'] else "  AUPRC: N/A")
    print(f"  F1:    {agg_val['f1_mean']:.3f} ± {agg_val['f1_std']:.3f}")
    print(f"  Balanced Acc: {agg_val['balanced_accuracy_mean']:.3f} ± {agg_val['balanced_accuracy_std']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a per-task model (joint/imit/free).")
    ap.add_argument("--features", required=True, help="Path to features_merged.csv (or parquet).")
    ap.add_argument("--labels", required=True, help="Path to labels_usable.csv (or labels_clean.csv).")
    ap.add_argument("--splits", required=True, help="Path to data/derived/splits.json (holdout or kfold)")
    ap.add_argument("--task", required=True, choices=["joint", "imit", "free"], help="Which task feature set to use.")
    ap.add_argument("--model", required=True, choices=["logreg", "xgb"], help="Model type.")
    ap.add_argument("--out_dir", default="data/derived/models", help="Directory to save model + metrics.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for metrics.")
    ap.add_argument("--random_state", type=int, default=1337, help="Random seed.")
    args = ap.parse_args()

    features_path = Path(args.features)
    labels_path = Path(args.labels)
    splits_path = Path(args.splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if features_path.suffix.lower() == ".parquet":
        feat_df = pd.read_parquet(features_path)
    else:
        feat_df = pd.read_csv(features_path)

    if "child_id" not in feat_df.columns:
        raise ValueError(f"features file missing child_id. Columns: {list(feat_df.columns)}")
    feat_df["child_id"] = feat_df["child_id"].astype(str)

    labels_df = load_labels(labels_path)
    splits = load_splits(splits_path)

    # Filter to labeled children only
    labeled_ids = set(labels_df["child_id"].tolist())
    feat_df = feat_df[feat_df["child_id"].isin(labeled_ids)].copy()

    # Merge labels
    df = feat_df.merge(labels_df, on="child_id", how="inner")
    if df.empty:
        raise ValueError("No rows left after merging features with labels. Check child_id formats.")

    # Select task features
    X_all, feature_cols = select_task_features(df, task=args.task)
    y_all = df["label"].to_numpy().astype(int)
    child_ids_all = df["child_id"].tolist()

    print(f"Task: {args.task}, Model: {args.model}")
    print(f"Samples: {len(df)}, Features: {len(feature_cols)}")
    print(f"Mode: {splits['mode']}")

    if splits["mode"] == "kfold":
        _train_kfold(X_all, y_all, child_ids_all, splits, args, feature_cols, out_dir)
    else:
        _train_holdout(X_all, y_all, child_ids_all, splits, args, feature_cols, out_dir)


if __name__ == "__main__":
    main()
