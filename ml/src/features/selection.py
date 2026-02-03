"""
Feature selection module for improving model performance.

Provides multiple feature selection strategies:
- Recursive Feature Elimination (RFE) with XGBoost
- SelectKBest with mutual information
- Variance threshold filtering
- Correlation-based redundancy removal
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import (  # type: ignore[import-untyped]
    RFE,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.model_selection import cross_val_score, StratifiedKFold  # type: ignore[import-untyped]


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    method: str
    n_original: int
    n_selected: int
    selected_features: List[str]
    feature_scores: Dict[str, float]
    cv_auroc_before: Optional[float]
    cv_auroc_after: Optional[float]


def _impute_data(X: np.ndarray) -> np.ndarray:
    """Impute missing values with median."""
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X)


def _drop_all_nan_columns(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns that are entirely NaN."""
    valid_cols = [col for col in X.columns if X[col].notna().any()]
    dropped = [col for col in X.columns if col not in valid_cols]
    if dropped:
        print(f"  Dropped {len(dropped)} all-NaN columns: {dropped[:5]}{'...' if len(dropped) > 5 else ''}")
    return X[valid_cols], dropped


def select_by_variance(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with low variance.

    Args:
        X: Feature DataFrame
        threshold: Minimum variance threshold (default: 0.01)

    Returns:
        Filtered DataFrame and list of selected feature names
    """
    # First drop all-NaN columns
    X_valid, _ = _drop_all_nan_columns(X)
    if X_valid.empty:
        return X_valid, []

    X_imputed = _impute_data(X_valid.values)

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_imputed)

    selected_mask = selector.get_support()
    selected_cols = [col for col, keep in zip(X_valid.columns, selected_mask) if keep]

    return X[selected_cols], selected_cols


def select_by_correlation(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Remove highly correlated features to reduce redundancy.

    Args:
        X: Feature DataFrame
        threshold: Correlation threshold for removal (default: 0.95)

    Returns:
        Filtered DataFrame, selected features, and removed features
    """
    # First drop all-NaN columns
    X_valid, _ = _drop_all_nan_columns(X)
    if X_valid.empty:
        return X_valid, [], list(X.columns)

    X_imputed = pd.DataFrame(_impute_data(X_valid.values), columns=X_valid.columns)

    corr_matrix = X_imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected_cols = [col for col in X.columns if col not in to_drop]

    return X[selected_cols], selected_cols, to_drop


def select_kbest_mi(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int = 20,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """
    Select top-k features using mutual information.

    Args:
        X: Feature DataFrame
        y: Binary labels
        k: Number of features to select
        random_state: Random seed for reproducibility

    Returns:
        FeatureSelectionResult with selected features and scores
    """
    X_imputed = _impute_data(X.values)

    # Compute mutual information scores
    mi_scores = mutual_info_classif(
        X_imputed, y,
        discrete_features=False,
        random_state=random_state,
    )

    # Create score dict
    score_dict = dict(zip(X.columns, mi_scores))
    score_dict = dict(sorted(score_dict.items(), key=lambda x: -x[1]))

    # Select top-k
    selector = SelectKBest(mutual_info_classif, k=min(k, len(X.columns)))
    selector.fit(X_imputed, y)

    selected_mask = selector.get_support()
    selected_cols = [col for col, keep in zip(X.columns, selected_mask) if keep]

    return FeatureSelectionResult(
        method="selectkbest_mi",
        n_original=len(X.columns),
        n_selected=len(selected_cols),
        selected_features=selected_cols,
        feature_scores=score_dict,
        cv_auroc_before=None,
        cv_auroc_after=None,
    )


def select_rfe_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    n_features_to_select: int = 15,
    step: int = 1,
    random_state: int = 42,
    cv_folds: int = 3,
) -> FeatureSelectionResult:
    """
    Select features using Recursive Feature Elimination with XGBoost.

    Args:
        X: Feature DataFrame
        y: Binary labels
        n_features_to_select: Target number of features
        step: Number of features to remove at each iteration
        random_state: Random seed
        cv_folds: Number of CV folds for evaluation

    Returns:
        FeatureSelectionResult with selected features and rankings
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError("XGBoost required for RFE. Install with: pip install xgboost") from e

    X_imputed = _impute_data(X.values)

    # Compute scale_pos_weight
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Base estimator
    base_estimator = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )

    # RFE
    rfe = RFE(
        estimator=base_estimator,
        n_features_to_select=min(n_features_to_select, len(X.columns)),
        step=step,
    )
    rfe.fit(X_imputed, y)

    selected_mask = rfe.support_
    selected_cols = [col for col, keep in zip(X.columns, selected_mask) if keep]

    # Ranking (lower is better)
    ranking = dict(zip(X.columns, rfe.ranking_.tolist()))
    # Convert to scores (higher is better)
    max_rank = max(ranking.values())
    scores = {col: float(max_rank - rank + 1) for col, rank in ranking.items()}
    scores = dict(sorted(scores.items(), key=lambda x: -x[1]))

    # Evaluate AUROC before/after
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    auroc_before = cross_val_score(
        base_estimator, X_imputed, y, cv=cv, scoring="roc_auc"
    ).mean()

    X_selected = X_imputed[:, selected_mask]
    auroc_after = cross_val_score(
        base_estimator, X_selected, y, cv=cv, scoring="roc_auc"
    ).mean()

    return FeatureSelectionResult(
        method="rfe_xgb",
        n_original=len(X.columns),
        n_selected=len(selected_cols),
        selected_features=selected_cols,
        feature_scores=scores,
        cv_auroc_before=float(auroc_before),
        cv_auroc_after=float(auroc_after),
    )


def combined_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    n_features: int = 15,
    method: str = "rfe",  # "rfe" or "mi"
    random_state: int = 42,
) -> Tuple[pd.DataFrame, FeatureSelectionResult]:
    """
    Combined feature selection pipeline:
    1. Drop all-NaN columns
    2. Remove low-variance features
    3. Remove highly correlated features
    4. Select top features using RFE or mutual information

    Args:
        X: Feature DataFrame
        y: Binary labels
        variance_threshold: Minimum variance
        correlation_threshold: Maximum correlation allowed
        n_features: Target number of features
        method: Selection method ("rfe" or "mi")
        random_state: Random seed

    Returns:
        Filtered DataFrame and selection result
    """
    print(f"  Starting with {len(X.columns)} features")

    # Step 0: Drop all-NaN columns
    X, nan_dropped = _drop_all_nan_columns(X)
    if X.empty:
        raise ValueError("All features are entirely NaN")
    print(f"  After dropping all-NaN: {len(X.columns)} features")

    # Step 1: Variance filter
    X_var, var_cols = select_by_variance(X, threshold=variance_threshold)
    print(f"  After variance filter: {len(var_cols)} features")

    # Step 2: Correlation filter
    X_corr, corr_cols, dropped = select_by_correlation(X_var, threshold=correlation_threshold)
    print(f"  After correlation filter: {len(corr_cols)} features (removed {len(dropped)})")

    # Step 3: Feature selection
    if method == "rfe":
        result = select_rfe_xgb(X_corr, y, n_features_to_select=n_features, random_state=random_state)
    else:
        result = select_kbest_mi(X_corr, y, k=n_features, random_state=random_state)

    print(f"  After {method} selection: {result.n_selected} features")
    if result.cv_auroc_before is not None:
        print(f"  CV AUROC: {result.cv_auroc_before:.4f} -> {result.cv_auroc_after:.4f}")

    return X[result.selected_features], result


def save_selected_features(
    features_path: Path,
    output_path: Path,
    selected_cols: List[str],
    child_id_col: str = "child_id",
) -> None:
    """
    Save a filtered features CSV with only selected columns.

    Args:
        features_path: Path to original features CSV
        output_path: Path to save filtered features
        selected_cols: List of feature columns to keep
        child_id_col: Name of child ID column (always preserved)
    """
    if features_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path)

    cols_to_keep = [child_id_col] + [c for c in selected_cols if c in df.columns]
    df_filtered = df[cols_to_keep]

    if output_path.suffix.lower() == ".parquet":
        df_filtered.to_parquet(output_path, index=False)
    else:
        df_filtered.to_csv(output_path, index=False)

    print(f"Saved {len(selected_cols)} selected features to {output_path}")


def main() -> None:
    """CLI for feature selection."""
    ap = argparse.ArgumentParser(description="Feature selection for task models.")
    ap.add_argument("--features", required=True, help="Path to features CSV/parquet")
    ap.add_argument("--labels", required=True, help="Path to labels CSV")
    ap.add_argument("--task", required=True, choices=["joint", "imit", "free"])
    ap.add_argument("--method", default="rfe", choices=["rfe", "mi"])
    ap.add_argument("--n_features", type=int, default=15, help="Number of features to select")
    ap.add_argument("--variance_threshold", type=float, default=0.01)
    ap.add_argument("--correlation_threshold", type=float, default=0.95)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_dir", default="data/derived/feature_selection")
    ap.add_argument("--save_filtered", action="store_true", help="Save filtered features CSV")
    args = ap.parse_args()

    # Load data
    features_path = Path(args.features)
    if features_path.suffix.lower() == ".parquet":
        feat_df = pd.read_parquet(features_path)
    else:
        feat_df = pd.read_csv(features_path)

    labels_df = pd.read_csv(args.labels)

    # Normalize child_id
    feat_df["child_id"] = feat_df["child_id"].astype(str)
    labels_df["child_id"] = labels_df["child_id"].astype(str)

    # Get label column
    label_col = "label" if "label" in labels_df.columns else "asd_label"

    # Merge
    df = feat_df.merge(labels_df[["child_id", label_col]], on="child_id", how="inner")
    df = df.rename(columns={label_col: "label"})

    # Select task features
    prefix_map = {"joint": "ja_", "imit": "imit_", "free": "fp_"}
    prefix = prefix_map[args.task]
    feature_cols = [c for c in df.columns if c.startswith(prefix)]

    print(f"Task: {args.task}")
    print(f"Method: {args.method}")
    print(f"Target features: {args.n_features}")

    X = df[feature_cols]
    y = np.asarray(df["label"].values, dtype=int)

    # Run selection
    X_selected, result = combined_selection(
        X, y,
        variance_threshold=args.variance_threshold,
        correlation_threshold=args.correlation_threshold,
        n_features=args.n_features,
        method=args.method,
        random_state=args.random_state,
    )

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "task": args.task,
        "method": result.method,
        "n_original": result.n_original,
        "n_selected": result.n_selected,
        "selected_features": result.selected_features,
        "feature_scores_top20": dict(list(result.feature_scores.items())[:20]),
        "cv_auroc_before": result.cv_auroc_before,
        "cv_auroc_after": result.cv_auroc_after,
    }

    result_path = out_dir / f"{args.task}_{args.method}_selection.json"
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved selection results to {result_path}")

    # Optionally save filtered features
    if args.save_filtered:
        filtered_path = out_dir / f"features_{args.task}_selected.csv"
        save_selected_features(
            features_path, filtered_path,
            result.selected_features,
        )

    # Print top features
    print("\nTop 10 features:")
    for i, (feat, score) in enumerate(list(result.feature_scores.items())[:10], 1):
        selected = "✓" if feat in result.selected_features else " "
        print(f"  {i:2d}. [{selected}] {feat}: {score:.4f}")


if __name__ == "__main__":
    main()
