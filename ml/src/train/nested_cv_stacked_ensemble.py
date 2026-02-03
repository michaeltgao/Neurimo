"""
Nested CV for stacked ensemble - unbiased evaluation.

Outer loop: Hold out test fold
Inner loop: Select best meta-learner using remaining folds
Then evaluate on held-out test fold

This prevents selection bias in meta-learner choice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

try:
    from .train_task_model import (
        load_labels,
        load_splits,
        _get_indices_for_ids,
        _safe_auroc,
    )
except ImportError:
    from train_task_model import (
        load_labels,
        load_splits,
        _get_indices_for_ids,
        _safe_auroc,
    )


def load_model_config(models_dir: Path, task: str, model: str) -> Dict[str, Any]:
    metrics_path = models_dir / f"{task}_{model}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)


def generate_oof_from_saved_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    models_dir: Path,
    task: str,
    model_type: str,
) -> np.ndarray:
    """Generate OOF predictions using saved fold models."""
    n_samples = len(df)
    oof_preds = np.zeros(n_samples)
    X = df[feature_cols].to_numpy()

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(val_idx) == 0:
            continue

        model_path = models_dir / f"{task}_{model_type}_fold{fold_idx}.joblib"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        X_val = X[val_idx]
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

    return oof_preds


def get_meta_learners():
    """Return a dict of meta-learner candidates."""
    return {
        "logreg_l2": LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000),
        "logreg_l1": LogisticRegression(C=1.0, penalty="l1", solver="saga", max_iter=1000),
        "rf_shallow": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        "gb_conservative": GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42),
        "svm_rbf": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "svm_linear": SVC(kernel="linear", C=1.0, probability=True, random_state=42),
    }


def nested_cv_evaluate(
    oof_stack: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_names: List[str],
) -> Dict[str, Any]:
    """
    Nested CV:
    - Outer loop: hold out one fold as test
    - Inner loop: select best meta-learner on remaining folds
    - Evaluate selected meta-learner on held-out test fold
    """
    n_folds = len(folds)
    outer_results = []
    meta_selections = []

    print(f"\nNested CV with {n_folds} outer folds...")
    print("=" * 60)

    for outer_fold_idx in range(n_folds):
        print(f"\nOuter fold {outer_fold_idx}: holding out as TEST")

        # Get outer test indices
        outer_test_fold = folds[outer_fold_idx]
        outer_test_idx, _ = _get_indices_for_ids(
            child_ids, outer_test_fold["val"], f"outer_test_{outer_fold_idx}"
        )

        # Get outer train indices (all other folds)
        outer_train_idx = []
        for i, fold in enumerate(folds):
            if i != outer_fold_idx:
                idx, _ = _get_indices_for_ids(child_ids, fold["val"], f"outer_train_{i}")
                outer_train_idx.extend(idx)
        outer_train_idx = np.array(outer_train_idx)

        X_outer_train = oof_stack[outer_train_idx]
        y_outer_train = y[outer_train_idx]
        X_outer_test = oof_stack[outer_test_idx]
        y_outer_test = y[outer_test_idx]

        # Scale features
        scaler = StandardScaler()
        X_outer_train_scaled = scaler.fit_transform(X_outer_train)
        X_outer_test_scaled = scaler.transform(X_outer_test)

        # Inner CV: select best meta-learner using outer train data
        print(f"  Inner CV on {len(outer_train_idx)} samples to select meta-learner...")

        meta_learners = get_meta_learners()
        inner_scores = {}

        for name, meta in meta_learners.items():
            # 3-fold inner CV
            try:
                scores = cross_val_score(
                    clone(meta), X_outer_train_scaled, y_outer_train,
                    cv=3, scoring="roc_auc"
                )
                inner_scores[name] = np.mean(scores)
            except Exception as e:
                inner_scores[name] = 0.5

        # Select best meta-learner
        best_meta_name = max(inner_scores, key=inner_scores.get)
        best_inner_score = inner_scores[best_meta_name]
        print(f"  Selected: {best_meta_name} (inner CV AUROC: {best_inner_score:.4f})")

        # Train selected meta-learner on full outer train, evaluate on outer test
        best_meta = clone(meta_learners[best_meta_name])
        best_meta.fit(X_outer_train_scaled, y_outer_train)

        test_preds = best_meta.predict_proba(X_outer_test_scaled)[:, 1]
        test_auroc = _safe_auroc(y_outer_test, test_preds)

        print(f"  Outer test AUROC: {test_auroc:.4f}")

        outer_results.append({
            "outer_fold": outer_fold_idx,
            "selected_meta": best_meta_name,
            "inner_cv_auroc": best_inner_score,
            "outer_test_auroc": test_auroc,
            "n_test": len(outer_test_idx),
        })
        meta_selections.append(best_meta_name)

    # Summary
    test_aurocs = [r["outer_test_auroc"] for r in outer_results]
    mean_auroc = np.mean(test_aurocs)
    std_auroc = np.std(test_aurocs)

    print("\n" + "=" * 60)
    print("NESTED CV SUMMARY")
    print("=" * 60)
    print(f"Mean test AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"Fold AUROCs: {[f'{a:.3f}' for a in test_aurocs]}")
    print(f"Meta-learner selections: {meta_selections}")

    # Count meta-learner selections
    from collections import Counter
    selection_counts = Counter(meta_selections)
    print(f"Selection frequency: {dict(selection_counts)}")

    return {
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "fold_aurocs": test_aurocs,
        "fold_details": outer_results,
        "meta_selection_counts": dict(selection_counts),
        "models_used": model_names,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--models_dir", default="data/derived/models")
    args = ap.parse_args()

    # Load data
    features_path = Path(args.features)
    if features_path.suffix.lower() == ".parquet":
        feat_df = pd.read_parquet(features_path)
    else:
        feat_df = pd.read_csv(features_path)

    feat_df["child_id"] = feat_df["child_id"].astype(str)
    labels_df = load_labels(Path(args.labels))
    splits = load_splits(Path(args.splits))

    folds = splits["folds"]
    models_dir = Path(args.models_dir)

    labeled_ids = set(labels_df["child_id"].tolist())
    feat_df = feat_df[feat_df["child_id"].isin(labeled_ids)].copy()
    df = feat_df.merge(labels_df, on="child_id", how="inner")

    y_all = df["label"].to_numpy().astype(int)
    child_ids_all = df["child_id"].tolist()

    print("=" * 60)
    print("NESTED CV STACKED ENSEMBLE EVALUATION")
    print("=" * 60)
    print(f"Samples: {len(y_all)}")

    # Model configs - use the best stacked ensemble models
    MODEL_CONFIGS = [
        {"task": "all", "model": "xgb", "name": "all_xgb"},
        {"task": "joint", "model": "xgb", "name": "joint_xgb"},
        {"task": "free", "model": "xgb", "name": "free_xgb"},
        {"task": "joint", "model": "logreg", "name": "joint_logreg"},
        {"task": "free", "model": "logreg", "name": "free_logreg"},
    ]

    # Generate OOF predictions
    print("\nGenerating OOF predictions from saved models...")
    oof_dict = {}

    for config in MODEL_CONFIGS:
        task = config["task"]
        model_type = config["model"]
        name = config["name"]

        try:
            metrics = load_model_config(models_dir, task, model_type)
            feature_cols = metrics.get("feature_cols", [])

            if not feature_cols:
                print(f"  {name}: No feature_cols, skipping")
                continue

            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                print(f"  {name}: Missing features, skipping")
                continue

            print(f"  {name}...", end=" ", flush=True)
            oof = generate_oof_from_saved_models(
                df, feature_cols, child_ids_all, folds, models_dir, task, model_type
            )
            oof_dict[name] = oof
            print("OK")

        except Exception as e:
            print(f"  {name}: Error - {e}")

    model_names = list(oof_dict.keys())
    oof_stack = np.column_stack([oof_dict[name] for name in model_names])

    print(f"\nUsing {len(model_names)} models: {model_names}")

    # Run nested CV
    results = nested_cv_evaluate(oof_stack, y_all, child_ids_all, folds, model_names)

    # Compare with baselines
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"3-model XGB weighted baseline:     0.8494")
    print(f"6-model weighted ensemble:         0.8501")
    print(f"Stacked (biased selection):        0.8744")
    print(f"Stacked (nested CV, unbiased):     {results['mean_auroc']:.4f} ± {results['std_auroc']:.4f}")

    improvement = (results['mean_auroc'] - 0.8501) / 0.8501 * 100
    print(f"\nImprovement vs weighted ensemble: {improvement:+.2f}%")

    # Save results
    out_path = models_dir / "nested_cv_stacked_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
