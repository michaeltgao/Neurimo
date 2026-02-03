"""
Reproduce the 3-model XGB ensemble baseline (0.847 AUROC) and search for better 4/5+ model combos.

Uses the exact feature columns from saved model metrics to ensure consistency.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

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
    """Load configuration from saved metrics file."""
    metrics_path = models_dir / f"{task}_{model}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)


def generate_oof_predictions(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_type: str,
    random_state: int,
) -> Tuple[np.ndarray, float, List[float]]:
    """Generate OOF predictions using exact feature columns."""
    n_samples = len(y)
    oof_preds = np.zeros(n_samples)
    fold_aurocs = []

    # Extract features
    X = df[feature_cols].to_numpy()

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        if model_type == "logreg":
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state + fold_idx,
                )),
            ])
        else:  # xgb
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.02,
                    max_depth=2,
                    subsample=0.7,
                    colsample_bytree=0.5,
                    reg_lambda=1.0,
                    min_child_weight=2,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state + fold_idx,
                    n_jobs=-1,
                    eval_metric="auc",
                )),
            ])

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        auroc = _safe_auroc(y_val, preds)
        if auroc:
            fold_aurocs.append(auroc)

    global_auroc = _safe_auroc(y, oof_preds)
    mean_fold_auroc = np.mean(fold_aurocs) if fold_aurocs else 0.5

    return oof_preds, global_auroc, fold_aurocs


def optimize_weights(
    oof_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    model_names: List[str],
    n_trials: int = 500,
    random_state: int = 1337,
) -> Tuple[Dict[str, float], float]:
    """Optimize ensemble weights using global AUROC."""
    oof_stack = np.column_stack([oof_dict[name] for name in model_names])
    n_models = len(model_names)

    if not OPTUNA_AVAILABLE:
        weights = np.ones(n_models) / n_models
        avg_preds = oof_stack.mean(axis=1)
        auroc = _safe_auroc(y, avg_preds)
        return dict(zip(model_names, weights)), auroc

    def objective(trial):
        raw_weights = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
        total = sum(raw_weights)
        if total < 1e-6:
            return 0.5
        weights = np.array(raw_weights) / total
        weighted_preds = (oof_stack * weights).sum(axis=1)
        auroc = _safe_auroc(y, weighted_preds)
        return auroc if auroc else 0.5

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw_weights = [study.best_params[f"w_{i}"] for i in range(n_models)]
    total = sum(raw_weights)
    best_weights = np.array(raw_weights) / total

    return dict(zip(model_names, best_weights)), study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--models_dir", default="data/derived/models")
    ap.add_argument("--random_state", type=int, default=1337)
    ap.add_argument("--n_trials", type=int, default=500)
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

    print("=" * 70)
    print("REPRODUCING 3-MODEL XGB ENSEMBLE BASELINE")
    print("=" * 70)
    print(f"Samples: {len(y_all)}")
    print()

    # Define models to test - use exact feature_cols from saved metrics
    MODEL_CONFIGS = [
        {"task": "all", "model": "xgb", "name": "all_xgb"},
        {"task": "joint", "model": "xgb", "name": "joint_xgb"},
        {"task": "free", "model": "xgb", "name": "free_xgb"},
        {"task": "joint", "model": "logreg", "name": "joint_logreg"},
        {"task": "imit", "model": "xgb", "name": "imit_xgb"},
        {"task": "imit", "model": "logreg", "name": "imit_logreg"},
        {"task": "free", "model": "logreg", "name": "free_logreg"},
    ]

    # Generate OOF predictions for each model using saved feature columns
    print("Generating OOF predictions (using saved feature_cols from metrics)...")
    oof_dict = {}
    model_aurocs = {}

    for config in MODEL_CONFIGS:
        task = config["task"]
        model_type = config["model"]
        name = config["name"]

        try:
            metrics = load_model_config(models_dir, task, model_type)
            feature_cols = metrics.get("feature_cols", [])

            if not feature_cols:
                print(f"  {name}: No feature_cols in metrics, skipping")
                continue

            # Check if all features exist in dataframe
            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                print(f"  {name}: Missing features {missing_cols}, skipping")
                continue

            print(f"  {name} ({len(feature_cols)} features)...", end=" ", flush=True)

            oof, global_auroc, fold_aurocs = generate_oof_predictions(
                df, feature_cols, y_all, child_ids_all, folds,
                model_type=model_type,
                random_state=args.random_state,
            )

            oof_dict[name] = oof
            model_aurocs[name] = {
                "global": global_auroc,
                "mean_fold": np.mean(fold_aurocs),
                "fold_aurocs": fold_aurocs,
            }
            print(f"Global AUROC: {global_auroc:.4f}, Mean fold: {np.mean(fold_aurocs):.4f}")

        except Exception as e:
            print(f"  {name}: Error - {e}")

    print()

    # Verify we have the 3 XGB models
    xgb_models = ["all_xgb", "joint_xgb", "free_xgb"]
    if not all(m in oof_dict for m in xgb_models):
        print("ERROR: Missing one or more XGB models!")
        return

    # Calculate 3-model XGB ensemble baseline
    print("=" * 70)
    print("3-MODEL XGB ENSEMBLE BASELINE")
    print("=" * 70)

    print("\nIndividual AUROCs:")
    for name in xgb_models:
        print(f"  {name}: Global={model_aurocs[name]['global']:.4f}, "
              f"Mean fold={model_aurocs[name]['mean_fold']:.4f}")

    # Simple average
    avg_preds = np.mean([oof_dict[m] for m in xgb_models], axis=0)
    avg_auroc = _safe_auroc(y_all, avg_preds)
    print(f"\nSimple average AUROC: {avg_auroc:.4f}")

    # Optimized weights
    print("\nOptimizing 3-model weights...")
    weights_3, auroc_3 = optimize_weights(
        oof_dict, y_all, xgb_models,
        n_trials=args.n_trials, random_state=args.random_state
    )
    print(f"Optimized 3-model AUROC: {auroc_3:.4f}")
    print("Weights:")
    for name, w in sorted(weights_3.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.3f}")

    # Now search for better 4, 5, 6, 7 model ensembles
    all_models = list(oof_dict.keys())
    n_models = len(all_models)

    print()
    print("=" * 70)
    print("SEARCHING FOR BETTER ENSEMBLES")
    print("=" * 70)

    all_results = []

    # Also add the 3-model XGB baseline to results
    all_results.append({
        "n_models": 3,
        "models": tuple(xgb_models),
        "auroc": auroc_3,
        "weights": weights_3,
        "is_baseline": True,
    })

    for n in range(4, n_models + 1):
        print(f"\n{n}-MODEL ENSEMBLES:")
        print("-" * 50)

        combos = list(combinations(all_models, n))
        print(f"Testing {len(combos)} combinations...")

        for combo in combos:
            weights, auroc = optimize_weights(
                oof_dict, y_all, list(combo),
                n_trials=args.n_trials, random_state=args.random_state
            )

            result = {
                "n_models": n,
                "models": combo,
                "auroc": auroc,
                "weights": weights,
            }
            all_results.append(result)

            improvement = (auroc - auroc_3) / auroc_3 * 100
            status = "✓ BEATS BASELINE" if auroc > auroc_3 else ""
            print(f"  {' + '.join(combo)}")
            print(f"    AUROC: {auroc:.4f} ({improvement:+.2f}% vs baseline) {status}")

    # Sort by AUROC
    all_results.sort(key=lambda x: -x["auroc"])

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nBaseline (3-model XGB): {auroc_3:.4f}")

    # Find ensembles that beat baseline
    better = [r for r in all_results if r["auroc"] > auroc_3 and not r.get("is_baseline")]

    if better:
        print(f"\n{len(better)} ensembles beat the baseline!")
        print("\nTop 5 improvements:")
        for i, r in enumerate(better[:5]):
            improvement = (r["auroc"] - auroc_3) / auroc_3 * 100
            print(f"\n{i+1}. AUROC: {r['auroc']:.4f} ({improvement:+.2f}%) - {r['n_models']} models")
            print(f"   Models: {' + '.join(r['models'])}")
            print(f"   Weights: {', '.join(f'{n}:{w:.2f}' for n, w in sorted(r['weights'].items(), key=lambda x: -x[1]) if w > 0.01)}")
    else:
        print("\nNo ensemble beats the 3-model XGB baseline.")

    # Best by size
    print("\n" + "=" * 70)
    print("BEST BY ENSEMBLE SIZE")
    print("=" * 70)

    for n in range(3, n_models + 1):
        size_results = [r for r in all_results if r["n_models"] == n]
        if size_results:
            best = max(size_results, key=lambda x: x["auroc"])
            improvement = (best["auroc"] - auroc_3) / auroc_3 * 100
            marker = " (BASELINE)" if best.get("is_baseline") else ""
            print(f"\nBest {n}-model: AUROC {best['auroc']:.4f} ({improvement:+.2f}%){marker}")
            print(f"  {' + '.join(best['models'])}")

    # Save results
    out_path = models_dir / "ensemble_search_with_baseline.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline_3xgb_auroc": auroc_3,
            "baseline_weights": weights_3,
            "model_individual_aurocs": model_aurocs,
            "all_results": all_results[:20],  # Top 20
            "best_overall": all_results[0],
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
