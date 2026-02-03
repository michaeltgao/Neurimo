"""
Search for optimal 4 and 5 model ensembles that beat the 3-model baseline (0.847 AUROC).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import shared utilities
try:
    from .train_task_model import (
        load_labels,
        load_splits,
        select_task_features,
        _remove_constant_columns,
        _get_indices_for_ids,
        _safe_auroc,
        build_model_with_params,
    )
except ImportError:
    from train_task_model import (
        load_labels,
        load_splits,
        select_task_features,
        _remove_constant_columns,
        _get_indices_for_ids,
        _safe_auroc,
        build_model_with_params,
    )


def load_tuned_params(models_dir: Path, task: str, model: str) -> Dict[str, Any] | None:
    """Load best hyperparameters from existing metrics file."""
    metrics_path = models_dir / f"{task}_{model}_metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if metrics.get("tuning") and metrics["tuning"].get("best_params"):
        return metrics["tuning"]["best_params"]
    return None


def prepare_task_features(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features for a specific task."""
    X, feature_cols = select_task_features(df, task=task)
    X, feature_cols, _ = _remove_constant_columns(X, feature_cols)
    return X, feature_cols


def generate_oof_predictions(
    X_task: pd.DataFrame,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_type: str,
    task: str,
    best_params: Dict[str, Any] | None,
    random_state: int,
) -> np.ndarray:
    """Generate out-of-fold predictions for a single base model."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    n_samples = len(y)
    oof_preds = np.zeros(n_samples)
    X_np = X_task.to_numpy()

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        X_train = X_np[train_idx]
        y_train = y[train_idx]
        X_val = X_np[val_idx]

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        if best_params:
            model = build_model_with_params(
                model_type,
                best_params,
                random_state=random_state + fold_idx,
                scale_pos_weight=scale_pos_weight if model_type in ["xgb", "lgbm"] else None,
            )
        else:
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
            elif model_type == "lgbm":
                from lightgbm import LGBMClassifier
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", LGBMClassifier(
                        n_estimators=500, learning_rate=0.02, max_depth=2,
                        num_leaves=4, subsample=0.7, colsample_bytree=0.5,
                        reg_lambda=1.0, min_child_samples=10,
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state + fold_idx,
                        n_jobs=-1, verbose=-1,
                    )),
                ])
            else:  # xgb
                from xgboost import XGBClassifier
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", XGBClassifier(
                        n_estimators=500, learning_rate=0.02, max_depth=2,
                        subsample=0.7, colsample_bytree=0.5, reg_lambda=1.0,
                        min_child_weight=2, scale_pos_weight=scale_pos_weight,
                        random_state=random_state + fold_idx,
                        n_jobs=-1, eval_metric="auc",
                    )),
                ])

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

    return oof_preds


def optimize_weights_cv(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    n_trials: int = 300,
    random_state: int = 1337,
) -> Tuple[np.ndarray, float]:
    """Optimize ensemble weights using CV AUROC."""
    if not OPTUNA_AVAILABLE:
        # Simple average
        n_models = oof_predictions.shape[1]
        weights = np.ones(n_models) / n_models
        avg_preds = oof_predictions.mean(axis=1)
        aurocs = []
        for fold_data in folds:
            val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_data['fold']}_val")
            if len(val_idx) > 0:
                auroc = _safe_auroc(y[val_idx], avg_preds[val_idx])
                if auroc:
                    aurocs.append(auroc)
        return weights, np.mean(aurocs) if aurocs else 0.5

    n_models = oof_predictions.shape[1]

    def objective(trial):
        raw_weights = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
        total = sum(raw_weights)
        if total < 1e-6:
            return 0.5
        weights = np.array(raw_weights) / total
        weighted_preds = (oof_predictions * weights).sum(axis=1)

        aurocs = []
        for fold_data in folds:
            val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_data['fold']}_val")
            if len(val_idx) > 0:
                auroc = _safe_auroc(y[val_idx], weighted_preds[val_idx])
                if auroc:
                    aurocs.append(auroc)
        return np.mean(aurocs) if aurocs else 0.5

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw_weights = [study.best_params[f"w_{i}"] for i in range(n_models)]
    total = sum(raw_weights)
    best_weights = np.array(raw_weights) / total

    return best_weights, study.best_value


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Search for optimal 4 and 5 model ensembles")
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--models_dir", default="data/derived/models")
    ap.add_argument("--random_state", type=int, default=1337)
    ap.add_argument("--n_trials", type=int, default=300, help="Optuna trials per ensemble")
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

    # All available models
    ALL_MODELS = [
        {"task": "all", "model": "xgb", "name": "all_xgb"},
        {"task": "joint", "model": "xgb", "name": "joint_xgb"},
        {"task": "joint", "model": "logreg", "name": "joint_logreg"},
        {"task": "imit", "model": "xgb", "name": "imit_xgb"},
        {"task": "imit", "model": "logreg", "name": "imit_logreg"},
        {"task": "free", "model": "xgb", "name": "free_xgb"},
        {"task": "free", "model": "logreg", "name": "free_logreg"},
    ]

    print("="*70)
    print("LARGER ENSEMBLE SEARCH")
    print("="*70)
    print(f"Baseline: 3-model XGB ensemble (all_xgb + joint_xgb + free_xgb) = 0.847 AUROC")
    print(f"Trials per ensemble: {args.n_trials}")
    print()

    # Generate OOF predictions for all models
    print("Generating OOF predictions for all models...")
    oof_dict = {}

    for config in ALL_MODELS:
        task = config["task"]
        model_type = config["model"]
        name = config["name"]

        print(f"  {name}...", end=" ", flush=True)

        X_task, _ = prepare_task_features(df, task)
        best_params = load_tuned_params(models_dir, task, model_type)

        oof = generate_oof_predictions(
            X_task, y_all, child_ids_all, folds,
            model_type=model_type, task=task,
            best_params=best_params, random_state=args.random_state,
        )

        global_auroc = _safe_auroc(y_all, oof)
        oof_dict[name] = oof
        print(f"AUROC: {global_auroc:.4f}" if global_auroc else "N/A")

    model_names = list(oof_dict.keys())
    print()

    # Search 4-model ensembles
    print("="*70)
    print("4-MODEL ENSEMBLES")
    print("="*70)

    results_4 = []
    combos_4 = list(combinations(model_names, 4))
    print(f"Testing {len(combos_4)} combinations...")

    for i, combo in enumerate(combos_4):
        oof_stack = np.column_stack([oof_dict[name] for name in combo])
        weights, auroc = optimize_weights_cv(
            oof_stack, y_all, child_ids_all, folds,
            n_trials=args.n_trials, random_state=args.random_state
        )
        results_4.append({
            "models": combo,
            "auroc": auroc,
            "weights": dict(zip(combo, weights.tolist())),
        })
        print(f"  {i+1}/{len(combos_4)}: {' + '.join(combo)} -> AUROC: {auroc:.4f}")

    # Sort by AUROC
    results_4.sort(key=lambda x: -x["auroc"])

    print("\nTop 5 4-model ensembles:")
    for r in results_4[:5]:
        print(f"  AUROC: {r['auroc']:.4f} - {' + '.join(r['models'])}")
        for name, w in sorted(r["weights"].items(), key=lambda x: -x[1]):
            print(f"    {name}: {w:.3f}")

    # Search 5-model ensembles
    print()
    print("="*70)
    print("5-MODEL ENSEMBLES")
    print("="*70)

    results_5 = []
    combos_5 = list(combinations(model_names, 5))
    print(f"Testing {len(combos_5)} combinations...")

    for i, combo in enumerate(combos_5):
        oof_stack = np.column_stack([oof_dict[name] for name in combo])
        weights, auroc = optimize_weights_cv(
            oof_stack, y_all, child_ids_all, folds,
            n_trials=args.n_trials, random_state=args.random_state
        )
        results_5.append({
            "models": combo,
            "auroc": auroc,
            "weights": dict(zip(combo, weights.tolist())),
        })
        print(f"  {i+1}/{len(combos_5)}: {' + '.join(combo)} -> AUROC: {auroc:.4f}")

    results_5.sort(key=lambda x: -x["auroc"])

    print("\nTop 5 5-model ensembles:")
    for r in results_5[:5]:
        print(f"  AUROC: {r['auroc']:.4f} - {' + '.join(r['models'])}")
        for name, w in sorted(r["weights"].items(), key=lambda x: -x[1]):
            print(f"    {name}: {w:.3f}")

    # Also try 6 and 7 model ensembles
    print()
    print("="*70)
    print("6 AND 7 MODEL ENSEMBLES")
    print("="*70)

    # 6 models
    combos_6 = list(combinations(model_names, 6))
    print(f"Testing {len(combos_6)} 6-model combinations...")
    results_6 = []
    for i, combo in enumerate(combos_6):
        oof_stack = np.column_stack([oof_dict[name] for name in combo])
        weights, auroc = optimize_weights_cv(
            oof_stack, y_all, child_ids_all, folds,
            n_trials=args.n_trials, random_state=args.random_state
        )
        results_6.append({
            "models": combo,
            "auroc": auroc,
            "weights": dict(zip(combo, weights.tolist())),
        })
        print(f"  {i+1}/{len(combos_6)}: AUROC: {auroc:.4f}")

    results_6.sort(key=lambda x: -x["auroc"])
    print(f"\nBest 6-model: AUROC: {results_6[0]['auroc']:.4f} - {' + '.join(results_6[0]['models'])}")

    # 7 models (all)
    print(f"\nTesting full 7-model ensemble...")
    oof_stack = np.column_stack([oof_dict[name] for name in model_names])
    weights, auroc = optimize_weights_cv(
        oof_stack, y_all, child_ids_all, folds,
        n_trials=args.n_trials, random_state=args.random_state
    )
    result_7 = {
        "models": tuple(model_names),
        "auroc": auroc,
        "weights": dict(zip(model_names, weights.tolist())),
    }
    print(f"  All 7 models: AUROC: {auroc:.4f}")

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBaseline (3-model XGB): 0.847 AUROC")
    print(f"\nBest 4-model: {results_4[0]['auroc']:.4f} AUROC")
    print(f"  Models: {' + '.join(results_4[0]['models'])}")
    print(f"  Improvement: {(results_4[0]['auroc'] - 0.847)*100:+.2f}%")

    print(f"\nBest 5-model: {results_5[0]['auroc']:.4f} AUROC")
    print(f"  Models: {' + '.join(results_5[0]['models'])}")
    print(f"  Improvement: {(results_5[0]['auroc'] - 0.847)*100:+.2f}%")

    print(f"\nBest 6-model: {results_6[0]['auroc']:.4f} AUROC")
    print(f"  Models: {' + '.join(results_6[0]['models'])}")
    print(f"  Improvement: {(results_6[0]['auroc'] - 0.847)*100:+.2f}%")

    print(f"\nFull 7-model: {result_7['auroc']:.4f} AUROC")
    print(f"  Improvement: {(result_7['auroc'] - 0.847)*100:+.2f}%")

    # Find overall best
    all_results = results_4 + results_5 + results_6 + [result_7]
    best = max(all_results, key=lambda x: x["auroc"])

    print()
    print("="*70)
    print("OVERALL BEST ENSEMBLE")
    print("="*70)
    print(f"AUROC: {best['auroc']:.4f}")
    print(f"Models ({len(best['models'])}): {' + '.join(best['models'])}")
    print("Weights:")
    for name, w in sorted(best["weights"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.4f}")
    print(f"\nImprovement over 3-model baseline: {(best['auroc'] - 0.847)*100:+.2f}%")

    # Save results
    out_path = models_dir / "larger_ensemble_search_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline_3model_auroc": 0.847,
            "best_4model": results_4[0],
            "best_5model": results_5[0],
            "best_6model": results_6[0],
            "full_7model": result_7,
            "overall_best": best,
            "all_4model_results": results_4,
            "all_5model_results": results_5,
            "all_6model_results": results_6,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
