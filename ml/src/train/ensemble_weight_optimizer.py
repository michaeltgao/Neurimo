"""
Optimize ensemble weights for 4+ model combinations.

Usage:
    python -m ml.src.train.ensemble_weight_optimizer \
        --features data/derived/features_merged.csv \
        --labels data/derived/labels_usable.csv \
        --splits data/derived/splits.json \
        --models_dir data/derived/models \
        --save_oof  # Save OOF predictions for reuse
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
) -> Tuple[np.ndarray, float]:
    """Generate OOF predictions and return mean fold AUROC."""
    n_samples = len(y)
    oof_preds = np.zeros(n_samples)
    X_np = X_task.to_numpy()
    fold_aurocs = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        X_train, y_train = X_np[train_idx], y[train_idx]
        X_val, y_val = X_np[val_idx], y[val_idx]

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        if best_params:
            model = build_model_with_params(
                model_type, best_params,
                random_state=random_state + fold_idx,
                scale_pos_weight=scale_pos_weight if model_type in ["xgb", "lgbm"] else None,
            )
        else:
            if model_type == "logreg":
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                random_state=random_state + fold_idx)),
                ])
            elif model_type == "lgbm":
                from lightgbm import LGBMClassifier
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", LGBMClassifier(
                        n_estimators=500, learning_rate=0.02, max_depth=2, num_leaves=4,
                        subsample=0.7, colsample_bytree=0.5, reg_lambda=1.0, min_child_samples=10,
                        scale_pos_weight=scale_pos_weight, random_state=random_state + fold_idx,
                        n_jobs=-1, verbose=-1)),
                ])
            else:  # xgb
                from xgboost import XGBClassifier
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", XGBClassifier(
                        n_estimators=500, learning_rate=0.02, max_depth=2, subsample=0.7,
                        colsample_bytree=0.5, reg_lambda=1.0, min_child_weight=2,
                        scale_pos_weight=scale_pos_weight, random_state=random_state + fold_idx,
                        n_jobs=-1, eval_metric="auc")),
                ])

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        auroc = _safe_auroc(y_val, preds)
        if auroc:
            fold_aurocs.append(auroc)

    mean_fold_auroc = np.mean(fold_aurocs) if fold_aurocs else 0.5
    return oof_preds, mean_fold_auroc


def optimize_weights_cv(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    n_trials: int = 500,
    random_state: int = 1337,
) -> Tuple[np.ndarray, float]:
    """Optimize ensemble weights using CV AUROC with multi-seed averaging."""
    if not OPTUNA_AVAILABLE:
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

    # Multi-seed optimization
    n_seeds = 3
    trials_per_seed = n_trials // n_seeds
    all_weights = []
    best_aurocs = []

    for seed_offset in range(n_seeds):
        seed = random_state + seed_offset * 100
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=trials_per_seed, show_progress_bar=False)

        raw_weights = [study.best_params[f"w_{i}"] for i in range(n_models)]
        total = sum(raw_weights)
        seed_weights = np.array(raw_weights) / total
        all_weights.append(seed_weights)
        best_aurocs.append(study.best_value)

    # Average weights across seeds
    best_weights = np.mean(all_weights, axis=0)
    best_weights = best_weights / best_weights.sum()

    # Calculate final AUROC with averaged weights
    weighted_preds = (oof_predictions * best_weights).sum(axis=1)
    aurocs = []
    for fold_data in folds:
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_data['fold']}_val")
        if len(val_idx) > 0:
            auroc = _safe_auroc(y[val_idx], weighted_preds[val_idx])
            if auroc:
                aurocs.append(auroc)

    return best_weights, np.mean(aurocs) if aurocs else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--models_dir", default="data/derived/models")
    ap.add_argument("--random_state", type=int, default=1337)
    ap.add_argument("--n_trials", type=int, default=500)
    ap.add_argument("--save_oof", action="store_true", help="Save OOF predictions to file")
    ap.add_argument("--load_oof", type=str, help="Load OOF predictions from file")
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

    print("="*70)
    print("ENSEMBLE WEIGHT OPTIMIZER")
    print("="*70)

    # Generate or load OOF predictions
    if args.load_oof:
        print(f"Loading OOF predictions from {args.load_oof}")
        oof_df = pd.read_csv(args.load_oof)
        model_names = [c for c in oof_df.columns if c.startswith(("all_", "joint_", "imit_", "free_"))]
        oof_dict = {name: oof_df[name].values for name in model_names}
    else:
        print("\nGenerating OOF predictions...")
        oof_dict = {}
        model_aurocs = {}

        for config in ALL_MODELS:
            task = config["task"]
            model_type = config["model"]
            name = config["name"]

            print(f"  {name}...", end=" ", flush=True)

            X_task, _ = prepare_task_features(df, task)
            best_params = load_tuned_params(models_dir, task, model_type)

            oof, mean_auroc = generate_oof_predictions(
                X_task, y_all, child_ids_all, folds,
                model_type=model_type, task=task,
                best_params=best_params, random_state=args.random_state,
            )

            oof_dict[name] = oof
            model_aurocs[name] = mean_auroc
            print(f"Mean fold AUROC: {mean_auroc:.4f}")

        model_names = list(oof_dict.keys())

        if args.save_oof:
            oof_path = models_dir / "all_models_oof_predictions.csv"
            oof_df = pd.DataFrame({
                "child_id": child_ids_all,
                "label": y_all,
                **oof_dict
            })
            oof_df.to_csv(oof_path, index=False)
            print(f"\nSaved OOF predictions to {oof_path}")

    model_names = list(oof_dict.keys())
    n_models = len(model_names)

    # Test all ensemble sizes from 3 to n_models
    print("\n" + "="*70)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*70)

    all_results = []

    for n in range(3, n_models + 1):
        print(f"\n{n}-MODEL ENSEMBLES:")
        print("-" * 50)

        combos = list(combinations(model_names, n))
        print(f"Testing {len(combos)} combinations...")

        for combo in combos:
            oof_stack = np.column_stack([oof_dict[name] for name in combo])
            weights, auroc = optimize_weights_cv(
                oof_stack, y_all, child_ids_all, folds,
                n_trials=args.n_trials, random_state=args.random_state
            )

            result = {
                "n_models": n,
                "models": combo,
                "auroc": auroc,
                "weights": dict(zip(combo, weights.tolist())),
            }
            all_results.append(result)

            print(f"  {' + '.join(combo)}")
            print(f"    AUROC: {auroc:.4f}")
            for name, w in sorted(result["weights"].items(), key=lambda x: -x[1]):
                if w > 0.01:
                    print(f"      {name}: {w:.3f}")

    # Sort and display best results
    all_results.sort(key=lambda x: -x["auroc"])

    print("\n" + "="*70)
    print("TOP 10 ENSEMBLES (ALL SIZES)")
    print("="*70)

    for i, r in enumerate(all_results[:10]):
        print(f"\n{i+1}. AUROC: {r['auroc']:.4f} ({r['n_models']} models)")
        print(f"   Models: {' + '.join(r['models'])}")
        print(f"   Weights: {', '.join(f'{n}:{w:.2f}' for n, w in sorted(r['weights'].items(), key=lambda x: -x[1]) if w > 0.01)}")

    # Best by size
    print("\n" + "="*70)
    print("BEST BY ENSEMBLE SIZE")
    print("="*70)

    for n in range(3, n_models + 1):
        size_results = [r for r in all_results if r["n_models"] == n]
        if size_results:
            best = max(size_results, key=lambda x: x["auroc"])
            print(f"\nBest {n}-model: AUROC {best['auroc']:.4f}")
            print(f"  {' + '.join(best['models'])}")

    # Save results
    out_path = models_dir / "optimized_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "all_results": all_results,
            "top_10": all_results[:10],
            "best_overall": all_results[0],
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
