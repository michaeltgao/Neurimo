"""
Generate OOF predictions from saved fold models and search for optimal ensembles.

Uses the actual saved .joblib models to ensure consistency with reported metrics.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from .train_task_model import load_labels, load_splits, _get_indices_for_ids, _safe_auroc
except ImportError:
    from train_task_model import load_labels, load_splits, _get_indices_for_ids, _safe_auroc  # type: ignore[import-not-found, no-redef]


def load_model_config(models_dir: Path, task: str, model: str) -> Dict[str, Any]:
    """Load configuration from saved metrics file."""
    metrics_path = models_dir / f"{task}_{model}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)


def generate_oof_from_saved_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    models_dir: Path,
    task: str,
    model_type: str,
) -> Tuple[np.ndarray, float, List[float]]:
    """Generate OOF predictions using saved fold models."""
    n_samples = len(y)
    oof_preds = np.zeros(n_samples)
    fold_aurocs = []

    X = df[feature_cols].to_numpy()

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(val_idx) == 0:
            continue

        # Load saved model for this fold
        model_path = models_dir / f"{task}_{model_type}_fold{fold_idx}.joblib"
        if not model_path.exists():
            print(f"    Warning: Model not found: {model_path}")
            continue

        model = joblib.load(model_path)

        X_val = X[val_idx]
        y_val = y[val_idx]

        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        auroc = _safe_auroc(y_val, preds)
        if auroc:
            fold_aurocs.append(auroc)

    global_auroc_result = _safe_auroc(y, oof_preds)
    global_auroc: float = global_auroc_result if global_auroc_result is not None else 0.5
    mean_fold_auroc = np.mean(fold_aurocs) if fold_aurocs else 0.5

    return oof_preds, global_auroc, fold_aurocs


def optimize_weights(
    oof_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    model_names: List[str],
    n_trials: int = 500,
    random_state: int = 1337,
    mask: np.ndarray | None = None,
) -> Tuple[Dict[str, float], float]:
    """Optimize ensemble weights using global AUROC.

    Args:
        oof_dict: Dict mapping model name to OOF predictions array
        y: True labels
        model_names: List of model names to include in ensemble
        n_trials: Number of Optuna trials
        random_state: Random seed for reproducibility
        mask: Optional boolean mask to select subset of samples for optimization

    Returns:
        Tuple of (weights dict, AUROC on the data used for optimization)
    """
    if mask is not None:
        oof_stack = np.column_stack([oof_dict[name][mask] for name in model_names])
        y_subset = y[mask]
    else:
        oof_stack = np.column_stack([oof_dict[name] for name in model_names])
        y_subset = y

    n_models = len(model_names)

    if not OPTUNA_AVAILABLE:
        weights = np.ones(n_models) / n_models
        avg_preds = oof_stack.mean(axis=1)
        auroc = _safe_auroc(y_subset, avg_preds)
        return dict(zip(model_names, weights)), auroc if auroc is not None else 0.5

    def objective(trial):
        raw_weights = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
        total = sum(raw_weights)
        if total < 1e-6:
            return 0.5
        weights = np.array(raw_weights) / total
        weighted_preds = (oof_stack * weights).sum(axis=1)
        auroc = _safe_auroc(y_subset, weighted_preds)
        return auroc if auroc else 0.5

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw_weights = [study.best_params[f"w_{i}"] for i in range(n_models)]
    total = sum(raw_weights)
    best_weights = np.array(raw_weights) / total

    return dict(zip(model_names, best_weights)), study.best_value


def nested_cv_ensemble_auroc(
    oof_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_names: List[str],
    n_trials: int = 500,
    random_state: int = 1337,
) -> Tuple[float, List[float], List[Dict[str, float]]]:
    """Compute unbiased ensemble AUROC using nested CV.

    For each outer fold:
      1. Optimize weights on OOF predictions from OTHER folds (inner)
      2. Apply those weights to get ensemble prediction for the outer fold

    This ensures weights are never optimized on the same data used for evaluation.

    Args:
        oof_dict: Dict mapping model name to OOF predictions array
        y: True labels
        child_ids: List of child IDs corresponding to each sample
        folds: List of fold dicts with 'val' key containing validation child IDs
        model_names: List of model names to include in ensemble
        n_trials: Number of Optuna trials per fold
        random_state: Random seed

    Returns:
        Tuple of (global_auroc, fold_aurocs, fold_weights)
    """
    n_samples = len(y)
    ensemble_preds = np.zeros(n_samples)
    fold_aurocs = []
    fold_weights = []

    # Build fold index mapping
    fold_masks = []
    for fold_data in folds:
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_data['fold']}_val")
        mask = np.zeros(n_samples, dtype=bool)
        mask[val_idx] = True
        fold_masks.append(mask)

    for outer_idx, outer_mask in enumerate(fold_masks):
        # Inner mask: all samples NOT in the outer fold
        inner_mask = ~outer_mask

        # Optimize weights on inner folds only
        weights, _ = optimize_weights(
            oof_dict, y, model_names,
            n_trials=n_trials,
            random_state=random_state + outer_idx,  # Different seed per fold
            mask=inner_mask,
        )
        fold_weights.append(weights)

        # Apply weights to outer fold (held-out from weight optimization)
        oof_stack_outer = np.column_stack([oof_dict[name][outer_mask] for name in model_names])
        weight_array = np.array([weights[name] for name in model_names])
        outer_preds = (oof_stack_outer * weight_array).sum(axis=1)

        ensemble_preds[outer_mask] = outer_preds

        # Compute fold AUROC
        fold_auroc = _safe_auroc(y[outer_mask], outer_preds)
        fold_aurocs.append(fold_auroc if fold_auroc else 0.5)

    # Global AUROC on all samples (each predicted with weights from other folds)
    global_auroc_result = _safe_auroc(y, ensemble_preds)
    global_auroc: float = global_auroc_result if global_auroc_result is not None else 0.5

    return global_auroc, fold_aurocs, fold_weights


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
    print("ENSEMBLE FROM SAVED MODELS")
    print("=" * 70)
    print(f"Samples: {len(y_all)}")
    print()

    # Define all models
    MODEL_CONFIGS = [
        {"task": "all", "model": "xgb", "name": "all_xgb"},
        {"task": "joint", "model": "xgb", "name": "joint_xgb"},
        {"task": "free", "model": "xgb", "name": "free_xgb"},
        {"task": "joint", "model": "logreg", "name": "joint_logreg"},
        {"task": "imit", "model": "xgb", "name": "imit_xgb"},
        {"task": "imit", "model": "logreg", "name": "imit_logreg"},
        {"task": "free", "model": "logreg", "name": "free_logreg"},
    ]

    # Generate OOF predictions from saved models
    print("Generating OOF predictions from saved models...")
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

            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                print(f"  {name}: Missing features {missing_cols}, skipping")
                continue

            # Check if fold models exist
            fold0_path = models_dir / f"{task}_{model_type}_fold0.joblib"
            if not fold0_path.exists():
                print(f"  {name}: No saved models found, skipping")
                continue

            print(f"  {name} ({len(feature_cols)} features)...", end=" ", flush=True)

            oof, global_auroc, fold_aurocs = generate_oof_from_saved_models(
                df, feature_cols, y_all, child_ids_all, folds,
                models_dir, task, model_type,
            )

            oof_dict[name] = oof
            model_aurocs[name] = {
                "global": global_auroc,
                "mean_fold": np.mean(fold_aurocs),
                "fold_aurocs": fold_aurocs,
            }

            # Compare with saved metrics
            saved_auroc = metrics.get("cv_summary", {}).get("val", {}).get("auroc_mean", 0)
            match = "✓" if abs(np.mean(fold_aurocs) - saved_auroc) < 0.001 else "≠"
            print(f"AUROC: {np.mean(fold_aurocs):.4f} (saved: {saved_auroc:.4f}) {match}")

        except Exception as e:
            print(f"  {name}: Error - {e}")

    print()

    # 3-model XGB baseline
    xgb_models = ["all_xgb", "joint_xgb", "free_xgb"]
    if not all(m in oof_dict for m in xgb_models):
        print("ERROR: Missing one or more XGB models!")
        return

    print("=" * 70)
    print("3-MODEL XGB ENSEMBLE BASELINE")
    print("=" * 70)

    print("\nIndividual AUROCs (from saved models):")
    for name in xgb_models:
        print(f"  {name}: Global={model_aurocs[name]['global']:.4f}, "
              f"Mean fold={model_aurocs[name]['mean_fold']:.4f}")

    # Simple average - compute fold-level AUROCs for fair comparison
    avg_preds = np.mean([oof_dict[m] for m in xgb_models], axis=0)
    avg_auroc_global = _safe_auroc(y_all, avg_preds)

    # Compute fold-level simple average AUROCs (same structure as nested CV)
    avg_fold_aurocs = []
    for fold_data in folds:
        val_idx, _ = _get_indices_for_ids(child_ids_all, fold_data["val"], f"fold{fold_data['fold']}_val")
        if len(val_idx) > 0:
            fold_avg_preds = avg_preds[val_idx]
            fold_auroc = _safe_auroc(y_all[val_idx], fold_avg_preds)
            avg_fold_aurocs.append(fold_auroc if fold_auroc else 0.5)

    print(f"\nSimple average AUROC (global): {avg_auroc_global:.4f}")
    print(f"Simple average fold AUROCs: {[f'{a:.4f}' for a in avg_fold_aurocs]}")
    print(f"Simple average mean fold AUROC: {np.mean(avg_fold_aurocs):.4f}")

    # Nested CV for unbiased optimized AUROC
    print("\nNested CV ensemble evaluation (unbiased)...")
    auroc_3_nested, fold_aurocs_3, fold_weights_3 = nested_cv_ensemble_auroc(
        oof_dict, y_all, child_ids_all, folds, xgb_models,
        n_trials=args.n_trials, random_state=args.random_state
    )
    print(f"Nested CV AUROC: {auroc_3_nested:.4f}")
    print(f"Fold AUROCs: {[f'{a:.4f}' for a in fold_aurocs_3]}")

    # Deployment weights (trained on ALL data) - for production use
    print("\nOptimizing deployment weights (on all data)...")
    weights_3_deploy, auroc_3_biased = optimize_weights(
        oof_dict, y_all, xgb_models,
        n_trials=args.n_trials, random_state=args.random_state
    )
    print(f"Deployment weights (biased AUROC={auroc_3_biased:.4f}, DO NOT report):")
    for name, w in sorted(weights_3_deploy.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.3f}")

    # Use nested CV AUROC as the true baseline
    auroc_3 = auroc_3_nested
    weights_3 = weights_3_deploy  # Use deployment weights for JSON output

    # Search for better ensembles
    all_models = list(oof_dict.keys())
    n_models = len(all_models)

    print()
    print("=" * 70)
    print("SEARCHING FOR BETTER ENSEMBLES")
    print("=" * 70)

    all_results = []
    all_results.append({
        "n_models": 3,
        "models": list(xgb_models),
        "auroc": auroc_3,  # Nested CV AUROC (unbiased)
        "weights": weights_3,  # Deployment weights
        "fold_aurocs": fold_aurocs_3,
        "is_baseline": True,
    })

    for n in range(4, n_models + 1):
        print(f"\n{n}-MODEL ENSEMBLES:")
        print("-" * 50)

        combos = list(combinations(all_models, n))
        print(f"Testing {len(combos)} combinations...")

        for combo in combos:
            combo_list = list(combo)

            # Nested CV for unbiased AUROC
            auroc_nested, fold_aurocs_combo, _ = nested_cv_ensemble_auroc(
                oof_dict, y_all, child_ids_all, folds, combo_list,
                n_trials=args.n_trials, random_state=args.random_state
            )

            # Deployment weights (on all data)
            weights_deploy, _ = optimize_weights(
                oof_dict, y_all, combo_list,
                n_trials=args.n_trials, random_state=args.random_state
            )

            result = {
                "n_models": n,
                "models": combo_list,  # List for JSON serialization
                "auroc": auroc_nested,  # Unbiased nested CV AUROC
                "fold_aurocs": fold_aurocs_combo,
                "weights": weights_deploy,  # Deployment weights for production
            }
            all_results.append(result)

            improvement = (auroc_nested - auroc_3) / auroc_3 * 100
            status = "✓ BEATS BASELINE" if auroc_nested > auroc_3 else ""
            print(f"  {' + '.join(combo_list)}")
            print(f"    AUROC: {auroc_nested:.4f} ({improvement:+.2f}% vs baseline) {status}")

    all_results.sort(key=lambda x: -x["auroc"])

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nBaseline (3-model XGB): {auroc_3:.4f}")

    better = [r for r in all_results if r["auroc"] > auroc_3 and not r.get("is_baseline")]

    if better:
        print(f"\n{len(better)} ensembles beat the baseline!")
        print("\nTop 5 improvements:")
        for i, r in enumerate(better[:5]):
            improvement = (r["auroc"] - auroc_3) / auroc_3 * 100
            print(f"\n{i+1}. AUROC: {r['auroc']:.4f} ({improvement:+.2f}%) - {r['n_models']} models")
            print(f"   Models: {' + '.join(r['models'])}")
            weights_str = ', '.join(f'{n}:{w:.2f}' for n, w in sorted(r['weights'].items(), key=lambda x: -x[1]) if w > 0.01)
            print(f"   Weights: {weights_str}")
    else:
        print("\nNo ensemble beats the 3-model XGB baseline.")

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

    # Save
    out_path = models_dir / "ensemble_from_saved_models.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline_3xgb_auroc": auroc_3,  # Nested CV (unbiased)
            "baseline_3xgb_fold_aurocs": fold_aurocs_3,
            "baseline_weights": weights_3,  # Deployment weights (for production)
            "simple_average": {
                "global_auroc": avg_auroc_global,
                "fold_aurocs": avg_fold_aurocs,
                "mean_fold_auroc": float(np.mean(avg_fold_aurocs)),
                "note": "Fixed 1/3 weights on same OOF predictions - truly unbiased"
            },
            "model_individual_aurocs": model_aurocs,
            "all_results": all_results[:20],
            "best_overall": all_results[0],
            "_note": "All AUROC values are unbiased. Simple average uses fixed equal weights. Nested CV uses per-fold optimized weights.",
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
