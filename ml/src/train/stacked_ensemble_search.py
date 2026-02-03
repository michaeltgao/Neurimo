"""
True stacked ensemble search - train meta-learners on OOF predictions.

Tests various meta-learners and model combinations to beat 0.8501 AUROC.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

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
) -> np.ndarray:
    """Generate OOF predictions using saved fold models."""
    n_samples = len(y)
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


def get_meta_learners(random_state: int = 1337) -> Dict[str, Any]:
    """Get dictionary of meta-learners to try."""
    learners = {
        # Logistic Regression variants
        'logreg_l2_C0.01': LogisticRegression(C=0.01, max_iter=5000, random_state=random_state),
        'logreg_l2_C0.1': LogisticRegression(C=0.1, max_iter=5000, random_state=random_state),
        'logreg_l2_C1': LogisticRegression(C=1.0, max_iter=5000, random_state=random_state),
        'logreg_l2_C10': LogisticRegression(C=10.0, max_iter=5000, random_state=random_state),
        'logreg_l1_C0.1': LogisticRegression(C=0.1, penalty='l1', solver='saga', max_iter=5000, random_state=random_state),
        'logreg_l1_C1': LogisticRegression(C=1.0, penalty='l1', solver='saga', max_iter=5000, random_state=random_state),
        'logreg_elasticnet': LogisticRegression(C=1.0, penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=5000, random_state=random_state),

        # Random Forest (shallow to avoid overfitting)
        'rf_shallow': RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_leaf=5, random_state=random_state),
        'rf_medium': RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=random_state),

        # Gradient Boosting (very conservative)
        'gb_conservative': GradientBoostingClassifier(n_estimators=50, max_depth=1, learning_rate=0.1, min_samples_leaf=10, random_state=random_state),
        'gb_medium': GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, min_samples_leaf=5, random_state=random_state),

        # SVM with probability
        'svm_rbf': SVC(C=1.0, kernel='rbf', probability=True, random_state=random_state),
        'svm_linear': SVC(C=1.0, kernel='linear', probability=True, random_state=random_state),

        # Simple MLP
        'mlp_tiny': MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, alpha=1.0, random_state=random_state),
        'mlp_small': MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=1000, alpha=0.5, random_state=random_state),
    }

    if HAS_XGB:
        learners.update({
            'xgb_conservative': XGBClassifier(n_estimators=50, max_depth=1, learning_rate=0.1, reg_lambda=2.0, random_state=random_state, eval_metric='auc'),
            'xgb_medium': XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, reg_lambda=1.0, random_state=random_state, eval_metric='auc'),
        })

    if HAS_LGBM:
        learners.update({
            'lgbm_conservative': LGBMClassifier(n_estimators=50, max_depth=1, learning_rate=0.1, reg_lambda=2.0, random_state=random_state, verbose=-1),
            'lgbm_medium': LGBMClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, reg_lambda=1.0, random_state=random_state, verbose=-1),
        })

    return learners


def evaluate_stacked_ensemble_cv(
    oof_stack: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    meta_learner,
    scale: bool = True,
) -> Tuple[float, List[float]]:
    """
    Evaluate stacked ensemble using nested CV.

    For each fold:
    1. Train meta-learner on OOF predictions from other folds
    2. Predict on this fold's OOF predictions
    3. Calculate AUROC
    """
    fold_aurocs = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        X_train = oof_stack[train_idx]
        y_train = y[train_idx]
        X_val = oof_stack[val_idx]
        y_val = y[val_idx]

        # Scale if requested
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Clone and fit meta-learner
        from sklearn.base import clone
        meta = clone(meta_learner)

        try:
            meta.fit(X_train, y_train)
            preds = meta.predict_proba(X_val)[:, 1]
            auroc = _safe_auroc(y_val, preds)
            if auroc:
                fold_aurocs.append(auroc)
        except Exception as e:
            print(f"      Error: {e}")
            continue

    mean_auroc = np.mean(fold_aurocs) if fold_aurocs else 0.5
    return mean_auroc, fold_aurocs


def tune_logreg_meta(
    oof_stack: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    n_trials: int = 100,
    random_state: int = 1337,
) -> Tuple[Dict[str, Any], float]:
    """Tune LogisticRegression meta-learner with Optuna."""
    if not OPTUNA_AVAILABLE:
        return {}, 0.5

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

        meta = LogisticRegression(
            C=C,
            penalty=penalty,
            solver="saga" if penalty == "l1" else "lbfgs",
            max_iter=5000,
            random_state=random_state,
        )

        auroc, _ = evaluate_stacked_ensemble_cv(
            oof_stack, y, child_ids, folds, meta, scale=True
        )
        return auroc

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def tune_xgb_meta(
    oof_stack: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    n_trials: int = 100,
    random_state: int = 1337,
) -> Tuple[Dict[str, Any], float]:
    """Tune XGBoost meta-learner with Optuna."""
    if not OPTUNA_AVAILABLE or not HAS_XGB:
        return {}, 0.5

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        meta = XGBClassifier(
            **params,
            random_state=random_state,
            eval_metric='auc',
            n_jobs=-1,
        )

        auroc, _ = evaluate_stacked_ensemble_cv(
            oof_stack, y, child_ids, folds, meta, scale=False
        )
        return auroc

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--models_dir", default="data/derived/models")
    ap.add_argument("--random_state", type=int, default=1337)
    ap.add_argument("--tune_trials", type=int, default=100)
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
    print("STACKED ENSEMBLE SEARCH")
    print("=" * 70)
    print(f"Samples: {len(y_all)}")
    print(f"Baseline (weighted avg): 0.8501 AUROC")
    print()

    # Load OOF predictions from saved models
    MODEL_CONFIGS = [
        {"task": "all", "model": "xgb", "name": "all_xgb"},
        {"task": "joint", "model": "xgb", "name": "joint_xgb"},
        {"task": "free", "model": "xgb", "name": "free_xgb"},
        {"task": "joint", "model": "logreg", "name": "joint_logreg"},
        {"task": "imit", "model": "xgb", "name": "imit_xgb"},
        {"task": "imit", "model": "logreg", "name": "imit_logreg"},
        {"task": "free", "model": "logreg", "name": "free_logreg"},
    ]

    print("Loading OOF predictions from saved models...")
    oof_dict = {}

    for config in MODEL_CONFIGS:
        task = config["task"]
        model_type = config["model"]
        name = config["name"]

        try:
            metrics = load_model_config(models_dir, task, model_type)
            feature_cols = metrics.get("feature_cols", [])

            if not feature_cols:
                continue

            fold0_path = models_dir / f"{task}_{model_type}_fold0.joblib"
            if not fold0_path.exists():
                continue

            oof = generate_oof_from_saved_models(
                df, feature_cols, y_all, child_ids_all, folds,
                models_dir, task, model_type,
            )
            oof_dict[name] = oof
            auroc = _safe_auroc(y_all, oof)
            print(f"  {name}: AUROC={auroc:.4f}")

        except Exception as e:
            print(f"  {name}: Error - {e}")

    model_names = list(oof_dict.keys())
    print()

    # Test different model combinations with various meta-learners
    print("=" * 70)
    print("TESTING META-LEARNERS ON DIFFERENT MODEL COMBINATIONS")
    print("=" * 70)

    meta_learners = get_meta_learners(args.random_state)
    all_results = []

    # Test combinations from 3 to all models
    for n in range(3, len(model_names) + 1):
        combos = list(combinations(model_names, n))

        for combo in combos:
            oof_stack = np.column_stack([oof_dict[name] for name in combo])

            print(f"\n{n}-model: {' + '.join(combo)}")
            print("-" * 50)

            combo_results = []

            for meta_name, meta_learner in meta_learners.items():
                try:
                    auroc, fold_aurocs = evaluate_stacked_ensemble_cv(
                        oof_stack, y_all, child_ids_all, folds,
                        meta_learner, scale=True
                    )

                    combo_results.append({
                        "meta_learner": meta_name,
                        "auroc": auroc,
                        "fold_aurocs": fold_aurocs,
                    })

                    status = "✓ BEATS 0.8501" if auroc > 0.8501 else ""
                    print(f"  {meta_name}: {auroc:.4f} {status}")

                except Exception as e:
                    print(f"  {meta_name}: Error - {e}")

            if combo_results:
                best = max(combo_results, key=lambda x: x["auroc"])
                all_results.append({
                    "n_models": n,
                    "models": combo,
                    "best_meta": best["meta_learner"],
                    "auroc": best["auroc"],
                    "all_metas": combo_results,
                })

    # Now tune the best combinations
    print()
    print("=" * 70)
    print("TUNING BEST COMBINATIONS")
    print("=" * 70)

    # Sort by AUROC and tune top 5
    all_results.sort(key=lambda x: -x["auroc"])

    tuned_results = []
    for r in all_results[:5]:
        combo = r["models"]
        oof_stack = np.column_stack([oof_dict[name] for name in combo])

        print(f"\nTuning {len(combo)}-model: {' + '.join(combo)}")
        print(f"  Best fixed meta: {r['best_meta']} = {r['auroc']:.4f}")

        # Tune LogReg
        print(f"  Tuning LogReg ({args.tune_trials} trials)...", end=" ", flush=True)
        logreg_params, logreg_auroc = tune_logreg_meta(
            oof_stack, y_all, child_ids_all, folds,
            n_trials=args.tune_trials, random_state=args.random_state
        )
        status = "✓ BEATS 0.8501" if logreg_auroc > 0.8501 else ""
        print(f"{logreg_auroc:.4f} {status}")

        # Tune XGB if available
        if HAS_XGB:
            print(f"  Tuning XGB ({args.tune_trials} trials)...", end=" ", flush=True)
            xgb_params, xgb_auroc = tune_xgb_meta(
                oof_stack, y_all, child_ids_all, folds,
                n_trials=args.tune_trials, random_state=args.random_state
            )
            status = "✓ BEATS 0.8501" if xgb_auroc > 0.8501 else ""
            print(f"{xgb_auroc:.4f} {status}")
        else:
            xgb_params, xgb_auroc = {}, 0.5

        best_tuned = max(logreg_auroc, xgb_auroc, r["auroc"])
        tuned_results.append({
            "models": combo,
            "n_models": len(combo),
            "best_fixed_auroc": r["auroc"],
            "best_fixed_meta": r["best_meta"],
            "tuned_logreg_auroc": logreg_auroc,
            "tuned_logreg_params": logreg_params,
            "tuned_xgb_auroc": xgb_auroc,
            "tuned_xgb_params": xgb_params,
            "best_overall_auroc": best_tuned,
        })

    # Final summary
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nBaseline (weighted average): 0.8501 AUROC")

    all_results.sort(key=lambda x: -x["auroc"])
    tuned_results.sort(key=lambda x: -x["best_overall_auroc"])

    print("\nTop 10 Stacked Ensembles (fixed meta-learners):")
    for i, r in enumerate(all_results[:10]):
        status = "✓ BEATS" if r["auroc"] > 0.8501 else ""
        print(f"  {i+1}. AUROC={r['auroc']:.4f} {status}")
        print(f"     Models: {' + '.join(r['models'])}")
        print(f"     Meta: {r['best_meta']}")

    print("\nTop 5 After Tuning:")
    for i, r in enumerate(tuned_results[:5]):
        status = "✓ BEATS" if r["best_overall_auroc"] > 0.8501 else ""
        print(f"  {i+1}. AUROC={r['best_overall_auroc']:.4f} {status}")
        print(f"     Models: {' + '.join(r['models'])}")
        if r["tuned_logreg_auroc"] >= r["tuned_xgb_auroc"]:
            print(f"     Best: Tuned LogReg {r['tuned_logreg_params']}")
        else:
            print(f"     Best: Tuned XGB")

    # Check if any beat the baseline
    best_stacked = max(all_results, key=lambda x: x["auroc"])
    best_tuned = max(tuned_results, key=lambda x: x["best_overall_auroc"])

    overall_best = max(best_stacked["auroc"], best_tuned["best_overall_auroc"])

    print()
    print("=" * 70)
    if overall_best > 0.8501:
        improvement = (overall_best - 0.8501) / 0.8501 * 100
        print(f"SUCCESS! Best stacked ensemble: {overall_best:.4f} (+{improvement:.2f}%)")
    else:
        print(f"No stacked ensemble beats 0.8501. Best: {overall_best:.4f}")
    print("=" * 70)

    # Save results
    out_path = models_dir / "stacked_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline_weighted_avg": 0.8501,
            "best_stacked_fixed": {
                "models": best_stacked["models"],
                "meta_learner": best_stacked["best_meta"],
                "auroc": best_stacked["auroc"],
            },
            "best_tuned": {
                "models": best_tuned["models"],
                "auroc": best_tuned["best_overall_auroc"],
                "logreg_params": best_tuned["tuned_logreg_params"],
                "xgb_params": best_tuned["tuned_xgb_params"],
            },
            "all_results": all_results[:20],
            "tuned_results": tuned_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
