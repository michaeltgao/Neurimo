"""
True Multi-Model Ensemble Training.

Combines predictions from multiple trained models (XGB, LogReg across tasks)
using a weighted meta-learner to achieve better accuracy than any single model.

This is a "late fusion" ensemble that stacks model predictions rather than features.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import shared utilities
try:
    from .train_task_model import (
        EvalMetrics,
        evaluate_binary,
        find_optimal_threshold,
        load_labels,
        load_splits,
        select_task_features,
        _remove_constant_columns,
        _get_indices_for_ids,
        aggregate_fold_metrics,
        _safe_auroc,
        build_model_with_params,
    )
except ImportError:
    from train_task_model import (
        EvalMetrics,
        evaluate_binary,
        find_optimal_threshold,
        load_labels,
        load_splits,
        select_task_features,
        _remove_constant_columns,
        _get_indices_for_ids,
        aggregate_fold_metrics,
        _safe_auroc,
        build_model_with_params,
    )

# Optional Optuna import
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# Define base model configurations - focused on proven strong models
# 6-model config includes imit_xgb which is now HEALTHY
BASE_MODEL_CONFIGS = [
    # Cross-task model - best single model
    {"task": "all", "model": "xgb", "name": "xgb_all"},           # ~0.795 AUROC

    # Joint attention - both classifiers
    {"task": "joint", "model": "xgb", "name": "xgb_joint"},       # ~0.766 AUROC
    {"task": "joint", "model": "logreg", "name": "logreg_joint"}, # ~0.646 AUROC

    # Imitation - now included since imit_xgb is HEALTHY
    {"task": "imit", "model": "xgb", "name": "xgb_imit"},         # ~0.633 AUROC

    # Free play - both classifiers
    {"task": "free", "model": "xgb", "name": "xgb_free"},         # ~0.734 AUROC
    {"task": "free", "model": "logreg", "name": "logreg_free"},   # ~0.644 AUROC
]


def load_tuned_params(models_dir: Path, task: str, model: str) -> Optional[Dict[str, Any]]:
    """Load best hyperparameters from existing metrics file."""
    metrics_path = models_dir / f"{task}_{model}_metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if metrics.get("tuning") and metrics["tuning"].get("best_params"):
        return metrics["tuning"]["best_params"]
    return None


def prepare_task_features(
    df: pd.DataFrame,
    task: str,
) -> Tuple[pd.DataFrame, List[str]]:
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
    best_params: Optional[Dict[str, Any]],
    random_state: int,
) -> Tuple[np.ndarray, List[float]]:
    """
    Generate out-of-fold predictions for a single base model.

    Returns:
        oof_preds: array of shape (n_samples,) with OOF predictions
        fold_aurocs: list of per-fold AUROC values (to match original metrics)
    """
    n_samples = len(y)
    oof_preds = np.zeros(n_samples)
    fold_aurocs = []

    X_np = X_task.to_numpy()

    # NO global imputation - let the Pipeline handle it per-fold
    # This matches the original training and avoids data leakage

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        X_train = X_np[train_idx]
        y_train = y[train_idx]
        X_val = X_np[val_idx]
        y_val = y[val_idx]

        # Compute scale_pos_weight
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Build model - Pipeline includes imputer
        if best_params:
            model = build_model_with_params(
                model_type,
                best_params,
                random_state=random_state + fold_idx,
                scale_pos_weight=scale_pos_weight if model_type in ["xgb", "lgbm"] else None,
            )
        else:
            # Use defaults - Pipeline includes imputer
            from sklearn.pipeline import Pipeline
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
                # Anti-overfitting defaults for small datasets
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.02,  # Slower learning
                        max_depth=2,  # Shallow trees
                        num_leaves=4,  # 2^max_depth
                        subsample=0.7,
                        colsample_bytree=0.5,  # More feature sampling
                        reg_lambda=1.0,
                        min_child_samples=10,  # Higher min samples per leaf
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state + fold_idx,
                        n_jobs=-1,
                        verbose=-1,
                    )),
                ])
            else:  # xgb
                from xgboost import XGBClassifier
                # Anti-overfitting defaults for small datasets
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.02,  # Slower learning
                        max_depth=2,  # Shallow trees
                        subsample=0.7,
                        colsample_bytree=0.5,  # More feature sampling
                        reg_lambda=1.0,
                        min_child_weight=2,  # Prevent memorization
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state + fold_idx,
                        n_jobs=-1,
                        eval_metric="auc",
                    )),
                ])

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        # Track per-fold AUROC to match original metrics format
        auroc = _safe_auroc(y_val, preds)
        fold_aurocs.append(auroc if auroc else 0.0)

    return oof_preds, fold_aurocs


def train_meta_learner(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    model_names: List[str],
    random_state: int,
    tune: bool = True,
    n_trials: int = 50,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Train meta-learner on stacked OOF predictions.

    Args:
        oof_predictions: Shape (n_samples, n_models) - stacked OOF preds
        y: Ground truth labels
        model_names: Names of base models for interpretation
        random_state: Random seed
        tune: Whether to tune meta-learner
        n_trials: Number of tuning trials

    Returns:
        Trained meta-learner and model weights dict
    """
    # Scale predictions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(oof_predictions)

    if tune and OPTUNA_AVAILABLE:
        def objective(trial):
            C = trial.suggest_float("C", 1e-4, 100.0, log=True)
            meta = LogisticRegression(
                C=C,
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            )

            # Internal CV
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            aurocs = []
            for tr_idx, va_idx in skf.split(X_scaled, y):
                meta.fit(X_scaled[tr_idx], y[tr_idx])
                p = meta.predict_proba(X_scaled[va_idx])[:, 1]
                auroc = _safe_auroc(y[va_idx], p)
                if auroc:
                    aurocs.append(auroc)
            return np.mean(aurocs) if aurocs else 0.5

        sampler = TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_C = study.best_params["C"]
        print(f"  Meta-learner best C: {best_C:.6f}")
    else:
        best_C = 1.0

    # Final meta-learner
    meta = LogisticRegression(
        C=best_C,
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
    )
    meta.fit(X_scaled, y)

    # Extract model weights (coefficients)
    weights = dict(zip(model_names, meta.coef_[0].tolist()))
    weights = dict(sorted(weights.items(), key=lambda x: -abs(x[1])))

    return meta, weights, scaler


def evaluate_ensemble_cv(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    random_state: int,
    threshold_metric: str = "f1",
    tune_meta: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float]]:
    """
    Evaluate true ensemble using CV.

    For each fold:
    1. Train meta-learner on OOF predictions from OTHER folds
    2. Predict on this fold's OOF predictions
    3. Evaluate

    Returns:
        Aggregated metrics, per-fold metrics, thresholds
    """
    n_folds = len(folds)
    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        # Train meta-learner on train OOF predictions
        X_train_oof = oof_predictions[train_idx]
        y_train = y[train_idx]
        X_val_oof = oof_predictions[val_idx]
        y_val = y[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_oof)
        X_val_scaled = scaler.transform(X_val_oof)

        # Simple meta-learner per fold (no tuning for speed)
        meta = LogisticRegression(
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state + fold_idx,
        )
        meta.fit(X_train_scaled, y_train)

        # Predict
        p_val = meta.predict_proba(X_val_scaled)[:, 1]

        # Find threshold
        threshold = find_optimal_threshold(y_val, p_val, metric=threshold_metric)
        fold_thresholds.append(threshold)

        # Evaluate
        metrics = asdict(evaluate_binary(y_val, p_val, threshold=threshold))
        fold_metrics.append(metrics)

        auroc_str = f"{metrics['auroc']:.3f}" if metrics['auroc'] else "N/A"
        print(f"  Fold {fold_idx}: AUROC={auroc_str}, Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

    # Aggregate
    agg = aggregate_fold_metrics(fold_metrics)

    return agg, fold_metrics, fold_thresholds


def simple_average_ensemble(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    threshold_metric: str = "f1",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float]]:
    """
    Simple averaging baseline - just average the OOF predictions.
    """
    # Average across models
    avg_preds = oof_predictions.mean(axis=1)

    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(val_idx) == 0:
            continue

        p_val = avg_preds[val_idx]
        y_val = y[val_idx]

        threshold = find_optimal_threshold(y_val, p_val, metric=threshold_metric)
        fold_thresholds.append(threshold)

        metrics = asdict(evaluate_binary(y_val, p_val, threshold=threshold))
        fold_metrics.append(metrics)

    agg = aggregate_fold_metrics(fold_metrics)
    return agg, fold_metrics, fold_thresholds


def xgb_meta_learner_ensemble(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_names: List[str],
    threshold_metric: str = "f1",
    random_state: int = 1337,
    n_trials: int = 50,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float], Dict[str, float]]:
    """
    Use XGBoost as meta-learner for more powerful stacking.
    XGBoost can capture non-linear relationships between base model predictions.
    """
    from xgboost import XGBClassifier

    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []
    feature_importances = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        X_train_oof = oof_predictions[train_idx]
        y_train = y[train_idx]
        X_val_oof = oof_predictions[val_idx]
        y_val = y[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_oof)
        X_val_scaled = scaler.transform(X_val_oof)

        # Compute class weight
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # XGBoost meta-learner with conservative hyperparams to avoid overfitting
        meta = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=2,  # Shallow to avoid overfitting on few features
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state + fold_idx,
            n_jobs=-1,
            eval_metric="logloss",
        )
        meta.fit(X_train_scaled, y_train)

        # Track feature importance
        feature_importances.append(meta.feature_importances_)

        # Predict
        p_val = meta.predict_proba(X_val_scaled)[:, 1]

        # Find threshold
        threshold = find_optimal_threshold(y_val, p_val, metric=threshold_metric)
        fold_thresholds.append(threshold)

        # Evaluate
        metrics = asdict(evaluate_binary(y_val, p_val, threshold=threshold))
        fold_metrics.append(metrics)

        auroc_str = f"{metrics['auroc']:.3f}" if metrics['auroc'] else "N/A"
        print(f"  Fold {fold_idx}: AUROC={auroc_str}, Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

    # Average feature importances across folds
    avg_importance = np.mean(feature_importances, axis=0)
    weights_dict = {name: float(imp) for name, imp in zip(model_names, avg_importance)}
    weights_dict = dict(sorted(weights_dict.items(), key=lambda x: -x[1]))

    agg = aggregate_fold_metrics(fold_metrics)
    return agg, fold_metrics, fold_thresholds, weights_dict


def optimized_weighted_ensemble(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_names: List[str],
    threshold_metric: str = "f1",
    n_trials: int = 500,  # Increased from 200 for better optimization
    random_state: int = 1337,
    n_seeds: int = 3,  # Multi-seed averaging for stability
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float], Dict[str, float]]:
    """
    Optimize weights directly using Optuna to maximize AUROC.
    Uses multi-seed averaging for more stable weight estimates.
    """
    if not OPTUNA_AVAILABLE:
        # Fallback to simple averaging
        agg, fold_metrics, thresholds = simple_average_ensemble(
            oof_predictions, y, child_ids, folds, threshold_metric
        )
        weights = {name: 1.0 / len(model_names) for name in model_names}
        return agg, fold_metrics, thresholds, weights

    n_models = oof_predictions.shape[1]

    def objective(trial):
        # Sample weights (will be normalized)
        raw_weights = []
        for i in range(n_models):
            w = trial.suggest_float(f"w_{i}", 0.0, 1.0)
            raw_weights.append(w)

        # Normalize weights
        total = sum(raw_weights)
        if total < 1e-6:
            return 0.5
        weights = np.array(raw_weights) / total

        # Weighted average
        weighted_preds = (oof_predictions * weights).sum(axis=1)

        # CV evaluation
        aurocs = []
        for fold_data in folds:
            fold_idx = fold_data["fold"]
            val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")
            if len(val_idx) == 0:
                continue
            auroc = _safe_auroc(y[val_idx], weighted_preds[val_idx])
            if auroc:
                aurocs.append(auroc)

        return np.mean(aurocs) if aurocs else 0.5

    # Multi-seed optimization for stability
    all_weights = []
    trials_per_seed = n_trials // n_seeds

    for seed_offset in range(n_seeds):
        seed = random_state + seed_offset * 100
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=trials_per_seed, show_progress_bar=True)

        # Get weights from this seed
        raw_weights = [study.best_params[f"w_{i}"] for i in range(n_models)]
        total = sum(raw_weights)
        seed_weights = np.array(raw_weights) / total
        all_weights.append(seed_weights)
        print(f"    Seed {seed_offset + 1}/{n_seeds}: best AUROC = {study.best_value:.4f}")

    # Average weights across seeds
    best_weights = np.mean(all_weights, axis=0)
    # Re-normalize after averaging
    best_weights = best_weights / best_weights.sum()
    weights_dict = {name: float(w) for name, w in zip(model_names, best_weights)}

    # Final weighted predictions
    weighted_preds = (oof_predictions * best_weights).sum(axis=1)

    # Evaluate per fold
    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")
        if len(val_idx) == 0:
            continue

        p_val = weighted_preds[val_idx]
        y_val = y[val_idx]

        threshold = find_optimal_threshold(y_val, p_val, metric=threshold_metric)
        fold_thresholds.append(threshold)

        metrics = asdict(evaluate_binary(y_val, p_val, threshold=threshold))
        fold_metrics.append(metrics)

    agg = aggregate_fold_metrics(fold_metrics)
    return agg, fold_metrics, fold_thresholds, weights_dict


def top_models_ensemble(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_names: List[str],
    top_k: int = 3,
    threshold_metric: str = "f1",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float], List[str]]:
    """
    Ensemble using only top-k models by OOF AUROC.
    """
    # Rank models by OOF AUROC
    aurocs = []
    for i, name in enumerate(model_names):
        auroc = _safe_auroc(y, oof_predictions[:, i])
        aurocs.append((i, name, auroc if auroc else 0.0))

    aurocs_sorted = sorted(aurocs, key=lambda x: -x[2])
    top_indices = [x[0] for x in aurocs_sorted[:top_k]]
    top_names = [x[1] for x in aurocs_sorted[:top_k]]

    # Average top models
    top_preds = oof_predictions[:, top_indices].mean(axis=1)

    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")
        if len(val_idx) == 0:
            continue

        p_val = top_preds[val_idx]
        y_val = y[val_idx]

        threshold = find_optimal_threshold(y_val, p_val, metric=threshold_metric)
        fold_thresholds.append(threshold)

        metrics = asdict(evaluate_binary(y_val, p_val, threshold=threshold))
        fold_metrics.append(metrics)

    agg = aggregate_fold_metrics(fold_metrics)
    return agg, fold_metrics, fold_thresholds, top_names


def stacked_xgb_ensemble(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    df: pd.DataFrame,
    model_names: List[str],
    threshold_metric: str = "f1",
    random_state: int = 1337,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float]]:
    """
    Stacking approach: Use base model predictions as additional features
    for the best task model (all_xgb).

    This allows the model to learn when to trust each base model.
    """
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline

    # Prepare all-task features
    X_all, feature_cols = prepare_task_features(df, "all")
    X_np = X_all.to_numpy()

    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        # Combine original features with OOF predictions
        X_train_combined = np.hstack([X_np[train_idx], oof_predictions[train_idx]])
        X_val_combined = np.hstack([X_np[val_idx], oof_predictions[val_idx]])
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Compute class weight
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Build stacked model
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

        model.fit(X_train_combined, y_train)
        p_val = model.predict_proba(X_val_combined)[:, 1]

        threshold = find_optimal_threshold(y_val, p_val, metric=threshold_metric)
        fold_thresholds.append(threshold)

        metrics = asdict(evaluate_binary(y_val, p_val, threshold=threshold))
        fold_metrics.append(metrics)

        auroc_str = f"{metrics['auroc']:.3f}" if metrics['auroc'] else "N/A"
        print(f"  Fold {fold_idx}: AUROC={auroc_str}, Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

    agg = aggregate_fold_metrics(fold_metrics)
    return agg, fold_metrics, fold_thresholds


def rank_weighted_ensemble(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_names: List[str],
    threshold_metric: str = "f1",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float], Dict[str, float]]:
    """
    Ensemble using rank-based weighting.
    Weights are proportional to (AUROC - 0.5) to emphasize models with higher performance.
    """
    # Calculate weights based on OOF AUROC
    aurocs = []
    for i, name in enumerate(model_names):
        auroc = _safe_auroc(y, oof_predictions[:, i])
        aurocs.append(auroc if auroc else 0.5)

    # Weight = (AUROC - 0.5) to give more weight to better models
    weights = np.array([max(0.01, a - 0.5) for a in aurocs])
    weights = weights / weights.sum()  # Normalize

    weights_dict = {name: float(w) for name, w in zip(model_names, weights)}

    # Weighted predictions
    weighted_preds = (oof_predictions * weights).sum(axis=1)

    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")
        if len(val_idx) == 0:
            continue

        p_val = weighted_preds[val_idx]
        y_val = y[val_idx]

        threshold = find_optimal_threshold(y_val, p_val, metric=threshold_metric)
        fold_thresholds.append(threshold)

        metrics = asdict(evaluate_binary(y_val, p_val, threshold=threshold))
        fold_metrics.append(metrics)

    agg = aggregate_fold_metrics(fold_metrics)
    return agg, fold_metrics, fold_thresholds, weights_dict


def diversity_weighted_ensemble(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    child_ids: List[str],
    folds: List[Dict[str, Any]],
    model_names: List[str],
    threshold_metric: str = "f1",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float], Dict[str, float]]:
    """
    Ensemble that considers both model accuracy and diversity.
    Uses a greedy selection approach to add diverse high-performers.
    """
    n_models = len(model_names)

    # Calculate base AUROCs
    aurocs = []
    for i in range(n_models):
        auroc = _safe_auroc(y, oof_predictions[:, i])
        aurocs.append(auroc if auroc else 0.5)

    # Sort by AUROC descending
    sorted_indices = np.argsort(aurocs)[::-1]

    # Greedy selection with diversity bonus
    selected = [sorted_indices[0]]  # Start with best model
    weights = {model_names[sorted_indices[0]]: 1.0}

    for idx in sorted_indices[1:]:
        # Check correlation with already selected models
        max_corr = 0
        for sel_idx in selected:
            corr = np.corrcoef(oof_predictions[:, idx], oof_predictions[:, sel_idx])[0, 1]
            max_corr = max(max_corr, abs(corr))

        # Add model if it's diverse enough (correlation < 0.9) and has decent AUROC
        if max_corr < 0.9 and aurocs[idx] > 0.55:
            selected.append(idx)
            # Weight based on AUROC and diversity bonus
            diversity_bonus = 1 - max_corr
            weights[model_names[idx]] = (aurocs[idx] - 0.5) * (1 + diversity_bonus)

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Create weight array
    weight_array = np.zeros(n_models)
    for name, w in weights.items():
        idx = model_names.index(name)
        weight_array[idx] = w

    # Weighted predictions
    weighted_preds = (oof_predictions * weight_array).sum(axis=1)

    fold_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        val_idx, _ = _get_indices_for_ids(child_ids, fold_data["val"], f"fold{fold_idx}_val")
        if len(val_idx) == 0:
            continue

        p_val = weighted_preds[val_idx]
        y_val = y[val_idx]

        threshold = find_optimal_threshold(y_val, p_val, metric=threshold_metric)
        fold_thresholds.append(threshold)

        metrics = asdict(evaluate_binary(y_val, p_val, threshold=threshold))
        fold_metrics.append(metrics)

    agg = aggregate_fold_metrics(fold_metrics)
    return agg, fold_metrics, fold_thresholds, weights


def main() -> None:
    ap = argparse.ArgumentParser(description="Train true multi-model ensemble.")
    ap.add_argument("--features", required=True, help="Path to features CSV/parquet")
    ap.add_argument("--labels", required=True, help="Path to labels CSV")
    ap.add_argument("--splits", required=True, help="Path to splits.json (must be kfold)")
    ap.add_argument("--models_dir", default="data/derived/models", help="Dir with existing model metrics")
    ap.add_argument("--out_dir", default="data/derived/models", help="Output directory")
    ap.add_argument("--random_state", type=int, default=1337)
    ap.add_argument("--threshold-metric", default="f1", choices=["f1", "balanced_accuracy", "youden_j"])
    ap.add_argument("--tune", action="store_true", default=True, help="Tune meta-learner")
    ap.add_argument("--no-tune", dest="tune", action="store_false")
    ap.add_argument("--tune_trials", type=int, default=50)

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

    if splits["mode"] != "kfold":
        raise ValueError("True ensemble requires kfold splits for OOF predictions")

    folds = splits["folds"]
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to labeled
    labeled_ids = set(labels_df["child_id"].tolist())
    feat_df = feat_df[feat_df["child_id"].isin(labeled_ids)].copy()

    # Merge
    df = feat_df.merge(labels_df, on="child_id", how="inner")
    if df.empty:
        raise ValueError("No rows after merge.")

    y_all = df["label"].to_numpy().astype(int)
    child_ids_all = df["child_id"].tolist()
    n_samples = len(y_all)

    print(f"Samples: {n_samples}")
    print(f"Folds: {len(folds)}")
    print(f"Base models: {len(BASE_MODEL_CONFIGS)}")
    print()

    # Generate OOF predictions for each base model
    print("Generating out-of-fold predictions for base models...")
    oof_predictions = []
    model_names = []
    model_fold_aurocs = {}  # Store per-fold AUROCs for comparison with original metrics

    for config in BASE_MODEL_CONFIGS:
        task = config["task"]
        model_type = config["model"]
        name = config["name"]

        print(f"\n  {name} ({task}/{model_type})...")

        # Prepare features for this task
        X_task, feature_cols = prepare_task_features(df, task)

        # Load tuned params if available
        best_params = load_tuned_params(models_dir, task, model_type)
        if best_params:
            print(f"    Using tuned params from {task}_{model_type}_metrics.json")

        # Generate OOF predictions
        oof, fold_aurocs = generate_oof_predictions(
            X_task, y_all, child_ids_all, folds,
            model_type=model_type,
            task=task,
            best_params=best_params,
            random_state=args.random_state,
        )

        # Calculate both metrics for comparison
        global_auroc = _safe_auroc(y_all, oof)
        mean_fold_auroc = np.mean(fold_aurocs)
        model_fold_aurocs[name] = fold_aurocs

        print(f"    Mean fold AUROC: {mean_fold_auroc:.4f} (matches original metrics)")
        print(f"    Global OOF AUROC: {global_auroc:.4f}" if global_auroc else "    Global OOF AUROC: N/A")

        oof_predictions.append(oof)
        model_names.append(name)

    # Stack predictions: (n_samples, n_models)
    oof_stacked = np.column_stack(oof_predictions)

    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)

    # 1. Simple averaging baseline
    print("\n1. Simple Averaging Ensemble:")
    avg_agg, avg_fold_metrics, avg_thresholds = simple_average_ensemble(
        oof_stacked, y_all, child_ids_all, folds, args.threshold_metric
    )
    print(f"   AUROC: {avg_agg['auroc_mean']:.4f} ± {avg_agg['auroc_std']:.4f}")
    print(f"   Accuracy: {avg_agg['accuracy_mean']:.4f} ± {avg_agg['accuracy_std']:.4f}")
    print(f"   Balanced Acc: {avg_agg['balanced_accuracy_mean']:.4f} ± {avg_agg['balanced_accuracy_std']:.4f}")
    print(f"   F1: {avg_agg['f1_mean']:.4f} ± {avg_agg['f1_std']:.4f}")

    # 2. Top-3 models ensemble
    print("\n2. Top-3 Models Ensemble:")
    top3_agg, top3_fold_metrics, top3_thresholds, top3_names = top_models_ensemble(
        oof_stacked, y_all, child_ids_all, folds, model_names,
        top_k=3, threshold_metric=args.threshold_metric
    )
    print(f"   Top 3: {top3_names}")
    print(f"   AUROC: {top3_agg['auroc_mean']:.4f} ± {top3_agg['auroc_std']:.4f}")
    print(f"   Accuracy: {top3_agg['accuracy_mean']:.4f} ± {top3_agg['accuracy_std']:.4f}")
    print(f"   Balanced Acc: {top3_agg['balanced_accuracy_mean']:.4f} ± {top3_agg['balanced_accuracy_std']:.4f}")
    print(f"   F1: {top3_agg['f1_mean']:.4f} ± {top3_agg['f1_std']:.4f}")

    # 3. Optimized weighted ensemble (multi-seed averaging)
    print("\n3. Optimized Weighted Ensemble (900 trials, 3-seed averaging):")
    opt_agg, opt_fold_metrics, opt_thresholds, opt_weights = optimized_weighted_ensemble(
        oof_stacked, y_all, child_ids_all, folds, model_names,
        threshold_metric=args.threshold_metric,
        n_trials=900,  # 300 per seed x 3 seeds
        n_seeds=3,
        random_state=args.random_state,
    )
    print(f"\n   AUROC: {opt_agg['auroc_mean']:.4f} ± {opt_agg['auroc_std']:.4f}")
    print(f"   Accuracy: {opt_agg['accuracy_mean']:.4f} ± {opt_agg['accuracy_std']:.4f}")
    print(f"   Balanced Acc: {opt_agg['balanced_accuracy_mean']:.4f} ± {opt_agg['balanced_accuracy_std']:.4f}")
    print(f"   F1: {opt_agg['f1_mean']:.4f} ± {opt_agg['f1_std']:.4f}")
    print(f"   Sensitivity: {opt_agg['recall_mean']:.4f} ± {opt_agg['recall_std']:.4f}")
    print(f"   Specificity: {opt_agg['specificity_mean']:.4f} ± {opt_agg['specificity_std']:.4f}")
    print("\n   Optimized weights:")
    for name, w in sorted(opt_weights.items(), key=lambda x: -x[1]):
        print(f"     {name}: {w:.4f}")

    # 4. XGBoost meta-learner ensemble
    print("\n4. XGBoost Meta-Learner Ensemble:")
    xgb_meta_agg, xgb_meta_fold_metrics, xgb_meta_thresholds, xgb_meta_weights = xgb_meta_learner_ensemble(
        oof_stacked, y_all, child_ids_all, folds, model_names,
        threshold_metric=args.threshold_metric,
        random_state=args.random_state,
    )
    print(f"\n   AUROC: {xgb_meta_agg['auroc_mean']:.4f} ± {xgb_meta_agg['auroc_std']:.4f}")
    print(f"   Accuracy: {xgb_meta_agg['accuracy_mean']:.4f} ± {xgb_meta_agg['accuracy_std']:.4f}")
    print(f"   Balanced Acc: {xgb_meta_agg['balanced_accuracy_mean']:.4f} ± {xgb_meta_agg['balanced_accuracy_std']:.4f}")
    print(f"   F1: {xgb_meta_agg['f1_mean']:.4f} ± {xgb_meta_agg['f1_std']:.4f}")
    print(f"   Sensitivity: {xgb_meta_agg['recall_mean']:.4f} ± {xgb_meta_agg['recall_std']:.4f}")
    print(f"   Specificity: {xgb_meta_agg['specificity_mean']:.4f} ± {xgb_meta_agg['specificity_std']:.4f}")
    print("\n   Feature importances:")
    for name, w in list(xgb_meta_weights.items())[:5]:
        print(f"     {name}: {w:.4f}")

    # 5. LogReg meta-learner ensemble
    print("\n5. LogReg Meta-Learner Ensemble:")
    meta_agg, meta_fold_metrics, meta_thresholds = evaluate_ensemble_cv(
        oof_stacked, y_all, child_ids_all, folds,
        random_state=args.random_state,
        threshold_metric=args.threshold_metric,
        tune_meta=args.tune,
    )
    print(f"\n   AUROC: {meta_agg['auroc_mean']:.4f} ± {meta_agg['auroc_std']:.4f}")
    print(f"   Accuracy: {meta_agg['accuracy_mean']:.4f} ± {meta_agg['accuracy_std']:.4f}")
    print(f"   Balanced Acc: {meta_agg['balanced_accuracy_mean']:.4f} ± {meta_agg['balanced_accuracy_std']:.4f}")
    print(f"   F1: {meta_agg['f1_mean']:.4f} ± {meta_agg['f1_std']:.4f}")
    print(f"   Sensitivity: {meta_agg['recall_mean']:.4f} ± {meta_agg['recall_std']:.4f}")
    print(f"   Specificity: {meta_agg['specificity_mean']:.4f} ± {meta_agg['specificity_std']:.4f}")

    # 6. Rank-weighted ensemble
    print("\n6. Rank-Weighted Ensemble:")
    rank_agg, rank_fold_metrics, rank_thresholds, rank_weights = rank_weighted_ensemble(
        oof_stacked, y_all, child_ids_all, folds, model_names,
        threshold_metric=args.threshold_metric,
    )
    print(f"   AUROC: {rank_agg['auroc_mean']:.4f} ± {rank_agg['auroc_std']:.4f}")
    print(f"   Accuracy: {rank_agg['accuracy_mean']:.4f} ± {rank_agg['accuracy_std']:.4f}")
    print(f"   Balanced Acc: {rank_agg['balanced_accuracy_mean']:.4f} ± {rank_agg['balanced_accuracy_std']:.4f}")
    print("   Weights:")
    for name, w in sorted(rank_weights.items(), key=lambda x: -x[1]):
        print(f"     {name}: {w:.4f}")

    # 7. Diversity-weighted ensemble
    print("\n7. Diversity-Weighted Ensemble:")
    div_agg, div_fold_metrics, div_thresholds, div_weights = diversity_weighted_ensemble(
        oof_stacked, y_all, child_ids_all, folds, model_names,
        threshold_metric=args.threshold_metric,
    )
    print(f"   AUROC: {div_agg['auroc_mean']:.4f} ± {div_agg['auroc_std']:.4f}")
    print(f"   Accuracy: {div_agg['accuracy_mean']:.4f} ± {div_agg['accuracy_std']:.4f}")
    print(f"   Balanced Acc: {div_agg['balanced_accuracy_mean']:.4f} ± {div_agg['balanced_accuracy_std']:.4f}")
    print("   Selected models & weights:")
    for name, w in sorted(div_weights.items(), key=lambda x: -x[1]):
        print(f"     {name}: {w:.4f}")

    # 8. Stacked XGB ensemble (features + predictions)
    print("\n8. Stacked XGB Ensemble (features + base model predictions):")
    stacked_agg, stacked_fold_metrics, stacked_thresholds = stacked_xgb_ensemble(
        oof_stacked, y_all, child_ids_all, folds, df, model_names,
        threshold_metric=args.threshold_metric,
        random_state=args.random_state,
    )
    print(f"\n   AUROC: {stacked_agg['auroc_mean']:.4f} ± {stacked_agg['auroc_std']:.4f}")
    print(f"   Accuracy: {stacked_agg['accuracy_mean']:.4f} ± {stacked_agg['accuracy_std']:.4f}")
    print(f"   Balanced Acc: {stacked_agg['balanced_accuracy_mean']:.4f} ± {stacked_agg['balanced_accuracy_std']:.4f}")
    print(f"   F1: {stacked_agg['f1_mean']:.4f} ± {stacked_agg['f1_std']:.4f}")

    # Train final meta-learner on all data
    print("\n9. Training final meta-learner on all data...")
    final_meta, model_weights, final_scaler = train_meta_learner(
        oof_stacked, y_all, model_names,
        random_state=args.random_state,
        tune=args.tune,
        n_trials=args.tune_trials,
    )

    print("\n   Model weights (by importance):")
    for name, weight in model_weights.items():
        print(f"     {name}: {weight:.4f}")

    # Compare with best single model (XGB all)
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    xgb_all_idx = model_names.index("xgb_all")
    # Use mean fold AUROC to match original metrics format (0.725)
    xgb_all_auroc = float(np.mean(model_fold_aurocs["xgb_all"]))
    xgb_all_global = _safe_auroc(y_all, oof_predictions[xgb_all_idx])
    print(f"\n   XGB-all baseline: Mean fold AUROC = {xgb_all_auroc:.4f}, Global OOF AUROC = {xgb_all_global:.4f}")

    # Find best ensemble
    results = [
        ("XGB-all (baseline)", xgb_all_auroc, 0.701),  # From metrics
        ("Simple Average", avg_agg['auroc_mean'], avg_agg['accuracy_mean']),
        ("Top-3 Models", top3_agg['auroc_mean'], top3_agg['accuracy_mean']),
        ("Optimized Weights", opt_agg['auroc_mean'], opt_agg['accuracy_mean']),
        ("XGB Meta-Learner", xgb_meta_agg['auroc_mean'], xgb_meta_agg['accuracy_mean']),
        ("LogReg Meta-Learner", meta_agg['auroc_mean'], meta_agg['accuracy_mean']),
        ("Rank-Weighted", rank_agg['auroc_mean'], rank_agg['accuracy_mean']),
        ("Diversity-Weighted", div_agg['auroc_mean'], div_agg['accuracy_mean']),
        ("Stacked XGB", stacked_agg['auroc_mean'], stacked_agg['accuracy_mean']),
    ]

    print(f"\n   {'Method':<25} {'AUROC':>10} {'Accuracy':>10}")
    print("   " + "-"*47)
    for name, auroc, acc in results:
        auroc_str = f"{auroc:.4f}" if auroc else "N/A"
        print(f"   {name:<25} {auroc_str:>10} {acc:>10.4f}")

    # Find best by AUROC (excluding baseline)
    best_by_auroc = max(results[1:], key=lambda x: x[1])
    best_by_acc = max(results[1:], key=lambda x: x[2])
    print(f"\n   Best by AUROC: {best_by_auroc[0]} with {best_by_auroc[1]:.4f} AUROC")
    print(f"   Best by Accuracy: {best_by_acc[0]} with {best_by_acc[2]:.4f} accuracy")
    print(f"   AUROC improvement over XGB-all baseline ({xgb_all_auroc:.4f}): {'+' if best_by_auroc[1] > xgb_all_auroc else ''}{(best_by_auroc[1] - xgb_all_auroc)*100:.2f}%")
    best_method = best_by_auroc

    # Save results
    metrics = {
        "model": "true_ensemble",
        "mode": "kfold",
        "n_folds": len(folds),
        "n_base_models": len(model_names),
        "base_models": BASE_MODEL_CONFIGS,
        "threshold_metric": args.threshold_metric,
        "simple_average": {
            "cv_summary": avg_agg,
            "per_fold": avg_fold_metrics,
            "thresholds": avg_thresholds,
        },
        "top3_models": {
            "cv_summary": top3_agg,
            "per_fold": top3_fold_metrics,
            "thresholds": top3_thresholds,
            "models_used": top3_names,
        },
        "optimized_weights": {
            "cv_summary": opt_agg,
            "per_fold": opt_fold_metrics,
            "thresholds": opt_thresholds,
            "weights": opt_weights,
        },
        "xgb_meta_learner": {
            "cv_summary": xgb_meta_agg,
            "per_fold": xgb_meta_fold_metrics,
            "thresholds": xgb_meta_thresholds,
            "feature_importances": xgb_meta_weights,
        },
        "logreg_meta_learner": {
            "cv_summary": meta_agg,
            "per_fold": meta_fold_metrics,
            "thresholds": meta_thresholds,
            "model_weights": model_weights,
        },
        "rank_weighted": {
            "cv_summary": rank_agg,
            "per_fold": rank_fold_metrics,
            "thresholds": rank_thresholds,
            "weights": rank_weights,
        },
        "diversity_weighted": {
            "cv_summary": div_agg,
            "per_fold": div_fold_metrics,
            "thresholds": div_thresholds,
            "weights": div_weights,
        },
        "stacked_xgb": {
            "cv_summary": stacked_agg,
            "per_fold": stacked_fold_metrics,
            "thresholds": stacked_thresholds,
        },
        "oof_aurocs_global": {name: float(_safe_auroc(y_all, oof)) for name, oof in zip(model_names, oof_predictions)},
        "oof_aurocs_mean_fold": {name: float(np.mean(model_fold_aurocs[name])) for name in model_names},
        "per_fold_aurocs": {name: model_fold_aurocs[name] for name in model_names},
        "best_ensemble": {
            "method": best_method[0],
            "auroc": best_method[1],
            "accuracy": best_method[2],
        },
    }

    metrics_path = out_dir / "true_ensemble_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Save final ensemble (with both meta-learner and optimized weights)
    ensemble_path = out_dir / "true_ensemble.joblib"
    joblib.dump({
        "meta_learner": final_meta,
        "scaler": final_scaler,
        "model_names": model_names,
        "base_model_configs": BASE_MODEL_CONFIGS,
        "logreg_model_weights": model_weights,
        "optimized_weights": opt_weights,  # Often best performer
        "best_method": best_method[0],
    }, ensemble_path)
    print(f"Saved ensemble: {ensemble_path}")


if __name__ == "__main__":
    main()
