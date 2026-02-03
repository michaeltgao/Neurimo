"""
Stacking ensemble training for improved AUROC.

Combines Logistic Regression and XGBoost base learners with a meta-learner
to leverage the strengths of both models.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
from sklearn.ensemble import StackingClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

# Import shared utilities from train_task_model
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
        _extract_feature_importance,
        aggregate_fold_metrics,
        _safe_auroc,
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
        _extract_feature_importance,
        aggregate_fold_metrics,
        _safe_auroc,
    )

# Optional Optuna import
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def build_base_estimators(
    random_state: int,
    scale_pos_weight: float,
    xgb_params: Optional[Dict[str, Any]] = None,
    logreg_params: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Pipeline]]:
    """
    Build base estimator pipelines for stacking.

    Returns list of (name, estimator) tuples.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError("XGBoost required. Install with: pip install xgboost") from e

    # Default XGB params
    default_xgb = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if xgb_params:
        default_xgb.update(xgb_params)

    # Default logreg params
    default_logreg = {
        "C": 1.0,
        "max_iter": 5000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": random_state,
    }
    if logreg_params:
        default_logreg.update(logreg_params)

    estimators = [
        ("logreg", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**default_logreg)),
        ])),
        ("xgb", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(**default_xgb)),
        ])),
    ]

    return estimators


def build_stacking_ensemble(
    random_state: int,
    scale_pos_weight: float,
    cv_folds: int = 5,
    xgb_params: Optional[Dict[str, Any]] = None,
    logreg_params: Optional[Dict[str, Any]] = None,
    meta_learner: str = "logreg",
    passthrough: bool = False,
) -> StackingClassifier:
    """
    Build a stacking ensemble with logreg + xgb base learners.

    Args:
        random_state: Random seed
        scale_pos_weight: Class imbalance weight for XGB
        cv_folds: Number of CV folds for stacking
        xgb_params: Custom XGB hyperparameters
        logreg_params: Custom logreg hyperparameters
        meta_learner: Meta-learner type ("logreg" or "xgb")
        passthrough: Whether to pass original features to meta-learner

    Returns:
        StackingClassifier instance
    """
    estimators = build_base_estimators(
        random_state, scale_pos_weight, xgb_params, logreg_params
    )

    # Meta-learner
    if meta_learner == "logreg":
        final_estimator = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        from xgboost import XGBClassifier
        final_estimator = XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv_folds,
        stack_method="predict_proba",
        passthrough=passthrough,
        n_jobs=-1,
    )

    return stack


def _optuna_ensemble_objective(
    trial: "optuna.Trial",
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    random_state: int,
    n_cv_folds: int = 3,
) -> float:
    """Optuna objective for tuning ensemble hyperparameters."""
    # XGB params
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500, step=100),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 2, 6),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.1, 10.0, log=True),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 7),
    }

    # Logreg params
    logreg_params = {
        "C": trial.suggest_float("logreg_C", 0.01, 10.0, log=True),
    }

    # Meta-learner choice
    meta_learner = trial.suggest_categorical("meta_learner", ["logreg", "xgb"])
    passthrough = trial.suggest_categorical("passthrough", [True, False])

    # Build and evaluate
    stack = build_stacking_ensemble(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        cv_folds=3,  # Internal CV for stacking
        xgb_params=xgb_params,
        logreg_params=logreg_params,
        meta_learner=meta_learner,
        passthrough=passthrough,
    )

    # Evaluate with CV
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    aurocs = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        y_tr, y_va = y_train[train_idx], y_train[val_idx]

        try:
            stack.fit(X_tr, y_tr)
            y_prob = stack.predict_proba(X_va)[:, 1]
            auroc = _safe_auroc(y_va, y_prob)
            if auroc is not None:
                aurocs.append(auroc)
        except Exception:
            continue

    return float(np.mean(aurocs)) if aurocs else 0.5


def run_optuna_ensemble_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    random_state: int,
    n_trials: int = 30,
    timeout: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Run Optuna hyperparameter tuning for ensemble.

    Returns:
        Tuple of (best_params dict, best_auroc)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Install with: pip install optuna")

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="ensemble_tuning",
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        return _optuna_ensemble_objective(
            trial, X_train, y_train, scale_pos_weight, random_state
        )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    return study.best_params, study.best_value


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
    scale_pos_weight: float,
    tune: bool = False,
    tune_trials: int = 30,
    calibrate: bool = False,
    xgb_params: Optional[Dict[str, Any]] = None,
    logreg_params: Optional[Dict[str, Any]] = None,
) -> Tuple[StackingClassifier, Dict[str, Any]]:
    """
    Train stacking ensemble with optional tuning and calibration.

    Returns:
        Tuple of (trained model, training info dict)
    """
    best_params = None

    if tune:
        print("  Running Optuna hyperparameter tuning for ensemble...")
        best_params, best_auroc = run_optuna_ensemble_tuning(
            X_train, y_train, scale_pos_weight, random_state,
            n_trials=tune_trials,
        )
        print(f"  Best ensemble params: {best_params}")
        print(f"  Best CV AUROC: {best_auroc:.4f}")

        # Extract params
        xgb_params = {k.replace("xgb_", ""): v for k, v in best_params.items() if k.startswith("xgb_")}
        logreg_params = {k.replace("logreg_", ""): v for k, v in best_params.items() if k.startswith("logreg_")}
        meta_learner = best_params.get("meta_learner", "logreg")
        passthrough = best_params.get("passthrough", False)
    else:
        meta_learner = "logreg"
        passthrough = False

    # Build ensemble
    ensemble = build_stacking_ensemble(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        cv_folds=5,
        xgb_params=xgb_params,
        logreg_params=logreg_params,
        meta_learner=meta_learner,
        passthrough=passthrough,
    )

    # Fit
    ensemble.fit(X_train, y_train)

    # Calibrate if requested (using CV-based calibration on training data)
    if calibrate:
        print("  Applying probability calibration...")
        # Create fresh ensemble and wrap with calibration
        fresh_ensemble = build_stacking_ensemble(
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            cv_folds=5,
            xgb_params=xgb_params,
            logreg_params=logreg_params,
            meta_learner=meta_learner,
            passthrough=passthrough,
        )
        calibrated = CalibratedClassifierCV(fresh_ensemble, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        ensemble = calibrated

    info = {
        "tuned": tune,
        "best_params": best_params,
        "calibrated": calibrate,
        "meta_learner": meta_learner,
        "passthrough": passthrough,
    }

    return ensemble, info


def _train_holdout_ensemble(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    child_ids_all: List[str],
    splits: Dict[str, Any],
    args: argparse.Namespace,
    feature_cols: List[str],
    out_dir: Path,
) -> None:
    """Train ensemble using holdout split."""
    train_idx, train_info = _get_indices_for_ids(child_ids_all, splits["train"], "train")
    val_idx, val_info = _get_indices_for_ids(child_ids_all, splits["val"], "val")
    test_idx, test_info = _get_indices_for_ids(child_ids_all, splits["test"], "test")

    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(idxs) == 0:
            raise ValueError(f"Split '{name}' has 0 samples.")

    X_train = X_all.iloc[train_idx].to_numpy()
    y_train = y_all[train_idx]
    X_val = X_all.iloc[val_idx].to_numpy()
    y_val = y_all[val_idx]
    X_test = X_all.iloc[test_idx].to_numpy()
    y_test = y_all[test_idx]

    # Impute for training
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # Class balance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"Training ensemble: {len(y_train)} train, {len(y_val)} val, {len(y_test)} test")

    # Train
    ensemble, train_info_dict = train_ensemble(
        X_train_imp, y_train, X_val_imp, y_val,
        random_state=args.random_state,
        scale_pos_weight=scale_pos_weight,
        tune=args.tune,
        tune_trials=args.tune_trials,
        calibrate=args.calibrate,
    )

    # Predict
    p_train = ensemble.predict_proba(X_train_imp)[:, 1]
    p_val = ensemble.predict_proba(X_val_imp)[:, 1]
    p_test = ensemble.predict_proba(X_test_imp)[:, 1]

    # Threshold optimization
    if args.auto_threshold:
        optimal_threshold = find_optimal_threshold(y_val, p_val, metric=args.threshold_metric)
        print(f"  Optimal threshold ({args.threshold_metric}): {optimal_threshold:.3f}")
    else:
        optimal_threshold = 0.5

    # Metrics
    metrics = {
        "task": args.task,
        "model": "ensemble_stack",
        "mode": "holdout",
        "n_features": len(feature_cols),
        "ensemble_info": train_info_dict,
        "threshold": float(optimal_threshold),
        "auto_threshold": args.auto_threshold,
        "split_validation": {
            "train": train_info,
            "val": val_info,
            "test": test_info,
        },
        "class_balance": {
            "train": {"n": len(y_train), "n_pos": n_pos, "n_neg": n_neg},
            "val": {"n": len(y_val), "n_pos": int((y_val == 1).sum())},
            "test": {"n": len(y_test), "n_pos": int((y_test == 1).sum())},
        },
        "splits": {
            "train": asdict(evaluate_binary(y_train, p_train, threshold=optimal_threshold)),
            "val": asdict(evaluate_binary(y_val, p_val, threshold=optimal_threshold)),
            "test": asdict(evaluate_binary(y_test, p_test, threshold=optimal_threshold)),
        },
        "feature_cols": feature_cols,
    }

    # Save
    tag = f"{args.task}_ensemble"
    model_path = out_dir / f"{tag}.joblib"
    metrics_path = out_dir / f"{tag}_metrics.json"

    joblib.dump({"ensemble": ensemble, "imputer": imputer}, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nVAL metrics:")
    print(json.dumps(metrics["splits"]["val"], indent=2))
    print("\nTEST metrics:")
    print(json.dumps(metrics["splits"]["test"], indent=2))


def _train_kfold_ensemble(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    child_ids_all: List[str],
    splits: Dict[str, Any],
    args: argparse.Namespace,
    feature_cols: List[str],
    out_dir: Path,
) -> None:
    """Train ensemble using k-fold cross-validation."""
    folds = splits["folds"]
    n_folds = len(folds)
    tag = f"{args.task}_ensemble"

    fold_val_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []

    # Global imputer fit on all data
    imputer = SimpleImputer(strategy="median")
    X_all_imp = imputer.fit_transform(X_all.to_numpy())

    # Compute global scale_pos_weight
    n_neg = int((y_all == 0).sum())
    n_pos = int((y_all == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Tune once on all data if requested
    best_params = None
    if args.tune:
        print(f"  Running Optuna tuning ({args.tune_trials} trials)...")
        best_params, best_auroc = run_optuna_ensemble_tuning(
            X_all_imp, y_all, scale_pos_weight, args.random_state,
            n_trials=args.tune_trials,
        )
        print(f"  Best params: {best_params}")
        print(f"  Best CV AUROC: {best_auroc:.4f}")

    print(f"Running {n_folds}-fold cross-validation for ensemble...")

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids_all, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids_all, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {fold_idx} has empty split.")

        X_train = X_all_imp[train_idx]
        y_train = y_all[train_idx]
        X_val = X_all_imp[val_idx]
        y_val = y_all[val_idx]

        # Build ensemble with tuned params if available
        if best_params:
            xgb_params = {k.replace("xgb_", ""): v for k, v in best_params.items() if k.startswith("xgb_")}
            logreg_params = {k.replace("logreg_", ""): v for k, v in best_params.items() if k.startswith("logreg_")}
            meta_learner = best_params.get("meta_learner", "logreg")
            passthrough = best_params.get("passthrough", False)
        else:
            xgb_params, logreg_params = None, None
            meta_learner, passthrough = "logreg", False

        # Compute fold scale_pos_weight
        fold_n_neg = int((y_train == 0).sum())
        fold_n_pos = int((y_train == 1).sum())
        fold_scale_pos_weight = fold_n_neg / fold_n_pos if fold_n_pos > 0 else 1.0

        ensemble = build_stacking_ensemble(
            random_state=args.random_state + fold_idx,
            scale_pos_weight=fold_scale_pos_weight,
            cv_folds=3,
            xgb_params=xgb_params,
            logreg_params=logreg_params,
            meta_learner=meta_learner,
            passthrough=passthrough,
        )
        ensemble.fit(X_train, y_train)

        # Calibrate if requested (using CV on training data)
        if args.calibrate:
            fresh_ensemble = build_stacking_ensemble(
                random_state=args.random_state + fold_idx,
                scale_pos_weight=fold_scale_pos_weight,
                cv_folds=3,
                xgb_params=xgb_params,
                logreg_params=logreg_params,
                meta_learner=meta_learner,
                passthrough=passthrough,
            )
            ensemble = CalibratedClassifierCV(fresh_ensemble, method="isotonic", cv=3)
            ensemble.fit(X_train, y_train)

        p_val = ensemble.predict_proba(X_val)[:, 1]

        # Threshold
        if args.auto_threshold:
            fold_threshold = find_optimal_threshold(y_val, p_val, metric=args.threshold_metric)
        else:
            fold_threshold = 0.5
        fold_thresholds.append(fold_threshold)

        val_metrics = asdict(evaluate_binary(y_val, p_val, threshold=fold_threshold))
        fold_val_metrics.append(val_metrics)

        # Save fold model
        fold_model_path = out_dir / f"{tag}_fold{fold_idx}.joblib"
        joblib.dump({"ensemble": ensemble, "imputer": imputer}, fold_model_path)

        auroc_str = f"{val_metrics['auroc']:.3f}" if val_metrics['auroc'] else "N/A"
        print(f"  Fold {fold_idx}: val_auroc={auroc_str}, val_f1={val_metrics['f1']:.3f}")

    # Aggregate
    agg_val = aggregate_fold_metrics(fold_val_metrics)

    metrics = {
        "task": args.task,
        "model": "ensemble_stack",
        "mode": "kfold",
        "n_folds": n_folds,
        "n_features": len(feature_cols),
        "tuning": {
            "enabled": args.tune,
            "best_params": best_params,
        } if args.tune else None,
        "calibration": args.calibrate,
        "threshold_mean": float(np.mean(fold_thresholds)),
        "threshold_std": float(np.std(fold_thresholds)),
        "cv_summary": {"val": agg_val},
        "per_fold": {"val": fold_val_metrics, "thresholds": fold_thresholds},
        "feature_cols": feature_cols,
    }

    metrics_path = out_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {n_folds} fold models: {out_dir}/{tag}_fold*.joblib")
    print(f"Saved metrics: {metrics_path}")
    print("\nCV VAL metrics (mean ± std):")
    print(f"  AUROC: {agg_val['auroc_mean']:.3f} ± {agg_val['auroc_std']:.3f}" if agg_val['auroc_mean'] else "  AUROC: N/A")
    print(f"  F1:    {agg_val['f1_mean']:.3f} ± {agg_val['f1_std']:.3f}")
    print(f"  Sensitivity: {agg_val['recall_mean']:.3f} ± {agg_val['recall_std']:.3f}")
    print(f"  Specificity: {agg_val['specificity_mean']:.3f} ± {agg_val['specificity_std']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train stacking ensemble (logreg + XGBoost).")
    ap.add_argument("--features", required=True, help="Path to features CSV/parquet")
    ap.add_argument("--labels", required=True, help="Path to labels CSV")
    ap.add_argument("--splits", required=True, help="Path to splits.json")
    ap.add_argument("--task", required=True, choices=["joint", "imit", "free"])
    ap.add_argument("--out_dir", default="data/derived/models")
    ap.add_argument("--random_state", type=int, default=1337)

    # Threshold
    ap.add_argument("--auto-threshold", action="store_true", default=True)
    ap.add_argument("--no-auto-threshold", dest="auto_threshold", action="store_false")
    ap.add_argument("--threshold-metric", default="f1", choices=["f1", "balanced_accuracy", "youden_j"])

    # Tuning
    ap.add_argument("--tune", action="store_true", help="Enable Optuna tuning")
    ap.add_argument("--tune_trials", type=int, default=30)

    # Calibration
    ap.add_argument("--calibrate", action="store_true", help="Apply probability calibration")

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

    # Filter to labeled
    labeled_ids = set(labels_df["child_id"].tolist())
    feat_df = feat_df[feat_df["child_id"].isin(labeled_ids)].copy()

    # Merge
    df = feat_df.merge(labels_df, on="child_id", how="inner")
    if df.empty:
        raise ValueError("No rows after merge.")

    # Select task features
    X_all, feature_cols = select_task_features(df, task=args.task)
    X_all, feature_cols, _ = _remove_constant_columns(X_all, feature_cols)

    y_all = df["label"].to_numpy().astype(int)
    child_ids_all = df["child_id"].tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: {args.task}")
    print(f"Samples: {len(df)}, Features: {len(feature_cols)}")
    print(f"Mode: {splits['mode']}")
    print(f"Tuning: {args.tune}, Calibration: {args.calibrate}")

    if splits["mode"] == "kfold":
        _train_kfold_ensemble(X_all, y_all, child_ids_all, splits, args, feature_cols, out_dir)
    else:
        _train_holdout_ensemble(X_all, y_all, child_ids_all, splits, args, feature_cols, out_dir)


if __name__ == "__main__":
    main()
