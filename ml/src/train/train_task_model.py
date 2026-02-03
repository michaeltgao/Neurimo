from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from sklearn.impute import SimpleImputer, KNNImputer  # type: ignore
from sklearn.experimental import enable_iterative_imputer  # type: ignore  # noqa: F401
from sklearn.impute import IterativeImputer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from sklearn.metrics import (  # type: ignore
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
)

# Optional Optuna import
try:
    import optuna  # type: ignore
    from optuna.samplers import TPESampler  # type: ignore
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class EvalMetrics:
    n: int
    pos_rate: float
    auroc: Optional[float]
    auprc: Optional[float]
    accuracy: float
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float  # sensitivity / TPR
    specificity: float  # TNR
    brier_score: float
    threshold_used: float


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    # AUROC undefined if only one class present in y_true
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Find optimal classification threshold on validation set.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for class 1
        metric: Optimization target - "f1", "balanced_accuracy", or "youden_j"

    Returns:
        Optimal threshold value
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = -1.0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == "youden_j":
            # Youden's J = sensitivity + specificity - 1
            sens = recall_score(y_true, y_pred, zero_division=0)
            spec = _compute_specificity(y_true, y_pred)
            score = sens + spec - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return float(best_threshold)


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
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=_compute_specificity(y_true, y_pred),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        threshold_used=float(threshold),
    )


def aggregate_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and std of metrics across folds."""
    keys = ["auroc", "auprc", "accuracy", "balanced_accuracy", "f1",
            "precision", "recall", "specificity", "brier_score"]
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
    """Load labels CSV with flexible column name support."""
    df = pd.read_csv(labels_path)

    # Support both 'label' and 'asd_label' column names
    label_col = None
    for candidate in ["label", "asd_label"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if "child_id" not in df.columns:
        raise ValueError(f"Labels CSV must have column 'child_id'. Got: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Labels CSV must have column 'label' or 'asd_label'. Got: {list(df.columns)}")

    df = df[["child_id", label_col]].copy()
    df = df.rename(columns={label_col: "label"})
    df["child_id"] = df["child_id"].astype(str)
    # ensure 0/1 ints
    df["label"] = df["label"].astype(int)
    return df


def select_task_features(features_df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    task: joint | imit | free | all

    'all' combines features from all tasks (ja_, imit_, fp_)
    """
    prefix_map = {
        "joint": ["ja_"],
        "imit": ["imit_"],
        "free": ["fp_"],
        "all": ["ja_", "imit_", "fp_"],  # Cross-task: use all features
    }
    if task not in prefix_map:
        raise ValueError(f"Unknown task={task}. Choose from {list(prefix_map.keys())}")

    prefixes = prefix_map[task]
    # keep only prefixed feature columns; always keep child_id for merges
    cols = [c for c in features_df.columns if any(c.startswith(p) for p in prefixes)]
    if not cols:
        raise ValueError(f"No columns found with prefixes {prefixes}. Available columns: {len(features_df.columns)}")

    X = features_df[cols].copy()
    return X, cols


def _remove_constant_columns(X: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Remove columns with zero variance (constant values or all NaN).

    Returns:
        Filtered DataFrame, remaining feature names, removed feature names
    """
    removed = []
    kept_cols = []

    for col in feature_cols:
        series = X[col]
        # Check if constant (including all-NaN)
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            removed.append(col)
        else:
            kept_cols.append(col)

    if removed:
        print(f"  Removed {len(removed)} constant/zero-variance columns")

    return X[kept_cols], kept_cols, removed


def _drop_high_missing_features(
    X: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Drop features with missing rate above threshold.

    Returns:
        Filtered DataFrame, remaining feature names, dropped feature names
    """
    missing_rates = X[feature_cols].isna().mean()
    dropped = [col for col in feature_cols if missing_rates[col] > threshold]
    kept_cols = [col for col in feature_cols if missing_rates[col] <= threshold]

    if dropped:
        print(f"  Dropped {len(dropped)} features with >{threshold:.0%} missing: {dropped}")

    return X[kept_cols], kept_cols, dropped


def _select_top_features_by_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    max_features: int,
    random_state: int = 1337,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Select top N features using mutual information.

    Returns:
        Filtered X array, selected feature names, selected indices
    """
    if max_features >= len(feature_cols):
        return X, feature_cols, list(range(len(feature_cols)))

    # Impute for feature selection (can't compute MI with NaN)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Use mutual information for feature selection
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=random_state),
        k=max_features,
    )
    selector.fit(X_imputed, y)

    # Get selected feature indices and names
    selected_mask = selector.get_support()
    selected_indices = [i for i, sel in enumerate(selected_mask) if sel]
    selected_cols = [feature_cols[i] for i in selected_indices]

    # Sort by importance score for reporting
    scores = selector.scores_
    sorted_features = sorted(zip(feature_cols, scores), key=lambda x: -x[1])
    print(f"  Selected top {max_features} features by mutual information:")
    for name, score in sorted_features[:max_features]:
        print(f"    {name}: {score:.4f}")

    return X[:, selected_indices], selected_cols, selected_indices


def _get_imputer(method: str, random_state: int = 1337) -> Any:
    """Get imputer based on method name."""
    if method == "median":
        return SimpleImputer(strategy="median")
    elif method == "knn":
        return KNNImputer(n_neighbors=5, weights="distance")
    elif method == "iterative":
        return IterativeImputer(random_state=random_state, max_iter=20)
    else:
        raise ValueError(f"Unknown impute_method: {method}. Choose from: median, knn, iterative")


def build_model(
    model_type: str,
    random_state: int,
    scale_pos_weight: Optional[float] = None,
    impute_method: str = "median",
    logreg_C: float = 1.0,
    logreg_penalty: str = "l2",
    xgb_learning_rate: float = 0.02,
    xgb_reg_lambda: float = 1.0,
    xgb_min_child_weight: int = 2,
    xgb_colsample_bytree: float = 0.5,
    xgb_subsample: float = 0.7,
) -> Pipeline:
    imputer = _get_imputer(impute_method, random_state)

    if model_type == "logreg":
        # Strong baseline: imputing + scaling + L2 logistic
        # Lower C = stronger regularization (helps with overfitting)
        solver = "lbfgs" if logreg_penalty == "l2" else "saga"
        return Pipeline(
            steps=[
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=logreg_C,
                    penalty=logreg_penalty,
                    max_iter=5000,
                    class_weight="balanced",
                    solver=solver,
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

        # Anti-overfitting defaults for small datasets (<500 samples)
        # Key changes: higher min_child_weight, lower learning_rate, more feature sampling
        xgb_params = {
            "n_estimators": 500,
            "learning_rate": xgb_learning_rate,
            "max_depth": 2,  # Shallow trees
            "subsample": xgb_subsample,
            "colsample_bytree": xgb_colsample_bytree,
            "reg_lambda": xgb_reg_lambda,
            "min_child_weight": xgb_min_child_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": random_state,
            "n_jobs": -1,
        }

        # Handle class imbalance via scale_pos_weight
        if scale_pos_weight is not None:
            xgb_params["scale_pos_weight"] = scale_pos_weight

        xgb = XGBClassifier(**xgb_params)

        return Pipeline(
            steps=[
                ("imputer", imputer),
                ("clf", xgb),
            ]
        )

    if model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
        except Exception as e:
            raise ImportError(
                "LightGBM not available. Install with `pip install lightgbm` (or add to your env)."
            ) from e

        # Good defaults for small/medium tabular dataset
        lgbm_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 3,
            "num_leaves": 8,  # 2^max_depth for balanced tree
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_samples": 5,
            "objective": "binary",
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,  # Suppress warnings
        }

        # Handle class imbalance via scale_pos_weight
        if scale_pos_weight is not None:
            lgbm_params["scale_pos_weight"] = scale_pos_weight

        lgbm = LGBMClassifier(**lgbm_params)  # type: ignore[arg-type]

        return Pipeline(
            steps=[
                ("imputer", imputer),
                ("clf", lgbm),
            ]
        )

    raise ValueError("model_type must be one of: logreg, xgb, lgbm")


def build_model_with_params(
    model_type: str,
    params: Dict[str, Any],
    random_state: int,
    scale_pos_weight: Optional[float] = None,
    impute_method: str = "median",
) -> Pipeline:
    """Build model with custom hyperparameters (used by Optuna tuning)."""
    imputer = _get_imputer(impute_method, random_state)

    if model_type == "logreg":
        return Pipeline(
            steps=[
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                    C=params.get("C", 1.0),
                )),
            ]
        )

    if model_type == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore[import-not-found]
        except Exception as e:
            raise ImportError("XGBoost not available.") from e

        # Anti-overfitting defaults for small datasets
        xgb_params = {
            "n_estimators": params.get("n_estimators", 500),
            "learning_rate": params.get("learning_rate", 0.02),
            "max_depth": params.get("max_depth", 2),
            "subsample": params.get("subsample", 0.7),
            "colsample_bytree": params.get("colsample_bytree", 0.5),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "reg_alpha": params.get("reg_alpha", 0.0),
            "min_child_weight": params.get("min_child_weight", 2),
            "gamma": params.get("gamma", 0.0),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": random_state,
            "n_jobs": -1,
        }
        if scale_pos_weight is not None:
            xgb_params["scale_pos_weight"] = scale_pos_weight

        return Pipeline(
            steps=[
                ("imputer", imputer),
                ("clf", XGBClassifier(**xgb_params)),
            ]
        )

    if model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
        except Exception as e:
            raise ImportError("LightGBM not available.") from e

        lgbm_params = {
            "n_estimators": params.get("n_estimators", 500),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 3),
            "num_leaves": params.get("num_leaves", 8),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "reg_alpha": params.get("reg_alpha", 0.0),
            "min_child_samples": params.get("min_child_samples", 5),
            "objective": "binary",
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        if scale_pos_weight is not None:
            lgbm_params["scale_pos_weight"] = scale_pos_weight

        return Pipeline(
            steps=[
                ("imputer", imputer),
                ("clf", LGBMClassifier(**lgbm_params)),  # type: ignore[arg-type]
            ]
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def _optuna_objective(
    trial: "optuna.Trial",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str,
    scale_pos_weight: float,
    random_state: int,
) -> float:
    """Optuna objective function for hyperparameter tuning."""
    if model_type == "logreg":
        params = {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        }
    elif model_type == "xgb":
        # Anti-overfitting search space for small datasets
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),  # Shallow trees only
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7),  # More feature sampling
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 6),  # Min 2 to prevent memorization
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = build_model_with_params(
        model_type, params, random_state, scale_pos_weight
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Optimize AUROC
    if len(np.unique(y_val)) < 2:
        return 0.5
    return float(roc_auc_score(y_val, y_prob))


def run_optuna_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    scale_pos_weight: float,
    random_state: int,
    n_trials: int = 50,
    n_cv_folds: int = 3,
    timeout: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Run Optuna hyperparameter tuning with internal cross-validation.

    Returns:
        Tuple of (best_params dict, best_auroc score)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Install with: pip install optuna")

    def objective(trial: "optuna.Trial") -> float:
        # Use internal CV for more robust tuning
        skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
        aurocs = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]

            auroc = _optuna_objective(
                trial, X_tr, y_tr, X_va, y_va,
                model_type, scale_pos_weight, random_state
            )
            aurocs.append(auroc)

        return float(np.mean(aurocs))

    # Create study with TPE sampler
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{model_type}_tuning",
    )

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    return study.best_params, study.best_value


def _get_indices_for_ids(
    child_ids_all: List[str],
    target_ids: List[str],
    split_name: str = "split",
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Get indices of child_ids_all that are in target_ids.

    Returns:
        Tuple of (indices, validation_info dict with missing/found counts)
    """
    available_set = set(child_ids_all)
    target_set = set(target_ids)

    found_ids = target_set & available_set
    missing_ids = target_set - available_set

    indices = [i for i, cid in enumerate(child_ids_all) if cid in target_set]

    validation_info = {
        "requested": len(target_ids),
        "found": len(found_ids),
        "missing": len(missing_ids),
        "missing_ids": sorted(list(missing_ids))[:10],  # First 10 for debugging
    }

    if missing_ids:
        warnings.warn(
            f"{split_name}: {len(missing_ids)}/{len(target_ids)} child_ids not found in features. "
            f"First few missing: {sorted(list(missing_ids))[:5]}"
        )

    return indices, validation_info


def _predict_prob(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Get predicted probabilities for class 1."""
    return model.predict_proba(X)[:, 1]


def _fit_with_sample_weight(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    early_stopping_rounds: Optional[int] = None,
) -> None:
    """Fit model with optional sample weights and early stopping.

    Args:
        model: Pipeline with imputer and classifier
        X: Training features
        y: Training labels
        sample_weight: Optional sample weights
        X_val: Validation features for early stopping
        y_val: Validation labels for early stopping
        early_stopping_rounds: Stop if no improvement for N rounds (XGBoost/LightGBM only)
    """
    fit_params: Dict[str, Any] = {}

    if sample_weight is not None:
        fit_params["clf__sample_weight"] = sample_weight

    # Early stopping for XGBoost/LightGBM
    clf = model.named_steps["clf"]
    if early_stopping_rounds is not None and X_val is not None and y_val is not None:
        if hasattr(clf, "set_params"):
            # Check if it's XGBoost or LightGBM
            clf_name = clf.__class__.__name__
            if clf_name in ("XGBClassifier", "LGBMClassifier"):
                clf.set_params(early_stopping_rounds=early_stopping_rounds)
                # Impute validation data using the imputer
                imputer = model.named_steps["imputer"]
                # Fit imputer on training data first, then transform both
                X_imputed = imputer.fit_transform(X)
                X_val_imputed = imputer.transform(X_val)
                # Fit classifier directly with eval_set
                if sample_weight is not None:
                    clf.fit(X_imputed, y, eval_set=[(X_val_imputed, y_val)],
                           sample_weight=sample_weight, verbose=False)
                else:
                    clf.fit(X_imputed, y, eval_set=[(X_val_imputed, y_val)], verbose=False)
                return

    # Standard fit (no early stopping)
    if fit_params:
        model.fit(X, y, **fit_params)
    else:
        model.fit(X, y)


def _extract_feature_importance(
    model: Pipeline,
    feature_cols: List[str],
) -> Dict[str, float]:
    """Extract feature importances from trained model."""
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        # XGBoost, RandomForest, etc.
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # LogisticRegression - use absolute coefficient values
        importances = np.abs(clf.coef_[0])
    else:
        return {}

    # Return as dict sorted by importance
    importance_dict = dict(zip(feature_cols, importances.tolist()))
    return dict(sorted(importance_dict.items(), key=lambda x: -x[1]))


def _compute_missing_rates(X: pd.DataFrame) -> Dict[str, float]:
    """Compute per-column missing rates."""
    missing = X.isna().mean()
    return {str(col): float(rate) for col, rate in missing.items() if rate > 0}


def _train_holdout(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    child_ids_all: List[str],
    splits: Dict[str, Any],
    args: argparse.Namespace,
    feature_cols: List[str],
    out_dir: Path,
    sample_weight_all: Optional[np.ndarray] = None,
) -> None:
    """Train and evaluate using holdout split."""
    train_idx, train_info = _get_indices_for_ids(child_ids_all, splits["train"], "train")
    val_idx, val_info = _get_indices_for_ids(child_ids_all, splits["val"], "val")
    test_idx, test_info = _get_indices_for_ids(child_ids_all, splits["test"], "test")

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

    sw_train = sample_weight_all[train_idx] if sample_weight_all is not None else None

    # Compute scale_pos_weight for XGBoost
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Optuna hyperparameter tuning
    best_params: Optional[Dict[str, Any]] = None
    if args.tune:
        print(f"  Running Optuna hyperparameter tuning ({args.tune_trials} trials)...")
        best_params, best_auroc = run_optuna_tuning(
            X_train, y_train,
            model_type=args.model,
            scale_pos_weight=scale_pos_weight,
            random_state=args.random_state,
            n_trials=args.tune_trials,
            n_cv_folds=args.tune_cv_folds,
            timeout=args.tune_timeout,
        )
        print(f"  Best params: {best_params}")
        print(f"  Best CV AUROC: {best_auroc:.4f}")

        model = build_model_with_params(
            args.model,
            best_params,
            random_state=args.random_state,
            scale_pos_weight=scale_pos_weight if args.model == "xgb" else None,
            impute_method=args.impute_method,
        )
    else:
        model = build_model(
            args.model,
            random_state=args.random_state,
            scale_pos_weight=scale_pos_weight if args.model in ("xgb", "lgbm") else None,
            impute_method=args.impute_method,
            logreg_C=args.logreg_C,
            logreg_penalty=args.logreg_penalty,
            xgb_learning_rate=args.xgb_learning_rate,
            xgb_reg_lambda=args.xgb_reg_lambda,
            xgb_min_child_weight=args.xgb_min_child_weight,
            xgb_colsample_bytree=args.xgb_colsample_bytree,
            xgb_subsample=args.xgb_subsample,
        )

    # Fit with early stopping if enabled (for XGBoost/LightGBM)
    early_stop_rounds = args.early_stopping if args.early_stopping > 0 else None
    _fit_with_sample_weight(
        model, X_train, y_train, sw_train,
        X_val=X_val, y_val=y_val, early_stopping_rounds=early_stop_rounds
    )

    # Apply probability calibration if requested (using CV-based calibration)
    if args.calibrate:
        print("  Applying probability calibration (isotonic)...")
        # Get transformed training data
        if "scaler" in model.named_steps:
            X_train_transformed = model.named_steps["scaler"].transform(
                model.named_steps["imputer"].transform(X_train)
            )
        else:
            X_train_transformed = model.named_steps["imputer"].transform(X_train)

        # Create a fresh base classifier with same params and calibrate with CV
        base_clf = model.named_steps["clf"]
        calibrated_clf = CalibratedClassifierCV(
            base_clf.__class__(**base_clf.get_params()),
            method="isotonic",
            cv=3,
        )
        calibrated_clf.fit(X_train_transformed, y_train)
        model.named_steps["clf"] = calibrated_clf

    p_train = _predict_prob(model, X_train)
    p_val = _predict_prob(model, X_val)
    p_test = _predict_prob(model, X_test)

    # Find optimal threshold on validation set
    if args.auto_threshold:
        optimal_threshold = find_optimal_threshold(y_val, p_val, metric=args.threshold_metric)
        print(f"  Optimal threshold ({args.threshold_metric}): {optimal_threshold:.3f}")
    else:
        optimal_threshold = args.threshold

    # Extract feature importance
    feature_importance = _extract_feature_importance(model, feature_cols)

    metrics = {
        "task": args.task,
        "model": args.model,
        "mode": "holdout",
        "n_features": int(len(feature_cols)),
        "features_prefix": {"joint": "ja_", "imit": "imit_", "free": "fp_", "all": "ja_,imit_,fp_"}[args.task],
        "threshold": float(optimal_threshold),
        "auto_threshold": args.auto_threshold,
        "threshold_metric": args.threshold_metric if args.auto_threshold else None,
        "sample_weight_col": args.quality_col,
        "sample_weight_floor": args.quality_floor if args.quality_col else None,
        "tuning": {
            "enabled": args.tune,
            "n_trials": args.tune_trials if args.tune else None,
            "best_params": best_params,
        } if args.tune else None,
        "calibration": args.calibrate,
        "split_validation": {
            "train": train_info,
            "val": val_info,
            "test": test_info,
        },
        "class_balance": {
            "train": {"n": len(y_train), "n_pos": n_pos, "n_neg": n_neg, "pos_rate": float(n_pos / len(y_train))},
            "val": {"n": len(y_val), "n_pos": int((y_val == 1).sum()), "n_neg": int((y_val == 0).sum())},
            "test": {"n": len(y_test), "n_pos": int((y_test == 1).sum()), "n_neg": int((y_test == 0).sum())},
        },
        "splits": {
            "train": asdict(evaluate_binary(y_train, p_train, threshold=optimal_threshold)),
            "val": asdict(evaluate_binary(y_val, p_val, threshold=optimal_threshold)),
            "test": asdict(evaluate_binary(y_test, p_test, threshold=optimal_threshold)),
        },
        "feature_cols": feature_cols,
        "feature_importance_top20": dict(list(feature_importance.items())[:20]),
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
    sample_weight_all: Optional[np.ndarray] = None,
) -> None:
    """Train and evaluate using k-fold cross-validation."""
    folds = splits["folds"]
    n_folds = len(folds)
    tag = f"{args.task}_{args.model}"

    fold_val_metrics: List[Dict[str, Any]] = []
    fold_train_metrics: List[Dict[str, Any]] = []
    fold_thresholds: List[float] = []
    all_feature_importance: Dict[str, List[float]] = {col: [] for col in feature_cols}

    # Run Optuna tuning once on all data (with internal CV) if requested
    best_params: Optional[Dict[str, Any]] = None
    if args.tune:
        print(f"  Running Optuna hyperparameter tuning ({args.tune_trials} trials)...")
        X_all_np = X_all.to_numpy()
        # Compute global scale_pos_weight
        n_neg_all = int((y_all == 0).sum())
        n_pos_all = int((y_all == 1).sum())
        global_scale_pos_weight = n_neg_all / n_pos_all if n_pos_all > 0 else 1.0

        best_params, best_auroc = run_optuna_tuning(
            X_all_np, y_all,
            model_type=args.model,
            scale_pos_weight=global_scale_pos_weight,
            random_state=args.random_state,
            n_trials=args.tune_trials,
            n_cv_folds=args.tune_cv_folds,
            timeout=args.tune_timeout,
        )
        print(f"  Best params: {best_params}")
        print(f"  Best internal CV AUROC: {best_auroc:.4f}")

    print(f"Running {n_folds}-fold cross-validation...")

    for fold_data in folds:
        fold_idx = fold_data["fold"]
        train_idx, _ = _get_indices_for_ids(child_ids_all, fold_data["train"], f"fold{fold_idx}_train")
        val_idx, _ = _get_indices_for_ids(child_ids_all, fold_data["val"], f"fold{fold_idx}_val")

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {fold_idx} has empty train or val split after filtering.")

        X_train = X_all.iloc[train_idx].to_numpy()
        y_train = y_all[train_idx]
        X_val = X_all.iloc[val_idx].to_numpy()
        y_val = y_all[val_idx]

        sw_train = sample_weight_all[train_idx] if sample_weight_all is not None else None

        # Compute scale_pos_weight for XGBoost
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        if best_params is not None:
            model = build_model_with_params(
                args.model,
                best_params,
                random_state=args.random_state + fold_idx,
                scale_pos_weight=scale_pos_weight if args.model == "xgb" else None,
                impute_method=args.impute_method,
            )
        else:
            model = build_model(
                args.model,
                random_state=args.random_state + fold_idx,
                scale_pos_weight=scale_pos_weight if args.model in ("xgb", "lgbm") else None,
                impute_method=args.impute_method,
                logreg_C=args.logreg_C,
                logreg_penalty=args.logreg_penalty,
                xgb_learning_rate=args.xgb_learning_rate,
                xgb_reg_lambda=args.xgb_reg_lambda,
                xgb_min_child_weight=args.xgb_min_child_weight,
                xgb_colsample_bytree=args.xgb_colsample_bytree,
                xgb_subsample=args.xgb_subsample,
            )
        # Fit with early stopping if enabled (for XGBoost/LightGBM)
        early_stop_rounds = args.early_stopping if args.early_stopping > 0 else None
        _fit_with_sample_weight(
            model, X_train, y_train, sw_train,
            X_val=X_val, y_val=y_val, early_stopping_rounds=early_stop_rounds
        )

        # Apply calibration if requested (using CV-based calibration)
        if args.calibrate:
            # Get transformed training data
            if "scaler" in model.named_steps:
                X_train_transformed = model.named_steps["scaler"].transform(
                    model.named_steps["imputer"].transform(X_train)
                )
            else:
                X_train_transformed = model.named_steps["imputer"].transform(X_train)

            # Create a fresh base classifier with same params and calibrate with CV
            base_clf = model.named_steps["clf"]
            calibrated_clf = CalibratedClassifierCV(
                base_clf.__class__(**base_clf.get_params()),
                method="isotonic",
                cv=3,
            )
            calibrated_clf.fit(X_train_transformed, y_train)
            model.named_steps["clf"] = calibrated_clf

        p_train = _predict_prob(model, X_train)
        p_val = _predict_prob(model, X_val)

        # Find optimal threshold on validation set
        if args.auto_threshold:
            fold_threshold = find_optimal_threshold(y_val, p_val, metric=args.threshold_metric)
        else:
            fold_threshold = args.threshold
        fold_thresholds.append(fold_threshold)

        train_metrics = asdict(evaluate_binary(y_train, p_train, threshold=fold_threshold))
        val_metrics = asdict(evaluate_binary(y_val, p_val, threshold=fold_threshold))

        fold_train_metrics.append(train_metrics)
        fold_val_metrics.append(val_metrics)

        # Accumulate feature importance
        fold_importance = _extract_feature_importance(model, feature_cols)
        for col in feature_cols:
            if col in fold_importance:
                all_feature_importance[col].append(fold_importance[col])

        # Save per-fold model
        fold_model_path = out_dir / f"{tag}_fold{fold_idx}.joblib"
        joblib.dump(model, fold_model_path)

        auroc_str = f"{val_metrics['auroc']:.3f}" if val_metrics['auroc'] is not None else "N/A"
        print(f"  Fold {fold_idx}: val_auroc={auroc_str}, val_f1={val_metrics['f1']:.3f}, threshold={fold_threshold:.3f}")

    # Aggregate metrics
    agg_val = aggregate_fold_metrics(fold_val_metrics)
    agg_train = aggregate_fold_metrics(fold_train_metrics)

    # Average feature importance across folds
    avg_feature_importance = {
        col: float(np.mean(vals)) for col, vals in all_feature_importance.items() if vals
    }
    avg_feature_importance = dict(sorted(avg_feature_importance.items(), key=lambda x: -x[1]))

    metrics = {
        "task": args.task,
        "model": args.model,
        "mode": "kfold",
        "n_folds": n_folds,
        "n_features": int(len(feature_cols)),
        "features_prefix": {"joint": "ja_", "imit": "imit_", "free": "fp_", "all": "ja_,imit_,fp_"}[args.task],
        "threshold_mean": float(np.mean(fold_thresholds)),
        "threshold_std": float(np.std(fold_thresholds)),
        "auto_threshold": args.auto_threshold,
        "threshold_metric": args.threshold_metric if args.auto_threshold else None,
        "sample_weight_col": args.quality_col,
        "sample_weight_floor": args.quality_floor if args.quality_col else None,
        "tuning": {
            "enabled": args.tune,
            "n_trials": args.tune_trials if args.tune else None,
            "best_params": best_params,
        } if args.tune else None,
        "calibration": args.calibrate,
        "cv_summary": {
            "train": agg_train,
            "val": agg_val,
        },
        "per_fold": {
            "train": fold_train_metrics,
            "val": fold_val_metrics,
            "thresholds": fold_thresholds,
        },
        "feature_cols": feature_cols,
        "feature_importance_top20": dict(list(avg_feature_importance.items())[:20]),
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
    print(f"  Sensitivity: {agg_val['recall_mean']:.3f} ± {agg_val['recall_std']:.3f}")
    print(f"  Specificity: {agg_val['specificity_mean']:.3f} ± {agg_val['specificity_std']:.3f}")
    print(f"  Brier Score: {agg_val['brier_score_mean']:.3f} ± {agg_val['brier_score_std']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a per-task model (joint/imit/free).")
    ap.add_argument("--features", required=True, help="Path to features_merged.csv (or parquet).")
    ap.add_argument("--labels", required=True, help="Path to labels_usable.csv (or labels_clean.csv).")
    ap.add_argument("--splits", required=True, help="Path to data/derived/splits.json (holdout or kfold)")
    ap.add_argument("--task", required=True, choices=["joint", "imit", "free", "all"], help="Which task feature set to use. 'all' combines features from all tasks.")
    ap.add_argument("--model", required=True, choices=["logreg", "xgb", "lgbm"], help="Model type.")
    ap.add_argument("--out_dir", default="data/derived/models", help="Directory to save model + metrics.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for metrics (if --no-auto-threshold).")
    ap.add_argument("--random_state", type=int, default=1337, help="Random seed.")

    # Threshold optimization
    ap.add_argument(
        "--auto-threshold", action="store_true", default=True,
        help="Automatically find optimal threshold on validation set (default: True)."
    )
    ap.add_argument(
        "--no-auto-threshold", dest="auto_threshold", action="store_false",
        help="Use fixed --threshold value instead of optimizing."
    )
    ap.add_argument(
        "--threshold-metric", type=str, default="f1", choices=["f1", "balanced_accuracy", "youden_j"],
        help="Metric to optimize when finding threshold (default: f1)."
    )

    # QC-aware training options
    ap.add_argument(
        "--quality_col", type=str, default=None,
        help="Column name for sample weights (e.g., qc_ja_quality, qc_overall_quality). "
             "If provided, samples are weighted by this column during training."
    )
    ap.add_argument(
        "--quality_floor", type=float, default=0.3,
        help="Minimum sample weight to prevent zero-weighting low-quality samples (default: 0.3)."
    )

    # Optuna hyperparameter tuning options
    ap.add_argument(
        "--tune", action="store_true", default=False,
        help="Enable Optuna hyperparameter tuning (requires optuna package)."
    )
    ap.add_argument(
        "--tune_trials", type=int, default=50,
        help="Number of Optuna trials for hyperparameter search (default: 50)."
    )
    ap.add_argument(
        "--tune_cv_folds", type=int, default=3,
        help="Number of CV folds for internal Optuna tuning (default: 3)."
    )
    ap.add_argument(
        "--tune_timeout", type=int, default=None,
        help="Timeout in seconds for Optuna tuning (default: None = no timeout)."
    )

    # Calibration options
    ap.add_argument(
        "--calibrate", action="store_true", default=False,
        help="Apply probability calibration (isotonic regression) after training."
    )

    # Early stopping options (anti-overfitting)
    ap.add_argument(
        "--early_stopping", type=int, default=20,
        help="Early stopping rounds for XGBoost/LightGBM (default: 20). Set to 0 to disable."
    )

    # Feature reduction options
    ap.add_argument(
        "--max_features", type=int, default=None,
        help="Maximum number of features to use (selects top N by mutual information). "
             "Recommended: 5-8 for small datasets."
    )
    ap.add_argument(
        "--feature_cols", type=str, default=None,
        help="Comma-separated list of exact feature columns to use (overrides --max_features). "
             "Example: 'ja_repetitive_motion_score,imit_arms_response_present'"
    )
    ap.add_argument(
        "--drop_missing_thresh", type=float, default=None,
        help="Drop features with missing rate above this threshold (e.g., 0.5 for 50%%). "
             "Recommended: 0.3-0.5 for better imputation quality."
    )

    # Imputation options
    ap.add_argument(
        "--impute_method", type=str, default="median", choices=["median", "knn", "iterative"],
        help="Imputation method: median (fast), knn (k-nearest neighbors), iterative (MICE). "
             "Default: median. For better accuracy with missing data, try knn or iterative."
    )

    # LogReg-specific options for regularization
    ap.add_argument(
        "--logreg_C", type=float, default=1.0,
        help="Regularization strength for LogReg (inverse of regularization). "
             "Lower = stronger regularization. Default: 1.0. Try 0.1 or 0.01 for overfitting."
    )
    ap.add_argument(
        "--logreg_penalty", type=str, default="l2", choices=["l1", "l2"],
        help="Penalty type for LogReg. l1=sparse features, l2=ridge. Default: l2."
    )

    # XGB-specific options for regularization
    ap.add_argument(
        "--xgb_learning_rate", type=float, default=0.02,
        help="XGB learning rate. Lower = slower learning, less overfitting. Default: 0.02."
    )
    ap.add_argument(
        "--xgb_reg_lambda", type=float, default=1.0,
        help="XGB L2 regularization. Higher = stronger regularization. Default: 1.0. Try 2-5 for overfitting."
    )
    ap.add_argument(
        "--xgb_min_child_weight", type=int, default=2,
        help="XGB min_child_weight. Higher = more conservative splits. Default: 2. Try 3-5 for overfitting."
    )
    ap.add_argument(
        "--xgb_colsample_bytree", type=float, default=0.5,
        help="XGB column sampling ratio. Lower = more regularization. Default: 0.5. Try 0.3-0.4."
    )
    ap.add_argument(
        "--xgb_subsample", type=float, default=0.7,
        help="XGB row sampling ratio. Lower = more regularization. Default: 0.7. Try 0.5-0.6."
    )
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

    # Enforce one row per child_id to prevent data leakage
    if not feat_df["child_id"].is_unique:
        duplicate_ids = feat_df[feat_df["child_id"].duplicated(keep=False)]["child_id"].unique()
        raise ValueError(
            f"Features file has {len(duplicate_ids)} duplicate child_ids. "
            f"This can cause data leakage between train/val/test splits. "
            f"First few duplicates: {list(duplicate_ids[:5])}. "
            f"Please aggregate to one row per child before training."
        )

    labels_df = load_labels(labels_path)
    splits = load_splits(splits_path)

    # Filter to labeled children only
    labeled_ids = set(labels_df["child_id"].tolist())
    feat_df = feat_df[feat_df["child_id"].isin(labeled_ids)].copy()

    # Merge labels
    df = feat_df.merge(labels_df, on="child_id", how="inner")
    if df.empty:
        raise ValueError("No rows left after merging features with labels. Check child_id formats.")

    # Select features: either explicit list or task-based selection
    if args.feature_cols:
        # Use explicitly specified feature columns
        feature_cols = [c.strip() for c in args.feature_cols.split(",")]
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found in data: {missing_cols}")
        X_all = df[feature_cols].copy()
        print(f"Using {len(feature_cols)} explicitly specified features")
    else:
        # Standard task-based feature selection
        X_all, feature_cols = select_task_features(df, task=args.task)

        # Remove constant/zero-variance columns
        X_all, feature_cols, _ = _remove_constant_columns(X_all, feature_cols)
        if not feature_cols:
            raise ValueError(f"No features remaining after removing constant columns for task '{args.task}'.")

        # Drop high-missing features if threshold specified
        if args.drop_missing_thresh is not None:
            X_all, feature_cols, _ = _drop_high_missing_features(
                X_all, feature_cols, threshold=args.drop_missing_thresh
            )
            if not feature_cols:
                raise ValueError(
                    f"No features remaining after dropping high-missing features. "
                    f"Try a higher --drop_missing_thresh value."
                )

    y_all = df["label"].to_numpy().astype(int)
    child_ids_all = df["child_id"].tolist()

    # Select top features by mutual information if max_features specified (only if not using explicit feature_cols)
    if not args.feature_cols and args.max_features is not None and args.max_features < len(feature_cols):
        X_all_np, feature_cols, _ = _select_top_features_by_importance(
            X_all.to_numpy(), y_all, feature_cols,
            max_features=args.max_features,
            random_state=args.random_state,
        )
        # Convert back to DataFrame for compatibility
        X_all = pd.DataFrame(X_all_np, columns=feature_cols)

    # Compute sample weights if quality column specified
    sample_weight_all: Optional[np.ndarray] = None
    if args.quality_col:
        if args.quality_col not in df.columns:
            raise ValueError(
                f"Quality column '{args.quality_col}' not found. "
                f"Available columns: {[c for c in df.columns if c.startswith('qc_')]}"
            )
        # Fill NaN with 0.5 (neutral), clip to [floor, 1.0]
        sample_weight_all = (
            df[args.quality_col]
            .fillna(0.5)
            .clip(args.quality_floor, 1.0)
            .to_numpy()
        )
        print(f"Using sample weights from '{args.quality_col}' (floor={args.quality_floor})")
        assert sample_weight_all is not None  # Assigned above
        print(f"  Weight stats: min={sample_weight_all.min():.3f}, max={sample_weight_all.max():.3f}, "
              f"mean={sample_weight_all.mean():.3f}")

    # Report missing rates
    missing_rates = _compute_missing_rates(X_all)
    if missing_rates:
        high_missing = {k: v for k, v in missing_rates.items() if v > 0.3}
        if high_missing:
            print(f"  Warning: {len(high_missing)} features have >30% missing values")

    print(f"Task: {args.task}, Model: {args.model}")
    print(f"Samples: {len(df)}, Features: {len(feature_cols)}")
    print(f"Mode: {splits['mode']}")
    print(f"Auto-threshold: {args.auto_threshold}" + (f" (metric: {args.threshold_metric})" if args.auto_threshold else ""))

    if splits["mode"] == "kfold":
        _train_kfold(X_all, y_all, child_ids_all, splits, args, feature_cols, out_dir, sample_weight_all)
    else:
        _train_holdout(X_all, y_all, child_ids_all, splits, args, feature_cols, out_dir, sample_weight_all)


if __name__ == "__main__":
    main()
