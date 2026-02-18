"""
Ensemble predictor for web app inference.

Loads the 3-model XGB simple average ensemble and provides predictions.

Usage:
    from ml.src.inference.ensemble_predictor import EnsemblePredictor

    predictor = EnsemblePredictor(models_dir="data/derived/models")

    # Get prediction for a single sample (dict of features)
    result = predictor.predict(features_dict)
    # result = {"probability": 0.73, "prediction": 1, "risk_level": "high"}

    # Get predictions for a DataFrame
    results = predictor.predict_batch(features_df)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.special import expit  # type: ignore

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    3-model XGB simple average ensemble predictor.

    Models: all_xgb, joint_xgb, free_xgb (equal 1/3 weights)
    AUROC: 0.847 (unbiased, validated via nested CV)

    Why simple average over optimized weights:
    - With small sample size (107), weight optimization overfits
    - Simple average is more robust to distribution shift
    - Achieves higher unbiased AUROC than nested CV optimized weights
    """

    # Best ensemble configuration - simple average of 3 XGB models
    ENSEMBLE_CONFIG: Dict[str, Any] = {
        "models": ["all_xgb", "joint_xgb", "free_xgb"],
        "weights": {
            "all_xgb": 1/3,
            "joint_xgb": 1/3,
            "free_xgb": 1/3,
        },
        "auroc": 0.847,
    }

    # Model to task mapping
    MODEL_TASK_MAP = {
        "all_xgb": ("all", "xgb"),
        "joint_xgb": ("joint", "xgb"),
        "free_xgb": ("free", "xgb"),
        "imit_xgb": ("imit", "xgb"),
        "imit_logreg": ("imit", "logreg"),
        "free_logreg": ("free", "logreg"),
    }

    def __init__(
        self,
        models_dir: Union[str, Path] = "data/derived/models",
        use_all_folds: bool = True,
        threshold: float = 0.5,
    ):
        """
        Initialize the ensemble predictor.

        Args:
            models_dir: Directory containing saved models and metrics
            use_all_folds: If True, average predictions across all 5 folds (more stable)
                          If False, use only fold 0 (faster)
            threshold: Classification threshold for binary predictions
        """
        self.models_dir = Path(models_dir)
        self.use_all_folds = use_all_folds
        self.threshold = threshold
        self.n_folds = 5

        self._models: Dict[str, List[Any]] = {}
        self._feature_cols: Dict[str, List[str]] = {}
        self._calibration: Dict[str, float] | None = None
        self._loaded = False

    def load(self) -> "EnsemblePredictor":
        """Load all models and feature columns."""
        if self._loaded:
            return self

        print("Loading ensemble models...")

        for model_name in self.ENSEMBLE_CONFIG["models"]:
            task, model_type = self.MODEL_TASK_MAP[model_name]

            # Load feature columns from metrics
            metrics_path = self.models_dir / f"{task}_{model_type}_metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Metrics not found: {metrics_path}")

            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            self._feature_cols[model_name] = metrics["feature_cols"]

            # Load fold models
            fold_models = []
            folds_to_load = range(self.n_folds) if self.use_all_folds else [0]

            for fold_idx in folds_to_load:
                model_path = self.models_dir / f"{task}_{model_type}_fold{fold_idx}.joblib"
                if not model_path.exists():
                    raise FileNotFoundError(f"Model not found: {model_path}")
                fold_models.append(joblib.load(model_path))

            self._models[model_name] = fold_models
            print(f"  Loaded {model_name}: {len(fold_models)} fold(s), {len(self._feature_cols[model_name])} features")

        # Load Platt scaling calibration if available
        cal_path = self.models_dir / "ensemble_calibration.json"
        if cal_path.exists():
            with open(cal_path, "r") as f:
                cal = json.load(f)
            self._calibration = {"a": cal["a"], "b": cal["b"]}
            print(f"  Loaded calibration: sigmoid({cal['a']:.2f} * raw + {cal['b']:.2f})")
        else:
            print("  No calibration file found — using raw predictions")

        self._loaded = True
        print(f"Ensemble loaded: {len(self._models)} models")
        return self

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def get_required_features(self) -> List[str]:
        """Get list of all required feature columns."""
        self._ensure_loaded()
        all_features = set()
        for cols in self._feature_cols.values():
            all_features.update(cols)
        return sorted(all_features)

    def predict_proba(self, features: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Get probability predictions for positive class.

        Args:
            features: Dict of feature values (single sample) or DataFrame (batch)

        Returns:
            Array of probabilities (shape: (n_samples,))
        """
        self._ensure_loaded()

        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        # Get predictions from each model
        model_preds = {}

        for model_name in self.ENSEMBLE_CONFIG["models"]:
            feature_cols = self._feature_cols[model_name]

            # Check for missing features
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing features for {model_name}: {missing}")

            X = df[feature_cols].to_numpy()

            # Get predictions from fold models and average
            fold_preds = []
            for model in self._models[model_name]:
                preds = model.predict_proba(X)[:, 1]
                fold_preds.append(preds)

            # Average across folds
            model_preds[model_name] = np.mean(fold_preds, axis=0)

        # Simple average ensemble (equal 1/3 weights)
        ensemble_raw = np.mean(list(model_preds.values()), axis=0)
        logger.info(f"  Ensemble raw: {ensemble_raw[0]:.4f}")

        # Apply Platt scaling calibration
        if self._calibration is not None:
            a, b = self._calibration["a"], self._calibration["b"]
            ensemble_pred = expit(a * ensemble_raw + b)
            logger.info(f"  Ensemble calibrated: {ensemble_pred[0]:.4f}")
        else:
            ensemble_pred = ensemble_raw

        return ensemble_pred

    def predict(self, features: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
        """
        Get full predictions with probability, class, and risk level.

        Args:
            features: Dict (single sample) or DataFrame (batch)

        Returns:
            Dict (single sample) or DataFrame (batch) with columns:
            - probability: float [0, 1]
            - prediction: int (0 or 1)
            - risk_level: str ("low", "moderate", "moderate-high", "high")
        """
        proba = self.predict_proba(features)

        # Single sample
        if isinstance(features, dict):
            p = float(proba[0])
            return {
                "probability": p,
                "prediction": int(p >= self.threshold),
                "risk_level": self._get_risk_level(p),
            }

        # Batch
        results = pd.DataFrame({
            "probability": proba,
            "prediction": (proba >= self.threshold).astype(int),
            "risk_level": [self._get_risk_level(p) for p in proba],
        })
        return results

    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level (4 buckets, score 0-100).

        Uses the rounded display score to ensure bucket labels match
        what users see (e.g., 75 displayed = moderate-high, 76+ = high).
        """
        # low: 0-25, moderate: 26-50, moderate-high: 51-75, high: 76-100
        score = round(probability * 100)
        if score <= 25:
            return "low"
        elif score <= 50:
            return "moderate"
        elif score <= 75:
            return "moderate-high"
        else:
            return "high"


# Convenience function for quick predictions
def get_predictor(models_dir: str = "data/derived/models") -> EnsemblePredictor:
    """Get a loaded predictor instance."""
    return EnsemblePredictor(models_dir=models_dir).load()


if __name__ == "__main__":
    # Example usage
    predictor = EnsemblePredictor(models_dir="data/derived/models")
    predictor.load()

    print("\nRequired features:")
    for feat in predictor.get_required_features():
        print(f"  - {feat}")
