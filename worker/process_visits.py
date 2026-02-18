"""
Worker: Process visits end-to-end.

Polls DB for visits ready for ML prediction and processes them.

Usage:
    python -m worker.process_visits          # poll loop
    python -m worker.process_visits --once   # single pass

Ready visit criteria:
- Has 3 uploaded videos (joint_attention, imitation, free_play)
- Has questionnaire
- No ml_prediction exists yet
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker, joinedload

from .config import (
    DATABASE_URL,
    POLL_INTERVAL_SECONDS,
    MODELS_DIR,
    VIDEO_STORAGE_DIR,
    REPO_ROOT,
)
from .feature_pipeline import get_feature_pipeline

# ORM models — resolved via PYTHONPATH (set to repo root)
from backend.app.models.base import Base
from backend.app.models.visit import Visit
from backend.app.models.video import Video
from backend.app.models.questionnaire import Questionnaire
from backend.app.models.ml_prediction import MLPrediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_video_path(storage_path: str) -> Path:
    """
    Resolve a video storage_path from the DB to an absolute path.

    The backend stores paths that may be absolute (Docker) or relative.
    Try several strategies to locate the actual file.
    """
    p = Path(storage_path)

    # Already absolute and exists
    if p.is_absolute() and p.exists():
        return p

    # Relative — resolve against repo root
    resolved = REPO_ROOT / p
    if resolved.exists():
        return resolved

    # Try just the filename in VIDEO_STORAGE_DIR
    fallback = VIDEO_STORAGE_DIR / p.name
    if fallback.exists():
        return fallback

    # Return the repo-root-resolved path (will fail later with a clear error)
    return resolved


class VisitWorker:
    """Worker that processes visits for ML predictions."""

    REQUIRED_TASKS = {"joint_attention", "imitation", "free_play"}

    def __init__(self):
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._predictor = None

    @property
    def predictor(self):
        """Lazy-load ML predictor."""
        if self._predictor is None:
            from ml.src.inference.ensemble_predictor import EnsemblePredictor

            self._predictor = EnsemblePredictor(models_dir=str(MODELS_DIR))
            self._predictor.load()
            logger.info("ML predictor loaded")
        return self._predictor

    def get_ready_visits(self, db: Session) -> List[Visit]:
        """
        Find visits ready for ML prediction.

        Criteria: 3 videos with distinct task types + questionnaire + no prediction yet.
        """
        stmt = (
            select(Visit)
            .options(
                joinedload(Visit.videos),
                joinedload(Visit.questionnaire),
                joinedload(Visit.ml_prediction),
            )
            .where(Visit.ml_prediction == None)  # noqa: E711
        )

        visits = db.execute(stmt).unique().scalars().all()

        ready = []
        for visit in visits:
            if not visit.questionnaire:
                continue
            task_types = {v.task_type for v in visit.videos}
            if task_types >= self.REQUIRED_TASKS:
                ready.append(visit)

        return ready

    def process_visit(self, db: Session, visit: Visit) -> bool:
        """
        Process a single visit: extract features → predict → save.

        Returns True if successful.
        """
        try:
            logger.info(f"Processing visit {visit.id} (child {visit.child_id})")

            # Use visit.id as the unique identifier for tracks / features
            visit_key = str(visit.id)

            # 1. Build video list with resolved paths
            pipeline = get_feature_pipeline()
            videos = [
                {
                    "task_type": v.task_type,
                    "storage_path": str(_resolve_video_path(v.storage_path)),
                }
                for v in visit.videos
                if v.task_type in self.REQUIRED_TASKS
            ]

            features = pipeline.process_videos(videos=videos, child_id=visit_key)

            # 2. Add questionnaire features
            q = visit.questionnaire
            features.update({
                "q_regression": int(q.regression),
                "q_seizures": int(q.seizures),
                "q_motor_delay": int(q.motor_delay),
                "q_global_delay": int(q.global_delay),
                "q_family_history": int(q.family_history_asd_ndd),
                "q_dysmorphic": int(q.dysmorphic_features),
                "q_macrocephaly": int(q.macrocephaly),
                "q_microcephaly": int(q.microcephaly),
            })

            # 3. Add age
            features["age_months"] = visit.age_months

            # 4. ML prediction
            result = self.predictor.predict(features)
            logger.info(f"  Prediction: {result}")

            prob = result["probability"]
            risk_bucket = self._probability_to_bucket(prob)
            explanations = self._generate_explanations(features, prob)

            # 5. Save to DB (upsert pattern for reprocessing)
            existing = db.query(MLPrediction).filter(
                MLPrediction.visit_id == visit.id
            ).first()

            if existing:
                # Update existing prediction
                existing.asd_risk_bucket = risk_bucket
                existing.probability = prob
                existing.explanations = explanations
                logger.info(f"  Updated existing prediction: {risk_bucket}")
            else:
                # Create new prediction
                prediction = MLPrediction(
                    visit_id=visit.id,
                    asd_risk_bucket=risk_bucket,
                    probability=prob,
                    explanations=explanations,
                )
                db.add(prediction)
                logger.info(f"  Saved new prediction: {risk_bucket}")

            db.commit()
            return True

        except Exception as e:
            logger.error(f"  Failed to process visit {visit.id}: {e}", exc_info=True)
            db.rollback()
            return False

    def _probability_to_bucket(self, prob: float) -> str:
        """Map probability to risk bucket (4 levels, score 0-100).

        Uses < instead of <= for upper bounds so that displayed scores
        match bucket labels (e.g., 75 displayed = moderate-high, 76+ = high).
        """
        # low: 0-25, moderate: 26-50, moderate-high: 51-75, high: 76-100
        score = round(prob * 100)
        if score <= 25:
            return "low"
        elif score <= 50:
            return "moderate"
        elif score <= 75:
            return "moderate-high"
        else:
            return "high"

    def _generate_explanations(
        self, features: dict, prob: float
    ) -> List[str]:
        """Generate rule-based explanation strings from extracted features."""
        explanations: List[str] = []

        # Joint attention
        if features.get("ja_attention_response_rate", 1.0) < 0.5:
            explanations.append("Lower than typical response to attention bids")
        if features.get("ja_follow_point_rate", 1.0) < 0.5:
            explanations.append("Reduced gaze-following when parent points")
        if features.get("ja_orient_success", 1.0) < 0.5:
            explanations.append("Limited orienting to name or social cues")

        # Imitation - check if demos were detected first
        demos_detected = (
            features.get("imit_clap_demo_present", 0) == 1 or
            features.get("imit_arms_demo_present", 0) == 1
        )
        if demos_detected:
            # Demos were detected - check for responses
            if features.get("imit_score", 1.0) == 0:
                explanations.append("No imitation of demonstrated actions observed")
            elif features.get("imit_arms_response_present", 1) == 0 and features.get("imit_arms_demo_present", 0) == 1:
                explanations.append("No arm-raise imitation response observed")
            if features.get("imit_clap_response_present", 1) == 0 and features.get("imit_clap_demo_present", 0) == 1:
                explanations.append("No clapping imitation after demonstration")

        # Free play
        if features.get("fp_repetitive_motion_time_frac", 0) > 0.3:
            explanations.append("Elevated repetitive motion patterns during free play")
        if features.get("fp_engaged_time_frac", 1.0) < 0.3:
            explanations.append("Limited social engagement during free play")
        if features.get("fp_hand_to_face_time_frac", 0) > 0.15:
            explanations.append("Frequent hand-to-face contact during free play")

        # Common features
        if features.get("ja_stillness_ratio", 0) > 0.5:
            explanations.append("Extended periods of stillness during joint attention")
        if features.get("fp_lack_of_eye_contact_duration_adj", 0) > 0.5:
            explanations.append("Reduced eye contact during free play")

        # Questionnaire flags
        if features.get("q_regression", 0):
            explanations.append("Developmental regression reported by caregiver")
        if features.get("q_motor_delay", 0):
            explanations.append("Motor delay reported by caregiver")

        return explanations

    def run_once(self) -> int:
        """Process all ready visits once. Returns count processed."""
        with self.SessionLocal() as db:
            visits = self.get_ready_visits(db)
            logger.info(f"Found {len(visits)} visits ready for processing")

            processed = 0
            for visit in visits:
                if self.process_visit(db, visit):
                    processed += 1

            return processed

    def run_forever(self, poll_interval: int = POLL_INTERVAL_SECONDS):
        """Run worker in polling loop."""
        logger.info(f"Starting worker (poll interval: {poll_interval}s)")

        while True:
            try:
                processed = self.run_once()
                if processed > 0:
                    logger.info(f"Processed {processed} visits")
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)

            time.sleep(poll_interval)


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Neurimo visit processing worker")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_SECONDS)
    args = parser.parse_args()

    worker = VisitWorker()

    if args.once:
        worker.run_once()
    else:
        worker.run_forever(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
