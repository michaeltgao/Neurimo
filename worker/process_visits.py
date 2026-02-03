"""
Worker: Process visits end-to-end.

Polls DB for visits ready for ML prediction and processes them.

Usage:
    python -m worker.process_visits

Ready visit criteria:
- Has 3 uploaded videos (joint_attention, imitation, free_play)
- Has questionnaire
- No ml_prediction exists yet
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from sqlalchemy import create_engine, select, and_
from sqlalchemy.orm import Session, sessionmaker, joinedload

from .config import DATABASE_URL, POLL_INTERVAL_SECONDS, MAX_RETRIES, MODELS_DIR
from .feature_pipeline import get_feature_pipeline

# Import models (adjust path as needed)
import sys
sys.path.insert(0, str(MODELS_DIR.parent.parent.parent / "backend"))
from backend.app.models.visit import Visit
from backend.app.models.video import Video
from backend.app.models.questionnaire import Questionnaire
from backend.app.models.ml_prediction import MLPrediction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class VisitWorker:
    """Worker that processes visits for ML predictions."""

    REQUIRED_TASKS = {"joint_attention", "imitation", "free_play"}

    def __init__(self):
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Lazy load predictor
        self._predictor = None

    @property
    def predictor(self):
        """Lazy load ML predictor."""
        if self._predictor is None:
            from ml.src.inference.ensemble_predictor import EnsemblePredictor
            self._predictor = EnsemblePredictor(models_dir=str(MODELS_DIR))
            self._predictor.load()
            logger.info("ML predictor loaded")
        return self._predictor

    def get_ready_visits(self, db: Session) -> List[Visit]:
        """
        Find visits ready for ML prediction.

        Criteria:
        - Has 3 videos with distinct task types
        - Has questionnaire
        - No ml_prediction exists
        """
        # Query visits with eager loading
        stmt = (
            select(Visit)
            .options(
                joinedload(Visit.videos),
                joinedload(Visit.questionnaire),
                joinedload(Visit.ml_prediction),
            )
            .where(Visit.ml_prediction == None)  # No prediction yet
        )

        visits = db.execute(stmt).unique().scalars().all()

        # Filter to those with 3 videos + questionnaire
        ready = []
        for visit in visits:
            if not visit.questionnaire:
                continue

            task_types = {v.task_type for v in visit.videos}
            if task_types == self.REQUIRED_TASKS:
                ready.append(visit)

        return ready

    def process_visit(self, db: Session, visit: Visit) -> bool:
        """
        Process a single visit: extract features → predict → save.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing visit {visit.id} (child {visit.child_id})")

            # 1. Extract features from videos
            pipeline = get_feature_pipeline()
            videos = [
                {"task_type": v.task_type, "storage_path": v.storage_path}
                for v in visit.videos
            ]

            features = pipeline.process_videos(
                videos=videos,
                child_id=str(visit.child_id),
            )

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

            # 4. Get ML prediction
            result = self.predictor.predict(features)
            logger.info(f"  Prediction: {result}")

            # 5. Map probability to risk bucket
            prob = result["probability"]
            risk_bucket = self._probability_to_bucket(prob)

            # 6. Generate explanations (placeholder)
            explanations = self._generate_explanations(features, prob)

            # 7. Save prediction to DB
            prediction = MLPrediction(
                visit_id=visit.id,
                asd_risk_bucket=risk_bucket,
                explanations=explanations,
            )
            db.add(prediction)
            db.commit()

            logger.info(f"  Saved prediction: {risk_bucket}")
            return True

        except Exception as e:
            logger.error(f"  Failed: {e}")
            db.rollback()
            return False

    def _probability_to_bucket(self, prob: float) -> str:
        """Map probability to risk bucket."""
        if prob < 0.25:
            return "low"
        elif prob < 0.50:
            return "medium"
        elif prob < 0.75:
            return "med-high"
        else:
            return "high"

    def _generate_explanations(
        self, features: dict, prob: float
    ) -> List[dict]:
        """
        Generate human-readable explanations for the prediction.

        TODO: Implement SHAP or feature importance based explanations.
        """
        explanations = []

        # Placeholder: flag key features
        if features.get("ja_attention_response_rate", 1.0) < 0.5:
            explanations.append({
                "feature": "attention_response",
                "message": "Lower than typical response to attention bids",
            })

        if features.get("imit_arms_response_present", 1) == 0:
            explanations.append({
                "feature": "imitation",
                "message": "No arm imitation response observed",
            })

        if features.get("fp_repetitive_motion_score", 0) > 0.3:
            explanations.append({
                "feature": "repetitive_motion",
                "message": "Elevated repetitive motion patterns",
            })

        return explanations

    def run_once(self) -> int:
        """
        Process all ready visits once.

        Returns:
            Number of visits processed
        """
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
                logger.error(f"Error in worker loop: {e}")

            time.sleep(poll_interval)


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser()
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
