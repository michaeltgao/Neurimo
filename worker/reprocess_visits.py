"""
Re-process visits to generate event CSV files for guided review.

This script:
1. Deletes ml_prediction records for specified visits (or all)
2. Optionally clears existing track directories
3. Re-runs the worker to process visits

Usage:
    python -m worker.reprocess_visits              # Re-process all visits
    python -m worker.reprocess_visits --visit-ids 32 33 34  # Specific visits
    python -m worker.reprocess_visits --clear-tracks        # Also clear track dirs
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from .config import DATABASE_URL, TRACKS_DIR
from .process_visits import VisitWorker

# ORM models
from backend.app.models.ml_prediction import MLPrediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def reprocess_visits(
    visit_ids: list[int] | None = None,
    clear_tracks: bool = False,
):
    """Re-process visits by clearing predictions and re-running worker."""
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as db:
        # Delete ml_prediction records
        if visit_ids:
            stmt = delete(MLPrediction).where(MLPrediction.visit_id.in_(visit_ids))
            result = db.execute(stmt)
            logger.info(f"Deleted {result.rowcount} ml_prediction records for visits: {visit_ids}")
        else:
            stmt = delete(MLPrediction)
            result = db.execute(stmt)
            logger.info(f"Deleted {result.rowcount} ml_prediction records (all)")

        db.commit()

    # Optionally clear track directories
    if clear_tracks:
        tracks_path = Path(TRACKS_DIR)
        if visit_ids:
            for vid in visit_ids:
                visit_dir = tracks_path / f"visit_{vid}"
                if visit_dir.exists():
                    shutil.rmtree(visit_dir)
                    logger.info(f"Cleared {visit_dir}")
        else:
            # Clear all visit_* directories
            for visit_dir in tracks_path.glob("visit_*"):
                shutil.rmtree(visit_dir)
                logger.info(f"Cleared {visit_dir}")

    # Re-run worker
    logger.info("Starting worker to re-process visits...")
    worker = VisitWorker()
    processed = worker.run_once()
    logger.info(f"Re-processed {processed} visits")

    return processed


def main():
    parser = argparse.ArgumentParser(description="Re-process visits for guided review")
    parser.add_argument(
        "--visit-ids",
        type=int,
        nargs="+",
        help="Specific visit IDs to re-process (default: all)",
    )
    parser.add_argument(
        "--clear-tracks",
        action="store_true",
        help="Clear existing track directories before re-processing",
    )
    args = parser.parse_args()

    reprocess_visits(
        visit_ids=args.visit_ids,
        clear_tracks=args.clear_tracks,
    )


if __name__ == "__main__":
    main()