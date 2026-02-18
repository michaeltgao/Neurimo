"""Worker configuration."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve repo root (neurimo/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from repo root
load_dotenv(dotenv_path=REPO_ROOT / ".env")

# Database — WORKER_DATABASE_URL overrides DATABASE_URL for local dev
# (.env has db:5432 for Docker; locally you may need localhost:5432)
DATABASE_URL = os.getenv("WORKER_DATABASE_URL", os.getenv("DATABASE_URL", ""))
if not DATABASE_URL:
    raise RuntimeError(
        "Set DATABASE_URL (or WORKER_DATABASE_URL for local dev) in .env"
    )

# Paths — absolute defaults relative to repo root
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(REPO_ROOT / "data" / "derived" / "models")))
VIDEO_STORAGE_DIR = Path(os.getenv("VIDEO_STORAGE_DIR", str(REPO_ROOT / "data" / "uploads")))
TRACKS_DIR = Path(os.getenv("TRACKS_DIR", str(REPO_ROOT / "data" / "derived" / "tracks")))

# Worker settings
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
