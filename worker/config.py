"""Worker configuration."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Paths
MODELS_DIR = Path(os.getenv("MODELS_DIR", "data/derived/models"))
VIDEO_STORAGE_DIR = Path(os.getenv("VIDEO_STORAGE_DIR", "data/videos"))
TRACKS_DIR = Path(os.getenv("TRACKS_DIR", "data/derived/tracks"))

# Worker settings
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
