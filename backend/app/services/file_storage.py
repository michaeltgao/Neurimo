import os
from pathlib import Path
from fastapi import UploadFile
import json
from typing import Any, Dict

VIDEO_STORAGE_PATH = os.getenv("VIDEO_STORAGE_PATH", "./data/uploads")

ALLOWED_TASKS = {"imitation", "joint_attention", "free_play"}

def ensure_storage_dir() -> Path:
    p = Path(VIDEO_STORAGE_PATH)
    p.mkdir(parents=True, exist_ok=True)
    return p

def safe_ext(filename: str) -> str:
    # default to .webm if unknown
    ext = Path(filename).suffix.lower()
    if ext not in {".webm", ".mp4", ".mov"}:
        ext = ".webm"
    return ext

async def save_visit_video(visit_id: str, task_type: str, file: UploadFile) -> str:
    if task_type not in ALLOWED_TASKS:
        raise ValueError(f"Invalid task_type: {task_type}")

    storage_dir = ensure_storage_dir()
    ext = safe_ext(file.filename or "")
    out_path = storage_dir / f"{visit_id}_{task_type}{ext}"

    # stream to disk
    with out_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return str(out_path)

DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

def save_json(rel_path: str, obj: Dict[str, Any]) -> str:
    """
    rel_path example: 'annotations/42.json'
    Returns the full relative path under data/, e.g. 'data/annotations/42.json'
    """
    ensure_dirs()
    full_path = DATA_DIR / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(obj, indent=2))
    return str(full_path)

def load_json(full_data_path: str) -> Dict[str, Any]:
    p = Path(full_data_path)
    if not p.exists():
        raise FileNotFoundError(full_data_path)
    return json.loads(p.read_text())

def video_filename(visit_id: int, task_type: str, ext: str = "mp4") -> str:
    return f"{visit_id}_{task_type}.{ext}"

def annotations_filename(video_id: int) -> str:
    return f"{video_id}.json"