from __future__ import annotations
from typing import Any, Dict

def make_placeholder_annotations(video_id: int, task_type: str) -> Dict[str, Any]:
    # v1.5+ you will add: landmarks (sparse), events, signals
    return {
        "version": "v1",
        "video_id": video_id,
        "task_type": task_type,
        "fps": None,
        "events": [],
        "signals": {},
        "landmarks": [],  # keep empty for now
        "notes": "Placeholder. Will be populated with landmarks/events in v1.5.",
    }
