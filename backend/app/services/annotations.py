from __future__ import annotations
from typing import Any, Dict

def make_placeholder_annotations(video_id: int, task_type: str) -> Dict[str, Any]:
    return {
        "version": "v1",
        "video_id": video_id,
        "task_type": task_type,
        "fps": None,
        "events": [],
        "signals": {},
        "landmarks": [],
        "notes": "",
    }
