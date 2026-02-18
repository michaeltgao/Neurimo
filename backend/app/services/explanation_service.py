"""Normalize explanation data from MLPrediction.explanations JSON column."""
from typing import Any, List


def format_explanations(raw: List[Any]) -> List[str]:
    """
    Convert raw explanations (stored as JSON) to a list of strings.

    Handles both formats:
    - Old: [{"feature": "...", "message": "..."}]
    - New: ["plain string", ...]
    """
    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and "message" in item:
            out.append(item["message"])
    return out
