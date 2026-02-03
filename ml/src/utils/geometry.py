"""
Geometry utilities for Neurimo ML pipeline.

Provides common functions for bounding box operations:
- IoU computation
- Distance calculations
- Validity checks
- Centroid extraction

These functions are used across perception and feature extraction modules.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two bboxes.

    Args:
        bbox1, bbox2: Arrays with [x0, y0, x1, y1, ...] (extra columns ignored)

    Returns:
        IoU value in [0, 1], 0 if either bbox is invalid
    """
    if not np.isfinite(bbox1[:4]).all() or not np.isfinite(bbox2[:4]).all():
        return 0.0

    x0_1, y0_1, x1_1, y1_1 = bbox1[:4]
    x0_2, y0_2, x1_2, y1_2 = bbox2[:4]

    xi0 = max(x0_1, x0_2)
    yi0 = max(y0_1, y0_2)
    xi1 = min(x1_1, x1_2)
    yi1 = min(y1_1, y1_2)

    inter_w = max(0.0, xi1 - xi0)
    inter_h = max(0.0, yi1 - yi0)
    inter_area = inter_w * inter_h

    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - inter_area

    if union_area < 1e-8:
        return 0.0
    return float(inter_area / union_area)


def bbox_center_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Euclidean distance between bbox centers.

    Args:
        bbox1, bbox2: Arrays with [x0, y0, x1, y1, ...]

    Returns:
        Distance between centers, inf if either bbox is invalid
    """
    if not np.isfinite(bbox1[:4]).all() or not np.isfinite(bbox2[:4]).all():
        return float("inf")
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def bbox_centroid(bbox: np.ndarray) -> np.ndarray:
    """
    Extract centroid (cx, cy) from bbox array.

    Args:
        bbox: (N, 5) or (N, 4) array with [x0, y0, x1, y1, ...]

    Returns:
        (N, 2) array of centroids [cx, cy]
    """
    cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
    cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
    return np.column_stack([cx, cy])


def bbox_valid(
    bbox: Optional[np.ndarray],
    conf_thr: float = 0.3,
    min_area: float = 1e-4,
) -> np.ndarray:
    """
    Returns boolean mask for frames with valid bbox.

    Valid = coordinates finite AND confidence >= threshold (if conf column exists)
    AND area > min_area

    Args:
        bbox: (N, 5) array [x0, y0, x1, y1, conf] or (N, 4) [x0, y0, x1, y1]
        conf_thr: Minimum confidence threshold
        min_area: Minimum bbox area (normalized)

    Returns:
        (N,) boolean mask
    """
    if bbox is None or bbox.shape[0] == 0:
        return np.array([], dtype=bool)

    coords_valid = np.isfinite(bbox[:, :4]).all(axis=1)

    if bbox.shape[1] >= 5:
        conf_valid = bbox[:, 4] >= conf_thr
    else:
        conf_valid = np.ones(len(bbox), dtype=bool)

    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    area_valid = area > min_area

    return coords_valid & conf_valid & area_valid


def bbox_area(bbox: np.ndarray) -> np.ndarray:
    """
    Compute normalized bbox area.

    Args:
        bbox: (N, 4+) array with [x0, y0, x1, y1, ...]

    Returns:
        (N,) array of areas
    """
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def point_in_bbox(x: float, y: float, bbox: np.ndarray) -> bool:
    """
    Check if point (x, y) is inside bbox [x0, y0, x1, y1, ...].

    Args:
        x, y: Point coordinates
        bbox: Bounding box array

    Returns:
        True if point is inside bbox
    """
    if not np.isfinite([x, y]).all() or not np.isfinite(bbox[:4]).all():
        return False
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
