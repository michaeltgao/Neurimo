"""Shared utility modules for Neurimo ML pipeline."""

from .signal_processing import (
    interp_nans,
    nanmed_smooth,
    ema_smooth,
    spectral_peak,
    windowed_periodicity,
    segments_from_bool,
    compute_speed,
    autocorr_peak,
)

from .geometry import (
    bbox_iou,
    bbox_center_distance,
    bbox_centroid,
    bbox_valid,
    bbox_area,
    point_in_bbox,
)

__all__ = [
    # Signal processing
    "interp_nans",
    "nanmed_smooth",
    "ema_smooth",
    "spectral_peak",
    "windowed_periodicity",
    "segments_from_bool",
    "compute_speed",
    "autocorr_peak",
    # Geometry
    "bbox_iou",
    "bbox_center_distance",
    "bbox_centroid",
    "bbox_valid",
    "bbox_area",
    "point_in_bbox",
]
