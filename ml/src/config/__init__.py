"""Configuration module for Neurimo ML pipeline."""

from .thresholds import (
    ActivityThresholds,
    PeriodicMotionThresholds,
    HandEventThresholds,
    ProximityThresholds,
    FreePlayEventConfig,
)

__all__ = [
    "ActivityThresholds",
    "PeriodicMotionThresholds",
    "HandEventThresholds",
    "ProximityThresholds",
    "FreePlayEventConfig",
]
