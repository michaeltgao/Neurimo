"""
Threshold configuration for Neurimo ML pipeline.

Provides structured, documented configuration classes for:
- Activity level detection
- Periodic/repetitive motion detection
- Hand event detection
- Parent-child proximity detection

Supports YAML serialization for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


@dataclass
class ActivityThresholds:
    """Thresholds for activity level detection."""
    # Speed thresholds (normalized coords/sec)
    low: float = 0.02
    high: float = 0.05

    # Minimum duration for activity burst events
    min_burst_duration: float = 0.5

    # EMA smoothing parameters
    ema_alpha: float = 0.4
    ema_max_hold_frames: int = 2


@dataclass
class PeriodicMotionThresholds:
    """Thresholds for repetitive/periodic motion detection."""
    # Frequency band (Hz)
    freq_min_hz: float = 1.5
    freq_max_hz: float = 5.5

    # Normalized periodicity score threshold (scale-invariant)
    normalized_score: float = 0.15

    # Minimum signal amplitude to consider
    min_amplitude: float = 0.006

    # Minimum duration for periodic bout
    min_duration: float = 1.0

    # Analysis window size (seconds)
    window_sec: float = 2.0


@dataclass
class HandEventThresholds:
    """Thresholds for hand-related events."""
    # Speed threshold for active hands (normalized coords/sec)
    speed_threshold: float = 0.04

    # Workspace region (hands must be in lower portion of frame/bbox)
    workspace_y_min: float = 0.4

    # Minimum duration for hand active events
    active_min_dur: float = 0.3

    # Hand-to-face distance threshold (normalized)
    face_distance: float = 0.12
    face_min_dur: float = 0.2

    # Hands-together distance threshold (wrist-to-wrist)
    together_distance: float = 0.10
    together_min_dur: float = 0.15


@dataclass
class ProximityThresholds:
    """Thresholds for parent-child proximity detection."""
    # IoU threshold for "close" proximity
    iou_threshold: float = 0.05

    # Center distance threshold (normalized)
    center_distance: float = 0.25

    # Minimum duration for proximity events
    min_duration: float = 0.5

    # Bbox confidence threshold
    bbox_conf_threshold: float = 0.2

    # Minimum bbox area to be considered valid
    min_bbox_area: float = 1e-4


@dataclass
class EngagementThresholds:
    """Thresholds for engagement/disengagement detection."""
    # Minimum duration for state changes
    state_window_sec: float = 0.5
    transition_min_duration: float = 0.2

    # Freeze detection (stillness)
    freeze_speed_threshold: float = 0.01
    freeze_min_duration: float = 1.0


@dataclass
class ResponseDetectionThresholds:
    """Thresholds for social response detection."""
    # Response window after parent bid
    response_window_sec: float = 3.0

    # Head turn threshold for orientation response
    head_turn_threshold: float = 0.04

    # Approach/avoid speed thresholds
    approach_speed: float = 0.02
    avoid_speed: float = 0.02

    # Proximity change detection window
    proximity_change_window: float = 1.0


@dataclass
class FreePlayEventConfig:
    """Complete configuration for free play event detection."""
    # Sub-configurations
    activity: ActivityThresholds = field(default_factory=ActivityThresholds)
    periodic: PeriodicMotionThresholds = field(default_factory=PeriodicMotionThresholds)
    hand: HandEventThresholds = field(default_factory=HandEventThresholds)
    proximity: ProximityThresholds = field(default_factory=ProximityThresholds)
    engagement: EngagementThresholds = field(default_factory=EngagementThresholds)
    response: ResponseDetectionThresholds = field(default_factory=ResponseDetectionThresholds)

    # Off-screen detection
    off_screen_min_dur: float = 0.5

    # Parent presence minimum duration
    parent_min_dur: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FreePlayEventConfig":
        """Create config from dictionary."""
        activity = ActivityThresholds(**d.get("activity", {}))
        periodic = PeriodicMotionThresholds(**d.get("periodic", {}))
        hand = HandEventThresholds(**d.get("hand", {}))
        proximity = ProximityThresholds(**d.get("proximity", {}))
        engagement = EngagementThresholds(**d.get("engagement", {}))
        response = ResponseDetectionThresholds(**d.get("response", {}))

        return cls(
            activity=activity,
            periodic=periodic,
            hand=hand,
            proximity=proximity,
            engagement=engagement,
            response=response,
            off_screen_min_dur=d.get("off_screen_min_dur", 0.5),
            parent_min_dur=d.get("parent_min_dur", 0.5),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "FreePlayEventConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d or {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_legacy_event_config(cls, legacy_cfg: Any) -> "FreePlayEventConfig":
        """
        Convert from legacy EventConfig dataclass.

        This allows gradual migration from the old flat config to the new structured format.
        """
        return cls(
            activity=ActivityThresholds(
                low=getattr(legacy_cfg, "activity_low_threshold", 0.02),
                high=getattr(legacy_cfg, "activity_high_threshold", 0.05),
                min_burst_duration=getattr(legacy_cfg, "activity_min_dur", 0.5),
                ema_alpha=0.4,
                ema_max_hold_frames=getattr(legacy_cfg, "activity_ema_window", 3),
            ),
            periodic=PeriodicMotionThresholds(
                freq_min_hz=getattr(legacy_cfg, "rep_freq_min", 1.5),
                freq_max_hz=getattr(legacy_cfg, "rep_freq_max", 5.5),
                normalized_score=getattr(legacy_cfg, "rep_normalized_threshold", 0.15),
                min_amplitude=getattr(legacy_cfg, "rep_min_amplitude", 0.006),
                min_duration=getattr(legacy_cfg, "rep_min_dur", 1.0),
                window_sec=getattr(legacy_cfg, "rep_window_sec", 2.0),
            ),
            hand=HandEventThresholds(
                speed_threshold=getattr(legacy_cfg, "hand_speed_threshold", 0.04),
                workspace_y_min=getattr(legacy_cfg, "hand_workspace_y_min", 0.4),
                active_min_dur=getattr(legacy_cfg, "hand_active_min_dur", 0.3),
                face_distance=getattr(legacy_cfg, "hand_face_distance_threshold", 0.12),
                face_min_dur=getattr(legacy_cfg, "hand_face_min_dur", 0.2),
                together_distance=getattr(legacy_cfg, "hands_together_threshold", 0.10),
                together_min_dur=getattr(legacy_cfg, "hands_together_min_dur", 0.15),
            ),
            proximity=ProximityThresholds(
                iou_threshold=getattr(legacy_cfg, "proximity_iou_threshold", 0.05),
                center_distance=getattr(legacy_cfg, "proximity_center_dist_threshold", 0.25),
                min_duration=getattr(legacy_cfg, "proximity_min_dur", 0.5),
            ),
            off_screen_min_dur=getattr(legacy_cfg, "off_screen_min_dur", 0.5),
            parent_min_dur=getattr(legacy_cfg, "parent_min_dur", 0.5),
        )
