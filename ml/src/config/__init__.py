"""Configuration module for Neurimo ML pipeline."""

from .thresholds import (
    ActivityThresholds,
    PeriodicMotionThresholds,
    HandEventThresholds,
    ProximityThresholds,
    FreePlayEventConfig,
)

# Audio event phrase lists (used by perception/audio_events.py)
ATTENTION_CALL_PHRASES = [
    "hey",
    "hey there",
    "hi",
    "hello",
    "listen",
    "come here",
]

LOOK_PHRASES = [
    "look",
    "look at",
    "look here",
    "over there",
    "see",
    "see that",
    "watch",
    "watch this",
]

__all__ = [
    "ActivityThresholds",
    "PeriodicMotionThresholds",
    "HandEventThresholds",
    "ProximityThresholds",
    "FreePlayEventConfig",
    "ATTENTION_CALL_PHRASES",
    "LOOK_PHRASES",
]
