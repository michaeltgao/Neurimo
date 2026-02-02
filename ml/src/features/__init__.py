"""Feature extraction modules for Neurimo ML pipeline."""

from .common import (
    CommonConfig,
    compute_common_features_from_tracks,
    extract_common_features_for_child,
    load_tracks_npz,
)
from .free_play import extract_free_play_features
from .imitation import extract_imitation_features
from .joint_attention import extract_joint_attention_features
from .merge import merge_all_features, save_features

__all__ = [
    # Common features
    "CommonConfig",
    "load_tracks_npz",
    "compute_common_features_from_tracks",
    "extract_common_features_for_child",
    # Task-specific features
    "extract_joint_attention_features",
    "extract_imitation_features",
    "extract_free_play_features",
    # Merge utilities
    "merge_all_features",
    "save_features",
]
