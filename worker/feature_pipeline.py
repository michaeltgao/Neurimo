"""
Feature extraction pipeline for processing visit videos.

Takes 3 videos (joint_attention, imitation, free_play) and extracts features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import TRACKS_DIR, MODELS_DIR


class FeaturePipeline:
    """
    Orchestrates video → features extraction for a visit.

    Pipeline:
    1. Run MediaPipe tracking on each video → tracks
    2. Extract task-specific events (pointing, imitation, free play)
    3. Extract common + task-specific features
    4. Merge into single feature dict
    """

    TASK_TYPES = ["joint_attention", "imitation", "free_play"]

    # Map task_type to feature prefix
    TASK_PREFIX_MAP = {
        "joint_attention": "ja_",
        "imitation": "imit_",
        "free_play": "fp_",
    }

    def __init__(self, tracks_dir: Path = TRACKS_DIR):
        self.tracks_dir = tracks_dir
        self._perception_loaded = False

    def _ensure_perception_loaded(self):
        """Lazy load perception modules (heavy imports)."""
        if self._perception_loaded:
            return

        # Import perception modules
        from ml.src.perception.mediapipe_runner import MediaPipeRunner
        from ml.src.perception.imitation import extract_imitation_events
        from ml.src.perception.free_play_events import extract_free_play_events
        from ml.src.perception.pointing import extract_pointing_events
        from ml.src.perception.audio_events import extract_audio_events

        self._mediapipe = MediaPipeRunner()
        self._extract_imitation = extract_imitation_events
        self._extract_free_play = extract_free_play_events
        self._extract_pointing = extract_pointing_events
        self._extract_audio = extract_audio_events

        # Import feature extraction
        from ml.src.features.common import extract_common_features_for_child
        from ml.src.features.joint_attention import extract_joint_attention_features
        from ml.src.features.imitation import extract_imitation_features
        from ml.src.features.free_play import extract_free_play_features

        self._extract_common = extract_common_features_for_child
        self._extract_ja_features = extract_joint_attention_features
        self._extract_imit_features = extract_imitation_features
        self._extract_fp_features = extract_free_play_features

        self._perception_loaded = True

    def process_videos(
        self,
        videos: List[Dict[str, Any]],
        child_id: str,
    ) -> Dict[str, Any]:
        """
        Process all videos for a visit and return merged features.

        Args:
            videos: List of video dicts with 'task_type' and 'storage_path'
            child_id: Child identifier

        Returns:
            Dict of all features ready for prediction
        """
        self._ensure_perception_loaded()

        # Organize videos by task type
        video_by_task = {v["task_type"]: v for v in videos}

        # Verify all 3 task types present
        missing = set(self.TASK_TYPES) - set(video_by_task.keys())
        if missing:
            raise ValueError(f"Missing videos for tasks: {missing}")

        features = {"child_id": child_id}

        # Process each video
        for task_type in self.TASK_TYPES:
            video = video_by_task[task_type]
            video_path = Path(video["storage_path"])

            # 1. Run MediaPipe tracking
            tracks = self._run_tracking(video_path, child_id, task_type)

            # 2. Extract task-specific events
            events = self._extract_events(video_path, tracks, task_type)

            # 3. Extract features
            task_features = self._extract_task_features(
                tracks, events, task_type, child_id
            )

            # 4. Merge with prefix
            features.update(task_features)

        return features

    def _run_tracking(
        self, video_path: Path, child_id: str, task_type: str
    ) -> pd.DataFrame:
        """Run MediaPipe tracking on video."""
        # Check for cached tracks
        tracks_path = self.tracks_dir / f"{child_id}_{task_type}_tracks.csv"

        if tracks_path.exists():
            return pd.read_csv(tracks_path)

        # Run tracking
        tracks = self._mediapipe.process_video(str(video_path))

        # Cache tracks
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        tracks.to_csv(tracks_path, index=False)

        return tracks

    def _extract_events(
        self, video_path: Path, tracks: pd.DataFrame, task_type: str
    ) -> Dict[str, Any]:
        """Extract task-specific events from tracks."""
        events = {}

        if task_type == "joint_attention":
            events["pointing"] = self._extract_pointing(tracks)
            events["audio"] = self._extract_audio(str(video_path))

        elif task_type == "imitation":
            events["imitation"] = self._extract_imitation(tracks)

        elif task_type == "free_play":
            events["free_play"] = self._extract_free_play(tracks)

        return events

    def _extract_task_features(
        self,
        tracks: pd.DataFrame,
        events: Dict[str, Any],
        task_type: str,
        child_id: str,
    ) -> Dict[str, Any]:
        """Extract features for a specific task."""
        # Common features (pose, face, motion)
        common = self._extract_common(tracks, child_id, task_type)

        # Task-specific features
        if task_type == "joint_attention":
            specific = self._extract_ja_features(
                tracks,
                events.get("pointing", pd.DataFrame()),
                events.get("audio", pd.DataFrame()),
            )
        elif task_type == "imitation":
            specific = self._extract_imit_features(
                tracks, events.get("imitation", pd.DataFrame())
            )
        elif task_type == "free_play":
            specific = self._extract_fp_features(
                tracks, events.get("free_play", pd.DataFrame())
            )
        else:
            specific = {}

        # Merge common + specific
        features = {**common, **specific}

        return features


# Singleton instance
_pipeline: FeaturePipeline | None = None


def get_feature_pipeline() -> FeaturePipeline:
    """Get singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FeaturePipeline()
    return _pipeline
