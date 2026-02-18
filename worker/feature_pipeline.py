"""
Feature extraction pipeline for processing visit videos.

Takes 3 videos (joint_attention, imitation, free_play), runs perception,
extracts features, and returns a merged feature dict for ML prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TRACKS_DIR

logger = logging.getLogger(__name__)

# ── Default perception parameters (from CLI defaults) ──────────────────

SAMPLE_EVERY_N = 2
MAX_FRAMES = 900

POINTING_DEFAULTS = dict(
    adult_region="top_or_side",
    min_vec_len=0.10,
    head_min_vis=0.3,
    adult_y_max=0.72,
    adult_side_x=0.18,
    adaptive_threshold=True,
    min_pointiness_ratio=1.8,
    min_hand_conf=0.55,
)

POINT_SEGMENT_DEFAULTS = dict(
    min_frames=6,
    max_angle_jitter_deg=10.0,
    max_gap_sec=0.25,
    merge_gap_sec=0.40,
    min_duration_sec=0.2,
    min_stability=0.45,
    min_mean_conf=0.5,
)

IMITATION_DEFAULTS = dict(
    response_window_sec=7.5,
    adult_speed_peak_thr_std=2.1,
    child_speed_peak_thr_std=2.1,
    adult_min_peaks=2,
    child_min_peaks=1,
    clap_min_claps_adult_dist=2,
    clap_min_claps_child_dist=1,
    clap_periodic_min_dist=8.0,
    clap_periodic_min_speed=6.0,
    adult_demo_min_speed_abs=0.06,
    adult_demo_min_speed_var=0.0006,
    child_resp_min_speed_abs=0.055,
    child_resp_allow_single_burst=False,
    arms_min_seg_dur=0.18,
    arms_raise_then_disappear=False,
)


class FeaturePipeline:
    """
    Orchestrates video → features extraction for a visit.

    Pipeline:
    1. Run MediaPipe tracking on each video → NPZ tracks
    2. Extract task-specific events (pointing, audio, imitation, free play)
    3. Extract common + task-specific features
    4. Merge into single feature dict
    """

    TASK_TYPES = ["joint_attention", "imitation", "free_play"]

    def __init__(self, tracks_dir: Path = TRACKS_DIR):
        self.tracks_dir = tracks_dir
        self._initialized = False

        # Heavy models (lazy-loaded)
        self._pose_landmarker = None
        self._hand_landmarker = None
        self._whisper_model = None

    def _ensure_initialized(self):
        """Lazy-load MediaPipe landmarkers and Whisper model."""
        if self._initialized:
            return

        logger.info("Initializing perception models...")

        # ── MediaPipe landmarkers ──
        from ml.src.perception.track_child import get_model_paths
        import mediapipe as mp

        pose_model_path, hand_model_path = get_model_paths()

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self._pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_model_path)),
                running_mode=VisionRunningMode.IMAGE,
                num_poses=2,
            )
        )
        self._hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(hand_model_path)),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=4,
            )
        )

        # ── Whisper model (for audio events in JA) ──
        from faster_whisper import WhisperModel

        self._whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        self._initialized = True
        logger.info("Perception models loaded")

    # ── Public API ──────────────────────────────────────────────────────

    def process_videos(
        self,
        videos: List[Dict[str, Any]],
        child_id: str,
    ) -> Dict[str, Any]:
        """
        Process all videos for a visit and return merged features.

        Args:
            videos: List of dicts with 'task_type' and 'storage_path'
            child_id: Identifier for track file naming (typically str(visit.id))

        Returns:
            Dict of all features ready for ML prediction
        """
        self._ensure_initialized()

        video_by_task = {v["task_type"]: v for v in videos}
        missing = set(self.TASK_TYPES) - set(video_by_task.keys())
        if missing:
            raise ValueError(f"Missing videos for tasks: {missing}")

        # Per-visit tracks directory avoids collisions
        visit_tracks_dir = self.tracks_dir / f"visit_{child_id}"
        visit_tracks_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Track all 3 videos → NPZ files ──
        for task_type in self.TASK_TYPES:
            video_path = Path(video_by_task[task_type]["storage_path"])
            npz_path = visit_tracks_dir / f"{child_id}_{task_type}.npz"
            if not npz_path.exists():
                logger.info(f"  Tracking {task_type}...")
                self._run_tracking(video_path, npz_path)
            else:
                logger.info(f"  Using cached tracks for {task_type}")

        # ── Step 2: Task-specific perception ──
        ja_path = Path(video_by_task["joint_attention"]["storage_path"])
        point_events_df = self._extract_pointing_events(ja_path)
        audio_events_df = self._extract_audio_events(ja_path, child_id)

        imit_path = Path(video_by_task["imitation"]["storage_path"])
        imit_events_df, imit_summary_df = self._extract_imitation_events(imit_path, child_id)

        fp_npz = visit_tracks_dir / f"{child_id}_free_play.npz"
        fp_events_df, fp_summary_df = self._extract_free_play_events(fp_npz, child_id)

        # ── Persist events for guided review ──
        self._save_guided_review_events(
            visit_tracks_dir, child_id, point_events_df, audio_events_df, imit_events_df, fp_events_df
        )

        # ── Step 3: Feature extraction ──
        from ml.src.features.common import extract_common_features_for_child
        from ml.src.features.joint_attention import extract_joint_attention_features, extract_response_outcomes
        from ml.src.features.imitation import extract_imitation_features, extract_imitation_outcomes
        from ml.src.features.free_play import extract_free_play_features, extract_free_play_outcomes

        qc_df = pd.DataFrame()  # common features work without QC data

        common_features = extract_common_features_for_child(
            child_id=child_id,
            tracks_dir=visit_tracks_dir,
            qc_df=qc_df,
        )

        ja_duration = common_features.get("ja_duration_sec")

        ja_features = extract_joint_attention_features(
            child_id=child_id,
            point_events_df=point_events_df,
            audio_events_df=audio_events_df,
            tracks_dir=visit_tracks_dir,
            video_duration_sec=ja_duration,
        )

        # ── Extract and save per-event response outcomes for guided review ──
        # This uses the SAME detection logic as feature extraction
        ja_outcomes = extract_response_outcomes(
            child_id=child_id,
            point_events_df=point_events_df,
            audio_events_df=audio_events_df,
            tracks_dir=visit_tracks_dir,
        )

        imit_features = extract_imitation_features(
            child_id=child_id,
            imit_summary_df=imit_summary_df,
        )

        # Extract imitation outcomes using same detection logic as features
        imit_outcomes = extract_imitation_outcomes(
            child_id=child_id,
            imit_summary_df=imit_summary_df,
            imit_events_df=imit_events_df,
        )

        fp_features = extract_free_play_features(
            child_id=child_id,
            fp_summary_df=fp_summary_df,
            fp_events_df=fp_events_df,
            tracks_dir=visit_tracks_dir,
        )

        # Extract free play outcomes using same threshold logic as features
        fp_outcomes = extract_free_play_outcomes(
            child_id=child_id,
            fp_summary_df=fp_summary_df,
            fp_events_df=fp_events_df,
        )

        # Combine all outcomes and save for guided review
        all_outcomes = ja_outcomes + imit_outcomes + fp_outcomes
        self._save_response_outcomes(visit_tracks_dir, all_outcomes)

        # ── Step 4: Merge all feature dicts ──
        features: Dict[str, Any] = {}
        features.update(common_features)
        features.update(ja_features)
        features.update(imit_features)
        features.update(fp_features)

        return features

    # ── Perception helpers ──────────────────────────────────────────────

    def _run_tracking(self, video_path: Path, npz_path: Path) -> None:
        """Run MediaPipe tracking and save NPZ."""
        from ml.src.perception.track_child import (
            extract_tracks_arrays_for_video,
            save_tracks_npz,
        )

        tracks_arrays, _quality = extract_tracks_arrays_for_video(
            video_path=str(video_path),
            sample_every_n=SAMPLE_EVERY_N,
            max_frames=MAX_FRAMES,
            pose_landmarker=self._pose_landmarker,
            hand_landmarker=self._hand_landmarker,
        )
        save_tracks_npz(npz_path, tracks_arrays)

    def _extract_pointing_events(self, video_path: Path) -> pd.DataFrame:
        """Extract pointing events from joint attention video."""
        from ml.src.perception.pointing import (
            process_video_with_orient,
            detect_point_segments,
            segment_to_pointing_event,
        )

        _head_samples, point_samples = process_video_with_orient(
            video_path=video_path,
            sample_every_n=SAMPLE_EVERY_N,
            max_frames=MAX_FRAMES,
            pose_landmarker=self._pose_landmarker,
            hand_landmarker=self._hand_landmarker,
            **POINTING_DEFAULTS,
        )

        segments = detect_point_segments(point_samples, **POINT_SEGMENT_DEFAULTS)
        events = [segment_to_pointing_event(seg) for seg in segments]

        if not events:
            return pd.DataFrame(columns=[
                "event_type", "t_start", "t_end", "point_angle_deg",
                "stability", "confidence", "method", "n_samples",
            ])

        return pd.DataFrame([
            {
                "event_type": e.event_type,
                "t_start": e.t_start,
                "t_end": e.t_end,
                "point_angle_deg": e.point_angle_deg,
                "stability": e.stability,
                "confidence": e.confidence,
                "method": e.method,
                "n_samples": e.n_samples,
            }
            for e in events
        ])

    def _extract_audio_events(
        self, video_path: Path, child_id: str
    ) -> pd.DataFrame:
        """Extract audio events (parent prompts) from JA video."""
        from ml.src.perception.audio_events import process_video_audio

        audio_events, _summary, _quality = process_video_audio(
            video_path=video_path,
            child_id=child_id,
            task_type="joint_attention",
            model=self._whisper_model,
        )

        if not audio_events:
            return pd.DataFrame(columns=[
                "child_id", "task_type", "event_type", "t_start", "t_end",
                "confidence", "matched_phrase", "stt_confidence",
                "energy_confidence", "is_vad_confirmed",
            ])

        return pd.DataFrame([
            {
                "child_id": child_id,
                "task_type": "joint_attention",
                "event_type": e.event_type,
                "t_start": e.t_start,
                "t_end": e.t_end,
                "confidence": e.combined_confidence,
                "matched_phrase": e.matched_phrase,
                "stt_confidence": e.stt_confidence,
                "energy_confidence": e.energy_confidence,
                "is_vad_confirmed": e.is_vad_confirmed,
            }
            for e in audio_events
        ])

    def _extract_imitation_events(
        self, video_path: Path, child_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run imitation detection and return events + summary DataFrames."""
        from ml.src.perception.imitation import process_video

        events, summary = process_video(
            child_id=child_id,
            video_path=video_path,
            hand_landmarker=self._hand_landmarker,
            pose_landmarker=self._pose_landmarker,
            sample_every_n=SAMPLE_EVERY_N,
            max_frames=MAX_FRAMES,
            **IMITATION_DEFAULTS,
        )

        # Build events DataFrame
        # Event class attributes: kind (event type), primitive (action type), meta (extra info)
        events_df = pd.DataFrame([
            {
                "child_id": e.child_id,
                "task_type": e.task_type,
                "event_type": e.kind,        # Event uses 'kind' not 'event_type'
                "action_type": e.primitive,  # Event uses 'primitive' not 'action_type'
                "subject": e.subject,
                "t_sec": e.t_sec,
                "confidence": e.confidence,
                "meta": e.meta,
            }
            for e in events
        ]) if events else pd.DataFrame()

        summary_df = pd.DataFrame([summary]) if summary else pd.DataFrame()

        return events_df, summary_df

    def _extract_free_play_events(
        self, npz_path: Path, child_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract free play events from tracked NPZ."""
        from ml.src.perception.free_play_events import (
            load_tracks,
            process_free_play_tracks,
            EventConfig,
        )

        tracks = load_tracks(npz_path)
        if tracks is None:
            return pd.DataFrame(), pd.DataFrame()

        event_segs, summary = process_free_play_tracks(child_id, tracks, EventConfig())

        events_df = pd.DataFrame([
            {
                "child_id": e.child_id,
                "task_type": e.task_type,
                "event_type": e.event_type,
                "t_start": e.t_start,
                "t_end": e.t_end,
                "confidence": e.confidence,
                "meta": e.meta,
            }
            for e in event_segs
        ]) if event_segs else pd.DataFrame()

        summary_df = pd.DataFrame([summary]) if summary else pd.DataFrame()

        return events_df, summary_df

    def _save_guided_review_events(
        self,
        visit_tracks_dir: Path,
        child_id: str,
        point_events_df: pd.DataFrame,
        audio_events_df: pd.DataFrame,
        imit_events_df: pd.DataFrame,
        fp_events_df: pd.DataFrame,
    ) -> None:
        """
        Save all task events for guided review.

        These CSVs are read by the backend's guided_review service
        to show flagged moments during video review.
        """
        # Save pointing events (for joint attention - parent points)
        point_csv = visit_tracks_dir / "pointing_events.csv"
        if not point_events_df.empty:
            point_events_df.to_csv(point_csv, index=False)
            logger.info(f"  Saved {len(point_events_df)} pointing events to {point_csv}")
        else:
            pd.DataFrame(columns=[
                "event_type", "t_start", "t_end", "point_angle_deg",
                "stability", "confidence", "method", "n_samples"
            ]).to_csv(point_csv, index=False)

        # Save audio events (for joint attention prompts)
        audio_csv = visit_tracks_dir / "audio_events.csv"
        if not audio_events_df.empty:
            audio_events_df.to_csv(audio_csv, index=False)
            logger.info(f"  Saved {len(audio_events_df)} audio events to {audio_csv}")
        else:
            pd.DataFrame(columns=[
                "child_id", "task_type", "event_type", "t_start", "t_end",
                "confidence", "matched_phrase"
            ]).to_csv(audio_csv, index=False)

        # Save imitation events (parent demos and child responses)
        imit_csv = visit_tracks_dir / "imitation_events.csv"
        if not imit_events_df.empty:
            imit_events_df.to_csv(imit_csv, index=False)
            logger.info(f"  Saved {len(imit_events_df)} imitation events to {imit_csv}")
        else:
            pd.DataFrame(columns=[
                "child_id", "task_type", "event_type", "action_type", "subject",
                "t_sec", "confidence", "meta"
            ]).to_csv(imit_csv, index=False)

        # Save free play behavioral events
        fp_csv = visit_tracks_dir / "behavioral_events.csv"
        if not fp_events_df.empty:
            fp_events_df.to_csv(fp_csv, index=False)
            logger.info(f"  Saved {len(fp_events_df)} behavioral events to {fp_csv}")
        else:
            pd.DataFrame(columns=[
                "child_id", "task_type", "event_type", "t_start", "t_end",
                "confidence", "meta"
            ]).to_csv(fp_csv, index=False)

    def _save_response_outcomes(
        self,
        visit_tracks_dir: Path,
        response_outcomes: List[Dict[str, Any]],
    ) -> None:
        """
        Save per-event response outcomes for guided review.

        This file contains the ML model's detection results for each cue,
        ensuring guided review shows exactly what the ML model detected.
        """
        outcomes_csv = visit_tracks_dir / "response_outcomes.csv"

        if response_outcomes:
            outcomes_df = pd.DataFrame(response_outcomes)
            outcomes_df.to_csv(outcomes_csv, index=False)
            logger.info(f"  Saved {len(outcomes_df)} response outcomes to {outcomes_csv}")
        else:
            pd.DataFrame(columns=[
                "cue_type", "event_type", "t_start_sec", "t_end_sec",
                "matched_phrase", "point_angle_deg", "responded",
                "latency_ms", "status"
            ]).to_csv(outcomes_csv, index=False)


# ── Singleton ──────────────────────────────────────────────────────────

_pipeline: FeaturePipeline | None = None


def get_feature_pipeline() -> FeaturePipeline:
    """Get singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FeaturePipeline()
    return _pipeline
