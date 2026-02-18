"""
Guided Review Data Service

Transforms ML pipeline outputs (audio events, pose tracks, behavioral events)
into the GuidedReviewData format expected by the frontend.

Data Sources (worker-processed only):
- tracks/visit_{visit_id}/audio_events.csv
- tracks/visit_{visit_id}/pointing_events.csv
- tracks/visit_{visit_id}/imitation_events.csv
- tracks/visit_{visit_id}/behavioral_events.csv
- tracks/visit_{visit_id}/{visit_id}_{task_type}.npz (pose tracks)

Note: Batch-processed data fallback has been removed to ensure data
integrity - only use data from videos processed by the worker pipeline.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Configuration - use environment variables with sensible defaults
# Docker: /app/data, Local: ../data relative to backend/
# In Docker, __file__ is /app/app/services/guided_review.py
# Locally, __file__ is .../neurimo/backend/app/services/guided_review.py
_THIS_FILE = Path(__file__).resolve()
_SERVICES_DIR = _THIS_FILE.parent  # app/services/
_APP_DIR = _SERVICES_DIR.parent    # app/
_ROOT_DIR = _APP_DIR.parent        # /app in Docker, backend/ locally

# Check if we're in Docker (/app/data exists) or local (need to go up one more level)
if (_ROOT_DIR / "data").exists():
    _DEFAULT_DATA_DIR = _ROOT_DIR / "data"
else:
    # Local: backend/ -> neurimo/ -> data/
    _DEFAULT_DATA_DIR = _ROOT_DIR.parent / "data"

_DATA_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA_DIR)))
DERIVED_DATA_DIR = _DATA_DIR / "derived"
AUDIO_EVENTS_FILE = DERIVED_DATA_DIR / "audio_events.csv"
FREE_PLAY_EVENTS_FILE = DERIVED_DATA_DIR / "free_play_events.csv"
TRACKS_DIR = Path(os.getenv("TRACKS_DIR", str(DERIVED_DATA_DIR / "tracks")))

# Response window configuration (milliseconds)
RESPONSE_WINDOW_MS = {
    "joint_attention": 3000,  # 3 seconds to respond to "look" prompt
    "imitation": 5000,        # 5 seconds to imitate action
    "free_play": 2000,        # 2 seconds for spontaneous behaviors
}

# Latency thresholds (milliseconds)
DELAYED_THRESHOLD_MS = 1500  # Response after this is considered "delayed"


@dataclass
class AudioEvent:
    """Audio event from ML pipeline."""
    child_id: str
    task_type: str
    event_type: str  # CALL_ATTENTION, LOOK
    t_start: float   # seconds
    t_end: float
    confidence: float
    matched_phrase: str


@dataclass
class BehavioralEvent:
    """Behavioral event from ML pipeline."""
    child_id: str
    task_type: str
    event_type: str  # OFF_SCREEN, ACTIVITY_HIGH, PERIODIC_MOTION, etc.
    t_start: float
    t_end: float
    confidence: float
    meta: str


@dataclass
class ImitationEvent:
    """Imitation event from ML pipeline."""
    child_id: str
    task_type: str
    event_type: str  # PARENT_ACTION_START, CHILD_RESPONSE_START, etc.
    action_type: str  # CLAP, ARMS_UP
    subject: str  # parent, child
    t_sec: float
    confidence: float
    trial_id: Optional[int] = None
    latency_sec: Optional[float] = None
    meta: str = ""  # Extra info from worker pipeline


@dataclass
class PointingEvent:
    """Pointing event from ML pipeline (parent pointing gesture)."""
    event_type: str  # POINT
    t_start: float   # seconds
    t_end: float
    point_angle_deg: float
    stability: float
    confidence: float
    method: str
    n_samples: int


@dataclass
class HeadTurnResponse:
    """Detected head turn response to a prompt."""
    detected: bool
    latency_ms: Optional[float] = None
    t_observed_ms: Optional[float] = None
    confidence: float = 0.0
    description: str = ""


@dataclass
class TrackingQuality:
    """Tracking quality metrics for a time window."""
    quality: str  # "good", "medium", "low"
    quality_pct: int  # 0-100
    face_visible: bool


@dataclass
class ResponseOutcome:
    """
    Per-event response outcome from ML feature extraction.

    This represents what the ML model detected for each cue,
    ensuring consistency between dashboard explanations and video replay.
    """
    cue_type: str  # "audio" or "point"
    event_type: str  # "CALL_ATTENTION", "LOOK", "POINT"
    t_start_sec: float
    t_end_sec: float
    matched_phrase: Optional[str]
    point_angle_deg: Optional[float]
    responded: bool
    latency_ms: Optional[float]
    status: str  # "observed", "delayed", "not_observed", "uncertain"


@dataclass
class ImitationOutcome:
    """
    Per-demo imitation outcome from ML feature extraction.

    Uses the same detection logic as the ML features to ensure
    consistency between dashboard and video replay.
    """
    action_type: str  # "CLAP" or "ARMS_UP"
    event_type: str  # "PARENT_CLAP" or "PARENT_ARMS_UP"
    t_sec: float
    responded: bool
    latency_ms: Optional[float]
    status: str  # "observed", "delayed", "not_observed"


@dataclass
class FreePlayOutcome:
    """
    Per-event free play outcome from ML feature extraction.

    Uses the same threshold logic as the ML features to determine
    which behavioral events should be flagged.
    """
    event_type: str  # "PERIODIC_MOTION", "HAND_TO_FACE"
    t_start_sec: float
    t_end_sec: float
    flagged: bool
    status: str  # "flagged" or "normal"


def load_audio_events_from_csv(csv_path: Path, filter_id: Optional[str] = None) -> List[AudioEvent]:
    """Load audio events from a CSV file, optionally filtering by child_id."""
    events: List[AudioEvent] = []

    if not csv_path.exists():
        return events

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # If filter_id is provided, filter by it; otherwise load all
            if filter_id and str(row.get("child_id")) != str(filter_id):
                continue
            events.append(AudioEvent(
                child_id=row.get("child_id", ""),
                task_type=row.get("task_type", ""),
                event_type=row.get("event_type", ""),
                t_start=float(row.get("t_start", 0)),
                t_end=float(row.get("t_end", 0)),
                confidence=float(row.get("confidence", 0.5)),
                matched_phrase=row.get("matched_phrase", ""),
            ))

    return sorted(events, key=lambda e: e.t_start)


def load_audio_events(visit_id: int) -> List[AudioEvent]:
    """
    Load audio events from worker-processed data only.

    Location: tracks/visit_{visit_id}/audio_events.csv
    """
    worker_csv = TRACKS_DIR / f"visit_{visit_id}" / "audio_events.csv"
    if worker_csv.exists():
        return load_audio_events_from_csv(worker_csv)
    return []


def _safe_float(value: str, default: float = 0.0) -> float:
    """Safely convert a value to float, handling empty strings and invalid values."""
    if not value or value.strip() == "" or value.lower() in ("none", "nan"):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_float_or_none(value: str) -> Optional[float]:
    """Safely convert a value to float, returning None for empty/invalid values."""
    if not value or value.strip() == "" or value.lower() in ("none", "nan"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_response_outcomes(visit_id: int) -> List[ResponseOutcome]:
    """
    Load per-event response outcomes from worker-processed data.

    These outcomes use the SAME detection logic as the ML feature extraction,
    ensuring consistency between dashboard explanations and video replay flags.

    Location: tracks/visit_{visit_id}/response_outcomes.csv

    Note: Only loads joint_attention outcomes (skips imitation and free_play rows).
    """
    outcomes: List[ResponseOutcome] = []
    csv_path = TRACKS_DIR / f"visit_{visit_id}" / "response_outcomes.csv"

    if not csv_path.exists():
        logger.debug(f"No response_outcomes.csv found at {csv_path}")
        return outcomes

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows for other task types (imitation, free_play have separate loaders)
                task_type = row.get("task_type", "")
                if task_type in ("imitation", "free_play"):
                    continue

                # Also skip rows that don't have t_start_sec (required for JA outcomes)
                t_start_str = row.get("t_start_sec", "")
                if not t_start_str or t_start_str.strip() == "":
                    continue

                # Parse latency_ms - handle empty strings and None
                latency_ms = _safe_float_or_none(row.get("latency_ms", ""))

                # Parse point_angle_deg - handle empty strings and None
                point_angle = _safe_float_or_none(row.get("point_angle_deg", ""))

                # Parse responded - handle string "True"/"False"
                responded_str = row.get("responded", "False")
                responded = responded_str.lower() == "true"

                outcomes.append(ResponseOutcome(
                    cue_type=row.get("cue_type", ""),
                    event_type=row.get("event_type", ""),
                    t_start_sec=_safe_float(row.get("t_start_sec", ""), 0.0),
                    t_end_sec=_safe_float(row.get("t_end_sec", ""), 0.0),
                    matched_phrase=row.get("matched_phrase") if row.get("matched_phrase") else None,
                    point_angle_deg=point_angle,
                    responded=responded,
                    latency_ms=latency_ms,
                    status=row.get("status", "uncertain"),
                ))

        logger.info(f"Loaded {len(outcomes)} response outcomes from {csv_path}")
    except Exception as e:
        logger.error(f"Error loading response outcomes from {csv_path}: {e}")

    return sorted(outcomes, key=lambda o: o.t_start_sec)


def load_imitation_outcomes(visit_id: int) -> List[ImitationOutcome]:
    """
    Load per-demo imitation outcomes from worker-processed data.

    These outcomes use the SAME detection logic as the ML feature extraction,
    ensuring consistency between dashboard explanations and video replay flags.

    Location: tracks/visit_{visit_id}/response_outcomes.csv
    """
    outcomes: List[ImitationOutcome] = []
    csv_path = TRACKS_DIR / f"visit_{visit_id}" / "response_outcomes.csv"

    if not csv_path.exists():
        logger.debug(f"No response_outcomes.csv found at {csv_path}")
        return outcomes

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only process imitation outcomes (have task_type=imitation and action_type)
                if row.get("task_type") != "imitation":
                    continue

                # Parse latency_ms using safe parser
                latency_ms = _safe_float_or_none(row.get("latency_ms", ""))

                # Parse responded
                responded_str = row.get("responded", "False")
                responded = responded_str.lower() == "true"

                outcomes.append(ImitationOutcome(
                    action_type=row.get("action_type", ""),
                    event_type=row.get("event_type", ""),
                    t_sec=_safe_float(row.get("t_sec", ""), 0.0),
                    responded=responded,
                    latency_ms=latency_ms,
                    status=row.get("status", "uncertain"),
                ))

        logger.info(f"Loaded {len(outcomes)} imitation outcomes from {csv_path}")
    except Exception as e:
        logger.error(f"Error loading imitation outcomes from {csv_path}: {e}")

    return sorted(outcomes, key=lambda o: o.t_sec)


def load_free_play_outcomes(visit_id: int) -> List[FreePlayOutcome]:
    """
    Load per-event free play outcomes from worker-processed data.

    These outcomes use the SAME threshold logic as the ML feature extraction,
    ensuring consistency between dashboard explanations and video replay flags.

    Location: tracks/visit_{visit_id}/response_outcomes.csv
    """
    outcomes: List[FreePlayOutcome] = []
    csv_path = TRACKS_DIR / f"visit_{visit_id}" / "response_outcomes.csv"

    if not csv_path.exists():
        logger.debug(f"No response_outcomes.csv found at {csv_path}")
        return outcomes

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only process free_play outcomes
                if row.get("task_type") != "free_play":
                    continue

                # Parse flagged
                flagged_str = row.get("flagged", "False")
                flagged = flagged_str.lower() == "true"

                outcomes.append(FreePlayOutcome(
                    event_type=row.get("event_type", ""),
                    t_start_sec=_safe_float(row.get("t_start_sec", ""), 0.0),
                    t_end_sec=_safe_float(row.get("t_end_sec", ""), 0.0),
                    flagged=flagged,
                    status=row.get("status", "normal"),
                ))

        logger.info(f"Loaded {len(outcomes)} free play outcomes from {csv_path}")
    except Exception as e:
        logger.error(f"Error loading free play outcomes from {csv_path}: {e}")

    return sorted(outcomes, key=lambda o: o.t_start_sec)


def load_imitation_events_from_csv(csv_path: Path, filter_id: Optional[str] = None, is_batch: bool = False) -> List[ImitationEvent]:
    """Load imitation events from a CSV file, optionally filtering by child_id.

    Args:
        csv_path: Path to CSV file
        filter_id: Optional child_id to filter by
        is_batch: If True, use batch column names (primitive/kind vs action_type/event_type)
    """
    events: List[ImitationEvent] = []

    if not csv_path.exists():
        return events

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filter_id and str(row.get("child_id")) != str(filter_id):
                continue

            # Handle different column names between worker and batch formats
            if is_batch:
                # Batch format: primitive, kind (e.g., PARENT_ACTION_START)
                action_type = row.get("primitive", "")
                event_type = row.get("kind", "")
                # Map subject: adult -> parent for consistency
                subject = row.get("subject", "")
                if subject == "adult":
                    subject = "parent"
            else:
                # Worker format: action_type, event_type
                action_type = row.get("action_type", "")
                event_type = row.get("event_type", "")
                subject = row.get("subject", "")

            # Handle empty confidence values (empty string in CSV)
            conf_str = row.get("confidence", "")
            confidence = float(conf_str) if conf_str else 0.5

            events.append(ImitationEvent(
                child_id=row.get("child_id", ""),
                task_type=row.get("task_type", ""),
                event_type=event_type,
                action_type=action_type,
                subject=subject,
                t_sec=float(row.get("t_sec", 0) or 0),
                confidence=confidence,
                trial_id=int(row.get("trial_id", 0)) if row.get("trial_id") else None,
                latency_sec=float(row.get("latency_sec", 0)) if row.get("latency_sec") else None,
                meta=row.get("meta", ""),
            ))

    return sorted(events, key=lambda e: e.t_sec)


def load_imitation_events(visit_id: int) -> List[ImitationEvent]:
    """
    Load imitation events from worker-processed data only.

    Location: tracks/visit_{visit_id}/imitation_events.csv
    """
    worker_csv = TRACKS_DIR / f"visit_{visit_id}" / "imitation_events.csv"
    if worker_csv.exists():
        return load_imitation_events_from_csv(worker_csv, is_batch=False)
    return []


def load_pointing_events_from_csv(csv_path: Path, filter_id: Optional[str] = None, is_batch: bool = False) -> List[PointingEvent]:
    """Load pointing events from a CSV file.

    Args:
        csv_path: Path to CSV file
        filter_id: Optional child_id to filter by (for batch data)
        is_batch: If True, use batch column names (angle_deg vs point_angle_deg)
    """
    events: List[PointingEvent] = []

    if not csv_path.exists():
        return events

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by child_id if provided
            if filter_id and str(row.get("child_id")) != str(filter_id):
                continue

            # Handle different column names between worker and batch formats
            if is_batch:
                # Batch format uses angle_deg
                point_angle_deg = float(row.get("angle_deg", 0))
            else:
                # Worker format uses point_angle_deg
                point_angle_deg = float(row.get("point_angle_deg", 0))

            # Get stability - may be empty in batch format
            stability_str = row.get("stability", "")
            stability = float(stability_str) if stability_str else 0.0

            # Get n_samples - may be empty in batch format
            n_samples_str = row.get("n_samples", "")
            n_samples = int(n_samples_str) if n_samples_str else 0

            events.append(PointingEvent(
                event_type=row.get("event_type", "POINT"),
                t_start=float(row.get("t_start", 0)),
                t_end=float(row.get("t_end", 0)),
                point_angle_deg=point_angle_deg,
                stability=stability,
                confidence=float(row.get("confidence", 0.5)),
                method=row.get("method", ""),
                n_samples=n_samples,
            ))

    return sorted(events, key=lambda e: e.t_start)


def load_pointing_events(visit_id: int) -> List[PointingEvent]:
    """
    Load pointing events from worker-processed data only.

    Location: tracks/visit_{visit_id}/pointing_events.csv
    """
    worker_csv = TRACKS_DIR / f"visit_{visit_id}" / "pointing_events.csv"
    if worker_csv.exists():
        return load_pointing_events_from_csv(worker_csv, is_batch=False)
    return []


def load_behavioral_events_from_csv(csv_path: Path, filter_id: Optional[str] = None) -> List[BehavioralEvent]:
    """Load behavioral events from a CSV file, optionally filtering by child_id."""
    events: List[BehavioralEvent] = []

    if not csv_path.exists():
        return events

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filter_id and str(row.get("child_id")) != str(filter_id):
                continue
            events.append(BehavioralEvent(
                child_id=row.get("child_id", ""),
                task_type=row.get("task_type", ""),
                event_type=row.get("event_type", ""),
                t_start=float(row.get("t_start", 0)),
                t_end=float(row.get("t_end", 0)),
                confidence=float(row.get("confidence", 0.5)),
                meta=row.get("meta", ""),
            ))

    return sorted(events, key=lambda e: e.t_start)


def load_behavioral_events(visit_id: int) -> List[BehavioralEvent]:
    """
    Load behavioral events from worker-processed data only.

    Location: tracks/visit_{visit_id}/behavioral_events.csv
    """
    worker_csv = TRACKS_DIR / f"visit_{visit_id}" / "behavioral_events.csv"
    if worker_csv.exists():
        return load_behavioral_events_from_csv(worker_csv)
    return []


def load_tracks_from_npz(npz_path: Path) -> Optional[Dict[str, Any]]:
    """Load pose tracks from an NPZ file."""
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path)
        return {
            "t_sec": data.get("t_sec", np.array([])),
            "pose": data.get("pose", np.array([])),
            "child_bbox": data.get("child_bbox", np.array([])),
            "parent_bbox": data.get("parent_bbox", np.array([])),
            "lh": data.get("lh", np.array([])),  # Left hand landmarks (21 x 4)
            "rh": data.get("rh", np.array([])),  # Right hand landmarks (21 x 4)
            "fps": float(data.get("fps", np.array([30.0]))[0]) if "fps" in data else 30.0,
        }
    except Exception:
        return None


def load_tracks(visit_id: int, task_type: str) -> Optional[Dict[str, Any]]:
    """
    Load pose tracks from worker-processed data only.

    Track files are created by the worker pipeline when videos are processed.
    Location: tracks/visit_{visit_id}/{visit_id}_{task_type}.npz

    Note: We intentionally do NOT fall back to batch-processed files
    as those use a different ID scheme from an external dataset and
    could return incorrect data for uploaded videos.
    """
    worker_npz = TRACKS_DIR / f"visit_{visit_id}" / f"{visit_id}_{task_type}.npz"
    tracks = load_tracks_from_npz(worker_npz)

    if tracks is None:
        logger.warning(
            f"No track data found for visit {visit_id}, task {task_type}. "
            f"Expected at: {worker_npz}. "
            "Run the worker pipeline to process uploaded videos."
        )

    return tracks


def compute_tracking_quality(
    tracks: Optional[Dict[str, Any]],
    t_start_sec: float,
    t_end_sec: float
) -> TrackingQuality:
    """Compute tracking quality metrics for a time window."""
    if tracks is None or len(tracks.get("t_sec", [])) == 0:
        return TrackingQuality(quality="low", quality_pct=0, face_visible=False)

    t = tracks["t_sec"]
    mask = (t >= t_start_sec) & (t <= t_end_sec)

    if not mask.any():
        return TrackingQuality(quality="low", quality_pct=0, face_visible=False)

    # Check child bbox visibility
    child_bbox = tracks.get("child_bbox", np.array([]))
    if len(child_bbox) > 0 and child_bbox.shape[0] == len(t):
        bbox_valid = np.isfinite(child_bbox[mask, 0])
        conf = child_bbox[mask, 4] if child_bbox.shape[1] > 4 else np.ones(mask.sum())
        visible_frames = (bbox_valid & (conf > 0.2)).sum()
        total_frames = mask.sum()
        quality_pct = int(100 * visible_frames / max(total_frames, 1))
    else:
        quality_pct = 50  # default if no bbox data

    # Determine quality level
    if quality_pct >= 80:
        quality = "good"
    elif quality_pct >= 50:
        quality = "medium"
    else:
        quality = "low"

    face_visible = quality_pct >= 30

    return TrackingQuality(quality=quality, quality_pct=quality_pct, face_visible=face_visible)


def detect_head_turn_response(
    tracks: Optional[Dict[str, Any]],
    prompt_time_sec: float,
    window_sec: float = 3.0
) -> HeadTurnResponse:
    """
    Detect if a head turn occurred in response to a prompt.

    Uses pose landmarks to detect significant head movement/orientation change.
    """
    if tracks is None or len(tracks.get("pose", [])) == 0:
        return HeadTurnResponse(detected=False, description="No tracking data")

    t = tracks["t_sec"]
    pose = tracks["pose"]  # (N, 33, 4) - MediaPipe pose landmarks

    # Find frames in response window
    mask = (t >= prompt_time_sec) & (t <= prompt_time_sec + window_sec)
    if not mask.any():
        return HeadTurnResponse(detected=False, description="No frames in window")

    # Get nose (landmark 0) and ears (landmarks 7, 8) for head orientation
    nose_x = pose[mask, 0, 0]  # nose x position
    left_ear_x = pose[mask, 7, 0]  # left ear x
    right_ear_x = pose[mask, 8, 0]  # right ear x

    # Check data validity
    if not np.isfinite(nose_x).any():
        return HeadTurnResponse(detected=False, description="Invalid pose data")

    # Compute head angle proxy (difference between ear positions)
    ear_diff = left_ear_x - right_ear_x
    ear_diff_valid = ear_diff[np.isfinite(ear_diff)]

    if len(ear_diff_valid) < 3:
        return HeadTurnResponse(detected=False, description="Insufficient ear data")

    # Compute velocity of head rotation
    ear_diff_interp = np.interp(
        np.arange(len(ear_diff)),
        np.where(np.isfinite(ear_diff))[0],
        ear_diff_valid
    )
    velocity = np.abs(np.diff(ear_diff_interp))

    # Detect significant movement (threshold based on typical head turn)
    turn_threshold = 0.02  # normalized coordinates
    significant_movement = velocity > turn_threshold

    if significant_movement.any():
        # Find first significant movement frame
        first_movement_idx = np.where(significant_movement)[0][0]
        frames_in_window = t[mask]
        response_time_sec = frames_in_window[first_movement_idx]
        latency_sec = response_time_sec - prompt_time_sec
        latency_ms = latency_sec * 1000
        t_observed_ms = response_time_sec * 1000

        # Confidence based on movement magnitude
        max_velocity = float(np.max(velocity))
        confidence = min(1.0, max_velocity / 0.05)

        return HeadTurnResponse(
            detected=True,
            latency_ms=latency_ms,
            t_observed_ms=t_observed_ms,
            confidence=confidence,
            description=f"Head turn at {latency_ms:.0f}ms"
        )

    return HeadTurnResponse(
        detected=False,
        confidence=0.3,
        description="No significant head movement detected"
    )


def determine_observation_status(
    response: HeadTurnResponse,
    quality: TrackingQuality,
    task_type: str
) -> Tuple[str, str]:
    """Determine the observation status and description."""
    if quality.quality == "low" or not quality.face_visible:
        return "uncertain", "Unable to determine (low tracking quality)"

    if not response.detected:
        return "not_observed", "No response detected in window"

    latency_ms = response.latency_ms or 0
    delayed_threshold = DELAYED_THRESHOLD_MS

    if latency_ms > delayed_threshold:
        return "delayed", f"Response at {latency_ms:.0f}ms (delayed)"

    return "observed", f"Response at {latency_ms:.0f}ms"


@dataclass
class DisengagementPeak:
    """Result of finding peak disengagement in tracking data."""
    found: bool
    t_sec: float = 0.0
    duration_sec: float = 0.0
    description: str = ""


# MediaPipe pose landmark indices
POSE_NOSE = 0
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
MIN_VISIBILITY = 0.3
DISENGAGE_HEAD_THRESHOLD = 0.04  # Head offset from center indicating looking away (lowered for sensitivity)


def find_peak_disengagement(
    tracks: Optional[Dict[str, Any]],
    min_duration_sec: float = 0.3,
) -> DisengagementPeak:
    """
    Find the peak disengagement moment in free play tracking data.

    Analyzes head orientation over time to find the longest/most significant
    period where the child is facing away from the camera/adult.

    Args:
        tracks: Tracking data with pose landmarks
        min_duration_sec: Minimum duration for a disengagement period to be flagged

    Returns:
        DisengagementPeak with the timestamp and duration of peak disengagement
    """
    if tracks is None or len(tracks.get("pose", [])) == 0:
        return DisengagementPeak(found=False, description="No tracking data")

    t = tracks["t_sec"]
    pose = tracks["pose"]  # (N, 33, 4) - MediaPipe pose landmarks

    if len(pose) < 10:
        return DisengagementPeak(found=False, description="Insufficient tracking frames")

    # Compute head orientation signal (nose_x - shoulder_midpoint_x)
    nose_x = pose[:, POSE_NOSE, 0]
    nose_vis = pose[:, POSE_NOSE, 3]
    l_shoulder_x = pose[:, POSE_L_SHOULDER, 0]
    l_shoulder_vis = pose[:, POSE_L_SHOULDER, 3]
    r_shoulder_x = pose[:, POSE_R_SHOULDER, 0]
    r_shoulder_vis = pose[:, POSE_R_SHOULDER, 3]

    # Valid mask: require key landmarks visible
    valid_mask = (
        (nose_vis >= MIN_VISIBILITY) &
        (l_shoulder_vis >= MIN_VISIBILITY) &
        (r_shoulder_vis >= MIN_VISIBILITY)
    )

    if valid_mask.sum() < 10:
        return DisengagementPeak(found=False, description="Insufficient valid frames")

    # Compute head signal (relative to shoulder midpoint)
    shoulder_mid_x = 0.5 * (l_shoulder_x + r_shoulder_x)
    head_signal = np.where(valid_mask, nose_x - shoulder_mid_x, 0)
    abs_head_signal = np.abs(head_signal)

    # Find frames where child is looking away (high absolute head offset)
    looking_away = abs_head_signal > DISENGAGE_HEAD_THRESHOLD

    # Find continuous periods of looking away
    disengagement_periods = []
    in_period = False
    period_start = 0

    for i in range(len(looking_away)):
        if looking_away[i] and valid_mask[i]:
            if not in_period:
                in_period = True
                period_start = i
        else:
            if in_period:
                in_period = False
                period_end = i
                duration = t[period_end - 1] - t[period_start]
                if duration >= min_duration_sec:
                    period_head = abs_head_signal[period_start:period_end]
                    peak_idx = period_start + np.argmax(period_head)
                    disengagement_periods.append({
                        "start": period_start,
                        "end": period_end,
                        "peak_idx": peak_idx,
                        "duration": duration,
                        "max_deviation": float(np.max(period_head)),
                    })

    # Handle period that extends to end of video
    if in_period:
        period_end = len(looking_away)
        duration = t[period_end - 1] - t[period_start]
        if duration >= min_duration_sec:
            period_head = abs_head_signal[period_start:period_end]
            peak_idx = period_start + np.argmax(period_head)
            disengagement_periods.append({
                "start": period_start,
                "end": period_end,
                "peak_idx": peak_idx,
                "duration": duration,
                "max_deviation": float(np.max(period_head)),
            })

    # If no periods found with strict criteria, find the moment with max head deviation
    # This ensures we always find something if there's valid tracking data
    if not disengagement_periods:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            # Find the frame with maximum head deviation
            valid_head = abs_head_signal[valid_mask]
            max_idx_in_valid = np.argmax(valid_head)
            peak_idx = valid_indices[max_idx_in_valid]
            max_dev = float(valid_head[max_idx_in_valid])

            # Only use if there's some meaningful deviation
            if max_dev > 0.02:
                return DisengagementPeak(
                    found=True,
                    t_sec=float(t[peak_idx]),
                    duration_sec=1.0,  # Approximate
                    description=f"Peak head turn at {float(t[peak_idx]):.1f}s",
                )

        return DisengagementPeak(found=False, description="No significant disengagement detected")

    # Find the period with highest peak deviation (most significant disengagement)
    best_period = max(disengagement_periods, key=lambda p: p["max_deviation"])

    peak_t = float(t[best_period["peak_idx"]])
    duration = best_period["duration"]

    return DisengagementPeak(
        found=True,
        t_sec=peak_t,
        duration_sec=duration,
        description=f"Peak disengagement at {peak_t:.1f}s ({duration:.1f}s period)",
    )


def get_status_display(status: str, risk_bucket: str = "moderate") -> Tuple[str, str, str]:
    """Get display properties for a status based on risk level.

    Args:
        status: The observation status
        risk_bucket: Risk level - "low", "moderate", "moderate-high", or "high"
    """
    # 4-tier presentation based on risk level
    if risk_bucket == "low":
        # Low risk (0-25%): Gray/neutral, informational
        status_config = {
            "observed": ("✓", "Observed", "#166534"),
            "delayed": ("○", "Slight Delay", "#6b7280"),
            "partial": ("○", "Partial", "#6b7280"),
            "not_observed": ("○", "Not Detected", "#6b7280"),
            "uncertain": ("○", "Unclear", "#6b7280"),
            "flagged": ("○", "Noted", "#6b7280"),
        }
    elif risk_bucket == "moderate":
        # Moderate risk (26-50%): Soft blue/gray, mild concern
        status_config = {
            "observed": ("✓", "Observed", "#166534"),
            "delayed": ("◐", "Delayed", "#1d4ed8"),  # Blue
            "partial": ("◐", "Partial", "#1d4ed8"),
            "not_observed": ("○", "Not Observed", "#1d4ed8"),
            "uncertain": ("?", "Uncertain", "#6b7280"),
            "flagged": ("◐", "Noted", "#1d4ed8"),
        }
    elif risk_bucket == "moderate-high":
        # Moderate-high risk (51-75%): Orange/amber, warning
        status_config = {
            "observed": ("✓", "Observed", "#166534"),
            "delayed": ("⚠", "Delayed", "#92400e"),
            "partial": ("◐", "Partial", "#92400e"),
            "not_observed": ("⚠", "Not Observed", "#92400e"),
            "uncertain": ("?", "Uncertain", "#6b7280"),
            "flagged": ("⚠", "Flagged", "#92400e"),
        }
    else:  # high
        # High risk (76-100%): Red, alarm
        status_config = {
            "observed": ("✓", "Observed", "#166534"),
            "delayed": ("⚠", "Delayed", "#991b1b"),
            "partial": ("◐", "Partial", "#991b1b"),
            "not_observed": ("✗", "Not Observed", "#991b1b"),
            "uncertain": ("?", "Uncertain", "#6b7280"),
            "flagged": ("✗", "Flagged", "#991b1b"),
        }
    return status_config.get(status, ("?", "Unknown", "#6b7280"))


def get_marker_color(status: str, risk_bucket: str = "moderate") -> str:
    """Get timeline marker color for a status based on risk level.

    Args:
        status: The observation status
        risk_bucket: Risk level - "low", "moderate", "moderate-high", or "high"
    """
    if risk_bucket == "low":
        # Low risk: neutral gray for issues
        colors = {
            "observed": "#22c55e",
            "delayed": "#9ca3af",
            "partial": "#9ca3af",
            "not_observed": "#9ca3af",
            "uncertain": "#9ca3af",
            "flagged": "#9ca3af",
        }
    elif risk_bucket == "moderate":
        # Moderate risk: soft blue for issues
        colors = {
            "observed": "#22c55e",
            "delayed": "#3b82f6",
            "partial": "#3b82f6",
            "not_observed": "#3b82f6",
            "uncertain": "#9ca3af",
            "flagged": "#3b82f6",
        }
    elif risk_bucket == "moderate-high":
        # Moderate-high risk: orange/amber
        colors = {
            "observed": "#22c55e",
            "delayed": "#f59e0b",
            "partial": "#f59e0b",
            "not_observed": "#f59e0b",
            "uncertain": "#9ca3af",
            "flagged": "#f59e0b",
        }
    else:  # high
        # High risk: red
        colors = {
            "observed": "#22c55e",
            "delayed": "#ef4444",
            "partial": "#ef4444",
            "not_observed": "#ef4444",
            "uncertain": "#9ca3af",
            "flagged": "#ef4444",
        }
    return colors.get(status, "#9ca3af")


def format_timestamp(ms: float) -> str:
    """Format milliseconds as mm:ss.s"""
    total_sec = ms / 1000
    minutes = int(total_sec // 60)
    seconds = total_sec % 60
    return f"{minutes}:{seconds:04.1f}"


def _find_response_outcome(
    outcomes: List[ResponseOutcome],
    event_type: str,
    t_start_sec: float,
    tolerance_sec: float = 0.5
) -> Optional[ResponseOutcome]:
    """Find matching response outcome for an event by type and timestamp."""
    for outcome in outcomes:
        if outcome.event_type == event_type and abs(outcome.t_start_sec - t_start_sec) < tolerance_sec:
            return outcome
    return None


def _build_joint_attention_flagged_moments(
    video_id: int,
    audio_events: List[AudioEvent],
    pointing_events: List[PointingEvent],
    tracks: Optional[Dict[str, Any]],
    duration_ms: float,
    response_window_ms: float,
    risk_bucket: str = "moderate",
    response_outcomes: Optional[List[ResponseOutcome]] = None,
) -> List[Dict[str, Any]]:
    """Build flagged moments for joint attention from audio + pointing events.

    Uses saved response_outcomes from ML feature extraction when available,
    ensuring consistency between dashboard explanations and video replay.

    Only includes actual issues (not_observed, delayed, uncertain) - skips successful responses.
    For low-risk children, uses softer colors and less alarming labels.
    """
    flagged_moments: List[Dict[str, Any]] = []
    use_saved_outcomes = response_outcomes is not None and len(response_outcomes) > 0

    if use_saved_outcomes:
        logger.info(f"  Using {len(response_outcomes)} saved response outcomes from ML feature extraction")
    else:
        logger.info("  No saved response outcomes - falling back to real-time detection")

    # Process audio events (parent verbal prompts like "look here")
    for audio_event in audio_events:
        prompt_time_ms = audio_event.t_start * 1000
        window_start_ms = prompt_time_ms
        window_end_ms = min(prompt_time_ms + response_window_ms, duration_ms)
        pause_at_ms = window_end_ms

        # Try to use saved outcome first (consistent with ML features)
        saved_outcome = None
        if use_saved_outcomes:
            saved_outcome = _find_response_outcome(
                response_outcomes, audio_event.event_type, audio_event.t_start
            )
            # When we have saved outcomes, ONLY use events with saved outcomes
            # Skip events that weren't analyzed by ML pipeline
            if not saved_outcome:
                continue

        if saved_outcome:
            # Use saved outcome from ML feature extraction
            status = saved_outcome.status
            latency_ms = saved_outcome.latency_ms
            if saved_outcome.responded and latency_ms is not None:
                observation_desc = f"Response at {latency_ms:.0f}ms"
                if status == "delayed":
                    observation_desc += " (delayed)"
            else:
                observation_desc = "No response detected in window"
            # For saved outcomes, assume good tracking (we had enough data to detect)
            quality = TrackingQuality(quality="good", quality_pct=80, face_visible=True)
            t_observed_ms = prompt_time_ms + latency_ms if latency_ms else None
        else:
            # Fall back to real-time detection (backwards compatibility - no saved outcomes)
            response = detect_head_turn_response(tracks, audio_event.t_start, response_window_ms / 1000)
            quality = compute_tracking_quality(tracks, audio_event.t_start, audio_event.t_start + response_window_ms / 1000)
            status, observation_desc = determine_observation_status(response, quality, "joint_attention")
            latency_ms = response.latency_ms
            t_observed_ms = response.t_observed_ms

        # Only include actual issues - skip successful responses
        if status == "observed":
            continue

        icon, label, color = get_status_display(status, risk_bucket)
        marker_color = get_marker_color(status, risk_bucket)

        flag_id = f"flag_{video_id}_{len(flagged_moments)}"
        prompt_id = f"prompt_{video_id}_{len(flagged_moments)}"

        flagged_moments.append({
            "id": flag_id,
            "promptId": prompt_id,
            "promptAtMs": prompt_time_ms,
            "windowStartMs": window_start_ms,
            "windowEndMs": window_end_ms,
            "pauseAtMs": pause_at_ms,
            "markerColor": marker_color,
            "expected": {
                "type": "head_turn",
                "description": f"Response to '{audio_event.matched_phrase}'",
                "windowDurationMs": response_window_ms,
            },
            "observed": {
                "status": status,
                "description": observation_desc,
                "tObservedMs": t_observed_ms,
                "latencyMs": latency_ms,
            },
            "trackingQuality": quality.quality_pct / 100.0,
            "trackingLabel": quality.quality,
            "faceVisibleDuringWindow": quality.face_visible,
            "pauseCard": {
                "status": status,
                "statusIcon": icon,
                "statusLabel": label,
                "statusColor": color,
                "prompt": {
                    "type": audio_event.event_type.replace("_", " ").title(),
                    "timestamp": format_timestamp(prompt_time_ms),
                    "confidence": audio_event.confidence,
                },
                "expectation": {
                    "description": "Child looks toward indicated direction",
                    "windowDuration": f"{response_window_ms / 1000:.1f}s",
                },
                "observation": {
                    "description": observation_desc,
                    "latencyMs": latency_ms,
                    "latencyDisplay": f"{latency_ms:.0f}ms" if latency_ms else None,
                },
                "tracking": {
                    "quality": quality.quality,
                    "qualityPct": quality.quality_pct,
                    "faceVisible": quality.face_visible,
                },
                "flagIndex": len(flagged_moments),
                "flagTotal": 0,  # Updated at end
            },
        })

    # Process pointing events (parent pointing gestures)
    for point_event in pointing_events:
        prompt_time_ms = point_event.t_start * 1000
        window_start_ms = prompt_time_ms
        window_end_ms = min(point_event.t_end * 1000 + response_window_ms, duration_ms)
        pause_at_ms = point_event.t_end * 1000  # Pause at end of pointing gesture

        # Try to use saved outcome first (consistent with ML features)
        saved_outcome = None
        if use_saved_outcomes:
            saved_outcome = _find_response_outcome(
                response_outcomes, "POINT", point_event.t_start
            )
            # When we have saved outcomes, ONLY use events with saved outcomes
            # Skip pointing events that weren't analyzed by ML pipeline
            if not saved_outcome:
                continue

        if saved_outcome:
            # Use saved outcome from ML feature extraction
            status = saved_outcome.status
            latency_ms = saved_outcome.latency_ms
            if saved_outcome.responded and latency_ms is not None:
                observation_desc = f"Response at {latency_ms:.0f}ms"
                if status == "delayed":
                    observation_desc += " (delayed)"
            else:
                observation_desc = "No response detected in window"
            quality = TrackingQuality(quality="good", quality_pct=80, face_visible=True)
            t_observed_ms = prompt_time_ms + latency_ms if latency_ms else None
        else:
            # Fall back to real-time detection (backwards compatibility - no saved outcomes)
            response = detect_head_turn_response(tracks, point_event.t_start, response_window_ms / 1000)
            quality = compute_tracking_quality(tracks, point_event.t_start, point_event.t_end + response_window_ms / 1000)
            status, observation_desc = determine_observation_status(response, quality, "joint_attention")
            latency_ms = response.latency_ms
            t_observed_ms = response.t_observed_ms

        # Only include actual issues - skip successful responses
        if status == "observed":
            continue

        icon, label, color = get_status_display(status, risk_bucket)
        marker_color = get_marker_color(status, risk_bucket)

        flag_id = f"flag_{video_id}_{len(flagged_moments)}"
        prompt_id = f"point_{video_id}_{len(flagged_moments)}"

        # Format pointing direction
        angle = point_event.point_angle_deg
        if angle < -45:
            direction = "left"
        elif angle > 45:
            direction = "right"
        elif angle < 0:
            direction = "slight left"
        elif angle > 0:
            direction = "slight right"
        else:
            direction = "forward"

        flagged_moments.append({
            "id": flag_id,
            "promptId": prompt_id,
            "promptAtMs": prompt_time_ms,
            "windowStartMs": window_start_ms,
            "windowEndMs": window_end_ms,
            "pauseAtMs": pause_at_ms,
            "markerColor": marker_color,
            "expected": {
                "type": "gaze_follow",
                "description": f"Follow point ({direction})",
                "windowDurationMs": response_window_ms,
            },
            "observed": {
                "status": status,
                "description": observation_desc,
                "tObservedMs": response.t_observed_ms,
                "latencyMs": response.latency_ms,
            },
            "trackingQuality": quality.quality_pct / 100.0,
            "trackingLabel": quality.quality,
            "faceVisibleDuringWindow": quality.face_visible,
            "pauseCard": {
                "status": status,
                "statusIcon": icon,
                "statusLabel": label,
                "statusColor": color,
                "prompt": {
                    "type": "Parent Point",
                    "timestamp": format_timestamp(prompt_time_ms),
                    "confidence": point_event.confidence,
                },
                "expectation": {
                    "description": f"Child follows point to the {direction}",
                    "windowDuration": f"{response_window_ms / 1000:.1f}s",
                },
                "observation": {
                    "description": observation_desc,
                    "latencyMs": response.latency_ms,
                    "latencyDisplay": f"{response.latency_ms:.0f}ms" if response.latency_ms else None,
                },
                "tracking": {
                    "quality": quality.quality,
                    "qualityPct": quality.quality_pct,
                    "faceVisible": quality.face_visible,
                },
                "flagIndex": len(flagged_moments),
                "flagTotal": 0,  # Updated at end
            },
        })

    # Sort by time so events are in chronological order
    flagged_moments.sort(key=lambda m: m["promptAtMs"])

    # Re-index after sorting and update flagTotal
    total = len(flagged_moments)
    for i, moment in enumerate(flagged_moments):
        moment["pauseCard"]["flagIndex"] = i
        moment["pauseCard"]["flagTotal"] = total

    return flagged_moments


def _find_imitation_outcome(
    outcomes: List[ImitationOutcome],
    action_type: str,
) -> Optional[ImitationOutcome]:
    """Find matching imitation outcome for an action type."""
    action_upper = action_type.upper()
    for outcome in outcomes:
        if outcome.action_type.upper() == action_upper:
            return outcome
    return None


def _build_imitation_flagged_moments(
    video_id: int,
    imitation_events: List[ImitationEvent],
    tracks: Optional[Dict[str, Any]],
    duration_ms: float,
    response_window_ms: float,
    risk_bucket: str = "moderate",
    imitation_outcomes: Optional[List[ImitationOutcome]] = None,
) -> List[Dict[str, Any]]:
    """Build flagged moments for imitation from parent demo + child response events.

    Uses saved imitation_outcomes from ML feature extraction when available,
    ensuring consistency between dashboard explanations and video replay.

    Only includes actual issues (not_observed, delayed, uncertain) - skips successful imitations.
    For low-risk children, uses softer colors and less alarming labels.
    """
    flagged_moments: List[Dict[str, Any]] = []
    use_saved_outcomes = imitation_outcomes is not None and len(imitation_outcomes) > 0

    if use_saved_outcomes:
        logger.info(f"  Using {len(imitation_outcomes)} saved imitation outcomes from ML feature extraction")
    else:
        logger.info("  No saved imitation outcomes - falling back to event matching")

    # Separate parent demos and child responses
    parent_demos: List[ImitationEvent] = []
    child_responses: List[ImitationEvent] = []

    logger.info(f"  Processing {len(imitation_events)} imitation events")

    for event in imitation_events:
        event_type_upper = event.event_type.upper()

        if event.subject == "parent" or event.subject == "adult":
            # Parent demo events - only use START events (avoid END events)
            # Match: PARENT_ACTION_START, DEMO_START
            # Skip: PARENT_ACTION_END, DEMO_END
            if "START" in event_type_upper and "END" not in event_type_upper:
                # Avoid duplicate entries (PARENT_ACTION_START and DEMO_START are often the same)
                if "DEMO_START" in event_type_upper:
                    # Skip DEMO_START if we already have PARENT_ACTION_START at same time
                    already_have = any(
                        p.action_type == event.action_type and abs(p.t_sec - event.t_sec) < 0.1
                        for p in parent_demos
                    )
                    if not already_have:
                        parent_demos.append(event)
                else:
                    # PARENT_ACTION_START or other START events
                    parent_demos.append(event)
        elif event.subject == "child":
            # Child response events
            if "ATTEMPT" in event_type_upper or "FAILURE" in event_type_upper or "RESPONSE" in event_type_upper:
                child_responses.append(event)

    # Sort by time
    parent_demos.sort(key=lambda e: e.t_sec)
    child_responses.sort(key=lambda e: e.t_sec)

    logger.info(f"  Found {len(parent_demos)} parent demos, {len(child_responses)} child responses")

    # When we have saved outcomes, iterate over them directly
    # (they contain demo timestamps and all needed info)
    if use_saved_outcomes and imitation_outcomes:
        for saved_outcome in imitation_outcomes:
            action_type = saved_outcome.action_type.upper()
            status = saved_outcome.status

            # Only include actual issues - skip successful imitations
            if status == "observed":
                continue

            # Use demo timestamp from saved outcome
            prompt_time_ms = saved_outcome.t_sec * 1000
            window_start_ms = prompt_time_ms
            window_end_ms = min(prompt_time_ms + response_window_ms, duration_ms)
            pause_at_ms = window_end_ms

            latency_ms = saved_outcome.latency_ms
            if saved_outcome.responded and latency_ms is not None:
                observation_desc = f"Child imitated {action_type.lower().replace('_', ' ')} at {latency_ms:.0f}ms"
                if status == "delayed":
                    observation_desc += " (delayed)"
                t_observed_ms = prompt_time_ms + latency_ms
            else:
                observation_desc = f"No imitation of {action_type.lower().replace('_', ' ')}"
                t_observed_ms = None

            # Compute tracking quality
            demo_t_sec = saved_outcome.t_sec
            quality = compute_tracking_quality(tracks, demo_t_sec, demo_t_sec + response_window_ms / 1000)

            icon, label, color = get_status_display(status, risk_bucket)
            marker_color = get_marker_color(status, risk_bucket)

            action_display = action_type.replace("_", " ").title()
            idx = len(flagged_moments)
            flag_id = f"flag_{video_id}_{idx}"
            prompt_id = f"prompt_{video_id}_{idx}"

            flagged_moments.append({
                "id": flag_id,
                "promptId": prompt_id,
                "promptAtMs": prompt_time_ms,
                "windowStartMs": window_start_ms,
                "windowEndMs": window_end_ms,
                "pauseAtMs": pause_at_ms,
                "markerColor": marker_color,
                "expected": {
                    "type": "imitation",
                    "description": f"Child imitates {action_display}",
                    "windowDurationMs": response_window_ms,
                },
                "observed": {
                    "status": status,
                    "description": observation_desc,
                    "tObservedMs": t_observed_ms,
                    "latencyMs": latency_ms,
                },
                "trackingQuality": quality.quality_pct / 100.0,
                "trackingLabel": quality.quality,
                "faceVisibleDuringWindow": quality.face_visible,
                "pauseCard": {
                    "status": status,
                    "statusIcon": icon,
                    "statusLabel": label,
                    "statusColor": color,
                    "prompt": {
                        "type": f"Parent {action_display}",
                        "timestamp": format_timestamp(prompt_time_ms),
                        "confidence": 0.9,  # Default confidence for saved outcomes
                    },
                    "expectation": {
                        "description": f"Child imitates {action_display.lower()} action",
                        "windowDuration": f"{response_window_ms / 1000:.1f}s",
                    },
                    "observation": {
                        "description": observation_desc,
                        "latencyMs": latency_ms,
                        "latencyDisplay": f"{latency_ms:.0f}ms" if latency_ms else None,
                    },
                    "tracking": {
                        "quality": quality.quality,
                        "qualityPct": quality.quality_pct,
                        "faceVisible": quality.face_visible,
                    },
                    "flagIndex": idx,
                    "flagTotal": 0,  # Updated below
                },
            })
    else:
        # Fall back to event-based matching (backwards compatibility)
        # Group demos by action_type
        action_demos: Dict[str, List[ImitationEvent]] = {}
        for demo in parent_demos:
            action = demo.action_type.upper()
            if action not in action_demos:
                action_demos[action] = []
            action_demos[action].append(demo)

        # Create flagged moment for each parent demo that wasn't successfully imitated
        for demo in parent_demos:
            action_type = demo.action_type.upper()
            prompt_time_ms = demo.t_sec * 1000
            window_start_ms = prompt_time_ms
            window_end_ms = min(prompt_time_ms + response_window_ms, duration_ms)
            pause_at_ms = window_end_ms

            # Find matching child response for this action type
            matching_child = None
            for child_event in child_responses:
                if child_event.action_type.upper() == action_type:
                    child_time = child_event.t_sec
                    # Check if child response is after the demo and within window
                    if child_time >= demo.t_sec and (child_time - demo.t_sec) * 1000 <= response_window_ms:
                        matching_child = child_event
                        break

            # Determine status based on child response
            if matching_child:
                event_type_upper = matching_child.event_type.upper()

                if "FAILURE" in event_type_upper:
                    # Explicit failure marked by ML pipeline
                    status = "not_observed"
                    observation_desc = f"No imitation of {action_type.lower().replace('_', ' ')}"
                    latency_ms = None
                    t_observed_ms = None
                else:
                    # Child attempted - calculate latency
                    latency_sec = matching_child.t_sec - demo.t_sec
                    latency_ms = latency_sec * 1000
                    t_observed_ms = matching_child.t_sec * 1000

                    if latency_ms <= DELAYED_THRESHOLD_MS:
                        status = "observed"
                        observation_desc = f"Child imitated {action_type.lower().replace('_', ' ')} at {latency_ms:.0f}ms"
                    else:
                        status = "delayed"
                        observation_desc = f"Delayed imitation at {latency_ms:.0f}ms"
            else:
                # No child response found
                status = "not_observed"
                observation_desc = f"No imitation of {action_type.lower().replace('_', ' ')}"
                latency_ms = None
                t_observed_ms = None

            quality = compute_tracking_quality(tracks, demo.t_sec, demo.t_sec + response_window_ms / 1000)
            if quality.quality == "low":
                status = "uncertain"
                observation_desc = "Unable to determine (low tracking quality)"

            # Only include actual issues - skip successful imitations
            if status == "observed":
                continue

            icon, label, color = get_status_display(status, risk_bucket)
            marker_color = get_marker_color(status, risk_bucket)

            action_display = action_type.replace("_", " ").title()
            idx = len(flagged_moments)
            flag_id = f"flag_{video_id}_{idx}"
            prompt_id = f"prompt_{video_id}_{idx}"

            flagged_moments.append({
                "id": flag_id,
                "promptId": prompt_id,
                "promptAtMs": prompt_time_ms,
                "windowStartMs": window_start_ms,
                "windowEndMs": window_end_ms,
                "pauseAtMs": pause_at_ms,
                "markerColor": marker_color,
                "expected": {
                    "type": "imitation",
                    "description": f"Child imitates {action_display}",
                    "windowDurationMs": response_window_ms,
                },
                "observed": {
                    "status": status,
                    "description": observation_desc,
                    "tObservedMs": t_observed_ms,
                    "latencyMs": latency_ms,
                },
                "trackingQuality": quality.quality_pct / 100.0,
                "trackingLabel": quality.quality,
                "faceVisibleDuringWindow": quality.face_visible,
                "pauseCard": {
                    "status": status,
                    "statusIcon": icon,
                    "statusLabel": label,
                    "statusColor": color,
                    "prompt": {
                        "type": f"Parent {action_display}",
                        "timestamp": format_timestamp(prompt_time_ms),
                        "confidence": demo.confidence,
                    },
                    "expectation": {
                        "description": f"Child imitates {action_display.lower()} action",
                        "windowDuration": f"{response_window_ms / 1000:.1f}s",
                    },
                    "observation": {
                        "description": observation_desc,
                        "latencyMs": latency_ms,
                        "latencyDisplay": f"{latency_ms:.0f}ms" if latency_ms else None,
                    },
                    "tracking": {
                        "quality": quality.quality,
                        "qualityPct": quality.quality_pct,
                        "faceVisible": quality.face_visible,
                    },
                    "flagIndex": idx,
                    "flagTotal": 0,  # Updated below
                },
            })

    # Update flagTotal for all moments
    total = len(flagged_moments)
    for i, moment in enumerate(flagged_moments):
        moment["pauseCard"]["flagIndex"] = i
        moment["pauseCard"]["flagTotal"] = total

    return flagged_moments


def _find_free_play_outcome(
    outcomes: List[FreePlayOutcome],
    event_type: str,
    t_start_sec: float,
    tolerance_sec: float = 0.5
) -> Optional[FreePlayOutcome]:
    """Find matching free play outcome for an event by type and timestamp."""
    event_upper = event_type.upper()
    for outcome in outcomes:
        if outcome.event_type.upper() == event_upper and abs(outcome.t_start_sec - t_start_sec) < tolerance_sec:
            return outcome
    return None


def _build_free_play_flagged_moments(
    video_id: int,
    behavioral_events: List[BehavioralEvent],
    tracks: Optional[Dict[str, Any]],
    duration_ms: float,
    flagged_behavior_types: Optional[set] = None,
    risk_bucket: str = "moderate",
    free_play_outcomes: Optional[List[FreePlayOutcome]] = None,
    limited_social_engagement: bool = False,
) -> List[Dict[str, Any]]:
    """Build flagged moments for free play from behavioral events.

    Uses saved free_play_outcomes from ML feature extraction when available,
    ensuring consistency between dashboard explanations and video replay.

    Only shows individual events for behaviors that are actually flagged in the
    dashboard/assessment (based on aggregate thresholds like >30% repetitive motion).
    For low-risk children, uses softer colors and less alarming labels.

    Args:
        flagged_behavior_types: Set of behavior types to show (e.g., {"PERIODIC_MOTION"}).
                               If None or empty and no saved outcomes, returns no flags.
        risk_bucket: Risk level for presentation - "low", "moderate", "moderate-high", or "high".
        free_play_outcomes: Optional list of saved outcomes from ML feature extraction.
        limited_social_engagement: If True, add a flag for limited social engagement.
                                   This is triggered when fp_engaged_time_frac < 0.3.
    """
    flagged_moments: List[Dict[str, Any]] = []

    # Check if we have saved outcomes from ML feature extraction
    # These are the source of truth - they use the same threshold logic as features
    has_saved_outcomes = free_play_outcomes is not None

    if has_saved_outcomes:
        # Use saved outcomes - only show events that are actually flagged
        flagged_outcomes = [o for o in free_play_outcomes if o.flagged]
        if flagged_outcomes:
            logger.info(f"  Using {len(flagged_outcomes)} flagged free play outcomes from ML feature extraction")
            flagged_event_types = {o.event_type.upper() for o in flagged_outcomes}
        else:
            # Outcomes exist but none are flagged - this means thresholds weren't met
            logger.info("  Saved outcomes exist but none flagged")
            flagged_event_types = set()
    else:
        # No saved outcomes file - this is a legacy/backwards compatibility case
        if flagged_behavior_types:
            logger.info(f"  No saved outcomes - using flagged_behavior_types: {flagged_behavior_types}")
            flagged_event_types = flagged_behavior_types
        else:
            logger.info("  No saved outcomes and no flagged behavior types")
            flagged_event_types = set()

    # Add flag for limited social engagement if ML detected it
    # Find the peak disengagement moment from head orientation tracking
    if limited_social_engagement:
        logger.info(f"  Looking for peak disengagement (limited_social_engagement=True)")
        disengage = find_peak_disengagement(tracks)
        logger.info(f"  Disengagement result: found={disengage.found}, desc={disengage.description}")

        if disengage.found:
            # Flag at the actual peak disengagement moment
            t_sec = disengage.t_sec
            t_ms = t_sec * 1000
            window_start = max(0, t_ms - 1500)  # 1.5s window around peak
            window_end = min(duration_ms, t_ms + 1500)

            quality = compute_tracking_quality(tracks, t_sec - 1.5, t_sec + 1.5)
            icon, label, color = get_status_display("flagged", risk_bucket)
            marker_color = get_marker_color("flagged", risk_bucket)

            flag_id = f"flag_{video_id}_social_engagement"
            prompt_id = f"prompt_{video_id}_social_engagement"

            flagged_moments.append({
                "id": flag_id,
                "promptId": prompt_id,
                "promptAtMs": t_ms,
                "windowStartMs": window_start,
                "windowEndMs": window_end,
                "pauseAtMs": t_ms,
                "markerColor": marker_color,
                "expected": {
                    "type": "behavior",
                    "description": "Social engagement during play",
                    "windowDurationMs": window_end - window_start,
                },
                "observed": {
                    "status": "flagged",
                    "description": f"Child facing away ({disengage.duration_sec:.1f}s period)",
                    "tObservedMs": t_ms,
                    "latencyMs": None,
                },
                "trackingQuality": quality.quality_pct / 100.0,
                "trackingLabel": quality.quality,
                "faceVisibleDuringWindow": quality.face_visible,
                "pauseCard": {
                    "status": "flagged",
                    "statusIcon": icon,
                    "statusLabel": label,
                    "statusColor": color,
                    "prompt": {
                        "type": "Limited Social Engagement",
                        "timestamp": format_timestamp(t_ms),
                        "confidence": 0.9,
                    },
                    "expectation": {
                        "description": "Child faces toward caregiver/toys",
                        "windowDuration": f"{(window_end - window_start) / 1000:.1f}s",
                    },
                    "observation": {
                        "description": f"Child facing away from adult ({disengage.duration_sec:.1f}s period)",
                        "latencyMs": None,
                        "latencyDisplay": None,
                    },
                    "tracking": {
                        "quality": quality.quality,
                        "qualityPct": quality.quality_pct,
                        "faceVisible": quality.face_visible,
                    },
                    "flagIndex": 0,
                    "flagTotal": 1,
                },
            })
            logger.info(f"  Added flag for limited social engagement at {t_sec:.1f}s (peak disengagement)")
        else:
            # No peak found in tracking - log but don't add flag
            logger.info(f"  Limited social engagement indicated but no peak found: {disengage.description}")

    if not flagged_event_types:
        return flagged_moments

    for event in behavioral_events:
        event_type_upper = event.event_type.upper()

        if has_saved_outcomes:
            # Check if this specific event was flagged in saved outcomes
            outcome = _find_free_play_outcome(free_play_outcomes, event_type_upper, event.t_start)
            if not outcome or not outcome.flagged:
                continue
        else:
            # Legacy fallback: type-based filtering (only when no outcomes file exists)
            if event_type_upper not in flagged_event_types:
                continue

        t_start_ms = event.t_start * 1000
        t_end_ms = event.t_end * 1000
        duration_event_ms = t_end_ms - t_start_ms

        # Compute tracking quality during event window
        quality = compute_tracking_quality(tracks, event.t_start, event.t_end)

        status = "flagged"

        # Format description based on event type
        if event_type_upper == "PERIODIC_MOTION":
            prompt_type = "Repetitive Motion"
            description = "Repetitive/periodic motion detected"
            expected_desc = "Varied movement patterns"
        elif event_type_upper == "HAND_TO_FACE":
            prompt_type = "Hand-to-Face Contact"
            description = "Frequent hand-to-face contact observed"
            expected_desc = "Occasional or no hand-to-face contact"
        else:
            prompt_type = event_type_upper.replace("_", " ").title()
            description = f"{prompt_type} detected"
            expected_desc = "Typical play behavior"

        icon, label, color = get_status_display(status, risk_bucket)
        marker_color = get_marker_color(status, risk_bucket)

        idx = len(flagged_moments)
        flag_id = f"flag_{video_id}_{idx}"
        prompt_id = f"prompt_{video_id}_{idx}"

        flagged_moments.append({
            "id": flag_id,
            "promptId": prompt_id,
            "promptAtMs": t_start_ms,
            "windowStartMs": t_start_ms,
            "windowEndMs": t_end_ms,
            "pauseAtMs": t_end_ms,
            "markerColor": marker_color,
            "expected": {
                "type": "behavior",
                "description": expected_desc,
                "windowDurationMs": duration_event_ms,
            },
            "observed": {
                "status": status,
                "description": description,
                "tObservedMs": t_start_ms,
                "latencyMs": None,
            },
            "trackingQuality": quality.quality_pct / 100.0,
            "trackingLabel": quality.quality,
            "faceVisibleDuringWindow": quality.face_visible,
            "pauseCard": {
                "status": status,
                "statusIcon": icon,
                "statusLabel": label,
                "statusColor": color,
                "prompt": {
                    "type": prompt_type,
                    "timestamp": format_timestamp(t_start_ms),
                    "confidence": event.confidence,
                },
                "expectation": {
                    "description": expected_desc,
                    "windowDuration": f"{duration_event_ms / 1000:.1f}s",
                },
                "observation": {
                    "description": description,
                    "latencyMs": None,
                    "latencyDisplay": None,
                },
                "tracking": {
                    "quality": quality.quality,
                    "qualityPct": quality.quality_pct,
                    "faceVisible": quality.face_visible,
                },
                "flagIndex": idx,
                "flagTotal": 0,
            },
        })

    # Sort by time and update flagTotal
    flagged_moments.sort(key=lambda m: m["promptAtMs"])
    total = len(flagged_moments)
    for i, moment in enumerate(flagged_moments):
        moment["pauseCard"]["flagIndex"] = i
        moment["pauseCard"]["flagTotal"] = total

    return flagged_moments


def _categorize_explanation(explanation: str) -> str:
    """Categorize an explanation by its source task type."""
    exp_lower = explanation.lower()

    # Joint attention indicators
    ja_keywords = [
        "attention bid", "gaze-following", "follow point", "orienting to name",
        "social cues", "stillness during joint attention", "joint attention"
    ]
    for kw in ja_keywords:
        if kw in exp_lower:
            return "joint_attention"

    # Imitation indicators
    imit_keywords = [
        "imitation", "arm-raise", "clapping", "demonstrated action"
    ]
    for kw in imit_keywords:
        if kw in exp_lower:
            return "imitation"

    # Free play indicators
    fp_keywords = [
        "free play", "repetitive motion", "hand-to-face", "eye contact during free",
        "social engagement"
    ]
    for kw in fp_keywords:
        if kw in exp_lower:
            return "free_play"

    # Questionnaire/general
    if "caregiver" in exp_lower or "reported" in exp_lower:
        return "questionnaire"

    return "general"


def _get_ml_prediction_info(visit_id: int) -> Tuple[set, str, Dict[str, List[str]]]:
    """Get ML prediction info for a visit.

    Returns:
        Tuple of (flagged_behavior_types, risk_bucket, categorized_explanations)
        - flagged_behavior_types: Set of behavior types to show in video replay
        - risk_bucket: "low", "moderate", "moderate-high", or "high"
        - categorized_explanations: Dict mapping task_type to list of explanations
    """
    flagged_types = set()
    risk_bucket = "moderate"  # Default to moderate if no prediction found
    categorized_explanations: Dict[str, List[str]] = {
        "joint_attention": [],
        "imitation": [],
        "free_play": [],
        "questionnaire": [],
        "general": [],
    }

    try:
        # Import here to avoid circular imports
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # Get database URL from environment or use default
        import os
        db_url = os.getenv("DATABASE_URL", "postgresql://neurimo:neurimo@localhost:5432/neurimo")

        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)

        with Session() as db:
            # Import with fallback for different environments (Docker vs local)
            try:
                from app.models.ml_prediction import MLPrediction
            except ImportError:
                from backend.app.models.ml_prediction import MLPrediction
            prediction = db.query(MLPrediction).filter(MLPrediction.visit_id == visit_id).first()

            if prediction:
                logger.info(f"  ML prediction found for visit {visit_id}: bucket={prediction.asd_risk_bucket}, prob={prediction.probability}")

                # Get risk bucket directly from prediction
                if prediction.asd_risk_bucket:
                    risk_bucket = prediction.asd_risk_bucket
                elif prediction.probability:
                    # Fallback: compute from probability
                    prob_pct = prediction.probability * 100
                    if prob_pct <= 25:
                        risk_bucket = "low"
                    elif prob_pct <= 50:
                        risk_bucket = "moderate"
                    elif prob_pct <= 75:
                        risk_bucket = "moderate-high"
                    else:
                        risk_bucket = "high"

                # Categorize and process explanations
                if prediction.explanations:
                    logger.info(f"  ML explanations: {prediction.explanations}")

                    for explanation in prediction.explanations:
                        # Categorize by task type
                        task_type = _categorize_explanation(explanation)
                        categorized_explanations[task_type].append(explanation)
                        logger.info(f"    '{explanation}' -> {task_type}")

                    explanations_lower = " ".join(prediction.explanations).lower()

                    # Free play specific flags for video replay:
                    # "Elevated repetitive motion patterns during free play"
                    if "repetitive motion" in explanations_lower:
                        flagged_types.add("PERIODIC_MOTION")
                        logger.info("    -> Flagging PERIODIC_MOTION for video replay")

                    # "Frequent hand-to-face contact during free play"
                    if "hand-to-face" in explanations_lower:
                        flagged_types.add("HAND_TO_FACE")
                        logger.info("    -> Flagging HAND_TO_FACE for video replay")

                    # "Limited social engagement during free play"
                    if "social engagement" in explanations_lower or "limited engagement" in explanations_lower:
                        flagged_types.add("ACTIVITY_HIGH")
                        flagged_types.add("OFF_SCREEN")
                        logger.info("    -> Flagging ACTIVITY_HIGH, OFF_SCREEN for video replay")
            else:
                logger.info(f"  No ML prediction found for visit {visit_id}")

    except Exception as e:
        logger.warning(f"Could not load ML prediction for visit {visit_id}: {e}")

    return flagged_types, risk_bucket, categorized_explanations


def build_guided_review_data(
    video_id: int,
    visit_id: int,
    task_type: str,
    video_url: str,
    duration_ms: float,
    age_bucket: str = "12-24 months"
) -> Dict[str, Any]:
    """
    Build complete GuidedReviewData for a video.

    Args:
        video_id: Database video ID
        visit_id: Database visit ID (used to find worker-processed data)
        task_type: "joint_attention", "imitation", or "free_play"
        video_url: URL to video file
        duration_ms: Video duration in milliseconds
        age_bucket: Age bucket string

    Returns:
        GuidedReviewData dict ready for frontend consumption
    """
    logger.info(f"Building guided review for video {video_id}, visit {visit_id}, task {task_type}")

    # Get ML prediction info (flagged behaviors, risk level, and categorized explanations)
    flagged_behavior_types, risk_bucket, categorized_explanations = _get_ml_prediction_info(visit_id)
    logger.info(f"  Risk bucket: {risk_bucket}, flagged behaviors: {flagged_behavior_types}")
    logger.info(f"  Explanations for {task_type}: {categorized_explanations.get(task_type, [])}")

    audio_events = load_audio_events(visit_id)
    pointing_events = load_pointing_events(visit_id)
    imitation_events = load_imitation_events(visit_id)
    behavioral_events = load_behavioral_events(visit_id)

    # Load saved outcomes from ML feature extraction for consistency
    response_outcomes = load_response_outcomes(visit_id)  # Joint attention outcomes
    imitation_outcomes = load_imitation_outcomes(visit_id)  # Imitation outcomes
    free_play_outcomes = load_free_play_outcomes(visit_id)  # Free play outcomes
    tracks = load_tracks(visit_id, task_type)

    logger.info(f"  Loaded events - audio: {len(audio_events)}, pointing: {len(pointing_events)}, "
                f"imitation: {len(imitation_events)}, behavioral: {len(behavioral_events)}")
    logger.info(f"  Loaded outcomes - JA: {len(response_outcomes)}, imitation: {len(imitation_outcomes)}, "
                f"free_play: {len(free_play_outcomes)}, tracks: {'yes' if tracks else 'no'}")

    # Get FPS from tracks or default
    fps = tracks.get("fps", 30.0) if tracks else 30.0

    # Build flagged moments based on task type
    flagged_moments: List[Dict[str, Any]] = []
    response_window_ms = RESPONSE_WINDOW_MS.get(task_type, 3000)

    # Check if ML detected limited social engagement (fp_engaged_time_frac < 0.3)
    # This is used for both free play flags and summary clinical note
    has_limited_social_engagement = any(
        "limited social engagement" in exp.lower()
        for exps in categorized_explanations.values() for exp in exps
    )
    if task_type == "free_play":
        logger.info(f"  Free play explanations: {categorized_explanations}")
        logger.info(f"  has_limited_social_engagement: {has_limited_social_engagement}")

    if task_type == "joint_attention":
        # Joint attention: use audio events (parent prompts) + pointing events
        # Pass response_outcomes for consistency with ML feature extraction
        flagged_moments = _build_joint_attention_flagged_moments(
            video_id, audio_events, pointing_events, tracks, duration_ms, response_window_ms, risk_bucket,
            response_outcomes=response_outcomes
        )
    elif task_type == "imitation":
        # Imitation: use imitation events (parent demos + child responses)
        # Pass imitation_outcomes for consistency with ML feature extraction
        flagged_moments = _build_imitation_flagged_moments(
            video_id, imitation_events, tracks, duration_ms, response_window_ms, risk_bucket,
            imitation_outcomes=imitation_outcomes
        )
    elif task_type == "free_play":
        # Free play: only show events for behaviors flagged in dashboard/assessment
        # Pass free_play_outcomes for consistency with ML feature extraction
        flagged_moments = _build_free_play_flagged_moments(
            video_id, behavioral_events, tracks, duration_ms, flagged_behavior_types, risk_bucket,
            free_play_outcomes=free_play_outcomes,
            limited_social_engagement=has_limited_social_engagement,
        )

    logger.info(f"  Built {len(flagged_moments)} flagged moments for {task_type}")

    # Compute overall tracking quality
    if tracks is not None and len(tracks.get("t_sec", [])) > 0:
        overall_quality = compute_tracking_quality(tracks, 0, duration_ms / 1000)
        # Calculate out of view percentage
        child_bbox = tracks.get("child_bbox", np.array([]))
        if len(child_bbox) > 0:
            bbox_valid = np.isfinite(child_bbox[:, 0])
            conf = child_bbox[:, 4] if child_bbox.shape[1] > 4 else np.ones(len(child_bbox))
            out_of_view = float((~bbox_valid | (conf <= 0.2)).sum() / len(child_bbox) * 100)
        else:
            out_of_view = 50.0
    else:
        overall_quality = TrackingQuality(quality="low", quality_pct=0, face_visible=False)
        out_of_view = 100.0

    # Build key drivers (notable issues) - pass ML risk bucket to modulate severity
    key_drivers = build_key_drivers(flagged_moments, behavioral_events, ml_risk_bucket=risk_bucket)

    # Build summary based on task type
    # Pass all outcomes (not just flagged) so summary includes successful responses
    # Pass ML risk bucket to modulate language severity
    summary = build_summary(
        flagged_moments, behavioral_events, task_type, key_drivers,
        response_outcomes=response_outcomes,
        imitation_outcomes=imitation_outcomes,
        free_play_outcomes=free_play_outcomes,
        ml_risk_bucket=risk_bucket,
        limited_social_engagement=has_limited_social_engagement,
    )

    # Task type display
    task_display_map = {
        "joint_attention": "Joint Attention",
        "imitation": "Imitation",
        "free_play": "Free Play",
    }

    # Get explanations for THIS task type only (for correct display)
    task_explanations = categorized_explanations.get(task_type, [])

    return {
        "videoId": video_id,
        "videoUrl": video_url,
        "durationMs": duration_ms,
        "fps": fps,
        "taskType": task_type,
        "taskTypeDisplay": task_display_map.get(task_type, task_type.replace("_", " ").title()),
        "ageBucket": age_bucket,
        "riskBucket": risk_bucket,
        "flaggedMoments": flagged_moments,
        "flaggedObservations": task_explanations,  # Only explanations for THIS task
        "allExplanations": categorized_explanations,  # Full categorized explanations
        "quality": {
            "overallQuality": overall_quality.quality,
            "trackingConfidenceAvg": overall_quality.quality_pct / 100.0,
            "faceVisibilityPct": overall_quality.quality_pct,
            "outOfViewPct": out_of_view,
        },
        "summary": summary,
        "keyDrivers": key_drivers,
    }


def build_summary(
    flagged_moments: List[Dict[str, Any]],
    behavioral_events: List[BehavioralEvent],
    task_type: str,
    key_drivers: List[Dict[str, Any]],
    response_outcomes: Optional[List[ResponseOutcome]] = None,
    imitation_outcomes: Optional[List[ImitationOutcome]] = None,
    free_play_outcomes: Optional[List[FreePlayOutcome]] = None,
    ml_risk_bucket: str = "moderate",
    limited_social_engagement: bool = False,
) -> Dict[str, Any]:
    """Build task-specific summary from outcomes and events.

    Uses the full outcomes (including successful responses) to calculate
    accurate totals and rates, not just flagged moments.

    The ML risk bucket modulates the severity of language - when overall
    risk is low, individual task observations are presented more neutrally.

    When saved outcomes are incomplete (e.g., missing pointing events),
    we merge counts from both saved outcomes AND flagged moments.
    """
    # Count outcomes based on task type
    outcomes: Dict[str, int] = {"observed": 0, "delayed": 0, "not_observed": 0, "uncertain": 0, "partial": 0}
    total = 0

    if task_type == "joint_attention":
        # Use saved outcomes for accurate counts (flagged moments now match saved outcomes)
        if response_outcomes:
            for outcome in response_outcomes:
                outcomes[outcome.status] = outcomes.get(outcome.status, 0) + 1
            total = len(response_outcomes)
        else:
            # Fallback when no saved outcomes: count from flagged moments
            for fm in flagged_moments:
                status = fm.get("observed", {}).get("status", "uncertain")
                outcomes[status] = outcomes.get(status, 0) + 1
            total = len(flagged_moments)

    elif task_type == "imitation" and imitation_outcomes:
        # Use actual imitation outcomes for accurate counts
        for outcome in imitation_outcomes:
            outcomes[outcome.status] = outcomes.get(outcome.status, 0) + 1
        total = len(imitation_outcomes)

    elif task_type == "free_play":
        # For free play, count from flagged moments (behavioral events)
        for fm in flagged_moments:
            status = fm.get("observed", {}).get("status", "flagged")
            # Map "flagged" to a reasonable category
            if status == "flagged":
                outcomes["not_observed"] = outcomes.get("not_observed", 0) + 1
            else:
                outcomes[status] = outcomes.get(status, 0) + 1
        total = len(flagged_moments)

    else:
        # Fallback: count from flagged moments only
        for fm in flagged_moments:
            status = fm.get("observed", {}).get("status", "uncertain")
            outcomes[status] = outcomes.get(status, 0) + 1
        total = len(flagged_moments)

    # Calculate risk score based on outcomes
    # Note: Delayed responses still show engagement - weight them lower than not_observed
    # Small sample sizes should not produce high risk scores
    if total > 0:
        raw_score = (
            outcomes["not_observed"] * 0.3 +  # Reduced from 0.4
            outcomes["delayed"] * 0.1 +        # Reduced from 0.2 - delayed still shows engagement
            outcomes["uncertain"] * 0.05       # Reduced - uncertain is just missing data
        ) / total

        # Apply sample size dampening - small samples shouldn't produce extreme scores
        # With 3 trials, multiply by 0.7; with 10+ trials, no dampening
        sample_factor = min(1.0, 0.5 + total * 0.05)
        risk_score = raw_score * sample_factor
    else:
        risk_score = 0.0

    # Risk bucket - raised thresholds to require more evidence
    if risk_score >= 0.5:
        risk_bucket = "elevated"  # Changed from "high" - less alarming
    elif risk_score >= 0.3:
        risk_bucket = "moderate"  # Raised threshold
    elif risk_score >= 0.15:
        risk_bucket = "low-moderate"
    else:
        risk_bucket = "low"

    # Cap per-task risk bucket when overall ML risk is low
    # Individual tasks shouldn't show alarming levels when overall assessment is low
    if ml_risk_bucket in ("low", "low-moderate"):
        # Cap at low-moderate when ML says overall risk is low
        if risk_bucket in ("elevated", "moderate"):
            risk_bucket = "low-moderate"

    summary: Dict[str, Any] = {
        "riskScore": risk_score,
        "riskConfidenceBand": 0.1,
        "riskBucket": risk_bucket,
        "keyDrivers": key_drivers,
        "totalPrompts": total,
        "totalFlags": len(flagged_moments),
    }

    if task_type == "joint_attention":
        summary["jointAttention"] = {
            "total": total,
            "observed": outcomes["observed"],
            "delayed": outcomes["delayed"],
            "notObserved": outcomes["not_observed"],
            "uncertain": outcomes["uncertain"],
            "clinicalNote": _generate_clinical_note(outcomes, "joint attention", ml_risk_bucket),
        }
    elif task_type == "imitation":
        summary["imitation"] = {
            "total": total,
            "full": outcomes["observed"],
            "partial": outcomes["partial"] + outcomes["delayed"],
            "none": outcomes["not_observed"],
            "rate": outcomes["observed"] / max(total, 1),
            "rateInclusive": (outcomes["observed"] + outcomes["partial"] + outcomes["delayed"]) / max(total, 1),
            "clinicalNote": _generate_clinical_note(outcomes, "imitation", ml_risk_bucket),
        }

    if task_type == "free_play":
        social_looks = sum(1 for e in behavioral_events if "LOOK" in e.event_type.upper())
        hand_events = sum(1 for e in behavioral_events if "HAND" in e.event_type.upper())
        periodic_events = [e for e in behavioral_events if "PERIODIC" in e.event_type.upper()]

        # Count flagged vs normal from outcomes
        flagged_count = sum(1 for o in (free_play_outcomes or []) if o.flagged)

        summary["freePlay"] = {
            "socialLooks": social_looks,
            "spontaneousPoints": 0,
            "toyTransitions": hand_events,
            "repetitiveEpisodes": len(periodic_events),
            "flaggedEvents": flagged_count,
            "clinicalNote": _generate_freeplay_note(behavioral_events, ml_risk_bucket, free_play_outcomes, limited_social_engagement),
        }

    return summary


def _generate_clinical_note(outcomes: Dict[str, int], task_name: str, ml_risk_bucket: str = "moderate") -> str:
    """Generate a clinical note based on outcomes.

    Uses balanced language that describes observations without being alarmist.
    Individual task observations are just one factor in overall assessment.

    When overall ML risk is low, language is softer to avoid alarm.
    """
    total = sum(outcomes.values())
    if total == 0:
        return f"No {task_name} trials detected"

    if outcomes["uncertain"] == total:
        return f"{task_name.title()} prompts detected; manual review recommended"

    observed_rate = outcomes["observed"] / total
    delayed_rate = outcomes.get("delayed", 0) / total

    # Soften language when overall risk is low
    is_low_risk = ml_risk_bucket in ("low", "low-moderate")

    if observed_rate >= 0.8:
        return f"Consistent {task_name} responses observed"
    elif observed_rate >= 0.5:
        return f"Variable {task_name} responses; some trials showed delays"
    elif observed_rate + delayed_rate >= 0.5:
        # If most responses happened but were delayed, softer language
        return f"Some {task_name} responses observed with variable timing"
    elif observed_rate >= 0.2:
        if is_low_risk:
            return f"Some {task_name} responses noted"
        return f"Some {task_name} responses observed; consider context factors"
    elif total <= 3:
        # Small sample size - be cautious about conclusions
        if is_low_risk:
            return f"Variable {task_name} responses in limited trials"
        return f"Few {task_name} responses in limited trials"
    else:
        if is_low_risk:
            return f"Variable {task_name} response pattern noted"
        return f"Few {task_name} responses detected; one factor among many"


def _generate_freeplay_note(
    events: List[BehavioralEvent],
    ml_risk_bucket: str = "moderate",
    free_play_outcomes: Optional[List[FreePlayOutcome]] = None,
    limited_social_engagement: bool = False,
) -> str:
    """Generate clinical note for free play observations.

    Only mentions behaviors that were actually flagged (met thresholds).
    Uses balanced language - repetitive behaviors are one data point among many.
    When overall ML risk is low, language is softer.
    """
    is_low_risk = ml_risk_bucket in ("low", "low-moderate")

    if not events:
        if limited_social_engagement:
            return "Limited social engagement during free play"
        return "No notable behavioral patterns detected during free play"

    # Only mention behaviors that were actually flagged (met thresholds)
    flagged_types: set = set()
    if free_play_outcomes:
        for outcome in free_play_outcomes:
            if outcome.flagged:
                flagged_types.add(outcome.event_type.upper())

    # Check if any periodic/repetitive movements were flagged
    has_flagged_periodic = any("PERIODIC" in t for t in flagged_types)

    if has_flagged_periodic:
        periodic = [e for e in events if "PERIODIC" in e.event_type.upper()]
        count = len(periodic)
        if is_low_risk:
            if count == 1:
                return "Brief repetitive movement observed"
            else:
                return f"Some repetitive movements observed during play"
        else:
            if count == 1:
                return "One instance of repetitive movement noted"
            else:
                return f"{count} instances of repetitive movement noted during play"

    # Check for flagged hand-to-face behaviors
    has_flagged_hand = any("HAND" in t for t in flagged_types)
    if has_flagged_hand:
        hand_events = [e for e in events if "HAND" in e.event_type.upper()]
        if len(hand_events) > 5:
            return "Frequent hand-to-face contact noted"
        return "Some hand-to-face contact observed"

    # Check for limited social engagement (detected by ML based on fp_engaged_time_frac < 0.3)
    if limited_social_engagement:
        if is_low_risk:
            return "Limited social engagement noted during play"
        return "Limited social engagement during free play"

    # No flagged behaviors - describe overall play pattern
    return "Typical free play behaviors observed"


def _get_driver_color(severity: str, ml_risk_bucket: str) -> str:
    """Get color for key driver text based on severity and ML risk bucket.

    Colors scale with overall ML risk level:
    - Low risk: Gray (informational)
    - Moderate risk: Blue (mild concern)
    - Moderate-high risk: Orange (warning)
    - High/elevated risk: Red (alarm)
    """
    # Color palettes by ML risk level
    if ml_risk_bucket in ("low", "low-moderate"):
        # Low overall risk: all issues shown in gray (informational)
        return "#6b7280"  # gray-500
    elif ml_risk_bucket == "moderate":
        # Moderate risk: blue for issues
        if severity == "medium":
            return "#2563eb"  # blue-600
        else:
            return "#6b7280"  # gray-500
    elif ml_risk_bucket in ("moderate-high", "elevated"):
        # Moderate-high risk: orange for issues
        if severity == "medium":
            return "#d97706"  # amber-600
        else:
            return "#92400e"  # amber-800
    else:  # high
        # High risk: red for issues
        if severity == "medium":
            return "#dc2626"  # red-600
        else:
            return "#991b1b"  # red-800


def build_key_drivers(
    flagged_moments: List[Dict[str, Any]],
    behavioral_events: List[BehavioralEvent],
    ml_risk_bucket: str = "moderate",
) -> List[Dict[str, Any]]:
    """Build list of key issues/drivers from the review.

    Key drivers are derived from the actual flagged observations, not generic categories.
    Labels are shortened versions of the observation descriptions.

    The ML risk bucket modulates severity - when overall risk is low, severity is reduced.
    """
    is_low_risk = ml_risk_bucket in ("low", "low-moderate")
    drivers: List[Dict[str, Any]] = []

    # Group flagged moments by their observed description to create meaningful drivers
    # This ensures drivers reflect actual observations, not generic labels
    observation_groups: Dict[str, List[Dict[str, Any]]] = {}

    for fm in flagged_moments:
        status = fm.get("observed", {}).get("status", "uncertain")
        observed_desc = fm.get("observed", {}).get("description", "")
        expected_type = fm.get("expected", {}).get("type", "")
        expected_desc = fm.get("expected", {}).get("description", "")

        # Skip observed/typical behaviors - only flag issues
        if status == "observed" and "flagged" not in str(fm.get("pauseCard", {}).get("status", "")):
            continue

        # Create a group key based on the actual observation
        if status == "not_observed":
            # Use expected description for no-response cases
            group_key = f"no_{expected_type}:{expected_desc}"
        elif status == "delayed":
            group_key = f"delayed:{expected_type}"
        elif status == "uncertain":
            group_key = "uncertain:tracking"
        elif status == "flagged" or "flagged" in str(fm.get("pauseCard", {}).get("status", "")):
            # For flagged behaviors (e.g., repetitive motion), use the observation
            group_key = f"flagged:{observed_desc}"
        else:
            continue

        if group_key not in observation_groups:
            observation_groups[group_key] = []
        observation_groups[group_key].append(fm)

    # Convert groups to drivers with descriptive labels
    for group_key, moments in observation_groups.items():
        parts = group_key.split(":", 1)
        category = parts[0]
        detail = parts[1] if len(parts) > 1 else ""

        # Generate concise label from the observation
        if category == "no_head_turn" or category == "no_gaze_follow":
            # Extract the prompt type for joint attention failures
            first_moment = moments[0]
            prompt_type = first_moment.get("pauseCard", {}).get("prompt", {}).get("type", "prompt")
            label = f"No response to {prompt_type.lower()}"
        elif category.startswith("no_"):
            # Generic no-response: shorten expected description
            label = _shorten_description(detail, prefix="No ")
        elif category == "delayed":
            label = f"Delayed {detail.replace('_', ' ')}" if detail else "Delayed response"
        elif category == "uncertain":
            label = "Low tracking quality"
        elif category == "flagged":
            # Flagged behavior - use shortened observation description
            label = _shorten_description(detail, max_words=4)
        else:
            label = _shorten_description(detail, max_words=4)

        # Determine severity based on count, category, and overall ML risk
        # When overall risk is low, keep all severities low
        if is_low_risk:
            # Low overall risk: all issues are informational, not alarming
            severity = "low"
        elif category.startswith("no_") or category == "flagged":
            # Require 3+ instances for "medium" severity
            if len(moments) >= 3:
                severity = "medium"
            else:
                severity = "low"
        elif category == "delayed":
            severity = "low"  # Delayed still shows engagement
        else:
            severity = "low"

        # Get color based on ML risk bucket (scales with overall risk)
        color = _get_driver_color(severity, ml_risk_bucket)

        # Create unique ID from label
        driver_id = label.lower().replace(" ", "_").replace("'", "")[:30]

        drivers.append({
            "id": driver_id,
            "label": label,
            "severity": severity,
            "color": color,
            "count": len(moments),
            "linkedFlagIds": [fm["id"] for fm in moments],
        })

    # Sort by severity (high first) then by count
    severity_order = {"high": 0, "medium": 1, "low": 2}
    drivers.sort(key=lambda d: (severity_order.get(d["severity"], 2), -d["count"]))

    return drivers


def _shorten_description(desc: str, max_words: int = 5, prefix: str = "") -> str:
    """Shorten a description to be concise for key driver labels."""
    if not desc:
        return "Issue detected"

    # Remove common filler words and parenthetical timing info
    import re
    desc = re.sub(r'\([^)]*\)', '', desc)  # Remove (2.1s) etc.
    desc = desc.strip()

    # Extract the key action/behavior
    words = desc.split()
    if len(words) <= max_words:
        return prefix + desc

    # Try to find the most meaningful part
    # Common patterns: "X detected", "X motion", "No X detected"
    short = " ".join(words[:max_words])

    return prefix + short


def compute_head_yaw(pose: np.ndarray) -> Optional[float]:
    """
    Compute head yaw angle from pose landmarks.

    Uses nose (landmark 0) and ears (landmarks 7, 8) to estimate head rotation.
    Returns angle in degrees: negative = looking left, positive = looking right.
    """
    if pose is None or len(pose) < 9:
        return None

    nose = pose[0]  # [x, y, z, visibility]
    left_ear = pose[7]
    right_ear = pose[8]

    # Check visibility
    if nose[3] < 0.3 or left_ear[3] < 0.3 or right_ear[3] < 0.3:
        return None

    # Compute ear midpoint
    ear_mid_x = (left_ear[0] + right_ear[0]) / 2

    # Yaw is estimated from nose offset from ear midpoint
    # Normalized coordinates, so multiply by a scale factor for degrees
    nose_offset = nose[0] - ear_mid_x

    # Also use ear-to-ear distance for normalization
    ear_distance = abs(left_ear[0] - right_ear[0])
    if ear_distance < 0.01:
        return None

    # Normalized offset -> approximate degrees (calibrated empirically)
    yaw_degrees = (nose_offset / ear_distance) * 45.0

    return float(yaw_degrees)


def get_overlay_data(
    video_id: int,
    visit_id: int,
    task_type: str,
    duration_ms: float,
) -> Dict[str, Any]:
    """
    Get frame-level overlay data for video annotation visualization.

    Args:
        video_id: Database video ID
        visit_id: Database visit ID
        task_type: "joint_attention", "imitation", or "free_play"
        duration_ms: Video duration in milliseconds

    Returns:
        Dict with fps, frames array, audio events, and expected windows
    """
    logger.info(f"Getting overlay data for video {video_id}, visit {visit_id}, task {task_type}")

    # Load tracks
    tracks = load_tracks(visit_id, task_type)

    # Load task-specific events
    # Audio events and pointing events are only relevant for joint_attention
    audio_events: List[AudioEvent] = []
    pointing_events: List[PointingEvent] = []
    imitation_events: List[ImitationEvent] = []

    if task_type == "joint_attention":
        audio_events = load_audio_events(visit_id)
        pointing_events = load_pointing_events(visit_id)
    elif task_type == "imitation":
        imitation_events = load_imitation_events(visit_id)

    fps = 30.0
    frames = []

    if tracks is not None:
        t_sec = tracks.get("t_sec", np.array([]))
        pose_data = tracks.get("pose", np.array([]))
        child_bbox = tracks.get("child_bbox", np.array([]))
        parent_bbox = tracks.get("parent_bbox", np.array([]))
        lh_data = tracks.get("lh", np.array([]))  # Left hand landmarks
        rh_data = tracks.get("rh", np.array([]))  # Right hand landmarks
        fps = tracks.get("fps", 30.0)

        n_frames = len(t_sec)
        logger.info(f"  Loaded {n_frames} frames, fps={fps}")

        for i in range(n_frames):
            frame: Dict[str, Any] = {
                "t_sec": float(t_sec[i]),
            }

            # Pose landmarks (33 x 4: x, y, z, visibility)
            if len(pose_data) > i:
                pose = pose_data[i]
                # Check if pose is valid (not all zeros/nan)
                if np.isfinite(pose).any() and pose.shape == (33, 4):
                    # Convert to list for JSON serialization
                    frame["pose"] = pose.tolist()
                    # Compute head yaw
                    frame["head_yaw"] = compute_head_yaw(pose)
                else:
                    frame["pose"] = None
                    frame["head_yaw"] = None
            else:
                frame["pose"] = None
                frame["head_yaw"] = None

            # Child bounding box [x, y, w, h, confidence]
            if len(child_bbox) > i:
                bbox = child_bbox[i]
                if np.isfinite(bbox).any() and len(bbox) >= 4:
                    frame["child_bbox"] = bbox.tolist()
                else:
                    frame["child_bbox"] = None
            else:
                frame["child_bbox"] = None

            # Parent bounding box [x, y, w, h, confidence]
            if len(parent_bbox) > i:
                pbbox = parent_bbox[i]
                if np.isfinite(pbbox).any() and len(pbbox) >= 4:
                    frame["parent_bbox"] = pbbox.tolist()
                else:
                    frame["parent_bbox"] = None
            else:
                frame["parent_bbox"] = None

            # Left hand landmarks (21 x 4: x, y, z, visibility)
            if len(lh_data) > i:
                lh = lh_data[i]
                if np.isfinite(lh).any() and lh.shape == (21, 4):
                    frame["left_hand"] = lh.tolist()
                else:
                    frame["left_hand"] = None
            else:
                frame["left_hand"] = None

            # Right hand landmarks (21 x 4: x, y, z, visibility)
            if len(rh_data) > i:
                rh = rh_data[i]
                if np.isfinite(rh).any() and rh.shape == (21, 4):
                    frame["right_hand"] = rh.tolist()
                else:
                    frame["right_hand"] = None
            else:
                frame["right_hand"] = None

            frames.append(frame)

    # Build task-specific events and expected windows
    events = []
    expected_windows = []
    response_window_ms = RESPONSE_WINDOW_MS.get(task_type, 3000)

    if task_type == "joint_attention":
        # Audio events (parent verbal prompts)
        for ae in audio_events:
            events.append({
                "type": ae.event_type,
                "t_start_sec": ae.t_start,
                "t_end_sec": ae.t_end,
                "confidence": ae.confidence,
                "matched_phrase": ae.matched_phrase,
            })
            expected_windows.append({
                "type": "orient_to_speaker",
                "trigger_type": ae.event_type,
                "trigger_phrase": ae.matched_phrase,
                "t_start_sec": ae.t_start,
                "t_end_sec": ae.t_start + response_window_ms / 1000,
                "expected_behavior": "Head turn toward speaker",
            })

        # Pointing events
        for pe in pointing_events:
            events.append({
                "type": "POINT",
                "t_start_sec": pe.t_start,
                "t_end_sec": pe.t_end,
                "confidence": pe.confidence,
                "matched_phrase": None,
            })
            expected_windows.append({
                "type": "gaze_follow",
                "trigger_type": "POINT",
                "trigger_phrase": None,
                "t_start_sec": pe.t_start,
                "t_end_sec": pe.t_end + response_window_ms / 1000,
                "expected_behavior": "Follow pointing gesture",
                "point_angle_deg": pe.point_angle_deg,
            })

    elif task_type == "imitation":
        # Imitation events (parent demos and child responses)
        for ie in imitation_events:
            events.append({
                "type": ie.event_type,
                "t_start_sec": ie.t_sec,
                "t_end_sec": ie.t_sec + 0.5,  # Approximate event duration
                "confidence": ie.confidence,
                "matched_phrase": ie.action_type,
                "subject": ie.subject,
            })
            # Only create expected windows for parent demos
            if ie.subject in ("parent", "adult") and "START" in ie.event_type.upper():
                expected_windows.append({
                    "type": "imitation",
                    "trigger_type": ie.event_type,
                    "trigger_phrase": ie.action_type,
                    "t_start_sec": ie.t_sec,
                    "t_end_sec": ie.t_sec + response_window_ms / 1000,
                    "expected_behavior": f"Child imitates {ie.action_type.lower().replace('_', ' ')}",
                })

    # free_play has no expected response windows - just pose/behavior tracking

    # Sort expected windows by start time
    expected_windows.sort(key=lambda w: w["t_start_sec"])

    logger.info(f"  Returning {len(frames)} frames, {len(events)} events, {len(expected_windows)} expected windows")

    return {
        "videoId": video_id,
        "fps": fps,
        "durationMs": duration_ms,
        "frames": frames,
        "events": events,
        "expectedWindows": expected_windows,
    }