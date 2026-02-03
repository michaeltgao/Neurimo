# ml/src/perception/mediapipe_runner.py
"""
MediaPipe Runner — video-based landmark detection with standardized output.

This module provides:
  - MediaPipe 33-landmark pose output with named accessors
  - Optional face landmarks for head orientation
  - Pose quality metrics for downstream filtering:
    - Gaze proxy quality: % frames with nose + ears
    - Physical contact quality: % frames with wrists
  - Raw MediaPipe results for backward compatibility

Coordinate System:
  - All coordinates normalized [0, 1] relative to frame dimensions
  - Origin (0, 0) is top-left corner
  - x increases rightward, y increases downward

MediaPipe 33 Pose Landmark Indices:
    0: nose
    1: left_eye_inner, 2: left_eye, 3: left_eye_outer
    4: right_eye_inner, 5: right_eye, 6: right_eye_outer
    7: left_ear, 8: right_ear
    9: mouth_left, 10: mouth_right
    11: left_shoulder, 12: right_shoulder
    13: left_elbow, 14: right_elbow
    15: left_wrist, 16: right_wrist
    17: left_pinky, 18: right_pinky
    19: left_index, 20: right_index
    21: left_thumb, 22: right_thumb
    23: left_hip, 24: right_hip
    25: left_knee, 26: right_knee
    27: left_ankle, 28: right_ankle
    29: left_heel, 30: right_heel
    31: left_foot_index, 32: right_foot_index
"""
from __future__ import annotations

import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import cv2
import mediapipe as mp  # type: ignore
import numpy as np
from mediapipe.tasks import python as mp_tasks  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore


# ============================================================
# Model URLs and cache
# ============================================================
MODEL_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

MODEL_FILENAMES = {
    "pose": "pose_landmarker_lite.task",
    "hand": "hand_landmarker.task",
    "face": "face_landmarker.task",
}

MODEL_CACHE_DIR = Path.home() / ".cache" / "mediapipe_models"


def _download_model(url: str, filename: str) -> Path:
    """Download model file if not cached, return path."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_CACHE_DIR / filename
    if not model_path.exists():
        print(f"Downloading {filename}...")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ssl_context) as response:
            model_path.write_bytes(response.read())
        print(f"Downloaded to {model_path}")
    return model_path


def get_model_path(model_type: str) -> Path:
    """Get path to a specific model, downloading if needed."""
    if model_type not in MODEL_URLS:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: {list(MODEL_URLS.keys())}")
    return _download_model(MODEL_URLS[model_type], MODEL_FILENAMES[model_type])


# ============================================================
# MediaPipe 33 Landmark Index Constants
# ============================================================
class PoseIdx:
    """MediaPipe pose landmark indices for easy reference."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    # Useful groupings for quality metrics
    GAZE_PROXY = [NOSE, LEFT_EAR, RIGHT_EAR]  # needed for gaze direction
    WRISTS = [LEFT_WRIST, RIGHT_WRIST]  # needed for physical contact
    STABLE_TORSO = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]  # stable landmarks
    FACE = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]  # face landmarks from pose


# Face mesh key landmark indices (478 total in MediaPipe face mesh)
class FaceIdx:
    """Key face mesh landmark indices for head orientation."""
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_EAR_TRAGION = 234
    RIGHT_EAR_TRAGION = 454
    CHIN = 152
    FOREHEAD = 10
    LEFT_CHEEK = 50
    RIGHT_CHEEK = 280


# ============================================================
# Output Dataclasses
# ============================================================
@dataclass
class PoseLandmarks33:
    """MediaPipe 33-landmark pose with named accessors.

    Stores landmarks as (33, 4) array: [x, y, z, visibility] per landmark.

    Coordinate system:
        - x, y: Normalized [0, 1], origin top-left
        - z: Relative depth in MediaPipe units (NOT meters).
             Smaller z = closer to camera. Scale varies with distance.
             Use for relative depth comparisons only, not absolute measurements.
        - visibility: [0, 1] confidence, or NaN if unavailable
    """
    landmarks: np.ndarray  # (33, 4): [x, y, z, vis]

    def __post_init__(self):
        if self.landmarks.shape != (33, 4):
            raise ValueError(f"Expected (33, 4) array, got {self.landmarks.shape}")

    def _get(self, idx: int) -> np.ndarray:
        """Get landmark as [x, y, z, vis] array."""
        return self.landmarks[idx]

    def _xy(self, idx: int) -> np.ndarray:
        """Get landmark xy coordinates."""
        return self.landmarks[idx, :2]

    def _vis(self, idx: int) -> float:
        """Get landmark visibility."""
        return float(self.landmarks[idx, 3])

    # --- Core face landmarks (for gaze proxy) ---
    @property
    def nose(self) -> np.ndarray:
        return self._get(PoseIdx.NOSE)

    @property
    def nose_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.NOSE)

    @property
    def left_ear(self) -> np.ndarray:
        return self._get(PoseIdx.LEFT_EAR)

    @property
    def left_ear_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.LEFT_EAR)

    @property
    def right_ear(self) -> np.ndarray:
        return self._get(PoseIdx.RIGHT_EAR)

    @property
    def right_ear_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.RIGHT_EAR)

    @property
    def left_eye(self) -> np.ndarray:
        return self._get(PoseIdx.LEFT_EYE)

    @property
    def right_eye(self) -> np.ndarray:
        return self._get(PoseIdx.RIGHT_EYE)

    # --- Upper body landmarks ---
    @property
    def left_shoulder(self) -> np.ndarray:
        return self._get(PoseIdx.LEFT_SHOULDER)

    @property
    def left_shoulder_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.LEFT_SHOULDER)

    @property
    def right_shoulder(self) -> np.ndarray:
        return self._get(PoseIdx.RIGHT_SHOULDER)

    @property
    def right_shoulder_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.RIGHT_SHOULDER)

    @property
    def left_elbow(self) -> np.ndarray:
        return self._get(PoseIdx.LEFT_ELBOW)

    @property
    def right_elbow(self) -> np.ndarray:
        return self._get(PoseIdx.RIGHT_ELBOW)

    # --- Wrist landmarks (for physical contact) ---
    @property
    def left_wrist(self) -> np.ndarray:
        return self._get(PoseIdx.LEFT_WRIST)

    @property
    def left_wrist_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.LEFT_WRIST)

    @property
    def right_wrist(self) -> np.ndarray:
        return self._get(PoseIdx.RIGHT_WRIST)

    @property
    def right_wrist_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.RIGHT_WRIST)

    # --- Hip landmarks ---
    @property
    def left_hip(self) -> np.ndarray:
        return self._get(PoseIdx.LEFT_HIP)

    @property
    def left_hip_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.LEFT_HIP)

    @property
    def right_hip(self) -> np.ndarray:
        return self._get(PoseIdx.RIGHT_HIP)

    @property
    def right_hip_xy(self) -> np.ndarray:
        return self._xy(PoseIdx.RIGHT_HIP)

    # --- Lower body landmarks ---
    @property
    def left_knee(self) -> np.ndarray:
        return self._get(PoseIdx.LEFT_KNEE)

    @property
    def right_knee(self) -> np.ndarray:
        return self._get(PoseIdx.RIGHT_KNEE)

    @property
    def left_ankle(self) -> np.ndarray:
        return self._get(PoseIdx.LEFT_ANKLE)

    @property
    def right_ankle(self) -> np.ndarray:
        return self._get(PoseIdx.RIGHT_ANKLE)

    # --- Derived measurements ---
    @property
    def shoulder_midpoint(self) -> np.ndarray:
        """Midpoint between shoulders (2D)."""
        return (self.left_shoulder_xy + self.right_shoulder_xy) / 2

    @property
    def hip_midpoint(self) -> np.ndarray:
        """Midpoint between hips (2D)."""
        return (self.left_hip_xy + self.right_hip_xy) / 2

    @property
    def ear_midpoint(self) -> np.ndarray:
        """Midpoint between ears (2D)."""
        return (self.left_ear_xy + self.right_ear_xy) / 2

    def get_ear_axis(self, min_vis: float = 0.3, min_ear_dist: float = 0.01) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get ear-to-ear axis for head orientation.

        Args:
            min_vis: Minimum visibility threshold for ear landmarks
            min_ear_dist: Minimum distance between ears (reject bad detections)

        Returns:
            (midpoint, direction) where direction is unit vector left→right ear.
            None if ears not valid or geometry is bad.
        """
        if not (self.is_landmark_valid(PoseIdx.LEFT_EAR, min_vis) and
                self.is_landmark_valid(PoseIdx.RIGHT_EAR, min_vis)):
            return None

        left = self.left_ear_xy
        right = self.right_ear_xy

        direction = right - left
        norm = np.linalg.norm(direction)

        # Reject if ears too close (bad detection)
        if norm < min_ear_dist:
            return None

        midpoint = (left + right) / 2
        return midpoint, direction / norm

    # --- Validity checks ---
    def is_landmark_valid(self, idx: int, min_vis: float = 0.3) -> bool:
        """Check if a landmark is valid (finite coords + sufficient visibility).

        A landmark is valid if:
        - x, y are finite and in [0, 1]
        - visibility >= min_vis (or visibility is NaN but coords are valid)
        """
        lm = self.landmarks[idx]
        x, y, vis = lm[0], lm[1], lm[3]

        # Check coords are finite and in valid range
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return False

        # Check visibility (NaN visibility with valid coords = assume valid)
        if np.isfinite(vis):
            return vis >= min_vis
        return True  # NaN vis but valid coords

    # --- Quality checks ---
    def has_gaze_proxy(self, min_vis: float = 0.3) -> bool:
        """Check if gaze proxy landmarks are usable.

        Requires nose + at least one of:
        - Both ears visible
        - Both eyes visible (fallback)

        Also validates geometry: ear distance must be reasonable.
        """
        nose_ok = self.is_landmark_valid(PoseIdx.NOSE, min_vis)
        if not nose_ok:
            return False

        # Check ears
        left_ear_ok = self.is_landmark_valid(PoseIdx.LEFT_EAR, min_vis)
        right_ear_ok = self.is_landmark_valid(PoseIdx.RIGHT_EAR, min_vis)
        both_ears = left_ear_ok and right_ear_ok

        if both_ears:
            # Validate geometry: ears should be reasonable distance apart
            ear_dist = np.linalg.norm(self.left_ear_xy - self.right_ear_xy)
            if ear_dist < 0.01:  # too close = bad detection
                both_ears = False

        # Fallback: both eyes
        left_eye_ok = self.is_landmark_valid(PoseIdx.LEFT_EYE, min_vis)
        right_eye_ok = self.is_landmark_valid(PoseIdx.RIGHT_EYE, min_vis)
        both_eyes = left_eye_ok and right_eye_ok

        return both_ears or both_eyes

    def has_wrists(self, min_vis: float = 0.3) -> bool:
        """Check if at least one wrist is valid (needed for physical contact)."""
        return (
            self.is_landmark_valid(PoseIdx.LEFT_WRIST, min_vis)
            or self.is_landmark_valid(PoseIdx.RIGHT_WRIST, min_vis)
        )

    def has_both_wrists(self, min_vis: float = 0.3) -> bool:
        """Check if both wrists are valid."""
        return (
            self.is_landmark_valid(PoseIdx.LEFT_WRIST, min_vis)
            and self.is_landmark_valid(PoseIdx.RIGHT_WRIST, min_vis)
        )

    def mean_visibility(self, indices: Optional[List[int]] = None) -> float:
        """Mean visibility of specified landmarks (or all if None).

        NaN visibility values are excluded from the mean.
        Returns 0.0 if no valid visibility values.
        """
        if indices is None:
            vis = self.landmarks[:, 3]
        else:
            vis = np.array([self._vis(i) for i in indices])

        valid = np.isfinite(vis)
        if not valid.any():
            return 0.0
        return float(np.nanmean(vis))

    def to_array(self) -> np.ndarray:
        """Return raw (33, 4) landmark array."""
        return self.landmarks.copy()

    def to_xy_array(self) -> np.ndarray:
        """Return (33, 2) array of just xy coordinates."""
        return self.landmarks[:, :2].copy()

    def to_xyv_array(self) -> np.ndarray:
        """Return (33, 3) array [x, y, visibility] (no z)."""
        return self.landmarks[:, [0, 1, 3]].copy()


@dataclass
class FaceLandmarks:
    """Face mesh landmarks for head orientation estimation.

    Key landmarks useful for:
    - Head pose estimation (yaw, pitch, roll)
    - Gaze direction proxy
    - Face orientation relative to camera
    """
    landmarks: np.ndarray  # (478, 3) or subset: [x, y, z] per landmark

    # Key landmarks extracted for convenience
    nose_tip: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    chin: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    forehead: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    left_eye_inner: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    left_eye_outer: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    right_eye_inner: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    right_eye_outer: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))

    @property
    def is_valid(self) -> bool:
        """Check if core face landmarks are present."""
        return (
            np.isfinite(self.nose_tip[:2]).all()
            and np.isfinite(self.left_eye_inner[:2]).all()
            and np.isfinite(self.right_eye_inner[:2]).all()
        )

    @property
    def eye_midpoint(self) -> np.ndarray:
        """Midpoint between inner eye corners (2D)."""
        return (self.left_eye_inner[:2] + self.right_eye_inner[:2]) / 2

    def get_nose_to_eye_vector(self) -> Optional[np.ndarray]:
        """Get 2D vector from eye midpoint to nose tip.

        This is a crude proxy for face orientation in the image plane.
        NOT a true "forward" direction - heavily affected by pitch, yaw, camera angle.

        Use for: detecting head motion/turns, relative orientation changes.
        NOT for: absolute gaze direction, 3D head pose.

        Returns (2,) unit vector pointing from eye center toward nose, or None.
        """
        if not self.is_valid:
            return None
        eye_mid = self.eye_midpoint
        nose = self.nose_tip[:2]
        direction = nose - eye_mid
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        return direction / norm

    def get_face_orientation_proxy_2d(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get crude 2D face orientation from landmarks.

        Returns:
            (direction, confidence) where:
            - direction: (2,) unit vector (nose-to-eye-center)
            - confidence: 0-1 based on landmark validity
            None if insufficient landmarks.

        NOTE: This is for detecting relative head motion, not absolute pose.
        For true head pose, use PnP with 3D face model.
        """
        vec = self.get_nose_to_eye_vector()
        if vec is None:
            return None

        # Confidence based on how many key landmarks are valid
        valid_count = sum([
            np.isfinite(self.nose_tip[:2]).all(),
            np.isfinite(self.left_eye_inner[:2]).all(),
            np.isfinite(self.right_eye_inner[:2]).all(),
            np.isfinite(self.chin[:2]).all(),
        ])
        confidence = valid_count / 4.0

        return vec, confidence


@dataclass
class PoseQualityMetrics:
    """Quality metrics for pose detection over a video.

    Used to filter videos for specific analyses:
    - frac_gaze_proxy_valid: for gaze direction estimation (needs nose + ears)
    - frac_wrists_valid: for physical contact detection (needs wrists)
    """
    # Fraction of frames with nose + both ears visible
    frac_gaze_proxy_valid: float

    # Fraction of frames with at least one wrist visible
    frac_wrists_valid: float

    # Fraction of frames with both wrists visible
    frac_both_wrists_valid: float

    # Fraction of frames with any pose detected
    frac_pose_detected: float

    # Counts
    total_frames: int
    frames_gaze_proxy_valid: int
    frames_wrists_valid: int

    def meets_gaze_threshold(self, min_frac: float = 0.5) -> bool:
        """Check if video has sufficient gaze proxy coverage."""
        return self.frac_gaze_proxy_valid >= min_frac

    def meets_wrist_threshold(self, min_frac: float = 0.5) -> bool:
        """Check if video has sufficient wrist coverage."""
        return self.frac_wrists_valid >= min_frac

    def summary(self) -> str:
        """Return human-readable quality summary."""
        return (
            f"Pose detected: {self.frac_pose_detected:.1%} | "
            f"Gaze proxy (nose+ears): {self.frac_gaze_proxy_valid:.1%} | "
            f"Wrists: {self.frac_wrists_valid:.1%}"
        )


@dataclass
class FrameResult:
    """Detection results for a single frame.

    Primary outputs:
        - frame_ts: Timestamp in seconds
        - frame_idx: Original frame index in video
        - pose: PoseLandmarks33 with named accessors (or None)

    Optional outputs:
        - face: FaceLandmarks for head orientation (if face detector enabled)

    Raw outputs (for backward compatibility):
        - raw_pose: Raw MediaPipe PoseLandmarkerResult
        - raw_hands: Raw MediaPipe HandLandmarkerResult
        - raw_face: Raw MediaPipe FaceLandmarkerResult
    """
    frame_ts: float
    frame_idx: int

    # Standardized pose output
    pose: Optional[PoseLandmarks33] = None

    # Optional face landmarks
    face: Optional[FaceLandmarks] = None

    # Raw MediaPipe results (backward compatibility)
    raw_pose: Any = None
    raw_hands: Any = None
    raw_face: Any = None

    @property
    def has_pose(self) -> bool:
        return self.pose is not None

    @property
    def has_gaze_proxy(self) -> bool:
        """Check if gaze proxy landmarks (nose + ears) are present."""
        return self.pose is not None and self.pose.has_gaze_proxy()

    @property
    def has_wrists(self) -> bool:
        """Check if wrist landmarks are present."""
        return self.pose is not None and self.pose.has_wrists()

    @property
    def has_face(self) -> bool:
        return self.face is not None and self.face.is_valid


@dataclass
class VideoResult:
    """Aggregated results for an entire video.

    Contains:
        - frames: List of per-frame FrameResult
        - quality: PoseQualityMetrics for filtering
        - video_info: Metadata (fps, dimensions, duration)
    """
    frames: List[FrameResult]
    quality: PoseQualityMetrics
    video_info: dict

    def get_pose_array(self) -> np.ndarray:
        """Get all poses as (N, 33, 4) array. NaN for missing frames."""
        n = len(self.frames)
        arr = np.full((n, 33, 4), np.nan, dtype=np.float32)
        for i, frame in enumerate(self.frames):
            if frame.pose is not None:
                arr[i] = frame.pose.landmarks
        return arr

    def get_timestamps(self) -> np.ndarray:
        """Get timestamps as (N,) array."""
        return np.array([f.frame_ts for f in self.frames], dtype=np.float32)

    def get_frame_indices(self) -> np.ndarray:
        """Get original frame indices as (N,) array."""
        return np.array([f.frame_idx for f in self.frames], dtype=np.int32)

    def get_pose_quality_mask(self, min_vis: float = 0.3) -> np.ndarray:
        """Get boolean mask of frames with usable pose.

        Args:
            min_vis: Minimum visibility threshold

        Returns:
            (N,) bool array where True = frame has valid pose
        """
        return np.array([
            f.pose is not None and f.pose.mean_visibility() >= min_vis
            for f in self.frames
        ], dtype=bool)

    def get_gaze_proxy_mask(self, min_vis: float = 0.3) -> np.ndarray:
        """Get boolean mask of frames usable for gaze proxy analysis.

        Returns:
            (N,) bool array where True = frame has nose + ears/eyes
        """
        return np.array([
            f.pose is not None and f.pose.has_gaze_proxy(min_vis)
            for f in self.frames
        ], dtype=bool)

    def get_wrist_mask(self, min_vis: float = 0.3, both: bool = False) -> np.ndarray:
        """Get boolean mask of frames usable for wrist/contact analysis.

        Args:
            min_vis: Minimum visibility threshold
            both: If True, require both wrists; if False, at least one

        Returns:
            (N,) bool array
        """
        if both:
            return np.array([
                f.pose is not None and f.pose.has_both_wrists(min_vis)
                for f in self.frames
            ], dtype=bool)
        return np.array([
            f.pose is not None and f.pose.has_wrists(min_vis)
            for f in self.frames
        ], dtype=bool)


# ============================================================
# Conversion utilities
# ============================================================
def _mediapipe_pose_to_landmarks33(pose_landmarks: Any) -> Optional[PoseLandmarks33]:
    """Convert MediaPipe pose landmarks to PoseLandmarks33.

    Visibility handling:
    - If visibility attr exists and is a valid float, use it
    - If missing or None, set to NaN (not 1.0!) so quality metrics are honest
    - Coordinates outside [0,1] or NaN are left as-is for downstream filtering
    """
    if pose_landmarks is None or len(pose_landmarks) == 0:
        return None

    arr = np.full((33, 4), np.nan, dtype=np.float32)
    for i, lm in enumerate(pose_landmarks):
        if i >= 33:
            break
        x, y, z = float(lm.x), float(lm.y), float(lm.z)
        arr[i, 0] = x
        arr[i, 1] = y
        arr[i, 2] = z

        # Handle visibility: NaN if missing, not 1.0
        vis = getattr(lm, 'visibility', None)
        if vis is not None:
            arr[i, 3] = float(vis)
        else:
            # No visibility provided - use geometric validity as proxy
            # Mark as valid (1.0) only if coords are in expected range
            if 0 <= x <= 1 and 0 <= y <= 1 and np.isfinite(z):
                arr[i, 3] = 1.0
            else:
                arr[i, 3] = np.nan

    return PoseLandmarks33(landmarks=arr)


def _extract_face_landmarks(face_result: Any) -> Optional[FaceLandmarks]:
    """Extract FaceLandmarks from MediaPipe face result."""
    if face_result is None:
        return None

    face_landmarks_list = getattr(face_result, 'face_landmarks', None)
    if not face_landmarks_list or len(face_landmarks_list) == 0:
        return None

    # Use first detected face
    landmarks = face_landmarks_list[0]
    n_landmarks = len(landmarks)

    # Build full array
    arr = np.full((n_landmarks, 3), np.nan, dtype=np.float32)
    for i, lm in enumerate(landmarks):
        arr[i] = [float(lm.x), float(lm.y), float(lm.z)]

    def safe_get(idx: int) -> np.ndarray:
        if idx < n_landmarks:
            return arr[idx].copy()
        return np.full(3, np.nan, dtype=np.float32)

    return FaceLandmarks(
        landmarks=arr,
        nose_tip=safe_get(FaceIdx.NOSE_TIP),
        chin=safe_get(FaceIdx.CHIN),
        forehead=safe_get(FaceIdx.FOREHEAD),
        left_eye_inner=safe_get(FaceIdx.LEFT_EYE_INNER),
        left_eye_outer=safe_get(FaceIdx.LEFT_EYE_OUTER),
        right_eye_inner=safe_get(FaceIdx.RIGHT_EYE_INNER),
        right_eye_outer=safe_get(FaceIdx.RIGHT_EYE_OUTER),
    )


def _choose_best_pose(pose_result: Any) -> Optional[Any]:
    """Select best pose from multiple detections.

    Selection criteria (in order of importance):
    1. Proximity to frame center (child is usually centered)
    2. Bbox size as tiebreaker (child often larger/closer)

    NOTE: Adults often have higher visibility scores, so we don't use
    visibility as the primary criterion.
    """
    if not pose_result or not getattr(pose_result, 'pose_landmarks', None):
        return None

    poses = pose_result.pose_landmarks
    if len(poses) == 0:
        return None
    if len(poses) == 1:
        return poses[0]

    best_pose = None
    best_score = -float('inf')

    for lms in poses:
        # Compute centroid from stable landmarks (shoulders + hips)
        stable_indices = [11, 12, 23, 24]  # shoulders and hips
        xs, ys = [], []
        for idx in stable_indices:
            if idx < len(lms):
                lm = lms[idx]
                x, y = float(lm.x), float(lm.y)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    xs.append(x)
                    ys.append(y)

        if len(xs) < 2:
            # Fallback: use all landmarks
            for lm in lms:
                x, y = float(lm.x), float(lm.y)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    xs.append(x)
                    ys.append(y)

        if not xs:
            continue

        cx, cy = np.mean(xs), np.mean(ys)

        # Score: proximity to center (0.5, 0.5)
        # Lower distance = higher score
        dist_to_center = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        center_score = 1.0 - dist_to_center  # max ~1.0 at center

        # Tiebreaker: bbox size (larger = closer/more prominent)
        x_range = max(xs) - min(xs) if xs else 0
        y_range = max(ys) - min(ys) if ys else 0
        size_score = x_range * y_range * 0.1  # small weight

        score = center_score + size_score

        if score > best_score:
            best_score = score
            best_pose = lms

    return best_pose


def _compute_quality_metrics(frames: List[FrameResult], min_vis: float = 0.3) -> PoseQualityMetrics:
    """Compute quality metrics from frame results."""
    total = len(frames)
    if total == 0:
        return PoseQualityMetrics(
            frac_gaze_proxy_valid=0.0,
            frac_wrists_valid=0.0,
            frac_both_wrists_valid=0.0,
            frac_pose_detected=0.0,
            total_frames=0,
            frames_gaze_proxy_valid=0,
            frames_wrists_valid=0,
        )

    pose_detected = 0
    gaze_proxy_valid = 0
    wrists_valid = 0
    both_wrists_valid = 0

    for frame in frames:
        if frame.pose is not None:
            pose_detected += 1
            if frame.pose.has_gaze_proxy(min_vis):
                gaze_proxy_valid += 1
            if frame.pose.has_wrists(min_vis):
                wrists_valid += 1
            if frame.pose.has_both_wrists(min_vis):
                both_wrists_valid += 1

    return PoseQualityMetrics(
        frac_gaze_proxy_valid=gaze_proxy_valid / total,
        frac_wrists_valid=wrists_valid / total,
        frac_both_wrists_valid=both_wrists_valid / total,
        frac_pose_detected=pose_detected / total,
        total_frames=total,
        frames_gaze_proxy_valid=gaze_proxy_valid,
        frames_wrists_valid=wrists_valid,
    )


# ============================================================
# MediaPipeRunner
# ============================================================
class MediaPipeRunner:
    """Video-based pose and face detection with standardized output.

    Features:
        - MediaPipe 33-landmark pose with named accessors
        - Optional face landmarks for head orientation
        - Pose quality metrics for downstream filtering
        - Raw MediaPipe results for backward compatibility

    Usage:
        runner = MediaPipeRunner(enable_pose=True, enable_face=True)

        # Process entire video with quality metrics
        result = runner.process_video("video.mp4")
        print(result.quality.summary())

        # Check quality thresholds
        if result.quality.meets_gaze_threshold(0.5):
            # Use for gaze analysis
            ...

        # Or iterate frame by frame
        for frame in runner.iter_video("video.mp4"):
            if frame.has_gaze_proxy:
                ear_axis = frame.pose.get_ear_axis()
                ...

        runner.close()

    Context manager:
        with MediaPipeRunner(enable_pose=True) as runner:
            result = runner.process_video("video.mp4")
    """

    def __init__(
        self,
        enable_pose: bool = True,
        enable_hands: bool = False,
        enable_face: bool = False,
        num_poses: int = 2,
        num_hands: int = 4,
        num_faces: int = 1,
    ):
        """Initialize with specified detectors.

        Args:
            enable_pose: Enable pose landmark detection
            enable_hands: Enable hand landmark detection
            enable_face: Enable face landmark detection (for head orientation)
            num_poses: Max poses to detect (2 = child + parent)
            num_hands: Max hands to detect
            num_faces: Max faces to detect
        """
        self._pose: Any = None
        self._hands: Any = None
        self._face: Any = None

        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        self.enable_face = enable_face

        if enable_pose:
            pose_path = get_model_path("pose")
            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=num_poses,
                )
            )

        if enable_hands:
            hand_path = get_model_path("hand")
            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=num_hands,
                )
            )

        if enable_face:
            face_path = get_model_path("face")
            self._face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(face_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=num_faces,
                )
            )

    def __enter__(self) -> "MediaPipeRunner":
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()

    def iter_video(
        self,
        video_path: str,
        sample_every_n: int = 1,
        max_frames: Optional[int] = None,
    ) -> Iterator[FrameResult]:
        """Iterate over video frames with standardized output.

        Args:
            video_path: Path to video file
            sample_every_n: Process every Nth frame (1 = all frames)
            max_frames: Stop after this many processed frames

        Yields:
            FrameResult with pose, face, and raw results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 30.0

        frame_idx = 0
        frames_used = 0

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                if sample_every_n > 1 and (frame_idx % sample_every_n != 0):
                    frame_idx += 1
                    continue

                # Prefer actual timestamp from video container (robust to VFR)
                # Fallback to frame_idx/fps if unavailable
                t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if t_msec > 0:
                    t_sec = t_msec / 1000.0
                else:
                    t_sec = frame_idx / fps

                current_frame_idx = frame_idx
                frame_idx += 1
                frames_used += 1

                if max_frames is not None and frames_used > max_frames:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Run detectors
                pose_result = self._pose.detect(mp_image) if self._pose else None
                hand_result = self._hands.detect(mp_image) if self._hands else None
                face_result = self._face.detect(mp_image) if self._face else None

                # Convert to standardized format
                pose_landmarks = None
                if pose_result is not None:
                    best_pose = _choose_best_pose(pose_result)
                    if best_pose is not None:
                        pose_landmarks = _mediapipe_pose_to_landmarks33(best_pose)

                face_landmarks = None
                if face_result is not None:
                    face_landmarks = _extract_face_landmarks(face_result)

                yield FrameResult(
                    frame_ts=t_sec,
                    frame_idx=current_frame_idx,
                    pose=pose_landmarks,
                    face=face_landmarks,
                    raw_pose=pose_result,
                    raw_hands=hand_result,
                    raw_face=face_result,
                )
        finally:
            cap.release()

    def process_video(
        self,
        video_path: str,
        sample_every_n: int = 1,
        max_frames: Optional[int] = None,
        quality_vis_threshold: float = 0.3,
    ) -> VideoResult:
        """Process entire video and compute quality metrics.

        Args:
            video_path: Path to video file
            sample_every_n: Process every Nth frame
            max_frames: Maximum frames to process
            quality_vis_threshold: Visibility threshold for quality metrics

        Returns:
            VideoResult with frames, quality metrics, and video info
        """
        video_info = self.get_video_info(video_path)

        frames = list(self.iter_video(
            video_path,
            sample_every_n=sample_every_n,
            max_frames=max_frames,
        ))

        quality = _compute_quality_metrics(frames, min_vis=quality_vis_threshold)

        return VideoResult(
            frames=frames,
            quality=quality,
            video_info=video_info,
        )

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata without processing frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps) if fps and fps > 0 else 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = frame_count / fps if fps > 0 else 0.0

            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration_sec": duration_sec,
                "width": width,
                "height": height,
            }
        finally:
            cap.release()

    def close(self) -> None:
        """Release all detector resources."""
        if self._pose:
            self._pose.close()
            self._pose = None
        if self._hands:
            self._hands.close()
            self._hands = None
        if self._face:
            self._face.close()
            self._face = None
