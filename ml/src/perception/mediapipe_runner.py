# ml/src/perception/mediapipe_runner.py
"""
MediaPipe Runner — a thin, shared utility for video-based landmark detection.

This module IS:
  - A camera + sensors driver
  - Responsible for model lifecycle (download, cache, load)
  - Responsible for frame iteration with consistent sampling
  - Responsible for detector invocation and timestamp calculation
  - Returns raw results (landmarks + timestamps)

This module is NOT:
  - A model or feature extractor
  - Protocol-specific logic
  - A place to decide "this is clapping / pointing / flapping"
"""
from __future__ import annotations

import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import mediapipe as mp  # type: ignore
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
    """Get path to a specific model, downloading if needed.

    Args:
        model_type: One of "pose", "hand", "face"
    """
    if model_type not in MODEL_URLS:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: {list(MODEL_URLS.keys())}")
    return _download_model(MODEL_URLS[model_type], MODEL_FILENAMES[model_type])


# ============================================================
# Raw frame result — what the runner yields
# ============================================================
@dataclass
class FrameResult:
    """Raw detection results for a single frame.

    All fields are raw MediaPipe result objects (or None if detector not enabled).
    Consumers are responsible for interpreting these results.

    Result types:
        pose: vision.PoseLandmarkerResult
        hands: vision.HandLandmarkerResult
        face: vision.FaceLandmarkerResult
    """
    t_sec: float
    frame_idx: int
    pose: Any  # PoseLandmarkerResult or None
    hands: Any  # HandLandmarkerResult or None
    face: Any  # FaceLandmarkerResult or None


# ============================================================
# MediaPipeRunner
# ============================================================
class MediaPipeRunner:
    """Thin utility that loads MediaPipe models and iterates video frames.

    Usage:
        runner = MediaPipeRunner(
            enable_pose=True,
            enable_hands=True,
            enable_face=False,
            num_poses=2,
            num_hands=4,
        )

        for result in runner.iter_video("video.mp4", sample_every_n=2):
            # result.t_sec, result.pose, result.hands, result.face
            ...

        runner.close()

    Or as a context manager:
        with MediaPipeRunner(enable_pose=True) as runner:
            for result in runner.iter_video("video.mp4"):
                ...
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
        """Initialize the runner with specified detectors.

        Args:
            enable_pose: Whether to enable pose landmark detection
            enable_hands: Whether to enable hand landmark detection
            enable_face: Whether to enable face landmark detection
            num_poses: Max number of poses to detect per frame
            num_hands: Max number of hands to detect per frame
            num_faces: Max number of faces to detect per frame
        """
        self._pose: Any = None  # PoseLandmarker or None
        self._hands: Any = None  # HandLandmarker or None
        self._face: Any = None  # FaceLandmarker or None

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
        """Iterate over video frames, running enabled detectors.

        Args:
            video_path: Path to video file
            sample_every_n: Process every Nth frame (1 = all frames)
            max_frames: Stop after this many processed frames (None = no limit)

        Yields:
            FrameResult for each processed frame containing raw detection results
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

                # Sampling
                if sample_every_n > 1 and (frame_idx % sample_every_n != 0):
                    frame_idx += 1
                    continue

                t_sec = frame_idx / fps
                current_frame_idx = frame_idx
                frame_idx += 1
                frames_used += 1

                if max_frames is not None and frames_used > max_frames:
                    break

                # Convert BGR -> RGB and create MediaPipe image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Run detectors
                pose_result = self._pose.detect(mp_image) if self._pose else None
                hand_result = self._hands.detect(mp_image) if self._hands else None
                face_result = self._face.detect(mp_image) if self._face else None

                yield FrameResult(
                    t_sec=t_sec,
                    frame_idx=current_frame_idx,
                    pose=pose_result,
                    hands=hand_result,
                    face=face_result,
                )
        finally:
            cap.release()

    def get_video_info(self, video_path: str) -> dict:
        """Get basic video metadata without processing frames.

        Returns:
            Dict with keys: fps, frame_count, duration_sec, width, height
        """
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
