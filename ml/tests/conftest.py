"""
Pytest fixtures for Neurimo ML tests.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_timestamps():
    """Generate 2-second sample at 30 fps."""
    return np.arange(60) / 30.0


@pytest.fixture
def sample_pose(sample_timestamps):
    """
    Generate synthetic pose data.

    Shape: (N, 33, 4) with [x, y, z, visibility]
    All landmarks visible with some movement.
    """
    n = len(sample_timestamps)
    pose = np.zeros((n, 33, 4), dtype=np.float32)

    # Set visibility to high for all landmarks
    pose[:, :, 3] = 0.9

    # Add some movement patterns
    t = sample_timestamps
    for i in range(33):
        pose[:, i, 0] = 0.5 + 0.1 * np.sin(2 * np.pi * t + i * 0.1)  # x
        pose[:, i, 1] = 0.5 + 0.05 * np.cos(2 * np.pi * t + i * 0.1)  # y
        pose[:, i, 2] = 0.0  # z

    return pose


@pytest.fixture
def sample_hands(sample_timestamps):
    """
    Generate synthetic hand data.

    Returns (lh, rh) each with shape (N, 21, 4)
    """
    n = len(sample_timestamps)
    lh = np.zeros((n, 21, 4), dtype=np.float32)
    rh = np.zeros((n, 21, 4), dtype=np.float32)

    lh[:, :, 3] = 0.9
    rh[:, :, 3] = 0.9

    t = sample_timestamps
    for i in range(21):
        lh[:, i, 0] = 0.3 + 0.05 * np.sin(3 * np.pi * t)
        lh[:, i, 1] = 0.6 + 0.03 * np.cos(3 * np.pi * t)
        rh[:, i, 0] = 0.7 + 0.05 * np.sin(3 * np.pi * t + np.pi)
        rh[:, i, 1] = 0.6 + 0.03 * np.cos(3 * np.pi * t + np.pi)

    return lh, rh


@pytest.fixture
def sample_child_bbox(sample_timestamps):
    """Generate synthetic child bbox data."""
    n = len(sample_timestamps)
    bbox = np.zeros((n, 5), dtype=np.float32)
    bbox[:, 0] = 0.2  # x0
    bbox[:, 1] = 0.1  # y0
    bbox[:, 2] = 0.8  # x1
    bbox[:, 3] = 0.9  # y1
    bbox[:, 4] = 0.95  # confidence
    return bbox


@pytest.fixture
def sample_parent_bbox(sample_timestamps):
    """Generate synthetic parent bbox data."""
    n = len(sample_timestamps)
    bbox = np.zeros((n, 5), dtype=np.float32)
    bbox[:, 0] = 0.6  # x0
    bbox[:, 1] = 0.0  # y0
    bbox[:, 2] = 1.0  # x1
    bbox[:, 3] = 0.5  # y1
    bbox[:, 4] = 0.90  # confidence
    return bbox


@pytest.fixture
def periodic_signal():
    """Generate a signal with known periodicity at 2.5 Hz."""
    fps = 30.0
    duration = 4.0
    t = np.arange(int(fps * duration)) / fps
    freq = 2.5
    signal = np.sin(2 * np.pi * freq * t)
    return signal, fps, freq


@pytest.fixture
def noisy_signal():
    """Generate a noisy signal without clear periodicity."""
    np.random.seed(42)
    return np.random.randn(120)


@pytest.fixture
def sample_free_play_summary_df():
    """Generate sample free_play_summary.csv data."""
    return pd.DataFrame({
        "child_id": ["test_001", "test_002"],
        "task_type": ["free_play", "free_play"],
        "duration_sec": [120.0, 90.0],
        "pose_present_ratio": [0.85, 0.72],
        "adult_present_ratio": [0.70, 0.60],
        "adult_hand_active_time_frac": [0.15, 0.20],
        "adult_hand_mean_activity": [0.03, 0.04],
        "freeze_time_frac": [0.10, 0.15],
        "hand_to_face_time_frac": [0.05, 0.08],
        "repetitive_motion_time_frac": [0.02, 0.05],
        "engaged_with_adult_time_frac": [0.40, 0.35],
        "disengaged_with_adult_time_frac": [0.20, 0.25],
        "hands_near_torso_time_frac": [0.12, 0.10],
        "repetitive_motion_freq_hz": [2.5, 3.0],
    })


@pytest.fixture
def sample_free_play_events_df():
    """Generate sample free_play_events.csv data."""
    return pd.DataFrame({
        "child_id": ["test_001"] * 5,
        "task_type": ["free_play"] * 5,
        "event_type": ["ACTIVITY_HIGH", "PERIODIC_MOTION", "HAND_TO_FACE", "CLOSE_PROXIMITY", "PARENT_PRESENT"],
        "t_start": [0.5, 10.0, 25.0, 40.0, 0.0],
        "t_end": [2.0, 12.5, 26.5, 55.0, 120.0],
        "confidence": [0.75, 0.60, 0.80, 0.75, 0.85],
        "meta": ["level=HIGH", "type=arm_flap", "dist<0.12", "bbox_overlap", "parent_bbox_detected"],
    })
