"""
Signal processing utilities for Neurimo ML pipeline.

Provides common functions for:
- NaN interpolation and smoothing
- Spectral analysis (periodicity detection)
- Time series segmentation
- Speed/velocity computation

These functions are used across perception and feature extraction modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class InterpolationConfig:
    """Configuration for NaN interpolation."""
    min_valid_fraction: float = 0.10
    min_valid_count: int = 8


def interp_nans(
    x: np.ndarray,
    min_valid_frac: float = 0.10,
    min_valid_count: int = 8,
) -> np.ndarray:
    """
    Linear interpolation over NaN values.

    Args:
        x: Input 1D array
        min_valid_frac: Minimum fraction of valid points required
        min_valid_count: Minimum absolute count of valid points

    Returns:
        Array with NaNs interpolated, or all-NaN if insufficient valid points
    """
    if x.size == 0:
        return x
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if good.sum() < max(min_valid_count, int(min_valid_frac * x.size)):
        return np.full_like(x, np.nan, dtype=float)
    out = x.astype(float).copy()
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def nanmed_smooth(x: np.ndarray, win: int = 2) -> np.ndarray:
    """
    Apply median smoothing, handling NaNs.

    Args:
        x: Input 1D array
        win: Window size (half-width on each side)

    Returns:
        Smoothed array with same shape as input
    """
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=float)
    n = x.size
    for i in range(n):
        j0 = max(0, i - win)
        j1 = min(n, i + win + 1)
        out[i] = float(np.nanmedian(x[j0:j1]))
    return out


def ema_smooth(
    x: np.ndarray,
    alpha: float = 0.3,
    max_hold_frames: int = 2,
) -> np.ndarray:
    """
    Exponential moving average smoothing with NaN handling.

    Args:
        x: Input signal
        alpha: Smoothing factor (0=no smoothing, 1=no memory)
        max_hold_frames: Maximum consecutive NaN frames to hold value.
                        Beyond this, output NaN to avoid fake activity during long gaps.

    Returns:
        Smoothed array with same shape as input
    """
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return np.full_like(x, np.nan)

    first_valid = np.where(valid)[0][0]
    state = x[first_valid]
    gap_count = 0

    for i in range(x.size):
        if np.isfinite(x[i]):
            state = alpha * x[i] + (1 - alpha) * state
            out[i] = state
            gap_count = 0
        else:
            gap_count += 1
            if gap_count <= max_hold_frames:
                out[i] = state
            else:
                out[i] = np.nan
    return out


def spectral_peak(
    signal: np.ndarray,
    fps: float,
    freq_min: float,
    freq_max: float,
    min_amplitude: float = 0.005,
) -> Tuple[float, float, float]:
    """
    Find dominant frequency in signal within [freq_min, freq_max] Hz band.

    Uses FFT with Hanning window for spectral analysis.

    Args:
        signal: Input signal (must be 1D, finite values)
        fps: Sampling rate in Hz
        freq_min: Minimum frequency of interest
        freq_max: Maximum frequency of interest
        min_amplitude: Minimum signal amplitude (std) required

    Returns:
        (frequency_hz, raw_power, normalized_score)
        - frequency_hz: Dominant frequency in band
        - raw_power: Power at dominant frequency
        - normalized_score: peak_power / (variance * size), scale-invariant
        Returns (0, 0, 0) if insufficient signal.
    """
    if signal.size < 30 or not np.isfinite(signal).all():
        return 0.0, 0.0, 0.0

    variance = float(np.var(signal))
    amplitude = float(np.std(signal))
    if amplitude < min_amplitude:
        return 0.0, 0.0, 0.0

    s = signal - float(np.mean(signal))
    window = np.hanning(s.size)
    y = np.fft.rfft(s * window)
    freqs = np.fft.rfftfreq(s.size, d=1.0 / fps)
    power = np.abs(y) ** 2
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    k = int(np.argmax(power[mask]))
    peak_power = float(power[mask][k])
    peak_freq = float(freqs[mask][k])

    eps = 1e-8
    normalized_score = peak_power / (variance * signal.size + eps)

    return peak_freq, peak_power, normalized_score


def windowed_periodicity(
    signal: np.ndarray,
    fps: float,
    window_sec: float,
    freq_min: float,
    freq_max: float,
    normalized_threshold: float = 0.15,
) -> np.ndarray:
    """
    Detect periodicity in sliding windows using normalized score.

    Args:
        signal: Input signal
        fps: Frames per second
        window_sec: Window size in seconds
        freq_min, freq_max: Frequency band to search
        normalized_threshold: Threshold for normalized periodicity score (scale-invariant)

    Returns:
        Per-frame boolean flags indicating periodic segments.
    """
    n = signal.size
    win_frames = max(int(window_sec * fps), 30)
    flags = np.zeros(n, dtype=bool)

    if n < win_frames:
        return flags

    for i in range(n - win_frames + 1):
        chunk = signal[i : i + win_frames]
        if not np.isfinite(chunk).all():
            continue
        freq, _power, norm_score = spectral_peak(chunk, fps, freq_min, freq_max)
        if norm_score > normalized_threshold and freq > 0:
            center = i + win_frames // 2
            radius = win_frames // 4
            flags[max(0, center - radius) : min(n, center + radius)] = True

    return flags


def autocorr_peak(
    x: np.ndarray,
    fps: float,
    min_lag_sec: float,
    max_lag_sec: float,
) -> float:
    """
    Find maximum autocorrelation value in lag range [min_lag_sec, max_lag_sec].

    Args:
        x: Input 1D signal (should be finite)
        fps: Effective sampling rate
        min_lag_sec: Minimum lag in seconds
        max_lag_sec: Maximum lag in seconds

    Returns:
        Maximum autocorrelation value in range, or 0.0 if insufficient data
    """
    if x.size < 20 or not np.isfinite(x).all():
        return 0.0
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0

    min_lag = int(max(1, round(min_lag_sec * fps)))
    max_lag = int(max(min_lag + 1, round(max_lag_sec * fps)))
    max_lag = min(max_lag, x.size - 2)
    if max_lag <= min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        v = float(np.dot(x[:-lag], x[lag:]) / denom)
        if v > best:
            best = v
    return float(best)


def segments_from_bool(
    t: np.ndarray,
    flags: np.ndarray,
    min_dur: float,
) -> List[Tuple[float, float]]:
    """
    Convert boolean flag array to list of (start, end) time segments.

    Args:
        t: Time array (same length as flags)
        flags: Boolean flags indicating events
        min_dur: Minimum duration for a valid segment

    Returns:
        List of (start_time, end_time) tuples
    """
    if t.size == 0:
        return []
    segs: List[Tuple[float, float]] = []
    in_seg = False
    s0 = 0.0
    for ti, fi in zip(t, flags):
        if fi and not in_seg:
            in_seg = True
            s0 = float(ti)
        if (not fi) and in_seg:
            e0 = float(ti)
            if e0 - s0 >= min_dur:
                segs.append((s0, e0))
            in_seg = False
    if in_seg:
        e0 = float(t[-1])
        if e0 - s0 >= min_dur:
            segs.append((s0, e0))
    return segs


def compute_speed(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute per-frame speed from (N, 2) xy coordinates.

    Args:
        xy: (N, 2) array of x, y coordinates
        t: (N,) array of timestamps

    Returns:
        (N,) array with speed values. speed[0] = speed[1] (padded).
    """
    if xy.shape[0] < 2:
        return np.zeros(max(1, xy.shape[0]), dtype=float)

    dt = np.diff(t)
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.sqrt(dx * dx + dy * dy) / np.maximum(dt, 1e-6)
    speed = np.concatenate([[speed[0] if speed.size else 0.0], speed])
    return speed.astype(float)
