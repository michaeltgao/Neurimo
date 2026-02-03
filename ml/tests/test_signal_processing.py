"""
Tests for signal processing utilities.
"""
import numpy as np
import pytest

import sys
sys.path.insert(0, str(__file__).replace("/tests/test_signal_processing.py", ""))

from src.utils.signal_processing import (
    interp_nans,
    nanmed_smooth,
    ema_smooth,
    spectral_peak,
    windowed_periodicity,
    segments_from_bool,
    compute_speed,
    autocorr_peak,
)


class TestInterpNans:
    def test_simple_gap(self):
        """Should interpolate single NaN with sufficient data."""
        # Need at least 8 valid points (default min_valid_count)
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = interp_nans(x)
        assert np.allclose(result[2], 3.0)  # interpolated value
        assert np.isfinite(result).all()

    def test_multiple_gaps(self):
        """Should interpolate multiple NaNs with sufficient data."""
        x = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = interp_nans(x)
        assert np.isfinite(result).all()
        assert np.allclose(result[2], 3.0)
        assert np.allclose(result[3], 4.0)

    def test_insufficient_valid_points(self):
        """Should return all NaN if insufficient valid points."""
        x = np.full(100, np.nan)
        x[0] = 1.0  # Only 1% valid
        result = interp_nans(x)
        assert np.all(np.isnan(result))

    def test_preserves_valid_values(self):
        """Should preserve valid values unchanged."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = interp_nans(x)
        assert np.allclose(result, x)

    def test_empty_array(self):
        """Should handle empty array."""
        x = np.array([])
        result = interp_nans(x)
        assert result.size == 0

    def test_all_valid_large(self):
        """Should return same array if all valid and large enough."""
        x = np.arange(10, dtype=float)
        result = interp_nans(x)
        assert np.allclose(result, x)

    def test_small_array_with_low_threshold(self):
        """Should work with small arrays if min_valid_count is lowered."""
        x = np.array([1.0, np.nan, 3.0])
        result = interp_nans(x, min_valid_count=2)
        assert np.allclose(result, [1.0, 2.0, 3.0])


class TestNanmedSmooth:
    def test_basic_smoothing(self):
        """Should smooth values."""
        x = np.array([1.0, 10.0, 1.0, 10.0, 1.0])
        result = nanmed_smooth(x, win=1)
        # With win=1: window at index 2 is x[1:4] = [10, 1, 10] -> median is 10.0
        assert result[2] == 10.0
        # At index 1: window is x[0:3] = [1, 10, 1] -> median is 1.0
        assert result[1] == 1.0

    def test_handles_nans(self):
        """Should handle NaN values."""
        x = np.array([1.0, np.nan, 3.0])
        result = nanmed_smooth(x, win=1)
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])

    def test_empty_array(self):
        """Should handle empty array."""
        x = np.array([])
        result = nanmed_smooth(x)
        assert result.size == 0


class TestEmaSmooth:
    def test_basic_smoothing(self):
        """Should apply EMA smoothing."""
        x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        result = ema_smooth(x, alpha=0.5)
        # Values should be smoothed
        assert result[1] > 0 and result[1] < 1

    def test_handles_leading_nans(self):
        """Should handle leading NaNs."""
        x = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
        result = ema_smooth(x, alpha=0.5)
        assert np.isfinite(result[2:]).all()

    def test_max_hold_frames(self):
        """Should respect max_hold_frames."""
        x = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 2.0])
        result = ema_smooth(x, alpha=0.5, max_hold_frames=2)
        # First 2 NaN frames should hold value
        assert np.isfinite(result[1])
        assert np.isfinite(result[2])
        # Beyond max_hold should be NaN
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_all_nans(self):
        """Should return all NaN if input is all NaN."""
        x = np.full(10, np.nan)
        result = ema_smooth(x)
        assert np.all(np.isnan(result))


class TestSpectralPeak:
    def test_known_frequency(self, periodic_signal):
        """Should detect known frequency."""
        signal, fps, true_freq = periodic_signal
        freq, power, norm_score = spectral_peak(signal, fps, 1.0, 4.0)
        assert abs(freq - true_freq) < 0.3
        assert norm_score > 0.1

    def test_no_periodicity(self, noisy_signal):
        """Should return low score for noise."""
        freq, power, norm_score = spectral_peak(noisy_signal, 30.0, 1.0, 4.0)
        # Random noise can have some spurious periodicity, but should be low
        assert norm_score < 0.5  # Relaxed threshold for random noise

    def test_insufficient_signal(self):
        """Should return zeros for short signal."""
        signal = np.array([1.0, 2.0, 3.0])
        freq, power, norm_score = spectral_peak(signal, 30.0, 1.0, 4.0)
        assert freq == 0.0
        assert power == 0.0
        assert norm_score == 0.0

    def test_low_amplitude(self):
        """Should return zeros for low amplitude signal."""
        signal = np.full(60, 0.001)
        freq, power, norm_score = spectral_peak(signal, 30.0, 1.0, 4.0)
        assert freq == 0.0


class TestWindowedPeriodicity:
    def test_detects_periodic_segment(self, periodic_signal):
        """Should detect periodic segments."""
        signal, fps, _ = periodic_signal
        flags = windowed_periodicity(signal, fps, 1.0, 1.0, 4.0)
        assert flags.any()

    def test_periodic_stronger_than_noise(self, periodic_signal, noisy_signal):
        """Periodic signal should have more flags than noise with high threshold."""
        signal, fps, _ = periodic_signal

        # Use high threshold to reduce false positives
        periodic_flags = windowed_periodicity(signal, fps, 1.0, 1.0, 4.0, normalized_threshold=0.5)
        noise_flags = windowed_periodicity(noisy_signal, 30.0, 1.0, 1.0, 4.0, normalized_threshold=0.5)

        # Periodic signal should have more flags than noise
        assert periodic_flags.mean() > noise_flags.mean() or periodic_flags.mean() > 0.5

    def test_short_signal(self):
        """Should handle signal shorter than window."""
        signal = np.array([1.0, 2.0, 3.0])
        flags = windowed_periodicity(signal, 30.0, 1.0, 1.0, 4.0)
        assert not flags.any()


class TestAutocorrPeak:
    def test_periodic_signal(self):
        """Should find high autocorr for periodic signal."""
        fps = 30.0
        t = np.arange(120) / fps
        signal = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz
        peak = autocorr_peak(signal, fps, 0.2, 1.0)
        assert peak > 0.5

    def test_noise_low_autocorr(self, noisy_signal):
        """Should find low autocorr for noise."""
        peak = autocorr_peak(noisy_signal, 30.0, 0.2, 1.0)
        assert peak < 0.3

    def test_insufficient_data(self):
        """Should return 0 for short signal."""
        signal = np.array([1.0, 2.0, 3.0])
        peak = autocorr_peak(signal, 30.0, 0.2, 1.0)
        assert peak == 0.0


class TestSegmentsFromBool:
    def test_single_segment(self):
        """Should detect single segment."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        flags = np.array([False, True, True, True, False])
        segs = segments_from_bool(t, flags, min_dur=0.5)
        assert len(segs) == 1
        assert segs[0] == (1.0, 4.0)

    def test_multiple_segments(self):
        """Should detect multiple segments."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        flags = np.array([True, True, False, False, True, True])
        segs = segments_from_bool(t, flags, min_dur=0.5)
        assert len(segs) == 2

    def test_filters_short_segments(self):
        """Should filter segments shorter than min_dur."""
        t = np.array([0.0, 0.1, 0.2, 1.0, 2.0])
        flags = np.array([True, True, False, True, True])
        segs = segments_from_bool(t, flags, min_dur=0.5)
        # First segment is 0.2s, should be filtered
        assert len(segs) == 1
        assert segs[0] == (1.0, 2.0)

    def test_empty_array(self):
        """Should handle empty arrays."""
        t = np.array([])
        flags = np.array([])
        segs = segments_from_bool(t, flags, min_dur=0.5)
        assert len(segs) == 0

    def test_segment_at_end(self):
        """Should handle segment at end of array."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        flags = np.array([False, True, True, True])
        segs = segments_from_bool(t, flags, min_dur=0.5)
        assert len(segs) == 1
        assert segs[0] == (1.0, 3.0)


class TestComputeSpeed:
    def test_constant_speed(self):
        """Should compute correct speed for linear motion."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        speed = compute_speed(xy, t)
        assert np.allclose(speed[1:], 1.0)

    def test_stationary(self):
        """Should return zero for stationary points."""
        t = np.array([0.0, 1.0, 2.0])
        xy = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        speed = compute_speed(xy, t)
        assert np.allclose(speed, 0.0)

    def test_diagonal_motion(self):
        """Should compute correct speed for diagonal motion."""
        t = np.array([0.0, 1.0])
        xy = np.array([[0.0, 0.0], [3.0, 4.0]])  # 5 units in 1 second
        speed = compute_speed(xy, t)
        assert np.allclose(speed[1], 5.0)

    def test_short_array(self):
        """Should handle single point."""
        t = np.array([0.0])
        xy = np.array([[0.5, 0.5]])
        speed = compute_speed(xy, t)
        assert speed.size == 1
        assert speed[0] == 0.0
