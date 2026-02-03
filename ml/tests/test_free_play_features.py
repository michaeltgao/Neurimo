"""
Tests for free play feature extraction.
"""
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(__file__).replace("/tests/test_free_play_features.py", ""))

from src.features.free_play import extract_free_play_features


class TestExtractFreePlayFeatures:
    def test_basic_extraction(self, sample_free_play_summary_df):
        """Should extract features for valid child."""
        features = extract_free_play_features("test_001", sample_free_play_summary_df)

        assert "fp_adult_present_ratio" in features
        assert "fp_freeze_time_frac" in features
        assert "fp_hand_to_face_time_frac" in features
        assert "fp_repetitive_motion_time_frac" in features

        # Check values are in expected range
        assert 0 <= features["fp_adult_present_ratio"] <= 1
        assert 0 <= features["fp_freeze_time_frac"] <= 1

    def test_missing_child(self, sample_free_play_summary_df):
        """Should return NaN for missing child."""
        features = extract_free_play_features("nonexistent", sample_free_play_summary_df)

        assert np.isnan(features["fp_adult_present_ratio"])
        assert np.isnan(features["fp_freeze_time_frac"])

    def test_clamp_values(self, sample_free_play_summary_df):
        """Should clamp ratio values to [0, 1]."""
        # Create df with out-of-range values
        df = sample_free_play_summary_df.copy()
        df.loc[0, "adult_present_ratio"] = 1.5  # Should be clamped to 1.0
        df.loc[0, "freeze_time_frac"] = -0.1  # Should be clamped to 0.0

        features = extract_free_play_features("test_001", df)

        assert features["fp_adult_present_ratio"] == 1.0
        assert features["fp_freeze_time_frac"] == 0.0

    def test_preserves_nan(self, sample_free_play_summary_df):
        """Should preserve NaN values (not clamp them)."""
        df = sample_free_play_summary_df.copy()
        df.loc[0, "adult_present_ratio"] = np.nan

        features = extract_free_play_features("test_001", df)

        assert np.isnan(features["fp_adult_present_ratio"])

    def test_handles_empty_string(self, sample_free_play_summary_df):
        """Should handle empty string values."""
        df = sample_free_play_summary_df.copy()
        # Convert column to object dtype to allow empty string
        df["adult_present_ratio"] = df["adult_present_ratio"].astype(object)
        df.loc[0, "adult_present_ratio"] = ""

        features = extract_free_play_features("test_001", df)

        assert np.isnan(features["fp_adult_present_ratio"])

    def test_duplicate_rows(self, sample_free_play_summary_df):
        """Should select row with best pose_present_ratio."""
        df = sample_free_play_summary_df.copy()
        # Add duplicate with lower quality
        new_row = df.iloc[0].copy()
        new_row["pose_present_ratio"] = 0.5
        new_row["adult_present_ratio"] = 0.3  # Different value
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        features = extract_free_play_features("test_001", df)

        # Should pick row with pose_present_ratio=0.85, adult_present_ratio=0.70
        assert features["fp_adult_present_ratio"] == 0.70

    def test_all_features_present(self, sample_free_play_summary_df):
        """Should return all expected features."""
        features = extract_free_play_features("test_001", sample_free_play_summary_df)

        expected_keys = [
            "fp_adult_present_ratio",
            "fp_adult_hand_active_time_frac",
            "fp_adult_hand_mean_activity",
            "fp_freeze_time_frac",
            "fp_hand_to_face_time_frac",
            "fp_repetitive_motion_time_frac",
            "fp_engaged_with_adult_time_frac",
            "fp_disengaged_with_adult_time_frac",
            "fp_hands_near_torso_time_frac",
            "fp_repetitive_motion_freq_hz",
        ]

        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_non_ratio_features_not_clamped(self, sample_free_play_summary_df):
        """Non-ratio features should not be clamped."""
        features = extract_free_play_features("test_001", sample_free_play_summary_df)

        # freq_hz and mean_activity are not ratios, should keep original values
        assert features["fp_repetitive_motion_freq_hz"] == 2.5
        assert features["fp_adult_hand_mean_activity"] == 0.03

    def test_string_child_id(self, sample_free_play_summary_df):
        """Should handle string child_id matching."""
        # Add a row with integer-like child_id (as string)
        df = sample_free_play_summary_df.copy()
        df.loc[0, "child_id"] = "123"

        features = extract_free_play_features("123", df)

        assert not np.isnan(features["fp_adult_present_ratio"])

    def test_integer_child_id(self, sample_free_play_summary_df):
        """Should handle integer child_id input."""
        df = sample_free_play_summary_df.copy()
        df.loc[0, "child_id"] = "123"

        features = extract_free_play_features(123, df)

        # Should convert to string and match
        assert not np.isnan(features["fp_adult_present_ratio"])
