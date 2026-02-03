"""
Tests for geometry utilities.
"""
import numpy as np
import pytest

import sys
sys.path.insert(0, str(__file__).replace("/tests/test_geometry.py", ""))

from src.utils.geometry import (
    bbox_iou,
    bbox_center_distance,
    bbox_centroid,
    bbox_valid,
    bbox_area,
    point_in_bbox,
)


class TestBboxIou:
    def test_perfect_overlap(self):
        """Same boxes should have IoU = 1."""
        bbox = np.array([0.0, 0.0, 1.0, 1.0])
        assert bbox_iou(bbox, bbox) == 1.0

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0."""
        bbox1 = np.array([0.0, 0.0, 0.5, 0.5])
        bbox2 = np.array([0.6, 0.6, 1.0, 1.0])
        assert bbox_iou(bbox1, bbox2) == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        bbox1 = np.array([0.0, 0.0, 0.6, 0.6])
        bbox2 = np.array([0.4, 0.4, 1.0, 1.0])
        iou = bbox_iou(bbox1, bbox2)
        assert 0 < iou < 1

    def test_invalid_bbox(self):
        """Invalid bbox should return 0."""
        bbox1 = np.array([np.nan, 0.0, 1.0, 1.0])
        bbox2 = np.array([0.0, 0.0, 1.0, 1.0])
        assert bbox_iou(bbox1, bbox2) == 0.0

    def test_with_confidence(self):
        """Should work with 5-element bbox (conf ignored)."""
        bbox1 = np.array([0.0, 0.0, 1.0, 1.0, 0.9])
        bbox2 = np.array([0.0, 0.0, 1.0, 1.0, 0.8])
        assert bbox_iou(bbox1, bbox2) == 1.0


class TestBboxCenterDistance:
    def test_same_center(self):
        """Same center should have distance = 0."""
        bbox1 = np.array([0.0, 0.0, 1.0, 1.0])
        bbox2 = np.array([0.25, 0.25, 0.75, 0.75])
        assert bbox_center_distance(bbox1, bbox2) == 0.0

    def test_different_centers(self):
        """Different centers should have positive distance."""
        bbox1 = np.array([0.0, 0.0, 0.2, 0.2])  # center at (0.1, 0.1)
        bbox2 = np.array([0.8, 0.8, 1.0, 1.0])  # center at (0.9, 0.9)
        dist = bbox_center_distance(bbox1, bbox2)
        expected = np.sqrt((0.9 - 0.1) ** 2 + (0.9 - 0.1) ** 2)
        assert np.isclose(dist, expected)

    def test_invalid_bbox(self):
        """Invalid bbox should return inf."""
        bbox1 = np.array([np.nan, 0.0, 1.0, 1.0])
        bbox2 = np.array([0.0, 0.0, 1.0, 1.0])
        assert bbox_center_distance(bbox1, bbox2) == float("inf")


class TestBboxCentroid:
    def test_single_bbox(self):
        """Should compute correct centroid."""
        bbox = np.array([[0.0, 0.0, 1.0, 1.0]])
        result = bbox_centroid(bbox)
        assert result.shape == (1, 2)
        assert np.allclose(result[0], [0.5, 0.5])

    def test_multiple_bboxes(self):
        """Should compute centroids for multiple boxes."""
        bbox = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.2, 0.3, 0.8, 0.9],
        ])
        result = bbox_centroid(bbox)
        assert result.shape == (2, 2)
        assert np.allclose(result[0], [0.5, 0.5])
        assert np.allclose(result[1], [0.5, 0.6])


class TestBboxValid:
    def test_valid_bbox(self):
        """Valid bbox should return True."""
        bbox = np.array([[0.0, 0.0, 1.0, 1.0, 0.9]])
        result = bbox_valid(bbox, conf_thr=0.5)
        assert result[0] == True

    def test_low_confidence(self):
        """Low confidence should return False."""
        bbox = np.array([[0.0, 0.0, 1.0, 1.0, 0.3]])
        result = bbox_valid(bbox, conf_thr=0.5)
        assert result[0] == False

    def test_nan_coordinates(self):
        """NaN coordinates should return False."""
        bbox = np.array([[np.nan, 0.0, 1.0, 1.0, 0.9]])
        result = bbox_valid(bbox)
        assert result[0] == False

    def test_small_area(self):
        """Small area should return False."""
        bbox = np.array([[0.5, 0.5, 0.5001, 0.5001, 0.9]])
        result = bbox_valid(bbox, min_area=0.01)
        assert result[0] == False

    def test_no_confidence_column(self):
        """Should work without confidence column."""
        bbox = np.array([[0.0, 0.0, 1.0, 1.0]])
        result = bbox_valid(bbox)
        assert result[0] == True

    def test_none_input(self):
        """None input should return empty array."""
        result = bbox_valid(None)
        assert result.size == 0

    def test_empty_input(self):
        """Empty array should return empty result."""
        bbox = np.array([]).reshape(0, 5)
        result = bbox_valid(bbox)
        assert result.size == 0


class TestBboxArea:
    def test_unit_square(self):
        """Unit square should have area = 1."""
        bbox = np.array([[0.0, 0.0, 1.0, 1.0]])
        result = bbox_area(bbox)
        assert result[0] == 1.0

    def test_multiple_bboxes(self):
        """Should compute areas for multiple boxes."""
        bbox = np.array([
            [0.0, 0.0, 1.0, 1.0],  # area = 1
            [0.0, 0.0, 0.5, 0.5],  # area = 0.25
        ])
        result = bbox_area(bbox)
        assert np.allclose(result, [1.0, 0.25])


class TestPointInBbox:
    def test_point_inside(self):
        """Point inside bbox should return True."""
        bbox = np.array([0.0, 0.0, 1.0, 1.0])
        assert point_in_bbox(0.5, 0.5, bbox) == True

    def test_point_outside(self):
        """Point outside bbox should return False."""
        bbox = np.array([0.0, 0.0, 1.0, 1.0])
        assert point_in_bbox(1.5, 0.5, bbox) == False

    def test_point_on_edge(self):
        """Point on edge should return True."""
        bbox = np.array([0.0, 0.0, 1.0, 1.0])
        assert point_in_bbox(0.0, 0.5, bbox) == True
        assert point_in_bbox(1.0, 0.5, bbox) == True

    def test_nan_point(self):
        """NaN point should return False."""
        bbox = np.array([0.0, 0.0, 1.0, 1.0])
        assert point_in_bbox(np.nan, 0.5, bbox) == False

    def test_nan_bbox(self):
        """NaN bbox should return False."""
        bbox = np.array([np.nan, 0.0, 1.0, 1.0])
        assert point_in_bbox(0.5, 0.5, bbox) == False
