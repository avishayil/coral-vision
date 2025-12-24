"""Core functionality tests for type utilities."""
from __future__ import annotations

from coral_vision.core.types import BBox, Detection


class TestTypes:
    """Tests for type definitions."""

    def test_bbox_clamp(self):
        """Test bounding box clamping."""
        bbox = BBox(xmin=-10, ymin=-5, xmax=650, ymax=490)
        clamped = bbox.clamp(640, 480)

        assert clamped.xmin == 0
        assert clamped.ymin == 0
        assert clamped.xmax == 640
        assert clamped.ymax == 480

    def test_bbox_is_valid(self):
        """Test bounding box validation."""
        valid = BBox(xmin=10, ymin=10, xmax=100, ymax=100)
        assert valid.is_valid()

        invalid1 = BBox(xmin=100, ymin=10, xmax=10, ymax=100)
        assert not invalid1.is_valid()

        invalid2 = BBox(xmin=10, ymin=100, xmax=100, ymax=10)
        assert not invalid2.is_valid()

    def test_detection_creation(self):
        """Test detection object creation."""
        bbox = BBox(xmin=10, ymin=10, xmax=100, ymax=100)
        detection = Detection(bbox=bbox, score=0.95)

        assert detection.bbox == bbox
        assert detection.score == 0.95
