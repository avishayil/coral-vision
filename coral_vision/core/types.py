"""Data types and structures for face recognition pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BBox:
    """Bounding box with pixel coordinates.

    Attributes:
        xmin: Left edge x-coordinate.
        ymin: Top edge y-coordinate.
        xmax: Right edge x-coordinate.
        ymax: Bottom edge y-coordinate.
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def clamp(self, width: int, height: int) -> "BBox":
        """Clamp bounding box coordinates to image dimensions.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            New BBox with clamped coordinates.
        """
        return BBox(
            xmin=max(0, min(self.xmin, width)),
            ymin=max(0, min(self.ymin, height)),
            xmax=max(0, min(self.xmax, width)),
            ymax=max(0, min(self.ymax, height)),
        )

    def is_valid(self) -> bool:
        """Check if bounding box has positive area.

        Returns:
            True if box has positive width and height.
        """
        return self.xmax > self.xmin and self.ymax > self.ymin


@dataclass(frozen=True)
class Detection:
    """Face detection result.

    Attributes:
        bbox: Bounding box in pixel coordinates.
        score: Confidence score (0.0 to 1.0).
    """

    bbox: BBox
    score: float


@dataclass(frozen=True)
class Match:
    """Face recognition match result.

    Attributes:
        person_id: Unique identifier of matched person.
        name: Display name of matched person.
        distance: L2 distance (lower is better match).
    """

    person_id: str
    name: str
    distance: float


@dataclass(frozen=True)
class FaceResult:
    """Recognition result for a single detected face.

    Attributes:
        bbox: Face bounding box in pixel coordinates.
        score: Detection confidence score.
        matches: List of top-k matching persons sorted by distance.
        predicted: Best match if distance is below threshold, None otherwise.
    """

    bbox: BBox
    score: float
    matches: list[Match]
    predicted: Match | None


@dataclass(frozen=True)
class ImageResult:
    """Recognition results for all faces in an image.

    Attributes:
        image_path: Path to the processed image.
        faces: List of face detection and recognition results.
    """

    image_path: str
    faces: list[FaceResult]


JSONDict = dict[str, Any]
