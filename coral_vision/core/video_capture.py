"""Video capture management for camera access."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class VideoCapture:
    """Manages video capture from webcam or other video sources.

    Attributes:
        camera_index: Index of the camera device (0 for default).
        width: Desired frame width.
        height: Desired frame height.
    """

    camera_index: int = 0
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        """Initialize the video capture."""
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the video capture device.

        Raises:
            RuntimeError: If no camera can be opened.
        """
        if self._cap is not None and self._cap.isOpened():
            return

        # Try to open the specified camera index
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            # Try alternative camera indices
            for idx in range(1, 5):
                self._cap = cv2.VideoCapture(idx)
                if self._cap.isOpened():
                    self.camera_index = idx
                    break

        if not self._cap.isOpened():
            raise RuntimeError("No camera found. Please ensure a camera is connected.")

        # Set frame dimensions
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video capture.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret:
            return False, None

        return True, frame

    def is_opened(self) -> bool:
        """Check if the video capture is open.

        Returns:
            True if capture is open, False otherwise.
        """
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        """Release the video capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoCapture":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()
