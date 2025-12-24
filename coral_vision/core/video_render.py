"""Video frame rendering utilities for drawing annotations."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class VideoRenderer:
    """Renders face recognition annotations on video frames."""

    def __init__(
        self,
        recognized_color: tuple[int, int, int] = (0, 255, 0),
        unknown_color: tuple[int, int, int] = (0, 0, 255),
        text_color: tuple[int, int, int] = (255, 255, 255),
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2,
    ) -> None:
        """Initialize the video renderer.

        Args:
            recognized_color: BGR color for recognized faces (default: green).
            unknown_color: BGR color for unknown faces (default: red).
            text_color: BGR color for text labels (default: white).
            box_thickness: Thickness of bounding box lines.
            font_scale: Scale factor for text font.
            font_thickness: Thickness of text font.
        """
        self.recognized_color = recognized_color
        self.unknown_color = unknown_color
        self.text_color = text_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_face(
        self,
        frame: np.ndarray,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        label: str,
        distance: Optional[float] = None,
        recognized: bool = False,
    ) -> None:
        """Draw a face bounding box and label on the frame.

        Args:
            frame: BGR frame to draw on (modified in place).
            xmin: Left edge of bounding box.
            ymin: Top edge of bounding box.
            xmax: Right edge of bounding box.
            ymax: Bottom edge of bounding box.
            label: Person name or "Unknown".
            distance: Recognition distance score (optional).
            recognized: Whether the face was recognized.
        """
        color = self.recognized_color if recognized else self.unknown_color

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, self.box_thickness)

        # Build label text
        label_text = label
        if distance is not None:
            label_text += f" ({distance:.3f})"
        else:
            label_text += " (No match)"

        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, self.font, self.font_scale, self.font_thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (xmin, ymin - text_height - baseline - 5),
            (xmin + text_width, ymin),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            label_text,
            (xmin, ymin - baseline - 2),
            self.font,
            self.font_scale,
            self.text_color,
            self.font_thickness,
        )

    def draw_error(self, frame: np.ndarray, message: str) -> None:
        """Draw an error message on the frame.

        Args:
            frame: BGR frame to draw on (modified in place).
            message: Error message to display.
        """
        cv2.putText(
            frame,
            f"Error: {message}",
            (10, 30),
            self.font,
            0.7,
            self.unknown_color,
            2,
        )

    def create_error_frame(self, width: int, height: int, message: str) -> np.ndarray:
        """Create a black frame with an error message.

        Args:
            width: Frame width.
            height: Frame height.
            message: Error message to display.

        Returns:
            Error frame as BGR numpy array.
        """
        error_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            error_frame,
            message,
            (50, height // 2),
            self.font,
            1,
            self.unknown_color,
            2,
        )
        return error_frame
