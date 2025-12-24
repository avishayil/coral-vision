"""Face embedding generation using MobileNet triplet model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coral_vision.core.tflite import TFLiteRunner


@dataclass(frozen=True)
class FaceEmbedder:
    """Face embedding generator using MobileNet triplet model.

    Attributes:
        runner: TFLite model runner for inference.
    """

    runner: TFLiteRunner

    def embed_face_chip(self, chip_96_rgb_uint8: np.ndarray) -> np.ndarray:
        """Generate embedding vector from 96x96 face chip.

        Args:
            chip_96_rgb_uint8: Face chip with shape (96, 96, 3), dtype uint8, RGB format.

        Returns:
            Embedding array with shape (1, D) as float.

        Raises:
            ValueError: If input shape is not (96, 96, 3).
        """
        if chip_96_rgb_uint8.shape != (96, 96, 3):
            raise ValueError(f"Expected (96,96,3), got {chip_96_rgb_uint8.shape}")
        if chip_96_rgb_uint8.dtype != np.uint8:
            chip_96_rgb_uint8 = chip_96_rgb_uint8.astype(np.uint8)

        # Model expects 0..1 float with batch dimension
        x = (chip_96_rgb_uint8.reshape(1, 96, 96, 3).astype(np.float32)) / 255.0
        emb = self.runner.invoke(x)
        return emb
