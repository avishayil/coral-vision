"""Face detection using SSD MobileNet V2 model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

from coral_vision.core.tflite import TFLiteRunner
from coral_vision.core.types import BBox, Detection


@dataclass(frozen=True)
class FaceDetector:
    """Face detector using SSD MobileNet V2 model.

    Attributes:
        runner: TFLite model runner for inference.
    """

    runner: TFLiteRunner

    def detect(self, image_rgb: Image.Image, threshold: float) -> List[Detection]:
        """Run SSD face detector and return detections in original image coordinates.

        Args:
            image_rgb: Input RGB image.
            threshold: Minimum confidence score.

        Returns:
            List of Detection objects with pixel coordinates.
        """
        orig_w, orig_h = image_rgb.size

        _, in_h, in_w, _ = self.runner.input_shape
        resized = image_rgb.resize((in_w, in_h), Image.Resampling.LANCZOS)

        # Detector expects uint8 image input in many TFLite SSD models
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        input_np = np.asarray(resized, dtype=np.uint8)
        if input_np.ndim == 3:
            input_np = np.expand_dims(input_np, axis=0)

        out = self._invoke_and_parse(input_np, orig_w, orig_h, threshold)
        return out

    def _invoke_and_parse(
        self, input_np: np.ndarray, orig_w: int, orig_h: int, threshold: float
    ) -> List[Detection]:
        # Many SSD postprocess models output:
        # 0: boxes (N,4) [ymin,xmin,ymax,xmax] normalized
        # 1: classes (N,)
        # 2: scores (N,)
        # 3: count (1,)
        #
        # Our TFLiteRunner returns only output_details[0] by default,
        # so we need access to raw interpreter outputs. Instead of complicating runner,
        # we parse by calling interpreter internals here.
        #
        # To keep it simple, we'll re-use runner._interpreter (private).
        interpreter = self.runner._interpreter  # noqa: SLF001
        input_details = interpreter.get_input_details()[0]
        tensor_index = input_details["index"]
        # Use set_tensor instead of direct tensor access to avoid reference conflicts
        interpreter.set_tensor(tensor_index, input_np)
        interpreter.invoke()

        def get_output(i: int) -> np.ndarray:
            od = interpreter.get_output_details()[i]
            # Copy output to avoid reference issues
            return np.squeeze(interpreter.get_tensor(od["index"]).copy())

        boxes = get_output(0)
        scores = get_output(2)
        count = int(get_output(3))

        detections: List[Detection] = []
        for i in range(count):
            score = float(scores[i])
            if score < threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            xmin_px = int(max(0.0, min(1.0, float(xmin))) * orig_w)
            xmax_px = int(max(0.0, min(1.0, float(xmax))) * orig_w)
            ymin_px = int(max(0.0, min(1.0, float(ymin))) * orig_h)
            ymax_px = int(max(0.0, min(1.0, float(ymax))) * orig_h)
            bbox = BBox(xmin=xmin_px, ymin=ymin_px, xmax=xmax_px, ymax=ymax_px).clamp(
                orig_w, orig_h
            )
            if bbox.is_valid():
                detections.append(Detection(bbox=bbox, score=score))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
