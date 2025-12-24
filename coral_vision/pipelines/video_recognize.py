"""Video recognition pipeline: real-time face detection and recognition on video streams."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator, Optional

import cv2
import numpy as np
from PIL import Image

from coral_vision.config import Paths, resolve_model_paths
from coral_vision.core.face_detect import FaceDetector
from coral_vision.core.face_embed import FaceEmbedder
from coral_vision.core.model_cache import get_cached_model
from coral_vision.core.recognition import EmbeddingDB
from coral_vision.core.video_capture import VideoCapture
from coral_vision.core.video_render import VideoRenderer

if TYPE_CHECKING:
    from coral_vision.core.storage_backend import StorageBackend


def _crop_face_chip(
    image_rgb: Image.Image, xmin: int, ymin: int, xmax: int, ymax: int
) -> np.ndarray:
    """Crop and resize a face region to 96x96 RGB chip.

    Args:
        image_rgb: Source PIL image in RGB format.
        xmin: Left edge of face bounding box.
        ymin: Top edge of face bounding box.
        xmax: Right edge of face bounding box.
        ymax: Bottom edge of face bounding box.

    Returns:
        96x96x3 uint8 numpy array in RGB format.
    """
    arr = np.asarray(image_rgb, dtype=np.uint8)
    crop = arr[ymin:ymax, xmin:xmax, :]
    chip = cv2.resize(crop, dsize=(96, 96), interpolation=cv2.INTER_CUBIC).astype(
        np.uint8
    )
    return chip


class VideoRecognitionPipeline:
    """Pipeline for real-time face recognition on video streams."""

    def __init__(
        self,
        *,
        paths: Paths,
        use_edgetpu: bool,
        storage: "StorageBackend",
        threshold: float = 0.6,
        per_person_k: int = 20,
        db_reload_interval: int = 30,
    ) -> None:
        """Initialize the video recognition pipeline.

        Args:
            paths: Configuration paths object.
            use_edgetpu: Whether to use Edge TPU acceleration.
            storage: Storage backend for loading embeddings.
            threshold: Maximum L2 distance for positive identification.
            per_person_k: Number of best embeddings to average per person.
            db_reload_interval: Reload embedding database every N frames.

        Raises:
            FileNotFoundError: If model files are missing.
        """
        model_paths = resolve_model_paths(paths)
        detector_model = (
            model_paths.detector_edgetpu if use_edgetpu else model_paths.detector_cpu
        )
        embedder_model = (
            model_paths.embedder_edgetpu if use_edgetpu else model_paths.embedder_cpu
        )

        if not detector_model.exists():
            raise FileNotFoundError(f"Detector model not found: {detector_model}")
        if not embedder_model.exists():
            raise FileNotFoundError(f"Embedder model not found: {embedder_model}")

        # Use cached models to avoid reloading
        self.detector = FaceDetector(
            get_cached_model(detector_model, use_edgetpu=use_edgetpu)
        )
        self.embedder = FaceEmbedder(
            get_cached_model(embedder_model, use_edgetpu=use_edgetpu)
        )
        self.storage = storage
        self.threshold = threshold
        self.per_person_k = per_person_k
        self.db_reload_interval = db_reload_interval

        self._embedding_db: Optional[EmbeddingDB] = None
        self._frame_count = 0
        self._last_db_load_time: float = 0.0
        self.renderer = VideoRenderer()

    def _ensure_embedding_db(self) -> EmbeddingDB:
        """Ensure embedding database is loaded and up-to-date.

        Uses time-based caching instead of frame-based to avoid unnecessary reloads
        when frame rate varies.

        Returns:
            Current embedding database.
        """
        current_time = time.time()
        time_since_load = current_time - self._last_db_load_time

        # Reload if database is None or if reload interval (in seconds) has passed
        # Convert db_reload_interval from frames to seconds (assuming ~30 FPS)
        reload_interval_seconds = self.db_reload_interval / 30.0

        if self._embedding_db is None or time_since_load >= reload_interval_seconds:
            self._embedding_db = EmbeddingDB.load_from_backend(self.storage)
            self._last_db_load_time = current_time
        return self._embedding_db

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Process a single video frame with face recognition.

        Args:
            frame_bgr: BGR frame from video capture.

        Returns:
            Annotated frame with bounding boxes and labels.
        """
        try:
            self._frame_count += 1
            db = self._ensure_embedding_db()

            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            # Detect faces
            detections = self.detector.detect(image_pil, threshold=0.5)

            # Process each detection
            for det in detections:
                bbox = det.bbox
                xmin, ymin, xmax, ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

                # Crop face chip
                chip = _crop_face_chip(image_pil, xmin, ymin, xmax, ymax)

                # Generate embedding
                emb = self.embedder.embed_face_chip(chip)

                # Match against database
                matches = db.match(emb, per_person_k=self.per_person_k, top_k=1)

                # Determine recognition result
                recognized = False
                label = "Unknown"
                distance = None

                if matches and matches[0].distance < self.threshold:
                    match = matches[0]
                    label = match.name
                    distance = match.distance
                    recognized = True

                # Draw annotation
                self.renderer.draw_face(
                    frame_bgr, xmin, ymin, xmax, ymax, label, distance, recognized
                )

        except Exception as e:
            self.renderer.draw_error(frame_bgr, str(e))

        return frame_bgr

    def generate_frames(
        self, camera_index: int = 0, width: int = 640, height: int = 480
    ) -> Iterator[bytes]:
        """Generate video frames with face recognition as JPEG bytes.

        Args:
            camera_index: Index of camera device.
            width: Desired frame width.
            height: Desired frame height.

        Yields:
            JPEG-encoded frame bytes in MJPEG format.
        """
        capture = VideoCapture(camera_index=camera_index, width=width, height=height)

        try:
            capture.open()

            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                # Process frame with face recognition
                annotated_frame = self.process_frame(frame)

                # Encode frame as JPEG
                ret, buffer = cv2.imencode(
                    ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

                # Yield frame in multipart format
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

        except Exception as e:
            # Send error frame
            error_frame = self.renderer.create_error_frame(width, height, str(e))
            ret, buffer = cv2.imencode(".jpg", error_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        finally:
            capture.release()
