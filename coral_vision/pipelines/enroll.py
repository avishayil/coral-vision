"""Person enrollment pipeline: face detection, embedding generation, and storage."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from coral_vision.config import Paths, resolve_model_paths
from coral_vision.core.face_detect import FaceDetector
from coral_vision.core.face_embed import FaceEmbedder
from coral_vision.core.image_io import iter_images, load_rgb
from coral_vision.core.model_cache import get_cached_model

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
    # PIL -> numpy RGB
    arr = np.asarray(image_rgb, dtype=np.uint8)
    crop = arr[ymin:ymax, xmin:xmax, :]
    # Resize to (96,96)
    chip = cv2.resize(crop, dsize=(96, 96), interpolation=cv2.INTER_CUBIC).astype(
        np.uint8
    )
    return chip


def enroll_person(
    *,
    paths: Paths,
    person_id: str,
    name: str,
    images_path: Path,
    use_edgetpu: bool,
    min_score: float,
    max_faces: int,
    keep_copies: bool,
    storage: "StorageBackend",
) -> dict[str, Any]:
    """Enroll a person from images: detect faces, create embeddings, save to database.

    For each input image:
      1. Detect face(s) using SSD MobileNet V2
      2. Crop detected faces to 96x96 chips
      3. Generate embeddings using MobileNet triplet model
      4. Save embeddings to pgvector database

    Args:
        paths: Configuration paths object.
        person_id: Unique identifier for the person.
        name: Display name for the person.
        images_path: Path to folder of images or single image file.
        use_edgetpu: Whether to use Edge TPU acceleration.
        min_score: Minimum detection confidence threshold.
        max_faces: Maximum faces to process per image.
        keep_copies: Whether to copy original images to person's directory (unused with pgvector).
        storage: pgvector storage backend.

    Returns:
        Dictionary with enrollment statistics including number of images
        processed, faces detected, and embeddings created.

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

    # Create person in storage
    storage.upsert_person(person_id, name)

    # Use cached models to avoid reloading
    detector = FaceDetector(get_cached_model(detector_model, use_edgetpu=use_edgetpu))
    embedder = FaceEmbedder(get_cached_model(embedder_model, use_edgetpu=use_edgetpu))

    processed = 0
    saved_faces = 0
    saved_embeddings = 0
    skipped_no_face = 0
    skipped_low_score = 0

    for img_path in iter_images(images_path):
        processed += 1

        image_rgb = load_rgb(img_path)
        detections = detector.detect(image_rgb, threshold=min_score)

        if not detections:
            skipped_no_face += 1
            continue

        # Limit faces per image
        chosen = detections[: max(1, max_faces)]

        for _face_idx, det in enumerate(chosen):
            if det.score < min_score:
                skipped_low_score += 1
                continue

            bbox = det.bbox
            chip = _crop_face_chip(
                image_rgb, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
            )

            # Generate and save embedding to database
            emb = embedder.embed_face_chip(chip)  # (1,D)
            storage.add_embedding(
                person_id, emb.reshape(-1), source_image=img_path.name
            )
            saved_embeddings += 1
            saved_faces += 1

    return {
        "person_id": person_id,
        "name": name,
        "use_edgetpu": use_edgetpu,
        "input": str(images_path),
        "processed_images": processed,
        "saved_faces": saved_faces,
        "saved_embeddings": saved_embeddings,
        "skipped_no_face": skipped_no_face,
        "skipped_low_score": skipped_low_score,
        "output_dir": "database",
    }
