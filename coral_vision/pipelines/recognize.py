"""Face recognition pipeline: detection, embedding, matching, and result output."""
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
from coral_vision.core.recognition import EmbeddingDB
from coral_vision.core.tts import Speaker

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


def recognize_folder(
    *,
    paths: Paths,
    input_path: Path,
    use_edgetpu: bool,
    threshold: float,
    top_k: int,
    per_person_k: int,
    say: bool,
    storage: "StorageBackend",
) -> dict[str, Any]:
    """Recognize faces in images: detect, embed, match, and return results.

    For each input image:
      1. Detect all faces using SSD MobileNet V2
      2. Crop faces to 96x96 chips
      3. Generate embeddings using MobileNet triplet model
      4. Match against enrolled person embeddings using L2 distance
      5. Return top-k matches per face, with best prediction if below threshold
      6. Optionally speak greeting using TTS

    Args:
        paths: Configuration paths object.
        input_path: Path to folder of images or single image file.
        use_edgetpu: Whether to use Edge TPU acceleration.
        threshold: Maximum L2 distance for positive identification.
        top_k: Number of top matches to return per face.
        per_person_k: Number of best embeddings to average per person.
        say: Whether to use TTS to greet recognized persons.
        storage: pgvector storage backend.

    Returns:
        Dictionary with recognition results including image paths,
        detected faces, matches, and predictions.

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

    # Load embeddings from storage backend
    db = EmbeddingDB.load_from_backend(storage)

    # Use cached models to avoid reloading
    detector = FaceDetector(get_cached_model(detector_model, use_edgetpu=use_edgetpu))
    embedder = FaceEmbedder(get_cached_model(embedder_model, use_edgetpu=use_edgetpu))
    speaker = Speaker() if say else None

    greeted: set[str] = set()
    image_results: list[dict[str, Any]] = []

    for img_path in iter_images(input_path):
        image_rgb = load_rgb(img_path)
        detections = detector.detect(
            image_rgb, threshold=0.5
        )  # detect broadly; classify by distance later

        faces_out: list[dict[str, Any]] = []
        for det in detections:
            bbox = det.bbox
            chip = _crop_face_chip(
                image_rgb, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
            )

            emb = embedder.embed_face_chip(chip)  # (1,D)
            matches = db.match(emb, per_person_k=per_person_k, top_k=top_k)

            predicted = matches[0] if matches else None
            accepted = predicted is not None and predicted.distance < threshold

            face_payload = {
                "bbox": {
                    "xmin": bbox.xmin,
                    "ymin": bbox.ymin,
                    "xmax": bbox.xmax,
                    "ymax": bbox.ymax,
                },
                "score": det.score,
                "matches": [
                    {"person_id": m.person_id, "name": m.name, "distance": m.distance}
                    for m in matches
                ],
                "predicted": (
                    {
                        "person_id": predicted.person_id,
                        "name": predicted.name,
                        "distance": predicted.distance,
                    }
                    if predicted
                    else None
                ),
                "accepted": accepted,
                "threshold": threshold,
            }
            faces_out.append(face_payload)

            if accepted and speaker and predicted:
                # greet each person once per run
                if predicted.person_id not in greeted:
                    greeted.add(predicted.person_id)
                    speaker.say_hello(predicted.name)

        image_results.append({"image_path": str(img_path), "faces": faces_out})

    return {
        "use_edgetpu": use_edgetpu,
        "input": str(input_path),
        "threshold": threshold,
        "top_k": top_k,
        "per_person_k": per_person_k,
        "results": image_results,
    }
