"""Configuration paths and model resolution for Coral Vision."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Configuration paths for data directory structure.

    Attributes:
        data_dir: Root data directory containing models.
    """

    data_dir: Path

    @property
    def models_dir(self) -> Path:
        """Return path to models directory."""
        return self.data_dir / "models"


@dataclass(frozen=True)
class ModelPaths:
    """Paths to TFLite models for CPU and Edge TPU inference.

    Attributes:
        detector_cpu: Path to CPU face detection model.
        detector_edgetpu: Path to Edge TPU face detection model.
        embedder_cpu: Path to CPU face embedding model.
        embedder_edgetpu: Path to Edge TPU face embedding model.
    """

    detector_cpu: Path
    detector_edgetpu: Path
    embedder_cpu: Path
    embedder_edgetpu: Path


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration for pgvector backend.

    Attributes:
        host: PostgreSQL host address.
        port: PostgreSQL port number.
        database: Database name.
        user: Database user.
        password: Database password.
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "coral_vision"
    user: str = "coral"
    password: str = "coral"


def get_data_dir_from_env(default: str = "./data") -> Path:
    """Get data directory path from environment variable.

    Args:
        default: Default path if DATA_DIR is not set.

    Returns:
        Path to data directory.
    """
    return Path(os.getenv("DATA_DIR", default))


def resolve_model_paths(paths: Paths) -> ModelPaths:
    """Resolve model file paths from data directory.

    Args:
        paths: Configuration paths object.

    Returns:
        ModelPaths object with resolved paths to all model files.
    """
    return ModelPaths(
        detector_cpu=paths.models_dir
        / "ssd_mobilenet_v2_face_quant_postprocess.tflite",
        detector_edgetpu=paths.models_dir
        / "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
        embedder_cpu=paths.models_dir
        / "Mobilenet1_triplet1589223569_triplet_quant.tflite",
        embedder_edgetpu=paths.models_dir
        / "Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite",
    )
