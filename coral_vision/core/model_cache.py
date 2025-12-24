"""Model caching utilities to avoid reloading models unnecessarily."""

from __future__ import annotations

import threading
from pathlib import Path

from coral_vision.core.logger import get_logger
from coral_vision.core.tflite import TFLiteRunner

logger = get_logger("model_cache")

# Global model cache with thread safety
_model_cache: dict[str, TFLiteRunner] = {}
_model_lock = threading.Lock()


def get_cached_model(
    model_path: Path,
    use_edgetpu: bool,
) -> TFLiteRunner:
    """Get or create a cached TFLite model runner.

    Models are cached by path and edgetpu flag to avoid reloading
    the same model multiple times.

    Args:
        model_path: Path to the TensorFlow Lite model file.
        use_edgetpu: Whether to use Edge TPU acceleration.

    Returns:
        Cached TFLiteRunner instance.
    """
    # Create cache key from model path and edgetpu flag
    cache_key = f"{model_path}:{use_edgetpu}"

    with _model_lock:
        if cache_key not in _model_cache:
            logger.debug(
                f"Loading model into cache: {model_path} (edgetpu={use_edgetpu})"
            )
            _model_cache[cache_key] = TFLiteRunner(model_path, use_edgetpu=use_edgetpu)
        else:
            logger.debug(f"Using cached model: {model_path} (edgetpu={use_edgetpu})")

        return _model_cache[cache_key]


def clear_model_cache() -> None:
    """Clear the model cache.

    Useful for testing or when models need to be reloaded.
    """
    with _model_lock:
        _model_cache.clear()
        logger.info("Model cache cleared")


def get_cache_size() -> int:
    """Get the number of models currently cached.

    Returns:
        Number of cached models.
    """
    with _model_lock:
        return len(_model_cache)
