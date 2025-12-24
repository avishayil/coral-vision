"""Video pipeline manager for better state management."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from coral_vision.core.logger import get_logger

if TYPE_CHECKING:
    from coral_vision.config import Paths
    from coral_vision.core.storage_backend import StorageBackend
    from coral_vision.pipelines.video_recognize import VideoRecognitionPipeline

logger = get_logger("pipeline_manager")


class VideoPipelineManager:
    """Thread-safe manager for video recognition pipelines.

    Manages multiple pipeline instances per session to avoid global state
    and improve testability.
    """

    def __init__(self) -> None:
        """Initialize pipeline manager."""
        self._pipelines: dict[str, "VideoRecognitionPipeline"] = {}
        self._lock = threading.Lock()

    def get_pipeline(
        self,
        session_id: str,
        paths: "Paths",
        use_edgetpu: bool,
        storage: "StorageBackend",
        threshold: float = 0.6,
    ) -> "VideoRecognitionPipeline":
        """Get or create a video pipeline for a session.

        Args:
            session_id: Unique session identifier.
            paths: Configuration paths object.
            use_edgetpu: Whether to use Edge TPU.
            storage: Storage backend instance.
            threshold: Recognition threshold.

        Returns:
            Video recognition pipeline instance.
        """
        with self._lock:
            cache_key = f"{session_id}:{threshold}"

            if cache_key not in self._pipelines:
                from coral_vision.pipelines.video_recognize import (
                    VideoRecognitionPipeline,
                )

                logger.debug(f"Creating new pipeline for session: {session_id}")
                self._pipelines[cache_key] = VideoRecognitionPipeline(
                    paths=paths,
                    use_edgetpu=use_edgetpu,
                    storage=storage,
                    threshold=threshold,
                )
            else:
                logger.debug(f"Reusing pipeline for session: {session_id}")

            return self._pipelines[cache_key]

    def remove_pipeline(self, session_id: str, threshold: float = 0.6) -> None:
        """Remove a pipeline for a session.

        Args:
            session_id: Session identifier.
            threshold: Threshold used for the pipeline.
        """
        with self._lock:
            cache_key = f"{session_id}:{threshold}"
            if cache_key in self._pipelines:
                del self._pipelines[cache_key]
                logger.debug(f"Removed pipeline for session: {session_id}")

    def clear_all(self) -> None:
        """Clear all pipelines."""
        with self._lock:
            self._pipelines.clear()
            logger.info("Cleared all pipelines")

    def get_pipeline_count(self) -> int:
        """Get the number of active pipelines.

        Returns:
            Number of active pipelines.
        """
        with self._lock:
            return len(self._pipelines)
