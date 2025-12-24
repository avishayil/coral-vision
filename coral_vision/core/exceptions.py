"""Custom exceptions for Coral Vision."""

from __future__ import annotations


class CoralVisionError(Exception):
    """Base exception for all Coral Vision errors."""

    pass


class RecognitionError(CoralVisionError):
    """Error during face recognition process."""

    pass


class DatabaseError(CoralVisionError):
    """Database operation error."""

    pass


class ModelLoadError(CoralVisionError):
    """Error loading TensorFlow Lite model."""

    pass


class StorageError(CoralVisionError):
    """Storage backend error."""

    pass


class ValidationError(CoralVisionError):
    """Input validation error."""

    pass


class AuthenticationError(CoralVisionError):
    """Authentication/authorization error."""

    pass
