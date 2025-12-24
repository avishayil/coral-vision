"""Input validation utilities for Coral Vision."""

from __future__ import annotations

import re

from coral_vision.core.exceptions import ValidationError


def validate_person_id(person_id: str) -> bool:
    """Validate person_id format.

    Args:
        person_id: Person identifier to validate.

    Returns:
        True if valid, False otherwise.

    Raises:
        ValidationError: If person_id is invalid.
    """
    if not person_id:
        raise ValidationError("person_id cannot be empty")

    if len(person_id) > 255:
        raise ValidationError("person_id cannot exceed 255 characters")

    # Only allow alphanumeric, underscore, hyphen
    if not re.match(r"^[a-zA-Z0-9_-]+$", person_id):
        raise ValidationError(
            "person_id can only contain alphanumeric characters, underscores, and hyphens"
        )

    return True


def validate_person_name(name: str) -> bool:
    """Validate person name format.

    Args:
        name: Person name to validate.

    Returns:
        True if valid, False otherwise.

    Raises:
        ValidationError: If name is invalid.
    """
    if not name:
        raise ValidationError("name cannot be empty")

    if len(name) > 255:
        raise ValidationError("name cannot exceed 255 characters")

    # Allow most characters but prevent SQL injection patterns
    if re.search(r'[<>"\';\\]', name):
        raise ValidationError("name contains invalid characters")

    return True


def validate_threshold(threshold: float) -> bool:
    """Validate recognition threshold value.

    Args:
        threshold: Threshold value to validate.

    Returns:
        True if valid.

    Raises:
        ValidationError: If threshold is invalid.
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError("threshold must be a number")

    if threshold < 0.0 or threshold > 1.0:
        raise ValidationError("threshold must be between 0.0 and 1.0")

    return True


def validate_file_extension(filename: str, allowed_extensions: set[str]) -> bool:
    """Validate file extension.

    Args:
        filename: Filename to validate.
        allowed_extensions: Set of allowed extensions (without dot).

    Returns:
        True if valid.

    Raises:
        ValidationError: If extension is invalid.
    """
    if not filename:
        raise ValidationError("filename cannot be empty")

    if "." not in filename:
        raise ValidationError("filename must have an extension")

    extension = filename.rsplit(".", 1)[1].lower()
    if extension not in allowed_extensions:
        raise ValidationError(
            f"Invalid file extension: {extension}. Allowed: {', '.join(allowed_extensions)}"
        )

    return True


def validate_file_size(file_size: int, max_size: int = 16 * 1024 * 1024) -> bool:
    """Validate file size.

    Args:
        file_size: File size in bytes.
        max_size: Maximum allowed size in bytes (default: 16MB).

    Returns:
        True if valid.

    Raises:
        ValidationError: If file size exceeds limit.
    """
    if file_size <= 0:
        raise ValidationError("file size must be greater than 0")

    if file_size > max_size:
        max_size_mb = max_size / (1024 * 1024)
        raise ValidationError(f"file size exceeds maximum of {max_size_mb}MB")

    return True
