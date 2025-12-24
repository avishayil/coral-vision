"""File validation utilities for image uploads."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from coral_vision.core.exceptions import ValidationError
from coral_vision.core.logger import get_logger
from coral_vision.core.validation import validate_file_extension, validate_file_size

logger = get_logger("file_validation")


def validate_image_file(
    file_content: bytes,
    filename: str,
    allowed_extensions: set[str] | None = None,
    max_size: int = 16 * 1024 * 1024,  # 16MB
) -> tuple[bool, str]:
    """Validate an image file comprehensively.

    This function performs multiple checks:
    1. File extension validation
    2. File size validation
    3. MIME type validation (actual image content)
    4. Image format verification (can be opened by PIL)

    Args:
        file_content: Raw file content as bytes.
        filename: Original filename.
        allowed_extensions: Set of allowed file extensions.
        max_size: Maximum file size in bytes.

    Returns:
        Tuple of (is_valid, error_message).
        is_valid: True if file passes all validations.
        error_message: Error message if validation fails, empty string if valid.
    """
    try:
        # 1. Validate file extension
        validate_file_extension(filename, allowed_extensions)

        # 2. Validate file size
        validate_file_size(len(file_content), max_size)

        # 3. Validate MIME type and image format
        try:
            # Try to open and verify the image
            image = Image.open(io.BytesIO(file_content))

            # Verify it's actually an image (not just a file with image extension)
            image.verify()

            # Reopen after verify (verify closes the image)
            image = Image.open(io.BytesIO(file_content))

            # Check if format is supported
            if image.format not in ["JPEG", "PNG", "BMP", "WEBP"]:
                return False, f"Unsupported image format: {image.format}"

            # Check image dimensions (prevent extremely large images)
            max_dimension = 10000  # 10k pixels
            if image.width > max_dimension or image.height > max_dimension:
                return (
                    False,
                    f"Image dimensions too large: {image.width}x{image.height}",
                )

            # Check if image is corrupted or empty
            if image.width == 0 or image.height == 0:
                return False, "Image has invalid dimensions"

            image.close()

        except Exception as e:
            logger.warning(f"Image validation failed for {filename}: {e}")
            return False, f"Invalid or corrupted image file: {str(e)}"

        return True, ""

    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        logger.error(f"Unexpected error validating file {filename}: {e}", exc_info=True)
        return False, f"File validation error: {str(e)}"


def validate_image_file_from_path(
    file_path: Path,
    allowed_extensions: set[str] | None = None,
    max_size: int = 16 * 1024 * 1024,
) -> tuple[bool, str]:
    """Validate an image file from file path.

    Args:
        file_path: Path to the image file.
        allowed_extensions: Set of allowed file extensions.
        max_size: Maximum file size in bytes.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    try:
        file_content = file_path.read_bytes()
        return validate_image_file(
            file_content,
            file_path.name,
            allowed_extensions,
            max_size,
        )
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        return False, f"Error reading file: {str(e)}"
