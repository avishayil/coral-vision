"""Image input/output utilities for loading and processing images."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path: Path):
    """Iterate over image files in a path.

    If path is a file, yields that file if it's an image.
    If path is a directory, recursively yields all image files.

    Args:
        path: File or directory path to search.

    Yields:
        Path objects for each image file found.
    """
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        yield path
        return

    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def load_rgb(path: Path) -> Image.Image:
    """Load an image and convert to RGB format.

    Args:
        path: Path to image file.

    Returns:
        PIL Image in RGB mode.
    """
    return Image.open(path).convert("RGB")
