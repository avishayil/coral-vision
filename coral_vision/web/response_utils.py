"""API response utilities for standardized responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flask import jsonify


@dataclass
class APIResponse:
    """Standardized API response structure."""

    success: bool
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"success": self.success}

        if self.data is not None:
            result["data"] = self.data

        if self.error:
            result["error"] = self.error

        if self.meta:
            result["meta"] = self.meta

        return result


def success_response(
    data: Any = None,
    status: int = 200,
    meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Create a successful API response.

    Args:
        data: Response data.
        status: HTTP status code.
        meta: Optional metadata.

    Returns:
        Tuple of (JSON response, status code).
    """
    response = APIResponse(success=True, data=data, meta=meta or {})
    return jsonify(response.to_dict()), status


def error_response(
    error: str,
    status: int = 400,
    meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Create an error API response.

    Args:
        error: Error message.
        status: HTTP status code.
        meta: Optional metadata.

    Returns:
        Tuple of (JSON response, status code).
    """
    response = APIResponse(success=False, error=error, meta=meta or {})
    return jsonify(response.to_dict()), status


def not_found_response(
    resource: str = "Resource",
    meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Create a 404 not found response.

    Args:
        resource: Resource name that was not found.
        meta: Optional metadata.

    Returns:
        Tuple of (JSON response, 404 status code).
    """
    return error_response(f"{resource} not found", status=404, meta=meta)


def validation_error_response(
    errors: list[str] | str,
    meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Create a validation error response.

    Args:
        errors: List of validation errors or single error string.
        meta: Optional metadata.

    Returns:
        Tuple of (JSON response, 400 status code).
    """
    if isinstance(errors, str):
        errors = [errors]

    meta = meta or {}
    meta["validation_errors"] = errors

    return error_response(
        "Validation failed",
        status=400,
        meta=meta,
    )
