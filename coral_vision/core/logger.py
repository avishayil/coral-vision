"""Logging configuration for Coral Vision."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_level: str | int = logging.INFO,
    log_file: Path | str | None = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_level: Logging level (string or int).
        log_file: Optional path to log file.
        log_to_console: Whether to log to console.

    Returns:
        Configured logger instance.
    """
    # Convert string level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("coral_vision")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (defaults to 'coral_vision').

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"coral_vision.{name}")
    return logging.getLogger("coral_vision")


# Initialize default logger
_default_log_file = os.getenv("LOG_FILE", None)
_default_log_level = os.getenv("LOG_LEVEL", "INFO")
_default_logger = setup_logging(
    log_level=_default_log_level,
    log_file=_default_log_file,
    log_to_console=True,
)
