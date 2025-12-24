"""Edge TPU detection and verification utilities."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from coral_vision.core.logger import get_logger

logger = get_logger("edgetpu")


def verify_edgetpu_availability(
    models_dir: Optional[Path] = None,
) -> tuple[bool, Optional[str]]:
    """Verify if Edge TPU is available and working.

    Args:
        models_dir: Optional path to models directory for device testing.

    Returns:
        Tuple of (is_available, warning_message).
        is_available: True if Edge TPU library is loaded successfully.
        warning_message: Warning message if Edge TPU is not available, None otherwise.
    """
    try:
        from tflite_runtime.interpreter import load_delegate

        # Try to load the Edge TPU delegate
        load_delegate("libedgetpu.so.1.0")
        logger.info("Edge TPU library loaded successfully")

        # Optionally test device access with a model
        if models_dir:
            test_model = (
                models_dir / "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
            )
            if test_model.exists():
                try:
                    from coral_vision.core.tflite import TFLiteRunner

                    # Create test runner to verify device access
                    TFLiteRunner(test_model, use_edgetpu=True)
                    logger.info("Edge TPU device verified and working")
                except Exception as test_e:
                    warning_msg = (
                        f"Edge TPU library loaded but device test failed: {test_e}. "
                        "This may indicate the Edge TPU device is not accessible."
                    )
                    logger.warning(warning_msg)
                    return True, warning_msg
            else:
                warning_msg = "Edge TPU library loaded but test model not found"
                logger.warning(warning_msg)
                return True, warning_msg

        return True, None

    except Exception as e:
        warning_msg = (
            f"Edge TPU requested but library not found ({type(e).__name__}: {e}). "
            "Will attempt to use Edge TPU models but may fall back to CPU mode. "
            "To fix: Install Edge TPU runtime: apt-get install libedgetpu1-std"
        )
        logger.warning(warning_msg)
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
        return False, warning_msg


def check_edgetpu_status(use_edgetpu: bool, models_dir: Optional[Path] = None) -> bool:
    """Check Edge TPU status and print warnings if needed.

    Args:
        use_edgetpu: Whether Edge TPU is requested.
        models_dir: Optional path to models directory for device testing.

    Returns:
        True if Edge TPU should be used (available or will fallback), False otherwise.
    """
    if not use_edgetpu:
        return False

    is_available, warning = verify_edgetpu_availability(models_dir)
    return True  # Always return True to allow fallback in TFLiteRunner
