"""TensorFlow Lite model runner with optional Edge TPU support."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate


@dataclass(frozen=True)
class TFLiteRunner:
    """TensorFlow Lite model runner with optional Edge TPU acceleration.

    Attributes:
        model_path: Path to the .tflite model file.
        use_edgetpu: Whether to use Edge TPU acceleration.
        delegate_path: Path to Edge TPU delegate library.
    """

    model_path: Path
    use_edgetpu: bool = False
    delegate_path: str = "libedgetpu.so.1.0"

    def __post_init__(self) -> None:
        """Initialize the TFLite interpreter and allocate tensors."""
        interpreter = self._create_interpreter()
        interpreter.allocate_tensors()

        object.__setattr__(self, "_interpreter", interpreter)
        object.__setattr__(self, "_input_details", interpreter.get_input_details()[0])
        object.__setattr__(self, "_output_details", interpreter.get_output_details()[0])

    def _create_interpreter(self) -> Interpreter:
        """Create TFLite interpreter with or without Edge TPU delegate.

        Returns:
            Configured TFLite interpreter.

        Note:
            If Edge TPU is requested but not available, automatically falls back to CPU
            and logs a warning. This allows Edge TPU models to run on CPU as fallback.
        """
        if self.use_edgetpu:
            try:
                delegate = load_delegate(self.delegate_path)
                return Interpreter(
                    model_path=str(self.model_path),
                    experimental_delegates=[delegate],
                )
            except Exception as e:
                # Edge TPU not available, fall back to CPU
                # Edge TPU models can usually run on CPU, so this is a safe fallback
                import warnings

                error_msg = f"Edge TPU delegate not available ({type(e).__name__}: {e}), falling back to CPU mode"
                from coral_vision.core.logger import get_logger

                logger = get_logger("tflite")
                logger.warning(error_msg)
                warnings.warn(error_msg, RuntimeWarning, stacklevel=2)
                # Update use_edgetpu flag to prevent future attempts
                object.__setattr__(self, "use_edgetpu", False)
                return Interpreter(model_path=str(self.model_path))
        return Interpreter(model_path=str(self.model_path))

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Get the input tensor shape.

        Returns:
            Tuple representing input tensor dimensions.
        """
        return tuple(self._input_details["shape"])

    def invoke(self, input_float: np.ndarray) -> np.ndarray:
        """Run inference on input tensor, handling quantization automatically.

        Args:
            input_float: Input tensor as float array.

        Returns:
            Output tensor as float array (dequantized if needed).
        """
        tensor_index = self._input_details["index"]
        in_scale, in_zero = self._input_details.get("quantization", (0.0, 0))

        # Use set_tensor instead of direct tensor access to avoid reference conflicts
        if in_scale and in_scale > 0:
            q = np.uint8(input_float / in_scale + in_zero)
            self._interpreter.set_tensor(tensor_index, q)
        else:
            self._interpreter.set_tensor(tensor_index, input_float)

        self._interpreter.invoke()

        out_index = self._output_details["index"]
        # Use get_tensor and copy to avoid reference issues
        out = self._interpreter.get_tensor(out_index).copy()

        out_scale, out_zero = self._output_details.get("quantization", (0.0, 0))
        if out_scale and out_scale > 0:
            out = out_scale * (out - out_zero)

        return out
