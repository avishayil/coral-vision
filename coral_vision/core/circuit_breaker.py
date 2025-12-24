"""Circuit breaker pattern implementation for resilient service calls."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, TypeVar

from coral_vision.core.logger import get_logger

logger = get_logger("circuit_breaker")

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures.

    The circuit breaker monitors failures and opens the circuit when
    failure threshold is reached, preventing further calls to a failing service.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Seconds to wait before attempting recovery (half-open).
            expected_exception: Exception types that count as failures.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.success_count = 0

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to call.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function result.

        Raises:
            CircuitBreakerOpenError: If circuit is open.
            Exception: Any exception raised by the function.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. "
                    f"Last failure: {self.last_failure_time}. "
                    f"Will retry after {self.recovery_timeout}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:  # Require 2 successes to close
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        else:
            # Reset failure count on success in CLOSED state
            self.failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state, go back to open
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning("Circuit breaker OPENED after failure in HALF_OPEN state")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and call is rejected."""

    pass


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception,
):
    """Decorator to add circuit breaker to a function.

    Args:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds to wait before attempting recovery.
        expected_exception: Exception types that count as failures.

    Returns:
        Decorated function with circuit breaker protection.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
        )

        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)

        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper

    return decorator
