"""
Pipeline retry policy for fault-tolerant stage execution.

Provides configurable retry behavior with exponential backoff,
error tracking, and manifest integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    """Exception that can trigger a retry."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class NonRetryableError(Exception):
    """Exception that should not trigger retries."""

    pass


class ErrorCategory(str, Enum):
    """Category of error for tracking and reporting."""

    MODEL_LOAD = "model_load"
    OOM = "out_of_memory"
    PARSE_FAILURE = "parse_failure"
    DATA_ERROR = "data_error"
    TRANSIENT_RAY = "transient_ray"
    NETWORK = "network"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    # Error handling
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            RetryableError,
            ConnectionError,
            TimeoutError,
            OSError,
        )
    )
    non_retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            NonRetryableError,
            ValueError,
            TypeError,
            KeyError,
        )
    )

    def compute_delay(self, attempt: int) -> float:
        """
        Compute delay before next retry attempt.

        Args:
            attempt: Current attempt number (1-indexed).

        Returns:
            Delay in seconds.
        """
        import random

        delay = self.initial_delay_seconds * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay_seconds)

        if self.jitter:
            # Add random jitter up to 25% of delay
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if an error should trigger a retry.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number.

        Returns:
            True if should retry, False otherwise.
        """
        if attempt >= self.max_attempts:
            return False

        if isinstance(exception, self.non_retryable_exceptions):
            return False

        if isinstance(exception, self.retryable_exceptions):
            return True

        # Default: retry unknown exceptions
        return True


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""

    attempt_number: int
    started_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    error: str | None = None
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    duration_ms: float = 0.0


@dataclass
class RetryContext:
    """Context for tracking retry state across attempts."""

    stage_name: str
    row_id: str | None = None
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    attempts: list[RetryAttempt] = field(default_factory=list)
    final_success: bool = False
    final_error: str | None = None

    @property
    def attempt_count(self) -> int:
        """Number of attempts made."""
        return len(self.attempts)

    @property
    def total_duration_ms(self) -> float:
        """Total duration across all attempts."""
        return sum(a.duration_ms for a in self.attempts)

    def record_attempt(
        self,
        success: bool,
        error: Exception | None = None,
        started_at: datetime | None = None,
    ) -> RetryAttempt:
        """
        Record a retry attempt.

        Args:
            success: Whether the attempt succeeded.
            error: Exception if failed.
            started_at: When the attempt started.

        Returns:
            The recorded attempt.
        """
        now = datetime.utcnow()
        started = started_at or now

        attempt = RetryAttempt(
            attempt_number=self.attempt_count + 1,
            started_at=started,
            completed_at=now,
            success=success,
            duration_ms=(now - started).total_seconds() * 1000,
        )

        if error:
            attempt.error = str(error)
            attempt.error_category = categorize_error(error)

        self.attempts.append(attempt)

        if success:
            self.final_success = True
        elif not self.policy.should_retry(error, attempt.attempt_number) if error else True:
            self.final_error = str(error) if error else "Unknown error"

        return attempt

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for manifest storage."""
        return {
            "stage_name": self.stage_name,
            "row_id": self.row_id,
            "attempt_count": self.attempt_count,
            "final_success": self.final_success,
            "final_error": self.final_error,
            "total_duration_ms": self.total_duration_ms,
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "started_at": a.started_at.isoformat(),
                    "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                    "success": a.success,
                    "error": a.error,
                    "error_category": a.error_category.value,
                    "duration_ms": a.duration_ms,
                }
                for a in self.attempts
            ],
        }


def categorize_error(error: Exception) -> ErrorCategory:
    """
    Categorize an error for reporting.

    Args:
        error: The exception to categorize.

    Returns:
        Error category.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # OOM detection
    if "out of memory" in error_str or "oom" in error_str or "cuda" in error_str:
        return ErrorCategory.OOM

    # Model loading errors
    if "model" in error_str and ("load" in error_str or "download" in error_str):
        return ErrorCategory.MODEL_LOAD

    # Parse failures
    if "json" in error_str or "parse" in error_str or "decode" in error_str:
        return ErrorCategory.PARSE_FAILURE

    # Network errors
    if "connection" in error_type or "timeout" in error_type:
        return ErrorCategory.NETWORK

    if "timeout" in error_str:
        return ErrorCategory.TIMEOUT

    # Ray-specific errors
    if "ray" in error_str or "actor" in error_str or "worker" in error_str:
        return ErrorCategory.TRANSIENT_RAY

    # Data errors
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return ErrorCategory.DATA_ERROR

    return ErrorCategory.UNKNOWN


def with_retry(
    policy: RetryPolicy | None = None,
    stage_name: str = "unknown",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add retry behavior to a function.

    Args:
        policy: Retry policy to use. Defaults to RetryPolicy().
        stage_name: Name of the stage for logging.

    Returns:
        Decorated function with retry behavior.

    Example:
        >>> @with_retry(RetryPolicy(max_attempts=3))
        ... def process_row(row):
        ...     return expensive_operation(row)
    """
    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            context = RetryContext(stage_name=stage_name, policy=policy)
            last_error: Exception | None = None

            for attempt in range(1, policy.max_attempts + 1):
                started_at = datetime.utcnow()

                try:
                    result = func(*args, **kwargs)
                    context.record_attempt(success=True, started_at=started_at)
                    return result

                except Exception as e:
                    last_error = e
                    context.record_attempt(success=False, error=e, started_at=started_at)

                    if not policy.should_retry(e, attempt):
                        raise

                    if attempt < policy.max_attempts:
                        delay = policy.compute_delay(attempt)
                        time.sleep(delay)

            # Should not reach here, but just in case
            if last_error:
                raise last_error
            raise RuntimeError("Retry exhausted without error")

        return wrapper

    return decorator


class RetryExecutor:
    """
    Executor for running functions with retry and tracking.

    Provides more control than the decorator for complex scenarios.
    """

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        on_retry: Callable[[RetryContext, Exception], None] | None = None,
        on_success: Callable[[RetryContext], None] | None = None,
        on_failure: Callable[[RetryContext, Exception], None] | None = None,
    ):
        """
        Initialize retry executor.

        Args:
            policy: Retry policy to use.
            on_retry: Callback before each retry.
            on_success: Callback on success.
            on_failure: Callback on final failure.
        """
        self.policy = policy or RetryPolicy()
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_failure = on_failure

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        stage_name: str = "unknown",
        row_id: str | None = None,
        **kwargs: Any,
    ) -> tuple[T | None, RetryContext]:
        """
        Execute function with retry.

        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            stage_name: Name of stage for context.
            row_id: Optional row ID for context.
            **kwargs: Keyword arguments for function.

        Returns:
            Tuple of (result or None, RetryContext).

        Raises:
            Exception: The last exception if all retries fail.
        """
        context = RetryContext(
            stage_name=stage_name,
            row_id=row_id,
            policy=self.policy,
        )
        last_error: Exception | None = None

        for attempt in range(1, self.policy.max_attempts + 1):
            started_at = datetime.utcnow()

            try:
                result = func(*args, **kwargs)
                context.record_attempt(success=True, started_at=started_at)

                if self.on_success:
                    self.on_success(context)

                return result, context

            except Exception as e:
                last_error = e
                context.record_attempt(success=False, error=e, started_at=started_at)

                if not self.policy.should_retry(e, attempt):
                    if self.on_failure:
                        self.on_failure(context, e)
                    raise

                if attempt < self.policy.max_attempts:
                    if self.on_retry:
                        self.on_retry(context, e)
                    delay = self.policy.compute_delay(attempt)
                    time.sleep(delay)

        # Final failure
        if self.on_failure and last_error:
            self.on_failure(context, last_error)

        if last_error:
            raise last_error

        raise RuntimeError("Retry exhausted without error")


# Default policies for common scenarios
CONSERVATIVE_RETRY = RetryPolicy(
    max_attempts=5,
    initial_delay_seconds=2.0,
    max_delay_seconds=120.0,
    exponential_base=2.0,
)

AGGRESSIVE_RETRY = RetryPolicy(
    max_attempts=10,
    initial_delay_seconds=0.5,
    max_delay_seconds=30.0,
    exponential_base=1.5,
)

NO_RETRY = RetryPolicy(max_attempts=1)
