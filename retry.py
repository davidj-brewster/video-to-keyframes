"""
Retry management for video processing operations with proper exponential backoff.
"""
import asyncio
import logging
import random
from typing import Callable, Optional, Set, TypeVar
from weakref import WeakSet
from config import VideoConfig
from core.errors import VideoError

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RetryManager:
    """Enhanced retry manager with proper exponential backoff and jitter."""

    def __init__(self, config: VideoConfig):
        self.config = config
        self._cleanup_registry: WeakSet[Callable] = WeakSet()
        self._active_operations: Set[str] = set()
        self._base_delay = config.retry_delay
        self._max_delay = 30.0  # Maximum delay between retries
        logger.info("Retry manager initialized with %d max attempts",
                   self.config.retry_attempts)

    def register_cleanup(self, operation_id: str, cleanup_func: Callable) -> None:
        """Register cleanup function for operation."""
        self._cleanup_registry.add(cleanup_func)
        self._active_operations.add(operation_id)
        logger.debug("Registered cleanup for operation: %s", operation_id)

    def unregister_cleanup(self, operation_id: str) -> None:
        """Unregister cleanup for completed operation."""
        self._active_operations.discard(operation_id)
        logger.debug("Unregistered cleanup for operation: %s", operation_id)

    def cleanup_all(self) -> None:
        """Execute all registered cleanup functions."""
        for cleanup_func in self._cleanup_registry:
            try:
                cleanup_func()
            except Exception as e:
                logger.error("Cleanup failed: %s", e, exc_info=True)

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate retry delay using exponential backoff with jitter.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            float: Delay in seconds
        """
        # Calculate exponential backoff
        delay = min(self._max_delay, self._base_delay * (2 ** attempt))

        # Add random jitter (±10% of delay)
        jitter = delay * 0.1
        delay += random.uniform(-jitter, jitter)

        return max(0, delay)  # Ensure non-negative delay

    async def retry_operation(
        self,
        operation_id: str,
        operation: Callable[..., T],
        cleanup_func: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> T:
        """Retry operation with exponential backoff and proper cleanup."""
        if cleanup_func:
            self.register_cleanup(operation_id, cleanup_func)

        last_exception = None

        for attempt in range(self.config.retry_attempts):
            try:
                result = await operation(*args, **kwargs)
                self.unregister_cleanup(operation_id)
                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    "Operation %s failed (attempt %d/%d): %s",
                    operation_id, attempt + 1, self.config.retry_attempts, e
                )

                if attempt < self.config.retry_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.debug(
                        "Retrying operation %s in %.2f seconds",
                        operation_id, delay
                    )
                    await asyncio.sleep(delay)

        # All retries failed
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as e:
                logger.error("Cleanup failed for %s: %s", operation_id, e)

        raise VideoError(
            f"Operation {operation_id} failed after {self.config.retry_attempts} attempts"
        ) from last_exception
