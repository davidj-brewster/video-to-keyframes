"""
Enhanced frame buffer with optimized memory management and minimal copying.
"""
import logging
import threading
from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from core.types import Frame
from models.metadata import FrameMetadata
from config import VideoConfig

class FrameBuffer:
    """Thread-safe frame buffer with optimized memory management."""

    def __init__(self, config: VideoConfig):
        """
        Initialize frame buffer.

        Args:
            config: Video processing configuration
        """
        self.config = config
        self._buffer: Dict[int, Tuple[Frame, FrameMetadata]] = {}
        self._lock = threading.Lock()
        self._memory_usage = 0
        self._logger = logging.getLogger(__name__)

    def add_frame(
        self,
        frame_number: int,
        frame: Frame,
        metadata: FrameMetadata
    ) -> bool:
        """
        Add frame to buffer with optimized memory handling.

        Uses view operations instead of copies where possible and
        properly accounts for all memory usage including metadata.

        Args:
            frame_number: Frame sequence number
            frame: Frame data
            metadata: Frame metadata

        Returns:
            bool: True if frame was added, False if skipped due to memory constraints
        """
        with self._lock:
            # Calculate total memory including metadata
            metadata_size = metadata.__sizeof__()
            frame_size = frame.nbytes
            total_size = frame_size + metadata_size

            # Check memory limit
            if (self.config.max_memory_usage is not None and
                self._memory_usage + total_size > self.config.max_memory_usage):
                self._logger.warning(
                    f"Memory limit reached. Skipping frame {frame_number}"
                )
                return False

            # Remove oldest frames if buffer full
            while len(self._buffer) >= self.config.buffer_size:
                oldest = min(self._buffer.keys())
                self._remove_frame(oldest)

            # Create memory-efficient view if possible
            if frame.flags['C_CONTIGUOUS']:
                # Use view for contiguous arrays
                frame_data = frame.view()
            else:
                # Create contiguous copy only if necessary
                frame_data = np.ascontiguousarray(frame)

            frame_data.flags.writeable = False  # Prevent modifications

            # Store frame and update memory tracking
            self._buffer[frame_number] = (frame_data, metadata)
            self._memory_usage += total_size

            self._logger.debug(
                "Added frame %d. Buffer size: %d frames, %.2f MB",
                frame_number,
                len(self._buffer),
                self._memory_usage / (1024 * 1024)
            )

            return True

    def get_frame(
        self,
        frame_number: int
    ) -> Optional[Tuple[Frame, FrameMetadata]]:
        """
        Thread-safe frame retrieval with copy-on-read.

        Args:
            frame_number: Frame number to retrieve

        Returns:
            Tuple of (frame, metadata) if found, None otherwise
        """
        with self._lock:
            data = self._buffer.get(frame_number)
            if data is None:
                return None

            frame, metadata = data
            # Return copy of frame to prevent buffer modifications
            return frame.copy(), metadata

    def _remove_frame(self, frame_number: int) -> None:
        """
        Remove frame and update memory tracking.

        Args:
            frame_number: Frame number to remove
        """
        if frame_number in self._buffer:
            frame, metadata = self._buffer[frame_number]
            self._memory_usage -= (frame.nbytes + metadata.__sizeof__())
            del self._buffer[frame_number]

    def clear(self) -> None:
        """Clear buffer and reset memory usage."""
        with self._lock:
            self._buffer.clear()
            self._memory_usage = 0
            self._logger.debug("Buffer cleared")

    @property
    def frames(self) -> Dict[int, Tuple[Frame, FrameMetadata]]:
        """
        Thread-safe access to frame buffer.

        Returns:
            Dictionary mapping frame numbers to (frame, metadata) tuples
        """
        with self._lock:
            # Return deep copy to prevent external modifications
            return {
                k: (v[0].copy(), v[1])
                for k, v in self._buffer.items()
            }

    @property
    def memory_usage(self) -> int:
        """
        Get current memory usage in bytes.

        Returns:
            Current memory usage in bytes
        """
        with self._lock:
            return self._memory_usage
