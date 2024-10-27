"""
Video resource management with enhanced error handling and thread safety.

Provides safe access to video resources with automatic cleanup and caching.
"""

import logging
import os
from pathlib import Path
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union
import signal

import cv2
import numpy as np
from numpy.typing import NDArray

from config import VideoConfig
from core.errors import VideoError, ResourceError, CodecError, TimeoutError, FrameError

# Configure module logger
logger = logging.getLogger(__name__)

# Type alias for clarity
Frame = NDArray[np.uint8]

@contextmanager
def timeout(seconds: float):
    """
    Platform-independent timeout context manager.

    Args:
        seconds: Timeout duration in seconds

    Raises:
        TimeoutError: If operation exceeds timeout
    """
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out", timeout_duration=seconds)

    if os.name == 'nt':  # Windows
        timer = threading.Timer(seconds, lambda: timeout_handler(None, None))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    else:  # Unix
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)


class VideoResource:
    """
    Thread-safe video resource manager.

    Provides:
    - Safe video file access with timeouts
    - Automatic resource cleanup
    - Frame caching with memory limits
    - Thread safety
    - Format validation
    """

    def __init__(self, path: Union[str, Path], config: VideoConfig):
        """
        Initialize video resource.

        Args:
            path: Path to video file
            config: Video configuration

        Raises:
            ValueError: If path is empty or config is invalid
        """
        if not path:
            raise ValueError("Video path cannot be empty")

        self.path = Path(path)
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._codec_info: Optional[Dict[str, any]] = None
        self._lock = threading.Lock()
        self._frame_cache: Dict[int, Frame] = {}
        self._total_cache_size = 0  # Track memory usage in bytes

        logger.info("Initializing video resource: %s", self.path)

    def __enter__(self) -> 'VideoResource':
        """
        Open video resource with validation.

        Returns:
            Self for context manager usage

        Raises:
            FileNotFoundError: If video file doesn't exist
            ResourceError: If video cannot be opened
            VideoError: For other initialization errors
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        try:
            with timeout(self.config.frame_timeout):  # Use frame timeout for opening
                self._capture = cv2.VideoCapture(str(self.path))
                if not self._capture.isOpened():
                    raise ResourceError(
                        "Failed to open video",
                        resource_path=str(self.path)
                    )
                self._validate_format()
                logger.info("Video resource opened successfully: %s", self.path)
        except Exception as e:
            if self._capture is not None:
                self._capture.release()
            logger.error("Failed to initialize video: %s", e, exc_info=True)
            raise VideoError(f"Failed to initialize video: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Release video resources with cleanup.

        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception value if error occurred
            exc_tb: Exception traceback if error occurred
        """
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as e:
                logger.error("Error releasing video capture: %s", e)
            finally:
                self._capture = None
        self._clear_cache()
        logger.info("Video resource released: %s", self.path)

    def _clear_cache(self) -> None:
        """Clear frame cache and reset memory tracking."""
        self._frame_cache.clear()
        self._total_cache_size = 0

    def _update_cache(self, frame_number: int, frame: Frame) -> None:
        """
        Update frame cache with memory management.

        Args:
            frame_number: Frame number to cache
            frame: Frame data to cache
        """
        if not self.config.enable_cache:
            return

        frame_size = frame.nbytes

        # Check memory limit
        if (self.config.max_memory_usage is not None and
            self._total_cache_size + frame_size > self.config.max_memory_usage):
            # Remove oldest frames until we have space
            while (len(self._frame_cache) > 0 and
                   self._total_cache_size + frame_size > self.config.max_memory_usage):
                oldest = min(self._frame_cache.keys())
                self._remove_from_cache(oldest)

        # Remove oldest frame if cache is full
        if len(self._frame_cache) >= self.config.cache_size:
            oldest = min(self._frame_cache.keys())
            self._remove_from_cache(oldest)

        # Add new frame
        self._frame_cache[frame_number] = frame.copy()
        self._total_cache_size += frame_size
        logger.debug(
            "Cached frame %d. Cache size: %d frames, %d bytes",
            frame_number, len(self._frame_cache), self._total_cache_size
        )

    def _remove_from_cache(self, frame_number: int) -> None:
        """
        Remove frame from cache and update memory tracking.

        Args:
            frame_number: Frame number to remove
        """
        if frame_number in self._frame_cache:
            frame = self._frame_cache[frame_number]
            self._total_cache_size -= frame.nbytes
            del self._frame_cache[frame_number]

    def read_frame(
        self,
        frame_number: Optional[int] = None
    ) -> Tuple[bool, Optional[Frame]]:
        """
        Read frame with thread safety and caching.

        Args:
            frame_number: Optional specific frame to read. If None, reads next frame.

        Returns:
            Tuple[bool, Optional[Frame]]: (success flag, frame if successful)

        Raises:
            ResourceError: If frame reading fails
            TimeoutError: If read operation times out
            FrameError: If frame number is invalid
        """
        with self._lock:
            try:
                if frame_number is not None:
                    # Validate frame number
                    if frame_number < 0 or (
                        self._codec_info and
                        frame_number >= self._codec_info['frame_count']
                    ):
                        raise FrameError(
                            "Invalid frame number",
                            frame_number=frame_number
                        )

                    # Check cache
                    if frame_number in self._frame_cache:
                        logger.debug("Cache hit for frame %d", frame_number)
                        return True, self._frame_cache[frame_number].copy()

                    # Seek to frame
                    with timeout(self.config.frame_timeout):
                        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number):
                            raise ResourceError(
                                "Failed to seek to frame",
                                resource_path=str(self.path)
                            )

                # Read frame
                with timeout(self.config.frame_timeout):
                    ret, frame = self._capture.read()

                    if not ret:
                        return False, None

                    # Cache frame if needed
                    if frame_number is not None:
                        self._update_cache(frame_number, frame)

                    return True, frame

            except TimeoutError as e:
                logger.error("Frame read timed out for frame %s", frame_number)
                raise TimeoutError(
                    f"Frame read timed out: {e}",
                    timeout_duration=self.config.frame_timeout
                )
            except Exception as e:
                logger.error(
                    "Error reading frame %s: %s",
                    frame_number if frame_number is not None else 'next',
                    e,
                    exc_info=True
                )
                raise

    def _validate_format(self) -> None:
        """
        Validate video format and codec configuration.

        Raises:
            ResourceError: If video capture not initialized
            CodecError: If video format is invalid
            TimeoutError: If validation times out
        """
        if self._capture is None:
            raise ResourceError(
                "Video capture not initialized",
                resource_path=str(self.path)
            )

        try:
            with timeout(self.config.frame_timeout):
                # Get video properties
                width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self._capture.get(cv2.CAP_PROP_FPS)
                frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

                if width <= 0 or height <= 0:
                    raise CodecError(
                        "Invalid video dimensions",
                        codec=self._get_codec_string()
                    )
                if fps <= 0:
                    raise CodecError(
                        "Invalid frame rate",
                        codec=self._get_codec_string()
                    )
                if frame_count <= 0:
                    raise CodecError(
                        "Invalid frame count",
                        codec=self._get_codec_string()
                    )

                # Get codec information
                fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
                codec_str = self._get_codec_string(fourcc)

                # Validate first frame can be read
                ret, _ = self.read_frame()
                if not ret:
                    raise CodecError(
                        "Cannot read video frames",
                        codec=codec_str
                    )

                self._codec_info = {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'frame_count': frame_count,
                    'fourcc': fourcc,
                    'codec': codec_str
                }

                logger.info(
                    "Video format validated - Dimensions: %dx%d, FPS: %.2f, "
                    "Frames: %d, Codec: %s",
                    width, height, fps, frame_count, codec_str
                )

        except TimeoutError:
            raise TimeoutError(
                "Format validation timed out",
                timeout_duration=self.config.frame_timeout
            )
        except Exception as e:
            logger.error("Format validation failed: %s", e, exc_info=True)
            raise

    def _get_codec_string(self, fourcc: Optional[int] = None) -> str:
        """
        Get string representation of codec.

        Args:
            fourcc: Optional fourcc code. If None, reads from capture.

        Returns:
            str: Codec string
        """
        if fourcc is None:
            fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def codec_info(self) -> Optional[Dict[str, any]]:
        """Get codec information if available."""
        return self._codec_info.copy() if self._codec_info else None
