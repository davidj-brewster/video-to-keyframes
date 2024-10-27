"""
Video I/O operations with enhanced error handling and metadata extraction.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2

from core.errors import VideoError, CodecError
from core.types import Frame
from models.metadata import VideoMetadata
from config import VideoConfig

logger = logging.getLogger(__name__)

class VideoReader:
    """Enhanced video reader with metadata extraction and error handling."""

    def __init__(self, path: str, config: Optional[VideoConfig] = None):
        """
        Initialize video reader.

        Args:
            path: Path to video file
            config: Optional video configuration
        """
        self.path = Path(path)
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def __enter__(self) -> 'VideoReader':
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """
        Open video file and validate format.

        Raises:
            VideoError: If video cannot be opened
            CodecError: If video format is invalid
        """
        if not self.path.exists():
            raise VideoError(f"Video file not found: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise VideoError(f"Failed to open video: {self.path}")

        self._validate_format()

    def close(self) -> None:
        """Close video file and release resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read_frame(self) -> Tuple[bool, Optional[Frame]]:
        """
        Read next frame from video.

        Returns:
            Tuple of (success flag, frame if successful)

        Raises:
            VideoError: If video is not opened
        """
        if self._capture is None:
            raise VideoError("Video is not opened")

        ret, frame = self._capture.read()
        return ret, frame if ret else None

    def get_metadata(self) -> VideoMetadata:
        """
        Get video metadata.

        Returns:
            VideoMetadata object

        Raises:
            VideoError: If metadata cannot be extracted
        """
        if self._metadata is None:
            if self._capture is None:
                raise VideoError("Video is not opened")

            # Extract basic properties
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self._capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # Get format information
            fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            # Read first frame to determine color
            ret, frame = self._capture.read()
            if not ret:
                raise VideoError("Cannot read video frames")

            is_color = len(frame.shape) == 3

            # Reset position
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self._metadata = VideoMetadata(
                filename=self.path.name,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=frame_count / fps if fps > 0 else 0,
                codec=codec,
                is_color=is_color,
                format=self.path.suffix[1:].lower()
            )

        return self._metadata

    def _validate_format(self) -> None:
        """
        Validate video format.

        Raises:
            CodecError: If video format is invalid
        """
        metadata = self.get_metadata()

        if metadata.width <= 0 or metadata.height <= 0:
            raise CodecError(
                "Invalid video dimensions",
                codec=metadata.codec
            )

        if metadata.fps <= 0:
            raise CodecError(
                "Invalid frame rate",
                codec=metadata.codec
            )

        if metadata.frame_count <= 0:
            raise CodecError(
                "Invalid frame count",
                codec=metadata.codec
            )
