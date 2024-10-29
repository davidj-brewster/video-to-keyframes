"""
Core type definitions and base classes.

This module defines the fundamental types and protocols used throughout the video
processing system, ensuring type safety and consistent interfaces.
"""
from dataclasses import dataclass
from typing import TypeVar, Protocol, runtime_checkable, Optional, Dict, Any, Tuple, List
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from models.metadata import FrameMetadata, VideoMetadata, AnalysisResult

# Type variables for generic operations
T = TypeVar('T')
R = TypeVar('R')

# Basic frame types
Frame = NDArray[np.uint8]  # Raw video frame
RGBFrame = NDArray[np.uint8]  # RGB color frame
GrayFrame = NDArray[np.uint8]  # Grayscale frame
Features = NDArray[np.float32]  # Feature vectors

# Processing state enums
class ProcessingState(Enum):
    """States for processing operations."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@runtime_checkable
class FrameProcessor(Protocol):
    """Protocol for frame processors."""
    def process_frame(self, frame: Frame) -> Frame:
        """
        Process a single frame.

        Args:
            frame: Input frame

        Returns:
            Processed frame
        """
        ...

@runtime_checkable
class FrameAnalyzer(Protocol):
    """Protocol for frame analyzers."""
    def analyze_frame(self, frame: Frame) -> float:
        """
        Analyze a single frame.

        Args:
            frame: Input frame

        Returns:
            Analysis score
        """
        ...

@runtime_checkable
class FrameFilter(Protocol):
    """Protocol for frame filters."""
    def should_keep_frame(self, frame: Frame, metadata: 'FrameMetadata') -> bool:
        """
        Determine if frame should be kept.

        Args:
            frame: Input frame
            metadata: Frame metadata

        Returns:
            True if frame should be kept
        """
        ...

@runtime_checkable
class VideoReader(Protocol):
    """Protocol for video readers."""
    def read_frame(self) -> Tuple[bool, Optional[Frame]]:
        """
        Read next frame from video.

        Returns:
            Tuple of (success flag, frame if successful)
        """
        ...

    def get_metadata(self) -> 'VideoMetadata':
        """
        Get video metadata.

        Returns:
            Video metadata
        """
        ...

@dataclass
class ProcessingResults:
    """Container for processing results."""
    frames_processed: int = 0
    frames_kept: int = 0
    processing_time: float = 0.0
    output_files: List[str] = None
    errors: List[Exception] = None
    state: ProcessingState = ProcessingState.IDLE
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.output_files is None:
            self.output_files = []
        if self.errors is None:
            self.errors = []
        if self.additional_info is None:
            self.additional_info = {}

__all__ = [
    # Type variables
    'T', 'R',

    # Frame types
    'Frame', 'RGBFrame', 'GrayFrame', 'Features',

    # Enums
    'ProcessingState',

    # Protocols
    'FrameProcessor', 'FrameAnalyzer', 'FrameFilter', 'VideoReader',

    # Data classes
    'ProcessingResults',

    # Re-exported types
    'FrameMetadata', 'VideoMetadata', 'AnalysisResult',
]
