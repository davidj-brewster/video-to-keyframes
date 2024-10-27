"""
Configuration management for video processing operations.

Provides structured configuration options with validation and sensible defaults
for video processing operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import logging
import cv2

logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    """
    Supported output formats with their specific configurations.

    Each format includes:
    - File extension
    - Compression parameters
    - Whether quality settings apply
    """

    PNG = ("png", [cv2.IMWRITE_PNG_COMPRESSION], False)  # PNG uses compression level, not quality
    JPEG = ("jpeg", [cv2.IMWRITE_JPEG_QUALITY], True)
    WEBP = ("webp", [cv2.IMWRITE_WEBP_QUALITY], True)

    def __init__(self, extension: str, params: list, uses_quality: bool):
        self.extension = extension
        self.params = params
        self.uses_quality = uses_quality

    @classmethod
    def from_string(cls, format_str: str) -> 'OutputFormat':
        """Convert string to OutputFormat with validation."""
        try:
            return cls[format_str.upper()]
        except KeyError:
            supported = [f.name.lower() for f in cls]
            raise ValueError(
                f"Unsupported format: {format_str}. "
                f"Supported formats: {', '.join(supported)}"
            )

    def get_save_params(self, quality: int) -> list:
        """Get OpenCV save parameters for this format."""
        if not self.uses_quality:
            if self == OutputFormat.PNG:
                # PNG uses compression level 0-9
                compression = min(9, max(0, quality // 10))
                return [cv2.IMWRITE_PNG_COMPRESSION, compression]
            return []
        return [self.params[0], quality]


@dataclass
class VideoConfig:
    """
    Configuration settings for video processing operations.

    Default values are optimized for common use cases while maintaining
    reasonable resource usage.
    """

    # Output settings
    output_format: OutputFormat = field(default=OutputFormat.PNG)
    compression_quality: int = field(default=9)  # 0-9 for PNG, 0-100 for JPEG/WEBP

    # Processing settings
    detect_keyframes: bool = field(default=True)
    similarity_threshold: float = field(default=0.95)

    # Resource management
    thread_count: int = field(default=1)
    buffer_size: int = field(default=30)
    cache_size: int = field(default=30)  # Match buffer size by default
    enable_cache: bool = field(default=True)
    max_memory_usage: Optional[int] = field(default=None)

    # Error handling
    retry_attempts: int = field(default=1)
    retry_delay: float = field(default=0.5)
    frame_timeout: float = field(default=5.0)  # Timeout for frame operations
    video_timeout: float = field(default=30.0)  # Timeout for full video operations

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
        logger.debug("Video configuration initialized: %s", self.__dict__)

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate compression/quality settings based on format
        if self.output_format == OutputFormat.PNG:
            if not 0 <= self.compression_quality <= 9:
                raise ValueError("PNG compression must be between 0 and 9")
        elif self.output_format in (OutputFormat.JPEG, OutputFormat.WEBP):
            if not 0 <= self.compression_quality <= 100:
                raise ValueError("JPEG/WEBP quality must be between 0 and 100")

        # Validate other parameters
        if self.retry_attempts < 1:
            raise ValueError("Retry attempts must be at least 1")
        if self.retry_delay < 0:
            raise ValueError("Retry delay must be non-negative")
        if self.buffer_size < 1:
            raise ValueError("Buffer size must be positive")
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        if self.frame_timeout <= 0:
            raise ValueError("Frame timeout must be positive")
        if self.video_timeout <= 0:
            raise ValueError("Video timeout must be positive")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.thread_count < 1:
            raise ValueError("Thread count must be at least 1")
        if self.max_memory_usage is not None and self.max_memory_usage <= 0:
            raise ValueError("Max memory usage must be positive")

        # Ensure cache size doesn't exceed buffer size
        if self.cache_size > self.buffer_size:
            logger.warning(
                "Cache size (%d) exceeds buffer size (%d). Setting cache size to %d",
                self.cache_size, self.buffer_size, self.buffer_size
            )
            self.cache_size = self.buffer_size

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VideoConfig':
        """Create configuration from dictionary."""
        # Handle output format conversion
        if 'output_format' in config_dict and isinstance(config_dict['output_format'], str):
            config_dict['output_format'] = OutputFormat.from_string(
                config_dict['output_format']
            )

        # Handle compression quality conversion for PNG
        if ('output_format' in config_dict and
            config_dict['output_format'] == OutputFormat.PNG and
            'compression_quality' in config_dict):
            quality = config_dict['compression_quality']
            if quality > 9:  # Assume it's in 0-100 range and convert
                config_dict['compression_quality'] = quality // 10

        return cls(**config_dict)
