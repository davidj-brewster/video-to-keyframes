"""
Custom exceptions for video processing operations.

Provides a hierarchy of specific exceptions for different types of failures
that can occur during video processing operations.
"""

class VideoError(Exception):
    """Base exception for all video processing errors."""
    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class CodecError(VideoError):
    """Raised when there are issues with video codecs or formats."""
    def __init__(self, message: str, codec: str = None, *args: object) -> None:
        super().__init__(message, *args)
        self.codec = codec


class ResourceError(VideoError):
    """Raised when there are issues managing video resources."""
    def __init__(self, message: str, resource_path: str = None, *args: object) -> None:
        super().__init__(message, *args)
        self.resource_path = resource_path


class TimeoutError(VideoError):
    """Raised when video operations exceed their time limit."""
    def __init__(self, message: str, timeout_duration: float = None, *args: object) -> None:
        super().__init__(message, *args)
        self.timeout_duration = timeout_duration


class FrameError(VideoError):
    """Raised when there are issues processing individual frames."""
    def __init__(self, message: str, frame_number: int = None, *args: object) -> None:
        super().__init__(message, *args)
        self.frame_number = frame_number
