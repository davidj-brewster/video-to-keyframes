"""
Metadata models for video and frame information.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class FrameMetadata:
    """Metadata for individual frames."""
    frame_number: int
    timestamp: float
    width: int
    height: int
    motion_score: Optional[float] = None
    quality_score: Optional[float] = None
    similarity_score: Optional[float] = None
    is_keyframe: bool = False
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.additional_info is None:
            self.additional_info = {}

@dataclass
class VideoMetadata:
    """Metadata for video files."""
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    is_color: bool
    format: str
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.additional_info is None:
            self.additional_info = {}

@dataclass
class AnalysisResult:
    """Result of frame analysis operations."""
    frame_metadata: FrameMetadata
    is_keyframe: bool
    analysis_success: bool
    output_path: Optional[str] = None
    error: Optional[Exception] = None
