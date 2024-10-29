"""
Main model coordination and integration.
"""
from typing import List, Optional, Callable
import logging
from core.errors import VideoError
from models.metadata import FrameMetadata, VideoMetadata
from analysis.analyzer import FrameAnalyzer
from video import VideoReader
from processor import VideoProcessor

class FrameExtractionModel:
    """
    Main model coordinating frame extraction and analysis.

    This class serves as the main integration point, coordinating:
    - Video I/O
    - Frame analysis
    - Processing decisions
    - Result management
    """

    def __init__(self, config):
        """Initialize model components."""
        logger = logging.getLogger(__name__)
        self.config = config
        self.analyzer = FrameAnalyzer(config)
        self.processor = VideoProcessor(config)
        self._metadata: Optional[VideoMetadata] = None
        logger.debug(f"FrameExtractionModel instantiated with config: {self.config}")

    async def process_video(
        self,
        input_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float], None]]
    ) -> List[str]:
        """
        Process video file and extract frames.

        Args:
            input_path: Path to input video
            output_dir: Directory for output frames
            progress_callback: Optional progress reporting function

        Returns:
            List of paths to extracted frames

        Raises:
            VideoError: If processing fails
        """
        logger = logging.getLogger(__name__)
        try:
            # Read video metadata
            with VideoReader(input_path) as reader:
                self._metadata = reader.get_metadata()

            # Process video
            output_files = await self.processor.process_video(
                input_path,
                output_dir,
                progress_callback
            )
            logger.debug(f"process_video completed with files {output_files} based on {self.config}")
            return output_files

        except Exception as e:
            raise VideoError(f"Video processing failed: {e}") from e

    @property
    def metadata(self) -> Optional[VideoMetadata]:
        """Get video metadata if available."""
        return self._metadata
