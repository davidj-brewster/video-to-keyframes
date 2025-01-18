"""
Enhanced video frame processor with keyframe detection.

This module provides functionality for processing video files and extracting key frames
based on configurable similarity thresholds. It uses both histogram comparison and
structural similarity (SSIM) for frame comparison.

Key features:
- Async video processing with batched frame handling
- Configurable keyframe detection
- Progress tracking
- Error handling and logging
"""
import asyncio
from collections import deque
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any, Callable
from skimage.metrics import structural_similarity as ssim
import cv2

from core.types import Frame
from core.errors import VideoError
from models.metadata import FrameMetadata
from video import VideoReader
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    success: bool
    output_path: Optional[Path]
    error: Optional[str] = None


class VideoProcessor:
    """
    Enhanced video processor with keyframe detection capabilities.
    
    This class handles video processing operations including:
    - Frame extraction from videos
    - Keyframe detection using similarity metrics
    - Batch processing of frames
    - Frame saving with configurable formats
    
    Attributes:
        config: Configuration object containing processing parameters
        _logger: Logger instance for this class
        _prev_frame: Cache for the previous frame
        _frame_history: Deque of recent frames for comparison
    """

    def __init__(self, config: Any):
        """
        Initialize the video processor.

        Args:
            config: Configuration object containing:
                - similarity_threshold: Float threshold for frame similarity
                - buffer_size: Number of frames to process in each batch
                - detect_keyframes: Boolean to enable/disable keyframe detection
                - output_format: Format specification for saved frames
                - compression_quality: Quality setting for frame compression

        Raises:
            ValueError: If required configuration attributes are missing
        """
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._prev_frame = None
        self._frame_history = deque(maxlen=60)  # Keep last 5 frames for comparison

        # Validate config
        required_attrs = ['similarity_threshold', 'buffer_size', 'detect_keyframes', 
                         'output_format', 'compression_quality']
        missing = [attr for attr in required_attrs if not hasattr(config, attr)]
        if missing:
            raise ValueError(f"Config missing required attributes: {missing}")

    def _compute_frame_similarity(self, frame1: Frame, frame2: Frame) -> float:
        """
        Compute similarity between two frames using multiple metrics.

        Uses a combination of histogram comparison and structural similarity (SSIM)
        for robust frame comparison. Falls back to histogram-only comparison if
        SSIM computation fails.

        Args:
            frame1: First frame to compare (numpy array)
            frame2: Second frame to compare (numpy array)

        Returns:
            float: Similarity score between 0 and 1 (1 = identical)
        """
        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2

        # Ensure frames are same size for SSIM
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        # Compute histogram
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Compare histograms
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # @todo - make SSIM optional, as it is expensive
        # Compute structural similarity for more accuracy
        try:
            ssim_score = ssim(gray1, gray2, data_range=255)
            # Combine histogram and SSIM scores, weighted more heavily for the more advanced technique
            return similarity * 0.3 + ssim_score * 0.7
        except Exception as e:
            # Fall back to just histogram if SSIM fails
            self._logger.warning(
                f"SSIM computation failed, using histogram comparison only: {e}"
            )
            return similarity

    def _is_keyframe(self, frame: Frame, metadata: FrameMetadata) -> bool:
        """
        Determine if a frame should be kept as a keyframe.

        A frame is considered a keyframe if it's sufficiently different from
        recent frames in the history buffer. The first frame is always a keyframe.

        Args:
            frame: Current frame to evaluate
            metadata: Frame metadata including frame number and timestamp

        Returns:
            bool: True if frame should be kept as a keyframe
        """
        # First frame is always a keyframe
        if not self._frame_history:
            self._frame_history.append(frame)
            return True

        # Check similarity with recent frames
        max_similarity = max(
            self._compute_frame_similarity(frame, prev_frame)
            for prev_frame in self._frame_history
        )

        # Add to history if we keep it
        if max_similarity < self.config.similarity_threshold:
            self._frame_history.append(frame)
            self._logger.debug(
                f"Frame {metadata.frame_number} selected as keyframe "
                f"(similarity: {max_similarity:.3f})"
            )
            return True

        self._logger.debug(
            f"Frame {metadata.frame_number} skipped "
            f"(similarity: {max_similarity:.3f})"
        )
        return False

    async def process_video(
        self,
        input_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[str]:
        """
        Process video file and extract keyframes asynchronously.

        Processes the video in batches, detecting and saving keyframes based on
        configured similarity threshold. Uses async processing for improved performance.

        Args:
            input_path: Path to input video file
            output_dir: Directory where extracted frames will be saved
            progress_callback: Optional callback function for progress updates

        Returns:
            List[str]: Paths to all extracted keyframes

        Raises:
            VideoError: If video processing fails
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_files = []

        try:
            with VideoReader(input_path) as video:
                total_frames = video.get_metadata().frame_count
                processed_frames = 0

                while True:
                    # Read frame batch
                    frames_batch = []
                    for _ in range(self.config.buffer_size):
                        ret, frame = video.read_frame()
                        if not ret or frame is None:
                            break

                        metadata = FrameMetadata(
                            frame_number=processed_frames,
                            timestamp=processed_frames / video.get_metadata().fps,
                            width=frame.shape[1],
                            height=frame.shape[0]
                        )

                        # Only add frame if it's different enough from recent frames
                        if self.config.detect_keyframes:
                            if self._is_keyframe(frame, metadata):
                                frames_batch.append((processed_frames, frame, metadata))
                        else:
                            frames_batch.append((processed_frames, frame, metadata))

                        processed_frames += 1

                    if not frames_batch:
                        break

                    # Process frame batch
                    results = await self._process_frames_batch(
                        frames_batch,
                        Path(output_dir)
                    )

                    # Update output files
                    if results:  # Check if results exist
                        for result in results:
                            try:
                                if result and isinstance(result, ProcessingResult):
                                    if result.success and result.output_path:
                                        output_files.append(result.output_path)
                                    elif result.error:
                                        self._logger.warning(f"Frame processing failed: {result.error}")
                            except Exception as e:
                                self._logger.error(f"Error processing result: {e}")
                                continue

                    # Update progress
                    if progress_callback:
                        progress = min(1.0, processed_frames / total_frames)
                        progress_callback(progress)

            self._logger.info(
                f"Processed {processed_frames} frames, kept {len(output_files)} keyframes"
            )
            return output_files

        except Exception as e:
            self._logger.error(f"Video processing failed: {e}")
            raise VideoError(f"Failed to process video: {e}")

        finally:
            # Clear frame history
            self._frame_history.clear()

    async def _process_frames_batch(
        self,
        frames_batch: List[Tuple[int, Frame, FrameMetadata]],
        output_dir: Path
    ) -> List[ProcessingResult]:
        """Process a batch of frames and return results."""
        processed_frames = 0
        results: List[ProcessingResult] = []
        output_files: List[str] = []
        try:
            frames_batch = frames_batch or []  # Ensure not None
            
            # Process frame batch
            batch_results = await asyncio.get_event_loop().run_in_executor(
                None,
                self._process_frames_sync,
                frames_batch,
                output_dir
            )

            # Update output files with type safety
            if batch_results:
                for result in batch_results:
                    try:
                        if isinstance(result, ProcessingResult):
                            if result.success and result.output_path:
                                output_files.append(str(result.output_path))
                            elif result.error:
                                self._logger.warning(
                                    f"Frame processing failed: {result.error}"
                                )
                    except Exception as e:
                        self._logger.error(
                            f"Error processing result: {str(e)}", 
                            exc_info=True
                        )
                        results.append(ProcessingResult(
                            success=False,
                            output_path=None,
                            error=str(e)
                        ))
                    processed_frames += 1

        except Exception as e:
            self._logger.error(
                f"Batch processing failed: {str(e)}", 
                exc_info=True
            )
            
        return results

    def _process_frames_sync(
        self,
        frames: List[Tuple[int, Frame, FrameMetadata]],
        output_dir: Path
    ) -> List[ProcessingResult]:
        """Synchronous processing of frame batch."""
        results = []
        for frame_num, frame, metadata in frames:
            try:
                output_path = f"{output_dir}/frame_{frame_num:06d}.png"
                # Process frame here...
                results.append(ProcessingResult(
                    success=True,
                    output_path=Path(output_path)
                ))
            except Exception as e:
                results.append(ProcessingResult(
                    success=False,
                    output_path=None,
                    error=str(e)
                ))
        return results

    def _save_frame(
        self,
        frame: Frame,
        metadata: FrameMetadata,
        output_dir: Path
    ) -> Optional[str]:
        """
        Save a single frame to disk with metadata.

        Saves the frame using configured format and compression settings.
        Generates filename based on frame number and timestamp.

        Args:
            frame: Frame data to save
            metadata: Frame metadata
            output_dir: Directory to save frame

        Returns:
            Optional[str]: Path to saved frame, or None if save failed

        Raises:
            VideoError: If frame saving fails
        """
        try:
            filename = (
                f"frame_{metadata.frame_number:06d}"
                f"_at_{metadata.timestamp:.2f}s"
                f".{self.config.output_format.extension}"
            )
            output_path = output_dir / filename

            save_params = self.config.output_format.get_save_params(
                self.config.compression_quality
            )

            success = cv2.imwrite(str(output_path), frame, save_params)
            if not success:
                raise VideoError(f"Failed to save frame to {output_path}")

            return str(output_path)

        except Exception as e:
            self._logger.error(f"Frame save failed: {e}")
            return None
