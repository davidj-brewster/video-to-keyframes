"""
Inference mode for automatically determining similarity threshold.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from core.errors import VideoError
from models.metadata import VideoMetadata
from config import VideoConfig
from video import VideoReader

@dataclass
class InferenceResult:
    """Result of similarity threshold inference."""
    optimal_threshold: float
    frame_count: int
    iterations: int
    search_path: List[Tuple[float, int]]  # List of (threshold, count) pairs tried

class SimilarityInference:
    """Infers optimal similarity threshold to achieve target frame count."""

    def __init__(self, config: VideoConfig):
        """Initialize inference engine."""
        self.config = config
        self._logger = logging.getLogger(__name__)
        self.max_iterations = 50
        self.tolerance = 0.002  # Allow 0.5% deviation from target

    async def infer_threshold(
        self,
        video_path: str,
        target_frames: int,
        progress_callback: Optional[callable] = None
    ) -> InferenceResult:
        """
        Find similarity threshold that produces desired number of frames.

        Args:
            video_path: Path to video file
            target_frames: Desired number of output frames
            progress_callback: Optional progress callback

        Returns:
            InferenceResult containing optimal threshold and search details
        """
        # Initial video analysis
        with VideoReader(video_path) as reader:
            metadata = reader.get_metadata()
            if target_frames > metadata.frame_count:
                raise VideoError(
                    f"Target frames ({target_frames}) exceeds video length "
                    f"({metadata.frame_count})"
                )

        # Binary search for optimal threshold
        low, high = 0.0, 1.0
        iterations = 0
        search_path = []

        while iterations < self.max_iterations:
            threshold = (low + high) / 2
            self._logger.debug(f"Trying threshold: {threshold:.5f}")

            # Update progress
            if progress_callback:
                progress = iterations / self.max_iterations
                progress_callback(progress)

            # Try current threshold
            frame_count = await self._count_frames_with_threshold(
                video_path, threshold
            )
            search_path.append((threshold, frame_count))

            # Check if we're within tolerance
            error_ratio = abs(frame_count - target_frames) / target_frames
            if error_ratio <= self.tolerance:
                self._logger.info(
                    f"Found acceptable threshold {threshold:.5f} "
                    f"producing {frame_count} frames "
                    f"(target: {target_frames})"
                )
                return InferenceResult(
                    optimal_threshold=threshold,
                    frame_count=frame_count,
                    iterations=iterations + 1,
                    search_path=search_path
                )
            self._logger.debug(f"INFO: n_frames: threshold: {threshold:.5f} producing {frame_count} frames (target: {target_frames})")
            self._logger.debug(f"TRACE: n_frames: search-path: {search_path}")
            # Update search bounds
            if frame_count < target_frames:
                self._logger.debug(f"DEBUG: low (was {low}) being updated to {threshold}")
                low = threshold
            else:
                self._logger.debug(f"DEBUG: high (was {high}) being updated to {threshold}")
                high = threshold

            iterations += 1

        # Use best attempt if max iterations reached
        self._logger.warning(
            f"Max iterations reached. Using best threshold: {threshold:.5f}"
        )
        return InferenceResult(
            optimal_threshold=threshold,
            frame_count=frame_count,
            iterations=iterations,
            search_path=search_path
        )

    async def _count_frames_with_threshold(
        self,
        video_path: str,
        similarity_threshold: float
    ) -> int:
        """
        Count frames that would be extracted with given threshold.

        Args:
            video_path: Path to video file
            similarity_threshold: Similarity threshold to test

        Returns:
            Number of frames that would be extracted
        """
        logger = logging.getLogger(__name__)
        # Create test config with current threshold
        test_config = VideoConfig.from_dict({
            **self.config.__dict__,
            'similarity_threshold': similarity_threshold,
            'enable_cache': False,  # Disable cache for inference
            'detect_keyframes': True
        })

        frame_count = 0
        frames_processed = 0
        prev_frame = None
        prev_frame_2 = None
        prev_gray_2 = None
        with VideoReader(video_path, test_config) as reader:
            while True:
                ret, frame = reader.read_frame()
                if not ret:
                    break

                if prev_frame is None:
                    frame_count += 1
                else:
                    # Simplified similarity check for speed
                    if len(frame.shape) == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        if prev_frame_2 is not None:
                            prev_gray_2 = cv2.cvtColor(prev_frame_2, cv2.COLOR_BGR2GRAY)
                    else:
                        gray, prev_gray = frame, prev_frame
                        if prev_frame_2 is not None:
                            prev_gray_2 =  prev_frame_2
                        else:
                            prev_gray_2 = prev_frame

                    hist1 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
                    hist3 = cv2.calcHist([prev_gray_2], [0], None, [256], [0, 256])
                    similarity1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    similarity2 = cv2.compareHist(hist1, hist3, cv2.HISTCMP_CORREL)

                    try:
                        ssim = cv2.compareSSIM(gray, prev_gray)
                        ssim2 = cv2.compareSSIM(gray, prev_gray_2)
                        # Combine histogram and SSIM scores
                        similarity1 = (similarity1 + ssim) / 2
                        similarity2 = (similarity2 + ssim) / 2
                    except:
                        # Fall back to just histogram if SSIM fails
                        logger.debug(f"DEBUG: nframes: VideoReader compareSSIM failed on frame {frames_processed} and previous")

                    if similarity1 < similarity_threshold or similarity2 < similarity_threshold:
                        frame_count += 1
                frames_processed += 1
                if prev_frame is not None:
                    prev_frame_2 = prev_frame.copy()
                prev_frame = frame.copy()

        return frame_count
