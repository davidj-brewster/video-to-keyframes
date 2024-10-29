"""
Inference mode for automatically determining similarity threshold.
"""
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
from skimage.metrics import structural_similarity as ssim

from core.errors import VideoError
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
        self.tolerance = 0.1  # Allow  deviation from target

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
        gray2 = None
        gray15 = None
        with VideoReader(video_path, test_config) as reader:
            while True:
                ret, frame = reader.read_frame()
                if not ret:
                    break

                if prev_frame is None:
                    frame_count += 1
                    frames_processed = 1
                    prev_frame2 = frame.copy()
                    prev_frame = frame.copy()
                elif frames_processed < 3: #ignore 2 frames after last key frame
                    frames_processed += 1
                    prev_frame = frame.copy()
                else:
                    # Simplified similarity check for speed
                    if len(frame.shape) == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray15=cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        gray2=cv2.cvtColor(prev_frame2, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame
                        if prev_frame is not None:
                            gray15=prev_frame
                        else:
                            logger.error(f"FATAL: None value for prev_frame")
                            sys.exit(1)

                        if prev_frame2 is not None:
                            gray2 = prev_frame2
                        else:
                            logger.error(f"FATAL: None values for gray15, gray2")
                            sys.exit(1)


                    hist_curr = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist_prev = cv2.calcHist([gray15], [0], None, [256], [0, 256])
                    hist_prev_prev = cv2.calcHist([gray2], [0], None, [256], [0, 256])
                    similarity2= 100
                    similarity15= 100
                    similarity2 = cv2.compareHist(hist_curr, hist_prev_prev, cv2.HISTCMP_CORREL)
                    similarity15 = cv2.compareHist(hist_prev, hist_prev_prev, cv2.HISTCMP_CORREL)

                    # @todo : make the use of SSIM optional throughout the program, it is expensive
                    try:
                        score2 = ssim(gray, gray2)
                        score15 = ssim(gray15, gray2)
                        # Combine histogram and SSIM scores
                        similarity2 = (similarity2 * 0.3+ score2 * 0.7)  #weigh advanced model more heavily than freqhist
                        similarity15 = (similarity15 * 0.3 + score15 * 0.7) #weigh advanced model more heavily than freqhist
                        #logging.trace(f"{similarity2} | {similarity15}")
                    except Exception as e:
                        # Fall back to just histogram if SSIM fails
                        logger.warn(f"WARN: nframes: VideoReader compareSSIM failed on frame {frames_processed} and previous")
                        logger.warn(f"{e}")

                    #Take shortcut given that if prev frame  satisfies threshold then 2nd prior frame almost cert. will
                    if (similarity2 < (similarity_threshold * 1.0)) or (similarity15 < (similarity_threshold * 1.0)):
                        frames_processed = 0
                        prev_frame2 = frame.copy()
                        frame_count += 1
                    frames_processed += 1
                    prev_frame = frame.copy()

        return frame_count
