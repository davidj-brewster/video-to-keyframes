"""
Enhanced video frame processor with keyframe detection.
"""
import asyncio
from collections import deque
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
import cv2

from core.types import Frame
from core.errors import VideoError
from models.metadata import FrameMetadata
from video import VideoReader


class VideoProcessor:
    """Enhanced video processor with keyframe detection."""

    def __init__(self, config):
        """Initialize processor components."""
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._prev_frame = None
        self._frame_history = deque(maxlen=60)  # Keep last 5 frames for comparison

    def _compute_frame_similarity(self, frame1: Frame, frame2: Frame) -> float:
        """
        Compute similarity between two frames.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2

        # Compute histogram
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Compare histograms
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # @todo - make SSIM optional, as it is expensive
        # Compute structural similarity for more accuracy
        try:
            ssim1 = ssim(gray1, gray2) 
            # Combine histogram and SSIM scores, weighted more heavily for the more advanced technique
            return similarity * 0.3 + ssim1 * 0.7
        except Exception as e:
            # Fall back to just histogram if SSIM fails
            logging.warn(f"processor.py: _compute_frame_similarity: Failed to compute SSIM between frames, using frame-similarity based on cv2.HISTCMP_CORREL: {e}")
            return similarity

    def _is_keyframe(self, frame: Frame, metadata: FrameMetadata) -> bool:
        """
        Determine if frame should be kept based on similarity to recent frames.

        Args:
            frame: Current frame
            metadata: Frame metadata

        Returns:
            True if frame should be kept
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
        progress_callback: Optional[callable] = None
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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
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
                        if not ret:
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
                    results = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._process_frames_batch,
                        frames_batch,
                        output_dir
                    )

                    # Update output files
                    for result in results:
                        if result['success'] and result['output_path']:
                            output_files.append(result['output_path'])

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

    def _process_frames_batch(
        self,
        frames_batch: List[Tuple[int, Frame, FrameMetadata]],
        output_dir: Path
    ) -> List[dict]:
        """Process a batch of frames."""
        results = []

        for frame_number, frame, metadata in frames_batch:
            try:
                # Save frame
                output_path = self._save_frame(frame, metadata, output_dir)
                if output_path:
                    results.append({
                        'success': True,
                        'output_path': output_path,
                        'metadata': metadata
                    })

            except Exception as e:
                self._logger.error(
                    f"Frame processing failed for frame {frame_number}: {e}"
                )
                results.append({
                    'success': False,
                    'error': str(e),
                    'metadata': metadata
                })

        return results

    def _save_frame(
        self,
        frame: Frame,
        metadata: FrameMetadata,
        output_dir: Path
    ) -> Optional[str]:
        """Save frame to disk."""
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
