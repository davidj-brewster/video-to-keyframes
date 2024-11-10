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
from typing import List, Optional, Tuple, Any, Dict, Callable
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

from core.types import Frame
from core.errors import VideoError
from models.metadata import FrameMetadata
from video import VideoReader


class VideoProcessor:
    """Enhanced video processor with keyframe detection capabilities."""
    
    def __init__(self, config: 'ProcessorConfig'):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._recent_frames: deque = deque(maxlen=5)
        
    async def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Path]:
        """Process video and extract frames."""
        video = VideoReader(video_path)
        output_files: List[Path] = []
        processed_frames = 0
        
        try:
            await self._initialize_processing(video)
            
            while True:
                # Process frames in batches
                frames_batch: List[Tuple[int, np.ndarray, FrameMetadata]] = []
                
                # Collect batch of frames
                for _ in range(self.config.buffer_size):
                    ret, frame = video.read_frame()
                    if not ret:
                        break
                        
                    metadata = FrameMetadata(
                        frame_number=processed_frames,
                        timestamp=processed_frames / video.fps,
                        width=frame.shape[1],
                        height=frame.shape[0]
                    )
                    
                    if self.config.detect_keyframes:
                        if self._is_keyframe(frame, metadata):
                            frames_batch.append((processed_frames, frame, metadata))
                    else:
                        frames_batch.append((processed_frames, frame, metadata))
                        
                    processed_frames += 1
                    
                if not frames_batch:
                    break
                    
                # Process batch
                try:
                    results = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._process_frames_batch,
                        frames_batch,
                        output_dir
                    )
                    
                    if results:
                        for result in results:
                            try:
                                if result and isinstance(result, dict):
                                    if result.get('success'):
                                        output_path = Path(result['output_path'])
                                        output_files.append(output_path)
                                    elif error_msg := result.get('error'):
                                        self._logger.warning(
                                            f"Frame processing failed: {error_msg}"
                                        )
                            except Exception as e:
                                self._logger.error(f"Error processing result: {e}")
                                
                    # Update progress
                    if progress_callback:
                        progress = min(1.0, processed_frames / video.total_frames)
                        progress_callback(progress)
                        
                except Exception as e:
                    self._logger.error(f"Batch processing error: {e}")
                    continue
                    
        finally:
            video.close()
            
        return output_files
        
    def _is_keyframe(self, frame: np.ndarray, metadata: FrameMetadata) -> bool:
        """Check if frame is different enough from recent frames."""
        if not self._recent_frames:
            self._recent_frames.append(frame)
            return True
            
        # Compare with most recent frame using SSIM
        prev_frame = self._recent_frames[-1]
        score = ssim(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
            full=False
        )
        
        if score < self.config.similarity_threshold:
            self._recent_frames.append(frame)
            return True
            
        return False
