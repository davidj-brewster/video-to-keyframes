"""
Frame analysis with motion and quality assessment.
"""
from collections import deque
import logging
from typing import Deque, Tuple

import cv2

from core.constants import (
    FRAME_HISTORY_SIZE,
    MOTION_ANALYSIS_WINDOW,
    ANALYSIS_WEIGHTS
)
from core.types import Frame
from models.frame import FrameData
from models.metadata import FrameMetadata
from analysis.motion import MotionDetector
from analysis.quality import QualityAnalyzer

class FrameAnalyzer:
    """Advanced frame analyzer with temporal analysis."""

    def __init__(self, config):
        """Initialize analyzer components."""
        self.config = config
        self._frame_history: Deque[Tuple[FrameData, FrameMetadata]] = deque(
            maxlen=FRAME_HISTORY_SIZE
        )
        self._logger = logging.getLogger(__name__)

        # Initialize components
        self.motion_detector = MotionDetector(config)
        self.quality_analyzer = QualityAnalyzer(config)
        self.feature_detector = cv2.SIFT_create()

    def is_keyframe(self, frame: FrameData, metadata: FrameMetadata) -> bool:
        """Determine if frame is a keyframe using comprehensive analysis."""
        if not self._frame_history:
            return True

        # Analyze recent history
        history_window = list(self._frame_history)[-MOTION_ANALYSIS_WINDOW:]

        # Compute motion and similarity scores
        motion_score = self.motion_detector.compute_score(
            frame,
            [f for f, _ in history_window]
        )

        similarity_score = max(
            self._compute_frame_similarity(frame, hist_frame)
            for hist_frame, _ in history_window
        )

        # Update metadata
        metadata.motion_score = motion_score
        metadata.similarity_score = 1.0 - similarity_score
        metadata.quality_score = self.quality_analyzer.compute_score(frame)

        # Update history
        self._frame_history.append((frame, metadata))

        # Weight and combine scores
        return (
            motion_score * ANALYSIS_WEIGHTS['motion'] +
            (1.0 - similarity_score) * ANALYSIS_WEIGHTS['feature']
        ) > 0.5

    def _compute_frame_similarity(
        self,
        frame1: FrameData,
        frame2: FrameData
    ) -> float:
        """Compute similarity between frames using multiple metrics."""
        hist_score = self._compute_histogram_similarity(frame1, frame2)
        feat_score = self._compute_feature_similarity(frame1, frame2)

        return (
            hist_score * ANALYSIS_WEIGHTS['histogram'] +
            feat_score * ANALYSIS_WEIGHTS['feature']
        )
