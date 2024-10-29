"""
Motion detection and analysis using optical flow and frame history.

This module provides sophisticated motion analysis by:
- Tracking motion patterns across multiple frames
- Using optical flow for detailed motion detection
- Weighting recent motion more heavily
- Detecting significant scene changes
"""

from collections import deque
import logging
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from core.constants import (
    MOTION_ANALYSIS_WINDOW,
    MOTION_THRESHOLDS,
    FRAME_HISTORY_SIZE
)
from models.frame import FrameData
from models.metadata import FrameMetadata

class MotionDetector:
    """
    Enhanced motion detector with temporal analysis.

    Uses a sliding window of frames to detect:
    - Gradual motion patterns
    - Sudden scene changes
    - Camera movements
    - Object motion
    """

    def __init__(self, config):
        """Initialize motion detector with configuration."""
        self.config = config
        self._frame_history: Deque[Tuple[FrameData, float]] = deque(
            maxlen=FRAME_HISTORY_SIZE
        )
        self._logger = logging.getLogger(__name__)
        self._flow_params = MOTION_THRESHOLDS['optical_flow']

    def compute_score(
        self,
        current_frame: FrameData,
        history_frames: List[FrameData],
        temporal_window: int = MOTION_ANALYSIS_WINDOW
    ) -> float:
        """
        Compute motion score using temporal analysis over multiple frames.

        Args:
            current_frame: Current frame being analyzed
            history_frames: List of previous frames for analysis
            temporal_window: Number of previous frames to analyze

        Returns:
            float: Motion score in range [0,1] where 1 indicates maximum motion
        """
        if not history_frames:
            return 0.0

        # Get pre-computed grayscale version
        curr_gray = current_frame.get_grayscale()
        motion_scores = []

        # Analyze motion against recent frame history
        for prev_frame in history_frames[-temporal_window:]:
            if prev_frame.shape != current_frame.shape:
                continue

            # Use pre-computed grayscale frames
            prev_gray = prev_frame.get_grayscale()

            try:
                # Calculate optical flow using Farneback algorithm
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    curr_gray,
                    None,
                    self._flow_params['pyr_scale'],
                    self._flow_params['levels'],
                    self._flow_params['winsize'],
                    self._flow_params['iterations'],
                    self._flow_params['poly_n'],
                    self._flow_params['poly_sigma'],
                    0
                )

                # Analyze flow patterns
                score = self._analyze_flow_patterns(flow, curr_gray.shape)
                motion_scores.append(score)

            except cv2.error as e:
                self._logger.warning(f"Flow computation failed: {e}")
                continue

        if not motion_scores:
            return 0.0

        # Weight recent motion more heavily using exponential decay
        weights = np.exp(np.linspace(-1, 0, len(motion_scores)))
        weighted_score = np.average(motion_scores, weights=weights)

        # Update frame history with motion score
        self._frame_history.append((current_frame, weighted_score))

        return max(0.0, min(1.0, weighted_score))

    def _analyze_flow_patterns(
        self,
        flow: NDArray[np.float32],
        shape: tuple
    ) -> float:
        """
        Analyze optical flow patterns to detect different types of motion.

        Detects:
        - Global camera motion
        - Local object motion
        - Scene transitions

        Args:
            flow: Optical flow field
            shape: Frame dimensions

        Returns:
            float: Normalized motion score
        """
        # Calculate flow magnitude and direction
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])

        # Analyze global motion (camera movement)
        mean_flow_x = np.mean(flow[..., 0])
        mean_flow_y = np.mean(flow[..., 1])
        global_motion = np.sqrt(mean_flow_x**2 + mean_flow_y**2)

        # Analyze local motion (object movement)
        local_motion = np.std(magnitude)

        # Detect coherent motion patterns
        flow_coherence = self._compute_flow_coherence(angle, magnitude)

        # Combine different motion aspects with weights
        weights = {
            'global': 0.3,  # Camera motion
            'local': 0.5,   # Object motion
            'coherent': 0.2  # Motion patterns
        }

        motion_score = (
            weights['global'] * global_motion +
            weights['local'] * local_motion +
            weights['coherent'] * flow_coherence
        )

        # Normalize by frame dimensions
        normalized_score = motion_score / (shape[0] * shape[1])
        return normalized_score * 10  # Scale to useful range

    def _compute_flow_coherence(
        self,
        angle: NDArray[np.float32],
        magnitude: NDArray[np.float32],
        threshold: float = 0.1
    ) -> float:
        """
        Compute coherence of motion patterns.

        Args:
            angle: Flow direction angles
            magnitude: Flow magnitudes
            threshold: Minimum magnitude to consider

        Returns:
            float: Coherence score [0,1]
        """
        # Only consider significant motion
        significant_motion = magnitude > threshold
        if not np.any(significant_motion):
            return 0.0

        # Compute angle histogram for significant motion
        hist, _ = np.histogram(
            angle[significant_motion],
            bins=16,
            range=(-np.pi, np.pi),
            weights=magnitude[significant_motion]
        )

        # Normalize histogram
        hist = hist / np.sum(hist)

        # Compute entropy as measure of coherence
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(16)  # Maximum possible entropy

        # Convert entropy to coherence score (lower entropy = higher coherence)
        coherence = 1.0 - (entropy / max_entropy)

        return coherence

    def is_significant_motion(
        self,
        score: float,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Determine if motion is significant.

        Args:
            score: Motion score to evaluate
            threshold: Optional custom threshold

        Returns:
            bool: True if motion is significant
        """
        threshold = threshold or MOTION_THRESHOLDS['significant_motion']
        return score > threshold
