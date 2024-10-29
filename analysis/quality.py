"""
Frame quality analysis using multiple metrics.

Provides comprehensive quality assessment including:
- Sharpness measurement
- Noise estimation
- Contrast evaluation
- Exposure analysis
"""

import logging
import cv2
import numpy as np
from numpy.typing import NDArray

from core.constants import (
    QUALITY_THRESHOLDS,
    DENOISING_PARAMS
)
from models.frame import FrameData

class QualityAnalyzer:
    """
    Enhanced frame quality analyzer.

    Analyzes multiple aspects of frame quality:
    - Sharpness using Laplacian variance
    - Noise using denoising difference
    - Contrast using intensity distribution
    - Exposure using histogram analysis
    """

    def __init__(self, config):
        """Initialize quality analyzer with configuration."""
        self.config = config
        self._logger = logging.getLogger(__name__)

        # Pre-compute kernels
        self._laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)

    def compute_score(self, frame: FrameData) -> float:
        """
        Compute comprehensive quality score.

        Args:
            frame: Frame to analyze

        Returns:
            float: Quality score in range [0,1]
        """
        if not frame.is_color:  # Some metrics require color information
            return self._compute_grayscale_quality(frame)

        scores = []
        weights = {
            'sharpness': 0.3,
            'contrast': 0.3,
            'noise': 0.2,
            'exposure': 0.2
        }

        # Get pre-computed grayscale for intensity-based metrics
        gray = frame.get_grayscale()

        # Compute individual quality metrics
        scores.append(
            self._compute_sharpness(gray) * weights['sharpness']
        )
        scores.append(
            self._compute_contrast(gray) * weights['contrast']
        )
        scores.append(
            self._compute_noise_score(frame.original) * weights['noise']
        )
        scores.append(
            self._compute_exposure_score(gray) * weights['exposure']
        )

        return np.mean(scores)

    def _compute_grayscale_quality(self, frame: FrameData) -> float:
        """
        Compute quality score for grayscale frame.

        Args:
            frame: Grayscale frame

        Returns:
            float: Quality score in range [0,1]
        """
        gray = frame.get_grayscale()

        # For grayscale, focus on sharpness and contrast
        sharpness_score = self._compute_sharpness(gray)
        contrast_score = self._compute_contrast(gray)

        return 0.5 * (sharpness_score + contrast_score)

    def _compute_sharpness(self, gray: NDArray[np.uint8]) -> float:
        """
        Compute sharpness score using Laplacian variance.

        The Laplacian operator detects edges and high-frequency content.
        Higher variance indicates more detail and better sharpness.

        Args:
            gray: Grayscale frame

        Returns:
            float: Sharpness score in range [0,1]
        """
        # Use custom kernel for better edge detection
        laplacian = cv2.filter2D(
            gray.astype(np.float32),
            cv2.CV_64F,
            self._laplacian_kernel
        )

        # Compute variance as measure of sharpness
        sharpness = np.var(laplacian) / (gray.shape[0] * gray.shape[1])

        # Normalize with empirically determined maximum
        return min(1.0, sharpness / QUALITY_THRESHOLDS['sharpness_max'])

    def _compute_contrast(self, gray: NDArray[np.uint8]) -> float:
        """
        Compute contrast score using intensity distribution.

        Uses standard deviation of intensities normalized by the
        theoretical maximum range for 8-bit images.

        Args:
            gray: Grayscale frame

        Returns:
            float: Contrast score in range [0,1]
        """
        # Compute intensity standard deviation
        std_dev = np.std(gray)

        # Normalize by theoretical maximum (half of 8-bit range)
        contrast = std_dev / QUALITY_THRESHOLDS['contrast_baseline']

        return min(1.0, contrast)

    def _compute_noise_score(self, frame: NDArray[np.uint8]) -> float:
        """
        Compute noise score using denoising difference.

        Uses Non-Local Means denoising to estimate noise level.
        Smaller difference between original and denoised image
        indicates less noise.

        Args:
            frame: Color frame

        Returns:
            float: Noise score in range [0,1] where 1 is least noisy
        """
        try:
            # Apply Non-Local Means denoising
            denoised = cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                h=DENOISING_PARAMS['h'],
                hColor=DENOISING_PARAMS['h'],
                templateWindowSize=DENOISING_PARAMS['templateWindowSize'],
                searchWindowSize=DENOISING_PARAMS['searchWindowSize']
            )

            # Compute mean absolute difference
            noise_level = np.mean(np.abs(frame - denoised)) / 255

            # Convert to quality score (less noise = higher score)
            return 1.0 - min(1.0, noise_level * QUALITY_THRESHOLDS['noise_scale'])

        except cv2.error as e:
            self._logger.warning(f"Denoising failed: {e}")
            return 0.0

    def _compute_exposure_score(self, gray: NDArray[np.uint8]) -> float:
        """
        Compute exposure score using histogram analysis.

        Evaluates the distribution of intensities to detect:
        - Under-exposure (too dark)
        - Over-exposure (too bright)
        - Good exposure (well-distributed intensities)

        Args:
            gray: Grayscale frame

        Returns:
            float: Exposure score in range [0,1]
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize

        # Compute mean intensity
        mean_intensity = np.average(np.arange(256), weights=hist)

        # Compute distance from optimal mean (128)
        optimal_mean = 128
        mean_score = 1.0 - abs(mean_intensity - optimal_mean) / optimal_mean

        # Check for clipping
        dark_clip = np.sum(hist[:10])  # First 10 bins
        bright_clip = np.sum(hist[-10:])  # Last 10 bins
        clip_penalty = (dark_clip + bright_clip) * 2

        return max(0.0, min(1.0, mean_score - clip_penalty))
