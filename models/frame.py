"""
Frame data structures and containers.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from core.types import Frame, GrayFrame

@dataclass
class FrameData:
    """Container for frame data with lazy grayscale conversion."""
    original: Frame
    grayscale: Optional[GrayFrame] = None
    is_color: bool = True
    
    def get_grayscale(self) -> GrayFrame:
        """Get or compute grayscale version of frame."""
        if self.grayscale is None:
            if self.is_color:
                self.grayscale = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
            else:
                self.grayscale = self.original
        return self.grayscale
    
    @property
    def shape(self) -> tuple:
        """Get frame dimensions."""
        return self.original.shape
    
    @classmethod
    def from_frame(cls, frame: Frame) -> 'FrameData':
        """Create FrameData from raw frame."""
        is_color = len(frame.shape) == 3
        return cls(
            original=frame,
            grayscale=None if is_color else frame,
            is_color=is_color
        )
