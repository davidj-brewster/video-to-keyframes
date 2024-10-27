"""
System-wide constants and configuration defaults.
"""

# Frame processing constants
FRAME_HISTORY_SIZE = 30
MOTION_ANALYSIS_WINDOW = 10

# Analysis weights
ANALYSIS_WEIGHTS = {
    'motion': 0.4,
    'feature': 0.3,
    'histogram': 0.3,
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'sharpness_max': 1000.0,  # Maximum expected Laplacian variance
    'contrast_baseline': 128.0,  # Standard deviation baseline for contrast
    'noise_scale': 2.0,  # Scaling factor for noise score
}

# Denoising parameters
DENOISING_PARAMS = {
    'h': 10,  # Filter strength
    'templateWindowSize': 7,  # Template patch size
    'searchWindowSize': 21,  # Search window size
}

# Motion thresholds
MOTION_THRESHOLDS = {
    'significant_motion': 0.3,
    'optical_flow': {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
    }
}
