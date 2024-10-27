#!/usr/bin/env python3
"""
Command-line interface for video frame extraction and analysis.

This script provides a robust CLI for the video processing system, allowing users
to process videos with configurable parameters for analysis, resource management,
and output settings.

Example usage:
    # Basic usage with default settings
    python video_cli.py input.mp4 output_dir/

    # Configure output format and quality
    python video_cli.py input.mp4 output_dir/ --format jpeg --quality 85

    # Advanced processing settings
    python video_cli.py input.mp4 output_dir/ --threads 4 --buffer-size 60 \\
        --enable-keyframes --similarity 0.95
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from config import VideoConfig, OutputFormat
from model import FrameExtractionModel
from common import setup_logging
from nframes import SimilarityInference

# Configure logging
logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with all supported options.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Extract and analyze video frames with advanced processing options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "input_video",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory for output frames"
    )

    # Output format options
    format_group = parser.add_argument_group("Output Format Options")
    format_group.add_argument(
        "--format",
        type=str,
        choices=[f.name.lower() for f in OutputFormat],
        default="png",
        help="Output format for extracted frames"
    )
    format_group.add_argument(
        "--quality",
        type=int,
        help="Quality/compression level (0-9 for PNG, 0-100 for JPEG/WebP)"
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--enable-keyframes",
        action="store_true",
        help="Enable intelligent keyframe detection"
    )
    proc_group.add_argument(
        "--similarity",
        type=float,
        default=0.95,
        help="Similarity threshold for keyframe detection (0.0-1.0)"
    )
    proc_group.add_argument(
        "--threads",
        type=int,
        help="Number of processing threads (default: CPU count - 1)"
    )

    # Resource management
    resource_group = parser.add_argument_group("Resource Management")
    resource_group.add_argument(
        "--buffer-size",
        type=int,
        default=30,
        help="Frame buffer size"
    )
    resource_group.add_argument(
        "--cache-size",
        type=int,
        default=30,
        help="Frame cache size"
    )
    resource_group.add_argument(
        "--max-memory",
        type=int,
        help="Maximum memory usage in MB (default: unlimited)"
    )
    resource_group.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable frame caching"
    )

    # Error handling
    error_group = parser.add_argument_group("Error Handling")
    error_group.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts for failed operations"
    )
    error_group.add_argument(
        "--retry-delay",
        type=float,
        default=0.5,
        help="Initial delay between retries in seconds"
    )
    error_group.add_argument(
        "--frame-timeout",
        type=float,
        default=5.0,
        help="Timeout for frame operations in seconds"
    )
    error_group.add_argument(
        "--video-timeout",
        type=float,
        default=30.0,
        help="Timeout for video operations in seconds"
    )

    # Logging options
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    log_group.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: log to console)"
    )

    inference_group = parser.add_argument_group("Inference Mode Options")
    inference_group.add_argument(
        "--target-frames",
        type=int,
        help="Enable inference mode to target specific number of output frames"
    )
    inference_group.add_argument(
        "--inference-tolerance",
        type=float,
        default=0.05,
        help="Acceptable deviation from target frame count (0.0-1.0)"
    )

    return parser

def validate_args(args: argparse.Namespace) -> None:
    """
    Validate and adjust command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid
    """
    # Validate input file
    input_path = Path(args.input_video)
    if not input_path.exists():
        raise ValueError(f"Input video does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    # Validate output directory
    output_path = Path(args.output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_path}")

    # Validate quality settings
    if args.quality is not None:
        format_enum = OutputFormat.from_string(args.format)
        if format_enum == OutputFormat.PNG and not 0 <= args.quality <= 9:
            raise ValueError("PNG quality must be between 0 and 9")
        elif format_enum in (OutputFormat.JPEG, OutputFormat.WEBP) and not 0 <= args.quality <= 100:
            raise ValueError("JPEG/WebP quality must be between 0 and 100")

    # Validate numeric parameters
    if args.similarity is not None and not 0 <= args.similarity <= 1:
        raise ValueError("Similarity threshold must be between 0.0 and 1.0")
    if args.threads is not None and args.threads < 1:
        raise ValueError("Thread count must be at least 1")
    if args.buffer_size < 1:
        raise ValueError("Buffer size must be at least 1")
    if args.cache_size < 0:
        raise ValueError("Cache size must be non-negative")
    if args.max_memory is not None and args.max_memory < 1:
        raise ValueError("Maximum memory must be at least 1 MB")

def create_config(args: argparse.Namespace) -> VideoConfig:
    """
    Create VideoConfig from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        VideoConfig: Configured video processing settings
    """
    # Convert format string to enum
    output_format = OutputFormat.from_string(args.format)

    # Create config with all parameters
    config_dict = {
        'output_format': output_format,
        'compression_quality': args.quality,
        'detect_keyframes': args.enable_keyframes,
        'similarity_threshold': args.similarity,
        'thread_count': args.threads,
        'buffer_size': args.buffer_size,
        'cache_size': args.cache_size,
        'enable_cache': not args.disable_cache,
        'max_memory_usage': args.max_memory * 1024 * 1024 if args.max_memory else None,
        'retry_attempts': args.retries,
        'retry_delay': args.retry_delay,
        'frame_timeout': args.frame_timeout,
        'video_timeout': args.video_timeout
    }

    # Remove None values to use defaults
    config_dict = {k: v for k, v in config_dict.items() if v is not None}

    return VideoConfig.from_dict(config_dict)

def setup_progress_callback() -> callable:
    """
    Create progress callback function.

    Returns:
        callable: Progress callback function
    """
    last_progress = [0]  # Use list for mutable state

    def progress_callback(progress: float) -> None:
        """
        Update progress bar.

        Args:
            progress: Progress value between 0 and 1
        """
        # Only update on significant changes (1% increments)
        current_percent = int(progress * 100)
        if current_percent > last_progress[0]:
            last_progress[0] = current_percent
            print(f"\rProgress: {current_percent}%", end="", file=sys.stderr)
            if progress >= 1:
                print(file=sys.stderr)  # New line at 100%

    return progress_callback

async def process_video(args: argparse.Namespace) -> None:
    """
    Process video with provided configuration.

    Args:
        args: Parsed command line arguments
    """
    try:
        # Validate arguments
        validate_args(args)

        # Create configuration
        config = create_config(args)
        logger.info("Created configuration: %s", config.__dict__)

        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=False)

        if args.target_frames:
            logger.info(
                f"Running inference mode to target {args.target_frames} frames"
            )

            progress_callback = setup_progress_callback()
            inference = SimilarityInference(config)
            inference_result = await inference.infer_threshold(
                str(args.input_video),
                args.target_frames,
                progress_callback
            )

            logger.info(
                f"Inference complete: threshold={inference_result.optimal_threshold:.3f}, "
                f"estimated frames={inference_result.frame_count}"
            )

            # Update config with inferred threshold
            config = VideoConfig.from_dict({
                **config.__dict__,
                'similarity_threshold': inference_result.optimal_threshold,
                'detect_keyframes': True
            })

            # Print inference details
            print("\nInference Results:")
            print(f"Optimal similarity threshold: {inference_result.optimal_threshold:.3f}")
            print(f"Estimated frame count: {inference_result.frame_count}")
            print(f"Search iterations: {inference_result.iterations}")
            print("\nProcessing video with inferred threshold...")

        # Create and run model
        model = FrameExtractionModel(config)
        progress_callback = setup_progress_callback()

        output_files = await model.process_video(
            str(args.input_video),
            str(output_path),
            progress_callback
        )

        # Report results
        logger.info("Processing complete. Extracted %d frames.", len(output_files))
        print(f"Successfully extracted {len(output_files)} frames to {output_path}")

    except Exception as e:
        logger.error("Processing failed: %s", e, exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=args.log_file
    )

    # Run async process_video
    asyncio.run(process_video(args))

if __name__ == "__main__":
    main()
