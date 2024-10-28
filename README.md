# Advanced Video Frame Extraction

A high-performance Python library for video frame extraction and analysis, suited to medical imagery or generic videos for example screen captures. This system provides adaptive key-frame analysis based on motion detection and multi-frame image differntials, efficient multi-threading amd memory management. it supported various Movie input and Image output formats. 

The module interaction diagram shows how the main components communicate during the processing of a video file.

```mermaid
sequenceDiagram
    participant CLI as Command Line (cli-script)
    participant Config as Configuration (config.py)
    participant Model as FrameExtractionModel (model.py)
    participant Buffer as FrameBuffer (buffer.py)
    participant Analyzer as FrameAnalyzer
    participant Output as Output Directory

    CLI->>Config: Load parameters
    CLI->>Model: Initialize FrameExtractionModel
    Model->>Buffer: Allocate FrameBuffer
    Model->>Analyzer: Initialize Analyzer

    CLI->>Model: Process video
    Model->>Buffer: Add frame to buffer
    Buffer-->>Analyzer: Pass frame for analysis
    Analyzer->>Model: Detect keyframes
    Model->>Output: Save keyframe to output directory
```

## Features

### Core Capabilities
- Intelligent keyframe detection using motion and content analysis over multiple previous frames
- Inference mode to calculate thresholds to generate _N_ keyframes from the video
- Concurrent frame processing with configurable thread pools
- Memory-efficient frame buffering and performance caching
- Multiple output format support (PNG, JPEG, WebP)
- Thorough debug logging mode 

### Analysis Features
- Motion detection using optical flow analysis
- Quality metrics:
  - Sharpness measurement
  - Noise estimation
  - Contrast evaluation
  - Exposure analysis
- Scene change detection
- Temporal pattern recognition

### Performance Features
- Configurable thread pool for parallel processing
- Memory-optimized frame buffer and recent frame cache
- Minimal frame copying through efficient view operations
- Retry mechanism with exponential backoff

## System Requirements

### Software Requirements
- Python 3.12+
- OpenCV Python (opencv-python)
- NumPy

## Usage

### Basic Usage
The video processor can be run from the command line with minimal configuration:

```bash
python cli-script.py input.mp4 output_dir/
```

### Required Arguments
- `input_video`: Path to input video file
- `output_dir`: Directory for output frames

### Advanced Usage Options

#### Output Format Configuration
Control the format and quality of extracted frames:
```bash
# Extract as JPEG with 85% quality
python cli-script.py input.mp4 output_dir/ \
    --format jpeg \
    --quality 85

# Extract as PNG with maximum compression
python cli-script.py input.mp4 output_dir/ \
    --format png \
    --quality 9
```

#### Frame Analysis Settings
Configure key-frame selection:
```bash
# Enable keyframe detection with custom similarity threshold
python cli-script.py input.mp4 output_dir/ \
    --enable-keyframes \
    --similarity 0.90
```

#### Performance Tuning
Optimize processing speed and resource usage:
```bash
# Configure threading and memory usage
python cli-script.py input.mp4 output_dir/ \
    --threads 4 \
    --buffer-size 60 \
    --max-memory 1024 \
    --disable-cache
```

#### Error Handling Configuration
Adjust retry behavior and timeouts:
```bash
# Configure robust error handling
python cli-script.py input.mp4 output_dir/ \
    --retries 5 \
    --retry-delay 1.0 \
    --frame-timeout 10.0 \
    --video-timeout 60.0
```

#### Logging Configuration
Control logging output and verbosity:
```bash
# Enable detailed logging to file
python cli-script.py input.mp4 output_dir/ \
    --log-level DEBUG \
    --log-file processing.log
```

### Complete Usage Example
```bash
python cli-script.py input.mp4 output_dir/ \
    --format jpeg \
    --quality 85 \
    --enable-keyframes \
    --similarity 0.95 \
    --threads 4 \
    --buffer-size 60 \
    --max-memory 1024 \
    --retries 3 \
    --log-level INFO
```

### Inference mode estimate threshold to generate specified number of key-frames
```bash
python ./cli-script.py 
    --format png 
    --inference-tolerance 0.00001 
    --target-frames 8  input.mp4 output/
```

### Sample output including inference mode

```bash
2024-10-27 13:44:13,490 - common - INFO - Logging configured at level 20
2024-10-27 13:44:13,490 - common - INFO - Video processing system initialized
2024-10-27 13:44:13,495 - __main__ - INFO - Created configuration: {'output_format': <OutputFormat.PNG: ('png', [16], False)>, 'compression_quality': 9, 'detect_keyframes': True, 'similarity_threshold': 0.999620166015625, 'thread_count': 1, 'buffer_size': 60, 'cache_size': 60, 'enable_cache': True, 'max_memory_usage': None, 'retry_attempts': 3, 'retry_delay': 0.5, 'frame_timeout': 5.0, 'video_timeout': 30.0}
2024-10-27 13:44:13,495 - __main__ - INFO - Running inference mode to target 8 frames
Progress: 22%2024-10-27 13:44:15,657 - nframes - INFO - Found acceptable threshold 0.99829 producing 8 frames (target: 8)
2024-10-27 13:44:15,657 - __main__ - INFO - Inference complete: threshold=0.998, estimated frames=8

Inference Results:
Optimal similarity threshold: 0.998
Estimated frame count: 8
Search iterations: 12

Processing video with inferred threshold...
Progress: 100%
2024-10-27 13:44:15,973 - processor - INFO - Processed 180 frames, kept 35 keyframes
2024-10-27 13:44:15,973 - __main__ - INFO - Processing complete. Extracted 35 frames.
Successfully extracted 35 frames to output
2024-10-27 13:44:15,974 - common - INFO - System cleanup completed
```
Note there is a discrepency between estimated and actual frames generated because the estimator uses a simplified method to estimate keyframe thresholds, whilst the full extraction compares not just a frame with its preceding 2 frames but a configurable number typically much higher. If this is problematic you can adjust the target frames accordingly.

### Parameters Reference

#### Output Format Options
- `--format`: Output format for frames
  - `png`: Lossless compression (default)
  - `jpeg`: Lossy compression, smaller files
  - `webp`: Modern format with good compression
- `--quality`: Quality/compression level
  - PNG: 0-9 (9 = max compression)
  - JPEG/WebP: 0-100 (100 = best quality)

#### Processing Options
- `--enable-keyframes`: Enable keyframe detection
- `--similarity`: Similarity threshold (0.0-1.0, default: 0.95)
- `--target-frames`: When specified enables automatic estimation of `--similarity` value so _n_ frames are produced
- `--threads`: Number of processing threads (default: CPU cores - 1)

#### Resource Management
- `--buffer-size`: Frame buffer size (default: 30)
- `--cache-size`: Frame cache size (default: 30)
- `--max-memory`: Maximum memory usage in MB (unbound if not speciifed)
- `--disable-cache`: Disable frame caching

#### Error Handling
- `--retries`: Number of retry attempts (default: 3)
- `--retry-delay`: Initial delay between retries in seconds (default: 0.5)
- `--frame-timeout`: Frame operation timeout in seconds (default: 5.0)
- `--video-timeout`: Video operation timeout in seconds (default: 30.0)

#### Logging Options
- `--log-level`: Logging verbosity
  - `DEBUG`: Detailed debugging information
  - `INFO`: General operation information
  - `WARNING`: Warning messages
  - `ERROR`: Error messages
  - `CRITICAL`: Critical issues
- `--log-file`: Path to log file (default: console output)

## Performance Considerations

### Memory Usage
- Frame buffer size directly impacts memory usage
- Monitor memory usage through built-in tracking

### CPU Utilization
- Thread count affects CPU usage and processing speed
- Default thread count is (CPU cores - 1)
- Adjust based on system capabilities and other workloads
- Consider I/O bottlenecks when configuring threads

### Storage I/O
- Buffer size affects disk I/O patterns
- Larger buffers reduce I/O frequency but increase memory usage
- Output format affects storage requirements:
  - PNG: Lossless, larger files
  - JPEG: Lossy, smaller files
  - WebP: Modern format, good compression

### Error Handling and Recovery
- Automatic retry with exponential backoff
- Proper resource cleanup on failures
- Comprehensive error logging
- Transaction-like operations with cleanup handlers

## Monitoring and Debugging

### Logging
- Available log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Performance metrics tracked via logs including:
  - Frame processing times
  - Memory usage statistics
  - Thread pool utilization
  - I/O operations monitoring
- Error tracking includes:
  - Detailed error messages and stack traces
  - Operation context
  - Cleanup operation status
  - Resource management events
