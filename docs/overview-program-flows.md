
# Video Frame Extraction and Analysis Program Overview

This document provides a visual and structural overview of the video frame extraction and analysis program, covering its core components, processing flows, and error-handling mechanisms.

---

## 1. Simplified Processing flowchart

This diagram shows the end-to-end flow using the CLI tool to interact with the module :- from loading the video file and configuration to extracting keyframes and saving the results.

```mermaid
flowchart LR
    A[Input Video File] --> B[Init,Metadata]
    B --> E[Inference]
    E --> F[Keyframe Threshold]
    F --> G[KeyFrames]
    B --> G[KeyFrames]
    B --> K[ALLFrames]
    K --> H[Save]
    G --> H[Save]

    style A font-size:10px
    style B font-size:10px
    style E font-size:10px
    style F font-size:10px
    style G font-size:10px
    style H font-size:10px
    style K font-size:10px
```

---

## 2. Module Interaction

Thus diagram shows how key the modules, such as `FrameExtractionModel`, `FrameBuffer`, `FrameAnalyzer`, communicate during the processing of a video file into a set of frames.

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
    Analyzer->>Model: Detect keyframe
    Model->>Output: Save keyframe to output directory
```

---

## 3. Keyframe Detection Workflow

This flowchart demonstrates the process of detecting keyframes within a video by comparing frames and selecting those that meet a defined similarity threshold.

```mermaid
flowchart TD
    A[Start Keyframe Detection] --> B[Load Frame from Buffer]
    B --> C{Check if Frame is Keyframe}
    C -->|Yes| D[Mark as Keyframe]
    C -->|No| E[Compare with Previous Frame]
    E --> F{Similarity < Threshold?}
    F -->|Yes| G[Save as Keyframe]
    F -->|No| H[Discard Frame]
    G --> I[Store Metadata for Keyframe]
    H --> B
    I --> J[End Keyframe Detection]
```

---

## 4. Error Handling Flow

The following flowchart outlines the error handling logic, detailing how different types of errors (e.g., `CodecError`, `TimeoutError`, `ResourceError`) are managed within the program.

```mermaid
flowchart TD
    A[Start Processing] --> B{Error Occurred?}
    B -->|No| C[Continue Processing]
    B -->|Yes| D{Error Type?}
    
    D -->|CodecError| E[Log Codec Issue]
    E --> F[Try Alternate Codec]
    F --> B

    D -->|ResourceError| G[Log Resource Issue]
    G --> H[Adjust Memory Allocation]
    H --> B

    D -->|TimeoutError| I[Log Timeout]
    I --> J[Retry Operation with Delay]
    J --> B

    C --> K[End Processing]
```

---

## 5. Configuration and Threshold Adjustment Flow

This flowchart details the configuration loading and dynamic adjustment process to ensure the program operates efficiently based on available resources and video properties.

```mermaid
flowchart TD
    A[Start Configuration] --> B[Load Config Parameters]
    B --> C{Validate Parameters?}
    C -->|Yes| D[Initialize with Default Config]
    C -->|No| E[Calculate Dynamic Adjustments]

    E --> F{System Resources Adequate?}
    F -->|Yes| G[Set Full Resolution]
    F -->|No| H[Reduce Buffer Size and Resolution]
    
    D --> I[Start Processing with Config]
    G --> I
    H --> I
    I --> J[Proceed to Keyframe Detection]
```

---

## 6. Class and Module Hierarchy

This class diagram represents the relationships between primary classes and modules, such as `FrameExtractionModel`, `FrameAnalyzer`, and `FrameBuffer`, showing the data flow and inheritance structures.

```mermaid
classDiagram
    FrameExtractionModel --> FrameAnalyzer
    FrameExtractionModel --> FrameBuffer
    FrameExtractionModel --> VideoMetadata
    FrameBuffer --> Frame
    FrameAnalyzer --> FrameMetadata
    FrameAnalyzer --> AnalysisResult
    VideoMetadata --> FrameMetadata
    FrameAnalyzer <|-- FrameProcessing : Inherits
    FrameProcessing : process_frame()
    FrameExtractionModel : initialize()
    FrameExtractionModel : process_video()
    FrameAnalyzer : analyze_frame()
    FrameBuffer : add_frame()
    FrameBuffer : release_frame()
    FrameBuffer : lock
```

