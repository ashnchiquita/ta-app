# Incremental rPPG Processing Optimization

## Overview

This implementation addresses the computation burst problem in real-time multi-person rPPG (remote photoplethysmography) systems by introducing incremental processing techniques.

## Problem Statement

The original implementation suffered from:
- **Computation Bursts**: Heavy processing every 180 frames (6 seconds)
- **FPS Drops**: Significant frame rate reduction during processing windows
- **Inefficient Resource Usage**: Idle periods followed by resource-intensive bursts
- **Poor Scalability**: Performance degradation with multiple subjects

## Solution: Incremental Processing

### Core Concept

Instead of processing all 180 frames at once, the system breaks down the computation into smaller chunks (e.g., 30 frames each) and distributes the processing load across time.

### Key Components

#### 1. Rolling Statistics (`RollingStatistics` class)
- Maintains running mean, standard deviation, and other statistics for the entire 180-frame window
- Updates incrementally as new frames arrive
- Removes old frame statistics when the window slides
- Ensures preprocessing uses consistent global statistics

#### 2. Incremental Chunk Processing (`IncrementalChunk` class)
- Represents a 30-frame chunk with its processed BVP signal
- Tracks processing status and timestamps
- Enables partial processing and result storage

#### 3. Incremental Processor (`IncrementalRPPGProcessor` class)
- Manages the entire incremental processing pipeline
- Processes chunks as they become ready
- Combines partial BVP signals for heart rate extraction
- Maintains per-face processing state

### Processing Flow

```
Frame 1-30:    Process Chunk 1 → BVP Segment 1
Frame 31-60:   Process Chunk 2 → BVP Segment 2
Frame 61-90:   Process Chunk 3 → BVP Segment 3
Frame 91-120:  Process Chunk 4 → BVP Segment 4
Frame 121-150: Process Chunk 5 → BVP Segment 5
Frame 151-180: Process Chunk 6 → BVP Segment 6

Heart Rate = FFT(Combine(BVP1, BVP2, BVP3, BVP4, BVP5, BVP6))
```

### Preprocessing Strategy

The challenge was maintaining the same preprocessing quality while working with chunks:

1. **Global Statistics**: Calculate mean/std over the entire 180-frame window
2. **Chunk Preprocessing**: Apply diff_normalize and standardize to 30-frame chunks using global statistics
3. **Consistency**: Ensures identical results to batch processing

#### Differential Normalization
```python
diff_normalized[j] = (frame[j+1] - frame[j]) / (frame[j+1] + frame[j] + ε)
normalized = diff_normalized / global_std
```

#### Standardization
```python
standardized = (chunk - global_mean) / (global_std + ε)
```

## Performance Optimizations

### 1. Memory Management (`MemoryManager` class)
- Monitors memory usage and forces garbage collection when needed
- Prevents memory leaks during long-running sessions
- Adaptive memory thresholds

### 2. Resource Pooling (`ResourcePool` class)
- Reuses numpy arrays to reduce allocation overhead
- Maintains pools of common array shapes
- Significantly reduces garbage collection pressure

### 3. Performance Monitoring (`PerformanceMonitor` class)
- Tracks FPS, processing times, and resource usage
- Provides detailed performance reports
- Enables adaptive optimization

### 4. Adaptive Optimization (`AdaptiveOptimizer` class)
- Automatically adjusts chunk sizes based on performance
- Balances latency vs. throughput
- Adapts to system capabilities

## Benefits

### Performance Improvements
- **Eliminated Computation Bursts**: Processing load distributed evenly
- **Stable FPS**: No more periodic frame rate drops
- **Lower Latency**: Heart rate updates available sooner
- **Better Scalability**: Handles multiple subjects more efficiently

### Resource Efficiency
- **Reduced Memory Usage**: Smaller working sets and better garbage collection
- **CPU Load Balancing**: Even distribution across cores
- **NPU Optimization**: Better utilization of Hailo NPU with serialized access

### Maintainability
- **Modular Design**: Clear separation of concerns
- **Extensible**: Easy to add new optimization strategies
- **Debuggable**: Comprehensive monitoring and logging

## Usage

### Basic Integration
```python
# In Pipeline.__init__()
if isinstance(rppg_signal_extractor, DeepLearningRPPGSignalExtractor):
    self.use_incremental = True
    self.incremental_processor = IncrementalRPPGProcessor(
        rppg_signal_extractor, hr_extractor, window_size, chunk_size
    )
```

### Adding Face Data
```python
# In Pipeline.add_face_data()
if self.use_incremental:
    self.incremental_processor.add_face_frame(face_id, roi, timestamp)
```

### Processing Faces
```python
# In Pipeline.process_faces()
if self.use_incremental:
    return self.process_faces_incremental(current_time)
```

### Core Approach

1. **Frame-by-Frame Processing**: Process each incoming frame immediately instead of waiting for full windows
2. **Incremental Statistics**: Maintain running statistics (mean, std, min, max) that update as new frames arrive
3. **Partial BVP Accumulation**: Extract BVP segments periodically and combine them at window completion
4. **Sliding Window Statistics**: Use efficient algorithms for statistics over sliding windows

### Key Components

#### 1. IncrementalStatistics Class
- **Welford's Online Algorithm**: For running mean and variance calculation
- **Sliding Window Min/Max**: Efficient min/max tracking using deques
- **Memory Efficient**: O(1) space complexity for statistics storage

#### 2. Modified Pipeline Processing
- **Automatic Detection**: Automatically enables incremental processing for DeepLearning models
- **Hybrid Support**: Maintains compatibility with conventional methods (POS, CHROM, ICA)
- **Configurable Segments**: Process every N frames (default: step_size/6)

#### 3. Statistics-Based Preprocessing
- **Global Statistics**: Uses window-wide statistics for consistent preprocessing
- **Differential Normalization**: Maintains DeepPhys preprocessing compatibility
- **Standardization**: Applies consistent normalization across segments

### Performance Benefits

1. **Smooth FPS**: Eliminates burst computation by distributing load across frames
2. **Reduced Latency**: Partial processing reduces final computation time
3. **Memory Efficient**: Clears processed segments to prevent memory buildup
4. **Scalable**: Handles up to 9 subjects with consistent performance

### Configuration

```python
# Incremental processing is automatically enabled for DeepLearning models
pipeline = Pipeline(
    rppg_signal_extractor=HEFDeepPhys(...),  # Auto-enables incremental processing
    hr_extractor=FFT(...),
    window_size=180,
    step_size=30  # Processing frequency: every step_size/6 frames
)
```

### Technical Details

#### Statistics Management
- **Running Mean**: Updated using Welford's algorithm for numerical stability
- **Running Variance**: Maintains M2 accumulator for variance calculation
- **Global Min/Max**: Tracks across entire sliding window for normalization

#### BVP Segment Processing
- **Segment Size**: `step_size // 6` frames per segment (typically 5-30 frames)
- **Preprocessing**: Uses current window statistics for consistent normalization
- **Combination**: Merges segments in correct temporal order
- **Memory Management**: Automatically clears old segments

#### Fallback Compatibility
- **Conventional Methods**: Still use traditional parallel processing
- **Mixed Environments**: Supports both incremental and traditional processing
- **Error Handling**: Graceful fallback to traditional processing on errors

### Implementation Files

1. **`system/pipeline.py`**: Main incremental processing logic
2. **`system/system.py`**: Integration with existing system (uncommented core processing)
3. **`components/rppg_signal_extractor/deep_learning/hef/deep_phys.py`**: Model compatibility

### Usage Notes

- **Automatic Activation**: Incremental processing activates automatically for DeepLearning models
- **Window Requirements**: Still requires full window_size frames before first HR estimation
- **Statistics Warmup**: Requires minimum 2 frames for meaningful statistics
- **Memory Efficiency**: Automatically manages memory by clearing processed segments

This implementation maintains the same API while providing significant performance improvements for real-time multi-subject rPPG estimation.
