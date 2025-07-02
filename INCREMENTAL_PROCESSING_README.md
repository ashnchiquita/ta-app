# Incremental rPPG Processing Implementation

## Problem Solved

The original implementation had significant FPS drops when processing multiple subjects because:
1. **Burst Processing**: All rPPG computation happened at the end of each 180-frame window
2. **Idle Time**: No preprocessing occurred during frame accumulation phase
3. **Resource Spikes**: Heavy computation created uneven processing loads

## Solution: Distributed Processing with Incremental Statistics

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
