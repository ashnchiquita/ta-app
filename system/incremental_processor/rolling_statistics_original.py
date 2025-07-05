import numpy as np
from collections import deque

class RollingStatisticsOriginal:
    """
    Original rolling statistics implementation using proper Welford's algorithm.
    This maintains running statistics with incremental frame addition/removal.
    """

    def __init__(self, window_size: int, step=30, frame_shape: tuple = (72,72,3)):
        self.window_size = window_size
        self.step = step
        h, w, c = frame_shape
        self.frame_size = h * w * c
        
        # Store frames for proper frame-level Welford's algorithm
        self.frames = deque(maxlen=window_size)
        
        # Welford's algorithm variables
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean
        
    def add_frame(self, frame: np.ndarray):
        """Add a new frame and update statistics using Welford's algorithm."""
        # Flatten the frame to work with individual pixel values
        frame_flat = frame.flatten()
        
        # If we're at capacity, remove the oldest frame's contribution
        if len(self.frames) == self.window_size:
            self._remove_oldest_frame()
        
        # Add the new frame
        self.frames.append(frame_flat.copy())
        self._add_frame_data(frame_flat)

    def _remove_oldest_frame(self):
        """Remove the oldest frame's contribution from statistics."""
        if not self.frames:
            return
            
        old_frame = self.frames[0]  # This will be popped when we append the new frame
        
        # Remove each pixel value from the running statistics
        for pixel_value in old_frame:
            self._remove_value(pixel_value)
    
    def _add_frame_data(self, frame_flat):
        """Add frame data using Welford's algorithm."""
        for pixel_value in frame_flat:
            self._add_value(pixel_value)
    
    def _add_value(self, value):
        """Add a single value using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
    
    def _remove_value(self, value):
        """Remove a single value using reverse Welford's algorithm."""
        if self.count <= 1:
            self.count = 0
            self.mean = 0.0
            self.m2 = 0.0
            return
            
        delta = value - self.mean
        self.count -= 1
        self.mean -= delta / self.count if self.count > 0 else 0
        delta2 = value - self.mean
        self.m2 -= delta * delta2

    def get_total_count(self) -> int:
        """Get total number of pixel values in the window."""
        return self.count
    
    def get_mean(self) -> float:
        """Get current mean."""
        return self.mean
    
    def get_variance(self) -> float:
        """Get current variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count  # Population variance
    
    def get_std(self) -> float:
        """Get current standard deviation."""
        variance = self.get_variance()
        print(f"Count: {self.count}, Mean: {self.mean:.6f}, Variance: {variance:.6f}")
        return np.sqrt(variance)
    
    def is_ready(self) -> bool:
        """Check if we have enough frames for stable statistics."""
        return len(self.frames) >= min(self.step, self.window_size // 2)
