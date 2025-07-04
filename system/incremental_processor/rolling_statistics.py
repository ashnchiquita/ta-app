import numpy as np
from collections import deque

class RollingStatistics:
    """Maintains rolling statistics for a sliding window."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.frames = deque(maxlen=window_size)
        self.sum = None
        self.sum_sq = None
        self.count = 0
        
    def add_frame(self, frame: np.ndarray):
        """Add a new frame and update statistics."""
        if self.sum is None:
            self.sum = np.zeros_like(frame, dtype=np.float32)
            self.sum_sq = np.zeros_like(frame, dtype=np.float32)
        
        # If we're at capacity, subtract the oldest frame
        if len(self.frames) == self.window_size:
            old_frame = self.frames[0]
            self.sum -= old_frame.astype(np.float32)
            self.sum_sq -= (old_frame.astype(np.float32) ** 2)
            self.count -= 1
        
        # Add the new frame
        frame_f32 = frame.astype(np.float32)
        self.frames.append(frame)
        self.sum += frame_f32
        self.sum_sq += frame_f32 ** 2
        self.count += 1
    
    def get_mean(self) -> np.ndarray:
        """Get current mean."""
        if self.count == 0:
            return np.zeros_like(self.sum)
        return self.sum / self.count
    
    def get_std(self) -> np.ndarray:
        """Get current standard deviation."""
        if self.count <= 1:
            return np.ones_like(self.sum)
        
        mean = self.get_mean()
        variance = (self.sum_sq / self.count) - (mean ** 2)
        # Avoid negative variance due to floating point errors
        variance = np.maximum(variance, 1e-8)
        return np.sqrt(variance)
    
    def is_ready(self) -> bool:
        """Check if we have enough frames for stable statistics."""
        return self.count >= min(30, self.window_size // 2)
