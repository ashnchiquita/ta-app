import numpy as np
from collections import deque

class RollingStatistics:
    """
    Optimized rolling statistics using a hybrid approach:
    - Maintains frame-level sums for efficiency
    - Uses numerically stable variance calculation
    """

    def __init__(self, window_size: int, step=30, frame_shape: tuple = (72,72,3)):
        self.window_size = window_size
        self.step = step
        h, w, c = frame_shape
        self.frame_size = h * w * c
        
        # Store frames for accurate variance calculation
        self.frames = deque(maxlen=window_size)
        
        # Quick access to current statistics
        self.current_mean = 0.0
        self.current_std = 0.0
        self.stats_valid = False
        
    def add_frame(self, frame: np.ndarray):
        """Add a new frame to the rolling window."""
        self.frames.append(frame.copy())
        self.stats_valid = False  # Mark stats as needing recalculation
        
    def _calculate_stats(self):
        """Calculate statistics using numerically stable method."""
        if not self.frames:
            self.current_mean = 0.0
            self.current_std = 0.0
            self.stats_valid = True
            return
            
        # Stack all frames and flatten
        all_frames = np.stack(list(self.frames))
        all_pixels = all_frames.flatten()
        
        # Calculate mean and std using numpy's numerically stable algorithms
        self.current_mean = np.mean(all_pixels)
        self.current_std = np.std(all_pixels)
        self.stats_valid = True
        
    def get_total_count(self) -> int:
        """Get total number of pixel values in the window."""
        return len(self.frames) * self.frame_size
    
    def get_mean(self) -> float:
        """Get current mean."""
        if not self.stats_valid:
            self._calculate_stats()
        return self.current_mean
    
    def get_std(self) -> float:
        """Get current standard deviation."""
        if not self.stats_valid:
            self._calculate_stats()
        print(f"Frames: {len(self.frames)}, Mean: {self.current_mean:.6f}, Std: {self.current_std:.6f}")
        return self.current_std
    
    def is_ready(self) -> bool:
        """Check if we have enough frames for stable statistics."""
        return len(self.frames) >= min(self.step, self.window_size // 2)
