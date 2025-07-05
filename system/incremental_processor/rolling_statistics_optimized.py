import numpy as np
from collections import deque

class RollingStatisticsOptimized:
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


class RollingStatisticsWelford:
    """
    Pure Welford's algorithm implementation for maximum numerical stability.
    Best for scenarios where you need the most accurate incremental statistics.
    """

    def __init__(self, window_size: int, step=30, frame_shape: tuple = (72,72,3)):
        self.window_size = window_size
        self.step = step
        h, w, c = frame_shape
        self.frame_size = h * w * c
        
        # Store individual pixel values for precise Welford's algorithm
        self.values = deque(maxlen=window_size * self.frame_size)
        
        # Welford's algorithm state
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        
    def add_frame(self, frame: np.ndarray):
        """Add a new frame using Welford's algorithm."""
        frame_flat = frame.flatten()
        
        # If we're at capacity, remove old values
        values_to_remove = max(0, len(self.values) + len(frame_flat) - self.values.maxlen)
        
        # Remove old values
        for _ in range(values_to_remove):
            if self.values:
                old_value = self.values.popleft()
                self._remove_value(old_value)
        
        # Add new values
        for pixel_value in frame_flat:
            self.values.append(pixel_value)
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
        new_mean = (self.count * self.mean + delta) / self.count if self.count > 0 else 0
        delta2 = value - new_mean
        self.m2 = max(0, self.m2 - delta * delta2)  # Ensure non-negative
        self.mean = new_mean
        
    def get_total_count(self) -> int:
        """Get total number of values in the window."""
        return self.count
    
    def get_mean(self) -> float:
        """Get current mean."""
        return self.mean
    
    def get_variance(self) -> float:
        """Get current variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count
    
    def get_std(self) -> float:
        """Get current standard deviation."""
        variance = self.get_variance()
        print(f"Count: {self.count}, Mean: {self.mean:.6f}, Variance: {variance:.6f}")
        return np.sqrt(variance)
    
    def is_ready(self) -> bool:
        """Check if we have enough data for stable statistics."""
        return self.count >= min(self.step, self.window_size * self.frame_size // 2)


class RollingStatisticsLightweight:
    """
    Lightweight rolling statistics that stores frame-level pre-computed statistics.
    This version is both fast and accurate by using NumPy's stable algorithms per frame.
    """

    def __init__(self, window_size: int, step=30, frame_shape: tuple = (72,72,3)):
        self.window_size = window_size
        self.step = step
        h, w, c = frame_shape
        self.frame_size = h * w * c
        
        # Store frame-level statistics computed with NumPy (stable)
        self.frame_data = deque(maxlen=window_size)  # Store (mean, var, count) tuples
        
        # Running totals for efficient global calculation
        self.total_count = 0
        self.total_sum = 0.0
        self.total_sum_squares = 0.0
        
    def add_frame(self, frame: np.ndarray):
        """Add a new frame using stable per-frame calculation."""
        frame_flat = frame.flatten()
        
        # Use NumPy's stable algorithms for this frame
        frame_mean = np.mean(frame_flat)
        frame_var = np.var(frame_flat)  # This is stable
        frame_count = len(frame_flat)
        
        # Calculate what we need for global statistics
        frame_sum = frame_mean * frame_count
        frame_sum_squares = frame_var * frame_count + frame_mean * frame_mean * frame_count
        
        # Remove old frame if at capacity
        if len(self.frame_data) == self.window_size:
            old_mean, old_var, old_count = self.frame_data.popleft()
            old_sum = old_mean * old_count
            old_sum_squares = old_var * old_count + old_mean * old_mean * old_count
            
            self.total_sum -= old_sum
            self.total_sum_squares -= old_sum_squares
            self.total_count -= old_count
        
        # Add new frame
        self.frame_data.append((frame_mean, frame_var, frame_count))
        self.total_sum += frame_sum
        self.total_sum_squares += frame_sum_squares
        self.total_count += frame_count
    
    def get_total_count(self) -> int:
        """Get total number of pixel values in the window."""
        return self.total_count
    
    def get_mean(self) -> float:
        """Get current mean - O(1) operation."""
        if self.total_count == 0:
            return 0.0
        return self.total_sum / self.total_count
    
    def get_variance(self) -> float:
        """Get current variance using stable calculation."""
        if self.total_count < 2:
            return 0.0
        
        # Use the stable E[X²] - μ² formula but with high precision
        mean = self.get_mean()
        mean_of_squares = self.total_sum_squares / self.total_count
        variance = mean_of_squares - (mean * mean)
        
        return max(0.0, variance)  # Ensure non-negative
    
    def get_std(self) -> float:
        """Get current standard deviation - O(1) operation."""
        variance = self.get_variance()
        print(f"Frames: {len(self.frame_data)}, Mean: {self.get_mean():.6f}, Variance: {variance:.6f}")
        return np.sqrt(variance)
    
    def is_ready(self) -> bool:
        """Check if we have enough frames for stable statistics."""
        return len(self.frame_data) >= min(self.step, self.window_size // 2)


class RollingStatisticsUltraLight:
    """
    Ultra-lightweight version that uses proper variance combination formulas.
    Very fast and surprisingly accurate for most use cases.
    """

    def __init__(self, window_size: int, step=30, frame_shape: tuple = (72,72,3)):
        self.window_size = window_size
        self.step = step
        h, w, c = frame_shape
        self.frame_size = h * w * c
        
        # Store frame-level statistics (lightweight)
        self.frame_stats = deque(maxlen=window_size)  # (mean, var, count) tuples
        
    def add_frame(self, frame: np.ndarray):
        """Add frame using ultra-lightweight approach with proper variance combination."""
        frame_flat = frame.flatten()
        frame_mean = np.mean(frame_flat)
        frame_var = np.var(frame_flat)  # NumPy's stable calculation
        frame_count = len(frame_flat)
        
        # Store frame statistics
        if len(self.frame_stats) == self.window_size:
            self.frame_stats.popleft()
        
        self.frame_stats.append((frame_mean, frame_var, frame_count))
        
    def get_total_count(self) -> int:
        """Get total number of pixel values in the window."""
        return sum(count for _, _, count in self.frame_stats)
    
    def get_mean(self) -> float:
        """Get current mean using proper weighted average."""
        if not self.frame_stats:
            return 0.0
        
        total_sum = sum(mean * count for mean, _, count in self.frame_stats)
        total_count = sum(count for _, _, count in self.frame_stats)
        
        return total_sum / total_count if total_count > 0 else 0.0
    
    def get_variance(self) -> float:
        """Get current variance using proper variance combination formula."""
        if len(self.frame_stats) < 1:
            return 0.0
        
        # Calculate global mean first
        global_mean = self.get_mean()
        total_count = sum(count for _, _, count in self.frame_stats)
        
        if total_count < 2:
            return 0.0
        
        # Combine variances using the correct formula:
        # Var(combined) = Σ[n_i * (var_i + (mean_i - global_mean)²)] / total_n
        combined_variance = 0.0
        
        for frame_mean, frame_var, frame_count in self.frame_stats:
            # Within-group variance contribution
            within_variance = frame_count * frame_var
            
            # Between-group variance contribution
            between_variance = frame_count * ((frame_mean - global_mean) ** 2)
            
            combined_variance += within_variance + between_variance
        
        return combined_variance / total_count
    
    def get_std(self) -> float:
        """Get current standard deviation."""
        variance = self.get_variance()
        print(f"Frames: {len(self.frame_stats)}, Mean: {self.get_mean():.6f}, Variance: {variance:.6f}")
        return np.sqrt(variance)
    
    def is_ready(self) -> bool:
        """Check if we have enough frames for stable statistics."""
        return len(self.frame_stats) >= min(self.step, self.window_size // 2)
