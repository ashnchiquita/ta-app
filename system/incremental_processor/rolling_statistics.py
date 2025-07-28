import numpy as np
from collections import deque

class RollingStatistics:
    def __init__(self, window_size: int, step=30, frame_shape: tuple = (72,72,3)):
        self.window_size = window_size
        self.step = step
        h, w, c = frame_shape
        self.frame_size = h * w * c
        
        # Store frame-level statistics computed with NumPy (stable)
        self.frame_data = deque(maxlen=window_size + 1)  # Store (mean, var, count) tuples
        
        # Store differential statistics
        self.diff_frame_data = deque(maxlen=window_size)  # Store diff statistics (mean, var, count)
        
        # Running totals for efficient global calculation
        self.total_count = 0
        self.total_sum = 0.0
        self.total_sum_squares = 0.0
        
        # Running totals for differential statistics
        self.diff_total_count = 0
        self.diff_total_sum = 0.0
        self.diff_total_sum_squares = 0.0
        self.diff_frames = deque(maxlen=window_size - 1)  # Store differential frames
        
        # Store previous frame for differential calculation
        self.prev_frame = None
        
        # Track current chunk frames for preprocessing
        self.current_chunk_frames = []

        self.passed_frame = None

    def get_internal_step(self) -> int:
        if self.passed_frame is not None:
            return self.step
        else:
            return self.step + 1

    def get_prev_frame(self):
        """Get the previous frame used for differential calculation."""
        return self.prev_frame
        
    def add_frame(self, frame: np.ndarray):
        """Add a new frame using stable per-frame calculation."""
        frame = frame.astype(np.float32)  # Ensure frame is in float32 for stability
        frame_flat = frame.flatten()
        
        # Add to current chunk frames
        self.current_chunk_frames.append(frame)
        
        # Use NumPy's stable algorithms for this frame
        frame_mean = np.mean(frame_flat)
        frame_var = np.var(frame_flat)  # This is stable
        frame_count = len(frame_flat)
        
        # Calculate what we need for global statistics
        frame_sum = frame_mean * frame_count
        frame_sum_squares = frame_var * frame_count + frame_mean * frame_mean * frame_count
        
        # Remove old frame if at capacity
        if len(self.frame_data) == self.frame_data.maxlen:
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
        
        # Handle differential statistics
        self._update_diff_statistics(frame)
        
        # Store current frame as previous for next iteration
        self.prev_frame = frame.copy()
    
    def _update_diff_statistics(self, current_frame):
        """Update differential statistics using current and previous frame."""
        if self.prev_frame is None:
            print("No previous frame available for differential calculation.")
            return  # Can't calculate diff for first frame
        
        # Calculate differential frame (same as in base.py)
        diff_frame = self._calculate_diff_frame(self.prev_frame, current_frame)
        diff_flat = diff_frame.flatten()
        
        # Calculate diff frame statistics
        diff_mean = np.mean(diff_flat)
        diff_var = np.var(diff_flat)
        diff_count = len(diff_flat)
        
        # Calculate what we need for global diff statistics
        diff_sum = diff_mean * diff_count
        diff_sum_squares = diff_var * diff_count + diff_mean * diff_mean * diff_count
        
        # Remove old diff frame if at capacity
        if len(self.diff_frame_data) == self.diff_frame_data.maxlen:
            old_diff_mean, old_diff_var, old_diff_count = self.diff_frame_data.popleft()
            old_diff_sum = old_diff_mean * old_diff_count
            old_diff_sum_squares = old_diff_var * old_diff_count + old_diff_mean * old_diff_mean * old_diff_count
            
            self.diff_total_sum -= old_diff_sum
            self.diff_total_sum_squares -= old_diff_sum_squares
            self.diff_total_count -= old_diff_count
        
        # Add new diff frame
        self.diff_frame_data.append((diff_mean, diff_var, diff_count))
        self.diff_frames.append(diff_frame)
        self.diff_total_sum += diff_sum
        self.diff_total_sum_squares += diff_sum_squares
        self.diff_total_count += diff_count
    
    def _calculate_diff_frame(self, prev_frame, current_frame):
        """Calculate differential frame as in base.py."""
        diff_frame = (current_frame - prev_frame) / (current_frame + prev_frame + 1e-7)
        return diff_frame
    
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
        return np.sqrt(variance)
    
    def get_diff_variance(self) -> float:
        """Get current differential variance."""
        if self.diff_total_count < 2:
            return 1.0  # Default to 1.0 to avoid division by zero
        
        # Use the stable E[X²] - μ² formula for diff data
        diff_mean = self.diff_total_sum / self.diff_total_count
        diff_mean_of_squares = self.diff_total_sum_squares / self.diff_total_count
        diff_variance = diff_mean_of_squares - (diff_mean * diff_mean)
        
        return max(1e-7, diff_variance)  # Ensure positive and not too small
    
    def get_diff_std(self) -> float:
        """Get current differential standard deviation - O(1) operation."""
        diff_variance = self.get_diff_variance()
        return np.sqrt(diff_variance)
    
    def is_ready(self) -> bool:
        """Check if we have enough frames for stable statistics."""
        return len(self.frame_data) >= self.get_internal_step()

    def is_preprocessing_ready(self) -> bool:
        return self.get_current_chunk_size() >= self.get_internal_step()

    def get_diff_chunk(self) -> np.ndarray:
        """Get latest self.step chunk of differential frames."""
        # if len(self.diff_frame_data) < self.step - 1:
        #     raise ValueError("Not enough differential frames to return a chunk.")

        diff_len = min(len(self.diff_frame_data), self.step)

        latest_frames = list(self.diff_frames)[-diff_len:]

        return np.array(latest_frames)
    
    def get_preprocessed_chunk(self) -> np.ndarray:
        """
        Get preprocessed chunk data ready for rPPG extraction.
        Uses the current chunk frames stored internally.
        Combines standardization and differential normalization.
            
        Returns:
            Preprocessed chunk data with standardized and diff-normalized channels concatenated
        """
        if not self.current_chunk_frames:
            raise ValueError("No current chunk frames available for preprocessing")

        if self.passed_frame is not None:
            self.current_chunk_frames.insert(0, self.passed_frame)  # Ensure last passed frame is included

        self.passed_frame = self.current_chunk_frames[-1]
        self.current_chunk_frames = self.current_chunk_frames[:-1]

        # Convert current chunk frames to numpy array
        chunk_array = np.array(self.current_chunk_frames, dtype=np.float32)
        
        # Get differential normalization
        diffnormalized_chunk = self._get_diff_normalized_chunk()
        
        # Get standardization
        standardized_chunk = self._get_standardized_chunk(chunk_array)

        # Concatenate along channel dimension
        preprocessed_chunk = np.concatenate([diffnormalized_chunk, standardized_chunk], axis=-1)
        
        return preprocessed_chunk
    
    def clear_current_chunk(self):
        """Clear the current chunk frames after processing."""
        self.current_chunk_frames = []
    
    def get_current_chunk_size(self) -> int:
        """Get the number of frames in the current chunk."""
        return len(self.current_chunk_frames)
    
    def _get_standardized_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Standardize chunk using global rolling statistics."""
        global_mean = self.get_mean()
        global_std = self.get_std()
        
        standardized = (chunk - global_mean) / (global_std + 1e-7)
        standardized[np.isnan(standardized)] = 0
        return standardized
    
    def _get_diff_normalized_chunk(self) -> np.ndarray:
        """Apply differential normalization using rolling statistics."""
        global_diff_std = self.get_diff_std()
        global_diff_chunk = self.get_diff_chunk()
        
        n, h, w, c = global_diff_chunk.shape
        
        # Normalize using the global differential std from rolling statistics
        diffnormalized_data = global_diff_chunk / global_diff_std
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        
        return diffnormalized_data
