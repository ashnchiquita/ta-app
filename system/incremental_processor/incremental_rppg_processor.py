"""
Incremental processor for rPPG signal extraction.
Breaks down large windows into smaller chunks to distribute computation load.
"""

import numpy as np
from collections import deque
import time
from typing import Dict, List, Tuple, Optional, Any
# from system.optimization_utils import get_memory_manager, get_resource_pool
# from system.incremental_processor.rolling_statistics import RollingStatistics
# from system.incremental_processor.rolling_statistics import RollingStatistics
# from system.incremental_processor.rolling_statistics_original import RollingStatisticsOriginal as RollingStatistics
from system.incremental_processor.rolling_statistics_optimized import RollingStatisticsLightweight as RollingStatistics
from system.incremental_processor.incremental_chunk import IncrementalChunk


class IncrementalRPPGProcessor:
    """
    Incremental processor for rPPG signal extraction.
    Breaks down large windows into smaller chunks to distribute computation.
    """
    
    def __init__(self, rppg_extractor, hr_extractor, 
                 window_size: int = 180, chunk_size: int = 30, step_size=180):
        self.rppg_extractor = rppg_extractor
        self.hr_extractor = hr_extractor
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.chunks_per_window = window_size // chunk_size
        self.step_size = step_size
        self.reset_chunk_step = self.step_size // chunk_size
        
        # Per-face data
        self.face_data: Dict[int, Dict[str, Any]] = {}
        
        # Optimization utilities
        # self.memory_manager = get_memory_manager()
        # self.resource_pool = get_resource_pool()
        
    def add_face_frame(self, face_id: int, roi_frame: np.ndarray, timestamp: float):
        """Add a new ROI frame for a face."""
        if face_id not in self.face_data:
            self.face_data[face_id] = {
                'statistics': RollingStatistics(self.window_size),
                'current_chunk_frames': [],
                'processed_chunks': deque(maxlen=self.chunks_per_window),
                'chunk_counter': 0,
                'last_hr': None,
                'last_hr_timestamp': 0
            }
        
        face_data = self.face_data[face_id]
        
        # Add frame to rolling statistics
        face_data['statistics'].add_frame(roi_frame)
        
        # Add frame to current chunk
        face_data['current_chunk_frames'].append(roi_frame)
        
        # Check if chunk is ready for processing
        if len(face_data['current_chunk_frames']) >= self.chunk_size:
            self._process_chunk(face_id, timestamp)

    def _process_chunk(self, face_id: int, timestamp: float):
        """Process a complete chunk for a face."""
        face_data = self.face_data[face_id]
        
        if not face_data['statistics'].is_ready():
            # Not enough data for stable statistics, skip this chunk
            print(f"current chunk cleared")
            face_data['current_chunk_frames'] = []
            return
        
        # Get current chunk frames
        chunk_frames = face_data['current_chunk_frames'][:self.chunk_size]
        chunk_id = face_data['chunk_counter']
        
        # Create chunk object
        chunk = IncrementalChunk(chunk_id, chunk_frames, timestamp=timestamp)
        
        try:
            # Preprocess the chunk using rolling statistics
            preprocessed_chunk = self._preprocess_chunk(
                chunk_frames, 
                face_data['statistics']
            )
            
            # Run inference on the chunk
            if hasattr(self.rppg_extractor, 'extract_chunk'):
                bvp_signal = self.rppg_extractor.extract_chunk(preprocessed_chunk)
            else:
                # Fallback to _npu_inference for compatibility
                bvp_signal = self.rppg_extractor._npu_inference(preprocessed_chunk)
            
            chunk.bvp_signal = bvp_signal
            chunk.processed = True
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id} for face {face_id}: {e}")
            chunk.processed = False
        
        # Store the processed chunk
        face_data['processed_chunks'].append(chunk)
        face_data['chunk_counter'] += 1
        
        # Clear current chunk
        face_data['current_chunk_frames'] = []
    
    def _preprocess_chunk(self, chunk_frames: List[np.ndarray], 
                         statistics: RollingStatistics) -> np.ndarray:
        """Preprocess a chunk using rolling statistics."""
        # Convert to numpy array
        chunk_array = np.array(chunk_frames, dtype=np.float32)
        
        # Get global statistics for the entire window
        global_mean = statistics.get_mean()
        global_std = statistics.get_std()

        print(f"Global mean: {global_mean}, Global std: {global_std}")
        
        # Apply preprocessing similar to the original method
        # but using global window statistics
        diffnormalized_chunk = self._diff_normalize_chunk(chunk_array, global_std)
        standardized_chunk = self._standardize_chunk(chunk_array, global_mean, global_std)
        
        # Concatenate along channel dimension
        preprocessed_chunk = np.concatenate([diffnormalized_chunk, standardized_chunk], axis=-1)
        
        return preprocessed_chunk
    
    def _standardize_chunk(self, chunk: np.ndarray, global_mean: np.ndarray, 
                          global_std: np.ndarray) -> np.ndarray:
        """Standardize chunk using global statistics."""
        standardized = (chunk - global_mean) / (global_std + 1e-7)
        standardized[np.isnan(standardized)] = 0
        return standardized
    
    def _diff_normalize_chunk(self, chunk: np.ndarray, global_std: np.ndarray) -> np.ndarray:
        """Apply differential normalization using global statistics."""
        n, h, w, c = chunk.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (
                (chunk[j + 1, :, :, :] - chunk[j, :, :, :]) / 
                (chunk[j + 1, :, :, :] + chunk[j, :, :, :] + 1e-7)
            )
        
        # Normalize using global standard deviation
        diffnormalized_data = diffnormalized_data / (global_std + 1e-7)
        
        # Add padding for the last frame
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        
        return diffnormalized_data
    
    def get_heart_rate(self, face_id: int) -> Optional[float]:
        """Get heart rate for a face if enough chunks are processed."""
        if face_id not in self.face_data:
            return None
        
        face_data = self.face_data[face_id]
        processed_chunks = [chunk for chunk in face_data['processed_chunks'] 
                          if chunk.processed]
        
        # Need full window of chunks for heart rate extraction
        if len(processed_chunks) < self.chunks_per_window:
            return None
        
        try:
            # Concatenate BVP signals from all chunks
            bvp_signals = [chunk.bvp_signal for chunk in processed_chunks[-self.chunks_per_window:]]
            combined_bvp = np.concatenate(bvp_signals, axis=0)
            
            # Extract heart rate
            heart_rate = self.hr_extractor.extract(combined_bvp)

            print(f"Extracted heart rate for face {face_id}: {heart_rate} bpm")
            
            # Update last heart rate
            face_data['last_hr'] = heart_rate
            face_data['last_hr_timestamp'] = time.time()

            # Reset chunk by self.reset_chunk_step
            for i in range(self.reset_chunk_step):
                if len(face_data['processed_chunks']) > 0:
                    face_data['processed_chunks'].popleft()
        
            return heart_rate
            
        except Exception as e:
            print(f"Error extracting heart rate for face {face_id}: {e}")
            return face_data['last_hr']  # Return last known good value
    
    def get_processing_info(self, face_id: int) -> Dict[str, Any]:
        """Get processing information for a face."""
        if face_id not in self.face_data:
            return {}
        
        face_data = self.face_data[face_id]
        processed_chunks = [chunk for chunk in face_data['processed_chunks'] 
                          if chunk.processed]
        
        return {
            'total_chunks': len(face_data['processed_chunks']),
            'processed_chunks': len(processed_chunks),
            'current_chunk_frames': len(face_data['current_chunk_frames']),
            'ready_for_hr': len(processed_chunks) >= self.chunks_per_window,
            'statistics_ready': face_data['statistics'].is_ready(),
            'last_hr': face_data['last_hr'],
            'progress': len(processed_chunks) / self.chunks_per_window
        }
    
    def cleanup_old_faces(self, current_time: float, timeout: float = 30.0):
        """Remove data for faces that haven't been seen recently."""
        faces_to_remove = []
        
        for face_id, face_data in self.face_data.items():
            if face_data['last_hr_timestamp'] != 0 and (current_time - face_data['last_hr_timestamp']) > timeout:
                faces_to_remove.append(face_id)
        
        for face_id in faces_to_remove:
            del self.face_data[face_id]
        
        # Perform memory optimization if needed
        # if faces_to_remove:
        #     self.memory_manager.optimize_memory()
