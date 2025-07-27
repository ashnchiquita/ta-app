"""
Incremental processor for rPPG signal extraction.
Breaks down large windows into smaller chunks to distribute computation load.
"""

import numpy as np
from collections import deque
import time
from typing import Dict, List, Optional, Any
from system.incremental_processor.rolling_statistics import RollingStatistics
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
        
        # Batch processing data
        self.pending_chunks: Dict[int, Dict[str, Any]] = {}  # face_id -> chunk info
        
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
            self._prepare_chunk_for_batch(face_id, timestamp)

    def _prepare_chunk_for_batch(self, face_id: int, timestamp: float):
        """Prepare a chunk for batch processing."""
        face_data = self.face_data[face_id]
        
        if not face_data['statistics'].is_ready():
            # Not enough data for stable statistics, skip this chunk
            print(f"current chunk cleared for face {face_id}")
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
            
            # Store in pending chunks for batch processing
            self.pending_chunks[face_id] = {
                'chunk': chunk,
                'preprocessed_data': preprocessed_chunk
            }
            
        except Exception as e:
            print(f"Error preprocessing chunk {chunk_id} for face {face_id}: {e}")
            chunk.processed = False
            face_data['processed_chunks'].append(chunk)
        
        # Update counters and clear current chunk
        face_data['chunk_counter'] += 1
        face_data['current_chunk_frames'] = []

    def process_pending_chunks(self):
        """Process all pending chunks in a single batch inference call."""
        if not self.pending_chunks:
            return
        
        print(f"Processing {len(self.pending_chunks)} pending chunks for faces: {list(self.pending_chunks.keys())}")
        
        # Check if we should use batch processing
        if hasattr(self.rppg_extractor, 'extract_chunks'):
            print("Using batch processing with extract_chunks")
            self._process_chunks_batch()
        else:
            print("Using individual processing (fallback)")
            self._process_chunks_individual()
        
        # Clear pending chunks
        self.pending_chunks.clear()

    def _process_chunks_batch(self):
        """Process multiple chunks using batch inference."""
        try:
            # Prepare batch data: face_id -> preprocessed_chunk
            batch_data = {}
            face_chunk_map = {}  # face_id -> chunk object
            face_id_list = []  # To maintain order for result mapping
            
            for face_id, chunk_info in self.pending_chunks.items():
                batch_data[face_id] = chunk_info['preprocessed_data']
                face_chunk_map[face_id] = chunk_info['chunk']
                face_id_list.append(face_id)
            
            print(f"Processing batch of {len(batch_data)} chunks using extract_chunks")
            
            # Call batch inference
            results = self.rppg_extractor.extract_chunks(batch_data)
            
            # Distribute results back to faces using ordered mapping
            # results has numeric keys (0, 1, 2, ...) corresponding to order of faces
            for i, face_id in enumerate(face_id_list):
                chunk = face_chunk_map[face_id]
                if i in results:
                    chunk.bvp_signal = results[i]
                    chunk.processed = True
                    # print(f"Successfully processed chunk for face {face_id}")
                else:
                    chunk.processed = False
                    print(f"Failed to get result for face {face_id}")
                
                # Store the processed chunk
                self.face_data[face_id]['processed_chunks'].append(chunk)
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Mark all chunks as failed
            for face_id, chunk_info in self.pending_chunks.items():
                chunk = chunk_info['chunk']
                chunk.processed = False
                self.face_data[face_id]['processed_chunks'].append(chunk)

    def _process_chunks_individual(self):
        """Process chunks individually (fallback method)."""
        for face_id, chunk_info in self.pending_chunks.items():
            chunk = chunk_info['chunk']
            preprocessed_chunk = chunk_info['preprocessed_data']
            
            try:
                # Run inference on the chunk
                if hasattr(self.rppg_extractor, 'extract_chunk'):
                    bvp_signal = self.rppg_extractor.extract_chunk(preprocessed_chunk)
                else:
                    # Fallback to _npu_inference for compatibility
                    bvp_signal = self.rppg_extractor._npu_inference(preprocessed_chunk)
                
                chunk.bvp_signal = bvp_signal
                chunk.processed = True
                
            except Exception as e:
                print(f"Error processing chunk for face {face_id}: {e}")
                chunk.processed = False
            
            # Store the processed chunk
            self.face_data[face_id]['processed_chunks'].append(chunk)
    
    def _preprocess_chunk(self, chunk_frames: List[np.ndarray], 
                         statistics: RollingStatistics) -> np.ndarray:
        """Preprocess a chunk using rolling statistics."""
        # Convert to numpy array
        chunk_array = np.array(chunk_frames, dtype=np.float32)
        
        # Get global statistics for the entire window
        global_mean = statistics.get_mean()
        global_std = statistics.get_std()
        global_diff_std = statistics.get_diff_std()
        global_diff_chunk = statistics.get_diff_chunk()

        diffnormalized_chunk = self._diff_normalize_chunk_try(global_diff_std, global_diff_chunk)
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
    
    def _diff_normalize_chunk_try(self, global_diff_std: float, global_diff_chunk: np.ndarray) -> np.ndarray:
        """Apply differential normalization using the correct approach from base.py."""
        n, h, w, c = global_diff_chunk.shape

        # Step 2: Normalize using the global differential std from rolling statistics
        # This is the key fix - use the std calculated from ALL differential frames in the window
        diffnormalized_data = global_diff_chunk / (global_diff_std)
        
        # Step 3: Add padding for the last frame (to maintain same length as input)
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

    def cleanup(self):
        """Cleanup incremental processor resources."""
        try:
            # Clear all face data
            if hasattr(self, 'face_data'):
                self.face_data.clear()
            
            # Clear pending chunks
            if hasattr(self, 'pending_chunks'):
                self.pending_chunks.clear()
                
        except Exception as e:
            print(f"Warning: Error during incremental processor cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
