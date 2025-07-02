from collections import deque
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import threading
from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor
from components.rppg_signal_extractor.deep_learning.hef.base import HEFModel

class IncrementalStatistics:
    """Manages incremental statistics for sliding window preprocessing"""
    def __init__(self, window_size, roi_shape):
        self.window_size = window_size
        self.roi_shape = roi_shape
        
        # Welford's algorithm for mean and variance
        self.count = 0
        self.mean = np.zeros(roi_shape, dtype=np.float64)
        self.m2 = np.zeros(roi_shape, dtype=np.float64)
        
        # Sliding window for min/max tracking
        self.data_buffer = deque(maxlen=window_size)
        self.min_vals = deque()
        self.max_vals = deque()
        
    def add_frame(self, frame):
        """Add a new frame and update statistics"""
        frame = frame.astype(np.float64)
        
        # Remove old frame if buffer is full
        if len(self.data_buffer) == self.window_size:
            old_frame = self.data_buffer[0]
            self._remove_frame_stats(old_frame)
        
        # Add new frame
        self.data_buffer.append(frame)
        self._add_frame_stats(frame)
        
        # Update min/max with sliding window
        self._update_minmax(frame)
    
    def _add_frame_stats(self, frame):
        """Add frame to Welford's algorithm"""
        self.count += 1
        delta = frame - self.mean
        self.mean += delta / self.count
        delta2 = frame - self.mean
        self.m2 += delta * delta2
    
    def _remove_frame_stats(self, frame):
        """Remove frame from Welford's algorithm"""
        if self.count <= 1:
            return
        
        delta = frame - self.mean
        self.mean = (self.count * self.mean - frame) / (self.count - 1)
        delta2 = frame - self.mean
        self.m2 -= delta * delta2
        self.count -= 1
    
    def _update_minmax(self, new_frame):
        """Update min/max using sliding window approach"""
        # Remove outdated min/max values
        while self.min_vals and len(self.data_buffer) > len(self.min_vals):
            self.min_vals.popleft()
        while self.max_vals and len(self.data_buffer) > len(self.max_vals):
            self.max_vals.popleft()
        
        # Add new frame min/max
        frame_min = np.min(new_frame)
        frame_max = np.max(new_frame)
        
        # Maintain monotonic deque for min
        while self.min_vals and self.min_vals[-1] >= frame_min:
            self.min_vals.pop()
        self.min_vals.append(frame_min)
        
        # Maintain monotonic deque for max  
        while self.max_vals and self.max_vals[-1] <= frame_max:
            self.max_vals.pop()
        self.max_vals.append(frame_max)
    
    def get_statistics(self):
        """Get current statistics for preprocessing"""
        if self.count < 2:
            return None, None, None, None
        
        mean = self.mean
        variance = self.m2 / (self.count - 1)
        std = np.sqrt(variance)
        
        # Global min/max across all frames in window
        if self.data_buffer:
            global_min = np.min([np.min(frame) for frame in self.data_buffer])
            global_max = np.max([np.max(frame) for frame in self.data_buffer])
        else:
            global_min = global_max = 0
        
        return mean, std, global_min, global_max

class Pipeline:
    def __init__(self, 
                    rppg_signal_extractor, 
                    hr_extractor,
                    window_size=180,
                    fps=30,
                    step_size=30):
        self.rppg_signal_extractor = rppg_signal_extractor
        self.hr_extractor = hr_extractor
        self.window_size = window_size
        self.step_size = step_size
        self.face_data = {}
        self.fps = fps
        
        # Enable incremental processing for deep learning models
        self.use_incremental = isinstance(rppg_signal_extractor, DeepLearningRPPGSignalExtractor)
        self.process_every_n_frames = max(1, step_size // 6)  # Process every ~5-30 frames
            
    def add_face_data(self, face_id, roi, timestamp):
        if face_id not in self.face_data:
            self.face_data[face_id] = {
                'timestamps': deque(maxlen=self.window_size),
                'roi_data': deque(maxlen=self.window_size),
                'heart_rate': None,
                'last_processed': 0,
                'frame_count': 0,
                'bvp_segments': deque(),
                'statistics': IncrementalStatistics(self.window_size, roi.shape) if self.use_incremental and roi is not None else None
            }
        
        # Add new data
        if roi is not None:    # Only add if ROI is valid
            self.face_data[face_id]['timestamps'].append(timestamp)
            self.face_data[face_id]['roi_data'].append(roi)
            self.face_data[face_id]['frame_count'] += 1
            
            # Update incremental statistics for deep learning models
            if self.use_incremental and self.face_data[face_id]['statistics'] is not None:
                self.face_data[face_id]['statistics'].add_frame(roi)
                
                # Process partial segments periodically
                if (self.face_data[face_id]['frame_count'] % self.process_every_n_frames == 0 and 
                    len(self.face_data[face_id]['roi_data']) >= self.process_every_n_frames):
                    self._process_incremental_segment(face_id)

    def process_faces(self):
        current_time = time.time()
        
        # Identify faces ready for processing
        faces_to_process = []
        for face_id, data in self.face_data.items():
            if (len(data['roi_data']) >= self.window_size and 
                current_time - data['last_processed'] >= (self.step_size / self.fps)):
                
                # For incremental processing, use combined BVP segments
                if self.use_incremental and data['bvp_segments']:
                    combined_bvp = self._combine_bvp_segments(face_id)
                    if combined_bvp is not None:
                        faces_to_process.append((face_id, combined_bvp, data, 'bvp'))
                else:
                    # Traditional processing: full ROI data
                    listed_roi_data = list(data['roi_data'])
                    try:
                        roi_data = np.array(listed_roi_data)
                    except Exception as e:  # Might happen if roi doesn't have static shape
                        roi_data = listed_roi_data
                    faces_to_process.append((face_id, roi_data, data, 'roi'))
        
        if not faces_to_process:
            return {}, 0

        # Process faces based on data type
        if len(faces_to_process) == 1:
            return self._process_faces_optimized(faces_to_process, current_time)
        
        if isinstance(self.rppg_signal_extractor, DeepLearningRPPGSignalExtractor):
            return self._process_faces_optimized(faces_to_process, current_time)

        # Conventional methods can still use parallel processing
        return self.process_faces_parallel(faces_to_process, current_time)
    
    def _process_faces_optimized(self, faces_to_process, current_time):
        """Optimized processing that handles both incremental and traditional approaches"""
        results = {}
        
        for face_id, data_input, face_data, data_type in faces_to_process:
            try:
                if data_type == 'bvp':
                    # Data is already BVP from incremental processing
                    heart_rate = self.hr_extractor.extract(data_input)
                else:
                    # Traditional processing: ROI -> BVP -> HR
                    pulse_signal, preprocess_time, inference_time = self.rppg_signal_extractor.extract(data_input)
                    heart_rate = self.hr_extractor.extract(pulse_signal)
                
                results[face_id] = heart_rate
                face_data['heart_rate'] = heart_rate
                face_data['last_processed'] = current_time
                
                # Clear old BVP segments after successful processing
                if data_type == 'bvp' and 'bvp_segments' in face_data:
                    face_data['bvp_segments'].clear()
                    
            except Exception as e:
                print(f"Error processing face {face_id}: {e}")
        
        core_time = time.time() - current_time
        return results, core_time

    def process_faces_producer_consumer(self, faces_to_process, current_time):
        """
        Async producer-consumer pattern:
        - Producer thread: extracts pulse signals from ROI data
        - Consumer thread: extracts heart rate from pulse signals
        - Ensures all faces are processed while maintaining async behavior
        """
        # Queues for producer-consumer communication
        pulse_queue = Queue()
        result_queue = Queue()
        
        # Track completion
        total_faces = len(faces_to_process)
        completed_faces = 0
        consumed_face_ids = set()
        
        def producer():
            """Producer: Extract pulse signals from ROI data"""
            for face_id, data_input, face_data, data_type in faces_to_process:
                try:
                    if data_type == 'bvp':
                        # Data is already BVP
                        pulse_queue.put((face_id, data_input, face_data))
                    else:
                        # Extract BVP from ROI data
                        pulse_signal, preprocess_time, inference_time = self.rppg_signal_extractor.extract(data_input)
                        pulse_queue.put((face_id, pulse_signal, face_data))
                except Exception as e:
                    print(f"Error extracting pulse signal for face {face_id}: {e}")
                    pulse_queue.put((face_id, None, face_data))  # Signal error but continue
        
        def consumer():
            """Consumer: Extract heart rates from pulse signals using multi-threading"""
            nonlocal completed_faces
            nonlocal consumed_face_ids
            
            # Adaptive worker count based on system and workload
            import os
            cpu_count = os.cpu_count() or 4
            
            if total_faces == 1:
                max_workers = 1
            elif isinstance(self.rppg_signal_extractor, DeepLearningRPPGSignalExtractor):
                # Check for Hailo NPU accelerator
                if isinstance(self.rppg_signal_extractor, HEFModel):
                    # Hailo NPU: Preprocessing can be parallel, but NPU inference is serialized
                    # Use more workers since preprocessing + postprocessing can be parallel
                    max_workers = min(total_faces, max(4, cpu_count // 2))
                else:
                    # Other ML models (GPU/CPU)
                    max_workers = min(total_faces, max(2, cpu_count // 3))
            else:
                # Traditional DSP (FFT, etc.) - good parallelism since they release GIL
                max_workers = min(total_faces, max(3, cpu_count - 1))
            
            # Cap at reasonable maximum to prevent resource exhaustion
            max_workers = min(max_workers, 10)  # Higher cap for NPU scenarios
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_face = {}
                
                while completed_faces < total_faces:
                    processed_any_work = False
                    
                    try:
                        if not pulse_queue.empty():
                            face_id, pulse_signal, face_data = pulse_queue.get_nowait()
                            processed_any_work = True

                            if pulse_signal is not None:
                                future = executor.submit(self.hr_extractor.extract, pulse_signal)
                                future_to_face[future] = (face_id, face_data)
                            else:
                                result_queue.put((face_id, None, face_data))
                                completed_faces += 1
                            pulse_queue.task_done()

                        completed_futures = []
                        for future in future_to_face:
                            if future.done():
                                completed_futures.append(future)
                        
                        for future in completed_futures:
                            face_id, face_data = future_to_face.pop(future)
                            processed_any_work = True
                            try:
                                heart_rate = future.result()
                                result_queue.put((face_id, heart_rate, face_data))
                            except Exception as e:
                                print(f"Error extracting heart rate for face {face_id}: {e}")
                                result_queue.put((face_id, None, face_data))
                            
                            completed_faces += 1
                        
                        # yield CPU to avoid tight loop
                        if not processed_any_work:
                            time.sleep(0.001)
                            
                    except Exception as e:
                        print(f"Consumer error: {e}")
                        break
                
                # Wait for any remaining futures to complete
                for future in future_to_face:
                    face_id, face_data = future_to_face[future]
                    try:
                        heart_rate = future.result()
                        result_queue.put((face_id, heart_rate, face_data))
                    except Exception as e:
                        print(f"Error extracting heart rate for face {face_id}: {e}")
                        result_queue.put((face_id, None, face_data))
        
        # Start producer and consumer threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        # Wait for both threads to complete
        producer_thread.join()
        consumer_thread.join()

        # Collect results
        processed_faces = 0
        results = {}
        while processed_faces < total_faces and not result_queue.empty():
            try:
                face_id, heart_rate, face_data = result_queue.get_nowait()
                
                if heart_rate is not None:
                    results[face_id] = heart_rate
                    face_data['heart_rate'] = heart_rate
                    face_data['last_processed'] = current_time
                
                # Clear BVP segments after processing
                if 'bvp_segments' in face_data:
                    face_data['bvp_segments'].clear()
                
                processed_faces += 1
                
            except Exception as e:
                print(f"Error collecting result: {e}")
                break
            except Empty:
                # Queue became empty, yield CPU briefly
                time.sleep(0.001)
                break
        
        core_time = time.time() - current_time
        return results, core_time
        
    def process_faces_batch_flattened(self, faces_to_process, current_time):
        # merge all ROIs into a single batch
        roi_batch = np.array([data_input for _, data_input, _, data_type in faces_to_process if data_type == 'roi'])
        if len(roi_batch) == 0:
            return {}, 0
            
        roi_batch = roi_batch.reshape(-1, *roi_batch.shape[2:])  # Flatten batch for processing

        pulse_signal, preprocess_time, inference_time = self.rppg_signal_extractor.extract(roi_batch)
        results = {}
        roi_idx = 0
        for face_id, data_input, face_data, data_type in faces_to_process:
            if data_type == 'roi':
                curr_signal = pulse_signal[180*roi_idx:180*(roi_idx+1)]
                heart_rate = self.hr_extractor.extract(curr_signal)
                results[face_id] = heart_rate
                face_data['heart_rate'] = heart_rate
                face_data['last_processed'] = current_time
                roi_idx += 1
            elif data_type == 'bvp':
                # Direct heart rate extraction
                heart_rate = self.hr_extractor.extract(data_input)
                results[face_id] = heart_rate
                face_data['heart_rate'] = heart_rate
                face_data['last_processed'] = current_time
                # Clear BVP segments
                if 'bvp_segments' in face_data:
                    face_data['bvp_segments'].clear()

        core_time = time.time() - current_time
        return results, core_time

    def process_faces_parallel(self, faces_to_process, current_time):
        results = {}
        max_workers = min(len(faces_to_process), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_face_map = {}
            
            for face_id, data_input, face_data, data_type in faces_to_process:
                if data_type == 'bvp':
                    # Direct HR extraction from BVP
                    future = executor.submit(self.hr_extractor.extract, data_input)
                else:
                    # Traditional ROI processing
                    future = executor.submit(self._thread_process_face, data_input)
                future_face_map[future] = (face_id, face_data, data_type)
            
            for future in as_completed(future_face_map):
                face_id, face_data, data_type = future_face_map[future]
                try:
                    heart_rate = future.result()
                    
                    results[face_id] = heart_rate
                    face_data['heart_rate'] = heart_rate
                    face_data['last_processed'] = current_time
                    
                    # Clear BVP segments after processing
                    if data_type == 'bvp' and 'bvp_segments' in face_data:
                        face_data['bvp_segments'].clear()
                        
                except Exception as e:
                    print(f"Error processing face {face_id}: {e}")

        core_time = time.time() - current_time
        return results, core_time

    def _thread_process_face(self, roi_data):
        pulse_signal, preprocess_time, inference_time = self.rppg_signal_extractor.extract(roi_data)
        heart_rate = self.hr_extractor.extract(pulse_signal)
        
        return heart_rate

    def process_faces_sequential(self, faces_to_process, current_time):
        
        results = {}

        for face_id, data_input, face_data, data_type in faces_to_process:
            try:
                if data_type == 'bvp':
                    # Data is already BVP, just extract heart rate
                    heart_rate = self.hr_extractor.extract(data_input)
                else:
                    # Traditional processing
                    pulse_signal, preprocess_time, inference_time = self.rppg_signal_extractor.extract(data_input)
                    heart_rate = self.hr_extractor.extract(pulse_signal)
            except Exception as e:
                print(f"Error processing face {face_id}: {e}")
                continue
            
            face_data['heart_rate'] = heart_rate
            face_data['last_processed'] = current_time
            results[face_id] = heart_rate
            
            # Clear BVP segments after processing
            if data_type == 'bvp' and 'bvp_segments' in face_data:
                face_data['bvp_segments'].clear()
        
        core_time = time.time() - current_time
        
        return results, core_time

    def _process_incremental_segment(self, face_id):
        """Process a small segment of frames for incremental BVP extraction"""
        try:
            data = self.face_data[face_id]
            roi_data = list(data['roi_data'])
            statistics = data['statistics']
            
            # Get current statistics for preprocessing
            mean, std, global_min, global_max = statistics.get_statistics()
            if mean is None:
                return  # Not enough data yet
            
            # Take the most recent segment for processing
            segment_size = min(self.process_every_n_frames, len(roi_data))
            recent_segment = roi_data[-segment_size:]
            
            # Preprocess the segment using current window statistics
            preprocessed_segment = self._preprocess_with_statistics(
                recent_segment, mean, std, global_min, global_max
            )
            
            # Extract BVP for this segment
            if preprocessed_segment is not None:
                bvp_segment, _, _ = self.rppg_signal_extractor.extract(preprocessed_segment)
                
                # Store the segment with timestamp
                segment_info = {
                    'bvp': bvp_segment,
                    'start_idx': len(roi_data) - segment_size,
                    'end_idx': len(roi_data),
                    'timestamp': time.time()
                }
                data['bvp_segments'].append(segment_info)
                
                # Keep only recent segments to avoid memory buildup
                max_segments = (self.window_size // self.process_every_n_frames) + 2
                while len(data['bvp_segments']) > max_segments:
                    data['bvp_segments'].popleft()
                    
        except Exception as e:
            print(f"Error in incremental processing for face {face_id}: {e}")
    
    def _preprocess_with_statistics(self, roi_segment, mean, std, global_min, global_max):
        """Preprocess ROI segment using provided statistics (mimicking DeepPhys preprocessing)"""
        try:
            roi_segment = np.array(roi_segment, dtype=np.float32)
            
            # Differential normalization (mimicking diff_normalize_data)
            if len(roi_segment) > 1:
                diff_data = np.diff(roi_segment, axis=0)
                # Normalize differences using global min/max
                if global_max > global_min:
                    diff_normalized = (diff_data - global_min) / (global_max - global_min)
                else:
                    diff_normalized = diff_data
                # Pad to match original length
                diff_normalized = np.concatenate([diff_normalized[0:1], diff_normalized], axis=0)
            else:
                diff_normalized = roi_segment
            
            # Standardization (mimicking standardized_data) 
            std_safe = np.where(std > 1e-8, std, 1.0)  # Avoid division by zero
            standardized = (roi_segment - mean) / std_safe
            
            # Concatenate along channel dimension (matches DeepPhys preprocessing)
            preprocessed = np.concatenate([diff_normalized, standardized], axis=-1)
            
            return preprocessed
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def _combine_bvp_segments(self, face_id):
        """Combine stored BVP segments into full window BVP"""
        try:
            data = self.face_data[face_id]
            segments = list(data['bvp_segments'])
            
            if not segments:
                return None
            
            # Sort segments by start index to ensure correct order
            segments.sort(key=lambda x: x['start_idx'])
            
            # Combine BVP segments
            combined_bvp = []
            for segment in segments:
                if segment['bvp'] is not None:
                    if len(segment['bvp'].shape) > 1:
                        segment_bvp = segment['bvp'].flatten()
                    else:
                        segment_bvp = segment['bvp']
                    combined_bvp.extend(segment_bvp)
            
            if not combined_bvp:
                return None
            
            # Ensure we have the right length (window_size)
            combined_bvp = np.array(combined_bvp)
            if len(combined_bvp) > self.window_size:
                combined_bvp = combined_bvp[-self.window_size:]  # Take most recent
            elif len(combined_bvp) < self.window_size:
                # Pad if necessary (shouldn't happen in normal operation)
                padding = self.window_size - len(combined_bvp)
                combined_bvp = np.pad(combined_bvp, (padding, 0), mode='edge')
            
            return combined_bvp
            
        except Exception as e:
            print(f"Error combining BVP segments for face {face_id}: {e}")
            return None
