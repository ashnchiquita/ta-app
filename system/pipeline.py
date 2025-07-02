from collections import deque
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import threading
from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor
from components.rppg_signal_extractor.deep_learning.hef.base import HEFModel
from system.incremental_processor import IncrementalRPPGProcessor
from system.performance_monitor import PerformanceMonitor

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
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Check if we should use incremental processing
        self.use_incremental = isinstance(rppg_signal_extractor, DeepLearningRPPGSignalExtractor)
        
        if self.use_incremental:
            # Use smaller chunk size for better load distribution
            chunk_size = max(1, min(30, window_size // 6))  # 6 chunks per window
            self.incremental_processor = IncrementalRPPGProcessor(
                rppg_signal_extractor, hr_extractor, window_size, chunk_size
            )
            print(f"Using incremental processing with chunk_size={chunk_size}")
        else:
            self.incremental_processor = None
            print("Using traditional batch processing")
            
    def add_face_data(self, face_id, roi, timestamp):
        if self.use_incremental:
            # Use incremental processor
            if roi is not None:
                self.incremental_processor.add_face_frame(face_id, roi, timestamp)
        else:
            # Traditional approach
            if face_id not in self.face_data:
                self.face_data[face_id] = {
                    'timestamps': deque(maxlen=self.window_size),
                    'roi_data': deque(maxlen=self.window_size),
                    'heart_rate': None,
                    'last_processed': 0
                }
            
            # Add new data
            if roi is not None:    # Only add if ROI is valid
                self.face_data[face_id]['timestamps'].append(timestamp)
                self.face_data[face_id]['roi_data'].append(roi)

    def process_faces(self):
        current_time = time.time()
        
        if self.use_incremental:
            return self.process_faces_incremental(current_time)
        else:
            return self.process_faces_traditional(current_time)
    
    def process_faces_incremental(self, current_time):
        """Process faces using incremental approach - no burst computation."""
        start_time = time.time()
        results = {}
        
        # Check each face for available heart rate
        face_count = 0
        for face_id in list(self.incremental_processor.face_data.keys()):
            heart_rate = self.incremental_processor.get_heart_rate(face_id)
            if heart_rate is not None:
                results[face_id] = heart_rate
                self.performance_monitor.record_hr_update(face_id)
            face_count += 1
        
        # Record performance metrics
        self.performance_monitor.record_faces_processed(face_count)
        
        # Cleanup old faces
        self.incremental_processor.cleanup_old_faces(current_time)
        
        # Minimal processing time since computation is distributed
        core_time = time.time() - start_time
        self.performance_monitor.record_processing_time(core_time)
        return results, core_time
    
    def process_faces_traditional(self, current_time):
        """Traditional batch processing approach."""
        # Identify faces ready for processing
        faces_to_process = []
        for face_id, data in self.face_data.items():
            if (len(data['roi_data']) >= self.window_size and 
                current_time - data['last_processed'] >= (self.step_size / self.fps)):
                
                listed_roi_data = list(data['roi_data'])
                try:
                    roi_data = np.array(listed_roi_data)
                except Exception as e:  # Might happen if roi doesn't have static shape
                    roi_data = listed_roi_data
                    
                faces_to_process.append((face_id, roi_data, data))
        
        if not faces_to_process:
            return {}, 0

        if len(faces_to_process) == 1:
            return self.process_faces_sequential(faces_to_process, current_time)
        
        if isinstance(self.rppg_signal_extractor, DeepLearningRPPGSignalExtractor):
            # Todo: batch processing slows down the system because of memory limitations. find out best batch size then try again
            return self.process_faces_producer_consumer(faces_to_process, current_time)

        # Conventional
        return self.process_faces_parallel(self, faces_to_process, current_time)

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
            for face_id, roi_data, data in faces_to_process:
                try:
                    pulse_signal, preprocess_time, inference_time = self.rppg_signal_extractor.extract(roi_data)
                    pulse_queue.put((face_id, pulse_signal, data))
                except Exception as e:
                    print(f"Error extracting pulse signal for face {face_id}: {e}")
                    pulse_queue.put((face_id, None, data))  # Signal error but continue
        
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
                            face_id, pulse_signal, data = pulse_queue.get_nowait()
                            processed_any_work = True

                            if pulse_signal is not None:
                                future = executor.submit(self.hr_extractor.extract, pulse_signal)
                                future_to_face[future] = (face_id, data)
                            else:
                                result_queue.put((face_id, None, data))
                                completed_faces += 1
                            pulse_queue.task_done()

                        completed_futures = []
                        for future in future_to_face:
                            if future.done():
                                completed_futures.append(future)
                        
                        for future in completed_futures:
                            face_id, data = future_to_face.pop(future)
                            processed_any_work = True
                            try:
                                heart_rate = future.result()
                                result_queue.put((face_id, heart_rate, data))
                            except Exception as e:
                                print(f"Error extracting heart rate for face {face_id}: {e}")
                                result_queue.put((face_id, None, data))
                            
                            completed_faces += 1
                        
                        # yield CPU to avoid tight loop
                        if not processed_any_work:
                            time.sleep(0.001)
                            
                    except Exception as e:
                        print(f"Consumer error: {e}")
                        break
                
                # Wait for any remaining futures to complete
                for future in future_to_face:
                    face_id, data = future_to_face[future]
                    try:
                        heart_rate = future.result()
                        result_queue.put((face_id, heart_rate, data))
                    except Exception as e:
                        print(f"Error extracting heart rate for face {face_id}: {e}")
                        result_queue.put((face_id, None, data))
        
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
                face_id, heart_rate, data = result_queue.get_nowait()
                
                if heart_rate is not None:
                    results[face_id] = heart_rate
                    data['heart_rate'] = heart_rate
                    data['last_processed'] = current_time
                
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
        roi_batch = np.array([roi for _, roi, _ in faces_to_process])
        roi_batch = roi_batch.reshape(-1, *roi_batch.shape[2:])  # Flatten batch for processing

        pulse_signal, preprocess_time, inference_time = self.rppg_signal_extractor.extract(roi_batch)
        results = {}
        for i, (face_id, _, data) in enumerate(faces_to_process):
            curr_signal = pulse_signal[180*i:180*(i+1)]
            heart_rate = self.hr_extractor.extract(curr_signal)
            results[face_id] = heart_rate
            data['heart_rate'] = heart_rate
            data['last_processed'] = current_time

        core_time = time.time() - current_time
        return results, core_time

    def process_faces_parallel(self, faces_to_process, current_time):
        results = {}
        max_workers = min(len(faces_to_process), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_face_map = {}
            
            for face_id, roi_data, data in faces_to_process:
                future = executor.submit(
                    self._thread_process_face, 
                    roi_data
                )
                future_face_map[future] = (face_id, data)
            
            for future in as_completed(future_face_map):
                face_id, data = future_face_map[future]
                try:
                    heart_rate = future.result()
                    
                    results[face_id] = heart_rate
                    data['heart_rate'] = heart_rate
                    data['last_processed'] = current_time
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

        for face_id, roi_data, data in faces_to_process:
            try:
                pulse_signal, preprocess_time, inference_time = self.rppg_signal_extractor.extract(roi_data)
                heart_rate = self.hr_extractor.extract(pulse_signal)
            except Exception as e:
                print(f"Error processing face {face_id}: {e}")
                continue
            
            data['heart_rate'] = heart_rate
            data['last_processed'] = current_time
            results[face_id] = heart_rate
        
        core_time = time.time() - current_time
        
        return results, core_time
    
    def get_performance_report(self):
        """Get performance report from the pipeline."""
        return self.performance_monitor.get_performance_summary()
    
    def print_performance_report(self):
        """Print detailed performance report."""
        self.performance_monitor.print_performance_report()
        
        if self.use_incremental:
            print("\nINCREMENTAL PROCESSOR STATUS:")
            print("-" * 30)
            for face_id, info in self.incremental_processor.face_data.items():
                processing_info = self.incremental_processor.get_processing_info(face_id)
                print(f"Face {face_id}:")
                print(f"  Progress: {processing_info.get('progress', 0)*100:.1f}%")
                print(f"  Processed Chunks: {processing_info.get('processed_chunks', 0)}")
                print(f"  Ready for HR: {processing_info.get('ready_for_hr', False)}")
                print(f"  Last HR: {processing_info.get('last_hr', 'N/A')}")
