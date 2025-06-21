from collections import deque
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor

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
            
    def add_face_data(self, face_id, roi, timestamp):
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
        if isinstance(self.rppg_signal_extractor, DeepLearningRPPGSignalExtractor):
            # If using multiple models, process in parallel
            return self.process_faces_batch()

        return self.process_faces_parallel()
        
        # return self.process_faces_sequential()

    def process_faces_batch(self):
        current_time = time.time()
        results = {}
        signal_extraction_time = 0
        hr_extraction_time = 0
        
        # filter to avoid unnecessary thread creation
        faces_to_process = []
        for face_id, data in self.face_data.items():
            if (len(data['roi_data']) >= self.window_size and 
                current_time - data['last_processed'] >= (self.step_size / self.fps)):
                roi_data = np.array(list(data['roi_data']))
                faces_to_process.append((face_id, roi_data, data))
        
        if not faces_to_process:
            return results, signal_extraction_time, hr_extraction_time

        # merge all ROIs into a single batch
        roi_batch = np.array([roi for _, roi, _ in faces_to_process])
        roi_batch = roi_batch.reshape(-1, *roi_batch.shape[2:])  # Flatten batch for processing

        print(f"Processing {len(faces_to_process)} faces in batch with shape: {roi_batch.shape}")

        t1 = time.time()
        pulse_signal = self.rppg_signal_extractor.extract(roi_batch)
        t2 = time.time()
        signal_extraction_time += t2 - t1

        for i, (face_id, _, data) in enumerate(faces_to_process):
            curr_signal = pulse_signal[180*i:180*(i+1)]
            heart_rate = self.hr_extractor.extract(curr_signal)
            results[face_id] = heart_rate
            data['heart_rate'] = heart_rate
            data['last_processed'] = current_time

        t3 = time.time()
        hr_extraction_time += t3 - t2

        return results, signal_extraction_time, hr_extraction_time

    def process_faces_parallel(self):
        current_time = time.time()
        results = {}
        signal_extraction_time = 0
        hr_extraction_time = 0
        
        # filter to avoid unnecessary thread creation
        faces_to_process = []
        for face_id, data in self.face_data.items():
            if (len(data['roi_data']) >= self.window_size and 
                current_time - data['last_processed'] >= (self.step_size / self.fps)):
                roi_data = np.array(list(data['roi_data']))
                faces_to_process.append((face_id, roi_data, data))
        
        if not faces_to_process:
            return results, signal_extraction_time, hr_extraction_time

        max_workers = min(len(faces_to_process), 10)
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
                    heart_rate, sig_time, hr_time = future.result()
                    
                    results[face_id] = heart_rate
                    signal_extraction_time += sig_time
                    hr_extraction_time += hr_time
                    
                    data['heart_rate'] = heart_rate
                    data['last_processed'] = current_time
                except Exception as e:
                    print(f"Error processing face {face_id}: {e}")
                    
        return results, signal_extraction_time, hr_extraction_time

    def _thread_process_face(self, roi_data):
        t1 = time.time()
        pulse_signal = self.rppg_signal_extractor.extract(roi_data)
        t2 = time.time()
        signal_time = t2 - t1
        
        heart_rate = self.hr_extractor.extract(pulse_signal)
        t3 = time.time()
        hr_time = t3 - t2
        
        return heart_rate, signal_time, hr_time

    def process_faces_sequential(self):
        current_time = time.time()
        results = {}

        signal_extraction_time = 0
        hr_extraction_time = 0
        
        for face_id, data in self.face_data.items():
            if (len(data['roi_data']) >= self.window_size and 
                current_time - data['last_processed'] >= (self.step_size / self.fps)):
                
                t1 = time.time()
                roi_data = np.array(list(data['roi_data']))
                pulse_signal = self.rppg_signal_extractor.extract(roi_data)
                t2 = time.time()
                signal_extraction_time += (t2 - t1)
                
                heart_rate = self.hr_extractor.extract(pulse_signal)
                t3 = time.time()
                hr_extraction_time += (t3 - t2)
                
                data['heart_rate'] = heart_rate
                data['last_processed'] = current_time
                results[face_id] = heart_rate
        
        return results, signal_extraction_time, hr_extraction_time
