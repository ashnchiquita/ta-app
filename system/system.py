import cv2
import numpy as np
import pandas as pd
import threading
import queue
import time
from components.face_detector.haar_cascade import HaarCascade
from components.face_detector.mediapipe import MediaPipe
from components.face_detector.hailo import HailoFaceDetector
from components.face_detector.degirum import DegirumFaceDetector
from components.face_tracker.centroid import Centroid
from components.roi_selector.fullface import FullFace
from components.rppg_signal_extractor.conventional.pos import POS
from components.hr_extractor.fft import FFT
from system.pipeline import Pipeline
from components.roi_selector.cheeks import Cheeks
from components.roi_selector.forehead import Forehead
from system.metrics import Metrics

class System:
    def __init__(self, 
                    camera_id=None,
                    video_file=None,
                    timestamp_file=None,
                    
                    face_detector=None,
                    face_tracker=None,
                    roi_selector=None,
                    rppg_signal_extractor=None,
                    hr_extractor=None,
                    
                    window_size=300,
                    fps=30,
                    step_size=30):
        
        self.camera_id = camera_id
        self.video_file = video_file
        self.timestamp_file = timestamp_file
        self.fps = fps

        if video_file is not None:
            print(f'Using video file: {video_file}')
            self.video_frames = np.load(video_file)
            print(f'Video frames loaded: {len(self.video_frames)} frames')
            self.total_frames = len(self.video_frames)
            self.current_frame_idx = 0
            self.cap = None

            if timestamp_file is not None:
                print(f'Using timestamps from: {timestamp_file}')
                self.timestamps_df = pd.read_csv(
                    timestamp_file, 
                    names=['frame_number', 'timestamp'],
                    dtype={'frame_number': int, 'timestamp': float},
                    header=0
                )
                self.use_timestamps = True
                self.start_time = None
            else:
                print('No timestamps provided, using constant FPS.')
                self.use_timestamps = False
                self.frame_interval = 1.0 / fps
        else:
            print(f'Using camera ID: {camera_id}')
            self.cap = cv2.VideoCapture(camera_id)
            self.video_frames = None
            self.use_timestamps = False
        
        # self.face_detector = face_detector or HaarCascade()
        self.face_detector = face_detector or HailoFaceDetector()
        self.face_tracker = face_tracker or Centroid()
        self.roi_selector = roi_selector or FullFace()
        self.rppg_signal_extractor = rppg_signal_extractor or POS(fps=fps)
        self.hr_extractor = hr_extractor or FFT(fps=fps)
        
        self.pipeline = Pipeline(
            self.rppg_signal_extractor,
            self.hr_extractor,
            window_size=window_size,
            fps=fps,
            step_size=step_size
        )

        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.running = False
        
        self.heart_rates = {}

        self.processing_metrics = Metrics()
        self.skipped_frames = 0
        
    def start(self):
        self.running = True
        
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.display_thread = threading.Thread(target=self.display_frames)
        
        self.capture_thread.start()
        self.processing_thread.start()
        self.display_thread.start()

        print("System started. Press 'q' to exit.")
            
    def stop(self):
        self.running = False
        
        self.capture_thread.join()
        self.processing_thread.join()
        self.display_thread.join()

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

        print("System stopped.")

        print(f"Skipped frames: {self.skipped_frames}")
        print(self.processing_metrics)
            
    def capture_frames(self):
        if self.video_frames is not None:
            self._capture_from_npy()
        else:
            self._capture_from_camera()

    def _capture_from_camera(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
                    
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # If queue is full, skip
                self.skipped_frames += 1
                print(f"[{self.skipped_frames}] Frame skipped due to full queue.")
                pass
    def _capture_from_npy(self):
        if self.use_timestamps:
            self._capture_with_timestamps()
        else:
            self._capture_with_constant_fps()

    def _capture_with_constant_fps(self):
        frame_start_time = time.time()
        
        while self.running and self.current_frame_idx < self.total_frames:
            frame = self.video_frames[self.current_frame_idx]
            
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                self.skipped_frames += 1
                print(f"[{self.skipped_frames}] Frame skipped due to full queue.")
            
            self.current_frame_idx += 1
            
            expected_time = frame_start_time + (self.current_frame_idx * self.frame_interval)
            current_time = time.time()
            
            if expected_time > current_time:
                time.sleep(expected_time - current_time)
        
        if self.current_frame_idx >= self.total_frames:
            print("Reached end of video file.")
            self.running = False
    
    def _capture_with_timestamps(self):
        if self.start_time is None:
            self.start_time = time.time()
            first_timestamp = self.timestamps_df.iloc[0]['timestamp']
            self.timestamp_offset = self.start_time - first_timestamp
        
        while self.running and self.current_frame_idx < self.total_frames:
            if self.current_frame_idx >= len(self.timestamps_df):
                print("Reached end of timestamp file.")
                self.running = False
                break
                
            frame = self.video_frames[self.current_frame_idx]
            
            expected_timestamp = self.timestamps_df.iloc[self.current_frame_idx]['timestamp'] + self.timestamp_offset
            current_time = time.time()
            
            # Wait until it's time to display this frame
            if expected_timestamp > current_time:
                time.sleep(expected_timestamp - current_time)
            
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                self.skipped_frames += 1
                print(f"[{self.skipped_frames}] Frame skipped due to full queue.")
            
            self.current_frame_idx += 1
        
        if self.current_frame_idx >= self.total_frames:
            print("Reached end of video file.")
            self.running = False
                    
    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                t0 = time.time()
                
                face_rects = self.face_detector.detect(frame)
                t1 = time.time()
                self.processing_metrics.processing_time['face_detection'] += t1 - t0
                
                centroid_map = self.face_tracker.update(face_rects)
                t2 = time.time()
                self.processing_metrics.processing_time['face_tracking'] += t2 - t1
                
                for face_rect in face_rects:
                    t3 = time.time()
                    x, y, x_end, y_end = face_rect
                    c_x = (x + x_end) // 2
                    c_y = (y + y_end) // 2

                    object_id = centroid_map[(c_x, c_y)]
                    if object_id is None:
                        raise ValueError(f"Centroid ({c_x}, {c_y}) not found in centroid map.")
                    
                    t4 = time.time()

                    self.processing_metrics.processing_time['face_tracking'] += t4 - t3
                    
                    roi = self.roi_selector.select(frame, face_rect)
                    t5 = time.time()
                    self.processing_metrics.processing_time['roi_selection'] += t5 - t4
                    
                    self.pipeline.add_face_data(object_id, roi, t0)
            
                new_results, signal_extraction_time, hr_extraction_time = self.pipeline.process_faces()

                self.processing_metrics.processing_time['signal_extraction'] += signal_extraction_time
                self.processing_metrics.processing_time['hr_extraction'] += hr_extraction_time
                
                for face_id, hr in new_results.items():
                    self.heart_rates[face_id] = hr
                
                self.result_queue.put((frame, face_rects, centroid_map, self.heart_rates.copy()))

                self.processing_metrics.processing_count += 1
                self.processing_metrics.total_processing_time += time.time() - t0
 
            except queue.Empty:
                continue

    def display_frames(self):
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        while self.running:
            try:
                frame, rects, centroid_map, heart_rates = self.result_queue.get(timeout=1)
                frame_count += 1
                
                current_time = time.time()
                elapsed_time = current_time - prev_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    prev_time = current_time
        
                for (x, y, x_end, y_end) in rects:
                    c_x = (x + x_end) // 2
                    c_y = (y + y_end) // 2
                    object_id = centroid_map[(c_x, c_y)]
                    
                    cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)
                                
                    text = f"ID: {object_id}"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if object_id in heart_rates:
                            hr_text = f"HR: {heart_rates[object_id]:.1f} BPM"
                            cv2.putText(frame, hr_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Multi-Person rPPG [Press 'q' to exit]", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                            
            except queue.Empty:
                continue
