import cv2
import numpy as np
import os
import pandas as pd
import threading
import queue
import time
from components.face_detector.haar_cascade import HaarCascade
from components.face_detector.mediapipe import MediaPipe
from components.face_detector.mt_cnn import MT_CNN
# from components.face_detector.degirum import DegirumFaceDetector
from components.face_tracker.centroid import Centroid
from components.roi_selector.fullface import FullFace
from components.roi_selector.fullface_square import FullFaceSquare
from components.rppg_signal_extractor.conventional.pos import POS
from components.rppg_signal_extractor.conventional.chrom import CHROM
from components.rppg_signal_extractor.conventional.ica import ICA
from components.hr_extractor.fft import FFT
from system.pipeline import Pipeline
from system.metrics import Metrics
import system.colors as colors
from components.rppg_signal_extractor.deep_learning.onnx.efficient_phys import EfficientPhys
from components.rppg_signal_extractor.deep_learning.onnx.deep_phys import DeepPhys as ONNXDeepPhys
from components.rppg_signal_extractor.deep_learning.hef.deep_phys import DeepPhys as HEFDeepPhys
from components.rppg_signal_extractor.deep_learning.onnx.tscan import TSCAN
from components.face_detector.hef.retina_face.retina_face import RetinaFace
from components.face_detector.hef.scrfd.scrfd import SCRFD
from constants import ONNX_DIR, HEF_DIR
from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor


class System:
    def __init__(self, 
                    camera_id=None,
                    video_file=None,
                    
                    face_detector=None,
                    face_tracker=None,
                    roi_selector=None,
                    rppg_signal_extractor=None,
                    hr_extractor=None,
                    
                    window_size=180,
                    fps=30,
                    step_size=180,
                    use_incremental=None,

                    log_dir="output",
                ):
        
        self.camera_id = camera_id
        self.video_file = video_file
        self.fps = fps

        # Logging
        self.fps_log = []
        self.heart_rates_log = []

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if video_file is not None:
            print(f'Using video file: {video_file}')
            
            # Check file extension to determine how to load the video
            if video_file.lower().endswith('.npy'):
                print('Loading NPY file into memory...')
                self.video_frames = np.load(video_file)
                # self.video_frames = self.video_frames[:31] # Limit to first 180 frames for test [Todo: remove this line]
                self.video_frames = np.flip(self.video_frames, axis=3)  # Convert BGR to RGB by flipping the last axis
                
                print(f'Video frames loaded: {len(self.video_frames)} frames')
                self.total_frames = len(self.video_frames)
                self.current_frame_idx = 0
                self.cap = None
                self.is_npy_file = True
            else:
                print('Loading video file with OpenCV...')
                self.cap = cv2.VideoCapture(video_file)
                if not self.cap.isOpened():
                    raise ValueError(f"Could not open video file: {video_file}")
                
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                print(f'Video file loaded: {self.total_frames} frames at {video_fps} FPS')
                
                # Use video file's FPS if not explicitly set
                if fps == 30:  # Default value
                    self.fps = video_fps
                    print(f'Using video file FPS: {self.fps}')
                
                self.current_frame_idx = 0
                self.video_frames = None
                self.is_npy_file = False

            self.frame_interval = 1.0 / self.fps
        else:
            print(f'Using camera ID: {camera_id}')
            self.cap = cv2.VideoCapture(camera_id)
            self.video_frames = None

        # self.face_detector = face_detector or HaarCascade()
        # self.face_detector = face_detector or MediaPipe()
        self.face_detector = face_detector or SCRFD(variant='2.5g')
        # self.face_detector = face_detector or MT_CNN()
        # self.face_detector = face_detector or DegirumFaceDetector()
        self.face_tracker = face_tracker or Centroid()
        # self.roi_selector = roi_selector or FullFace()
        self.roi_selector = roi_selector or FullFaceSquare(target_size=(72,72), larger_box_coef=1.5)
        # self.rppg_signal_extractor = rppg_signal_extractor or POS(fps=fps)
        self.rppg_signal_extractor = rppg_signal_extractor or HEFDeepPhys(fps=fps, model_path=os.path.join(HEF_DIR, "PURE_DeepPhys_quantized_20250706-000109.hef"))
        # self.rppg_signal_extractor = rppg_signal_extractor or ONNXDeepPhys(fps=fps, model_path=os.path.join(ONNX_DIR, "PURE_DeepPhys.onnx"))
        # self.rppg_signal_extractor = rppg_signal_extractor or EfficientPhys(fps=fps, model_path=os.path.join(ONNX_DIR, "PURE_EfficientPhys.onnx"))
        # self.rppg_signal_extractor = rppg_signal_extractor or TSCAN(fps=fps, model_path=os.path.join(ONNX_DIR, "PURE_TSCAN.onnx"))

        diff_flag = isinstance(self.rppg_signal_extractor, DeepLearningRPPGSignalExtractor)
        print(f"Using diff_flag={diff_flag} for rPPG signal extraction.")
        self.hr_extractor = hr_extractor or FFT(fps=fps, diff_flag=diff_flag, use_bandpass=True, use_detrend=True)

        self.pipeline = Pipeline(
            self.rppg_signal_extractor,
            self.hr_extractor,
            window_size=window_size,
            fps=fps,
            step_size=step_size,
            use_incremental=use_incremental
        )

        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.running = False
        
        self.heart_rates = {}

        self.processing_metrics = Metrics()
        self.skipped_frames = 0
        
        # Debug info
        print(f"Initialized System with:")
        print(f"  Face Detector: {type(self.face_detector).__name__}")
        print(f"  rPPG Extractor: {type(self.rppg_signal_extractor).__name__}")
        print(f"  Window Size: {window_size}")
        print(f"  Use Incremental: {use_incremental}")
            
    def start(self):
        self.running = True
        
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.display_thread = threading.Thread(target=self.display_frames)
        
        self.capture_thread.start()
        self.processing_thread.start()
        self.display_thread.start()

        print("System started. Press 'q' to exit.")

        self.processing_metrics.start_time = time.time()
            
    def stop(self):
        print("Stopping system...")
        self.running = False
        self.processing_metrics.end_time = time.time()
        
        # Give threads a moment to finish their current operations
        time.sleep(0.1)
        
        # Join threads with timeout to prevent indefinite blocking
        threads_info = [
            (self.capture_thread, "capture"),
            (self.processing_thread, "processing"), 
            (self.display_thread, "display")
        ]
        
        for thread, name in threads_info:
            if thread.is_alive():
                print(f"Waiting for {name} thread to finish...")
                thread.join(timeout=5.0)  # 5 second timeout
                if thread.is_alive():
                    print(f"Warning: {name} thread did not finish within timeout")
                else:
                    print(f"{name} thread finished successfully")

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

        # Cleanup all components
        self._cleanup_components()

        print("System stopped.")

        print(f"Skipped frames: {self.skipped_frames}")
        self.processing_metrics.skipped_frames = self.skipped_frames
        print(self.processing_metrics)

        self.store_all_logs()
            
    def capture_frames(self):
        if self.video_frames is not None:
            self._capture_from_npy()
        elif self.video_file is not None:
            self._capture_from_video_file()
        else:
            self._capture_from_camera()

    def _capture_from_camera(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
                    
            try:
                # convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # If queue is full, skip and yield CPU briefly
                self.skipped_frames += 1
                print(f"[{self.skipped_frames}] Frame skipped due to full queue.")
                time.sleep(0.001)  # Brief yield to prevent tight loop
            
    def _capture_from_npy(self):
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

    def _capture_from_video_file(self):
        frame_start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Reached end of video file.")
                self.running = False
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
        
        print(f"AVI capture finished. Processed {self.current_frame_idx} frames.")
                    
    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Check for cleanup signal
                if frame is None:
                    break
                    
                t0 = time.time()

                # Face Detection
                face_rects = self.face_detector.detect(frame)
                t1 = time.time()
                self.processing_metrics.processing_time['face_detection'] += t1 - t0

                # Face Tracking
                objects = self.face_tracker.update(face_rects)
                t2 = time.time()
                self.processing_metrics.processing_time['face_tracking'] += t2 - t1

                # ROI Selection
                roi_coords = {}  # Store ROI coordinates for display
                for object_id, data in objects.items():
                    face_rect = data['rect']

                    t3 = time.time()
                    roi, roi_coord = self.roi_selector.select(frame, face_rect)
                    t4 = time.time()
                    self.processing_metrics.processing_time['roi_selection'] += t4 - t3
                    
                    if roi is not None:
                        roi_coords[object_id] = roi_coord
                        self.pipeline.add_face_data(object_id, roi, t0)

                new_results, core_time = self.pipeline.process_faces()
                self.processing_metrics.processing_time['core_time'] += core_time

                currtime = time.time()

                for face_id, hr in new_results.items():
                    self.heart_rates_log.append((currtime, face_id, hr))
                    self.heart_rates[face_id] = hr
                
                self.result_queue.put((frame, objects, self.heart_rates.copy(), roi_coords))

                self.processing_metrics.processing_count += 1
                self.processing_metrics.total_processing_time += time.time() - t0
 
            except queue.Empty:
                # Yield CPU when no frames available
                time.sleep(0.001)
                continue
            except Exception as e:
                if self.running:  # Only print errors if system is still supposed to be running
                    print(f"Error in process_frames: {e}")
                break
        
        print("Processing thread finished.")

    def display_frames(self):
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        while self.running:
            try:
                frame, objects, heart_rates, roi_coords = self.result_queue.get(timeout=1)
                
                # Check for cleanup signal
                if frame is None:
                    break
                    
                frame_count += 1

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV display
                
                current_time = time.time()
                elapsed_time = current_time - prev_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    prev_time = current_time
                    self.fps_log.append((time.time(), fps))

                for object_id in objects.keys():
                    if object_id not in roi_coords:
                        continue
                    roi_x, roi_y, roi_x_end, roi_y_end = roi_coords[object_id]

                    color = colors.get_annotation_color(object_id)
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x_end, roi_y_end), color, 2)
                                
                    text = f"{object_id} | "
                    if object_id in heart_rates:
                        text += f"{heart_rates[object_id]:.1f}"
                    else:
                        text += "-"
                        text += "-"
                    cv2.putText(frame, text, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                    
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Multi-Person rPPG [Press 'q' to exit]", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                            
            except queue.Empty:
                # Yield CPU when no display data available
                time.sleep(0.001)
                continue
            except Exception as e:
                if self.running:  # Only print errors if system is still supposed to be running
                    print(f"Error in display_frames: {e}")
                break
        
        print("Display thread finished.")

    def _cleanup_components(self):
        """Cleanup all components and release resources."""
        try:
            # Clear queues first to unblock any waiting threads
            self._clear_queues()
            
            # Cleanup rPPG signal extractor
            if hasattr(self.rppg_signal_extractor, 'cleanup'):
                self.rppg_signal_extractor.cleanup()
            
            # Cleanup face detector
            if hasattr(self.face_detector, 'cleanup'):
                self.face_detector.cleanup()
            
            # Cleanup pipeline
            if hasattr(self.pipeline, 'cleanup'):
                self.pipeline.cleanup()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def _clear_queues(self):
        """Clear all queues to unblock waiting threads."""
        try:
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
            
            # Clear result queue
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    break
                    
            # Add dummy items to unblock any threads waiting on get()
            try:
                self.frame_queue.put(None, block=False)
            except:
                pass
            try:
                self.result_queue.put((None, None, None, None), block=False)
            except:
                pass
                
        except Exception as e:
            print(f"Warning: Error clearing queues: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'running') and self.running:
            self.stop()

    def store_fps(self):
        """Store FPS timestamps and values to a CSV file."""
        
        df = pd.DataFrame(self.fps_log, columns=['timestamp', 'fps'])

        output_file = os.path.join(self.log_dir, "fps.csv")

        df.to_csv(output_file, index=False)
        print(f"FPS data stored to {output_file}")

    def store_heart_rate(self):
        """Store heart rate timestamps and values to a CSV file."""
        if not self.heart_rates_log:
            print("No heart rate data to store.")
            return

        df = pd.DataFrame(self.heart_rates_log, columns=['timestamp', 'face_id', 'heart_rate'])

        output_file = os.path.join(self.log_dir, "heart_rate.csv")
        df.to_csv(output_file, index=False)
        print(f"Heart rate data stored to {output_file}")

    def store_processing_metrics(self):
        metrics_data = self.processing_metrics.to_tuple()
        output_file = os.path.join(self.log_dir, "processing_metrics.csv")
        
        df = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
        df.to_csv(output_file, index=False)
        print(f"Processing metrics stored to {output_file}")

    def store_all_logs(self):
        self.store_fps()
        self.store_heart_rate()
        self.store_processing_metrics()
        
