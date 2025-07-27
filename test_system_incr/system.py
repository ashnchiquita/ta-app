import cv2
import numpy as np
import os
import pandas as pd
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
from test_system.pipeline import Pipeline
from test_system.metrics import Metrics
from components.rppg_signal_extractor.deep_learning.onnx.efficient_phys import EfficientPhys
from components.rppg_signal_extractor.deep_learning.onnx.deep_phys import DeepPhys as ONNXDeepPhys
from components.rppg_signal_extractor.deep_learning.hef.deep_phys import DeepPhys as HEFDeepPhys
from components.rppg_signal_extractor.deep_learning.onnx.tscan import TSCAN
from components.face_detector.hef.scrfd.scrfd import SCRFD
from constants import ONNX_DIR, HEF_DIR
from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor
from components.manager.hailo_target_manager import HailoTargetManager

class System:
    def __init__(self, 
                    video_file=None,
                    timestamp_file=None,
                    
                    face_detector=None,
                    face_tracker=None,
                    roi_selector=None,
                    rppg_signal_extractor=None,
                    hr_extractor=None,
                    
                    window_size=180,
                    fps=30,
                    step_size=180):
        
        self.video_file = video_file
        self.timestamp_file = timestamp_file
        self.fps = fps

        if video_file is None:
            raise ValueError("video_file must be provided")
            
        print(f'Using video file: {video_file}')
        self.video_frames = np.load(video_file)
        self.video_frames = np.flip(self.video_frames, axis=3)  # Convert BGR to RGB by flipping the last axis
        
        print(f'Video frames loaded: {len(self.video_frames)} frames')
        self.total_frames = len(self.video_frames)

        # self.face_detector = face_detector or HaarCascade()
        # self.face_detector = face_detector or MediaPipe()
        self.face_detector = face_detector or SCRFD(variant='500m')
        # self.face_detector = face_detector or MT_CNN()
        # self.face_detector = face_detector or DegirumFaceDetector()
        self.face_tracker = face_tracker or Centroid()
        # self.roi_selector = roi_selector or FullFace()
        self.roi_selector = roi_selector or FullFaceSquare(target_size=(72,72), larger_box_coef=1.5)
        # self.rppg_signal_extractor = rppg_signal_extractor or POS(fps=fps)
        self.rppg_signal_extractor = rppg_signal_extractor or HEFDeepPhys(fps=fps, model_path=os.path.join(HEF_DIR, "PURE_DeepPhys_quantized_20250629-180552.hef"))
        # self.rppg_signal_extractor = rppg_signal_extractor or ONNXDeepPhys(fps=fps, model_path=os.path.join(ONNX_DIR, "PURE_DeepPhys.onnx"))
        # self.rppg_signal_extractor = rppg_signal_extractor or EfficientPhys(fps=fps, model_path=os.path.join(ONNX_DIR, "PURE_EfficientPhys.onnx"))
        # self.rppg_signal_extractor = rppg_signal_extractor or TSCAN(fps=fps, model_path=os.path.join(ONNX_DIR, "PURE_TSCAN.onnx"))

        diff_flag = isinstance(self.rppg_signal_extractor, DeepLearningRPPGSignalExtractor)
        self.hr_extractor = hr_extractor or FFT(fps=fps, diff_flag=diff_flag, use_bandpass=True)

        self.pipeline = Pipeline(
            self.rppg_signal_extractor,
            self.hr_extractor,
            window_size=window_size,
            fps=fps,
            step_size=step_size
        )

        self.heart_rates = {}
        self.heart_rates_frame_idx = {}
        self.processing_metrics = Metrics()
        
    def run(self):
        """Run the system sequentially without threading"""
        print("Starting sequential processing...")
        
        start_time = time.time()
        
        for frame_idx in range(self.total_frames):
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{self.total_frames}")
            
            frame = self.video_frames[frame_idx]
            
            # Get timestamp for this frame
            timestamp = frame_idx / self.fps  # Use frame index as time reference
            
            self._process_single_frame(frame, timestamp, frame_idx)
            
            # Memory management - clear cache periodically
            if frame_idx % 1000 == 0 and frame_idx > 0:
                import gc
                gc.collect()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Processing completed in {total_time:.2f} seconds")
        print(f"Average processing FPS: {self.total_frames / total_time:.2f}")
        self.processing_metrics.fill_average()
        print(self.processing_metrics)

        target_manager = HailoTargetManager()
        target_manager._cleanup()
        return self.heart_rates
    
    def _process_single_frame(self, frame, timestamp, frame_idx):
        """Process a single frame through the complete pipeline"""
        t0 = time.time()

        # Face Detection
        face_rects = self.face_detector.detect(frame)
        t1 = time.time()
        self.processing_metrics.processing_time['face_detection'] += t1 - t0

        # Face Tracking
        objects = self.face_tracker.update(face_rects)
        t2 = time.time()
        self.processing_metrics.processing_time['face_tracking'] += t2 - t1

        # ROI Selection for each tracked face
        for object_id, data in objects.items():
            face_rect = data['rect']

            t3 = time.time()
            roi, _ = self.roi_selector.select(frame, face_rect)
            t4 = time.time()
            self.processing_metrics.processing_time['roi_selection'] += t4 - t3
            
            if roi is not None:
                # Add ROI data to pipeline for windowing
                self.pipeline.add_face_data(object_id, roi, timestamp)

        # CORE: rPPG Signal + HR Extraction (with windowing and step system)
        new_results, signal_extraction_time, hr_extraction_time, signal_preprocess_time, signal_inference_time, total_processed_faces = self.pipeline.process_faces()

        self.processing_metrics.processing_time['signal_extraction'] += signal_extraction_time
        self.processing_metrics.processing_time['hr_extraction'] += hr_extraction_time
        self.processing_metrics.processing_time['signal_extraction_preprocess'] += signal_preprocess_time
        self.processing_metrics.processing_time['signal_extraction_inference'] += signal_inference_time
        self.processing_metrics.total_processed_faces += total_processed_faces

        # Update heart rates
        for face_id, hr in new_results.items():
            if face_id not in self.heart_rates:
                self.heart_rates[face_id] = [hr]
                self.heart_rates_frame_idx[face_id] = [frame_idx]
            else:
                self.heart_rates[face_id].append(hr)
                self.heart_rates_frame_idx[face_id].append(frame_idx)

        self.processing_metrics.processing_count += 1
        self.processing_metrics.total_processing_time += time.time() - t0
    
    def get_pipeline_stats(self):
        """Get statistics about the pipeline state"""
        stats = {}
        for face_id, data in self.pipeline.face_data.items():
            stats[face_id] = {
                'roi_count': len(data['roi_data']),
                'timestamp_count': len(data['timestamps']),
                'last_heart_rate': data['heart_rate'],
                'last_processed': data['last_processed'],
                'ready_for_processing': len(data['roi_data']) >= self.pipeline.window_size
            }
        return stats
