import numpy as np
import threading
from components.rppg_signal_extractor.deep_learning.hef.base import HEFModel
from hailo_platform import (
    InferVStreams,
)
import time

# Global NPU lock to serialize network activation across all instances
_NPU_LOCK = threading.Lock()

class DeepPhys(HEFModel):
    def __init__(self, model_path: str, fps: float=30.0):
        super().__init__(model_path, fps)
        self.n_frames = 180

    def preprocess(self, roi_data):
        """
        Preprocess the ROI data for DeepPhys model.
        Supports both full window processing and incremental segment processing.
        """
        if roi_data is None or len(roi_data) == 0:
            raise ValueError("ROI data is empty or None")
        
        roi_data = np.array(roi_data, dtype=np.float32)
        data_length = len(roi_data)
        
        # Support incremental processing: process any length, not just multiples of n_frames
        if data_length >= self.n_frames and data_length % self.n_frames == 0:
            # Traditional full window processing
            n_clips = data_length // self.n_frames
            
            if n_clips == 1:
                diffnomz_clip = self.diff_normalize_data(roi_data)
                stdz_clip = self.standardized_data(roi_data)
                preprocessed_clip = np.concatenate([diffnomz_clip, stdz_clip], axis=-1)
                return preprocessed_clip
            
            # Multiple clips processing
            preprocessed_data = []
            for i in range(n_clips):
                start_idx = i * self.n_frames
                end_idx = start_idx + self.n_frames
                clip = roi_data[start_idx:end_idx]
                diffnomz_clip = self.diff_normalize_data(clip)
                stdz_clip = self.standardized_data(clip)
                preprocessed_clip = np.concatenate([diffnomz_clip, stdz_clip], axis=-1)
                preprocessed_data.append(preprocessed_clip)
            
            return np.concatenate(preprocessed_data, axis=0)
        
        else:
            # Incremental segment processing - process any length
            diffnomz_clip = self.diff_normalize_data(roi_data)
            stdz_clip = self.standardized_data(roi_data)
            preprocessed_clip = np.concatenate([diffnomz_clip, stdz_clip], axis=-1)
            return preprocessed_clip
    
    def extract(self, roi_data):
        try:
            # Preprocessing can run in parallel - no NPU access needed
            t0 = time.time()
            preprocessed_data = self.preprocess(roi_data)

            t1 = time.time()
            preprocess_time = t1 - t0
            
            result = self._npu_inference(preprocessed_data)
            t2 = time.time()
            inference_time = t2 - t1

            return result, preprocess_time, inference_time
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract rPPG signal: {str(e)}")
    
    def _npu_inference(self, preprocessed_data):
        """
        NPU inference with global lock to handle hardware limitation.
        Only one network can be active at a time across all instances.
        """
        with _NPU_LOCK:
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    outputs = infer_pipeline.infer({self.input_name: preprocessed_data})

            # Extract the rPPG signal
            rppg_signal = outputs[self.output_name]

            # Flatten
            if len(rppg_signal.shape) > 1:
                rppg_signal = rppg_signal.flatten()
            
            return rppg_signal

    def get_dummy_input(self, n_d=180):
        """
        Generate dummy input data for testing the model.
        This should match the expected input shape of the model.
        """
        dummy_input = np.random.rand(n_d, 72, 72, 6).astype(np.float32)
        return { self.input_name: dummy_input}
