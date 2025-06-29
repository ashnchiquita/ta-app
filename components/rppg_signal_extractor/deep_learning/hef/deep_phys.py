import numpy as np
from components.rppg_signal_extractor.deep_learning.hef.base import HEFModel
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InputVStreamParams,
    InferVStreams,
    OutputVStreamParams,
    VDevice,
)

class DeepPhys(HEFModel):
    def __init__(self, model_path: str, fps: float=30.0):
        super().__init__(model_path, fps)
        self.n_frames = 180

    def preprocess(self, roi_data):
        """
        Preprocess the ROI data for DeepPhys model.
        This method should be implemented based on the model's requirements.
        """
        if roi_data is None or len(roi_data) == 0:
            raise ValueError("ROI data is empty or None")
        if len(roi_data) % self.n_frames != 0:
            raise ValueError(f"ROI data must have {self.n_frames} frames")

        # Convert to numpy array if not already
        roi_data = np.array(roi_data, dtype=np.float32)
        
        # Standardize each frame
        preprocessed_data = []

        n_clips = len(roi_data) // self.n_frames

        for i in range(n_clips):
            start_idx = i * self.n_frames
            end_idx = start_idx + self.n_frames
            clip = roi_data[start_idx:end_idx]
            diffnomz_clip = self.diff_normalize_data(clip)
            stdz_clip = self.standardized_data(clip)
            preprocessed_clip = np.concatenate([diffnomz_clip, stdz_clip], axis=-1)

            preprocessed_data.append(preprocessed_clip)

        # Concatenate all clips
        preprocessed_data = np.concatenate(preprocessed_data, axis=0)

        return preprocessed_data
    
    def extract(self, roi_data):
        try:
            print("Starting rPPG signal extraction using DeepPhys...")
            preprocessed_data = self.preprocess(roi_data)

            print(f"Preprocessed data shape: {preprocessed_data.shape}")
            
            # outputs = self.infer_pipeline.infer({self.input_name: preprocessed_data})
            
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    outputs = infer_pipeline.infer({self.input_name: preprocessed_data})

            # Extract the rPPG signal
            rppg_signal = outputs[self.output_name]

            # Flatten
            if len(rppg_signal.shape) > 1:
                rppg_signal = rppg_signal.flatten()
            
            return rppg_signal
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract rPPG signal: {str(e)}")

    def get_dummy_input(self):
        """
        Generate dummy input data for testing the model.
        This should match the expected input shape of the model.
        """
        dummy_input = np.random.rand(180, 72, 72, 6).astype(np.float32)
        return { self.input_name: dummy_input}
