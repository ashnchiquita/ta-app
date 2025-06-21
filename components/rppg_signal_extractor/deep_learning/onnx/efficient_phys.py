import numpy as np
from components.rppg_signal_extractor.deep_learning.onnx.base import ONNXModel

class EfficientPhys(ONNXModel):
    def __init__(self, model_path: str, fps: float=30.0):
        super().__init__(model_path, fps)
        self.n_frames = 180

    def preprocess(self, roi_data):
        """
        Preprocess the ROI data for EfficientPhys model.
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
            print(f"Processing clip {i+1}/{n_clips} with shape: {clip.shape}")
            preprocessed_clip = self.standardized_data(clip)
            preprocessed_clip = np.transpose(preprocessed_clip, (0, 3, 1, 2))  # Change to (D, C, H, W) format

            print(f"Preprocessed clip shape: {preprocessed_clip.shape}")
            preprocessed_data.append(preprocessed_clip)

        # Concatenate all clips
        preprocessed_data = np.concatenate(preprocessed_data, axis=0)

        print(f"Total preprocessed data shape: {preprocessed_data.shape}")

        last_frame = np.expand_dims(preprocessed_data[-1, :, :, :], 0)
        last_frame = np.repeat(last_frame, 1, axis=0)
        preprocessed_data = np.concatenate((preprocessed_data, last_frame), axis=0)

        print(f"Final preprocessed data shape after concatenation: {preprocessed_data.shape}")
        return preprocessed_data
    
    def extract(self, roi_data):
        try:
            print("Starting rPPG signal extraction using EfficientPhys...")
            preprocessed_data = self.preprocess(roi_data)

            print(f"Preprocessed data shape: {preprocessed_data.shape}")
            
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: preprocessed_data})
            
            # Extract the rPPG signal (assuming first output is the signal)
            rppg_signal = outputs[0]
            print(f"Extracted rPPG signal shape: {rppg_signal.shape}")

            # Flatten
            if len(rppg_signal.shape) > 1:
                rppg_signal = rppg_signal.flatten()
            
            return rppg_signal
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract rPPG signal: {str(e)}")

    @staticmethod
    def get_dummy_input():
        """
        Generate dummy input data for testing the model.
        This should match the expected input shape of the model.
        """
        dummy_input = np.random.rand(181, 3, 72, 72).astype(np.float32)
        return { "input": dummy_input }
