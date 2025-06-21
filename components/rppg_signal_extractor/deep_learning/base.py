from components.rppg_signal_extractor.base import RPPGSignalExtractor
import numpy as np
import os

class DeepLearningRPPGSignalExtractor(RPPGSignalExtractor):
    def __init__(self, model_path: str, fps: float = 30.0):
        super().__init__(fps)
        self.model_path = model_path
        print(f"Initializing Deep Learning RPPG Signal Extractor with model: {model_path}")
        self.model_name = os.path.basename(model_path).split('.')[0]

        self._load_model()
        
    def _load_model(self):
        """Load the deep learning model."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def preprocess(self, roi_data):
        """
        Preprocess the ROI data for the ONNX model.
        This method should be implemented based on the model's requirements.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def extract(self, roi_data):
        """
        Extract the rPPG signal from the ROI data using the ONNX model.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @staticmethod
    def get_dummy_input():
        raise NotImplementedError("This method should be implemented in subclasses to return dummy data for testing.")

    @staticmethod
    def standardized_data(single_clip):
        """Z-score standardization for video data."""
        single_clip = single_clip - np.mean(single_clip)
        single_clip = single_clip / np.std(single_clip)
        single_clip[np.isnan(single_clip)] = 0
        return single_clip

    @staticmethod
    def diff_normalize_data(single_clip):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = single_clip.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (single_clip[j + 1, :, :, :] - single_clip[j, :, :, :]) / (
                    single_clip[j + 1, :, :, :] + single_clip[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data
