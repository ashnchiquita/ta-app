import onnxruntime as ort
from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor
import os

class ONNXModel(DeepLearningRPPGSignalExtractor):
    def __init__(self, model_path: str, fps: float=30.0):
        if not model_path.endswith('.onnx'):
            raise ValueError("Model path must point to an ONNX file.")
        
        super().__init__(model_path, fps)

    def _load_model(self):
        """Load the ONNX model."""
        try:
            self.model = ort.InferenceSession(self.model_path)
            print(f"ONNX model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {self.model_path}: {str(e)}")

        # load with dummy data to pass lazy loading
        dummy_input = self.get_dummy_input()
        self.model.run(None, dummy_input)
    