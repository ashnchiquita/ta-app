from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor

class HEFModel(DeepLearningRPPGSignalExtractor):
    def __init__(self, model_path: str, fps: float=30.0):
        if not model_path.endswith('.hef'):
            raise ValueError("Model path must point to an ONNX file.")
        
        super().__init__(fps)
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1].split('.')[0]  # Extract model name from path
        self._load_model()

    def _load_model(self):
        """Load the HEF model."""
        raise NotImplementedError("HEF model loading is not implemented yet.")
    