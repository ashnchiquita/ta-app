from components.rppg_signal_extractor.base import RPPGSignalExtractor

class DeepPhys(RPPGSignalExtractor):
    def __init__(self, model_path: str, fps: float=30.0):
        super().__init__(fps)
        self.model_path = model_path

        # TODO: Load the model
        self.model = None    

    # TODO: Implement the extraction method
    def extract(self, roi_data):
        raise NotImplementedError("DeepPhys extraction is not implemented yet.")
