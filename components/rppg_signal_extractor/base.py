from abc import ABC, abstractmethod

class RPPGSignalExtractor(ABC):
    def __init__(self, fps: float=30.0):
        self.fps = fps
        
    @abstractmethod
    def extract(self, roi_data):
        pass
