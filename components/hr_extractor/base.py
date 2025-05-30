from abc import ABC, abstractmethod

class HRExtractor(ABC):
    def __init__(self, fps: float=30.0):
        self.fps = fps
    
    @abstractmethod
    def extract(self, pulse_signal):
        pass
