from abc import ABC, abstractmethod

class ROISelector(ABC):
    def __init__(self, target_size=(72, 72)):
        self.target_size = target_size
    
    @abstractmethod
    def select(self, frame, face_rect):
        pass
