from abc import ABC, abstractmethod

class ROISelector(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def select(self, frame, face_rect):
        pass
