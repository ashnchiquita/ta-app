from abc import ABC, abstractmethod

class FaceDetector(ABC):
  @abstractmethod
  def detect(self, frame):
    pass
