from abc import ABC, abstractmethod

class FaceTracker(ABC):
  @abstractmethod
  def update(self, rects):
    pass
