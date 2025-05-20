import cv2
from components.roi_selector.base import ROISelector

class FullFace(ROISelector):
  def select(self, frame, face_rect):
    x, y, x_end, y_end = face_rect
    roi = frame[y:y_end, x:x_end]
    
    if roi.size == 0:
      return None
    
    return cv2.resize(roi, self.target_size)
