import cv2
import mediapipe as mp
from components.face_detector.base import FaceDetector

class MediaPipe(FaceDetector):
  def __init__(self, min_confidence: float=0.5):
    self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_confidence)

  def detect(self, frame):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.detector.process(rgb_image)

    # [TODO] possibility to parallelize this
    faces = []
    for detection in results.detections:
      bbox = detection.location_data.relative_bounding_box
      h, w, _ = frame.shape
      x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
      faces.append((x, y, width, height))

    return faces
