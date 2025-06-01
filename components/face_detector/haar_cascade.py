import cv2
from components.face_detector.base import FaceDetector

class HaarCascade(FaceDetector):
    def __init__(self, scale_factor: float=1.1, min_neighbors: int=7):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors)

        return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]
