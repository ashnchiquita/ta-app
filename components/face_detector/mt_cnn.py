from mtcnn import MTCNN
import cv2
from components.face_detector.base import FaceDetector

class MT_CNN(FaceDetector):
    def __init__(self, min_confidence=0.5):
        self.detector = MTCNN(device="CPU:0", stages="face_detection_only")
        self.min_confidence = min_confidence

    def detect(self, frame):
        results = self.detector.detect_faces(frame, box_format="xyxy")
        if not results:
            return []
        faces = [
            tuple(face['box']) for face in results if face is not None and face['confidence'] >= self.min_confidence
        ]
        return faces
