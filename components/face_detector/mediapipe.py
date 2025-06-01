import cv2
import mediapipe as mp
from components.face_detector.base import FaceDetector

class MediaPipe(FaceDetector):
    def __init__(self, min_confidence: float=0.5):
        self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_confidence)

    def detect(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)

        faces = []
        if results.detections is not None:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                frame_h, frame_w, _ = frame.shape
                x_min, y_min = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h)
                x_max, y_max = int((bbox.xmin + bbox.width) * frame_w), int((bbox.ymin + bbox.height) * frame_h)
                faces.append((x_min, y_min, x_max, y_max))

        return faces
