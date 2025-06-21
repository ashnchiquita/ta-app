from components.face_detector.base import FaceDetector
import degirum as dg
import numpy as np
from constants import DEGIRUM_ZOO_DIR

class HailoFaceDetector(FaceDetector):
    def __init__(self, model_name="yolov8n_relu6_face--640x640_quant_hailort_hailo8l_1", 
                    confidence_threshold=0.5):
        self.face_model = dg.load_model(
            model_name=model_name,
            inference_host_address="@local",
            zoo_url=DEGIRUM_ZOO_DIR,
            token="",
            overlay_color=(0, 255, 0)
        )

        self.face_model.output_confidence_threshold = confidence_threshold

        # warm up the model because it use lazy initialization
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.face_model(dummy_frame)    
            
    def detect(self, frame):
        try:
            detection_result = self.face_model(frame)
            
            faces = []
            if detection_result.results:
                for detection in detection_result.results:
                    bbox = detection['bbox']
                    # no need to check for confidence here, as we set the threshold in the model
                    x_min, y_min, x_max, y_max = bbox
                    faces.append((int(x_min), int(y_min), int(x_max), int(y_max)))
                            
            return faces
                
        except Exception as e:
            print(f"Error in Hailo face detection: {e}")
            return []
