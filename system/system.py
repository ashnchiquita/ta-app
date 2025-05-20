import cv2
import numpy as np
import threading
import queue
import time
from components.face_detector.haar_cascade import HaarCascade
from components.face_tracker.centroid import Centroid
from components.roi_selector.fullface import FullFace
from components.rppg_signal_extractor.conventional.pos import POS
from components.hr_extractor.fft import FFT
from system.pipeline import Pipeline
from components.roi_selector.cheeks import Cheeks
from components.roi_selector.forehead import Forehead

class System:
  def __init__(self, 
                camera_id=0,
                face_detector=None,
                face_tracker=None,
                roi_selector=None,
                rppg_signal_extractor=None,
                hr_extractor=None,
                window_size=300,
                fps=30,
                step_size=30):
    
    self.camera_id = camera_id
    self.cap = cv2.VideoCapture(camera_id)
    
    self.face_detector = face_detector or HaarCascade()
    self.face_tracker = face_tracker or Centroid()
    self.roi_selector = roi_selector or FullFace()
    self.rppg_signal_extractor = rppg_signal_extractor or POS(fps=fps)
    self.hr_extractor = hr_extractor or FFT(fps=fps)
    
    self.pipeline = Pipeline(
        self.rppg_signal_extractor,
        self.hr_extractor,
        window_size=window_size,
        fps=fps,
        step_size=step_size
    )

    self.frame_queue = queue.Queue(maxsize=30)
    self.result_queue = queue.Queue()
    self.running = False
    
    self.heart_rates = {}
    
  def start(self):
    self.running = True
    
    self.capture_thread = threading.Thread(target=self.capture_frames)
    self.processing_thread = threading.Thread(target=self.process_frames)
    self.display_thread = threading.Thread(target=self.display_frames)
    
    self.capture_thread.start()
    self.processing_thread.start()
    self.display_thread.start()

    print("System started. Press 'q' to exit.")
      
  def stop(self):
    self.running = False
    
    self.capture_thread.join()
    self.processing_thread.join()
    self.display_thread.join()
    
    self.cap.release()
    cv2.destroyAllWindows()

    print("System stopped.")
      
  def capture_frames(self):
    print("Starting frame capture...")
    while self.running:
      print("Capturing frames...")
      ret, frame = self.cap.read()
      if not ret:
        self.running = False
        break
          
      try:
        self.frame_queue.put(frame, block=False)
      except queue.Full:
        # If queue is full, skip
        pass
              
  def process_frames(self):
    while self.running:
      try:
        frame = self.frame_queue.get(timeout=1)
        timestamp = time.time()
        
        face_rects = self.face_detector.detect(frame)
        objects = self.face_tracker.update(face_rects)
        
        for (object_id, centroid) in objects.items():
          for face_rect in face_rects:
            x, y, x_end, y_end = face_rect
            c_x = (x + x_end) // 2
            c_y = (y + y_end) // 2
            
            if np.linalg.norm(np.array([c_x, c_y]) - np.array(centroid)) < 30:
              roi = self.roi_selector.select(frame, face_rect)
              self.pipeline.add_face_data(object_id, roi, timestamp)
              break
        
        new_results = self.pipeline.process_faces()
        
        for face_id, hr in new_results.items():
          self.heart_rates[face_id] = hr
        
        self.result_queue.put((frame, face_rects, objects, self.heart_rates.copy()))
          
      except queue.Empty:
        continue
              
  def display_frames(self):
    while self.running:
      try:
        frame, rects, objects, heart_rates = self.result_queue.get(timeout=1)
        
        for (object_id, centroid) in objects.items():
          for (x, y, x_end, y_end) in rects:
            c_x = (x + x_end) // 2
            c_y = (y + y_end) // 2
            
            if np.linalg.norm(np.array([c_x, c_y]) - np.array(centroid)) < 30:
              cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)
                
              if isinstance(self.roi_selector, FullFace):
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x_end, y_end), (0, 255, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                  
              elif isinstance(self.roi_selector, Cheeks):
                face_width = x_end - x
                face_height = y_end - y
                
                left_cheek_x = x + int(face_width * 0.1)
                right_cheek_x = x + int(face_width * 0.6)
                cheek_y = y + int(face_height * 0.4)
                cheek_width = int(face_width * 0.3)
                cheek_height = int(face_height * 0.3)
                
                left_cheek_points = np.array([
                  [left_cheek_x, cheek_y],
                  [left_cheek_x + cheek_width, cheek_y],
                  [left_cheek_x + cheek_width, cheek_y + cheek_height],
                  [left_cheek_x, cheek_y + cheek_height]
                ])
                cv2.polylines(frame, [left_cheek_points], True, (0, 255, 255), 2)
                
                right_cheek_points = np.array([
                  [right_cheek_x, cheek_y],
                  [right_cheek_x + cheek_width, cheek_y],
                  [right_cheek_x + cheek_width, cheek_y + cheek_height],
                  [right_cheek_x, cheek_y + cheek_height]
                ])
                cv2.polylines(frame, [right_cheek_points], True, (0, 255, 255), 2)
                  
              elif isinstance(self.roi_selector, Forehead):
                forehead_x = x + int((x_end - x) * 0.2)
                forehead_width = int((x_end - x) * 0.6)
                forehead_y = y + int((y_end - y) * 0.05)
                forehead_height = int((y_end - y) * 0.25)
                  
                forehead_points = np.array([
                  [forehead_x, forehead_y],
                  [forehead_x + forehead_width, forehead_y],
                  [forehead_x + forehead_width, forehead_y + forehead_height],
                  [forehead_x, forehead_y + forehead_height]
                ])
                cv2.polylines(frame, [forehead_points], True, (0, 255, 255), 2)
                    
              text = f"ID: {object_id}"
              cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
              
              if object_id in heart_rates:
                  hr_text = f"HR: {heart_rates[object_id]:.1f} BPM"
                  cv2.putText(frame, hr_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
              break
        
        cv2.imshow("Multi-Person rPPG [Press 'q' to exit]", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
          self.running = False
              
      except queue.Empty:
        continue
