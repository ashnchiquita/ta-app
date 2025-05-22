from system.system import System
import time
from components.face_detector.haar_cascade import HaarCascade
from components.face_tracker.centroid import Centroid
from components.roi_selector.cheeks import Cheeks
from components.rppg_signal_extractor.conventional.chrom import CHROM
from components.hr_extractor.fft import FFT

def run_default_system():
  rppg_system = System(camera_id=0)
  try:
    rppg_system.start()
    
    while rppg_system.running:
      time.sleep(0.1)
  except KeyboardInterrupt:
    pass
  finally:
    rppg_system.stop()

def run_custom_system():
    face_detector = HaarCascade(scale_factor=1.2, min_neighbors=6)
    face_tracker = Centroid(max_disappeared=20)
    roi_selector = Cheeks(target_size=(64, 64))
    rppg_signal_extractor = CHROM()
    hr_extractor = FFT()
    
    rppg_system = System(
      camera_id=0,
      face_detector=face_detector,
      face_tracker=face_tracker,
      roi_selector=roi_selector,
      rppg_signal_extractor=rppg_signal_extractor,
      hr_extractor=hr_extractor,
      window_size=300,
      step_size=30
    )
    
    try:
      rppg_system.start()
      
      while rppg_system.running:
        time.sleep(0.1)
    except KeyboardInterrupt:
      pass
    finally:
      rppg_system.stop()
      
if __name__ == "__main__":
  run_default_system()
  # run_custom_system()
