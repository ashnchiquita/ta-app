class Metrics:
  def __init__(self):
    self.processing_count = 0
    self.total_processing_time = 0
    self.processing_time = {
      'face_detection': 0,
      'face_tracking': 0,
      'roi_selection': 0,
      'signal_extraction': 0,
      'hr_extraction': 0
    }

  def __str__(self):
    return f"Processing Count: {self.processing_count}\nTotal Processing Time: {self.total_processing_time} seconds\nFace Detection Time: {self.processing_time['face_detection']} seconds\nFace Tracking Time: {self.processing_time['face_tracking']} seconds\nROI Selection Time: {self.processing_time['roi_selection']} seconds\nSignal Extraction Time: {self.processing_time['signal_extraction']} seconds\nHR Extraction Time: {self.processing_time['hr_extraction']} seconds"
