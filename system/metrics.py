class Metrics:
    def __init__(self):
        self.processing_count = 0
        self.total_processing_time = 0
        self.processing_time = {
            'face_detection': 0,
            'face_tracking': 0,
            'roi_selection': 0,
            'core_time': 0,
        }
        self.skipped_frames = 0

        self.start_time = 0
        self.end_time = 0

    def __str__(self):
        return f"Processing Count: {self.processing_count}\nTotal Processing Time: {self.total_processing_time} seconds\nFace Detection Time: {self.processing_time['face_detection']} seconds\nFace Tracking Time: {self.processing_time['face_tracking']} seconds\nROI Selection Time: {self.processing_time['roi_selection']} seconds\nCore Time: {self.processing_time['core_time']} seconds\nSkipped Frames: {self.skipped_frames}\nStart Time: {self.start_time}\nEnd Time: {self.end_time}\nAverageFPS: {self.processing_count / (self.end_time - self.start_time)}"
