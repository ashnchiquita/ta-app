import numpy as np
from scipy import signal
from components.hr_extractor.base import HRExtractor

class PeakDetection(HRExtractor):
    def extract(self, pulse_signal):
        peaks, _ = signal.find_peaks(pulse_signal, distance=self.fps/4)
        if len(peaks) < 2:
            return 0
                
        peak_times = peaks / self.fps
        intervals = np.diff(peak_times)

        avg_interval = np.mean(intervals)
        heart_rate = 60 / avg_interval
        
        return heart_rate
