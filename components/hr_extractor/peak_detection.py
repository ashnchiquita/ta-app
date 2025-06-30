import numpy as np
import scipy.signal
from components.hr_extractor.base import HRExtractor

class PeakDetection(HRExtractor):
    @staticmethod
    def _calculate_peak_hr(ppg_signal, fps):
        """Calculate heart rate based on PPG using peak detection."""
        ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
        hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fps)
        return hr_peak

    def calculate_hr(self, predictions, fps=30):
        return PeakDetection._calculate_peak_hr(predictions, fps=fps)
