import numpy as np
from scipy import signal
from components.rppg_signal_extractor.base import RPPGSignalExtractor

class CHROM(RPPGSignalExtractor):
    def extract(self, roi_data):
        rgb_signals = np.mean(roi_data, axis=(1, 2))
        rgb_norm = rgb_signals / np.mean(rgb_signals, axis=0)
        
        X_chrom = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
        Y_chrom = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
        
        std_X = np.std(X_chrom)
        std_Y = np.std(Y_chrom)
        
        alpha = std_X / std_Y
        pulse = X_chrom - alpha * Y_chrom
        
        # Bandpass filtering (0.7-4Hz, typical heart rate range 42-240 BPM)
        b, a = signal.butter(3, [0.7/self.fps*2, 4/self.fps*2], btype='bandpass')
        pulse_filtered = signal.filtfilt(b, a, pulse)
        
        return pulse_filtered
