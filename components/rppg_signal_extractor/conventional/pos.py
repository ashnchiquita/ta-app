import numpy as np
from scipy import signal
from components.rppg_signal_extractor.base import RPPGSignalExtractor

class POS(RPPGSignalExtractor):
    def extract(self, roi_data):
        # Handle varying ROI sizes by computing spatial mean for each frame
        if isinstance(roi_data, list):
            rgb_signals = []
            for frame in roi_data:
                frame_mean = np.mean(frame.reshape(-1, frame.shape[-1]), axis=0)
                rgb_signals.append(frame_mean)
            
            rgb_signals = np.array(rgb_signals)
        else: # Assuming roi_data is a numpy array
            rgb_signals = np.mean(roi_data, axis=(1, 2))

        # Normalize signals to reduce illumination variations
        rgb_norm = rgb_signals / np.mean(rgb_signals, axis=0)
        
        X = rgb_norm[:, 0] - rgb_norm[:, 1]    # R-G
        Y = rgb_norm[:, 0] + rgb_norm[:, 1] - 2 * rgb_norm[:, 2]    # R+G-2B
        
        alpha = np.std(X) / np.std(Y)
        pulse = X - alpha * Y
        
        # Bandpass filtering (0.7-4Hz, typical heart rate range 42-240 BPM)
        b, a = signal.butter(3, [0.7/self.fps*2, 4/self.fps*2], btype='bandpass')
        pulse_filtered = signal.filtfilt(b, a, pulse)
        
        return pulse_filtered
