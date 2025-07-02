from abc import ABC, abstractmethod
import numpy as np
import scipy
from scipy.signal import butter
from scipy.sparse import spdiags

class HRExtractor(ABC):
    def __init__(self, fps: float=30.0, diff_flag=False, use_bandpass=False, use_detrend=False):
        self.fps = fps
        self.diff_flag = diff_flag
        self.use_bandpass = use_bandpass
        self.use_detrend = use_detrend

    @staticmethod
    def _next_power_of_2(x):
        """Calculate the nearest power of 2."""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    @staticmethod
    def _detrend(input_signal, lambda_value):
        """Detrend PPG signal."""
        signal_length = input_signal.shape[0]
        # observation matrix
        H = np.identity(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = spdiags(diags_data, diags_index,
                    (signal_length - 2), signal_length).toarray()
        detrended_signal = np.dot(
            (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
        return detrended_signal

    @staticmethod
    def power2db(mag):
        """Convert power to db."""
        return 10 * np.log10(mag)
        
    @abstractmethod
    def calculate_hr(self, predictions, fps=30):
        """Calculate heart rate from predictions."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def extract(self, pulse_signal):
        return self.calculate_per_bvp(pulse_signal, fps=self.fps, diff_flag=self.diff_flag, use_bandpass=self.use_bandpass, use_detrend=self.use_detrend)

    def calculate_per_bvp(self, predictions, fps=30, diff_flag=True, use_bandpass=True, use_detrend=True):
        """Calculate video-level HR and SNR"""
        if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
            predictions = np.cumsum(predictions)
        if use_detrend:
            predictions = HRExtractor._detrend(predictions, 100)
        if use_bandpass:
            # bandpass filter between [0.75, 2.5] Hz, equals [45, 150] beats per min
            # bandpass filter between [0.6, 3.3] Hz, equals [36, 198] beats per min
            #
            # Note: to more closely match results in the NeurIPS 2023 toolbox paper,
            # we recommend using 0.75 in place of 0.6 and 2.5 in place of 3.3 in the 
            # below line.
            [b, a] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
            predictions = scipy.signal.filtfilt(b, a, np.double(predictions))


        hr_pred = self.calculate_hr(predictions, fps=fps)
        return hr_pred
