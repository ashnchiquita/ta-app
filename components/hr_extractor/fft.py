import numpy as np
import scipy.signal
from components.hr_extractor.base import HRExtractor

class FFT(HRExtractor):
    @staticmethod
    def _calculate_fft_hr(ppg_signal, fps=60, low_pass=0.75, high_pass=2.5):
        # Note: to more closely match results in the NeurIPS 2023 toolbox paper,
        # we recommend low_pass=0.75 and high_pass=2.5 instead of the defaults above.
        """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
        ppg_signal = np.expand_dims(ppg_signal, 0)
        N = HRExtractor._next_power_of_2(ppg_signal.shape[1])
        f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fps, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
        return fft_hr
        
    def calculate_hr(self, predictions, fps=30):
        return FFT._calculate_fft_hr(predictions, fps=fps, low_pass=0.75, high_pass=2.5)
