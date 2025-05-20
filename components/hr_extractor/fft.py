import numpy as np
from components.hr_extractor.base import HRExtractor

class FFT(HRExtractor):
  def extract(self, pulse_signal):
    fft_size = len(pulse_signal)
    fft_data = np.abs(np.fft.rfft(pulse_signal, fft_size))
    freqs = np.fft.rfftfreq(fft_size, d=1.0/self.fps)
    
    # Expected heart rate range: 40-240 BPM
    valid_range = np.where((freqs >= 0.7) & (freqs <= 4.0))
    max_idx = np.argmax(fft_data[valid_range])
    hr_freq = freqs[valid_range][max_idx]
    heart_rate = hr_freq * 60  # Hz -> BPM
    
    return heart_rate
