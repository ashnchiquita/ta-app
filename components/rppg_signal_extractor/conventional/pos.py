"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""

import math
import numpy as np
from scipy import signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from components.rppg_signal_extractor.base import RPPGSignalExtractor
from components.rppg_signal_extractor.conventional import utils

class POS(RPPGSignalExtractor):
    def extract(self, roi_data, method='simple'):
        """
        Extract pulse signal using POS algorithm.
        
        Args:
            roi_data: ROI data (list of arrays or numpy array)
            method: 'simple' (fastest), 'optimized' (balanced), 'original' (slowest but most accurate)
        """
        return POS.POS_WANG(roi_data, self.fps)

    @staticmethod
    def avg_roi_data(roi_data):
        """Calculates the average value of each frame."""
        if isinstance(roi_data, list):
            RGB = []
            for roi in roi_data:
                summation = np.sum(np.sum(roi, axis=0), axis=0)
                RGB.append(summation / (roi.shape[0] * roi.shape[1]))
            return np.asarray(RGB)
        else:  # If frames is a numpy array, process it directly
            RGB = np.mean(roi_data, axis=(1, 2))
            return RGB

    @staticmethod
    def POS_WANG(roi_data, fps):
        WinSec = 1.6
        RGB = POS.avg_roi_data(roi_data)
        N = RGB.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * fps)

        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
                Cn = np.mat(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])

        BVP = H
        BVP = utils.detrend(np.mat(BVP).H, 100)
        BVP = np.asarray(np.transpose(BVP))[0]
        b, a = signal.butter(1, [0.75 / fps * 2, 3 / fps * 2], btype='bandpass')
        BVP = signal.filtfilt(b, a, BVP.astype(np.double))
        return BVP
