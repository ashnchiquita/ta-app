# The Chrominance Method from: De Haan, G., & Jeanne, V. (2013). 
# Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886. 
# DOI: 10.1109/TBME.2013.2266196
import math
import numpy as np
from scipy import signal
from components.rppg_signal_extractor.base import RPPGSignalExtractor
import time

class CHROM(RPPGSignalExtractor):
    def extract(self, roi_data):
        """
        Extract rPPG signal using CHROME-DEHAAN method.
        
        Args:
            roi_data: List of frame ROIs or numpy array of shape (frames, height, width, channels)
        
        Returns:
            BVP signal (Blood Volume Pulse)
        """
        
        # Use the original CHROME_DEHAAN algorithm with self.fps
        return CHROM.CHROME_DEHAAN(roi_data, self.fps)

    @staticmethod
    def CHROME_DEHAAN(roi_data, fps):
        t1 = time.time()
        
        LPF = 0.7
        HPF = 2.5
        WinSec = 1.6

        RGB = CHROM.avg_roi_data(roi_data)
        t2 = time.time()
        
        FN = RGB.shape[0]
        NyquistF = 1/2*fps
        B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

        WinL = math.ceil(WinSec*fps)
        if(WinL % 2):
            WinL = WinL+1
        NWin = math.floor((FN-WinL//2)/(WinL//2))
        WinS = 0
        WinM = int(WinS+WinL//2)
        WinE = WinS+WinL
        totallen = (WinL//2)*(NWin+1)
        S = np.zeros(totallen)

        for i in range(NWin):
            RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
            RGBNorm = np.zeros((WinE-WinS, 3))
            for temp in range(WinS, WinE):
                RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)
            Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
            Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
            Xf = signal.filtfilt(B, A, Xs, axis=0)
            Yf = signal.filtfilt(B, A, Ys)

            Alpha = np.std(Xf) / np.std(Yf)
            SWin = Xf-Alpha*Yf
            SWin = np.multiply(SWin, signal.windows.hann(WinL))

            temp = SWin[:int(WinL//2)]
            S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
            S[WinM:WinE] = SWin[int(WinL//2):]
            WinS = WinM
            WinM = WinS+WinL//2
            WinE = WinS+WinL
        BVP = S

        t3 = time.time()
        preprocessing_time = t2 - t1
        inference_time = t3 - t2
        return BVP, preprocessing_time, inference_time

    @staticmethod
    def avg_roi_data(roi_data):
        "Calculates the average value of each frame."
        if isinstance(roi_data, list):
            RGB = []
            for frame in roi_data:
                sum = np.sum(np.sum(frame, axis=0), axis=0)
                RGB.append(sum/(frame.shape[0]*frame.shape[1]))
            return np.asarray(RGB)
        else: # If frames is a numpy array, process it directly
            RGB = np.mean(roi_data, axis=(1, 2))
            return RGB 
