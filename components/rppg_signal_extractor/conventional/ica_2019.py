"""
Modified ICA implementation based on:
- Original ICA paper: Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010)
- Modified approach: Green channel focus with ICA + HHT two-level noise reduction
"""
import math
import numpy as np
from scipy import linalg
from scipy import signal
from components.rppg_signal_extractor.conventional import utils
from components.rppg_signal_extractor.base import RPPGSignalExtractor
import time
from PyEMD import EMD  # For HHT implementation


class ICA2019(RPPGSignalExtractor):
    def extract(self, roi_data):
        return ICA2019.ICA_Modified(roi_data, self.fps)

    @staticmethod
    def ICA_Modified(roi_data, fps):
        # Cut off frequency
        t0 = time.time()
        LPF = 0.7
        HPF = 2.5
        
        # RGB decomposition
        RGB = ICA2019.avg_roi_data(roi_data)
        
        t1 = time.time()
        NyquistF = 1 / 2 * fps
        
        # Process all channels for ICA (required for blind source separation)
        BGRNorm = np.zeros(RGB.shape)
        Lambda = 100
        
        for c in range(3):
            BGRDetrend = utils.detrend(RGB[:, c], Lambda)
            BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
        
        # Special processing for green channel (the paper's modification)
        green_channel_idx = 1  # Assuming BGR format
        green_detrend = utils.detrend(RGB[:, green_channel_idx], Lambda)
        green_normalized = (green_detrend - np.mean(green_detrend)) / np.std(green_detrend)
        
        # First level: ICA on all RGB channels
        _, S = ICA2019.ica(np.mat(BGRNorm).H, 3)
        
        # Modified component selection - prioritize green channel characteristics
        MaxPx = np.zeros((1, 3))
        green_correlation = np.zeros((1, 3))  # New: correlation with green channel
        
        for c in range(3):
            # Original frequency domain analysis
            FF = np.fft.fft(S[c, :])
            F = np.arange(0, FF.shape[1]) / FF.shape[1] * fps * 60
            FF = FF[:, 1:]
            FF = FF[0]
            N = FF.shape[0]
            Px = np.abs(FF[:math.floor(N / 2)])
            Px = np.multiply(Px, Px)
            Px = Px / np.sum(Px, axis=0)
            MaxPx[0, c] = np.max(Px)
            
            # New: correlation with processed green channel
            green_correlation[0, c] = np.abs(np.corrcoef(S[c, :].real.flatten(), 
                                                        green_normalized.flatten())[0, 1])
        
        # Combined selection criteria (frequency + green correlation)
        combined_score = MaxPx[0] * green_correlation[0]  # Weight both factors
        MaxComp = np.argmax(combined_score)
        BVP_I = S[MaxComp, :]

        BVP_I_flattened = np.asarray(np.real(BVP_I).astype(np.double)).flatten()
        
        # HHT (BEFORE filtering, as per block diagram)
        BVP_hht = ICA2019.apply_hht(BVP_I_flattened, fps)

        # Moving Average & Bandpass Filter (AFTER HHT, as per block diagram)
        # Add moving average
        window_size = int(fps * 2)  # 2-second window
        BVP_ma = np.convolve(BVP_hht, np.ones(window_size)/window_size, mode='same')

        # Bandpass filter
        B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
        BVP_filtered = signal.filtfilt(B, A, BVP_ma)
        t2 = time.time()
        
        preprocessing_time = t1 - t0
        inference_time = t2 - t1
        return BVP_filtered, preprocessing_time, inference_time


    @staticmethod
    def apply_hht(signal_data, fps):
        """
        Apply Hilbert-Huang Transform for noise reduction
        This includes EMD (Empirical Mode Decomposition) followed by Hilbert Transform
        """
        try:
            # Empirical Mode Decomposition
            emd = EMD()
            IMFs = emd(signal_data)
            
            # Select relevant IMFs (typically the first few contain the physiological signal)
            # This is a simplified approach - you may need to adjust based on your specific requirements
            if len(IMFs) > 0:
                # Combine the first few IMFs that likely contain the heart rate signal
                # Typically IMFs 1-3 contain physiological signals for rPPG
                num_imfs_to_use = min(3, len(IMFs))
                reconstructed_signal = np.sum(IMFs[:num_imfs_to_use], axis=0)
                
                # Apply Hilbert Transform for instantaneous frequency analysis
                analytic_signal = signal.hilbert(reconstructed_signal)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fps
                
                # Return the reconstructed signal (you could also return frequency-based features)
                return reconstructed_signal
            else:
                # Fallback to original signal if EMD fails
                return signal_data
                
        except Exception as e:
            print(f"HHT processing failed: {e}. Returning filtered signal.")
            return signal_data

    @staticmethod
    def avg_roi_data(roi_data):
        """Calculates the average value of each frame."""
        if isinstance(roi_data, list):
            RGB = []
            for roi in roi_data:
                sum = np.sum(np.sum(roi, axis=0), axis=0)
                RGB.append(sum / (roi.shape[0] * roi.shape[1]))
            return np.asarray(RGB)
        else:  # If frames is a numpy array, process it directly
            RGB = np.mean(roi_data, axis=(1, 2))
            return RGB

    @staticmethod
    def ica(X, Nsources, Wprev=0):
        """ICA implementation (same as original)"""
        nRows = X.shape[0]
        nCols = X.shape[1]
        if nRows > nCols:
            print("Warning - The number of rows cannot be greater than the number of columns.")
            print("Please transpose input.")

        if Nsources > min(nRows, nCols):
            Nsources = min(nRows, nCols)
            print('Warning - The number of sources cannot exceed number of observation channels.')
            print('The number of sources will be reduced to the number of observation channels ', Nsources)

        Winv, Zhat = ICA2019.jade(X, Nsources, Wprev)
        W = np.linalg.pinv(Winv)
        return W, Zhat

    @staticmethod
    def jade(X, m, Wprev):
        """JADE algorithm implementation (same as original)"""
        n = X.shape[0]
        T = X.shape[1]
        nem = m
        seuil = 1 / math.sqrt(T) / 100
        
        if m < n:
            D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
            Diag = D
            k = np.argsort(Diag)
            pu = Diag[k]
            ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
            bl = np.true_divide(np.ones((m, 1)), ibl.reshape(-1, 1))
            W = np.matmul(np.diag(bl.flatten()), np.transpose(U[0:n, k[n - m:n]]))
            IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl.flatten()))
        else:
            IW = linalg.sqrtm(np.matmul(X, X.H) / T)
            W = np.linalg.inv(IW)

        Y = np.mat(np.matmul(W, X))
        R = np.matmul(Y, Y.H) / T
        C = np.matmul(Y, Y.T) / T
        Q = np.zeros((m * m * m * m, 1))
        index = 0

        for lx in range(m):
            Y1 = Y[lx, :]
            for kx in range(m):
                Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
                for jx in range(m):
                    Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                    for ix in range(m):
                        Q[index] = np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(
                            T)) - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                        index += 1
        
        # Compute and Reshape the significant Eigen
        D, U = np.linalg.eig(Q.reshape(m * m, m * m))
        Diag = abs(D)
        K = np.argsort(Diag)
        la = Diag[K]
        M = np.zeros((m, nem * m), dtype=complex)
        Z = np.zeros(m)
        h = m * m - 1
        
        for u in range(0, nem * m, m):
            Z = U[:, K[h]].reshape((m, m))
            M[:, u:u + m] = la[h] * Z
            h = h - 1
        
        # Approximate the Diagonalization of the Eigen Matrices:
        B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
        Bt = np.mat(B).H

        encore = 1
        if Wprev == 0:
            V = np.eye(m).astype(complex)
        else:
            V = np.linalg.inv(Wprev)
        
        # Main Loop:
        while encore:
            encore = 0
            for p in range(m - 1):
                for q in range(p + 1, m):
                    Ip = np.arange(p, nem * m, m)
                    Iq = np.arange(q, nem * m, m)
                    g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                    temp1 = np.matmul(g, g.H)
                    temp2 = np.matmul(B, temp1)
                    temp = np.matmul(temp2, Bt)
                    D, vcp = np.linalg.eig(np.real(temp))
                    K = np.argsort(D)
                    la = D[K]
                    angles = vcp[:, K[2]]
                    if angles[0, 0] < 0:
                        angles = -angles
                    c = np.sqrt(0.5 + angles[0, 0] / 2)
                    s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                    if abs(s) > seuil:
                        encore = 1
                        pair = [p, q]
                        G = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                        V[:, pair] = np.matmul(V[:, pair], G)
                        M[pair, :] = np.matmul(G.H, M[pair, :])
                        temp1 = c * M[:, Ip] + s * M[:, Iq]
                        temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                        M[:, Ip] = temp1
                        M[:, Iq] = temp2

        # Whiten the Matrix
        # Estimation of the Mixing Matrix and Signal Separation
        A = np.matmul(IW, V)
        S = np.matmul(np.mat(V).H, Y)
        return A, S
