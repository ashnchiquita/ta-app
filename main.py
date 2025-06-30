from system.system import System
import time
from components.face_detector.haar_cascade import HaarCascade
from components.face_tracker.centroid import Centroid
from components.rppg_signal_extractor.conventional.chrom import CHROM
from components.hr_extractor.fft import FFT

def run_default_system():
    rppg_system = System(camera_id=0)
    try:
        rppg_system.start()
        
        while rppg_system.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rppg_system.stop()

def run_default_system_npy():
    rppg_system = System(
        video_file='/home/pme/ta/data/camera/26/video_26_20250503_170735.npy'
        # video_file='C:\\Users\\dyogggeming\\TA\\tes\\rppg-data\\fix\\camera\\29\\video_29_20250503_171818.npy',
    )
    try:
        rppg_system.start()
        
        while rppg_system.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rppg_system.stop()

def run_default_system_npy_timestamp():
    rppg_system = System(
        video_file='/home/pme/ta/data/camera/00/video_00_20250503_152120.npy',
        timestamp_file='/home/pme/ta/data/camera/00/timestamps_00_20250503_152120.csv',
        # video_file='C:\\Users\\dyogggeming\\TA\\tes\\rppg-data\\fix\\camera\\29\\video_29_20250503_171818.npy',
        # timestamps='C:\\Users\\dyogggeming\\TA\\tes\\rppg-data\\fix\\camera\\29\\timestamps_29_20250503_171818.csv'
    )
    try:
        rppg_system.start()
        
        while rppg_system.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rppg_system.stop()
            
if __name__ == "__main__":
    # run_default_system()
    run_default_system_npy()
    # run_default_system_npy_timestamp()
