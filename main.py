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

VIDEOS = [
    '/home/pme/ta/data/camera/00/video_00_20250503_152120.npy',
    '/home/pme/ta/data/camera/01/video_01_20250503_152345.npy',
    '/home/pme/ta/data/camera/02/video_02_20250503_152754.npy',
    '/home/pme/ta/data/camera/03/video_03_20250503_153102.npy',
    '/home/pme/ta/data/camera/04/video_04_20250503_153508.npy',
    '/home/pme/ta/data/camera/05/video_05_20250503_153825.npy',
    '/home/pme/ta/data/camera/06/video_06_20250503_154102.npy',
    '/home/pme/ta/data/camera/07/video_07_20250503_154629.npy',
    '/home/pme/ta/data/camera/08/video_08_20250503_155339.npy',
    '/home/pme/ta/data/camera/09/video_09_20250503_155545.npy',
    '/home/pme/ta/data/camera/10/video_10_20250503_155820.npy',
    '/home/pme/ta/data/camera/11/video_11_20250503_160024.npy',
    '/home/pme/ta/data/camera/12/video_12_20250503_160447.npy',
    '/home/pme/ta/data/camera/13/video_13_20250503_160703.npy',
    '/home/pme/ta/data/camera/14/video_14_20250503_160909.npy',
    '/home/pme/ta/data/camera/15/video_15_20250503_161440.npy',
    '/home/pme/ta/data/camera/16/video_16_20250503_161812.npy',
    '/home/pme/ta/data/camera/17/video_17_20250503_162022.npy',
    '/home/pme/ta/data/camera/18/video_18_20250503_162823.npy',
    '/home/pme/ta/data/camera/19/video_19_20250503_163047.npy',
    '/home/pme/ta/data/camera/20/video_20_20250503_163304.npy',
    '/home/pme/ta/data/camera/21/video_21_20250503_164031.npy',
    '/home/pme/ta/data/camera/22/video_22_20250503_164308.npy',
    '/home/pme/ta/data/camera/23/video_23_20250503_164535.npy',
    '/home/pme/ta/data/camera/24/video_24_20250503_165937.npy',
    '/home/pme/ta/data/camera/25/video_25_20250503_170153.npy',
    '/home/pme/ta/data/camera/26/video_26_20250503_170735.npy'
]
def run_default_system_npy():
    rppg_system = System(
        video_file=VIDEOS[26],
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
