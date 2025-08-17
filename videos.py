import os
import time

MY_VIDEOS = [
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
    '/home/pme/ta/data/camera/26/video_26_20250503_170735.npy',
]

ubfc_exceptions = [2, 6, 7, 19, 21, 28, 29]
ubfc_rppg_subjects = [i for i in range(1, 50) if i not in ubfc_exceptions]
UBFC_RPPG_VIDEOS = {}
for subject in ubfc_rppg_subjects:
    UBFC_RPPG_VIDEOS[subject] = f'/media/pme/SSD NUR/Datasets/rPPG/UBFC-rPPG/subject{subject}/vid.avi'

UBFC_RPPG_CUSTOM_VIDEOS = {}
UBFC_RPPG_CUSTOM_VERSIONS = ['a', 'b', 'c']
for version in UBFC_RPPG_CUSTOM_VERSIONS:
    UBFC_RPPG_CUSTOM_VIDEOS[version] = {}
    
    for i in range(1, 10):
        UBFC_RPPG_CUSTOM_VIDEOS[version][i] = f'/media/pme/SSD NUR/Multisubject/Sintetik/{version}/collage_{i}.npy'

def is_valid_my_video_index(idx):
    return 0 <= idx < len(MY_VIDEOS) and os.path.exists(MY_VIDEOS[idx])

def is_valid_ubfc_rppg_subject_index(idx):
    return idx in ubfc_rppg_subjects and os.path.exists(UBFC_RPPG_VIDEOS[idx])

def is_valid_ubfc_rppg_custom_video_index(version, idx):
    return version in UBFC_RPPG_CUSTOM_VIDEOS and idx in UBFC_RPPG_CUSTOM_VIDEOS[version] and os.path.exists(UBFC_RPPG_CUSTOM_VIDEOS[version][idx])

def get_my_videos_log_dir(idx, resolution_factor):
    video_name = os.path.basename(MY_VIDEOS[idx]).split('_')[1]
    return f'/home/pme/ta/ta-app/logs/my-videos_r{resolution_factor}/{video_name}'

def get_ubfc_rppg_log_dir(subject_idx, resolution_factor):
    return f'/home/pme/ta/ta-app/logs/ubfc-rppg_r{resolution_factor}/{subject_idx}'

def get_ubfc_rppg_custom_log_dir(version, video_idx, resolution_factor):
    return f'/home/pme/ta/ta-app/logs/ubfc-rppg-custom_r{resolution_factor}/{version}/{video_idx}'

def get_camera_log_dir(resolution_factor):
    return f'/home/pme/ta/ta-app/logs/camera_r{resolution_factor}/{time.time()}/'
