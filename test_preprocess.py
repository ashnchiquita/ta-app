import numpy as np
import csv
import os
from components.face_detector.hef.scrfd.scrfd import SCRFD
from components.roi_selector.fullface_square import FullFaceSquare
# from system.incremental_processor.rolling_statistics_original import RollingStatisticsOriginal as RollingStatistics
from system.incremental_processor.rolling_statistics import RollingStatistics as RollingStatistics
from system.incremental_processor.rolling_statistics import RollingStatisticsUltraLight as RollingStatisticsWelford
from system.incremental_processor.rolling_statistics import RollingStatisticsOptimized

# from system.incremental_processor.rolling_statistics_optimized import RollingStatisticsOptimized, RollingStatisticsWelford

def stdz_mean_std(single_clip):
    """Calculate mean and standard deviation of the video data."""
    mean = np.mean(single_clip)
    std = np.std(single_clip)

    clip_sum_sq = np.sum(single_clip ** 2)
    print(f"Clip sum of squares: {clip_sum_sq}")

    total = np.prod(single_clip.shape)
    print(f"Total number of elements in the clip: {total}")

    # manual std calc - FIXED VERSION
    # Method 1: Using the mathematically stable formula
    mean_precise = np.sum(single_clip) / total  # More precise mean calculation
    variance_manual_1 = clip_sum_sq / total - (mean_precise ** 2)
    print(f"Manual variance (method 1): {variance_manual_1}")
    
    # Method 2: Using the standard deviation definition (more numerically stable)
    variance_manual_2 = np.sum((single_clip - mean) ** 2) / total
    print(f"Manual variance (method 2): {variance_manual_2}")
    
    # Method 3: Using numpy's variance for comparison
    variance_numpy = np.var(single_clip)
    print(f"NumPy variance: {variance_numpy}")
    
    # Manual standard deviation
    std_manual = np.sqrt(variance_manual_2)
    print(f"Manual std: {std_manual}")
    print(f"NumPy std: {std}")
    
    return mean, std

def diff_normalize_std(single_clip):
    """Calculate discrete difference in video data along the time-axis and normalize by its standard deviation."""
    n, h, w, c = single_clip.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (single_clip[j + 1, :, :, :] - single_clip[j, :, :, :]) / (
                single_clip[j + 1, :, :, :] + single_clip[j, :, :, :] + 1e-7)

    std = np.std(diffnormalized_data)

    return std

def load_video_frames(video_file):
    video_frames = np.load(video_file)
    video_frames = np.flip(video_frames, axis=3)

    print(f"Video frames loaded: {len(video_frames)} frames")
    # face detection
    face_detector = SCRFD(variant='500m')

    # roi
    roi_selector = FullFaceSquare(target_size=(72,72), larger_box_coef=1.5)

    count = 0

    # Apply face detection and ROI selection
    roi_data = []
    for frame in video_frames:
        faces = face_detector.detect(frame)
        if faces:
            roi, _ = roi_selector.select(frame, faces[0])
            roi_data.append(roi)

        count += 1

        if count > 0 and count % 100 == 0:
            print(f"Processed {count} frames")

    roi_data = np.array(roi_data)
    print(f"ROI frames extracted: {len(roi_data)} frames")

    # save
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    roi_save_path = f"{video_name}_roi_data.npy"
    np.save(roi_save_path, roi_data)
    print(f"ROI data saved to {roi_save_path}")

    return roi_data

def load_from_saved(video_file):
    roi_data = np.load(video_file)

    return roi_data

def test_preprocess_gt(video_file, roi_frames, step=30, window_size=180):
    n_frames = roi_frames.shape[0]
    steps = n_frames // step
    
    # Create CSV file name based on video file name
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    csv_filename = f"{video_name}_preprocess_stats_gt_step{step}_windowsize{window_size}.csv"
    
    # Write data to CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['step_number', 'start_idx', 'end_idx', 'stdz_mean', 'stdz_std', 'diffnorm_std'])

        diff_size = window_size - step
        
        for i in range(steps):
            start_idx = i * step - diff_size

            if start_idx < 0:
                start_idx = 0
            
            end_idx = (i + 1) * step
            
            clip = roi_frames[start_idx:end_idx]

            stdz_mean, stdz_std = stdz_mean_std(clip)
            diffnorm_std = diff_normalize_std(clip)
            
            # Write data row
            writer.writerow([i, start_idx, end_idx, stdz_mean, stdz_std, diffnorm_std])

            print(f"Step {i+1}/{steps} done")
    
    print(f"Data saved to {csv_filename}")

def test_preprocess_rolling(video_file, roi_frames, step=30, window_size=180):
    rolling = RollingStatistics(window_size=window_size, step=step, frame_shape=(72, 72, 3))

    n_frames = roi_frames.shape[0]
    steps = n_frames // step
    
    # Create CSV file name based on video file name
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    csv_filename = f"{video_name}_preprocess_stats_rolling_step{step}_windowsize{window_size}.csv"
    
    # Write data to CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['step_number', 'start_idx', 'end_idx', 'stdz_mean', 'stdz_std', 'diffnorm_std'])

        diff_size = window_size - step
        
        for i in range(steps):
            start_idx = i * step - diff_size

            if start_idx < 0:
                start_idx = 0
            
            end_idx = (i + 1) * step
            
            clip = roi_frames[start_idx:end_idx]

            for frame in clip:
                rolling.add_frame(frame)

            if not rolling.is_ready():
                print(f"Rolling statistics not ready for step {i+1}/{steps}, skipping")
                continue

            global_mean = rolling.get_mean()
            global_std = rolling.get_std()
            
            # Write data row
            writer.writerow([i, start_idx, end_idx, global_mean, global_std, global_std])

            print(f"Step {i+1}/{steps} done")
    
    print(f"Data saved to {csv_filename}")

def test_rolling_statistics_comparison(video_file, roi_frames, step=30, window_size=180):
    """Compare all rolling statistics implementations."""
    print("=== Testing Rolling Statistics Implementations ===")
    
    # Initialize all implementations
    rolling_original = RollingStatistics(window_size=window_size, step=step, frame_shape=(72, 72, 3))
    rolling_optimized = RollingStatisticsOptimized(window_size=window_size, step=step, frame_shape=(72, 72, 3))
    rolling_welford = RollingStatisticsWelford(window_size=window_size, step=step, frame_shape=(72, 72, 3))
    
    n_frames = roi_frames.shape[0]
    steps = min(100, n_frames // step)  # Test first 5 steps
    
    # Create CSV file name based on video file name
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    csv_filename = f"{video_name}_rolling_comparison_step{step}_windowsize{window_size}.csv"
    
    # Write comparison data to CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            'step_number', 'start_idx', 'end_idx', 
            'gt_mean', 'gt_std',
            'original_mean', 'original_std', 'original_ready',
            'optimized_mean', 'optimized_std', 'optimized_ready',
            'welford_mean', 'welford_std', 'welford_ready'
        ])

        diff_size = window_size - step
        
        for i in range(steps):
            start_idx = i * step - diff_size
            if start_idx < 0:
                start_idx = 0
            end_idx = (i + 1) * step
            
            clip = roi_frames[start_idx:end_idx]
            
            print(f"\n--- Step {i+1}/{steps} ---")
            print(f"Clip shape: {clip.shape}")
            
            # Ground truth calculation
            gt_mean = np.mean(clip)
            gt_std = np.std(clip)
            print(f"Ground Truth - Mean: {gt_mean:.6f}, Std: {gt_std:.6f}")
            
            # Add frames to all rolling statistics
            for frame in clip:
                rolling_original.add_frame(frame)
                rolling_optimized.add_frame(frame)
                rolling_welford.add_frame(frame)
            
            # Get results from all implementations
            try:
                print("\nOriginal Implementation:")
                orig_mean = rolling_original.get_mean()
                orig_std = rolling_original.get_std()
                orig_ready = rolling_original.is_ready()
                print(f"Ready: {orig_ready}")
            except Exception as e:
                print(f"Original failed: {e}")
                orig_mean = orig_std = float('nan')
                orig_ready = False
            
            try:
                print("\nOptimized Implementation:")
                opt_mean = rolling_optimized.get_mean()
                opt_std = rolling_optimized.get_std()
                opt_ready = rolling_optimized.is_ready()
                print(f"Ready: {opt_ready}")
            except Exception as e:
                print(f"Optimized failed: {e}")
                opt_mean = opt_std = float('nan')
                opt_ready = False
            
            try:
                print("\nWelford Implementation:")
                wel_mean = rolling_welford.get_mean()
                wel_std = rolling_welford.get_std()
                wel_ready = rolling_welford.is_ready()
                print(f"Ready: {wel_ready}")
            except Exception as e:
                print(f"Welford failed: {e}")
                wel_mean = wel_std = float('nan')
                wel_ready = False
            
            # Write data row
            writer.writerow([
                i, start_idx, end_idx, 
                gt_mean, gt_std,
                orig_mean, orig_std, orig_ready,
                opt_mean, opt_std, opt_ready,
                wel_mean, wel_std, wel_ready
            ])
    
    print(f"\nComparison data saved to {csv_filename}")

    

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

if __name__ == "__main__":
    video_file = VIDEOS[0]  # Change this to test different videos
    roi_frames = load_from_saved("video_00_20250503_152120_roi_data.npy")
    
    # Test ground truth calculation
    # test_preprocess_gt(video_file, roi_frames, step=30, window_size=180)
    
    # Test and compare all rolling statistics implementations
    test_rolling_statistics_comparison(video_file, roi_frames, step=30, window_size=180)
    
    # Optional: Test original rolling implementation
    # test_preprocess_rolling(video_file, roi_frames, step=30, window_size=180)
