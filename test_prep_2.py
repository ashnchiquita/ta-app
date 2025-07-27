import numpy as np
import csv
import os
import psutil
import gc
from components.face_detector.hef.scrfd.scrfd import SCRFD
from components.roi_selector.fullface_square import FullFaceSquare
from system.incremental_processor.rolling_statistics import RollingStatistics as RollingStatistics

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024**2):.1f} MB")
    
    # System memory
    virtual_memory = psutil.virtual_memory()
    print(f"System memory: {virtual_memory.percent:.1f}% used ({virtual_memory.used / (1024**3):.1f}/{virtual_memory.total / (1024**3):.1f} GB)")

def diff_normalize_std(single_clip):
    """Calculate discrete difference in video data along the time-axis and normalize by its standard deviation."""
    n, h, w, c = single_clip.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (single_clip[j + 1, :, :, :] - single_clip[j, :, :, :]) / (
                single_clip[j + 1, :, :, :] + single_clip[j, :, :, :] + 1e-7)

    std = np.std(diffnormalized_data)

    return diffnormalized_data, std

def load_video_frames(video_file):
    if video_file.endswith('.npy'):
        return load_video_frames_npy(video_file)
    elif video_file.endswith('.avi'):
        return load_video_frames_avi(video_file)

def load_video_frames_avi(video_file):
    import cv2
    print_memory_usage()
    
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    # Get video properties first
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {width}x{height}")
    
    # Check if ROI file already exists
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    roi_save_path = f"{video_name}_roi_data.npy"
    
    if os.path.exists(roi_save_path):
        cap.release()
        print(f"Loading existing ROI data from {roi_save_path}")
        return np.load(roi_save_path)
    
    # Pre-allocate numpy array instead of using list
    try:
        # Try to allocate the full array (this might fail if not enough memory)
        video_frames = np.zeros((total_frames, height, width, 3), dtype=np.uint8)
        print(f"Pre-allocated array: {video_frames.nbytes / (1024**2):.1f} MB")
    except MemoryError:
        cap.release()
        print("Not enough memory to load entire video. Processing in streaming mode...")
        return load_video_frames_avi_streaming(video_file)
    
    print_memory_usage()
    
    # Load frames into pre-allocated array
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        video_frames[frame_idx] = frame
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Loaded {frame_idx}/{total_frames} frames")
            print_memory_usage()

    cap.release()
    
    # Trim array if we read fewer frames than expected
    if frame_idx < total_frames:
        video_frames = video_frames[:frame_idx]
    
    print(f"Video frames loaded: {len(video_frames)} frames")
    print_memory_usage()
    
    # Now process for face detection and ROI extraction (similar to NPY version)
    face_detector = SCRFD(variant='500m')
    roi_selector = FullFaceSquare(target_size=(72,72), larger_box_coef=1.5)

    roi_data = []
    batch_size = 50
    num_frames = len(video_frames)
    
    for i in range(0, num_frames, batch_size):
        end_idx = min(i + batch_size, num_frames)
        print(f"Processing batch {i//batch_size + 1}/{(num_frames + batch_size - 1)//batch_size} (frames {i}-{end_idx-1})")
        
        # Process batch
        batch_frames = np.flip(video_frames[i:end_idx], axis=3)  # RGB to BGR flip
        
        for frame_idx, frame in enumerate(batch_frames):
            faces = face_detector.detect(frame)
            if faces:
                roi, _ = roi_selector.select(frame, faces[0])
                roi_data.append(roi)
            else:
                print(f"Warning: No face detected in frame {i + frame_idx}")
                
            if (i + frame_idx + 1) % 100 == 0:
                print(f"Processed {i + frame_idx + 1} frames")
                print_memory_usage()
        
        # Clear batch from memory
        del batch_frames
        gc.collect()

    # Clear the original video frames from memory before creating ROI array
    del video_frames
    gc.collect()
    print_memory_usage()

    roi_data = np.array(roi_data)
    print(f"ROI frames extracted: {len(roi_data)} frames")

    # Save ROI data
    np.save(roi_save_path, roi_data)
    print(f"ROI data saved to {roi_save_path}")

    return roi_data
      
def load_video_frames_npy(video_file):
    print_memory_usage()
    
    # Memory-mapped loading to avoid loading entire video into memory
    video_frames = np.load(video_file, mmap_mode='r')
    print(f"Video frames loaded (memory-mapped): {len(video_frames)} frames")
    
    # Check if ROI file already exists
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    roi_save_path = f"{video_name}_roi_data.npy"
    
    if os.path.exists(roi_save_path):
        print(f"Loading existing ROI data from {roi_save_path}")
        return np.load(roi_save_path)
    
    print_memory_usage()
    
    # face detection
    face_detector = SCRFD(variant='500m')
    # roi
    roi_selector = FullFaceSquare(target_size=(72,72), larger_box_coef=1.5)

    count = 0
    # Apply face detection and ROI selection with limited memory usage
    roi_data = []
    
    # Process in smaller batches to avoid memory issues
    batch_size = 50
    num_frames = len(video_frames)
    
    for i in range(0, num_frames, batch_size):
        end_idx = min(i + batch_size, num_frames)
        print(f"Processing batch {i//batch_size + 1}/{(num_frames + batch_size - 1)//batch_size} (frames {i}-{end_idx-1})")
        
        # Load and process batch
        batch_frames = np.array([np.flip(video_frames[j], axis=2) for j in range(i, end_idx)])
        
        for frame_idx, frame in enumerate(batch_frames):
            faces = face_detector.detect(frame)
            if faces:
                roi, _ = roi_selector.select(frame, faces[0])
                roi_data.append(roi)
            else:
                # If no face detected, create empty ROI or skip
                print(f"Warning: No face detected in frame {i + frame_idx}")
                # You might want to handle this case differently
                
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} frames")
                print_memory_usage()
        
        # Clear batch from memory
        del batch_frames
        
        # Force garbage collection to free memory
        gc.collect()

    roi_data = np.array(roi_data)
    print(f"ROI frames extracted: {len(roi_data)} frames")

    # save
    np.save(roi_save_path, roi_data)
    print(f"ROI data saved to {roi_save_path}")

    return roi_data

def load_from_saved(video_file):
    roi_data = np.load(video_file)

    return roi_data

import numpy as np

def save_ndarray_phone_friendly(ndarray, filename='ndarray_phone_friendly.txt'):
    """
    Save numpy ndarray in a phone-friendly format with clear structure
    """
    with open(filename, 'w') as f:
        # Header information
        f.write("NUMPY ARRAY DATA\n")
        f.write("=" * 30 + "\n")
        f.write(f"Shape: {ndarray.shape}\n")
        f.write(f"Data Type: {ndarray.dtype}\n")
        f.write(f"Total Elements: {ndarray.size}\n")
        f.write("=" * 30 + "\n\n")
        
        # Set print options
        np.set_printoptions(threshold=np.inf, linewidth=80, suppress=True)
        
        # Write each "frame" (first dimension) separately for clarity
        for i in range(ndarray.shape[0]):
            f.write(f"FRAME {i+1}/{ndarray.shape[0]}\n")
            f.write("-" * 20 + "\n")
            
            # Write the 72x72x3 data for this frame
            frame_string = np.array2string(ndarray[i], separator=', ')
            f.write(frame_string)
            f.write("\n\n")
    
    print(f"Phone-friendly array saved to {filename}")
    return filename

def test_rolling_statistics_comparison(video_file, roi_frames, step=30, window_size=180):
    """Compare all rolling statistics implementations."""
    print("=== Testing Rolling Statistics Implementations ===")
    
    # Initialize all implementations
    rolling_original = RollingStatistics(window_size=window_size, step=step, frame_shape=(72, 72, 3))
    
    n_frames = roi_frames.shape[0]
    # Limit the number of steps to prevent memory issues
    max_steps = min(20, n_frames // step)  # Reduced from 100 to 20
    print(f"Testing {max_steps} steps out of possible {n_frames // step}")
    
    # Create CSV file name based on video file name
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    csv_filename = f"{video_name}_rolling_comparison2_step{step}_windowsize{window_size}.csv"
    
    # Write comparison data to CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            'step_number', 'start_idx', 'end_idx', 
            'gt_mean', 'gt_std', 'gt_diffnorm_std',
            'rolling_mean', 'rolling_std', 'rolling_diffnorm_std',
            'diff_rmse'
        ])

        diff_size = window_size - step
        
        for i in range(max_steps):
            start_idx = i * step - diff_size
            if start_idx < 0:
                start_idx = 0
            end_idx = (i + 1) * step
            
            # Create a copy to avoid modifying the original data
            clip = roi_frames[start_idx:end_idx].copy()
            
            print(f"\n--- Step {i+1}/{max_steps} ---")
            print(f"Clip shape: {clip.shape}")
            
            try:
                # Ground truth calculation
                gt_mean = np.mean(clip)
                gt_std = np.std(clip)
                gt_diffnorm, gt_diffnorm_std = diff_normalize_std(clip)
                print(f"Ground Truth - Mean: {gt_mean:.6f}, Std: {gt_std:.6f}, DiffNorm Std: {gt_diffnorm_std:.6f}")

                gt_diffnorm = gt_diffnorm[-(step - 1):]

                # Add frames to rolling statistics
                for frame in clip:
                    rolling_original.add_frame(frame)
                
                # Get results from rolling implementation
                orig_mean = rolling_original.get_mean()
                orig_std = rolling_original.get_std()
                orig_diffnorm_std = rolling_original.get_diff_std()
                orig_diffnorm = rolling_original.get_diff_chunk()

                print(f"Rolling - Mean: {orig_mean:.6f}, Std: {orig_std:.6f}, DiffNorm Std: {orig_diffnorm_std:.6f}")

                # Calculate RMSE
                diff_diff = orig_diffnorm - gt_diffnorm
                diff_rmse = np.sqrt(np.mean(diff_diff ** 2))
                print(f"DiffNorm RMSE: {diff_rmse:.6f}")
                
                # Write data row
                writer.writerow([
                    i, start_idx, end_idx, 
                    gt_mean, gt_std, gt_diffnorm_std,
                    orig_mean, orig_std, orig_diffnorm_std,
                    diff_rmse
                ])
                
            except Exception as e:
                print(f"Error in step {i}: {e}")
                # Write error row
                writer.writerow([
                    i, start_idx, end_idx, 
                    'ERROR', 'ERROR', 'ERROR',
                    'ERROR', 'ERROR', 'ERROR',
                    'ERROR'
                ])
            
            # Clean up memory
            del clip
            import gc
            gc.collect()
    
    print(f"\nComparison data saved to {csv_filename}")

    

VIDEOS = [
    '/home/pme/ta/ta-app/UBFC-rPPG/raw/subject1/vid.avi',
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

def load_video_frames_avi_streaming(video_file):
    """Process AVI video in streaming mode without loading all frames into memory."""
    import cv2
    
    print("Processing video in streaming mode (memory-constrained)...")
    
    # Check if ROI file already exists
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    roi_save_path = f"{video_name}_roi_data.npy"
    
    if os.path.exists(roi_save_path):
        print(f"Loading existing ROI data from {roi_save_path}")
        return np.load(roi_save_path)
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames in streaming mode")
    
    # Initialize face detector and ROI selector
    face_detector = SCRFD(variant='500m')
    roi_selector = FullFaceSquare(target_size=(72,72), larger_box_coef=1.5)
    
    roi_data = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB then to BGR for face detection
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Face detection and ROI extraction
        faces = face_detector.detect(frame_bgr)
        if faces:
            roi, _ = roi_selector.select(frame_bgr, faces[0])
            roi_data.append(roi)
        else:
            print(f"Warning: No face detected in frame {frame_idx}")
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
            print_memory_usage()
            gc.collect()  # Force garbage collection every 100 frames
    
    cap.release()
    
    roi_data = np.array(roi_data)
    print(f"ROI frames extracted: {len(roi_data)} frames")
    
    # Save ROI data
    np.save(roi_save_path, roi_data)
    print(f"ROI data saved to {roi_save_path}")
    
    return roi_data

if __name__ == "__main__":
    import gc
    
    # Force garbage collection at start
    gc.collect()
    
    video_file = VIDEOS[0]  # Change this to test different videos
    
    # Check if ROI data already exists
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    roi_save_path = f"{video_name}_roi_data.npy"
    
    if os.path.exists(roi_save_path):
        print(f"Loading existing ROI data from {roi_save_path}")
        roi_frames = load_from_saved(roi_save_path)
    else:
        print("Processing video to extract ROI data...")
        roi_frames = load_video_frames(video_file)
    
    roi_frames = roi_frames.astype(np.float32)  # Ensure frames are in float32 format for processing
    
    print(f"ROI frames shape: {roi_frames.shape}")
    print(f"Memory usage estimate: {roi_frames.nbytes / (1024**2):.1f} MB")
    
    test_rolling_statistics_comparison(video_file, roi_frames, step=30, window_size=180)
