from system.system import System
import time
import sys
import argparse
from components.face_detector.haar_cascade import HaarCascade
from components.face_tracker.centroid import Centroid
from components.rppg_signal_extractor.conventional.chrom import CHROM
from components.hr_extractor.fft import FFT

def run_default_system():
    rppg_system = None
    try:
        rppg_system = System(camera_id=0)
        rppg_system.start()
        
        while rppg_system.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rppg_system is not None:
            rppg_system.stop()
        
        # Force cleanup of hardware resources
        _cleanup_hardware_resources()

def _cleanup_hardware_resources():
    """Force cleanup of all hardware resources."""
    try:
        from components.manager.hailo_target_manager import HailoTargetManager
        
        # Get the singleton instance and force release
        if HailoTargetManager._instance is not None:
            HailoTargetManager._instance.release()
            HailoTargetManager._instance = None
            HailoTargetManager._target = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Brief pause to allow hardware cleanup
        time.sleep(2)
        
        print("Hardware resources cleaned up")
        
    except Exception as e:
        print(f"Warning: Error during hardware cleanup: {e}")

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
def run_default_system_npy(video_index=0):
    """Run system with NPY video file by index.
    
    Args:
        video_index (int): Index of video file to use (0-26)
    """
    if video_index < 0 or video_index >= len(VIDEOS):
        raise ValueError(f"Video index must be between 0 and {len(VIDEOS)-1}, got {video_index}")
    
    rppg_system = None
    try:
        selected_video = VIDEOS[video_index]
        print(f"Running with video {video_index}: {selected_video}")
        
        rppg_system = System(
            video_file=selected_video,
            # video_file='C:\\Users\\dyogggeming\\TA\\tes\\rppg-data\\fix\\camera\\29\\video_29_20250503_171818.npy',
        )
        rppg_system.start()
        
        while rppg_system.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rppg_system is not None:
            rppg_system.stop()
        
        # Force cleanup of hardware resources
        _cleanup_hardware_resources()

def run_default_system_npy_timestamp(video_index=0):
    """Run system with NPY video file and timestamp file by index.
    
    Args:
        video_index (int): Index of video file to use (0-26)
    """
    if video_index < 0 or video_index >= len(VIDEOS):
        raise ValueError(f"Video index must be between 0 and {len(VIDEOS)-1}, got {video_index}")
    
    rppg_system = None
    try:
        selected_video = VIDEOS[video_index]
        # Construct timestamp file path based on video path
        video_dir = selected_video.replace('.npy', '').replace('video_', '')
        camera_num = f"{video_index:02d}"
        timestamp_file = f'/home/pme/ta/data/camera/{camera_num}/timestamps_{camera_num}_20250503_152120.csv'
        
        print(f"Running with video {video_index}: {selected_video}")
        print(f"Using timestamp file: {timestamp_file}")
        
        rppg_system = System(
            video_file=selected_video,
            timestamp_file=timestamp_file,
            # video_file='C:\\Users\\dyogggeming\\TA\\tes\\rppg-data\\fix\\camera\\29\\video_29_20250503_171818.npy',
            # timestamps='C:\\Users\\dyogggeming\\TA\\tes\\rppg-data\\fix\\camera\\29\\timestamps_29_20250503_171818.csv'
        )
        rppg_system.start()
        
        while rppg_system.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rppg_system is not None:
            rppg_system.stop()
        
        # Force cleanup of hardware resources
        _cleanup_hardware_resources()
            
def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run rPPG system with video files')
    parser.add_argument('--video-index', '-v', type=int, default=0, 
                       help=f'Video index to use (0-{len(VIDEOS)-1}), default: 0')
    parser.add_argument('--mode', '-m', choices=['camera', 'npy', 'npy-timestamp'], 
                       default='npy', help='Run mode: camera, npy, or npy-timestamp')
    parser.add_argument('--list-videos', '-l', action='store_true', 
                       help='List all available videos and exit')
    
    args = parser.parse_args()
    
    if args.list_videos:
        print("Available videos:")
        for i, video in enumerate(VIDEOS):
            print(f"  {i:2d}: {video}")
        return
    
    if args.video_index < 0 or args.video_index >= len(VIDEOS):
        print(f"Error: Video index must be between 0 and {len(VIDEOS)-1}")
        sys.exit(1)
    
    print(f"Running in '{args.mode}' mode with video index {args.video_index}")
    
    try:
        if args.mode == 'camera':
            run_default_system()
        elif args.mode == 'npy':
            run_default_system_npy(args.video_index)
        elif args.mode == 'npy-timestamp':
            run_default_system_npy_timestamp(args.video_index)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    # Legacy calls for backward compatibility:
    # run_default_system()
    # run_default_system_npy()
    # run_default_system_npy_timestamp()
