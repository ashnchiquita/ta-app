from system.system import System
import time
import sys
import argparse
from videos import MY_VIDEOS, UBFC_RPPG_VIDEOS, is_valid_my_video_index, is_valid_ubfc_rppg_subject_index, ubfc_exceptions, get_my_videos_log_dir, get_ubfc_rppg_log_dir

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


    
def run_default_system(video_file, log_dir):
    """Run system with NPY video file by index.
    
    Args:
        video_index (int): Index of video file to use (0-26)
    """

    rppg_system = None
    try:
        rppg_system = System(
            video_file=video_file,
            log_dir=log_dir,
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
                       help=f'Video index to use. For my-videos (0-{len(MY_VIDEOS)-1}). For ubfc-rppg ({UBFC_RPPG_VIDEOS.keys()[0]}-{UBFC_RPPG_VIDEOS.keys()[-1]} except {ubfc_exceptions}). Default: 0')
    parser.add_argument('--mode', '-m', choices=['camera', 'my-videos', 'ubfc-rppg'], 
                       default='my-videos', help='Run mode: camera, my-videos, or ubfc-rppg')
    parser.add_argument('--list-videos', '-l', action='store_true', 
                       help='List all available videos and exit')
    
    args = parser.parse_args()
    
    if args.list_videos:
        print("Available MY VIDEOS:")
        for i, video in enumerate(MY_VIDEOS):
            print(f"  {i:2d}: {video}")
        print("\nAvailable UBFC-rPPG Videos:")
        for subject, videos in UBFC_RPPG_VIDEOS.items():
            print(f"  {subject:2d}: {videos}")
        return
    
    print(f"Running in '{args.mode}' mode with video index {args.video_index}")
    
    try:
        if args.mode == 'camera':
            run_default_system()
        elif args.mode == 'my-videos':
            if not is_valid_my_video_index(args.video_index):
                raise ValueError(f"Invalid video index {args.video_index} for my-videos")
            video_file = MY_VIDEOS[args.video_index]
            log_dir = get_my_videos_log_dir(args.video_index)
            run_default_system(video_file, log_dir)
        elif args.mode == 'ubfc-rppg':
            if not is_valid_ubfc_rppg_subject_index(args.video_index):
                raise ValueError(f"Invalid subject index {args.video_index} for ubfc-rppg")
            video_file = UBFC_RPPG_VIDEOS[args.video_index]
            log_dir = get_ubfc_rppg_log_dir(args.video_index)
            run_default_system(video_file, log_dir)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
