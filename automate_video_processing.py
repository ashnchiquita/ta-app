#!/usr/bin/env python3
"""
Automation script for running video processing with retry logic.

This script runs the main.py script for a list of video indices with the following features:
1. Runs python main.py -m my-videos -v {video_index} for each index in the list
2. Waits 1 minute between runs by default
3. After each run, checks processing-metrics.csv for df['Value'][6]
4. If df['Value'][6] is not zero, waits 5 minutes and retries the same video
5. Continues to next video only when df['Value'][6] becomes zero

Usage:
    python automate_video_processing.py --video-list 0 1 2 3 4
    python automate_video_processing.py --video-list 0 1 2 --max-retries 5
"""

import argparse
import subprocess
import time
import pandas as pd
import os
import sys
from typing import List
from videos import get_my_videos_log_dir, MY_VIDEOS

MODE = 'ubfc-rppg-custom'
VERSION = 'b'
RESOLUTION_FACTOR = 0.1
PREFIX_PATH = f'/home/pme/ta/ta-app/logs/{MODE}_r{RESOLUTION_FACTOR}'
if VERSION:
    PREFIX_PATH += f"/{VERSION}"

class VideoProcessingAutomator:
    def __init__(self, video_indices: List[int], max_retries: int = 3):
        """
        Initialize the automator.
        
        Args:
            video_indices: List of video indices to process
            max_retries: Maximum number of retries per video (default: 3)
        """
        self.video_indices = video_indices
        self.max_retries = max_retries
        self.success_count = 0
        self.failed_videos = []
        
    def run_video_processing(self, video_index: int) -> bool:
        """
        Run the main.py script for a specific video index.
        
        Args:
            video_index: The video index to process
            
        Returns:
            bool: True if the command executed successfully, False otherwise
        """
        try:
            cmd = ["python", "main.py", "-m", MODE, "-v", str(video_index), "-r", str(RESOLUTION_FACTOR)]
            if VERSION:
                cmd += ["-V", VERSION]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes timeout
            
            if result.returncode == 0:
                print(f"✓ Video {video_index} processing completed successfully")
                return True
            else:
                print(f"✗ Video {video_index} processing failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ Video {video_index} processing timed out (5 minutes)")
            return False
        except Exception as e:
            print(f"✗ Error running video {video_index}: {e}")
            return False
    
    def check_processing_metrics(self, video_index: int) -> bool:
        """
        Check if processing-metrics.csv shows df['Value'][6] == 0.
        
        Args:
            video_index: The video index to check
            
        Returns:
            bool: True if df['Value'][6] == 0, False otherwise
        """
        try:
            log_dir = os.path.join(PREFIX_PATH, f"{video_index:01d}")
            metrics_file = os.path.join(log_dir, "processing_metrics.csv")
            
            if not os.path.exists(metrics_file):
                print(f"⚠ Metrics file not found: {metrics_file}")
                return False
            
            # Read the CSV file
            df = pd.read_csv(metrics_file)
            
            # Check if we have at least 7 rows (index 6)
            if len(df) < 7:
                print(f"⚠ Metrics file has insufficient rows (need at least 7, got {len(df)})")
                return False
            
            # Check if Value column exists
            if 'Value' not in df.columns:
                print(f"⚠ 'Value' column not found in metrics file")
                return False
            
            value_6 = df['Value'][6]
            print(f"📊 Video {video_index}: df['Value'][6] = {value_6}")
            
            return value_6 == 0
            
        except Exception as e:
            print(f"✗ Error checking metrics for video {video_index}: {e}")
            return False
    
    def wait_with_countdown(self, minutes: int = 0, reason: str = "", seconds: int = 0):
        """
        Wait for specified time with a countdown display.
        
        Args:
            minutes: Number of minutes to wait (default: 0)
            reason: Reason for waiting (for display)
            seconds: Number of seconds to wait (default: 0)
        """
        total_seconds = minutes * 60 + seconds
        
        if minutes > 0 and seconds > 0:
            time_str = f"{minutes} minute(s) and {seconds} second(s)"
        elif minutes > 0:
            time_str = f"{minutes} minute(s)"
        else:
            time_str = f"{seconds} second(s)"
            
        print(f"⏳ Waiting {time_str} {reason}...")
        
        for remaining in range(total_seconds, 0, -1):
            mins, secs = divmod(remaining, 60)
            timer = f"{mins:02d}:{secs:02d}"
            print(f"\r⏳ Time remaining: {timer}", end="", flush=True)
            time.sleep(1)
        
        print("\r✓ Wait completed!              ")
    
    def process_video_with_retry(self, video_index: int) -> bool:
        """
        Process a single video with retry logic.
        
        Args:
            video_index: The video index to process
            
        Returns:
            bool: True if successful (df['Value'][6] == 0), False if failed after max retries
        """
        print(f"\n{'='*60}")
        print(f"🎬 Processing Video {video_index}")
        print(f"{'='*60}")
        
        # # Validate video index
        # if video_index < 0 or video_index >= len(MY_VIDEOS):
        #     print(f"✗ Invalid video index {video_index} (valid range: 0-{len(MY_VIDEOS)-1})")
        #     return False
        
        # if not os.path.exists(MY_VIDEOS[video_index]):
        #     print(f"✗ Video file not found: {MY_VIDEOS[video_index]}")
        #     return False
        
        retry_count = 0
        
        while retry_count <= self.max_retries:
            if retry_count > 0:
                print(f"\n🔄 Retry attempt {retry_count}/{self.max_retries} for video {video_index}")
            
            # Run the video processing
            success = self.run_video_processing(video_index)
            
            if not success:
                print(f"✗ Video processing failed for video {video_index}")
                if retry_count < self.max_retries:
                    print("⏳ Waiting 2 minutes before retry...")
                    self.wait_with_countdown(2, "before retry")
                    retry_count += 1
                    continue
                else:
                    print(f"✗ Max retries reached for video {video_index}")
                    return False
            
            # Check processing metrics
            if self.check_processing_metrics(video_index):
                print(f"✓ Video {video_index} completed successfully (df['Value'][6] == 0)")
                return True
            else:
                print(f"⚠ Video {video_index} needs retry (df['Value'][6] != 0)")
                if retry_count < self.max_retries:
                    self.wait_with_countdown(5, "before retry due to non-zero value")
                    retry_count += 1
                else:
                    print(f"✗ Max retries reached for video {video_index}")
                    return False
        
        return False
    
    def run_automation(self):
        """
        Run the complete automation process for all video indices.
        """
        print(f"🚀 Starting video processing automation")
        print(f"📋 Video indices to process: {self.video_indices}")
        print(f"🔄 Max retries per video: {self.max_retries}")
        print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        for i, video_index in enumerate(self.video_indices):
            print(f"\n📈 Progress: {i+1}/{len(self.video_indices)} videos")
            
            success = self.process_video_with_retry(video_index)
            
            if success:
                self.success_count += 1
            else:
                self.failed_videos.append(video_index)
            
            # Wait 30 seconds before next video (except for the last one)
            if i < len(self.video_indices) - 1:
                self.wait_with_countdown(0, "before next video", 30)
        
        # Print final summary
        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{'='*60}")
        print(f"📊 AUTOMATION SUMMARY")
        print(f"{'='*60}")
        print(f"⏰ Total duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"✓ Successful videos: {self.success_count}/{len(self.video_indices)}")
        print(f"✗ Failed videos: {len(self.failed_videos)}")
        
        if self.failed_videos:
            print(f"❌ Failed video indices: {self.failed_videos}")
        else:
            print("🎉 All videos processed successfully!")
        
        print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Automate video processing with retry logic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automate_video_processing.py --video-list 0 1 2 3 4
  python automate_video_processing.py --video-list 0 1 2 --max-retries 5
  python automate_video_processing.py --video-list 10 11 12 13 14 15
        """
    )
    
    parser.add_argument(
        '--video-list', '-v',
        type=int,
        nargs='+',
        required=True,
        help='List of video indices to process (e.g., 0 1 2 3 4)'
    )
    
    parser.add_argument(
        '--max-retries', '-r',
        type=int,
        default=3,
        help='Maximum number of retries per video (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Validate video indices
    # invalid_indices = [idx for idx in args.video_list if idx < 0 or idx >= len(MY_VIDEOS)]
    # if invalid_indices:
    #     print(f"❌ Error: Invalid video indices: {invalid_indices}")
    #     print(f"Valid range: 0-{len(MY_VIDEOS)-1}")
    #     sys.exit(1)
    
    # Remove duplicates and sort
    video_indices = sorted(list(set(args.video_list)))
    
    try:
        automator = VideoProcessingAutomator(video_indices, args.max_retries)
        automator.run_automation()
        
        # Exit with error code if any videos failed
        if automator.failed_videos:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠ Automation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
