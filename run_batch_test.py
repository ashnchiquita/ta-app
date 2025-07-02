#!/usr/bin/env python3
"""
Batch testing script for running video processing on all test videos
and saving metrics to CSV file.
"""

import os
import csv
import time
from datetime import datetime
import json
from test_system.system import System
from components.rppg_signal_extractor.conventional.pos import POS
from components.rppg_signal_extractor.conventional.ica_2019 import ICA2019
from components.hr_extractor.fft import FFT
from components.roi_selector.fullface import FullFace
from components.face_detector.mt_cnn import MT_CNN

def make_json_csv_safe(data):
    """Convert data to JSON string that's safe for CSV storage"""
    if data is None:
        return None
    json_str = json.dumps(data, separators=(',', ':'))
    # Replace commas with semicolons to avoid CSV issues
    return json_str.replace(',', ';')


def restore_json_from_csv(csv_safe_str):
    """
    Convert CSV-safe JSON string back to original data
    
    Example usage:
    # When reading from CSV:
    heart_rates = restore_json_from_csv(row['heart_rates'])
    heart_rates_frame_idx = restore_json_from_csv(row['heart_rates_frame_idx'])
    """
    if csv_safe_str is None:
        return None
    # Replace semicolons back to commas
    json_str = csv_safe_str.replace(';', ',')
    return json.loads(json_str)

# Video paths from main_test.py
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

def get_a_system(video_path):
    rppg_system = System(
            video_file=video_path,
            
            face_detector=None, # scrfd 500m
            face_tracker=None, # centroid
            roi_selector=None, # full face square
            rppg_signal_extractor=None, # pure quantized
            hr_extractor=None, # fft
            
            window_size=180,
            fps=30,
            step_size=180
        )
    return rppg_system

def get_b_system(video_path):
    rppg_system = System(
            video_file=video_path,
            
            face_detector=None, # scrfd 500m
            face_tracker=None, # centroid
            roi_selector=None, # full face square
            rppg_signal_extractor=POS(fps=30),
            hr_extractor=None, # fft
            
            window_size=180,
            fps=30,
            step_size=180
        )
    return rppg_system

def get_baseline_system(video_path):
    rppg_system = System(
            video_file=video_path,
            
            face_detector=MT_CNN(), # mtcnn
            face_tracker=None, # centroid
            roi_selector=FullFace(), # full face
            rppg_signal_extractor=ICA2019(fps=30),
            hr_extractor=FFT(fps=30, diff_flag=False, use_bandpass=False, use_detrend=False), # fft

            window_size=180,
            fps=30,
            step_size=180
        )
    return rppg_system

def run_single_video_test(video_path, video_index):
    """
    Run test for a single video and return metrics data
    
    Args:
        video_path (str): Path to the video file
        video_index (int): Index of the video (0-26)
    
    Returns:
        dict: Metrics data for this video
    """
    print(f"\n{'='*60}")
    print(f"Processing Video {video_index}: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    try:
        # Create system with default configuration
        rppg_system = get_baseline_system(video_path)
        
        # Run the system
        start_time = time.time()
        heart_rates = rppg_system.run()
        end_time = time.time()
        
        # Extract metrics
        metrics = rppg_system.processing_metrics
        heart_rates = rppg_system.heart_rates
        heart_rates_frame_idx = rppg_system.heart_rates_frame_idx

        
        heart_rates_json = make_json_csv_safe(heart_rates)
        heart_rates_frame_idx_json = make_json_csv_safe(heart_rates_frame_idx)
        
        
        # Create metrics data dictionary
        metrics_data = {
            'video_index': video_index,
            'video_path': video_path,
            'processing_count': metrics.processing_count,
            'total_processing_time': metrics.total_processing_time,
            'avg_total_processing_time': metrics.avg_total_processing_time,
            'total_test_time': end_time - start_time,
            'face_detection_time': metrics.processing_time['face_detection'],
            'face_tracking_time': metrics.processing_time['face_tracking'],
            'roi_selection_time': metrics.processing_time['roi_selection'],
            'signal_extraction_time': metrics.processing_time['signal_extraction'],
            'signal_extraction_preprocess_time': metrics.processing_time['signal_extraction_preprocess'],
            'signal_extraction_inference_time': metrics.processing_time['signal_extraction_inference'],
            'hr_extraction_time': metrics.processing_time['hr_extraction'],
            'avg_face_detection_time': metrics.avg_processing_time['face_detection'],
            'avg_face_tracking_time': metrics.avg_processing_time['face_tracking'],
            'avg_roi_selection_time': metrics.avg_processing_time['roi_selection'],
            'avg_signal_extraction_time': metrics.avg_processing_time['signal_extraction'],
            'avg_signal_extraction_preprocess_time': metrics.avg_processing_time['signal_extraction_preprocess'],
            'avg_signal_extraction_inference_time': metrics.avg_processing_time['signal_extraction_inference'],
            'avg_hr_extraction_time': metrics.avg_processing_time['hr_extraction'],
            'detected_faces': len(heart_rates) if heart_rates else 0,
            'total_processed_faces': metrics.total_processed_faces,
            'success': True,
            'heart_rates': heart_rates_json,
            'heart_rates_frame_idx': heart_rates_frame_idx_json,
            'error_message': None,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Video {video_index} completed successfully")
        print(f"   Processing count: {metrics.processing_count}")
        print(f"   Total time: {end_time - start_time:.2f}s")
        print(f"   Detected faces: {len(heart_rates) if heart_rates else 0}")
        
        return metrics_data
        
    except Exception as e:
        print(f"❌ Error processing video {video_index}: {str(e)}")
        
        # Return error metrics data
        return {
            'video_index': video_index,
            'video_path': video_path,
            'processing_count': 0,
            'total_processing_time': 0,
            'avg_total_processing_time': 0,
            'total_test_time': 0,
            'face_detection_time': 0,
            'face_tracking_time': 0,
            'roi_selection_time': 0,
            'signal_extraction_time': 0,
            'signal_extraction_preprocess_time': 0,
            'signal_extraction_inference_time': 0,
            'hr_extraction_time': 0,
            'avg_face_detection_time': 0,
            'avg_face_tracking_time': 0,
            'avg_roi_selection_time': 0,
            'avg_signal_extraction_time': 0,
            'avg_signal_extraction_preprocess_time': 0,
            'avg_signal_extraction_inference_time': 0,
            'avg_hr_extraction_time': 0,
            'detected_faces': 0,
            'total_processed_faces': 0,
            'success': False,
            'heart_rates': None,
            'heart_rates_frame_idx': None,
            'error_message': str(e),
            'timestamp': datetime.now().isoformat()
        }


def save_metrics_to_csv(all_metrics, output_file):
    """
    Save all metrics data to CSV file
    
    Args:
        all_metrics (list): List of metrics dictionaries
        output_file (str): Path to output CSV file
    """
    if not all_metrics:
        print("No metrics data to save")
        return
    
    # Define CSV headers based on the metrics data structure
    headers = [
        'video_index',
        'video_path',
        'processing_count',
        'total_processing_time',
        'avg_total_processing_time',
        'total_test_time',
        'face_detection_time',
        'face_tracking_time', 
        'roi_selection_time',
        'signal_extraction_time',
        'signal_extraction_preprocess_time',
        'signal_extraction_inference_time',
        'hr_extraction_time',
        'avg_face_detection_time',
        'avg_face_tracking_time',
        'avg_roi_selection_time',
        'avg_signal_extraction_time',
        'avg_signal_extraction_preprocess_time',
        'avg_signal_extraction_inference_time',
        'avg_hr_extraction_time',
        'detected_faces',
        'total_processed_faces',
        'success',
        'heart_rates',
        'heart_rates_frame_idx',
        'error_message',
        'timestamp'
    ]
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_metrics)
        
        print(f"\n✅ Metrics saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving CSV file: {str(e)}")


def run_batch_tests(start_index=0, end_index=None, output_file=None):
    """
    Run batch tests on all videos
    
    Args:
        start_index (int): Starting video index (default: 0)
        end_index (int): Ending video index (default: None, processes all)
        output_file (str): Output CSV file path (default: auto-generated)
    """
    if end_index is None:
        end_index = len(VIDEOS) - 1
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"video_metrics_{timestamp}.csv"
    
    print(f"🚀 Starting batch testing")
    print(f"   Videos to process: {start_index} to {end_index} ({end_index - start_index + 1} videos)")
    print(f"   Output file: {output_file}")
    
    all_metrics = []
    successful_tests = 0
    failed_tests = 0
    
    batch_start_time = time.time()
    
    for i in range(start_index, end_index + 1):
        if i >= len(VIDEOS):
            print(f"⚠️  Video index {i} is out of range (max: {len(VIDEOS) - 1})")
            continue
            
        video_path = VIDEOS[i]
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path}")
            # Create error entry
            metrics_data = {
                'video_index': i,
                'video_path': video_path,
                'processing_count': 0,
                'total_processing_time': 0,
                'avg_total_processing_time': 0,
                'total_test_time': 0,
                'face_detection_time': 0,
                'face_tracking_time': 0,
                'roi_selection_time': 0,
                'signal_extraction_time': 0,
                'signal_extraction_preprocess_time': 0,
                'signal_extraction_inference_time': 0,
                'hr_extraction_time': 0,
                'avg_face_detection_time': 0,
                'avg_face_tracking_time': 0,
                'avg_roi_selection_time': 0,
                'avg_signal_extraction_time': 0,
                'avg_signal_extraction_preprocess_time': 0,
                'avg_signal_extraction_inference_time': 0,
                'avg_hr_extraction_time': 0,
                'detected_faces': 0,
                'total_processed_faces': 0,
                'success': False,
                'heart_rates': None,
                'heart_rates_frame_idx': None,
                'error_message': 'Video file not found',
                'timestamp': datetime.now().isoformat()
            }
            all_metrics.append(metrics_data)
            failed_tests += 1
            continue
        
        # Run test for this video
        metrics_data = run_single_video_test(video_path, i)
        all_metrics.append(metrics_data)
        
        if metrics_data['success']:
            successful_tests += 1
        else:
            failed_tests += 1

        # Save intermediate results every 6 videos
        if (i - start_index + 1) % 6 == 0:
            intermediate_file = f"intermediate_{output_file}"
            save_metrics_to_csv(all_metrics, intermediate_file)
            print(f"💾 Intermediate results saved to {intermediate_file}")
    
    batch_end_time = time.time()
    total_batch_time = batch_end_time - batch_start_time
    
    # Save final results
    save_metrics_to_csv(all_metrics, output_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos processed: {successful_tests + failed_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"Total batch time: {total_batch_time:.2f} seconds")
    print(f"Average time per video: {total_batch_time / (successful_tests + failed_tests):.2f} seconds")
    print(f"Results saved to: {output_file}")
    
    if failed_tests > 0:
        print(f"\n⚠️  {failed_tests} tests failed. Check the CSV file for error details.")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    start_idx = 0
    end_idx = None
    output_file = None
    
    if len(sys.argv) > 1:
        start_idx = int(sys.argv[1])
    if len(sys.argv) > 2:
        end_idx = int(sys.argv[2])
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    # Run batch tests
    run_batch_tests(start_idx, end_idx, output_file)
