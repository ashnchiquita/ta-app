#!/usr/bin/env python3
"""
Script to calculate average FPS from all fps.csv files in the output directory
and save the results to a summary CSV file.
"""

import os
import pandas as pd
import csv
from pathlib import Path
from typing import List, Tuple


def find_fps_files(output_dir: str) -> List[Tuple[str, str]]:
    """
    Find all fps.csv files in the output directory.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        List of tuples containing (folder_name, fps_file_path)
    """
    fps_files = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Output directory {output_dir} does not exist!")
        return fps_files
    
    for subfolder in output_path.iterdir():
        if subfolder.is_dir():
            fps_file = subfolder / "fps.csv"
            if fps_file.exists():
                fps_files.append((subfolder.name, str(fps_file)))
            else:
                print(f"Warning: fps.csv not found in {subfolder.name}")
    
    return fps_files


def calculate_average_fps(fps_file_path: str) -> float:
    """
    Calculate the average FPS from a fps.csv file.
    
    Args:
        fps_file_path: Path to the fps.csv file
        
    Returns:
        Average FPS value
    """
    try:
        df = pd.read_csv(fps_file_path)
        
        if 'fps' not in df.columns:
            print(f"Warning: 'fps' column not found in {fps_file_path}")
            return 0.0
        
        # Remove any NaN values and calculate average
        fps_values = df['fps'].dropna()
        
        if len(fps_values) == 0:
            print(f"Warning: No valid FPS values found in {fps_file_path}")
            return 0.0
        
        avg_fps = fps_values.mean()
        return avg_fps
        
    except Exception as e:
        print(f"Error processing {fps_file_path}: {e}")
        return 0.0


def save_results_to_csv(results: List[Tuple[str, float]], output_file: str):
    """
    Save the average FPS results to a CSV file.
    
    Args:
        results: List of tuples containing (folder_name, average_fps)
        output_file: Path to the output CSV file
    """
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['folder', 'average_fps'])
            
            # Sort results by folder name (numerically if possible)
            try:
                sorted_results = sorted(results, key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
            except ValueError:
                sorted_results = sorted(results, key=lambda x: x[0])
            
            for folder, avg_fps in sorted_results:
                writer.writerow([folder, avg_fps])
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")


def main():
    """Main function to orchestrate the FPS calculation process."""
    # Configuration
    output_dir = "output"
    result_file = "average_fps_summary.csv"
    
    print("Starting FPS calculation process...")
    print(f"Scanning directory: {output_dir}")
    
    # Find all fps.csv files
    fps_files = find_fps_files(output_dir)
    
    if not fps_files:
        print("No fps.csv files found!")
        return
    
    print(f"Found {len(fps_files)} fps.csv files")
    
    # Calculate average FPS for each file
    results = []
    for folder_name, fps_file_path in fps_files:
        print(f"Processing {folder_name}...")
        avg_fps = calculate_average_fps(fps_file_path)
        results.append((folder_name, avg_fps))
        print(f"  Average FPS: {avg_fps:.2f}")
    
    # Save results to CSV
    save_results_to_csv(results, result_file)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if results:
        total_avg = sum(result[1] for result in results) / len(results)
        max_fps = max(results, key=lambda x: x[1])
        min_fps = min(results, key=lambda x: x[1])
        
        print(f"Total files processed: {len(results)}")
        print(f"Overall average FPS: {total_avg:.2f}")
        print(f"Highest average FPS: {max_fps[1]:.2f} (folder: {max_fps[0]})")
        print(f"Lowest average FPS: {min_fps[1]:.2f} (folder: {min_fps[0]})")
    
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
