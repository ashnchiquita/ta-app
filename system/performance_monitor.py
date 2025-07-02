"""
Performance monitor for rPPG processing pipeline.
Tracks various metrics to analyze system performance.
"""

import time
import threading
from collections import deque, defaultdict
from typing import Dict, List, Optional
import statistics


class PerformanceMonitor:
    """Monitor and track performance metrics for the rPPG pipeline."""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.lock = threading.Lock()
        
        # Timing metrics
        self.frame_times = deque(maxlen=history_size)
        self.processing_times = deque(maxlen=history_size)
        self.inference_times = deque(maxlen=history_size)
        self.hr_extraction_times = deque(maxlen=history_size)
        
        # Pipeline metrics
        self.faces_processed = deque(maxlen=history_size)
        self.chunk_processing_times = defaultdict(lambda: deque(maxlen=history_size))
        self.hr_updates = deque(maxlen=history_size)
        
        # System metrics
        self.memory_usage = deque(maxlen=history_size)
        self.cpu_usage = deque(maxlen=history_size)
        
        # Start time
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
    def record_frame(self):
        """Record a new frame timing."""
        current_time = time.time()
        with self.lock:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            self.last_frame_time = current_time
    
    def record_processing_time(self, processing_time: float):
        """Record processing time for a frame."""
        with self.lock:
            self.processing_times.append(processing_time)
    
    def record_inference_time(self, inference_time: float):
        """Record inference time."""
        with self.lock:
            self.inference_times.append(inference_time)
    
    def record_hr_extraction_time(self, hr_time: float):
        """Record heart rate extraction time."""
        with self.lock:
            self.hr_extraction_times.append(hr_time)
    
    def record_faces_processed(self, num_faces: int):
        """Record number of faces processed."""
        with self.lock:
            self.faces_processed.append(num_faces)
    
    def record_chunk_processing(self, face_id: int, processing_time: float):
        """Record chunk processing time for a specific face."""
        with self.lock:
            self.chunk_processing_times[face_id].append(processing_time)
    
    def record_hr_update(self, face_id: int):
        """Record heart rate update."""
        with self.lock:
            self.hr_updates.append((time.time(), face_id))
    
    def get_fps(self) -> float:
        """Calculate current FPS."""
        with self.lock:
            if len(self.frame_times) < 2:
                return 0.0
            return 1.0 / statistics.mean(list(self.frame_times)[-10:])
    
    def get_average_processing_time(self) -> float:
        """Get average processing time per frame."""
        with self.lock:
            if not self.processing_times:
                return 0.0
            return statistics.mean(self.processing_times)
    
    def get_average_inference_time(self) -> float:
        """Get average inference time."""
        with self.lock:
            if not self.inference_times:
                return 0.0
            return statistics.mean(self.inference_times)
    
    def get_hr_update_rate(self) -> float:
        """Get heart rate update frequency."""
        with self.lock:
            if len(self.hr_updates) < 2:
                return 0.0
            
            recent_updates = [t for t, _ in self.hr_updates 
                            if time.time() - t < 10.0]
            
            if len(recent_updates) < 2:
                return 0.0
            
            return len(recent_updates) / 10.0
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        with self.lock:
            summary = {
                'fps': self.get_fps(),
                'avg_processing_time': self.get_average_processing_time(),
                'avg_inference_time': self.get_average_inference_time(),
                'hr_update_rate': self.get_hr_update_rate(),
                'uptime': time.time() - self.start_time,
                'total_frames': len(self.frame_times)
            }
            
            if self.faces_processed:
                summary['avg_faces_per_frame'] = statistics.mean(self.faces_processed)
            
            if self.processing_times:
                summary['processing_time_std'] = statistics.stdev(self.processing_times) if len(self.processing_times) > 1 else 0
            
            return summary
    
    def print_performance_report(self):
        """Print detailed performance report."""
        summary = self.get_performance_summary()
        
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        print(f"FPS: {summary['fps']:.2f}")
        print(f"Average Processing Time: {summary['avg_processing_time']*1000:.2f} ms")
        print(f"Average Inference Time: {summary['avg_inference_time']*1000:.2f} ms")
        print(f"Heart Rate Update Rate: {summary['hr_update_rate']:.2f} Hz")
        print(f"Uptime: {summary['uptime']:.1f} seconds")
        print(f"Total Frames Processed: {summary['total_frames']}")
        
        if 'avg_faces_per_frame' in summary:
            print(f"Average Faces per Frame: {summary['avg_faces_per_frame']:.2f}")
        
        if 'processing_time_std' in summary:
            print(f"Processing Time Std Dev: {summary['processing_time_std']*1000:.2f} ms")
        
        print("="*50)


class PerformanceOptimizer:
    """Adaptive performance optimization for the rPPG pipeline."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.target_fps = 30.0
        self.adjustment_history = deque(maxlen=10)
        
    def should_adjust_chunk_size(self) -> Optional[int]:
        """Determine if chunk size should be adjusted based on performance."""
        current_fps = self.monitor.get_fps()
        
        if current_fps < self.target_fps * 0.8:  # FPS too low
            # Suggest larger chunk size for better throughput
            return min(60, int(self.target_fps * 2))
        elif current_fps > self.target_fps * 1.2:  # FPS higher than needed
            # Suggest smaller chunk size for lower latency
            return max(15, int(self.target_fps * 0.5))
        
        return None
    
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count based on current workload."""
        import os
        cpu_count = os.cpu_count() or 4
        avg_faces = self.monitor.get_performance_summary().get('avg_faces_per_frame', 1)
        
        if avg_faces <= 1:
            return 1
        elif avg_faces <= 3:
            return min(2, cpu_count // 2)
        else:
            return min(4, cpu_count - 1)
