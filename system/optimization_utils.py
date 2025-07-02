"""
Memory and resource optimization utilities for rPPG processing.
"""

import gc
import threading
import time
import numpy as np
from typing import Dict, Any, Optional
import psutil
import os


class MemoryManager:
    """Manages memory usage and garbage collection for optimal performance."""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.last_gc_time = time.time()
        self.gc_interval = 5.0  # Force GC every 5 seconds if needed
        self.memory_alerts = 0
        
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        return {
            'memory_mb': memory_info.rss / (1024 * 1024),
            'memory_percent': memory_percent,
            'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def should_force_gc(self) -> bool:
        """Determine if garbage collection should be forced."""
        current_time = time.time()
        memory_info = self.check_memory_usage()
        
        # Force GC if memory usage is high or enough time has passed
        if (memory_info['memory_percent'] > self.max_memory_percent or 
            current_time - self.last_gc_time > self.gc_interval):
            return True
        
        return False
    
    def force_garbage_collection(self):
        """Force garbage collection and update timing."""
        gc.collect()
        self.last_gc_time = time.time()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization and return statistics."""
        before_memory = self.check_memory_usage()
        
        if self.should_force_gc():
            self.force_garbage_collection()
        
        after_memory = self.check_memory_usage()
        
        # Alert if memory usage is still high
        if after_memory['memory_percent'] > self.max_memory_percent:
            self.memory_alerts += 1
            if self.memory_alerts % 10 == 0:  # Alert every 10 occurrences
                print(f"WARNING: High memory usage {after_memory['memory_percent']:.1f}%")
        
        return {
            'before_memory_mb': before_memory['memory_mb'],
            'after_memory_mb': after_memory['memory_mb'],
            'freed_memory_mb': before_memory['memory_mb'] - after_memory['memory_mb'],
            'memory_percent': after_memory['memory_percent']
        }


class ResourcePool:
    """Pool of reusable resources to reduce allocation overhead."""
    
    def __init__(self):
        self.array_pool = {}
        self.lock = threading.Lock()
        self.max_pool_size = 50  # Maximum number of arrays per shape
        
    def get_array(self, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Get a reusable array from the pool or create new one."""
        key = (shape, dtype)
        
        with self.lock:
            if key in self.array_pool and self.array_pool[key]:
                array = self.array_pool[key].pop()
                array.fill(0)  # Reset array content
                return array
        
        # Create new array if none available
        return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray):
        """Return an array to the pool for reuse."""
        key = (array.shape, array.dtype)
        
        with self.lock:
            if key not in self.array_pool:
                self.array_pool[key] = []
            
            # Only keep arrays if pool isn't full
            if len(self.array_pool[key]) < self.max_pool_size:
                self.array_pool[key].append(array)
    
    def clear_pool(self):
        """Clear the resource pool."""
        with self.lock:
            self.array_pool.clear()
    
    def get_pool_stats(self) -> Dict:
        """Get statistics about the resource pool."""
        with self.lock:
            stats = {
                'total_shapes': len(self.array_pool),
                'total_arrays': sum(len(arrays) for arrays in self.array_pool.values()),
                'shapes': list(self.array_pool.keys())
            }
        return stats


class AdaptiveOptimizer:
    """Adaptive optimization that adjusts parameters based on system performance."""
    
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.performance_history = []
        self.adjustment_cooldown = 10.0  # Seconds between adjustments
        self.last_adjustment = 0
        
    def should_adjust_parameters(self, current_fps: float, 
                                cpu_usage: float, memory_usage: float) -> Dict[str, Any]:
        """Determine if and how to adjust parameters."""
        current_time = time.time()
        
        # Don't adjust too frequently
        if current_time - self.last_adjustment < self.adjustment_cooldown:
            return {}
        
        adjustments = {}
        
        # FPS-based adjustments
        if current_fps < self.target_fps * 0.8:  # Too slow
            if memory_usage > 70:
                adjustments['reduce_chunk_size'] = True
                adjustments['force_gc'] = True
            if cpu_usage > 80:
                adjustments['reduce_thread_count'] = True
                
        elif current_fps > self.target_fps * 1.2:  # Too fast
            adjustments['increase_chunk_size'] = True
            
        # Resource-based adjustments
        if memory_usage > 85:
            adjustments['aggressive_gc'] = True
            adjustments['reduce_cache_size'] = True
            
        if cpu_usage > 90:
            adjustments['reduce_parallelism'] = True
        
        if adjustments:
            self.last_adjustment = current_time
            
        return adjustments


# Global instances for reuse
_MEMORY_MANAGER = MemoryManager()
_RESOURCE_POOL = ResourcePool()
_ADAPTIVE_OPTIMIZER = AdaptiveOptimizer()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    return _MEMORY_MANAGER


def get_resource_pool() -> ResourcePool:
    """Get the global resource pool instance."""
    return _RESOURCE_POOL


def get_adaptive_optimizer() -> AdaptiveOptimizer:
    """Get the global adaptive optimizer instance."""
    return _ADAPTIVE_OPTIMIZER
