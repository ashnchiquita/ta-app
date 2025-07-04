from typing import List, Optional
import numpy as np
import time

class IncrementalChunk:
    """Represents a chunk of processed data."""
    
    def __init__(self, chunk_id: int, frames: List[np.ndarray], 
                 bvp_signal: Optional[np.ndarray] = None, 
                 timestamp: float = None):
        self.chunk_id = chunk_id
        self.frames = frames
        self.bvp_signal = bvp_signal
        self.timestamp = timestamp or time.time()
        self.processed = bvp_signal is not None
