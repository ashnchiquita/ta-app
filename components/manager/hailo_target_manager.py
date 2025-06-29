import atexit
from typing import Optional
from hailo_platform import VDevice, HailoSchedulingAlgorithm


class HailoTargetManager:
    """Singleton class for managing Hailo target devices."""
    
    _instance: Optional['HailoTargetManager'] = None
    _target: Optional[VDevice] = None
    
    def __new__(cls) -> 'HailoTargetManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if self._target is None:
            self._initialize_target()
            # Register cleanup function to be called on exit
            atexit.register(self._cleanup)
    
    def _initialize_target(self) -> None:
        """Initialize the Hailo target device."""
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
        self._target = VDevice(params=params)
    
    @property
    def target(self) -> VDevice:
        """Get the Hailo target device."""
        if self._target is None:
            raise RuntimeError("Hailo target device not initialized")
        return self._target
    
    def _cleanup(self) -> None:
        """Release the target device."""
        if self._target is not None:
            try:
                self._target.release()
                self._target = None
            except Exception as e:
                print(f"Error releasing Hailo target device: {e}")
    
    def release(self) -> None:
        """Manually release the target device."""
        self._cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self._cleanup()
