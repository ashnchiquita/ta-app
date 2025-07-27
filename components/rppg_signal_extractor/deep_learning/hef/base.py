from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InputVStreamParams,
    OutputVStreamParams,
)
from components.manager.hailo_target_manager import HailoTargetManager

class HEFModel(DeepLearningRPPGSignalExtractor):
    def __init__(self, model_path: str, fps: float=30.0):
        if not model_path.endswith('.hef'):
            raise ValueError("Model path must point to an HEF file.")
        self.target = HailoTargetManager().target
        super().__init__(model_path, fps)

    def _load_model(self):
        """Load the HEF model."""
        hef = HEF(self.model_path)

        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = self.target.configure(hef, configure_params)
        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

        self.input_name = hef.get_input_vstream_infos()[0].name
        self.output_name = hef.get_output_vstream_infos()[0].name

    def cleanup(self):
        """Cleanup HEF model resources."""
        try:
            # Call parent cleanup
            super().cleanup()
            
            # Clean up network group resources
            if hasattr(self, 'network_group') and self.network_group is not None:
                # Note: Network groups are automatically cleaned up when target is released
                self.network_group = None
                
            self.input_vstreams_params = None
            self.output_vstreams_params = None
            self.network_group_params = None
            
        except Exception as e:
            print(f"Warning: Error during HEF model cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
