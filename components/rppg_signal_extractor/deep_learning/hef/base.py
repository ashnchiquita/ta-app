from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InputVStreamParams,
    InferVStreams,
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

        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(self.get_dummy_input())
                print(f"Stream output shape is {infer_results[self.output_name].shape}")
