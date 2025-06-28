from components.rppg_signal_extractor.deep_learning.base import DeepLearningRPPGSignalExtractor

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InputVStreamParams,
    InferVStreams,
    OutputVStreamParams,
    VDevice,
)

class HEFModel(DeepLearningRPPGSignalExtractor):
    def __init__(self, model_path: str, target, fps: float=30.0):
        if not model_path.endswith('.hef'):
            raise ValueError("Model path must point to an HEF file.")
        self.target = target
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

        # self.infer_pipeline = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        # self.activated_network = self.network_group.activate(self.network_group_params)

        # self.infer_pipeline.__enter__()
        # self.activated_network.__enter__()

        # # lazy loading
        # _ = self.infer_pipeline.infer(self.get_dummy_input())

        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(self.get_dummy_input())
                print(f"Stream output shape is {infer_results[self.output_name].shape}")
                
    def cleanup(self):
        """Cleanup resources."""
        # self.infer_pipeline.__exit__(None, None, None)
 
        # self.activated_network.__exit__(None, None, None)

        # self.target.release()
        # del self.target

        super().cleanup()

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Ensure cleanup on exit."""
        self.cleanup()
        return super().__exit__(exc_type, exc_value, traceback)