from components.face_detector.base import FaceDetector

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InputVStreamParams,
    InferVStreams,
    OutputVStreamParams,
)

class HEFModel(FaceDetector):
    def __init__(self, model_path: str, target):
        if not model_path.endswith('.hef'):
            raise ValueError("Model path must point to an HEF file.")
        self.target = target
        self.model_path = model_path

        self._load_model()

    def _load_model(self):
        """Load the HEF model."""
        hef = HEF(self.model_path)

        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = self.target.configure(hef, configure_params)
        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

        self.input_names = [info.name for info in hef.get_input_vstream_infos()]
        self.output_names = [info.name for info in hef.get_output_vstream_infos()]
