import os
import numpy as np
from components.face_detector.hef.preprocess import Preprocessor
from components.face_detector.hef.retina_face.post_process import PostProcessor
from components.face_detector.hef.base import HEFModel
import json
from constants import DEGIRUM_ZOO_DIR

from hailo_platform import (
    InferVStreams,
)

class RetinaFace(HEFModel):
    def __init__(self, target):
        model_name = "retinaface_mobilenet--736x1280_quant_hailort_hailo8l_1"
        zoo_path = os.path.join(DEGIRUM_ZOO_DIR, model_name)
        hef_path = os.path.join(zoo_path, f"{model_name}.hef")
        json_config_path = os.path.join(zoo_path, f"{model_name}.json")

        with open(json_config_path, 'r') as f:
            json_config = json.dumps(json.load(f))

        self.preprocessor = Preprocessor(input_shape_nhwc=(1, 736, 1280, 3))
        self.postprocessor = PostProcessor(json_config, zoo_path)

        self.layer_order = [
            'retinaface_mobilenet_v1/conv41', 'retinaface_mobilenet_v1/conv42', 'retinaface_mobilenet_v1/conv43',
            'retinaface_mobilenet_v1/conv32', 'retinaface_mobilenet_v1/conv33', 'retinaface_mobilenet_v1/conv34',
            'retinaface_mobilenet_v1/conv23', 'retinaface_mobilenet_v1/conv24', 'retinaface_mobilenet_v1/conv25'
        ]

        super().__init__(
            model_path=hef_path,
            target=target
        )

    def detect(self, frame):
        """
        Detect faces in the given frame using the RetinaFace model.
        
        Args:
            frame (numpy.ndarray): The input frame in which to detect faces.
        
        Returns:
            list: A list of detected face bounding boxes.
        """
        # Preprocess
        prep_frame, converter = self.preprocessor.forward(frame)

        batched_image = np.expand_dims(prep_frame, axis=0)  # Add batch dimension
        input_data = {self.input_names[0]: batched_image}

        # Inference
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)

        # Postprocess
        sorted_results = [infer_results[layer] for layer in self.layer_order]

        reshaped_results = []
        for layer, result in zip(self.layer_order, sorted_results):
            batch, height, width, channels = result.shape
            reshaped = result.reshape(batch, height * width, channels)
            reshaped_results.append(reshaped)

        postproc_result = self.postprocessor.forward(reshaped_results)

        faces = []
        for detection in postproc_result:
            x_min, y_min, x_max, y_max = detection['bbox']  # bbox format: [x_min, y_min, x_max, y_max]
            x_min, y_min = converter(x_min, y_min)
            x_max, y_max = converter(x_max, y_max)

            faces.append((int(x_min), int(y_min), int(x_max), int(y_max)))

        return faces
