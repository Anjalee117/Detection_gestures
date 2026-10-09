from pathlib import Path

import numpy as np


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path=None,
        num_threads=1,
    ):
        import tensorflow as tf

        if model_path is None:
            model_path = Path(__file__).resolve().with_name("keypoint_classifier.tflite")
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path), num_threads=num_threads
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if tuple(self.input_details[0]["shape"]) != (1, 42):
            raise ValueError("Classifier input must have shape (1, 42)")
        if tuple(self.output_details[0]["shape"]) != (1, 26):
            raise ValueError("Classifier output must have shape (1, 26)")
        if self.input_details[0]["dtype"] != np.float32 or self.output_details[0]["dtype"] != np.float32:
            raise ValueError("Classifier input/output must use float32")

    def __call__(
        self,
        landmark_list,
    ):
        return self.predict(landmark_list)[0]

    def predict(self, landmark_list):
        features = np.asarray(landmark_list, dtype=np.float32)
        if features.shape != (42,) or not np.isfinite(features).all():
            raise ValueError("Expected 42 finite landmark values")
        input_details_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_details_tensor_index, features[None, :]
        )
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]["index"]

        result = self.interpreter.get_tensor(output_details_tensor_index)

        probabilities = np.squeeze(result)
        if probabilities.shape != (26,) or not np.isfinite(probabilities).all():
            raise ValueError("Classifier returned invalid probabilities")
        if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1, atol=1e-4):
            raise ValueError("Classifier output must contain normalized probabilities")
        result_index = int(np.argmax(probabilities))

        return result_index, float(probabilities[result_index])
