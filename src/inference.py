import numpy as np
import onnxruntime as ort
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = PROJECT_ROOT / "models" / "mnist_model.onnx"


class ONNXPredictor:
    def __init__(self, model_path: Path | str | None = None):
        if model_path is None:
            model_path = DEFAULT_ONNX
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: np.ndarray) -> tuple[int, float, np.ndarray]:
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, ...]
        elif image.ndim == 3:
            image = image[np.newaxis, ...]

        image = image.astype(np.float32)

        logits = self.session.run(None, {self.input_name: image})[0]
        probs = self._softmax(logits[0])
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        return pred, confidence, probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
