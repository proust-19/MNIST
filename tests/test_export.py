from pathlib import Path
from src.export import export_to_onnx, MODELS_DIR


def test_export_creates_onnx():
    pth = MODELS_DIR / "mnist_cnn.pth"
    if not pth.exists():
        return

    onnx_path = export_to_onnx()
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0


def test_onnx_loadable():
    import onnxruntime as ort

    onnx_path = MODELS_DIR / "mnist_model.onnx"
    if not onnx_path.exists():
        return

    sess = ort.InferenceSession(str(onnx_path))
    assert sess.get_inputs()[0].name == "input"
    assert sess.get_outputs()[0].name == "output"
