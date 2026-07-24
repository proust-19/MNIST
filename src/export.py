import torch
import torch.onnx
from pathlib import Path

from .model import CNN
from .config import DEVICE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def export_to_onnx(pth_path: Path | None = None, onnx_path: Path | None = None) -> Path:
    if pth_path is None:
        pth_path = MODELS_DIR / "mnist_cnn.pth"  # existing trained model
    if onnx_path is None:
        onnx_path = MODELS_DIR / "mnist_model.onnx"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(pth_path, map_location=DEVICE, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28).to(DEVICE)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Exported ONNX model to {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    export_to_onnx()
