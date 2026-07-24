import io
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.export import MODELS_DIR
from src.inference import DEFAULT_ONNX, ONNXPredictor

predictor: ONNXPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = ONNXPredictor(DEFAULT_ONNX)
    yield


app = FastAPI(title="MNIST Inference Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/model-info")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    onnx_path = DEFAULT_ONNX
    pth_path = MODELS_DIR / "mnist_cnn.pth"

    input_meta = predictor.session.get_inputs()[0]
    output_meta = predictor.session.get_outputs()[0]

    return {
        "model": "mnist_cnn",
        "framework": "onnxruntime",
        "onnx_file": str(onnx_path.name),
        "pth_file": str(pth_path.name),
        "input_name": input_meta.name,
        "input_shape": input_meta.shape,
        "output_name": output_meta.name,
        "output_shape": output_meta.shape,
        "num_classes": 10,
        "image_size": "28x28 grayscale",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(status_code=400, detail="Upload a PNG, JPEG, or WebP image")

    start = time.perf_counter()

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - 0.1307) / 0.3081

    pred, confidence, probs = predictor.predict(img_array)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "prediction": pred,
        "confidence": round(confidence * 100, 2),
        "probabilities": {str(i): round(float(probs[i]) * 100, 2) for i in range(10)},
        "latency_ms": round(elapsed_ms, 2),
    }
