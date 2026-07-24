import io
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info():
    r = client.get("/model-info")
    if r.status_code == 503:
        return
    data = r.json()
    assert data["num_classes"] == 10


def test_predict():
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post("/predict", files={"file": ("test.png", buf, "image/png")})
    if r.status_code == 503:
        return
    data = r.json()
    assert "prediction" in data
    assert "confidence" in data
    assert 0 <= data["prediction"] <= 9
