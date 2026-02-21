import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model import CNN
from src.config import DEVICE
from src.data_loader import PROJECT_ROOT 


# Load model
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_cnn.pth"
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded: {MODEL_PATH} | Device: {DEVICE}")


def center_and_resize(canvas_280: np.ndarray) -> np.ndarray:
    img = canvas_280.copy()

    # remove faint noise
    _, th = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(th)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = th[y:y+h, x:x+w]

    h0, w0 = cropped.shape
    scale = 20.0 / max(h0, w0)
    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    out[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # center-of-mass shift
    M = cv2.moments(out)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        shift_x = 14 - cx
        shift_y = 14 - cy
        T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        out = cv2.warpAffine(out, T, (28, 28), borderValue=0)

    return out


def preprocess(canvas_280: np.ndarray) -> torch.Tensor:
    img28 = center_and_resize(canvas_280)
    x = img28.astype(np.float32) / 255.0

    x = (x - 0.1307) / 0.3081

    x_tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return x_tensor, img28


@torch.no_grad()
def predict(canvas_280: np.ndarray):
    x, img28 = preprocess(canvas_280)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item()) * 100.0
    return pred, conf, img28, probs.cpu().numpy()


def draw_live():
    title = "Live MNIST Draw | c=clear q=quit"
    canvas = np.zeros((280, 280), dtype=np.uint8)
    cv2.namedWindow(title)

    brush = 10  # try 10..18 if needed

    last_pred = None
    last_conf = 0.0
    last_time = 0.0
    interval = 0.12  # seconds between predictions (CPU-friendly)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(canvas, (x, y), brush, 255, -1)

    cv2.setMouseCallback(title, on_mouse)

    while True:
        now = time.time()

        # predict only if enough time passed + something drawn
        if now - last_time >= interval:
            last_time = now
            if canvas.mean() > 2:
                last_pred, last_conf, img28, _ = predict(canvas)
                # show what model sees
                view = cv2.resize(img28, (140, 140), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Model sees (28x28)", view)
            else:
                last_pred, last_conf = None, 0.0
                cv2.imshow("Model sees (28x28)", np.zeros((140, 140), dtype=np.uint8))

        # display
        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        if last_pred is None:
            text = "Draw a digit..."
        else:
            text = f"Pred: {last_pred}  Conf: {last_conf:.1f}%"

        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display, "Hold left click to draw | c clear | q quit", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        cv2.imshow(title, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            canvas.fill(0)
            last_pred, last_conf = None, 0.0
            print("🔄 Cleared")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("🎨 Drawing app started!")
    print("   - Draw digits with your mouse")
    print("   - Press 'p' to predict")
    print("   - Press 'c' to clear")
    print("   - Press 'q' to quit")
    draw_live()
