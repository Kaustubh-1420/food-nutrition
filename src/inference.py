"""ONNX Runtime inference wrapper. Pure CPU, no torch dependency."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    MIN_CONFIDENCE,
    MODEL_PATH,
    TOP_K,
)

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def _resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    short = int(size * 1.15)
    w, h = img.size
    scale = short / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = _resize_and_center_crop(img, IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # HWC
    arr = arr.transpose(2, 0, 1)                       # CHW
    arr = (arr - _MEAN) / _STD
    return arr[None, ...]                              # NCHW


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class FoodClassifier:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Train and export first "
                "(see training/README.md)."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0  # let ORT pick
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img: Image.Image, top_k: int = TOP_K) -> list[tuple[str, float]]:
        x = preprocess(img)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = _softmax(logits)
        idx = np.argsort(probs)[::-1][:top_k]
        results = [(CLASSES[i], float(probs[i])) for i in idx if probs[i] >= MIN_CONFIDENCE]
        # Always return at least the top-1 even if below threshold so the UI can flag low-confidence.
        if not results:
            top1 = int(idx[0])
            results = [(CLASSES[top1], float(probs[top1]))]
        return results
