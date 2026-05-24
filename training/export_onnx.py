"""Export the best PyTorch checkpoint to ONNX, simplify, and verify parity.

Usage:
    python -m training.export_onnx \\
        --ckpt models/checkpoints/best.pt \\
        --out models/model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from onnxsim import simplify

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import IMG_SIZE, NUM_CLASSES  # noqa: E402

OPSET = 18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/model.onnx"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    assert ckpt["num_classes"] == NUM_CLASSES, "class count mismatch with src.config"
    assert ckpt["img_size"] == IMG_SIZE, "image size mismatch with src.config"

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    tmp = args.out.with_suffix(".raw.onnx")
    torch.onnx.export(
        model, dummy, tmp,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True,
        dynamo=False,
    )

    raw = onnx.load(tmp)
    simplified, ok = simplify(raw)
    assert ok, "onnxsim failed"
    onnx.save(simplified, args.out)
    tmp.unlink()

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    diff = np.max(np.abs(torch_out - onnx_out))
    print(f"max |torch - onnx| = {diff:.2e}")
    assert diff < 1e-3, f"parity check failed: {diff}"

    size_mb = args.out.stat().st_size / 1e6
    print(f"exported {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
