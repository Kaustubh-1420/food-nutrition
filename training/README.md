# Training pipeline

End-to-end: download dataset → split → fine-tune EfficientNet-B2 → export ONNX.

Run on Colab (T4 free tier) or Lightning AI. Local Mac works for prep/export but not full training.

## 1. Install training deps

```bash
pip install -r requirements-train.txt
```

## 2. Get the dataset

Kaggle: [Indian Food Images Dataset](https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification) (or the equivalent ~80-class version that matches `src/config.CLASSES`).

```bash
# Via Kaggle CLI (after `kaggle login`):
kaggle datasets download -d l33tc0d3r/indian-food-classification -p data/raw --unzip
```

Expected layout afterwards:

```
data/raw/
  adhirasam/*.jpg
  aloo_gobi/*.jpg
  ...
```

If the folder names from the download don't exactly match `src/config.CLASSES`, rename them — the class list is the source of truth (it controls model output indices and nutrition lookups).

## 3. Audit and split

```bash
python -m training.prepare_data --raw data/raw --out data/splits
```

This writes `train.csv`, `val.csv`, `test.csv`, and `class_counts.csv`. It will warn loudly about missing classes or classes with <50 images. **Address those before training** — usually by scraping more images for the thin classes.

## 4. Train

```bash
python -m training.train \
  --splits data/splits \
  --out models/checkpoints \
  --epochs 25 --batch-size 32 --lr 3e-4
```

On a T4 expect ~6–10 minutes per epoch with ~5k training images, so a full run is ~3 hours.

The script saves `models/checkpoints/best.pt` whenever val macro-F1 improves. Realistic target: ≥ 0.75 macro-F1, ≥ 0.80 top-1 accuracy. If you're well under that, the dataset is the bottleneck (more data per thin class), not the model.

## 5. Export to ONNX

```bash
python -m training.export_onnx \
  --ckpt models/checkpoints/best.pt \
  --out models/model.onnx
```

This simplifies the graph and verifies parity with the PyTorch model (max abs diff < 1e-3).

## 6. Smoke-test inference locally

```bash
pip install -r requirements.txt   # CPU-only deps
python -c "from PIL import Image; from src.inference import FoodClassifier; \
           print(FoodClassifier().predict(Image.open('examples/test.jpg')))"
```

## 7. Ship it

The Gradio app reads `models/model.onnx` directly. Push that file (Git LFS) to your HF Space or upload via the Spaces UI. See top-level `README.md` for HF Spaces config.
