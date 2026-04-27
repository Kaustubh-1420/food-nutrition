---
title: Indian Food Nutrition
emoji: 🍛
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Indian Food → Nutrition

Upload a photo of an Indian dish, get an estimated nutritional breakdown
(calories, protein, carbs, fat, fiber) for a typical serving — adjustable
with a portion slider.

## How it works

1. **Classify** — a fine-tuned EfficientNet-B2 (exported to ONNX) identifies
   the dish from one of 79 Indian food classes.
2. **Look up** — predicted dish maps to a curated nutrition record sourced
   from the [Indian Food Composition Tables (IFCT 2017)](https://www.nin.res.in/)
   published by the National Institute of Nutrition.
3. **Scale** — values are scaled by the user's portion multiplier.

Inference runs on CPU via ONNX Runtime; ~150 ms per image on the HF Spaces
free tier.

## Repository layout

```
app.py                  Gradio entry point (HF Spaces reads this)
src/
  config.py             Class list, paths, image preprocessing constants
  inference.py          ONNX Runtime wrapper, no torch dependency
  nutrition.py          Static-DB lookup + portion scaling
data/
  nutrition_db.json     Curated per-100g nutrition for all classes
training/
  prepare_data.py       Dataset audit + stratified split
  train.py              EfficientNet-B2 fine-tune (Colab/Lightning AI)
  export_onnx.py        PyTorch → ONNX with parity check
  README.md             Full training instructions
models/
  model.onnx            Trained model (not in git; produced by export_onnx)
```

## Limitations (v1)

- **Single dish per photo.** A thali with multiple items will return the
  most prominent dish only. Multi-label classification is planned for v2.
- **Fixed standard servings** with a manual portion multiplier — no portion
  estimation from image scale.
- **79 dishes** drawn from the Kaggle Indian Food Images dataset. Regional
  variants outside this list aren't covered.
- **Recipe variance** — home recipes differ; nutrition values are rough
  averages, not lab measurements. Treat results as estimates.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

You need `models/model.onnx` for real predictions. See
`training/README.md` to produce one.

## Credits

- Nutrition reference: IFCT 2017, NIN Hyderabad
- Dataset: Kaggle Indian Food Images Dataset
- Model: EfficientNet-B2 (Tan & Le, 2019), via `timm`
