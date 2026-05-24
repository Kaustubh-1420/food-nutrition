"""Gradio app — Indian food photo to nutrition breakdown.

HF Spaces entry point. Looks for an ONNX model at models/model.onnx.
If absent, the app launches in a clearly-labelled demo mode using the
top-1 dish based on filename heuristics so the UI is still inspectable.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from src.config import EXAMPLES_DIR, MODEL_PATH, display_name
from src.inference import FoodClassifier
from src.llm_summary import summarize
from src.nutrition import lookup

MODEL_AVAILABLE = Path(MODEL_PATH).exists()
_classifier: FoodClassifier | None = None


def _get_classifier() -> FoodClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FoodClassifier()
    return _classifier


HEADER_OK = "## Indian Food → Nutrition\nUpload a photo of a single dish. Adjust portion if needed."
HEADER_NO_MODEL = (
    "## Indian Food → Nutrition (model not loaded)\n"
    "No `models/model.onnx` found. Train one with the scripts in `training/` "
    "and copy the exported ONNX to `models/model.onnx`."
)


def predict(image: Image.Image, portion: float) -> tuple[dict, list, str, str]:
    if image is None:
        return {}, [], "Upload a photo to get started.", ""
    if not MODEL_AVAILABLE:
        return {}, [], "Model file missing — see the header above.", ""

    clf = _get_classifier()
    preds = clf.predict(image)

    label_dict = {display_name(name): conf for name, conf in preds}

    top_dish, top_conf = preds[0]
    nut = lookup(top_dish, portion=portion)
    if nut is None:
        return label_dict, [], f"No nutrition data for `{top_dish}`.", ""

    note_lines = [
        f"**{nut.display}** · category: {nut.category} · confidence: {top_conf * 100:.1f}%",
        f"Portion: {portion:.1f}× standard serving ({nut.serving_g:.0f} g).",
    ]
    if top_conf < 0.40:
        note_lines.append("Low confidence — model is uncertain. Check alternates above.")

    summary = summarize(nut, top_conf)
    summary_md = f"### Quick take\n{summary}" if summary else ""

    return label_dict, nut.as_table_rows(), "\n\n".join(note_lines), summary_md


def build_ui() -> gr.Blocks:
    examples = []
    if EXAMPLES_DIR.exists():
        examples = [[str(p), 1.0] for p in sorted(EXAMPLES_DIR.glob("*"))
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    with gr.Blocks(title="Indian Food Nutrition", theme=gr.themes.Soft()) as demo:
        gr.Markdown(HEADER_OK if MODEL_AVAILABLE else HEADER_NO_MODEL)

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Food photo", height=320)
                portion = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Portion (×)")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                labels_out = gr.Label(num_top_classes=3, label="Top predictions")
                table_out = gr.Dataframe(
                    headers=["Nutrient", "Amount"],
                    label="Nutrition (for chosen portion)",
                    interactive=False, wrap=True,
                )
                note_out = gr.Markdown()
                summary_out = gr.Markdown()

        outputs = [labels_out, table_out, note_out, summary_out]
        btn.click(predict, [image_in, portion], outputs)
        image_in.change(predict, [image_in, portion], outputs)
        portion.change(predict, [image_in, portion], outputs)

        if examples:
            gr.Examples(examples=examples, inputs=[image_in, portion])

        gr.Markdown(
            "---\nNutrition values approximated from IFCT 2017 (NIN, India). "
            "Single-dish identification only in v1 — multi-dish thali support is on the roadmap."
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
