"""Natural-language nutrition summary via Hugging Face Inference API.

Free, no Anthropic key required — uses the HF_TOKEN already configured
for the Space. Returns "" on any error (caller treats empty as "skip").
"""
from __future__ import annotations

import os

from huggingface_hub import InferenceClient

from .nutrition import Nutrition

MODEL = "meta-llama/Llama-3.3-70B-Instruct"

SYSTEM_PROMPT = (
    "You are a practical nutrition assistant. Given an Indian dish and its "
    "nutrition values for one serving, write 2-3 sentences. Be concrete and "
    "useful — call out what stands out (high protein, heavy in fat, low fiber, "
    "etc.) and one practical takeaway (e.g. pair with X, good post-workout, "
    "skip if cutting). Avoid clinical jargon and disclaimers. No bullet points."
)


def summarize(nut: Nutrition, confidence: float) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return ""

    user_msg = (
        f"Dish: {nut.display}\n"
        f"Serving: {nut.serving_g:.0f} g\n"
        f"Calories: {nut.calories:.0f} kcal\n"
        f"Protein: {nut.protein_g:.1f} g\n"
        f"Carbs: {nut.carbs_g:.1f} g\n"
        f"Fat: {nut.fat_g:.1f} g\n"
        f"Fiber: {nut.fiber_g:.1f} g\n"
        f"Model confidence: {confidence * 100:.0f}%"
    )

    try:
        client = InferenceClient(model=MODEL, token=token)
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=140,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"_(summary unavailable: {type(e).__name__})_"
