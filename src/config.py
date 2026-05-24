"""Single source of truth for class IDs, paths, and inference settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.onnx"
NUTRITION_DB_PATH = PROJECT_ROOT / "data" / "nutrition_db.json"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

IMG_SIZE = 260
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TOP_K = 3
MIN_CONFIDENCE = 0.10

# Canonical class list — matches the Kaggle "Indian Food Classification" dataset
# (l33tc0d3r/indian-food-classification) with pizza and burger dropped.
# Order is fixed: index = class id used by the trained model.
CLASSES = [
    "butter_naan",
    "chai",
    "chapati",
    "chole_bhature",
    "dal_makhani",
    "dhokla",
    "fried_rice",
    "idli",
    "jalebi",
    "kaathi_rolls",
    "kadai_paneer",
    "kulfi",
    "masala_dosa",
    "momos",
    "paani_puri",
    "pakode",
    "pav_bhaji",
    "samosa",
]

NUM_CLASSES = len(CLASSES)


def display_name(class_key: str) -> str:
    return class_key.replace("_", " ").title()
