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

# Canonical class list — matches Kaggle "Indian Food Images Dataset" folder names.
# Order is fixed: index = class id used by the trained model.
CLASSES = [
    "adhirasam", "aloo_gobi", "aloo_matar", "aloo_methi", "aloo_shimla_mirch",
    "aloo_tikki", "anarsa", "ariselu", "bandar_laddu", "basundi",
    "bhatura", "bhindi_masala", "biryani", "boondi", "butter_chicken",
    "chak_hao_kheer", "cham_cham", "chana_masala", "chapati", "chhena_kheeri",
    "chicken_razala", "chicken_tikka", "chicken_tikka_masala", "chikki",
    "daal_baati_churma", "daal_puri", "dal_makhani", "dal_tadka",
    "dharwad_pedha", "doodhpak", "double_ka_meetha", "dum_aloo",
    "gajar_ka_halwa", "gavvalu", "ghevar", "gulab_jamun", "imarti",
    "jalebi", "kachori", "kadai_paneer", "kadhi_pakoda", "kajjikaya",
    "kakinada_khaja", "kalakand", "karela_bharta", "kofta", "kuzhi_paniyaram",
    "lassi", "ledikeni", "litti_chokha", "lyangcha", "maach_jhol",
    "makki_di_roti_sarson_da_saag", "malapua", "misi_roti", "misti_doi",
    "modak", "mysore_pak", "naan", "navrattan_korma", "palak_paneer",
    "paneer_butter_masala", "phirni", "pithe", "poha", "poornalu",
    "pootharekulu", "qubani_ka_meetha", "rabri", "ras_malai", "rasgulla",
    "sandesh", "shankarpali", "sheer_korma", "sheera", "shrikhand",
    "sohan_halwa", "sohan_papdi", "sutar_feni", "unni_appam",
]

NUM_CLASSES = len(CLASSES)


def display_name(class_key: str) -> str:
    return class_key.replace("_", " ").title()
