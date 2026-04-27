"""Audit the Kaggle Indian Food Images dataset and write stratified splits.

Expected layout:
    data/raw/<class_name>/<image>.jpg

Outputs:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
    data/splits/class_counts.csv

Usage:
    python -m training.prepare_data --raw data/raw --out data/splits --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASSES  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15  # remainder is test
MIN_IMAGES_WARN = 50


def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            found[cls] = []
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        found[cls] = imgs
    return found


def stratified_split(items: list[Path], rng: random.Random) -> tuple[list[Path], list[Path], list[Path]]:
    items = items.copy()
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def write_csv(path: Path, rows: list[tuple[str, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label_id", "label_name"])
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    by_class = collect_images(args.raw)
    counts = {cls: len(imgs) for cls, imgs in by_class.items()}

    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        print(f"WARNING: {len(missing)} classes missing in {args.raw}: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    thin = [c for c, n in counts.items() if 0 < n < MIN_IMAGES_WARN]
    if thin:
        print(f"WARNING: {len(thin)} classes have <{MIN_IMAGES_WARN} images: " +
              ", ".join(f"{c}({counts[c]})" for c in thin))

    splits: dict[str, list[tuple[str, int, str]]] = {"train": [], "val": [], "test": []}
    for label_id, cls in enumerate(CLASSES):
        imgs = by_class[cls]
        if len(imgs) < 3:
            continue
        tr, va, te = stratified_split(imgs, rng)
        for bucket, paths in (("train", tr), ("val", va), ("test", te)):
            splits[bucket].extend((str(p), label_id, cls) for p in paths)

    for bucket, rows in splits.items():
        out = args.out / f"{bucket}.csv"
        write_csv(out, rows)
        print(f"{bucket}: {len(rows):>5} samples -> {out}")

    counts_path = args.out / "class_counts.csv"
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    with counts_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_id", "label_name", "total", "train", "val", "test"])
        per_split: dict[str, Counter] = {k: Counter(r[2] for r in v) for k, v in splits.items()}
        for label_id, cls in enumerate(CLASSES):
            w.writerow([
                label_id, cls, counts[cls],
                per_split["train"][cls], per_split["val"][cls], per_split["test"][cls],
            ])
    print(f"class counts -> {counts_path}")


if __name__ == "__main__":
    main()
