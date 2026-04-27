"""Fine-tune EfficientNet-B2 on the Indian food dataset.

Pipeline:
  - 2 warmup epochs with backbone frozen (head only)
  - Then unfreeze all, cosine LR schedule
  - Class-weighted CE loss to handle imbalance
  - Mixed precision on CUDA, single-precision on CPU/MPS
  - Best checkpoint by val macro-F1

Usage (Colab T4):
    python -m training.train \\
        --splits data/splits --out models/checkpoints \\
        --epochs 25 --batch-size 32 --lr 3e-4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, NUM_CLASSES  # noqa: E402

MODEL_NAME = "tf_efficientnet_b2.ns_jft_in1k"


class FoodDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.rows: list[tuple[str, int]] = []
        with csv_path.open() as f:
            r = csv.reader(f)
            next(r)
            for path, label_id, _ in r:
                self.rows.append((path, int(label_id)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def class_weights(train_csv: Path) -> torch.Tensor:
    labels: list[int] = []
    with train_csv.open() as f:
        r = csv.reader(f)
        next(r)
        for _, label_id, _ in r:
            labels.append(int(label_id))
    counts = Counter(labels)
    total = sum(counts.values())
    w = torch.zeros(NUM_CLASSES)
    for cid in range(NUM_CLASSES):
        c = counts.get(cid, 0)
        w[cid] = (total / (NUM_CLASSES * c)) if c > 0 else 0.0
    return w


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if "classifier" in name or "fc" in name or "head" in name:
            p.requires_grad = True
        else:
            p.requires_grad = trainable


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    preds, gts = [], []
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        p = logits.argmax(1)
        preds.extend(p.cpu().tolist())
        gts.extend(y.cpu().tolist())
        correct += (p == y).sum().item()
        total += y.numel()
    acc = correct / max(total, 1)
    f1 = f1_score(gts, preds, average="macro", zero_division=0)
    return acc, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("data/splits"))
    ap.add_argument("--out", type=Path, default=Path("models/checkpoints"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = device.type == "cuda"
    print(f"device: {device}, AMP: {use_amp}, classes: {NUM_CLASSES}")

    train_tf, val_tf = build_transforms()
    train_ds = FoodDataset(args.splits / "train.csv", train_tf)
    val_ds = FoodDataset(args.splits / "val.csv", val_tf)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    weights = class_weights(args.splits / "train.csv").to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs - args.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = -1.0
    history: list[dict] = []

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            set_backbone_trainable(model, trainable=False)
            phase = "warmup-head-only"
        elif epoch == args.warmup_epochs:
            set_backbone_trainable(model, trainable=True)
            phase = "full-finetune"
        else:
            phase = "full-finetune"

        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"ep{epoch:02d} [{phase}]")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{running / n_seen:.4f}")

        if epoch >= args.warmup_epochs:
            sched.step()

        train_loss = running / max(n_seen, 1)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        lr_now = optim.param_groups[0]["lr"]
        print(f"ep{epoch:02d} | loss {train_loss:.4f} | val acc {val_acc:.4f} | val F1 {val_f1:.4f} | lr {lr_now:.2e} | {elapsed:.0f}s")
        history.append({
            "epoch": epoch, "phase": phase, "train_loss": train_loss,
            "val_acc": val_acc, "val_f1": val_f1, "lr": lr_now,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt = {
                "state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "classes": CLASSES,
                "img_size": IMG_SIZE,
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
            }
            torch.save(ckpt, args.out / "best.pt")
            print(f"  -> saved best (F1 {val_f1:.4f})")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
