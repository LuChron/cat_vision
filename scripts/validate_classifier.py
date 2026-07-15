#!/usr/bin/env python3
"""Evaluate the trained classifier on a validation dataset.

Usage:
  python -m scripts.validate_classifier --data-dir data/split/val

Output: per-class accuracy + confusion matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/split/val"),
                        help="Path to validation set (with class subfolders)")
    parser.add_argument("--model", type=Path, default=Path("models/classifier/best_effnet_b2_cat_breeds.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.model, map_location=device)
    classes = ckpt.get("classes")
    if not classes:
        raise ValueError("Checkpoint missing 'classes' field.")

    model = efficientnet_b2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Model: {args.model.name}")
    print(f"Classes: {classes}")
    print()

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(args.data_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Validation images: {len(dataset)}")
    print(f"{'─' * 50}")

    # Evaluate
    correct, total = 0, 0
    per_class = {cls: {"correct": 0, "total": 0} for cls in classes}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for p, l in zip(preds, labels):
                cls = classes[int(l.item())]
                per_class[cls]["total"] += 1
                if int(p.item()) == int(l.item()):
                    per_class[cls]["correct"] += 1
                    correct += 1
                total += 1

    # Report
    print(f"\n{'Class':20s} {'Correct':>10s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"{'─' * 50}")
    for cls in classes:
        c = per_class[cls]["correct"]
        t = per_class[cls]["total"]
        acc = c / t if t > 0 else 0
        print(f"{cls:20s} {c:>5d}/{t:<3d} {t:>8d} {acc:>8.2%}")
    print(f"{'─' * 50}")
    print(f"{'Total':20s} {correct:>5d}/{total:<3d} {total:>8d} {correct/total:>8.2%}")


if __name__ == "__main__":
    main()
