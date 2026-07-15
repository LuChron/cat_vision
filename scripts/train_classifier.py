#!/usr/bin/env python3
"""Fine-tune ImageNet-pretrained EfficientNet-B2 on the five cat-breed classes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/split"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classifier"),
                        help="Experiment metrics; copy the best .pth to models/classifier for runtime")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, scaler, device, training: bool) -> tuple[float, float]:
    model.train(training)
    loss_sum = correct = total = 0
    progress = tqdm(loader, leave=False, desc="train" if training else "val")
    for images, labels in progress:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is unavailable. Recheck the catvision environment installation.")
    print("Device:", torch.cuda.get_device_name(0))

    weights = EfficientNet_B2_Weights.DEFAULT
    # Keep most of the cat in view: head and facial markings are important for
    # the Ragdoll/Persian distinction. The geometric and image-quality changes
    # mimic a printed card photographed from a real scene without creating
    # separate duplicate files on disk.
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomResizedCrop(260, scale=(0.88, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=6, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.35),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.20),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.18, contrast=0.30, saturation=0.10, hue=0.02)
        ], p=0.80),
        # Simulate camera auto-exposure washing out white fur: brighten while
        # reducing contrast and saturation, without changing the class label.
        transforms.RandomApply([
            transforms.ColorJitter(brightness=(1.10, 1.45), contrast=(0.65, 1.00), saturation=(0.70, 1.00))
        ], p=0.30),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])
    val_transform = weights.transforms()
    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(args.data_dir / "val", transform=val_transform)
    if train_ds.classes != val_ds.classes or len(train_ds.classes) != 5:
        raise ValueError(f"Expected matching five classes; train={train_ds.classes}, val={val_ds.classes}")

    class_counts = {name: train_ds.targets.count(index) for index, name in enumerate(train_ds.classes)}
    sample_weights = [1.0 / class_counts[train_ds.classes[label]] for label in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print("Train images per class:", class_counts)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    model = efficientnet_b2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = args.output_dir / "best_effnet_b2_cat_breeds.pth"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False)
        scheduler.step()
        for key, value in (("train_loss", train_loss), ("train_acc", train_acc), ("val_loss", val_loss), ("val_acc", val_acc)):
            history[key].append(value)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_acc={train_acc:.2%}, val_acc={val_acc:.2%}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"arch": "efficientnet_b2", "model_state": model.state_dict(), "classes": train_ds.classes,
                        "class_to_idx": train_ds.class_to_idx, "epoch": epoch, "best_val_acc": best_val_acc}, best_path)

    class_map_path = args.output_dir / "class_to_idx.json"
    class_map_path.write_text(json.dumps(train_ds.class_to_idx, indent=2), encoding="utf-8")
    results = {"classes": train_ds.classes, "train_images_per_class": class_counts, "best_validation_accuracy": best_val_acc,
               "final_training_accuracy": history["train_acc"][-1], "history": history,
               "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr}
    (args.output_dir / "training_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1); plt.plot(history["train_acc"], label="train"); plt.plot(history["val_acc"], label="validation"); plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="validation"); plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(args.output_dir / "training_curves.png", dpi=180); plt.close()
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model: {best_path}")


if __name__ == "__main__":
    main()
