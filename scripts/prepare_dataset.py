#!/usr/bin/env python3
"""Create a reproducible 85/15 train/validation split from class folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


DEFAULT_CLASSES = ["ragdoll", "singapura", "persian", "sphynx", "pallas"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/split"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already contains files. Use --overwrite to rebuild it.")
        shutil.rmtree(args.output_dir)

    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int]] = {}

    for class_name in DEFAULT_CLASSES:
        source = args.raw_dir / class_name
        if not source.is_dir():
            raise FileNotFoundError(f"Missing class folder: {source}")

        files = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        unique_files: list[Path] = []
        seen_hashes: set[str] = set()
        for file_path in files:
            fingerprint = image_hash(file_path)
            if fingerprint not in seen_hashes:
                seen_hashes.add(fingerprint)
                unique_files.append(file_path)

        if len(unique_files) < args.min_per_class:
            raise ValueError(
                f"{class_name}: only {len(unique_files)} unique images; need at least {args.min_per_class}."
            )

        rng.shuffle(unique_files)
        val_count = max(1, round(len(unique_files) * args.val_ratio))
        val_files = unique_files[:val_count]
        train_files = unique_files[val_count:]

        for split_name, split_files in (("train", train_files), ("val", val_files)):
            destination = args.output_dir / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for index, file_path in enumerate(split_files):
                suffix = file_path.suffix.lower()
                shutil.copy2(file_path, destination / f"{class_name}_{index:04d}{suffix}")

        summary[class_name] = {
            "raw_files": len(files),
            "unique_files": len(unique_files),
            "train": len(train_files),
            "val": len(val_files),
        }
        print(f"{class_name}: {summary[class_name]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dataset_counts.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Created split at: {args.output_dir}")


if __name__ == "__main__":
    main()
