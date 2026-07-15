#!/usr/bin/env python3
"""Automatically create CNN training crops with the same YOLO cascade as runtime."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from catvision.detector import detect_with_cascade, load_detectors


CLASSES = ("ragdoll", "singapura", "persian", "sphynx", "pallas")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cropped"))
    parser.add_argument("--report", type=Path, default=Path("outputs/crop_report.json"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def crop_with_margin(image, box, margin: float):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return image[max(0, y1 - dy):min(height, y2 + dy), max(0, x1 - dx):min(width, x2 + dx)]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already contains files. Use --overwrite to rebuild it.")
        shutil.rmtree(args.output_dir)
    detectors = load_detectors()
    report = {}
    for class_name in CLASSES:
        source_dir = args.raw_dir / class_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {source_dir}")
        target_dir = args.output_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(path for path in source_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        missed, saved = [], 0
        for image_path in images:
            frame = cv2.imread(str(image_path))
            detection = None if frame is None else detect_with_cascade(frame, detectors, args.conf)
            if detection is None:
                missed.append(str(image_path))
                continue
            best_index = int(detection.result.boxes.conf.argmax().item())
            crop = crop_with_margin(frame, detection.result.boxes.xyxy[best_index].cpu().numpy(), args.margin)
            if crop.size:
                cv2.imwrite(str(target_dir / f"{class_name}_{saved:04d}.jpg"), crop)
                saved += 1
        report[class_name] = {"input_images": len(images), "cropped_images": saved,
                              "missed_images": missed}
        print(f"{class_name}: cropped {saved}/{len(images)}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Crop report: {args.report}")


if __name__ == "__main__":
    main()
