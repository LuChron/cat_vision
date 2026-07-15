#!/usr/bin/env python3
"""Run the complete YOLO + CNN pipeline over every image in one directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
from torchvision.models import EfficientNet_B2_Weights

from catvision.detector import load_detectors
from catvision.runtime import load_classifier, predict_frame


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch_inference"))
    parser.add_argument("--model", type=Path, default=Path("models/classifier/best_effnet_b2_cat_breeds.pth"))
    parser.add_argument("--class-map", type=Path, default=Path("config/class_to_idx.json"))
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = sorted(path for path in args.input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images in {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    detectors = load_detectors()
    classifier, classes = load_classifier(args.model, args.class_map, device)
    transform = EfficientNet_B2_Weights.DEFAULT.transforms()
    annotated_dir = args.output_dir / "annotated"
    rows = []
    for index, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            rows.append({"image": str(image_path), "status": "unreadable"})
            continue
        annotated, prediction = predict_frame(frame, detectors, classifier, classes, transform, device, args.conf)
        relative = image_path.relative_to(args.input_dir).with_suffix(".jpg")
        output_image = annotated_dir / relative
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image), annotated)
        row = {"image": str(image_path), "annotated_image": str(output_image),
               "status": "ok" if prediction else "no_cat_detected"}
        if prediction:
            row.update(prediction)
        rows.append(row)
        print(f"{index}/{len(images)} {image_path.name}: {row['status']} {row.get('breed', '')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = ["image", "annotated_image", "status", "breed", "classification_confidence",
              "detection_confidence", "detector", "box"]
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} results to: {args.output_dir}")


if __name__ == "__main__":
    main()
