#!/usr/bin/env python3
"""Use pretrained YOLO to crop the cat region from each labelled source image.

The source folder name remains the breed label. Images where YOLO cannot find a
cat are reported and deliberately excluded: silently training on them would make
the CNN's training input differ from its robot-time input.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASSES = ["ragdoll", "singapura", "persian", "sphynx", "pallas"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COCO_CAT_CLASS_ID = 15
# Ensemble cascade: try larger models if nano misses
YOLO_CASCADE = ["yolo26n.pt", "yolo11s.pt", "yolo11m.pt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cropped"))
    parser.add_argument("--report", type=Path, default=Path("outputs/crop_report.json"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def crop_with_margin(image, box, margin: float):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(width, x2 + dx), min(height, y2 + dy)
    return image[y1:y2, x1:x2]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Use --overwrite to rebuild it.")
        shutil.rmtree(args.output_dir)

    # Load ensemble: specify a model or use full cascade
    model_names = [args.model] if args.model else YOLO_CASCADE
    models = [(YOLO(name), name) for name in model_names]
    print(f"Detection cascade: {' → '.join(model_names)}")

    report: dict[str, dict[str, object]] = {}
    for class_name in CLASSES:
        source_dir = args.raw_dir / class_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {source_dir}")
        source_images = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        target_dir = args.output_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        missed: list[str] = []
        saved = 0
        for image_path in source_images:
            image = cv2.imread(str(image_path))
            if image is None:
                missed.append(f"{image_path} (unreadable)")
                continue

            # Cascade: try each model until one detects a cat
            best_box = None
            for model, mname in models:
                result = model.predict(image, classes=[COCO_CAT_CLASS_ID], conf=args.conf,
                                       imgsz=640, verbose=False)[0]
                if len(result.boxes) > 0:
                    best_index = int(result.boxes.conf.argmax().item())
                    best_box = result.boxes.xyxy[best_index].cpu().numpy()
                    if mname != model_names[0]:
                        print(f"  Cascade fallback {mname} → {image_path.name}")
                    break

            if best_box is None:
                missed.append(str(image_path))
                continue

            crop = crop_with_margin(image, best_box, args.margin)
            if crop.size == 0:
                missed.append(f"{image_path} (empty crop)")
                continue
            cv2.imwrite(str(target_dir / f"{class_name}_{saved:04d}.jpg"), crop)
            saved += 1
        report[class_name] = {"input_images": len(source_images), "cropped_images": saved, "missed_images": missed}
        print(f"{class_name}: cropped {saved}/{len(source_images)}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Crop report: {args.report}")


if __name__ == "__main__":
    main()
