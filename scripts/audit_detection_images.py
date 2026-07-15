#!/usr/bin/env python3
"""Audit one labelled image folder before CNN training without changing its source files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from catvision.detector import detect_with_cascade, load_detectors


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="One labelled class folder, e.g. data/raw/pallas")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--review-below", type=float, default=0.40,
                        help="Also copy detected images whose winning detection confidence is below this value")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_for_review(source: Path, destination: Path, index: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination / f"{index:04d}_{source.name}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already exists; use --overwrite to rebuild the review.")
        shutil.rmtree(args.output_dir)
    images = sorted(path for path in args.input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No image files found in {args.input_dir}")

    detectors = load_detectors()
    rows = []
    for index, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            copy_for_review(image_path, args.output_dir / "unreadable", index)
            rows.append({"file": str(image_path), "status": "unreadable"})
            continue
        detection = detect_with_cascade(frame, detectors, args.conf)
        if detection is None:
            copy_for_review(image_path, args.output_dir / "missed", index)
            rows.append({"file": str(image_path), "status": "missed"})
            continue
        best_index = int(detection.result.boxes.conf.argmax().item())
        score = float(detection.result.boxes.conf[best_index].item())
        status = "detected" if score >= args.review_below else "low_confidence"
        if status == "low_confidence":
            copy_for_review(image_path, args.output_dir / "low_confidence", index)
        rows.append({"file": str(image_path), "status": status, "detector": detection.model_name,
                     "threshold": detection.threshold, "confidence": score})
        print(f"{index:04d}/{len(images)} {status:14s} {score:.2f} {image_path.name}")

    summary = {status: sum(row["status"] == status for row in rows)
               for status in ("detected", "low_confidence", "missed", "unreadable")}
    report = {"input_dir": str(args.input_dir), "normal_threshold": args.conf,
              "review_below": args.review_below,
              "total": len(rows), "summary": summary, "images": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Summary:", summary)
    print("Review folder:", args.output_dir)


if __name__ == "__main__":
    main()
