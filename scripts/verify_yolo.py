#!/usr/bin/env python3
"""Measure the simple YOLO26n -> YOLO11m cat-detection cascade on scene images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from catvision.detector import DEFAULT_MODELS, detect_with_cascade, load_detectors, predict_cat


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("test_scene"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolo_check"))
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = sorted(path for path in args.scene_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {args.scene_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detectors = load_detectors()
    primary_hits = cascade_hits = 0
    rows = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        primary_result = predict_cat(detectors[0][0], frame, args.conf)
        primary_score = None if not len(primary_result.boxes) else float(primary_result.boxes.conf.max().item())
        primary_hits += primary_score is not None
        detection = detect_with_cascade(frame, detectors, args.conf)
        if detection is None:
            annotated = frame.copy()
            cv2.putText(annotated, "MISS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            winner, score = None, None
        else:
            cascade_hits += 1
            winner = detection.model_name
            score = float(detection.result.boxes.conf.max().item())
            annotated = detection.result.plot()
        cv2.imwrite(str(args.output_dir / f"{path.stem}_annotated.jpg"), annotated)
        rows.append({"image": str(path), "yolo26n_score": primary_score, "winner": winner, "winner_score": score})
        print(f"{'OK' if detection else 'MISS':4s} {path.name:35s} winner={winner} score={score}")
    total = len(rows)
    report = {"models": list(DEFAULT_MODELS), "threshold": args.conf, "total_scenes": total,
              "yolo26n_detections": primary_hits, "cascade_detections": cascade_hits, "scenes": rows}
    report_path = args.output_dir / "yolo_detection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"YOLO26n: {primary_hits}/{total} = {primary_hits / total:.0%}")
    print(f"Two-model cascade: {cascade_hits}/{total} = {cascade_hits / total:.0%}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
