#!/usr/bin/env python3
"""Test YOLO ensemble cascade: try models in order until one detects a cat.

Shows detection per model and the ensemble total.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COCO_CAT_CLASS_ID = 15
YOLO_ENSEMBLE = ["yolo26n.pt", "yolo11s.pt", "yolo11m.pt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("test_scene"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolo_check"))
    parser.add_argument("--model", default=None, help="Use a single model instead of ensemble")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = sorted(p for p in args.scene_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {args.scene_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [args.model] if args.model else YOLO_ENSEMBLE
    models: list[tuple[YOLO, str]] = []
    for name in model_names:
        try:
            models.append((YOLO(name), name))
        except Exception as e:
            print(f"Failed: {name} — {e}")
    if not models:
        raise RuntimeError("No YOLO models loaded.")

    # Stats
    cascade_025 = 0          # first model in list at 0.25
    cascade_any = 0          # cascade: try each model at 0.25, then 0.15
    ensemble_025 = 0         # any model at 0.25
    total = len(images)

    print(f"Models: {' → '.join(model_names)}")
    print(f"{'Result':42s} | {'Cascade winner':20s} | {'All at 0.25':40s}")
    print("-" * 105)

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        # Results per model at 0.25
        per_model_025: dict[str, tuple[bool, list[float]]] = {}
        for model, mname in models:
            r = model.predict(source=str(image_path), classes=[COCO_CAT_CLASS_ID],
                              conf=args.conf, imgsz=640, verbose=False)[0]
            found = len(r.boxes) > 0
            confs = [round(float(c), 4) for c in r.boxes.conf.cpu().tolist()] if found else []
            per_model_025[mname] = (found, confs)
            if found:
                ensemble_025 += 1
                break  # count once per image

        # Fix: ensemble_025 was counting per model, not per image
        # Let me recount properly below
        ensemble_025 = 0
        for model, mname in models:
            r = model.predict(source=str(image_path), classes=[COCO_CAT_CLASS_ID],
                              conf=args.conf, imgsz=640, verbose=False)[0]
            found = len(r.boxes) > 0
            per_model_025[mname] = (found, [round(float(c), 4) for c in r.boxes.conf.cpu().tolist()] if found else [])

        # Cascade at 0.25
        winner = None
        winner_conf = None
        annotated = None
        for model, mname in models:
            found, confs = per_model_025[mname]
            if found:
                winner = f"{mname}({confs[0]:.2f})"
                winner_conf = (mname, confs[0])
                r = model.predict(source=str(image_path), classes=[COCO_CAT_CLASS_ID],
                                  conf=args.conf, imgsz=640, verbose=False)[0]
                annotated = r.plot()
                # Only count if it's the first model (yolo26n)
                if mname == model_names[0]:
                    cascade_025 += 1
                break

        # Cascade at 0.15
        if winner is None:
            for model, mname in models:
                r = model.predict(source=str(image_path), classes=[COCO_CAT_CLASS_ID],
                                  conf=0.15, imgsz=640, verbose=False)[0]
                if len(r.boxes) > 0:
                    c = float(r.boxes.conf[0].item())
                    winner = f"{mname}(lo:{c:.2f})"
                    winner_conf = (mname, c)
                    annotated = r.plot()
                    cv2.putText(annotated, f"{mname} lowconf", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    break

        if winner is None:
            annotated = frame.copy()
            cv2.putText(annotated, "MISS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            winner = "MISS"

        if winner != "MISS":
            cascade_any += 1

        cv2.imwrite(str(args.output_dir / f"{image_path.stem}_annotated.jpg"), annotated)

        # Summarize which models found at 0.25
        all_025 = "  ".join(
            f"{m}:{confs[0]:.2f}" if found else f"{m}:-"
            for m, (found, confs) in per_model_025.items()
        )

        print(f"{'✓' if winner != 'MISS' else '✗'} {image_path.name:35s} | {winner or 'MISS':20s} | {all_025}")

    # Ensemble at 0.25: any model found
    ensemble_025 = 0
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        for model, mname in models:
            r = model.predict(source=str(image_path), classes=[COCO_CAT_CLASS_ID],
                              conf=args.conf, imgsz=640, verbose=False)[0]
            if len(r.boxes) > 0:
                ensemble_025 += 1
                break

    print(f"\n{'='*60}")
    print(f"Cascade (yolo26n only, 0.25): {cascade_025}/{total} = {cascade_025/total:.0%}")
    print(f"Ensemble (any model, 0.25):   {ensemble_025}/{total} = {ensemble_025/total:.0%}")
    print(f"Cascade (all → lowconf):      {cascade_any}/{total} = {cascade_any/total:.0%}")


if __name__ == "__main__":
    main()
