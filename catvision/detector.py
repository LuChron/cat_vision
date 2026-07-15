"""Minimal two-model cat detector used everywhere in this project."""

from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO


COCO_CAT_CLASS_ID = 15
# These are intentionally local project paths.  The weight files are not stored
# in Git; see models/README.md for the short hand-off checklist.
DEFAULT_MODELS = ("models/detector/yolo26n.pt", "models/detector/yolo11m.pt")


@dataclass
class Detection:
    result: object
    model_name: str
    threshold: float


def load_detectors(model_names: tuple[str, ...] = DEFAULT_MODELS) -> list[tuple[YOLO, str]]:
    detectors = [(YOLO(name), name) for name in model_names]
    for _, name in detectors:
        print(f"Loaded YOLO model: {name}")
    return detectors


def predict_cat(model: YOLO, frame, confidence: float, image_size: int = 640):
    return model.predict(frame, classes=[COCO_CAT_CLASS_ID], conf=confidence, imgsz=image_size, verbose=False)[0]


def detect_with_cascade(frame, detectors: list[tuple[YOLO, str]], confidence: float = 0.25) -> Detection | None:
    """Try YOLO26n first; only use YOLO11m after a genuine miss."""
    for model, name in detectors:
        result = predict_cat(model, frame, confidence)
        if len(result.boxes):
            return Detection(result, name, confidence)
    return None
