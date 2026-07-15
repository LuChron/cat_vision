#!/usr/bin/env python3
"""
Run YOLO cat detection + EfficientNet-B0 breed classification.

Examples:
  python robot_realtime_infer.py --image robot_frame.jpg
  python robot_realtime_infer.py --camera 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b2
from ultralytics import YOLO


COCO_CAT_CLASS_ID = 15
# Ensemble of YOLO models — each complements the other's misses
YOLO_ENSEMBLE = ["yolo26n.pt", "yolo11s.pt", "yolo11m.pt"]
# Minimum classification confidence for fallback predictions
FALLBACK_CLS_CONF = 0.6


def load_yolo_ensemble(custom_model: str | None = None) -> list[tuple[YOLO, str]]:
    """Load the YOLO ensemble. Returns list of (model, name) pairs."""
    names = [custom_model] if custom_model else YOLO_ENSEMBLE
    models: list[tuple[YOLO, str]] = []
    for name in names:
        try:
            model = YOLO(name)
            models.append((model, name))
            print(f"Loaded YOLO model: {name}")
        except Exception as exc:
            print(f"Failed to load YOLO model {name}: {exc}")
    if not models:
        raise RuntimeError("Could not load any YOLO model from the ensemble list.")
    return models")


def build_classifier(num_classes: int) -> nn.Module:
    model = efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_classifier(model_path: Path, class_map_path: Path, device: torch.device):
    with class_map_path.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = build_classifier(num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names


def crop_with_margin(frame, xyxy, margin: float = 0.08):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * margin)
    dy = int(bh * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize((288, 288)),
            transforms.CenterCrop(260),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ── Fallback detection helpers ──────────────────────────────────────────────

def classify_crop(crop_bgr, classifier, class_names, transform, device) -> tuple[str | None, float]:
    """Run classifier on a BGR crop. Returns (breed, confidence) or (None, 0)."""
    if crop_bgr is None or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
        return None, 0.0
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            conf = float(probs[pred_idx].item())
        return class_names[pred_idx], conf
    except Exception:
        return None, 0.0


def detect_by_contours(frame, classifier, class_names, transform, device,
                       min_cls_conf: float = FALLBACK_CLS_CONF):
    """Find rectangular photo-like regions via edge detection, classify each.

    Returns annotated frame and prediction dict, or (frame, None).
    """
    candidates: list[tuple[int, int, int, int]] = []
    h, w = frame.shape[:2]
    total_area = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for low, high in [(30, 100), (50, 150)]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, low, high)
        dilated = cv2.dilate(edged, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if 4 <= len(approx) <= 10:
                x, y, cw, ch = cv2.boundingRect(approx)
                area = cw * ch
                if area < 0.01 * total_area or area > 0.90 * total_area:
                    continue
                aspect = cw / max(ch, 1)
                if 0.4 < aspect < 2.5:
                    candidates.append((x, y, x + cw, y + ch))

    best_breed, best_conf, best_box = None, 0.0, None
    for x1, y1, x2, y2 in candidates:
        crop = frame[y1:y2, x1:x2]
        breed, conf = classify_crop(crop, classifier, class_names, transform, device)
        if breed and conf > best_conf:
            best_breed, best_conf, best_box = breed, conf, (x1, y1, x2, y2)

    if best_breed and best_conf >= min_cls_conf:
        x1, y1, x2, y2 = best_box
        label = f"{best_breed} contour cls={best_conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        return frame, {
            "breed": best_breed, "classification_confidence": best_conf,
            "detection_confidence": None, "method": "contour", "box": [x1, y1, x2, y2],
        }
    return frame, None


def classify_center_crop(frame, classifier, class_names, transform, device,
                         min_cls_conf: float = FALLBACK_CLS_CONF):
    """Last resort: classify a center crop of the whole frame."""
    h, w = frame.shape[:2]
    margin = 0.30
    x1 = int(w * margin)
    y1 = int(h * margin)
    x2 = int(w * (1 - margin))
    y2 = int(h * (1 - margin))
    crop = frame[y1:y2, x1:x2]
    breed, conf = classify_crop(crop, classifier, class_names, transform, device)
    if breed and conf >= min_cls_conf:
        label = f"{breed} center cls={conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return frame, {
            "breed": breed, "classification_confidence": conf,
            "detection_confidence": None, "method": "center_crop", "box": [x1, y1, x2, y2],
        }
    return frame, None


def predict_frame(frame, detectors, classifier, class_names, transform, device, conf: float):
    """Multi-tier detection: try each model in cascade; first detection wins."""
    # ── Tier 1: YOLO at conf=0.25, cascade through all ensemble models ──
    for detector, model_name in detectors:
        results = detector.predict(frame, classes=[COCO_CAT_CLASS_ID], conf=conf,
                                   imgsz=640, verbose=False)
        if len(results[0].boxes) > 0:
            return _annotate_yolo(frame, results[0], classifier, class_names,
                                  transform, device, model_name)

    # ── Tier 2: YOLO at lowered threshold, cascade again ──
    for detector, model_name in detectors:
        results = detector.predict(frame, classes=[COCO_CAT_CLASS_ID], conf=0.15,
                                   imgsz=640, verbose=False)
        if len(results[0].boxes) > 0:
            return _annotate_yolo(frame, results[0], classifier, class_names,
                                  transform, device, f"{model_name}_lowconf")

    # ── Tier 3: contour-based rectangle search ──
    annotated, pred = detect_by_contours(frame, classifier, class_names, transform, device)
    if pred:
        return annotated, pred

    # ── Tier 4: center-crop (last resort) ──
    annotated, pred = classify_center_crop(frame, classifier, class_names, transform, device)
    if pred:
        return annotated, pred

    return frame, None


def _annotate_yolo(frame, result, classifier, class_names, transform, device, method: str):
    """Run classification on the highest-confidence YOLO cat box and annotate."""
    boxes = result.boxes
    best_i = int(boxes.conf.argmax().item())
    xyxy = boxes.xyxy[best_i].cpu().numpy()
    det_conf = float(boxes.conf[best_i].item())

    crop, (x1, y1, x2, y2) = crop_with_margin(frame, xyxy)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    image_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        cls_conf = float(probs[pred_idx].item())

    breed = class_names[pred_idx]
    label = f"{breed} cls={cls_conf:.2f} det={det_conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, {
        "breed": breed, "classification_confidence": cls_conf,
        "detection_confidence": det_conf, "method": method, "box": [x1, y1, x2, y2],
    }


def run_image(args, detectors, classifier, class_names, transform, device):
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    annotated, pred = predict_frame(
        frame,
        detectors,
        classifier,
        class_names,
        transform,
        device,
        args.conf,
    )
    cv2.imwrite(str(args.output), annotated)
    print("Prediction:", pred if pred else "No cat detected")
    print("Saved:", args.output)


def run_camera(args, detectors, classifier, class_names, transform, device):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        annotated, pred = predict_frame(
            frame,
            detectors,
            classifier,
            class_names,
            transform,
            device,
            args.conf,
        )
        if pred:
            print(pred)

        cv2.imshow("YOLO + CNN cat breed inference", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to one scene image")
    source.add_argument("--camera", type=int, help="Camera index, usually 0")
    parser.add_argument("--model", type=Path, default=Path("outputs/classifier/best_effnet_b2_cat_breeds.pth"))
    parser.add_argument("--class-map", type=Path, default=Path("outputs/classifier/class_to_idx.json"))
    parser.add_argument("--yolo", default=None, help="YOLO weight name/path, e.g. yolo11s.pt (uses ensemble by default)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence")
    parser.add_argument("--output", type=Path, default=Path("pipeline_result.jpg"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    detectors = load_yolo_ensemble(args.yolo)
    classifier, class_names = load_classifier(args.model, args.class_map, device)
    transform = make_transform()

    if args.image:
        run_image(args, detectors, classifier, class_names, transform, device)
    else:
        run_camera(args, detectors, classifier, class_names, transform, device)


if __name__ == "__main__":
    main()
