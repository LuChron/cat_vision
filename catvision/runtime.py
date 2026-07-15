#!/usr/bin/env python3
"""Run the validated YOLO cascade followed by EfficientNet-B2 breed classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from catvision.detector import detect_with_cascade, load_detectors


def build_classifier(num_classes: int) -> nn.Module:
    model = efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_classifier(model_path: Path, class_map_path: Path, device: torch.device):
    class_to_idx = json.loads(class_map_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and checkpoint.get("arch") not in (None, "efficientnet_b2"):
        raise ValueError(f"Expected EfficientNet-B2 checkpoint, got {checkpoint.get('arch')}")
    classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
    if not classes:
        classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    model = build_classifier(len(classes))
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    return model.to(device).eval(), classes


def crop_with_margin(frame, box, margin: float = 0.08):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1, x2, y2 = max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def predict_frame(frame, detectors, classifier, classes, transform, device, conf):
    detection = detect_with_cascade(frame, detectors, conf)
    if detection is None:
        return frame, None
    best_index = int(detection.result.boxes.conf.argmax().item())
    box = detection.result.boxes.xyxy[best_index].cpu().numpy()
    det_conf = float(detection.result.boxes.conf[best_index].item())
    crop, (x1, y1, x2, y2) = crop_with_margin(frame, box)
    tensor = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(classifier(tensor), dim=1)[0]
    class_index = int(probabilities.argmax().item())
    class_confidence = float(probabilities[class_index].item())
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{classes[class_index]} cls={class_confidence:.2f} det={det_conf:.2f}",
                (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return frame, {"breed": classes[class_index], "classification_confidence": class_confidence,
                   "detection_confidence": det_conf, "detector": detection.model_name, "box": [x1, y1, x2, y2]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--camera", type=int)
    parser.add_argument("--model", type=Path, default=Path("models/classifier/best_effnet_b2_cat_breeds.pth"))
    parser.add_argument("--class-map", type=Path, default=Path("config/class_to_idx.json"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("pipeline_result.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    detectors = load_detectors()
    classifier, classes = load_classifier(args.model, args.class_map, device)
    transform = EfficientNet_B2_Weights.DEFAULT.transforms()
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise FileNotFoundError(args.image)
        annotated, prediction = predict_frame(frame, detectors, classifier, classes, transform, device, args.conf)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), annotated):
            raise RuntimeError(f"Could not write output image: {args.output}")
        print("Prediction:", prediction if prediction else "No cat detected")
        print("Saved:", args.output)
        return
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    print("Press q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        annotated, prediction = predict_frame(frame, detectors, classifier, classes, transform, device, args.conf)
        if prediction:
            print(prediction)
        cv2.imshow("YOLO cascade + EfficientNet-B2", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
