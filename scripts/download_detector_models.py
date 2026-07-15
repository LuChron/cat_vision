#!/usr/bin/env python3
"""Download the public YOLO detector weights required by this project."""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO


MODEL_NAMES = ("yolo26n.pt", "yolo11m.pt")
OUTPUT_DIR = Path("models/detector")


def main() -> None:
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_dir = Path.cwd()
    try:
        # Ultralytics downloads standard model names into the current directory.
        os.chdir(output_dir)
        for model_name in MODEL_NAMES:
            model = YOLO(model_name)
            print(f"Ready: {Path(model.ckpt_path).resolve()}")
    finally:
        os.chdir(previous_dir)


if __name__ == "__main__":
    main()
