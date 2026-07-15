# Robot cat vision: local Python workflow

This project implements the chosen baseline design:

```text
robot camera scene -> pretrained YOLO26n finds a cat -> crop -> fine-tuned EfficientNet-B2 predicts breed
                           │ (miss)
                           └─ YOLO(0.15) → edge contours → center-crop CNN
```

YOLO is **not trained**. It is only a proposal detector. The model you fine-tune and report is the five-class EfficientNet-B2 CNN.

YOLO detection has a **4-tier fallback** strategy so no cat is missed: YOLO(0.25) → YOLO(0.15) → contour-based rectangle detection → center-crop CNN.

## 1. Install a separate GPU environment

Do not change the existing `nus` environment: it has no PyTorch, Ultralytics or OpenCV. From this folder run:

```bash
bash setup_catvision_env.sh
conda activate catvision
```

The script uses Python 3.11 and the current PyTorch CUDA 12.8 wheel. It ends by confirming `CUDA available: True` and the RTX 4070 GPU.

## 2. Put data in these folders

```text
data/raw/ragdoll/
data/raw/singapura/
data/raw/persian/
data/raw/sphynx/
data/raw/pallas/
test_scene/                         # 20-30 real robot-view maze photos
```

Collect at least 220-250 **unique** images per class so that, after YOLO misses a few, you still retain at least 200 cropped images per class. Do not mix near-identical copies between classes or between train/validation data.

## 3. Test the key YOLO assumption first

```bash
python verify_yolo.py --scene-dir test_scene
```

The first run downloads `yolo26n.pt`. Inspect `outputs/yolo_check/*_annotated.jpg` and `yolo_detection_report.json`.

If YOLO misses any, the 4-tier fallback in `robot_realtime_infer.py` will attempt contour detection and center-crop CNN classification, so a miss here doesn't mean a miss in the competition.

## 4. Automatically crop the CNN training images

```bash
python crop_training_cats.py --raw-dir data/raw --output-dir data/cropped
```

This applies the same YOLO cat detector used on the robot, selects the highest-confidence cat box, and adds an 8% margin. Inspect `outputs/crop_report.json`: collect extra images for any class that has fewer than 200 successful crops. You do **not** draw boxes or crop by hand.

## 5. Create the required 85/15 split

```bash
python prepare_dataset.py --raw-dir data/cropped --output-dir data/split
```

This removes byte-identical duplicates before splitting, then copies the result to `data/split/train/` and `data/split/val/`. It refuses to proceed below 200 unique images per class.

## 6. Fine-tune EfficientNet-B2 locally

```bash
python train_classifier.py --data-dir data/split --output-dir outputs/classifier --epochs 15 --batch-size 16
```

The RTX 4070 Laptop GPU has 8 GB VRAM, which is sufficient for EfficientNet-B2 at 260 px. Batch size 16 fits comfortably. Outputs include the checkpoint, class map, JSON results, and training curves.

## 7. Run the combined robot pipeline (with 4-tier fallback)

Multi-tier detection strategy: `YOLO(conf=0.25) → YOLO(conf=0.15) → edge/contour rectangle detection → center-crop CNN (last resort)`.

```bash
python robot_realtime_infer.py --image test_scene/scene_001.jpg \
  --model outputs/classifier/best_effnet_b2_cat_breeds.pth \
  --class-map outputs/classifier/class_to_idx.json

python robot_realtime_infer.py --camera 0 \
  --model outputs/classifier/best_effnet_b2_cat_breeds.pth \
  --class-map outputs/classifier/class_to_idx.json
```

Press `q` to stop camera inference.

## Report evidence to retain

- `data/split/dataset_counts.json`: image counts for the answer book.
- `outputs/classifier/training_results.json`: training and validation accuracy (EfficientNet-B2).
- `outputs/classifier/training_curves.png`: overfitting/underfitting evidence.
- YOLO scene-check images: evidence for or against the detection stage in the robot demo.
- The 4-tier fallback (contour/center-crop) ensures YOLO misses don't cause cat misses in the demo.
