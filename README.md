# Cat Vision Pipeline

This is a local five-class cat-breed recognition project. It recognises `pallas`, `persian`, `ragdoll`, `singapura`, and `sphynx`.

```text
input image or camera frame
  -> YOLO26n detects a cat (confidence 0.25)
  -> if it misses, YOLO11m retries (confidence 0.25)
  -> crop the detected cat with an 8% margin
  -> EfficientNet-B2 predicts the breed
```

The detector uses only this two-model cascade. If neither YOLO model detects a cat, the system reports `No cat detected`; it does not classify the background.

## Quick start

The two YOLO detector weights and the fine-tuned CNN checkpoint are included in Git.

Create the environment once on a CUDA machine:

```bash
bash setup_catvision_env.sh
conda activate catvision
```

Run one image from the project root:

```bash
python -m catvision.runtime --image /path/to/image.jpg
```

The annotated image is written to `outputs/runtime/pipeline_result.jpg`. Camera/ROS integration is outside this repository; the external program can pass saved frames to this script or reuse `predict_frame()` from `catvision.runtime`.

## Batch test

Put test images in any folder, then run:

```bash
python -m catvision.batch_infer --input-dir /path/to/images --output-dir outputs/batch_run
```

It writes annotated images to `outputs/batch_run/annotated/` and one result per image to `results.csv` and `results.json`.

## Current validated result

On the current 12 on-site camera snapshots, YOLO26n alone detected 10/12 cats; the YOLO26n -> YOLO11m cascade detected 12/12 at confidence `0.25`. This is a small local test set, not a guarantee for every lighting condition or camera angle.

The EfficientNet-B2 checkpoint was trained for 15 epochs on automatically YOLO-cropped images. Its best validation accuracy was **98.92%** on the current split. The Pallas validation set is small, so more varied real-scene examples would improve confidence in that class.

## Repository layout

```text
catvision/       Detector, single-image inference, and batch inference
scripts/         Dataset audit, automatic crop, split, training, validation, YOLO check
config/          Versioned class-index mapping used by the checkpoint
models/          Downloaded YOLO weights and the versioned CNN checkpoint
data/            Local raw/cropped/split datasets; excluded from Git
outputs/         Generated visualisations, reports, and experiment results; excluded from Git
```

## Development and training commands

All commands must be run from the project root after `conda activate catvision`.

Check the detector on camera snapshots:

```bash
python -m scripts.verify_yolo --scene-dir test_scene
```

Audit one labelled source folder without modifying it:

```bash
python -m scripts.audit_detection_images \
  --input-dir data/raw/pallas \
  --output-dir outputs/pallas_audit --overwrite
```

Create CNN training crops automatically, then make a reproducible 85/15 split:

```bash
python -m scripts.crop_training_cats --raw-dir data/raw --output-dir data/cropped --overwrite
python -m scripts.prepare_dataset --raw-dir data/cropped --output-dir data/split --min-per-class 1 --overwrite
```

Train and evaluate the CNN:

```bash
python -m scripts.train_classifier --data-dir data/split --output-dir outputs/classifier --epochs 15 --batch-size 16
python -m scripts.validate_classifier --data-dir data/split/val
```

Training applies online augmentation only to the training split: small perspective and affine changes, blur, and lighting/contrast changes. It also includes occasional overexposure (brighter, lower-contrast, lower-saturation images) so white-fur details are less sensitive to camera auto-exposure. Validation images are never augmented.

After retraining, copy `outputs/classifier/best_effnet_b2_cat_breeds.pth` to `models/classifier/best_effnet_b2_cat_breeds.pth`. Keep `config/class_to_idx.json` aligned with the new checkpoint's `class_to_idx` field before deployment.

## Git policy

Source code, the class mapping, setup script, documentation, and selected model weights are versioned. Datasets, snapshots, generated outputs, Python caches, notebooks, and older experimental weights are intentionally ignored. This keeps the repository small and prevents accidentally committing multi-gigabyte data.
