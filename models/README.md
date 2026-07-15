# Model files

The fine-tuned CNN checkpoint is versioned in Git:

```text
models/
├── detector/
│   ├── yolo26n.pt
│   └── yolo11m.pt
└── classifier/
    └── best_effnet_b2_cat_breeds.pth
```

After cloning the repository, download the two public YOLO files with:

```bash
python -m scripts.download_detector_models
```

`config/class_to_idx.json` and `best_effnet_b2_cat_breeds.pth` are versioned with the source code and must stay paired. The YOLO files are excluded from Git because they are downloaded public weights.
