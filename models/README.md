# Model files (not committed to Git)

Place these three files here before running the robot pipeline:

```text
models/
├── detector/
│   ├── yolo26n.pt
│   └── yolo11m.pt
└── classifier/
    └── best_effnet_b2_cat_breeds.pth
```

`config/class_to_idx.json` is versioned with the source code and must stay paired with the classifier checkpoint. Model weights are intentionally excluded from Git because they are large generated/downloaded artifacts. Transfer the three files through the team shared drive or a USB drive.
