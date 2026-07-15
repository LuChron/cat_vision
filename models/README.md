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

All three weights above are versioned with this repository. `config/class_to_idx.json` must stay paired with `best_effnet_b2_cat_breeds.pth`.
