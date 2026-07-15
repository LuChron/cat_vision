#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-catvision}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda create -y -n "$ENV_NAME" python=3.11 pip
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -n "$ENV_NAME" python -m pip install -r "$PROJECT_DIR/requirements.txt"
conda run -n "$ENV_NAME" python -c "import torch, torchvision, ultralytics, cv2; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('ultralytics:', ultralytics.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"
