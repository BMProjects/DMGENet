#!/usr/bin/env bash
# setup_env.sh — create the Python environment used by the DMGENet repository
# Managed with uv; installs the CUDA 12.8 PyTorch build (cu128)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "[setup] Creating virtual environment with uv: $VENV_DIR"
uv venv --python 3.10 "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"

echo "[setup] Installing PyTorch (cu128)..."
uv pip install --python "$PYTHON" \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo "[setup] Installing remaining dependencies..."
uv pip install --python "$PYTHON" \
    numpy pandas scipy scikit-learn \
    matplotlib requests dtaidistance thop

echo "[setup] Verifying the installation..."
$PYTHON -c "
import torch, numpy, pandas, scipy, sklearn
print(f'  torch  {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'  numpy  {numpy.__version__}')
print(f'  pandas {pandas.__version__}')
print('  scipy / sklearn ok')
"

echo "[setup] Done. Python executable: $PYTHON"
