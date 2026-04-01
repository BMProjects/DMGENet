#!/usr/bin/env bash
# setup_env.sh — 创建 DMGENet 所需的 Python 虚拟环境
# 使用 uv 管理，CUDA 12.8 对应 PyTorch cu128
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "[setup] 使用 uv 创建虚拟环境: $VENV_DIR"
uv venv --python 3.10 "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"

echo "[setup] 安装 PyTorch (cu128)..."
uv pip install --python "$PYTHON" \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo "[setup] 安装其他依赖..."
uv pip install --python "$PYTHON" \
    numpy pandas scipy scikit-learn \
    matplotlib requests dtaidistance thop

echo "[setup] 验证安装..."
$PYTHON -c "
import torch, numpy, pandas, scipy, sklearn
print(f'  torch  {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'  numpy  {numpy.__version__}')
print(f'  pandas {pandas.__version__}')
print('  scipy / sklearn ok')
"

echo "[setup] 完成! Python: $PYTHON"
