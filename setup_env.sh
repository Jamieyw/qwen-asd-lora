#!/bin/bash
###############################################################################
# setup_env.sh — Set up conda environment on the cluster
#
# Run this ONCE on the cluster (in an interactive session, NOT on login node):
#   srun --partition=short --mem=8G --time=01:00:00 --pty bash
#   bash setup_env.sh
#
# This installs all dependencies including the special transformers build
# needed for Qwen2.5-Omni support.
###############################################################################

set -e  # Exit on any error

ENV_NAME="qwen-asd"

echo "========================================"
echo "Setting up conda environment: $ENV_NAME"
echo "========================================"

# Load conda module (adjust to your cluster)
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true

# Remove existing environment if it exists (uncomment to force reinstall)
# conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create fresh environment
echo "Creating conda environment with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# Activate it
source activate $ENV_NAME 2>/dev/null || conda activate $ENV_NAME

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the special transformers build for Qwen2.5-Omni
# This is REQUIRED — the standard PyPI transformers does NOT support Qwen2.5-Omni
echo ""
echo "Installing transformers (Qwen2.5-Omni preview build)..."
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview

# Install other dependencies
echo ""
echo "Installing remaining dependencies..."
pip install accelerate>=0.27.0
pip install "peft>=0.10.0,<0.14.0"
pip install "qwen-omni-utils[decord]"
pip install datasets>=2.18.0
pip install pandas numpy
pip install soundfile librosa Pillow
pip install scikit-learn
pip install tqdm

# Verify installation
echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import transformers
print(f'Transformers: {transformers.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

import accelerate
print(f'Accelerate: {accelerate.__version__}')

from transformers import Qwen2_5OmniForConditionalGeneration
print('Qwen2.5-Omni model class: OK')

print()
print('All dependencies installed successfully!')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo "Activate with: conda activate $ENV_NAME"
echo "========================================"
