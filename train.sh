#!/bin/bash
###############################################################################
# train.sh — SLURM sbatch script for LoRA fine-tuning Qwen2.5-Omni-3B
#
# Submit with:  sbatch train.sh
# Monitor with: squeue -u $USER
# View output:  tail -f logs/<job_id>.out
# Cancel:       scancel <job_id>
###############################################################################

#SBATCH --job-name=qwen-asd            # Job name (shows in squeue)
#SBATCH --partition=gpu                 # GPU partition — adjust if your cluster uses a different name
#SBATCH --gres=gpu:a100:1              # Request 1 A100 GPU (40GB). Fastest available on Explorer
#SBATCH --mem=64G                       # RAM (need enough for data loading + model)
#SBATCH --cpus-per-task=4               # CPU cores for data loading workers
#SBATCH --time=04:00:00                 # 4 hour time limit
#SBATCH --output=logs/%j.out            # stdout log (%j = job ID)
#SBATCH --error=logs/%j.err             # stderr log

###############################################################################
# Environment setup
###############################################################################

# Create logs directory (in case it doesn't exist)
mkdir -p logs

# Print job info for debugging
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

# Load modules
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || echo "No cuda module found"
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || echo "No conda module found"

# Properly initialize and activate conda
# (sbatch doesn't source .bashrc, so conda activate won't work without this)
eval "$(conda shell.bash hook)"
conda activate qwen-asd

# Print environment info
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Check GPU memory
nvidia-smi

###############################################################################
# Run training
###############################################################################

echo ""
echo "========================================"
echo "Starting training..."
echo "========================================"

python train.py \
    --model_name "Qwen/Qwen2.5-Omni-3B" \
    --data_dir ./asd-data \
    --output_dir ./output \
    --epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --logging_steps 10 \
    --save_steps 500 \
    --timing_test_steps 50 \
    --gradient_checkpointing

echo ""
echo "========================================"
echo "Training finished at: $(date)"
echo "Exit code: $?"
echo "========================================"
