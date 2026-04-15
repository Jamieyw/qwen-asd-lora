#!/bin/bash
###############################################################################
# train_v2.sh — SLURM sbatch script for improved LoRA fine-tuning
#
# Changes from train.sh:
#   - Uses train_v2.py (label smoothing + audio encoder LoRA + tuned hyperparams)
#   - Lower LR (5e-5), higher rank (16), more dropout (0.1)
#   - Label smoothing 0.1 to prevent overconfident predictions
#   - LoRA on last 8 audio encoder layers with 10x lower LR
#
# Submit with:  sbatch train_v2.sh
# Monitor with: squeue -u $USER
# View output:  tail -f logs/<job_id>.out
###############################################################################

#SBATCH --job-name=qwen-asd-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

###############################################################################
# Environment setup
###############################################################################

mkdir -p logs

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || echo "No cuda module found"
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || echo "No conda module found"

eval "$(conda shell.bash hook)"
conda activate qwen-asd

echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

nvidia-smi

###############################################################################
# Run training (v2)
###############################################################################

echo ""
echo "========================================"
echo "Starting training (v2 — label smoothing + audio encoder LoRA)..."
echo "========================================"

python train_v2.py \
    --model_name "Qwen/Qwen2.5-Omni-3B" \
    --data_dir ./asd-data \
    --output_dir ./output \
    --epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_steps 50 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smoothing 0.1 \
    --unfreeze_audio_layers 8 \
    --audio_lr_scale 0.1 \
    --logging_steps 10 \
    --save_steps 500 \
    --timing_test_steps 50 \
    --gradient_checkpointing

echo ""
echo "========================================"
echo "Training finished at: $(date)"
echo "Exit code: $?"
echo "========================================"
