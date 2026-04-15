#!/bin/bash
#SBATCH --job-name=qwen-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

mkdir -p logs

echo "========================================"
echo "Baseline Evaluation Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate qwen-asd

echo "Evaluating base Qwen2.5-Omni-3B (no LoRA adapter)..."
python evaluate.py \
    --model_name "Qwen/Qwen2.5-Omni-3B" \
    --data_dir ./asd-data \
    --output_dir ./output \
    --no_adapter

echo ""
echo "========================================"
echo "Baseline evaluation finished at: $(date)"
echo "========================================"
