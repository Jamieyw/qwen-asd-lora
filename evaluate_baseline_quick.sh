#!/bin/bash
#SBATCH --job-name=qwen-quick
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/baseline_quick_%j.out
#SBATCH --error=logs/baseline_quick_%j.err

mkdir -p logs

echo "========================================"
echo "Quick Baseline Test (50 samples) Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate qwen-asd

echo "Evaluating base Qwen2.5-Omni-3B (no LoRA, 50 samples only)..."
python evaluate.py \
    --model_name "Qwen/Qwen2.5-Omni-3B" \
    --data_dir ./asd-data \
    --output_dir ./output \
    --no_adapter \
    --max_samples 50

echo ""
echo "========================================"
echo "Quick baseline finished at: $(date)"
echo "========================================"
