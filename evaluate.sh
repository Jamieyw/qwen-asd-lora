#!/bin/bash
###############################################################################
# evaluate.sh — SLURM sbatch script for evaluating the fine-tuned model
#
# Submit with:  sbatch evaluate.sh
# Or run interactively with:
#   srun --partition=gpu --gres=gpu:a100:1 --mem=32G --time=00:30:00 --pty bash
#   conda activate qwen-asd
#   python evaluate.py
###############################################################################

#SBATCH --job-name=qwen-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

mkdir -p logs

echo "========================================"
echo "Evaluation Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null
source activate qwen-asd 2>/dev/null || conda activate qwen-asd

nvidia-smi

# Run evaluation with LoRA adapter
echo ""
echo "Evaluating fine-tuned model..."
python evaluate.py \
    --model_name "Qwen/Qwen2.5-Omni-3B" \
    --adapter_path ./output/best_model \
    --data_dir ./asd-data \
    --output_dir ./output \

# Run baseline evaluation (without adapter) for comparison
echo ""
echo "Evaluating baseline (no adapter)..."
python evaluate.py \
    --model_name "Qwen/Qwen2.5-Omni-3B" \
    --data_dir ./asd-data \
    --output_dir ./output \
    --fp16 \
    --no_adapter

echo ""
echo "========================================"
echo "Evaluation finished at: $(date)"
echo "========================================"
