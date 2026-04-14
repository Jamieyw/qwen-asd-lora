# LoRA Fine-Tuning Qwen2.5-Omni-3B for Active Speaker Detection

CS6140 Machine Learning — Final Project

Fine-tune [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) using LoRA on the [UniTalk-ASD](https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD) dataset to detect which person in a video is currently speaking, based on face images and audio.

## Overview

**Active Speaker Detection (ASD)** is the task of determining whether a visible person in a video is the one producing the speech. This project fine-tunes a multimodal language model to solve this as a binary classification task (SPEAKING vs NOT_SPEAKING).

**Why Qwen2.5-Omni-3B?** It is the only Qwen model that processes both video/images and audio simultaneously, using TMRoPE (Time-aligned Multimodal RoPE) to synchronize visual and audio inputs — exactly what ASD requires.

**Why LoRA?** Full fine-tuning of a 3B parameter model requires ~48GB of GPU memory. LoRA freezes the original weights and trains small adapter matrices (~0.1% of parameters), reducing memory to ~20-27GB and fitting on a single V100.

## Project Structure

```
.
├── README.md              # This file
├── PROJECT_PLAN.md        # Detailed project plan
├── LORA_GUIDE.md          # Step-by-step LoRA explanation
├── requirements.txt       # Python dependencies
├── setup_env.sh           # Conda environment setup for cluster
├── prepare_data.py        # Download and subset the dataset
├── train.py               # LoRA fine-tuning script
├── train.sh               # SLURM sbatch job for training
├── evaluate.py            # Evaluation with metrics
└── evaluate.sh            # SLURM sbatch job for evaluation
```

## Dataset

[UniTalk-ASD](https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD) contains ~48,693 face tracks from videos, each with:
- **Face crops** (JPG at 25fps) — sequential images of one person's face
- **Audio** (WAV) — the corresponding audio segment
- **Labels** — per-frame SPEAKING_AUDIBLE (1) or NOT_SPEAKING (0)

We use **~2,000 training tracks (~5%)** with 10 frames each, plus 500 validation tracks, to keep training under 4 hours.

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | ~3M / 3B (~0.1%) |

## Quick Start (Local Testing)

```bash
# Install dependencies
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install -r requirements.txt

# Prepare a small data subset
python prepare_data.py --num_samples 100 --val_samples 20

# Train (requires GPU)
python train.py --epochs 1 --timing_test_steps 0 --fp16
```

## Running on SLURM Cluster (NEU Explorer)

### 1. Transfer files

```bash
scp -r ./* your_username@explorer.northeastern.edu:~/qwen-asd/
```

### 2. Set up environment

```bash
ssh your_username@explorer.northeastern.edu
srun --partition=short --mem=8G --time=01:00:00 --pty bash
cd ~/qwen-asd
bash setup_env.sh
```

### 3. Prepare data

```bash
conda activate qwen-asd
python prepare_data.py
```

### 4. Submit training job

```bash
sbatch train.sh
```

### 5. Monitor

```bash
squeue -u $USER              # check job status
tail -f logs/<job_id>.out    # follow output
```

### 6. Evaluate

```bash
sbatch evaluate.sh
```

## Training Details

- **Model:** Qwen2.5-Omni-3B (3 billion parameters, multimodal)
- **Method:** LoRA via PEFT library applied to the thinker component
- **GPU:** V100 32GB (fp16) or H200
- **Effective batch size:** 8 (batch_size=1 x gradient_accumulation=8)
- **Epochs:** 3
- **Learning rate:** 2e-4 with linear warmup
- **Estimated training time:** ~2-4 hours on V100

The training script includes a **50-step timing test** at startup that estimates total training time before committing to the full run.

## Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- Confusion matrix
- Comparison against baseline (model without fine-tuning)

## References

- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [UniTalk-ASD Paper](https://arxiv.org/html/2505.21954v1)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
